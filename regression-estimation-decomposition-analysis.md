# Statsmodels 回归估计异常路径分析报告

## 1. 引言

本报告深入分析 statsmodels 回归估计在**异常路径**上的统一接口机制，重点关注：
1. **秩亏/近奇异矩阵**的检测与处理
2. **QR 求解失败**的退化处理
3. **稳健协方差依赖 pinv**的实现细节
4. **错误/退化信息**的缓存、传递与展示

---

## 2. 秩亏与近奇异矩阵的检测机制

### 2.1 核心概念

**秩亏矩阵** (Rank-deficient Matrix)：设计矩阵 X 的列之间存在完美线性依赖，导致 `rank(X) < p`（p 为自变量个数）。

**近奇异矩阵** (Near-singular Matrix)：设计矩阵 X 的列之间存在高度线性依赖（多重共线性），虽然数学上满秩，但数值上接近奇异，表现为**条件数很大**。

### 2.2 PINV 方法的内在鲁棒性

**实现位置**：`statsmodels/tools/tools.py:244-265`

PINV 方法通过 SVD 分解和小奇异值截断，**内在地**处理秩亏和近奇异问题：

```python
def pinv_extended(x, rcond=1e-15):
    """
    返回伪逆矩阵以及计算中使用的奇异值
    """
    x = np.asarray(x)
    x = x.conjugate()
    
    # ========== 步骤 1: SVD 分解 ==========
    # X = U @ diag(S) @ Vt
    u, s, vt = np.linalg.svd(x, False)
    s_orig = np.copy(s)  # 保存原始奇异值
    
    m = u.shape[0]
    n = vt.shape[1]
    
    # ========== 步骤 2: 小奇异值截断 (关键!) ==========
    # cutoff = rcond * max(singular_values)
    # 默认 rcond = 1e-15
    cutoff = rcond * np.maximum.reduce(s)
    
    for i in range(min(n, m)):
        if s[i] > cutoff:
            s[i] = 1./s[i]   # 大奇异值: 正常取倒数
        else:
            s[i] = 0.         # 小奇异值: 置零 (秩亏处理)
    
    # ========== 步骤 3: 计算伪逆 ==========
    # pinv(X) = Vt.T @ diag(1/S) @ U.T
    res = np.dot(np.transpose(vt), np.multiply(s[:, np.newaxis],
                                               np.transpose(u)))
    
    # 返回伪逆和原始奇异值
    return res, s_orig
```

### 2.3 奇异值的缓存机制

**实现位置**：`statsmodels/regression/linear_model.py:349-387`

```python
if method == "pinv":
    if not (hasattr(self, "pinv_wexog") and ...):
        # 计算伪逆和奇异值
        self.pinv_wexog, singular_values = pinv_extended(self.wexog)
        
        # ========== 关键缓存 1: 奇异值 ==========
        # 用于后续诊断 (条件数、特征值等)
        self.wexog_singular_values = singular_values
        
        # ========== 关键缓存 2: 矩阵秩 ==========
        # 基于奇异值计算，准确反映数值秩
        self.rank = np.linalg.matrix_rank(np.diag(singular_values))

elif method == "qr":
    if not (hasattr(self, "exog_Q") and ...):
        Q, R = np.linalg.qr(self.wexog)
        # ...
        
        # ========== QR 方法同样缓存奇异值 ==========
        # 从 R 矩阵计算奇异值 (用于一致性)
        self.wexog_singular_values = np.linalg.svd(R, 0, 0)
        
        # 基于 R 计算秩
        self.rank = np.linalg.matrix_rank(R)
```

### 2.4 诊断信息的计算与传递

#### 特征值计算

**实现位置**：`statsmodels/regression/linear_model.py:2044-2054`

```python
@cache_readonly
def eigenvals(self):
    """
    返回按降序排列的特征值
    特征值 = 奇异值²
    """
    if self._wexog_singular_values is not None:
        # 从缓存的奇异值计算
        eigvals = self._wexog_singular_values**2
    else:
        # 后备方案：直接计算
        wx = self.model.wexog
        eigvals = np.linalg.eigvalsh(wx.T @ wx)
    
    # 按降序排列
    return np.sort(eigvals)[::-1]
```

#### 条件数计算

**实现位置**：`statsmodels/regression/linear_model.py:2056-2067`

```python
@cache_readonly
def condition_number(self):
    """
    返回外生矩阵的条件数
    
    条件数 = sqrt(最大特征值 / 最小特征值)
           = 最大奇异值 / 最小奇异值
    
    解释:
    - 条件数小 (< 100): 无严重共线性
    - 条件数中等 (100-1000): 中等共线性
    - 条件数大 (> 1000): 严重共线性
    """
    eigvals = self.eigenvals
    return np.sqrt(eigvals[0] / eigvals[-1])
```

#### 奇异值的传递链

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      奇异值传递链                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  fit() 方法 (模型实例)                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ PINV 方法:                                                           │  │
│  │   self.pinv_wexog, singular_values = pinv_extended(self.wexog)    │  │
│  │   self.wexog_singular_values = singular_values  ←── 缓存           │  │
│  │                                                                      │  │
│  │ QR 方法:                                                             │  │
│  │   Q, R = np.linalg.qr(self.wexog)                                   │  │
│  │   self.wexog_singular_values = np.linalg.svd(R, 0, 0)  ←── 缓存  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  RegressionResults.__init__()                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ # 从模型继承奇异值                                                    │  │
│  │ if hasattr(model, "wexog_singular_values"):                         │  │
│  │     self._wexog_singular_values = model.wexog_singular_values      │  │
│  │ else:                                                                 │  │
│  │     self._wexog_singular_values = None                              │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  结果对象属性 (延迟计算)                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ @cache_readonly                                                       │  │
│  │ def eigenvals(self):                                                  │  │
│  │     if self._wexog_singular_values is not None:                      │  │
│  │         eigvals = self._wexog_singular_values**2                     │  │
│  │     return np.sort(eigvals)[::-1]                                     │  │
│  │                                                                      │  │
│  │ @cache_readonly                                                       │  │
│  │ def condition_number(self):                                           │  │
│  │     eigvals = self.eigenvals                                          │  │
│  │     return np.sqrt(eigvals[0] / eigvals[-1])                         │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.5 诊断信息在汇总输出中的展示

**实现位置**：`statsmodels/regression/linear_model.py:2875-2953`

```python
def summary(self, yname=None, xname=None, title=None, alpha=0.05, slim=False):
    # ...
    
    # ========== 计算诊断指标 ==========
    eigvals = self.eigenvals
    condno = self.condition_number
    
    # ========== 缓存诊断信息 ==========
    self.diagn = dict(
        jb=jb,                    # Jarque-Bera 正态性检验
        jbpv=jbpv,                # JB 检验 p 值
        skew=skew,                # 偏度
        kurtosis=kurtosis,        # 峰度
        omni=omni,                # Omnibus 正态性检验
        omnipv=omnipv,            # Omnibus p 值
        condno=condno,            # 条件数 ←── 关键诊断
        mineigval=eigvals[-1],    # 最小特征值 ←── 关键诊断
    )
    
    # ========== 在汇总表中显示 ==========
    if not slim:
        diagn_right = [
            ("Durbin-Watson:", ["%#8.3f" % durbin_watson(self.wresid)]),
            ("Jarque-Bera (JB):", ["%#8.3f" % jb]),
            ("Prob(JB):", ["%#8.3g" % jbpv]),
            ("Cond. No.", ["%#8.3g" % condno]),  # 显示条件数
        ]
```

**典型的汇总输出示例**：

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.999
Model:                            OLS   Adj. R-squared:                  0.999
Method:                 Least Squares   F-statistic:                 1.735e+04
Date:                Wed, 01 May 2026   Prob (F-statistic):           1.23e-45
Time:                        10:30:00   Log-Likelihood:                -88.572
No. Observations:                  50   AIC:                             185.1
Df Residuals:                      46   BIC:                             192.8
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          2.3456      0.123     19.070      0.000       2.098       2.593
x1             0.5678      0.045     12.618      0.000       0.477       0.658
x2             0.8901      0.234      3.804      0.000       0.419       1.361
x3             0.1234      0.567      0.218      0.828      -1.019       1.266
==============================================================================
Omnibus:                        1.234   Durbin-Watson:                   2.012
Prob(Omnibus):                  0.540   Jarque-Bera (JB):                1.345
Skew:                           0.234   Prob(JB):                        0.510
Kurtosis:                       2.567   Cond. No.                     1.23e+04  ←── 条件数!
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.23e+04. This might indicate that there are
strong multicollinearity or other numerical problems.  ←── 自动警告!
```

---

## 3. QR 求解失败的退化处理

### 3.1 线性模型 fit() 中的 QR 路径

**实现位置**：`statsmodels/regression/linear_model.py:367-388`

```python
elif method == "qr":
    if not (hasattr(self, "exog_Q") and hasattr(self, "exog_R") and ...):
        Q, R = np.linalg.qr(self.wexog)
        self.exog_Q, self.exog_R = Q, R
        self.normalized_cov_params = np.linalg.inv(np.dot(R.T, R))
        self.wexog_singular_values = np.linalg.svd(R, 0, 0)
        self.rank = np.linalg.matrix_rank(R)
    else:
        Q, R = self.exog_Q, self.exog_R
    
    # ========== 潜在风险点 ==========
    # np.linalg.solve 要求 R 是非奇异的
    # 如果 R 奇异，会抛出 np.linalg.LinAlgError
    
    self.pinv_wexog = np.linalg.pinv(self.wexog)  # 兼容处理
    self.effects = effects = np.dot(Q.T, self.wendog)
    beta = np.linalg.solve(R, effects)  # ←── 可能失败!
```

### 3.2 异常处理模式：yule_walker 示例

**实现位置**：`statsmodels/regression/linear_model.py:1575-1584`

虽然 `fit()` 方法中的 QR 路径没有显式的 try-except，但 statsmodels 在其他地方展示了**标准的异常处理模式**：

```python
def yule_walker(x, order=1, method="adjusted", df=None, inv=False, demean=True):
    # ... 计算自相关矩阵 R ...
    
    R = toeplitz(r[:-1])  # Yule-Walker 方程的系数矩阵
    
    try:
        # 尝试直接求解
        rho = np.linalg.solve(R, r[1:])
    except np.linalg.LinAlgError as err:
        # ========== 异常处理策略 ==========
        if "Singular matrix" in str(err):
            # 1. 发出警告 (非致命)
            warnings.warn(
                "Matrix is singular. Using pinv.", 
                SingularMatrixWarning, 
                stacklevel=2
            )
            # 2. 回退到伪逆方法 (退化但可用)
            rho = np.linalg.pinv(R) @ r[1:]
        else:
            # 其他类型错误重新抛出
            raise
```

### 3.3 QR 方法的兼容设计

虽然 `fit()` 中的 QR 路径没有显式的异常处理，但它有两个重要的**兼容设计**：

#### 设计 1：强制计算 pinv_wexog

```python
elif method == "qr":
    # ... QR 分解 ...
    
    # ========== 关键兼容设计 ==========
    # Needed for some covariance estimators, see GH #8157
    self.pinv_wexog = np.linalg.pinv(self.wexog)
    
    # ... 继续计算 ...
```

**为什么重要**：
- 稳健协方差估计（HC0-HC3）依赖 `pinv_wexog`
- 即使使用 QR 方法，也必须计算伪逆
- 这为后续的退化处理提供了基础

#### 设计 2：秩的显式计算

```python
elif method == "qr":
    # ...
    self.rank = np.linalg.matrix_rank(R)
    # ...
```

**为什么重要**：
- `np.linalg.matrix_rank()` 有内置的数值容忍度
- 它能检测到"数值上秩亏"的矩阵
- 这个秩值会传递给结果对象，影响自由度计算

### 3.4 潜在的改进空间

当前实现的潜在问题：

```python
# 当前 QR 路径
beta = np.linalg.solve(R, effects)  # 可能抛出 LinAlgError

# 更健壮的实现应该是:
try:
    beta = np.linalg.solve(R, effects)
except np.linalg.LinAlgError:
    warnings.warn(
        "QR solve failed, falling back to pinv.",
        SingularMatrixWarning,
        stacklevel=2
    )
    beta = np.dot(self.pinv_wexog, self.wendog)  # 使用已计算的 pinv
```

---

## 4. 稳健协方差对 pinv 的依赖机制

### 4.1 问题背景：GH #8157

在 `fit()` 方法的 QR 路径中有一行关键注释：

```python
# Needed for some covariance estimators, see GH #8157
self.pinv_wexog = np.linalg.pinv(self.wexog)
```

这揭示了一个重要的**设计约束**：即使选择 QR 方法，某些协方差估计器仍然依赖伪逆。

### 4.2 HCCM 方法的实现

**实现位置**：`statsmodels/regression/linear_model.py:2069-2115`

#### 核心 HCCM 计算函数

```python
def _HCCM(self, scale):
    """
    异方差一致协方差矩阵 (Heteroskedasticity-Consistent Covariance Matrix)
    
    公式: cov(β) = pinv(X) @ diag(scale) @ pinv(X).T
    
    其中 scale 是残差的某种变换 (不同 HC 类型有不同的 scale)
    """
    H = np.dot(self.model.pinv_wexog, scale[:, None] * self.model.pinv_wexog.T)
    return H
```

#### HC0-HC3 的具体实现

```python
@cache_readonly
def cov_HC0(self):
    """
    White (1980) 异方差稳健协方差
    scale = resid²
    """
    self.het_scale = self.wresid**2
    cov_HC0 = self._HCCM(self.het_scale)  # 调用 _HCCM，使用 pinv_wexog
    return cov_HC0

@cache_readonly
def cov_HC1(self):
    """
    MacKinnon-White (1985) 调整
    scale = (n/(n-p)) * resid²
    """
    self.het_scale = self.nobs / (self.df_resid) * (self.wresid**2)
    cov_HC1 = self._HCCM(self.het_scale)
    return cov_HC1

@cache_readonly
def cov_HC2(self):
    """
    杠杆调整 (Leverage adjustment)
    scale = resid² / (1 - h_ii)
    其中 h_ii 是帽子矩阵的对角元
    """
    wexog = self.model.wexog
    # 计算 h_ii = diag(X @ (X'X)^-1 @ X')
    h = self._abat_diagonal(wexog, self.normalized_cov_params)
    self.het_scale = self.wresid**2 / (1 - h)
    cov_HC2 = self._HCCM(self.het_scale)  # 仍然使用 pinv_wexog!
    return cov_HC2

@cache_readonly
def cov_HC3(self):
    """
    残差删除 (Residual deletion)
    scale = (resid / (1 - h_ii))²
    更保守的估计
    """
    wexog = self.model.wexog
    h = self._abat_diagonal(wexog, self.normalized_cov_params)
    self.het_scale = (self.wresid / (1 - h)) ** 2
    cov_HC3 = self._HCCM(self.het_scale)
    return cov_HC3
```

### 4.3 为什么稳健协方差依赖 pinv？

#### 数学原理

标准协方差矩阵：
```
cov(β) = σ² * (X'X)^-1
```

稳健协方差矩阵（三明治估计）：
```
cov_robust(β) = (X'X)^-1 @ X' @ Ω @ X @ (X'X)^-1

其中 Ω 是异方差形式的对角矩阵
```

使用伪逆的形式：
```
cov_pinv(β) = pinv(X) @ Ω @ pinv(X).T
```

**等价性**：当 X 满秩时，`pinv(X) = (X'X)^-1 @ X'`，所以两种形式等价。

#### 为什么不使用 normalized_cov_params？

`normalized_cov_params` 存储的是 `(X'X)^-1` 或其等价形式，但 HCCM 公式需要：

```python
# 使用 pinv 的形式 (更通用)
H = pinv(X) @ diag(scale) @ pinv(X).T

# 等价形式 (需要更多计算)
H = (X'X)^-1 @ X' @ diag(scale) @ X @ (X'X)^-1
```

使用 `pinv_wexog` 的优势：
1. **更简洁**：一次矩阵乘法即可完成
2. **更通用**：即使 X 秩亏也能工作
3. **更高效**：伪逆已经缓存，无需重新计算

### 4.4 依赖链完整分析

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   稳健协方差对 pinv 的依赖链                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  用户调用: results = model.fit(cov_type="HC0")                            │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ fit() 方法内部                                                        │  │
│  │ ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │ │ 无论 method 是 "pinv" 还是 "qr":                                │ │  │
│  │ │                                                                  │ │  │
│  │ │ PINV 路径:                                                       │ │  │
│  │ │   self.pinv_wexog = pinv_extended(self.wexog)[0]  ←── 计算   │ │  │
│  │ │                                                                  │ │  │
│  │ │ QR 路径:                                                         │ │  │
│  │ │   Q, R = np.linalg.qr(self.wexog)                               │ │  │
│  │ │   # ...                                                          │ │  │
│  │ │   self.pinv_wexog = np.linalg.pinv(self.wexog)  ←── 强制计算!  │ │  │
│  │ │   # 注释: Needed for some covariance estimators, see GH #8157   │ │  │
│  │ └─────────────────────────────────────────────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ RegressionResults.__init__()                                         │  │
│  │ ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │ │ if cov_type != "nonrobust":                                      │ │  │
│  │ │     self.get_robustcov_results(                                   │ │  │
│  │ │         cov_type=cov_type, use_self=True, ...                    │ │  │
│  │ │     )                                                             │ │  │
│  │ └─────────────────────────────────────────────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ 访问稳健协方差 (例如 results.cov_HC0 或 results.bse)                 │  │
│  │ ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │ │ @cache_readonly                                                  │ │  │
│  │ │ def cov_HC0(self):                                               │ │  │
│  │ │     self.het_scale = self.wresid**2                             │ │  │
│  │ │     cov_HC0 = self._HCCM(self.het_scale)                        │ │  │
│  │ │     return cov_HC0                                               │ │  │
│  │ └─────────────────────────────────────────────────────────────────┘ │  │
│  │                              │                                       │  │
│  │                              ▼                                       │  │
│  │ ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │ │ def _HCCM(self, scale):                                          │ │  │
│  │ │     # 关键: 依赖 self.model.pinv_wexog                           │ │  │
│  │ │     H = np.dot(self.model.pinv_wexog,                            │ │  │
│  │ │              scale[:, None] * self.model.pinv_wexog.T)           │ │  │
│  │ │     return H                                                      │ │  │
│  │ └─────────────────────────────────────────────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.5 如果没有 pinv_wexog 会怎样？

假设 QR 方法中没有这行代码：

```python
# 假设这行被删除
# self.pinv_wexog = np.linalg.pinv(self.wexog)
```

那么当用户尝试使用稳健协方差时：

```python
model = sm.OLS(y, X).fit(method="qr")
print(model.cov_HC0)  # 会发生什么?
```

**答案**：会抛出 `AttributeError`，因为 `self.model.pinv_wexog` 不存在。

这就是为什么 GH #8157 修复强制要求 QR 方法也计算 `pinv_wexog`。

---

## 5. 异常信息的完整传递路径

### 5.1 警告类型体系

**实现位置**：`statsmodels/tools/sm_exceptions.py:148-151`

```python
class SingularMatrixWarning(ModelWarning):
    """
    非致命的矩阵求逆问题，会影响输出结果
    """
```

**警告继承体系**：
```
Warning
└── UserWarning
    └── ModelWarning (基础内部警告类)
        ├── SingularMatrixWarning  ←── 矩阵奇异警告
        ├── ConvergenceWarning
        ├── CollinearityWarning
        └── ...
```

### 5.2 实际触发警告的场景

#### 场景 1：Yule-Walker 方程求解

```python
try:
    rho = np.linalg.solve(R, r[1:])
except np.linalg.LinAlgError as err:
    if "Singular matrix" in str(err):
        warnings.warn(
            "Matrix is singular. Using pinv.", 
            SingularMatrixWarning, 
            stacklevel=2
        )
        rho = np.linalg.pinv(R) @ r[1:]
```

#### 场景 2：混合线性模型中的协方差奇异

**实现位置**：`statsmodels/regression/mixed_linear_model.py:1285-1289`

```python
# 检查协方差矩阵是否奇异
if np.linalg.cond(cov) > 1e10:  # 条件数过大
    warnings.warn(
        _warn_cov_sing,  # "The covariance matrix of the random effects..."
        SingularMatrixWarning, 
        stacklevel=2
    )
```

#### 场景 3：广义估计方程中的奇异

**实现位置**：`statsmodels/genmod/generalized_estimating_equations.py:1525-1528`

```python
try:
    # 某些矩阵运算
except np.linalg.LinAlgError:
    msg = ("The working covariance matrix is singular. "
           "This may indicate that the model is not appropriate "
           "for the data.")
    warnings.warn(msg, SingularMatrixWarning, stacklevel=2)
```

### 5.3 诊断信息的多层访问

#### 层 1：直接访问属性

```python
results = model.fit()

# 数值诊断
print(f"条件数: {results.condition_number}")
print(f"特征值: {results.eigenvals}")
print(f"矩阵秩: {results.rank}")
print(f"样本量: {results.nobs}")
print(f"模型自由度: {results.df_model}")
print(f"残差自由度: {results.df_resid}")
```

#### 层 2：诊断字典

```python
# 在 summary() 调用后填充
results.summary()

print(results.diagn)
# 输出:
# {
#     'jb': 1.234,           # Jarque-Bera 统计量
#     'jbpv': 0.54,          # JB p 值
#     'skew': 0.234,         # 偏度
#     'kurtosis': 2.567,     # 峰度
#     'omni': 1.234,         # Omnibus 统计量
#     'omnipv': 0.54,        # Omnibus p 值
#     'condno': 12345.6,     # 条件数 ←── 关键
#     'mineigval': 0.00123,  # 最小特征值 ←── 关键
# }
```

#### 层 3：汇总输出

```python
print(results.summary())
# 包含:
# - 条件数显示
# - 如果条件数过大，自动添加警告注释
```

### 5.4 条件数阈值与解释

| 条件数范围 | 解释 | 建议行动 |
|-----------|------|----------|
| < 100 | 无严重共线性 | 无需特殊处理 |
| 100 - 1000 | 中等共线性 | 检查变量相关性，考虑 PCA |
| 1000 - 10000 | 较强共线性 | 可能影响系数稳定性 |
| > 10000 | 严重共线性 | 强烈建议检查模型设定 |

**注意**：这些阈值是经验法则，具体问题需要具体分析。

---

## 6. 异常路径的统一接口设计总结

### 6.1 设计原则

#### 原则 1：优雅退化而非崩溃

```
设计目标:
┌─────────────────────────────────────────────────────────────────┐
│  错误类型              当前行为              理想行为             │
├─────────────────────────────────────────────────────────────────┤
│  秩亏矩阵              PINV: 自动处理        保持                 │
│                        QR: 可能崩溃          退化到 PINV         │
│                                                                   │
│  近奇异矩阵            PINV: 截断小奇异值    保持                 │
│                        QR: 可能不准确        提供警告             │
│                                                                   │
│  稳健协方差请求        强制计算 pinv        保持                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 原则 2：诊断信息可访问

无论估计是否完全"成功"，用户都应该能够访问：
- 数值诊断（条件数、特征值、秩）
- 统计诊断（正态性检验、自相关检验）
- 警告信息（通过 Python warnings 机制）

#### 原则 3：接口一致性

- `fit(method="pinv")` 和 `fit(method="qr")` 返回**相同类型**的结果对象
- 结果对象具有**相同的属性和方法**
- 即使内部使用不同的数值方法，外部接口保持一致

### 6.2 统一接口的实现保障

#### 保障 1：缓存机制

```python
# 模型实例中的缓存
self.pinv_wexog           # 伪逆矩阵 (所有方法最终都有)
self.wexog_singular_values # 奇异值 (所有方法最终都有)
self.normalized_cov_params # 归一化协方差 (所有方法最终都有)
self.rank                  # 矩阵秩 (所有方法最终都有)

# 结果对象中的缓存
self._wexog_singular_values # 继承的奇异值
self.eigenvals               # 延迟计算的特征值
self.condition_number        # 延迟计算的条件数
self.diagn                   # 汇总后的诊断字典
```

#### 保障 2：兼容计算

QR 方法中的关键兼容设计：

```python
elif method == "qr":
    # ... QR 分解 ...
    
    # 保障 1: 强制计算 pinv_wexog
    self.pinv_wexog = np.linalg.pinv(self.wexog)
    
    # 保障 2: 计算奇异值 (与 PINV 方法一致)
    self.wexog_singular_values = np.linalg.svd(R, 0, 0)
    
    # 保障 3: 计算秩
    self.rank = np.linalg.matrix_rank(R)
```

#### 保障 3：统一结果对象

```python
# 无论使用哪种方法，都创建相同类型的结果对象
if isinstance(self, OLS):
    lfit = OLSResults(
        self,
        beta,
        normalized_cov_params=self.normalized_cov_params,
        cov_type=cov_type,
        cov_kwds=cov_kwds,
        use_t=use_t,
    )
else:
    lfit = RegressionResults(
        self,
        beta,
        normalized_cov_params=self.normalized_cov_params,
        cov_type=cov_type,
        cov_kwds=cov_kwds,
        use_t=use_t,
        **kwargs,
    )

# 统一包装
return RegressionResultsWrapper(lfit)
```

### 6.3 与正常路径的对比

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    正常路径 vs 异常路径                                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  正常路径 (满秩矩阵):                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ 1. fit() 计算 β, normalized_cov_params, rank                         │  │
│  │ 2. 创建 RegressionResults                                             │  │
│  │ 3. 用户访问 results.params, results.bse, results.pvalues             │  │
│  │ 4. 条件数小，无警告                                                    │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  异常路径 (秩亏/近奇异矩阵):                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ PINV 方法:                                                            │  │
│  │ 1. pinv_extended() 自动截断小奇异值                                   │  │
│  │ 2. 计算 β (最小范数解)                                                │  │
│  │ 3. 奇异值被缓存，可通过 condition_number 访问                         │  │
│  │ 4. summary() 显示条件数，如果过大则添加警告注释                       │  │
│  │                                                                      │  │
│  │ QR 方法:                                                              │  │
│  │ 1. np.linalg.solve() 可能抛出 LinAlgError (如果 R 严格奇异)         │  │
│  │ 2. 即使成功，pinv_wexog 也被强制计算 (用于稳健协方差)                │  │
│  │ 3. 奇异值和条件数同样可访问                                           │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  关键统一点:                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ ✓ 无论路径如何，结果对象类型相同                                      │  │
│  │ ✓ 无论路径如何，诊断属性 (condition_number, eigenvals) 都可访问      │  │
│  │ ✓ 无论路径如何，稳健协方差都能工作 (因为 pinv_wexog 存在)            │  │
│  │ ✓ 警告通过统一的 warnings 机制传递                                    │  │
│  │ ✓ 条件数等诊断信息在 summary() 中统一显示                             │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 代码参考位置速查

| 功能 | 文件路径 | 行号 |
|------|----------|------|
| **秩亏/近奇异处理** | | |
| pinv_extended 函数 | `tools/tools.py` | 244-265 |
| PINV 方法奇异值缓存 | `regression/linear_model.py` | 349-365 |
| QR 方法兼容处理 | `regression/linear_model.py` | 367-388 |
| eigenvals 属性 | `regression/linear_model.py` | 2044-2054 |
| condition_number 属性 | `regression/linear_model.py` | 2056-2067 |
| **异常处理** | | |
| SingularMatrixWarning 定义 | `tools/sm_exceptions.py` | 148-151 |
| yule_walker 异常处理模式 | `regression/linear_model.py` | 1575-1584 |
| **稳健协方差依赖** | | |
| _HCCM 核心函数 | `regression/linear_model.py` | 2070-2072 |
| cov_HC0 实现 | `regression/linear_model.py` | 2078-2085 |
| cov_HC1 实现 | `regression/linear_model.py` | 2087-2094 |
| cov_HC2 实现 | `regression/linear_model.py` | 2096-2105 |
| cov_HC3 实现 | `regression/linear_model.py` | 2107-2115 |
| **诊断信息展示** | | |
| summary() 诊断字典 | `regression/linear_model.py` | 2880-2890 |
| summary() 条件数显示 | `regression/linear_model.py` | 2948-2953 |

---

## 8. 总结与关键洞察

### 8.1 核心设计哲学

Statsmodels 在异常路径上的统一接口设计遵循以下哲学：

1. **数值鲁棒性优先**：
   - PINV 方法通过 SVD 和奇异值截断内在地处理秩亏
   - 即使选择 QR 方法，也强制计算伪逆以保证兼容性

2. **诊断透明性**：
   - 奇异值、特征值、条件数等关键诊断信息始终可访问
   - 汇总输出自动显示条件数，并在必要时添加警告

3. **接口一致性**：
   - 无论使用哪种估计方法，返回相同类型的结果对象
   - 结果对象具有相同的属性和方法签名

### 8.2 关键设计决策

| 决策 | 原因 | 影响 |
|------|------|------|
| QR 方法强制计算 `pinv_wexog` | 稳健协方差依赖伪逆 | 兼容性提高，但计算开销增加 |
| 奇异值始终缓存 | 诊断信息需要 | 用户可以随时检查数值问题 |
| 条件数在 summary() 中显示 | 帮助用户诊断问题 | 提高模型可解释性 |
| 使用 warnings 而非异常 | 非致命问题不应中断流程 | 优雅退化，用户可选择处理 |

### 8.3 潜在改进空间

1. **QR 路径的显式异常处理**：
   - 当前 `np.linalg.solve(R, effects)` 可能在 R 严格奇异时崩溃
   - 应该添加 try-except 并回退到 PINV 方法

2. **近奇异时的主动警告**：
   - 当前只有条件数显示，没有主动警告
   - 可以考虑在条件数超过阈值时发出 `CollinearityWarning`

3. **秩亏时的系数解释**：
   - 秩亏时系数估计不唯一
   - 可以在结果对象中添加更多诊断信息

---

## 附录 A：快速检查表

当遇到数值问题时，按以下顺序检查：

### A.1 诊断检查清单

```python
results = model.fit()

# 1. 检查条件数
cond = results.condition_number
print(f"条件数: {cond:.2e}")
if cond > 1000:
    print("⚠️  条件数较大，可能存在多重共线性")

# 2. 检查特征值
eig = results.eigenvals
print(f"特征值范围: [{eig[-1]:.2e}, {eig[0]:.2e}]")
if eig[-1] < 1e-10:
    print("⚠️  最小特征值接近零，矩阵接近奇异")

# 3. 检查秩
print(f"矩阵秩: {results.rank}")
print(f"自变量个数: {results.model.exog.shape[1]}")
if results.rank < results.model.exog.shape[1]:
    print("⚠️  矩阵秩亏!")

# 4. 查看完整汇总
print(results.summary())
```

### A.2 方法选择建议

| 场景 | 推荐方法 | 理由 |
|------|----------|------|
| 满秩矩阵，追求速度 | `method="qr"` | QR 分解通常更快 |
| 可能存在共线性 | `method="pinv"` (默认) | SVD 更数值稳定 |
| 需要使用稳健协方差 | 任意 | 两种方法最终都计算 pinv |
| 需要 ANOVA 分析 | `method="qr"` | QR 方法计算 effects |

### A.3 常见警告类型

| 警告类 | 触发场景 | 严重程度 |
|--------|----------|----------|
| `SingularMatrixWarning` | 矩阵求逆失败，使用 pinv 回退 | 中 |
| `CollinearityWarning` | 变量高度相关 | 中 |
| `ConvergenceWarning` | 优化算法未收敛 | 高 |
| `HessianInversionWarning` | Hessian 不可逆，标准误不可用 | 高 |
