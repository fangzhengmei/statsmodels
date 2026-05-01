# Statsmodels 回归估计数值分解策略分析报告

## 1. 整体架构概览

Statsmodels 的线性回归估计系统采用了**分层继承 + 统一接口**的设计模式，确保不同的估计方法能够在相同的框架下协同工作。

### 1.1 类继承体系

```
base.LikelihoodModel
    └── RegressionModel (基类，定义统一接口)
            ├── GLS (广义最小二乘)
            │       └── GLSAR (带自回归误差的 GLS)
            └── WLS (加权最小二乘)
                    └── OLS (普通最小二乘)
```

### 1.2 核心设计理念

通过以下机制实现接口统一：
1. **统一入口**：`fit()` 方法定义算法骨架
2. **策略模式**：`method` 参数选择数值分解策略
3. **模板方法**：`whiten()` 方法由各子类实现数据预处理
4. **统一输出**：所有策略产生相同的中间结果格式

---

## 2. 数值分解策略详解

Statsmodels 支持两种主要的线性回归数值分解策略：**伪逆法 (PINV)** 和 **QR 分解法**。

### 2.1 伪逆法 (Pseudoinverse Method)

**实现位置**：`statsmodels/regression/linear_model.py:349-365`

#### 核心算法逻辑

```python
if method == "pinv":
    if not (hasattr(self, "pinv_wexog") and 
            hasattr(self, "normalized_cov_params") and 
            hasattr(self, "rank")):
        
        # 1. 使用 SVD 计算 Moore-Penrose 伪逆
        self.pinv_wexog, singular_values = pinv_extended(self.wexog)
        
        # 2. 计算归一化协方差参数: pinv(X) @ pinv(X).T
        self.normalized_cov_params = np.dot(
            self.pinv_wexog, np.transpose(self.pinv_wexog)
        )
        
        # 3. 缓存奇异值供后续使用
        self.wexog_singular_values = singular_values
        
        # 4. 基于奇异值计算矩阵秩
        self.rank = np.linalg.matrix_rank(np.diag(singular_values))
    
    # 5. 计算回归系数: β = pinv(X) @ y
    beta = np.dot(self.pinv_wexog, self.wendog)
```

#### PINV 扩展实现

**实现位置**：`statsmodels/tools/tools.py:244-265`

```python
def pinv_extended(x, rcond=1e-15):
    """
    返回伪逆矩阵以及计算中使用的奇异值
    代码改编自 NumPy
    """
    x = np.asarray(x)
    x = x.conjugate()
    
    # SVD 分解: X = U @ diag(S) @ Vt
    u, s, vt = np.linalg.svd(x, False)
    s_orig = np.copy(s)
    
    m = u.shape[0]
    n = vt.shape[1]
    
    # 数值稳定性: 截断小奇异值
    cutoff = rcond * np.maximum.reduce(s)
    for i in range(min(n, m)):
        if s[i] > cutoff:
            s[i] = 1./s[i]   # 大奇异值取倒数
        else:
            s[i] = 0.         # 小奇异值置零
    
    # 计算伪逆: pinv(X) = Vt.T @ diag(1/S) @ U.T
    res = np.dot(np.transpose(vt), np.multiply(s[:, np.newaxis],
                                               np.transpose(u)))
    return res, s_orig
```

#### 数学原理

最小二乘问题的正规方程：
```
X.T @ X @ β = X.T @ y
```

当 X 不满秩时，使用 Moore-Penrose 伪逆求解：
```
β = pinv(X) @ y
  = V @ diag(1/s) @ U.T @ y   (通过 SVD 计算)
```

其中 SVD 分解为：`X = U @ diag(S) @ V.T`

---

### 2.2 QR 分解法 (QR Factorization Method)

**实现位置**：`statsmodels/regression/linear_model.py:367-388`

#### 核心算法逻辑

```python
elif method == "qr":
    if not (hasattr(self, "exog_Q") and 
            hasattr(self, "exog_R") and 
            hasattr(self, "normalized_cov_params") and 
            hasattr(self, "rank")):
        
        # 1. QR 分解: X = Q @ R
        Q, R = np.linalg.qr(self.wexog)
        self.exog_Q, self.exog_R = Q, R
        
        # 2. 计算归一化协方差参数: inv(R.T @ R)
        self.normalized_cov_params = np.linalg.inv(np.dot(R.T, R))
        
        # 3. 从 R 计算奇异值 (用于诊断)
        self.wexog_singular_values = np.linalg.svd(R, 0, 0)
        
        # 4. 基于 R 计算矩阵秩
        self.rank = np.linalg.matrix_rank(R)
    else:
        Q, R = self.exog_Q, self.exog_R
    
    # 关键处理 1: 仍然计算 pinv 用于某些协方差估计器
    # 见 GH #8157 - 某些协方差估计器需要 pinv
    self.pinv_wexog = np.linalg.pinv(self.wexog)
    
    # 关键处理 2: 存储 effects 用于 ANOVA 分析
    self.effects = effects = np.dot(Q.T, self.wendog)
    
    # 5. 解三角系统: R @ β = Q.T @ y
    beta = np.linalg.solve(R, effects)
```

#### 数学原理

QR 分解将设计矩阵分解为正交矩阵 Q 和上三角矩阵 R：
```
X = Q @ R
```

最小二乘问题转化为：
```
min ||y - Xβ||² = min ||y - Q R β||²
                = min ||Q.T y - R β||²  (Q 正交保持范数)
```

由于 R 是上三角矩阵，可以通过回代法高效求解：
```
R @ β = Q.T @ y  →  直接回代求解
```

---

## 3. 统一接口机制详解

### 3.1 fit() 方法：统一入口

**实现位置**：`statsmodels/regression/linear_model.py:284-415`

所有回归模型共享同一个 `fit()` 方法，通过 `method` 参数选择不同的数值分解策略：

```python
def fit(
    self,
    method: Literal["pinv", "qr"] = "pinv",      # 数值分解策略选择
    cov_type: Literal["nonrobust", "HC0", "HC1", "HC2", "HC3", 
                       "HAC", "hac-panel", "hac-groupsum", "cluster"] = "nonrobust",
    cov_kwds=None,
    use_t: bool | None = None,
    **kwargs,
) -> RegressionResults:
    """
    完整的模型拟合
    
    参数
    ----------
    method : str, 可选
        可以是 "pinv" 或 "qr"。
        "pinv": 使用 Moore-Penrose 伪逆求解最小二乘问题
        "qr": 使用 QR 分解
    cov_type : str, 可选
        协方差估计器类型
    ...
    """
    # ========== 阶段 1: 数值分解 (策略选择) ==========
    if method == "pinv":
        # PINV 策略实现
        self.pinv_wexog, singular_values = pinv_extended(self.wexog)
        self.normalized_cov_params = np.dot(self.pinv_wexog, self.pinv_wexog.T)
        self.rank = np.linalg.matrix_rank(np.diag(singular_values))
        beta = np.dot(self.pinv_wexog, self.wendog)
        
    elif method == "qr":
        # QR 策略实现
        Q, R = np.linalg.qr(self.wexog)
        self.exog_Q, self.exog_R = Q, R
        self.normalized_cov_params = np.linalg.inv(np.dot(R.T, R))
        self.rank = np.linalg.matrix_rank(R)
        # 解三角系统
        effects = np.dot(Q.T, self.wendog)
        beta = np.linalg.solve(R, effects)
        # 兼容处理: 仍然计算 pinv
        self.pinv_wexog = np.linalg.pinv(self.wexog)
        
    else:
        raise ValueError('method has to be "pinv" or "qr"')
    
    # ========== 阶段 2: 统一自由度计算 ==========
    if self._df_model is None:
        self._df_model = float(self.rank - self.k_constant)
    if self._df_resid is None:
        self.df_resid = self.nobs - self.rank
    
    # ========== 阶段 3: 统一结果对象创建 ==========
    if isinstance(self, OLS):
        # OLS 专用结果对象
        lfit = OLSResults(
            self,
            beta,
            normalized_cov_params=self.normalized_cov_params,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            use_t=use_t,
        )
    else:
        # 通用结果对象
        lfit = RegressionResults(
            self,
            beta,
            normalized_cov_params=self.normalized_cov_params,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            use_t=use_t,
            **kwargs,
        )
    
    # 包装结果对象，提供更友好的 API
    return RegressionResultsWrapper(lfit)
```

### 3.2 数据预处理的统一流程

**实现位置**：`statsmodels/regression/linear_model.py:225-234`

无论使用哪种分解方法，数据都经过相同的预处理流程：

```python
class RegressionModel(base.LikelihoodModel):
    def __init__(self, endog, exog, **kwargs):
        super().__init__(endog, exog, **kwargs)
        self.pinv_wexog: Float64Array | None = None
        # 注册数据属性，用于数据处理
        self._data_attr.extend(["pinv_wexog", "wendog", "wexog", "weights"])

    def initialize(self):
        """初始化模型组件 - 所有模型共享"""
        # 1. 白化（whiten）设计矩阵 - 各模型自行实现
        self.wexog = self.whiten(self.exog)
        self.wendog = self.whiten(self.endog)
        
        # 2. 统一计算样本量
        self.nobs = float(self.wexog.shape[0])
        
        # 3. 初始化自由度和秩
        self._df_model = None
        self._df_resid = None
        self.rank = None
```

### 3.3 白化方法的策略差异

`whiten()` 方法是**模板方法模式**的典型应用，基类定义接口，子类实现具体变换：

| 模型 | whiten 方法实现 | 变换目的 | 代码位置 |
|------|----------------|----------|----------|
| **OLS** | `return x` | 无需变换 | `linear_model.py:1036-1054` |
| **WLS** | `x * sqrt(weights)` | 处理异方差 | `linear_model.py:815-834` |
| **GLS** | `cholsigmainv @ x` | 处理一般协方差结构 | `linear_model.py:587-614` |
| **GLSAR** | 减去自回归项 | 处理自相关误差 | `linear_model.py:1456-1479` |

#### OLS 白化实现

```python
def whiten(self, x):
    """OLS model whitener does nothing."""
    return x
```

#### WLS 白化实现

```python
def whiten(self, x):
    """Whitener for WLS model, multiplies each column by sqrt(self.weights)."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x * np.sqrt(self.weights)
    elif x.ndim == 2:
        return np.sqrt(self.weights)[:, None] * x
```

#### GLS 白化实现

```python
def whiten(self, x):
    """GLS whiten method - 使用 Cholesky 逆变换"""
    x = np.asarray(x)
    if self.sigma is None or self.sigma.shape == ():
        return x
    elif self.sigma.ndim == 1:
        # 对角协方差 (等价于 WLS)
        if x.ndim == 1:
            return x * self.cholsigmainv
        else:
            return x * self.cholsigmainv[:, None]
    else:
        # 一般协方差矩阵
        return np.dot(self.cholsigmainv, x)
```

---

## 4. 结果对象的统一生成

### 4.1 结果类继承体系

```
base.LikelihoodModelResults
    └── RegressionResults (通用回归结果)
            └── OLSResults (OLS 专用结果，增加特定功能)
```

### 4.2 结果对象初始化

**实现位置**：`statsmodels/regression/linear_model.py:1659-1757`

```python
class RegressionResults(base.LikelihoodModelResults):
    """
    汇总线性回归模型的拟合结果
    处理对比检验、协方差估计等
    """
    
    def __init__(
        self,
        model,                    # 模型实例
        params,                   # 估计的系数 β
        normalized_cov_params,   # 归一化协方差矩阵
        scale=1.0,               # 残差尺度
        cov_type="nonrobust",    # 协方差类型
        cov_kwds=None,           # 协方差额外参数
        use_t=None,              # 是否使用 t 分布
        **kwargs,
    ):
        # 调用父类初始化
        super().__init__(model, params, normalized_cov_params, scale)
        
        self._cache = {}
        
        # 继承模型的奇异值 (用于诊断)
        if hasattr(model, "wexog_singular_values"):
            self._wexog_singular_values = model.wexog_singular_values
        else:
            self._wexog_singular_values = None
        
        # 继承模型的自由度
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        
        # ========== 协方差类型处理 ==========
        if cov_type == "nonrobust":
            # 标准协方差: cov(β) = scale * normalized_cov_params
            self.cov_type = "nonrobust"
            self.cov_kwds = {
                "description": "Standard Errors assume that the covariance matrix of "
                               "the errors is correctly specified."
            }
            if use_t is None:
                use_t = True  # 默认使用 t 分布
            self.use_t = use_t
        else:
            # 稳健协方差: 调用专门的估计方法
            if cov_kwds is None:
                cov_kwds = {}
            if "use_t" in cov_kwds:
                use_t_2 = cov_kwds.pop("use_t")
                if use_t is None:
                    use_t = use_t_2
            
            # 调用稳健协方差估计方法
            self.get_robustcov_results(
                cov_type=cov_type, use_self=True, use_t=use_t, **cov_kwds
            )
        
        # 处理额外参数
        for key, value in kwargs.items():
            setattr(self, key, value)
```

### 4.3 协方差矩阵的统一计算

无论使用哪种分解方法，协方差矩阵的计算遵循统一的逻辑：

#### 基本公式

```
cov(β) = scale * normalized_cov_params

其中：
- scale = SSR / df_resid  (残差均方)
- normalized_cov_params:
  * PINV 方法: pinv(X) @ pinv(X).T
  * QR 方法: inv(R.T @ R)
```

#### 稳健协方差的统一接口

`get_robustcov_results()` 方法支持多种协方差类型：

| cov_type | 方法名称 | 适用场景 |
|----------|----------|----------|
| `nonrobust` | 标准 OLS 协方差 | 同方差、无自相关 |
| `HC0` | White 稳健协方差 | 存在异方差 |
| `HC1` | MacKinnon-White 调整 | 异方差 + 小样本校正 |
| `HC2` | 杠杆调整稳健协方差 | 异方差 + 高杠杆点 |
| `HC3` | 残差删除稳健协方差 | 异方差 (最保守) |
| `HAC` | Newey-West 协方差 | 异方差 + 自相关 |
| `cluster` | 聚类稳健标准误 | 面板数据/聚类结构 |
| `hac-panel` | 面板 HAC | 面板数据 |
| `hac-groupsum` | Driscoll-Kraay | 面板数据 (大 N 大 T) |

---

## 5. 两种分解策略的对比分析

### 5.1 数学等价性

从数学上讲，当设计矩阵 X 满秩时，两种方法应该得到相同的结果：

**PINV 方法**：
```
β_pinv = pinv(X) @ y
      = (X.T @ X)^-1 @ X.T @ y  (当 X 满秩时)
```

**QR 方法**：
```
X = Q @ R  (Q 正交, R 上三角)

由于 Q 正交:
X.T @ X = R.T @ Q.T @ Q @ R = R.T @ R
X.T @ y = R.T @ Q.T @ y

所以:
β_qr = solve(R, Q.T @ y) 
     = inv(R.T @ R) @ X.T @ y 
     = (X.T @ X)^-1 @ X.T @ y
```

**结论**：满秩时，`β_pinv = β_qr`

### 5.2 数值稳定性对比

| 特性 | PINV (SVD 基) | QR 分解 |
|------|---------------|---------|
| **奇异值处理** | 显式截断小奇异值 (`rcond` 参数控制) | 依赖 `np.linalg.solve` 的精度 |
| **秩检测** | 基于奇异值，更准确可靠 | 基于 R 的对角元大小 |
| **计算复杂度** | O(n×p²) | O(n×p²) |
| **内存使用** | 需要存储 U, S, Vt | 需要存储 Q, R |
| **数值稳定性** | 对多重共线性更稳定 | 对接近奇异矩阵可能有问题 |
| **默认选择** | ✅ statsmodels 默认 | 可选方法 |

### 5.3 代码中的差异处理

```python
# QR 方法的额外兼容处理
if method == "qr":
    # ... QR 分解 ...
    
    # === 差异 1: 仍然计算 pinv ===
    # 见 GH #8157 - 某些协方差估计器需要 pinv
    self.pinv_wexog = np.linalg.pinv(self.wexog)
    
    # === 差异 2: 存储 effects ===
    # effects = Q.T @ y，用于 ANOVA 分析
    self.effects = effects = np.dot(Q.T, self.wendog)
```

### 5.4 选择建议

**使用 PINV (默认) 的场景**：
- 存在多重共线性风险
- 需要准确的秩检测
- 对数值稳定性要求高

**使用 QR 的场景**：
- 设计矩阵满秩或接近满秩
- 需要进行 ANOVA 分析
- 追求微小的计算效率优势

---

## 6. 完整估计流程图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      用户调用: model.fit(method="pinv/qr")                 │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  阶段 1: 模型初始化 (所有模型共享流程)                                      │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ 1. 调用 model.initialize()                                          │  │
│  │    - wexog = whiten(exog)  [各模型自定义实现]                        │  │
│  │    - wendog = whiten(endog) [各模型自定义实现]                       │  │
│  │    - 更新 nobs, 初始化 df_model, df_resid, rank                     │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  阶段 2: 数值分解 (根据 method 分支)                                       │
│                                                                             │
│  ┌──────────────────────┐              ┌──────────────────────┐          │
│  │   method == "pinv"   │              │    method == "qr"    │          │
│  └──────────────────────┘              └──────────────────────┘          │
│           │                                        │                       │
│           ▼                                        ▼                       │
│  ┌─────────────────────────┐       ┌─────────────────────────┐          │
│  │ pinv_extended(wexog)    │       │ np.linalg.qr(wexog)    │          │
│  │ - SVD 分解: X=U@S@Vt    │       │ - QR 分解: X=Q@R        │          │
│  │ - 截断小奇异值          │       │ - Q 正交, R 上三角      │          │
│  │ - 返回 pinv + 奇异值    │       │ - 从 R 计算奇异值       │          │
│  └─────────────────────────┘       └─────────────────────────┘          │
│           │                                        │                       │
│           ▼                                        ▼                       │
│  ┌─────────────────────────┐       ┌─────────────────────────┐          │
│  │ β = pinv @ wendog       │       │ effects = Q.T @ wendog  │          │
│  │                         │       │ β = solve(R, effects)   │          │
│  └─────────────────────────┘       └─────────────────────────┘          │
│           │                                        │                       │
│           └────────────────────┬───────────────────┘                       │
│                                ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 统一输出中间结果:                                                    │   │
│  │ - β: 回归系数 (完全相同)                                             │   │
│  │ - normalized_cov_params: 归一化协方差 (数学等价)                     │   │
│  │ - rank: 矩阵秩                                                        │   │
│  │ - wexog_singular_values: 奇异值 (用于诊断)                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  阶段 3: 结果对象创建 (统一接口)                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ 1. 计算自由度: df_model = rank - k_constant                         │  │
│  │                        df_resid = nobs - rank                        │  │
│  │                                                                         │  │
│  │ 2. 创建结果对象:                                                        │  │
│  │    - OLS 模型 → OLSResults (OLS 专用)                                 │  │
│  │    - 其他模型 → RegressionResults (通用)                              │  │
│  │                                                                         │  │
│  │ 3. 处理协方差类型:                                                      │  │
│  │    - nonrobust: 使用 normalized_cov_params                           │  │
│  │    - 其他: 调用 get_robustcov_results() 计算稳健协方差                │  │
│  │                                                                         │  │
│  │ 4. 包装结果: RegressionResultsWrapper (提供友好 API)                  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      返回: RegressionResultsWrapper                        │
│  包含属性:                                                                  │
│  - params: 系数估计                                                        │
│  - bse: 标准误 (sqrt(diag(cov_params())))                                  │
│  - tvalues / pvalues: 推断统计量                                           │
│  - rsquared / rsquared_adj: 拟合优度                                       │
│  - resid / fittedvalues: 残差/拟合值                                       │
│  - scale / ssr / ess: 方差分解                                             │
│  - df_model / df_resid: 自由度                                             │
│  - cov_type / cov_kwds: 协方差信息                                         │
│                                                                             │
│  包含方法:                                                                  │
│  - summary(): 生成汇总输出                                                  │
│  - predict(): 预测                                                          │
│  - t_test() / f_test(): 假设检验                                           │
│  - conf_int(): 置信区间                                                     │
│  - get_prediction(): 预测区间                                               │
│  - get_robustcov_results(): 切换协方差类型                                 │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 关键设计模式总结

### 7.1 模板方法模式 (Template Method)

`RegressionModel.fit()` 定义了算法的骨架，将具体的数值分解步骤延迟到方法参数控制：

```
fit() 方法骨架:
┌─────────────────────────────────────────┐
│ 1. 检查缓存 (固定)                        │
│ 2. 选择数值分解策略 (可替换)              │
│    - PINV: pinv_extended()               │
│    - QR: np.linalg.qr() + solve()        │
│ 3. 计算 β (固定: β = ... @ y)            │
│ 4. 计算自由度 (固定)                      │
│ 5. 创建结果对象 (固定)                    │
└─────────────────────────────────────────┘
```

### 7.2 策略模式 (Strategy Pattern)

通过 `method` 参数在运行时选择不同的数值分解策略：

| 元素 | 实现 |
|------|------|
| **策略接口** | 产生相同的输出: `β`, `normalized_cov_params`, `rank` |
| **具体策略 1** | `pinv` 方法: 使用 SVD 计算伪逆 |
| **具体策略 2** | `qr` 方法: 使用 QR 分解 |
| **上下文** | `RegressionModel.fit()` 方法 |

### 7.3 工厂方法模式 (Factory Method)

结果对象的创建使用了工厂方法思想：

```python
if isinstance(self, OLS):
    lfit = OLSResults(...)  # OLS 专用结果对象
else:
    lfit = RegressionResults(...)  # 通用结果对象
```

### 7.4 装饰器模式 (Decorator Pattern)

`RegressionResultsWrapper` 装饰原始结果对象，提供更友好的 API：

```python
return RegressionResultsWrapper(lfit)
```

---

## 8. 代码参考位置速查

| 功能 | 文件路径 | 行号 |
|------|----------|------|
| RegressionModel 基类定义 | `linear_model.py` | 213-282 |
| fit() 方法主入口 | `linear_model.py` | 284-415 |
| PINV 策略实现 | `linear_model.py` | 349-365 |
| QR 策略实现 | `linear_model.py` | 367-388 |
| pinv_extended 函数 | `tools.py` | 244-265 |
| initialize() 方法 | `linear_model.py` | 225-234 |
| OLS.whiten() | `linear_model.py` | 1036-1054 |
| WLS.whiten() | `linear_model.py` | 815-834 |
| GLS.whiten() | `linear_model.py` | 587-614 |
| RegressionResults 类 | `linear_model.py` | 1659-2806 |
| get_robustcov_results() | `linear_model.py` | 2497-2806 |

---

## 9. 设计优势与总结

### 9.1 设计优势

1. **用户友好**：
   - 无需关心底层实现细节
   - 只需选择 `method` 参数即可切换策略
   - 所有模型使用相同的 `fit()` 接口

2. **可维护性**：
   - 数值计算与统计推断逻辑分离
   - 新增分解策略只需修改 `fit()` 方法的对应分支
   - 不影响结果接口和后处理逻辑

3. **可扩展性**：
   - 新增模型只需继承 `RegressionModel` 并实现 `whiten()` 方法
   - 自动获得所有数值分解策略的支持
   - 协方差估计器独立开发，通过 `get_robustcov_results()` 集成

### 9.2 核心要点总结

Statsmodels 的回归估计系统通过以下机制实现接口统一：

1. **统一入口**：所有模型共享 `fit()` 方法，通过 `method` 参数选择策略
2. **统一预处理**：`whiten()` 模板方法处理数据变换，`initialize()` 标准化初始化
3. **统一输出**：无论使用哪种方法，都产生相同的中间结果格式
4. **统一结果**：`RegressionResults` 类封装所有后处理逻辑，提供一致的推断 API

这种设计使得：
- **用户**：学习成本低，接口一致
- **开发者**：代码复用率高，易于维护扩展
- **代码质量**：逻辑清晰，职责分离
