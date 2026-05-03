# Statsmodels 线性回归估计器深度分析

本文档深入分析 statsmodels 中线性回归估计器的实现细节，包括求解方法、GLS/WLS 转换机制、M-估计量迭代逻辑以及递推与滚动回归的架构对比。

---

## 一、伪逆、QR、Cholesky 三条路径在病态矩阵下的表现

### 1.1 实现位置与调用路径

在 `statsmodels/regression/linear_model.py` 的 `RegressionModel.fit()` 方法中（第 284-415 行），实现了两种主要求解方法：

#### 伪逆法 (Pseudoinverse) - method="pinv"

```python
# 第 349-365 行
if method == "pinv":
    if not (hasattr(self, "pinv_wexog") and ...):
        self.pinv_wexog, singular_values = pinv_extended(self.wexog)
        self.normalized_cov_params = np.dot(
            self.pinv_wexog, np.transpose(self.pinv_wexog)
        )
        self.wexog_singular_values = singular_values
        self.rank = np.linalg.matrix_rank(np.diag(singular_values))
    beta = np.dot(self.pinv_wexog, self.wendog)
```

**技术要点**：
- 使用 `pinv_extended` 函数（来自 `statsmodels.tools.tools`），底层基于 SVD 分解
- 返回奇异值用于计算矩阵秩和检测共线性
- `normalized_cov_params = pinv(X) @ pinv(X).T`，这是 Moore-Penrose 伪逆的协方差估计

#### QR 分解法 - method="qr"

```python
# 第 367-389 行
elif method == "qr":
    if not (hasattr(self, "exog_Q") and ...):
        Q, R = np.linalg.qr(self.wexog)
        self.exog_Q, self.exog_R = Q, R
        self.normalized_cov_params = np.linalg.inv(np.dot(R.T, R))
        self.wexog_singular_values = np.linalg.svd(R, 0, 0)
        self.rank = np.linalg.matrix_rank(R)
    else:
        Q, R = self.exog_Q, self.exog_R
    self.pinv_wexog = np.linalg.pinv(self.wexog)  # 第 384 行，为协方差计算备用
    self.effects = effects = np.dot(Q.T, self.wendog)
    beta = np.linalg.solve(R, effects)
```

**技术要点**：
- 使用 `np.linalg.qr` 进行 QR 分解
- 求解使用 `np.linalg.solve(R, effects)`，即回代求解 Rx = Q'y
- 协方差矩阵计算为 `inv(R'R)`，这需要 R 是非奇异的

#### Cholesky 分解的使用场景

Cholesky 分解在当前 OLS.fit() 中**未作为主要求解选项**，但在以下关键位置使用：

**位置 1：GLS 协方差矩阵预处理**（`linear_model.py` 第 176-210 行）

```python
def _get_sigma(sigma, nobs):
    """返回 sigma 及其 Cholesky 分解的逆"""
    if sigma.ndim == 2:
        # 第 201-203 行：对 sigma 做 Cholesky 分解，然后求三角矩阵的逆
        cholsigmainv, info = dtrtri(
            cholesky(sigma, lower=True), lower=True, overwrite_c=True
        )
        if info > 0:
            raise np.linalg.LinAlgError(
                "Cholesky decomposition of sigma yields a singular matrix"
            )
```

**位置 2：隐含在 `inv(R'R)` 计算中**

当使用 QR 方法时，`normalized_cov_params = np.linalg.inv(np.dot(R.T, R))`（第 376 行）。由于 R 是上三角矩阵，R'R 是对称正定矩阵，理论上可以用 Cholesky 分解求逆，但实际调用的是通用的 `np.linalg.inv`。

### 1.2 病态矩阵下的行为差异

| 方法 | 数学基础 | 病态矩阵表现 | 数值稳定性 |
|------|---------|-------------|-----------|
| **伪逆 (SVD)** | $X^+ = V\Sigma^+U^T$ | 通过截断小奇异值自动处理秩亏 | ⭐⭐⭐ 最稳定 |
| **QR 分解** | $X = QR, \hat{\beta} = R^{-1}Q'y$ | 当 $R$ 接近奇异时，`solve(R, ...)` 可能失败或产生大误差 | ⭐⭐ 中等稳定 |
| **Cholesky** | $X'X = LL', \hat{\beta} = L'^{-1}L^{-1}X'y$ | 要求 $X'X$ 正定，病态时分解失败或精度损失 | ⭐ 对病态最敏感 |

**关键代码证据**：

1. **伪逆的秩检测机制**（第 363 行）：
```python
self.rank = np.linalg.matrix_rank(np.diag(singular_values))
```
伪逆法通过奇异值显式计算秩，能够自动处理列共线性。

2. **QR 方法的潜在问题**：
   - 第 376 行：`np.linalg.inv(np.dot(R.T, R))` 直接求逆，当 R 病态时精度差
   - 第 387 行：`np.linalg.solve(R, effects)` 使用回代，R 接近奇异时会警告或失败

3. **Cholesky 的严格正定要求**（第 204-207 行）：
```python
if info > 0:
    raise np.linalg.LinAlgError(
        "Cholesky decomposition of sigma yields a singular matrix"
    )
```
Cholesky 分解在 `_get_sigma` 中对 sigma 矩阵严格要求正定，奇异时直接抛异常。

### 1.3 实际使用建议

- **默认方法**：statsmodels 默认使用 `method="pinv"`（第 286 行），正是因为其对病态矩阵的鲁棒性
- **QR 方法**：当设计矩阵条件数适中时更快，但需要注意 `np.linalg.solve` 的数值警告
- **Cholesky**：仅用于 GLS 的协方差白化步骤，不用于直接求解 OLS

---

## 二、GLS 和 WLS 如何转换为等价 OLS

### 2.1 核心设计模式：Template Method + 白化 (Whitening)

整个回归框架采用**模板方法模式**，基类 `RegressionModel` 定义算法骨架，子类通过覆写 `whiten()` 方法实现不同的变换策略。

### 2.2 类继承关系

```
RegressionModel (基类)
├── GLS (广义最小二乘)
│   └── GLSAR (带自回归误差的 GLS)
└── WLS (加权最小二乘)
    └── OLS (普通最小二乘)
```

**注意**：OLS 继承自 WLS，因为 OLS 是权重全为 1 的特殊 WLS。

### 2.3 转换机制详解

#### 关键入口：initialize() 方法（`linear_model.py` 第 225-234 行）

```python
def initialize(self):
    """Initialize model components."""
    self.wexog = self.whiten(self.exog)    # 白化解释变量
    self.wendog = self.whiten(self.endog)    # 白化被解释变量
    # overwrite nobs from class Model:
    self.nobs = float(self.wexog.shape[0])
    # ... 其他初始化
```

**核心思想**：在 `fit()` 被调用前，所有数据已被白化。`fit()` 方法实际操作的是 `self.wexog` 和 `self.wendog`，而非原始的 `self.exog` 和 `self.endog`。

#### fit() 方法使用白化后的数据（第 356, 365, 374, 387 行）

```python
# 伪逆方法使用 wexog
self.pinv_wexog, singular_values = pinv_extended(self.wexog)  # 第 356 行
beta = np.dot(self.pinv_wexog, self.wendog)                    # 第 365 行

# QR 方法同样使用 wexog
Q, R = np.linalg.qr(self.wexog)                                 # 第 374 行
beta = np.linalg.solve(R, effects)                              # 第 387 行，effects 基于 wendog
```

### 2.4 WLS 的 whitening 实现

**位置**：`linear_model.py` 第 815-834 行

```python
def whiten(self, x):
    """Whitener for WLS model, multiplies each column by sqrt(self.weights)."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x * np.sqrt(self.weights)
    elif x.ndim == 2:
        return np.sqrt(self.weights)[:, None] * x
```

**数学原理**：
- WLS 最小化：$\sum_{i=1}^n w_i (y_i - x_i'\beta)^2$
- 等价于对变换后数据做 OLS：
  - $\tilde{y}_i = \sqrt{w_i} y_i$
  - $\tilde{x}_{ij} = \sqrt{w_i} x_{ij}$
- 变换后最小化：$\sum_{i=1}^n (\tilde{y}_i - \tilde{x}_i'\beta)^2$，即标准 OLS

### 2.5 GLS 的 whitening 实现

**位置**：`linear_model.py` 第 587-614 行

```python
def whiten(self, x):
    """GLS whiten method. Returns np.dot(cholsigmainv, X)."""
    x = np.asarray(x)
    if self.sigma is None or self.sigma.shape == ():
        return x
    elif self.sigma.ndim == 1:
        # sigma 是对角方差向量 → 退化为 WLS
        if x.ndim == 1:
            return x * self.cholsigmainv
        else:
            return x * self.cholsigmainv[:, None]
    else:
        # sigma 是完整协方差矩阵 → 真正的 GLS 变换
        return np.dot(self.cholsigmainv, x)
```

**配合 `_get_sigma` 理解**（第 176-210 行）：

```python
def _get_sigma(sigma, nobs):
    # ...
    if sigma.ndim == 1:
        # 对角情况：cholsigmainv = 1 / sqrt(sigma)
        cholsigmainv = 1 / np.sqrt(sigma)                    # 第 194 行
    else:
        # 完整矩阵：Cholesky 分解 sigma = LL'，则 cholsigmainv = L^{-1}
        cholsigmainv, info = dtrtri(
            cholesky(sigma, lower=True), lower=True, ...     # 第 201-203 行
        )
```

**数学原理**：
- GLS 假设：$Var(\varepsilon) = \sigma^2 \Sigma$
- GLS 最小化：$(y - X\beta)'\Sigma^{-1}(y - X\beta)$
- Cholesky 分解：$\Sigma = LL'$，因此 $\Sigma^{-1} = L'^{-1}L^{-1}$
- 令变换矩阵 $P = L^{-1}$（即代码中的 `cholsigmainv`）
- 白化后数据：$\tilde{y} = Py$, $\tilde{X} = PX$
- 变换后目标：$(\tilde{y} - \tilde{X}\beta)'(\tilde{y} - \tilde{X}\beta)$，即标准 OLS

### 2.6 OLS 的特殊情况

**位置**：`linear_model.py` 第 1036-1054 行

```python
def whiten(self, x):
    """OLS model whitener does nothing. Returns the input array unmodified."""
    return x
```

OLS 的 `whiten` 是恒等变换，验证了"OLS 是权重全为 1 的 WLS，也是 sigma=I 的 GLS"这一设计。

### 2.7 正则化拟合中的复用证据

在 `fit_regularized` 方法中，这种复用更加明显：

**GLS.fit_regularized**（第 690-730 行）：
```python
def fit_regularized(self, ...):
    # ... 调整 alpha 后
    rslt = OLS(self.wendog, self.wexog).fit_regularized(...)  # 第 714 行
    # 直接用白化后的数据创建 OLS 模型！
```

**WLS.fit_regularized**（第 897-930 行）：
```python
def fit_regularized(self, ...):
    # ...
    rslt = OLS(self.wendog, self.wexog).fit_regularized(...)  # 第 914 行
    # 同样复用 OLS 的正则化拟合
```

**关键洞察**：
> 无论是 `fit()` 还是 `fit_regularized()`，GLS/WLS 的核心策略都是：**通过 `whiten()` 将问题转化为标准 OLS 问题，然后完全复用 OLS 的求解 machinery**。

---

## 三、M-估计量迭代重加权 (IRLS) 的逻辑分布

### 3.1 整体架构

M-估计量通过 **RLM (Robust Linear Model)** 类实现，位于 `statsmodels/robust/robust_linear_model.py`。核心算法是 **迭代重加权最小二乘 (IRLS)**。

### 3.2 IRLS 主循环

**位置**：`robust_linear_model.py` 第 195-349 行的 `fit()` 方法

```python
def fit(self, maxiter=50, tol=1e-8, scale_est="mad", ...):
    # ===== 第 1 步：初始估计 =====
    if start_params is None:
        wls_results = lm.WLS(self.endog, self.exog).fit()  # 第 270 行：用 OLS 作初始值
    else:
        # ... 自定义初始值处理
    
    # ===== 第 2 步：初始尺度估计 =====
    if not init and not start_scale:
        self.scale = self._estimate_scale(wls_results.resid)  # 第 288 行
    # scale_est 支持 "mad" (中位数绝对偏差) 或 HuberScale()
    
    # ===== 第 3 步：IRLS 迭代循环 =====
    while not converged:                                    # 第 311 行
        # 3.1 检查尺度是否为 0（完美拟合）
        if self.scale == 0.0:
            warnings.warn("Estimated scale is 0.0...", ConvergenceWarning)
            break
        
        # 3.2 核心：根据残差计算权重 ←─ M-估计量的核心
        self.weights = self.M.weights(
            wls_results.resid / self.scale                  # 第 323 行：标准化残差
        )
        
        # 3.3 加权最小二乘更新
        wls_results = reg_tools._MinimalWLS(
            self.endog, self.exog, 
            weights=self.weights, check_weights=True
        ).fit()                                            # 第 324-326 行
        
        # 3.4 可选：更新尺度估计
        if update_scale is True:
            self.scale = self._estimate_scale(wls_results.resid)  # 第 328 行
        
        # 3.5 收敛检查
        iteration += 1
        converged = _check_convergence(criterion, iteration, tol, maxiter)  # 第 331 行
```

### 3.3 权重函数的分散实现

IRLS 的核心——**权重计算**——分散在 `statsmodels/robust/norms.py` 的各个 `RobustNorm` 子类中。

#### 基类定义（`norms.py` 第 16-93 行）

```python
class RobustNorm:
    def rho(self, z):
        """准则函数：-2 loglike used in M-estimator"""
        raise NotImplementedError
    
    def psi(self, z):
        """影响函数：psi = rho'（rho 的导数）"""
        raise NotImplementedError
    
    def weights(self, z):
        """IRLS 权重：psi(z) / z"""
        raise NotImplementedError
    
    def psi_deriv(self, z):
        """psi 的导数：用于鲁棒协方差估计"""
        raise NotImplementedError
```

#### 关键关系

```
权重 w(z) = ψ(z) / z
其中 ψ(z) = ρ'(z) 是准则函数的一阶导数
```

这确保了 IRLS 的不动点是 M-估计方程 $\sum \psi(r_i/\sigma) x_i = 0$ 的解。

### 3.4 各 M-估计量的权重函数对比

| 估计量 | 类名 | 权重函数 w(z) | 特性 |
|--------|------|--------------|------|
| **最小二乘** | `LeastSquares` | $w(z) = 1$ | 参考基准，无降权 |
| **Huber T** | `HuberT` | $w(z) = \min(1, t/|z|)$ | 软降权，对大残差线性降权 |
| **Hampel** | `Hampel` | 三段式降权 | 硬降权，超过 c 权重为 0 |
| **Tukey 双权重** | `TukeyBiweight` | $w(z) = (1 - (z/c)^2)^2 \cdot I(|z| \leq c)$ | 硬降权，最流行的选择 |
| **Andrew 波** | `AndrewWave` | $w(z) = \sin(z/a)/(z/a) \cdot I(|z| \leq a\pi)$ | 硬降权，周期性权重 |
| **Ramsay Ea** | `RamsayE` | $w(z) = \exp(-a|z|)$ | 软降权，指数衰减 |
| **修剪均值** | `TrimmedMean` | $w(z) = I(|z| \leq c)$ | 硬降权，简单截断 |
| **学生 T** | `StudentT` | $w(z) = df / (df + (z/c)^2)$ | 软降权，基于 t 分布 |

#### 代码示例：HuberT 的权重（`norms.py` 第 274-307 行）

```python
def weights(self, z):
    r"""Huber's t weighting function for the IRLS algorithm.
    
    w(z) = 1          for |z| <= t
    w(z) = t/|z|      for |z| > t
    """
    z_isscalar = np.isscalar(z)
    z = np.atleast_1d(z)
    test = self._subset(z)           # |z| <= t
    absz = np.abs(z)
    absz[test] = 1.0                 # 对于 |z| <= t，避免除以 z
    v = test + (1 - test) * self.t / absz
    # 即：v = 1 或 t/|z|
    if z_isscalar:
        v = v[0]
    return v
```

#### 代码示例：TukeyBiweight 的权重（`norms.py` 第 1023-1043 行）

```python
def weights(self, z):
    r"""Tukey's biweight weighting function.
    
    w(z) = (1 - (z/c)^2)^2    for |z| <= c
    w(z) = 0                    for |z| > c
    """
    z = np.asarray(z)
    subset = self._subset(z)      # |z| <= c
    return (1 - (z / self.c)**2)**2 * subset
```

### 3.5 IRLS 逻辑的三个分散位置

IRLS 算法的逻辑并非集中在一处，而是分散在三个层次：

| 位置 | 文件/类 | 职责 |
|------|---------|------|
| **第一层** | `RLM.fit()` | 迭代控制、初始值、尺度估计、收敛检查 |
| **第二层** | `RobustNorm.weights()` | 具体 M-估计量的权重函数定义 |
| **第三层** | `_MinimalWLS.fit()` | 每次迭代中的 WLS 求解（轻量级实现） |

#### 第三层：_MinimalWLS 的作用

在第 324-326 行调用的 `reg_tools._MinimalWLS` 是一个轻量级 WLS 实现，避免了每次迭代都创建完整的 `WLS` 模型对象的开销。

### 3.6 收敛检查机制

**位置**：`robust_linear_model.py` 第 31-33 行

```python
def _check_convergence(criterion, iteration, tol, maxiter):
    cond = np.abs(criterion[iteration] - criterion[iteration - 1])
    return not (np.any(cond > tol) and iteration < maxiter)
```

支持的收敛准则（通过 `conv` 参数选择）：
- `"dev"`：偏差函数值（deviance）
- `"coefs"`：系数变化
- `"weights"`：权重变化
- `"sresid"`：标准化残差变化

### 3.7 尺度估计的重要性

尺度估计 `self.scale` 在 IRLS 中至关重要，因为权重是基于**标准化残差** $r/\sigma$ 计算的。

**位置**：`robust_linear_model.py` 第 178-193 行

```python
def _estimate_scale(self, resid):
    if isinstance(self.scale_est, str):
        if self.scale_est.lower() == "mad":
            return scale.mad(resid, center=0)  # 中位数绝对偏差
    elif isinstance(self.scale_est, scale.HuberScale):
        return self.scale_est(self.df_resid, self.nobs, resid)  # Huber 提案 2
    else:
        return self.scale_est(resid) * np.sqrt(self.nobs / self.df_resid)
```

**MAD (Median Absolute Deviation)** 是默认选择，因为它对异常值鲁棒：
$$\text{MAD} = 1.4826 \times \text{median}(|r_i - \text{median}(r)|)$$
常数 1.4826 使 MAD 在正态数据下一致估计 $\sigma$。

---

## 四、递推最小二乘与滚动回归的架构对比

### 4.1 定位与继承关系

| 方法 | 文件名 | 基类 | 核心思想 |
|------|--------|------|---------|
| **递推最小二乘** | `regression/recursive_ls.py` | `MLEModel` (状态空间模型) | 扩展窗口，Kalman 滤波递推 |
| **滚动回归** | `regression/rolling.py` | 独立实现，无继承 | 固定窗口，增量更新矩估计 |

### 4.2 递推最小二乘 (RecursiveLS) 实现

#### 核心洞察：状态空间表示

**位置**：`recursive_ls.py` 第 34-145 行

```python
class RecursiveLS(MLEModel):
    """Recursive least squares
    
    Notes
    -----
    Recursive least squares (RLS) corresponds to expanding window ordinary
    least squares (OLS).
    
    This model applies the Kalman filter to compute recursive estimates of the
    coefficients and recursive residuals.
    """
    def __init__(self, endog, exog, constraints=None, **kwargs):
        # ... 数据标准化 ...
        
        # ===== 关键：设置状态空间表示 =====
        super().__init__(
            endog, k_states=self.k_exog, exog=exog, **kwargs)  # 第 119 行
        
        # 状态转移矩阵：恒等矩阵（系数随时间不变）
        self["transition"] = np.eye(self.k_states)  # 第 133, 138 行
        
        # 观测矩阵：随时间变化的设计矩阵 X_t
        self["design"] = np.zeros((self.k_endog, self.k_states, self.nobs))
        self["design", 0] = self.exog[:, :, None].T  # 第 129-130 行
        
        # 观测方差设为 1（集中处理）
        self["obs_cov", 0, 0] = 1.  # 第 137 行
        
        # 使用扩散初始化和集中尺度
        self.ssm.filter_univariate = True
        self.ssm.filter_concentrated = True
```

#### 状态空间模型对应的 RLS

将线性回归表示为状态空间模型：

- **状态向量**：$\alpha_t = \beta$（回归系数，假定随时间常数）
- **状态方程**：$\alpha_{t+1} = \alpha_t$（转移矩阵为单位矩阵）
- **观测方程**：$y_t = x_t' \alpha_t + \varepsilon_t$

然后用 **Kalman 滤波** 递推计算：
- 滤波状态估计：$a_{t|t} = E[\alpha_t | Y_t]$
- 滤波协方差：$P_{t|t} = Var[\alpha_t | Y_t]$

这恰好对应使用前 t 个观测的 OLS 估计量！

#### fit() 方法：调用平滑器

**位置**：`recursive_ls.py` 第 157-170 行

```python
def fit(self):
    """Fits the model by application of the Kalman filter"""
    smoother_results = self.smooth(return_ssm=True)  # 第 165 行
    
    with self.ssm.fixed_scale(smoother_results.scale):
        res = self.smooth()                           # 第 168 行
    
    return res
```

这里调用的是 `MLEModel.smooth()`，底层执行 Kalman 滤波和平滑。

#### 递归系数的提取

**位置**：`recursive_ls.py` 第 299-335 行

```python
@property
def recursive_coefficients(self):
    """Estimates of regression coefficients, recursively estimated"""
    out = Bunch(
        filtered=self.filtered_state[start:end],           # 滤波估计
        filtered_cov=self.filtered_state_cov[start:end, start:end],
        smoothed=None, smoothed_cov=None,
        offset=offset
    )
    if self.smoothed_state is not None:
        out.smoothed = self.smoothed_state[start:end]     # 平滑估计
    # ...
    return out
```

`filtered_state[:, t]` 就是使用前 t 个观测得到的递归 OLS 估计。

### 4.3 滚动回归 (RollingOLS/RollingWLS) 实现

#### 核心洞察：增量更新矩估计

与 RecursiveLS 完全不同，RollingOLS 采用**固定窗口**和**矩更新**策略。

**位置**：`rolling.py` 第 289-382 行的 `fit()` 方法

```python
def fit(self, method="inv", reset=None, ...):
    # ===== 初始化滚动存储 =====
    store = RollingStore(
        params=np.full((nobs, k), np.nan),
        ssr=np.full(nobs, np.nan),
        # ...
    )
    
    # ===== 第一个窗口 =====
    first = self._min_nobs if self._expanding else w
    xpx, xpy, nobs = self._reset(first)  # 第 357 行：计算初始 X'X 和 X'y
    self._fit_single(first, xpx, xpy, nobs, store, params_only, method)
    
    # ===== 滚动更新 =====
    for i in range(first + 1, self._x.shape[0] + 1):  # 第 361 行
        # 可选：定期重置以避免数值误差累积
        if i % reset == 0:
            xpx, xpy, nobs = self._reset(i)
        else:
            # ===== 核心增量更新 =====
            # 移除移出窗口的观测
            if not self._is_nan[i - w - 1] and i > w:
                remove_x = wx[i - w - 1 : i - w]
                xpx -= remove_x.T @ remove_x      # 第 369 行：减去 x_old x_old'
                xpy -= remove_x.T @ wy[i - w - 1 : i - w]  # 减去 x_old y_old
                nobs -= 1
            
            # 添加新进入窗口的观测
            if not self._is_nan[i - 1]:
                add_x = wx[i - 1 : i]
                xpx += add_x.T @ add_x          # 第 374 行：加上 x_new x_new'
                xpy += add_x.T @ wy[i - 1 : i]  # 加上 x_new y_new
                nobs += 1
        
        # ===== 拟合当前窗口 =====
        self._fit_single(i, xpx, xpy, nobs, store, params_only, method)
```

#### 数学原理：递推更新 OLS

对于固定窗口大小 w，当窗口从 $[t-w, t-1]$ 移动到 $[t-w+1, t]$ 时：

- 移出观测：$x_{t-w}, y_{t-w}$
- 移入观测：$x_t, y_t$

矩的更新：
$$X'X_{new} = X'X_{old} - x_{t-w}x_{t-w}' + x_t x_t'$$
$$X'y_{new} = X'y_{old} - x_{t-w}y_{t-w} + x_t y_t$$

然后求解：
$$\hat{\beta} = (X'X)^{-1} X'y$$

#### 三种求解方法

在 `_fit_single` 中（第 226-264 行）支持三种方法：

```python
def _fit_single(self, idx, wxpwx, wxpwy, nobs, store, params_only, method):
    try:
        if (method == "inv") or not params_only:
            wxpwxi = np.linalg.inv(wxpwx)      # 方法 1：直接求逆
        if method == "inv":
            params = wxpwxi @ wxpwy
        else:
            _, wy, wx, _, _ = self._get_data(idx)
            if method == "lstsq":
                params = lstsq(wx, wy)[0]       # 方法 2：最小二乘求解器
            else:  # 'pinv'
                wxpwxiwxp = np.linalg.pinv(wx)  # 方法 3：伪逆
                params = wxpwxiwxp @ wy
    except np.linalg.LinAlgError:
        return  # 奇异时跳过
```

#### 重置机制 (reset)

注意第 364-365 行的 `reset` 参数：
```python
if i % reset == 0:
    xpx, xpy, nobs = self._reset(i)
```

这是为了**防止数值误差累积**。增量更新中的浮点舍入误差会随时间累积，定期通过 `_reset()` 重新计算完整的 $X'X$ 和 $X'y$ 可以重置数值精度。

### 4.4 关键架构对比总结

| 维度 | 递推最小二乘 (RecursiveLS) | 滚动回归 (RollingOLS) |
|------|---------------------------|----------------------|
| **继承体系** | 继承 `MLEModel`（状态空间框架） | 独立实现，无继承 |
| **窗口类型** | 扩展窗口 (expanding window) | 固定窗口 (rolling window) + 可选扩展 |
| **核心算法** | Kalman 滤波递推 | 矩估计增量更新 |
| **数值稳定性** | 依赖 Kalman 滤波的数值稳定性 | 依赖 `reset` 定期重置精度 |
| **状态存储** | 完整状态空间结果（滤波/平滑状态、协方差） | 仅存储回归结果（params, ssr, 等） |
| **特殊功能** | 递归残差、CUSUM 检验、结构突变检验 | 仅系数估计和标准统计量 |
| **代码复杂度** | 高（复用复杂的状态空间 machinery） | 低（独立、透明的循环） |

### 4.5 两者是否共用核心结构？

**答案：完全不共用**，证据如下：

1. **基类不同**：
   - `RecursiveLS(MLEModel)` → 状态空间世界
   - `RollingOLS` → 独立实现（第 441 行 `class RollingOLS(RollingWLS)`，而 `RollingWLS` 不继承任何回归模型类）

2. **核心算法完全不同**：
   - RecursiveLS 的核心是 `self.smooth()` → Kalman 滤波
   - RollingOLS 的核心是 `for` 循环中的 `xpx -= ...` 和 `xpx += ...`

3. **数据表示不同**：
   - RecursiveLS 需要构造 `"design"`、`"transition"` 等状态空间矩阵
   - RollingOLS 直接操作 `self._wx`、`self._wy` 等原始数据

4. **结果对象不同**：
   - `RecursiveLSResults(MLEResults)` → 状态空间结果
   - `RollingRegressionResults` → 独立结果类（第 463 行）

**设计意图**：
- RecursiveLS 旨在**利用成熟的状态空间框架**，获得递归残差、CUSUM 检验等高级功能
- RollingOLS 旨在**高效计算固定窗口回归**，代码简洁直接，无需状态空间的复杂性

---

## 五、总结与设计洞察

### 5.1 求解方法的层次选择

```
数值稳定性递增
    ↑
Cholesky → QR → 伪逆 (SVD)
    ↓
计算效率递增
```

statsmodels 默认选择伪逆，体现了"稳定性优先于效率"的统计软件设计哲学。

### 5.2 GLS/WLS 的优雅设计

通过 **Template Method + 白化** 模式：
- 基类 `RegressionModel.fit()` 完全不知道 GLS/WLS 的区别
- 所有差异被 `whiten()` 方法封装
- 正则化、预测等方法自动获得 GLS/WLS 支持

这是 **开闭原则 (Open/Closed Principle)** 的典范：对扩展开放（新增 WLS/GLS 只需覆写 whiten），对修改关闭（无需修改 fit 逻辑）。

### 5.3 IRLS 的关注点分离

M-估计量的设计体现了**策略模式 (Strategy Pattern)**：
- `RLM` 类管理迭代流程（上下文）
- `RobustNorm` 族封装具体的权重策略
- 新增 M-估计量只需新增一个 `RobustNorm` 子类

### 5.4 递归与滚动："重用"与"专用"的权衡

| 方法 | 设计选择 | 优点 | 缺点 |
|------|---------|------|------|
| **RecursiveLS** | 复用状态空间框架 | 功能丰富（Kalman 平滑、检验）、可扩展性强 | 代码复杂、依赖大型状态空间模块 |
| **RollingOLS** | 独立专用实现 | 代码简洁、易于理解、无额外依赖 | 功能相对局限 |

这体现了软件工程中的经典权衡：**重用带来复杂性，专用带来简洁性**。statsmodels 根据两种方法的不同使用场景做出了合理的不同选择。

### 5.5 关键文件索引

| 功能 | 文件路径 | 关键类/函数 |
|------|---------|------------|
| OLS/GLS/WLS 核心 | `statsmodels/regression/linear_model.py` | `OLS`, `WLS`, `GLS`, `RegressionModel.fit()` |
| 伪逆工具 | `statsmodels/tools/tools.py` | `pinv_extended` |
| M-估计量框架 | `statsmodels/robust/robust_linear_model.py` | `RLM`, `RLM.fit()` |
| M-估计量权重 | `statsmodels/robust/norms.py` | `HuberT`, `TukeyBiweight`, `RobustNorm.weights()` |
| 递推最小二乘 | `statsmodels/regression/recursive_ls.py` | `RecursiveLS` |
| 滚动回归 | `statsmodels/regression/rolling.py` | `RollingOLS`, `RollingWLS` |
