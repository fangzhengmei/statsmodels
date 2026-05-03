# Statsmodels GLM 家族与链接函数笛卡尔积设计分析

## 1. 概述

本文深入分析 statsmodels 中广义线性模型（GLM）的家族（Family）与链接函数（Link）的笛卡尔积设计模式。通过分析源码，揭示：

- **统一接口设计**：Family 和 Link 基类如何定义标准协议
- **差异化实现**：各家族和链接函数如何在统一接口下实现差异化逻辑
- **笛卡尔积组合机制**：如何支持任意 Family-Link 组合，同时提供安全约束
- **参数估计协作**：IRLS 算法中各模块如何跨层协作
- **预测机制协作**：从线性预测到置信区间的完整流程

---

## 2. 设计架构概览

### 2.1 核心模块定位

| 模块 | 文件路径 | 职责 |
|------|-----------|------|
| GLM 主类 | `statsmodels/genmod/generalized_linear_model.py` | 组合 Family 和 Link，协调参数估计与预测 |
| Family 基类与实现 | `statsmodels/genmod/families/family.py` | 定义指数族分布协议，实现各分布特性 |
| Link 基类与实现 | `statsmodels/genmod/families/links.py` | 定义链接函数协议，实现各链接函数数学计算 |
| Variance 函数 | `statsmodels/genmod/families/varfuncs.py` | 定义方差函数，描述 Var(Y) 与均值 μ 的关系 |

### 2.2 整体架构图

```
                    ┌─────────────────────────────────────────────────────┐
                    │                      GLM 主类                         │
                    │  (generalized_linear_model.py)                        │
                    ├─────────────────────────────────────────────────────┤
                    │  组合关系：                                            │
                    │  ┌──────────────┐         ┌──────────────────────┐  │
                    │  │   Family     │◄───────►│      Link            │  │
                    │  │ (指数族分布)  │  组合   │   (链接函数)          │  │
                    │  └──────────────┘         └──────────────────────┘  │
                    │         │                        │                    │
                    │         ▼                        ▼                    │
                    │  ┌──────────────┐         ┌──────────────────────┐  │
                    │  │  Variance    │         │  数学计算协议         │  │
                    │  │  方差函数     │         │  __call__, inverse,  │  │
                    │  │              │         │  deriv, deriv2, ...   │  │
                    │  └──────────────┘         └──────────────────────┘  │
                    └─────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │   IRLS 拟合  │  │   预测计算    │  │  结果分析    │
            │  参数估计    │  │  均值/方差   │  │  残差/诊断   │
            └──────────────┘  └──────────────┘  └──────────────┘
```

### 2.3 设计模式应用

**策略模式 (Strategy Pattern)**：
- `Family` 是策略接口，各具体家族（Gaussian, Poisson, Binomial 等）是具体策略
- `Link` 是策略接口，各具体链接函数（Logit, Log, Identity 等）是具体策略
- GLM 类作为上下文（Context），在运行时组合不同的 Family 和 Link 策略

**组合模式 (Composition Pattern)**：
- Family 内部组合了 Link 实例
- GLM 内部组合了 Family 实例
- 形成 `GLM → Family → Link` 的组合链

---

## 3. Family 模块深度分析

### 3.1 Family 基类接口定义

**文件位置**：`statsmodels/genmod/families/family.py:28`

`Family` 基类定义了指数族分布的统一协议：

```python
class Family:
    """
    The parent class for one-parameter exponential families.
    """
    # 类属性：各子类覆盖
    valid = [-np.inf, np.inf]  # 均值的有效范围
    links = []                  # 可用的链接函数类型列表
    
    def __init__(self, link, variance, check_link=True):
        self.link = link        # 持有 Link 实例
        self.variance = variance  # 持有 VarianceFunction 实例
```

### 3.2 核心方法协议

| 方法 | 签名 | 职责 | 调用 Link |
|------|------|------|-----------|
| `starting_mu(y)` | `(y: ndarray) -> ndarray` | IRLS 算法的初始 μ 值 | 否 |
| `weights(mu)` | `(mu: ndarray) -> ndarray` | 计算 IRLS 权重 | **是** (link.deriv) |
| `deviance(...)` | `(endog, mu, ...) -> float` | 计算偏差（用于收敛判断） | 否 |
| `loglike_obs(...)` | `(endog, mu, ...) -> ndarray` | 逐观测对数似然 | 否 |
| `loglike(...)` | `(endog, mu, ...) -> float` | 总对数似然 | 否 |
| `fitted(lin_pred)` | `(lin_pred: ndarray) -> ndarray` | 由线性预测计算均值 μ | **是** (link.inverse) |
| `predict(mu)` | `(mu: ndarray) -> ndarray` | 由均值 μ 计算线性预测 | **是** (link.__call__) |
| `resid_dev(...)` | `(endog, mu, ...) -> ndarray` | 偏差残差 | 否 |
| `resid_anscombe(...)` | `(endog, mu, ...) -> ndarray` | Anscombe 残差 | 否 |
| `get_distribution(...)` | `(mu, scale, ...) -> scipy.stats` | 获取 scipy 分布实例 | 否 |

### 3.3 IRLS 权重计算的协作

**关键实现** (`family.py:127-147`):

```python
def weights(self, mu):
    r"""
    Weights for IRLS steps
    
    Notes
    -----
    .. math::
       w = 1 / (g'(\mu)^2  * Var(\mu))
    """
    return 1.0 / (self.link.deriv(mu) ** 2 * self.variance(mu))
```

这是整个架构中最核心的协作点：
1. **`link.deriv(mu)`**：调用链接函数的一阶导数 g'(μ)
2. **`self.variance(mu)`**：调用方差函数 Var(μ)
3. **组合**：w = 1 / (g'(μ)² * Var(μ))

### 3.4 八个具体 Family 实现

| Family 类 | 默认 Link | 默认 Variance | 有效范围 | 特性 |
|-----------|-----------|---------------|----------|------|
| `Gaussian` | `Identity()` | `constant` (1) | (-∞, +∞) | 正态分布，OLS 特例 |
| `Poisson` | `Log()` | `mu` (μ) | [0, +∞) | 计数数据 |
| `Binomial` | `Logit()` | `binary` (μ(1-μ)) | [0, 1] | 二元/比例数据 |
| `Gamma` | `InversePower()` | `mu_squared` (μ²) | [0, +∞) | 正值右偏数据 |
| `InverseGaussian` | `InverseSquared()` | `mu_cubed` (μ³) | [0, +∞) | 逆高斯分布 |
| `NegativeBinomial` | `Log()` | `nbinom` (μ+αμ²) | [0, +∞) | 过离散计数数据 |
| `Tweedie` | `Log()` | 幂方差 | [0, +∞) | 复合 Poisson 等 |

### 3.5 差异化实现示例：Poisson vs Gaussian

**Poisson 的对数似然** (`family.py:452-482`):

```python
def loglike_obs(self, endog, mu, var_weights=1.0, scale=1.0):
    r"""
    ll_i = var_weights_i / scale * (endog_i * ln(mu_i) - mu_i - ln Γ(endog_i + 1))
    """
    return (
        var_weights / scale * (endog * np.log(mu) - mu - special.gammaln(endog + 1))
    )
```

**Gaussian 的对数似然** (`family.py:605-653`):

```python
def loglike_obs(self, endog, mu, var_weights=1.0, scale=1.0):
    r"""
    ll_i = -1/2 * (var_weights * (Y_i - mu_i)^2 / scale + log(2π * scale / var_weights))
    """
    ll_obs = -var_weights * (endog - mu) ** 2 / scale
    ll_obs += -np.log(scale / var_weights) - np.log(2 * np.pi)
    ll_obs /= 2
    return ll_obs
```

---

## 4. Link 模块深度分析

### 4.1 Link 基类接口定义

**文件位置**：`statsmodels/genmod/families/links.py:23`

```python
class Link:
    """
    A generic link function for one-parameter exponential family.
    """
    def __call__(self, p):          # g(p) = z: 均值 → 线性预测
        return NotImplementedError
    
    def inverse(self, z):            # g⁻¹(z) = p: 线性预测 → 均值
        return NotImplementedError
    
    def deriv(self, p):              # g'(p): 一阶导数
        return NotImplementedError
    
    def deriv2(self, p):             # g''(p): 二阶导数
        # 默认数值微分实现
        return _approx_fprime_cs_scalar(p, self.deriv)
    
    def inverse_deriv(self, z):      # d/dz [g⁻¹(z)]: 逆链接的一阶导数
        # 默认实现: 1 / g'(g⁻¹(z))
        return 1 / self.deriv(self.inverse(z))
    
    def inverse_deriv2(self, z):     # d²/dz² [g⁻¹(z)]: 逆链接的二阶导数
        # 默认实现: -g''(g⁻¹(z)) / g'(g⁻¹(z))³
        iz = self.inverse(z)
        return -self.deriv2(iz) / self.deriv(iz) ** 3
```

### 4.2 链接函数协议的数学意义

```
        均值 μ ∈ (定义域)              线性预测 η ∈ (-∞, +∞)
              │                                    ▲
              │                                    │
              │  g(μ) = η: __call__(μ)            │  g⁻¹(η) = μ: inverse(η)
              ├────────────────────────────────────┤
              │                                    │
              │  g'(μ): deriv(μ)                   │  d/dη g⁻¹(η): inverse_deriv(η)
              │  用于 IRLS 权重计算                 │  用于预测方差计算
              │                                    │
              │  g''(μ): deriv2(μ)                 │  d²/dη² g⁻¹(η): inverse_deriv2(η)
              │  用于观测 Hessian 计算              │  (可选，用于高级推断)
              ▼                                    ▼
```

### 4.3 十二个具体 Link 实现

| Link 类 | 数学定义 g(μ) | 逆函数 g⁻¹(η) | 主要适用 Family |
|---------|--------------|----------------|-----------------|
| `Logit` | log(μ/(1-μ)) | exp(η)/(1+exp(η)) | Binomial |
| `Probit` | Φ⁻¹(μ) | Φ(η) | Binomial |
| `Cauchy` | Cauchy.ppf(μ) | Cauchy.cdf(η) | Binomial |
| `Log` | log(μ) | exp(η) | Poisson, Gamma, 所有正值分布 |
| `LogC` | log(1-μ) | 1 - exp(η) | Binomial |
| `CLogLog` | log(-log(1-μ)) | 1 - exp(-exp(η)) | Binomial |
| `LogLog` | -log(-log(μ)) | exp(-exp(-η)) | Binomial |
| `Identity` | μ | η | Gaussian (OLS), 所有 |
| `Power(power)` | μ^power | η^(1/power) | 通用幂族 |
| `InversePower` | 1/μ (power=-1) | 1/η | Gamma |
| `Sqrt` | √μ (power=0.5) | η² | Poisson |
| `InverseSquared` | 1/μ² (power=-2) | 1/√η | InverseGaussian |
| `NegativeBinomial` | log(μ/(μ+1/α)) | 复杂形式 | NegativeBinomial |

### 4.4 差异化实现示例：Logit vs Log

**Logit 链接** (`links.py:132-260`):

```python
class Logit(Link):
    def __call__(self, p):
        """g(p) = log(p / (1 - p))"""
        p = self._clean(p)  # 裁剪到 (eps, 1-eps)
        return np.log(p / (1.0 - p))
    
    def inverse(self, z):
        """g⁻¹(z) = exp(z)/(1+exp(z))"""
        z = np.asarray(z)
        t = np.exp(-z)
        return 1.0 / (1.0 + t)  # 数值稳定实现
    
    def deriv(self, p):
        """g'(p) = 1 / (p * (1 - p))"""
        p = self._clean(p)
        return 1.0 / (p * (1 - p))
    
    def inverse_deriv(self, z):
        """d/dz [g⁻¹(z)] = exp(z)/(1+exp(z))² = μ(1-μ)"""
        t = np.exp(z)
        return t / (1 + t) ** 2
```

**Log 链接** (`links.py:481-592`):

```python
class Log(Link):
    def __call__(self, p):
        """g(p) = log(p)"""
        x = self._clean(p)  # 裁剪到 (eps, +∞)
        return np.log(x)
    
    def inverse(self, z):
        """g⁻¹(z) = exp(z)"""
        return np.exp(z)
    
    def deriv(self, p):
        """g'(p) = 1/p"""
        p = self._clean(p)
        return 1.0 / p
    
    def inverse_deriv(self, z):
        """d/dz [g⁻¹(z)] = exp(z)"""
        return np.exp(z)
```

### 4.5 CDFLink 类的设计：通用 CDF 链接

**文件位置**：`links.py:728-875`

```python
class CDFLink(Logit):
    """
    使用 scipy.stats 分布的 CDF 作为链接函数
    """
    def __init__(self, dbn=scipy.stats.norm):
        self.dbn = dbn  # dbn 是任意 scipy.stats 连续分布
    
    def __call__(self, p):
        """g(p) = dbn.ppf(p)  (分位数函数)"""
        p = self._clean(p)
        return self.dbn.ppf(p)
    
    def inverse(self, z):
        """g⁻¹(z) = dbn.cdf(z)  (累积分布函数)"""
        return self.dbn.cdf(z)
    
    def deriv(self, p):
        """g'(p) = 1 / dbn.pdf(dbn.ppf(p))"""
        p = self._clean(p)
        return 1.0 / self.dbn.pdf(self.dbn.ppf(p))
```

**子类 Probit** (`links.py:878-905`):

```python
class Probit(CDFLink):
    """Probit = 标准正态分布的 CDF 链接"""
    def __init__(self):
        super().__init__(dbn=scipy.stats.norm)
    
    # 覆盖优化逆链接二阶导数
    def inverse_deriv2(self, z):
        return -z * self.dbn.pdf(z)  # d²/dz² Φ(z) = -z * φ(z)
```

---

## 5. 笛卡尔积组合机制

### 5.1 组合约束的声明

每个 Family 通过类属性声明可用链接：

```python
class Poisson(Family):
    links = [L.Log, L.Identity, L.Sqrt]      # 技术上可用的链接
    variance = V.mu
    valid = [0, np.inf]
    safe_links = [L.Log]                     # 推荐的"安全"链接
```

**文档中的组合表** (`generalized_linear_model.py:227-239`):

```
============= ===== === ===== ====== ======= === ==== ====== ====== ====
Family        ident log logit probit cloglog pow opow nbinom loglog logc
============= ===== === ===== ====== ======= === ==== ====== ====== ====
Gaussian      x     x   x     x      x       x   x     x      x
inv Gaussian  x     x                        x
binomial      x     x   x     x      x       x   x           x      x
Poisson       x     x                        x
neg binomial  x     x                        x        x
gamma         x     x                        x
Tweedie       x     x                        x
============= ===== === ===== ====== ======= === ==== ====== ====== ====
```

### 5.2 链接验证机制

**GLM 初始化时的验证** (`generalized_linear_model.py:315-327`):

```python
def __init__(self, endog, exog, family=None, ...):
    # 检查链接是否在 safe_links 中
    if (family is not None) and not isinstance(
        family.link, tuple(family.safe_links)
    ):
        warnings.warn(
            f"The {type(family.link).__name__} link function "
            "does not respect the domain of the "
            f"{type(family).__name__} family.",
            DomainWarning,
            stacklevel=2,
        )
```

**Family 内部的链接设置** (`family.py:54-89`):

```python
def _setlink(self, link):
    self._link = link
    if self._check_link:
        # 验证是否为 Link 实例
        if not isinstance(link, L.Link):
            raise TypeError("The input should be a valid Link object.")
        # 验证是否在可用链接列表中
        if hasattr(self, "links"):
            validlink = max([isinstance(link, _) for _ in self.links])
            if not validlink:
                msg = "Invalid link for family, should be in %s. (got %s)"
                raise ValueError(msg % (repr(self.links), link))
```

### 5.3 组合流程图

```
用户调用: sm.GLM(endog, exog, family=sm.families.Binomial(link=sm.families.links.Probit()))
                              │
                              ▼
                    ┌─────────────────┐
                    │  Binomial.__init__│
                    │  - link = Probit() │
                    │  - variance = Binary() │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Family._setlink │
                    │ 检查: isinstance(Probit, Binomial.links)? │
                    │ Binomial.links = [Logit, Probit, Cauchy, Log, ...] │
                    │ ✓ Probit 在列表中 ✓ │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ GLM.__init__    │
                    │ 检查: isinstance(Probit, Binomial.safe_links)? │
                    │ Binomial.safe_links = [Logit, CDFLink] │
                    │ ✓ Probit 是 CDFLink 子类 ✓ │
                    │ 无警告，继续 │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────────────────────────────────────┐
                    │ 组合结果: GLM 持有 Family，Family 持有 Link     │
                    │  GLM.family = Binomial(link=Probit())          │
                    │  GLM.family.link = Probit()                      │
                    │  GLM.family.variance = Binary(n=1)              │
                    └─────────────────────────────────────────────────┘
```

---

## 6. 参数估计中的跨模块协作

### 6.1 IRLS 算法核心流程

**文件位置**：`generalized_linear_model.py:1420-1525`

IRLS（Iteratively Reweighted Least Squares）是 GLM 参数估计的核心算法，每轮迭代都涉及 Family、Link、Variance 的深度协作。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IRLS 迭代流程                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  初始化:                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ start_params = None 时:                                               │  │
│  │   mu = family.starting_mu(endog)    ← Family 提供初始值             │  │
│  │   lin_pred = family.predict(mu)     ← 调用 link(mu)                 │  │
│  │                                                                      │  │
│  │ 或 start_params 已给定时:                                            │  │
│  │   lin_pred = X @ start_params + offset_exposure                     │  │
│  │   mu = family.fitted(lin_pred)      ← 调用 link.inverse(lin_pred)  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ 迭代循环 (for iteration in range(maxiter)):                         │  │
│  │                                                                      │  │
│  │  Step 1: 计算 IRLS 权重                                             │  │
│  │  ─────────────────────                                              │  │
│  │  weights = iweights * n_trials * family.weights(mu)                │  │
│  │                                      │                               │  │
│  │              family.weights(mu) 展开为:                             │  │
│  │              1.0 / (link.deriv(mu) ** 2 * variance(mu))           │  │
│  │                      │                   │                          │  │
│  │                      ▼                   ▼                          │  │
│  │              Link.deriv(mu)     VarianceFunction(mu)              │  │
│  │              g'(μ)                 Var(μ)                           │  │
│  │                                                                      │  │
│  │ Step 2: 构造工作因变量 (Working Response)                           │  │
│  │ ───────────────────────────────────────                             │  │
│  │  wlsendog = lin_pred + link.deriv(mu) * (endog - mu)              │  │
│  │                      │                                              │  │
│  │                      ▼                                              │  │
│  │              泰勒展开近似: z ≈ η + (y - μ) * g'(μ)                 │  │
│  │                                                                      │  │
│  │ Step 3: WLS 回归更新参数                                            │  │
│  │ ──────────────────────────────                                      │  │
│  │  wls_mod = _MinimalWLS(wlsendog, exog, weights)                   │  │
│  │  wls_results = wls_mod.fit()                                       │  │
│  │                                                                      │  │
│  │ Step 4: 更新线性预测和均值                                          │  │
│  │ ────────────────────────────────                                    │  │
│  │  lin_pred = X @ wls_results.params + offset_exposure               │  │
│  │  mu = family.fitted(lin_pred)    ← 调用 link.inverse(lin_pred)    │  │
│  │                                                                      │  │
│  │ Step 5: 收敛检查                                                    │  │
│  │ ─────────────────────                                               │  │
│  │  dev = family.deviance(endog, mu, ...)                             │  │
│  │  converged = allclose(dev_prev, dev_current)                       │  │
│  │                                                                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 关键代码解析

**Step 1: 权重计算** (`generalized_linear_model.py:1475`):

```python
self.weights = self.iweights * self.n_trials * self.family.weights(mu)
```

`family.weights(mu)` 的实现 (`family.py:147`):

```python
return 1.0 / (self.link.deriv(mu) ** 2 * self.variance(mu))
```

**协作链**:
```
GLM._fit_irls()
    └──► family.weights(mu)
              ├──► link.deriv(mu)        # Link 提供 g'(μ)
              └──► variance(mu)           # VarianceFunction 提供 Var(μ)
```

**Step 2: 工作因变量** (`generalized_linear_model.py:1476-1480`):

```python
wlsendog = (
    lin_pred
    + self.family.link.deriv(mu) * (self.endog - mu)  # 直接调用 link.deriv
    - self._offset_exposure
)
```

**Step 4: 更新均值** (`generalized_linear_model.py:1487`):

```python
mu = self.family.fitted(lin_pred)
```

`family.fitted()` 的实现 (`family.py:230-247`):

```python
def fitted(self, lin_pred):
    """
    Fitted values based on linear predictors lin_pred.
    
    Returns
    -------
    mu : ndarray
        The mean response variables given by the inverse of the link function.
    """
    fits = self.link.inverse(lin_pred)  # 调用 link.inverse
    return fits
```

### 6.3 梯度优化方法的协作

当使用 `method='newton'` 等梯度方法时，协作模式略有不同但仍依赖相同的接口：

**分数函数计算** (`generalized_linear_model.py:545-577`):

```python
def score_factor(self, params, scale=None):
    mu = self.predict(params)  # 调用 family.fitted(lin_pred)
    
    # 协作: link.deriv 和 family.variance
    score_factor = (self.endog - mu) / self.family.link.deriv(mu)
    score_factor /= self.family.variance(mu)
    score_factor *= self.iweights * self.n_trials
    
    return score_factor
```

**Hessian 因子计算** (`generalized_linear_model.py:579-634`):

```python
def hessian_factor(self, params, scale=None, observed=True):
    mu = self.predict(params)
    
    # 期望信息矩阵因子
    eim_factor = 1 / (self.family.link.deriv(mu) ** 2 * self.family.variance(mu))
    eim_factor *= self.iweights * self.n_trials
    
    if not observed:
        return eim_factor  # EIM
    
    # 观测信息矩阵需要 link.deriv2 和 variance.deriv
    tmp = (self.family.variance(mu) * self.family.link.deriv2(mu)
           + self.family.variance.deriv(mu) * self.family.link.deriv(mu))
    
    oim_factor = eim_factor * (1 + score_factor * tmp / (iweights * n_trials))
    return oim_factor
```

---

## 7. 预测机制中的跨模块协作

### 7.1 预测层级架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         预测调用层级                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  第一层: 用户接口 (GLMResults)                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  results.predict(exog)          # 简单预测                        │   │
│  │  results.get_prediction(exog)   # 带推断的预测（置信区间等）     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│              │                                                          │
│              ▼                                                          │
│  第二层: 模型类 (GLM)                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  GLM.predict(params, exog, which='mean')                        │   │
│  │      │                                                           │   │
│  │      ├──► which='mean':   family.fitted(lin_pred)               │   │
│  │      ├──► which='linear': 返回 lin_pred                         │   │
│  │      └──► which='var_unscaled': family.variance(mean)           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│              │                                                          │
│              ▼                                                          │
│  第三层: 策略实现 (Family + Link)                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  family.fitted(lin_pred) = link.inverse(lin_pred)               │   │
│  │  family.variance(mu)   = VarianceFunction(mu)                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 GLM.predict 核心实现

**文件位置**：`generalized_linear_model.py:1015-1101`

```python
def predict(
    self, params, exog=None, exposure=None, offset=None, 
    which="mean", linear=None
):
    """
    Return predicted values for a design matrix
    
    Parameters
    ----------
    which : 'mean', 'linear', 'var_unscaled'
        - 'mean': E(y|x) = g⁻¹(Xβ)
        - 'linear': Xβ (线性预测)
        - 'var_unscaled': Var(y) = variance(g⁻¹(Xβ)) (不含 scale)
    """
    # 计算线性预测: η = Xβ + offset + exposure
    linpred = np.dot(exog, params) + offset + exposure
    
    if which == "mean":
        # μ = g⁻¹(η)
        return self.family.fitted(linpred)  # 调用 link.inverse
    elif which == "linear":
        return linpred
    elif which == "var_unscaled":
        # Var(y) = variance(μ)
        mean = self.family.fitted(linpred)
        var_ = self.family.variance(mean)  # 调用方差函数
        return var_
    else:
        raise ValueError(f'The which value "{which}" is not recognized')
```

### 7.3 预测推断的协作：get_prediction

**文件位置**：`generalized_linear_model.py:2252-2394` 和 `_prediction_inference.py:436-514`

```python
def get_prediction_glm(
    self, exog=None, transform=True, row_labels=None, 
    linpred=None, link=None, pred_kwds=None
):
    """
    计算带推断的预测结果（方差、置信区间等）
    """
    # Step 1: 计算预测均值
    predicted_mean = self.model.predict(self.params, exog, **pred_kwds)
    #                                   │
    #                                   └──► family.fitted(lin_pred)
    
    # Step 2: 计算预测方差 (Delta 方法)
    covb = self.cov_params()
    
    # 关键协作: 使用逆链接的导数
    link_deriv = self.model.family.link.inverse_deriv(linpred.predicted_mean)
    #                               │
    #                               └──► d/dη g⁻¹(η)
    
    # Var(μ̂) = [d/dη g⁻¹(η)]² * Var(Xβ̂)
    var_pred_mean = link_deriv**2 * (exog * np.dot(covb, exog.T).T).sum(1)
    
    # Step 3: 构造预测结果对象
    return PredictionResultsMean(
        predicted_mean,
        var_pred_mean,
        var_resid=self.scale,
        df=self.df_resid,
        link=link,  # 传入 link 用于置信区间的逆变换
    )
```

### 7.4 Delta 方法的数学原理

预测均值的方差通过 Delta 方法计算：

```
设: η̂ = Xβ̂, μ̂ = g⁻¹(η̂)

则: Var(μ̂) ≈ [d/dη g⁻¹(η̂)]² * Var(η̂)
           = [inverse_deriv(η̂)]² * X * Cov(β̂) * X'

其中:
- inverse_deriv(η) = d/dη g⁻¹(η)  (由 Link 类提供)
- Cov(β̂) 由参数估计得到
```

**代码实现** (`_prediction_inference.py:495-496`):

```python
link_deriv = self.model.family.link.inverse_deriv(linpred.predicted_mean)
var_pred_mean = link_deriv**2 * (exog * np.dot(covb, exog.T).T).sum(1)
```

---

## 8. 残差计算中的协作

### 8.1 残差类型与协作

GLMResults 提供多种残差类型，每种都涉及不同程度的模块协作：

| 残差类型 | 计算方式 | 协作模块 |
|----------|----------|----------|
| `resid_response` | y - μ | 无（直接计算） |
| `resid_pearson` | (y - μ) / √Var(μ) | **VarianceFunction** |
| `resid_working` | (y - μ) * g'(μ) | **Link.deriv** |
| `resid_deviance` | 分布特定 | **Family._resid_dev** |
| `resid_anscombe` | 分布特定 | **Family.resid_anscombe** |

### 8.2 关键实现

**Pearson 残差** (`generalized_linear_model.py:1894-1907`):

```python
@cached_data
def resid_pearson(self):
    """
    Pearson residuals: (endog - mu) / sqrt(variance(mu))
    """
    return (
        np.sqrt(self._n_trials)
        * (self._endog - self.mu)
        * np.sqrt(self._var_weights)
        / np.sqrt(self.family.variance(self.mu))  # 调用方差函数
    )
```

**Working 残差** (`generalized_linear_model.py:1909-1919`):

```python
@cached_data
def resid_working(self):
    """
    Working residuals: resid_response * link.deriv(mu)
    """
    val = self.resid_response * self.family.link.deriv(self.mu)  # 调用 link.deriv
    val *= self._n_trials
    return val
```

**Deviance 残差** (`generalized_linear_model.py:1953-1962`):

```python
@cached_data
def resid_deviance(self):
    dev = self.family.resid_dev(
        self._endog, self.fittedvalues, var_weights=self._var_weights, scale=1.0
    )
    return dev
```

`family.resid_dev` 调用 `_resid_dev`，各 Family 有不同实现：

```python
# Poisson 的 _resid_dev
def _resid_dev(self, endog, mu):
    endog_mu = self._clean(endog / mu)
    resid_dev = endog * np.log(endog_mu) - (endog - mu)
    return 2 * resid_dev

# Gaussian 的 _resid_dev
def _resid_dev(self, endog, mu):
    return (endog - mu) ** 2

# Binomial 的 _resid_dev
def _resid_dev(self, endog, mu):
    endog_mu = self._clean(endog / (mu + 1e-20))
    n_endog_mu = self._clean((1.0 - endog) / (1.0 - mu + 1e-20))
    resid_dev = endog * np.log(endog_mu) + (1 - endog) * np.log(n_endog_mu)
    return 2 * self.n * resid_dev
```

---

## 9. 设计模式总结

### 9.1 策略模式的应用

| 组件 | 策略接口 | 具体策略 | 上下文 |
|------|----------|----------|--------|
| 分布家族 | `Family` | Gaussian, Poisson, Binomial, Gamma, ... | GLM |
| 链接函数 | `Link` | Logit, Log, Identity, Probit, ... | Family |
| 方差函数 | `VarianceFunction` | constant, mu, mu_squared, binary, ... | Family |

### 9.2 接口一致性

所有策略都严格遵守统一接口：

**Family 必须实现**:
- `starting_mu(y)`: 初始值
- `_resid_dev(endog, mu)`: 偏差贡献
- `loglike_obs(endog, mu, ...)`: 对数似然
- `resid_anscombe(...)`: Anscombe 残差
- `get_distribution(...)`: scipy 分布

**Link 必须实现**:
- `__call__(p)`: g(p)
- `inverse(z)`: g⁻¹(z)
- `deriv(p)`: g'(p)

（`deriv2`, `inverse_deriv`, `inverse_deriv2` 有默认数值实现，可覆盖优化）

**VarianceFunction 必须实现**:
- `__call__(mu)`: Var(μ)
- `deriv(mu)`: d/dμ Var(μ)

### 9.3 组合灵活性与约束

**灵活性**:
- 任意 Family 可以与任意 Link 组合（技术上）
- 组合在运行时动态确定
- 用户可以自定义 Link 或 Family

**约束机制**:
- `Family.links`: 技术上支持的链接列表
- `Family.safe_links`: 推荐的安全链接
- GLM 初始化时检查并发出警告
- `_check_link` 参数可禁用检查

---

## 10. 关键代码位置索引

| 功能 | 文件位置 | 行号 |
|------|----------|------|
| GLM 主类定义 | `genmod/generalized_linear_model.py` | 85 |
| GLM 初始化与链接验证 | `genmod/generalized_linear_model.py` | 299-409 |
| IRLS 拟合算法 | `genmod/generalized_linear_model.py` | 1420-1525 |
| GLM 预测方法 | `genmod/generalized_linear_model.py` | 1015-1101 |
| GLMResults 类 | `genmod/generalized_linear_model.py` | 1767 |
| Family 基类 | `genmod/families/family.py` | 28 |
| Family.weights 核心协作 | `genmod/families/family.py` | 127-147 |
| Gaussian 家族 | `genmod/families/family.py` | 543 |
| Poisson 家族 | `genmod/families/family.py` | 384 |
| Binomial 家族 | `genmod/families/family.py` | 898 |
| Gamma 家族 | `genmod/families/family.py` | 713 |
| Link 基类 | `genmod/families/links.py` | 23 |
| Logit 链接 | `genmod/families/links.py` | 132 |
| Log 链接 | `genmod/families/links.py` | 481 |
| CDFLink 基类 | `genmod/families/links.py` | 728 |
| Probit 链接 | `genmod/families/links.py` | 878 |
| VarianceFunction 基类 | `genmod/families/varfuncs.py` | 9 |
| Power 方差族 | `genmod/families/varfuncs.py` | 63 |
| Binomial 方差 | `genmod/families/varfuncs.py` | 146 |
| NegativeBinomial 方差 | `genmod/families/varfuncs.py` | 217 |
| GLM 预测推断 | `base/_prediction_inference.py` | 436 |

---

## 11. 结论

statsmodels 的 GLM 设计是**策略模式**与**组合模式**的经典应用：

### 11.1 设计优点

1. **高度可扩展**: 新增 Family 或 Link 只需继承基类并实现接口
2. **运行时灵活**: 组合在运行时确定，支持笛卡尔积
3. **类型安全**: 通过 `isinstance` 检查和 `safe_links` 约束提供安全保障
4. **数值稳定性**: 各 Link 子类提供优化的逆链接和导数实现（而非依赖默认数值微分）

### 11.2 协作模式

整个架构通过**统一接口**实现松耦合协作：

```
                    ┌─────────────────────────────────────────┐
                    │              GLM (上下文)               │
                    │  协调者：知道何时调用什么               │
                    └──────────────────┬──────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
    ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
    │    Family       │◄────►│     Link        │◄────►│VarianceFunction │
    │  (分布策略)      │      │  (链接策略)      │      │  (方差策略)      │
    ├─────────────────┤      ├─────────────────┤      ├─────────────────┤
    │ - starting_mu   │      │ - __call__      │      │ - __call__      │
    │ - weights       │──────►│ - deriv         │──────►│ - deriv         │
    │ - deviance      │      │ - deriv2        │      └─────────────────┘
    │ - loglike       │      │ - inverse       │
    │ - fitted        │──────►│ - inverse_deriv │
    │ - predict       │      │ - inverse_deriv2│
    └─────────────────┘      └─────────────────┘
```

### 11.3 关键协作点总结

| 场景 | 协作链 | 关键代码 |
|------|--------|----------|
| IRLS 权重 | GLM → Family.weights → Link.deriv + Variance | `1 / (g'(μ)² * Var(μ))` |
| 工作因变量 | GLM → Link.deriv | `η + (y-μ) * g'(μ)` |
| μ 更新 | GLM → Family.fitted → Link.inverse | `g⁻¹(Xβ)` |
| 收敛检查 | GLM → Family.deviance | 分布特定 |
| 预测均值 | GLM.predict → Family.fitted → Link.inverse | `g⁻¹(Xβ)` |
| 预测方差 | get_prediction → Link.inverse_deriv | `[d/dη g⁻¹(η)]² * Var(Xβ)` |
| Pearson 残差 | GLMResults → Family.variance | `(y-μ) / √Var(μ)` |
| Working 残差 | GLMResults → Link.deriv | `(y-μ) * g'(μ)` |

这种设计使得 statsmodels 的 GLM 既灵活又严谨，完美体现了**面向对象设计原则**在统计软件中的应用。
