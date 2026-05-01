# Statsmodels 核心模型基类体系技术分析报告

## 目录

1. [概述](#1-概述)
2. [模型继承层级与职责分离](#2-模型继承层级与职责分离)
3. [拟合流程：模板方法模式的应用](#3-拟合流程模板方法模式的应用)
4. [结果类：按需计算的缓存属性设计](#4-结果类按需计算的缓存属性设计)
5. [关键设计模式总结](#5-关键设计模式总结)
6. [参考文献](#6-参考文献)

---

## 1. 概述

Statsmodels 是 Python 中功能强大的统计建模库，其核心设计体现了**面向对象的分层抽象思想**和**软件设计模式**的精妙应用。本报告深入分析其三个核心设计维度：

1. **继承层级设计**：从通用抽象到具体模型的职责分离
2. **拟合骨架方法**：模板方法模式实现可复用的优化流程
3. **结果类缓存机制**：描述符模式实现按需计算与内存效率

---

## 2. 模型继承层级与职责分离

### 2.1 核心继承图谱

```
┌─────────────────────────────────────────────────────────────────────┐
│                              Model (抽象基类)                          │
│  职责：数据处理、公式接口(from_formula)、基础属性管理                  │
│  核心方法：__init__, _handle_data, from_formula, fit, predict       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         LikelihoodModel (似然模型框架)                │
│  职责：最大似然估计框架、通用fit()骨架、梯度/海森抽象                 │
│  核心方法：loglike, score, hessian, information, fit()              │
└─────────────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  RegressionModel │  │       GLM        │  │  DiscreteModel   │
│   (线性回归族)    │  │ (广义线性模型)   │  │  (离散选择模型)   │
└──────────────────┘  └──────────────────┘  └──────────────────┘
           │                                       │
           ▼                                       ▼
   ┌──────────────┐                       ┌──────────────┐
   │     GLS      │                       │ BinaryModel  │  CountModel
   │ (广义最小二乘)│                       │ (二元模型)    │ (计数模型)
   └──────────────┘                       └──────────────┘
           │                                       │
           ▼                                       ▼
   ┌──────────────┐                       ┌──────────────┐
   │     WLS      │                       │   Logit      │   Poisson
   │ (加权最小二乘)│                       │   Probit     │   NegBin
   └──────────────┘                       └──────────────┘
           │
           ▼
   ┌──────────────┐
   │     OLS      │
   │(普通最小二乘) │
   └──────────────┘
```

### 2.2 各层级职责详解

#### 2.2.1 Model 类：最底层的抽象

**位置**: `statsmodels/base/model.py:65`

**核心职责**：

| 职责 | 实现方式 | 关键代码位置 |
|------|---------|-------------|
| **数据处理** | `_handle_data()` 统一处理数据 | model.py:143 |
| **缺失值处理** | 支持 'none', 'drop', 'raise' 策略 | model.py:49-53 |
| **公式接口** | `from_formula()` 类方法实现 R 式公式 | model.py:157 |
| **属性管理** | `_data_attr` 记录数据相关属性 | model.py:107 |
| **预测抽象** | `predict()` 作为占位方法等待子类实现 | model.py:270 |

**关键设计点**：

```python
# model.py:100-115
def __init__(self, endog, exog=None, **kwargs):
    missing = kwargs.pop("missing", "none")
    hasconst = kwargs.pop("hasconst", None)
    # 统一数据处理入口
    self.data = self._handle_data(endog, exog, missing, hasconst, **kwargs)
    self.k_constant = self.data.k_constant
    self.exog = self.data.exog
    self.endog = self.data.endog
    # ...
```

**从公式创建模型**：

```python
# model.py:157-248
@classmethod
def from_formula(cls, formula, data, subset=None, drop_cols=None, *args, **kwargs):
    """
    Create a Model from a formula and dataframe.
    
    示例:
    >>> model = sm.OLS.from_formula('y ~ x1 + x2', data=df)
    """
    # 使用 FormulaManager 处理公式
    mgr = FormulaManager()
    # ... 公式解析和矩阵构建
    return mod
```

#### 2.2.2 LikelihoodModel 类：似然估计框架

**位置**: `statsmodels/base/model.py:279`

**核心职责**：定义基于最大似然估计的模型框架，提供通用的拟合骨架。

**似然相关的抽象方法**：

```python
# model.py:300-360
class LikelihoodModel(Model):
    def loglike(self, params):
        """对数似然函数 - 子类必须实现"""
        raise NotImplementedError
    
    def score(self, params):
        """得分函数 (梯度) - 对数似然的一阶导数"""
        raise NotImplementedError
    
    def hessian(self, params):
        """海森矩阵 - 对数似然的二阶导数"""
        raise NotImplementedError
    
    def information(self, params):
        """Fisher信息矩阵 - 负海森矩阵"""
        raise NotImplementedError
```

**为什么这些方法是抽象的？**

不同模型有完全不同的似然形式：

| 模型类型 | 对数似然函数特点 |
|---------|-----------------|
| **OLS** | 正态分布假设，RSS 最小化等价于 MLE |
| **Logit** | 伯努利分布，对数似然为对数几率和 |
| **Poisson** | 泊松分布，对数似然涉及指数项 |
| **GLM** | 通过联结函数和指数族分布统一 |

#### 2.2.3 RegressionModel 类：线性回归族

**位置**: `statsmodels/regression/linear_model.py:213`

**继承链**: `Model → LikelihoodModel → RegressionModel → GLS → WLS → OLS`

**特殊设计**：

RegressionModel 虽然继承自 LikelihoodModel，但**完全重写了 `fit()` 方法**，不使用似然优化，而是直接使用最小二乘求解：

```python
# linear_model.py:284-390
def fit(self, method="pinv", cov_type="nonrobust", ...):
    """
    线性回归的 fit 方法不使用似然优化
    而是直接通过矩阵运算求解最小二乘
    """
    if method == "pinv":
        # 使用 Moore-Penrose 伪逆求解
        self.pinv_wexog, singular_values = pinv_extended(self.wexog)
        self.normalized_cov_params = np.dot(
            self.pinv_wexog, np.transpose(self.pinv_wexog)
        )
        # β = (X'X)⁻¹X'y
        beta = np.dot(self.pinv_wexog, self.wendog)
    
    elif method == "qr":
        # 使用 QR 分解求解
        Q, R = np.linalg.qr(self.wexog)
        # Rβ = Q'y
        effects = np.dot(Q.T, self.wendog)
        beta = np.linalg.solve(R, effects)
    # ...
```

**白化机制 (Whitening)**：

```python
# linear_model.py:273-282
def whiten(self, x):
    """
    白化方法 - 必须由子类实现
    用于处理异方差和序列相关
    """
    raise NotImplementedError("Subclasses must implement.")
```

| 模型 | `whiten()` 实现 | 用途 |
|------|----------------|------|
| **OLS** | 直接返回 x (不做变换) | 球形误差假定 |
| **WLS** | 乘以 `1/sqrt(weights)` | 处理异方差 |
| **GLS** | Cholesky 分解变换 | 处理一般协方差结构 |

#### 2.2.4 GLM 类：广义线性模型

**位置**: `statsmodels/genmod/generalized_linear_model.py:85`

**核心特点**：

1. **指数族分布框架**：通过 `family` 参数指定分布族
2. **联结函数**：通过 `family.link` 指定联结函数
3. **双重拟合方式**：支持 IRLS 和梯度优化

```python
# generalized_linear_model.py:85-295
class GLM(base.LikelihoodModel):
    def __init__(self, endog, exog, family=None, offset=None, 
                 exposure=None, freq_weights=None, var_weights=None, ...):
        # family: 分布族 (Gaussian, Binomial, Poisson, Gamma, Tweedie)
        # offset: 固定偏移项
        # exposure: 暴露时间 (仅对数联结有效)
        # freq_weights: 频率权重
        # var_weights: 方差权重
        # ...
```

**似然计算**：

```python
# generalized_linear_model.py:487-499
def loglike(self, params, scale=None):
    """
    GLM 的对数似然通过分布族和联结函数间接计算
    """
    # 1. 计算线性预测器: η = Xβ + offset + log(exposure)
    lin_pred = np.dot(self.exog, params) + self._offset_exposure
    
    # 2. 通过联结函数反变换得到均值: μ = g⁻¹(η)
    expval = self.family.link.inverse(lin_pred)
    
    # 3. 调用分布族的 loglike 方法
    return self.family.loglike(
        self.endog, expval, self.var_weights, self.freq_weights, scale
    )
```

#### 2.2.5 DiscreteModel 类：离散选择模型

**位置**: `statsmodels/discrete/discrete_model.py:185`

**子类体系**：

```
DiscreteModel
├── BinaryModel          # 二元选择
│   ├── Logit            # 逻辑回归
│   └── Probit           # 概率单位回归
├── CountModel           # 计数模型
│   ├── Poisson          # 泊松回归
│   └── NegativeBinomial # 负二项回归
└── MultinomialModel     # 多分类模型
    └── MNLogit          # 多分类逻辑回归
```

**完美分离检测**：

```python
# discrete_model.py:227-240
def _check_perfect_pred(self, params, *args):
    """
    检测完美分离 (Perfect Separation)
    当某个预测变量能完美预测因变量时发生
    """
    endog = self.endog
    fittedvalues = self.predict(params)
    if np.allclose(fittedvalues - endog, 0):
        msg = (
            "Perfect separation or prediction detected, "
            "parameter may not be identified"
        )
        warnings.warn(msg, PerfectSeparationWarning, stacklevel=2)
```

**在 fit() 中自动注入**：

```python
# discrete_model.py:242-274
@Appender(base.LikelihoodModel.fit.__doc__)
def fit(self, start_params=None, method="newton", ...):
    # 自动添加完美分离检测回调
    if callback is None:
        callback = self._check_perfect_pred
    # 调用父类的 fit
    mlefit = super().fit(..., callback=callback, ...)
    return mlefit
```

### 2.3 继承层级设计的优点

| 设计原则 | 体现方式 |
|---------|---------|
| **单一职责** | 每一层只负责特定抽象级别 |
| **开放封闭** | 新增模型只需继承并实现抽象方法 |
| **里氏替换** | 任何 LikelihoodModel 子类都可调用 fit() |
| **依赖倒置** | 高层依赖抽象 (LikelihoodModel)，而非具体实现 |
| **接口分离** | 各层只暴露必要接口 |

---

## 3. 拟合流程：模板方法模式的应用

### 3.1 模板方法模式概述

**模板方法模式**定义了一个算法的骨架，将一些步骤延迟到子类中实现。Statsmodels 的 `LikelihoodModel.fit()` 是这一模式的经典应用。

### 3.2 LikelihoodModel.fit() 通用骨架

**位置**: `statsmodels/base/model.py:362-651`

**完整流程图**：

```
┌──────────────────────────────────────────────────────────────────────┐
│                    LikelihoodModel.fit() 执行流程                      │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step 1: 参数初始化                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ if start_params is None:                                          │ │
│  │     if hasattr(self, "start_params"):                             │ │
│  │         start_params = self.start_params  # 子类自定义初始值       │ │
│  │     else:                                                          │ │
│  │         start_params = [0.0] * self.exog.shape[1]  # 零向量默认   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step 2: 求解器选择                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ method 参数决定优化算法:                                           │ │
│  │                                                                   │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │ │
│  │ │ 'newton' │  │  'bfgs'  │  │   'nm'   │  │  'basinhopping'│  │ │
│  │ │ Newton-  │  │ Quasi-   │  │ Nelder-  │  │  全局优化     │   │ │
│  │ │ Raphson  │  │ Newton   │  │ Mead     │  │               │   │ │
│  │ └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │ │
│  │                                                                   │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │ │
│  │ │ 'lbfgs'  │  │  'cg'    │  │  'ncg'   │  │  'minimize'   │   │ │
│  │ │ Limited  │  │ Conj-    │  │ Newton-  │  │  scipy通用    │   │ │
│  │ │ memory   │  │ ugate    │  │ Conj-    │  │  包装器       │   │ │
│  │ │ BFGS     │  │ Gradient │  │ ugate    │  │               │   │ │
│  │ └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step 3: 目标函数构造                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ # 最大化对数似然 = 最小化负对数似然 (平均后)                       │ │
│  │ def f(params, *args):                                             │ │
│  │     return -self.loglike(params, *args) / nobs                   │ │
│  │                                                                   │ │
│  │ # 根据方法调整 score 和 hessian 的符号                            │ │
│  │ if method == "newton":                                            │ │
│  │     def score(params, *args):                                     │ │
│  │         return self.score(params, *args) / nobs    # 正梯度      │ │
│  │     def hess(params, *args):                                      │ │
│  │         return self.hessian(params, *args) / nobs                │ │
│  │ else:                                                              │ │
│  │     def score(params, *args):                                     │ │
│  │         return -self.score(params, *args) / nobs   # 负梯度      │ │
│  │     def hess(params, *args):                                      │ │
│  │         return -self.hessian(params, *args) / nobs               │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step 4: 执行优化                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ optimizer = Optimizer()                                           │ │
│  │ xopt, retvals, optim_settings = optimizer._fit(                   │ │
│  │     f,                            # 目标函数                       │ │
│  │     score,                        # 梯度函数                       │ │
│  │     start_params,                 # 初始参数                       │ │
│  │     fargs,                        # 额外参数                       │ │
│  │     kwargs,                       # 求解器参数                     │ │
│  │     hessian=hess,                 # 海森函数                       │ │
│  │     method=method,                # 求解器选择                     │ │
│  │     disp=disp,                    # 是否显示信息                   │ │
│  │     maxiter=maxiter,              # 最大迭代次数                   │ │
│  │     callback=callback,            # 迭代回调                       │ │
│  │     retall=retall,                # 是否记录每步解                 │ │
│  │     full_output=full_output,      # 是否完整输出                   │ │
│  │ )                                                                 │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step 5: 收敛性检查                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ if warn_convergence and not retvals["converged"]:                │ │
│  │     warnings.warn(                                                 │ │
│  │         "Maximum Likelihood optimization failed to converge.",    │ │
│  │         ConvergenceWarning                                         │ │
│  │     )                                                              │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step 6: 协方差矩阵计算                                                │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ # 情况1: 自定义协方差函数                                          │ │
│  │ if cov_params_func:                                                │ │
│  │     Hinv = cov_params_func(self, xopt, retvals)                   │ │
│  │                                                                   │ │
│  │ # 情况2: Newton法已计算海森                                        │ │
│  │ elif method == "newton" and full_output:                          │ │
│  │     Hinv = np.linalg.inv(-retvals["Hessian"]) / nobs             │ │
│  │                                                                   │ │
│  │ # 情况3: 手动计算海森逆                                           │ │
│  │ elif not skip_hessian:                                             │ │
│  │     H = -1 * self.hessian(xopt)  # Fisher信息 = -Hessian         │ │
│  │     # 检查正定性                                                   │ │
│  │     eigvals, eigvecs = np.linalg.eigh(H)                          │ │
│  │     if np.min(eigvals) > 0:  # 正定                               │ │
│  │         Hinv = eigvecs.dot(np.diag(1.0 / eigvals)).dot(eigvecs.T)│ │
│  │     else:  # 奇异                                                   │ │
│  │         warnings.warn("Inverting hessian failed...")              │ │
│  │         Hinv = None                                                │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step 7: 结果封装                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ mlefit = LikelihoodModelResults(self, xopt, Hinv, scale=1.0)   │ │
│  │                                                                   │ │
│  │ # 附加求解器信息                                                  │ │
│  │ mlefit.mle_retvals = retvals      # 求解器返回值                 │ │
│  │ mlefit.mle_settings = optim_settings  # 求解器设置               │ │
│  │                                                                   │ │
│  │ return mlefit                                                    │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.3 GLM 的 IRLS 拟合

**位置**: `statsmodels/genmod/generalized_linear_model.py:1420-1500`

GLM 默认使用 **迭代重加权最小二乘 (IRLS)** 而非梯度优化：

```python
# generalized_linear_model.py:1178-1290
def fit(self, start_params=None, method="IRLS", ...):
    """
    GLM 支持两种拟合方式:
    - 'IRLS': 迭代重加权最小二乘 (默认)
    - 其他: 梯度优化 (调用 LikelihoodModel.fit)
    """
    if method.lower() == "irls":
        return self._fit_irls(...)
    else:
        return self._fit_gradient(...)
```

**IRLS 算法核心流程**：

```python
# generalized_linear_model.py:1420-1499
def _fit_irls(self, start_params=None, maxiter=100, tol=1e-8, ...):
    """
    迭代重加权最小二乘 (Iteratively Reweighted Least Squares)
    
    核心思想: 将 GLM 转化为一系列加权最小二乘问题迭代求解
    """
    
    # 初始化
    if start_params is None:
        start_params = np.zeros(self.exog.shape[1])
        mu = self.family.starting_mu(self.endog)  # 初始均值
        lin_pred = self.family.predict(mu)         # 初始线性预测器
    else:
        lin_pred = np.dot(wlsexog, start_params) + self._offset_exposure
        mu = self.family.fitted(lin_pred)
    
    # 迭代历史记录
    history = dict(params=[np.inf, start_params], deviance=[np.inf, dev])
    converged = False
    
    # 主循环
    for iteration in range(maxiter):
        # Step 1: 计算工作权重 (基于当前均值)
        # w = 方差权重 * 频率权重 * 分布权重
        self.weights = self.iweights * self.n_trials * self.family.weights(mu)
        
        # Step 2: 计算工作因变量 (Z 变量)
        # z = η + g'(μ)(y - μ)
        wlsendog = (
            lin_pred
            + self.family.link.deriv(mu) * (self.endog - mu)
            - self._offset_exposure
        )
        
        # Step 3: 加权最小二乘求解
        wls_mod = reg_tools._MinimalWLS(wlsendog, wlsexog, self.weights)
        wls_results = wls_mod.fit(method=wls_method)
        
        # Step 4: 更新线性预测器和均值
        lin_pred = np.dot(self.exog, wls_results.params) + self._offset_exposure
        mu = self.family.fitted(lin_pred)  # μ = g⁻¹(η)
        
        # Step 5: 检查收敛 (基于 deviance)
        converged = _check_convergence(criterion, iteration + 1, atol, rtol)
        if converged:
            break
    
    # 最终结果封装
    return self._estimate_x2_scale(wls_results, ...)
```

**IRLS 与梯度优化的比较**：

| 特性 | IRLS | 梯度优化 |
|------|------|---------|
| **收敛性** | 对 GLM 通常二次收敛 | 依赖求解器选择 |
| **计算复杂度** | 每次迭代需矩阵求逆 | 每次迭代只需函数/梯度计算 |
| **适用范围** | 仅限 GLM 框架 | 任意似然模型 |
| **内存效率** | 需存储完整权重矩阵 | 更高效 |
| **稳健性** | 对初始值较敏感 | 更灵活 |

### 3.4 各模型 fit() 方法的差异化

| 模型类 | fit() 策略 | 关键差异 |
|--------|-----------|---------|
| **LikelihoodModel** | 通用梯度优化骨架 | 模板方法，定义算法结构 |
| **RegressionModel** | 直接最小二乘求解 | 完全重写，不使用似然 |
| **GLM** | IRLS (默认) + 梯度 | 双重策略，IRLS 更高效 |
| **DiscreteModel** | 梯度优化 + 完美分离检测 | 注入 callback 进行额外检查 |

**设计思想**：**好莱坞原则** - "不要调用我们，我们调用你"

- 父类 `LikelihoodModel.fit()` 定义了"何时做"
- 子类通过实现 `loglike()`, `score()`, `hessian()` 定义"怎么做"
- 子类也可选择完全重写 `fit()` (如 RegressionModel)

---

## 4. 结果类：按需计算的缓存属性设计

### 4.1 结果类继承体系

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Results (基类)                              │
│  职责: 基本结果封装、预测接口、摘要框架                               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LikelihoodModelResults                            │
│  职责: 似然模型结果、统计推断 (t值、p值、标准误)                     │
│  缓存属性: llf, bse, tvalues, pvalues                               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────┬──────────────────────┬──────────────────────┐
│  RegressionResults   │    GLMResults        │  DiscreteResults     │
│  (线性回归结果)       │  (广义线性模型结果)   │  (离散模型结果)       │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

### 4.2 缓存装饰器的实现

**位置**: `statsmodels/tools/decorators.py`

#### 4.2.1 核心装饰器类型

```python
# decorators.py:77-112
class CachedAttribute:
    """
    延迟计算的属性描述符
    
    工作原理:
    1. 首次访问时计算属性值
    2. 将结果存储到实例的 _cache 字典
    3. 后续访问直接从缓存返回
    """
    
    def __init__(self, func, cachename=None):
        self.fget = func                    # 被装饰的方法
        self.name = func.__name__           # 属性名
        self.cachename = cachename or "_cache"  # 缓存字典名
    
    def __get__(self, obj, type=None):
        """
        描述符获取协议
        """
        if obj is None:
            return self.fget  # 类访问时返回函数本身
        
        # 获取或创建缓存字典
        _cache = getattr(obj, self.cachename, None)
        if _cache is None:
            setattr(obj, self.cachename, {})
            _cache = getattr(obj, self.cachename)
        
        # 检查缓存
        name = self.name
        _cachedval = _cache.get(name, None)
        
        # 缓存未命中: 计算并存储
        if _cachedval is None:
            _cachedval = self.fget(obj)  # 执行实际计算
            _cache[name] = _cachedval     # 存入缓存
        
        return _cachedval
    
    def __set__(self, obj, value):
        """
        只读缓存: 写入时发出警告
        """
        errmsg = "The attribute '%s' cannot be overwritten" % self.name
        warnings.warn(errmsg, CacheWriteWarning, stacklevel=2)
```

#### 4.2.2 可写缓存变体

```python
# decorators.py:107-112
class CachedWritableAttribute(CachedAttribute):
    """
    可写入的缓存属性
    允许手动设置缓存值
    """
    def __set__(self, obj, value):
        _cache = getattr(obj, self.cachename)
        name = self.name
        _cache[name] = value  # 直接写入缓存
```

#### 4.2.3 装饰器别名

```python
# decorators.py:136-151
# 使用 pandas 的 cached_property (文档兼容性更好)
cache_readonly = PandasCacheReadonly

# 语义别名 - 用于标识不同用途
cached_value = PandasCacheReadonly  # 计算得到的值 (如统计量)
cached_data = PandasCacheReadonly   # 数据属性 (如残差、拟合值)
```

### 4.3 结果类中的缓存应用

**位置**: `statsmodels/base/model.py:1276-1600+`

#### 4.3.1 基础统计量的缓存

```python
# model.py:1503-1541
class LikelihoodModelResults(Results):
    
    @cached_value
    def llf(self):
        """对数似然值 - 仅在首次访问时计算"""
        return self.model.loglike(self.params)
    
    @cached_value
    def bse(self):
        """参数标准误"""
        # Issue 3299: 处理无协方差的情况
        if (not hasattr(self, "cov_params_default")) and (
            self.normalized_cov_params is None
        ):
            return np.full(len(self.params), np.nan)
        
        # 从协方差矩阵提取对角元并开方
        return np.sqrt(np.diag(self.cov_params()))
    
    @cached_value
    def tvalues(self):
        """t统计量 = 参数估计 / 标准误"""
        return self.params / self.bse
    
    @cached_value
    def pvalues(self):
        """p值 - 基于 t分布或正态分布"""
        if self.use_t:
            # t 分布
            df_resid = getattr(self, "df_resid_inference", self.df_resid)
            return stats.t.sf(np.abs(self.tvalues), df_resid) * 2
        else:
            # 正态分布
            return stats.norm.sf(np.abs(self.tvalues)) * 2
```

#### 4.3.2 信息准则的缓存

```python
# model.py:2516-2560
class ResultMixin:
    """
    结果混合类 - 提供额外的统计量
    """
    
    @cache_readonly
    def df_modelwc(self):
        """用于计算 AIC/BIC 的参数数量 (带常数项修正)"""
        k_extra = getattr(self.model, "k_extra", 0)
        if hasattr(self, "df_model"):
            # df_model + 常数项 + 额外参数
            return self.df_model + self.k_constant + k_extra
        return self.params.size
    
    @cache_readonly
    def aic(self):
        """
        Akaike 信息准则
        AIC = -2 * logL + 2 * k
        """
        return -2 * self.llf + 2 * self.df_modelwc
    
    @cache_readonly
    def bic(self):
        """
        Bayesian 信息准则
        BIC = -2 * logL + log(n) * k
        """
        return -2 * self.llf + np.log(self.nobs) * self.df_modelwc
```

#### 4.3.3 高级统计量的延迟计算

```python
# model.py:2546-2600
class ResultMixin:
    
    @cache_readonly
    def score_obsv(self):
        """
        观测水平的得分向量 (Jacobian)
        用于 BHHH 协方差估计
        """
        return self.model.score_obs(self.params)
    
    @cache_readonly
    def hessv(self):
        """
        观测水平的海森矩阵
        用于信息矩阵计算
        """
        return self.model.hessian(self.params)
    
    @cache_readonly
    def covjac(self):
        """
        基于 Jacobian 外积的协方差 (BHHH 估计)
        """
        # 计算: (J'J)⁻¹
        from statsmodels.tools.tools import pinv_extended
        score_obs = self.score_obsv
        if score_obs.ndim == 1:
            score_obs = score_obs[:, None]
        H = np.dot(score_obs.T, score_obs)
        return pinv_extended(H)[0]
```

### 4.4 缓存机制的执行流程

```
首次访问 results.llf
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  1. 触发 CachedAttribute.__get__()                           │
│     - 获取实例 results 的 _cache 属性                         │
│     - _cache 不存在则创建 {}                                  │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 检查缓存是否存在                                          │
│     - 查找 _cache.get('llf', None)                           │
│     - 返回 None (首次访问)                                    │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  3. 执行实际计算                                              │
│     - 调用 self.fget(obj) = results.llf 的原始方法           │
│     - 执行: self.model.loglike(self.params)                  │
│     - 这可能涉及复杂的似然计算 (如求和、矩阵运算)             │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 存储缓存                                                  │
│     - _cache['llf'] = 计算结果                               │
│     - 返回计算结果                                            │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
再次访问 results.llf
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  5. 缓存命中                                                  │
│     - _cache['llf'] 已存在                                   │
│     - 直接返回缓存值，跳过计算                                │
└─────────────────────────────────────────────────────────────┘
```

### 4.5 缓存一致性与内存管理

#### 4.5.1 数据感知的缓存清除

```python
# model.py:1127-1129
class Results:
    def __init__(self, model, params, **kwd):
        # ...
        # 标记需要随数据变化清除的缓存
        self._data_in_cache = ["fittedvalues", "resid", "wresid"]
```

#### 4.5.2 可写缓存的应用场景

```python
# decorators.py:127-133
class cache_writable(_cache_readonly):
    """
    可写入的缓存装饰器
    
    使用场景:
    1. 某些统计量可能需要在外部计算后手动设置
    2. 允许用户覆盖默认计算
    3. 用于调试或特殊场景
    """
    def __call__(self, func):
        return CachedWritableAttribute(func, cachename=self.cachename)
```

#### 4.5.3 缓存设计的优点

| 优点 | 说明 |
|------|------|
| **计算效率** | 避免重复计算昂贵的统计量 |
| **内存效率** | 不使用的统计量永远不会被计算 |
| **一致性保证** | 所有统计量基于相同的 params 计算 |
| **API 友好** | 用户无需关心何时计算，直接访问即可 |
| **可追溯** | 缓存存储在 `_cache` 字典，便于调试 |

### 4.6 实际使用示例

```python
import statsmodels.api as sm
import numpy as np

# 准备数据
np.random.seed(42)
X = np.random.randn(100, 3)
X = sm.add_constant(X)
y = 1 + 2*X[:,1] + 0.5*X[:,2] + np.random.randn(100)

# 拟合模型
model = sm.OLS(y, X)
results = model.fit()

# 此时: 没有任何统计量被计算 (除了 fit 过程中必需的)

# 首次访问 - 触发计算
print("首次访问 llf:")
print(f"  llf = {results.llf:.4f}")  # 计算并缓存

# 再次访问 - 直接使用缓存
print("\n再次访问 llf:")
print(f"  llf = {results.llf:.4f}")  # 从缓存返回，无计算

# 访问其他统计量 - 按需计算
print("\n访问 tvalues:")
print(f"  tvalues = {results.tvalues}")  # 计算 tvalues (依赖 bse)
print(f"  bse = {results.bse}")          # bse 已被计算 (tvalues 的依赖)

# 信息准则 - 延迟计算
print("\n访问 AIC/BIC:")
print(f"  AIC = {results.aic:.4f}")    # 计算 aic (依赖 llf, df_modelwc)
print(f"  BIC = {results.bic:.4f}")    # 计算 bic (依赖 llf, df_modelwc)

# 查看缓存内容
print("\n缓存字典内容:")
for key, value in results._results._cache.items():
    print(f"  {key}: {type(value).__name__}")
```

---

## 5. 关键设计模式总结

### 5.1 设计模式应用总览

| 设计模式 | 应用位置 | 解决的问题 |
|---------|---------|-----------|
| **模板方法模式** | `LikelihoodModel.fit()` | 定义算法骨架，延迟步骤实现 |
| **策略模式** | 多求解器选择 (`method` 参数) | 运行时切换优化算法 |
| **描述符模式** | `CachedAttribute` 类 | 实现属性的延迟计算 |
| **装饰器模式** | `@cache_readonly` 等 | 语法糖，简化描述符使用 |
| **工厂方法** | `Model.from_formula()` | 灵活创建模型实例 |
| **混合类 (Mixin)** | `ResultMixin` | 代码复用，避免多重继承 |
| **模板方法 (变体)** | `loglike()`, `score()` 抽象方法 | 强制子类实现核心逻辑 |

### 5.2 模板方法模式详解

**核心思想**: 父类定义算法结构，子类实现具体步骤

```python
# 伪代码: LikelihoodModel.fit() 的模板结构
class LikelihoodModel(Model):
    def fit(self, ...):
        """
        这是一个模板方法
        定义了似然模型拟合的通用骨架
        """
        # 固定步骤 1: 初始化参数 (可被子类覆盖)
        start_params = self._get_start_params(start_params)
        
        # 固定步骤 2: 选择优化器 (策略模式)
        optimizer = self._select_optimizer(method)
        
        # 变化步骤: 这些方法由子类实现
        def objective(params):
            return -self.loglike(params)  # 抽象方法
        
        def gradient(params):
            return self.score(params)      # 抽象方法
        
        def hessian(params):
            return self.hessian(params)   # 抽象方法
        
        # 固定步骤 3: 执行优化
        xopt = optimizer.minimize(objective, gradient, hessian, ...)
        
        # 固定步骤 4: 后处理
        Hinv = self._compute_covariance(xopt, ...)
        
        # 固定步骤 5: 封装结果
        return self._create_results(xopt, Hinv, ...)
```

### 5.3 描述符模式详解

**核心思想**: 将属性的访问逻辑委托给专门的对象

```python
# 普通属性访问
results.llf  # 直接读取实例的 __dict__['llf']

# 描述符属性访问
results.llf  # 触发 CachedAttribute.__get__(results, LikelihoodModelResults)
             # 内部逻辑:
             # 1. 检查 results._cache 是否存在
             # 2. 检查 _cache['llf'] 是否存在
             # 3. 不存在则计算并存储
             # 4. 返回结果
```

**为什么使用描述符而不是 `@property`？**

| 特性 | `@property` | `CachedAttribute` 描述符 |
|------|-------------|-------------------------|
| **延迟计算** | 每次访问都计算 | 首次计算，后续缓存 |
| **缓存控制** | 无 | 支持缓存清除、替换 |
| **可组合性** | 低 | 高 (可继承、可定制) |
| **内省能力** | 一般 | 强 (可检查缓存状态) |
| **使用复杂度** | 简单 | 稍复杂 |

### 5.4 继承层级的设计智慧

#### 5.4.1 层次抽象的优势

```
┌─────────────────────────────────────────────────────────────────────────┐
│  为什么需要这么多层继承？                                                  │
└─────────────────────────────────────────────────────────────────────────┘

问题: 直接从 Model 到 OLS 行不行？

尝试方案 A: 单层继承
┌─────────────────────────────────────────────────────────────────────────┐
│  Model ──→ OLS, Logit, Poisson, GLM, ...                                │
│                                                                          │
│  问题:                                                                    │
│  - OLS 使用最小二乘，Logit 使用似然优化                                 │
│  - 如何复用似然相关的代码？                                              │
│  - 如何区分不同的拟合策略？                                              │
└─────────────────────────────────────────────────────────────────────────┘

尝试方案 B: 两层继承
┌─────────────────────────────────────────────────────────────────────────┐
│  Model ──→ LikelihoodModel ──→ Logit, Poisson, GLM                     │
│       └─→ RegressionModel ──→ OLS, WLS, GLS                            │
│                                                                          │
│  问题:                                                                    │
│  - GLM 虽然是似然模型，但有特殊的 IRLS 拟合                             │
│  - 离散模型有完美分离检测等特殊需求                                      │
│  - 需要进一步细分                                                        │
└─────────────────────────────────────────────────────────────────────────┘

实际方案 C: 多层继承 (Statsmodels 的选择)
┌─────────────────────────────────────────────────────────────────────────┐
│  Model                                                                    │
│    └── LikelihoodModel                                                   │
│          ├── RegressionModel                                             │
│          │     └── GLS                                                   │
│          │          └── WLS                                              │
│          │               └── OLS                                         │
│          ├── GLM (特殊拟合: IRLS)                                        │
│          └── DiscreteModel (特殊检测: 完美分离)                          │
│                ├── BinaryModel (Logit, Probit)                           │
│                ├── CountModel (Poisson, NegBin)                          │
│                └── MultinomialModel (MNLogit)                            │
│                                                                          │
│  优势:                                                                    │
│  - 每一层只处理一个抽象级别                                              │
│  - 代码复用最大化                                                        │
│  - 变化点隔离 (开放封闭原则)                                              │
│  - 易于扩展新模型                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 5.4.2 扩展新模型的指南

假设你想添加一个新模型 `MyNewModel`：

```python
# 步骤 1: 确定模型类型，选择合适的基类

# 选项 A: 基于似然，使用标准梯度优化
from statsmodels.base.model import LikelihoodModel

class MyNewModel(LikelihoodModel):
    """
    只需要实现核心抽象方法
    fit() 骨架自动复用
    """
    
    def loglike(self, params):
        # 你的对数似然实现
        pass
    
    def score(self, params):
        # 你的梯度实现 (可选，会用数值近似)
        pass
    
    def hessian(self, params):
        # 你的海森实现 (可选)
        pass
    
    # fit() 自动继承，无需修改

# 选项 B: 需要特殊拟合逻辑
class MyNewModel(LikelihoodModel):
    def fit(self, **kwargs):
        """
        完全重写 fit()，使用自定义算法
        例如: MCMC, EM 算法等
        """
        # 自定义拟合逻辑
        params = self._my_custom_optimization()
        
        # 但仍可复用结果封装
        return LikelihoodModelResults(self, params, ...)

# 选项 C: 属于现有模型族
class MyNewCountModel(CountModel):
    """
    继承 CountModel，自动获得:
    - 似然框架
    - 完美分离检测
    - 正则化支持
    """
    def loglike(self, params):
        # 特定分布的似然
        pass
```

---

## 6. 参考文献

### 6.1 代码位置索引

| 模块 | 文件位置 | 关键内容 |
|------|---------|---------|
| **核心基类** | `statsmodels/base/model.py` | Model, LikelihoodModel, Results, LikelihoodModelResults |
| **缓存装饰器** | `statsmodels/tools/decorators.py` | CachedAttribute, cache_readonly, cached_value |
| **线性回归** | `statsmodels/regression/linear_model.py` | RegressionModel, GLS, WLS, OLS |
| **广义线性模型** | `statsmodels/genmod/generalized_linear_model.py` | GLM, IRLS 算法 |
| **离散模型** | `statsmodels/discrete/discrete_model.py` | DiscreteModel, Logit, Probit, Poisson |

### 6.2 设计模式参考

- **Gamma E, Helm R, Johnson R, Vlissides J**. Design Patterns: Elements of Reusable Object-Oriented Software. Addison-Wesley, 1994.
  - 模板方法模式 (第 3 章)
  - 策略模式 (第 5 章)
  - 装饰器模式 (第 4 章)

### 6.3 统计方法参考

- **McCullagh P, Nelder JA**. Generalized Linear Models. Chapman & Hall/CRC, 1989. (GLM 理论基础)
- **Green WF**. Econometric Analysis. Prentice Hall, 2003. (离散选择模型)
- **Cameron AC, Trivedi PK**. Regression Analysis of Count Data. Cambridge University Press, 1998. (计数模型)

---

## 附录 A: 快速参考表

### A.1 模型继承速查

```python
# 基础
Model()                          # 最抽象，仅数据处理
└── LikelihoodModel()            # 似然框架，fit() 骨架
    ├── RegressionModel()        # 线性回归族
    │   └── GLS()
    │       └── WLS()
    │           └── OLS()
    ├── GLM()                    # 广义线性模型
    └── DiscreteModel()          # 离散选择模型
        ├── BinaryModel()        # Logit, Probit
        ├── CountModel()         # Poisson, NegBin
        └── MultinomialModel()   # MNLogit
```

### A.2 缓存装饰器速查

| 装饰器 | 可写 | 用途 | 示例 |
|--------|------|------|------|
| `@cache_readonly` | 否 | 通用只读缓存 | `aic`, `bic` |
| `@cached_value` | 否 | 计算值缓存 | `llf`, `bse` |
| `@cached_data` | 否 | 数据缓存 | `resid`, `fittedvalues` |
| `@cache_writable` | 是 | 可写缓存 | 特殊场景 |

### A.3 fit() 求解器速查

| 求解器 | 方法名 | 需要梯度 | 需要海森 | 适用场景 |
|--------|--------|---------|---------|---------|
| Newton-Raphson | `'newton'` | 是 | 是 | 局部二次收敛 |
| BFGS | `'bfgs'` | 是 | 否 | 拟牛顿，通用 |
| L-BFGS-B | `'lbfgs'` | 是 | 否 | 大规模问题，有界约束 |
| Nelder-Mead | `'nm'` | 否 | 否 | 无梯度，鲁棒但慢 |
| Powell | `'powell'` | 否 | 否 | 无梯度，方向集 |
| CG | `'cg'` | 是 | 否 | 共轭梯度 |
| NCG | `'ncg'` | 是 | 可选 | Newton-共轭梯度 |
| Basin-Hopping | `'basinhopping'` | 是 | 否 | 全局优化 |
| IRLS | `'IRLS'` | N/A | N/A | GLM 专用 |

---

**报告生成时间**: 2025  
**分析版本**: Statsmodels 0.14.x  
**分析范围**: 核心模型基类体系
