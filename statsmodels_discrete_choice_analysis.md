# Statsmodels 离散选择模型分析报告

## 1. 类层次结构与继承关系

### 1.1 整体架构

Statsmodels 的离散选择模型采用了清晰的继承层次结构，通过多层抽象实现代码复用和扩展性：

```
LikelihoodModel (base.model)
    └── DiscreteModel (discrete_model.py:185)
            ├── BinaryModel (discrete_model.py:521)
            │       ├── Logit (discrete_model.py:2622)
            │       └── Probit (discrete_model.py:2929)
            │
            ├── MultinomialModel (discrete_model.py:758)
            │       └── MNLogit (discrete_model.py:3255)
            │
            └── CountModel (discrete_model.py:1037)
                    ├── Poisson (discrete_model.py:1338)
                    ├── GeneralizedPoisson (discrete_model.py:1962)
                    └── NegativeBinomial (count_model.py)
```

### 1.2 各层职责

| 类名 | 职责 | 关键方法 |
|------|------|----------|
| `LikelihoodModel` | 似然模型基类 | `fit()`, `loglike()`, `score()`, `hessian()` |
| `DiscreteModel` | 离散模型抽象基类 | `_derivative_exog()`, `_check_perfect_pred()` |
| `BinaryModel` | 二元选择模型基类 | 实现二元模型通用逻辑 |
| `CountModel` | 计数模型基类 | 处理 offset/exposure 等计数特性 |
| `MultinomialModel` | 多分类模型基类 | 处理多分类数据转换 |

## 2. 似然函数估计体系

### 2.1 似然函数体系组织

Statsmodels 的似然函数估计采用了**模板方法模式**，基类定义框架，子类实现具体细节：

#### 2.1.1 核心接口设计

每个离散模型必须实现以下核心方法：

| 方法 | 说明 | 位置 |
|------|------|------|
| `loglike(params)` | 计算总对数似然 | 子类实现 |
| `loglikeobs(params)` | 计算每个观测的对数似然 | 子类实现 |
| `score(params)` | 计算得分向量（梯度） | 子类实现 |
| `score_obs(params)` | 计算每个观测的得分 | 子类实现 |
| `hessian(params)` | 计算海森矩阵 | 子类实现 |
| `cdf(X)` | 累积分布函数 | 子类实现 |
| `pdf(X)` | 概率密度/质量函数 | 子类实现 |

#### 2.1.2 具体模型的似然实现

**Logit 模型**（`discrete_model.py:2705-2732`）：
```python
def loglike(self, params):
    """
    对数似然函数：ln L = Σ ln Λ(q_i x_i'β)
    其中 q = 2y - 1，利用 Logistic 分布的对称性
    """
    q = 2 * self.endog - 1
    linpred = self.predict(params, which="linear")
    return np.sum(np.log(self.cdf(q * linpred)))
```

**Probit 模型**（`discrete_model.py:2997-3022`）：
```python
def loglike(self, params):
    """
    对数似然函数：ln L = Σ ln Φ(q_i x_i'β)
    其中 q = 2y - 1，利用正态分布的对称性
    """
    q = 2 * self.endog - 1
    linpred = self.predict(params, which="linear")
    return np.sum(np.log(np.clip(self.cdf(q * linpred), FLOAT_EPS, 1)))
```

**Poisson 模型**（`discrete_model.py:1427-1454`）：
```python
def loglike(self, params):
    """
    对数似然函数：ln L = Σ[-λ_i + y_i x_i'β - ln(y_i!)]
    其中 λ_i = exp(x_i'β + offset + exposure)
    """
    offset = getattr(self, "offset", 0)
    exposure = getattr(self, "exposure", 0)
    XB = np.dot(self.exog, params) + offset + exposure
    endog = self.endog
    return np.sum(
        -np.exp(np.clip(XB, None, EXP_UPPER_LIMIT))
        + endog * XB
        - gammaln(endog + 1)
    )
```

### 2.2 得分（梯度）计算

**Logit 得分**（`discrete_model.py:2765-2788`）：
```python
def score(self, params):
    """
    得分向量：∂ln L/∂β = Σ(y_i - Λ_i)x_i
    其中 Λ_i = Λ(x_i'β) 是 Logistic CDF
    """
    y = self.endog
    X = self.exog
    fitted = self.predict(params)
    return np.dot(y - fitted, X)
```

**Probit 得分**（`discrete_model.py:3053-3081`）：
```python
def score(self, params):
    """
    得分向量：∂ln L/∂β = Σ[q_i φ(q_i x_i'β)/Φ(q_i x_i'β)]x_i
    其中 φ 是正态 PDF，Φ 是正态 CDF
    """
    y = self.endog
    X = self.exog
    XB = self.predict(params, which="linear")
    q = 2 * y - 1
    L = q * self.pdf(q * XB) / np.clip(self.cdf(q * XB), FLOAT_EPS, 1 - FLOAT_EPS)
    return np.dot(L, X)
```

### 2.3 海森矩阵计算

**Logit 海森矩阵**（`discrete_model.py:2846-2867`）：
```python
def hessian(self, params):
    """
    海森矩阵：∂²ln L/∂β∂β' = -Σ Λ_i(1-Λ_i)x_i x_i'
    """
    X = self.exog
    L = self.predict(params)
    return -np.dot(L * (1 - L) * X.T, X)
```

**Poisson 海森矩阵**（`discrete_model.py:1727-1754`）：
```python
def hessian(self, params):
    """
    海森矩阵：∂²ln L/∂β∂β' = -Σ λ_i x_i x_i'
    其中 λ_i = exp(x_i'β)
    """
    offset = getattr(self, "offset", 0)
    exposure = getattr(self, "exposure", 0)
    X = self.exog
    L = np.exp(np.dot(X, params) + exposure + offset)
    return -np.dot(L * X.T, X)
```

### 2.4 最大似然估计流程

估计流程在 `LikelihoodModel.fit()`（`base.model.py:362-576`）中实现，核心步骤：

1. **初始化参数**：
   - 如果未提供 `start_params`，使用零向量或模型特定的初始值
   - Poisson 模型使用 `_get_start_params_null()` 计算初始值

2. **定义优化目标函数**：
   ```python
   def f(params, *args):
       return -self.loglike(params, *args) / nobs  # 负对数似然（平均）
   
   def score(params, *args):
       return -self.score(params, *args) / nobs  # 负得分
   
   def hess(params, *args):
       return -self.hessian(params, *args) / nobs  # 负海森
   ```

3. **选择优化算法**：
   - `newton`：Newton-Raphson（使用海森矩阵）
   - `bfgs`：Broyden-Fletcher-Goldfarb-Shanno
   - `nm`：Nelder-Mead（单纯形法）
   - `lbfgs`：Limited-memory BFGS
   - `powell`：Powell 方法
   - `cg`：共轭梯度法
   - `ncg`：Newton-共轭梯度法

4. **收敛检查**：
   - 使用 `callback` 函数检查完美分离问题（`_check_perfect_pred`）
   - 完美分离时发出警告或抛出异常

## 3. 边际效应计算机制

### 3.1 边际效应的核心概念

边际效应（Marginal Effects）衡量解释变量变化对响应变量概率的影响。Statsmodels 支持四种类型的边际效应：

| 类型 | 公式 | 解释 |
|------|------|------|
| `dydx` | dy/dx | x 变化 1 单位时 y 的变化 |
| `eyex` | d(lny)/d(lnx) | 弹性：x 变化 1% 时 y 变化的百分比 |
| `dyex` | dy/d(lnx) | 半弹性：x 变化 1% 时 y 的变化 |
| `eydx` | d(lny)/dx | 半弹性：x 变化 1 单位时 y 变化的百分比 |

### 3.2 边际效应的计算位置

支持在以下位置计算边际效应：

| 位置 | 说明 |
|------|------|
| `overall` | 所有观测的平均边际效应（默认） |
| `mean` | 在解释变量均值处的边际效应 |
| `median` | 在解释变量中位数处的边际效应 |
| `zero` | 在解释变量为零处的边际效应 |
| `all` | 每个观测的边际效应 |

### 3.3 边际效应的实现架构

#### 3.3.1 整体流程

```
用户调用 results.get_margeff()
    ↓
DiscreteResults.get_margeff() (discrete_model.py:5293)
    ↓
创建 DiscreteMargins 实例 (discrete_margins.py:433)
    ↓
DiscreteMargins.get_margeff() (discrete_margins.py:688)
    ↓
model._derivative_exog() 计算基础边际效应
    ↓
处理虚拟变量和计数变量的离散变化
    ↓
margeff_cov_with_se() 使用 Delta 方法计算标准误
```

#### 3.3.2 核心实现类：DiscreteMargins

`DiscreteMargins` 类（`discrete_margins.py:433`）负责计算和管理边际效应：

```python
class DiscreteMargins:
    """Get marginal effects of a Discrete Choice model."""
    
    def __init__(self, results, args, kwargs=None):
        self._cache = {}
        self.results = results
        self.get_margeff(*args, **kwargs)
```

#### 3.3.3 边际效应计算主方法

`get_margeff()` 方法（`discrete_margins.py:688-823`）的核心流程：

1. **参数检查和预处理**：
   ```python
   _check_margeff_args(at, method)  # 验证参数有效性
   effects_idx, const_idx = _get_const_index(exog)  # 识别常数项
   ```

2. **识别离散变量**：
   ```python
   if dummy:
       dummy_idx, dummy = _get_dummy_index(exog, const_idx)  # 识别虚拟变量
   if count:
       count_idx, count = _get_count_index(exog, const_idx)  # 识别计数变量
   ```

3. **准备计算边际效应的解释变量**：
   ```python
   exog = _get_margeff_exog(exog, at, atexog, effects_idx)
   ```

4. **计算基础边际效应**：
   ```python
   effects = model._derivative_exog(params, exog, method, dummy_idx, count_idx)
   ```

5. **处理离散变量的边际效应**：
   - 虚拟变量：计算从 0 到 1 的变化（`_get_dummy_effects`）
   - 计数变量：计算 ±1 变化的平均影响（`_get_count_effects`）

6. **汇总边际效应**：
   ```python
   effects = _effects_at(effects, at)  # 根据 at 参数汇总
   ```

7. **计算标准误（Delta 方法）**：
   ```python
   if at != "all":
       margeff_cov, margeff_se = margeff_cov_with_se(
           model, params, exog, results.cov_params(), at,
           model._derivative_exog, dummy_idx, count_idx, method, J
       )
   ```

### 3.4 不同模型的边际效应计算

#### 3.4.1 二元模型（BinaryModel）

`BinaryModel._derivative_exog()`（`discrete_model.py:670-703`）：

```python
def _derivative_exog(
    self, params, exog=None, transform="dydx", dummy_idx=None, count_idx=None, offset=None
):
    """
    计算 dF(XB)/dX，其中 F(.) 是预测概率
    
    对于二元模型：
    dF/dX = pdf(XB) * params
    
    变换支持：
    - dydx: 直接返回
    - dyex: 乘以 exog（对应 d(lnx)）
    - eydx: 除以预测值（对应 d(lny)）
    - eyex: 乘以 exog 并除以预测值
    """
    if exog is None:
        exog = self.exog
    
    # 计算线性预测 XB
    linpred = self.predict(params, exog, offset=offset, which="linear")
    
    # 基础边际效应：pdf(XB) * params
    margeff = np.dot(self.pdf(linpred)[:, None], params[None, :])
    
    # 应用变换
    if "ex" in transform:
        margeff *= exog  # dy/d(lnx) = x * dy/dx
    if "ey" in transform:
        margeff /= self.predict(params, exog)[:, None]  # d(lny)/dx = (1/y) * dy/dx
    
    # 处理离散变量
    return self._derivative_exog_helper(
        margeff, params, exog, dummy_idx, count_idx, transform
    )
```

**Logit 模型的边际效应**：
- PDF：`pdf(X) = exp(-X)/(1+exp(-X))² = Λ(X)(1-Λ(X))`
- 边际效应：`dΛ(XB)/dX = Λ(XB)(1-Λ(XB)) * params`

**Probit 模型的边际效应**：
- PDF：`pdf(X) = φ(X)`（正态密度函数）
- 边际效应：`dΦ(XB)/dX = φ(XB) * params`

#### 3.4.2 多分类模型（MultinomialModel）

`MultinomialModel._derivative_exog()`（`discrete_model.py:979-1030`）：

```python
def _derivative_exog(
    self, params, exog=None, transform="dydx", dummy_idx=None, count_idx=None
):
    """
    多分类 Logit 模型的边际效应
    
    公式：P[j] * (params[j] - sum_k P[k] * params[k])
    
    其中：
    - P[j] 是选择第 j 类的概率
    - params[j] 是第 j 类的参数向量（base category 为 0）
    """
    J = int(self.J)  # 类别数
    K = int(self.K)  # 变量数
    
    if exog is None:
        exog = self.exog
    if params.ndim == 1:
        params = params.reshape(K, J - 1, order="F")
    
    # 添加 base category 的零参数
    zeroparams = np.c_[np.zeros(K), params]
    
    # 计算各类概率
    cdf = self.cdf(np.dot(exog, params))
    
    # 计算加权平均参数
    iterm = np.array([cdf[:, [i]] * zeroparams[:, i] for i in range(int(J))]).sum(0)
    
    # 计算边际效应
    margeff = np.array([cdf[:, [j]] * (zeroparams[:, j] - iterm) for j in range(J)])
    
    # 调整维度顺序：nobs, K, J
    margeff = np.transpose(margeff, (1, 2, 0))
    
    # 应用变换
    if "ex" in transform:
        margeff *= exog
    if "ey" in transform:
        margeff /= self.predict(params, exog)[:, None, :]
    
    # 处理离散变量
    margeff = self._derivative_exog_helper(
        margeff, params, exog, dummy_idx, count_idx, transform
    )
    return margeff.reshape(len(exog), -1, order="F")
```

#### 3.4.3 计数模型（CountModel）

`CountModel._derivative_exog()`（`discrete_model.py:1220-1247`）：

```python
def _derivative_exog(
    self, params, exog=None, transform="dydx", dummy_idx=None, count_idx=None
):
    """
    计数模型的边际效应
    
    对于 Poisson 模型：
    E[y|X] = exp(XB)
    dE[y|X]/dX = exp(XB) * params = E[y|X] * params
    """
    if exog is None:
        exog = self.exog
    
    # 处理额外参数（如 NegativeBinomial 的 dispersion 参数）
    k_extra = getattr(self, "k_extra", 0)
    params_exog = params if k_extra == 0 else params[:-k_extra]
    
    # 基础边际效应：E[y|X] * params
    margeff = self.predict(params, exog)[:, None] * params_exog[None, :]
    
    # 应用变换
    if "ex" in transform:
        margeff *= exog
    if "ey" in transform:
        margeff /= self.predict(params, exog)[:, None]  # 对于 eydx，结果就是 params
    
    # 处理离散变量
    return self._derivative_exog_helper(
        margeff, params, exog, dummy_idx, count_idx, transform
    )
```

### 3.5 离散变量的边际效应处理

对于虚拟变量和计数变量，Statsmodels 采用**离散变化**而非导数：

#### 3.5.1 虚拟变量（Dummy Variables）

`_get_dummy_effects()`（`discrete_margins.py:177-194`）：

```python
def _get_dummy_effects(effects, exog, dummy_ind, method, model, params):
    """
    虚拟变量的边际效应：计算从 0 到 1 的变化
    
    ΔF = F(XB | d=1) - F(XB | d=0)
    """
    for i in dummy_ind:
        exog0 = exog.copy()
        exog0[:, i] = 0
        effect0 = model.predict(params, exog0)
        
        exog0[:, i] = 1
        effect1 = model.predict(params, exog0)
        
        # 对于弹性变换，使用对数差分
        if "ey" in method:
            effect0 = np.log(effect0)
            effect1 = np.log(effect1)
        
        effects[:, i] = effect1 - effect0
    return effects
```

#### 3.5.2 计数变量（Count Variables）

`_get_count_effects()`（`discrete_margins.py:156-174`）：

```python
def _get_count_effects(effects, exog, count_ind, method, model, params):
    """
    计数变量的边际效应：计算 ±1 变化的平均影响
    
    ΔF = [F(XB | d+1) - F(XB | d-1)] / 2
    """
    for i in count_ind:
        exog0 = exog.copy()
        exog0[:, i] -= 1
        effect0 = model.predict(params, exog0)
        
        exog0[:, i] += 2
        effect1 = model.predict(params, exog0)
        
        # 对于弹性变换，使用对数差分
        if "ey" in method:
            effect0 = np.log(effect0)
            effect1 = np.log(effect1)
        
        effects[:, i] = (effect1 - effect0) / 2
    return effects
```

### 3.6 Delta 方法计算标准误

`margeff_cov_params()`（`discrete_margins.py:271-349`）实现了 Delta 方法：

```python
def margeff_cov_params(
    model, params, exog, cov_params, at, derivative, dummy_ind, count_ind, method, J
):
    """
    使用 Delta 方法计算边际效应的方差-协方差矩阵
    
    公式：
    Asy.Var[MargEff] = [d margeff / d params] * V * [d margeff / d params]'
    
    其中：
    - V 是参数的方差-协方差矩阵（cov_params）
    - [d margeff / d params] 是边际效应对参数的 Jacobian 矩阵
    """
    if callable(derivative):
        # 数值计算 Jacobian 矩阵
        params = params.ravel("F")
        try:
            jacobian_mat = approx_fprime_cs(params, derivative, args=(exog, method))
        except TypeError:  # norm.cdf 不支持复数
            jacobian_mat = approx_fprime(params, derivative, args=(exog, method))
        
        # 对于 overall，取平均
        if at == "overall":
            jacobian_mat = np.mean(jacobian_mat, axis=1)
        
        # 处理离散变量的 Jacobian
        if dummy_ind is not None:
            jacobian_mat = _margeff_cov_params_dummy(
                model, jacobian_mat, params, exog, dummy_ind, method, J
            )
        if count_ind is not None:
            jacobian_mat = _margeff_cov_params_count(
                model, jacobian_mat, params, exog, count_ind, method, J
            )
    else:
        jacobian_mat = derivative
    
    # 计算方差-协方差矩阵
    return np.dot(np.dot(jacobian_mat, cov_params), jacobian_mat.T)
```

## 4. 共享设计模式与架构特征

### 4.1 模板方法模式（Template Method Pattern）

Statsmodels 的离散模型广泛使用了模板方法模式：

**基类定义框架**：
- `DiscreteModel` 定义了离散模型的通用接口
- `fit()` 方法在 `LikelihoodModel` 中实现，子类只需提供 `loglike()`、`score()`、`hessian()`

**子类实现细节**：
- `Logit` 实现了 Logistic 分布的 `cdf()`、`pdf()`、`loglike()`
- `Probit` 实现了正态分布的 `cdf()`、`pdf()`、`loglike()`
- `Poisson` 实现了 Poisson 分布的相关方法

**示例**：`fit()` 方法的模板化实现

```python
# LikelihoodModel.fit() 定义框架
def fit(self, start_params=None, method="newton", ...):
    # 1. 初始化参数
    if start_params is None:
        start_params = [0.0] * self.exog.shape[1]
    
    # 2. 定义目标函数（使用子类的 loglike）
    def f(params, *args):
        return -self.loglike(params, *args) / nobs
    
    def score(params, *args):
        return -self.score(params, *args) / nobs
    
    # 3. 调用优化器
    # ...
    
# 子类只需实现具体的 loglike、score、hessian
class Logit(BinaryModel):
    def loglike(self, params):
        # Logit 特定实现
        q = 2 * self.endog - 1
        linpred = self.predict(params, which="linear")
        return np.sum(np.log(self.cdf(q * linpred)))
    
    def score(self, params):
        # Logit 特定实现
        y = self.endog
        X = self.exog
        fitted = self.predict(params)
        return np.dot(y - fitted, X)
```

### 4.2 继承层次结构

**多层继承实现代码复用**：

```
LikelihoodModel（似然估计框架）
    ↓
DiscreteModel（离散模型通用逻辑）
    ├── 完美分离检查
    ├── 边际效应计算框架
    └── 结果类包装
    ↓
BinaryModel（二元模型特性）
    ├── 二元数据验证
    ├── 二元预测逻辑
    └── 二元边际效应计算
    ↓
Logit / Probit（具体分布）
    ├── CDF/PDF 实现
    ├── 似然函数实现
    └── 得分/海森实现
```

**关键继承点**：

1. **`LikelihoodModel` → `DiscreteModel`**：
   - 继承了 `fit()` 方法
   - 添加了 `_check_perfect_pred()` 完美分离检查
   - 定义了 `_derivative_exog()` 边际效应接口

2. **`DiscreteModel` → `BinaryModel`**：
   - 添加了二元数据验证（endog 必须是 0/1）
   - 实现了二元模型的 `predict()` 方法
   - 实现了二元模型的 `_derivative_exog()` 方法

3. **`BinaryModel` → `Logit/Probit`**：
   - 实现了具体的 `cdf()` 和 `pdf()` 方法
   - 实现了具体的 `loglike()`、`score()`、`hessian()` 方法

### 4.3 结果类与模型类分离

Statsmodels 采用了**模型-结果分离**的设计：

**模型类**：负责数据处理、似然计算、模型设定
- `Logit`、`Probit`、`Poisson` 等

**结果类**：负责存储估计结果、提供推断方法
- `DiscreteResults`、`BinaryResults`、`CountResults` 等

**结果类层次**：
```
LikelihoodModelResults (base.model)
    └── DiscreteResults (discrete_model.py:4905)
            ├── BinaryResults (discrete_model.py:5753)
            │       ├── LogitResults
            │       └── ProbitResults
            └── CountResults (discrete_model.py:5522)
                    ├── PoissonResults
                    └── ...
```

**结果类的职责**：
- 存储参数估计值、标准误、协方差矩阵
- 提供模型诊断统计量（伪 R²、似然比检验等）
- 提供预测方法（`predict()`、`get_prediction()`）
- 提供边际效应计算（`get_margeff()`）
- 提供结果汇总（`summary()`）

**示例**：`DiscreteResults.get_margeff()`

```python
def get_margeff(self, at="overall", method="dydx", atexog=None, dummy=False, count=False):
    """
    结果类的边际效应方法，委托给 DiscreteMargins 类
    """
    if getattr(self.model, "offset", None) is not None:
        raise NotImplementedError("Margins with offset are not available.")
    from statsmodels.discrete.discrete_margins import DiscreteMargins
    
    # 创建 DiscreteMargins 实例处理具体计算
    return DiscreteMargins(self, (at, method, atexog, dummy, count))
```

### 4.4 装饰器模式

Statsmodels 使用 `@Appender` 装饰器实现文档字符串的复用：

```python
from statsmodels.compat.pandas import Appender

class Logit(BinaryModel):
    @cache_readonly
    def link(self):
        from statsmodels.genmod.families import links
        return links.Logit()
    
    @Appender(DiscreteModel.fit.__doc__)
    def fit(self, start_params=None, method="newton", ...):
        """
        Logit 模型的 fit 方法
        文档字符串通过 @Appender 从基类追加
        """
        bnryfit = super().fit(
            start_params=start_params, method=method, ...
        )
        discretefit = LogitResults(self, bnryfit)
        return BinaryResultsWrapper(discretefit)
```

### 4.5 缓存机制

使用 `@cache_readonly` 装饰器实现结果缓存：

```python
from statsmodels.tools.decorators import cache_readonly

class DiscreteResults(base.LikelihoodModelResults):
    @cache_readonly
    def prsquared(self):
        """McFadden's pseudo-R-squared."""
        return 1 - self.llf / self.llnull
    
    @cache_readonly
    def llr(self):
        """Likelihood ratio chi-squared statistic."""
        return -2 * (self.llnull - self.llf)
    
    @cache_readonly
    def llr_pvalue(self):
        """p-value for likelihood ratio test."""
        return stats.distributions.chi2.sf(self.llr, self.df_model)
```

**缓存的优势**：
- 避免重复计算（如 `llnull` 需要拟合零模型）
- 提高访问效率
- 保持结果一致性

### 4.6 Wrapper 模式

使用 Wrapper 类提供用户友好的接口：

```python
class BinaryResultsWrapper(lm.RegressionResultsWrapper):
    """
    包装 BinaryResults，提供更友好的用户接口
    """
    pass

# 使用示例
model = Logit(endog, exog)
results = model.fit()  # 返回 BinaryResultsWrapper 实例
print(results.summary())  # 通过 wrapper 访问方法
margeff = results.get_margeff()  # 通过 wrapper 访问方法
```

**Wrapper 的作用**：
- 分离内部实现和外部接口
- 提供类型安全的方法访问
- 支持属性的延迟加载

## 5. 关键代码位置总结

| 功能 | 文件位置 | 关键类/方法 |
|------|----------|-------------|
| **似然估计框架** | `statsmodels/base/model.py` | `LikelihoodModel.fit()` |
| **离散模型基类** | `statsmodels/discrete/discrete_model.py:185` | `DiscreteModel` |
| **二元模型基类** | `statsmodels/discrete/discrete_model.py:521` | `BinaryModel` |
| **计数模型基类** | `statsmodels/discrete/discrete_model.py:1037` | `CountModel` |
| **多分类模型基类** | `statsmodels/discrete/discrete_model.py:758` | `MultinomialModel` |
| **Logit 模型** | `statsmodels/discrete/discrete_model.py:2622` | `Logit` |
| **Probit 模型** | `statsmodels/discrete/discrete_model.py:2929` | `Probit` |
| **Poisson 模型** | `statsmodels/discrete/discrete_model.py:1338` | `Poisson` |
| **MNLogit 模型** | `statsmodels/discrete/discrete_model.py:3255` | `MNLogit` |
| **边际效应计算** | `statsmodels/discrete/discrete_margins.py` | `DiscreteMargins` |
| **Delta 方法** | `statsmodels/discrete/discrete_margins.py:271` | `margeff_cov_params()` |
| **结果基类** | `statsmodels/discrete/discrete_model.py:4905` | `DiscreteResults` |
| **边际效应方法** | `statsmodels/discrete/discrete_model.py:5293` | `DiscreteResults.get_margeff()` |

## 6. 设计亮点与可扩展性

### 6.1 设计亮点

1. **清晰的继承层次**：通过多层继承实现代码复用，每层职责明确
2. **模板方法模式**：基类定义算法框架，子类实现具体细节
3. **模型-结果分离**：模型类负责估计，结果类负责推断和展示
4. **灵活的边际效应**：支持多种变换类型、多种计算位置、离散变量特殊处理
5. **完善的数值计算**：使用 Delta 方法计算标准误，支持数值微分
6. **健壮的估计过程**：包含完美分离检查、多种优化算法选择

### 6.2 可扩展性

**添加新的离散模型**：
1. 继承 `DiscreteModel` 或其子类
2. 实现 `cdf()`、`pdf()` 方法
3. 实现 `loglike()`、`loglikeobs()` 方法
4. 实现 `score()`、`hessian()` 方法（或使用数值微分）
5. 实现 `_derivative_exog()` 方法（或使用继承的实现）
6. 创建对应的结果类（如需要）

**添加新的边际效应类型**：
1. 在 `_check_margeff_args()` 中添加参数验证
2. 在 `_derivative_exog()` 中添加变换逻辑
3. 在 `DiscreteMargins` 中更新文档和汇总

**自定义优化算法**：
1. 在 `LikelihoodModel.fit()` 中添加新方法
2. 或通过 `fit()` 的 `method` 参数使用 scipy.optimize 的算法

## 7. 总结

Statsmodels 的离散选择模型采用了精心设计的面向对象架构：

1. **似然函数体系**：通过模板方法模式，基类提供估计框架，子类实现具体分布的似然、得分和海森矩阵。

2. **边际效应计算**：
   - 连续变量：使用导数 `pdf(XB) * params`
   - 离散变量：使用离散变化（0→1 或 ±1）
   - 支持四种变换类型（dydx, eyex, dyex, eydx）
   - 使用 Delta 方法计算标准误

3. **设计模式**：
   - 模板方法模式：定义估计流程框架
   - 继承层次：实现代码复用和扩展
   - 模型-结果分离：清晰的职责划分
   - 装饰器模式：文档字符串复用
   - 缓存机制：提高计算效率

这种设计使得 statsmodels 的离散选择模型既易于使用（统一的接口），又易于扩展（添加新模型只需实现特定方法）。
