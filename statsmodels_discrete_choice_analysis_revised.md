# Statsmodels 离散选择模型分析报告（修订版）

## 1. 类层次结构与继承关系

### 1.1 整体架构

Statsmodels 的离散选择模型采用了清晰的继承层次结构，通过多层抽象实现代码复用和扩展性。以下是**修正后的准确继承关系**：

```
LikelihoodModel (base.model.py:279)
    └── DiscreteModel (discrete_model.py:185)
            ├── BinaryModel (discrete_model.py:521)
            │       ├── Logit (discrete_model.py:2622)
            │       ├── Probit (discrete_model.py:2929)
            │       └── MultinomialModel (discrete_model.py:758)  ← 关键修正：继承自 BinaryModel
            │               └── MNLogit (discrete_model.py:3255)
            │
            └── CountModel (discrete_model.py:1037)
                    ├── Poisson (discrete_model.py:1338)
                    ├── GeneralizedPoisson (discrete_model.py:1962)
                    ├── NegativeBinomial (discrete_model.py:3597)  ← 关键修正：在 discrete_model.py
                    ├── NegativeBinomialP (discrete_model.py:4224)
                    └── GenericZeroInflated (count_model.py:48)  ← count_model.py 只有零膨胀模型
```

**重要说明**：
- `MultinomialModel` 继承自 `BinaryModel`，而不是直接继承自 `DiscreteModel`
- 主要计数模型（`Poisson`、`NegativeBinomial`、`GeneralizedPoisson`、`NegativeBinomialP`）都在 `discrete_model.py` 中
- `count_model.py` 中只有 `GenericZeroInflated` 及相关的零膨胀模型

### 1.2 各层职责

| 类名 | 父类 | 职责 | 关键方法 |
|------|------|------|----------|
| `LikelihoodModel` | `Model` | 似然模型基类 | `fit()`, `loglike()`, `score()`, `hessian()` |
| `DiscreteModel` | `LikelihoodModel` | 离散模型抽象基类 | `_derivative_exog()`, `_check_perfect_pred()` |
| `BinaryModel` | `DiscreteModel` | 二元选择模型基类 | 二元数据验证、二元预测逻辑 |
| `MultinomialModel` | `BinaryModel` | 多分类模型基类 | 多分类数据转换、多分类预测 |
| `CountModel` | `DiscreteModel` | 计数模型基类 | 处理 offset/exposure 等计数特性 |

## 2. 似然函数估计体系

### 2.1 似然函数体系组织

Statsmodels 的似然函数估计采用了**模板方法模式**，基类定义框架，子类实现具体细节：

#### 2.1.1 核心接口设计

每个离散模型必须实现以下核心方法：

| 方法 | 说明 | 实现位置 |
|------|------|----------|
| `loglike(params)` | 计算总对数似然 | 各具体模型类 |
| `loglikeobs(params)` | 计算每个观测的对数似然 | 各具体模型类 |
| `score(params)` | 计算得分向量（梯度） | 各具体模型类 |
| `score_obs(params)` | 计算每个观测的得分 | 各具体模型类 |
| `hessian(params)` | 计算海森矩阵 | 各具体模型类 |
| `cdf(X)` | 累积分布函数 | 各具体模型类 |
| `pdf(X)` | 概率密度/质量函数 | 各具体模型类（MNLogit 除外） |

**注意**：`MNLogit.pdf()` 方法抛出 `NotImplementedError`，因为多分类 Logit 模型的 PDF 概念与二元模型不同。

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

**MNLogit 模型**（`discrete_model.py:3342-3372`）：
```python
def loglike(self, params):
    """
    对数似然函数：ln L = Σ_i Σ_j d_ij ln[exp(β_j'x_i) / Σ_k exp(β_k'x_i)]
    
    其中 d_ij = 1 如果个体 i 选择了选项 j，否则为 0
    """
    params = params.reshape(self.K, -1, order="F")
    d = self.wendog
    logprob = np.log(self.cdf(np.dot(self.exog, params)))
    return np.sum(d * logprob)
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

**NegativeBinomial 模型**（`discrete_model.py:3597`，继承自 `CountModel`）：
- NB1 和 NB2 两种参数化方式
- 包含额外的离散参数（dispersion parameter）

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

**MNLogit 得分**（`discrete_model.py:3428-3480`）：
```python
def score(self, params):
    """
    多分类 Logit 模型的得分向量
    
    对于每个选项 j（非 base category）：
    ∂ln L/∂β_j = Σ_i (d_ij - P_ij) x_i
    
    其中 P_ij 是个体 i 选择选项 j 的概率
    """
    params = params.reshape(self.K, -1, order="F")
    X = self.exog
    d = self.wendog
    prob = self.cdf(np.dot(X, params))
    # 计算每个非 base category 的得分
    # ...
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
   - `newton`：Newton-Raphson（使用海森矩阵，默认）
   - `bfgs`：Broyden-Fletcher-Goldfarb-Shanno
   - `nm`：Nelder-Mead（单纯形法）
   - `lbfgs`：Limited-memory BFGS
   - `powell`：Powell 方法
   - `cg`：共轭梯度法
   - `ncg`：Newton-共轭梯度法
   - `basinhopping`：全局 basin-hopping 求解器
   - `minimize`：scipy minimize 的通用包装器

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

**重要说明**：`MultinomialModel` 继承自 `BinaryModel`，但覆盖了 `_derivative_exog()` 方法以实现多分类边际效应。

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

## 4. 结果类层次结构（修正版）

### 4.1 结果类继承关系

**重要修正**：结果类的继承关系与模型类不完全一致。

```
LikelihoodModelResults (base.model.py:1276)
    └── DiscreteResults (discrete_model.py:4905)
            ├── BinaryResults (discrete_model.py:5753)
            │       ├── LogitResults (discrete_model.py:5885)
            │       └── ProbitResults (discrete_model.py:5927)
            │
            ├── CountResults (discrete_model.py:5522)
            │       ├── PoissonResults (discrete_model.py:5638)
            │       ├── NegativeBinomialResults (discrete_model.py:5565)
            │       │       └── NegativeBinomialPResults (discrete_model.py:5596)
            │       └── GeneralizedPoissonResults (discrete_model.py:5603)
            │
            └── MultinomialResults (discrete_model.py:5969)  ← 关键修正：直接继承自 DiscreteResults
                    └── ...
```

**重要说明**：
- 模型类：`MultinomialModel` 继承自 `BinaryModel`
- 结果类：`MultinomialResults` 直接继承自 `DiscreteResults`（而非 `BinaryResults`）

这种设计是因为多分类模型的结果结构（多方程参数）与二元模型有显著差异。

### 4.2 结果类职责

| 结果类 | 父类 | 对应模型 | 主要职责 |
|--------|------|----------|----------|
| `DiscreteResults` | `LikelihoodModelResults` | 所有离散模型 | 基础结果存储、伪 R²、似然比检验、`get_margeff()` |
| `BinaryResults` | `DiscreteResults` | `Logit`, `Probit` | 二元模型特定方法 |
| `CountResults` | `DiscreteResults` | `Poisson`, `NegativeBinomial` 等 | 计数模型特定方法（如 `predict_prob()`） |
| `MultinomialResults` | `DiscreteResults` | `MNLogit` | 多分类模型特定方法 |

### 4.3 Wrapper 类

Statsmodels 使用 Wrapper 模式提供用户友好的接口：

```
RegressionResultsWrapper (linear_model.py)
    ├── BinaryResultsWrapper (discrete_model.py:6219)
    ├── CountResultsWrapper (discrete_model.py:6159)
    ├── MultinomialResultsWrapper (discrete_model.py:6239)
    └── PoissonResultsWrapper (discrete_model.py:6187)
        └── ...
```

## 5. 共享设计模式与架构特征

### 5.1 模板方法模式（Template Method Pattern）

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

### 5.2 继承层次结构

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
    ├── 二元边际效应计算
    └── MultinomialModel（多分类特性）
            ├── 多分类数据转换
            └── 多分类预测逻辑
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

3. **`BinaryModel` → `MultinomialModel`**：
   - 覆盖了 `_handle_data()` 处理多分类数据
   - 覆盖了 `initialize()` 设置多分类特定属性（J, K）
   - 覆盖了 `predict()` 实现多分类预测
   - 覆盖了 `_derivative_exog()` 实现多分类边际效应

4. **`BinaryModel` → `Logit/Probit`**：
   - 实现了具体的 `cdf()` 和 `pdf()` 方法
   - 实现了具体的 `loglike()`、`score()`、`hessian()` 方法

### 5.3 模型-结果分离

Statsmodels 采用了**模型-结果分离**的设计：

**模型类**：负责数据处理、似然计算、模型设定
- `Logit`、`Probit`、`Poisson`、`MNLogit` 等

**结果类**：负责存储估计结果、提供推断方法
- `DiscreteResults`、`BinaryResults`、`CountResults`、`MultinomialResults` 等

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

### 5.4 装饰器模式

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

### 5.5 缓存机制

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

### 5.6 Wrapper 模式

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

## 6. 关键代码位置总结（修正版）

### 6.1 模型类位置

| 模型类 | 父类 | 文件位置 | 行号 |
|--------|------|----------|------|
| `DiscreteModel` | `LikelihoodModel` | `statsmodels/discrete/discrete_model.py` | 185 |
| `BinaryModel` | `DiscreteModel` | `statsmodels/discrete/discrete_model.py` | 521 |
| `MultinomialModel` | `BinaryModel` | `statsmodels/discrete/discrete_model.py` | 758 |
| `CountModel` | `DiscreteModel` | `statsmodels/discrete/discrete_model.py` | 1037 |
| `Logit` | `BinaryModel` | `statsmodels/discrete/discrete_model.py` | 2622 |
| `Probit` | `BinaryModel` | `statsmodels/discrete/discrete_model.py` | 2929 |
| `MNLogit` | `MultinomialModel` | `statsmodels/discrete/discrete_model.py` | 3255 |
| `Poisson` | `CountModel` | `statsmodels/discrete/discrete_model.py` | 1338 |
| `GeneralizedPoisson` | `CountModel` | `statsmodels/discrete/discrete_model.py` | 1962 |
| `NegativeBinomial` | `CountModel` | `statsmodels/discrete/discrete_model.py` | 3597 |
| `NegativeBinomialP` | `CountModel` | `statsmodels/discrete/discrete_model.py` | 4224 |
| `GenericZeroInflated` | `CountModel` | `statsmodels/discrete/count_model.py` | 48 |

### 6.2 结果类位置

| 结果类 | 父类 | 文件位置 | 行号 |
|--------|------|----------|------|
| `DiscreteResults` | `LikelihoodModelResults` | `statsmodels/discrete/discrete_model.py` | 4905 |
| `BinaryResults` | `DiscreteResults` | `statsmodels/discrete/discrete_model.py` | 5753 |
| `CountResults` | `DiscreteResults` | `statsmodels/discrete/discrete_model.py` | 5522 |
| `MultinomialResults` | `DiscreteResults` | `statsmodels/discrete/discrete_model.py` | 5969 |
| `LogitResults` | `BinaryResults` | `statsmodels/discrete/discrete_model.py` | 5885 |
| `ProbitResults` | `BinaryResults` | `statsmodels/discrete/discrete_model.py` | 5927 |
| `PoissonResults` | `CountResults` | `statsmodels/discrete/discrete_model.py` | 5638 |
| `NegativeBinomialResults` | `CountResults` | `statsmodels/discrete/discrete_model.py` | 5565 |

### 6.3 边际效应相关位置

| 功能 | 文件位置 | 行号 |
|------|----------|------|
| `DiscreteMargins` 类 | `statsmodels/discrete/discrete_margins.py` | 433 |
| `DiscreteMargins.get_margeff()` | `statsmodels/discrete/discrete_margins.py` | 688 |
| `margeff_cov_params()` (Delta 方法) | `statsmodels/discrete/discrete_margins.py` | 271 |
| `DiscreteResults.get_margeff()` | `statsmodels/discrete/discrete_model.py` | 5293 |
| `BinaryModel._derivative_exog()` | `statsmodels/discrete/discrete_model.py` | 670 |
| `MultinomialModel._derivative_exog()` | `statsmodels/discrete/discrete_model.py` | 979 |
| `CountModel._derivative_exog()` | `statsmodels/discrete/discrete_model.py` | 1220 |

### 6.4 似然估计框架位置

| 功能 | 文件位置 | 行号 |
|------|----------|------|
| `LikelihoodModel` 类 | `statsmodels/base/model.py` | 279 |
| `LikelihoodModel.fit()` | `statsmodels/base/model.py` | 362 |
| `LikelihoodModelResults` 类 | `statsmodels/base/model.py` | 1276 |

## 7. 设计亮点与可扩展性

### 7.1 设计亮点

1. **清晰的继承层次**：通过多层继承实现代码复用，每层职责明确
   - 模型类：`MultinomialModel` 继承自 `BinaryModel`，复用二元模型的基础结构
   - 结果类：`MultinomialResults` 直接继承自 `DiscreteResults`，适应多分类结果的特殊性

2. **模板方法模式**：基类定义算法框架，子类实现具体细节
   - `fit()` 方法在基类实现，子类只需提供 `loglike()`、`score()`、`hessian()`

3. **模型-结果分离**：模型类负责估计，结果类负责推断和展示
   - 清晰的职责划分
   - 结果类可以独立扩展（如添加新的诊断统计量）

4. **灵活的边际效应**：支持多种变换类型、多种计算位置、离散变量特殊处理
   - 四种变换类型：`dydx`、`eyex`、`dyex`、`eydx`
   - 五种计算位置：`overall`、`mean`、`median`、`zero`、`all`
   - 离散变量使用离散变化而非导数

5. **完善的数值计算**：使用 Delta 方法计算标准误，支持数值微分
   - `margeff_cov_params()` 实现 Delta 方法
   - 当无法使用复数微分时，自动切换到实数微分

6. **健壮的估计过程**：包含完美分离检查、多种优化算法选择
   - `_check_perfect_pred()` 检测完美分离
   - 支持 Newton-Raphson、BFGS、Nelder-Mead 等多种优化器

### 7.2 可扩展性

**添加新的离散模型**：
1. 继承 `DiscreteModel` 或其子类
2. 实现 `cdf()`、`pdf()` 方法（如适用）
3. 实现 `loglike()`、`loglikeobs()` 方法
4. 实现 `score()`、`hessian()` 方法（或使用数值微分）
5. 实现 `_derivative_exog()` 方法（或使用继承的实现）
6. 创建对应的结果类（如需要，继承 `DiscreteResults` 或其子类）

**添加新的边际效应类型**：
1. 在 `_check_margeff_args()` 中添加参数验证
2. 在 `_derivative_exog()` 中添加变换逻辑
3. 在 `DiscreteMargins` 中更新文档和汇总

**自定义优化算法**：
1. 在 `LikelihoodModel.fit()` 中添加新方法
2. 或通过 `fit()` 的 `method` 参数使用 scipy.optimize 的算法

## 8. 修正说明

本报告对初版报告中的以下两处事实偏差进行了修正：

### 8.1 修正 1：MultinomialModel 的继承关系

**初版错误**：
```
DiscreteModel
    ├── BinaryModel
    ├── MultinomialModel  ← 错误：直接继承自 DiscreteModel
    └── CountModel
```

**修正后**：
```
DiscreteModel
    ├── BinaryModel
    │       └── MultinomialModel  ← 正确：继承自 BinaryModel
    └── CountModel
```

**代码依据**：`discrete_model.py:758`
```python
class MultinomialModel(BinaryModel):
```

### 8.2 修正 2：计数模型的文件位置

**初版错误**：
- `NegativeBinomial` 在 `count_model.py` 中

**修正后**：
- 主要计数模型（`Poisson`、`NegativeBinomial`、`GeneralizedPoisson`、`NegativeBinomialP`）都在 `discrete_model.py` 中
- `count_model.py` 中只有 `GenericZeroInflated` 及相关的零膨胀模型

**代码依据**：
- `discrete_model.py:1338`：`class Poisson(CountModel)`
- `discrete_model.py:3597`：`class NegativeBinomial(CountModel)`
- `count_model.py:48`：`class GenericZeroInflated(CountModel)`

### 8.3 附加修正：结果类的继承关系

**重要说明**：
- 模型类：`MultinomialModel` 继承自 `BinaryModel`
- 结果类：`MultinomialResults` 直接继承自 `DiscreteResults`（而非 `BinaryResults`）

这种不对称设计是因为多分类模型的结果结构（多方程参数、多类别边际效应）与二元模型有显著差异。

**代码依据**：
- `discrete_model.py:758`：`class MultinomialModel(BinaryModel)`
- `discrete_model.py:5969`：`class MultinomialResults(DiscreteResults)`

## 9. 总结

Statsmodels 的离散选择模型采用了精心设计的面向对象架构：

### 9.1 准确的继承层次

```
LikelihoodModel
    └── DiscreteModel
            ├── BinaryModel
            │       ├── Logit
            │       ├── Probit
            │       └── MultinomialModel
            │               └── MNLogit
            │
            └── CountModel
                    ├── Poisson (discrete_model.py)
                    ├── NegativeBinomial (discrete_model.py)
                    └── GenericZeroInflated (count_model.py)
```

### 9.2 似然函数体系

通过模板方法模式，基类提供估计框架，子类实现具体分布的似然、得分和海森矩阵。

### 9.3 边际效应计算

- 连续变量：使用导数 `pdf(XB) * params`
- 离散变量：使用离散变化（0→1 或 ±1）
- 支持四种变换类型（dydx, eyex, dyex, eydx）
- 使用 Delta 方法计算标准误

### 9.4 设计模式

- 模板方法模式：定义估计流程框架
- 继承层次：实现代码复用和扩展
- 模型-结果分离：清晰的职责划分
- 装饰器模式：文档字符串复用
- 缓存机制：提高计算效率

这种设计使得 statsmodels 的离散选择模型既易于使用（统一的接口），又易于扩展（添加新模型只需实现特定方法）。
