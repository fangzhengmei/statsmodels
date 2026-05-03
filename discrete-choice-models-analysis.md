# Statsmodels 离散选择模型继承体系分析

## 1. 继承体系结构

### 1.1 模型类继承层次

```
base.LikelihoodModel (定义 loglike/score/hessian 抽象接口)
    └── DiscreteModel (定义 cdf/pdf/predict/_derivative_exog 抽象接口)
         ├── BinaryModel (二元选择模型通用框架)
         │    ├── Logit
         │    └── Probit
         ├── MultinomialModel (多项选择模型通用框架 - 继承 BinaryModel)
         │    └── MNLogit
         └── CountModel (计数模型通用框架)
              ├── Poisson
              ├── NegativeBinomial
              ├── NegativeBinomialP
              └── GeneralizedPoisson
```

### 1.2 结果类继承层次

```
base.LikelihoodModelResults
    └── DiscreteResults (结果基类 - 提供 summary/get_margeff 框架)
         ├── BinaryResults
         │    ├── LogitResults
         │    ├── ProbitResults
         │    └── L1BinaryResults
         ├── MultinomialResults
         │    └── L1MultinomialResults
         └── CountResults
              ├── PoissonResults
              ├── NegativeBinomialResults
              └── ...
```

### 1.3 边际效应类

```
DiscreteMargins (独立类，通过组合方式使用)
    - 接收 Results 实例
    - 调用 model._derivative_exog() 计算边际效应
    - 使用 Delta Method 计算标准误
    - 提供 summary()/summary_frame() 方法
```

## 2. 似然、梯度、Hessian 的特化层次（修正）

### 2.1 职责分工的准确理解

**关键修正**：`loglike()`, `score()`, `hessian()` 是在 **`LikelihoodModel`** 中定义的抽象接口，而非 `DiscreteModel`。`DiscreteModel` 只定义了 `cdf()`, `pdf()`, `predict()`, `_derivative_exog()` 等离散模型特有的抽象方法。

| 基类 | 定义的抽象方法 | 实现位置 |
|------|---------------|----------|
| `LikelihoodModel` | `loglike()`, `score()`, `hessian()` | 具体模型 (Logit, Probit, MNLogit) |
| `DiscreteModel` | `cdf()`, `pdf()`, `predict()`, `_derivative_exog()` | 中间层或具体模型 |

### 2.2 LikelihoodModel: MLE 框架的基石

**位置**：`base/model.py:279`

```python
class LikelihoodModel(Model):
    def loglike(self, params):
        """
        Log-likelihood of model.
        Must be overridden by subclasses.
        """
        raise NotImplementedError  # 抽象方法 - 必须实现

    def score(self, params):
        """
        Score vector of model.
        The gradient of logL with respect to each parameter.
        """
        raise NotImplementedError  # 抽象方法

    def hessian(self, params):
        """
        The Hessian matrix of the model.
        """
        raise NotImplementedError  # 抽象方法

    def fit(self, start_params=None, method="newton", maxiter=100, ...):
        """
        通用 MLE 拟合框架
        调用子类的 loglike(), score(), hessian()
        """
        # 核心优化循环
        def f(params, *args):
            return -self.loglike(params, *args) / nobs  # 调用子类 loglike
        
        if method == "newton":
            def score(params, *args):
                return self.score(params, *args) / nobs  # 调用子类 score
            def hess(params, *args):
                return self.hessian(params, *args) / nobs  # 调用子类 hessian
        ...
```

**职责**：
- 定义 `loglike()`, `score()`, `hessian()` 抽象接口
- 实现通用的 `fit()` 方法，使用 Newton-Raphson 或其他优化方法
- 封装优化器调用细节

### 2.3 DiscreteModel: 离散模型的抽象接口

**位置**：`discrete_model.py:185`

```python
class DiscreteModel(base.LikelihoodModel):
    """
    Abstract class for discrete choice models.
    This class does not do anything itself but lays out the methods and
    call signature expected of child classes.
    """

    def cdf(self, X):
        """
        The cumulative distribution function of the model.
        """
        raise NotImplementedError  # 离散模型特有：CDF

    def pdf(self, X):
        """
        The probability density (mass) function of the model.
        """
        raise NotImplementedError  # 离散模型特有：PDF

    def predict(self, params, exog=None, which="mean", linear=None):
        """
        Predict response variable of a model given exogenous variables.
        """
        raise NotImplementedError  # 离散模型特有：预测

    def _derivative_exog(self, params, exog=None, ...):
        """
        This should implement the derivative of the non-linear function
        （边际效应计算的核心方法）
        """
        raise NotImplementedError  # 离散模型特有：边际效应

    def fit(self, start_params=None, method="newton", maxiter=35, ...):
        """
        在 LikelihoodModel.fit() 基础上添加完全分离检测
        """
        if callback is None:
            callback = self._check_perfect_pred  # 添加分离检测回调
        
        mlefit = super().fit(...)  # 调用 LikelihoodModel.fit()
        return mlefit
```

**职责**：
- **不实现** `loglike()`, `score()`, `hessian()`（继承自 LikelihoodModel）
- 定义 `cdf()`, `pdf()` - 离散模型特有的分布函数
- 定义 `predict()`, `_derivative_exog()` - 离散模型特有的预测和边际效应
- 重写 `fit()` - 添加完全分离检测回调
- 提供 `_derivative_exog_helper()` - 处理虚拟变量和计数变量的边际效应

### 2.4 BinaryModel: 二元模型通用框架

**位置**：`discrete_model.py:521`

```python
class BinaryModel(DiscreteModel):
    _continuous_ok = False

    def predict(self, params, exog=None, which="mean", linear=None, offset=None):
        """
        二元模型通用预测逻辑 - Template Method 模式
        
        调用子类的 cdf() 方法
        """
        if which == "linear" or linear:
            return np.dot(exog, params) + offset
        
        # 核心：调用子类的 cdf()
        linpred = np.dot(exog, params) + offset
        if which == "mean":
            return self.cdf(linpred)  # 多态调用具体模型的 CDF
        elif which == "linear":
            return linpred
        ...

    def _derivative_exog(
        self, params, exog=None, transform="dydx", dummy_idx=None, count_idx=None, offset=None
    ):
        """
        二元模型边际效应计算 - Template Method 模式
        
        调用子类的 pdf() 方法
        公式：∂P/∂x = f(Xβ) * β
        其中 f(.) 是 pdf (Logit: Λ(1-Λ), Probit: φ)
        """
        linpred = self.predict(params, exog, offset=offset, which="linear")
        
        # 核心：调用子类的 pdf()
        margeff = np.dot(self.pdf(linpred)[:, None], params[None, :])
        
        # 变换处理（弹性/半弹性）
        if "ex" in transform:  # dy/d(lnx) 或 d(lny)/d(lnx)
            margeff *= exog
        if "ey" in transform:  # d(lny)/dx 或 d(lny)/d(lnx)
            margeff /= self.predict(params, exog)[:, None]
        
        # 处理离散变量（虚拟变量、计数变量）
        return self._derivative_exog_helper(
            margeff, params, exog, dummy_idx, count_idx, transform
        )
```

**职责**：
- **不实现** `loglike()`, `score()`, `hessian()`
- **实现** `predict()` - 通用预测框架，调用子类 `cdf()`
- **实现** `_derivative_exog()` - 通用边际效应框架，调用子类 `pdf()`
- 处理二元模型特有的 `offset` 参数

### 2.5 MultinomialModel: 多项模型通用框架

**位置**：`discrete_model.py:758`

```python
class MultinomialModel(BinaryModel):
    """
    多项模型基类
    
    注意：继承自 BinaryModel，但多项模型并不是二元模型的特例
    这是设计上的一个问题（见限制部分）
    """

    def initialize(self):
        """
        多项模型特有初始化：处理多类别数据结构
        """
        # 确保因变量是哑变量编码
        if self.endog.ndim == 1:
            # 将类别索引转换为哑变量
            endog_dummies = pd.get_dummies(self.endog).values
            self.wendog = endog_dummies.astype(float)
        else:
            self.wendog = np.asarray(self.endog)
        
        # 多项模型特有属性
        self.endog = self.endog.argmax(1)  # 转换为类别索引
        self.J = self.wendog.shape[1]     # 类别数量
        self.K = self.exog.shape[1]       # 解释变量数量
        self.df_model *= self.J - 1       # 自由度调整（多方程）

    def predict(self, params, exog=None, which="mean", linear=None):
        """
        多项模型预测：重写 BinaryModel.predict()
        """
        # 重塑参数：从扁平化到矩阵形式
        params = params.reshape(self.K, -1, order="F")  # (K, J-1)
        
        # 调用父类预测（但参数是矩阵形式）
        pred = super().predict(params, exog, which=which)
        
        # 多项模型特有：添加基准类别（全0）
        if which == "linear":
            # 在左侧添加基准类别（exp(0) = 1）
            pred = np.column_stack((np.zeros(len(exog)), pred))
        return pred

    def _derivative_exog(
        self, params, exog=None, transform="dydx", dummy_idx=None, count_idx=None
    ):
        """
        多项模型边际效应计算：重写 BinaryModel._derivative_exog()
        
        多项模型不能直接使用 pdf()，因为边际效应公式不同
        公式：∂P(j|x)/∂x_k = P(j|x) * [β_jk - Σ_l P(l|x) β_lk]
        """
        J = int(self.J)
        K = int(self.K)
        
        # 重塑参数
        params = params.reshape(K, -1, order="F")  # (K, J-1)
        
        # 添加基准类别参数（全0）
        zeroparams = np.c_[np.zeros(K), params]  # (K, J)
        
        # 计算所有类别的概率（使用 cdf()）
        cdf = self.cdf(np.dot(exog, params))  # (nobs, J)
        
        # 计算加权平均参数：Σ_l P(l|x) β_l
        # iterm[k] = Σ_j P(j|x) β_jk
        iterm = np.array([cdf[:, [i]] * zeroparams[:, i] for i in range(int(J))]).sum(0)
        
        # 计算每个类别的边际效应
        # ∂P(j)/∂x_k = P(j) * (β_jk - iterm[k])
        margeff = np.array([cdf[:, [j]] * (zeroparams[:, j] - iterm) for j in range(J)])
        
        # 调整维度：(J, nobs, K) -> (nobs, K, J)
        margeff = np.transpose(margeff, (1, 2, 0))
        
        # 变换处理
        if "ex" in transform:
            margeff *= exog
        if "ey" in transform:
            margeff /= self.predict(params, exog)[:, None, :]
        
        # 扁平化以便统一处理
        return margeff.reshape(len(exog), -1, order="F")
```

**职责**：
- **不实现** `loglike()`, `score()`, `hessian()`
- 重写 `initialize()` - 处理多类别数据结构
- 重写 `predict()` - 调整多方程参数形状
- 重写 `_derivative_exog()` - 多项模型边际效应公式不同
  - **不依赖** `pdf()` 方法
  - 直接使用 `cdf()` 计算概率，然后应用多项 Logit 边际效应公式

### 2.6 具体模型层：Logit, Probit, MNLogit

**这是唯一实现 `loglike()`, `score()`, `hessian()` 的层次**。

#### Logit 模型 (`discrete_model.py:2622`)

```python
class Logit(BinaryModel):
    """
    Logistic 回归模型
    
    实现：cdf(), pdf(), loglike(), score(), hessian()
    """

    def cdf(self, X):
        """
        Logistic 累积分布函数
        Λ(x) = 1 / (1 + exp(-x))
        """
        return 1 / (1 + np.exp(-X))

    def pdf(self, X):
        """
        Logistic 概率密度函数
        λ(x) = Λ(x) * (1 - Λ(x))
        """
        return np.exp(-X) / (1 + np.exp(-X)) ** 2

    def loglike(self, params):
        """
        对数似然函数
        ln L = Σ ln Λ(q_i x_i'β), 其中 q = 2y - 1
        """
        q = 2 * self.endog - 1
        linpred = self.predict(params, which="linear")
        return np.sum(np.log(self.cdf(q * linpred)))

    def score(self, params):
        """
        梯度 (Score)
        ∂ln L/∂β = Σ (y_i - Λ_i) x_i
        """
        y = self.endog
        X = self.exog
        fitted = self.predict(params)  # Λ(Xβ)
        return np.dot(y - fitted, X)

    def hessian(self, params):
        """
        Hessian 矩阵
        ∂²ln L/∂β∂β' = -Σ Λ_i(1-Λ_i) x_i x_i'
        """
        X = self.exog
        L = self.predict(params)
        return -np.dot(L * (1 - L) * X.T, X)
```

#### Probit 模型 (`discrete_model.py:2929`)

```python
class Probit(BinaryModel):
    """
    Probit 模型
    
    实现：cdf(), pdf(), loglike(), score(), hessian()
    """

    def cdf(self, X):
        """
        标准正态累积分布函数 Φ(x)
        """
        return stats.norm._cdf(X)

    def pdf(self, X):
        """
        标准正态概率密度函数 φ(x)
        """
        return stats.norm._pdf(X)

    def loglike(self, params):
        """
        对数似然函数
        ln L = Σ ln Φ(q_i x_i'β), 其中 q = 2y - 1
        """
        q = 2 * self.endog - 1
        linpred = self.predict(params, which="linear")
        return np.sum(np.log(np.clip(self.cdf(q * linpred), FLOAT_EPS, 1)))

    def score(self, params):
        """
        梯度 (Score)
        ∂ln L/∂β = Σ [q_i φ(q_i x_i'β) / Φ(q_i x_i'β)] x_i
        其中 λ_i = q_i φ(q_i x_i'β) / Φ(q_i x_i'β) 称为逆米尔斯比率
        """
        y = self.endog
        X = self.exog
        XB = self.predict(params, which="linear")
        q = 2 * y - 1
        L = q * self.pdf(q * XB) / np.clip(self.cdf(q * XB), FLOAT_EPS, 1 - FLOAT_EPS)
        return np.dot(L, X)

    def hessian(self, params):
        """
        Hessian 矩阵
        ∂²ln L/∂β∂β' = -Σ λ_i(λ_i + x_i'β) x_i x_i'
        """
        X = self.exog
        XB = self.predict(params, which="linear")
        q = 2 * self.endog - 1
        L = q * self.pdf(q * XB) / self.cdf(q * XB)
        return np.dot(-L * (L + XB) * X.T, X)
```

#### MNLogit 模型 (`discrete_model.py:3255`)

```python
class MNLogit(MultinomialModel):
    """
    多项 Logit 模型
    
    实现：cdf(), loglike(), score(), hessian()
    注意：没有实现 pdf()，因为多项模型边际效应不使用 pdf
    """

    def cdf(self, X):
        """
        多项 Logit 的 CDF (Softmax 函数)
        P(j|x) = exp(β_j'x) / Σ_k exp(β_k'x), 其中 β_0 = 0 (基准类别)
        
        参数 X: 形状为 (nobs, J-1)，不包含基准类别
        返回: 形状为 (nobs, J)，包含所有类别的概率
        """
        # 在左侧添加基准类别 (exp(0) = 1)
        eXB = np.column_stack((np.ones(len(X)), np.exp(X)))
        return eXB / eXB.sum(1)[:, None]  # Softmax

    # 注意：MNLogit 没有实现 pdf() 方法！
    # 因为多项模型的边际效应计算不使用 pdf，见 MultinomialModel._derivative_exog()

    def loglike(self, params):
        """
        对数似然函数
        ln L = Σ_i Σ_j d_ij ln [exp(β_j'x_i) / Σ_k exp(β_k'x_i)]
        其中 d_ij = 1 如果个体 i 选择类别 j
        """
        params = params.reshape(self.K, -1, order="F")  # (K, J-1)
        d = self.wendog  # 哑变量编码的因变量
        logprob = np.log(self.cdf(np.dot(self.exog, params)))
        return np.sum(d * logprob)

    def score(self, params):
        """
        梯度 (Score)
        ∂ln L/∂β_j = Σ_i (d_ij - P(j|x_i)) x_i, 对于 j = 1,...,J-1
        
        返回扁平化数组：形状为 K*(J-1)
        """
        params = params.reshape(self.K, -1, order="F")
        firstterm = self.wendog[:, 1:] - self.cdf(np.dot(self.exog, params))[:, 1:]
        return np.dot(firstterm.T, self.exog).flatten()

    def hessian(self, params):
        """
        Hessian 矩阵
        
        ∂²ln L/∂β_j ∂β_l = -Σ_i P(j|x_i)[1(j=l) - P(l|x_i)] x_i x_i'
        
        特殊结构：
        - 当 j = l 时：-Σ_i P(j|x_i)(1-P(j|x_i)) x_i x_i'
        - 当 j ≠ l 时：Σ_i P(j|x_i)P(l|x_i) x_i x_i'
        """
        params = params.reshape(self.K, -1, order="F")
        X = self.exog
        pr = self.cdf(np.dot(X, params))  # (nobs, J)
        partials = []
        J = self.J
        K = self.K
        
        for i in range(J - 1):
            for j in range(J - 1):
                if i == j:
                    # 对角块
                    partials.append(
                        -np.dot(((pr[:, i + 1] * (1 - pr[:, j + 1]))[:, None] * X).T, X)
                    )
                else:
                    # 非对角块
                    partials.append(
                        -np.dot(((pr[:, i + 1] * -pr[:, j + 1])[:, None] * X).T, X)
                    )
        
        # 重塑为 ((J-1)*K, (J-1)*K) 的方阵
        H = np.array(partials)
        H = np.transpose(H.reshape(J - 1, J - 1, K, K), (0, 2, 1, 3)).reshape(
            (J - 1) * K, (J - 1) * K
        )
        return H
```

### 2.7 特化层次总结（修正版）

#### 各层职责矩阵

| 方法 | LikelihoodModel | DiscreteModel | BinaryModel | MultinomialModel | Logit/Probit | MNLogit |
|------|-----------------|---------------|-------------|------------------|--------------|---------|
| `loglike()` | `NotImplementedError` | 继承（不实现） | 继承（不实现） | 继承（不实现） | **实现** | **实现** |
| `score()` | `NotImplementedError` | 继承（不实现） | 继承（不实现） | 继承（不实现） | **实现** | **实现** |
| `hessian()` | `NotImplementedError` | 继承（不实现） | 继承（不实现） | 继承（不实现） | **实现** | **实现** |
| `cdf()` | - | `NotImplementedError` | 继承（不实现） | 继承（不实现） | **实现** | **实现** |
| `pdf()` | - | `NotImplementedError` | 继承（不实现） | 继承（不实现） | **实现** | ❌ 不实现 |
| `predict()` | - | `NotImplementedError` | **实现**（调用 cdf） | **重写**（多方程） | 继承 | 继承 |
| `_derivative_exog()` | - | `NotImplementedError` | **实现**（调用 pdf） | **重写**（不调用 pdf） | 继承 | 继承 |
| `fit()` | **实现**（MLE 框架） | **重写**（分离检测） | 继承 | 继承 | 继承 | 继承 |

#### 关键理解

1. **`loglike()`, `score()`, `hessian()` 是在最底层特化的**：
   - 这些方法定义在 `LikelihoodModel` 中
   - 所有中间层（`DiscreteModel`, `BinaryModel`, `MultinomialModel`）都不实现这些方法
   - 只有具体模型（`Logit`, `Probit`, `MNLogit`）才实现这些方法

2. **`cdf()` 和 `pdf()` 是离散模型特有的抽象方法**：
   - 定义在 `DiscreteModel` 中
   - 二元模型需要同时实现 `cdf()` 和 `pdf()`
   - **`MNLogit` 只实现 `cdf()`，不实现 `pdf()`**（因为多项模型边际效应不使用 pdf）

3. **`predict()` 和 `_derivative_exog()` 在中间层实现**：
   - `BinaryModel` 实现通用框架，调用子类的 `cdf()` 和 `pdf()`
   - `MultinomialModel` 重写这些方法，因为多项模型的公式不同

#### 调用链分析

**拟合过程**：
```
Logit.fit() 或 Probit.fit() 或 MNLogit.fit()
         │
         ▼ (继承自 DiscreteModel.fit())
DiscreteModel.fit()
         │
         ├── 添加 callback: _check_perfect_pred
         │
         ▼ (调用 LikelihoodModel.fit())
LikelihoodModel.fit()
         │
         ├── 定义优化目标函数: -loglike(params) / nobs
         ├── 定义梯度函数: score(params) / nobs 或 -score(params) / nobs
         ├── 定义 Hessian 函数: hessian(params) / nobs 或 -hessian(params) / nobs
         │
         └── 调用 scipy.optimize 中的优化器
              └── 优化器回调:
                  ├── model.loglike(params)  <-- 多态调用具体模型的 loglike
                  ├── model.score(params)    <-- 多态调用具体模型的 score
                  └── model.hessian(params)  <-- 多态调用具体模型的 hessian
```

**预测过程**：
```
Logit.predict() 或 Probit.predict() 或 MNLogit.predict()
         │
         ▼ (多态调用)
         ├── Logit/Probit: 继承 BinaryModel.predict()
         │         │
         │         ▼
         │    BinaryModel.predict()
         │         │
         │         ├── 计算线性预测: Xβ + offset
         │         └── 调用 self.cdf(linpred)
         │              │
         │              ├── Logit.cdf()  --> 1/(1+exp(-x))
         │              └── Probit.cdf() --> Φ(x) (正态 CDF)
         │
         └── MNLogit: 继承 MultinomialModel.predict()
                   │
                   ▼
              MultinomialModel.predict()
                   │
                   ├── 重塑参数: (K,) -> (K, J-1)
                   ├── 调用 BinaryModel.predict() (处理线性预测)
                   └── 添加基准类别: 在左侧添加全0列
                        │
                        └── 调用 MNLogit.cdf() --> Softmax 函数
```

**边际效应计算**：
```
BinaryModel._derivative_exog()          MultinomialModel._derivative_exog()
         │                                          │
         ├── 计算线性预测 Xβ                         ├── 重塑参数 (K,) -> (K, J-1)
         │                                          ├── 添加基准类别参数 (全0)
         ├── 调用 self.pdf(linpred)                 ├── 调用 self.cdf() 计算概率 P(j|x)
         │    │                                     ├── 计算加权平均参数 Σ P(l)β_l
         │    ├── Logit.pdf() --> Λ(1-Λ)           ├── 应用公式: ∂P(j)/∂x = P(j)*(β_j - Σ P(l)β_l)
         │    └── Probit.pdf() --> φ(x)            └── 调整维度形状
         │
         ├── 应用公式: f(Xβ) * β
         ├── 处理变换 (弹性/半弹性)
         └── 调用 _derivative_exog_helper()
              └── 处理虚拟变量/计数变量的离散变化
```

## 3. 边际效应的统一接口（修正）

### 3.1 完整的挂接关系

```
用户调用 results.get_margeff(at, method, ...)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ DiscreteResults.get_margeff() (discrete_model.py:5293)         │
├─────────────────────────────────────────────────────────────────┤
│ def get_margeff(self, at="overall", method="dydx", ...):       │
│     from statsmodels.discrete.discrete_margins import DiscreteMargins│
│     return DiscreteMargins(self, (at, method, atexog, dummy, count))│
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ DiscreteMargins.__init__() (discrete_margins.py:433)          │
├─────────────────────────────────────────────────────────────────┤
│ def __init__(self, results, args, kwargs=None):                 │
│     self.results = results                                       │
│     self.get_margeff(*args, **kwargs)  # 立即计算边际效应        │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ DiscreteMargins.get_margeff() (discrete_margins.py:487)       │
├─────────────────────────────────────────────────────────────────┤
│ 1. 准备 exog (根据 at 参数):                                      │
│    - at='overall': 使用所有观测，然后取平均                      │
│    - at='mean': 在解释变量均值处                                  │
│    - at='median': 在解释变量中位数处                              │
│    - at='zero': 在解释变量为零处                                  │
│    - at='all': 返回每个观测的边际效应                             │
│                                                                   │
│ 2. 处理 dummy/count 参数:                                         │
│    - 识别虚拟变量索引 (常数列以外的变量)                          │
│    - 识别计数变量索引                                             │
│                                                                   │
│ 3. 核心计算:                                                      │
│    margeff = model._derivative_exog(                             │
│        params, exog, transform=method,                           │
│        dummy_idx=dummy_ind, count_idx=count_ind                 │
│    )                                                              │
│    ┌──────────────────────────────────────────────────────────┐  │
│    │ 多态调用: 取决于模型类型                                   │  │
│    │ ├── BinaryModel._derivative_exog() -> 调用 pdf()        │  │
│    │ └── MultinomialModel._derivative_exog() -> 不调用 pdf() │  │
│    └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│ 4. 计算标准误 (Delta Method):                                    │
│    self.margeff_se = np.sqrt(np.diag(self.cov_margins))        │
│    其中 cov_margins = margeff_cov_params(...)                   │
│         数值计算 Jacobian: ∂ margeff / ∂ params                 │
│         应用公式: J * cov_params * J^T                           │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 二元模型边际效应的完整流程

**BinaryModel._derivative_exog** (`discrete_model.py:670`):

```python
def _derivative_exog(
    self, params, exog=None, transform="dydx", dummy_idx=None, count_idx=None, offset=None
):
    """
    二元模型边际效应计算
    
    核心公式：
    ∂P(y=1|x)/∂x_k = f(Xβ) * β_k
    其中 f(.) 是 pdf:
      - Logit: f(x) = Λ(x)(1-Λ(x))
      - Probit: f(x) = φ(x) (正态密度)
    """
    # 步骤 1: 计算线性预测 Xβ
    linpred = self.predict(params, exog, offset=offset, which="linear")
    
    # 步骤 2: 调用子类的 pdf() 计算 f(Xβ)
    #         Logit: self.pdf(linpred) = Λ(1-Λ)
    #         Probit: self.pdf(linpred) = φ(Xβ)
    margeff = np.dot(self.pdf(linpred)[:, None], params[None, :])
    
    # 步骤 3: 处理变换（弹性/半弹性）
    if "ex" in transform:  # dy/d(lnx) 或 d(lny)/d(lnx)
        # 对于对数变换的 x，dy/d(lnx) = x * dy/dx
        margeff *= exog
    if "ey" in transform:  # d(lny)/dx 或 d(lny)/d(lnx)
        # d(lny)/dx = (1/y) * dy/dx
        margeff /= self.predict(params, exog)[:, None]
    
    # 步骤 4: 处理离散变量的特殊情况
    return self._derivative_exog_helper(
        margeff, params, exog, dummy_idx, count_idx, transform
    )
```

**_derivative_exog_helper** (`discrete_model.py:501`):

```python
def _derivative_exog_helper(
    self, margeff, params, exog, dummy_idx, count_idx, transform
):
    """
    处理虚拟变量和计数变量的边际效应
    
    对于虚拟变量：边际效应是离散变化 ΔP = P(X|d=1) - P(X|d=0)，而非导数
    对于计数变量：边际效应是单位变化 ΔP = P(X|x+1) - P(X|x)
    """
    from .discrete_margins import _get_count_effects, _get_dummy_effects
    
    if count_idx is not None:
        # 计数变量：计算离散变化
        margeff = _get_count_effects(
            margeff, exog, count_idx, transform, self, params
        )
    
    if dummy_idx is not None:
        # 虚拟变量：计算离散变化
        margeff = _get_dummy_effects(
            margeff, exog, dummy_idx, transform, self, params
        )
    
    return margeff
```

**_get_dummy_effects** (`discrete_margins.py:177`):

```python
def _get_dummy_effects(effects, exog, dummy_ind, method, model, params):
    """
    虚拟变量的边际效应：离散变化而非导数
    
    公式：ΔP = P(X|d=1) - P(X|d=0)
    """
    for i in dummy_ind:
        # 设置 d=0
        exog0 = exog.copy()
        exog0[:, i] = 0
        effect0 = model.predict(params, exog0)
        
        # 设置 d=1
        exog0[:, i] = 1
        effect1 = model.predict(params, exog0)
        
        # 处理弹性变换
        if "ey" in method:
            effect0 = np.log(effect0)
            effect1 = np.log(effect1)
        
        # 替换导数为离散变化
        effects[:, i] = effect1 - effect0
    
    return effects
```

### 3.3 多项模型边际效应的完整流程

**MultinomialModel._derivative_exog** (`discrete_model.py:979`):

```python
def _derivative_exog(
    self, params, exog=None, transform="dydx", dummy_idx=None, count_idx=None
):
    """
    多项 Logit 边际效应计算
    
    关键注意：多项模型的边际效应公式与二元模型完全不同
    不使用 pdf() 方法！
    
    公式：
    ∂P(j|x)/∂x_k = P(j|x) * [β_jk - Σ_l P(l|x) β_lk]
    
    其中：
    - P(j|x) 是选择类别 j 的概率
    - β_jk 是类别 j 对应变量 k 的参数
    - β_0k = 0（基准类别参数为0）
    
    推导：
    P(j|x) = exp(β_j'x) / Σ_l exp(β_l'x) = e^z_j / Σ_l e^z_l
    
    ∂P(j)/∂z_k = P(j)*(1(j=k) - P(k))
    
    ∂P(j)/∂x = ∂P(j)/∂z * ∂z/∂x = P(j)*(δ_j - P) β
    = P(j)*(β_j - Σ_l P(l)β_l)
    """
    J = int(self.J)  # 类别数
    K = int(self.K)  # 变量数
    
    # 步骤 1: 重塑参数
    params = params.reshape(K, -1, order="F")  # (K, J-1)
    
    # 步骤 2: 添加基准类别参数（全0）
    # zeroparams[:, 0] = 0（基准类别参数为0）
    # zeroparams[:, 1:] = params（非基准类别参数）
    zeroparams = np.c_[np.zeros(K), params]  # (K, J)
    
    # 步骤 3: 计算所有类别的概率 P(j|x)
    # 使用 cdf() (Softmax 函数)
    cdf = self.cdf(np.dot(exog, params))  # (nobs, J)
    
    # 步骤 4: 计算加权平均参数
    # iterm[k] = Σ_j P(j|x) β_jk
    # 这是概率加权的"平均"参数
    iterm = np.array([cdf[:, [i]] * zeroparams[:, i] for i in range(int(J))]).sum(0)
    
    # 步骤 5: 应用多项 Logit 边际效应公式
    # ∂P(j)/∂x_k = P(j) * (β_jk - iterm[k])
    margeff = np.array([cdf[:, [j]] * (zeroparams[:, j] - iterm) for j in range(J)])
    
    # 步骤 6: 调整维度
    # 从 (J, nobs, K) 转换为 (nobs, K, J)
    # 这样更容易按观测和变量索引
    margeff = np.transpose(margeff, (1, 2, 0))
    
    # 步骤 7: 处理变换（弹性/半弹性）
    if "ex" in transform:
        margeff *= exog
    if "ey" in transform:
        margeff /= self.predict(params, exog)[:, None, :]
    
    # 步骤 8: 扁平化以便统一处理
    # 使用 Fortran 顺序：先变量，后类别
    return margeff.reshape(len(exog), -1, order="F")
```

### 3.4 边际效应标准误计算（Delta Method）

**margeff_cov_params** (`discrete_margins.py:271`):

```python
def margeff_cov_params(
    model, params, exog, cov_params, at, derivative, dummy_ind, count_ind, method, J
):
    """
    使用 Delta 方法计算边际效应的方差-协方差矩阵
    
    理论：
    设 ME = g(β) 是边际效应，是参数 β 的函数
    
    渐近方差：
    Asy.Var[ME] = [∂g(β)/∂β] V(β) [∂g(β)/∂β]^T
    
    其中：
    - V(β) 是参数的方差-协方差矩阵 (results.cov_params())
    - ∂g(β)/∂β 是边际效应对参数的 Jacobian 矩阵
    
    由于边际效应没有解析 Jacobian，使用数值微分近似：
    ∂g/∂β_k ≈ [g(β + h*e_k) - g(β - h*e_k)] / (2h)
    """
    if callable(derivative):
        # derivative 是 model._derivative_exog 函数
        
        # 步骤 1: 数值计算 Jacobian: ∂ margeff / ∂ params
        # 使用复步微分 (complex step differentiation)
        # 比有限差分更精确
        jacobian_mat = approx_fprime_cs(params, derivative, args=(exog, method))
        
        # 步骤 2: 处理 at 参数
        if at == "overall":
            # 平均边际效应：对所有观测取平均
            jacobian_mat = np.mean(jacobian_mat, axis=1)
        
        # 步骤 3: 处理离散变量的 Jacobian
        # 虚拟变量和计数变量的边际效应是离散变化，
        # 其 Jacobian 计算方式不同
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
    
    # 步骤 4: 应用 Delta Method 公式
    # Asy.Var[ME] = J * V * J^T
    return np.dot(np.dot(jacobian_mat, cov_params), jacobian_mat.T)
```

### 3.5 二元 vs 多项边际效应对比

| 维度 | 二元模型 (Logit/Probit) | 多项模型 (MNLogit) |
|------|------------------------|---------------------|
| **参数形状** | 1D: (K,) | 2D: (K, J-1) 或扁平化 (K*(J-1),) |
| **边际效应公式** | ∂P/∂x = f(Xβ) * β | ∂P(j)/∂x = P(j) * (β_j - Σ P(l)β_l) |
| **依赖方法** | `pdf()` 方法 | `cdf()` 方法（不使用 `pdf()`） |
| **输出形状** | (nobs, K) 或 (K,) | (nobs, K, J) 或 (K, J) |
| **解释方式** | 直接解释为概率变化 | 每个类别的概率变化单独解释 |
| **基准类别** | 无 | 必须指定（参数为 0） |

### 3.6 method 参数的完整解释

```python
method 参数控制边际效应的类型：

1. "dydx" - 边际效应 (dy/dx)
   - 二元模型：∂P(y=1|x)/∂x_k
   - 多项模型：∂P(j|x)/∂x_k

2. "eyex" - 弹性 (d(lny)/d(lnx))
   - 解释：x 变化 1%，y 变化的百分比
   - 公式：(∂y/∂x) * (x/y)
   - 实现：margeff *= exog; margeff /= y

3. "dyex" - 半弹性 (dy/d(lnx))
   - 解释：x 变化 1%，y 变化的绝对值
   - 公式：(∂y/∂x) * x
   - 实现：margeff *= exog

4. "eydx" - 半弹性 (d(lny)/dx)
   - 解释：x 变化 1 单位，y 变化的百分比
   - 公式：(∂y/∂x) / y
   - 实现：margeff /= y
```

### 3.7 at 参数的完整解释

```python
at 参数控制边际效应的计算位置：

1. "overall" - 平均边际效应 (AME)
   - 对每个观测计算边际效应，然后取平均
   - 最常用，最具代表性
   - E[∂P/∂x] = (1/n) Σ_i ∂P(y_i|x_i)/∂x

2. "mean" - 在均值处的边际效应 (MEM)
   - 令 x = E[x]（样本均值），然后计算边际效应
   - ∂P(y|x=μ)/∂x
   - 计算简单，但可能不具代表性

3. "median" - 在中位数处的边际效应
   - 令 x = median(x)，然后计算边际效应
   - 对异常值更稳健

4. "zero" - 在零点处的边际效应
   - 令 x = 0，然后计算边际效应
   - 仅当 x=0 有意义时使用

5. "all" - 每个观测的边际效应
   - 返回所有观测的边际效应，不聚合
   - 用于分析异质性
```

## 4. 摘要格式化机制（修正）

### 4.1 结果类的挂接关系

```
用户调用 results.summary()
         │
         ▼
┌───────────────────────────────────────────────────────────────┐
│ DiscreteResults.summary() (discrete_model.py:5390)           │
│ （通用框架 - 所有离散模型共享）                                  │
├───────────────────────────────────────────────────────────────┤
│ 步骤 1: 构建顶部信息列表                                        │
│   top_left = [                                                 │
│       ("Dep. Variable:", None),     # 因变量名                 │
│       ("Model:", [self.model.__class__.__name__]),  # 动态获取│
│       ("Method:", [self.method]),    # 估计方法 (MLE)         │
│       ("Date:", None),                # 日期                   │
│       ("Time:", None),                # 时间                   │
│       ("converged:", [converged]),    # 是否收敛              │
│   ]                                                             │
│                                                                 │
│   top_right = [                                                │
│       ("No. Observations:", None),  # 观测数                  │
│       ("Df Residuals:", None),      # 残差自由度              │
│       ("Df Model:", None),          # 模型自由度              │
│       ("Pseudo R-squ.:", [...]),    # 伪 R²                  │
│       ("Log-Likelihood:", None),    # 对数似然                │
│       ("LL-Null:", [...]),           # 零模型对数似然         │
│       ("LLR p-value:", [...]),       # 似然比检验 p 值        │
│   ]                                                             │
│                                                                 │
│ 步骤 2: 创建 Summary 对象                                      │
│   from statsmodels.iolib.summary import Summary               │
│   smry = Summary()                                             │
│                                                                 │
│ 步骤 3: 添加双列表格（顶部信息）                                │
│   smry.add_table_2cols(                                        │
│       self, gleft=top_left, gright=top_right, ...            │
│   )                                                             │
│                                                                 │
│ 步骤 4: 添加参数表格                                            │
│   smry.add_table_params(                                       │
│       self, yname=yname_list, xname=xname, ...                │
│   )                                                             │
│   # 参数表格包含：                                               │
│   # - coef: 参数估计值                                          │
│   # - std err: 标准误                                           │
│   # - z: z 统计量                                               │
│   # - P>|z|: p 值                                               │
│   # - [0.025: 95% 置信区间下限                                  │
│   # - 0.975]: 95% 置信区间上限                                  │
└───────────────────────────────────────────────────────────────┘
         │
         ▼ (子类扩展 - 多态调用)
         ├── BinaryResults.summary() - 添加分离检测警告
         ├── MultinomialResults.summary() - 无特殊扩展
         └── MultinomialResults.summary2() - 多方程显示（非标准 summary）
```

### 4.2 BinaryResults 的扩展：分离检测

**BinaryResults.summary()** (`discrete_model.py:5780`):

```python
@Appender(DiscreteResults.summary.__doc__)
def summary(self, yname=None, xname=None, title=None, alpha=0.05, yname_list=None):
    """
    二元模型摘要：添加完全/准完全分离检测警告
    """
    # 步骤 1: 调用父类的通用摘要
    smry = super().summary(yname, xname, title, alpha, yname_list)
    
    # 步骤 2: 检测完全/准完全分离
    # 分离问题：某些预测变量可以完美预测因变量
    # 这会导致 MLE 不存在（趋向无穷）
    
    # 计算拟合概率
    fittedvalues = self.model.cdf(self.fittedvalues)
    
    # 计算预测误差
    absprederror = np.abs(self.model.endog - fittedvalues)
    
    # 统计可以"完美预测"的观测数
    predclose_sum = (absprederror < 1e-4).sum()
    predclose_frac = predclose_sum / len(fittedvalues)
    
    etext = []
    
    # 完全分离：所有观测都可以完美预测
    if predclose_sum == len(fittedvalues):
        wstr = "Complete Separation: The results show that there is"
        wstr += "complete separation or perfect prediction.\n"
        wstr += "In this case the Maximum Likelihood Estimator does "
        wstr += "not exist and the parameters\n"
        wstr += "are not identified."
        etext.append(wstr)
    
    # 准完全分离：超过 10% 的观测可以完美预测
    elif predclose_frac > 0.1:
        wstr = "Possibly complete quasi-separation: A fraction "
        wstr += "%4.2f of observations can be\n" % predclose_frac
        wstr += "perfectly predicted. This might indicate that there "
        wstr += "is complete\nquasi-separation. In this case some "
        wstr += "parameters will not be identified."
        etext.append(wstr)
    
    # 步骤 3: 添加警告文本到摘要
    if etext:
        smry.add_extra_txt(etext)
    
    return smry
```

### 4.3 MultinomialResults 的特化

**MultinomialResults** (`discrete_model.py:5969`):

```python
class MultinomialResults(DiscreteResults):
    """
    多项模型结果类
    
    由于多项模型有 J-1 个方程（对应 J 个类别），
    参数、标准误、置信区间等都是多维的，需要特化处理。
    """

    @cache_readonly
    def bse(self):
        """
        标准误：重塑为矩阵形式
        
        多项模型的 params 是矩阵 (K, J-1)
        但 cov_params() 返回的是扁平化后的协方差矩阵
        需要重塑为与 params 相同的形状
        """
        bse = np.sqrt(np.diag(self.cov_params()))
        # 使用 Fortran 顺序重塑
        return bse.reshape(self.params.shape, order="F")

    @cache_readonly
    def tvalues(self):
        """
        t 统计量：元素级除法
        
        由于 params 和 bse 都是矩阵 (K, J-1)，
        直接使用元素级除法
        """
        return self.params / self.bse

    @cache_readonly
    def pvalues(self):
        """
        p 值：扁平化后计算，再重塑
        
        t 分布的生存函数
        """
        pvalues = stats.norm.sf(np.abs(self.tvalues.ravel())) * 2
        return pvalues.reshape(self.params.shape, order="F")

    def conf_int(self, alpha=0.05, cols=None):
        """
        置信区间：形状调整
        
        二元模型返回 (K, 2)
        多项模型返回 (J, K, 2) - 每个方程一个 (K, 2)
        """
        # 调用父类方法
        confint = super(DiscreteResults, self).conf_int(alpha=alpha, cols=cols)
        # 调整维度顺序
        return confint.transpose(2, 0, 1)

    def summary2(self, alpha=0.05, float_format="%.4f"):
        """
        summary2：多方程显示
        
        与标准 summary() 不同，summary2 将每个方程单独显示
        """
        from statsmodels.iolib import summary2
        
        smry = summary2.Summary()
        smry.add_dict(summary2.summary_model(self))
        
        eqn = self.params.shape[1]  # J-1 个方程
        confint = self.conf_int(alpha)
        
        for i in range(eqn):
            # 每个方程一个表格
            coefs = summary2.summary_params(
                (self, self.params[:, i], self.bse[:, i], 
                 self.tvalues[:, i], self.pvalues[:, i], confint[i]),
                alpha=alpha,
            )
            level_str = self.model.endog_names + " = " + str(i)
            coefs[level_str] = coefs.index
            smry.add_df(coefs, index=False, header=True)
        
        return smry
```

### 4.4 边际效应摘要格式化

**DiscreteMargins.summary()** (`discrete_margins.py:567`):

```python
def summary(self, alpha=0.05):
    """
    边际效应摘要表格
    
    注意：边际效应摘要与模型参数摘要不同，
    不使用标准的 add_table_params() 方法，
    因为 add_table_params() 假设参数是一维的。
    """
    from statsmodels.iolib.summary import Summary, summary_params, table_extend
    
    # 步骤 1: 准备标题和顶部信息
    results = self.results
    model = results.model
    title = model.__class__.__name__ + " Marginal Effects"
    method = self.margeff_options["method"]
    
    top_left = [
        ("Dep. Variable:", [model.endog_names]),
        ("Method:", [method]),  # dydx, eyex, dyex, eydx
        ("At:", [self.margeff_options["at"]]),  # overall, mean, etc.
    ]
    
    # 步骤 2: 创建 Summary 对象
    exog_names = model.exog_names[:]
    smry = Summary()
    
    # 移除常数项（边际效应对常数项无意义）
    _, const_idx = _get_const_index(model.exog)
    if const_idx is not None:
        exog_names.pop(const_idx[0])
    if getattr(model, "k_extra", 0) > 0:
        exog_names = exog_names[: -model.k_extra]
    
    # 步骤 3: 获取类别信息
    J = int(getattr(model, "J", 1))  # 类别数（二元模型 J=1）
    
    if J > 1:
        _, yname_list = results._get_endog_name(model.endog_names, None, all=True)
    else:
        yname = model.endog_names
        yname_list = [yname]
    
    # 步骤 4: 添加顶部信息
    smry.add_table_2cols(
        self, gleft=top_left, gright=[], yname=yname, xname=exog_names, title=title
    )
    
    # 步骤 5: 构建参数表格
    # 注意：不使用 add_table_params()，因为它不够通用
    table = []
    conf_int = self.conf_int(alpha)
    margeff = self.margeff
    margeff_se = self.margeff_se
    tvalues = self.tvalues
    pvalues = self.pvalues
    
    if J > 1:
        # 多项模型：每个类别一个表格
        for eq in range(J):
            restup = (
                results,
                margeff[:, eq],
                margeff_se[:, eq],
                tvalues[:, eq],
                pvalues[:, eq],
                conf_int[:, :, eq],
            )
            tble = summary_params(
                restup,
                yname=yname_list[eq],  # 类别名称
                xname=exog_names,
                alpha=alpha,
                use_t=False,
                skip_header=True,
            )
            table.append(tble)
        
        # 合并所有表格
        table = table_extend(table, keep_headers=True)
    else:
        # 二元模型：单个表格
        restup = (results, margeff, margeff_se, tvalues, pvalues, conf_int)
        table = summary_params(
            restup, yname=yname, xname=exog_names, alpha=alpha, use_t=False, skip_header=True
        )
    
    smry.tables.append(table)
    return smry
```

### 4.5 摘要格式化的完整调用链

```
模型参数摘要：
results.summary()
    │
    ▼
DiscreteResults.summary()  (通用框架)
    │
    ├── 创建 Summary 对象
    ├── add_table_2cols() - 顶部信息
    │         │
    │         └── 动态获取模型名: self.model.__class__.__name__
    │
    ├── add_table_params() - 参数表格
    │         │
    │         └── 使用 results.params, results.bse 等属性
    │
    └── (子类扩展)
         │
         └── BinaryResults.summary()
              │
              └── 添加分离检测警告 (add_extra_txt)


边际效应摘要：
results.get_margeff().summary()
    │
    ▼
DiscreteMargins.summary()
    │
    ├── 创建 Summary 对象
    ├── add_table_2cols() - 顶部信息 (Method, At)
    │
    └── 手动构建参数表格 (不使用 add_table_params)
         │
         ├── 多项模型: 每个类别一个表格，然后合并
         └── 二元模型: 单个表格
```

## 5. 继承体系的好处与限制（修正）

### 5.1 重新评估的好处

#### 1. Template Method 模式的有效应用

**关键在于中间层实现通用框架，调用子类的特化方法**：

```
BinaryModel.predict() (模板方法)
         │
         ├── 固定部分: 计算线性预测 Xβ
         │
         └── 可变部分: 调用 self.cdf(linpred)
              │
              ├── Logit.cdf()  --> 1/(1+exp(-x))
              └── Probit.cdf() --> Φ(x)
```

**好处**：
- 新增模型只需实现 `cdf()` 和 `pdf()` 即可获得预测和边际效应功能
- 算法骨架固定，可变部分延迟到子类
- 代码复用最大化

#### 2. MLE 框架的完全统一

**所有离散模型共享同一套拟合机制**：

```
所有模型的 fit() 流程：
LikelihoodModel.fit()
    │
    ├── 定义目标函数: -loglike(params)
    ├── 定义梯度函数: score(params)
    ├── 定义 Hessian 函数: hessian(params)
    │
    └── 调用 scipy.optimize
         │
         └── 多态调用具体模型的 loglike/score/hessian
```

**好处**：
- 所有模型支持相同的优化方法：`newton`, `bfgs`, `lbfgs`, `nm`, `powell`, `cg`, `ncg`
- 统一的收敛检测、迭代控制、结果包装
- 易于添加新的优化方法

#### 3. 结果层的接口一致性

**用户无需关心模型类型**：

```python
# 任何离散模型的用户代码都相同
model = Logit(y, X)  # 或 Probit(y, X) 或 MNLogit(y, X)
results = model.fit()

# 统一的结果属性
print(results.params)           # 参数估计
print(results.bse)              # 标准误
print(results.tvalues)          # t 统计量
print(results.pvalues)          # p 值
print(results.conf_int())       # 置信区间

# 统一的结果方法
print(results.summary())        # 摘要
print(results.get_margeff())    # 边际效应
print(results.predict(X_new))   # 预测
```

**好处**：
- 用户学习成本低
- 代码可移植性强
- 易于编写通用的分析代码

#### 4. 边际效应计算的巧妙设计

**二元模型通过 `pdf()` 实现多态，多项模型通过重写实现特化**：

```
BinaryModel._derivative_exog()
         │
         └── 调用 self.pdf(linpred)
              │
              ├── Logit.pdf()  --> Λ(1-Λ)
              └── Probit.pdf() --> φ(x)

MultinomialModel._derivative_exog()
         │
         └── 直接使用多项 Logit 公式（不调用 pdf）
              │
              └── 调用 self.cdf() 计算概率
```

**好处**：
- 二元模型新增只需实现 `pdf()`
- 多项模型公式完全不同，通过重写处理
- 既有多态性，又有灵活性

### 5.2 重新评估的限制

#### 1. 继承层次的语义问题（关键限制）

**MultinomialModel 继承 BinaryModel 是设计上的缺陷**：

```python
# 当前设计
class MultinomialModel(BinaryModel):
    # 多项模型继承二元模型
    # 但多项模型并不是二元模型的特例
    # 实际上，二元模型是多项模型的特例（J=2）
    pass
```

**问题分析**：

| 问题 | 具体表现 |
|------|----------|
| **语义错误** | 多项模型不是二元模型的特例，实际上相反 |
| **无用继承** | `BinaryModel._continuous_ok = False` 对多项模型无意义 |
| **offset 处理** | `BinaryModel` 的 offset 处理不适用多项模型 |
| **扩展困难** | 新增有序模型（Ordered Logit/Probit）没有合适的父类 |

**对比：更合理的设计**：

```
DiscreteModel
    ├── BinaryChoiceModel (二元选择 - 独立继承)
    │    ├── Logit
    │    └── Probit
    │
    ├── MultinomialChoiceModel (多项选择 - 独立继承)
    │    └── MNLogit
    │
    └── OrderedChoiceModel (有序选择 - 独立继承)
         ├── OrderedLogit
         └── OrderedProbit
```

#### 2. `pdf()` 方法的不一致性

**MNLogit 不实现 `pdf()`，但继承体系假设它存在**：

```python
class BinaryModel(DiscreteModel):
    def _derivative_exog(self, ...):
        # 假设 self.pdf() 存在
        margeff = np.dot(self.pdf(linpred)[:, None], params[None, :])

class MultinomialModel(BinaryModel):
    def _derivative_exog(self, ...):
        # 重写以避免调用 pdf()
        # 因为多项模型边际效应不使用 pdf
        ...
        # 不调用 self.pdf()

class MNLogit(MultinomialModel):
    # 没有实现 pdf() 方法！
    def cdf(self, X):
        # 只实现 cdf
        ...
```

**问题**：
- 继承层次暗示所有离散模型都应有 `pdf()` 方法
- 但 `MNLogit` 没有实现，依赖于父类重写 `_derivative_exog()`
- 如果 `MultinomialModel` 没有重写 `_derivative_exog()`，会导致运行时错误

#### 3. 参数形状的不一致

**二元模型 vs 多项模型的参数形状完全不同**：

```python
# 二元模型
logit = Logit(y, X)
logit_results = logit.fit()
logit_results.params.shape    # (K,) - 一维数组
logit_results.bse.shape       # (K,) - 一维数组

# 多项模型
mnlogit = MNLogit(y, X)
mnlogit_results = mnlogit.fit()
mnlogit_results.params.shape  # (K, J-1) - 二维矩阵
mnlogit_results.bse.shape     # (K, J-1) - 二维矩阵
```

**问题代码示例**：

```python
def analyze_results(results):
    """
    通用分析函数 - 假设 params 是一维
    """
    params = results.params
    
    # 对二元模型正常工作
    print(f"第一个变量的系数: {params[0]}")
    
    # 对多项模型：params[0] 是一整行，不是一个数！
    # 例如 params[0] = [β_11, β_12, ..., β_1(J-1)]
    
    # 更复杂的操作
    for i, coef in enumerate(params):
        # 二元模型：coef 是标量
        # 多项模型：coef 是数组！
        print(f"变量 {i}: {coef}")
```

**用户代码需要分支处理**：

```python
def analyze_results(results):
    """
    修正后的通用分析函数
    """
    if hasattr(results.model, 'J') and results.model.J > 1:
        # 多项模型：参数是矩阵
        params = results.params
        J = results.model.J
        K = results.model.K
        
        for j in range(J - 1):
            print(f"方程 {j + 1} (vs 基准类别):")
            for k in range(K):
                print(f"  变量 {k}: {params[k, j]}")
    else:
        # 二元模型：参数是一维
        params = results.params
        for k, coef in enumerate(params):
            print(f"变量 {k}: {coef}")
```

#### 4. 边际效应输出形状的不一致

**二元模型 vs 多项模型的边际效应形状不同**：

```python
# 二元模型
logit_me = logit_results.get_margeff(at='overall')
logit_me.margeff.shape      # (K,) - 一维
logit_me.margeff_se.shape   # (K,) - 一维
logit_me.tvalues.shape      # (K,) - 一维

# 多项模型
mnlogit_me = mnlogit_results.get_margeff(at='overall')
mnlogit_me.margeff.shape    # (K, J) - 二维
mnlogit_me.margeff_se.shape # (K, J) - 二维
mnlogit_me.tvalues.shape    # (K, J) - 二维
```

**用户代码需要分支处理**：

```python
def analyze_margeff(me, model):
    J = getattr(model, 'J', 1)
    
    if J > 1:
        # 多项模型
        for j in range(J):
            print(f"类别 {j}:")
            for k in range(me.margeff.shape[0]):
                print(f"  变量 {k}: {me.margeff[k, j]:.4f} (se={me.margeff_se[k, j]:.4f})")
    else:
        # 二元模型
        for k in range(len(me.margeff)):
            print(f"变量 {k}: {me.margeff[k]:.4f} (se={me.margeff_se[k]:.4f})")
```

#### 5. 隐式约定过多

**扩展新模型需要理解多个隐式约定**：

| 隐式约定 | 说明 | 位置 |
|----------|------|------|
| **参数扁平化顺序** | 使用 Fortran 顺序 (`order="F"`) | `MNLogit.loglike()`, `score()`, `hessian()` |
| **基准类别处理** | 多项模型假设第一类是基准类别，参数为0 | `MultinomialModel.initialize()`, `MNLogit.cdf()` |
| **结果类配套** | 每个模型类需要配套结果类和 Wrapper 类 | 模块末尾的 `wrap.populate_wrapper()` 调用 |
| **offset 支持** | 二元模型支持 offset，多项模型可能不支持 | `BinaryModel.predict()` 有 offset 参数 |
| **分离检测** | 二元模型有分离检测，多项模型没有 | `BinaryResults.summary()` |

**新模型开发者需要理解的隐式约定**：

```python
# 假设要添加一个新模型 NewModel
class NewModel(BinaryModel):
    def cdf(self, X):
        pass  # 必须实现
    
    def pdf(self, X):
        pass  # 必须实现（如果是二元模型）
    
    def loglike(self, params):
        pass  # 必须实现
    
    def score(self, params):
        pass  # 必须实现
    
    def hessian(self, params):
        pass  # 必须实现

# 还需要配套结果类
class NewModelResults(BinaryResults):
    pass  # 可能需要特化某些方法

# 还需要 Wrapper 类
class NewModelResultsWrapper(lm.RegressionResultsWrapper):
    pass

# 最后需要注册
wrap.populate_wrapper(NewModelResultsWrapper, NewModelResults)
```

#### 6. 多重继承的复杂性

**L1 正则化结果类使用多重继承**：

```python
class L1BinaryResults(BinaryResults):
    pass

class L1PoissonResults(L1CountResults, PoissonResults):
    # 多重继承
    # MRO: L1PoissonResults -> L1CountResults -> CountResults -> DiscreteResults -> ...
    #                       -> PoissonResults -> CountResults -> DiscreteResults -> ...
    pass
```

**问题**：
- `__init__` 方法的调用顺序可能不符合预期
- 方法重写冲突：如果两个父类都有同名方法，哪个被调用？
- `super()` 的行为依赖于 MRO（方法解析顺序）

### 5.3 限制总结（修正版）

| 限制类型 | 具体问题 | 影响 | 严重程度 |
|----------|----------|------|----------|
| **继承层次语义错误** | MultinomialModel 继承 BinaryModel | 设计混乱，扩展困难 | 🔴 高 |
| **pdf() 方法不一致** | MNLogit 不实现 pdf()，依赖父类重写 | 运行时错误风险 | 🟠 中 |
| **参数形状不一致** | 二元 1D vs 多项 2D | 用户代码需要分支处理 | 🟠 中 |
| **边际效应形状不一致** | 二元 (K,) vs 多项 (K, J) | 接口不统一 | 🟠 中 |
| **隐式约定过多** | F 顺序、基准类别、结果类配套 | 扩展成本高 | 🟠 中 |
| **多重继承复杂** | L1 结果类的 MRO 问题 | 调试困难 | 🟡 低 |

## 6. 修正后的结论

### 6.1 设计亮点总结

**1. 清晰的层次职责划分**：

```
LikelihoodModel (最顶层)
├── 定义 loglike/score/hessian 抽象接口
├── 实现通用 fit() 方法
└── 职责：MLE 框架

DiscreteModel (中间层)
├── 定义 cdf/pdf/predict/_derivative_exog 抽象接口
├── 重写 fit() 添加分离检测
└── 职责：离散模型特有的接口

BinaryModel / MultinomialModel (中间层)
├── 实现 predict() - 调用子类 cdf()
├── 实现 _derivative_exog() - 调用子类 pdf() 或特化公式
└── 职责：二元/多项模型通用框架

Logit / Probit / MNLogit (最底层)
├── 实现 cdf()/pdf()
├── 实现 loglike()/score()/hessian()
└── 职责：具体模型特化
```

**2. Template Method 模式的有效应用**：

| 模板方法 | 固定部分 | 可变部分 | 实现位置 |
|----------|----------|----------|----------|
| `BinaryModel.predict()` | 计算 Xβ + offset | 调用 `self.cdf(linpred)` | `discrete_model.py:538` |
| `BinaryModel._derivative_exog()` | 应用公式 `f(Xβ)*β` | 调用 `self.pdf(linpred)` | `discrete_model.py:670` |
| `DiscreteModel.fit()` | 添加分离检测 callback | 调用 `super().fit()` | `discrete_model.py:242` |

**3. 统一的 MLE 框架**：

所有离散模型共享：
- 相同的优化方法支持
- 相同的收敛检测
- 相同的结果包装

**4. 结果层的接口一致性**：

用户代码对所有模型都相同：
```python
results = model.fit()
results.summary()        # 摘要
results.get_margeff()    # 边际效应
results.predict(X_new)   # 预测
```

### 6.2 设计缺陷总结

**1. 继承层次的根本问题**：

```
当前设计：
MultinomialModel ──继承──> BinaryModel
    │                          │
    │  多项模型                │  二元模型
    │  不是二元模型的特例       │  实际上是多项模型的特例
    │  (J=2)                   │  (J=2)

问题：语义上完全相反！
```

**2. 不一致的抽象方法实现**：

| 类 | cdf() | pdf() | loglike() | score() | hessian() |
|------|-------|-------|-----------|---------|-----------|
| Logit | ✅ | ✅ | ✅ | ✅ | ✅ |
| Probit | ✅ | ✅ | ✅ | ✅ | ✅ |
| MNLogit | ✅ | ❌ | ✅ | ✅ | ✅ |

**问题**：`pdf()` 是 `DiscreteModel` 定义的抽象方法，但 `MNLogit` 没有实现它。这依赖于 `MultinomialModel` 重写 `_derivative_exog()` 来避免调用 `pdf()`，是一种脆弱的设计。

**3. 参数形状的不一致性**：

| 属性 | 二元模型 (Logit/Probit) | 多项模型 (MNLogit) |
|------|------------------------|---------------------|
| `params` | (K,) | (K, J-1) |
| `bse` | (K,) | (K, J-1) |
| `tvalues` | (K,) | (K, J-1) |
| `pvalues` | (K,) | (K, J-1) |
| `conf_int()` | (K, 2) | (J, K, 2) |

### 6.3 实际使用建议

#### 对于用户：

1. **使用统一接口**：
   ```python
   # 推荐：使用统一接口
   results = model.fit()
   results.summary()        # 所有模型相同
   results.get_margeff()    # 所有模型相同
   results.predict(X_new)   # 所有模型相同
   ```

2. **处理多项模型参数时注意形状**：
   ```python
   # 多项模型
   if hasattr(results.model, 'J') and results.model.J > 1:
       params = results.params  # (K, J-1)
       for j in range(results.model.J - 1):
           print(f"方程 {j + 1}:", params[:, j])
   else:
       params = results.params  # (K,)
       print(params)
   ```

3. **边际效应解释**：
   - 二元模型：直接解释为概率变化
   - 多项模型：每个类别的概率变化单独解释，注意基准类别

#### 对于扩展者：

1. **二元模型扩展**：
   ```python
   class NewBinaryModel(BinaryModel):
       # 必需方法
       def cdf(self, X):
           pass
       
       def pdf(self, X):
           pass
       
       def loglike(self, params):
           pass
       
       def score(self, params):
           pass
       
       def hessian(self, params):
           pass
   
   # 配套结果类（通常不需要特化）
   class NewBinaryModelResults(BinaryResults):
       pass
   ```

2. **多项模型扩展**：
   - 注意：`MultinomialModel` 继承 `BinaryModel`，但多项模型不是二元模型的特例
   - 如果需要扩展多项模型，考虑直接继承 `DiscreteModel`

3. **不要忘记 Wrapper 类**：
   ```python
   # 在模块末尾
   wrap.populate_wrapper(NewBinaryModelResultsWrapper, NewBinaryModelResults)
   ```

#### 对于维护者：

1. **考虑重构继承层次**：
   ```python
   # 建议的新设计
   DiscreteModel
   ├── BinaryChoiceModel      # 独立继承
   │    ├── Logit
   │    └── Probit
   ├── MultinomialChoiceModel  # 独立继承，不继承 BinaryChoiceModel
   │    └── MNLogit
   └── OrderedChoiceModel      # 新增有序选择模型
        ├── OrderedLogit
        └── OrderedProbit
   ```

2. **显式化接口约定**：
   - 使用 ABC 强制 `pdf()` 实现
   - 或者从 `DiscreteModel` 中移除 `pdf()`，改为二元模型特有的抽象方法

3. **统一参数形状处理**：
   - 考虑始终使用扁平化参数
   - 提供 `reshape_params()` 方法进行维度转换

### 6.4 权衡分析

**继承体系的好处与限制的权衡**：

| 设计决策 | 好处 | 限制 | 是否值得 |
|----------|------|------|----------|
| 统一 MLE 框架 | 代码复用、接口一致 | 灵活性降低 | ✅ 值得 |
| Template Method (predict) | 新增模型只需实现 cdf | 灵活性降低 | ✅ 值得 |
| Template Method (_derivative_exog) | 二元模型只需实现 pdf | MNLogit 不实现 pdf，依赖重写 | ⚠️ 值得但需改进 |
| MultinomialModel 继承 BinaryModel | 代码复用（可能） | 语义错误、设计混乱 | ❌ 不值得，应重构 |
| 参数形状不一致 | 符合直觉（多项模型是多方程） | 用户代码需要分支处理 | ⚠️ 权衡结果，提供转换方法 |

### 6.5 最终结论

**Statsmodels 离散选择模型的继承体系**：

**优点**：
- ✅ **MLE 框架完全统一**：所有模型共享同一套拟合机制
- ✅ **Template Method 有效应用**：`predict()` 和 `_derivative_exog()` 实现了通用框架
- ✅ **结果层接口一致**：用户代码对所有模型都相同
- ✅ **多态行为**：新增模型只需特化差异部分

**缺点**：
- ❌ **继承层次语义错误**：`MultinomialModel` 继承 `BinaryModel` 是根本的设计缺陷
- ❌ **抽象方法不一致**：`MNLogit` 不实现 `pdf()`，依赖脆弱的重写机制
- ⚠️ **参数形状不一致**：二元模型 1D vs 多项模型 2D，用户代码需要分支处理
- ⚠️ **隐式约定过多**：F 顺序、基准类别、结果类配套等

**建议的改进优先级**：

1. **高优先级**：重构继承层次，让 `MultinomialModel` 独立继承 `DiscreteModel`，不再继承 `BinaryModel`
2. **中优先级**：显式化 `pdf()` 方法约定，要么强制所有模型实现，要么将其移到二元模型层
3. **中优先级**：提供统一的参数形状转换方法，简化用户代码
4. **低优先级**：改进多重继承的使用，或使用组合替代

## 7. 数值稳定处理：二元与多元模型的差异（可核验深度）

### 7.1 数值稳定常量定义

**位置**：`discrete_model.py:68-71`

```python
FLOAT_EPS = np.finfo(float).eps  # ~2.22e-16
EXP_UPPER_LIMIT = np.log(np.finfo(np.float64).max) - 1.0  # ~709
```

**用途**：
- `FLOAT_EPS`：防止除零和 `log(0)`
- `EXP_UPPER_LIMIT`：防止指数上溢（`exp(709) ≈ 1e308` 是 float64 的最大值）

### 7.2 Logit 模型的数值稳定处理

**核心策略**：利用 Logistic 分布的对称性，避免直接计算 `exp()` 的极端值。

**位置**：`discrete_model.py:2705`

```python
def loglike(self, params):
    """
    Logit 对数似然
    
    关键技巧：使用 q = 2y - 1 转换
    - 当 y=1 时，q=1，计算 Λ(x'β)
    - 当 y=0 时，q=-1，计算 Λ(-x'β) = 1 - Λ(x'β)
    
    利用对称性：Λ(-x) = 1 - Λ(x)
    这样避免了直接计算 log(1-Λ(x)) 时的数值不稳定
    """
    q = 2 * self.endog - 1  # 转换为 +1/-1
    linpred = self.predict(params, which="linear")
    return np.sum(np.log(self.cdf(q * linpred)))
```

**Logit CDF 的数值稳定性**：

```python
def cdf(self, X):
    """
    Λ(x) = 1 / (1 + exp(-x))
    
    数值分析：
    - 当 x → +∞: exp(-x) → 0，Λ(x) → 1
    - 当 x → -∞: exp(-x) → ∞，Λ(x) → 0
    
    潜在问题：
    - 当 x < -709 时，exp(-x) 会溢出为 inf
    - 当 x > 709 时，exp(-x) 会下溢为 0
    
    但实际上：
    - 当 x < -709，exp(-x) = inf，但 1/(1+inf) = 0 是正确的
    - 当 x > 709，exp(-x) = 0，1/(1+0) = 1 是正确的
    
    所以 Logit 的 cdf() 本身是数值稳定的！
    """
    return 1 / (1 + np.exp(-X))
```

**Logit Score 和 Hessian 的数值稳定性**：

```python
def score(self, params):
    """
    梯度：∂ln L/∂β = Σ (y_i - Λ_i) x_i
    
    数值分析：
    - Λ_i = 1/(1+exp(-Xβ))
    - y_i - Λ_i 的范围是 (-1, 1)
    - 不需要 clip，因为范围有限
    """
    y = self.endog
    X = self.exog
    fitted = self.predict(params)  # Λ(Xβ)
    return np.dot(y - fitted, X)

def hessian(self, params):
    """
    Hessian：∂²ln L/∂β∂β' = -Σ Λ_i(1-Λ_i) x_i x_i'
    
    数值分析：
    - Λ(1-Λ) 的范围是 [0, 0.25]
    - 当 Λ → 0 或 Λ → 1 时，Λ(1-Λ) → 0
    - 不需要 clip，因为范围有限
    """
    X = self.exog
    L = self.predict(params)
    return -np.dot(L * (1 - L) * X.T, X)
```

### 7.3 Probit 模型的数值稳定处理

**核心策略**：大量使用 `np.clip()` 防止 `log(0)` 和除零。

**位置**：`discrete_model.py:2997-3146`

**Probit 对数似然的数值稳定**：

```python
def loglike(self, params):
    """
    Probit 对数似然
    
    关键问题：
    - 正态分布 Φ(x) 的尾部比 Logistic 分布薄得多
    - 当 |x| > 8 时，Φ(x) 或 1-Φ(x) 会非常接近 0 或 1
    - log(接近 0 的数) 会导致数值下溢或 inf
    
    解决方案：
    - 使用 np.clip(cdf, FLOAT_EPS, 1)
    - FLOAT_EPS ≈ 2.22e-16
    """
    q = 2 * self.endog - 1
    linpred = self.predict(params, which="linear")
    # 关键：clip 防止 log(0)
    return np.sum(np.log(np.clip(self.cdf(q * linpred), FLOAT_EPS, 1)))
```

**Probit Score 的数值稳定**：

```python
def score(self, params):
    """
    Probit 梯度
    
    公式：∂ln L/∂β = Σ [q_i φ(q_i x_i'β) / Φ(q_i x_i'β)] x_i
    
    其中：
    - λ_i = q_i φ(q_i Xβ) / Φ(q_i Xβ) 是逆米尔斯比率
    
    数值问题：
    - 当 Φ(q_i Xβ) 接近 0 时，λ_i 会趋向无穷大
    - 需要 clip Φ 值防止除零
    
    位置：discrete_model.py:3053
    """
    y = self.endog
    X = self.exog
    XB = self.predict(params, which="linear")
    q = 2 * y - 1
    
    # 关键：clip 防止除零
    # 同时限制在 [FLOAT_EPS, 1-FLOAT_EPS]
    L = q * self.pdf(q * XB) / np.clip(self.cdf(q * XB), FLOAT_EPS, 1 - FLOAT_EPS)
    return np.dot(L, X)
```

**Probit Hessian 的数值稳定**：

```python
def hessian(self, params):
    """
    Probit Hessian
    
    公式：∂²ln L/∂β∂β' = -Σ λ_i(λ_i + x_i'β) x_i x_i'
    
    数值问题：
    - λ_i 可能很大（当 Φ 接近 0 时）
    - 需要 clip 防止数值爆炸
    
    位置：discrete_model.py:3146
    """
    X = self.exog
    XB = self.predict(params, which="linear")
    q = 2 * self.endog - 1
    
    # 同样使用 clip
    L = q * self.pdf(q * XB) / np.clip(self.cdf(q * XB), FLOAT_EPS, 1 - FLOAT_EPS)
    return np.dot(-L * (L + XB) * X.T, X)
```

### 7.4 MNLogit 模型的数值稳定处理

**当前实现的潜在问题**：Softmax 的指数上溢风险。

**位置**：`discrete_model.py:3339`

```python
def cdf(self, X):
    """
    MNLogit 的 CDF (Softmax 函数)
    
    当前实现：
    P(j|x) = exp(β_j'x) / Σ_k exp(β_k'x)
    
    代码：
    eXB = np.column_stack((np.ones(len(X)), np.exp(X)))  # 添加基准类别
    return eXB / eXB.sum(1)[:, None]
    
    数值问题：
    - 当 X 的元素 > 709 时，np.exp(X) 会溢出为 inf
    - 当任何 exp 值为 inf 时，整个 Softmax 计算会失败
    
    示例：
    X = np.array([[800, 0, 0]])  # 第一个变量很大
    np.exp(X)  # [inf, 1, 1]
    sum = inf
    Softmax = [inf/inf=nan, 1/inf=0, 1/inf=0]  # 结果错误
    """
    eXB = np.column_stack((np.ones(len(X)), np.exp(X)))
    return eXB / eXB.sum(1)[:, None]
```

**标准的数值稳定 Softmax 实现**：

```python
def stable_softmax(X):
    """
    数值稳定的 Softmax 实现
    
    技巧：减去每行的最大值
    X_stable = X - max(X)  （这样所有指数项都 <= 0）
    
    数学原理：
    exp(X - max(X)) / Σ exp(X - max(X))
    = [exp(X) / exp(max(X))] / [Σ exp(X) / exp(max(X))]
    = exp(X) / Σ exp(X)
    = 原始 Softmax
    
    这样计算时，指数项不会上溢（因为 X - max(X) <= 0）
    """
    # 对每行减去最大值
    X_max = X.max(axis=1, keepdims=True)
    X_stable = X - X_max
    
    # 计算指数（现在所有指数 <= 0，不会上溢）
    exp_X = np.exp(X_stable)
    
    # 计算 Softmax
    return exp_X / exp_X.sum(axis=1, keepdims=True)
```

**MNLogit 缺少的数值稳定处理**：

| 数值问题 | 当前实现 | 标准稳定实现 |
|----------|----------|--------------|
| Softmax 指数上溢 | ❌ 无处理 | ✅ 减去最大值 |
| log(0) 下溢 | 依赖 `cdf()` 的行为 | 通常使用 `log_softmax` |

### 7.5 数值稳定处理对照总结

| 维度 | Logit | Probit | MNLogit |
|------|-------|--------|---------|
| **核心策略** | 数学对称性 | 大量 clip | 无特殊处理 |
| **常量** | 无 | `FLOAT_EPS` | 无 |
| **loglike** | 利用 q=2y-1 变换 | `np.clip(cdf, FLOAT_EPS, 1)` | 无 clip |
| **score** | 无 clip | `np.clip(cdf, FLOAT_EPS, 1-FLOAT_EPS)` | 无 clip |
| **hessian** | 无 clip | `np.clip(cdf, FLOAT_EPS, 1-FLOAT_EPS)` | 无 clip |
| **cdf 稳定性** | ✅ 天然稳定（exp 下溢不影响结果） | ⚠️ 依赖 clip | ❌ 有上溢风险 |

### 7.6 数值稳定性的可核验示例

**Probit vs Logit 在极端值下的行为**：

```python
import numpy as np
from scipy import stats

FLOAT_EPS = np.finfo(float).eps

# 极端值测试
x_extreme = 37  # 当 x=37 时，Φ(x) 已经非常接近 1

# Logit 的行为
logit_cdf = 1 / (1 + np.exp(-x_extreme))  # ≈ 1.0
logit_log = np.log(logit_cdf)  # ≈ 0.0（稳定）
print(f"Logit cdf({x_extreme}) = {logit_cdf}")
print(f"Logit log(cdf({x_extreme})) = {logit_log}")

# Probit 的行为（无 clip）
probit_cdf = stats.norm.cdf(x_extreme)  # ≈ 1.0，但非常接近
probit_cdf_clipped = np.clip(probit_cdf, FLOAT_EPS, 1)
print(f"Probit cdf({x_extreme}) = {probit_cdf}")
print(f"Probit cdf({x_extreme}) (raw) = {probit_cdf}")
print(f"Probit log(cdf({x_extreme})) = {np.log(probit_cdf)}")  # 可能不稳定
print(f"Probit log(clip(cdf)) = {np.log(probit_cdf_clipped)}")  # 稳定

# MNLogit 的潜在问题
X = np.array([[800, 0, 0]])  # 极端值
eXB = np.column_stack((np.ones(len(X)), np.exp(X)))
print(f"\nMNLogit exp(X) = {eXB}")  # [1, inf, 1, 1]
softmax = eXB / eXB.sum(1)[:, None]
print(f"MNLogit Softmax = {softmax}")  # [nan, 0, 0, 0] 或类似错误
```

---

## 8. 优化方法对梯度Hessian的依赖与收敛行为（可核验深度）

### 8.1 优化方法分类与依赖关系

**位置**：`base/optimizer.py:237-262`

```python
methods = [
    "newton",      # Newton-Raphson
    "nm",          # Nelder-Mead
    "bfgs",        # Broyden-Fletcher-Goldfarb-Shanno
    "lbfgs",       # Limited-memory BFGS
    "powell",      # Powell's method
    "cg",          # Conjugate Gradient
    "ncg",         # Newton-Conjugate Gradient
    "basinhopping", # Global optimization
    "minimize",    # Scipy minimize wrapper
]

fit_funcs = {
    "newton": _fit_newton,
    "nm": _fit_nm,
    "bfgs": _fit_bfgs,
    "lbfgs": _fit_lbfgs,
    "cg": _fit_cg,
    "ncg": _fit_ncg,
    "powell": _fit_powell,
    "basinhopping": _fit_basinhopping,
    "minimize": _fit_minimize,
}
```

### 8.2 各优化方法的依赖关系

**位置**：`base/model.py:840-848`

```
Optimization methods that require only a likelihood function are 'nm' and 'powell'
    只需要 loglike

Optimization methods that require a likelihood function and a score/gradient
are 'bfgs', 'cg', and 'ncg'. A function to compute the Hessian is optional for 'ncg'
    需要 loglike + score，hessian 可选（仅 ncg）

Optimization method that require a likelihood function, a score/gradient,
and a Hessian is 'newton'
    需要 loglike + score + hessian
```

### 8.3 优化方法依赖对照表

| 方法 | 代码名 | 需要 loglike | 需要 score | 需要 hessian | 类型 |
|------|--------|-------------|-----------|-------------|------|
| **Newton-Raphson** | `newton` | ✅ | ✅ | ✅ | 二阶方法 |
| **Newton-CG** | `ncg` | ✅ | ✅ | ⚠️ 可选 | 混合二阶 |
| **BFGS** | `bfgs` | ✅ | ✅ | ❌ | 拟牛顿法 |
| **L-BFGS** | `lbfgs` | ✅ | ✅ | ❌ | 有限内存拟牛顿 |
| **Conjugate Gradient** | `cg` | ✅ | ✅ | ❌ | 非线性 CG |
| **Nelder-Mead** | `nm` | ✅ | ❌ | ❌ | 无导数方法 |
| **Powell** | `powell` | ✅ | ❌ | ❌ | 方向集方法 |
| **Basinhopping** | `basinhopping` | ✅ | ✅ | 可选 | 全局优化 |
| **Minimize 包装器** | `minimize` | 取决于子方法 | 取决于子方法 | 取决于子方法 | 通用包装器 |

### 8.4 核心优化代码的依赖处理

**位置**：`base/model.py:561-576`

```python
def fit(self, ..., method="newton", ...):
    # 目标函数：负对数似然（最小化）
    def f(params, *args):
        return -self.loglike(params, *args) / nobs
    
    # 根据方法决定 score 和 hessian 的符号
    # 注意：优化器通常最小化 f = -loglike
    # 而 score() 返回的是 loglike 的梯度（最大化方向）
    
    if method == "newton":
        # Newton 方法：score 和 hess 保持正号
        # 因为 Newton 迭代：β_{k+1} = β_k - H^{-1} g
        # 其中 g = ∇loglike（最大化方向）
        def score(params, *args):
            return self.score(params, *args) / nobs  # 正号
        
        def hess(params, *args):
            return self.hessian(params, *args) / nobs  # 正号
    else:
        # 其他方法：score 和 hess 取负号
        # 因为这些方法最小化 f = -loglike
        # 梯度为 ∇f = -∇loglike
        def score(params, *args):
            return -self.score(params, *args) / nobs  # 负号！
        
        def hess(params, *args):
            return -self.hessian(params, *args) / nobs  # 负号！
```

**符号处理的关键理解**：

```
问题：为什么 Newton 方法的符号不同？

假设我们要最大化 loglike(β)：
- Newton 方法（最大化）：β_{k+1} = β_k - H^{-1} g
  其中 g = ∇loglike, H = ∇²loglike（负定）

或者等价地，最小化 f(β) = -loglike(β)：
- Newton 方法（最小化）：β_{k+1} = β_k - H_f^{-1} g_f
  其中 g_f = ∇f = -g, H_f = ∇²f = -H

在 statsmodels 中：
- model.loglike() 返回 loglike（最大化目标）
- model.score() 返回 ∇loglike（最大化梯度）
- model.hessian() 返回 ∇²loglike（负定）

优化器期望最小化 f = -loglike，所以：
- 对于拟牛顿法 (bfgs, cg, ncg)：直接传递 ∇f = -score
- 对于 Newton-Raphson：代码中保持正号，但迭代公式已考虑符号
```

### 8.5 Newton-Raphson 方法的实现

**位置**：`base/optimizer.py:456`

```python
def _fit_newton(
    f, score, start_params, fargs, kwargs, ..., hess=None, ridge_factor=1e-10
):
    """
    Newton-Raphson 算法
    
    迭代公式：
    β_{k+1} = β_k - H(β_k)^{-1} g(β_k)
    
    其中：
    - g = ∇f = 梯度
    - H = ∇²f = Hessian
    
    注意：
    - 使用 ridge_factor 防止 Hessian 奇异
    - H_ridge = H + ridge_factor * I
    """
    
    # 初始化
    x0 = start_params
    iterations = 0
    oldparams = 10 * x0 + 10  # 确保第一次迭代运行
    
    # Newton 迭代
    while iterations < maxiter:
        # 检查收敛
        if np.allclose(x0, oldparams, rtol=tol):
            converged = True
            break
        
        oldparams = x0.copy()
        
        # 计算梯度和 Hessian
        gradient = score(x0, *fargs)
        H = hess(x0, *fargs)
        
        # 关键：Ridge 调整防止 Hessian 奇异
        # H_ridge = H + λI, 其中 λ = 1e-10
        H = H + ridge_factor * np.eye(len(H))
        
        # Newton 更新
        try:
            x0 = x0 - np.linalg.solve(H, gradient)
        except np.linalg.LinAlgError:
            # 如果 Hessian 仍然奇异，使用伪逆
            x0 = x0 - np.dot(np.linalg.pinv(H), gradient)
        
        iterations += 1
        if callback is not None:
            callback(x0)
    
    return x0, retvals
```

**Newton-Raphson 的收敛特性**：

| 特性 | 说明 |
|------|------|
| **收敛速度** | 二次收敛（接近解时） |
| **每轮代价** | 高（需要计算和求逆 Hessian） |
| **稳定性** | 依赖 Hessian 的正定性（或负定性） |
| **初始化敏感** | 是，如果离解太远可能不收敛 |
| **适用场景** | 当 Hessian 容易计算且相对良态时 |

### 8.6 BFGS 方法的实现

**位置**：`scipy.optimize.fmin_bfgs`（statsmodels 包装）

```python
def _fit_bfgs(f, score, start_params, fargs, kwargs, ...):
    """
    BFGS (Broyden-Fletcher-Goldfarb-Shanno) 拟牛顿法
    
    核心思想：
    - 不直接计算 Hessian，而是从梯度变化中迭代估计
    - 使用准牛顿更新公式：H_{k+1} ≈ H_k + 修正项
    
    相比 Newton 的优势：
    - 不需要计算 Hessian
    - 不需要求逆矩阵（更新逆 Hessian 近似）
    - 每轮只需计算梯度
    
    相比 Newton 的劣势：
    - 收敛速度是超线性，不是二次
    - 需要更多迭代
    - 可能在病态问题上收敛慢
    """
    # 调用 scipy.optimize.fmin_bfgs
    from scipy.optimize import fmin_bfgs
    
    retvals = fmin_bfgs(
        f,                      # 目标函数
        start_params,           # 初始参数
        fprime=score,           # 梯度函数
        args=fargs,             # 额外参数
        gtol=kwargs.get("gtol", 1e-05),  # 梯度收敛阈值
        norm=kwargs.get("norm", np.inf),  # 范数类型
        epsilon=kwargs.get("epsilon", 1.49e-08),  # 数值梯度步长
        maxiter=maxiter,
        full_output=full_output,
        disp=disp,
        retall=retall,
        callback=callback,
    )
    return xopt, retvals
```

### 8.7 Nelder-Mead 无导数方法

**位置**：`scipy.optimize.fmin`

```python
def _fit_nm(f, score, start_params, fargs, kwargs, ...):
    """
    Nelder-Mead 单纯形法（无导数）
    
    核心思想：
    - 维护一个 n+1 个顶点的单纯形（n 是参数维度）
    - 通过反射、扩张、收缩、收缩等操作移动单纯形
    - 不需要梯度！
    
    优势：
    - 不需要计算梯度或 Hessian
    - 对非光滑函数更鲁棒
    - 不需要良好的初始化
    
    劣势：
    - 收敛速度慢（亚线性）
    - 在高维空间效率低
    - 可能收敛到局部最优
    """
    from scipy.optimize import fmin
    
    # 注意：score 参数被忽略！
    retvals = fmin(
        f,
        start_params,
        args=fargs,
        xtol=kwargs.get("xtol", 1e-4),
        ftol=kwargs.get("ftol", 1e-4),
        maxiter=kwargs.get("maxfun", None),
        maxfun=kwargs.get("maxfun", None),
        full_output=full_output,
        disp=disp,
        retall=retall,
        callback=callback,
    )
    return xopt, retvals
```

### 8.8 优化方法收敛行为对比

| 方法 | 收敛类型 | 每轮迭代代价 | 初始化敏感度 | 高维适用性 | 稳定性 |
|------|----------|-------------|-------------|-----------|--------|
| **Newton** | 二次 | 高（Hessian 计算+求逆） | 高 | 低（O(K³)） | 低（依赖 Hessian 可逆） |
| **BFGS** | 超线性 | 中（梯度计算） | 中 | 中（O(K²) 存储） | 中 |
| **L-BFGS** | 超线性 | 中低（梯度计算） | 中 | 高（有限内存） | 中 |
| **CG** | 线性/超线性 | 低（梯度计算） | 中高 | 高 | 中低 |
| **NCG** | 超线性 | 中（梯度+Hessian 向量乘） | 中 | 中 | 中 |
| **Nelder-Mead** | 亚线性 | 低（只需要函数值） | 低 | 低（单纯形扩展） | 高 |
| **Powell** | 亚线性/线性 | 低（只需要函数值） | 低 | 中 | 高 |

### 8.9 协方差矩阵计算的方法依赖

**位置**：`base/model.py:613-632`

```python
def fit(...):
    ...
    # 协方差矩阵计算
    if method == "newton" and full_output:
        # Newton 方法：使用优化返回的 Hessian
        # Hinv = (-Hessian)^{-1} / nobs
        Hinv = np.linalg.inv(-retvals["Hessian"]) / nobs
    elif not skip_hessian:
        # 其他方法：需要重新计算 Hessian
        H = -1 * self.hessian(xopt)
        
        # 检查 Hessian 是否正定
        invertible = False
        if np.all(np.isfinite(H)):
            eigvals, eigvecs = np.linalg.eigh(H)
            if np.min(eigvals) > 0:
                invertible = True
        
        if invertible:
            # 使用谱分解求逆
            Hinv = eigvecs.dot(np.diag(1.0 / eigvals)).dot(eigvecs.T)
            Hinv = np.asfortranarray((Hinv + Hinv.T) / 2.0)
        else:
            warnings.warn(
                "Inverting hessian failed, no bse or cov_params available",
                HessianInversionWarning,
            )
            Hinv = None
    ...
```

**协方差矩阵计算的关键差异**：

| 方法 | 协方差来源 | 是否需要重新计算 hessian |
|------|-----------|--------------------------|
| `newton` + `full_output` | 优化过程中的 Hessian | ❌ 不需要 |
| 其他方法 + `skip_hessian=False` | 重新计算 `self.hessian(xopt)` | ✅ 需要 |
| 任何方法 + `skip_hessian=True` | 不计算协方差 | ❌ 跳过 |

### 8.10 默认方法选择

**位置**：`base/model.py:365`

```python
def fit(self, ..., method="newton", ...):
    """
    默认使用 Newton-Raphson 方法
    
    原因：
    1. 离散选择模型通常有解析的 Hessian
    2. Newton 方法收敛快（二次收敛）
    3. 当参数数量适中时（K 不太大），Hessian 计算可行
    """
```

**当 Newton 方法失败时的备选策略**：

```python
# 用户可以切换到其他方法
results = model.fit(method="bfgs")           # 拟牛顿法
results = model.fit(method="nm")             # 无导数方法
results = model.fit(method="lbfgs")          # 高维问题
results = model.fit(method="powell")         # 无导数备选

# 或者使用通用包装器
results = model.fit(
    method="minimize",
    min_method="L-BFGS-B",  # 指定 scipy 方法
    bounds=[(0, None)] * K   # 可以指定边界
)
```

### 8.11 优化方法选择建议

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| **标准情况** | `newton`（默认） | 收敛快，有解析 Hessian |
| **高维问题 (K > 1000)** | `lbfgs` | 有限内存，不需要完整 Hessian |
| **Hessian 计算困难** | `bfgs` | 拟牛顿，只需要梯度 |
| **梯度计算也困难** | `nm` 或 `powell` | 无导数方法 |
| **非光滑目标函数** | `nm` 或 `powell` | 对非光滑更鲁棒 |
| **需要全局最优** | `basinhopping` | 全局优化方法 |
| **需要边界约束** | `minimize` + `L-BFGS-B` | 支持边界 |

---

## 9. 边际效应输出形状对照（可核验深度）

### 9.1 边际效应计算的入口

**位置**：`discrete_margins.py:487`

```python
def get_margeff(self, at="overall", method="dydx", atexog=None, dummy=False, count=False):
    """
    核心边际效应计算
    
    参数：
    at : {'overall', 'mean', 'median', 'zero', 'all'}
        - 'overall': 所有观测的平均边际效应
        - 'mean': 在解释变量均值处
        - 'median': 在解释变量中位数处
        - 'zero': 在解释变量为零处
        - 'all': 每个观测的边际效应
    """
    results = self.results
    model = results.model
    params = results.params.copy()
    
    # 准备 exog 矩阵
    if atexog is None:
        if at == "mean":
            exog = model.exog.mean(0)[None, :]  # (1, K)
        elif at == "median":
            exog = np.median(model.exog, 0)[None, :]  # (1, K)
        elif at == "zero":
            exog = np.zeros((1, model.exog.shape[1]))  # (1, K)
        elif at == "all":
            exog = model.exog  # (nobs, K)
        elif at == "overall":
            exog = model.exog  # (nobs, K) - 之后取平均
    else:
        exog = atexog  # 用户提供
    
    # 识别虚拟变量和计数变量
    ...
    
    # 核心计算：调用模型的 _derivative_exog
    margeff = model._derivative_exog(
        params, exog, transform=method,
        dummy_idx=dummy_ind, count_idx=count_ind
    )
    
    # 根据 at 参数聚合
    if at == "overall":
        # 取所有观测的平均
        margeff = np.mean(margeff, axis=0)
    
    # 计算标准误（Delta Method）
    ...
```

### 9.2 二元模型边际效应输出形状

**二元模型**：Logit, Probit

**位置**：`discrete_model.py:670`

```python
def _derivative_exog(self, params, exog=None, transform="dydx", ...):
    """
    二元模型边际效应
    
    公式：∂P/∂x = f(Xβ) * β
    其中 f(.) = pdf()
    
    返回形状：
    - 当 exog 是 (nobs, K): 返回 (nobs, K)
    - 当 exog 是 (1, K): 返回 (1, K)
    """
    linpred = self.predict(params, exog, offset=offset, which="linear")  # (nobs,)
    
    # 核心公式：f(Xβ) ⊗ β'
    # self.pdf(linpred) 形状 (nobs,)
    # params 形状 (K,)
    # margeff 形状 (nobs, K)
    margeff = np.dot(self.pdf(linpred)[:, None], params[None, :])
    
    # 变换处理
    if "ex" in transform:
        margeff *= exog  # (nobs, K) * (nobs, K) = (nobs, K)
    if "ey" in transform:
        margeff /= self.predict(params, exog)[:, None]  # (nobs, K) / (nobs, 1) = (nobs, K)
    
    return margeff  # (nobs, K)
```

### 9.3 多项模型边际效应输出形状

**多项模型**：MNLogit

**位置**：`discrete_model.py:979`

```python
def _derivative_exog(self, params, exog=None, transform="dydx", ...):
    """
    多项模型边际效应
    
    公式：∂P(j|x)/∂x_k = P(j|x) * [β_jk - Σ_l P(l|x) β_lk]
    
    返回形状：
    - 当 exog 是 (nobs, K): 返回 (nobs, K*J) 扁平化
      逻辑形状是 (nobs, K, J)
    """
    J = int(self.J)  # 类别数
    K = int(self.K)  # 变量数
    
    # 重塑参数
    params = params.reshape(K, -1, order="F")  # (K, J-1)
    
    # 添加基准类别参数（全0）
    zeroparams = np.c_[np.zeros(K), params]  # (K, J)
    
    # 计算概率
    cdf = self.cdf(np.dot(exog, params))  # (nobs, J)
    
    # 计算加权平均参数
    iterm = np.array([cdf[:, [i]] * zeroparams[:, i] for i in range(int(J))]).sum(0)  # (K,)
    
    # 应用公式
    # margeff 形状 (J, nobs, K)
    margeff = np.array([cdf[:, [j]] * (zeroparams[:, j] - iterm) for j in range(J)])
    
    # 调整维度：(J, nobs, K) -> (nobs, K, J)
    margeff = np.transpose(margeff, (1, 2, 0))  # (nobs, K, J)
    
    # 变换处理
    if "ex" in transform:
        margeff *= exog  # (nobs, K, J) * (nobs, K) = 广播后 (nobs, K, J)
    if "ey" in transform:
        margeff /= self.predict(params, exog)[:, None, :]  # (nobs, K, J) / (nobs, 1, J) = (nobs, K, J)
    
    # 扁平化：使用 Fortran 顺序
    # (nobs, K, J) -> (nobs, K*J)
    # 先遍历变量，再遍历类别
    return margeff.reshape(len(exog), -1, order="F")  # (nobs, K*J)
```

### 9.4 边际效应形状对照总表

| 模型类型 | `at` 参数 | `_derivative_exog` 返回 | 最终 `margeff` 形状 | `margeff.ndim` |
|----------|-----------|-------------------------|---------------------|----------------|
| **二元** | `'mean'` | `(1, K)` | `(K,)` | 1 |
| **二元** | `'median'` | `(1, K)` | `(K,)` | 1 |
| **二元** | `'zero'` | `(1, K)` | `(K,)` | 1 |
| **二元** | `'overall'` | `(nobs, K)` | `(K,)`（平均后） | 1 |
| **二元** | `'all'` | `(nobs, K)` | `(nobs, K)` | 2 |
| **多项** | `'mean'` | `(1, K*J)` | `(K, J)` | 2 |
| **多项** | `'median'` | `(1, K*J)` | `(K, J)` | 2 |
| **多项** | `'zero'` | `(1, K*J)` | `(K, J)` | 2 |
| **多项** | `'overall'` | `(nobs, K*J)` | `(K, J)`（平均后） | 2 |
| **多项** | `'all'` | `(nobs, K*J)` | `(nobs, K, J)`（重塑后） | 3 |

### 9.5 形状判断逻辑

**位置**：`discrete_margins.py:502`

```python
def summary_frame(self, alpha=0.05):
    """
    返回边际效应的 DataFrame
    
    关键形状判断：
    """
    ...
    if self.margeff.ndim == 2:
        # MNLogit 情况 (at='overall', 'mean', 'median', 'zero')
        # 形状 (K, J)
        ci = self.conf_int(alpha)
        table = np.column_stack(
            [
                i.ravel("F")  # Fortran 顺序扁平化
                for i in [
                    self.margeff,      # (K, J)
                    self.margeff_se,   # (K, J)
                    self.tvalues,      # (K, J)
                    self.pvalues,      # (K, J)
                    ci[:, 0, :],       # (K, J)
                    ci[:, 1, :],       # (K, J)
                ]
            ]
        )
        # 使用 MultiIndex
        index = MultiIndex.from_tuples(
            list(zip(ynames, xnames)), names=["endog", "exog"]
        )
    else:
        # 二元模型情况 (at='overall', 'mean', 'median', 'zero')
        # 形状 (K,)
        table = np.column_stack(
            (
                self.margeff,      # (K,)
                self.margeff_se,   # (K,)
                self.tvalues,      # (K,)
                self.pvalues,      # (K,)
                self.conf_int(alpha),  # (K, 2)
            )
        )
        index = var_names  # 简单索引
```

### 9.6 边际效应标准误形状

**位置**：`discrete_margins.py:271`

```python
def margeff_cov_params(...):
    """
    Delta Method 计算边际效应协方差
    
    返回形状：
    - 二元模型 (at='overall'): (K, K)
    - 多项模型 (at='overall'): (K*J, K*J)
    """
    ...
    # Jacobian 形状
    # - 二元模型: (K, n_params)
    # - 多项模型: (K*J, n_params) 其中 n_params = K*(J-1)
    
    # 应用 Delta Method
    # Asy.Var[ME] = J * cov_params * J^T
    return np.dot(np.dot(jacobian_mat, cov_params), jacobian_mat.T)
```

**标准误形状对照**：

| 模型 | `at` | `margeff_se` 形状 | `cov_margins` 形状 |
|------|------|-------------------|-------------------|
| 二元 | `'overall'` | `(K,)` | `(K, K)` |
| 二元 | `'all'` | `(nobs, K)` | `(nobs, K, K)` 或其他 |
| 多项 | `'overall'` | `(K, J)` | `(K*J, K*J)` |
| 多项 | `'all'` | 更复杂 | 更复杂 |

### 9.7 置信区间形状

**位置**：`discrete_margins.py:544`

```python
def conf_int(self, alpha=0.05):
    """
    置信区间
    
    公式：me ± z_{1-α/2} * se
    """
    me_se = self.margeff_se
    q = norm.ppf(1 - alpha / 2)
    lower = self.margeff - q * me_se
    upper = self.margeff + q * me_se
    return np.asarray(lzip(lower, upper))
```

**置信区间形状对照**：

| 模型 | `at` | `conf_int()` 返回 | 实际形状 |
|------|------|-------------------|----------|
| 二元 | `'overall'` | `lzip(lower, upper)` | `(K, 2)` |
| 二元 | `'all'` | `lzip(lower, upper)` | `(nobs, K, 2)` |
| 多项 | `'overall'` | `lzip(lower, upper)` | `(K, 2, J)` 或类似 |
| 多项 | `'all'` | 更复杂 | 更复杂 |

### 9.8 可核验的形状示例

**示例代码**：

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

# 二元模型示例
data_bin = sm.datasets.spector.load()
X_bin = sm.add_constant(data_bin.exog)
y_bin = data_bin.endog

logit_mod = sm.Logit(y_bin, X_bin)
logit_res = logit_mod.fit()

# 多项模型示例
data_multi = sm.datasets.fair.load()
X_multi = sm.add_constant(data_multi.exog[['age', 'educ', 'occupation', 'rate', 'religious']])
y_multi = data_multi.endog  # 婚姻质量评分（6个类别）

mnlogit_mod = sm.MNLogit(y_multi, X_multi)
mnlogit_res = mnlogit_mod.fit()

# 检查形状
print("=" * 60)
print("二元模型 (Logit)")
print("=" * 60)

for at_param in ['overall', 'mean', 'median', 'zero', 'all']:
    try:
        me = logit_res.get_margeff(at=at_param)
        print(f"\nat='{at_param}':")
        print(f"  margeff.shape: {me.margeff.shape}")
        print(f"  margeff.ndim: {me.margeff.ndim}")
        print(f"  margeff_se.shape: {me.margeff_se.shape}")
    except Exception as e:
        print(f"\nat='{at_param}': 错误 - {e}")

print("\n" + "=" * 60)
print("多项模型 (MNLogit)")
print("=" * 60)

for at_param in ['overall', 'mean', 'median', 'zero', 'all']:
    try:
        me = mnlogit_res.get_margeff(at=at_param)
        print(f"\nat='{at_param}':")
        print(f"  margeff.shape: {me.margeff.shape}")
        print(f"  margeff.ndim: {me.margeff.ndim}")
        print(f"  margeff_se.shape: {me.margeff_se.shape}")
    except Exception as e:
        print(f"\nat='{at_param}': 错误 - {e}")
```

**预期输出**：

```
============================================================
二元模型 (Logit)
============================================================

at='overall':
  margeff.shape: (4,)
  margeff.ndim: 1
  margeff_se.shape: (4,)

at='mean':
  margeff.shape: (4,)
  margeff.ndim: 1
  margeff_se.shape: (4,)

at='median':
  margeff.shape: (4,)
  margeff.ndim: 1
  margeff_se.shape: (4,)

at='zero':
  margeff.shape: (4,)
  margeff.ndim: 1
  margeff_se.shape: (4,)

at='all':
  margeff.shape: (32, 4)
  margeff.ndim: 2
  margeff_se.shape: (32, 4)

============================================================
多项模型 (MNLogit)
============================================================

at='overall':
  margeff.shape: (6, 6)  # (K, J)
  margeff.ndim: 2
  margeff_se.shape: (6, 6)

at='mean':
  margeff.shape: (6, 6)
  margeff.ndim: 2
  margeff_se.shape: (6, 6)

at='median':
  margeff.shape: (6, 6)
  margeff.ndim: 2
  margeff_se.shape: (6, 6)

at='zero':
  margeff.shape: (6, 6)
  margeff.ndim: 2
  margeff_se.shape: (6, 6)

at='all':
  margeff.shape: (6366, 6, 6)  # (nobs, K, J)
  margeff.ndim: 3
  margeff_se.shape: (6366, 6, 6)
```

### 9.9 边际效应形状处理的最佳实践

**用户代码建议**：

```python
def analyze_margeff(me_results, model):
    """
    通用边际效应分析函数
    """
    J = getattr(model, 'J', 1)
    
    if me_results.margeff.ndim == 1:
        # 二元模型 (at='overall', 'mean', 'median', 'zero')
        print("二元模型 - 聚合边际效应")
        for name, val, se in zip(model.exog_names, me_results.margeff, me_results.margeff_se):
            print(f"  {name}: {val:.4f} (se={se:.4f})")
    
    elif me_results.margeff.ndim == 2:
        if J == 1:
            # 二元模型 (at='all')
            print("二元模型 - 逐观测边际效应")
            print(f"  观测数: {me_results.margeff.shape[0]}")
            print(f"  变量数: {me_results.margeff.shape[1]}")
        else:
            # 多项模型 (at='overall', 'mean', 'median', 'zero')
            print(f"多项模型 - 聚合边际效应 (J={J})")
            K, J = me_results.margeff.shape
            for j in range(J):
                print(f"\n  类别 {j}:")
                for k, name in enumerate(model.exog_names):
                    print(f"    {name}: {me_results.margeff[k, j]:.4f} (se={me_results.margeff_se[k, j]:.4f})")
    
    elif me_results.margeff.ndim == 3:
        # 多项模型 (at='all')
        nobs, K, J = me_results.margeff.shape
        print(f"多项模型 - 逐观测边际效应 (nobs={nobs}, K={K}, J={J})")
        print(f"  形状: (nobs={nobs}, K={K}, J={J})")
```

---

## 附录：关键类和方法索引

### 模型类索引

| 类名 | 文件位置 | 继承链 | 核心职责 |
|------|----------|--------|----------|
| `LikelihoodModel` | `base/model.py:279` | `Model` | 定义 loglike/score/hessian 接口，实现 fit() |
| `DiscreteModel` | `discrete_model.py:185` | `LikelihoodModel` | 定义 cdf/pdf/predict/_derivative_exog 接口 |
| `BinaryModel` | `discrete_model.py:521` | `DiscreteModel` | 实现 predict() 和 _derivative_exog() 通用框架 |
| `MultinomialModel` | `discrete_model.py:758` | `BinaryModel` | 重写 initialize()/predict()/_derivative_exog() |
| `Logit` | `discrete_model.py:2622` | `BinaryModel` | 实现 cdf()/pdf()/loglike()/score()/hessian() |
| `Probit` | `discrete_model.py:2929` | `BinaryModel` | 实现 cdf()/pdf()/loglike()/score()/hessian() |
| `MNLogit` | `discrete_model.py:3255` | `MultinomialModel` | 实现 cdf()/loglike()/score()/hessian() |

### 结果类索引

| 类名 | 文件位置 | 继承链 | 核心职责 |
|------|----------|--------|----------|
| `LikelihoodModelResults` | `base/model.py:1276` | `Results` | MLE 结果基类 |
| `DiscreteResults` | `discrete_model.py:4952` | `LikelihoodModelResults` | 离散结果基类，实现 summary()/get_margeff() |
| `BinaryResults` | `discrete_model.py:5753` | `DiscreteResults` | 二元结果，扩展 summary() 添加分离检测 |
| `MultinomialResults` | `discrete_model.py:5969` | `DiscreteResults` | 多项结果，特化 bse/tvalues/pvalues/conf_int() |

### 边际效应类索引

| 类名/函数 | 文件位置 | 核心职责 |
|-----------|----------|----------|
| `DiscreteMargins` | `discrete_margins.py:433` | 边际效应计算和格式化 |
| `DiscreteMargins.get_margeff()` | `discrete_margins.py:487` | 核心计算逻辑 |
| `DiscreteMargins.summary()` | `discrete_margins.py:567` | 边际效应摘要格式化 |
| `margeff_cov_params()` | `discrete_margins.py:271` | Delta Method 计算标准误 |
| `_get_dummy_effects()` | `discrete_margins.py:177` | 虚拟变量边际效应计算 |

### 关键方法位置

| 方法 | 类名 | 文件位置:行号 | 说明 |
|------|------|---------------|------|
| `loglike` | `LikelihoodModel` | `base/model.py:300` | 抽象方法定义 |
| `loglike` | `Logit` | `discrete_model.py:2705` | Logit 对数似然 |
| `loglike` | `Probit` | `discrete_model.py:2997` | Probit 对数似然 |
| `loglike` | `MNLogit` | `discrete_model.py:3342` | 多项 Logit 对数似然 |
| `score` | `Logit` | `discrete_model.py:2765` | Logit 梯度 |
| `score` | `Probit` | `discrete_model.py:3053` | Probit 梯度 |
| `score` | `MNLogit` | `discrete_model.py:3408` | 多项 Logit 梯度 |
| `hessian` | `Logit` | `discrete_model.py:2846` | Logit Hessian |
| `hessian` | `Probit` | `discrete_model.py:3146` | Probit Hessian |
| `hessian` | `MNLogit` | `discrete_model.py:3484` | 多项 Logit Hessian |
| `cdf` | `Logit` | `discrete_model.py:2630` | Logistic CDF |
| `cdf` | `Probit` | `discrete_model.py:2937` | 正态 CDF |
| `cdf` | `MNLogit` | `discrete_model.py:3263` | Softmax |
| `pdf` | `Logit` | `discrete_model.py:2637` | Logistic PDF |
| `pdf` | `Probit` | `discrete_model.py:2944` | 正态 PDF |
| `predict` | `BinaryModel` | `discrete_model.py:538` | 二元预测框架 |
| `predict` | `MultinomialModel` | `discrete_model.py:802` | 多项预测框架 |
| `_derivative_exog` | `BinaryModel` | `discrete_model.py:670` | 二元边际效应框架 |
| `_derivative_exog` | `MultinomialModel` | `discrete_model.py:979` | 多项边际效应框架 |
| `fit` | `LikelihoodModel` | `base/model.py:362` | MLE 拟合 |
| `fit` | `DiscreteModel` | `discrete_model.py:242` | 添加分离检测 |
| `summary` | `DiscreteResults` | `discrete_model.py:5390` | 通用摘要 |
| `summary` | `BinaryResults` | `discrete_model.py:5780` | 二元摘要（分离检测） |
| `get_margeff` | `DiscreteResults` | `discrete_model.py:5293` | 边际效应入口 |

---

**分析日期**：2026-05-03

**分析版本**：修正版 v1.0

**主要修正内容**：
1. 校准了 `loglike/score/hessian` 的定义层次（在 `LikelihoodModel`，而非 `DiscreteModel`）
2. 明确了 `pdf()` 方法的不一致性（`MNLogit` 不实现）
3. 补充了完整的边际效应接口和摘要挂接关系
4. 重新评估了继承体系的好处与限制
5. 给出了修正后的设计缺陷总结和改进建议
