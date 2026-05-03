# Statsmodels 离散选择模型继承体系分析

## 1. 继承体系结构

### 1.1 模型类继承层次

```
base.LikelihoodModel
    └── DiscreteModel (抽象基类)
         ├── BinaryModel (二元选择模型基类)
         │    ├── Logit
         │    └── Probit
         ├── MultinomialModel (多项选择模型基类)
         │    └── MNLogit
         └── CountModel (计数模型基类)
              ├── Poisson
              ├── NegativeBinomial
              ├── NegativeBinomialP
              └── GeneralizedPoisson
```

### 1.2 结果类继承层次

```
base.LikelihoodModelResults
    └── DiscreteResults (结果基类)
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

## 2. 似然、梯度、Hessian 的特化层次

### 2.1 抽象基层：DiscreteModel

`DiscreteModel` 定义了离散选择模型的抽象接口，位于 `discrete_model.py:185`：

```python
class DiscreteModel(base.LikelihoodModel):
    def cdf(self, X):
        raise NotImplementedError  # 累积分布函数
    
    def pdf(self, X):
        raise NotImplementedError  # 概率密度/质量函数
    
    def loglike(self, params):
        # 在基类中未实现，由子类特化
        pass
    
    def score(self, params):
        # 在基类中未实现，由子类特化
        pass
    
    def hessian(self, params):
        # 在基类中未实现，由子类特化
        pass
```

**职责**：
- 定义 `fit()` 和 `fit_regularized()` 方法框架
- 提供数据初始化、秩检查等通用功能
- 声明模型特化所需的抽象方法

### 2.2 中间层：BinaryModel 和 MultinomialModel

**BinaryModel** (`discrete_model.py:521`)：

```python
class BinaryModel(DiscreteModel):
    def predict(self, params, exog=None, which="mean", linear=None, offset=None):
        # 二元模型通用预测逻辑
        linpred = np.dot(exog, params) + offset
        if which == "mean":
            return self.cdf(linpred)  # 调用具体模型的 cdf
        ...
    
    def _derivative_exog(self, params, exog=None, transform="dydx", ...):
        # 边际效应计算的通用框架
        linpred = self.predict(params, exog, offset=offset, which="linear")
        margeff = np.dot(self.pdf(linpred)[:, None], params[None, :])
        # 调用具体模型的 pdf
        ...
```

**MultinomialModel** (`discrete_model.py:758`)：

```python
class MultinomialModel(BinaryModel):
    def initialize(self):
        # 多项模型特有：处理多类别数据
        self.endog = self.endog.argmax(1)  # 转换为类别索引
        self.J = self.wendog.shape[1]     # 类别数量
        self.K = self.exog.shape[1]       # 解释变量数量
        self.df_model *= self.J - 1       # 自由度调整
    
    def predict(self, params, exog=None, which="mean", linear=None):
        # 多项模型预测：添加基准类别 (全0)
        pred = super().predict(params, exog, which=which)
        if which == "linear":
            pred = np.column_stack((np.zeros(len(exog)), pred))
        return pred
```

### 2.3 具体模型特化层

#### Logit 模型 (`discrete_model.py:2622`)

**CDF 特化**：
```python
def cdf(self, X):
    """Logistic 累积分布函数"""
    return 1 / (1 + np.exp(-X))  # Λ(x) = 1/(1+e^(-x))
```

**PDF 特化**：
```python
def pdf(self, X):
    """Logistic 概率密度函数"""
    return np.exp(-X) / (1 + np.exp(-X)) ** 2  # λ(x) = Λ(x)(1-Λ(x))
```

**对数似然特化**：
```python
def loglike(self, params):
    """
    ln L = Σ ln Λ(q_i x_i'β), 其中 q = 2y - 1
    利用 Logistic 分布对称性简化计算
    """
    q = 2 * self.endog - 1
    linpred = self.predict(params, which="linear")
    return np.sum(np.log(self.cdf(q * linpred)))
```

**梯度 (Score) 特化**：
```python
def score(self, params):
    """
    ∂ln L/∂β = Σ (y_i - Λ_i) x_i
    """
    y = self.endog
    X = self.exog
    fitted = self.predict(params)  # Λ(Xβ)
    return np.dot(y - fitted, X)
```

**Hessian 特化**：
```python
def hessian(self, params):
    """
    ∂²ln L/∂β∂β' = -Σ Λ_i(1-Λ_i) x_i x_i'
    """
    X = self.exog
    L = self.predict(params)
    return -np.dot(L * (1 - L) * X.T, X)
```

#### Probit 模型 (`discrete_model.py:2929`)

**CDF 特化**：
```python
def cdf(self, X):
    """标准正态累积分布函数 Φ(x)"""
    return stats.norm._cdf(X)
```

**PDF 特化**：
```python
def pdf(self, X):
    """标准正态概率密度函数 φ(x)"""
    return stats.norm._pdf(X)
```

**对数似然特化**：
```python
def loglike(self, params):
    """
    ln L = Σ ln Φ(q_i x_i'β), 其中 q = 2y - 1
    """
    q = 2 * self.endog - 1
    linpred = self.predict(params, which="linear")
    # 使用 FLOAT_EPS 避免数值下溢
    return np.sum(np.log(np.clip(self.cdf(q * linpred), FLOAT_EPS, 1)))
```

**梯度 (Score) 特化**：
```python
def score(self, params):
    """
    ∂ln L/∂β = Σ [q_i φ(q_i x_i'β) / Φ(q_i x_i'β)] x_i
    其中 λ_i = q_i φ(q_i x_i'β) / Φ(q_i x_i'β) 称为逆米尔斯比率
    """
    y = self.endog
    X = self.exog
    XB = self.predict(params, which="linear")
    q = 2 * y - 1
    L = q * self.pdf(q * XB) / np.clip(self.cdf(q * XB), FLOAT_EPS, 1 - FLOAT_EPS)
    return np.dot(L, X)
```

**Hessian 特化**：
```python
def hessian(self, params):
    """
    ∂²ln L/∂β∂β' = -Σ λ_i(λ_i + x_i'β) x_i x_i'
    比 Logit 更复杂，涉及逆米尔斯比率的乘积
    """
    X = self.exog
    XB = self.predict(params, which="linear")
    q = 2 * self.endog - 1
    L = q * self.pdf(q * XB) / self.cdf(q * XB)  # λ_i
    return np.dot(-L * (L + XB) * X.T, X)
```

#### MNLogit 模型 (`discrete_model.py:3255`)

**CDF 特化**：
```python
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
```

**对数似然特化**：
```python
def loglike(self, params):
    """
    ln L = Σ_i Σ_j d_ij ln [exp(β_j'x_i) / Σ_k exp(β_k'x_i)]
    其中 d_ij = 1 如果个体 i 选择类别 j
    """
    params = params.reshape(self.K, -1, order="F")  # 重塑为 (K, J-1)
    d = self.wendog  # 哑变量编码的因变量
    logprob = np.log(self.cdf(np.dot(self.exog, params)))
    return np.sum(d * logprob)
```

**梯度 (Score) 特化**：
```python
def score(self, params):
    """
    ∂ln L/∂β_j = Σ_i (d_ij - P(j|x_i)) x_i, 对于 j = 1,...,J-1
    
    返回扁平化数组以便优化器使用：形状为 K*(J-1)
    """
    params = params.reshape(self.K, -1, order="F")
    # d_ij - P(j|x_i), 排除基准类别
    firstterm = self.wendog[:, 1:] - self.cdf(np.dot(self.exog, params))[:, 1:]
    # 计算后扁平化
    return np.dot(firstterm.T, self.exog).flatten()
```

**Hessian 特化**：
```python
def hessian(self, params):
    """
    ∂²ln L/∂β_j ∂β_l = -Σ_i P(j|x_i)[1(j=l) - P(l|x_i)] x_i x_i'
    
    Hessian 具有特殊结构：
    - 当 j = l 时：-Σ_i P(j|x_i)(1-P(j|x_i)) x_i x_i'
    - 当 j ≠ l 时：Σ_i P(j|x_i)P(l|x_i) x_i x_i'
    
    重塑为 ((J-1)*K, (J-1)*K) 的方阵
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
                # 对角块：-Σ P(j)(1-P(j)) x x'
                partials.append(
                    -np.dot(((pr[:, i + 1] * (1 - pr[:, j + 1]))[:, None] * X).T, X)
                )
            else:
                # 非对角块：Σ P(j)P(l) x x'
                partials.append(
                    -np.dot(((pr[:, i + 1] * -pr[:, j + 1])[:, None] * X).T, X)
                )
    
    # 重塑为方阵
    H = np.array(partials)
    H = np.transpose(H.reshape(J - 1, J - 1, K, K), (0, 2, 1, 3)).reshape(
        (J - 1) * K, (J - 1) * K
    )
    return H
```

### 2.4 特化层次总结

| 层次 | 类 | 特化内容 |
|------|-----|----------|
| 抽象接口 | `DiscreteModel` | 定义 `cdf`, `pdf`, `loglike`, `score`, `hessian` 抽象方法 |
| 中间框架 | `BinaryModel` | 实现通用 `predict`、`_derivative_exog`，调用子类 `cdf/pdf` |
| 中间框架 | `MultinomialModel` | 处理多类别数据结构，调整自由度，扩展预测逻辑 |
| 具体模型 | `Logit/Probit` | 特化 `cdf`, `pdf`, `loglike`, `score`, `hessian` |
| 具体模型 | `MNLogit` | 特化 `cdf`, `loglike`, `score`, `hessian`，处理多方程参数 |

## 3. 边际效应的统一接口

### 3.1 统一入口：get_margeff()

所有离散选择模型的结果类都继承自 `DiscreteResults`，提供统一的边际效应接口：

```python
# DiscreteResults.get_margeff() (discrete_model.py:5293)
def get_margeff(self, at="overall", method="dydx", atexog=None, dummy=False, count=False):
    """
    获取拟合模型的边际效应
    
    参数:
        at : {'overall', 'mean', 'median', 'zero', 'all'}
            - 'overall': 所有观测的平均边际效应
            - 'mean': 在解释变量均值处的边际效应
            - 'median': 在解释变量中位数处的边际效应
            - 'zero': 在解释变量为零处的边际效应
            - 'all': 每个观测的边际效应
            
        method : {'dydx', 'eyex', 'dyex', 'eydx'}
            - 'dydx': dy/dx (边际效应)
            - 'eyex': d(lny)/d(lnx) (弹性)
            - 'dyex': dy/d(lnx) (半弹性)
            - 'eydx': d(lny)/dx (半弹性)
    """
    from statsmodels.discrete.discrete_margins import DiscreteMargins
    return DiscreteMargins(self, (at, method, atexog, dummy, count))
```

### 3.2 二元模型边际效应计算

**BinaryModel._derivative_exog** (`discrete_model.py:670`)：

```python
def _derivative_exog(
    self, params, exog=None, transform="dydx", dummy_idx=None, count_idx=None, offset=None
):
    """
    计算 dF(XB)/dX，其中 F(.) 是预测概率
    
    二元模型边际效应公式：
    dy/dx = f(Xβ) * β
    其中 f(.) 是 pdf (Logit: Λ(1-Λ), Probit: φ)
    """
    linpred = self.predict(params, exog, offset=offset, which="linear")
    # 核心公式：f(Xβ) ⊗ β'
    margeff = np.dot(self.pdf(linpred)[:, None], params[None, :])
    
    # 变换处理
    if "ex" in transform:  # dy/d(lnx) 或 d(lny)/d(lnx)
        margeff *= exog
    if "ey" in transform:  # d(lny)/dx 或 d(lny)/d(lnx)
        margeff /= self.predict(params, exog)[:, None]
    
    # 处理虚拟变量和计数变量的离散变化
    return self._derivative_exog_helper(
        margeff, params, exog, dummy_idx, count_idx, transform
    )
```

**Logit vs Probit 边际效应差异**：

| 模型 | PDF | 边际效应公式 |
|------|-----|-------------|
| Logit | Λ(x)(1-Λ(x)) | ∂P(y=1|x)/∂x_k = Λ(Xβ)(1-Λ(Xβ)) β_k |
| Probit | φ(x) | ∂P(y=1|x)/∂x_k = φ(Xβ) β_k |

### 3.3 多项模型边际效应计算

**MultinomialModel._derivative_exog** (`discrete_model.py:979`)：

```python
def _derivative_exog(
    self, params, exog=None, transform="dydx", dummy_idx=None, count_idx=None
):
    """
    多项 Logit 边际效应计算
    
    公式：∂P(j|x)/∂x_k = P(j|x) * [β_jk - Σ_l P(l|x) β_lk]
    
    其中 β_0 = 0 (基准类别参数)
    """
    J = int(self.J)  # 类别数
    K = int(self.K)  # 变量数
    
    # 添加基准类别参数 (全0)
    zeroparams = np.c_[np.zeros(K), params]  # 形状 (K, J)
    
    # 计算所有类别的概率 P(j|x)
    cdf = self.cdf(np.dot(exog, params))  # 形状 (nobs, J)
    
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

### 3.4 离散变量的特殊处理

**_get_dummy_effects** (`discrete_margins.py:177`)：

```python
def _get_dummy_effects(effects, exog, dummy_ind, method, model, params):
    """
    虚拟变量的边际效应：离散变化而非导数
    
    计算 ΔP = P(X|d=1) - P(X|d=0)
    """
    for i in dummy_ind:
        exog0 = exog.copy()
        exog0[:, i] = 0
        effect0 = model.predict(params, exog0)
        exog0[:, i] = 1
        effect1 = model.predict(params, exog0)
        # 处理弹性变换
        if "ey" in method:
            effect0 = np.log(effect0)
            effect1 = np.log(effect1)
        effects[:, i] = effect1 - effect0
    return effects
```

### 3.5 边际效应标准误计算

**Delta Method** (`discrete_margins.py:271`)：

```python
def margeff_cov_params(
    model, params, exog, cov_params, at, derivative, dummy_ind, count_ind, method, J
):
    """
    使用 Delta 方法计算边际效应的方差-协方差矩阵
    
    Asy.Var[MargEff] = [∂ margeff / ∂ params] V [∂ margeff / ∂ params]'
    
    其中 V 是参数的方差-协方差矩阵
    """
    if callable(derivative):
        # 数值计算 Jacobian: ∂ margeff / ∂ params
        jacobian_mat = approx_fprime_cs(params, derivative, args=(exog, method))
        if at == "overall":
            jacobian_mat = np.mean(jacobian_mat, axis=1)
        # 处理离散变量
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
    
    # Delta 方法公式
    return np.dot(np.dot(jacobian_mat, cov_params), jacobian_mat.T)
```

### 3.6 边际效应统一接口总结

```
用户调用 results.get_margeff()
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    DiscreteMargins 类                        │
│  (discrete_margins.py:433)                                  │
├─────────────────────────────────────────────────────────────┤
│  属性：                                                      │
│  - margeff: 边际效应值                                       │
│  - margeff_se: 标准误 (Delta Method)                        │
│  - tvalues: t 统计量                                         │
│  - pvalues: p 值                                             │
│  - cov_margins: 边际效应协方差矩阵                           │
├─────────────────────────────────────────────────────────────┤
│  方法：                                                      │
│  - summary(): 返回格式化的摘要表格                           │
│  - summary_frame(): 返回 DataFrame                          │
│  - conf_int(): 置信区间                                      │
└─────────────────────────────────────────────────────────────┘
         │
         ▼ (调用 model._derivative_exog)
┌─────────────────┐    ┌─────────────────────┐
│  BinaryModel    │    │  MultinomialModel   │
│  _derivative_   │    │  _derivative_       │
│  exog           │    │  exog                │
├─────────────────┤    ├─────────────────────┤
│ 调用 pdf() 计算 │    │ 使用多项 Logit 特   │
│ f(Xβ)*β         │    │ 有公式计算          │
└─────────────────┘    └─────────────────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐    ┌─────────────────────┐
│   Logit.pdf()   │    │   MNLogit.cdf()    │
│   Probit.pdf()  │    │   (用于计算概率)    │
└─────────────────┘    └─────────────────────┘
```

## 4. 摘要格式化机制

### 4.1 结果层的继承与扩展

**DiscreteResults.summary()** (`discrete_model.py:5390`)：

```python
def summary(self, yname=None, xname=None, title=None, alpha=0.05, yname_list=None):
    """
    通用摘要格式化框架
    """
    # 顶部左侧信息
    top_left = [
        ("Dep. Variable:", None),
        ("Model:", [self.model.__class__.__name__]),  # 动态获取模型名
        ("Method:", [self.method]),
        ("Date:", None),
        ("Time:", None),
        ("converged:", ["%s" % self.mle_retvals["converged"]]),
    ]
    
    # 顶部右侧信息
    top_right = [
        ("No. Observations:", None),
        ("Df Residuals:", None),
        ("Df Model:", None),
        ("Pseudo R-squ.:", ["%#6.4g" % self.prsquared]),
        ("Log-Likelihood:", None),
        ("LL-Null:", ["%#8.5g" % self.llnull]),
        ("LLR p-value:", ["%#6.4g" % self.llr_pvalue]),
    ]
    
    # 创建 Summary 对象
    from statsmodels.iolib.summary import Summary
    smry = Summary()
    
    # 添加双列表格 (顶部信息)
    smry.add_table_2cols(
        self,
        gleft=top_left,
        gright=top_right,
        yname=yname,
        xname=xname,
        title=title or self.model.__class__.__name__ + " Regression Results",
    )
    
    # 添加参数表格 (系数、标准误、z 值、p 值、置信区间)
    smry.add_table_params(
        self, yname=yname_list, xname=xname, alpha=alpha, use_t=self.use_t
    )
    
    return smry
```

### 4.2 二元模型的特殊扩展

**BinaryResults.summary()** (`discrete_model.py:5780`)：

```python
@Appender(DiscreteResults.summary.__doc__)
def summary(self, yname=None, xname=None, title=None, alpha=0.05, yname_list=None):
    # 调用父类的通用摘要
    smry = super().summary(yname, xname, title, alpha, yname_list)
    
    # 二元模型特有：检测完全/准完全分离
    fittedvalues = self.model.cdf(self.fittedvalues)
    absprederror = np.abs(self.model.endog - fittedvalues)
    predclose_sum = (absprederror < 1e-4).sum()
    predclose_frac = predclose_sum / len(fittedvalues)
    
    etext = []
    if predclose_sum == len(fittedvalues):
        # 完全分离
        wstr = "Complete Separation: The results show that there is"
        wstr += "complete separation or perfect prediction.\n"
        wstr += "In this case the Maximum Likelihood Estimator does "
        wstr += "not exist and the parameters\n"
        wstr += "are not identified."
        etext.append(wstr)
    elif predclose_frac > 0.1:
        # 准完全分离
        wstr = "Possibly complete quasi-separation: A fraction "
        wstr += "%4.2f of observations can be\n" % predclose_frac
        wstr += "perfectly predicted. This might indicate that there "
        wstr += "is complete\nquasi-separation. In this case some "
        wstr += "parameters will not be identified."
        etext.append(wstr)
    
    if etext:
        smry.add_extra_txt(etext)
    return smry
```

### 4.3 多项模型的特殊处理

**MultinomialResults 特化**：

```python
# conf_int 特化 (discrete_model.py:6050)
def conf_int(self, alpha=0.05, cols=None):
    """
    多项模型的置信区间：返回形状为 (J, K, 2)
    而非二元模型的 (K, 2)
    """
    confint = super(DiscreteResults, self).conf_int(alpha=alpha, cols=cols)
    return confint.transpose(2, 0, 1)

# bse 特化 (discrete_model.py:6038)
@cache_readonly
def bse(self):
    """
    多项模型的标准误：重塑为 (K, J-1) 矩阵
    """
    bse = np.sqrt(np.diag(self.cov_params()))
    return bse.reshape(self.params.shape, order="F")

# summary2 特化 (discrete_model.py:6082)
def summary2(self, alpha=0.05, float_format="%.4f"):
    """
    多项模型的 summary2：每个方程单独显示
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

**DiscreteMargins.summary()** (`discrete_margins.py:567`)：

```python
def summary(self, alpha=0.05):
    """
    边际效应摘要表格
    """
    from statsmodels.iolib.summary import Summary, summary_params, table_extend
    
    # 标题
    title = model.__class__.__name__ + " Marginal Effects"
    
    # 顶部信息
    top_left = [
        ("Dep. Variable:", [model.endog_names]),
        ("Method:", [method]),  # dydx, eyex 等
        ("At:", [self.margeff_options["at"]]),  # overall, mean 等
    ]
    
    smry = Summary()
    smry.add_table_2cols(
        self, gleft=top_left, gright=[], yname=yname, xname=exog_names, title=title
    )
    
    # 多项模型：每个类别一个表格
    if J > 1:
        table = []
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

### 4.5 摘要格式化流程

```
用户调用 results.summary()
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│                  DiscreteResults.summary()                  │
│  (通用框架)                                                  │
├────────────────────────────────────────────────────────────┤
│  1. 构建 top_left / top_right 信息列表                     │
│  2. 创建 Summary 对象                                       │
│  3. 调用 smry.add_table_2cols() 添加顶部信息               │
│  4. 调用 smry.add_table_params() 添加参数表格              │
└────────────────────────────────────────────────────────────┘
         │
         ▼ (子类扩展)
┌──────────────────────┐    ┌───────────────────────────┐
│  BinaryResults       │    │  MultinomialResults       │
│  .summary()          │    │  .summary() / .summary2() │
├──────────────────────┤    ├───────────────────────────┤
│  添加分离检测警告     │    │  处理多方程参数显示        │
│  - 完全分离          │    │  - conf_int 形状调整      │
│  - 准完全分离        │    │  - bse 矩阵重塑           │
└──────────────────────┘    └───────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│              statsmodels.iolib.summary.Summary             │
├────────────────────────────────────────────────────────────┤
│  - add_table_2cols(): 双列信息表 (模型诊断信息)           │
│  - add_table_params(): 参数估计表 (系数、标准误等)        │
│  - add_extra_txt(): 附加文本 (警告信息)                   │
│  - tables[]: 表格列表                                       │
└────────────────────────────────────────────────────────────┘
```

## 5. 继承体系的好处与限制

### 5.1 好处

#### 1. 代码复用与一致性

**统一的 MLE 框架**：
```python
# DiscreteModel.fit() 提供统一的最大似然估计框架
def fit(self, start_params=None, method="newton", maxiter=35, ...):
    # 使用 Newton-Raphson 或其他优化方法
    # 调用子类的 loglike, score, hessian
    mlefit = super().fit(...)  # 调用 LikelihoodModel.fit
    return mlefit
```

**好处**：
- 所有模型共享相同的优化算法实现
- 只需在子类实现 `loglike`, `score`, `hessian`
- 统一的收敛检查、迭代控制

#### 2. 接口一致性

用户代码无需关心模型类型：
```python
# 对于任何离散选择模型，用户接口相同
model = Logit(y, X)  # 或 Probit(y, X), 或 MNLogit(y, X)
results = model.fit()

# 统一的结果接口
print(results.summary())        # 摘要
print(results.params)           # 参数估计
print(results.get_margeff())    # 边际效应
print(results.predict(X_new))   # 预测
```

#### 3. 渐进式特化

**层次化设计允许部分特化**：

| 层次 | 特化程度 | 示例 |
|------|----------|------|
| `DiscreteModel` | 定义接口 | 抽象方法声明 |
| `BinaryModel` | 部分实现 | 通用 `predict` 调用子类 `cdf` |
| `Logit` | 完全特化 | 具体的 `cdf`, `loglike`, `score` |

**好处**：
- 中间层可以实现通用逻辑，减少代码重复
- 新模型只需特化差异部分
- 易于维护和扩展

#### 4. 多态行为

**相同方法名，不同实现**：

```python
def compute_marginal_effects(model_results, params, exog):
    """
    通用函数，适用于任何离散选择模型
    """
    # 多态调用：根据实际模型类型调用不同实现
    return model_results.model._derivative_exog(params, exog)

# 使用时无需区分模型类型
logit_results = Logit(y, X).fit()
probit_results = Probit(y, X).fit()
mnlogit_results = MNLogit(y, X).fit()

# 同一函数处理所有模型
me_logit = compute_marginal_effects(logit_results, ...)
me_probit = compute_marginal_effects(probit_results, ...)
me_mnlogit = compute_marginal_effects(mnlogit_results, ...)
```

#### 5. 易于测试和验证

**统一测试框架**：
```python
# 可以针对抽象接口编写测试
class TestDiscreteModel:
    def test_loglike_sign(self, model):
        # 所有模型的对数似然应该为负
        params = np.zeros(model.exog.shape[1])
        ll = model.loglike(params)
        assert ll <= 0
    
    def test_hessian_negative_semidefinite(self, model):
        # 所有模型的 Hessian 应该是半负定的
        params = np.zeros(model.exog.shape[1])
        H = model.hessian(params)
        eigvals = np.linalg.eigvalsh(H)
        assert np.all(eigvals <= 1e-10)  # 考虑数值误差
```

### 5.2 限制

#### 1. 继承层次的僵化

**问题示例：MultinomialModel 继承 BinaryModel**

```python
# 当前设计
class MultinomialModel(BinaryModel):
    # 多项模型继承二元模型
    # 但多项模型并不是二元模型的"特殊情况"
    pass
```

**问题**：
- 语义上不合理：多项模型不是二元模型的特例
- `BinaryModel` 中的 `_continuous_ok`、`offset` 处理等对多项模型无意义
- 如果要添加新的模型类型（如有序模型），继承关系会变得复杂

**更好的设计可能是**：
```python
# 更清晰的层次
class DiscreteModel:
    pass

class BinaryChoiceModel(DiscreteModel):  # 二元选择
    pass

class MultinomialChoiceModel(DiscreteModel):  # 多项选择 (独立)
    pass

class OrderedChoiceModel(DiscreteModel):  # 有序选择
    pass
```

#### 2. 方法特化的脆弱性

**问题：Hessian 的数值计算 vs 解析计算**

```python
# 某些模型可能没有解析 Hessian
class SomeComplexModel(DiscreteModel):
    def loglike(self, params):
        # 复杂的对数似然
        ...
    
    def score(self, params):
        # 解析梯度
        ...
    
    # 没有实现 hessian()
    # 依赖基类的数值近似
```

**问题**：
- 如果基类期望 `hessian()` 存在，会导致运行时错误
- Newton 方法需要 Hessian，否则会降级使用其他方法
- 数值 Hessian 计算慢且可能不准确

**当前处理方式**：
```python
# LikelihoodModel.fit() 允许指定 method
# method='newton' 需要 hessian
# method='bfgs' 只需要 score
# method='nm' 不需要梯度
```

#### 3. 参数形状的不一致

**二元模型 vs 多项模型**：

```python
# 二元模型：参数是一维数组
logit = Logit(y, X)
logit.fit().params.shape  # (K,)

# 多项模型：参数是二维数组 (或扁平化的一维数组)
mnlogit = MNLogit(y, X)
mnlogit.fit().params.shape  # (K, J-1) 或 (K*(J-1),)
```

**问题代码**：
```python
def generic_function(results):
    # 假设 params 是一维
    params = results.params
    # 对多项模型会出错
    print(params[0])  # 对于 MNLogit，这是一整行
```

**需要特殊处理**：
```python
# MultinomialResults 中的特化
@cache_readonly
def bse(self):
    bse = np.sqrt(np.diag(self.cov_params()))
    # 需要重塑为与 params 相同的形状
    return bse.reshape(self.params.shape, order="F")
```

#### 4. 边际效应接口的复杂性

**问题：二元 vs 多项的边际效应形状**

```python
# 二元模型边际效应：形状 (nobs, K) 或 (K,)
logit_me = logit_results.get_margeff()
logit_me.margeff.shape  # (K,) 当 at='overall'

# 多项模型边际效应：形状 (nobs, K, J) 或 (K, J)
mnlogit_me = mnlogit_results.get_margeff()
mnlogit_me.margeff.shape  # (K, J) 当 at='overall'
```

**用户代码需要分支处理**：
```python
me = results.get_margeff()
if hasattr(results.model, 'J') and results.model.J > 1:
    # 多项模型：处理多维度
    for j in range(results.model.J):
        print(f"类别 {j} 的边际效应: {me.margeff[:, j]}")
else:
    # 二元模型：简单处理
    print(f"边际效应: {me.margeff}")
```

#### 5. 扩展新模型的难度

**添加新模型需要**：

1. **继承正确的基类**：
   ```python
   # 例如添加 Cloglog 模型
   class Cloglog(BinaryModel):
       # 需要理解 BinaryModel 的期望
       def cdf(self, X):
           return 1 - np.exp(-np.exp(X))  # Gumbel CDF
       
       def pdf(self, X):
           return np.exp(X - np.exp(X))
       
       # 还需要实现 loglike, score, hessian...
   ```

2. **理解隐式约定**：
   - `predict()` 的返回值含义
   - `_derivative_exog()` 的期望形状
   - 参数的扁平化顺序 (F 顺序 vs C 顺序)

3. **结果类的配套**：
   ```python
   class CloglogResults(BinaryResults):
       # 可能需要特化某些方法
       pass
   
   # 还需要 Wrapper 类
   class CloglogResultsWrapper(lm.RegressionResultsWrapper):
       pass
   
   wrap.populate_wrapper(CloglogResultsWrapper, CloglogResults)
   ```

#### 6. 多重继承的复杂性

**示例：L1 正则化结果类**：

```python
class L1PoissonResults(L1CountResults, PoissonResults):
    # 多重继承
    # 需要处理方法解析顺序 (MRO)
    pass
```

**问题**：
- `__init__` 方法的调用顺序
- 方法重写的冲突
- `super()` 的行为可能不符合预期

### 5.3 限制总结

| 限制类型 | 具体问题 | 影响 |
|----------|----------|------|
| 继承层次僵化 | MultinomialModel 继承 BinaryModel 语义不清 | 扩展困难，理解成本高 |
| 方法特化脆弱 | 依赖解析 Hessian，缺失时行为不确定 | 运行时错误，性能下降 |
| 参数形状不一致 | 二元模型 1D，多项模型 2D | 用户代码需要分支处理 |
| 边际效应复杂 | 输出形状因模型类型而异 | 接口不统一，使用困难 |
| 扩展难度大 | 需要理解隐式约定，配套多个类 | 新模型开发成本高 |
| 多重继承复杂 | L1 结果类使用多重继承 | MRO 问题，调试困难 |

## 6. 设计亮点与改进建议

### 6.1 设计亮点

#### 1. Template Method 模式的巧妙运用

**`BinaryModel.predict()` 作为模板方法**：

```python
def predict(self, params, exog=None, which="mean", ...):
    linpred = np.dot(exog, params) + offset  # 固定部分
    if which == "mean":
        return self.cdf(linpred)  # 调用子类的具体实现
    elif which == "linear":
        return linpred
    ...
```

**优点**：
- 固定算法骨架，可变部分延迟到子类
- `cdf()` 是唯一需要特化的方法
- 新增模型只需实现 `cdf()` 即可获得预测功能

#### 2. Hook 方法的设计

**`_derivative_exog_helper` 作为扩展点**：

```python
def _derivative_exog_helper(
    self, margeff, params, exog, dummy_idx, count_idx, transform
):
    """
    处理虚拟变量和计数变量的边际效应
    可由子类重写以改变行为
    """
    from .discrete_margins import _get_count_effects, _get_dummy_effects
    
    if count_idx is not None:
        margeff = _get_count_effects(margeff, exog, count_idx, transform, self, params)
    if dummy_idx is not None:
        margeff = _get_dummy_effects(margeff, exog, dummy_idx, transform, self, params)
    
    return margeff
```

#### 3. Wrapper 模式的结果封装

```python
# 结果类和 Wrapper 类分离
class BinaryResults(DiscreteResults):
    # 核心逻辑
    pass

class BinaryResultsWrapper(lm.RegressionResultsWrapper):
    # 用于 pandas 等的适配
    _attrs = {
        "resid_dev": "rows",
        "resid_generalized": "rows",
        ...
    }

# 自动填充 Wrapper
wrap.populate_wrapper(BinaryResultsWrapper, BinaryResults)
```

**优点**：
- 核心逻辑与接口适配分离
- 易于添加新的接口适配
- 保持结果类的纯净

### 6.2 改进建议

#### 1. 重构继承层次

**建议的新层次**：

```
DiscreteModel (抽象基类)
    ├── BinaryChoiceModel (二元选择)
    │    ├── Logit
    │    ├── Probit
    │    └── Cloglog
    ├── MultinomialChoiceModel (多项选择)
    │    └── MNLogit
    ├── OrderedChoiceModel (有序选择)
    │    ├── OrderedLogit
    │    └── OrderedProbit
    └── CountModel (计数模型)
         ├── Poisson
         └── NegativeBinomial
```

**改动要点**：
- `MultinomialModel` 不再继承 `BinaryModel`
- 提取 `ChoiceModel` 中间层（如适用）
- 明确各层的职责边界

#### 2. 使用组合替代继承

**边际效应计算的组合设计**：

```python
# 当前设计：继承层次中的方法
class BinaryModel:
    def _derivative_exog(self, ...):
        # 二元模型实现
        pass

class MultinomialModel:
    def _derivative_exog(self, ...):
        # 多项模型实现
        pass

# 建议设计：策略模式
class MarginalEffectCalculator:
    """边际效应计算策略接口"""
    def compute(self, model, params, exog, transform):
        raise NotImplementedError

class BinaryMarginalEffectCalculator(MarginalEffectCalculator):
    def compute(self, model, params, exog, transform):
        # 二元模型边际效应
        linpred = np.dot(exog, params)
        margeff = np.dot(model.pdf(linpred)[:, None], params[None, :])
        return margeff

class MultinomialMarginalEffectCalculator(MarginalEffectCalculator):
    def compute(self, model, params, exog, transform):
        # 多项模型边际效应
        J = model.J
        zeroparams = np.c_[np.zeros(model.K), params]
        cdf = model.cdf(np.dot(exog, params))
        # ... 多项模型特有计算
        return margeff

# 在模型中组合使用
class Logit(BinaryChoiceModel):
    def __init__(self, ...):
        super().__init__(...)
        self._margeff_calculator = BinaryMarginalEffectCalculator()
    
    def _derivative_exog(self, ...):
        return self._margeff_calculator.compute(self, ...)
```

**优点**：
- 运行时可切换策略
- 新增边际效应计算方法无需修改模型类
- 更好的单一职责原则

#### 3. 统一参数形状

**建议：始终使用扁平化参数，但提供重塑方法**：

```python
class DiscreteModel:
    @property
    def param_shape(self):
        """返回参数的逻辑形状"""
        raise NotImplementedError
    
    def reshape_params(self, flat_params):
        """将扁平化参数重塑为逻辑形状"""
        return flat_params.reshape(self.param_shape, order="F")
    
    def flatten_params(self, shaped_params):
        """将逻辑形状参数扁平化"""
        return shaped_params.flatten(order="F")

# 使用示例
class MNLogit(MultinomialChoiceModel):
    @property
    def param_shape(self):
        return (self.K, self.J - 1)

# 用户代码
params = results.params  # 始终是扁平化的 (K*(J-1),)
shaped = results.model.reshape_params(params)  # (K, J-1)
```

#### 4. 显式接口定义

**使用抽象基类 (ABC) 强制接口**：

```python
from abc import ABC, abstractmethod

class DiscreteModel(ABC, base.LikelihoodModel):
    @abstractmethod
    def cdf(self, X):
        """累积分布函数"""
        pass
    
    @abstractmethod
    def pdf(self, X):
        """概率密度/质量函数"""
        pass
    
    @abstractmethod
    def loglike(self, params):
        """对数似然"""
        pass
    
    # 可选方法使用 None 或默认实现
    def hessian(self, params):
        """
        Hessian 矩阵（可选）
        如果未实现，使用数值近似
        """
        from statsmodels.tools.numdiff import approx_hess
        return approx_hess(params, self.loglike)
```

**优点**：
- 编译时（定义时）检查接口完整性
- 明确区分必需方法和可选方法
- 更好的文档化

## 7. 总结

### 7.1 核心设计模式

| 模式 | 应用场景 | 实现位置 |
|------|----------|----------|
| **Template Method** | 统一的 MLE 估计流程 | `DiscreteModel.fit()` |
| **Factory Method** | 结果对象创建 | `fit()` 返回具体 Results 类 |
| **Strategy** | 边际效应计算（隐式） | `_derivative_exog` 多态 |
| **Wrapper** | 结果类接口适配 | `*ResultsWrapper` 类 |
| **Hook Method** | 扩展点设计 | `_derivative_exog_helper` |

### 7.2 关键特化点

| 特化内容 | 特化层次 | 特化方式 |
|----------|----------|----------|
| CDF/PDF | 具体模型层 | `Logit.cdf()`, `Probit.cdf()`, `MNLogit.cdf()` |
| 对数似然 | 具体模型层 | 各模型 `loglike()` 方法 |
| 梯度 (Score) | 具体模型层 | 各模型 `score()` 方法 |
| Hessian | 具体模型层 | 各模型 `hessian()` 方法 |
| 预测逻辑 | 中间层 (`BinaryModel`) | 模板方法调用子类 `cdf()` |
| 边际效应计算 | 中间层 + 具体模型 | `_derivative_exog` 调用子类 `pdf()` |
| 摘要格式化 | 结果层 | `DiscreteResults.summary()` 框架 + 子类扩展 |

### 7.3 权衡取舍

**好处**：
- ✅ 代码复用：统一的 MLE 框架、预测逻辑
- ✅ 接口一致：用户无需关心模型类型差异
- ✅ 渐进扩展：新增模型只需特化差异部分
- ✅ 多态行为：通用函数可处理所有模型类型

**限制**：
- ❌ 继承层次语义不清：多项模型继承二元模型
- ❌ 参数形状不一致：二元 1D vs 多项 2D
- ❌ 隐式约定多：需要理解 F 顺序、参数扁平化等
- ❌ 多重继承复杂：L1 结果类的 MRO 问题

### 7.4 实际使用建议

1. **对于用户**：
   - 利用统一接口：`fit()`, `summary()`, `get_margeff()`, `predict()`
   - 注意多项模型的参数形状：使用 `reshape` 或访问方程
   - 边际效应解释：二元模型直接解释，多项模型注意类别对应

2. **对于扩展者**：
   - 继承正确的基类：二元模型继承 `BinaryModel`
   - 必需方法：`cdf()`, `pdf()`, `loglike()`, `score()`, `hessian()`
   - 配套结果类：继承相应的 `*Results` 类
   - 不要忘记 Wrapper 类：使用 `wrap.populate_wrapper()`

3. **对于维护者**：
   - 考虑重构继承层次：分离 `BinaryChoice` 和 `MultinomialChoice`
   - 引入策略模式：边际效应、梯度计算等可替换组件
   - 显式化接口：使用 ABC 和类型注解
   - 统一参数形状：始终扁平化，提供重塑方法

## 附录：关键类索引

### 模型类

| 类名 | 文件位置 | 继承链 | 主要职责 |
|------|----------|--------|----------|
| `DiscreteModel` | `discrete_model.py:185` | `base.LikelihoodModel` | 抽象基类，定义接口 |
| `BinaryModel` | `discrete_model.py:521` | `DiscreteModel` | 二元模型通用框架 |
| `MultinomialModel` | `discrete_model.py:758` | `BinaryModel` | 多项模型通用框架 |
| `CountModel` | `discrete_model.py:1037` | `DiscreteModel` | 计数模型通用框架 |
| `Logit` | `discrete_model.py:2622` | `BinaryModel` | Logistic 回归 |
| `Probit` | `discrete_model.py:2929` | `BinaryModel` | Probit 模型 |
| `MNLogit` | `discrete_model.py:3255` | `MultinomialModel` | 多项 Logit 模型 |

### 结果类

| 类名 | 文件位置 | 继承链 | 主要职责 |
|------|----------|--------|----------|
| `DiscreteResults` | `discrete_model.py:4952` | `base.LikelihoodModelResults` | 结果基类 |
| `BinaryResults` | `discrete_model.py:5753` | `DiscreteResults` | 二元结果 |
| `MultinomialResults` | `discrete_model.py:5969` | `DiscreteResults` | 多项结果 |
| `CountResults` | `discrete_model.py:5522` | `DiscreteResults` | 计数结果 |
| `DiscreteMargins` | `discrete_margins.py:433` | - | 边际效应计算 |

### 关键方法位置

| 方法 | 类名 | 行号 | 说明 |
|------|------|------|------|
| `loglike` | `Logit` | 2705 | Logit 对数似然 |
| `loglike` | `Probit` | 2997 | Probit 对数似然 |
| `loglike` | `MNLogit` | 3342 | 多项 Logit 对数似然 |
| `score` | `Logit` | 2765 | Logit 梯度 |
| `score` | `Probit` | 3053 | Probit 梯度 |
| `score` | `MNLogit` | 3408 | 多项 Logit 梯度 |
| `hessian` | `Logit` | 2846 | Logit Hessian |
| `hessian` | `Probit` | 3146 | Probit Hessian |
| `hessian` | `MNLogit` | 3484 | 多项 Logit Hessian |
| `_derivative_exog` | `BinaryModel` | 670 | 二元边际效应计算 |
| `_derivative_exog` | `MultinomialModel` | 979 | 多项边际效应计算 |
| `get_margeff` | `DiscreteResults` | 5293 | 边际效应统一入口 |
| `summary` | `DiscreteResults` | 5390 | 通用摘要格式化 |
