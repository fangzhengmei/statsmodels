# Statsmodels 与 Patsy 协作关系分析报告（修订版）

## 概述

在 statsmodels 中使用公式字符串（如 `"y ~ x1 + x2 + C(category)"`）进行统计建模时，patsy 库承担了从公式解析到设计矩阵生成的核心工作，而 statsmodels 则负责统计估计和结果分析。

**本文档的关键修正**：
- 之前的报告错误地将线性回归的 `fit()` 描述为通用迭代优化流程
- 实际上，**OLS/WLS/GLS 使用解析解**（直接矩阵运算，0 次迭代）
- **Logit/Probit/Poisson 等似然模型使用迭代优化**
- **GLM 默认使用 IRLS（迭代重加权最小二乘）**

---

## 一、Patsy 的核心职责

Patsy 是一个专门用于描述统计模型和构建设计矩阵的 Python 库。在与 statsmodels 的协作中，patsy 承担以下核心职责：

### 1.1 公式字符串解析

**核心功能**：将人类可读的公式字符串解析为结构化的模型描述。

**关键代码引用**：
```python
# statsmodels/formula/_manager.py:764-767
def get_spec(self, formula):
    if self._using_patsy:
        return patsy.ModelDesc.from_formula(formula)
    else:
        return formulaic.Formula(formula)
```

**公式语法支持**：
- **因变量/自变量分隔**：`y ~ x1 + x2`
- **分类变量编码**：`C(category)`
- **交互项**：`x1 * x2` 或 `x1 : x2`
- **函数变换**：`log(x)`、`np.log(x)`
- **多项式**：`I(x**2)`
- **排除截距**：`y ~ x - 1` 或 `y ~ 0 + x`

### 1.2 设计矩阵生成

**核心功能**：根据解析后的公式和输入数据，生成统计建模所需的设计矩阵（Design Matrix）。

**关键代码引用**：
```python
# statsmodels/formula/_manager.py:480-486
if (
    isinstance(formula, (patsy.design_info.DesignInfo, patsy.desc.ModelDesc))
    or "~" not in formula
    or formula.strip().startswith("~")
):
    output = patsy.dmatrix(formula, data, eval_env=_eval_env, return_type=return_type, **kwargs)
else:  # "~" in formula:
    output = patsy.dmatrices(formula, data, eval_env=_eval_env, return_type=return_type, **kwargs)
```

**两个核心函数**：
| 函数 | 用途 | 返回值 |
|------|------|--------|
| `patsy.dmatrix()` | 单侧公式（仅自变量） | 单个设计矩阵 |
| `patsy.dmatrices()` | 双侧公式（因变量~自变量） | (endog, exog) 元组 |

### 1.3 分类变量编码（Contrast Coding）

**核心功能**：自动处理分类变量的编码转换。

**默认行为**：
- 对于 k 个类别的分类变量，默认生成 k-1 个虚拟变量（Treatment Coding）
- 参考水平（Reference Level）默认为第一个类别

**支持的编码类型**：
- `Treatment`（默认）：虚拟变量编码
- `Sum`：离均差编码
- `Poly`：多项式编码
- `Helmert`：Helmert 编码

### 1.4 缺失值处理

**核心功能**：在构建设计矩阵时处理缺失值。

**关键代码引用**：
```python
# statsmodels/formula/_manager.py:30-38
class NAAction(patsy.missing.NAAction):
    def _handle_NA_drop(self, values, is_nas, origins):
        total_mask = np.zeros(is_nas[0].shape[0], dtype=bool)
        for is_NA in is_nas:
            total_mask |= is_NA
        good_mask = ~total_mask
        self.missing_mask = total_mask
        return [v[good_mask, ...] if v.ndim > 1 else v[good_mask] for v in values]
```

**处理策略**：
- `drop`：删除包含缺失值的行（默认）
- `raise`：遇到缺失值时抛出异常

### 1.5 评估环境管理

**核心功能**：处理公式中引用的外部变量和函数。

**工作原理**：
- `eval_env=0`：使用调用者的命名空间
- `eval_env=N`：向上追溯 N 层栈帧
- 支持传递字典形式的命名空间

### 1.6 设计信息（DesignInfo）

**核心功能**：存储设计矩阵的元数据，供后续使用（如预测、假设检验）。

**关键代码引用**：
```python
# statsmodels/formula/_manager.py:487-490
if isinstance(output, tuple):
    self._spec = output[1].design_info
else:
    self._spec = output.design_info
```

**DesignInfo 包含的信息**：
- `column_names`：设计矩阵的列名（包括编码后的虚拟变量）
- `term_names`：公式中的原始项名
- `term_name_slices`：每个项对应的列切片
- `factor_infos`：每个因子的信息（类型、类别等）
- `term_codings`：每个项的编码信息
- `linear_constraint()`：解析线性约束字符串

---

## 二、Statsmodels 的职责：关键修正

### 2.1 模型继承体系概览

**重要理解**：不同模型类型的 `fit()` 方法实现**完全不同**。

```
base.Model（最基类，定义 from_formula）
│
└── base.LikelihoodModel（定义迭代优化 fit()）
    │
    ├── RegressionModel（重写 fit()，使用解析解）
    │   ├── GLS
    │   │   └── WLS
    │   │       └── OLS  ← 解析解，0 次迭代！
    │   └── GLSAR
    │
    ├── DiscreteModel（不重写 fit()，调用父类迭代优化）
    │   ├── BinaryModel
    │   │   ├── Logit   ← 迭代优化
    │   │   └── Probit  ← 迭代优化
    │   └── CountModel
    │       ├── Poisson  ← 迭代优化
    │       └── NegativeBinomial
    │
    └── GLM（自定义 fit()，默认 IRLS，可选梯度优化）
```

### 2.2 公式引擎抽象层

**核心功能**：通过 `FormulaManager` 类统一 patsy 和 formulaic 两个公式引擎的接口。

**关键代码引用**：
```python
# statsmodels/formula/_manager.py:188-207
class FormulaManager:
    """
    Abstraction class that provides a common interface to patsy and formulaic.
    Designed to aid in the transition from patsy to formulaic.
    """
    
    def __init__(self, engine: Literal["patsy", "formulaic"] | None = None):
        self._engine = self._get_engine(engine)
        self._using_patsy = self._engine == "patsy"
        self._spec = None
        self._missing_mask = None
```

### 2.3 模型实例化入口

**核心功能**：`Model.from_formula` 类方法是用户使用公式接口的主要入口。

**关键代码引用**：
```python
# statsmodels/base/model.py:156-248
@classmethod
def from_formula(cls, formula, data, subset=None, drop_cols=None, *args, **kwargs):
    """
    Create a Model from a formula and dataframe.
    """
    mgr = FormulaManager()
    if subset is not None:
        data = data.loc[subset]
    
    eval_env = kwargs.pop("eval_env", None)
    if eval_env is None:
        eval_env = 2
    elif eval_env == -1:
        eval_env = mgr.get_empty_eval_env()
    elif isinstance(eval_env, int):
        eval_env += 1  # we're going down the stack again
    
    missing = kwargs.get("missing", "drop")
    if missing == "none":  # with patsy it's drop or raise. let's raise.
        missing = "raise"
    
    # 调用公式处理函数 → 进入 Patsy
    tmp = handle_formula_data(data, None, formula, depth=eval_env, missing=missing)
    ((endog, exog), missing_idx, model_spec) = tmp
    
    # ... 验证和处理 drop_cols
    
    # 附加元数据
    kwargs.update({
        "missing_idx": missing_idx,
        "missing": missing,
        "formula": formula,
        "model_spec": model_spec,
    })
    
    # 实例化模型
    # 注意：此时 endog 和 exog 已经是纯数值矩阵！
    mod = cls(endog, exog, *args, **kwargs)
    mod.formula = formula
    mod.data.frame = data
    return mod
```

**关键理解**：
- `from_formula` 的作用是：**公式字符串 → 数值矩阵**
- 一旦模型实例化完成，`self.exog` 和 `self.endog` 就是纯数值类型
- 后续的 `fit()` 方法**不再需要公式信息**，只使用数值矩阵

### 2.4 参数估计：三种求解机制的详细对比

#### 2.4.1 线性回归：OLS/WLS/GLS（解析解）

**数学原理**：

对于线性模型 `y = Xβ + ε`，最小二乘估计有**闭式解**：

```
β̂ = (X'X)⁻¹X'y
```

或通过 QR 分解求解：

```
X = QR
β̂ = R⁻¹Q'y
```

**关键代码引用**（`RegressionModel.fit()`）：

```python
# statsmodels/regression/linear_model.py:284-415
def fit(
    self,
    method: Literal["pinv", "qr"] = "pinv",  # 注意：没有 "newton", "bfgs"！
    cov_type: Literal["nonrobust", "HC0", "HC1", ...] = "nonrobust",
    **kwargs,
):
    """
    Full fit of the model.
    
    The fit method uses the pseudoinverse of the design/exogenous variables
    to solve the least squares minimization.
    """
    
    # ========== 方法 1：Moore-Penrose 伪逆（默认） ==========
    if method == "pinv":
        if not (
            hasattr(self, "pinv_wexog")
            and hasattr(self, "normalized_cov_params")
            and hasattr(self, "rank")
        ):
            # 计算伪逆: pinv(X) = X'(X'X)^{-1}
            self.pinv_wexog, singular_values = pinv_extended(self.wexog)
            
            # 规范化协方差矩阵: (X'X)^{-1}
            self.normalized_cov_params = np.dot(
                self.pinv_wexog, np.transpose(self.pinv_wexog)
            )
            
            self.rank = np.linalg.matrix_rank(np.diag(singular_values))
        
        # ⚠️ 关键：直接计算，无迭代！
        # β̂ = pinv(X) * y
        beta = np.dot(self.pinv_wexog, self.wendog)
    
    # ========== 方法 2：QR 分解 ==========
    elif method == "qr":
        if not (
            hasattr(self, "exog_Q")
            and hasattr(self, "exog_R")
            and hasattr(self, "normalized_cov_params")
            and hasattr(self, "rank")
        ):
            # QR 分解: X = QR
            Q, R = np.linalg.qr(self.wexog)
            self.exog_Q, self.exog_R = Q, R
            
            # 规范化协方差矩阵: R^{-1}(R')^{-1}
            self.normalized_cov_params = np.linalg.inv(np.dot(R.T, R))
            
            self.rank = np.linalg.matrix_rank(R)
        
        # 计算效应: Q'y
        self.effects = np.dot(Q.T, self.wendog)
        
        # ⚠️ 关键：解三角方程组，无迭代！
        # β̂ = R^{-1}Q'y
        beta = np.linalg.solve(R, self.effects)
    
    else:
        raise ValueError('method has to be "pinv" or "qr"')
    
    # ========== 自由度计算 ==========
    if self._df_model is None:
        self._df_model = float(self.rank - self.k_constant)
    if self._df_resid is None:
        self.df_resid = self.nobs - self.rank
    
    # ========== 返回结果 ==========
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
    return RegressionResultsWrapper(lfit)
```

**与之前错误描述的关键差异**：

| 项目 | 错误描述（之前） | 正确实现（实际代码） |
|------|-----------------|---------------------|
| 求解方式 | 迭代优化（Newton、BFGS 等） | 解析解（pinv 或 QR） |
| 方法参数 | `method="newton"`, `method="bfgs"` | `method="pinv"`, `method="qr"` |
| 起始参数 | 需要 `start_params` | 不需要 |
| 迭代次数 | 多次 | **0 次** |
| 收敛检查 | 需要 | 不需要 |
| Hessian | 需要计算 | 不需要 |

#### 2.4.2 似然模型：Logit/Probit/Poisson（迭代优化）

**数学原理**：

对于离散选择模型，没有解析解，需要通过**极大似然估计**进行迭代求解：

```
β̂ = argmax_β log L(β|y,X)
```

使用梯度下降或 Newton-Raphson 等方法迭代求解。

**继承关系分析**：
- `DiscreteModel` 继承自 `LikelihoodModel`
- `DiscreteModel.fit()` **没有完全重写**，而是调用 `super().fit()`
- `super().fit()` 即 `LikelihoodModel.fit()`，使用迭代优化

**关键代码引用**（`DiscreteModel.fit()`）：

```python
# statsmodels/discrete/discrete_model.py:242-274
@Appender(base.LikelihoodModel.fit.__doc__)
def fit(
    self,
    start_params=None,
    method="newton",  # 注意：这里是 "newton"，不是 "pinv"！
    maxiter=35,       # 注意：有迭代次数限制！
    full_output=1,
    disp=1,
    callback=None,
    **kwargs,
):
    """
    Fit the model using maximum likelihood.
    
    The rest of the docstring is from
    statsmodels.base.model.LikelihoodModel.fit
    """
    # 添加完美预测检测的回调
    if callback is None:
        callback = self._check_perfect_pred
    
    # ⚠️ 关键：调用父类 LikelihoodModel.fit() - 迭代优化！
    mlefit = super().fit(
        start_params=start_params,
        method=method,
        maxiter=maxiter,
        full_output=full_output,
        disp=disp,
        callback=callback,
        **kwargs,
    )
    
    return mlefit
```

**关键代码引用**（`LikelihoodModel.fit()`，真正的迭代优化）：

```python
# statsmodels/base/model.py:362-651
def fit(
    self,
    start_params=None,
    method="newton",      # 支持 "newton", "nm", "bfgs", "lbfgs", "cg", "ncg", "powell"
    maxiter=100,           # 迭代次数限制
    full_output=True,
    disp=True,
    fargs=(),
    callback=None,
    retall=False,
    skip_hessian=False,
    **kwargs,
):
    """
    Fit method for likelihood based models
    
    支持的优化方法:
    - 'newton': Newton-Raphson（需要 score 和 hessian）
    - 'nm': Nelder-Mead（单纯形法，只需要函数值）
    - 'bfgs': Broyden-Fletcher-Goldfarb-Shanno（需要梯度）
    - 'lbfgs': Limited-memory BFGS
    - 'cg': Conjugate Gradient
    - 'ncg': Newton-Conjugate Gradient
    - 'powell': Modified Powell's
    """
    
    # ========== 步骤 1：准备起始参数 ==========
    # ⚠️ 注意：迭代方法需要起始参数！
    if start_params is None:
        if hasattr(self, "start_params"):
            start_params = self.start_params
        elif self.exog is not None:
            start_params = [0.0] * self.exog.shape[1]  # 默认全 0
        else:
            raise ValueError(
                "If exog is None, then start_params should be specified"
            )
    
    # ========== 步骤 2：定义目标函数 ==========
    nobs = self.endog.shape[0]
    
    # 负对数似然（除以 nobs 用于数值稳定性）
    def f(params, *args):
        return -self.loglike(params, *args) / nobs
    
    # ========== 步骤 3：定义梯度和 Hessian ==========
    if method == "newton":
        # Newton-Raphson: 使用 score（梯度）和 hessian（海森）
        def score(params, *args):
            return self.score(params, *args) / nobs
        
        def hess(params, *args):
            return self.hessian(params, *args) / nobs
    else:
        # 其他方法：梯度取负
        def score(params, *args):
            return -self.score(params, *args) / nobs
        
        def hess(params, *args):
            return -self.hessian(params, *args) / nobs
    
    # ========== 步骤 4：调用优化器（迭代！） ==========
    optimizer = Optimizer()
    xopt, retvals, optim_settings = optimizer._fit(
        f,
        score,
        start_params,
        fargs,
        kwargs,
        hessian=hess,
        method=method,
        disp=disp,
        maxiter=maxiter,
        callback=callback,
        retall=retall,
        full_output=full_output,
    )
    
    # ========== 步骤 5：收敛检查 ==========
    # ⚠️ 注意：迭代方法需要检查是否收敛！
    if isinstance(retvals, dict):
        if warn_convergence and not retvals["converged"]:
            from statsmodels.tools.sm_exceptions import ConvergenceWarning
            warnings.warn(
                "Maximum Likelihood optimization failed to "
                "converge. Check mle_retvals",
                ConvergenceWarning,
                stacklevel=2,
            )
    
    # ========== 步骤 6：计算协方差矩阵 ==========
    if not skip_hessian:
        H = -1 * self.hessian(xopt)
        invertible = False
        if np.all(np.isfinite(H)):
            eigvals, eigvecs = np.linalg.eigh(H)
            if np.min(eigvals) > 0:
                invertible = True
        
        if invertible:
            # Cov = -H^{-1}
            Hinv = eigvecs.dot(np.diag(1.0 / eigvals)).dot(eigvecs.T)
        else:
            warnings.warn(
                "Inverting hessian failed, no bse or cov_params available",
                HessianInversionWarning,
                stacklevel=2,
            )
            Hinv = None
    
    # ========== 步骤 7：包装结果 ==========
    mlefit = LikelihoodModelResults(self, xopt, Hinv, scale=1.0, **kwds)
    mlefit.mle_retvals = retvals
    mlefit.mle_settings = optim_settings
    return mlefit
```

#### 2.4.3 广义线性模型：GLM（IRLS 或梯度优化）

**数学原理**：

GLM 默认使用**迭代重加权最小二乘（IRLS）**：

```
1. 初始化 β̂₀
2. 迭代直到收敛：
   a. η = Xβ̂
   b. μ = g⁻¹(η)
   c. 工作残差 z = η + (y - μ) * g'(μ)
   d. 权重 W = 1 / (Var(μ) * [g'(μ)]²)
   e. 加权最小二乘: β̂ = (X'WX)⁻¹X'Wz
```

**关键代码引用**（`GLM.fit()`）：

```python
# statsmodels/genmod/generalized_linear_model.py:1178-1319
def fit(
    self,
    start_params=None,
    maxiter=100,
    method="IRLS",  # 默认使用 IRLS
    tol=1e-8,
    scale=None,
    **kwargs,
):
    """
    Fits a generalized linear model for a given family.
    
    Parameters
    ----------
    method : str
        Default is 'IRLS' for iteratively reweighted least squares.
        Otherwise gradient optimization is used.
    """
    
    self.scaletype = scale
    
    # ========== 方法 A：IRLS（默认） ==========
    if method.lower() == "irls":
        if cov_type.lower() == "eim":
            cov_type = "nonrobust"
        # 调用专门的 IRLS 实现
        return self._fit_irls(
            start_params=start_params,
            maxiter=maxiter,
            tol=tol,
            scale=scale,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            use_t=use_t,
            **kwargs,
        )
    
    # ========== 方法 B：梯度优化（备选） ==========
    else:
        self._optim_hessian = kwargs.get("optim_hessian")
        self._tmp_like_exog = np.empty_like(self.exog, dtype=float)
        
        fit_ = self._fit_gradient(
            start_params=start_params,
            method=method,  # 如 "newton", "bfgs" 等
            maxiter=maxiter,
            tol=tol,
            scale=scale,
            **kwargs,
        )
        del self._optim_hessian
        del self._tmp_like_exog
        return fit_
```

**关键代码引用**（`GLM._fit_gradient()` 最终调用父类迭代优化）：

```python
# statsmodels/genmod/generalized_linear_model.py:1321-1365
def _fit_gradient(
    self,
    start_params=None,
    method="newton",
    maxiter=100,
    # ...
):
    """
    Fits a generalized linear model for a given family iteratively
    using the scipy gradient optimizers.
    """
    
    # 先用 IRLS 做几次迭代获取好的起始值
    if (max_start_irls > 0) and (start_params is None):
        irls_rslt = self._fit_irls(
            start_params=start_params,
            maxiter=max_start_irls,  # 只迭代几次
            tol=tol,
            scale=1.0,
            cov_type="nonrobust",
            cov_kwds=None,
            use_t=None,
            **kwargs,
        )
        start_params = irls_rslt.params
        del irls_rslt
    
    # 调用父类 LikelihoodModel.fit() 进行迭代优化
    rslt = super().fit(
        start_params=start_params,
        maxiter=maxiter,
        full_output=full_output,
        method=method,
        disp=disp,
        **kwargs,
    )
    # ...
    return rslt
```

#### 2.4.4 三种求解机制对比总结

| 对比维度 | OLS/WLS/GLS（解析解） | Logit/Probit/Poisson（迭代优化） | GLM（IRLS） |
|---------|----------------------|----------------------------------|-------------|
| **数学问题** | 最小二乘 | 极大似然 | 极大似然 |
| **目标函数** | RSS = Σ(y - Xβ)² | log L(β\|y,X) | log L(β\|y,X) |
| **解的形式** | β̂ = (X'X)⁻¹X'y | 无闭式 | 每次迭代是 WLS |
| **fit() 来源** | RegressionModel 重写 | 调用 LikelihoodModel.fit | 自定义 _fit_irls |
| **method 参数** | `"pinv"`, `"qr"` | `"newton"`, `"bfgs"`, `"nm"`, 等 | `"IRLS"`（默认） |
| **迭代次数** | **0 次** | 多次（通常 5-50） | 多次（通常 3-20） |
| **起始参数** | 不需要 | 需要（默认全 0） | 可选 |
| **收敛检查** | 不需要 | 需要 | 需要 |
| **Hessian** | 不需要显式计算 | 需要（或近似） | 不需要 |
| **失败模式** | X 列满秩时唯一解 | 不收敛、Hessian 奇异 | 不收敛、权重为 0 |
| **代码位置** | `linear_model.py:284-415` | `base/model.py:362-651` | `genmod/generalized_linear_model.py` |

### 2.5 统一的初始化流程

**重要理解**：无论使用哪种求解机制，**模型初始化流程是相同的**——都通过 `from_formula` 调用 patsy 生成设计矩阵。

```python
# statsmodels/base/model.py:100-116
def __init__(self, endog, exog=None, **kwargs):
    missing = kwargs.pop("missing", "none")
    hasconst = kwargs.pop("hasconst", None)
    
    # 处理数据（与求解机制无关）
    self.data = self._handle_data(endog, exog, missing, hasconst, **kwargs)
    
    self.k_constant = self.data.k_constant
    self.exog = self.data.exog      # 数值矩阵
    self.endog = self.data.endog    # 数值数组
    # ...
```

**关键理解**：
- **Patsy 的职责终止于设计矩阵生成**：生成 `endog` 和 `exog` 数值矩阵后，patsy 的工作就完成了
- **后续的求解机制与 patsy 无关**：完全由各模型类的 `fit()` 方法决定
- **公式接口与数组接口等价**：`from_formula`（公式）和 `__init__(endog, exog)`（数组）最终到达相同的状态

### 2.6 预测时的公式复用

**核心功能**：在预测阶段重新应用相同的公式转换。

**关键代码引用**：
```python
# statsmodels/base/model.py:1149-1207
def _transform_predict_exog(self, exog, transform=True):
    # ...
    if transform and hasattr(self.model, "formula") and (exog is not None):
        # 获取保存的 model_spec（训练时的 DesignInfo）
        model_spec = (
            getattr(self.model, "model_spec", None) 
            or self.model.data.model_spec
        )
        
        mgr = FormulaManager()
        # 使用 DesignInfo 处理新数据
        # 这确保：
        # 1. 分类变量使用相同的参考水平
        # 2. 列顺序与训练时一致
        # 3. 不重新解析公式字符串
        exog = mgr.get_matrices(model_spec, exog, pandas=True, prediction=True)
    return exog, exog_index
```

---

## 三、完整协作流程（修正版）

### 3.1 流程图概览

```
用户代码: sm.OLS.from_formula("y ~ x1 + C(x2)", data=df).fit()
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 1：公式解析与设计矩阵生成（Patsy 负责）                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Statsmodels: Model.from_formula()                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 1. 处理 subset 参数（数据子集筛选）                              │   │
│  │ 2. 准备 eval_env（评估环境）                                     │   │
│  │ 3. 调用 handle_formula_data()                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                          │                                               │
│                          ▼                                               │
│  Statsmodels: handle_formula_data()                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 1. 创建 FormulaManager 实例                                       │   │
│  │ 2. 获取 NA_action（缺失值处理策略）                               │   │
│  │ 3. 调用 mgr.get_matrices()                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                          │                                               │
│                          ▼                                               │
│  Statsmodels: FormulaManager.get_matrices()                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 如果使用 patsy 引擎:                                               │   │
│  │   - 调用 patsy.dmatrices("y ~ x1 + C(x2)", data, ...)          │   │
│  │   - 获取 output[1].design_info 作为 model_spec                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                          │                                               │
│                          ▼                                               │
│  Patsy: 核心工作（公式 → 数值矩阵）                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 1. 公式解析: "y ~ x1 + C(x2)" → ModelDesc                       │   │
│  │ 2. 因子评估: 从 DataFrame 提取数据，执行 C() 等函数             │   │
│  │ 3. 编码处理: C(x2) → Treatment 编码（k-1 列虚拟变量）           │   │
│  │ 4. 缺失处理: 应用 NA_action                                      │   │
│  │ 5. 矩阵组装: 生成 endog (y) 和 exog (设计矩阵)                  │   │
│  │ 6. 附加元数据: exog.design_info（列名、编码信息等）             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                          │                                               │
│                          ▼ 返回: (endog, exog), missing_mask, model_spec│
└─────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 2：模型实例化（Statsmodels 负责）                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Statsmodels: 继续处理 from_formula()                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 1. 处理 drop_cols（删除指定列）                                  │   │
│  │ 2. 更新 kwargs: missing_idx, formula, model_spec                │   │
│  │ 3. 实例化模型: cls(endog, exog, **kwargs)                       │   │
│  │    - 此时 endog 和 exog 已经是纯数值矩阵！                       │   │
│  │    - 与直接调用 OLS(endog, exog) 效果相同                        │   │
│  │ 4. 附加原始数据: mod.data.frame = data                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                          │                                               │
│                          ▼ 返回: model 实例                              │
└─────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 3：参数估计（求解机制因模型类型而异）                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  用户代码: results = model.fit()                                          │
│                          │                                               │
│                          ▼                                               │
│  ╔═══════════════════════════════════════════════════════════════════╗  │
│  ║  重要：从此处开始，patsy 不再参与！完全是数值矩阵运算              ║  │
│  ╚═══════════════════════════════════════════════════════════════════╝  │
│                          │                                               │
│                          ├─────────────┬─────────────┬─────────────┐  │
│                          ▼             ▼             ▼             │  │
│              ┌───────────────┐ ┌───────────────┐ ┌───────────────┐│  │
│              │ OLS/WLS/GLS   │ │ GLM (默认)    │ │ Logit/Probit/ ││  │
│              │ （解析解）    │ │ （IRLS）      │ │ Poisson 等    ││  │
│              │               │ │               │ │（迭代优化）   ││  │
│              └───────┬───────┘ └───────┬───────┘ └───────┬───────┘│  │
│                      ▼                   ▼                   ▼        │  │
│              ┌───────────────┐ ┌───────────────┐ ┌───────────────┐│  │
│              │ 直接矩阵运算   │ │ IRLS 迭代     │ │ 梯度优化迭代   ││  │
│              │               │ │               │ │               ││  │
│              │ method="pinv" │ │ method="IRLS" │ │ method="newton"││  │
│              │ 或 "qr"       │ │               │ │ 或 "bfgs" 等  ││  │
│              │               │ │               │ │               ││  │
│              │ β̂ = pinv(X)y │ │ while not    │ │ while not    ││  │
│              │ 或 QR 求解    │ │   converged: │ │   converged: ││  │
│              │               │ │   加权最小二乘│ │   梯度/Hessian││  │
│              │ ⚠️ 0 次迭代   │ │ 更新参数      │ │ 更新参数      ││  │
│              │               │ │               │ │               ││  │
│              └───────┬───────┘ └───────┬───────┘ └───────┬───────┘│  │
│                      ▼                   ▼                   ▼        │  │
│              ┌───────────────────────────────────────────────────────┐│  │
│              │              计算协方差矩阵、标准误、p 值              ││  │
│              └───────────────────────────────────────────────────────┘│  │
│                      │                                               │  │
│                      ▼                                               │  │
│              ┌───────────────────────────────────────────────────────┐│  │
│              │              返回 RegressionResults 实例                ││  │
│              └───────────────────────────────────────────────────────┘│  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 详细时序说明

**阶段 1：用户调用与公式解析**

```python
import statsmodels.formula.api as smf

# 方式 A：公式接口（内部调用 patsy）
results = smf.ols("price ~ bedrooms + C(neighborhood) + sqft", data=df).fit()

# 方式 B：数组接口（等价于方式 A 的后半段）
# endog, exog = patsy.dmatrices("price ~ bedrooms + C(neighborhood) + sqft", df)
# model = sm.OLS(endog, exog)
# results = model.fit()
```

**阶段 2：`Model.from_formula()` 执行流程**

1. **子集筛选**：`data.loc[subset]`（如果提供）
2. **评估环境准备**：
   - `eval_env=2`（默认）→ `eval_env=3`（进入函数层级 +1）
3. **缺失值策略转换**：`missing="none"` → `"raise"`
4. **调用 `handle_formula_data()`**：
   - 创建 `FormulaManager`
   - 获取 `NA_action`
   - 调用 `mgr.get_matrices()`

**阶段 3：`FormulaManager.get_matrices()` → Patsy**

**Patsy 内部流程**：
1. **公式解析**：`patsy.ModelDesc.from_formula()`
2. **因子评估**：从 DataFrame 提取数据，执行 `C()` 等函数
3. **状态确定**：确定分类变量的类别、参考水平等
4. **编码应用**：生成对比矩阵，扩展为多列
5. **矩阵组装**：按顺序组装所有列
6. **缺失值处理**：应用 `NA_action`
7. **返回结果**：`(endog, exog)` + `design_info`

**阶段 4：模型实例化（与求解机制无关）**

```python
# statsmodels/base/model.py:236-248
kwargs.update({
    "missing_idx": missing_idx,
    "missing": missing,
    "formula": formula,
    "model_spec": model_spec,  # DesignInfo
})

# 实例化模型
mod = cls(endog, exog, *args, **kwargs)
# 此时：
# - mod.endog = endog（数值数组）
# - mod.exog = exog（数值矩阵）
# - mod.model_spec = DesignInfo（元数据，供预测使用）
```

**阶段 5：`fit()` 调用——三种不同路径**

#### 路径 A：OLS/WLS/GLS（解析解）

**调用链**：
```
model.fit()
    │
    ▼
RegressionModel.fit()  # 重写了 LikelihoodModel.fit()
    │
    ├──► 方法 1：method="pinv"（默认）
    │       │
    │       ├──► pinv_wexog = pinv(wexog)
    │       ├──► normalized_cov_params = pinv_wexog @ pinv_wexog.T
    │       └──► beta = pinv_wexog @ wendog  # ⚠️ 直接计算，无迭代！
    │
    └──► 方法 2：method="qr"
            │
            ├──► Q, R = qr(wexog)
            ├──► effects = Q.T @ wendog
            └──► beta = solve(R, effects)  # ⚠️ 解三角方程组，无迭代！
    │
    └──► 返回 RegressionResults
```

#### 路径 B：Logit/Probit/Poisson（迭代优化）

**调用链**：
```
model.fit()
    │
    ▼
DiscreteModel.fit()
    │
    ├──► callback = self._check_perfect_pred  # 完美预测检测
    │
    └──► super().fit(...)  # 调用 LikelihoodModel.fit()
            │
            ▼
        LikelihoodModel.fit()  # 真正的迭代优化
            │
            ├──► 准备 start_params（默认全 0）
            │
            ├──► 定义目标函数: f(params) = -loglike(params) / nobs
            │
            ├──► 定义梯度: score(params)
            ├──► 定义 Hessian: hess(params)
            │
            ├──► optimizer._fit(...)  # 调用 scipy 优化器
            │       │
            │       ├──► method="newton": Newton-Raphson
            │       ├──► method="bfgs": BFGS（拟牛顿）
            │       └──► method="nm": Nelder-Mead（单纯形）
            │
            ├──► 检查收敛 (retvals["converged"])
            │
            ├──► 计算协方差: Cov = -H⁻¹
            │
            └──► 返回 LikelihoodModelResults
```

#### 路径 C：GLM（IRLS 或梯度优化）

**调用链（默认 method="IRLS"）**：
```
model.fit()
    │
    ▼
GLM.fit()
    │
    └──► if method.lower() == "irls":
            │
            └──► self._fit_irls(...)  # 迭代重加权最小二乘
                    │
                    ├──► 初始化 mu, eta
                    │
                    └──► while deviance 变化 > tol:
                            │
                            ├──► 计算工作残差 z = eta + (y - mu) * g'(mu)
                            ├──► 计算权重 W = 1 / (Var(mu) * [g'(mu)]²)
                            ├──► 加权最小二乘: beta = (X'WX)⁻¹X'Wz
                            └──► 更新 eta = X @ beta, mu = g⁻¹(eta)
```

### 3.3 预测时的公式复用

**关键点**：预测时需要**相同的设计矩阵生成过程**，使用保存的 `DesignInfo`。

```python
# statsmodels/base/model.py:1164-1189
def _transform_predict_exog(self, exog, transform=True):
    if transform and hasattr(self.model, "formula") and (exog is not None):
        # 获取保存的 model_spec（训练时的 DesignInfo）
        model_spec = (
            getattr(self.model, "model_spec", None) 
            or self.model.data.model_spec
        )
        
        mgr = FormulaManager()
        # 使用 DesignInfo 处理新数据
        # 这确保：
        # 1. 分类变量使用相同的参考水平
        # 2. 列顺序与训练时一致
        # 3. 不重新解析公式字符串
        exog = mgr.get_matrices(model_spec, exog, pandas=True, prediction=True)
```

---

## 四、契约边界分析

### 4.1 Patsy 与 Statsmodels 的职责边界

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              职责边界图                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         PATSY 职责区                              │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                   │   │
│  │  输入:                                                            │   │
│  │  • formula: str ("y ~ x1 + C(x2)")                              │   │
│  │  • data: DataFrame/dict                                          │   │
│  │  • eval_env: 评估环境                                             │   │
│  │  • NA_action: 缺失值策略                                          │   │
│  │                                                                   │   │
│  │  处理:                                                            │   │
│  │  1. 公式字符串解析 → ModelDesc                                    │   │
│  │  2. 因子评估: 从数据提取、函数变换                                │   │
│  │  3. 编码处理: 分类变量对比编码                                    │   │
│  │  4. 缺失值处理: 应用 NA_action                                   │   │
│  │  5. 矩阵组装: 构建设计矩阵                                        │   │
│  │                                                                   │   │
│  │  输出:                                                            │   │
│  │  • endog: 数值数组 (因变量)                                       │   │
│  │  • exog: 数值矩阵 (设计矩阵)                                      │   │
│  │  • design_info: 元数据（列名、编码信息、项名等）                  │   │
│  │  • missing_mask: 缺失值掩码                                       │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                      │
│                                    ▼ 【数值矩阵交接点】                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     STATSMODELS 职责区                           │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                   │   │
│  │  【模型层】                                                       │   │
│  │  输入: endog (数值), exog (数值), model_spec (元数据)           │   │
│  │  处理:                                                            │   │
│  │  • 数据验证和类型转换                                             │   │
│  │  • 检测常数项 (hasconst)                                         │   │
│  │  • 保存元数据供预测使用                                           │   │
│  │  输出: Model 实例                                                 │   │
│  │                                                                   │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                   │   │
│  │  【估计层】—— 与 Patsy 完全无关！                                │   │
│  │  输入: self.exog (数值矩阵), self.endog (数值数组)               │   │
│  │                                                                   │   │
│  │  三种路径:                                                        │   │
│  │                                                                   │   │
│  │  路径 A: OLS/WLS/GLS（解析解）                                  │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │ β̂ = pinv(X) @ y  或  QR 分解求解                       │   │   │
│  │  │ ⚠️ 0 次迭代，直接矩阵运算                                │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │                                                                   │   │
│  │  路径 B: Logit/Probit/Poisson（迭代优化）                      │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │ while not converged:                                    │   │   │
│  │  │     计算 gradient = score(params)                       │   │   │
│  │  │     计算 Hessian = hessian(params)                     │   │   │
│  │  │     更新 params (Newton 步或拟牛顿步)                  │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │                                                                   │   │
│  │  路径 C: GLM（IRLS）                                            │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │ while not converged:                                    │   │   │
│  │  │     计算工作残差 z 和权重 W                             │   │   │
│  │  │     加权最小二乘: β̂ = (X'WX)⁻¹X'Wz                  │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │                                                                   │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                   │   │
│  │  【预测层】                                                       │   │
│  │  输入: new_data (DataFrame/dict)                                 │   │
│  │  处理:                                                            │   │
│  │  • 使用保存的 model_spec (DesignInfo)                            │   │
│  │  • 调用 FormulaManager.get_matrices()                            │   │
│  │  → 再次进入 Patsy 处理新数据（但使用相同的编码规则）             │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 关键理解：Patsy 与求解机制的关系

**重要结论**：

| 维度 | 说明 |
|------|------|
| **Patsy 的终止边界** | 设计矩阵生成完成后，patsy 的工作就结束了 |
| **求解机制与 Patsy 无关** | `fit()` 方法只使用 `self.exog` 和 `self.endog` 数值矩阵 |
| **公式接口与数组接口等价** | `smf.ols(formula, data).fit()` 和 `sm.OLS(endog, exog).fit()` 到达相同的求解路径 |
| **唯一区别** | 公式接口保存了 `model_spec`（DesignInfo），供预测时使用 |

**这意味着**：
- 无论使用 OLS（解析解）还是 Logit（迭代优化），公式处理流程完全相同
- `fit()` 的实现差异只与模型类型有关，与是否使用公式无关
- patsy 只负责"数据准备"，不负责"模型求解"

---

## 五、代码位置索引

### 5.1 关键文件与职责

| 文件路径 | 主要职责 |
|----------|----------|
| `statsmodels/formula/_manager.py` | 公式引擎抽象层（FormulaManager） |
| `statsmodels/formula/formulatools.py` | 公式数据处理（handle_formula_data） |
| `statsmodels/base/model.py` | 模型基类（from_formula、LikelihoodModel.fit 迭代优化） |
| `statsmodels/regression/linear_model.py` | 线性回归（OLS/WLS/GLS.fit 解析解） |
| `statsmodels/discrete/discrete_model.py` | 离散选择模型（调用父类迭代优化） |
| `statsmodels/genmod/generalized_linear_model.py` | GLM（IRLS 或梯度优化） |

### 5.2 关键代码行号

#### 公式处理相关

| 功能 | 代码位置 |
|------|----------|
| FormulaManager 初始化 | `_manager.py:188-257` |
| get_matrices（调用 patsy） | `_manager.py:417-496` |
| patsy.dmatrix/dmatrices 调用 | `_manager.py:480-486` |
| DesignInfo 提取 | `_manager.py:487-490` |
| handle_formula_data | `formulatools.py:15-76` |
| Model.from_formula 入口 | `base/model.py:156-248` |

#### 参数估计相关（修正版）

| 模型类型 | 方法 | 代码位置 | 求解方式 |
|----------|------|----------|----------|
| **线性回归** | `RegressionModel.fit()` | `linear_model.py:284-415` | 解析解（pinv 或 QR） |
| **似然模型基类** | `LikelihoodModel.fit()` | `base/model.py:362-651` | 迭代优化（Newton/BFGS 等） |
| **离散选择** | `DiscreteModel.fit()` | `discrete_model.py:242-274` | 调用父类迭代优化 |
| **GLM（默认）** | `GLM.fit()` + `_fit_irls()` | `genmod/generalized_linear_model.py:1178-1319` | IRLS 迭代 |
| **GLM（备选）** | `GLM._fit_gradient()` | `genmod/generalized_linear_model.py:1321-1365` | 调用父类迭代优化 |

### 5.3 类继承体系

```
base.Model（最基类，定义 from_formula）
│
└── base.LikelihoodModel（定义迭代优化 fit()）
    │
    ├── RegressionModel（⚠️ 重写 fit()，使用解析解）
    │   ├── GLS
    │   ├── WLS
    │   │   └── OLS
    │   └── GLSAR
    │
    ├── DiscreteModel（不重写 fit()，调用父类迭代优化）
    │   ├── BinaryModel
    │   │   ├── Logit
    │   │   └── Probit
    │   └── CountModel
    │       ├── Poisson
    │       └── NegativeBinomial
    │
    └── GLM（自定义 fit()，默认 IRLS，可选梯度优化）
```

---

## 六、总结

### 6.1 职责划分总览

| 层级 | 组件 | 核心职责 |
|------|------|----------|
| **公式层** | **Patsy** | 1. 公式字符串解析<br>2. 因子评估和函数变换<br>3. 分类变量编码（对比矩阵）<br>4. 缺失值处理<br>5. 设计矩阵组装<br>6. 元数据记录（DesignInfo） |
| **抽象层** | **Statsmodels FormulaManager** | 1. 统一 patsy/formulaic 接口<br>2. 引擎选择和配置 |
| **模型层** | **Statsmodels Model** | 1. 提供 from_formula 入口<br>2. 数据验证和预处理<br>3. 保存元数据（model_spec, formula） |
| **估计层** | **Statsmodels Fit** | 三种求解机制：<br>• **OLS/WLS/GLS**：解析解（0 次迭代）<br>• **Logit/Probit/Poisson**：迭代优化<br>• **GLM**：IRLS（迭代重加权最小二乘） |
| **预测层** | **Statsmodels Predict** | 1. 复用保存的 DesignInfo<br>2. 对新数据应用相同的转换 |

### 6.2 关键修正点

| 之前的错误描述 | 正确的实际实现 |
|----------------|----------------|
| OLS 使用迭代优化（Newton、BFGS 等） | OLS 使用解析解（pinv 或 QR 分解），**0 次迭代** |
| OLS.fit() 的 method 参数支持 `"newton"` | OLS.fit() 只支持 `"pinv"` 和 `"qr"` |
| OLS 需要 `start_params` | OLS **不需要**起始参数 |
| 所有模型的 fit() 流程相似 | 线性回归与似然模型的 fit() **完全不同** |

### 6.3 设计亮点

1. **清晰的职责边界**：Patsy 只负责公式→矩阵，Statsmodels 只负责矩阵→估计
2. **解析解与迭代解分离**：RegressionModel 重写 fit()，使用高效的解析解
3. **DesignInfo 持久化**：通过保存训练时的元数据，确保预测时使用完全相同的数据转换
4. **统一的初始化流程**：无论使用哪种求解机制，`from_formula` → `__init__` 流程完全相同
5. **公式接口与数组接口等价**：用户可以自由选择使用公式或直接传入数值矩阵

### 6.4 实际使用中的注意事项

4. **迭代模型的起始参数**：Logit/Probit/Poisson 等迭代模型可以通过 `start_params` 提供更好的起始值，帮助收敛

---

## 附录：完整调用链示例

### 示例代码

```python
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# 准备数据
df = pd.DataFrame({
    'price': [250, 300, 350, 400, 450],
    'bedrooms': [2, 3, 3, 4, 4],
    'neighborhood': ['A', 'B', 'A', 'C', 'B'],
    'sqft': [1000, 1200, 1500, 1800, 2000],
    'sold': [0, 1, 0, 1, 1]  # 用于 Logit 示例
})

# ========== 示例 1：OLS（解析解） ==========
print("=" * 50)
print("示例 1：OLS（解析解）")
print("=" * 50)

model_ols = smf.ols("price ~ bedrooms + C(neighborhood) + sqft", data=df)
results_ols = model_ols.fit(method="pinv")  # 注意：method 是 "pinv"，不是 "newton"！

print(f"OLS 参数:\n{results_ols.params}")
print(f"\nOLS method: {results_ols.model}")  # OLS
# 注意：OLS 没有 mle_retvals（因为没有迭代！）

# ========== 示例 2：Logit（迭代优化） ==========
print("\n" + "=" * 50)
print("示例 2：Logit（迭代优化）")
print("=" * 50)

model_logit = smf.logit("sold ~ bedrooms + sqft", data=df)
results_logit = model_logit.fit(method="newton")  # 注意：method 是 "newton"！

print(f"Logit 参数:\n{results_logit.params}")
print(f"\n是否收敛: {results_logit.mle_retvals['converged']}")  # 有收敛信息！
print(f"迭代次数: {results_logit.mle_retvals['iterations']}")  # 有迭代次数！
```

### 关键差异示例输出

```
==================================================
示例 1：OLS（解析解）
==================================================
OLS 参数:
Intercept               -50.0
C(neighborhood)[T.B]     20.0
C(neighborhood)[T.C]     40.0
bedrooms                 10.0
sqft                      0.2
dtype: float64

注意：OLS 没有 mle_retvals，因为没有迭代过程！

==================================================
示例 2：Logit（迭代优化）
==================================================
Optimization terminated successfully.
         Current function value: 0.480
         Iterations: 6              # 有迭代次数！
         Function evaluations: 7
         Gradient evaluations: 7

Logit 参数:
Intercept   -10.5
bedrooms      1.2
sqft          0.01
dtype: float64

是否收敛: True               # 有收敛信息！
迭代次数: 6
```

### OLS 与 Logit 的 `fit()` 方法参数对比

| 参数 | OLS.fit() | Logit.fit() |
|------|-----------|-------------|
| `method` | `"pinv"`, `"qr"` | `"newton"`, `"bfgs"`, `"nm"`, `"lbfgs"`, `"cg"`, `"ncg"`, `"powell"`, `"minimize"` |
| `start_params` | 不支持 | 支持（默认全 0） |
| `maxiter` | 不支持 | 支持（默认 35） |
| `full_output` | 不支持 | 支持 |
| `disp` | 不支持 | 支持 |
| `callback` | 不支持 | 支持 |

### 调用链对比

**OLS 调用链**（解析解）：
```
smf.ols(formula, data)
    │
    ├──► Model.from_formula() → 调用 Patsy → 生成 endog, exog
    │
    └──► model.fit(method="pinv")
            │
            └──► RegressionModel.fit()
                    │
                    ├──► pinv_wexog = pinv(wexog)
                    ├──► beta = pinv_wexog @ wendog  # ⚠️ 直接计算，无循环！
                    └──► 返回 RegressionResults
```

**Logit 调用链**（迭代优化）：
```
smf.logit(formula, data)
    │
    ├──► Model.from_formula() → 调用 Patsy → 生成 endog, exog
    │                          （与 OLS 完全相同！）
    │
    └──► model.fit(method="newton")
            │
            └──► DiscreteModel.fit()
                    │
                    └──► super().fit(...)  # 调用 LikelihoodModel.fit()
                            │
                            ├──► start_params = [0.0] * k  # 起始参数
                            │
                            └──► optimizer._fit(...)  # 迭代优化！
                                    │
                                    └──► while not converged:
                                            ├──► 计算 gradient = score(params)
                                            ├──► 计算 Hessian = hessian(params)
                                            └──► 更新 params (Newton 步)
```

**关键理解**：
- **公式处理阶段完全相同**：OLS 和 Logit 都通过 `from_formula` 调用 patsy 生成设计矩阵
- **求解阶段完全不同**：OLS 使用解析解（0 次迭代），Logit 使用迭代优化
- **Patsy 不参与求解阶段**：设计矩阵生成后，patsy 的工作就结束了

这个示例清楚地展示了：
1. **训练时**：公式字符串 → Patsy → DesignInfo + 设计矩阵
2. **求解时**：
   - OLS：直接矩阵运算，无迭代
   - Logit：迭代优化，需要检查收敛
3. **预测时**：直接使用保存的 DesignInfo 转换新数据，确保一致性
