# Statsmodels 与 Patsy 协作关系分析报告

## 概述

在 statsmodels 中使用公式字符串（如 `"y ~ x1 + x2 + C(category)"`）进行统计建模时，patsy 库承担了从公式解析到设计矩阵生成的核心工作，而 statsmodels 则负责统计估计和结果分析。本文档详细分析两个库的职责划分、协作流程以及契约边界。

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

**关键代码引用**：
```python
# statsmodels/formula/_manager.py:937-969
def get_contrast_matrix(self, term, factor, model_spec):
    if self._using_patsy:
        return model_spec.term_codings[term][0].contrast_matrices[factor].matrix
    else:
        # formulaic 的处理逻辑
        cat = self.get_factor_categories(factor, model_spec)
        reduced_rank = True
        # ...
        return np.asarray(
            model_spec.factor_contrasts[factor].get_coding_matrix(
                reduced_rank=reduced_rank
            )
        )
```

### 1.4 缺失值处理

**核心功能**：在构建设计矩阵时处理缺失值。

**关键代码引用**：
```python
# statsmodels/formula/_manager.py:30-38
class NAAction(patsy.missing.NAAction):
    # monkey-patch so we can handle missing values in 'extra' arrays later
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
- `NA_types`：定义哪些值被视为缺失（默认 `None`, `NaN`）

### 1.5 评估环境管理

**核心功能**：处理公式中引用的外部变量和函数。

**关键代码引用**：
```python
# statsmodels/formula/_manager.py:466-472
if isinstance(eval_env, Mapping):
    _eval_env = patsy.eval.EvalEnvironment(
        [{key: val} for key, val in eval_env.items()]
    )
else:
    _eval_env = eval_env
    if isinstance(eval_env, patsy.eval.EvalEnvironment):
        warnings.warn(EVAL_ENV_WARNING, FutureWarning, stacklevel=2)
```

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
- `column_names`：设计矩阵的列名
- `term_names`：公式中的项名
- `term_name_slices`：每个项对应的列切片
- `factor_infos`：每个因子的信息（类型、类别等）
- `term_codings`：每个项的编码信息
- `linear_constraint()`：解析线性约束字符串

---

## 二、Statsmodels 的职责

Statsmodels 作为统计建模框架，承担以下职责：

### 2.1 公式引擎抽象层

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

**引擎选择逻辑**：
```python
# statsmodels/formula/_manager.py:17-27
DEFAULT_FORMULA_ENGINE = os.environ.get("SM_FORMULA_ENGINE", None)
if DEFAULT_FORMULA_ENGINE not in ("formulaic", "patsy", None):
    raise ValueError(f"Invalid value for SM_FORMULA_ENGINE: {DEFAULT_FORMULA_ENGINE}")

ensure_patsy_compat()

try:
    import patsy
    import patsy.missing
    DEFAULT_FORMULA_ENGINE = DEFAULT_FORMULA_ENGINE or "patsy"
    # ...
    HAVE_PATSY = True
except ImportError:
    DEFAULT_FORMULA_ENGINE = DEFAULT_FORMULA_ENGINE or "formulaic"
```

### 2.2 数据与公式的桥梁

**核心功能**：`handle_formula_data` 函数连接公式和数据。

**关键代码引用**：
```python
# statsmodels/formula/formulatools.py:15-76
def handle_formula_data(Y, X, formula, depth=0, missing="drop"):
    """
    Returns endog, exog, and the model specification from arrays and formula.
    """
    na_action = FormulaManager().get_na_action(action=missing)
    mgr = FormulaManager()
    if X is not None:
        result = mgr.get_matrices(
            formula, (Y, X), eval_env=depth, pandas=True, na_action=na_action
        )
    else:
        # ...
        result = mgr.get_matrices(
            formula, Y, eval_env=depth, pandas=True, na_action=na_action
        )
    
    missing_mask = mgr.missing_mask
    # ...
    model_spec = mgr.spec
    return result, missing_mask, model_spec
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
    
    tmp = handle_formula_data(data, None, formula, depth=eval_env, missing=missing)
    ((endog, exog), missing_idx, model_spec) = tmp
    
    # ... 验证和处理 drop_cols
    
    kwargs.update({
        "missing_idx": missing_idx,
        "missing": missing,
        "formula": formula,  # attach formula for unpckling
        "model_spec": model_spec,
    })
    
    mod = cls(endog, exog, *args, **kwargs)
    mod.formula = formula
    mod.data.frame = data
    return mod
```

### 2.4 统计估计与推断

**核心功能**：使用 patsy 生成的设计矩阵进行统计建模。

**关键代码引用（OLS 拟合流程）**：
```python
# 模型初始化 - statsmodels/base/model.py:100-116
def __init__(self, endog, exog=None, **kwargs):
    missing = kwargs.pop("missing", "none")
    hasconst = kwargs.pop("hasconst", None)
    self.data = self._handle_data(endog, exog, missing, hasconst, **kwargs)
    self.k_constant = self.data.k_constant
    self.exog = self.data.exog
    self.endog = self.data.endog
    # ...
```

**似然模型拟合**：
```python
# statsmodels/base/model.py:362-651
def fit(self, start_params=None, method="newton", maxiter=100, ...):
    # 1. 准备起始参数
    if start_params is None:
        if hasattr(self, "start_params"):
            start_params = self.start_params
        elif self.exog is not None:
            start_params = [0.0] * self.exog.shape[1]
    
    # 2. 定义目标函数（负对数似然）
    def f(params, *args):
        return -self.loglike(params, *args) / nobs
    
    # 3. 选择优化方法
    optimizer = Optimizer()
    xopt, retvals, optim_settings = optimizer._fit(
        f, score, start_params, fargs, kwargs,
        hessian=hess, method=method, ...
    )
    
    # 4. 计算协方差矩阵
    if not skip_hessian:
        H = -1 * self.hessian(xopt)
        # 检查正定性并求逆
        # ...
        Hinv = eigvecs.dot(np.diag(1.0 / eigvals)).dot(eigvecs.T)
    
    # 5. 包装结果
    mlefit = LikelihoodModelResults(self, xopt, Hinv, scale=1.0, **kwds)
    return mlefit
```

### 2.5 预测时的公式复用

**核心功能**：在预测阶段重新应用相同的公式转换。

**关键代码引用**：
```python
# statsmodels/base/model.py:1149-1207
def _transform_predict_exog(self, exog, transform=True):
    # ...
    if transform and hasattr(self.model, "formula") and (exog is not None):
        model_spec = (
            getattr(self.model, "model_spec", None) or self.model.data.model_spec
        )
        mgr = FormulaManager()
        # ...
        try:
            exog = mgr.get_matrices(model_spec, exog, pandas=True, prediction=True)
        except Exception as exc:
            # 错误处理
            raise exc.__class__(msg) from exc
        # ...
    return exog, exog_index
```

---

## 三、完整协作流程

### 3.1 流程图概览

```
用户代码: sm.OLS.from_formula("y ~ x1 + C(x2)", data=df)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Statsmodels: Model.from_formula()                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 处理 subset 参数（数据子集筛选）                  │   │
│  │ 2. 准备 eval_env（评估环境）                         │   │
│  │ 3. 调用 handle_formula_data()                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
┌─────────────────────────────────────────────────────────────┘
│  Statsmodels: handle_formula_data()
│  ┌─────────────────────────────────────────────────────┐
│  │ 1. 创建 FormulaManager 实例                         │
│  │ 2. 获取 NA_action（缺失值处理策略）                 │
│  │ 3. 调用 mgr.get_matrices()                         │
│  └─────────────────────────────────────────────────────┘
│                          │
│                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Statsmodels: FormulaManager.get_matrices()                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 如果使用 patsy 引擎:                                 │   │
│  │   - 调用 patsy.dmatrices() 或 patsy.dmatrix()      │   │
│  │   - 获取 output[1].design_info 作为 model_spec     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Patsy: 核心工作                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 公式解析: ModelDesc.from_formula()               │   │
│  │    - 解析 "y ~ x1 + C(x2)"                          │   │
│  │    - 识别 LHS (y) 和 RHS (x1, C(x2))               │   │
│  │                                                        │   │
│  │ 2. 因子评估: EvalFactor.eval()                       │   │
│  │    - 从 DataFrame 中提取数据                         │   │
│  │    - 执行函数变换（如 log(x)）                       │   │
│  │                                                        │   │
│  │ 3. 分类编码: ContrastMatrix                          │   │
│  │    - 对 C(x2) 应用 Treatment 编码                    │   │
│  │    - 生成 k-1 个虚拟变量列                           │   │
│  │                                                        │   │
│  │ 4. 缺失值处理: NAAction._handle_NA_drop()           │   │
│  │    - 识别包含 NA 的行                                 │   │
│  │    - 删除或抛出异常                                   │   │
│  │                                                        │   │
│  │ 5. 构建设计矩阵: DesignMatrixBuilder                 │   │
│  │    - 组装所有列                                       │   │
│  │    - 附加 DesignInfo 元数据                          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼ (返回: (endog, exog), missing_mask, model_spec)
┌─────────────────────────────────────────────────────────────┐
│  Statsmodels: 继续处理                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 处理 drop_cols（删除指定列）                     │   │
│  │ 2. 更新 kwargs: missing_idx, formula, model_spec   │   │
│  │ 3. 实例化模型: cls(endog, exog, **kwargs)          │   │
│  │ 4. 附加原始数据: mod.data.frame = data              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼ (返回: model 实例)
┌─────────────────────────────────────────────────────────────┐
│  用户代码: results = model.fit()                             │
│                          │                                   │
│                          ▼                                   │
│  Statsmodels: 模型拟合和推断                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 优化算法寻找参数估计值                            │   │
│  │ 2. 计算标准误、p值、置信区间                         │   │
│  │ 3. 生成 Results 实例                                 │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 详细时序说明

**阶段 1：用户调用**
```python
import statsmodels.formula.api as smf
results = smf.ols("price ~ bedrooms + C(neighborhood) + sqft", data=df).fit()
```

**阶段 2：Statsmodels 入口处理**

`statsmodels/base/model.py:156-248` 的 `from_formula` 方法执行：
1. **子集筛选**：如果提供了 `subset` 参数，使用 `data.loc[subset]` 筛选数据
2. **评估环境准备**：
   - 默认 `eval_env=2`（向上追溯 2 层栈帧）
   - `eval_env=-1` 表示使用空环境
   - 每次进入新函数层级，`eval_env += 1`
3. **缺失值策略转换**：`missing="none"` 转换为 `"raise"`（因为 patsy 不支持 none）

**阶段 3：调用公式处理函数**

`statsmodels/formula/formulatools.py:15-76` 的 `handle_formula_data` 函数：
1. 创建 `FormulaManager` 实例
2. 获取 `NA_action`（根据 `missing` 参数）
3. 调用 `mgr.get_matrices()`

**阶段 4：FormulaManager 分发到 Patsy**

`statsmodels/formula/_manager.py:417-496` 的 `get_matrices` 方法：

**判断使用哪个公式函数**：
```python
if (
    isinstance(formula, (patsy.design_info.DesignInfo, patsy.desc.ModelDesc))
    or "~" not in formula
    or formula.strip().startswith("~")
):
    # 单侧公式：仅 dmatrix
    output = patsy.dmatrix(formula, data, ...)
else:
    # 双侧公式：dmatrices 返回 (endog, exog)
    output = patsy.dmatrices(formula, data, ...)
```

**参数传递给 patsy**：
- `eval_env`：评估环境（调整后的栈帧深度或字典）
- `return_type`：`"dataframe"` 或 `"matrix"`
- `NA_action`：缺失值处理策略

**阶段 5：Patsy 内部处理流程**

1. **公式解析** (`patsy/desc.py:ModelDesc.from_formula`)
   - 使用 `patsy.parse_formula()` 解析字符串
   - 构建 `ModelDesc` 对象，包含 `lhs_termlist` 和 `rhs_termlist`
   - 每个 `Term` 包含多个 `EvalFactor`

2. **因子评估** (`patsy/eval.py:EvalFactor`)
   - 从 DataFrame 或评估环境中提取数据
   - 执行公式中的函数调用（如 `log(x)`、`C(x)`）
   - `C()` 是特殊函数，标记分类变量并指定编码方式

3. **状态转换** (`patsy/build.py:DesignMatrixBuilder`)
   - 为每个因子确定"状态"（如分类变量的类别、中心化参数等）
   - 这些状态存储在 `DesignInfo` 中，供预测时复用

4. **编码应用** (`patsy/contrasts.py`)
   - 对于分类因子，应用对比矩阵
   - 默认 `Treatment` 编码：k 个类别 → k-1 列
   - 截距列默认添加（除非公式中有 `-1` 或 `0 +`）

5. **矩阵组装** (`patsy/build.py`)
   - 按公式顺序组装所有列
   - 处理缺失值（根据 `NA_action`）
   - 返回 `DesignMatrix` 对象（带有 `design_info` 属性）

**阶段 6：返回结果给 Statsmodels**

`get_matrices` 方法的返回处理：
```python
if isinstance(output, tuple):
    # dmatrices 返回 (lhs, rhs)
    self._spec = output[1].design_info  # 保存 RHS 的 DesignInfo
else:
    # dmatrix 返回单个矩阵
    self._spec = output.design_info

# 记录缺失值掩码
if isinstance(na_action, NAAction):
    self._missing_mask = getattr(na_action, "missing_mask", None)

return output
```

**阶段 7：模型实例化**

`from_formula` 方法的最后步骤：
```python
# 附加公式相关信息到 kwargs
kwargs.update({
    "missing_idx": missing_idx,
    "missing": missing,
    "formula": formula,      # 用于反序列化
    "model_spec": model_spec,  # DesignInfo，用于预测
})

# 实例化具体模型类（如 OLS）
mod = cls(endog, exog, *args, **kwargs)

# 附加额外信息
mod.formula = formula
mod.data.frame = data  # 保存原始数据引用

return mod
```

**阶段 8：用户调用 fit()**

用户执行 `model.fit()` 后，statsmodels 进行：
1. **数据准备**：`endog` 和 `exog` 已经是数值矩阵
2. **参数估计**：使用优化算法最小化损失函数（OLS 使用最小二乘，MLE 使用极大似然）
3. **推断统计**：计算标准误、t 值、p 值、置信区间
4. **结果包装**：返回 `RegressionResults` 实例

**阶段 9：预测时的公式复用**

当用户调用 `results.predict(new_data)` 时：
```python
# statsmodels/base/model.py:1149-1207
def _transform_predict_exog(self, exog, transform=True):
    if transform and hasattr(self.model, "formula") and (exog is not None):
        # 获取保存的 model_spec (DesignInfo)
        model_spec = getattr(self.model, "model_spec", None)
        
        mgr = FormulaManager()
        # 使用相同的公式规格处理新数据
        exog = mgr.get_matrices(model_spec, exog, pandas=True, prediction=True)
        # ...
```

**关键点**：预测时使用的是保存的 `model_spec`（即 `DesignInfo`），而不是重新解析公式字符串。这确保了：
- 分类变量使用相同的编码（相同的参考水平）
- 连续变量使用相同的变换参数（如相同的中心化均值）
- 列顺序与训练时一致

---

## 四、契约边界分析

### 4.1 输入契约

**Statsmodels → Patsy 的输入参数**：

| 参数 | 类型 | 说明 | 代码位置 |
|------|------|------|----------|
| `formula` | `str` 或 `patsy.ModelDesc` 或 `patsy.DesignInfo` | 公式字符串或预解析的公式对象 | `_manager.py:474-486` |
| `data` | `DataFrame` 或 `dict` 或 `(Y, X)` 元组 | 包含变量的数据 | `_manager.py:446-465`, `formulatools.py:45-66` |
| `eval_env` | `int` 或 `dict` 或 `patsy.EvalEnvironment` | 公式中函数/变量的评估环境 | `_manager.py:458-472` |
| `return_type` | `Literal["dataframe", "matrix"]` | 返回矩阵的类型 | `_manager.py:461` |
| `NA_action` | `patsy.missing.NAAction` | 缺失值处理策略 | `_manager.py:463-464` |

**关键代码**：
```python
# statsmodels/formula/_manager.py:460-486
if self._using_patsy:
    return_type = "dataframe" if pandas else "matrix"
    kwargs = {}
    if na_action:
        kwargs["NA_action"] = na_action
    if isinstance(eval_env, Mapping):
        _eval_env = patsy.eval.EvalEnvironment(
            [{key: val} for key, val in eval_env.items()]
        )
    else:
        _eval_env = eval_env
    # ...
    output = patsy.dmatrices(formula, data, eval_env=_eval_env, 
                              return_type=return_type, **kwargs)
```

### 4.2 输出契约

**Patsy → Statsmodels 的输出**：

#### 4.2.1 主要返回值

**情况 1：双侧公式（`dmatrices`）**
```python
# 返回值: (endog, exog) 元组
# endog: DesignMatrix 或 DataFrame - 因变量矩阵
# exog: DesignMatrix 或 DataFrame - 自变量设计矩阵
```

**情况 2：单侧公式（`dmatrix`）**
```python
# 返回值: exog（单个 DesignMatrix 或 DataFrame）
```

#### 4.2.2 元数据：DesignInfo

**这是最重要的契约对象**，存储在 `exog.design_info` 属性中。

**关键属性**（`statsmodels/formula/_manager.py` 中的使用方式）：

| 属性 | 类型 | 用途 | 使用位置 |
|------|------|------|----------|
| `.column_names` | `list[str]` | 设计矩阵的列名（包括编码后的虚拟变量） | `_manager.py:812` |
| `.term_names` | `list[str]` | 公式中的原始项名（如 `"C(x2)"`） | `_manager.py:794` |
| `.term_name_slices` | `dict[str, slice]` | 每个项对应的列切片 | `_manager.py:831` |
| `.terms` | `list[Term]` | 项对象列表 | `_manager.py:706, 724` |
| `.factor_infos` | `dict` | 每个因子的元信息（类型、类别等） | `_manager.py:933` |
| `.term_codings` | `dict` | 每个项的编码信息（对比矩阵等） | `_manager.py:956` |

**关键方法**：

| 方法 | 功能 | 使用位置 |
|------|------|----------|
| `.slice(term)` | 获取某个项对应的列切片 | `_manager.py:875` |
| `.linear_constraint(constraints)` | 解析线性约束字符串 | `_manager.py:618-626` |
| `.describe()` | 返回公式的可读描述 | `_manager.py:912` |
| `.subset(cols)` | 创建列子集的新 DesignInfo | `base/model.py:234` |

**DesignInfo 在预测中的关键作用**：
```python
# statsmodels/base/model.py:1177-1189
def _transform_predict_exog(self, exog, transform=True):
    if transform and hasattr(self.model, "formula"):
        # 使用保存的 model_spec（即训练时的 DesignInfo）
        model_spec = getattr(self.model, "model_spec", None)
        mgr = FormulaManager()
        # 直接使用 model_spec 处理新数据，而不是重新解析公式
        exog = mgr.get_matrices(model_spec, exog, pandas=True, prediction=True)
```

#### 4.2.3 缺失值信息

```python
# statsmodels/formula/_manager.py:37
self.missing_mask = total_mask  # 布尔数组，True 表示该行被删除

# statsmodels/formula/formulatools.py:68-70
missing_mask = mgr.missing_mask
if not np.any(missing_mask):
    missing_mask = None
```

### 4.3 数据流契约

```
┌────────────────────────────────────────────────────────────────┐
│                        数据流方向                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Statsmodels 用户层                                            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  smf.ols("y ~ x1 + C(x2)", data=df)                    │  │
│  │         │                                                │  │
│  │         ▼                                                │  │
│  │  Model.from_formula()                                   │  │
│  │  - 准备 eval_env (int 或 dict)                         │  │
│  │  - 准备 missing 策略                                     │  │
│  │  - 准备 subset 数据筛选                                  │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                     │
│                          ▼ (公式字符串, DataFrame, eval_env)  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Statsmodels 抽象层 (FormulaManager)                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  - 选择引擎: patsy 或 formulaic                         │  │
│  │  - 统一参数格式                                          │  │
│  │  - 包装返回值                                            │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                     │
│                          ▼                                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Patsy 层                                                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  输入:                                                    │  │
│  │  - formula: str ("y ~ x1 + C(x2)")                      │  │
│  │  - data: DataFrame 或 dict                               │  │
│  │  - eval_env: int 或 EvalEnvironment                      │  │
│  │  - NA_action: NAAction 实例                              │  │
│  │                                                           │  │
│  │  处理:                                                    │  │
│  │  1. 公式解析 → ModelDesc                                 │  │
│  │  2. 因子评估 → 提取/变换数据                            │  │
│  │  3. 编码处理 → 分类变量对比编码                         │  │
│  │  4. 缺失处理 → 应用 NA_action                           │  │
│  │  5. 矩阵组装 → 构建设计矩阵                             │  │
│  │                                                           │  │
│  │  输出:                                                    │  │
│  │  - (endog, exog): (DesignMatrix, DesignMatrix)          │  │
│  │    或单个 DesignMatrix (单侧公式)                        │  │
│  │  - exog.design_info: DesignInfo (元数据)                │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                     │
│                          ▼ (设计矩阵 + DesignInfo)            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Statsmodels 模型层                                            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  输入:                                                    │  │
│  │  - endog: 数值数组 (因变量)                              │  │
│  │  - exog: 数值数组 (设计矩阵)                             │  │
│  │  - model_spec: DesignInfo (元数据)                      │  │
│  │  - missing_idx: 缺失值掩码                               │  │
│  │                                                           │  │
│  │  处理:                                                    │  │
│  │  - 数据验证和类型转换                                    │  │
│  │  - 检测常数项                                            │  │
│  │  - 保存元数据供后续使用                                  │  │
│  │                                                           │  │
│  │  输出:                                                    │  │
│  │  - Model 实例 (可调用 fit())                             │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                     │
│                          ▼                                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Statsmodels 估计层                                            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  model.fit()                                             │  │
│  │  - 使用数值矩阵进行估计                                  │  │
│  │  - 不再需要公式信息                                      │  │
│  │  - 但保留 model_spec 供预测使用                         │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 4.4 关键契约接口总结

#### 4.4.1 公式接口

**Patsy 提供的核心函数**：
```python
# 双侧公式：因变量 ~ 自变量
patsy.dmatrices(formula, data, eval_env=0, return_type="matrix", NA_action=None)
# 返回: (lhs_design_matrix, rhs_design_matrix)

# 单侧公式：仅自变量（用于预测或仅生成 X）
patsy.dmatrix(formula_like, data, eval_env=0, return_type="matrix")
# 返回: design_matrix

# 解析公式但不构建矩阵
patsy.ModelDesc.from_formula(formula, eval_env=0)
# 返回: ModelDesc 对象
```

**Statsmodels 对这些函数的包装**：
```python
# statsmodels/formula/_manager.py:417-496
def get_matrices(self, formula, data, eval_env=0, pandas=True, 
                  na_action=None, prediction=False):
    if self._using_patsy:
        return_type = "dataframe" if pandas else "matrix"
        # ...
        if "~" not in formula or formula.strip().startswith("~"):
            output = patsy.dmatrix(formula, data, eval_env=_eval_env, 
                                    return_type=return_type, **kwargs)
        else:
            output = patsy.dmatrices(formula, data, eval_env=_eval_env, 
                                      return_type=return_type, **kwargs)
        # 提取 DesignInfo
        if isinstance(output, tuple):
            self._spec = output[1].design_info
        else:
            self._spec = output.design_info
        return output
```

#### 4.4.2 设计信息接口

**DesignInfo 的核心作用是"记住"训练时的数据处理方式**，确保预测时使用完全相同的转换。

**Statsmodels 中保存 DesignInfo 的位置**：
```python
# 位置 1: Model 实例
# statsmodels/base/model.py:236-242
kwargs.update({
    "missing_idx": missing_idx,
    "missing": missing,
    "formula": formula,
    "model_spec": model_spec,  # 这里
})
mod = cls(endog, exog, *args, **kwargs)

# 位置 2: Model.data 属性
# statsmodels/base/data.py 中也会保存

# 位置 3: Results 实例
# 通过 model 属性访问
results.model.model_spec
```

**使用 DesignInfo 进行预测**：
```python
# statsmodels/formula/_manager.py:417-496 中的 prediction 分支
if prediction:
    if hasattr(_formula, "rhs"):
        _formula = _formula.rhs  # 只使用公式右侧

# 可以直接传入 DesignInfo 给 get_matrices
# 这会复用训练时的所有编码参数
```

#### 4.4.3 线性约束接口

**Patsy 提供线性约束字符串解析**：
```python
# statsmodels/formula/_manager.py:618-626
if self._using_patsy:
    from patsy.design_info import DesignInfo
    # 使用 DesignInfo 解析约束字符串
    lc = DesignInfo(variable_names).linear_constraint(constraints)
    return LinearConstraintValues(
        constraint_matrix=lc.coefs,
        constraint_values=lc.constants,
        variable_names=lc.variable_names,
    )
```

**使用示例**：
```python
# 用户可以这样写
model.fit_constrained("x1 = x2")
# patsy 会解析为对应的约束矩阵
```

---

## 五、代码位置索引

### 5.1 Statsmodels 关键文件

| 文件路径 | 职责 |
|----------|------|
| `statsmodels/formula/_manager.py` | 公式引擎抽象层（FormulaManager） |
| `statsmodels/formula/formulatools.py` | 公式数据处理（handle_formula_data） |
| `statsmodels/formula/api.py` | 公式 API 入口（导出 from_formula 快捷方式） |
| `statsmodels/formula/__init__.py` | 公式模块初始化（options 配置） |
| `statsmodels/base/model.py` | 模型基类（from_formula 方法） |
| `statsmodels/compat/patsy.py` | Patsy 兼容性补丁 |

### 5.2 关键代码行号

**FormulaManager 核心方法**：
- `__init__`: `_manager.py:206-257` - 初始化和引擎选择
- `get_matrices`: `_manager.py:417-496` - 调用 patsy 生成设计矩阵
- `get_linear_constraints`: `_manager.py:598-655` - 线性约束解析
- `get_model_spec`: `_manager.py:835-857` - 从 DataFrame 获取 DesignInfo
- `get_na_action`: `_manager.py:726-748` - 获取缺失值处理策略

**Model.from_formula 流程**：
- 入口: `base/model.py:156`
- 子集处理: `base/model.py:198-199`
- eval_env 处理: `base/model.py:200-206`
- 缺失值处理: `base/model.py:207-209`
- 调用 handle_formula_data: `base/model.py:211`
- 验证 endog: `base/model.py:213-220`
- 处理 drop_cols: `base/model.py:221-234`
- 附加元数据: `base/model.py:236-243`
- 实例化模型: `base/model.py:244`
- 附加额外信息: `base/model.py:245-247`

**预测时的公式处理**：
- `_transform_predict_exog`: `base/model.py:1149-1207`
- 使用保存的 model_spec: `base/model.py:1161-1163`
- 调用 get_matrices: `base/model.py:1181`

### 5.3 Patsy 集成点

| 集成点 | 代码位置 | 说明 |
|--------|----------|------|
| Patsy 导入 | `_manager.py:24-28` | 尝试导入 patsy，设置 HAVE_PATSY |
| NAAction 继承 | `_manager.py:30-38` | 继承并 monkey-patch patsy 的 NAAction |
| dmatrix 调用 | `_manager.py:480-482` | 单侧公式使用 dmatrix |
| dmatrices 调用 | `_manager.py:484-486` | 双侧公式使用 dmatrices |
| DesignInfo 提取 | `_manager.py:487-490` | 从返回值提取 design_info |
| ModelDesc.from_formula | `_manager.py:765` | 解析公式字符串 |
| DesignInfo.linear_constraint | `_manager.py:621` | 解析线性约束 |
| 兼容性补丁 | `compat/patsy.py:15-22` | 修复 pandas CategoricalDtype 检测 |

---

## 六、总结

### 6.1 职责划分总览

| 层级 | 组件 | 核心职责 |
|------|------|----------|
| **公式层** | **Patsy** | 1. 公式字符串解析<br>2. 因子评估和函数变换<br>3. 分类变量编码（对比矩阵）<br>4. 缺失值处理<br>5. 设计矩阵组装<br>6. 元数据记录（DesignInfo） |
| **抽象层** | **Statsmodels FormulaManager** | 1. 统一 patsy/formulaic 接口<br>2. 引擎选择和配置<br>3. 参数格式转换<br>4. 结果统一包装 |
| **模型层** | **Statsmodels Model** | 1. 提供 from_formula 入口<br>2. 数据验证和预处理<br>3. 保存元数据（model_spec, formula）<br>4. 实例化模型对象 |
| **估计层** | **Statsmodels Fit** | 1. 使用数值矩阵进行参数估计<br>2. 统计推断（标准误、p值等）<br>3. 结果包装 |
| **预测层** | **Statsmodels Predict** | 1. 复用保存的 DesignInfo<br>2. 对新数据应用相同的转换<br>3. 确保一致性 |

### 6.2 关键契约边界

**输入边界（Statsmodels → Patsy）**：
- **公式**：字符串或预解析的公式对象
- **数据**：DataFrame、字典或元组
- **评估环境**：整数（栈帧深度）或字典
- **缺失值策略**：NAAction 实例

**输出边界（Patsy → Statsmodels）**：
- **设计矩阵**：DesignMatrix 或 DataFrame（数值类型）
- **元数据**：DesignInfo（最重要的契约对象）
  - 列名映射
  - 项名映射
  - 编码信息
  - 对比矩阵
- **缺失值信息**：missing_mask 布尔数组

**持久化边界**：
- `formula` 字符串：用于人类可读和调试
- `model_spec` (DesignInfo)：用于预测和约束
- `data.frame`：原始数据引用

### 6.3 设计亮点

1. **引擎抽象**：FormulaManager 允许在 patsy 和 formulaic 之间切换，而用户代码无需修改
2. **DesignInfo 持久化**：通过保存训练时的元数据，确保预测时使用完全相同的数据转换
3. **分层设计**：公式处理与统计估计完全解耦，模型的 fit() 方法只处理数值矩阵
4. **缺失值追踪**：通过 monkey-patch 的 NAAction，能够追踪哪些行因缺失值被删除
5. **灵活的评估环境**：支持栈帧深度和字典两种评估环境，适应不同使用场景

### 6.4 实际使用中的注意事项

1. **预测时必须使用 DataFrame**：如果模型是通过公式创建的，预测时传入的新数据也必须是 DataFrame（或字典），且列名必须与训练时一致

2. **分类变量的一致性**：预测时分类变量可以包含训练时未见过的类别，但这些类别会被编码为全 0 列（取决于编码方式）

3. **公式中的函数**：公式中使用的函数（如 `log()`、`C()`）在预测时也需要可用（在评估环境中）

4. **缺失值处理**：训练时的 `missing="drop"` 策略会在预测时同样应用

5. **截距处理**：公式中的 `-1` 或 `0 +` 会被正确反映在设计矩阵和后续的统计检验中

---

## 附录：完整调用链示例

### 示例代码
```python
import pandas as pd
import statsmodels.formula.api as smf

# 准备数据
df = pd.DataFrame({
    'price': [250, 300, 350, 400, 450],
    'bedrooms': [2, 3, 3, 4, 4],
    'neighborhood': ['A', 'B', 'A', 'C', 'B'],
    'sqft': [1000, 1200, 1500, 1800, 2000]
})

# 使用公式建模
model = smf.ols("price ~ bedrooms + C(neighborhood) + sqft", data=df)
results = model.fit()

# 预测
new_data = pd.DataFrame({
    'bedrooms': [3],
    'neighborhood': ['B'],
    'sqft': [1300]
})
prediction = results.predict(new_data)
```

### 调用链追踪

```
1. smf.ols(...)
   │
   ├──► statsmodels/formula/api.py:14
   │    ols = lm_.OLS.from_formula
   │
   └──► statsmodels/base/model.py:156 (Model.from_formula)
        │
        ├──► 处理 subset (None)
        ├──► eval_env = 2 → 3
        ├──► missing = "drop" → "raise"（转换）
        │
        └──► statsmodels/formula/formulatools.py:15 (handle_formula_data)
             │
             ├──► FormulaManager() 初始化
             ├──► get_na_action(action="drop")
             │
             └──► statsmodels/formula/_manager.py:417 (get_matrices)
                  │
                  ├──► 判断: "~" 在公式中 → 使用 dmatrices
                  │
                  └──► patsy.dmatrices(
                       formula="price ~ bedrooms + C(neighborhood) + sqft",
                       data=df,
                       eval_env=3,
                       return_type="dataframe",
                       NA_action=NAAction(...)
                   )
                  │
                  ├──► Patsy 内部处理（解析→评估→编码→组装）
                  │
                  └──► 返回 (endog, exog)，两者都是 DataFrame
                       ├──► endog: 单列 'price'
                       └──► exog: 包含以下列的 DataFrame:
                            - Intercept (截距)
                            - C(neighborhood)[T.B]
                            - C(neighborhood)[T.C]
                            - bedrooms
                            - sqft
                       └──► exog.design_info: DesignInfo 实例
                            ├──► .column_names: ['Intercept', 'C(neighborhood)[T.B]', ...]
                            ├──► .term_names: ['Intercept', 'C(neighborhood)', 'bedrooms', 'sqft']
                            ├──► .factor_infos: 每个因子的详细信息
                            └──► .term_codings: 分类变量的对比矩阵
                  │
                  └──► get_matrices 保存 self._spec = exog.design_info
             │
             └──► handle_formula_data 返回:
                  ((endog, exog), missing_mask=None, model_spec=DesignInfo)
        │
        ├──► 验证 endog 维度 (通过)
        ├──► 处理 drop_cols (None)
        │
        ├──► 更新 kwargs:
        │    - missing_idx = None
        │    - missing = "drop"
        │    - formula = "price ~ bedrooms + C(neighborhood) + sqft"
        │    - model_spec = DesignInfo
        │
        └──► 实例化 OLS:
             mod = OLS(endog, exog, **kwargs)
        │
        ├──► mod.formula = 公式字符串
        ├──► mod.data.frame = df（原始数据引用）
        │
        └──► 返回 model 实例
│
└──► model.fit()  # 标准 OLS 拟合，只使用数值矩阵
     │
     └──► 返回 RegressionResults 实例
          │
          └──► results.model.model_spec = DesignInfo（供预测使用）
│
└──► results.predict(new_data)
     │
     └──► statsmodels/base/model.py:1209 (Results.predict)
          │
          └──► _transform_predict_exog(new_data, transform=True)
               │
               ├──► 检查: 模型有 formula 属性 → 是
               ├──► 获取 model_spec（从 results.model）
               │
               └──► FormulaManager().get_matrices(
                    model_spec,  # 注意：这里传入的是 DesignInfo，不是字符串！
                    new_data,
                    pandas=True,
                    prediction=True
                )
               │
               ├──► Patsy 使用保存的 DesignInfo:
               │    - C(neighborhood) 使用相同的参考水平 'A'
               │    - 列顺序与训练时完全一致
               │    - 不重新解析公式字符串
               │
               └──► 返回转换后的 exog:
                    DataFrame，列顺序: Intercept, C(neighborhood)[T.B], ...
│
└──► 最终预测值
```

这个示例清楚地展示了：
1. **训练时**：公式字符串 → DesignInfo + 设计矩阵
2. **保存时**：DesignInfo 被附加到 model 对象
3. **预测时**：直接使用 DesignInfo 转换新数据，确保一致性
