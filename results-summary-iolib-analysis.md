# Statsmodels Results-Summary-IOLib 架构分析

## 目录

1. [整体架构概览](#整体架构概览)
2. [参数估计与诊断指标的数据流](#参数估计与诊断指标的数据流)
3. [惰性计算与缓存机制](#惰性计算与缓存机制)
4. [新旧两套摘要接口的分工](#新旧两套摘要接口的分工)
5. [专项指标的通用结构适配](#专项指标的通用结构适配)
6. [展示层与计算层的解耦设计](#展示层与计算层的解耦设计)
7. [关键类与模块关系图](#关键类与模块关系图)

---

## 整体架构概览

Statsmodels 的结果展示系统采用三层架构：

```
┌─────────────────────────────────────────────────────────────┐
│                      展示层 (Presentation)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Summary    │  │  Summary2    │  │   SimpleTable    │  │
│  │ (iolib/summary.py)│  │(iolib/summary2.py)│  │ (iolib/table.py)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │ 数据传递
┌───────────────────────────▼─────────────────────────────────┐
│                     结果层 (Results)                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Results 类层次结构                        │  │
│  │  Results                                              │  │
│  │    └── LikelihoodModelResults                        │  │
│  │          ├── RegressionResults (线性回归)             │  │
│  │          ├── GLMResults (广义线性模型)                 │  │
│  │          ├── DiscreteResults (离散模型)               │  │
│  │          └── MLEResults (状态空间/时间序列)           │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │ 计算逻辑
┌───────────────────────────▼─────────────────────────────────┐
│                     计算层 (Computation)                      │
│  模型拟合、参数估计、统计检验、诊断指标计算                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 参数估计与诊断指标的数据流

### 核心数据属性

所有 Results 类都持有以下核心数据属性：

| 属性 | 类型 | 说明 | 来源 |
|------|------|------|------|
| `params` | ndarray | 参数估计值 | 拟合结果直接存储 |
| `bse` | ndarray | 标准误 | 协方差矩阵对角线开方 |
| `tvalues` | ndarray | t/z 统计量 | params / bse |
| `pvalues` | ndarray | p 值 | 基于 t/z 分布计算 |
| `conf_int()` | method | 置信区间 | 基于 alpha 计算 |

### 诊断指标分类

```
诊断指标
├── 拟合优度指标
│   ├── rsquared / rsquared_adj (线性模型)
│   ├── prsquared (离散模型，伪R²)
│   └── llf / llnull (对数似然值)
│
├── 信息准则
│   ├── aic (Akaike)
│   ├── bic (Bayesian)
│   └── hqic (Hannan-Quinn，仅时间序列)
│
├── 残差诊断
│   ├── resid / wresid / sresid
│   ├── fittedvalues
│   └── resid_pearson / resid_deviance (GLM)
│
└── 模型基本信息
    ├── nobs (样本量)
    ├── df_model / df_resid (自由度)
    └── scale (尺度参数)
```

### 数据流机制

数据从 Results 类流向 iolib 表格层的过程：

**1. Results 类构建数据列表** (`linear_model.py:2900-2923`)

```python
# 顶部表格左侧数据
top_left = [
    ("Dep. Variable:", None),
    ("Model:", None),
    ("Method:", ["Least Squares"]),
    ("Date:", None),
    ("Time:", None),
    ("No. Observations:", None),
    ("Df Residuals:", None),
    ("Df Model:", None),
]

# 顶部表格右侧数据（包含诊断指标）
top_right = [
    ("R-squared:", ["%#8.3f" % self.rsquared]),
    ("Adj. R-squared:", ["%#8.3f" % self.rsquared_adj]),
    ("F-statistic:", ["%#8.4g" % self.fvalue]),
    ("Prob (F-statistic):", ["%#6.3g" % self.f_pvalue]),
    ("Log-Likelihood:", None),
    ("AIC:", ["%#8.4g" % self.aic]),
    ("BIC:", ["%#8.4g" % self.bic]),
]
```

**2. 调用 iolib 的 Summary 类** (`linear_model.py:2958-2981`)

```python
from statsmodels.iolib.summary import Summary

smry = Summary()
smry.add_table_2cols(
    self,
    gleft=top_left,
    gright=top_right,
    yname=yname,
    xname=xname,
    title=title,
)
smry.add_table_params(
    self, yname=yname, xname=xname, alpha=alpha, use_t=self.use_t
)
```

**3. SimpleTable 作为中间数据结构** (`iolib/table.py`)

- `SimpleTable` 是一个通用的表格容器
- 支持多种输出格式：text, csv, latex, html
- 数据以行×列的二维列表形式存储

**4. 默认值填充机制** (`iolib/summary.py:293-304`)

当数据项为 `None` 时，使用 `default_items` 字典中的 lambda 函数动态获取：

```python
default_items = {
    "Dependent Variable:": lambda: [yname],
    "Model:": lambda: [results.model.__class__.__name__],
    "Date:": lambda: [date],
    "Time:": lambda: time_of_day,
    "Number of Obs:": lambda: [results.nobs],
    "Df Model:": lambda: [d_or_f(results.df_model)],
    "Df Residuals:": lambda: [d_or_f(results.df_resid)],
    "Log-Likelihood:": lambda: ["%#8.5g" % results.llf],
}
```

---

## 惰性计算与缓存机制

### 核心装饰器：`@cache_readonly`

**实现位置**: `statsmodels/tools/decorators.py`

**工作原理**:

```python
class CachedAttribute:
    def __get__(self, obj, type=None):
        # 获取或创建缓存字典
        _cache = getattr(obj, self.cachename, None)
        if _cache is None:
            setattr(obj, self.cachename, {})
            _cache = getattr(obj, self.cachename)
        
        # 检查缓存是否存在
        _cachedval = _cache.get(self.name, None)
        if _cachedval is None:
            # 首次访问时计算并缓存
            _cachedval = self.fget(obj)
            _cache[self.name] = _cachedval
        
        return _cachedval
```

**注意**: Statsmodels 实际使用的是 pandas 的 `cache_readonly` 实现（`decorators.py:137`）：
```python
cache_readonly = PandasCacheReadonly
```

### 缓存属性示例

**离散模型中的应用** (`discrete/discrete_model.py:4953-5110`)

```python
@cache_readonly
def prsquared(self):
    """McFadden's pseudo-R-squared. `1 - (llf / llnull)`"""
    return 1 - self.llf / self.llnull

@cache_readonly
def llr(self):
    """Likelihood ratio chi-squared statistic"""
    return -2 * (self.llnull - self.llf)

@cache_readonly
def llr_pvalue(self):
    """Likelihood ratio test p-value"""
    return stats.distributions.chi2.sf(self.llr, self.df_model)

@cache_readonly
def llnull(self):
    """Null model log-likelihood (computes null model on first access)"""
    # 首次访问时拟合空模型，计算开销大
    mod_null = model.__class__(model.endog, np.ones(self.nobs), **kwds)
    res_null = mod_null.fit(...)
    return res_null.llf

@cache_readonly
def aic(self):
    """Akaike information criterion"""
    return -2 * (self.llf - (self.df_model + 1 + k_extra))

@cache_readonly
def bic(self):
    """Bayesian information criterion"""
    return -2 * self.llf + np.log(self.nobs) * (self.df_model + 1 + k_extra)
```

### 缓存失效机制

**手动清除缓存** (`discrete/discrete_model.py:5002-5015`)

```python
def set_null_options(self, llnull=None, attach_results=True, **kwargs):
    # 清除依赖于 llnull 的所有缓存
    self._cache.pop("llnull", None)
    self._cache.pop("llr", None)
    self._cache.pop("llr_pvalue", None)
    self._cache.pop("prsquared", None)
    
    if hasattr(self, "res_null"):
        del self.res_null
    
    # 可直接设置缓存值，跳过计算
    if llnull is not None:
        self._cache["llnull"] = llnull
```

**数据相关缓存** (`base/model.py:1129`)

```python
class Results:
    def __init__(self, model, params, **kwd):
        # 需要从缓存中清除的变量列表
        self._data_in_cache = ["fittedvalues", "resid", "wresid"]
```

### 缓存策略总结

| 策略 | 应用场景 | 实现方式 |
|------|----------|----------|
| 惰性计算 | 计算开销大的指标（如 `llnull` 需要重新拟合模型） | `@cache_readonly` |
| 预计算 | 拟合时直接可得的结果（如 `params`, `llf`） | 直接存储为实例属性 |
| 按需清除 | 依赖关系变化时（如 `set_null_options`） | `_cache.pop()` |

---

## 新旧两套摘要接口的分工

### 接口概览

| 特性 | summary() (旧版/主流) | summary2() (新版/实验性) |
|------|----------------------|---------------------------|
| 模块 | `statsmodels.iolib.summary` | `statsmodels.iolib.summary2` |
| 状态 | 稳定/主流 | 实验性/备选 |
| 数据结构 | SimpleTable | pandas DataFrame |
| 多模型对比 | 不支持 | 支持 (`summary_col`) |

### summary() 接口分析

**核心类**: `statsmodels.iolib.summary.Summary`

**主要方法**:

```python
class Summary:
    def __init__(self):
        self.tables = []          # SimpleTable 列表
        self.extra_txt = None     # 额外说明文本
    
    def add_table_2cols(self, res, title=None, gleft=None, gright=None, ...):
        # 添加双列表格（模型基本信息 + 诊断指标）
        table = summary_top(res, gleft=gleft, gright=gright, ...)
        self.tables.append(table)
    
    def add_table_params(self, res, yname=None, xname=None, alpha=.05, use_t=True):
        # 添加参数估计表格
        table = summary_params(res, yname=yname, xname=xname, ...)
        self.tables.append(table)
    
    def add_extra_txt(self, etext):
        # 添加额外文本（如警告、注释）
        self.extra_txt = "\n".join(etext)
    
    # 输出方法
    def as_text(self): ...
    def as_latex(self): ...
    def as_csv(self): ...
    def as_html(self): ...
```

**数据组装流程** (`linear_model.py:2900-3015`):

```
1. 构建 top_left 列表 (模型基本信息)
        ↓
2. 构建 top_right 列表 (拟合优度、信息准则)
        ↓
3. 构建 diagn_left/diagn_right 列表 (残差诊断)
        ↓
4. 创建 Summary 实例
        ↓
5. add_table_2cols(gleft=top_left, gright=top_right)
        ↓
6. add_table_params()  # 参数估计表
        ↓
7. add_table_2cols(gleft=diagn_left, gright=diagn_right)  # 诊断表
        ↓
8. add_extra_txt(etext)  # 警告/注释
```

### summary2() 接口分析

**核心类**: `statsmodels.iolib.summary2.Summary`

**设计特点**:

1. **基于 pandas DataFrame**
   ```python
   def add_df(self, df, index=True, header=True, float_format="%.4f", align="r"):
       # 直接添加 DataFrame
       self.tables.append(df)
       self.settings.append(settings)
   ```

2. **更灵活的数据添加方式**
   ```python
   def add_dict(self, d, ncols=2, align="l", float_format="%.4f"):
       # 将字典转为表格
   ```

3. **多模型对比功能** (`summary_col`)
   ```python
   def summary_col(results, float_format="%.4f", model_names=(), stars=False,
                   info_dict=None, regressor_order=(), ...):
       # 并排展示多个模型的系数和标准误
   ```

**与 summary() 的区别** (`linear_model.py:3079-3090`):

```python
# summary2 使用不同的 Summary 类
from statsmodels.iolib import summary2

smry = summary2.Summary()
smry.add_base(
    results=self,
    alpha=alpha,
    float_format=float_format,
    xname=xname,
    yname=yname,
    title=title,
)
smry.add_dict(diagnostic)  # 使用字典而非元组列表
```

### 两套接口的分工总结

| 维度 | summary() | summary2() |
|------|-----------|------------|
| **设计目标** | 单个模型的完整摘要 | 灵活的表格组装、多模型对比 |
| **数据结构** | SimpleTable (自定义) | pandas DataFrame |
| **参数表** | `summary_params()` → SimpleTable | `summary_params()` → DataFrame |
| **模型信息** | `summary_top()` + `default_items` | `summary_model()` + try-except |
| **多模型** | 不支持 | `summary_col()` 并排展示 |
| **使用频率** | 绝大多数模型实现 | 部分模型实现 |

### 实际使用中的共存策略

大多数模型同时实现两套接口：

```python
class RegressionResults(LikelihoodModelResults):
    def summary(self, ...):
        """主流接口，返回 iolib.summary.Summary"""
        from statsmodels.iolib.summary import Summary
        ...
    
    def summary2(self, ...):
        """实验性接口，返回 iolib.summary2.Summary"""
        from statsmodels.iolib import summary2
        ...
```

但状态空间模型 (`MLEResults`) 只实现了 `summary()`，没有 `summary2()`。

---

## 专项指标的通用结构适配

### 信息准则的实现方式

**时间序列模型 (State Space)** (`mlemodel.py:5368-5371`):

```python
top_right = [
    ("No. Observations:", [self.nobs]),
    ("Log Likelihood", ["%#5.3f" % self.llf]),
]
if hasattr(self, "rsquared"):
    top_right.append(("R-squared:", ["%#8.3f" % self.rsquared]))
top_right += [
    ("AIC", ["%#5.3f" % self.aic]),
    ("BIC", ["%#5.3f" % self.bic]),
    ("HQIC", ["%#5.3f" % self.hqic]),  # 时间序列特有
]
```

**线性回归模型** (`linear_model.py:2915-2923`):

```python
top_right = [
    ("R-squared" + rsquared_type + ":", ["%#8.3f" % self.rsquared]),
    ("Adj. R-squared" + rsquared_type + ":", ["%#8.3f" % self.rsquared_adj]),
    ("F-statistic:", ["%#8.4g" % self.fvalue]),
    ("Prob (F-statistic):", ["%#6.3g" % self.f_pvalue]),
    ("Log-Likelihood:", None),
    ("AIC:", ["%#8.4g" % self.aic]),
    ("BIC:", ["%#8.4g" % self.bic]),
]
```

**离散模型** (`discrete/discrete_model.py:5430-5435`):

```python
top_right = [
    ("Pseudo R-squ.:", ["%#6.4g" % self.prsquared]),  # 伪R²
    ("Log-Likelihood:", None),
    ("LL-Null:", ["%#8.5g" % self.llnull]),
    ("LLR p-value:", ["%#6.4g" % self.llr_pvalue]),
    ("AIC:", ["%#8.4g" % self.aic]),
    ("BIC:", ["%#8.4g" % self.bic]),
]
```

### 伪 R² (Pseudo R-squared)

**离散模型中的实现** (`discrete/discrete_model.py:4953-4958`):

```python
@cache_readonly
def prsquared(self):
    """
    McFadden's pseudo-R-squared. `1 - (llf / llnull)`
    
    注意：需要先计算 llnull（空模型的对数似然）
    """
    return 1 - self.llf / self.llnull
```

**在摘要中展示** (`discrete/discrete_model.py:5431`):

```python
top_right = [
    ("Pseudo R-squ.:", ["%#6.4g" % self.prsquared]),
    ...
]
```

### summary2 中的动态获取机制

`summary2.py` 采用更灵活的方式：通过 try-except 动态尝试获取属性 (`summary2.py:286-333`):

```python
def summary_model(results):
    """创建包含模型信息的字典"""
    info = {}
    info["Model:"] = lambda x: x.model.__class__.__name__
    info["Model Family:"] = lambda x: x.family.__class.__name__  # GLM特有
    info["Link Function:"] = lambda x: x.family.link.__class__.__name__  # GLM特有
    info["Dependent Variable:"] = lambda x: x.model.endog_names
    info["No. Observations:"] = lambda x: "%#6d" % x.nobs
    info["Df Model:"] = lambda x: "%#6d" % x.df_model
    info["Df Residuals:"] = lambda x: "%#6d" % x.df_resid
    
    # 拟合优度指标（多种类型，按需获取）
    rsquared_type = "" if results.k_constant else " (uncentered)"
    info["R-squared" + rsquared_type + ":"] = lambda x: "%#8.3f" % x.rsquared
    info["Adj. R-squared" + rsquared_type + ":"] = lambda x: "%#8.3f" % x.rsquared_adj
    info["Pseudo R-squared:"] = lambda x: "%#8.3f" % x.prsquared  # 离散模型
    
    # 信息准则
    info["AIC:"] = lambda x: "%8.4f" % x.aic
    info["BIC:"] = lambda x: "%8.4f" % x.bic
    info["Log-Likelihood:"] = lambda x: "%#8.5g" % x.llf
    info["LL-Null:"] = lambda x: "%#8.5g" % x.llnull  # 部分模型
    info["LLR p-value:"] = lambda x: "%#8.5g" % x.llr_pvalue
    
    # GLM 特有
    info["Deviance:"] = lambda x: "%#8.5g" % x.deviance
    info["Pearson chi2:"] = lambda x: "%#6.3g" % x.pearson_chi2
    
    # 线性模型特有
    info["F-statistic:"] = lambda x: "%#8.4g" % x.fvalue
    info["Prob (F-statistic):"] = lambda x: "%#6.3g" % x.f_pvalue
    info["Scale:"] = lambda x: "%#8.5g" % x.scale
    
    out = {}
    for key, func in info.items():
        try:
            out[key] = func(results)
        except (AttributeError, KeyError, NotImplementedError):
            # 忽略不存在的属性
            pass
    return out
```

### 专项指标分类与适配策略

| 指标类型 | 涉及模型 | 实现方式 | 展示位置 |
|----------|----------|----------|----------|
| **AIC** | 所有似然模型 | `@cache_readonly` | top_right |
| **BIC** | 所有似然模型 | `@cache_readonly` | top_right |
| **HQIC** | 状态空间模型 | 属性 | top_right (时间序列特有) |
| **R²** | 线性模型 | 属性 | top_right |
| **Adj. R²** | 线性模型 | 属性 | top_right |
| **Pseudo R²** | 离散模型 | `@cache_readonly` (基于 llnull) | top_right |
| **F统计量** | 线性模型 | 属性 | top_right |
| **LLR p值** | 离散模型 | `@cache_readonly` | top_right |
| **Deviance** | GLM | 属性 | 动态获取 |
| **Pearson χ²** | GLM | 属性 | 动态获取 |

### 适配模式总结

**模式 1: 显式构建列表** (`summary()` 接口)

每个模型类型在 `summary()` 方法中显式构建 `top_left`/`top_right` 列表，只包含该模型实际存在的指标。

**优点**: 精确控制，不会出现缺失值
**缺点**: 代码重复，每个模型都要重写类似逻辑

**模式 2: 动态获取 + try-except** (`summary2()` 接口)

在 `summary_model()` 中定义所有可能的指标，通过 try-except 过滤不存在的属性。

**优点**: 代码集中，易于扩展
**缺点**: 依赖属性命名约定

---

## 展示层与计算层的解耦设计

### 层次架构

```
┌──────────────────────────────────────────────────────────────┐
│                        展示层 (Presentation)                   │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Output Formatters                           │ │
│  │  as_text()  as_html()  as_latex()  as_csv()            │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Summary 容器类                              │ │
│  │  statsmodels.iolib.summary.Summary                      │ │
│  │  statsmodels.iolib.summary2.Summary                     │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Table 数据结构                              │ │
│  │  statsmodels.iolib.table.SimpleTable                    │ │
│  └─────────────────────────────────────────────────────────┘ │
└────────────────────────────┬─────────────────────────────────┘
                             │ 数据契约
┌────────────────────────────▼─────────────────────────────────┐
│                        结果层 (Results)                        │
│                                                               │
│  职责: 持有数据，提供计算方法                                 │
│  - params, bse, tvalues, pvalues (核心参数)                 │
│  - aic, bic, rsquared, prsquared (诊断指标)                  │
│  - summary() / summary2() (桥梁方法)                         │
└────────────────────────────┬─────────────────────────────────┘
                             │ 计算契约
┌────────────────────────────▼─────────────────────────────────┐
│                        计算层 (Computation)                    │
│                                                               │
│  职责: 模型拟合、统计计算                                      │
│  - Model.fit() → 参数估计                                    │
│  - 协方差矩阵计算                                             │
│  - 假设检验 (t检验、F检验、似然比检验)                        │
└──────────────────────────────────────────────────────────────┘
```

### 解耦的关键设计

#### 1. Results 类不依赖展示层

**证据**: Results 类的导入关系

```python
# base/model.py - 不导入任何 iolib 模块
# 只有在 summary() 方法被调用时才动态导入

def summary(self, yname=None, xname=None, title=None, alpha=0.05):
    # 延迟导入，避免循环依赖
    from statsmodels.iolib.summary import Summary
    ...
```

#### 2. Summary 类通过契约访问数据

Summary 类不直接依赖具体的 Results 子类，而是通过**属性访问契约**获取数据：

```python
# iolib/summary.py:389 - summary_params 函数
def summary_params(results, yname=None, xname=None, alpha=.05, use_t=True, ...):
    # 契约：results 必须提供以下属性
    params = np.asarray(results.params)      # 必须有
    std_err = np.asarray(results.bse)        # 必须有
    tvalues = np.asarray(results.tvalues)    # 必须有
    pvalues = np.asarray(results.pvalues)    # 必须有
    conf_int = np.asarray(results.conf_int(alpha))  # 必须有方法
```

**契约清单** (Results 类需要提供的属性/方法):

| 属性/方法 | 必需 | 用途 |
|-----------|------|------|
| `params` | 是 | 参数估计值 |
| `bse` | 是 | 标准误 |
| `tvalues` | 是 | t/z 统计量 |
| `pvalues` | 是 | p 值 |
| `conf_int(alpha)` | 是 | 置信区间 |
| `model` | 是 | 模型引用 (用于获取 names) |
| `nobs` | 是 | 样本量 |
| `df_model` | 是 | 模型自由度 |
| `df_resid` | 是 | 残差自由度 |
| `llf` | 否 | 对数似然 |
| `aic` | 否 | AIC 准则 |
| `bic` | 否 | BIC 准则 |
| `use_t` | 否 | 使用 t 分布还是正态分布 |

#### 3. 双向不依赖的设计

```
展示层 (iolib.summary)
    ↓ 依赖
数据契约 (属性访问协议)
    ↑ 实现
结果层 (base.model.Results)
    ↓ 依赖
计算层 (模型拟合逻辑)
```

**展示层不依赖具体 Results 子类**：
- `summary_params()` 函数接受任何提供所需属性的对象
- 不检查 `isinstance(results, RegressionResults)`

**Results 类不依赖展示层实现**：
- 只有调用 `summary()` 时才导入 iolib
- 可以替换展示层实现而不影响 Results 类

#### 4. 桥梁方法：`summary()`

`summary()` 方法是连接计算层和展示层的唯一桥梁：

```python
# 典型的 summary() 实现模式
def summary(self, yname=None, xname=None, title=None, alpha=0.05):
    # 1. 从 self 提取/计算所需数据
    #    - 直接访问属性: self.params, self.rsquared
    #    - 调用方法: self.conf_int(alpha)
    #    - 计算诊断指标: jarque_bera(self.wresid)
    
    # 2. 构建表格数据结构
    top_left = [...]
    top_right = [...]
    diagn_left = [...]
    diagn_right = [...]
    
    # 3. 导入展示层类（延迟导入）
    from statsmodels.iolib.summary import Summary
    
    # 4. 组装展示对象
    smry = Summary()
    smry.add_table_2cols(self, gleft=top_left, gright=top_right, ...)
    smry.add_table_params(self, ...)
    
    # 5. 返回展示对象
    return smry
```

### SimpleTable：独立的数据结构

`SimpleTable` 是一个完全独立的表格类，不依赖任何模型相关代码：

```python
# iolib/table.py - 仅依赖标准库
import csv
from itertools import cycle, zip_longest
from statsmodels.compat.python import lmap, lrange

# 不导入任何 model 或 results 相关模块
```

**职责单一原则**:
- `SimpleTable` 只负责：数据存储、格式化、多格式输出
- 不负责：数据计算、模型逻辑

### 解耦带来的好处

1. **可测试性**: 可以独立测试展示层
   ```python
   # 无需拟合模型即可测试 SimpleTable
   mydata = [[11, 12], [21, 22]]
   tbl = SimpleTable(mydata, ["Col1", "Col2"], ["Row1", "Row2"])
   print(tbl.as_text())
   ```

2. **可扩展性**: 新增输出格式无需修改 Results 类
   - 只需在 `Summary` 或 `SimpleTable` 中添加新的 `as_*()` 方法

3. **可替换性**: 可以完全替换展示层
   - 例如用 `summary2` 替换 `summary`，Results 类基本无需修改

4. **关注点分离**:
   - 统计学家关注：模型拟合、参数估计、假设检验
   - 开发者关注：表格格式化、输出样式

### 依赖关系图

```
statsmodels/
├── base/
│   └── model.py          # Results 类 (结果层)
│       │
│       └── 调用 summary() 时才导入 → 延迟依赖
│
├── iolib/
│   ├── summary.py        # Summary 类 (展示层)
│   │   └── 依赖: table.py, 通过属性访问协议使用 results
│   │
│   ├── summary2.py       # Summary2 类 (展示层)
│   │   └── 依赖: table.py, pandas
│   │
│   └── table.py          # SimpleTable (数据结构)
│       └── 无外部依赖 (仅标准库)
│
└── 各模型模块/
    ├── linear_model.py   # RegressionResults
    ├── discrete/discrete_model.py  # DiscreteResults
    ├── genmod/generalized_linear_model.py  # GLMResults
    └── tsa/statespace/mlemodel.py  # MLEResults
        └── 均继承自 base.model.LikelihoodModelResults
```

---

## 关键类与模块关系图

### 类继承层次

```
Results (base/model.py:1112)
│
├── 核心属性
│   ├── params: ndarray
│   ├── model: Model
│   ├── _cache: dict
│   └── _data_in_cache: list
│
└── LikelihoodModelResults (base/model.py:1276)
    │
    ├── 新增属性
    │   ├── llf: float (对数似然)
    │   ├── aic: float (@cache_readonly)
    │   ├── bic: float (@cache_readonly)
    │   └── llnull: float (@cache_readonly, 部分子类)
    │
    ├── GenericLikelihoodModelResults (base/model.py:2802)
    │   └── 继承: LikelihoodModelResults + ResultMixin
    │   └── 用途: GenericLikelihoodModel 的默认结果类
    │
    ├── RegressionResults (regression/linear_model.py)
    │   ├── rsquared / rsquared_adj
    │   ├── fvalue / f_pvalue
    │   └── eigenvals / condition_number
    │
    ├── GLMResults (genmod/generalized_linear_model.py)
    │   ├── deviance
    │   ├── pearson_chi2
    │   └── family / link
    │
    ├── DiscreteResults (discrete/discrete_model.py:4905)  ← 离散模型独立分支
    │   ├── 继承: base.LikelihoodModelResults (直接继承，不经过 GenericLikelihoodModelResults)
    │   ├── 子分支:
    │   │   ├── CountResults → PoissonResults, NegativeBinomialResults, 等
    │   │   ├── BinaryResults → LogitResults, ProbitResults
    │   │   ├── OrderedResults
    │   │   └── MultinomialResults → MNLogitResults
    │   └── 特有属性/方法:
    │       ├── prsquared (伪R²，@cache_readonly)
    │       ├── llnull (空模型对数似然，@cache_readonly)
    │       ├── llr, llr_pvalue (似然比检验，@cache_readonly)
    │       └── set_null_options() (缓存管理)
    │
    └── MLEResults (tsa/statespace/mlemodel.py:2560)
        ├── hqic (Hannan-Quinn 准则，时间序列特有)
        ├── filter_results
        └── param_names
```

### 模块依赖关系

```
┌──────────────────────────────────────────────────────────────┐
│                        用户代码                                │
│  model.fit() → results → results.summary() → print()         │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                    Results 类 (结果层)                         │
│                                                               │
│  持有：params, bse, tvalues, pvalues, llf, aic, bic, ...   │
│  方法：summary(), summary2()                                 │
└────────────────────────────┬─────────────────────────────────┘
                             │ 调用时才导入
        ┌────────────────────┴────────────────────┐
        ▼                                         ▼
┌───────────────────┐                   ┌───────────────────┐
│   iolib.summary   │                   │  iolib.summary2   │
│   (主流接口)       │                   │  (实验性接口)      │
├───────────────────┤                   ├───────────────────┤
│ • Summary 类      │                   │ • Summary 类      │
│ • summary_top()   │                   │ • add_base()      │
│ • summary_params()│                   │ • summary_model() │
│ • default_items   │                   │ • summary_col()   │
└─────────┬─────────┘                   └─────────┬─────────┘
          │                                         │
          └──────────────────┬──────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                    iolib.table.SimpleTable                     │
│                                                               │
│  职责：数据存储、格式化、多格式输出                            │
│  输出：as_text(), as_csv(), as_html(), as_latex_tabular()   │
└──────────────────────────────────────────────────────────────┘
```

### 关键文件位置

| 文件 | 职责 | 关键类/函数 |
|------|------|-------------|
| `statsmodels/base/model.py` | 结果层基类 | `Results`, `LikelihoodModelResults`, `GenericLikelihoodModelResults` |
| `statsmodels/iolib/summary.py` | 主流展示层 | `Summary`, `summary_top()`, `summary_params()` |
| `statsmodels/iolib/summary2.py` | 实验性展示层 | `Summary`, `summary_model()`, `summary_col()` |
| `statsmodels/iolib/table.py` | 表格数据结构 | `SimpleTable` |
| `statsmodels/tools/decorators.py` | 缓存装饰器 | `@cache_readonly`, `CachedAttribute` |
| `statsmodels/regression/linear_model.py` | 线性回归结果 | `RegressionResults.summary()` |
| `statsmodels/discrete/discrete_model.py` | 离散模型结果 | `DiscreteResults`, 伪R²实现 |
| `statsmodels/tsa/statespace/mlemodel.py` | 时间序列结果 | `MLEResults.summary()` |
| `statsmodels/genmod/generalized_linear_model.py` | GLM结果 | `GLMResults` |

---

## 总结

### 架构设计的核心原则

1. **延迟依赖**: Results 类不在模块层面导入 iolib，只在 `summary()` 调用时延迟导入
2. **契约设计**: 展示层通过属性访问契约与结果层交互，不依赖具体类
3. **惰性计算**: 计算开销大的指标使用 `@cache_readonly` 实现按需计算+缓存
4. **职责分离**:
   - 计算层：模型拟合、参数估计
   - 结果层：数据持有、指标计算
   - 展示层：表格组装、格式输出

### 新旧接口对比

| 方面 | summary() | summary2() |
|------|-----------|------------|
| 稳定性 | 稳定，所有模型实现 | 实验性，部分模型实现 |
| 数据结构 | SimpleTable | pandas DataFrame |
| 多模型对比 | 不支持 | 支持 (summary_col) |
| 灵活性 | 较低（硬编码列表） | 较高（动态获取） |

### 专项指标的适配模式

1. **信息准则** (AIC, BIC, HQIC): 统一作为 Results 属性，`@cache_readonly` 装饰
2. **伪 R²**: 离散模型特有，命名为 `prsquared` 以区别于 `rsquared`
3. **GLM 特有指标**: `deviance`, `pearson_chi2` 作为属性存在
4. **展示策略**: 
   - `summary()`: 每个模型显式构建自己的指标列表
   - `summary2()`: 通过 try-except 动态过滤不存在的指标

### 缓存策略

```
┌──────────────────────────────────────────────────────────────┐
│                    缓存策略决策树                              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  指标获取方式？                                               │
│  ├── 直接存储 (拟合时已计算)                                  │
│  │   └── 例: params, llf, nobs, df_model                     │
│  │                                                             │
│  └── 需要计算                                                 │
│      ├── 计算开销小？                                         │
│      │   ├── 是 → property (每次计算)                        │
│      │   │   └── 例: tvalues = params / bse                  │
│      │   │                                                     │
│      │   └── 否 → @cache_readonly (惰性计算+缓存)            │
│      │       ├── 需要拟合模型？                               │
│      │       │   └── 例: llnull (空模型重新拟合)            │
│      │       │                                                 │
│      │       └── 数学计算？                                   │
│      │           └── 例: aic, bic, prsquared, llr_pvalue    │
│      │                                                         │
│      └── 缓存失效？                                           │
│          └── 手动清除: _cache.pop("llnull", None)            │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

*分析日期: 2026-05-03*
*基于 statsmodels 代码库分析*
