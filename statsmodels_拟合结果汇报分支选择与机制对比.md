# Statsmodels 拟合结果汇报：分支选择与机制对比深度分析

## 一、汇报路径全景图

### 1.1 两套独立的汇报机制

Statsmodels 实现了**两套独立但相互关联**的结果汇报机制：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           汇报机制架构全景                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        统一统计量计算层                                 │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │   │
│  │  │  params  │ │   bse    │ │ tvalues  │ │ pvalues  │              │   │
│  │  │ (系数)    │ │ (标准误)  │ │ (t/z值)  │ │ (p值)    │              │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘              │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │   │
│  │  │ conf_int │ │   llf    │ │   aic    │ │   bic    │              │   │
│  │  │ (置信区间) │ │ (对数似然)│ │ (信息准则)│ │ (信息准则)│              │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘              │   │
│  │                                                                       │   │
│  │  位置: statsmodels/base/model.py (LikelihoodModelResults 类)         │   │
│  └─────────────────────────────────┬───────────────────────────────────┘   │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        模型特定统计量扩展层                             │   │
│  │                                                                       │   │
│  │  线性回归: rsquared, rsquared_adj, fvalue, f_pvalue                  │   │
│  │  离散模型: prsquared (伪R²), llnull, llr_pvalue                      │   │
│  │  状态空间: hqic, Ljung-Box, Jarque-Bera 等诊断统计量                  │   │
│  │                                                                       │   │
│  └─────────────────────────────────┬───────────────────────────────────┘   │
│                                    │                                          │
│                    ┌───────────────┴───────────────┐                      │
│                    │                               │                      │
│                    ▼                               ▼                      │
│  ┌─────────────────────────────┐   ┌─────────────────────────────┐       │
│  │     机制 A: summary.py       │   │     机制 B: summary2.py      │       │
│  │     (主流/稳定版本)           │   │     (实验性版本)             │       │
│  ├─────────────────────────────┤   ├─────────────────────────────┤       │
│  │ • SimpleTable 表格容器       │   │ • pandas DataFrame 中间层   │       │
│  │ • add_table_2cols()         │   │ • add_df(), add_array()     │       │
│  │ • add_table_params()        │   │ • add_dict()                 │       │
│  │ • 固定双列布局               │   │ • 灵活布局                   │       │
│  │ • 被大多数模型使用           │   │ • summary_col() 多模型比较   │       │
│  └─────────────────────────────┘   └─────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 路径选择触发条件

| 调用方式 | 触发场景 | 选择的机制 | 触发代码位置 |
|---------|---------|-----------|-------------|
| `results.summary()` | 用户显式调用 | **机制 A** | 各模型类的 `summary()` 方法 |
| `results.summary2()` | 用户显式调用 | **机制 B** | 部分模型类的 `summary2()` 方法 |
| `print(results)` | 隐式调用 `__str__()` | **机制 A** | `Summary.__str__()` → `as_text()` |
| Jupyter 显示 | 调用 `_repr_html_()` | **机制 A** | `Summary._repr_html_()` |
| `summary_col([...])` | 多模型并排比较 | **机制 B** | `statsmodels.iolib.summary2.summary_col()` |

---

## 二、不同模型的汇报路径实现

### 2.1 线性回归模型：双路径支持

**文件**: `statsmodels/regression/linear_model.py`

线性回归模型（OLS/GLS/WLS）同时实现了两套汇报机制：

#### 路径 A: `summary()` 方法 (主流机制)

**位置**: `linear_model.py:2822-3015`

```python
def summary(self, yname=None, xname=None, title=None, alpha=0.05, slim=False):
    # 1. 计算诊断统计量（模型特定）
    jb, jbpv, skew, kurtosis = jarque_bera(self.wresid)
    omni, omnipv = omni_normtest(self.wresid)
    eigvals = self.eigenvals
    condno = self.condition_number
    
    # 2. 构建顶部信息表（双列布局）
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
    
    top_right = [
        ("R-squared:", ["%#8.3f" % self.rsquared]),
        ("Adj. R-squared:", ["%#8.3f" % self.rsquared_adj]),
        ("F-statistic:", ["%#8.4g" % self.fvalue]),
        ("Prob (F-statistic):", ["%#6.3g" % self.f_pvalue]),
        ("Log-Likelihood:", None),
        ("AIC:", ["%#8.4g" % self.aic]),
        ("BIC:", ["%#8.4g" % self.bic]),
    ]
    
    # 3. 创建 Summary 实例并添加表格
    from statsmodels.iolib.summary import Summary
    smry = Summary()
    smry.add_table_2cols(self, gleft=top_left, gright=top_right, ...)  # 顶部双列表
    smry.add_table_params(self, ...)  # 参数表
    smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right, ...)  # 诊断表
    
    # 4. 添加警告文本
    smry.add_extra_txt(etext)
    
    return smry
```

#### 路径 B: `summary2()` 方法 (实验性机制)

**位置**: `linear_model.py:3017-3130`

```python
def summary2(self, yname=None, xname=None, title=None, alpha=0.05, float_format="%.4f"):
    # 1. 计算相同的诊断统计量（复用统计量计算层）
    jb, jbpv, skew, kurtosis = jarque_bera(self.wresid)
    # ... 相同的统计量计算
    
    # 2. 使用 summary2 机制
    from statsmodels.iolib import summary2
    smry = summary2.Summary()
    
    # 3. 使用 add_base() 自动添加基础信息和参数表
    smry.add_base(results=self, alpha=alpha, float_format=float_format, ...)
    
    # 4. 使用 add_dict() 添加诊断信息
    smry.add_dict(diagnostic)
    
    # 5. 使用 add_text() 添加警告
    for line in etext:
        smry.add_text(line)
    
    return smry
```

### 2.2 离散选择模型：双路径支持

**文件**: `statsmodels/discrete/discrete_model.py`

离散模型（Logit/Probit/MNLogit）同样实现了两套机制：

#### 路径 A: `summary()` 方法

**位置**: `discrete_model.py:5390-5469`

```python
def summary(self, yname=None, xname=None, title=None, alpha=0.05, yname_list=None):
    # 离散模型特有的统计量
    top_left = [
        ("Dep. Variable:", None),
        ("Model:", [self.model.__class__.__name__]),
        ("Method:", [self.method]),  # MLE 等方法
        ("Date:", None),
        ("Time:", None),
        ("converged:", ["%s" % self.mle_retvals["converged"]]),  # 收敛状态
    ]
    
    top_right = [
        ("No. Observations:", None),
        ("Df Residuals:", None),
        ("Df Model:", None),
        ("Pseudo R-squ.:", ["%#6.4g" % self.prsquared]),  # 伪 R²
        ("Log-Likelihood:", None),
        ("LL-Null:", ["%#8.5g" % self.llnull]),  # 空模型对数似然
        ("LLR p-value:", ["%#6.4g" % self.llr_pvalue]),  # LLR 检验 p 值
    ]
    
    # 使用机制 A
    from statsmodels.iolib.summary import Summary
    smry = Summary()
    smry.add_table_2cols(...)
    smry.add_table_params(...)
    return smry
```

#### 路径 B: `summary2()` 方法

**位置**: `discrete_model.py:5471-5519`

```python
def summary2(self, yname=None, xname=None, title=None, alpha=0.05, float_format="%.4f"):
    from statsmodels.iolib import summary2
    smry = summary2.Summary()
    smry.add_base(results=self, alpha=alpha, float_format=float_format, ...)
    return smry
```

### 2.3 状态空间模型：仅路径 A

**文件**: `statsmodels/tsa/statespace/mlemodel.py`

状态空间模型（SARIMAX/VARMAX/Dynamic Factor）**仅实现了机制 A**，且实现更为复杂：

**位置**: `mlemodel.py:5266-5493`

```python
def summary(self, alpha=0.05, start=None, title=None, model_name=None,
            display_params=True, display_diagnostics=True, ...):
    # 状态空间模型特有的复杂处理
    
    # 1. 多内生变量处理
    if self.model.k_endog > display_max_endog:
        yname = '"' + endog_names[0] + f'", and {k} more'
    
    # 2. 样本区间显示（时间序列特有）
    if self.model._index_dates:
        sample = ["%02d-%02d-%02d" % (d.month, d.day, d.year), ...]
    else:
        sample = [str(start), " - " + str(self.nobs)]
    
    # 3. 状态空间特有统计量
    top_right = [
        ("No. Observations:", [self.nobs]),
        ("Log Likelihood", ["%#5.3f" % self.llf]),
        ("AIC", ["%#5.3f" % self.aic]),
        ("BIC", ["%#5.3f" % self.bic]),
        ("HQIC", ["%#5.3f" % self.hqic]),  # 状态空间特有：HQIC 准则
    ]
    
    # 4. 诊断测试（状态空间特有）
    if display_diagnostics:
        het = self.test_heteroskedasticity(method="breakvar")
        lb = self.test_serial_correlation(method="ljungbox", lags=[1])
        jb = self.test_normality(method="jarquebera")
        
        # 多内生变量时的诊断表布局变化
        if self.model.k_endog <= display_max_endog:
            # 双列布局
            diagn_left = [("Ljung-Box (L1) (Q):", ...), ...]
            diagn_right = [("Jarque-Bera (JB):", ...), ...]
            summary.add_table_2cols(...)
        else:
            # 宽表布局（每个内生变量一行）
            data = pd.DataFrame(np.c_[lb[:, :2, -1], het[:, :2], jb[:, :4]], ...)
            table = SimpleTable(params_data, params_header, ...)
            summary.tables.insert(table_ix, table)
    
    # 5. 使用机制 A
    from statsmodels.iolib.summary import Summary
    summary = Summary()
    summary.add_table_2cols(...)
    summary.add_table_params(...)
    return summary
```

### 2.4 模型路径选择汇总

| 模型家族 | 实现 `summary()` | 实现 `summary2()` | 推荐使用 |
|---------|-----------------|------------------|----------|
| 线性回归 (OLS/GLS/WLS) | ✅ | ✅ | `summary()` |
| 离散选择 (Logit/Probit) | ✅ | ✅ | `summary()` |
| 广义线性模型 (GLM) | ✅ | ❌ | `summary()` |
| 稳健线性模型 (RLM) | ✅ | ❌ | `summary()` |
| 状态空间 (SARIMAX/VARMAX) | ✅ | ❌ | `summary()` |
| 多方程模型 (VAR/VECM) | ✅ | ❌ | `summary()` |

---

## 三、统计量复用与分流机制

### 3.1 统一的统计量计算层

所有汇报机制都复用同一套统计量计算逻辑，位于 `statsmodels/base/model.py` 的 `LikelihoodModelResults` 类：

#### 核心统计量（所有模型共享）

**位置**: `base/model.py:1503-1541`

```python
@cached_value
def bse(self):
    """标准误：协方差矩阵对角线的平方根"""
    return np.sqrt(np.diag(self.cov_params()))

@cached_value
def tvalues(self):
    """t/z 值：系数 / 标准误"""
    return self.params / self.bse

@cached_value
def pvalues(self):
    """p 值：基于 t 分布或正态分布"""
    if self.use_t:
        df_resid = getattr(self, "df_resid_inference", self.df_resid)
        return stats.t.sf(np.abs(self.tvalues), df_resid) * 2
    else:
        return stats.norm.sf(np.abs(self.tvalues)) * 2

@cached_value
def llf(self):
    """对数似然值"""
    return self.model.loglike(self.params)
```

#### 统计量计算依赖关系

```
┌─────────────────────────────────────────────────────────────────────┐
│                      统计量计算依赖关系图                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   params (系数估计值) ◄─────────── fit() 方法直接返回               │
│       │                                                             │
│       ▼                                                             │
│   normalized_cov_params (标准化协方差)                               │
│       │                                                             │
│       ├──► cov_params() = normalized_cov_params × scale            │
│       │           │                                                 │
│       │           ▼                                                 │
│       │        bse = sqrt(diag(cov_params()))                      │
│       │           │                                                 │
│       │           ├──► tvalues = params / bse                      │
│       │           │           │                                     │
│       │           │           └──► pvalues (t分布/正态分布)        │
│       │           │                                                 │
│       │           └──► conf_int() = params ± t_crit × bse          │
│       │                                                             │
│       └──► llf = model.loglike(params)                              │
│                   │                                                 │
│                   ├──► aic = -2*llf + 2*k                          │
│                   ├──► bic = -2*llf + k*log(n)                      │
│                   └──► 模型特定的拟合优度统计量                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 模型特定统计量扩展

不同模型家族在统一统计量基础上扩展特定统计量：

#### 线性回归模型扩展

**位置**: `linear_model.py:1871-2000`

```python
@cache_readonly
def rsquared(self):
    """R² = 1 - SSR/SST"""
    return 1 - self.ssr / self.centered_tss

@cache_readonly
def rsquared_adj(self):
    """调整 R²"""
    return 1 - (self.ssr / self.df_resid) / (self.centered_tss / self.df_model)

@cache_readonly
def fvalue(self):
    """F 统计量"""
    return (self.mse_model / self.scale) if self.scale else np.inf

@cache_readonly
def aic(self):
    """AIC 信息准则"""
    return -2 * self.llf + 2 * (self.df_model + 1)

@cache_readonly
def bic(self):
    """BIC 信息准则"""
    return -2 * self.llf + np.log(self.nobs) * (self.df_model + 1)
```

#### 离散模型扩展

**位置**: `discrete_model.py` (各结果类)

```python
@cache_readonly
def prsquared(self):
    """伪 R² = 1 - llf/llnull"""
    return 1 - self.llf / self.llnull

@cache_readonly
def llr_pvalue(self):
    """似然比检验 p 值"""
    from scipy.stats import chi2
    return chi2.sf(self.llr, self.df_model)
```

### 3.3 两套机制的统计量提取方式

#### 机制 A 的提取方式：懒加载 + 默认值字典

**位置**: `summary.py:291-335`

```python
def summary_top(results, title=None, gleft=None, gright=None, ...):
    # 使用 lambda 函数实现懒加载
    default_items = {
          "Dependent Variable:": lambda: [yname],
          "Model:": lambda: [results.model.__class__.__name__],
          "Date:": lambda: [date],
          "No. Observations:": lambda: [d_or_f(results.nobs)],
          "Df Model:": lambda: [d_or_f(results.df_model)],
          "Df Residuals:": lambda: [d_or_f(results.df_resid)],
          "Log-Likelihood:": lambda: ["%#8.5g" % results.llf]
    }
    
    # 替换 None 值为默认值（调用 lambda）
    for item, value in gen_left:
        if value is None:
            value = default_items[item]()  # 懒加载执行
        gen_left_.append((item, value))
```

#### 机制 B 的提取方式：直接访问 + DataFrame 封装

**位置**: `summary2.py:286-333`

```python
def summary_model(results):
    """创建模型信息字典"""
    info = {}
    info["Model:"] = lambda x: x.model.__class__.__name__
    info["No. Observations:"] = lambda x: "%#6d" % x.nobs
    info["Df Model:"] = lambda x: "%#6d" % x.df_model
    info["R-squared:"] = lambda x: "%#8.3f" % x.rsquared
    info["AIC:"] = lambda x: "%8.4f" % x.aic
    info["BIC:"] = lambda x: "%8.4f" % x.bic
    # ... 更多统计量
    
    out = {}
    for key, func in info.items():
        try:
            out[key] = func(results)  # 直接访问属性
        except (AttributeError, KeyError, NotImplementedError):
            pass  # 不存在则跳过
    return out
```

**参数表提取**: `summary2.py:336-393`

```python
def summary_params(results, ..., use_t=True, ...):
    # 直接从 results 对象提取统计量
    params = results.params
    bse = results.bse
    tvalues = results.tvalues
    pvalues = results.pvalues
    conf_int = results.conf_int(alpha)
    
    # 封装为 DataFrame
    data = np.array([params, bse, tvalues, pvalues]).T
    data = np.hstack([data, conf_int])
    data = pd.DataFrame(data)
    
    return data
```

### 3.4 统计量分流决策逻辑

不同汇报场景选择展示不同的统计量子集：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         统计量分流决策逻辑                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   统一统计量池                                                               │
│   ┌─────────┬─────────┬─────────┬─────────┬─────────┐                   │
│   │ params  │   bse   │ tvalues │ pvalues │ conf_int│  ◄─── 参数表必备    │
│   └─────────┴─────────┴─────────┴─────────┴─────────┘                   │
│   ┌─────────┬─────────┬─────────┬─────────┬─────────┐                   │
│   │   llf   │   aic   │   bic   │  nobs   │ df_model│  ◄─── 模型信息表必备 │
│   └─────────┴─────────┴─────────┴─────────┴─────────┘                   │
│                                                                             │
│                         │                                                   │
│                         ▼                                                   │
│              ┌───────────────────────┐                                     │
│              │   模型类型判断         │                                     │
│              └───────────┬───────────┘                                     │
│                          │                                                   │
│            ┌─────────────┼─────────────┐                                   │
│            │             │             │                                   │
│            ▼             ▼             ▼                                   │
│     ┌───────────┐ ┌───────────┐ ┌───────────┐                             │
│     │ 线性回归   │ │ 离散模型   │ │ 状态空间   │                             │
│     └─────┬─────┘ └─────┬─────┘ └─────┬─────┘                             │
│           │             │             │                                   │
│           ▼             ▼             ▼                                   │
│     ┌───────────┐ ┌───────────┐ ┌───────────┐                             │
│     │ rsquared  │ │ prsquared │ │  hqic     │                             │
│     │ fvalue    │ │ llnull    │ │ Ljung-Box│                             │
│     │ f_pvalue  │ │ llr_pvalue│ │ Jarque-Bera│                           │
│     └───────────┘ └───────────┘ │ Heterosk. │                             │
│                                   └───────────┘                             │
│                                                                             │
│                         │                                                   │
│                         ▼                                                   │
│              ┌───────────────────────┐                                     │
│              │   输出格式判断         │                                     │
│              └───────────┬───────────┘                                     │
│                          │                                                   │
│            ┌─────────────┼─────────────┐                                   │
│            │             │             │                                   │
│            ▼             ▼             ▼                                   │
│     ┌───────────┐ ┌───────────┐ ┌───────────┐                             │
│     │  Text     │ │  HTML     │ │  LaTeX    │                             │
│     │  格式     │ │  格式     │ │  格式     │                             │
│     └───────────┘ └───────────┘ └───────────┘                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 四、两套机制设计取舍对比

### 4.1 核心设计差异

| 设计维度 | 机制 A (`summary.py`) | 机制 B (`summary2.py`) |
|---------|----------------------|-----------------------|
| **表格容器** | `SimpleTable` (自定义类) | `pandas.DataFrame` |
| **依赖** | 仅 `numpy` | `numpy` + `pandas` |
| **布局方式** | 固定双列布局 (`add_table_2cols`) | 灵活布局 (`add_df`, `add_array`) |
| **多模型比较** | ❌ 不原生支持 | ✅ `summary_col()` 原生支持 |
| **参数表方法** | `add_table_params()` (专用) | `add_base()` + `summary_params()` (通用) |
| **状态** | 稳定 / 主流 | 实验性 / 备选 |
| **文档标记** | 标准方法 | "Experimental" |

### 4.2 代码风格对比

#### 机制 A：显式构建，细粒度控制

**优点**：
- 完全控制每一行内容
- 双列布局紧凑美观
- 无需 pandas 依赖

**缺点**：
- 代码冗长
- 每次需要手动构建 `top_left`, `top_right` 列表
- 多模型比较需要自己实现

```python
# 机制 A 风格：手动构建每一项
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

top_right = [
    ("R-squared:", ["%#8.3f" % self.rsquared]),
    ("Adj. R-squared:", ["%#8.3f" % self.rsquared_adj]),
    ("F-statistic:", ["%#8.4g" % self.fvalue]),
    # ... 更多项
]

smry = Summary()
smry.add_table_2cols(self, gleft=top_left, gright=top_right, ...)
smry.add_table_params(self, ...)
```

#### 机制 B：简洁封装，依赖 DataFrame

**优点**：
- 代码简洁 (`add_base()` 一行搞定)
- 原生支持多模型比较 (`summary_col`)
- DataFrame 易于后处理

**缺点**：
- 依赖 pandas
- 定制化需要操作 DataFrame
- 标记为 "Experimental"

```python
# 机制 B 风格：简洁封装
smry = summary2.Summary()
smry.add_base(results=self, alpha=alpha, float_format=float_format, ...)
smry.add_dict(diagnostic)
```

### 4.3 多模型比较：机制 B 的独家优势

**位置**: `summary2.py:470-626`

`summary_col()` 是机制 B 最有价值的功能，支持将多个模型结果并排展示：

```python
def summary_col(results, float_format="%.4f", model_names=(), stars=False,
                info_dict=None, regressor_order=(), ...):
    """
    Summarize multiple results instances side-by-side (coefs and SEs)
    
    示例输出:
    =================================================================
                      Model I         Model II         Model III
    -----------------------------------------------------------------
    const           1.234***       1.456***         1.678***
                  (0.123)         (0.134)           (0.145)
    x1              0.567**        0.678**
                  (0.234)         (0.245)
    x2                              0.890*           0.901*
                                    (0.456)          (0.467)
    -----------------------------------------------------------------
    R-squared       0.890           0.912            0.923
    No. Obs         100             100              100
    =================================================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01
    """
    # 1. 提取每个模型的参数和标准误
    cols = [_col_params(x, stars=stars, float_format=float_format,
                        include_r2=include_r2) for x in results]
    
    # 2. 合并为一个 DataFrame（自动对齐变量）
    def merg(x, y):
        return x.merge(y, how="outer", right_index=True, left_index=True)
    
    summ = reduce(merg, cols)
    
    # 3. 添加模型信息（N, R² 等）
    if info_dict:
        cols = [_col_info(x, info_dict.get(x.model.__class__.__name__,
                                           info_dict)) for x in results]
        info = reduce(merg, cols)
        # ... 合并到主表
    
    # 4. 封装为 Summary 对象
    smry = Summary()
    smry._merge_latex = True
    smry.add_df(summ, header=True, align="l")
    smry.add_text("Standard errors in parentheses.")
    if stars:
        smry.add_text("* p<.1, ** p<.05, ***p<.01")
    
    return smry
```

### 4.4 设计取舍决策树

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      汇报机制选择决策树                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  开始：需要生成结果汇报                                                       │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────┐                                       │
│  │ 需要比较多个模型结果？            │                                       │
│  └───────────────┬─────────────────┘                                       │
│                  │                                                          │
│         ┌────────┴────────┐                                                 │
│         │                 │                                                 │
│         ▼                 ▼                                                 │
│    ┌─────────┐       ┌─────────┐                                           │
│    │   是    │       │   否    │                                           │
│    └────┬────┘       └────┬────┘                                           │
│         │                 │                                                 │
│         ▼                 ▼                                                 │
│  机制 B: summary_col()   ┌─────────────────────────────┐                    │
│  (多模型并排比较)         │ 需要最大程度的布局定制？      │                    │
│                          └───────────────┬─────────────┘                    │
│                                          │                                  │
│                                 ┌────────┴────────┐                         │
│                                 │                 │                         │
│                                 ▼                 ▼                         │
│                            ┌─────────┐       ┌─────────┐                   │
│                            │   是    │       │   否    │                   │
│                            └────┬────┘       └────┬────┘                   │
│                                 │                 │                         │
│                                 ▼                 ▼                         │
│                           机制 A           ┌──────────────────┐            │
│                        (细粒度控制)        │ 可以接受 pandas 依赖?│            │
│                                        └─────────┬────────┘            │
│                                                  │                     │
│                                         ┌────────┴────────┐           │
│                                         │                 │           │
│                                         ▼                 ▼           │
│                                    ┌─────────┐       ┌─────────┐      │
│                                    │   是    │       │   否    │      │
│                                    └────┬────┘       └────┬────┘      │
│                                         │                 │           │
│                                         ▼                 ▼           │
│                                    机制 B            机制 A           │
│                                (简洁/实验性)       (稳定/不依赖pandas)  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 五、扩展点与自定义开发指南

### 5.1 机制 A 的扩展方式

#### 方式 1：在模型类中覆盖 `summary()` 方法

这是最常用的扩展方式，参考线性回归模型的实现：

```python
def summary(self, yname=None, xname=None, title=None, alpha=0.05, slim=False):
    # 1. 计算额外的统计量
    custom_stat = self._compute_custom_stat()
    
    # 2. 扩展顶部信息表
    top_left = [
        ("Dep. Variable:", None),
        ("Model:", [self.model.__class__.__name__]),
        ("Custom Stat:", ["%.4f" % custom_stat]),  # 添加自定义项
        # ...
    ]
    
    # 3. 创建 Summary 并添加表格
    from statsmodels.iolib.summary import Summary
    smry = Summary()
    smry.add_table_2cols(self, gleft=top_left, gright=top_right, ...)
    
    # 4. 完全自定义表格（使用 SimpleTable）
    from statsmodels.iolib.table import SimpleTable
    custom_data = [...]
    custom_header = [...]
    custom_stubs = [...]
    custom_table = SimpleTable(custom_data, custom_header, custom_stubs, ...)
    smry.tables.append(custom_table)  # 直接添加到 tables 列表
    
    return smry
```

#### 方式 2：使用 `add_extra_txt()` 添加警告/注释

```python
etext = []
if self.some_condition:
    etext.append("警告：这是一个自定义警告信息")
if self.another_condition:
    etext.append("注意：这个结果需要特别解释")

if etext:
    etext = [f"[{i + 1}] {text}" for i, text in enumerate(etext)]
    etext.insert(0, "自定义注释:")
    smry.add_extra_txt(etext)
```

### 5.2 机制 B 的扩展方式

#### 方式 1：使用 `add_df()` 添加任意 DataFrame

```python
def summary2(self, ...):
    from statsmodels.iolib import summary2
    smry = summary2.Summary()
    
    # 添加基础信息
    smry.add_base(results=self, ...)
    
    # 添加自定义 DataFrame
    import pandas as pd
    custom_df = pd.DataFrame({
        "Statistic": ["Custom A", "Custom B", "Custom C"],
        "Value": [self.stat_a, self.stat_b, self.stat_c]
    })
    smry.add_df(custom_df, index=False, header=True, float_format="%.4f")
    
    # 使用 add_dict() 添加键值对
    custom_dict = {
        "Custom Stat X:": "%.4f" % self.stat_x,
        "Custom Stat Y:": "%.4f" % self.stat_y,
    }
    smry.add_dict(custom_dict, ncols=2)
    
    return smry
```

#### 方式 2：自定义 `info_dict` 用于 `summary_col()`

```python
from statsmodels.iolib.summary2 import summary_col

# 定义自定义信息字典
info_dict = {
    "N": lambda x: x.nobs,
    "R2": lambda x: getattr(x, "rsquared", np.nan),
    "Adj. R2": lambda x: getattr(x, "rsquared_adj", np.nan),
    "F-stat": lambda x: getattr(x, "fvalue", np.nan),
    # 可以为特定模型定义不同的信息
    "OLS": {
        "R2": lambda x: x.rsquared,
        "Adj. R2": lambda x: x.rsquared_adj,
        "AIC": lambda x: x.aic,
    }
}

# 多模型比较
results = [model1.fit(), model2.fit(), model3.fit()]
smry = summary_col(
    results,
    model_names=["Model 1", "Model 2", "Model 3"],
    stars=True,  # 显示显著性星号
    info_dict=info_dict,
    float_format="%.4f",
)
print(smry)
```

### 5.3 扩展点对比

| 扩展需求 | 机制 A 方案 | 机制 B 方案 | 推荐 |
|---------|------------|------------|------|
| 添加自定义统计量行 | 扩展 `top_left`/`top_right` 列表 | `add_dict()` 或构造 DataFrame | 视场景而定 |
| 完全自定义表格 | 直接构造 `SimpleTable` 并 append | 构造 DataFrame 并 `add_df()` | 机制 B 更简单 |
| 多模型比较 | 需手动实现 | 直接使用 `summary_col()` | **机制 B** |
| 添加警告文本 | `add_extra_txt()` | `add_text()` | 相似 |
| 控制数字格式 | 格式化字符串嵌入列表 | `float_format` 参数 | 机制 B 更统一 |
| 不依赖 pandas | ✅ 原生支持 | ❌ 需要 pandas | **机制 A** |

---

## 六、关键代码位置索引

| 功能模块 | 文件路径 | 关键行号 |
|---------|---------|---------|
| **机制 A - Summary 类** | `statsmodels/iolib/summary.py` | 742-910 |
| **机制 A - add_table_2cols** | `statsmodels/iolib/summary.py` | 776-802 |
| **机制 A - add_table_params** | `statsmodels/iolib/summary.py` | 804-837 |
| **机制 A - summary_top** | `statsmodels/iolib/summary.py` | 273-386 |
| **机制 A - summary_params** | `statsmodels/iolib/summary.py` | 389-472 |
| **机制 B - Summary 类** | `statsmodels/iolib/summary2.py` | 16-247 |
| **机制 B - add_base** | `statsmodels/iolib/summary2.py` | 127-154 |
| **机制 B - summary_col** | `statsmodels/iolib/summary2.py` | 470-626 |
| **机制 B - summary_model** | `statsmodels/iolib/summary2.py` | 286-333 |
| **机制 B - summary_params** | `statsmodels/iolib/summary2.py` | 336-393 |
| **SimpleTable 类** | `statsmodels/iolib/table.py` | 124-926 |
| **核心统计量计算** | `statsmodels/base/model.py` | 1503-1541 |
| **线性回归 summary()** | `statsmodels/regression/linear_model.py` | 2822-3015 |
| **线性回归 summary2()** | `statsmodels/regression/linear_model.py` | 3017-3130 |
| **离散模型 summary()** | `statsmodels/discrete/discrete_model.py` | 5390-5469 |
| **状态空间 summary()** | `statsmodels/tsa/statespace/mlemodel.py` | 5266-5493 |

---

## 七、总结与建议

### 7.1 核心发现

1. **两套机制共享同一统计量计算层**：无论使用 `summary()` 还是 `summary2()`，底层的 `params`, `bse`, `tvalues`, `pvalues`, `llf`, `aic`, `bic` 等统计量都来自 `LikelihoodModelResults` 类的同一套实现。

2. **机制选择取决于使用场景**：
   - **单模型标准汇报**: 使用 `summary()` (机制 A)，稳定且不依赖 pandas
   - **多模型并排比较**: 必须使用 `summary_col()` (机制 B)
   - **需要 DataFrame 后处理**: 选择 `summary2()`

3. **统计量分流基于模型类型**：
   - 线性回归展示 `rsquared`, `fvalue` 等
   - 离散模型展示 `prsquared`, `llr_pvalue` 等
   - 状态空间模型展示 `hqic` 和诊断测试结果

4. **状态空间模型仅实现机制 A**：由于其复杂性（多内生变量、诊断测试的条件布局），状态空间模型没有实现 `summary2()` 方法。

### 7.2 使用建议

| 场景 | 推荐方法 | 原因 |
|-----|---------|------|
| 日常分析单模型 | `results.summary()` | 稳定、成熟、输出美观 |
| 论文表格准备 | `results.summary()` + 复制粘贴 | 标准格式被广泛接受 |
| 模型比较与选择 | `summary_col([...])` | 多模型并排对比，带显著性星号 |
| 自动化批处理 | `summary2()` 提取 DataFrame | 易于程序处理 |
| 自定义扩展 | 覆盖 `summary()` 方法 | 最大程度控制布局 |
| 避免 pandas 依赖 | 坚持使用机制 A | 无需额外依赖 |

### 7.3 架构评价

Statsmodels 的汇报机制设计体现了**实用主义**的设计哲学：

✅ **优点**:
- 统计量计算与展示分离，关注点分离
- 两套机制共享核心统计量，避免重复计算
- `summary_col()` 是非常实用的多模型比较功能
- 延迟计算 (`@cached_value`) 优化性能

⚠️ **可改进点**:
- 两套机制并存增加了学习成本
- `summary2()` 标记为 "Experimental" 但功能实用，状态不明确
- 状态空间模型没有实现 `summary2()`，造成不一致
- 机制 B 强依赖 pandas，在无 pandas 环境无法使用
