# Statsmodels 拟合结果汇报机制深度分析

## 一、整体架构概览

Statsmodels 的拟合结果汇报系统是一个分层设计的架构，从原始估计量到格式化输出表格经历了多个处理阶段。以下是核心模块的分布：

| 模块路径 | 主要职责 |
|---------|---------|
| `statsmodels/iolib/summary.py` | 摘要生成核心功能，定义 Summary 类和各种表格构建函数 |
| `statsmodels/iolib/table.py` | SimpleTable 类，负责表格的渲染和多种格式输出 |
| `statsmodels/base/model.py` | 模型和结果基类，定义统计指标的计算方法 |
| `statsmodels/regression/linear_model.py` | 线性回归模型的具体实现，包含定制化的 summary 方法 |
| `statsmodels/discrete/discrete_model.py` | 离散选择模型的结果汇报 |
| `statsmodels/tsa/statespace/mlemodel.py` | 时间序列状态空间模型的结果汇报 |

---

## 二、原始估计量的收集与处理流程

### 2.1 拟合过程中的参数估计

当调用 `model.fit()` 方法后，参数估计值会被收集到结果对象中。以线性回归为例：

**文件**: `statsmodels/regression/linear_model.py:396-414`

```python
if isinstance(self, OLS):
    lfit = OLSResults(
        self,
        beta,  # 参数估计值
        normalized_cov_params=self.normalized_cov_params,
        cov_type=cov_type,
        cov_kwds=cov_kwds,
        use_t=use_t,
    )
```

**核心数据流转**：
1. **原始参数估计** (`beta`): 通过最小二乘或极大似然估计得到的系数向量
2. **标准化协方差矩阵** (`normalized_cov_params`): $(X'X)^{-1}$ 或其等价形式
3. **尺度参数** (`scale`): 用于缩放协方差矩阵（对于线性回归是残差均方）

### 2.2 结果对象的初始化

**文件**: `statsmodels/base/model.py:1434-1464`

```python
def __init__(self, model, params, normalized_cov_params=None, scale=1.0, **kwargs):
    super().__init__(model, params)
    self.normalized_cov_params = normalized_cov_params
    self.scale = scale
    self._use_t = False
    
    # 处理协方差类型
    if "cov_type" in kwargs:
        cov_type = kwargs.get("cov_type", "nonrobust")
        if cov_type == "nonrobust":
            self.cov_type = "nonrobust"
            self.cov_kwds = {
                "description": "Standard Errors assume that the covariance matrix "
                               "of the errors is correctly specified."
            }
        else:
            # 使用稳健协方差估计
            from statsmodels.base.covtype import get_robustcov_results
            get_robustcov_results(
                self, cov_type=cov_type, use_self=True, use_t=use_t, **cov_kwds
            )
```

---

## 三、统计指标的计算与汇总机制

### 3.1 核心统计指标的延迟计算

Statsmodels 使用 `@cached_value` 装饰器实现统计指标的延迟计算和缓存：

**文件**: `statsmodels/base/model.py:1508-1541`

```python
@cached_value
def bse(self):
    """The standard errors of the parameter estimates."""
    # Issue 3299
    if (not hasattr(self, "cov_params_default")) and (
        self.normalized_cov_params is None
    ):
        bse_ = np.empty(len(self.params))
        bse_[:] = np.nan
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bse_ = np.sqrt(np.diag(self.cov_params()))
    return bse_

@cached_value
def tvalues(self):
    """Return the t-statistic for a given parameter estimate."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return self.params / self.bse

@cached_value
def pvalues(self):
    """The two-tailed p values for the t-stats of the params."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if self.use_t:
            df_resid = getattr(self, "df_resid_inference", self.df_resid)
            return stats.t.sf(np.abs(self.tvalues), df_resid) * 2
        else:
            return stats.norm.sf(np.abs(self.tvalues)) * 2
```

### 3.2 统计指标计算流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    统计指标计算流程图                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  params (参数估计值)                                                 │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────┐                                                 │
│  │  cov_params()   │◄── normalized_cov_params × scale              │
│  │  协方差矩阵计算   │                                                 │
│  └────────┬────────┘                                                 │
│           │                                                           │
│           ▼                                                           │
│  ┌─────────────────┐                                                 │
│  │     bse         │ = sqrt(diag(cov_params))                       │
│  │   标准误计算      │                                                 │
│  └────────┬────────┘                                                 │
│           │                                                           │
│           ▼                                                           │
│  ┌─────────────────┐     ┌─────────────────┐                        │
│  │   tvalues       │────►│    pvalues      │                        │
│  │ = params / bse  │     │ t分布或正态分布  │                        │
│  └─────────────────┘     └─────────────────┘                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 线性回归模型的扩展统计指标

**文件**: `statsmodels/regression/linear_model.py:2874-2953`

线性回归模型在 `summary()` 方法中计算了更多诊断统计量：

```python
def summary(self, yname=None, xname=None, title=None, alpha=0.05, slim=False):
    from statsmodels.stats.stattools import (
        durbin_watson,
        jarque_bera,
        omni_normtest,
    )
    
    # 正态性检验
    jb, jbpv, skew, kurtosis = jarque_bera(self.wresid)
    omni, omnipv = omni_normtest(self.wresid)
    
    # 多重共线性诊断
    eigvals = self.eigenvals
    condno = self.condition_number
    
    # 模型拟合指标（右上表格）
    top_right = [
        ("R-squared:", ["%#8.3f" % self.rsquared]),
        ("Adj. R-squared:", ["%#8.3f" % self.rsquared_adj]),
        ("F-statistic:", ["%#8.4g" % self.fvalue]),
        ("Prob (F-statistic):", ["%#6.3g" % self.f_pvalue]),
        ("Log-Likelihood:", None),
        ("AIC:", ["%#8.4g" % self.aic]),
        ("BIC:", ["%#8.4g" % self.bic]),
    ]
    
    # 残差诊断统计量
    diagn_left = [
        ("Omnibus:", ["%#6.3f" % omni]),
        ("Prob(Omnibus):", ["%#6.3f" % omnipv]),
        ("Skew:", ["%#6.3f" % skew]),
        ("Kurtosis:", ["%#6.3f" % kurtosis]),
    ]
    
    diagn_right = [
        ("Durbin-Watson:", ["%#8.3f" % durbin_watson(self.wresid)]),
        ("Jarque-Bera (JB):", ["%#8.3f" % jb]),
        ("Prob(JB):", ["%#8.3g" % jbpv]),
        ("Cond. No.", ["%#8.3g" % condno]),
    ]
```

---

## 四、表格构建与格式化输出实现

### 4.1 Summary 类：表格容器

**文件**: `statsmodels/iolib/summary.py:742-910`

`Summary` 类是一个容器类，用于组织多个 `SimpleTable` 实例：

```python
class Summary:
    """
    Result summary
    
    Attributes
    ----------
    tables : list of tables
        Contains the list of SimpleTable instances
    extra_txt : str
        extra lines that are added to the text output
    """
    def __init__(self):
        self.tables = []
        self.extra_txt = None
    
    def add_table_2cols(self, res, title=None, gleft=None, gright=None,
                        yname=None, xname=None):
        """添加双列表格（模型信息表通常使用这种格式）"""
        table = summary_top(res, title=title, gleft=gleft, gright=gright,
                            yname=yname, xname=xname)
        self.tables.append(table)
    
    def add_table_params(self, res, yname=None, xname=None, alpha=.05,
                         use_t=True):
        """添加参数估计表格"""
        if res.params.ndim == 1:
            table = summary_params(res, yname=yname, xname=xname, alpha=alpha,
                                   use_t=use_t)
        elif res.params.ndim == 2:
            _, table = summary_params_2dflat(res, endog_names=yname,
                                             exog_names=xname,
                                             alpha=alpha, use_t=use_t)
        self.tables.append(table)
    
    def add_extra_txt(self, etext):
        """添加额外的警告文本"""
        self.extra_txt = "\n".join(etext)
```

### 4.2 SimpleTable 类：表格渲染引擎

**文件**: `statsmodels/iolib/table.py:124-926`

`SimpleTable` 是核心的表格渲染类，支持多种输出格式：

```python
class SimpleTable(list):
    """
    Produce a simple ASCII, CSV, HTML, or LaTeX table from a
    rectangular (2d!) array of data.
    """
    
    def __init__(self, data, headers=None, stubs=None, title="",
                 datatypes=None, csv_fmt=None, txt_fmt=None, ltx_fmt=None,
                 html_fmt=None, celltype=None, rowtype=None, **fmt_dict):
        self.title = title
        # 初始化各种格式配置
        self._txt_fmt = default_txt_fmt.copy()
        self._latex_fmt = default_latex_fmt.copy()
        self._csv_fmt = default_csv_fmt.copy()
        self._html_fmt = default_html_fmt.copy()
        
        # 数据转换为行
        rows = self._data2rows(data)
        list.__init__(self, rows)
        self._add_headers_stubs(headers, stubs)
```

### 4.3 多格式输出机制

**文件**: `statsmodels/iolib/summary.py:850-910`

```python
def as_text(self):
    """返回文本格式"""
    txt = summary_return(self.tables, return_fmt="text")
    if self.extra_txt is not None:
        txt = txt + "\n\n" + self.extra_txt
    return txt

def as_latex(self):
    """返回 LaTeX 格式"""
    latex = summary_return(self.tables, return_fmt="latex")
    if self.extra_txt is not None:
        latex = latex + "\n\n" + self.extra_txt.replace("\n", " \\newline\n ")
    return latex

def as_csv(self):
    """返回 CSV 格式"""
    csv = summary_return(self.tables, return_fmt="csv")
    if self.extra_txt is not None:
        csv = csv + "\n\n" + self.extra_txt
    return csv

def as_html(self):
    """返回 HTML 格式"""
    html = summary_return(self.tables, return_fmt="html")
    if self.extra_txt is not None:
        html = html + "<br/><br/>" + self.extra_txt.replace("\n", "<br/>")
    return html
```

### 4.4 表格格式化函数

**文件**: `statsmodels/iolib/summary.py:389-472`

参数表格的构建函数：

```python
def summary_params(results, yname=None, xname=None, alpha=.05, use_t=True,
                   skip_header=False, title=None):
    '''create a summary table for the parameters'''
    
    # 从结果对象提取统计指标
    params = np.asarray(results.params)
    std_err = np.asarray(results.bse)
    tvalues = np.asarray(results.tvalues)
    pvalues = np.asarray(results.pvalues)
    conf_int = np.asarray(results.conf_int(alpha))
    
    # 根据使用 t 分布还是正态分布设置表头
    if use_t:
        param_header = ["coef", "std err", "t", "P>|t|",
                        "[" + str(alpha/2), str(1-alpha/2) + "]"]
    else:
        param_header = ["coef", "std err", "z", "P>|z|",
                        "[" + str(alpha/2), str(1-alpha/2) + "]"]
    
    # 获取变量名
    _, xname = _getnames(results, yname=yname, xname=xname)
    
    # 格式化数据
    params_data = lzip([forg(params[i], prec=4) for i in exog_idx],
                       [forg(std_err[i]) for i in exog_idx],
                       [forg(tvalues[i]) for i in exog_idx],
                       ["%#6.3f" % (pvalues[i]) for i in exog_idx],
                       [forg(conf_int[i, 0]) for i in exog_idx],
                       [forg(conf_int[i, 1]) for i in exog_idx])
    
    # 创建 SimpleTable
    parameter_table = SimpleTable(params_data,
                                  param_header,
                                  params_stubs,
                                  title=title,
                                  txt_fmt=fmt_params)
    
    return parameter_table
```

---

## 五、完整流程示例

### 5.1 以 OLS 回归为例的完整流程

当执行以下代码时：

```python
import statsmodels.api as sm
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
```

**完整的处理流程如下**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OLS 拟合结果汇报完整流程图                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 模型拟合阶段                                                              │
│  ─────────────────                                                           │
│  model.fit()                                                                 │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────┐                                             │
│  │  计算 beta = (X'X)⁻¹X'y    │                                             │
│  │  计算 normalized_cov_params │                                             │
│  │  创建 OLSResults 对象       │                                             │
│  └───────────────┬─────────────┘                                             │
│                  │                                                             │
│                  ▼                                                             │
│                                                                             │
│  2. 统计指标延迟计算阶段                                                       │
│  ────────────────────────                                                    │
│  访问 results.params    ──► 直接返回拟合的系数向量                           │
│  访问 results.bse       ──► 计算 sqrt(diag(cov_params()))                   │
│  访问 results.tvalues   ──► 计算 params / bse                               │
│  访问 results.pvalues   ──► 根据 t 分布计算双侧 p 值                         │
│  访问 results.conf_int()──► 计算置信区间                                     │
│                                                                             │
│                  │                                                             │
│                  ▼                                                             │
│                                                                             │
│  3. Summary 构建阶段                                                           │
│  ────────────────────                                                         │
│  results.summary()                                                            │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  3.1 计算诊断统计量                                                    │   │
│  │      - Jarque-Bera 正态性检验                                         │   │
│  │      - Omnibus 检验                                                   │   │
│  │      - Durbin-Watson 自相关检验                                       │   │
│  │      - 条件数、特征值（多重共线性诊断）                                 │   │
│  └───────────────────────────┬─────────────────────────────────────────┘   │
│                              │                                                 │
│                              ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  3.2 构建三个表格                                                      │   │
│  │                                                                     │   │
│  │  表格1: 模型信息表（双列布局）                                          │   │
│  │  ┌─────────────────────┬─────────────────────┐                      │   │
│  │  │ Dep. Variable:  y   │ R-squared:      0.95│                      │   │
│  │  │ Model:          OLS │ Adj. R-squared: 0.94│                      │   │
│  │  │ Method:   Least Sq. │ F-statistic:   123.4│                      │   │
│  │  │ Date:     ...       │ Prob (F-stat): 0.000│                      │   │
│  │  └─────────────────────┴─────────────────────┘                      │   │
│  │                                                                     │   │
│  │  表格2: 参数估计表                                                    │   │
│  │  ┌─────────────────────────────────────────────────┐                │   │
│  │  │         coef   std err    t  P>|t|  [0.025  0.975]            │   │
│  │  │ const    1.234    0.123  10.03  0.000   0.987   1.481        │   │
│  │  │ x1       0.567    0.045  12.60  0.000   0.476   0.658        │   │
│  │  └─────────────────────────────────────────────────┘                │   │
│  │                                                                     │   │
│  │  表格3: 诊断统计表（双列布局）                                          │   │
│  │  ┌─────────────────────┬─────────────────────┐                      │   │
│  │  │ Omnibus:       2.34 │ Durbin-Watson:  1.98│                      │   │
│  │  │ Prob(Omnibus): 0.31 │ Jarque-Bera (JB): 1.23│                      │   │
│  │  │ Skew:          0.12 │ Prob(JB):       0.54│                      │   │
│  │  │ Kurtosis:      2.89 │ Cond. No.      12.3│                      │   │
│  │  └─────────────────────┴─────────────────────┘                      │   │
│  └───────────────────────────┬─────────────────────────────────────────┘   │
│                              │                                                 │
│                              ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  3.3 添加警告文本                                                      │   │
│  │      - 无常数项时的 R² 警告                                            │   │
│  │      - 协方差类型说明                                                   │   │
│  │      - 多重共线性警告（条件数过大）                                     │   │
│  └───────────────────────────┬─────────────────────────────────────────┘   │
│                              │                                                 │
│                              ▼                                                 │
│                                                                             │
│  4. 输出格式化阶段                                                            │
│  ────────────────────                                                         │
│  print(summary) 调用 Summary.__str__()                                       │
│       │                                                                      │
│       ▼                                                                      │
│  Summary.as_text()                                                           │
│       │                                                                      │
│       ▼                                                                      │
│  summary_return(tables, return_fmt="text")                                   │
│       │                                                                      │
│       ▼                                                                      │
│  每个 SimpleTable.as_text() 被调用                                           │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  SimpleTable.as_text() 渲染流程:                                      │   │
│  │  1. 计算每列宽度                                                       │   │
│  │  2. 格式化每个单元格（应用 data_fmts）                                  │   │
│  │  3. 添加表格装饰线（table_dec_above, table_dec_below）                │   │
│  │  4. 添加标题                                                           │   │
│  │  5. 合并所有行为字符串                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 关键文件与行号参考

| 功能模块 | 文件路径 | 关键行号 |
|---------|---------|---------|
| Summary 类定义 | `statsmodels/iolib/summary.py` | 742-910 |
| 参数表格构建 | `statsmodels/iolib/summary.py` | 389-472 |
| 顶部信息表格 | `statsmodels/iolib/summary.py` | 273-386 |
| SimpleTable 类 | `statsmodels/iolib/table.py` | 124-926 |
| OLS summary 方法 | `statsmodels/regression/linear_model.py` | 2822-3015 |
| LikelihoodModelResults | `statsmodels/base/model.py` | 1276-1700 |
| 统计指标计算 | `statsmodels/base/model.py` | 1503-1541 |

---

## 六、设计亮点与扩展机制

### 6.1 延迟计算与缓存

使用 `@cached_value` 装饰器实现了统计指标的按需计算和缓存，避免了重复计算：

```python
@cached_value
def bse(self):
    # 只在第一次访问时计算
    return np.sqrt(np.diag(self.cov_params()))
```

### 6.2 多态设计

不同模型可以覆盖 `summary()` 方法来定制输出：

- **线性回归**: 包含 R²、F 统计量、Durbin-Watson 等
- **Logit/Probit**: 包含 Pseudo R²、LR 统计量等
- **时间序列模型**: 包含 ARIMA 阶数、信息准则等

### 6.3 多种输出格式

通过 `SimpleTable` 的多格式渲染能力，同一数据可以输出为：
- **Text**: 控制台显示
- **HTML**: Jupyter Notebook 显示 (`_repr_html_`)
- **LaTeX**: 学术论文排版 (`_repr_latex_`)
- **CSV**: 数据导出

### 6.4 模块化架构

整个系统采用了清晰的分层架构：

```
┌──────────────────────────────────────────────────────┐
│                    用户层                              │
│         print(results.summary())                      │
└───────────────────────┬──────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────┐
│                    Summary 层                         │
│         表格容器 + 多格式输出方法                      │
└───────────────────────┬──────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────┐
│                   SimpleTable 层                      │
│         表格渲染 + 格式配置 + 单元格格式化             │
└───────────────────────┬──────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────┐
│                 统计指标计算层                         │
│    Results 类中的 bse, tvalues, pvalues 等           │
└───────────────────────┬──────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────┐
│                    数据层                              │
│    params, normalized_cov_params, scale               │
└──────────────────────────────────────────────────────┘
```

---

## 七、总结

Statsmodels 的拟合结果汇报系统是一个设计精良的分层架构：

1. **数据层**: 存储原始估计值 (`params`) 和协方差矩阵 (`normalized_cov_params`)
2. **计算层**: 通过 `@cached_value` 实现统计指标的延迟计算
3. **表格层**: `SimpleTable` 提供灵活的表格渲染能力
4. **容器层**: `Summary` 组织多个表格并提供多格式输出接口
5. **定制层**: 各模型通过覆盖 `summary()` 方法实现定制化输出

这种设计使得系统既具有统一的接口，又允许各模型根据自身特点定制输出内容，同时支持多种输出格式，满足了从探索性分析到学术出版的各种需求。
