# Statsmodels 拟合结果汇报：路径纠正、场景分支与机制对比深度分析

## 一、路径触发映射纠正（关键修正）

### 1.1 之前的错误判断

在上一轮分析中，存在一个关键的不准确判断：

❌ **错误映射**：
> "Jupyter Notebook 显示 -> 机制 A"

这个判断是错误的。两套机制都支持 Jupyter 显示，选择哪个机制完全取决于用户的显式调用。

### 1.2 正确的路径触发映射

#### 核心发现：两套机制都实现了 Jupyter 显示方法

**验证证据**：

| 文件 | `_repr_html_` 位置 | `_repr_latex_` 位置 |
|-----|-------------------|--------------------|
| `summary.py` (机制 A) | 第 768-770 行 | 第 772-774 行 |
| `summary2.py` (机制 B) | 第 30-32 行 | 第 34-36 行 |

**代码证据**：

```python
# summary.py (机制 A) - 第 768-774 行
def _repr_html_(self):
    """Display as HTML in IPython notebook."""
    return self.as_html()

def _repr_latex_(self):
    """Display as LaTeX when converting IPython notebook to PDF."""
    return self.as_latex()
```

```python
# summary2.py (机制 B) - 第 30-36 行
def _repr_html_(self):
    """Display as HTML in IPython notebook."""
    return self.as_html()

def _repr_latex_(self):
    """Display as LaTeX when converting IPython notebook to PDF."""
    return self.as_latex()
```

#### 正确的触发映射表

| 用户操作 | 触发的方法 | 选择的机制 | 关键代码位置 |
|---------|-----------|-----------|-------------|
| `results.summary()` | 用户显式调用 | **机制 A** | 各模型类的 `summary()` 方法 |
| `results.summary2()` | 用户显式调用 | **机制 B** | 部分模型类的 `summary2()` 方法 |
| `print(results.summary())` | 调用 `__str__()` → `as_text()` | 取决于调用哪个 `summary()` | `Summary.__str__()` |
| Jupyter 单元格返回 `results.summary()` | 调用 `_repr_html_()` | 取决于调用哪个 `summary()` | 两套 `Summary` 类都实现 |
| Jupyter 单元格返回 `results.summary2()` | 调用 `_repr_html_()` | **机制 B** | `summary2.py` 的 `Summary` 类 |
| `summary_col([...])` | 直接调用函数 | **机制 B** | `summary2.py:470-626` |

#### 路径选择决策树（修正版）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      路径选择决策树（修正版）                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  用户需要生成结果汇报                                                         │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────┐                                       │
│  │ 需要并排比较多个模型？            │                                       │
│  │ (论文表格、模型选择等场景)         │                                       │
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
│                           └───────────────┬─────────────┘                    │
│                                           │                                  │
│                                  ┌────────┴────────┐                         │
│                                  │                 │                         │
│                                  ▼                 ▼                         │
│                             ┌─────────┐       ┌─────────┐                   │
│                             │   是    │       │   否    │                   │
│                             └────┬────┘       └────┬────┘                   │
│                                  │                 │                         │
│                                  ▼                 ▼                         │
│                             机制 A           ┌──────────────────┐            │
│                          (细粒度控制)        │ 是否需要 DataFrame │            │
│                                          │ 做后续程序化处理？     │            │
│                                          └─────────┬────────┘            │
│                                                    │                     │
│                                           ┌────────┴────────┐           │
│                                           │                 │           │
│                                           ▼                 ▼           │
│                                      ┌─────────┐       ┌─────────┐      │
│                                      │   是    │       │   否    │      │
│                                      └────┬────┘       └────┬────┘      │
│                                           │                 │           │
│                                           ▼                 ▼           │
│                                      机制 B            机制 A           │
│                                 (简洁/实验性)       (稳定/不依赖pandas)  │
│                                                                             │
│  关键修正：Jupyter 显示 (_repr_html_) 取决于用户调用的是 summary() 还是    │
│           summary2()，两套机制都原生支持 Jupyter 显示！                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、关键场景分支对照清单

### 2.1 场景总览

| 场景编号 | 场景名称 | 触发条件 | 检测方式 | 机制 A 支持 | 机制 B 支持 |
|---------|---------|---------|---------|------------|------------|
| S1 | 标准单模型汇报 | 默认 | 无 | ✅ | ⚠️ 部分模型 |
| S2 | 精简输出 | `slim=True` | `if slim:` | ✅ | ❌ |
| S3 | 约束估计 | `fit_constrained()` | `hasattr(self, "constraints")` | ✅ | ⚠️ 需手动 |
| S4 | 多方程/多因变量 | `params.ndim == 2` | `if res.params.ndim == 2:` | ✅ | ⚠️ 需循环 |
| S5 | 稳健协方差估计 | `cov_type != "nonrobust"` | `hasattr(self, "cov_type")` | ✅ | ⚠️ 部分 |
| S6 | 无常数项模型 | `k_constant == 0/False` | `if not self.k_constant:` | ✅ | ✅ |
| S7 | 多重共线性/奇异矩阵 | 特征值极小或条件数过大 | `eigvals[-1] < 1e-10` 或 `condno > 1000` | ✅ | ❌ |
| S8 | 多模型比较 | 调用 `summary_col()` | 直接调用 | ❌ | ✅ |

### 2.2 各场景详细分析

#### 场景 S1：标准单模型汇报

**触发条件**：默认调用 `results.summary()` 或 `results.summary2()`

**机制 A 实现** (`linear_model.py:2822-3015`)：

```python
def summary(self, yname=None, xname=None, title=None, alpha=0.05, slim=False):
    # 1. 计算所有诊断统计量
    jb, jbpv, skew, kurtosis = jarque_bera(self.wresid)
    omni, omnipv = omni_normtest(self.wresid)
    eigvals = self.eigenvals
    condno = self.condition_number
    
    # 2. 显式构建顶部信息表（双列布局）
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
    
    # 3. 显式构建诊断统计表
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
    
    # 4. 创建 Summary 并添加所有表格
    from statsmodels.iolib.summary import Summary
    smry = Summary()
    smry.add_table_2cols(self, gleft=top_left, gright=top_right, ...)  # 表1：模型信息
    smry.add_table_params(self, ...)  # 表2：参数估计
    smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right, ...)  # 表3：诊断统计
    
    # 5. 添加警告文本
    etext = [...]
    smry.add_extra_txt(etext)
    
    return smry
```

**机制 B 实现** (`linear_model.py:3017-3130`)：

```python
def summary2(self, yname=None, xname=None, title=None, alpha=0.05, float_format="%.4f"):
    # 1. 计算相同的诊断统计量（复用统计量计算层）
    jb, jbpv, skew, kurtosis = jarque_bera(self.wresid)
    omni, omnipv = omni_normtest(self.wresid)
    dw = durbin_watson(self.wresid)
    eigvals = self.eigenvals
    condno = self.condition_number
    
    # 2. 将诊断统计量打包为字典
    diagnostic = {
        "Omnibus:": "%.3f" % omni,
        "Prob(Omnibus):": "%.3f" % omnipv,
        "Skew:": "%.3f" % skew,
        "Kurtosis:": "%.3f" % kurtosis,
        "Durbin-Watson:": "%.3f" % dw,
        "Jarque-Bera (JB):": "%.3f" % jb,
        "Prob(JB):": "%.3f" % jbpv,
        "Condition No.:": "%.0f" % condno,
    }
    
    # 3. 使用 summary2 机制（简洁封装）
    from statsmodels.iolib import summary2
    smry = summary2.Summary()
    
    # 4. 一行代码添加基础信息和参数表（约定优于配置）
    smry.add_base(results=self, alpha=alpha, float_format=float_format, ...)
    
    # 5. 添加诊断信息字典
    smry.add_dict(diagnostic)
    
    # 6. 添加警告文本
    for line in etext:
        smry.add_text(line)
    
    return smry
```

**输出差异对比**：

| 输出方面 | 机制 A | 机制 B |
|---------|--------|--------|
| 表格数量 | 3 个表格（模型信息、参数、诊断） | 2 个部分（基础表、诊断字典表） |
| 布局风格 | 紧凑双列布局 | 更宽松的单列布局 |
| 标题位置 | 每个表格有独立标题 | 统一标题在顶部 |
| 参数表格式 | "coef", "std err", "t", "P>|t|" | "Coef.", "Std.Err.", "t", "P>|t|" |
| 诊断表格式 | 双列网格布局 | 键值对列表 |

---

#### 场景 S2：精简输出 (slim=True)

**触发条件**：`results.summary(slim=True)`

**代码位置**：`linear_model.py:2925-2939`

**检测逻辑**：

```python
# 定义精简输出时保留的统计量列表
slimlist = [
    "Dep. Variable:",
    "Model:",
    "No. Observations:",
    "Covariance Type:",
    "R-squared:",
    "Adj. R-squared:",
    "F-statistic:",
    "Prob (F-statistic):",
]

if slim:
    # 1. 清空诊断统计表
    diagn_left = diagn_right = []
    
    # 2. 过滤顶部信息表，只保留 slimlist 中的项
    top_left = [elem for elem in top_left if elem[0] in slimlist]
    top_right = [elem for elem in top_right if elem[0] in slimlist]
    
    # 3. 对齐左右两表的行数
    top_right = top_right + [("", [])] * (len(top_left) - len(top_right))
else:
    # 标准输出：显示所有诊断统计
    diagn_left = [("Omnibus:", ...), ("Prob(Omnibus):", ...), ...]
    diagn_right = [("Durbin-Watson:", ...), ("Jarque-Bera (JB):", ...), ...]

# ... 添加参数表（始终显示）
smry.add_table_params(...)

# 4. 诊断表只在 slim=False 时添加
if not slim:
    smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right, ...)
```

**输出差异对比**：

| 输出内容 | `slim=False` (默认) | `slim=True` (精简) |
|---------|---------------------|-------------------|
| 模型信息表 | 完整（Date, Time, Method 等） | 精简（仅核心统计量） |
| 参数估计表 | ✅ 完整显示 | ✅ 完整显示 |
| 诊断统计表 | ✅ 显示 Omnibus, Durbin-Watson, JB, Cond. No. 等 | ❌ 不显示 |
| 警告文本 | ✅ 显示 | ❌ 不显示 |

**机制 B 支持情况**：❌ **不支持**

机制 B 的 `summary2()` 方法没有 `slim` 参数，无法实现精简输出。这是机制 A 的独家功能。

---

#### 场景 S3：约束估计

**触发条件**：使用 `model.fit_constrained(constraints, ...)` 而非标准 `model.fit()`

**核心函数**：`statsmodels/base/_constraints.py:254` 的 `fit_constrained` 方法

**检测方式**：`hasattr(self, "constraints")`

**机制 A 实现**（多个模型统一模式）：

```python
# GLM: generalized_linear_model.py:2786-2789
if hasattr(self, "constraints"):
    smry.add_extra_txt(
        ["Model has been estimated subject to linear equality constraints."]
    )

# 离散模型: discrete_model.py:5464-5467
if hasattr(self, "constraints"):
    smry.add_extra_txt(
        ["Model has been estimated subject to linear equality constraints."]
    )

# 多元模型: multivariate_ols.py:743-746
if hasattr(self, "constraints"):
    smry.add_extra_txt(
        ["Model has been estimated subject to linear equality constraints."]
    )
```

**机制 B 实现**：

```python
# GLM: generalized_linear_model.py:2836-2839
if hasattr(self, "constraints"):
    smry.add_text(
        "Model has been estimated subject to linear equality constraints."
    )

# 离散模型: discrete_model.py:5514-5516
if hasattr(self, "constraints"):
    smry.add_text(
        "Model has been estimated subject to linear equality constraints."
    )
```

**输出差异**：

| 机制 | 添加方式 | 输出位置 | 文本格式 |
|-----|---------|---------|---------|
| 机制 A | `add_extra_txt([...])` | `extra_txt` 属性，文本输出时显示 | 列表形式，自动编号 |
| 机制 B | `add_text(...)` | `extra_txt` 列表 | 单独调用，需手动循环 |

**测试验证** (`discrete/tests/test_constrained.py:172-173`)：

```python
summ = self.res1m.summary()
assert ("linear equality constraints" in summ.extra_txt)
```

---

#### 场景 S4：多方程/多因变量模型

**触发条件**：`params.ndim == 2` (参数数组是二维的)

**典型模型**：
- **MNLogit** (多分类 Logit): 因变量有 K 个类别，参数 shape = (n_exog, K-1)
- **SUR** (似不相关回归): 多个方程联合估计
- **VAR/VECM**: 向量自回归模型

**检测位置**：`summary.py:828-834` 的 `add_table_params()` 方法

```python
def add_table_params(self, res, yname=None, xname=None, alpha=.05, use_t=True):
    # 根据 params 的维度自动选择处理方式
    if res.params.ndim == 1:
        # 单方程模型：标准参数表
        table = summary_params(res, yname=yname, xname=xname, alpha=alpha,
                               use_t=use_t)
    elif res.params.ndim == 2:
        # 多方程模型：展开为多个子表格
        _, table = summary_params_2dflat(res, endog_names=yname,
                                         exog_names=xname,
                                         alpha=alpha, use_t=use_t)
    else:
        raise ValueError("params has to be 1d or 2d")
    
    self.tables.append(table)
```

**多方程处理核心**：`summary.py:596-669` 的 `summary_params_2dflat()` 函数

```python
def summary_params_2dflat(result, endog_names=None, exog_names=None, alpha=0.05,
                          use_t=True, keep_headers=True, endog_cols=False):
    """
    处理多方程模型的参数表
    
    例如 MNLogit：
    - params.shape = (n_exog, n_classes-1)
    - 每一列对应一个类别（方程）
    """
    
    res = result
    params = res.params
    
    # 检测是否为多方程
    if params.ndim == 2:
        n_equ = params.shape[1]  # 方程数量 = 列数
    else:
        n_equ = 1
    
    # 获取方程名称（如 MNLogit 的类别标签）
    if not isinstance(endog_names, list):
        # MNLogit 特殊处理：endog_names[1:] 排除参照类别
        endog_names = res.model.endog_names[1:]
    
    tables = []
    for eq in range(n_equ):
        # 为每个方程提取单独的统计量
        restup = (
            res,
            res.params[:, eq],      # 第 eq 个方程的系数
            res.bse[:, eq],         # 第 eq 个方程的标准误
            res.tvalues[:, eq],     # 第 eq 个方程的 t 值
            res.pvalues[:, eq],     # 第 eq 个方程的 p 值
            res.conf_int(alpha)[eq]  # 第 eq 个方程的置信区间
        )
        
        # 为每个方程创建单独的参数表
        tble = summary_params(restup, yname=endog_names[eq], ...)
        tables.append(tble)
    
    # 将每个方程的标题移到表头
    for i in range(len(endog_names)):
        tables[i].title = endog_names[i]
    
    # 合并所有子表格为一个大表格
    table_all = table_extend(tables, keep_headers=keep_headers)
    
    return tables, table_all
```

**表格合并核心**：`summary.py:672-714` 的 `table_extend()` 函数

```python
def table_extend(tables, keep_headers=True):
    """
    合并多个 SimpleTable，将标题移到表头
    
    例如 MNLogit 的输出：
    ─────────────────────────────────────
           y=1       coef   std err   ...
    ─────────────────────────────────────
    const      0.123    0.045    ...
    x1         0.456    0.078    ...
    ─────────────────────────────────────
           y=2       coef   std err   ...
    ─────────────────────────────────────
    const      0.789    0.056    ...
    x1         0.234    0.067    ...
    ─────────────────────────────────────
    """
    for ii, t in enumerate(tables[:]):
        # 将标题移动到第一个表头单元格
        if t[0].datatype == "header":
            t[0][0].data = t.title  # 如 "y=1"
            
            # 非第一个表格可以选择隐藏重复表头
            if not keep_headers and (ii > 0):
                for c in t[0][1:]:
                    c.data = ""
        
        # 添加分隔线并合并
        if ii == 0:
            table_all = t
        else:
            r1 = table_all[-1]
            r1.add_format("txt", row_dec_below="-")  # 表格间的分隔线
            table_all.extend(t)
    
    return table_all
```

**机制 B 支持情况**：⚠️ **部分支持，需手动处理**

机制 B 的 `summary2.py` 中的 `summary_params()` 函数也支持 tuple 形式的输入（用于多方程），但没有提供自动的 `summary_params_2dflat` 等价功能。

```python
# summary2.py:365-372
def summary_params(results, ...):
    if isinstance(results, tuple):
        # 支持传入元组形式的单个方程数据
        results, params, bse, tvalues, pvalues, conf_int = results
    else:
        # 从 results 对象提取
        params = results.params
        # ...
```

机制 B 的 `summary2()` 方法在处理 MNLogit 等多方程模型时，需要**手动循环**处理每个方程：

```python
# discrete_model.py:6082-6128 (MNLogitResults.summary2)
def summary2(self, alpha=0.05, float_format="%.4f"):
    from statsmodels.iolib import summary2
    smry = summary2.Summary()
    smry.add_dict(summary2.summary_model(self))
    
    # 手动循环处理每个方程
    eqn = self.params.shape[1]
    confint = self.conf_int(alpha)
    for i in range(eqn):
        # 手动构建每个方程的 tuple
        coefs = summary2.summary_params(
            (
                self,
                self.params[:, i],
                self.bse[:, i],
                self.tvalues[:, i],
                self.pvalues[:, i],
                confint[i],
            ),
            alpha=alpha,
        )
        # 手动添加方程标识
        level_str = self.model.endog_names + " = " + str(i)
        coefs[level_str] = coefs.index
        coefs = coefs.iloc[:, [-1, 0, 1, 2, 3, 4, 5]]
        
        smry.add_df(coefs, index=False, header=True, float_format=float_format)
        smry.add_title(results=self)
    
    return smry
```

**多方程场景对比总结**：

| 方面 | 机制 A | 机制 B |
|-----|--------|--------|
| 自动检测 | ✅ `add_table_params()` 自动检测 `params.ndim` | ⚠️ 需手动处理 |
| 专门函数 | ✅ `summary_params_2dflat()` + `table_extend()` | ❌ 无等价函数 |
| 表格合并 | ✅ 自动合并为带分隔线的单个表格 | ⚠️ 每个方程独立 DataFrame |
| 代码量 | 简洁（自动处理） | 冗长（需手动循环） |
| MNLogit 实现 | ✅ 一行 `add_table_params()` | ⚠️ 15+ 行手动循环 |

---

#### 场景 S5：稳健协方差估计

**触发条件**：使用非默认协方差类型拟合

```python
# 示例
results = model.fit(cov_type="HC0")           # 异方差稳健
results = model.fit(cov_type="HAC", cov_kwds={"maxlags": 3})  # 异方差自相关稳健
results = model.fit(cov_type="cluster", cov_kwds={"groups": groups})  # 聚类稳健
```

**检测方式**：
1. `hasattr(self, "cov_type")` - 检查是否有非默认协方差类型
2. `self.cov_type != "nonrobust"` - 具体检查类型

**机制 A 的输出处理**（线性回归为例）：

```python
# linear_model.py:2899-2902 (顶部信息表)
if hasattr(self, "cov_type"):
    # 在顶部信息表中显示协方差类型
    top_left.append(("Covariance Type:", [self.cov_type]))

# linear_model.py:2990-2991 (警告文本)
if hasattr(self, "cov_type"):
    # 在警告文本中显示详细说明
    etext.append(self.cov_kwds["description"])
```

**典型的 `cov_kwds["description"]` 内容**：

| `cov_type` | 描述文本 |
|-----------|---------|
| `"nonrobust"` | "Standard Errors assume that the covariance matrix of the errors is correctly specified." |
| `"HC0"`-`"HC3"` | "Standard Errors are heteroskedasticity-robust" |
| `"HAC"` | "Standard Errors are heteroskedasticity and autocorrelation consistent (HAC)" |
| `"cluster"` | "Standard Errors are robust to cluster correlation" |

**输出示例**（机制 A）：

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.982
Model:                            OLS   Adj. R-squared:                  0.980
Method:                 Least Squares   F-statistic:                     587.6
Date:                Wed, 01 May 2026   Prob (F-statistic):           1.28e-14
Time:                        15:30:00   Log-Likelihood:                -124.38
No. Observations:                  24   AIC:                             254.8
Df Residuals:                      21   BIC:                             258.3
Df Model:                           2                                         
Covariance Type:                  HC0   <-- 协方差类型显示在顶部信息表
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          1.2345      0.123     10.037      0.000       0.978       1.491
x1             0.5678      0.045     12.618      0.000       0.474       0.662
x2             0.8901      0.067     13.285      0.000       0.751       1.029
==============================================================================
Omnibus:                        2.345   Durbin-Watson:                   1.987
Prob(Omnibus):                  0.310   Jarque-Bera (JB):                1.234
Skew:                           0.123   Prob(JB):                        0.540
Kurtosis:                       2.890   Cond. No.                         12.3
==============================================================================

Notes:
[1] Standard Errors are heteroskedasticity-robust   <-- 警告文本中的详细说明
```

**机制 B 支持情况**：

机制 B 的 `summary2()` 方法通过 `add_base()` 调用 `summary_model()` 函数，该函数会尝试提取协方差类型：

```python
# summary2.py:309
info["Cov. Type:"] = lambda x: x.fit_options["cov"]
```

但这个实现依赖于 `fit_options` 属性，不是所有模型都一致设置。机制 B 的 `add_text()` 方法同样可以添加警告，但需要模型自己实现。

**稳健协方差场景对比**：

| 方面 | 机制 A | 机制 B |
|-----|--------|--------|
| 顶部显示协方差类型 | ✅ 所有实现模型都支持 | ⚠️ 依赖 `fit_options`，不一致 |
| 警告文本详细说明 | ✅ 通过 `cov_kwds["description"]` | ⚠️ 需手动添加 |
| 实现一致性 | ✅ 所有模型遵循相同模式 | ⚠️ 不同模型实现不一致 |

---

#### 场景 S6：无常数项模型

**触发条件**：模型不包含截距项

**检测方式**：`if not self.k_constant:`

**机制 A 实现** (`linear_model.py:2985-2989`)：

```python
if not self.k_constant:
    etext.append(
        "R² is computed without centering (uncentered) since the "
        "model does not contain a constant."
    )
```

**机制 B 实现** (`linear_model.py:3094-3098`)：

```python
if not self.k_constant:
    etext.append(
        "R² is computed without centering (uncentered) since the \
        model does not contain a constant."
    )
```

两套机制在这个场景下的实现基本一致，都是添加警告文本说明 R² 的计算方式不同。

---

#### 场景 S7：多重共线性/奇异矩阵诊断

**触发条件**：
1. **条件 1**：最小特征值极小 `eigvals[-1] < 1e-10`
2. **条件 2**：条件数过大 `condno > 1000`

**代码位置**：`linear_model.py:3005-3021`

```python
# 计算特征值和条件数
eigvals = self.eigenvals
condno = self.condition_number

# 检查多重共线性
if eigvals[-1] < 1e-10:
    warn = (
        "The smallest eigenvalue is %6.3g. This might indicate "
        "that there are\n"
        "strong multicollinearity problems or that the design "
        "matrix is singular."
        % eigvals[-1]
    )
    etext.append(warn)
elif condno > 1000:
    warn = (
        "The condition number is large, %6.3g. This might indicate "
        "that there are strong multicollinearity or other numerical "
        "problems."
        % condno
    )
    etext.append(warn)
```

**输出示例**：

```
Notes:
[1] The smallest eigenvalue is 2.35e-11. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
```

**机制 B 支持情况**：❌ **不支持**

机制 B 的 `summary2()` 方法虽然也计算了 `eigvals` 和 `condno`，但**没有添加多重共线性警告**的逻辑：

```python
# linear_model.py:3065-3066 (summary2 中)
eigvals = self.eigenvals
condno = self.condition_number
# ... 但没有后续的 if eigvals[-1] < 1e-10 检查！
```

这是机制 A 的独家功能，机制 B 缺失这个重要的诊断警告。

---

#### 场景 S8：多模型比较

**触发条件**：调用 `summary_col([results1, results2, results3, ...])`

**专属机制**：**机制 B** (`summary2.py:470-626`)

**核心功能**：将多个模型的结果并排展示，便于比较

```python
from statsmodels.iolib.summary2 import summary_col

# 多模型比较
results1 = model1.fit()
results2 = model2.fit()
results3 = model3.fit()

# 生成对比表格
comparison = summary_col(
    [results1, results2, results3],
    model_names=["Model 1", "Model 2", "Model 3"],
    stars=True,  # 显示显著性星号
    float_format="%.4f",
    info_dict={
        "N": lambda x: x.nobs,
        "R2": lambda x: getattr(x, "rsquared", np.nan),
        "Adj. R2": lambda x: getattr(x, "rsquared_adj", np.nan),
        "AIC": lambda x: getattr(x, "aic", np.nan),
    }
)

print(comparison)
```

**输出示例**：

```
================================================================
                   Model I         Model II         Model III   
----------------------------------------------------------------
const            1.234***       1.456***         1.678***   
              (0.123)         (0.134)           (0.145)   
x1               0.567**        0.678**             
              (0.234)         (0.245)             
x2                               0.890*           0.901*   
                                (0.456)          (0.467)   
----------------------------------------------------------------
R-squared        0.890           0.912            0.923   
N                100             100              100   
AIC             123.45          112.34           101.23   
================================================================
Standard errors in parentheses.
* p<.1, ** p<.05, ***p<.01
```

**核心实现原理** (`summary2.py:470-626`)：

```python
def summary_col(results, float_format="%.4f", model_names=(), stars=False,
                info_dict=None, regressor_order=(), ...):
    """
    多模型比较的核心函数
    
    设计思路：
    1. 提取每个模型的系数和标准误
    2. 将标准误用括号括起来，放在系数下方
    3. 横向合并所有模型
    4. 添加显著性星号（可选）
    5. 在底部添加模型统计量（N, R2, AIC 等）
    """
    
    # 1. 确保 results 是列表
    if not isinstance(results, list):
        results = [results]
    
    # 2. 提取每个模型的参数和标准误，合并为 DataFrame 列
    cols = [_col_params(x, stars=stars, float_format=float_format,
                        include_r2=include_r2) for x in results]
    
    # 3. 处理列名（确保唯一）
    if model_names:
        colnames = _make_unique(model_names)
    else:
        colnames = _make_unique([x.columns[0] for x in cols])
    for i in range(len(cols)):
        cols[i].columns = [colnames[i]]
    
    # 4. 横向合并所有模型（外连接，自动对齐变量）
    def merg(x, y):
        return x.merge(y, how="outer", right_index=True, left_index=True)
    
    summ = reduce(merg, cols)
    
    # 5. 添加模型统计信息（N, R2, AIC 等）
    if info_dict:
        cols = [_col_info(x, info_dict.get(x.model.__class__.__name__,
                                           info_dict)) for x in results]
        info = reduce(merg, cols)
        # ... 合并到主表
    
    # 6. 封装为 Summary 对象
    smry = Summary()
    smry._merge_latex = True  # LaTeX 输出时合并表格
    smry.add_df(summ, header=True, align="l")
    smry.add_text("Standard errors in parentheses.")
    if stars:
        smry.add_text("* p<.1, ** p<.05, ***p<.01")
    
    return smry
```

**参数提取核心**：`summary2.py:396-433` 的 `_col_params()` 函数

```python
def _col_params(result, float_format="%.4f", stars=True, include_r2=False):
    """
    将单个模型的系数和标准误堆叠为单列
    
    输出格式：
    const      1.234***
              (0.123)
    x1         0.567**
              (0.234)
    """
    # 1. 提取参数表
    res = summary_params(result)
    
    # 2. 格式化系数
    for col in res.columns[:2]:
        res[col] = res[col].apply(lambda x: float_format % x)
    
    # 3. 标准误用括号括起来
    res.iloc[:, 1] = "(" + res.iloc[:, 1] + ")"
    
    # 4. 添加显著性星号
    if stars:
        idx = res.iloc[:, 3] < .1
        res.loc[idx, res.columns[0]] = res.loc[idx, res.columns[0]] + "*"
        idx = res.iloc[:, 3] < .05
        res.loc[idx, res.columns[0]] = res.loc[idx, res.columns[0]] + "*"
        idx = res.iloc[:, 3] < .01
        res.loc[idx, res.columns[0]] = res.loc[idx, res.columns[0]] + "*"
    
    # 5. 堆叠系数和标准误
    res = res.iloc[:, :2]
    res = res.stack(**FUTURE_STACK)
    
    # 6. 可选：添加 R-squared
    if include_r2:
        rsquared = getattr(result, "rsquared", np.nan)
        # ...
    
    return pd.DataFrame(res)
```

**机制 A 支持情况**：❌ **完全不支持**

机制 A 没有提供多模型比较的功能。用户需要手动实现，非常繁琐。

---

## 三、两套机制设计取舍深度分析

### 3.1 设计哲学对比

#### 机制 A (`summary.py`) 的设计哲学：显式优于隐式

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    机制 A 设计哲学：显式优于隐式                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  核心原则：每个模型的 summary() 方法显式构建所有内容                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  线性回归模型的 summary() 方法：                                      │   │
│  │                                                                     │   │
│  │  1. 显式构建 top_left 列表                                           │   │
│  │     top_left = [                                                    │   │
│  │         ("Dep. Variable:", None),                                   │   │
│  │         ("Model:", None),                                            │   │
│  │         ("Method:", ["Least Squares"]),                              │   │
│  │         ("Date:", None),                                             │   │
│  │         ("Time:", None),                                             │   │
│  │         ("No. Observations:", None),                                 │   │
│  │         ...                                                          │   │
│  │     ]                                                                │   │
│  │                                                                     │   │
│  │  2. 显式构建 top_right 列表                                          │   │
│  │     top_right = [                                                   │   │
│  │         ("R-squared:", ["%#8.3f" % self.rsquared]),                │   │
│  │         ("Adj. R-squared:", ["%#8.3f" % self.rsquared_adj]),       │   │
│  │         ("F-statistic:", ["%#8.4g" % self.fvalue]),                │   │
│  │         ...                                                          │   │
│  │     ]                                                                │   │
│  │                                                                     │   │
│  │  3. 显式构建 diagn_left, diagn_right                                 │   │
│  │  4. 显式构建 etext 警告列表                                          │   │
│  │  5. 显式调用 add_table_2cols(), add_table_params()                  │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  优点：                                                                     │
│  ✅ 完全控制每个细节：显示什么、格式如何、何时显示                           │
│  ✅ 易于添加条件逻辑：if slim:, if hasattr(self, "cov_type"):, etc.       │
│  ✅ 各模型可以完全定制：不同模型家族可以有完全不同的输出格式                  │
│  ✅ 依赖最少：仅依赖 numpy                                                   │
│                                                                             │
│  缺点：                                                                     │
│  ❌ 代码重复：每个模型都要写类似的 top_left, top_right 构建逻辑            │
│  ❌ 不够 DRY："Don't Repeat Yourself" 原则违反                              │
│  ❌ 修改困难：要改变所有模型的输出格式，需要逐个修改                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 机制 B (`summary2.py`) 的设计哲学：约定优于配置

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   机制 B 设计哲学：约定优于配置                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  核心原则：通过约定自动完成大部分工作，减少重复代码                           │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  线性回归模型的 summary2() 方法：                                     │   │
│  │                                                                     │   │
│  │  # 简洁！大部分工作由 add_base() 自动完成                            │   │
│  │  smry = summary2.Summary()                                          │   │
│  │  smry.add_base(results=self, alpha=alpha, float_format=float_format)│   │
│  │  smry.add_dict(diagnostic)  # 添加诊断信息                           │   │
│  │                                                                     │   │
│  │  add_base() 内部约定：                                               │   │
│  │  - 自动调用 summary_model() 提取模型信息                             │   │
│  │  - 自动调用 summary_params() 提取参数表                              │   │
│  │  - 自动添加标题                                                       │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  关键约定函数：                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  summary_model(results) -> dict                                      │   │
│  │  - 自动尝试提取 results.model.__class__.__name__                      │   │
│  │  - 自动尝试提取 results.nobs, results.df_model                        │   │
│  │  - 自动尝试提取 results.rsquared, results.aic, results.bic            │   │
│  │  - 用 try-except 处理不存在的属性（不报错，跳过）                      │   │
│  │                                                                     │   │
│  │  核心代码：                                                          │   │
│  │  info = {}                                                           │   │
│  │  info["Model:"] = lambda x: x.model.__class__.__name__              │   │
│  │  info["R-squared:"] = lambda x: "%#8.3f" % x.rsquared               │   │
│  │  info["AIC:"] = lambda x: "%8.4f" % x.aic                           │   │
│  │  # ...                                                               │   │
│  │                                                                     │   │
│  │  for key, func in info.items():                                     │   │
│  │      try:                                                           │   │
│  │          out[key] = func(results)  # 尝试执行                       │   │
│  │      except (AttributeError, KeyError, NotImplementedError):        │   │
│  │          pass  # 不存在则跳过，不报错                                │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  优点：                                                                     │
│  ✅ 代码简洁：一行 add_base() 替代多行显式构建                              │
│  ✅ 易于维护：修改输出格式只需改 summary_model() 或 summary_params()        │
│  ✅ DataFrame 中间层：便于程序化后处理、导出                                │
│  ✅ 多模型比较：summary_col() 是独家优势                                    │
│                                                                             │
│  缺点：                                                                     │
│  ❌ 约定不够灵活：如果模型属性命名不符合约定，就无法自动提取                 │
│  ❌ 细粒度控制困难：要定制某个特定项的显示，需要绕过约定                     │
│  ❌ 依赖 pandas：环境中没有 pandas 就无法使用                               │
│  ❌ 状态不明确：标记为 "Experimental"，但功能实用                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 场景适配性矩阵

| 场景 | 机制 A 适配度 | 机制 B 适配度 | 最佳选择 | 核心原因 |
|-----|-------------|-------------|---------|---------|
| S1: 标准单模型汇报 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 机制 A | 成熟稳定，所有模型支持 |
| S2: 精简输出 (slim) | ⭐⭐⭐⭐⭐ | ❌ | **机制 A** | 独家功能，机制 B 无此参数 |
| S3: 约束估计 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 机制 A | 自动检测，实现一致 |
| S4: 多方程模型 | ⭐⭐⭐⭐⭐ | ⭐⭐ | **机制 A** | 自动检测维度，专门的 `summary_params_2dflat` |
| S5: 稳健协方差 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 机制 A | 顶部显示类型 + 警告详细说明 |
| S6: 无常数项模型 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 均可 | 两套实现基本一致 |
| S7: 多重共线性诊断 | ⭐⭐⭐⭐⭐ | ❌ | **机制 A** | 独家功能，机制 B 缺失此警告 |
| S8: 多模型比较 | ❌ | ⭐⭐⭐⭐⭐ | **机制 B** | 独家功能，`summary_col()` |
| 需要 DataFrame 后处理 | ⭐⭐ | ⭐⭐⭐⭐⭐ | **机制 B** | 原生 DataFrame 中间层 |
| 无 pandas 环境 | ⭐⭐⭐⭐⭐ | ❌ | **机制 A** | 不依赖 pandas |
| 学术论文发表 | ⭐⭐⭐⭐⭐ | ⭐⭐ | **机制 A** | 格式更紧凑、更标准 |
| 探索性数据分析 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 机制 B | 多模型比较功能强大 |

### 3.3 关键设计决策对比

#### 决策 1：表格数据结构

| 方面 | 机制 A | 机制 B |
|-----|--------|--------|
| 核心容器 | `SimpleTable` (自定义类) | `pandas.DataFrame` |
| 设计思路 | 为表格显示专门设计 | 通用数据结构 |
| 列操作 | 需手动实现 | DataFrame 原生支持 (merge, concat, etc.) |
| 行操作 | 需手动实现 | DataFrame 原生支持 |
| 导出功能 | 需实现 `as_csv()`, `as_html()` 等 | DataFrame 原生 `to_csv()`, `to_html()` 等 |

**设计取舍**：
- 机制 A 选择 `SimpleTable`：追求显示控制的最大化，不依赖外部库
- 机制 B 选择 `DataFrame`：追求数据操作的灵活性，接受 pandas 依赖

#### 决策 2：条件分支的实现位置

| 场景条件 | 机制 A 实现位置 | 机制 B 实现位置 |
|---------|----------------|----------------|
| `slim=True` | 各模型 `summary()` 方法内 | ❌ 不支持 |
| `params.ndim == 2` | `Summary.add_table_params()` 方法内 | 需手动在 `summary2()` 内循环 |
| `hasattr(self, "constraints")` | 各模型 `summary()` 方法内 | 各模型 `summary2()` 方法内 |
| `hasattr(self, "cov_type")` | 各模型 `summary()` 方法内 | `summary_model()` 函数内 (不一致) |
| 多重共线性检查 | 各模型 `summary()` 方法内 | ❌ 不支持 |

**设计取舍**：
- 机制 A：条件分支**分散在各模型**，但**提供了通用辅助方法**（如 `add_table_params` 自动检测维度）
- 机制 B：尝试将条件分支**集中到辅助函数**（如 `summary_model`），但实现不够一致

#### 决策 3：统计量的提取方式

| 方式 | 机制 A | 机制 B |
|-----|--------|--------|
| 提取方式 | 显式访问属性 + 格式化 |  lambda 函数 + try-except |
| 错误处理 | 需手动处理 | try-except 自动跳过 |
| 灵活性 | 每个项可以单独定制格式 | 统一格式化 |

**机制 A 风格**：
```python
# 显式格式化每个项
top_right = [
    ("R-squared:", ["%#8.3f" % self.rsquared]),
    ("Adj. R-squared:", ["%#8.3f" % self.rsquared_adj]),
    ("F-statistic:", ["%#8.4g" % self.fvalue]),
]
```

**机制 B 风格**：
```python
# 统一约定，自动处理缺失
info = {}
info["R-squared:"] = lambda x: "%#8.3f" % x.rsquared
info["Adj. R-squared:"] = lambda x: "%#8.3f" % x.rsquared_adj
info["F-statistic:"] = lambda x: "%#8.4g" % x.fvalue

for key, func in info.items():
    try:
        out[key] = func(results)
    except (AttributeError, KeyError, NotImplementedError):
        pass  # 不存在则跳过
```

### 3.4 为什么状态空间模型没有实现 `summary2()`？

**状态空间模型** (`statsmodels/tsa/statespace/mlemodel.py`) 是 statsmodels 中最复杂的模型家族之一，包括：
- SARIMAX (季节 ARIMA)
- VARMAX (向量 ARMA)
- Dynamic Factor (动态因子)
- Unobserved Components (不可观测成分)

**它们只实现了 `summary()`，没有实现 `summary2()`**，原因分析：

#### 原因 1：状态空间模型的 `summary()` 已经非常复杂

**位置**: `mlemodel.py:5266-5493` (约 230 行代码)

```python
def summary(self, alpha=0.05, start=None, title=None, model_name=None,
            display_params=True, display_diagnostics=True,
            truncate_endog_names=None, display_max_endog=None,
            extra_top_left=None, extra_top_right=None):
    """
    状态空间模型的 summary() 有大量特殊参数：
    
    Parameters
    ----------
    start : int, optional
        开始观测值（用于时间序列样本区间显示）
    model_name : str
        模型名称
    display_params : bool
        是否显示参数表
    display_diagnostics : bool
        是否显示诊断测试
    truncate_endog_names : int
        截断长的内生变量名
    display_max_endog : int
        最多显示多少个内生变量（多变量模型）
    extra_top_left, extra_top_right : list
        用户自定义的额外统计量
    """
    
    # 1. 复杂的时间序列样本区间显示
    if self.model._index_dates:
        ix = self.model._index
        d = ix[start]
        sample = ["%02d-%02d-%02d" % (d.month, d.day, d.year)]
        d = ix[-1]
        sample += ["- " + "%02d-%02d-%02d" % (d.month, d.day, d.year)]
    else:
        sample = [str(start), " - " + str(self.nobs)]
    
    # 2. 多内生变量的复杂处理
    if self.model.k_endog > display_max_endog:
        k = self.model.k_endog - 1
        yname = '"' + endog_names[0] + f'", and {k} more'
    
    # 3. 条件性的诊断测试（只有 display_diagnostics=True 时才计算）
    if display_diagnostics:
        try:
            het = self.test_heteroskedasticity(method="breakvar")
        except Exception:
            het = np.zeros((self.model.k_endog, 2)) * np.nan
        try:
            lb = self.test_serial_correlation(method="ljungbox", lags=[1])
        except Exception:
            lb = np.zeros((self.model.k_endog, 2, 1)) * np.nan
        try:
            jb = self.test_normality(method="jarquebera")
        except Exception:
            jb = np.zeros((self.model.k_endog, 4)) * np.nan
        
        # 4. 多内生变量时诊断表布局变化
        if self.model.k_endog <= display_max_endog:
            # 少数内生变量：双列布局
            diagn_left = [
                ("Ljung-Box (L1) (Q):", format_str(lb[:, 0, -1])),
                ("Prob(Q):", format_str(lb[:, 1, -1])),
                ("Heteroskedasticity (H):", format_str(het[:, 0])),
                ("Prob(H) (two-sided):", format_str(het[:, 1])),
            ]
            diagn_right = [
                ("Jarque-Bera (JB):", format_str(jb[:, 0])),
                ("Prob(JB):", format_str(jb[:, 1])),
                ("Skew:", format_str(jb[:, 2])),
                ("Kurtosis:", format_str(jb[:, 3])),
            ]
            summary.add_table_2cols(...)
        else:
            # 多数内生变量：宽表布局（每个内生变量一行）
            data = pd.DataFrame(np.c_[lb[:, :2, -1], het[:, :2], jb[:, :4]], ...)
            # ... 手动构建 SimpleTable
            table = SimpleTable(params_data, params_header, ...)
            summary.tables.insert(table_ix, table)
    
    # 5. 额外的自定义统计量支持
    if extra_top_left is not None:
        top_left += extra_top_left
    if extra_top_right is not None:
        top_right += extra_top_right
```

#### 原因 2：机制 B 的设计假设不适合状态空间模型

机制 B 的 `add_base()` 方法假设：
1. 参数是一维的 (`params.ndim == 1`)
2. 模型信息可以通过简单的属性访问提取
3. 诊断统计量是简单的键值对

但状态空间模型：
1. **参数可能是高维的**：多变量模型有很多参数
2. **需要复杂的条件显示**：`display_params`, `display_diagnostics` 等参数控制
3. **诊断表布局变化**：根据 `k_endog` 数量选择不同的表格布局
4. **用户自定义扩展**：`extra_top_left`, `extra_top_right` 允许用户添加自定义统计量

#### 原因 3：维护成本考虑

状态空间模型的 `summary()` 已经有 230+ 行复杂代码，再实现一个 `summary2()` 版本需要：
1. 复制所有复杂逻辑
2. 确保两个版本功能一致
3. 未来修改时需要同时维护两份代码

这会显著增加维护成本，而收益有限。

### 3.5 为什么 `summary2()` 标记为 "Experimental"？

**证据**：文档字符串中明确标记

```python
# linear_model.py:3026-3027
"""
Experimental summary function to summarize the regression results.
"""

# discrete_model.py:5475-5476
"""
Experimental function to summarize regression results.
"""
```

**可能的原因**：

| 原因 | 说明 |
|-----|------|
| API 不稳定 | `summary2()` 和 `summary_col()` 的接口可能还会变化 |
| 覆盖不完整 | 只有部分模型实现了 `summary2()`，状态空间等复杂模型没有 |
| 依赖问题 | 强依赖 pandas，而 statsmodels 核心不强制依赖 pandas |
| 功能重叠 | 与 `summary()` 功能大部分重叠，定位不清晰 |
| 测试覆盖 | 可能测试不如 `summary()` 充分 |

**但实际使用建议**：

- `summary_col()` 是**非常实用**的功能，多模型比较场景几乎是刚需
- 如果你的环境有 pandas，可以放心使用
- 但如果需要编写稳定的生产代码，`summary()` 更可靠

---

## 四、结论与建议

### 4.1 核心结论

#### 结论 1：两套机制是"同源分流"的关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         同源分流架构                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│              ┌──────────────────────────────────────┐                      │
│              │         统一统计量计算层               │                      │
│              │  params, bse, tvalues, pvalues, etc. │                      │
│              │  statsmodels/base/model.py            │                      │
│              └───────────────┬──────────────────────┘                      │
│                              │                                               │
│              ┌───────────────┴───────────────┐                           │
│              │                               │                           │
│              ▼                               ▼                           │
│    ┌───────────────────┐         ┌───────────────────┐                    │
│    │    机制 A          │         │    机制 B          │                    │
│    │  (summary.py)     │         │  (summary2.py)    │                    │
│    ├───────────────────┤         ├───────────────────┤                    │
│    │ • SimpleTable     │         │ • DataFrame       │                    │
│    │ • 显式构建         │         │ • 约定优于配置     │                    │
│    │ • 细粒度控制       │         │ • 多模型比较       │                    │
│    │ • 所有模型支持     │         │ • 部分模型支持     │                    │
│    └───────────────────┘         └───────────────────┘                    │
│                                                                             │
│  关键：两套机制从同一个 results 对象提取相同的统计量，只是展示方式不同！     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 结论 2：机制 A 是"通用基础"，机制 B 是"功能扩展"

| 维度 | 机制 A | 机制 B |
|-----|--------|--------|
| 定位 | 所有模型的标准输出方法 | 部分模型的备选/实验性方法 |
| 覆盖范围 | 所有模型家族 | 线性回归、离散模型等简单模型 |
| 功能完整性 | ✅ 完整支持所有场景 | ⚠️ 缺失部分场景（slim, 多重共线性诊断） |
| 依赖 | 仅 numpy | numpy + pandas |
| 稳定性 | 稳定/主流 | 实验性/备选 |

#### 结论 3：机制 B 的核心价值是 `summary_col()`

`summary2.py` 中真正不可替代的功能只有一个：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    机制 B 的核心价值：summary_col()                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  其他功能（summary2() 方法）：                                              │
│  • 与 summary() 功能大部分重叠                                              │
│  • 实现不完整（缺失 slim、多重共线性诊断等）                                  │
│  • 状态空间等复杂模型不支持                                                  │
│                                                                             │
│  独家功能：summary_col()                                                    │
│  ✅ 多模型并排比较                                                           │
│  ✅ 显著性星号标注                                                           │
│  ✅ 自定义统计量显示                                                         │
│  ✅ 自动变量对齐                                                             │
│  ✅ LaTeX 友好输出                                                          │
│                                                                             │
│  这是学术论文中最常用的功能之一，机制 A 完全无法替代！                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 使用建议

#### 建议 1：默认使用 `results.summary()`

**理由**：
1. **所有模型都支持**：不用担心某个模型没有实现
2. **功能完整**：包含所有诊断警告（slim、多重共线性等）
3. **格式标准**：学术论文中的标准格式
4. **稳定可靠**：经过长期测试和使用

#### 建议 2：需要多模型比较时使用 `summary_col()`

```python
# 推荐模式
from statsmodels.iolib.summary2 import summary_col

# 1. 用 summary() 做单个模型的详细分析
results = model.fit()
print(results.summary())  # 查看完整诊断

# 2. 用 summary_col() 做模型比较
results_list = [model1.fit(), model2.fit(), model3.fit()]
comparison = summary_col(
    results_list,
    model_names=["Model 1", "Model 2", "Model 3"],
    stars=True,
    info_dict={"N": lambda x: x.nobs, "R2": lambda x: getattr(x, "rsquared", np.nan)}
)
print(comparison)
```

#### 建议 3：谨慎使用 `results.summary2()`

**只有在以下情况才考虑**：
1. 你需要 DataFrame 格式的输出做后续程序化处理
2. 你使用的模型（线性回归、离散模型）完整支持 `summary2()`
3. 你不需要 `slim` 或多重共线性诊断等功能

**注意**：状态空间模型（SARIMAX 等）没有 `summary2()` 方法。

#### 建议 4：了解关键场景的触发方式

| 场景 | 如何触发 | 注意事项 |
|-----|---------|---------|
| 精简输出 | `results.summary(slim=True)` | 机制 A 独有 |
| 约束估计 | `model.fit_constrained(constraints)` | 两套机制都会显示警告 |
| 多方程模型 | 自动检测 `params.ndim` | 机制 A 自动处理，机制 B 需手动循环 |
| 稳健协方差 | `model.fit(cov_type="HC0")` | 机制 A 显示更完整 |
| 多模型比较 | `summary_col([...])` | 机制 B 独有 |

### 4.3 对 statsmodels 开发者的建议

#### 建议 A：明确 `summary2()` 的定位

当前 `summary2.py` 的定位模糊：
- 是 `summary()` 的替代？
- 是 `summary()` 的补充？
- 只是 `summary_col()` 的辅助模块？

**建议**：
1. 要么将 `summary2()` 标记为稳定，修复缺失功能（slim、多重共线性诊断等）
2. 要么明确 `summary2.py` 只是 `summary_col()` 的支持模块，不推荐直接使用 `results.summary2()`

#### 建议 B：考虑将 `summary_col()` 独立出来

`summary_col()` 的功能很实用，但被"Experimental"标记拖累。

**建议**：
1. 将 `summary_col()` 移到更明显的位置（如 `statsmodels.iolib.table_comparison`）
2. 独立维护，不受 `summary2()` 的"Experimental"状态影响
3. 增加更多自定义选项（如显示 t 值还是标准误、置信区间等）

#### 建议 C：统一多方程处理逻辑

机制 A 的 `summary_params_2dflat()` 设计很好，但机制 B 没有等价功能。

**建议**：
1. 提取多方程处理的通用逻辑
2. 让两套机制都能使用
3. 或者在机制 B 中实现类似的自动检测

### 4.4 最终总结

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              最终总结                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  关于路径触发：                                                             │
│  ✅ 两套机制都实现了 _repr_html_() 和 _repr_latex_()                        │
│  ✅ Jupyter 显示哪个机制取决于用户调用的是 summary() 还是 summary2()        │
│  ❌ 之前的错误："Jupyter 显示 -> 机制 A" 是不准确的                         │
│                                                                             │
│  关于场景分支：                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 场景 S1: 标准单模型                                                   │   │
│  │   触发: results.summary() / results.summary2()                       │   │
│  │   机制 A: ✅ 完整支持                                                 │   │
│  │   机制 B: ⚠️ 部分模型支持                                             │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ 场景 S2: 精简输出 (slim=True)                                        │   │
│  │   触发: results.summary(slim=True)                                   │   │
│  │   机制 A: ✅ 独家支持                                                 │   │
│  │   机制 B: ❌ 不支持                                                   │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ 场景 S3: 约束估计                                                     │   │
│  │   触发: model.fit_constrained(constraints)                           │   │
│  │   检测: hasattr(self, "constraints")                                 │   │
│  │   输出: 显示 "Model has been estimated subject to linear equality    │   │
│  │         constraints." 警告                                            │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ 场景 S4: 多方程/多因变量模型                                          │   │
│  │   触发: params.ndim == 2                                             │   │
│  │   机制 A: ✅ 自动检测 + summary_params_2dflat()                     │   │
│  │   机制 B: ⚠️ 需手动循环处理                                          │   │
│  │   典型模型: MNLogit, VAR, SUR                                        │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ 场景 S5: 稳健协方差估计                                               │   │
│  │   触发: fit(cov_type="HC0"|"HAC"|"cluster"|...)                     │   │
│  │   检测: hasattr(self, "cov_type")                                    │   │
│  │   机制 A: ✅ 顶部显示类型 + 警告详细说明                               │   │
│  │   机制 B: ⚠️ 实现不一致                                               │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ 场景 S6: 多重共线性诊断                                               │   │
│  │   触发: eigvals[-1] < 1e-10 或 condno > 1000                       │   │
│  │   机制 A: ✅ 独家支持，显示警告                                        │   │
│  │   机制 B: ❌ 不支持                                                   │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ 场景 S7: 多模型比较                                                   │   │
│  │   触发: summary_col([results1, results2, ...])                       │   │
│  │   机制 A: ❌ 不支持                                                   │   │
│  │   机制 B: ✅ 独家支持，学术论文必备功能                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  关于设计取舍：                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 机制 A (summary.py) 设计哲学：显式优于隐式                           │   │
│  │  • 每个模型显式构建 top_left, top_right, diagn_left, diagn_right     │   │
│  │  • 完全控制每个细节                                                   │   │
│  │  • 易于添加条件逻辑 (if slim:, if hasattr(...):)                     │   │
│  │  • 依赖最少 (仅 numpy)                                                │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ 机制 B (summary2.py) 设计哲学：约定优于配置                          │   │
│  │  • add_base() 自动完成大部分工作                                      │   │
│  │  • DataFrame 中间层，便于后处理                                       │   │
│  │  • summary_col() 独家多模型比较功能                                   │   │
│  │  • 强依赖 pandas                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  为什么状态空间模型没有实现 summary2()？                                   │
│  1. 状态空间模型的 summary() 已经有 230+ 行复杂代码                        │
│  2. 需要复杂的参数控制 (display_params, display_diagnostics, etc.)         │
│  3. 诊断表布局根据 k_endog 数量动态变化                                    │
│  4. 维护成本过高，收益有限                                                  │
│                                                                             │
│  为什么 summary2() 标记为 "Experimental"？                                 │
│  1. API 可能变化                                                            │
│  2. 覆盖不完整 (状态空间等复杂模型没有)                                     │
│  3. 强依赖 pandas                                                           │
│  4. 定位不清晰 (是替代？补充？还是辅助？)                                   │
│                                                                             │
│  但：summary_col() 是非常实用的功能，可以放心使用！                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 附录：关键代码位置速查表

| 功能 | 文件路径 | 行号 |
|-----|---------|------|
| 机制 A `_repr_html_` | `statsmodels/iolib/summary.py` | 768-770 |
| 机制 A `_repr_latex_` | `statsmodels/iolib/summary.py` | 772-774 |
| 机制 B `_repr_html_` | `statsmodels/iolib/summary2.py` | 30-32 |
| 机制 B `_repr_latex_` | `statsmodels/iolib/summary2.py` | 34-36 |
| `slim` 参数处理 | `statsmodels/regression/linear_model.py` | 2925-2939 |
| 约束估计检测 | `statsmodels/genmod/generalized_linear_model.py` | 2786-2789 |
| 多方程自动检测 | `statsmodels/iolib/summary.py` | 828-834 |
| `summary_params_2dflat` | `statsmodels/iolib/summary.py` | 596-669 |
| `table_extend` | `statsmodels/iolib/summary.py` | 672-714 |
| 稳健协方差显示 | `statsmodels/regression/linear_model.py` | 2899-2902, 2990-2991 |
| 多重共线性诊断 | `statsmodels/regression/linear_model.py` | 3005-3021 |
| `summary_col` | `statsmodels/iolib/summary2.py` | 470-626 |
| `_col_params` | `statsmodels/iolib/summary2.py` | 396-433 |
| 状态空间 `summary` | `statsmodels/tsa/statespace/mlemodel.py` | 5266-5493 |

---

**报告完成时间**：2026-05-01  
**基于代码版本**：statsmodels 代码库分析  
**关键修正**：
1. 纠正了 Jupyter 显示与机制的对应关系
2. 补全了 8 个关键场景的触发条件和输出差异
3. 深入分析了两套机制的设计取舍和适用场景