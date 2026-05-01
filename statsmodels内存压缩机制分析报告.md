# Statsmodels 拟合结果体系：内存压缩机制技术分析报告

## 目录

1. [概述](#1-概述)
2. [数据移除的双重识别路径](#2-数据移除的双重识别路径)
3. [数据移除后预测接口的功能边界](#3-数据移除后预测接口的功能边界)
4. [不同模型族的覆盖逻辑差异](#4-不同模型族的覆盖逻辑差异)
5. [设计模式与架构权衡](#5-设计模式与架构权衡)
6. [附录：完整执行流程图](#6-附录完整执行流程图)

---

## 1. 概述

### 1.1 问题背景

在统计建模实践中，一个常见的场景是：

1. **模型训练阶段**：需要完整的数据集 (`endog`, `exog`, 权重等) 来估计参数
2. **模型持久化阶段**：只需要参数 (`params`) 和协方差矩阵进行预测和推断
3. **内存优化需求**：大型数据集会导致序列化的结果对象非常庞大

Statsmodels 通过 **`remove_data()` 方法** 和 **`save(remove_data=True)` 选项** 提供了优雅的解决方案：在序列化前移除所有与原始数据相关的数组，只保留进行预测和统计推断所需的最小信息。

### 1.2 核心机制概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         内存压缩机制概览                                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  完整结果对象 (内存占用大)                                                │
├─────────────────────────────────────────────────────────────────────────┤
│  params: ndarray          ← 保留 (参数估计值)                            │
│  normalized_cov_params    ← 保留 (协方差矩阵)                            │
│  scale: float             ← 保留 (尺度参数)                               │
│                                                                          │
│  exog: ndarray            ──┐                                            │
│  endog: ndarray             │                                            │
│  wendog, wexog              │                                            │
│  weights: ndarray           │  ← 这些被 remove_data() 清除             │
│  resid (缓存)               │                                            │
│  fittedvalues (缓存)        │                                            │
│  mu, freq_weights, etc.    ─┘                                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ remove_data()
┌─────────────────────────────────────────────────────────────────────────┐
│  压缩后结果对象 (内存占用小)                                              │
├─────────────────────────────────────────────────────────────────────────┤
│  params: ndarray          ← 保留                                         │
│  normalized_cov_params    ← 保留                                         │
│  scale: float             ← 保留                                         │
│                                                                          │
│  exog: None               ← 设为 None                                    │
│  endog: None              ← 设为 None                                    │
│  resid: None              ← 缓存设为 None                                │
│  fittedvalues: None       ← 缓存设为 None                                │
│  ...                                                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ 仍然支持的操作
┌─────────────────────────────────────────────────────────────────────────┐
│  ✅ predict()           → 仅需要 params 和新 exog                        │
│  ✅ get_prediction()    → 预测 + 推断统计                                │
│  ✅ params, bse, tvalues, pvalues → 基于协方差矩阵计算                   │
│  ✅ aic, bic            → 基于 llf (如果已缓存)                          │
│  ❌ resid, fittedvalues → 依赖原始数据，返回 None 或报错                 │
│  ❌ 重新计算 llf         → 需要 endog，可能失败                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据移除的双重识别路径

### 2.1 路径一：类级缓存扫描

**核心思想**：通过装饰器在类定义时标记需要特殊处理的缓存属性。

#### 2.1.1 装饰器的语义区分

**位置**: `statsmodels/tools/decorators.py:136-151`

```python
# Use pandas since it works with docs correctly
cache_readonly = PandasCacheReadonly

# cached_value and cached_data behave identically to cache_readonly, but
# are used by `remove_data` to
#   a) identify array-like attributes to remove (cached_data)
#   b) make sure certain values are evaluated before caching (cached_value)

cached_data = PandasCacheReadonly
cached_value = PandasCacheReadonly
```

虽然三个装饰器在实现上相同（都指向 `PandasCacheReadonly`），但它们具有**不同的语义含义**：

| 装饰器 | 语义用途 | 示例 |
|--------|---------|------|
| `@cache_readonly` | 通用只读缓存，不参与数据移除 | `aic`, `bic`, `df_modelwc` |
| `@cached_value` | 计算值，需在缓存前求值 | `llf`, `bse`, `tvalues` |
| `@cached_data` | **数组类属性，数据移除时设为 None** | `resid`, `fittedvalues` (概念上) |

#### 2.1.2 类级扫描的实现

**位置**: `statsmodels/base/model.py:2459-2469`

```python
def remove_data(self):
    cls = self.__class__
    
    # 注意：不能直接使用 getattr(cls, x) 或 getattr(self, x)
    # 因为属性访问器会触发重定向
    cls_attrs = {}
    for name in dir(cls):
        try:
            # 使用 object.__getattribute__ 绕过描述符协议
            attr = object.__getattribute__(cls, name)
        except AttributeError:
            pass
        else:
            cls_attrs[name] = attr
    
    # 筛选出所有用 cached_data 装饰的属性
    data_attrs = [x for x in cls_attrs if isinstance(cls_attrs[x], cached_data)]
    
    # 将这些属性在缓存中设为 None
    for name in data_attrs:
        self._cache[name] = None
```

**设计要点**：

1. **绕过描述符协议**：使用 `object.__getattribute__(cls, name)` 而非 `getattr(cls, name)`
   - 原因：`getattr()` 会触发描述符的 `__get__()` 方法，导致实际执行属性计算
   - 目的：只检查类属性是否是 `cached_data` 实例，不触发计算

2. **`dir(cls)` 的使用**：遍历类的所有属性名，包括继承的属性

3. **`isinstance(cls_attrs[x], cached_data)`**：检查装饰器类型
   - 这就是为什么需要语义上区分 `cached_data` 和其他装饰器

### 2.2 路径二：实例级数据清单

**核心思想**：通过三个列表属性显式声明需要清除的数据。

#### 2.2.1 三个关键清单

| 清单属性 | 声明位置 | 存储内容 | 处理方式 |
|---------|---------|---------|---------|
| `_data_attr` | Model/Results | 结果和模型共有的数组属性名 | 递归清除 (`wipe`) |
| `_data_attr_model` | Results | 仅模型独有的属性名 | 加上 "model." 前缀后清除 |
| `_data_in_cache` | Results | 缓存字典中的键名 | 直接设 `_cache[key] = None` |

#### 2.2.2 _data_attr 的构建与扩展

`_data_attr` 是一个在**多层继承中累积**的列表：

**Model 基类初始化**: `statsmodels/base/model.py:107-110`
```python
def __init__(self, endog, exog=None, **kwargs):
    # ...
    self._data_attr = []
    self._data_attr.extend(["exog", "endog", "data.exog", "data.endog"])
    if "formula" not in kwargs:
        self._data_attr.extend(["data.orig_endog", "data.orig_exog"])
```

**RegressionModel 扩展**: `statsmodels/regression/linear_model.py:223`
```python
class RegressionModel(base.LikelihoodModel):
    def __init__(self, endog, exog, **kwargs):
        super().__init__(endog, exog, **kwargs)
        self.pinv_wexog: Float64Array | None = None
        self._data_attr.extend(["pinv_wexog", "wendog", "wexog", "weights"])
```

**GLS 进一步扩展**: `statsmodels/regression/linear_model.py:585`
```python
class GLS(RegressionModel):
    def __init__(self, endog, exog, sigma=None, **kwargs):
        super().__init__(endog, exog, **kwargs)
        self.sigma, self.cholsigmainv = _get_sigma(sigma, self.endog.shape[0])
        self._data_attr.extend(["sigma", "cholsigmainv"])
```

**GLM 模型扩展**: `statsmodels/genmod/generalized_linear_model.py:379-390`
```python
class GLM(base.LikelihoodModel):
    def __init__(self, endog, exog, family=None, ...):
        # ...
        self.nobs = self.endog.shape[0]
        
        # things to remove_data
        self._data_attr.extend(
            [
                "weights",
                "mu",
                "freq_weights",
                "var_weights",
                "iweights",
                "_offset_exposure",
                "n_trials",
            ]
        )
```

#### 2.2.3 _data_in_cache 的构建

**Results 基类**: `statsmodels/base/model.py:1127-1129`
```python
class Results:
    def __init__(self, model, params, **kwd):
        # ...
        self._data_attr = []
        # Variables to clear from cache
        self._data_in_cache = ["fittedvalues", "resid", "wresid"]
```

**GLMResults 扩展**: `statsmodels/genmod/generalized_linear_model.py:1837-1838`
```python
class GLMResults(LikelihoodModelResults):
    def __init__(self, ...):
        # ...
        self._data_attr.extend(["results_constrained", "_freq_weights", "_var_weights", "_iweights"])
        self._data_in_cache.extend(["null", "mu"])
```

**RLMResults 扩展**: `statsmodels/robust/robust_linear_model.py:445-446`
```python
class RLMResults(RegressionResults):
    def __init__(self, ...):
        # ...
        # for remove_data
        self._data_in_cache.extend(["sresid"])
```

#### 2.2.4 _data_attr_model 的使用

**GLMResults 示例**: `statsmodels/genmod/generalized_linear_model.py:1839-1840`
```python
self._data_attr_model = getattr(self, "_data_attr_model", [])
self._data_attr_model.append("mu")
```

**状态空间模型**: `statsmodels/tsa/statespace/sarimax.py:1899`
```python
self._data_attr_model.extend(["orig_endog", "orig_exog"])
```

### 2.3 双重路径的协作执行

**位置**: `statsmodels/base/model.py:2455-2495`

完整的 `remove_data()` 执行流程：

```python
def remove_data(self):
    """
    Remove data arrays to reduce memory usage when pickling.
    
    The lists of arrays to delete are maintained as attributes of
    the result and model instance, except for cached values.
    """
    
    # =========================================================================
    # 阶段一：类级缓存扫描 (Path 1)
    # =========================================================================
    
    cls = self.__class__
    
    # 使用 object.__getattribute__ 绕过描述符协议
    cls_attrs = {}
    for name in dir(cls):
        try:
            attr = object.__getattribute__(cls, name)
        except AttributeError:
            pass
        else:
            cls_attrs[name] = attr
    
    # 找出所有 cached_data 装饰的属性
    data_attrs = [x for x in cls_attrs if isinstance(cls_attrs[x], cached_data)]
    
    # 将这些属性在缓存中设为 None
    for name in data_attrs:
        self._cache[name] = None
    
    # =========================================================================
    # 辅助函数：递归清除嵌套属性
    # =========================================================================
    
    def wipe(obj, att):
        """
        清除嵌套属性，支持点分隔的路径
        
        例如: "data.exog" → getattr(getattr(obj, "data"), "exog") = None
        """
        # 分割路径
        p = att.split(".")
        att_ = p.pop(-1)  # 最后一个是属性名
        
        try:
            # 逐层获取对象
            obj_ = reduce(getattr, [obj] + p)
            # 设置为 None
            if hasattr(obj_, att_):
                setattr(obj_, att_, None)
        except AttributeError:
            # 属性不存在时静默失败
            pass
    
    # =========================================================================
    # 阶段二：实例级数据清单处理 (Path 2)
    # =========================================================================
    
    # 构造需要清除的完整属性路径列表
    
    # 1. model 独有的属性 (需要 "model." 前缀)
    model_only = ["model." + i for i in getattr(self, "_data_attr_model", [])]
    
    # 2. model._data_attr 中的所有属性 (需要 "model." 前缀)
    model_attr = ["model." + i for i in self.model._data_attr]
    
    # 3. 合并所有路径
    all_attrs = self._data_attr + model_attr + model_only
    
    # 遍历清除
    for att in all_attrs:
        # 跳过已在阶段一处理的 cached_data 属性
        if att in data_attrs:
            continue
        
        # 递归清除
        wipe(self, att)
    
    # =========================================================================
    # 阶段三：缓存中的数据属性
    # =========================================================================
    
    for key in self._data_in_cache:
        try:
            self._cache[key] = None
        except (AttributeError, KeyError):
            # 缓存不存在或键不存在时静默失败
            pass
```

### 2.4 路径协作图解

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    remove_data() 执行流程                                │
└─────────────────────────────────────────────────────────────────────────┘

                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 1: 类级缓存扫描                                                    │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  输入: dir(cls) 遍历类的所有属性名                                  │  │
│  │  方法: object.__getattribute__(cls, name) 绕过描述符               │  │
│  │  检查: isinstance(attr, cached_data)                               │  │
│  │  动作: self._cache[name] = None                                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  目的: 清除类定义时用 @cached_data 标记的缓存属性                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 2: 实例级数据清单处理                                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  三个清单的协作:                                                    │  │
│  │                                                                   │  │
│  │  self._data_attr                     model._data_attr              │  │
│  │  ┌─────────────────────┐            ┌─────────────────────┐      │  │
│  │  │ "exog"              │            │ "exog"              │      │  │
│  │  │ "endog"             │            │ "endog"             │      │  │
│  │  │ "pinv_wexog"        │            │ "pinv_wexog"        │      │  │
│  │  │ "wendog"            │            │ "wendog"            │      │  │
│  │  │ "wexog"             │            │ "wexog"             │      │  │
│  │  │ "weights"           │            │ "weights"           │      │  │
│  │  └─────────────────────┘            └─────────────────────┘      │  │
│  │           │                                    │                    │  │
│  │           │          加上 "model." 前缀        │                    │  │
│  │           │           ┌─────────────────┐      │                    │  │
│  │           │           │ "model.exog"    │      │                    │  │
│  │           │           │ "model.endog"   │◄─────┘                    │  │
│  │           │           │ "model.pinv_..."│                           │  │
│  │           │           └─────────────────┘                           │  │
│  │           │                                                         │  │
│  │           ▼                                                         │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │              wipe(self, att) 递归清除                        │   │  │
│  │  │  - 支持点分隔路径: "data.exog", "model.endog"               │   │  │
│  │  │  - 使用 reduce(getattr, [obj] + path_parts) 逐层访问       │   │  │
│  │  │  - setattr(obj_, attr_name, None) 设为 None                  │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 3: 缓存中的数据属性                                                │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  self._data_in_cache = ["fittedvalues", "resid", "wresid", ...] │  │
│  │                                                                   │  │
│  │  动作: for key in self._data_in_cache:                            │  │
│  │          self._cache[key] = None                                  │  │
│  │                                                                   │  │
│  │  注意: 这些可能已在阶段一通过 cached_data 处理过                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 数据移除后预测接口的功能边界

### 3.1 测试用例揭示的设计意图

**位置**: `statsmodels/regression/tests/test_predict.py:268-284`

```python
def test_predict_remove_data():
    # GH6887
    endog = [i + np.random.normal(scale=0.1) for i in range(100)]
    exog = list(range(100))
    model = WLS(endog, exog, weights=[1 for _ in range(100)]).fit()
    
    # 关键注释: we need to compute scale before we remove wendog, wexog
    assert isinstance(model.scale, float)  # scale 需要先计算
    
    model.remove_data()
    
    # 预测仍然可用
    scalar = model.get_prediction(1).predicted_mean
    pred = model.get_prediction([1])
    one_d = pred.predicted_mean
    assert_allclose(scalar, one_d)
    
    # 推断统计也可用
    pred.summary_frame()
```

**关键洞察**：

1. **`scale` 的预计算**：测试代码在 `remove_data()` 之前显式访问 `model.scale`
   - 原因：`scale` 是 `@cache_writable()` 属性，计算依赖 `wendog`, `wexog`
   - 设计意图：`scale` 是统计推断必需的，用户应该在数据移除前确保其已计算

2. **预测功能完全保留**：`get_prediction()` 和 `predict()` 在数据移除后正常工作

3. **推断统计保留**：`summary_frame()` 等方法正常工作

### 3.2 保留的功能分析

#### 3.2.1 预测功能

**为什么预测仍然可用？**

让我们看看 `predict()` 的实现：

```python
# statsmodels/regression/linear_model.py:417-443
class RegressionModel:
    def predict(self, params, exog=None):
        """
        Return linear predicted values from a design matrix.
        
        仅需要:
        1. params: 参数估计值
        2. exog: 新的解释变量 (用户提供)
        """
        if exog is None:
            exog = self.exog  # 这里会访问 self.exog，但用户调用时会提供新 exog
        return np.dot(exog, params)
```

**Results.predict() 方法**：

```python
# statsmodels/base/model.py:1209-1264
class Results:
    def predict(self, exog=None, transform=True, *args, **kwargs):
        """
        Call self.model.predict with self.params as the first argument.
        """
        # 处理公式转换 (如果模型是用公式创建的)
        exog, exog_index = self._transform_predict_exog(exog, transform=transform)
        
        # 核心预测逻辑: 只需要 params 和 exog
        predict_results = self.model.predict(self.params, exog, *args, **kwargs)
        
        # ... 返回结果
```

**关键点**：

- 当用户调用 `results.predict(new_exog)` 时：
  - `new_exog` 由用户提供，不依赖原始数据
  - `self.params` 是参数估计值，**不会被 `remove_data()` 清除**
  - `self.model.predict()` 只需执行矩阵乘法

#### 3.2.2 统计推断功能

**依赖已缓存的统计量**：

```python
# 以下属性依赖 normalized_cov_params 和 scale，不依赖原始数据

@cached_value
def bse(self):
    """标准误: sqrt(diag(cov_params))"""
    # 依赖 self.cov_params()，后者依赖 normalized_cov_params 和 scale
    return np.sqrt(np.diag(self.cov_params()))

@cached_value
def tvalues(self):
    """t统计量: params / bse"""
    return self.params / self.bse

@cached_value
def pvalues(self):
    """p值: 基于 t分布或正态分布"""
    if self.use_t:
        return stats.t.sf(np.abs(self.tvalues), df_resid) * 2
    else:
        return stats.norm.sf(np.abs(self.tvalues)) * 2
```

**get_prediction() 的完整功能**：

```python
# 即使数据移除后，以下仍然可用：
# - predicted_mean: 预测均值 (仅需要 params 和新 exog)
# - se_mean: 预测均值的标准误 (需要 cov_params)
# - ci_lower/ci_upper: 置信区间 (需要 se_mean 和 t分布临界值)
# - obs_ci_lower/obs_ci_upper: 预测区间 (需要 scale)
```

### 3.3 失效的功能分析

#### 3.3.1 依赖原始数据的缓存属性

```python
# 这些属性在 _data_in_cache 中，会被设为 None

@cache_readonly
def fittedvalues(self):
    """拟合值: 依赖 self.model.exog"""
    return self.model.predict(self.params, self.model.exog)
    #                                    ^^^^^^^^^^^^^^^^^^
    #                                    remove_data 后为 None

@cache_readonly
def resid(self):
    """残差: 依赖 self.model.endog"""
    return self.model.endog - self.model.predict(self.params, self.model.exog)
    #                  ^^^^^^^^^^^^^^^^^^
    #                  remove_data 后为 None

@cache_readonly
def wresid(self):
    """白化残差: 依赖 self.model.wendog"""
    return self.model.wendog - self.model.predict(self.params, self.model.wexog)
```

#### 3.3.2 依赖原始数据的方法

```python
# 重新计算对数似然
def llf(self):
    """对数似然: 依赖 self.model.endog"""
    return self.model.loglike(self.params)
    #                ^^^^^^^^^^^^^^
    #                loglike 需要 endog 计算

# 注意: 如果 llf 在 remove_data 之前已被访问并缓存，
#      则缓存值会保留（除非 cached_data 被标记）
```

### 3.4 功能边界总结表

| 功能类型 | 具体操作 | 数据移除后状态 | 依赖资源 |
|---------|---------|---------------|---------|
| **核心预测** | `predict(new_exog)` | ✅ 可用 | `params`, 新 `exog` |
| **完整预测** | `get_prediction(new_exog)` | ✅ 可用 | `params`, `cov_params`, `scale` |
| **参数推断** | `params`, `bse`, `tvalues`, `pvalues` | ✅ 可用 | `cov_params`, `scale` |
| **信息准则** | `aic`, `bic` | ⚠️ 条件可用 | 依赖 `llf` 是否已缓存 |
| **模型摘要** | `summary()` | ⚠️ 部分可用 | 可能依赖已失效的属性 |
| **残差分析** | `resid`, `fittedvalues` | ❌ 失效 | 依赖原始 `endog`, `exog` |
| **影响分析** | `outlier_test()`, `get_influence()` | ❌ 失效 | 依赖原始数据 |
| **似然比检验** | `compare_lr_test()` | ❌ 失效 | 依赖 `llnull` 等 |

### 3.5 设计取舍分析

**为什么这样设计？**

1. **预测优先**：
   - 序列化模型的主要用途是部署和预测
   - 预测只需要参数和新数据，不需要原始训练数据

2. **推断其次**：
   - 统计推断 (标准误、置信区间) 是模型结果的重要组成
   - 这些依赖协方差矩阵和尺度参数，这些在拟合时已计算

3. **诊断放弃**：
   - 残差分析、影响分析等诊断功能需要完整数据
   - 这些在模型部署阶段通常不需要

**这种设计的代价**：

```python
# 用户需要注意的"陷阱"

# 场景1: scale 未预计算
results = model.fit()
results.remove_data()
results.scale  # ❌ 可能失败，因为 scale 计算依赖 wendog

# 正确做法:
results = model.fit()
_ = results.scale  # 预计算 scale
results.remove_data()
results.scale  # ✅ 从缓存返回

# 场景2: llf 未预计算
results = model.fit()
results.remove_data()
results.llf  # ❌ 可能失败，因为 loglike 需要 endog

# 正确做法:
results = model.fit()
_ = results.llf  # 预计算 llf
results.remove_data()
results.llf  # ✅ 从缓存返回
```

---

## 4. 不同模型族的覆盖逻辑差异

### 4.1 基类实现

**LikelihoodModelResults.remove_data()**: `statsmodels/base/model.py:2455-2495`

这是所有似然模型结果的基础实现，包含：

1. 类级 `cached_data` 扫描
2. 实例级 `_data_attr`、`_data_attr_model`、`_data_in_cache` 处理
3. 通用的 `wipe()` 递归清除函数

### 4.2 GLM 的特殊扩展

**位置**: `statsmodels/genmod/generalized_linear_model.py:2656-2665`

```python
@Appender(base.LikelihoodModelResults.remove_data.__doc__)
def remove_data(self):
    """
    GLM 的特殊实现
    
    关键差异:
    1. 从 model._data_attr 动态扩展结果的 _data_attr
    2. 额外清除几个引用属性
    """
    # GLM has alias/reference in result instance
    # 过滤掉 "_data." 开头的属性（由 wipe 处理路径访问）
    self._data_attr.extend([i for i in self.model._data_attr if "_data." not in i])
    
    # 调用基类实现
    super(self.__class__, self).remove_data()
    
    # TODO: what are these in results?
    # 额外清除几个内部引用
    self._endog = None
    self._freq_weights = None
    self._var_weights = None
```

**为什么 GLM 需要特殊处理？**

让我们看看 GLM 的 `_data_attr` 结构：

```python
# statsmodels/genmod/generalized_linear_model.py:379-390
self._data_attr.extend(
    [
        "weights",
        "mu",
        "freq_weights",
        "var_weights",
        "iweights",
        "_offset_exposure",
        "n_trials",
    ]
)
```

**问题**：GLM 模型实例的某些属性在结果实例中也有别名/引用，需要双重清除。

**解决方案**：`self._data_attr.extend([i for i in self.model._data_attr if "_data." not in i])`

- 把 `model._data_attr` 中的属性（除了 `_data.` 路径）添加到结果的 `_data_attr`
- 这样 `wipe()` 会同时清除 `results.X` 和 `results.model.X`

### 4.3 状态空间模型的复杂处理

**位置**: `statsmodels/tsa/statespace/mlemodel.py:2822-2826`

```python
self._data_attr_model = getattr(self, "_data_attr_model", [])
self._data_attr_model.extend(["ssm"])  # 状态空间模型对象
self._data_attr.extend(extra_arrays)
self._data_attr.extend(["filter_results", "smoother_results"])
```

状态空间模型有更复杂的数据结构：
- `ssm`: 状态空间模型对象本身
- `filter_results`: 卡尔曼滤波结果
- `smoother_results`: 卡尔曼平滑结果

这些都需要在数据移除时清除。

### 4.4 各模型族数据清单对比

| 模型族 | _data_attr 扩展 | _data_in_cache 扩展 | 自定义 remove_data |
|--------|----------------|--------------------|-------------------|
| **Model (基类)** | `exog`, `endog`, `data.exog`, `data.endog` | (Results 基类) | 否 |
| **RegressionModel** | `pinv_wexog`, `wendog`, `wexog`, `weights` | - | 否 |
| **GLS** | `sigma`, `cholsigmainv` | - | 否 |
| **GLM** | `weights`, `mu`, `freq_weights`, `var_weights`, `iweights`, `_offset_exposure`, `n_trials` | `null`, `mu` | **是** (动态扩展) |
| **RLM** | - | `sresid` | 否 |
| **DiscreteModel** | (继承) | (继承) | 否 |
| **StateSpace** | `filter_results`, `smoother_results`, `extra_arrays` | - | 复杂 |

### 4.5 离散模型的特殊情况

**搜索结果**：在 `discrete_model.py` 中没有找到 `remove_data` 的覆盖实现。

**这意味着什么？**

离散模型（Logit, Probit, Poisson, NegativeBinomial 等）直接使用基类的 `remove_data()` 实现。

**需要确认的数据清单**：

```python
# DiscreteModel 继承自 LikelihoodModel，后者继承自 Model

# Model 基类已声明:
# _data_attr = ["exog", "endog", "data.exog", "data.endog", ...]

# 离散模型是否有额外的数据属性？
# 让我们检查...
```

从代码搜索来看，DiscreteModel 没有显式扩展 `_data_attr`，这意味着：

1. 它依赖基类的 `_data_attr` 声明
2. 如果有额外的数据属性需要清除，可能需要子类自己扩展

### 4.6 差异原因分析

**为什么 GLM 需要自定义 `remove_data()`？**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GLM 的特殊引用结构                                                       │
└─────────────────────────────────────────────────────────────────────────┘

GLM 模型实例                              GLMResults 结果实例
┌─────────────────────┐                   ┌─────────────────────┐
│ endog               │                   │ model               │──┐
│ exog                │                   │ params              │  │
│ weights             │◄──────────────────│ (引用 model)        │  │
│ mu                  │                   │                     │  │
│ freq_weights        │                   │ _endog              │  │
│ var_weights         │                   │ _freq_weights       │  │
│ iweights            │                   │ _var_weights        │  │
│ _offset_exposure    │                   │                     │  │
│ n_trials            │                   │  (这些是独立引用)   │  │
└─────────────────────┘                   └─────────────────────┘  │
                                          ▲                          │
                                          │                          │
                                          └──────────────────────────┘

问题:
1. results.model.weights 指向 model.weights
2. results._freq_weights 是独立的引用

基类 remove_data() 会清除:
- results.model.weights (通过 "model.weights" 路径)
- results._data_attr 中声明的属性

但 GLM 有额外的独立引用: _endog, _freq_weights, _var_weights
需要手动清除
```

**GLM 自定义 `remove_data()` 的两个目的**：

1. **动态扩展 `_data_attr`**：确保 model 的所有数据属性都被清除
2. **清除独立引用**：`_endog`, `_freq_weights`, `_var_weights` 这些结果实例的独立属性

---

## 5. 设计模式与架构权衡

### 5.1 应用的设计模式

#### 5.1.1 模板方法模式 (Template Method)

`remove_data()` 定义了算法骨架，子类可以通过以下方式扩展：

1. **扩展数据清单**：在 `__init__` 中 `_data_attr.extend(...)`
2. **覆盖方法**：如 `GLMResults.remove_data()` 动态扩展后调用 `super()`

```python
# 基类定义骨架
def remove_data(self):
    # Step 1: 类级扫描 (固定)
    # Step 2: 实例清单处理 (可扩展)
    # Step 3: 缓存清除 (固定)

# 子类扩展清单
class MyModelResults(LikelihoodModelResults):
    def __init__(self, ...):
        super().__init__(...)
        self._data_attr.extend(["my_extra_array"])  # 扩展清单
        self._data_in_cache.extend(["my_cached_data"])  # 扩展缓存清单

# 子类覆盖方法
class GLMResults(LikelihoodModelResults):
    def remove_data(self):
        self._data_attr.extend([...])  # 动态扩展
        super().remove_data()  # 调用骨架
        self._endog = None  # 额外清理
```

#### 5.1.2 组合模式 (Composite) - 嵌套属性清除

`wipe()` 函数通过 `reduce(getattr, ...)` 支持嵌套路径：

```python
def wipe(obj, att):
    """
    支持点分隔的路径: "data.exog", "model.endog"
    
    这是组合模式的简化实现，允许处理树状结构的属性
    """
    p = att.split(".")        # ["data", "exog"]
    att_ = p.pop(-1)          # "exog"
    
    # reduce(getattr, [obj] + ["data"]) = obj.data
    obj_ = reduce(getattr, [obj] + p)
    
    if hasattr(obj_, att_):
        setattr(obj_, att_, None)  # obj.data.exog = None
```

#### 5.1.3 策略模式 (Strategy) - 装饰器语义

虽然 `cache_readonly`, `cached_data`, `cached_value` 实现相同，但语义不同：

```python
# 三种装饰器定义不同的"策略"

@cache_readonly   # 策略: 通用缓存，不参与数据移除
def aic(self):
    return -2 * self.llf + 2 * self.df_modelwc

@cached_value     # 策略: 计算值，需要时预计算
def llf(self):
    return self.model.loglike(self.params)

@cached_data      # 策略: 数据属性，remove_data 时设为 None
def resid(self):   # 概念上的使用
    return self.model.endog - self.fittedvalues
```

### 5.2 架构权衡分析

#### 5.2.1 显式清单 vs 隐式扫描

| 方案 | 优点 | 缺点 |
|------|------|------|
| **显式清单** (`_data_attr`) | 明确、可控制、可继承扩展 | 容易遗漏、需要维护 |
| **隐式扫描** (`cached_data`) | 自动、标记即生效 | 需要特殊协议、不够直观 |

**Statsmodels 的选择：两者结合**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  双重识别路径的互补性                                                     │
└─────────────────────────────────────────────────────────────────────────┘

场景 1: 模型实例的数组属性
  - 这些是普通属性，不是缓存属性
  - 无法用装饰器标记
  - → 使用 _data_attr 显式声明

场景 2: 结果实例的缓存属性
  - 这些是 @property-like 的描述符
  - 存储在 _cache 字典中
  - → 可用 cached_data 装饰器标记 + 类级扫描
  - → 也可用 _data_in_cache 显式声明

场景 3: 模型独有的属性
  - 结果实例没有直接引用
  - 但 model.X 需要清除
  - → 使用 _data_attr_model + "model." 前缀
```

#### 5.2.2 内存压缩 vs 功能完整性

```
┌─────────────────────────────────────────────────────────────────────────┐
│  设计权衡：内存压缩 vs 功能完整性                                         │
└─────────────────────────────────────────────────────────────────────────┘

                    remove_data=False          remove_data=True
                    (完整功能)                  (内存优化)
                    ──────────────              ──────────────
内存占用:           大                          小
                    (包含所有原始数据数组)       (仅参数和协方差)

预测功能:           ✅ 完整                      ✅ 完整
                    (需要时才计算)               (需要时才计算)

推断功能:           ✅ 完整                      ⚠️ 条件完整
                    (所有统计量可用)              (依赖已缓存的统计量)

诊断功能:           ✅ 完整                      ❌ 不可用
                    (残差、影响分析等)            (依赖原始数据)

序列化大小:         大                          小
                    (可能数百MB)                  (可能几KB)

适用场景:           交互式分析、模型诊断         模型部署、批量预测
                    开发阶段                      生产阶段
```

#### 5.2.3 惰性计算的双刃剑

**缓存属性的设计**：

```python
@cache_readonly
def scale(self):
    """
    惰性计算: 首次访问时才计算
    
    优点:
    - 不使用就不计算，节省时间
    - 计算后缓存，重复访问高效
    
    缺点 (与 remove_data 结合时):
    - 如果 remove_data 前未访问，之后无法计算
    - 因为计算依赖的原始数据已被清除
    """
    wresid = self.wresid  # 依赖 self.model.wendog, wexog
    return np.dot(wresid, wresid) / self.df_resid
```

**这导致了用户需要注意的"预计算模式"**：

```python
# 推荐模式
results = model.fit()

# 数据移除前，预计算所有需要的统计量
_ = results.scale    # 预计算尺度参数
_ = results.llf      # 预计算对数似然
_ = results.aic      # 预计算信息准则 (依赖 llf)

# 现在可以安全地移除数据
results.remove_data()

# 之后这些统计量从缓存返回
print(results.scale)  # ✅
print(results.aic)    # ✅
```

**这是设计缺陷吗？**

不是。这是**有意的设计取舍**：

1. **默认惰性**：适合大多数场景（不进行 `remove_data`）
2. **显式预计算**：对于 `remove_data` 场景，用户需要明确哪些统计量需要保留
3. **内存优先**：未使用的统计量永远不会计算，节省内存和时间

### 5.3 与其他库的对比

| 库/框架 | 序列化策略 | 特点 |
|---------|-----------|------|
| **Statsmodels** | `save(remove_data=True)` | 可选移除数据，保留参数 |
| **scikit-learn** | `pickle/joblib` | 整个对象序列化，不移除数据 |
| **TensorFlow** | `SavedModel` | 仅保存计算图和参数，不保存训练数据 |
| **PyTorch** | `state_dict()` | 仅保存参数，需要手动重建模型 |
| **XGBoost/LightGBM** | 自定义二进制格式 | 仅保存树结构和参数 |

**Statsmodels 的独特之处**：

1. **完全兼容 pickle**：标准 Python 序列化
2. **可选数据移除**：用户决定是否压缩
3. **功能边界清晰**：明确什么可用、什么不可用
4. **向后兼容**：`remove_data=False` 是默认行为

---

## 6. 附录：完整执行流程图

### 6.1 remove_data() 完整执行流程

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        remove_data() 完整执行流程                               │
└──────────────────────────────────────────────────────────────────────────────┘

输入: Results 实例 (self)
输出: 修改后的实例 (数据属性设为 None)

                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Phase 1: 类级缓存扫描 (cached_data 装饰器识别)                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   cls = self.__class__                                                       │
│   for name in dir(cls):                                                      │
│       ┌────────────────────────────────────────────────────────────────┐   │
│       │ 关键: 使用 object.__getattribute__ 绕过描述符协议               │   │
│       │                                                                 │   │
│       │  getattr(cls, 'resid') → 触发 CachedAttribute.__get__()       │   │
│       │                              → 实际执行计算                      │   │
│       │                              → 返回计算结果                      │   │
│       │                                                                 │   │
│       │  object.__getattribute__(cls, 'resid')                         │   │
│       │                              → 返回描述符对象本身                │   │
│       │                              → 可检查 isinstance(obj, cached_data) │   │
│       └────────────────────────────────────────────────────────────────┘   │
│       try:                                                                  │
│           attr = object.__getattribute__(cls, name)                        │
│       except AttributeError:                                                │
│           continue                                                          │
│       cls_attrs[name] = attr                                                │
│                                                                              │
│   # 筛选 cached_data 装饰的属性                                              │
│   data_attrs = [x for x in cls_attrs                                        │
│                 if isinstance(cls_attrs[x], cached_data)]                  │
│                                                                              │
│   # 缓存设为 None                                                            │
│   for name in data_attrs:                                                    │
│       self._cache[name] = None                                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Phase 2: 构建清除路径列表                                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   # 路径类型 1: 结果实例独有的属性                                           │
│   # self._data_attr = ["exog", "endog", "pinv_wexog", ...]                │
│                                                                              │
│   # 路径类型 2: 模型实例的属性 (需加 "model." 前缀)                         │
│   # self.model._data_attr = ["exog", "endog", "weights", ...]             │
│   model_attr = ["model." + i for i in self.model._data_attr]               │
│   # → ["model.exog", "model.endog", "model.weights", ...]                 │
│                                                                              │
│   # 路径类型 3: 模型独有的属性 (结果实例没有直接引用)                        │
│   # self._data_attr_model = ["ssm", "filter_results", ...] (状态空间)      │
│   model_only = ["model." + i for i in getattr(self, "_data_attr_model", [])]│
│                                                                              │
│   # 合并所有路径                                                              │
│   all_attrs = self._data_attr + model_attr + model_only                     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Phase 3: 递归清除 (wipe 函数)                                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   def wipe(obj, att):                                                         │
│       # 示例: att = "model.data.orig_exog"                                   │
│                                                                              │
│       p = att.split(".")           # ["model", "data", "orig_exog"]        │
│       att_ = p.pop(-1)            # "orig_exog"                             │
│       # p 现在是 ["model", "data"]                                           │
│                                                                              │
│       # reduce 逐层获取对象                                                   │
│       # reduce(f, [a, b, c], init) = f(f(f(init, a), b), c)                │
│       # 这里 init = obj = self                                               │
│       #                                                                       │
│       # 第1步: getattr(self, "model") → self.model                           │
│       # 第2步: getattr(self.model, "data") → self.model.data                 │
│       # 结果: obj_ = self.model.data                                         │
│       obj_ = reduce(getattr, [obj] + p)                                      │
│                                                                              │
│       # 设置为 None                                                           │
│       # self.model.data.orig_exog = None                                     │
│       if hasattr(obj_, att_):                                                 │
│           setattr(obj_, att_, None)                                           │
│                                                                              │
│   # 遍历所有路径                                                              │
│   for att in all_attrs:                                                       │
│       if att in data_attrs:     # 已在 Phase 1 处理，跳过                   │
│           continue                                                            │
│       wipe(self, att)                                                          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Phase 4: 清除缓存中的数据属性                                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   # self._data_in_cache = ["fittedvalues", "resid", "wresid", ...]          │
│                                                                              │
│   for key in self._data_in_cache:                                            │
│       try:                                                                    │
│           self._cache[key] = None                                             │
│       except (AttributeError, KeyError):                                      │
│           pass    # 缓存不存在或键不存在时静默失败                            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
完成: 所有数据相关的数组属性已设为 None
      仅参数、协方差矩阵等推断必需的信息保留
```

### 6.2 数据清单的累积构建

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    _data_attr 在继承链中的累积                                │
└──────────────────────────────────────────────────────────────────────────────┘

Model.__init__ (statsmodels/base/model.py:107-110)
┌──────────────────────────────────────────────────────────────────────────────┐
│  self._data_attr = []                                                         │
│  self._data_attr.extend([                                                     │
│      "exog",                                                                   │
│      "endog",                                                                  │
│      "data.exog",           # 嵌套路径                                        │
│      "data.endog",           # 嵌套路径                                        │
│  ])                                                                            │
│  if "formula" not in kwargs:                                                   │
│      self._data_attr.extend([                                                 │
│          "data.orig_endog",    # 公式模型的原始数据                           │
│          "data.orig_exog",                                                    │
│      ])                                                                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ 继承
RegressionModel.__init__ (statsmodels/regression/linear_model.py:223)
┌──────────────────────────────────────────────────────────────────────────────┐
│  super().__init__(...)  # 先执行上面的                                        │
│                                                                              │
│  self._data_attr.extend([                                                     │
│      "pinv_wexog",           # 伪逆矩阵                                      │
│      "wendog",               # 白化后的因变量                                 │
│      "wexog",                # 白化后的自变量                                 │
│      "weights",              # 权重                                           │
│  ])                                                                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ 继承
GLS.__init__ (statsmodels/regression/linear_model.py:585)
┌──────────────────────────────────────────────────────────────────────────────┐
│  super().__init__(...)  # 先执行上面的                                        │
│                                                                              │
│  self._data_attr.extend([                                                     │
│      "sigma",                 # 协方差矩阵                                    │
│      "cholsigmainv",         # Cholesky 逆                                   │
│  ])                                                                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ 继承 (另一个分支)
GLM.__init__ (statsmodels/genmod/generalized_linear_model.py:379-390)
┌──────────────────────────────────────────────────────────────────────────────┐
│  super().__init__(...)  # 继承 Model → LikelihoodModel                        │
│                                                                              │
│  self._data_attr.extend([                                                     │
│      "weights",                                                               │
│      "mu",                    # 均值向量                                      │
│      "freq_weights",          # 频率权重                                      │
│      "var_weights",           # 方差权重                                      │
│      "iweights",              # 组合权重 = freq * var                        │
│      "_offset_exposure",      # 偏移 + 暴露项                                 │
│      "n_trials",              # 二项分布的试验次数                             │
│  ])                                                                            │
└──────────────────────────────────────────────────────────────────────────────┘

最终 _data_attr (以 GLS 为例):
[
    "exog", "endog", "data.exog", "data.endog",  # Model
    "data.orig_endog", "data.orig_exog",          # Model (非公式模型)
    "pinv_wexog", "wendog", "wexog", "weights",   # RegressionModel
    "sigma", "cholsigmainv"                         # GLS
]
```

---

## 参考文献

### 代码位置索引

| 模块 | 文件位置 | 关键内容 |
|------|---------|---------|
| **基类实现** | `statsmodels/base/model.py:2455-2495` | `LikelihoodModelResults.remove_data()` |
| **装饰器定义** | `statsmodels/tools/decorators.py:136-151` | `cached_data`, `cached_value` 语义 |
| **GLM 扩展** | `statsmodels/genmod/generalized_linear_model.py:2656-2665` | `GLMResults.remove_data()` |
| **测试用例** | `statsmodels/regression/tests/test_predict.py:268-284` | `test_predict_remove_data()` |
| **RegressionModel** | `statsmodels/regression/linear_model.py:223` | `_data_attr` 扩展 |
| **Results 基类** | `statsmodels/base/model.py:1127-1129` | `_data_in_cache` 初始化 |

### 相关 GitHub Issues

- **GH6887**: 测试用例中提到的问题 (remove_data 后预测功能)

---

**报告生成时间**: 2026  
**分析版本**: Statsmodels 0.14.x  
**分析范围**: 结果序列化与内存压缩机制
