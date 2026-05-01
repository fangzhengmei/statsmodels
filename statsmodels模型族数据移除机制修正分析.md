# Statsmodels 模型族数据移除机制：修正与深度分析

## 目录

1. [修正背景与范围](#1-修正背景与范围)
2. [GLM 独立引用属性的完整清单与耦合分析](#2-glm-独立引用属性的完整清单与耦合分析)
3. [状态空间模型：声明式扩展的验证](#3-状态空间模型声明式扩展的验证)
4. [命令式覆盖 vs 声明式扩展：设计对比](#4-命令式覆盖-vs-声明式扩展设计对比)
5. [修正后的完整机制总结](#5-修正后的完整机制总结)

---

## 1. 修正背景与范围

### 1.1 需要修正的两处不准确

在之前的分析中，存在两处需要核实和修正的描述：

| 序号 | 原描述 | 需要核实的问题 |
|------|--------|---------------|
| 1 | GLM 额外清零 `_endog`, `_freq_weights`, `_var_weights` | **完整清单是什么？** 这些属性反映了什么样的耦合关系？ |
| 2 | 状态空间模型"覆盖了数据移除方法" | **是否真的覆盖了？** 还是通过声明清单委托基类？两种方式的设计差异是什么？ |

### 1.2 核实方法

通过代码级别的精确分析：

1. **GLM 分析**：
   - 检查 `GLMResults.__init__` 中从模型复制了哪些属性
   - 检查 `GLMResults.remove_data()` 中手动清零了哪些属性
   - 检查 `_data_attr` 中声明了哪些属性

2. **状态空间模型分析**：
   - 搜索 `def remove_data` 是否存在覆盖
   - 检查 `_data_attr`, `_data_attr_model` 的声明位置
   - 分析这种设计的特点

---

## 2. GLM 独立引用属性的完整清单与耦合分析

### 2.1 完整属性清单的核实

#### 2.1.1 从模型复制的属性

**位置**: `statsmodels/genmod/generalized_linear_model.py:1805-1840`

```python
class GLMResults(base.LikelihoodModelResults):
    def __init__(
        self,
        model,
        params,
        normalized_cov_params,
        scale,
        cov_type="nonrobust",
        cov_kwds=None,
        use_t=None,
    ):
        super().__init__(
            model, params, normalized_cov_params=normalized_cov_params, scale=scale
        )
        self.family = model.family
        
        # ─────────────────────────────────────────────────────────────────
        # 关键：从模型复制数据属性到结果实例
        # ─────────────────────────────────────────────────────────────────
        
        self._endog = model.endog                    # 独立引用 1
        self.nobs = model.endog.shape[0]
        self._freq_weights = model.freq_weights      # 独立引用 2
        self._var_weights = model.var_weights        # 独立引用 3
        self._iweights = model.iweights              # 独立引用 4
        
        if isinstance(self.family, families.Binomial):
            self._n_trials = self.model.n_trials     # 独立引用 5 (仅二项分布)
        else:
            self._n_trials = 1
        
        # ... 其他初始化 ...
```

**共 5 个独立引用属性**：

| 属性 | 来源 | 说明 |
|------|------|------|
| `_endog` | `model.endog` | 因变量数据 |
| `_freq_weights` | `model.freq_weights` | 频率权重 |
| `_var_weights` | `model.var_weights` | 方差权重 |
| `_iweights` | `model.iweights` | 组合权重 = freq * var |
| `_n_trials` | `model.n_trials` | 二项分布的试验次数 |

#### 2.1.2 `_data_attr` 中声明的属性

**位置**: `statsmodels/genmod/generalized_linear_model.py:1834-1840`

```python
# for remove data and pickle without large arrays
self._data_attr.extend(
    ["results_constrained", "_freq_weights", "_var_weights", "_iweights"]
)
self._data_in_cache.extend(["null", "mu"])
self._data_attr_model = getattr(self, "_data_attr_model", [])
self._data_attr_model.append("mu")
```

**注意不一致性**：

| 属性 | 在 `__init__` 中复制 | 在 `_data_attr` 中声明 |
|------|---------------------|-----------------------|
| `_endog` | ✅ | ❌ **未声明** |
| `_freq_weights` | ✅ | ✅ |
| `_var_weights` | ✅ | ✅ |
| `_iweights` | ✅ | ✅ |
| `_n_trials` | ✅ | ❌ **未声明** |

#### 2.1.3 `remove_data()` 中实际清零的属性

**位置**: `statsmodels/genmod/generalized_linear_model.py:2656-2667`

```python
@Appender(base.LikelihoodModelResults.remove_data.__doc__)
def remove_data(self):
    # GLM has alias/reference in result instance
    # ─────────────────────────────────────────────────────────────────
    # 动态扩展：从 model._data_attr 复制属性到结果的 _data_attr
    # ─────────────────────────────────────────────────────────────────
    self._data_attr.extend([i for i in self.model._data_attr if "_data." not in i])
    
    # 调用基类实现
    super(self.__class__, self).remove_data()

    # TODO: what are these in results?
    # ─────────────────────────────────────────────────────────────────
    # 手动清零：这 5 个属性
    # ─────────────────────────────────────────────────────────────────
    self._endog = None
    self._freq_weights = None
    self._var_weights = None
    self._iweights = None
    self._n_trials = None
```

### 2.2 为什么需要手动清零？

**代码中的不一致性揭示了问题**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GLMResults 的数据引用结构                                               │
└─────────────────────────────────────────────────────────────────────────┘

GLM 模型实例                              GLMResults 结果实例
┌─────────────────────┐                   ┌─────────────────────────────┐
│                     │                   │                             │
│   endog             │                   │   model ────────────────────┼──► 引用 model
│   exog              │                   │                             │
│   freq_weights      │                   │   _endog = model.endog      │──► 独立复制！
│   var_weights       │                   │   _freq_weights = model.freq │──► 独立复制！
│   iweights          │                   │   _var_weights = model.var  │──► 独立复制！
│   n_trials          │                   │   _iweights = model.iweights │──► 独立复制！
│                     │                   │   _n_trials = model.n_trials │──► 独立复制！
└─────────────────────┘                   └─────────────────────────────┘

问题：
1. results.model.endog 是引用，会被基类的 wipe("model.endog") 清除
2. results._endog 是独立的数组引用，不会被自动清除！
3. 同样的问题存在于 _freq_weights, _var_weights, _iweights, _n_trials
```

**基类 `wipe()` 处理的路径**：

```python
# 基类 remove_data() 会处理：
# - "model.endog" → wipe(self, "model.endog") → self.model.endog = None
# - "model.freq_weights" → self.model.freq_weights = None
# - ...

# 但不会处理：
# - "_endog" → 除非在 self._data_attr 中声明
# - "_n_trials" → 除非在 self._data_attr 中声明
```

**`_data_attr` 声明的不一致**：

```python
# 在 __init__ 中：
self._data_attr.extend(
    ["results_constrained", "_freq_weights", "_var_weights", "_iweights"]
)
# 注意：_endog 和 _n_trials 没有被添加！

# 这意味着：
# - _freq_weights, _var_weights, _iweights 会被基类 wipe() 清除
# - _endog 和 _n_trials 不会被自动清除！
```

### 2.3 这反映了什么样的耦合关系？

#### 2.3.1 双重引用模式

GLMResults 采用了**双重引用**模式：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  引用类型 1: 通过 model 的间接引用                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   results.model.endog                                                    │
│   results.model.exog                                                     │
│   results.model.freq_weights                                             │
│   results.model.var_weights                                              │
│   results.model.iweights                                                 │
│   results.model.n_trials                                                 │
│                                                                          │
│   这些是通过 self.model 的间接引用，基类机制可以处理                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  引用类型 2: 结果实例的独立复制                                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   results._endog          ← 从 model.endog 复制                         │
│   results._freq_weights   ← 从 model.freq_weights 复制                  │
│   results._var_weights    ← 从 model.var_weights 复制                   │
│   results._iweights       ← 从 model.iweights 复制                      │
│   results._n_trials       ← 从 model.n_trials 复制                      │
│                                                                          │
│   这些是独立的引用！基类机制不会自动处理                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 2.3.2 为什么要复制这些属性？

代码中的 TODO 注释揭示了设计者的疑惑：

```python
# are these intermediate results needed or can we just
# call the model's attributes?

# 翻译：这些中间结果是必需的吗，还是我们可以直接调用模型的属性？
```

**可能的设计意图**：

| 原因 | 分析 |
|------|------|
| **性能优化** | 避免每次访问都通过 `self.model.X` 的间接引用 |
| **独立性** | 结果对象应该相对独立，不依赖模型实例的生命周期 |
| **历史遗留** | 早期设计决策，现在难以修改 |

#### 2.3.3 耦合关系图解

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GLMResults 与 GLM 的耦合关系                                            │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │   GLM 模型实例   │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
    ┌─────────────────┐    │        ┌─────────────────┐
    │  直接引用路径    │    │        │  独立复制路径    │
    └────────┬────────┘    │        └────────┬────────┘
             │             │                 │
             ▼             │                 ▼
    ┌──────────────────────────────────────────────────┐
    │            GLMResults 结果实例                     │
    │                                                      │
    │  self.model.endog      ◄── 引用，会被自动清除      │
    │  self.model.exog       ◄── 引用，会被自动清除      │
    │  ...                   ◄── ...                      │
    │                                                      │
    │  self._endog          ◄── 独立复制，需手动清除     │
    │  self._freq_weights   ◄── 独立复制，需手动清除     │
    │  self._var_weights    ◄── 独立复制，需手动清除     │
    │  self._iweights       ◄── 独立复制，需手动清除     │
    │  self._n_trials       ◄── 独立复制，需手动清除     │
    └──────────────────────────────────────────────────┘

耦合强度：██████████████████ 强耦合

问题：
1. 结果对象不仅通过 self.model 引用模型
2. 还直接复制了模型的数据属性
3. 这导致数据移除时需要双重清理
```

### 2.4 GLM `remove_data()` 的完整执行流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GLMResults.remove_data() 完整执行流程                                    │
└─────────────────────────────────────────────────────────────────────────┘

输入: GLMResults 实例
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 1: 动态扩展 _data_attr                                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   self._data_attr.extend([                                               │
│       i for i in self.model._data_attr                                  │
│         if "_data." not in i                                             │
│   ])                                                                      │
│                                                                          │
│   从 model._data_attr 复制属性（排除嵌套路径如 "data.exog"）             │
│   这确保 model 的数据属性也被纳入清除范围                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 2: 调用基类 remove_data()                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   super(self.__class__, self).remove_data()                             │
│                                                                          │
│   执行标准三阶段流程：                                                     │
│   1. 类级 cached_data 扫描                                                │
│   2. 实例级 _data_attr, _data_attr_model 处理                            │
│   3. 缓存中 _data_in_cache 的键清除                                       │
│                                                                          │
│   这会清除：                                                              │
│   - self.model.endog, self.model.exog, ...  (通过 "model.X" 路径)      │
│   - self._freq_weights, self._var_weights, self._iweights (已在 _data_attr) │
│   - 但不会清除 self._endog 和 self._n_trials！                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 3: 手动清零独立引用                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   self._endog = None          # 未在 _data_attr 中声明，需手动处理     │
│   self._freq_weights = None   # 虽然在 _data_attr 中，防御性再设一次   │
│   self._var_weights = None    # 同上                                    │
│   self._iweights = None       # 同上                                    │
│   self._n_trials = None      # 未在 _data_attr 中声明，需手动处理     │
│                                                                          │
│   注意：_freq_weights, _var_weights, _iweights 实际上会被基类清除       │
│        这里的手动设置是冗余的，可能是防御性编程                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
完成: 所有数据引用已清零
```

---

## 3. 状态空间模型：声明式扩展的验证

### 3.1 是否覆盖了 `remove_data()`？

**搜索验证**：

```
在 statsmodels/tsa/statespace/ 目录下搜索 "def remove_data"

结果: No matches found
```

**结论**：状态空间模型**没有**覆盖 `remove_data()` 方法。

### 3.2 如何实现数据移除？

**位置**: `statsmodels/tsa/statespace/mlemodel.py:2821-2825`

```python
# Handle removing data
self._data_attr_model = getattr(self, "_data_attr_model", [])
self._data_attr_model.extend(["ssm"])
self._data_attr.extend(extra_arrays)
self._data_attr.extend(["filter_results", "smoother_results"])
```

这是在**结果初始化阶段**进行的声明式扩展。

### 3.3 状态空间模型的数据清单

| 清单类型 | 声明内容 | 说明 |
|---------|---------|------|
| `_data_attr_model` | `["ssm"]` | 状态空间模型对象本身 |
| `_data_attr` | `extra_arrays` + `["filter_results", "smoother_results"]` | 滤波结果、平滑结果、额外数组 |

### 3.4 声明式扩展的执行流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│  状态空间模型：声明式扩展的执行流程                                        │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────┐
                    │  结果对象初始化阶段    │
                    └──────────┬───────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 1: 在 __init__ 中声明待清除清单                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   # mlemodel.py 中的代码                                                 │
│   self._data_attr_model = getattr(self, "_data_attr_model", [])         │
│   self._data_attr_model.extend(["ssm"])                                  │
│                                                                          │
│   self._data_attr.extend(extra_arrays)                                   │
│   self._data_attr.extend(["filter_results", "smoother_results"])        │
│                                                                          │
│   关键点：                                                                 │
│   - 这些是在结果实例创建时就声明好的                                      │
│   - 不需要覆盖 remove_data() 方法                                         │
│   - 完全委托基类的标准流程                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               │  稍后用户调用
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 2: 用户调用 remove_data()                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   results.remove_data()                                                  │
│   或                                                                      │
│   results.save("file.pkl", remove_data=True)                            │
│                                                                          │
│   实际上调用的是基类 LikelihoodModelResults.remove_data()                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 3: 基类标准流程处理声明的清单                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   # 基类 remove_data() 中的处理                                           │
│                                                                          │
│   # 处理 _data_attr_model（加上 "model." 前缀）                          │
│   model_only = ["model." + i for i in self._data_attr_model]           │
│   # → ["model.ssm"]                                                      │
│                                                                          │
│   # 处理 _data_attr                                                       │
│   for att in self._data_attr + ...:                                      │
│       wipe(self, att)                                                    │
│   # → 清除 extra_arrays, filter_results, smoother_results               │
│                                                                          │
│   # wipe() 递归清除                                                       │
│   # - wipe(self, "model.ssm") → self.model.ssm = None                   │
│   # - wipe(self, "filter_results") → self.filter_results = None         │
│   # - 等等                                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
完成: 所有声明的属性已被基类自动清除
```

### 3.5 状态空间模型为什么不需要覆盖？

**关键洞察**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  状态空间模型 vs GLM：数据引用结构对比                                    │
└─────────────────────────────────────────────────────────────────────────┘

状态空间模型结果实例                    GLMResults 结果实例
┌─────────────────────────────┐         ┌─────────────────────────────┐
│                             │         │                             │
│  model ─────────────────────┼──► ssm  │  model ────────────────────┼──► model
│                             │         │                             │
│  filter_results             │         │  _endog = model.endog      │──► 独立复制！
│  smoother_results           │         │  _freq_weights = model.freq │──► 独立复制！
│  extra_arrays               │         │  _var_weights = model.var  │──► 独立复制！
│                             │         │  _iweights = model.iweights │──► 独立复制！
│                             │         │  _n_trials = model.n_trials │──► 独立复制！
│                             │         │                             │
│  所有大数据都通过：           │         │  既有 model 引用，           │
│  - self.model.ssm           │         │  又有独立复制！               │
│  - self.filter_results      │         │                             │
│  - self.smoother_results    │         │                             │
│                             │         │                             │
│  可以通过声明清单自动处理：   │         │  需要手动覆盖 remove_data()  │
│  - _data_attr_model = ["ssm"]│         │  来清除独立复制的属性        │
│  - _data_attr = [...]       │         │                             │
└─────────────────────────────┘         └─────────────────────────────┘
```

**状态空间模型的设计优势**：

| 特点 | 说明 |
|------|------|
| **无独立复制** | 结果对象不复制模型的数据属性，只通过 `self.model` 引用 |
| **结果数据明确** | `filter_results`, `smoother_results` 等是结果对象自己的属性 |
| **声明即生效** | 在 `__init__` 中声明到 `_data_attr` 或 `_data_attr_model`，基类自动处理 |
| **无需要覆盖** | 不需要自定义 `remove_data()` 方法 |

---

## 4. 命令式覆盖 vs 声明式扩展：设计对比

### 4.1 两种方式的代码对比

#### 4.1.1 GLM：命令式覆盖 (Imperative)

```python
class GLMResults(base.LikelihoodModelResults):
    def __init__(self, ...):
        # ... 部分属性声明到 _data_attr ...
        self._data_attr.extend(
            ["results_constrained", "_freq_weights", "_var_weights", "_iweights"]
        )
        # 注意：_endog 和 _n_trials 没有声明！
    
    def remove_data(self):  # ‼️ 覆盖基类方法
        # 命令 1: 动态扩展 _data_attr
        self._data_attr.extend([
            i for i in self.model._data_attr if "_data." not in i
        ])
        
        # 命令 2: 调用基类
        super().remove_data()
        
        # 命令 3: 手动清零独立引用
        self._endog = None
        self._freq_weights = None
        self._var_weights = None
        self._iweights = None
        self._n_trials = None
```

#### 4.1.2 状态空间模型：声明式扩展 (Declarative)

```python
class MLEResults(base.LikelihoodModelResults):
    def __init__(self, ...):
        # ... 在初始化时声明所有需要清除的属性 ...
        
        # 声明 1: 模型独有的属性
        self._data_attr_model = getattr(self, "_data_attr_model", [])
        self._data_attr_model.extend(["ssm"])
        
        # 声明 2: 结果自有的属性
        self._data_attr.extend(extra_arrays)
        self._data_attr.extend(["filter_results", "smoother_results"])
        
        # ‼️ 不需要覆盖 remove_data()！
        # 基类会自动处理声明的清单
```

### 4.2 设计维度对比

| 设计维度 | GLM 命令式覆盖 | 状态空间 声明式扩展 |
|---------|---------------|-------------------|
| **核心思想** | 描述"怎么做" (How) | 描述"是什么" (What) |
| **代码位置** | 逻辑分散在 `__init__` 和 `remove_data()` | 逻辑集中在 `__init__` |
| **一致性** | ❌ 部分属性在 `_data_attr`，部分手动清零 | ✅ 所有属性都通过清单声明 |
| **可维护性** | ⚠️ 需要记住哪些属性需要特殊处理 | ✅ 新增属性只需添加到清单 |
| **可追溯性** | ⚠️ 清除逻辑分散，难以追踪 | ✅ 清除逻辑集中，一目了然 |
| **灵活性** | ✅ 可以实现复杂的动态逻辑 | ⚠️ 依赖基类能力，复杂场景受限 |

### 4.3 为什么 GLM 采用命令式？

**可能的原因分析**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GLM 命令式设计的历史/技术原因                                            │
└─────────────────────────────────────────────────────────────────────────┘

原因 1: 双重引用结构
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   GLMResults 既有 model 引用，又有独立复制的属性：                        │
│   - self._endog = model.endog                                            │
│   - self._freq_weights = model.freq_weights                              │
│   - ...                                                                   │
│                                                                          │
│   这些独立复制的属性如果不在 `__init__` 时正确声明到 `_data_attr`，      │
│   就需要在 `remove_data()` 中手动处理。                                   │
│                                                                          │
│   代码显示：_endog 和 _n_trials 没有在 _data_attr 中声明！              │
│   这可能是历史遗留的 bug 或设计疏忽。                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

原因 2: 动态扩展需求
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   GLM.remove_data() 中的动态扩展：                                        │
│                                                                          │
│   self._data_attr.extend([                                               │
│       i for i in self.model._data_attr if "_data." not in i             │
│   ])                                                                      │
│                                                                          │
│   这是在运行时从 model._data_attr 复制属性到结果的 _data_attr。          │
│   这在状态空间模型的声明式设计中也可以做到：                               │
│   - 在 __init__ 时就执行这个扩展                                          │
│   - 而不是等到 remove_data()                                              │
│                                                                          │
│   所以这不是必须用命令式的理由。                                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

原因 3: 历史演进与向后兼容
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   GLM 是 statsmodels 中较早的模块：                                        │
│   - 可能在声明式机制完善之前就已实现                                        │
│   - 为了向后兼容，保持了命令式覆盖                                          │
│   - 状态空间模型是后来加入的，采用了更现代的声明式设计                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.4 两种方式的适用场景

| 场景 | 推荐方式 | 理由 |
|------|---------|------|
| **简单数据结构** | 声明式 | 清晰、易维护 |
| **静态属性清单** | 声明式 | 初始化时一次声明即可 |
| **需要动态逻辑** | 命令式 | 如运行时决定清除哪些 |
| **有独立复制属性** | ⚠️ 需谨慎 | 确保在 `__init__` 时正确声明，否则需要命令式 |
| **新开发代码** | 声明式优先 | 符合现代设计理念 |
| **遗留代码** | 命令式（保持一致） | 避免破坏现有逻辑 |

### 4.5 理想的统一设计

如果重新设计 GLM，可以采用声明式：

```python
# 理想的 GLMResults 设计（声明式）

class GLMResults(base.LikelihoodModelResults):
    def __init__(self, ...):
        super().__init__(...)
        
        # 方案 A: 不独立复制，直接通过 model 引用
        # 这样就不需要额外处理了！
        # 但可能需要修改其他依赖 _endog 等属性的代码
        
        # 方案 B: 如果必须独立复制，确保完整声明
        self._endog = model.endog
        self._freq_weights = model.freq_weights
        self._var_weights = model.var_weights
        self._iweights = model.iweights
        self._n_trials = model.n_trials
        
        # 完整声明所有独立复制的属性到 _data_attr
        self._data_attr.extend([
            "results_constrained",
            "_endog",           # 之前遗漏的！
            "_freq_weights",
            "_var_weights",
            "_iweights",
            "_n_trials",        # 之前遗漏的！
        ])
        
        # 同时从 model._data_attr 扩展（在初始化时就做）
        self._data_attr.extend([
            i for i in self.model._data_attr if "_data." not in i
        ])
        
        # 声明缓存中的数据属性
        self._data_in_cache.extend(["null", "mu"])
        
        # 声明模型独有的属性
        self._data_attr_model = getattr(self, "_data_attr_model", [])
        self._data_attr_model.append("mu")
        
        # 不需要覆盖 remove_data()！
        # 基类会自动处理所有声明的属性
```

---

## 5. 修正后的完整机制总结

### 5.1 关键修正点

| 原描述 | 修正后 |
|--------|--------|
| GLM 额外清零 3 个属性 | **5 个属性**: `_endog`, `_freq_weights`, `_var_weights`, `_iweights`, `_n_trials` |
| 状态空间模型覆盖了 `remove_data()` | **没有覆盖**，完全通过声明清单委托基类 |
| GLM 和状态空间都是覆盖方式 | **GLM 是命令式覆盖**，**状态空间是声明式扩展** |

### 5.2 GLM 独立引用属性的完整分析

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GLMResults 数据引用完整清单                                              │
└─────────────────────────────────────────────────────────────────────────┘

从模型复制的独立属性（共 5 个）：
┌───────────────┬───────────────────┬─────────────────┬─────────────────┐
│ 属性名        │ 来源              │ 在 _data_attr   │ 在 remove_data   │
│               │                   │ 中声明了吗？    │ 中手动清零了吗？ │
├───────────────┼───────────────────┼─────────────────┼─────────────────┤
│ _endog        │ model.endog       │ ❌ 否           │ ✅ 是           │
│ _freq_weights │ model.freq_weights│ ✅ 是           │ ✅ 是（冗余）   │
│ _var_weights  │ model.var_weights │ ✅ 是           │ ✅ 是（冗余）   │
│ _iweights     │ model.iweights    │ ✅ 是           │ ✅ 是（冗余）   │
│ _n_trials     │ model.n_trials    │ ❌ 否           │ ✅ 是           │
└───────────────┴───────────────────┴─────────────────┴─────────────────┘

问题：
1. _endog 和 _n_trials 没有在 _data_attr 中声明
   → 必须在 remove_data() 中手动清零

2. _freq_weights, _var_weights, _iweights 已在 _data_attr 中声明
   → 基类会自动清除，手动清零是冗余的

3. 这种不一致反映了代码可能存在历史遗留问题
```

### 5.3 两种设计方式的对比总结

```
┌─────────────────────────────────────────────────────────────────────────┐
│  命令式覆盖 (GLM)                    vs    声明式扩展 (状态空间模型)      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   怎么做？                                是什么？                        │
│   ┌─────────────────────────┐             ┌─────────────────────────┐  │
│   │ def remove_data():      │             │ def __init__(...):      │  │
│   │     命令 1              │             │     声明 A              │  │
│   │     命令 2              │             │     声明 B              │  │
│   │     命令 3              │             │     声明 C              │  │
│   │     ...                 │             │                         │  │
│   │                         │             │  # 基类自动处理          │  │
│   └─────────────────────────┘             └─────────────────────────┘  │
│                                                                          │
│   逻辑分散                                逻辑集中                        │
│   部分声明 + 部分手动                      全部声明                        │
│   易出错、难维护                          清晰、易维护                    │
│   适合复杂动态场景                        适合静态清单场景                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.4 设计建议

对于新开发的模型结果类：

1. **优先采用声明式扩展**
   - 在 `__init__` 中将所有需要清除的属性声明到 `_data_attr` 或 `_data_attr_model`
   - 不需要覆盖 `remove_data()`

2. **避免独立复制数据属性**
   - 尽量通过 `self.model.X` 引用，而不是 `self._X = model.X` 复制
   - 如果必须复制，确保同时添加到 `_data_attr`

3. **对于 GLM 这样的遗留代码**
   - 保持现有行为不变（向后兼容）
   - 如果重构，建议统一为声明式：
     - 在 `__init__` 中将 `_endog` 和 `_n_trials` 添加到 `_data_attr`
     - 移除 `remove_data()` 中的冗余手动清零
     - 将动态扩展 `_data_attr` 的逻辑移到 `__init__`

---

## 附录：代码位置速查

| 模块 | 文件位置 | 关键内容 |
|------|---------|---------|
| **GLM 结果初始化** | `statsmodels/genmod/generalized_linear_model.py:1805-1840` | `_endog`, `_freq_weights` 等独立复制 |
| **GLM 数据移除** | `statsmodels/genmod/generalized_linear_model.py:2656-2667` | `remove_data()` 覆盖实现 |
| **状态空间声明** | `statsmodels/tsa/statespace/mlemodel.py:2821-2825` | `_data_attr`, `_data_attr_model` 声明 |
| **基类数据移除** | `statsmodels/base/model.py:2455-2495` | `LikelihoodModelResults.remove_data()` 标准流程 |

---

**报告生成时间**: 2026  
**分析版本**: Statsmodels 0.14.x  
**修正内容**: GLM 独立引用属性清单、状态空间模型实现方式确认、两种设计方式对比
