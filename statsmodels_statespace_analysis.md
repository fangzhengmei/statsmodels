# Statsmodels 状态空间框架实现分析

## 目录
1. [架构概述](#1-架构概述)
2. [状态空间模型定义](#2-状态空间模型定义)
3. [状态转移方程实现](#3-状态转移方程实现)
4. [Kalman 滤波实现](#4-kalman-滤波实现)
5. [后向平滑实现](#5-后向平滑实现)
6. [数据流转机制](#6-数据流转机制)
7. [完整流程图](#7-完整流程图)

---

## 1. 架构概述

### 1.1 核心类继承关系

```
Representation (状态空间表示)
    ↓
KalmanFilter (卡尔曼滤波)
    ↓
KalmanSmoother (卡尔曼平滑)
    ↓
SimulationSmoother (模拟平滑)
```

### 1.2 主要模块

| 模块 | 文件 | 主要功能 |
|------|------|----------|
| Representation | `representation.py` | 状态空间矩阵表示，数据绑定 |
| KalmanFilter | `kalman_filter.py` | 卡尔曼滤波算法实现 |
| KalmanSmoother | `kalman_smoother.py` | 卡尔曼平滑算法实现 |
| MLEModel | `mlemodel.py` | 最大似然估计模型封装 |
| Initialization | `initialization.py` | 状态初始化管理 |
| SimulationSmoother | `simulation_smoother.py` | 模拟平滑器 |

### 1.3 底层 Cython 实现

底层计算通过 Cython 实现，按数据类型区分：

```python
# 来自 tools.py
prefix_statespace_map = {
    "s": _representation.sStatespace,    # float32
    "d": _representation.dStatespace,    # float64
    "c": _representation.cStatespace,    # complex64
    "z": _representation.zStatespace     # complex128
}

prefix_kalman_filter_map = {
    "s": _kalman_filter.sKalmanFilter,
    "d": _kalman_filter.dKalmanFilter,
    "c": _kalman_filter.cKalmanFilter,
    "z": _kalman_filter.zKalmanFilter
}
```

---

## 2. 状态空间模型定义

### 2.1 数学形式

状态空间模型的一般形式由两个方程组成：

**观测方程 (Measurement Equation):**
```
y_t = Z_t α_t + d_t + ε_t
```

**状态转移方程 (Transition Equation):**
```
α_{t+1} = T_t α_t + c_t + R_t η_t
```

其中：
- `ε_t ~ N(0, H_t)` - 观测噪声
- `η_t ~ N(0, Q_t)` - 状态噪声

### 2.2 矩阵维度定义

在 `Representation` 类中定义了以下矩阵（`representation.py:191-203`）：

| 矩阵名 | 变量名 | 维度 | 说明 |
|--------|--------|------|------|
| Z | `design` | (k_endog × k_states × nobs) | 设计矩阵 |
| d | `obs_intercept` | (k_endog × nobs) | 观测方程截距 |
| H | `obs_cov` | (k_endog × k_endog × nobs) | 观测噪声协方差 |
| T | `transition` | (k_states × k_states × nobs) | 状态转移矩阵 |
| c | `state_intercept` | (k_states × nobs) | 状态方程截距 |
| R | `selection` | (k_states × k_posdef × nobs) | 选择矩阵 |
| Q | `state_cov` | (k_posdef × k_posdef × nobs) | 状态噪声协方差 |

### 2.3 MatrixWrapper 描述符

矩阵使用 `MatrixWrapper` 描述符进行管理，自动处理：
- Fortran 顺序内存布局
- 时间不变矩阵的自动扩展（最后一维为 1）
- 形状验证

```python
# representation.py:37-85
class MatrixWrapper:
    def __init__(self, name, attribute):
        self.name = name
        self.attribute = attribute
        self._attribute = "_" + attribute

    def __get__(self, obj, objtype):
        return getattr(obj, self._attribute, None)

    def __set__(self, obj, value):
        value = np.asarray(value, order="F")
        # 自动处理时间不变矩阵的扩展
        if value.ndim == 2:
            value = np.array(value[:, :, None], order="F")
        setattr(obj, self._attribute, value)
```

---

## 3. 状态转移方程实现

### 3.1 状态转移的数学表达

状态转移方程的核心是：

```
α_{t+1} = T_t α_t + c_t + R_t η_t
```

其中：
- `T_t` - 状态转移矩阵，控制状态如何从 t 时刻演化到 t+1 时刻
- `c_t` - 状态方程的截距项
- `R_t` - 选择矩阵，将噪声映射到状态空间
- `η_t ~ N(0, Q_t)` - 状态噪声

### 3.2 在 Kalman 滤波中的应用

状态转移发生在**预测步骤**中，在 `_kalman_filter.pyx` 的 Cython 实现中执行：

```python
# 预测步骤 (伪代码表示)
# α_{t|t-1} = T_{t-1} α_{t-1|t-1} + c_{t-1}
predicted_state = transition @ filtered_state_prev + state_intercept

# P_{t|t-1} = T_{t-1} P_{t-1|t-1} T_{t-1}' + R_{t-1} Q_{t-1} R_{t-1}'
predicted_cov = transition @ filtered_cov_prev @ transition.T + \
                selection @ state_cov @ selection.T
```

### 3.3 时间不变与时间变化矩阵

系统支持两种矩阵类型：

1. **时间不变矩阵**：最后一维为 1
   ```python
   # 例如 AR(1) 模型的转移矩阵始终相同
   transition = np.array([[0.5]])  # shape: (1, 1, 1)
   ```

2. **时间变化矩阵**：最后一维为 nobs
   ```python
   # 例如时变参数模型
   transition = np.random.randn(k_states, k_states, nobs)
   ```

在 `Representation.time_invariant` 属性中检查：

```python
# representation.py:757-774
@property
def time_invariant(self):
    if self._time_invariant is None:
        return (
            self._design.shape[2]
            == self._obs_intercept.shape[1]
            == self._obs_cov.shape[2]
            == self._transition.shape[2]
            == ...
        )
    return self._time_invariant
```

---

## 4. Kalman 滤波实现

### 4.1 滤波算法概述

Kalman 滤波是一个递归算法，包含两个主要步骤：

1. **预测步骤 (Prediction)**: 使用 t-1 时刻的信息预测 t 时刻
2. **更新步骤 (Update)**: 使用 t 时刻的观测值修正预测

### 4.2 标准 Kalman 滤波递推公式

**初始化：**
```
α_{1|0} = a_1  (初始状态均值)
P_{1|0} = P_1  (初始状态协方差)
```

**预测步骤 (t = 1, ..., n):**
```
# 状态预测
α_{t|t-1} = T_{t-1} α_{t-1|t-1} + c_{t-1}

# 协方差预测
P_{t|t-1} = T_{t-1} P_{t-1|t-1} T_{t-1}' + R_{t-1} Q_{t-1} R_{t-1}'

# 观测预测
y_{t|t-1} = Z_t α_{t|t-1} + d_t

# 预测误差协方差
F_t = Z_t P_{t|t-1} Z_t' + H_t

# 预测误差
v_t = y_t - y_{t|t-1}
```

**更新步骤:**
```
# Kalman 增益
K_t = P_{t|t-1} Z_t' F_t^{-1}

# 状态更新
α_{t|t} = α_{t|t-1} + K_t v_t

# 协方差更新
P_{t|t} = P_{t|t-1} - K_t Z_t P_{t|t-1}
```

### 4.3 代码实现流程

在 `KalmanFilter.filter()` 方法中（`kalman_filter.py:930-984`）：

```python
def filter(self, ...):
    # 1. 处理内存保存选项
    if conserve_memory is None:
        conserve_memory = self.conserve_memory | MEMORY_NO_SMOOTHING
    
    # 2. 执行滤波
    kfilter = self._filter(...)
    
    # 3. 创建结果对象
    results = self.results_class(self)
    results.update_representation(self)
    results.update_filter(kfilter)
    
    return results
```

底层调用 `_filter()` 方法：

```python
# kalman_filter.py:909-928
def _filter(self, ...):
    # 1. 初始化滤波器
    prefix, dtype, create_filter, create_statespace = (
        self._initialize_filter(...)
    )
    kfilter = self._kalman_filters[prefix]
    
    # 2. 初始化状态
    self._initialize_state(prefix=prefix, complex_step=complex_step)
    
    # 3. 运行滤波 (调用 Cython 实现)
    kfilter()  # 这是 __call__ 方法
    
    return kfilter
```

### 4.4 滤波方法选项

系统支持多种滤波方法（通过位掩码控制）：

```python
# kalman_filter.py:19-29
FILTER_CONVENTIONAL = 0x01     # 常规滤波
FILTER_EXACT_INITIAL = 0x02    # 精确初始滤波
FILTER_AUGMENTED = 0x04        # 增强滤波
FILTER_SQUARE_ROOT = 0x08      # 平方根滤波
FILTER_UNIVARIATE = 0x10       # 单变量滤波
FILTER_COLLAPSED = 0x20        # 压缩滤波
FILTER_EXTENDED = 0x40         # 扩展 Kalman 滤波
FILTER_UNSCENTED = 0x80        # 无迹 Kalman 滤波
FILTER_CONCENTRATED = 0x100    # 集中似然
FILTER_CHANDRASEKHAR = 0x200   # Chandrasekhar 递归
```

### 4.5 内存管理选项

控制保存哪些中间结果：

```python
# kalman_filter.py:39-56
MEMORY_STORE_ALL = 0              # 保存所有
MEMORY_NO_FORECAST_MEAN = 0x01   # 不保存预测均值
MEMORY_NO_FORECAST_COV = 0x02    # 不保存预测协方差
MEMORY_NO_PREDICTED_MEAN = 0x04  # 不保存预测状态均值
MEMORY_NO_PREDICTED_COV = 0x08   # 不保存预测状态协方差
MEMORY_NO_FILTERED_MEAN = 0x10   # 不保存滤波状态均值
MEMORY_NO_FILTERED_COV = 0x20    # 不保存滤波状态协方差
MEMORY_NO_LIKELIHOOD = 0x40      # 不保存似然值
MEMORY_NO_GAIN = 0x80             # 不保存 Kalman 增益
MEMORY_NO_SMOOTHING = 0x100       # 不保存平滑所需变量
```

---

## 5. 后向平滑实现

### 5.1 平滑算法概述

Kalman 平滑使用**全部观测数据** `y_1, ..., y_n` 来估计每个时刻的状态，相比滤波只使用到 `y_1, ..., y_t` 的信息更全面。

### 5.2 RTS 平滑器 (Rauch-Tung-Striebel)

**后向递推公式 (t = n-1, ..., 1):**

```
# 平滑增益
L_t = P_{t|t} T_t' P_{t+1|t}^{-1}

# 平滑状态均值
α_{t|n} = α_{t|t} + L_t (α_{t+1|n} - α_{t+1|t})

# 平滑状态协方差
P_{t|n} = P_{t|t} + L_t (P_{t+1|n} - P_{t+1|t}) L_t'
```

**初值 (t = n):**
```
α_{n|n} = α_{n|n}  (滤波结果)
P_{n|n} = P_{n|n}  (滤波结果)
```

### 5.3 代码实现流程

在 `KalmanSmoother.smooth()` 方法中（`kalman_smoother.py:376-428`）：

```python
def smooth(self, ...):
    # 1. 先运行滤波
    kfilter = self._filter(**kwargs)
    
    # 2. 创建结果对象
    results = self.results_class(self)
    if update_representation:
        results.update_representation(self)
    if update_filter:
        results.update_filter(kfilter)
    
    # 3. 运行平滑
    if smoother_output is None:
        smoother_output = self.smoother_output
    smoother = self._smooth(smoother_output, results=results, **kwargs)
    
    # 4. 更新平滑结果
    if update_smoother:
        results.update_smoother(smoother)
    
    return results
```

底层调用 `_smooth()` 方法：

```python
# kalman_smoother.py:354-374
def _smooth(self, ...):
    # 1. 初始化平滑器
    prefix, dtype, create_smoother, create_filter, create_statespace = (
        self._initialize_smoother(...)
    )
    
    # 2. 获取平滑器对象
    smoother = self._kalman_smoothers[prefix]
    
    # 3. 运行平滑 (调用 Cython 实现)
    smoother()  # __call__ 方法
    
    return smoother
```

### 5.4 平滑输出选项

控制平滑器计算哪些结果：

```python
# kalman_smoother.py:21-29
SMOOTHER_STATE = 0x01              # 平滑状态
SMOOTHER_STATE_COV = 0x02          # 平滑状态协方差
SMOOTHER_DISTURBANCE = 0x04        # 平滑扰动
SMOOTHER_DISTURBANCE_COV = 0x08    # 平滑扰动协方差
SMOOTHER_STATE_AUTOCOV = 0x10      # 平滑状态自协方差
SMOOTHER_ALL = (
    SMOOTHER_STATE | SMOOTHER_STATE_COV | 
    SMOOTHER_DISTURBANCE | SMOOTHER_DISTURBANCE_COV | 
    SMOOTHER_STATE_AUTOCOV
)
```

### 5.5 结果更新机制

在 `SmootherResults.update_smoother()` 方法中处理平滑结果：

```python
# kalman_smoother.py:607-738
def update_smoother(self, smoother):
    # 根据选项复制相应的输出
    attributes = []
    
    if self.smoother_state or self.smoother_disturbance:
        attributes.append("scaled_smoothed_estimator")
    if self.smoother_state_cov or self.smoother_disturbance_cov:
        attributes.append("scaled_smoothed_estimator_cov")
    if self.smoother_state:
        attributes.append("smoothed_state")
    if self.smoother_state_cov:
        attributes.append("smoothed_state_cov")
    # ...
    
    # 复制数据到结果对象
    for name in self._smoother_attributes:
        if name in attributes:
            setattr(self, name, np.array(getattr(smoother, name, None), copy=True))
```

---

## 6. 数据流转机制

### 6.1 完整数据流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         用户调用层                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │  fit()   │  │ filter() │  │ smooth() │  │ loglike()│               │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
└───────┼──────────────┼──────────────┼──────────────┼─────────────────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MLEModel 层                                      │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  1. handle_params() - 处理参数转换                                │   │
│  │  2. update() - 更新状态空间矩阵 (关键步骤)                        │   │
│  │  3. 调用底层 ssm.filter() / ssm.smooth()                          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└───────┬───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      KalmanFilter/KalmanSmoother 层                      │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  _initialize_filter() / _initialize_smoother()                   │   │
│  │  ├─ _initialize_representation() - 初始化矩阵表示                 │   │
│  │  ├─ 创建/获取 Cython 滤波器对象                                    │   │
│  │  └─ _initialize_state() - 初始化状态                              │   │
│  │                                                                    │   │
│  │  执行滤波/平滑: kfilter() / smoother()                            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└───────┬───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Cython 底层实现层                                 │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  _representation.pyx: {s,d,c,z}Statespace                        │   │
│  │  └─ 矩阵存储和基础操作                                              │   │
│  │                                                                    │   │
│  │  _kalman_filter.pyx: {s,d,c,z}KalmanFilter                       │   │
│  │  └─ Kalman 滤波递推计算 (核心数值计算)                             │   │
│  │                                                                    │   │
│  │  _kalman_smoother.pyx: {s,d,c,z}KalmanSmoother                   │   │
│  │  └─ Kalman 平滑后向递推计算                                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└───────┬───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         结果收集层                                        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  FilterResults / SmootherResults                                  │   │
│  │  ├─ update_representation() - 复制模型表示                        │   │
│  │  ├─ update_filter() - 复制滤波结果                                │   │
│  │  └─ update_smoother() - 复制平滑结果                              │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 关键数据流详解

#### 6.2.1 参数更新流 (`update()` 方法)

在 `MLEModel` 的子类中（如 `SARIMAX`），`update()` 方法负责将参数映射到状态空间矩阵：

```python
# 伪代码示例 (SARIMAX.update())
def update(self, params, **kwargs):
    # 1. 解包参数
    ar_params = params[:p]    # AR 系数
    ma_params = params[p:p+q]  # MA 系数
    variance = params[-1]      # 噪声方差
    
    # 2. 构造转移矩阵 T (companion form)
    self.ssm['transition'] = companion_matrix(ar_params)
    
    # 3. 构造设计矩阵 Z
    self.ssm['design'] = [1, 0, ..., 0]  # 选择第一个状态
    
    # 4. 构造选择矩阵 R 和状态协方差 Q
    self.ssm['selection'] = eye(k_states)[:, :k_posdef]
    self.ssm['state_cov'] = variance * eye(k_posdef)
    
    # 5. 观测协方差 H (通常为 0)
    self.ssm['obs_cov'] = zeros((k_endog, k_endog))
```

#### 6.2.2 状态初始化流

状态初始化在 `_initialize_state()` 中处理：

```python
# representation.py:1111-1126
def _initialize_state(self, prefix=None, complex_step=False):
    if isinstance(self.initialization, Initialization):
        if not self.initialization.initialized:
            raise RuntimeError("Initialization is incomplete.")
        # 调用 Cython 初始化
        self._statespaces[prefix].initialize(
            self.initialization, complex_step=complex_step
        )
```

初始化支持四种类型：

| 类型 | 说明 | 代码位置 |
|------|------|----------|
| `'known'` | 已知均值和协方差 | `initialization.py:526-550` |
| `'diffuse'` | 精确扩散初始化 | `initialization.py:551-558` |
| `'approximate_diffuse'` | 近似扩散 (大方差) | `initialization.py:559-561` |
| `'stationary'` | 平稳初始化 (解 Lyapunov 方程) | `initialization.py:562-568` |

#### 6.2.3 滤波结果数据流

滤波结果通过 `FilterResults.update_filter()` 收集：

```python
# kalman_filter.py (在 FilterResults 类中)
def update_filter(self, kalman_filter):
    # 复制滤波结果
    self.filtered_state = np.array(kalman_filter.filtered_state, copy=True)
    self.filtered_state_cov = np.array(kalman_filter.filtered_state_cov, copy=True)
    self.predicted_state = np.array(kalman_filter.predicted_state, copy=True)
    self.predicted_state_cov = np.array(kalman_filter.predicted_state_cov, copy=True)
    
    # 复制预测相关
    self.forecasts = np.array(kalman_filter.forecasts, copy=True)
    self.forecasts_error = np.array(kalman_filter.forecasts_error, copy=True)
    self.forecasts_error_cov = np.array(kalman_filter.forecasts_error_cov, copy=True)
    
    # 复制似然和增益
    self.loglikelihood = np.array(kalman_filter.loglikelihood, copy=True)
    self.kalman_gain = np.array(kalman_filter.gain, copy=True)
```

#### 6.2.4 平滑结果数据流

平滑结果通过 `SmootherResults.update_smoother()` 收集（见 5.5 节）。

### 6.3 矩阵数据的内部表示

#### 6.3.1 Python 层存储

在 `Representation` 类中，矩阵以 `_` 为前缀的属性存储：

```python
# representation.py:336-341
# 初始化时创建
for name, shape in self.shapes.items():
    if name == "obs":
        continue
    setattr(self, "_" + name, np.zeros(shape, dtype=dtype, order="F"))
```

例如：
- `self._design` → 设计矩阵 Z
- `self._transition` → 转移矩阵 T
- `self._selection` → 选择矩阵 R
- `self._state_cov` → 状态协方差 Q

#### 6.3.2 Cython 层共享

通过 `_initialize_representation()` 创建共享内存：

```python
# representation.py:1042-1109
def _initialize_representation(self, prefix=None):
    # 1. 确定数据类型
    dtype = tools.prefix_dtype_map[prefix]
    
    # 2. 创建/更新 dtype 特定的矩阵副本
    if prefix not in self._representations:
        self._representations[prefix] = {}
        for matrix in self.shapes.keys():
            if matrix == "obs":
                self._representations[prefix][matrix] = self.obs.astype(dtype)
            else:
                self._representations[prefix][matrix] = getattr(
                    self, "_" + matrix
                ).astype(dtype)
    
    # 3. 创建/获取 Cython Statespace 对象
    if create:
        cls = self.prefix_statespace_map[prefix]
        self._statespaces[prefix] = cls(
            self._representations[prefix]["obs"],
            self._representations[prefix]["design"],
            self._representations[prefix]["obs_intercept"],
            self._representations[prefix]["obs_cov"],
            self._representations[prefix]["transition"],
            self._representations[prefix]["state_intercept"],
            self._representations[prefix]["selection"],
            self._representations[prefix]["state_cov"],
        )
```

**关键点**：NumPy 数组和 Cython 数组共享内存，修改 Python 层的矩阵会直接影响 Cython 层的计算。

### 6.4 结果对象的数据存储

#### FilterResults 存储的滤波数据

| 属性 | 形状 | 说明 |
|------|------|------|
| `filtered_state` | (k_states, nobs+1) | 滤波状态均值 α_{t\|t} |
| `filtered_state_cov` | (k_states, k_states, nobs+1) | 滤波状态协方差 P_{t\|t} |
| `predicted_state` | (k_states, nobs+1) | 预测状态均值 α_{t\|t-1} |
| `predicted_state_cov` | (k_states, k_states, nobs+1) | 预测状态协方差 P_{t\|t-1} |
| `forecasts` | (k_endog, nobs) | 一步预测 y_{t\|t-1} |
| `forecasts_error` | (k_endog, nobs) | 预测误差 v_t |
| `forecasts_error_cov` | (k_endog, k_endog, nobs) | 预测误差协方差 F_t |
| `kalman_gain` | (k_states, k_endog, nobs) | Kalman 增益 K_t |
| `loglikelihood` | (nobs,) | 每期对数似然 |

#### SmootherResults 额外存储的平滑数据

| 属性 | 形状 | 说明 |
|------|------|------|
| `smoothed_state` | (k_states, nobs) | 平滑状态均值 α_{t\|n} |
| `smoothed_state_cov` | (k_states, k_states, nobs) | 平滑状态协方差 P_{t\|n} |
| `smoothed_state_autocov` | (k_states, k_states, nobs-1) | 平滑状态自协方差 |
| `smoothed_measurement_disturbance` | (k_endog, nobs) | 平滑观测扰动 |
| `smoothed_state_disturbance` | (k_posdef, nobs) | 平滑状态扰动 |
| `scaled_smoothed_estimator` | (k_states, nobs+1) | 缩放平滑估计量 r_t |
| `scaled_smoothed_estimator_cov` | (k_states, k_states, nobs+1) | 缩放平滑估计量协方差 N_t |

---

## 7. 完整流程图

### 7.1 从模型定义到估计的完整流程

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           第一阶段：模型定义                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  用户代码:                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  from statsmodels.tsa.statespace import SARIMAX                      │   │
│  │  model = SARIMAX(endog, order=(1,0,1))  # 创建模型实例              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  SARIMAX.__init__()                                                         │
│  ├─ 调用父类 MLEModel.__init__()                                            │
│  │   ├─ prepare_data() - 准备数据                                          │
│  │   └─ initialize_statespace() - 初始化状态空间                            │
│  │        └─ 创建 SimulationSmoother 实例 (self.ssm)                       │
│  │             └─ 继承链: SimulationSmoother → KalmanSmoother →          │
│  │                          KalmanFilter → Representation                   │
│  │                                                                           │
│  └─ 设置模型特定的状态空间矩阵结构 (AR/MA 阶数决定 k_states)               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           第二阶段：参数估计 (fit())                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  用户代码:                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  results = model.fit()  # 开始拟合                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  MLEModel.fit()                                                              │
│  ├─ 获取初始参数 start_params                                                 │
│  ├─ 调用 scipy.optimize 进行优化                                             │
│  │   └─ 目标函数: 最大化 loglike(params)                                      │
│  │                                                                           │
│  └─ 优化完成后，执行最终的滤波和平滑                                          │
│       ├─ func = self.filter 或 self.smooth                                  │
│       └─ res = func(mlefit.params, ...)                                     │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────   │
│  优化循环中的每次迭代:                                                        │
│                                                                              │
│  loglike(params)                                                             │
│  ├─ handle_params(params) - 参数验证和转换                                   │
│  ├─ update(params) - 更新状态空间矩阵 (关键!)                                │
│  │   └─ SARIMAX.update():                                                    │
│  │      ├─ 从 params 提取 AR/MA 系数和方差                                    │
│  │      └─ 设置 self.ssm['transition'], self.ssm['selection'], 等          │
│  │                                                                           │
│  └─ self.ssm.loglike() - 执行滤波计算似然                                    │
│       └─ 调用 KalmanFilter.loglike()                                         │
│            └─ 调用 _filter() 执行 Cython 滤波                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         第三阶段：滤波计算 (Kalman Filter)                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  KalmanFilter._filter()                                                      │
│  ├─ _initialize_filter()                                                     │
│  │   ├─ _initialize_representation()                                         │
│  │   │   ├─ 确定 dtype (BLAS prefix: 's','d','c','z')                       │
│  │   │   ├─ 创建/更新 self._representations[prefix]                          │
│  │   │   │   └─ 复制矩阵到 dtype 特定的数组                                   │
│  │   │   └─ 创建 self._statespaces[prefix] (Cython 对象)                     │
│  │   │       └─ 与 Python 数组共享内存                                        │
│  │   │                                                                       │
│  │   ├─ 创建 self._kalman_filters[prefix] (Cython 滤波器)                   │
│  │   │   └─ 接收 _statespace 对象作为参数                                     │
│  │   │                                                                       │
│  │   └─ _initialize_state()                                                  │
│  │       └─ 调用 Initialization 对象设置初始状态                              │
│  │           ├─ initial_state = a_1                                          │
│  │           └─ initial_state_cov = P_1                                      │
│  │                                                                           │
│  └─ kfilter()  # 调用 Cython KalmanFilter.__call__()                         │
│       └─ 执行滤波递推:                                                        │
│           for t in 1..nobs:                                                  │
│               # 预测步骤                                                      │
│               α_{t|t-1} = T α_{t-1|t-1} + c                                │
│               P_{t|t-1} = T P_{t-1|t-1} T' + R Q R'                        │
│                                                                              │
│               # 计算预测误差                                                  │
│               v_t = y_t - (Z α_{t|t-1} + d)                                │
│               F_t = Z P_{t|t-1} Z' + H                                      │
│                                                                              │
│               # 更新步骤                                                      │
│               K_t = P_{t|t-1} Z' F_t^{-1}                                   │
│               α_{t|t} = α_{t|t-1} + K_t v_t                                │
│               P_{t|t} = P_{t|t-1} - K_t Z P_{t|t-1}                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         第四阶段：平滑计算 (Kalman Smoother)                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  KalmanSmoother.smooth()                                                     │
│  ├─ 先执行滤波 (必须先有滤波结果)                                             │
│  │   └─ kfilter = self._filter(**kwargs)                                    │
│  │                                                                           │
│  ├─ _initialize_smoother()                                                   │
│  │   └─ 创建 self._kalman_smoothers[prefix] (Cython 平滑器)                 │
│  │       └─ 接收 _statespace 和 _kalman_filter 对象                          │
│  │                                                                           │
│  └─ smoother()  # 调用 Cython KalmanSmoother.__call__()                     │
│       └─ 执行后向平滑递推:                                                    │
│           # 初值: 使用滤波的最后结果                                          │
│           α_{n|n} = α_{n|n}  (来自滤波)                                     │
│           P_{n|n} = P_{n|n}  (来自滤波)                                     │
│                                                                              │
│           for t in n-1..1:                                                   │
│               # 平滑增益                                                      │
│               L_t = P_{t|t} T_t' P_{t+1|t}^{-1}                            │
│                                                                              │
│               # 平滑状态                                                      │
│               α_{t|n} = α_{t|t} + L_t (α_{t+1|n} - α_{t+1|t})             │
│                                                                              │
│               # 平滑协方差                                                    │
│               P_{t|n} = P_{t|t} + L_t (P_{t+1|n} - P_{t+1|t}) L_t'         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            第五阶段：结果收集                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MLEResults (继承 SmootherResults, FilterResults)                           │
│                                                                              │
│  results.update_representation(model)                                        │
│  ├─ 复制模型维度: nobs, k_endog, k_states, k_posdef                          │
│  ├─ 复制状态空间矩阵: design, transition, selection, state_cov, 等           │
│  └─ 复制初始化信息: initialization, initial_state, initial_state_cov          │
│                                                                              │
│  results.update_filter(kalman_filter)                                        │
│  ├─ 复制滤波结果: filtered_state, filtered_state_cov                         │
│  ├─ 复制预测结果: predicted_state, predicted_state_cov                        │
│  ├─ 复制预测相关: forecasts, forecasts_error, forecasts_error_cov            │
│  ├─ 复制 Kalman 增益: kalman_gain                                            │
│  └─ 复制似然值: loglikelihood                                                 │
│                                                                              │
│  results.update_smoother(smoother)                                           │
│  ├─ 复制平滑状态: smoothed_state, smoothed_state_cov                         │
│  ├─ 复制平滑扰动: smoothed_measurement_disturbance,                          │
│  │                  smoothed_state_disturbance                                │
│  ├─ 复制平滑协方差: smoothed_measurement_disturbance_cov,                    │
│  │                    smoothed_state_disturbance_cov                          │
│  └─ 复制辅助变量: scaled_smoothed_estimator,                                │
│                    scaled_smoothed_estimator_cov                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 关键类关系图

```
                    ┌──────────────────┐
                    │   Representation │
                    │  (状态空间表示)   │
                    └─────────┬────────┘
                              │ 继承
                              ▼
                    ┌──────────────────┐
                    │   KalmanFilter   │
                    │   (卡尔曼滤波)    │
                    └─────────┬────────┘
                              │ 继承
                              ▼
                    ┌──────────────────┐
                    │  KalmanSmoother  │
                    │   (卡尔曼平滑)    │
                    └─────────┬────────┘
                              │ 继承
                              ▼
                    ┌──────────────────┐
                    │SimulationSmoother│
                    │   (模拟平滑器)    │
                    └──────────────────┘
                              │
                              │ 组合
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          MLEModel                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  self.ssm = SimulationSmoother(...)                       │  │
│  │                                                             │  │
│  │  核心方法:                                                   │  │
│  │  - update(params) → 更新 self.ssm 的矩阵                   │  │
│  │  - filter(params) → 调用 self.ssm.filter()                 │  │
│  │  - smooth(params) → 调用 self.ssm.smooth()                 │  │
│  │  - loglike(params) → 调用 self.ssm.loglike()               │  │
│  │  - fit() → 优化参数，返回 MLEResults                        │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 继承
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    具体模型 (SARIMAX, VARMAX, 等)                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  覆盖方法:                                                   │  │
│  │  - __init__() → 设置特定模型的矩阵结构                      │  │
│  │  - update() → 实现参数到矩阵的映射                          │  │
│  │  - start_params → 提供初始参数                              │  │
│  │  - transform_params() / untransform_params() → 参数变换     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 附录

### A. 核心文件位置

| 文件 | 路径 | 说明 |
|------|------|------|
| `representation.py` | `statsmodels/tsa/statespace/` | 状态空间表示基类 |
| `kalman_filter.py` | `statsmodels/tsa/statespace/` | Kalman 滤波实现 |
| `kalman_smoother.py` | `statsmodels/tsa/statespace/` | Kalman 平滑实现 |
| `mlemodel.py` | `statsmodels/tsa/statespace/` | MLE 估计模型封装 |
| `initialization.py` | `statsmodels/tsa/statespace/` | 状态初始化 |
| `simulation_smoother.py` | `statsmodels/tsa/statespace/` | 模拟平滑 |
| `tools.py` | `statsmodels/tsa/statespace/` | 工具函数和类型映射 |
| `_kalman_filter.pyx` | `statsmodels/tsa/statespace/_filters/` | Cython 滤波实现 |
| `_kalman_smoother.pyx` | `statsmodels/tsa/statespace/_smoothers/` | Cython 平滑实现 |
| `_representation.pyx` | `statsmodels/tsa/statespace/` | Cython 表示实现 |

### B. 参考资料

1. Durbin, James, and Siem Jan Koopman. 2012. *Time Series Analysis by State Space Methods: Second Edition*. Oxford University Press.

2. Harvey, Andrew C. 1990. *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.

3. Rauch, H. E., F. Tung, and C. T. Striebel. 1965. "Maximum Likelihood Estimates of Linear Dynamic Systems." AIAA Journal 3 (8): 1445-1450.

---

*报告生成时间: 2026-05-01*
