# Statsmodels 状态空间模型子系统架构深度分析

## 1. 引言

Statsmodels 的状态空间模型子系统是一个设计精妙、层次清晰的软件工程典范。该子系统通过统一的抽象接口，将多种时间序列模型（ARIMA、SARIMAX、VARMAX、结构时间序列等）纳入统一的计算框架，实现了算法复用和高度可扩展性。

本报告深入分析该子系统的四个核心架构层面：
1. **系数矩阵的统一抽象表达**
2. **卡尔曼滤波器的多精度代码生成机制**
3. **滤波-平滑-仿真平滑器的层次递进关系**
4. **与通用似然模型体系的集成方式**

---

## 2. 系数矩阵的统一抽象表达

### 2.1 状态空间模型的标准形式

Statsmodels 采用 Durbin & Koopman (2012) 的标准状态空间表示：

```
观测方程:   y_t = Z_t α_t + d_t + ε_t,   ε_t ~ N(0, H_t)
状态方程:   α_{t+1} = T_t α_t + c_t + R_t η_t,  η_t ~ N(0, Q_t)
```

其中各矩阵的维度定义如下：

| 矩阵名 | Python属性名 | 维度 | 作用 |
|--------|-------------|------|------|
| Z_t | `design` | (k_endog × k_states × nobs) | 设计矩阵，连接状态与观测 |
| d_t | `obs_intercept` | (k_endog × nobs) | 观测方程截距项 |
| H_t | `obs_cov` | (k_endog × k_endog × nobs) | 观测噪声协方差 |
| T_t | `transition` | (k_states × k_states × nobs) | 状态转移矩阵 |
| c_t | `state_intercept` | (k_states × nobs) | 状态方程截距项 |
| R_t | `selection` | (k_states × k_posdef × nobs) | 选择矩阵，筛选状态噪声 |
| Q_t | `state_cov` | (k_posdef × k_posdef × nobs) | 状态噪声协方差 |

### 2.2 MatrixWrapper 描述符机制

核心抽象通过 `MatrixWrapper` 描述符类实现（位于 `representation.py:37-85`）：

```python
class MatrixWrapper:
    def __init__(self, name, attribute):
        self.name = name
        self.attribute = attribute
        self._attribute = "_" + attribute  # 内部存储属性名
    
    def __get__(self, obj, objtype):
        return getattr(obj, self._attribute, None)
    
    def __set__(self, obj, value):
        value = np.asarray(value, order="F")  # 强制 Fortran 顺序
        shape = obj.shapes[self.attribute]
        
        if len(shape) == 3:
            value = self._set_matrix(obj, value, shape)
        else:
            value = self._set_vector(obj, value, shape)
        
        setattr(obj, self._attribute, value)
        obj.shapes[self.attribute] = value.shape
```

**设计亮点**：
1. **自动维度扩展**：时间不变矩阵（2D）自动扩展为 3D（最后维度为 1）
2. **Fortran 顺序强制**：确保与底层 Cython 代码的内存布局兼容
3. **形状验证**：通过 `validate_matrix_shape` 确保用户输入符合预期维度

### 2.3 时变矩阵的优雅处理

系统支持时变矩阵（每个时间点有不同的系数矩阵）和时不变矩阵的统一表示：

```python
# 时不变矩阵: 最后维度为 1
self.shapes = {
    "design": (self.k_endog, self.k_states, 1),
    "obs_intercept": (self.k_endog, 1),
    # ...
}

# MatrixWrapper._set_matrix 中的自动扩展逻辑
if value.ndim == 2:
    value = np.array(value[:, :, None], order="F")
```

通过索引访问时，系统会智能处理时变/时不变情况：

```python
def __getitem__(self, key):
    # ...
    matrix = getattr(self, "_" + key)
    if matrix.shape[-1] == 1:  # 时不变
        return matrix[(slice(None),) * (matrix.ndim - 1) + (0,)]
    else:  # 时变
        return matrix
```

### 2.4 不同时序模型的矩阵填充示例

以 SARIMAX 模型为例，只需填充特定矩阵即可复用整套算法：

```python
# SARIMAX 模型的状态空间表示
# - transition 矩阵: 伴随矩阵形式，包含 AR 系数
# - selection 矩阵: 单位矩阵，选择 MA 部分
# - state_cov 矩阵: 包含创新方差
# - design 矩阵: [1, 0, ..., 0]，选择第一个状态

# Harvey 表示法的 AR(1) 模型示例:
# transition = [[φ1, 1],
#               [0,  0]]
# design = [[1, 0]]
# selection = [[1],
#              [0]]
# state_cov = [[σ²]]
```

**架构价值**：任何时序模型只要能转化为上述 8 个矩阵的形式，即可立即获得：
- 卡尔曼滤波
- 卡尔曼平滑
- 仿真平滑
- 最大似然估计
- 预测
- 诊断

---

## 3. 卡尔曼滤波器的多精度代码生成机制

### 3.1 问题背景

卡尔曼滤波涉及大量矩阵运算，需要：
1. **多种数值精度**：float32 (s)、float64 (d)、complex64 (c)、complex128 (z)
2. **高性能计算**：使用 BLAS/LAPACK 优化
3. **代码复用**：避免为每种精度重复编写相同逻辑

### 3.2 Tempita 模板引擎

Statsmodels 使用 **Tempita** 模板语言实现代码生成，核心配置位于 `_filters/meson.build`：

```python
# meson.build 中的模板处理
_conventional_pyx = custom_target(
    '_conventional_pyx',
    input: '_conventional.pyx.in',      # 模板文件
    output: '_conventional.pyx',         # 生成文件
    command: [tempita, '@INPUT@', '--outfile', '@OUTPUT@'],
)
```

### 3.3 模板结构解析

模板文件 `_conventional.pyx.in` 的核心结构：

```cython
{{py:
# 定义四种精度类型
TYPES = {
    "s": ("np.float32_t", "np.float32", "np.NPY_FLOAT32"),
    "d": ("np.float64_t", "float", "np.NPY_FLOAT64"),
    "c": ("np.complex64_t", "np.complex64", "np.NPY_COMPLEX64"),
    "z": ("np.complex128_t", "complex", "np.NPY_COMPLEX128"),
}
}}

# 循环生成每种精度的代码
{{for prefix, types in TYPES.items()}}
{{py:cython_type, dtype, typenum = types}}

# 类型组合处理（float32 降级为 float64 计算）
{{py:
combined_prefix = prefix
combined_cython_type = cython_type
if prefix == 'c':
    combined_prefix = 'z'
    combined_cython_type = 'np.complex128_t'
if prefix == 's':
    combined_prefix = 'd'
    combined_cython_type = 'np.float64_t'
}}

# 生成该精度的所有函数
cdef int {{prefix}}forecast_conventional({{prefix}}KalmanFilter kfilter, ...):
    # 使用 BLAS 前缀调用
    blas.{{prefix}}copy(...)
    blas.{{prefix}}gemv(...)
    # ...

cdef int {{prefix}}updating_conventional(...):
    # ...

cdef {{cython_type}} {{prefix}}loglikelihood_conventional(...):
    loglikelihood = -0.5*(model._k_endog*{{combined_prefix}}log(2*M_PI) + determinant)
    # ...

{{endfor}}
```

### 3.4 生成的代码示例

处理后生成的 `_conventional.pyx` 将包含四组函数：

```cython
# float32 版本
cdef int sforecast_conventional(sKalmanFilter kfilter, ...):
    blas.scopy(...)

# float64 版本  
cdef int dforecast_conventional(dKalmanFilter kfilter, ...):
    blas.dcopy(...)

# complex64 版本
cdef int cforecast_conventional(cKalmanFilter kfilter, ...):
    blas.ccopy(...)

# complex128 版本
cdef int zforecast_conventional(zKalmanFilter kfilter, ...):
    blas.zcopy(...)
```

### 3.5 运行时类型调度

通过 `tools.py` 中的前缀映射实现运行时动态选择：

```python
# tools.py:28-70
prefix_dtype_map = {
    "s": np.float32, "d": np.float64, 
    "c": np.complex64, "z": np.complex128
}

prefix_statespace_map = {
    "s": _representation.sStatespace,
    "d": _representation.dStatespace,
    "c": _representation.cStatespace,
    "z": _representation.zStatespace
}

prefix_kalman_filter_map = {
    "s": _kalman_filter.sKalmanFilter,
    "d": _kalman_filter.dKalmanFilter,
    "c": _kalman_filter.cKalmanFilter,
    "z": _kalman_filter.zKalmanFilter
}
```

运行时自动选择最优精度：

```python
# representation.py:731-747
@property
def prefix(self):
    """BLAS prefix of currently active representation matrices"""
    arrays = (
        self._design, self._obs_intercept, self._obs_cov,
        self._transition, self._state_intercept, 
        self._selection, self._state_cov,
    )
    if self.endog is not None:
        arrays = (self.endog,) + arrays
    return find_best_blas_type(arrays)[0]
```

### 3.6 架构优势

| 特性 | 说明 |
|------|------|
| **零代码重复** | 一套模板生成四种精度实现 |
| **BLAS 原生调用** | 直接使用 scipy.linalg.cython_blas |
| **类型安全** | Cython 静态类型检查 |
| **智能降级** | float32 内部使用 float64 计算保证精度 |
| **运行时透明** | 用户无需关心精度，自动选择最优 |

---

## 4. 滤波-平滑-仿真平滑器的层次递进关系

### 4.1 类继承层次结构

系统通过清晰的类继承链实现能力递进：

```
Representation (基础表示层)
      ↓
KalmanFilter (滤波层)
      ↓
KalmanSmoother (平滑层)
      ↓
SimulationSmoother (仿真平滑层)
```

### 4.2 各层能力分析

#### 第一层：Representation（`representation.py:87-1128`）

**核心职责**：状态空间模型的数学表示

```python
class Representation:
    """State space representation of a time series process"""
    
    # 8 个核心矩阵（通过 MatrixWrapper 管理）
    design = MatrixWrapper("design", "design")
    obs_intercept = MatrixWrapper("observation intercept", "obs_intercept")
    obs_cov = MatrixWrapper("observation covariance matrix", "obs_cov")
    transition = MatrixWrapper("transition", "transition")
    state_intercept = MatrixWrapper("state intercept", "state_intercept")
    selection = MatrixWrapper("selection", "selection")
    state_cov = MatrixWrapper("state covariance matrix", "state_cov")
    
    # 核心方法
    def bind(self, endog):          # 绑定观测数据
    def initialize(self, method):   # 初始化状态分布
    def clone(self, endog):         # 克隆模型（用于预测）
```

**初始化方法支持**：
- `'known'`：已知初始状态和协方差
- `'stationary'`：平稳初始化
- `'diffuse'`：精确漫延初始化
- `'approximate_diffuse'`：近似漫延初始化（大方差）

#### 第二层：KalmanFilter（`kalman_filter.py:62-...`）

**核心职责**：实现卡尔曼滤波递推

```python
class KalmanFilter(Representation):
    """State space representation with Kalman filter"""
    
    # 滤波方法选项（位掩码）
    filter_method = FILTER_CONVENTIONAL  # 传统卡尔曼滤波
    # 其他选项:
    # FILTER_UNIVARIATE - 单变量滤波
    # FILTER_CONCENTRATED - 集中似然
    # FILTER_CHANDRASEKHAR - Chandrasekhar 递推
    
    # 内存控制选项
    conserve_memory = MEMORY_STORE_ALL
    
    # 核心方法
    def filter(self, **kwargs):         # 执行滤波
    def loglike(self, **kwargs):         # 计算对数似然
    def loglikeobs(self, **kwargs):      # 计算每时刻的对数似然
```

**滤波输出**：
- 预测状态：$a_{t|t-1}$ (`predicted_state`)
- 预测协方差：$P_{t|t-1}$ (`predicted_state_cov`)
- 滤波状态：$a_{t|t}$ (`filtered_state`)
- 滤波协方差：$P_{t|t}$ (`filtered_state_cov`)
- 预测误差：$v_t$ (`forecast_error`)
- 预测误差协方差：$F_t$ (`forecast_error_cov`)
- 卡尔曼增益：$K_t$ (`kalman_gain`)

#### 第三层：KalmanSmoother（`kalman_smoother.py:37-...`）

**核心职责**：实现卡尔曼平滑（使用全部数据估计状态）

```python
class KalmanSmoother(KalmanFilter):
    """State space representation with Kalman filter and smoother"""
    
    # 平滑输出选项（位掩码）
    smoother_output = SMOOTHER_ALL
    
    SMOOTHER_STATE = 0x01           # 平滑状态
    SMOOTHER_STATE_COV = 0x02       # 平滑状态协方差
    SMOOTHER_DISTURBANCE = 0x04     # 平滑扰动
    SMOOTHER_DISTURBANCE_COV = 0x08 # 平滑扰动协方差
    SMOOTHER_STATE_AUTOCOV = 0x10   # 平滑状态自协方差
    
    # 平滑方法
    smooth_method = SMOOTH_CONVENTIONAL  # Durbin-Koopman 方法
    # SMOOTH_CLASSICAL - 经典方法
    # SMOOTH_ALTERNATIVE - 修正 Bryson-Frazier
    # SMOOTH_UNIVARIATE - 单变量平滑
    
    # 核心方法
    def smooth(self, **kwargs):     # 执行平滑
```

**平滑输出**（基于全部样本 $T$ 时刻的信息）：
- 平滑状态：$a_{t|T}$ (`smoothed_state`)
- 平滑协方差：$P_{t|T}$ (`smoothed_state_cov`)
- 平滑扰动：$\hat{\varepsilon}_{t|T}$, $\hat{\eta}_{t|T}$ (`smoothed_observation_disturbance`, `smoothed_state_disturbance`)
- 滞后一期自协方差：$Cov(a_t, a_{t-1}|Y_T)$ (`smoothed_state_autocov`)

#### 第四层：SimulationSmoother（`simulation_smoother.py:49-...`）

**核心职责**：实现仿真平滑器（从条件后验分布采样）

```python
class SimulationSmoother(KalmanSmoother):
    """State space representation with simulation smoother"""
    
    # 仿真输出选项
    simulation_output = SIMULATION_ALL
    SIMULATION_STATE = 0x01        # 仿真状态
    SIMULATION_DISTURBANCE = 0x04  # 仿真扰动
    
    # 核心方法
    def simulation_smoother(self, **kwargs):  # 获取仿真平滑器
    def _simulate(self, nsimulations, ...):   # 执行仿真
    def simulator(self, nsimulations, ...):    # 获取仿真器
```

**两种仿真方法**：
1. **KFS 方法**：基于 Kalman 滤波和平滑的仿真平滑器
2. **CFA 方法**：Cholesky 因子算法（适用于特定模型，速度更快）

**仿真能力**：
- 从 $p(\alpha | y)$ 采样状态轨迹
- 从 $p(\varepsilon, \eta | y)$ 采样扰动
- 生成新的时间序列实现

### 4.3 层次递进关系总结

```
┌─────────────────────────────────────────────────────────────────┐
│                    SimulationSmoother                            │
│  能力：条件后验采样、贝叶斯分析、缺失数据插补、多重插补          │
│  算法：Durbin-Koopman 仿真平滑器 + CFA 快速算法                 │
├─────────────────────────────────────────────────────────────────┤
│                      KalmanSmoother                              │
│  能力：全样本状态估计、扰动估计、自协方差计算                    │
│  算法：RTS 固定区间平滑器、扰动平滑器                            │
├─────────────────────────────────────────────────────────────────┤
│                       KalmanFilter                               │
│  能力：实时滤波、似然计算、一步预测                              │
│  算法：传统卡尔曼滤波、单变量滤波、Chandrasekhar 递推           │
├─────────────────────────────────────────────────────────────────┤
│                       Representation                             │
│  能力：矩阵管理、数据绑定、模型克隆、初始化                      │
│  核心：8 个状态空间矩阵的统一抽象                                │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 应用场景对比

| 层级 | 典型应用 | 信息利用 |
|------|----------|----------|
| KalmanFilter | 实时信号处理、在线预测 | 仅使用到 t 时刻的信息 |
| KalmanSmoother | 历史数据分析、状态诊断、参数估计 | 使用全部 T 个时刻的信息 |
| SimulationSmoother | 贝叶斯推断、MCMC、缺失数据插补、预测区间 | 从条件后验分布采样 |

---

## 5. 与通用似然模型体系的集成

### 5.1 MLEModel 类的设计

`MLEModel` 是连接状态空间框架与 statsmodels 通用 MLE 体系的桥梁（位于 `mlemodel.py:93-...`）：

```python
class MLEModel(tsbase.TimeSeriesModel):
    """State space model for maximum likelihood estimation"""
    
    def __init__(self, endog, k_states, exog=None, dates=None, freq=None, **kwargs):
        # 1. 调用父类（TimeSeriesModel）初始化
        super().__init__(endog=endog, exog=exog, dates=dates, freq=freq, missing="none")
        
        # 2. 准备数据
        self.endog, self.exog = self.prepare_data()
        
        # 3. 初始化状态空间表示（注意：使用最完整的 SimulationSmoother）
        self.initialize_statespace(**kwargs)
    
    def initialize_statespace(self, **kwargs):
        """Initialize the state space representation"""
        endog = self.endog.T  # 转换为 F-ordered 宽格式
        
        # 实例化最完整的 SimulationSmoother
        self.ssm = SimulationSmoother(
            endog.shape[0], self.k_states, nobs=endog.shape[1], **kwargs
        )
        self.ssm.bind(endog)
```

**关键设计**：`MLEModel` 内部持有 `SimulationSmoother` 实例（而非继承），这种组合模式提供了更好的灵活性。

### 5.2 参数到矩阵的映射：update 方法

子类必须实现 `update` 方法，将参数向量映射到状态空间矩阵：

```python
# 这是一个抽象方法，子类必须实现
def update(self, params, transformed=True, includes_fixed=False, complex_step=False):
    """
    Update the statespace representation with new parameters.
    
    Parameters
    ----------
    params : array_like
        Unconstrained parameters (if transformed=False) or
        constrained parameters (if transformed=True).
    """
    raise NotImplementedError
```

**SARIMAX 模型的 update 示例**：

```python
# sarimax.py 中的简化逻辑
def update(self, params, **kwargs):
    params = super().update(params, **kwargs)
    
    # 1. 提取参数
    if self.k_ar_params > 0:
        ar_params = params[:self.k_ar_params]
    if self.k_ma_params > 0:
        ma_params = params[self.k_ar_params:self.k_ar_params + self.k_ma_params]
    # ... 其他参数
    
    # 2. 填充 transition 矩阵（伴随矩阵形式）
    if self.k_ar > 0:
        self.ssm['transition', 0, :self.k_ar] = ar_params
    
    # 3. 填充 selection 矩阵
    if self.k_ma > 0:
        self.ssm['selection', :self.k_ma, 0] = ma_params
    
    # 4. 填充 state_cov 矩阵（创新方差）
    self.ssm['state_cov', 0, 0] = params[self.k_params - 1] ** 2
```

### 5.3 似然计算流程

```
loglike(params)
    │
    ├──► handle_params(params)  # 处理固定参数、变换参数
    │
    ├──► update(params)         # 关键：参数 → 状态空间矩阵
    │
    └──► ssm.loglike()          # 调用 KalmanFilter 计算似然
              │
              ├──► 运行卡尔曼滤波
              │
              └──► 累加预测误差的对数似然
```

**核心代码**（`mlemodel.py:987-1040`）：

```python
def loglike(self, params, *args, **kwargs):
    """Loglikelihood evaluation"""
    # 1. 处理参数（固定参数、变换等）
    transformed, includes_fixed, complex_step, kwargs = _handle_args(...)
    
    params = self.handle_params(
        params, transformed=transformed, includes_fixed=includes_fixed
    )
    
    # 2. 更新状态空间矩阵
    self.update(
        params, transformed=True, includes_fixed=True, complex_step=complex_step
    )
    
    # 3. 调用底层 KalmanFilter 计算似然
    if complex_step:
        kwargs["inversion_method"] = INVERT_UNIVARIATE | SOLVE_LU
    
    loglike = self.ssm.loglike(complex_step=complex_step, **kwargs)
    
    return loglike
```

### 5.4 参数估计流程（fit 方法）

```python
def fit(self, start_params=None, method="lbfgs", maxiter=50, **kwargs):
    """Fits the model by maximum likelihood via Kalman filter"""
    
    # 1. 获取起始参数
    if start_params is None:
        start_params = self.start_params  # 子类提供
    
    # 2. 准备优化参数（处理固定参数、变换）
    start_params = self.handle_params(start_params, transformed=True, ...)
    if transformed:
        start_params = self.untransform_params(start_params)
    
    # 3. 调用父类 LikelihoodModel.fit()
    # 这会使用 scipy.optimize 最小化 -loglike(params)
    mlefit = super().fit(
        start_params,
        method=method,
        fargs=fargs,
        maxiter=maxiter,
        **kwargs
    )
    
    # 4. 最终滤波/平滑，构造结果对象
    if low_memory:
        func = self.filter
    else:
        func = self.smooth
    
    res = func(
        mlefit.params,
        transformed=False,
        cov_type=cov_type,
        cov_kwds=cov_kwds,
    )
    
    return res
```

### 5.5 参数变换机制

为了在无约束空间进行优化，系统实现了参数变换：

```python
# 模板方法，子类可覆盖
def transform_params(self, unconstrained):
    """
    Transform unconstrained parameters to constrained space.
    
    Example: AR 系数需要满足平稳性条件
    """
    return unconstrained

def untransform_params(self, constrained):
    """
    Transform constrained parameters to unconstrained space.
    """
    return constrained
```

**SARIMAX 中的实际变换**：

```python
# 确保 AR 系数平稳
def transform_params(self, unconstrained):
    # 使用 Yuule-Walker 或其他方法确保平稳
    constrained = unconstrained.copy()
    if self.enforce_stationarity and self.k_ar_params > 0:
        ar_params = unconstrained[:self.k_ar_params]
        # 变换到平稳域
        constrained[:self.k_ar_params] = constrain_stationary_univariate(ar_params)
    # ...
    return constrained
```

### 5.6 协方差矩阵计算

支持多种协方差估计方法：

```python
# fit() 方法中的 cov_type 参数
cov_type : str, optional
    The method for calculating the covariance matrix:
    
    - 'opg'   : 外积梯度估计器 (Outer Product of Gradients)
    - 'oim'   : 观测信息矩阵 (Observed Information Matrix)
    - 'approx': 数值 Hessian 近似
    - 'robust': 准极大似然 (QML) 鲁棒协方差
    - 'none'  : 不计算协方差
```

**数值导数实现**（`mlemodel.py:1115-...`）：

```python
def _forecasts_error_partial_derivatives(self, params, ...):
    """计算预测误差对参数的偏导数（用于 OIM）"""
    
    if approx_complex_step:
        # 复步微分法（更精确）
        epsilon = _get_epsilon(params, 2, None, n)
        increments = np.identity(n) * 1j * epsilon
        
        for i, ih in enumerate(increments):
            self.update(params + ih, complex_step=True, ...)
            _res = self.ssm.filter(complex_step=True)
            
            # 利用复步导数性质: f'(x) ≈ Im[f(x + ih)] / h
            partials[:, :, i] = _res.forecasts_error.imag / epsilon[i]
    else:
        # 有限差分法
        # ...
```

### 5.7 完整集成架构图

```
┌──────────────────────────────────────────────────────────────────────┐
│                         statsmodels 通用 MLE 框架                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐       │
│  │ LikelihoodModel │    │ TimeSeriesModel│    │    Model (base)   │       │
│  │  - loglike()    │    │  - 时间序列处理  │    │  - fit() 优化框架  │       │
│  │  - score()      │    │  - 日期/频率    │    │  - 结果包装        │       │
│  │  - hessian()    │    │  - 滞后处理     │    │                   │       │
│  └───────┬──────┘    └───────┬──────┘    └──────────────────┘       │
└──────────┼────────────────────┼───────────────────────────────────────┘
           │                    │
           └────────┬───────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           MLEModel (适配器)                            │
│                                                                       │
│  职责：                                                                 │
│  1. 继承 TimeSeriesModel，融入通用 MLE 体系                            │
│  2. 组合 SimulationSmoother，桥接状态空间算法                          │
│  3. 定义参数 ↔ 矩阵映射接口（update 方法）                              │
│  4. 提供 fit() / filter() / smooth() 高层 API                          │
│                                                                       │
│  核心方法：                                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  fit(params)                                                  │    │
│  │    ├──► scipy.optimize 最小化 -loglike(params)              │    │
│  │    └──► 返回 MLEResults（含滤波/平滑结果）                   │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  loglike(params)                                              │    │
│  │    ├──► update(params)  # 参数 → 状态空间矩阵               │    │
│  │    └──► ssm.loglike()   # Kalman 滤波计算                   │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  update(params)  [抽象方法]                                   │    │
│  │    └──► 子类实现：SARIMAX.update, VARMAX.update, ...        │    │
│  └─────────────────────────────────────────────────────────────┘    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            │ 组合（self.ssm = SimulationSmoother(...)）
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    状态空间算法层（Representation 家族）               │
│                                                                       │
│  SimulationSmoother ──► KalmanSmoother ──► KalmanFilter ──► Representation
│                                                                       │
│  能力：仿真采样        能力：全样本平滑    能力：实时滤波    能力：矩阵表示
└──────────────────────────────────────────────────────────────────────┘
```

---

## 6. 架构设计总结

### 6.1 核心设计模式

| 设计模式 | 应用位置 | 实现效果 |
|----------|----------|----------|
| **描述符模式** | `MatrixWrapper` | 统一矩阵访问，自动处理维度/内存顺序 |
| **模板方法模式** | `MLEModel.update()` | 定义参数映射接口，子类实现具体逻辑 |
| **组合模式** | `MLEModel.ssm` | 灵活组合状态空间算法，而非继承 |
| **分层继承** | Representation → Filter → Smoother → SimulationSmoother | 能力递进，代码复用 |
| **代码生成** | Tempita + .pyx.in | 一套模板生成四种精度实现 |
| **位掩码选项** | `filter_method`, `conserve_memory` | 灵活配置，无需多个方法重载 |

### 6.2 关键工程决策

1. **Fortran 顺序内存布局**
   - 所有矩阵强制 `order="F"`
   - 与 BLAS/LAPACK 接口完全兼容
   - 避免不必要的内存拷贝

2. **精度前缀调度**
   - 编译时：Tempita 生成 s/d/c/z 四组函数
   - 运行时：`find_best_blas_type()` 自动选择
   - 用户透明，性能最优

3. **内存控制**
   - `conserve_memory` 位掩码选项
   - 可选择性不存储预测协方差、卡尔曼增益等
   - 大规模数据时显著降低内存占用

4. **缺失值处理**
   - 卡尔曼滤波天然支持缺失值
   - 观测为 NaN 时，跳过更新步骤
   - 无需额外的数据插补预处理

### 6.3 扩展性分析

**添加新的时序模型**：
1. 继承 `MLEModel`
2. 实现 `update()` 方法（参数 → 矩阵）
3. 实现 `start_params` 属性（起始参数）
4. 可选：实现 `transform_params`/`untransform_params`（参数变换）
5. **自动获得**：滤波、平滑、仿真、MLE 估计、预测、诊断

**添加新的滤波算法**：
1. 在 `_filters/` 添加新的 `.pyx.in` 模板
2. 更新 `kalman_filter.py` 中的 `filter_method` 选项
3. **自动获得**：四精度支持、平滑器兼容

---

## 7. 参考文献

1. Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford University Press.

2. Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.

3. Kim, C. J., & Nelson, C. R. (1999). *State-Space Models with Regime Switching*. MIT Press.

4. Statsmodels 源代码：`statsmodels/tsa/statespace/` 目录

---

## 附录：核心文件索引

| 文件路径 | 主要内容 |
|----------|----------|
| `representation.py` | `Representation` 类，矩阵抽象，数据绑定 |
| `kalman_filter.py` | `KalmanFilter` 类，滤波算法，选项配置 |
| `kalman_smoother.py` | `KalmanSmoother` 类，平滑算法 |
| `simulation_smoother.py` | `SimulationSmoother` 类，仿真平滑器 |
| `mlemodel.py` | `MLEModel` 类，MLE 估计集成 |
| `tools.py` | 前缀映射，辅助函数，伴随矩阵生成 |
| `_filters/*.pyx.in` | 卡尔曼滤波 Cython 模板 |
| `_smoothers/*.pyx.in` | 卡尔曼平滑 Cython 模板 |
| `initialization.py` | 状态初始化方法 |
| `sarimax.py` | SARIMAX 模型实现（示例） |
| `structural.py` | 结构时间序列模型 |
| `varmax.py` | VARMAX 模型 |
| `dynamic_factor.py` | 动态因子模型 |
