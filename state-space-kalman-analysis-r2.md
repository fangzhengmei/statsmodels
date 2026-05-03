# Statsmodels 状态空间模型职责边界与调用关系分析

## 目录
1. [类继承体系与职责边界
2. [参数更新完整链路
3. [参数变换逻辑复用路径
4. [各模型实现对比
5. [完整代码引用索引

---

## 1. 类继承体系与职责边界

### 1.1 完整继承层次

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        通用基础设施层                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────┐                                                │
│  │  Representation   │                                                │
│  └─────────┬─────────┘                                                │
│            │ 职责：状态空间矩阵的存储与访问管理                          │
│            │ - MatrixWrapper 描述符管理矩阵内存                          │
│            │ - 管理 design, transition, selection 等矩阵                 │
│            ▼                                                        │
│  ┌───────────────────┐                                                │
│  │   KalmanFilter    │                                                │
│  └─────────┬─────────┘                                                │
│            │ 职责：卡尔曼滤波算法实现                                      │
│            │ - 常规滤波、单变量滤波                                   │
│            │ - 缺失值处理逻辑                                             │
│            │ - 对数似然计算                                           │
│            ▼                                                        │
│  ┌───────────────────┐                                                │
│  │ SimulationSmoother │                                               │
│  └─────────┬─────────┘                                                │
│            │ 职责：模拟平滑器扩展                                       │
│            │ - 状态平滑                                                     │
│            │ - 模拟平滑采样                                               │
│            ▼                                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                        MLE 估计框架层                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────┐                                                │
│  │    MLEModel     │                                                │
│  └─────────┬─────────┘                                                │
│            │ 职责：最大似然估计框架                                   │
│            │ - fit()：协调优化器与滤波                                 │
│            │ - loglike()：计算对数似然                                       │
│            │ - handle_params()：参数预处理                             │
│            │ - update()：基类方法（需子类覆盖）                            │
│            ▼                                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                        具体模型实现层                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│     ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│     │   SARIMAX    │    │   VARMAX     │    │ UnobservedComponents │    │   DynamicFactor   │
│     │  (单变量)     │    │  (多变量)    │    │    (结构模型)        │    │   (动态因子模型)   │
│     └──────────────┘    └──────────────┘    └──────────────────┘    └──────────────────┘
│                                                                         │
│  共同职责：                                                              │
│  - 实现特定状态空间矩阵构建                                             │
│  - 覆盖 update()：参数 → 状态空间矩阵                                    │
│  - 覆盖 transform_params()：参数约束变换                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 各层职责详解

#### 1.2.1 Representation 层

**文件位置**：`representation.py:87-500+`

**核心职责**：状态空间矩阵的内存管理与访问

**关键特性**：

1. **MatrixWrapper 描述符** (`representation.py:37-85`)
   ```python
   class MatrixWrapper:
       """
       管理状态空间矩阵的描述符
       
       负责：
       - 矩阵形状验证
       - 自动处理时变/时不变矩阵转换
       - Fortran 顺序内存布局优化
       """
       
       def __get__(self, obj, objtype):
           matrix = getattr(obj, self._attribute, None)
           return matrix
       
       def __set__(self, obj, value):
           value = np.asarray(value, order="F")
           # 形状验证
           # 时不变矩阵自动扩展为三维
           setattr(obj, self._attribute, value)
   ```

2. **状态空间矩阵定义** (`representation.py:220-251`)
   ```python
   design = MatrixWrapper("design", "design")           # Z_t: (k_endog × k_states × nobs)
   obs_intercept = MatrixWrapper("observation intercept", "obs_intercept")  # d_t
   obs_cov = MatrixWrapper("observation covariance matrix", "obs_cov")      # H_t
   transition = MatrixWrapper("transition", "transition")    # T_t
   state_intercept = MatrixWrapper("state intercept", "state_intercept")  # c_t
   selection = MatrixWrapper("selection", "selection")        # R_t
   state_cov = MatrixWrapper("state covariance matrix", "state_cov")      # Q_t
   ```

3. **矩阵访问语法糖**
   ```python
   # 通过索引访问
   model.ssm["transition"]           # 获取整个过渡矩阵
   model.ssm["transition", 0, :]   # 获取过渡矩阵第一行
   model.ssm["transition", :, 0]     # 获取过渡矩阵第一列
   ```

#### 1.2.2 KalmanFilter 层

**文件位置**：`kalman_filter.py` + `_filters/_conventional.pyx.in` + `_filters/_univariate.pyx.in`

**核心职责**：卡尔曼滤波算法与缺失值处理

**关键特性**：

1. **滤波方法选择**
   - `FILTER_CONVENTIONAL`：标准卡尔曼滤波
   - `FILTER_UNIVARIATE`：单变量滤波（处理缺失值）
   - `FILTER_COLLAPSED`：压缩滤波
   - `FILTER_CONCENTRATED`：集中尺度滤波

2. **缺失值处理核心**
   ```cython
   // 完全缺失时：预测值设为0，状态不更新
   cdef int {{prefix}}updating_missing_conventional(...):
       // 直接复制预测状态到滤波状态
       blas.{{prefix}}copy(&kfilter.k_states, 
             kfilter._input_state, &inc, 
             kfilter._filtered_state, &inc)
   ```

#### 1.2.3 MLEModel 层

**文件位置**：`mlemodel.py`

**核心职责**：最大似然估计框架协调

**关键方法**：

| 方法 | 行号 | 职责 | 是否需子类覆盖 |
|-----|------|------|---------------|
| `fit()` | 540-787 | 协调优化器与滤波 | 否 |
| `loglike()` | 987-1040 | 计算对数似然 | 否 |
| `filter()` | 848-919 | 执行卡尔曼滤波 | 否 |
| `handle_params()` | 1930-1963 | 参数预处理（变换、固定参数） | 否 |
| `update()` | 1965-1991 | 参数→状态空间矩阵 | **是** |
| `transform_params()` | (基类无) | 参数约束变换 | **是** |
| `start_params` | (属性) | 初始参数猜测 | **是** |

#### 1.2.4 具体模型层

**各模型职责**：

| 模型 | 适用场景 | 状态空间特点 |
|-----|---------|-------------|
| **SARIMAX** | 单变量 ARIMA/SARIMA | 伴随矩阵形式，支持差分纳入状态向量 |
| **VARMAX** | 多变量 VARMA | 分块伴随矩阵 |
| **UnobservedComponents** | 结构时间序列分解 | 趋势、季节、循环分量分别建模 |
| **DynamicFactor** | 动态因子模型 | 因子 + 观测方程两层结构 |

---

## 2. 参数更新完整链路

### 2.1 全局数据流图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              用户调用 model.fit()                              │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        MLEModel.fit() (mlemodel.py:540)                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  步骤 1: 获取初始参数                                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ if start_params is None:                                               │ │
│  │     start_params = self.start_params  ← 子类提供                       │ │
│  │     transformed = True                                                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  步骤 2: 设置优化器参数                                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ if method == "lbfgs":                                                  │ │
│  │     approx_grad = True    # 使用数值梯度                                  │ │
│  │     epsilon = 1e-5        # 梯度步长                                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  步骤 3: 调用 scipy.optimize.fmin_l_bfgs_b                              │
│                                                                              │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼ 优化器反复调用
┌──────────────────────────────────────────────────────────────────────────────┐
│                    优化器迭代：while not converged                         │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼ 每次迭代调用 loglike(params)
┌──────────────────────────────────────────────────────────────────────────────┐
│                      MLEModel.loglike() (mlemodel.py:987)                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  步骤 A: 参数预处理                                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ params = self.handle_params(                                           │ │
│  │     params, transformed=transformed, includes_fixed=includes_fixed     │ │
│  │ )                                                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  步骤 B: 更新状态空间矩阵                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ self.update(                                                           │ │
│  │     params, transformed=True, includes_fixed=True, complex_step=False   │ │
│  │ )                    ↑ 这是关键！子类实现                                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  步骤 C: 执行卡尔曼滤波计算似然                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ loglike = self.ssm.loglike(complex_step=complex_step, **kwargs)    │ │
│  │                                                                       │ │
│  │ 内部流程:                                                               │ │
│  │ ssm.loglike() → KalmanFilter.filter() → 实际滤波计算                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└───────────────────────────────┬──────────────────────────────────────────────┘
```

### 2.2 handle_params 详细流程

**文件位置**：`mlemodel.py:1930-1963`

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    MLEModel.handle_params() 详细流程                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  输入: params (优化器提供的参数向量)                                          │
│        transformed=True/False (是否已变换)                                  │
│        includes_fixed=True/False (是否包含固定参数)                          │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐
│  │ 步骤 1: 类型转换                                                        │
│  │ if np.issubdtype(params.dtype, np.integer):                            │
│  │     params = params.astype(np.float64)                                   │
│  └────────────────────────────────────────────────────────────────────────┘
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐
│  │ 步骤 2: 处理固定参数 (如果需要)                                           │
│  │ if not includes_fixed and self._has_fixed_params:                        │
│  │     # 扩展参数向量，插入固定参数                                          │
│  │     new_params = np.zeros(k_params, dtype=params.dtype) * np.nan          │
│  │     new_params[self._free_params_index] = params                          │
│  │     params = new_params                                                     │
│  └────────────────────────────────────────────────────────────────────────┘
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐
│  │ 步骤 3: 参数约束变换 (关键!)                                               │
│  │ if not transformed:                                                       │
│  │     # 先填充固定参数的值（变换可能需要它们）                               │
│  │     if not includes_fixed and self._has_fixed_params:                    │
│  │         params[self._fixed_params_index] = list(self._fixed_params.values())│
│  │                                                                      │
│  │     # 调用子类的 transform_params()                                       │
│  │     if return_jacobian:                                                  │
│  │         transform_score = self.transform_jacobian(params)                 │
│  │     params = self.transform_params(params)                                │
│  └────────────────────────────────────────────────────────────────────────┘
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐
│  │ 步骤 4: 最终设置固定参数                                                │
│  │ if not includes_fixed and self._has_fixed_params:                        │
│  │     params[self._fixed_params_index] = list(self._fixed_params.values())      │
│  └────────────────────────────────────────────────────────────────────────┘
│                                                                              │
│  输出: params (已变换后的参数向量)                                            │
│        (可选) transform_score (变换的雅可比矩阵)                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 SARIMAX.update() 详细流程

**文件位置**：`sarimax.py:1530-1719`

这是参数更新链路中最关键的一环，负责将参数向量映射到状态空间矩阵。

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      SARIMAX.update() 详细流程                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  输入: params (已变换的约束参数向量)                                          │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐
│  │ 步骤 1: 参数预处理（通过 handle_params）                                  │
│  │ params = self.handle_params(                                              │
│  │     params, transformed=transformed, includes_fixed=includes_fixed     │
│  │ )                                                                      │
│  └────────────────────────────────────────────────────────────────────────┘
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐
│  │ 步骤 2: 分段提取参数                                                    │
│  │                                                                      │
│  │ params = [                                                            │
│  │     # 1. 趋势参数 (k_trend个)                                        │
│  │     δ₀, δ₁, ...,                                                    │
│  │                                                                      │
│  │     # 2. MLE 回归系数 (如果 mle_regression=True)                     │
│  │     β₁, β₂, ..., β_{k_exog},                                        │
│  │                                                                      │
│  │     # 3. AR 参数 (k_ar_params个)                                      │
│  │     φ₁, φ₂, ..., φ_{k_ar_params},                                    │
│  │                                                                      │
│  │     # 4. MA 参数 (k_ma_params个)                                      │
│  │     θ₁, θ₂, ..., θ_{k_ma_params},                                    │
│  │                                                                      │
│  │     # 5. 季节 AR 参数 (k_seasonal_ar_params个)                         │
│  │     Φ₁, Φ₂, ..., Φ_{k_seasonal_ar_params},                            │
│  │                                                                      │
│  │     # 6. 季节 MA 参数 (k_seasonal_ma_params个)                         │
│  │     Θ₁, Θ₂, ..., Θ_{k_seasonal_ma_params},                            │
│  │                                                                      │
│  │     # 7. 方差参数                                                      │
│  │     σ_β² (时变回归方差),                                               │
│  │     σ_ε² (测量误差方差),                                                │
│  │     σ_η² (状态噪声方差)                                                 │
│  │ ]                                                                      │
│  │ 提取方式:                                                              │
│  │ start = end = 0                                                         │
│  │ end += self._k_trend              # 趋势参数结束位置                         │
│  │ params_trend = params[start:end]                                         │
│  │ start += self._k_trend                                                   │
│  │                                                                      │
│  │ end += self._k_exog             # 回归系数结束位置                       │
│  │ params_exog = params[start:end]                                          │
│  │ start += self._k_exog                                                    │
│  │                                                                      │
│  │ # ... 依此类推                                                          │
│  └────────────────────────────────────────────────────────────────────────┘
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐
│  │ 步骤 3: 构建滞后多项式                                                  │
│  │                                                                      │
│  │ // AR 多项式: φ(L) = 1 - φ₁L - φ₂L² - ... - φₚLᵖ                      │
│  │ if self.k_ar > 0:                                                       │
│  │     # polynomial_ar 初始化为 [1, 0, 0, ..., 0]                              │
│  │     # _polynomial_ar_idx 是需要填充的位置索引                            │
│  │     self._polynomial_ar[self._polynomial_ar_idx] = -params_ar           │
│  │     # 注意负号！这是多项式的约定                                           │
│  │                                                                      │
│  │ // MA 多项式: θ(L) = 1 + θ₁L + θ₂L² + ... + θ_qL^q                      │
│  │ if self.k_ma > 0:                                                       │
│  │     self._polynomial_ma[self._polynomial_ma_idx] = params_ma              │
│  │     # 注意没有负号！                                                    │
│  └────────────────────────────────────────────────────────────────────────┘
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐
│  │ 步骤 4: 计算简化形式多项式                                              │
│  │                                                                      │
│  │ // 将常规和季节多项式相乘                                              │
│  │ if self.k_seasonal_ar > 0:                                            │
│  │     // φ(L) * Φ(L^s)                                                   │
│  │     reduced_polynomial_ar = -np.polymul(                             │
│  │         self._polynomial_ar, self._polynomial_seasonal_ar            │
│  │     )                                                                  │
│  │ else:                                                                   │
│  │     reduced_polynomial_ar = -self._polynomial_ar                       │
│  │                                                                      │
│  │ // 同样处理 MA 多项式                                                   │
│  │ if self.k_seasonal_ma > 0:                                            │
│  │     reduced_polynomial_ma = np.polymul(                              │
│  │         self._polynomial_ma, self._polynomial_seasonal_ma             │
│  │     )                                                                  │
│  └────────────────────────────────────────────────────────────────────────┘
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐
│  │ 步骤 5: 更新状态空间矩阵（核心！）                                        │
│  │                                                                      │
│  │ // 5.1 更新观测截距 (外生回归)                                            │
│  │ if self.mle_regression:                                                │
│  │     // y_t = X_t β + u_t                                              │
│  │     // 通过观测截距实现: d_t = X_t β                                          │
│  │     self.ssm["obs_intercept"] = np.dot(self.exog, params_exog)[None, :]   │
│  │                                                                      │
│  │ // 5.2 更新状态截距 (趋势)                                              │
│  │ if self._k_trend > 0:                                                   │
│  │     data = np.dot(self._trend_data, params_trend)                      │
│  │     if not self.hamilton_representation:                                 │
│  │         // Harvey 表示: 趋势进入状态截距                                  │
│  │         self.ssm["state_intercept", self._k_states_diff, :] = data      │
│  │     else:                                                               │
│  │         // Hamilton 表示: 趋势进入观测截距                                │
│  │         // 需要调整为均值形式                                              │
│  │         data /= np.sum(-reduced_polynomial_ar)                          │
│  │         self.ssm["obs_intercept"] += data[None, :]                       │
│  │                                                                      │
│  │ // 5.3 更新观测协方差 (测量误差)                                          │
│  │ if self.measurement_error:                                             │
│  │     self.ssm["obs_cov", 0, 0] = params_measurement_variance             │
│  │                                                                      │
│  │ // 5.4 更新过渡矩阵 (AR 参数)                                           │
│  │ if self.k_ar > 0 or self.k_seasonal_ar > 0:                              │
│  │     // 预计算的索引，避免每次重新计算                                    │
│  │     // Harvey: transition_ar_params_idx = ("transition", start:end, col)       │
│  │     // Hamilton: transition_ar_params_idx = ("transition", col, start:end)│
│  │                                                                      │
│  │     // 第一列（或第一行）填充简化多项式系数                                │
│  │     // reduced_polynomial_ar[0] 是常数项 1，跳过                        │
│  │     self.ssm[self.transition_ar_params_idx] = reduced_polynomial_ar[1:]│
│  │                                                                      │
│  │ // 5.5 更新选择矩阵或设计矩阵 (MA 参数)                                 │
│  │ if self.k_ma > 0 or self.k_seasonal_ma > 0:                            │
│  │     if not self.hamilton_representation:                                 │
│  │         // Harvey 表示: MA 参数在 selection 矩阵                          │
│  │         self.ssm[self.selection_ma_params_idx] = reduced_polynomial_ma[1:] │
│  │     else:                                                               │
│  │         // Hamilton 表示: MA 参数在 design 矩阵                         │
│  │         self.ssm[self.design_ma_params_idx] = reduced_polynomial_ma[1:]  │
│  │                                                                      │
│  │ // 5.6 更新状态协方差                                                    │
│  │ if self.k_posdef > 0:                                                   │
│  │     if not self.concentrate_scale:                                        │
│  │         self["state_cov", 0, 0] = params_variance                      │
│  │     if self.state_regression and self.time_varying_regression:            │
│  │         self.ssm[self._exog_variance_idx] = params_exog_variance           │
│  └────────────────────────────────────────────────────────────────────────┘
│                                                                              │
│  输出: params (返回参数向量)                                                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 完整调用链代码引用

| 层级 | 方法 | 文件:行号 | 关键操作 |
|-----|------|---------|---------|
| 用户层 | `model.fit()` | - | 入口点 |
| MLE层 | `MLEModel.fit()` | `mlemodel.py:540` | 调用优化器 |
| MLE层 | `MLEModel.loglike()` | `mlemodel.py:987` | 似然计算入口 |
| MLE层 | `MLEModel.handle_params()` | `mlemodel.py:1930` | 参数变换处理 |
| 模型层 | `SARIMAX.transform_params()` | `sarimax.py:1299` | 参数约束变换 |
| 模型层 | `SARIMAX.update()` | `sarimax.py:1530` | 参数→状态空间矩阵 |
| 工具层 | `companion_matrix()` | `tools.py:127` | 伴随矩阵创建 |
| 滤波层 | `KalmanFilter.loglike()` | `kalman_filter.py` | 滤波计算似然 |
| Cython层 | `forecast_conventional()` | `_conventional.pyx.in:108` | 预测步 |
| Cython层 | `updating_conventional()` | `_conventional.pyx.in` | 更新步 |

---

## 3. 参数变换逻辑复用路径

### 3.1 核心工具函数位置

**文件**：`tools.py`

**函数列表**：

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         tools.py 核心变换函数                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐
│  │ 单变量平稳性约束                                                      │
│  ├────────────────────────────────────────────────────────────────────────┤
│  │                                                                      │
│  │ def constrain_stationary_univariate(unconstrained):                    │
│  │     """                                                                   │
│  │     输入: 无约束参数向量 x ∈ ℝᵏ                                           │
│  │                                                                      │
│  │     输出: 约束参数向量 φ，满足 AR(p) 平稳性条件                            │
│  │           (所有特征根在单位圆内)                                            │
│  │                                                                      │
│  │     算法: Monahan (1984) 方法                                         │
│  │                                                                      │
│  │     步骤:                                                              │
│  │     1. r_k = x_k / √(1 + x_k²)  →  将 x_k ∈ ℝ 映射到 (-1, 1)    │
│  │        (r_k 是偏自相关系数 PACF)                                     │
│  │                                                                      │
│  │     2. Levinson-Durbin 递推:                                           │
│  │        y[k, k] = r[k]                                                 │
│  │        y[k, i] = y[k-1, i] + r[k] * y[k-1, k-i-1]                │
│  │                                                                      │
│  │     3. 返回 -y[n-1, :]  (负号匹配多项式约定)                          │
│  │                                                                      │
│  │ def unconstrain_stationary_univariate(constrained):                     │
│  │     // 逆变换: 从约束参数恢复无约束参数                                │
│  │                                                                      │
│  └────────────────────────────────────────────────────────────────────────┘
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐
│  │ 多变量平稳性约束                                                      │
│  ├────────────────────────────────────────────────────────────────────────┤
│  │                                                                      │
│  │ def constrain_stationary_multivariate(unconstrained, cov):             │
│  │     """                                                                   │
│  │     输入: 无约束系数矩阵 (k_endog × k_endog*p)                         │
│  │           状态协方差矩阵 Σ                                                 │
│  │                                                                      │
│  │     输出: 约束 VAR 系数矩阵，满足平稳性条件                              │
│  │           (companion 矩阵的特征根模 < 1)                                    │
│  │                                                                      │
│  │     算法: Ansley & Kohn (1986) 方法                                 │
│  │                                                                      │
│  │     参考: Lemma 2.2 in Ansley and Kohn (1986)                          │
│  │                                                                      │
│  │ def unconstrain_stationary_multivariate(constrained, cov):               │
│  │     // 逆变换                                                         │
│  │                                                                      │
│  └────────────────────────────────────────────────────────────────────────┘
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐
│  │ 伴随矩阵创建                                                              │
│  ├────────────────────────────────────────────────────────────────────────┤
│  │                                                                      │
│  │ def companion_matrix(polynomial):                                      │
│  │     """                                                                   │
│  │     对于多项式 c(L) = c₀ + c₁L + c₂L² + ... + cₚLᵖ                   │
│  │                                                                      │
│  │     创建伴随矩阵:                                                        │
│  │     [φ₁  1  0  ...  0]                                              │
│  │     [φ₂  0  1  ...  0]                                              │
│  │     [φ₃  0  0  ...  0]                                              │
│  │     [...            1]                                              │
│  │     [φₚ  0  0  ...  0]                                              │
│  │                                                                      │
│  │     其中 φᵢ = -cᵢ / c₀                                               │
│  │                                                                      │
│  └────────────────────────────────────────────────────────────────────────┘
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 各模型复用情况对比

#### 3.2.1 SARIMAX 模型

**文件位置**：`sarimax.py:1299-1505`

**参数结构与变换映射**：

| 参数类型 | 变换函数 | 变换方式 | 行号 |
|---------|---------|-----------|------|
| **趋势参数** | 无 | 直接使用 | 1332-1335 |
| **回归系数** | 无 | 直接使用 | 1338-1341 |
| **AR 参数** | `constrain_stationary_univariate` | 直接 | 1343-1352 |
| **MA 参数** | `constrain_stationary_univariate` | **取负** | 1354-1363 |
| **季节 AR** | `constrain_stationary_univariate` | 直接 | 1366-1374 |
| **季节 MA** | `constrain_stationary_univariate` | **取负** | 1377-1385 |
| **方差参数** | 平方运算 | `x**2** | 1387-1399 |

**关键代码**：

```python
# AR 参数: 直接使用
constrained[start:end] = constrain_stationary_univariate(unconstrained[start:end])

# MA 参数: 取负使用
constrained[start:end] = -constrain_stationary_univariate(unconstrained[start:end])
```

**为什么 MA 参数要取负？**

```
MA(q) 模型: y_t = ε_t + θ₁ ε_{t-1} + ... + θ_q ε_{t-q}

滞后多项式: θ(L) = 1 + θ₁ L + θ₂ L² + ... + θ_q L^q

可逆性条件: θ(z) = 0 的所有根都在单位圆外

令 ψ_i = -θ_i，则多项式变为:
θ(L) = 1 - ψ₁ L - ψ₂ L² - ... - ψ_q L^q

这与 AR 多项式形式完全相同！

因此:
- MA 的可逆性条件 ⟺ ψ 是平稳 AR 系数
- 可以复用 constrain_stationary_univariate 函数
- 最后取负得到 θ
```

#### 3.2.2 VARMAX 模型

**文件位置**：`varmax.py:518-699`

**参数结构与变换映射**：

| 参数类型 | 变换函数 | 变换方式 | 行号 |
|---------|---------|-----------|------|
| **截距项** | 无 | 直接使用 | 543-544 |
| **VAR 参数** | `constrain_stationary_multivariate` | 结合协方差 | 546-565 |
| **VMA 参数** | `constrain_stationary_multivariate` | 结合单位阵 | 567-577 |
| **回归系数** | 无 | 直接使用 | 579-581 |
| **对角协方差** | 平方运算 | `x**2` | 585-587 |
| **非结构化协方差** | Cholesky 分解 | `L @ L.T` | 589-591 |
| **测量误差方差** | 平方运算 | `x**2` | 593-597 |

**关键代码**：

```python
# VAR 参数变换 (需要协方差矩阵)
coefficients = unconstrained[self._params_ar].reshape(
    self.k_endog, self.k_endog * self.k_ar)
coefficient_matrices, variance = (
    constrain_stationary_multivariate(coefficients, state_cov))

# VMA 参数变换 (使用单位阵作为协方差)
state_cov = np.eye(self.k_endog, dtype=unconstrained.dtype)
coefficient_matrices, variance = (
    constrain_stationary_multivariate(coefficients, state_cov))
```

#### 3.2.3 UnobservedComponents (结构模型)

**文件位置**：`structural.py:1053-1135`

**参数结构与变换映射**：

| 参数类型 | 变换函数 | 变换方式 | 行号 |
|---------|---------|-----------|------|
| **观测方差** | 平方运算 | `x**2` | 1061-1063 |
| **状态方差** | 平方运算 | `x**2` | 1061-1063 |
| **周期频率** | sigmoid 变换 | 映射到区间 | 1065-1071 |
| **周期阻尼** | sigmoid 变换 | 映射到 (0,1) | 1074-1077 |
| **AR 系数** | `constrain_stationary_univariate` | 直接 | 1079-1086 |
| **回归系数** | 无 | 直接使用 | 1088-1091 |

**关键代码**：

```python
# 周期频率: 映射到 [low, high] 区间
constrained[offset] = (1 / (1 + np.exp(-unconstrained[offset]))) * (
    high - low
) + low

# 周期阻尼: 映射到 (0, 1)
constrained[offset] = 1 / (1 + np.exp(-unconstrained[offset]))

# AR 系数: 复用单变量平稳性约束
constrained[offset : offset + self.ar_order] = (
    constrain_stationary_univariate(
        unconstrained[offset : offset + self.ar_order]
    )
)
```

#### 3.2.4 DynamicFactor (动态因子模型)

**文件位置**：`dynamic_factor.py:658-822`

**参数结构与变换映射**：

| 参数类型 | 变换函数 | 变换方式 | 行号 |
|---------|---------|-----------|------|
| **因子载荷** | 无 | 直接使用 | 684-687 |
| **外生回归** | 无 | 直接使用 | 689-692 |
| **误差协方差** | 平方/Cholesky | 视类型 | 694-702 |
| **因子 VAR** | `constrain_stationary_multivariate` | 多变量 | 704-721 |
| **误差 VAR (联合)** | `constrain_stationary_multivariate` | 多变量 | 725-739 |
| **误差 AR (分离)** | `constrain_stationary_univariate` | 单变量循环 | 741-749 |

**关键代码**：

```python
# 因子 VAR: 多变量变换
unconstrained_matrices = (
    unconstrained[self._params_factor_transition].reshape(
        self.k_factors, self._factor_order))
coefficient_matrices, variance = (
    constrain_stationary_multivariate(unconstrained_matrices, cov))

# 误差 VAR (分离 AR 规格): 单变量循环
coefficients = unconstrained[self._params_error_transition].copy()
for i in range(self.k_endog):
    start = i * self.error_order
    end = (i + 1) * self.error_order
    coefficients[start:end] = constrain_stationary_univariate(
        coefficients[start:end])
```

### 3.3 复用关系汇总表

| 变换函数 | SARIMAX | VARMAX | UnobservedComponents | DynamicFactor |
|---------|---------|--------|---------------------|---------------|
| `constrain_stationary_univariate` | ✅ AR/MA/季节 | ❌ | ✅ AR 分量 | ✅ 误差 AR |
| `constrain_stationary_multivariate` | ❌ | ✅ VAR/VMA | ❌ | ✅ 因子 VAR |
| 平方运算 (方差) | ✅ | ✅ | ✅ | ✅ |
| sigmoid (区间) | ❌ | ❌ | ✅ 周期 | ❌ |
| Cholesky 分解 | ❌ | ✅ 非结构化协方差 | ❌ | ✅ 非结构化协方差 |

### 3.4 变换逻辑设计模式

**模板方法模式**：

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         MLEModel (基类)                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  def loglike(self, params, ...):                                             │
│      # 固定流程（模板方法）                                                     │
│      params = self.handle_params(params, ...)    ← 固定步骤                  │
│      self.update(params, ...)                     ← 钩子方法（子类覆盖）        │
│      return self.ssm.loglike(...)               ← 固定步骤               │
│                                                                              │
│  def handle_params(self, params, ...):                                        │
│      # 固定流程                                                             │
│      if not transformed:                                                      │
│          params = self.transform_params(params)   ← 钩子方法（子类覆盖）       │
│      ...                                                                      │
│                                                                              │
│  # 以下为钩子方法（基类提供默认或需子类覆盖）                                │
│  ┌────────────────────────────────────────────────────────────────────────┐
│  │ def transform_params(self, unconstrained):                                 │
│  │     # 基类默认: 不做变换                                               │
│  │     return unconstrained                                                │
│  │                                                                      │
│  │ def update(self, params, ...):                                          │
│  │     # 基类默认: 仅处理参数                                             │
│  │     return self.handle_params(params, ...)                               │
│  └────────────────────────────────────────────────────────────────────────┘
│                                                                              │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ 继承
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         具体模型 (SARIMAX 等)                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  def transform_params(self, unconstrained):                                 │
│      # 覆盖: 实现特定的参数约束变换                                          │
│      constrained = np.zeros(...)                                                 │
│      # 复用 tools.py 中的变换函数                                            │
│      constrained[ar_idx] = constrain_stationary_univariate(...)              │
│      constrained[ma_idx] = -constrain_stationary_univariate(...)            │
│      constrained[var_idx] = unconstrained[var_idx]**2                        │
│      return constrained                                                       │
│                                                                              │
│  def update(self, params, ...):                                               │
│      # 覆盖: 实现参数到状态空间矩阵的映射                                    │
│      # 提取各部分参数                                                          │
│      # 构建滞后多项式                                                        │
│      # 更新 self.ssm["...                                                      │
│      return params                                                            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 各模型实现对比

### 4.1 状态空间矩阵构建对比

| 特性 | SARIMAX | VARMAX | UnobservedComponents | DynamicFactor |
|-----|---------|--------|---------------------|---------------|
| **过渡矩阵** | 伴随矩阵 | 分块伴随矩阵 | 分量对角块 | 因子分块 |
| **设计矩阵** | [1, 0, ..., 0] | 单位矩阵前几行 | 各分量载荷 | 因子载荷矩阵 |
| **选择矩阵** | 伴随矩阵形式 | 单位矩阵前几列 | 分量选择 | 分块选择 |
| **状态协方差** | 单元素 | 多元素 | 分量方差 | 因子+误差 |
| **AR/MA 参数位置** | transition + selection | transition | 无 | 无 |
| **差分处理** | 纳入状态向量 | 不支持 | 无 | 无 |
| **时变矩阵** | 支持 (trend/exog) | 支持 (trend) | 支持 | 支持 |

### 4.2 参数向量结构对比

#### SARIMAX 参数向量

```
SARIMAX(p,d,q)(P,D,Q)s 与 exog, trend='ct'

params = [
    # 1. 趋势参数
    δ₀ (constant),
    δ₁ (linear),
    
    # 2. MLE 回归系数 (如果 mle_regression=True)
    β₁, β₂, ..., β_{k_exog},
    
    # 3. AR 参数
    φ₁, φ₂, ..., φ_p,
    
    # 4. MA 参数
    θ₁, θ₂, ..., θ_q,
    
    # 5. 季节 AR 参数
    Φ₁, Φ₂, ..., Φ_P,
    
    # 6. 季节 MA 参数
    Θ₁, Θ₂, ..., Θ_Q,
    
    # 7. 方差参数
    σ_η² (状态噪声),
    σ_ε² (测量误差, 如果 measurement_error=True)
]
```

#### VARMAX 参数向量

```
VARMAX(p,q) 与 trend='c'

params = [
    # 1. 截距项 (k_endog 个)
    ν₁, ν₂, ..., ν_{k_endog},
    
    # 2. VAR 参数 (k_endog × k_endog × p 个)
    A₁₁¹, A₁₂¹, ..., A₁_{k_endog}¹,  # A₁ 矩阵第一行
    A₂₁¹, A₂₂¹, ..., A₂_{k_endog}¹,  # A₁ 矩阵第二行
    ...
    A_{k_endog}_1¹, ...,              # A₁ 矩阵最后一行
    ... (重复 p 次)
    
    # 3. VMA 参数 (k_endog × k_endog × q 个)
    M₁₁¹, M₁₂¹, ...,                   # M₁ 矩阵
    ...
    
    # 4. 回归系数 (如果有 exog)
    B₁₁, B₁₂, ...,
    
    # 5. 状态协方差 (下三角元素)
    # diagonal: L₁₁, L₂₂, ..., L_{k_endog}{k_endog}
    # 或下三角: L₁₁, L₂₁, L₂₂, ...
    
    # 6. 测量误差方差 (如果 measurement_error=True)
    σ_ε₁², σ_ε₂², ...
]
```

#### UnobservedComponents 参数向量

```
UnobservedComponents 包含: local linear trend + seasonal(12) + cycle + AR(2)

params = [
    # 1. 观测方差 (irregular)
    σ_ε²,
    
    # 2. 状态方差
    σ_η² (level),
    σ_ζ² (slope),
    σ_ω² (seasonal),
    σ_κ² (cycle),
    
    # 3. 周期参数
    λ_c (cycle frequency),
    φ_c (cycle damping, 如果 damped_cycle=True),
    
    # 4. AR 系数
    φ₁, φ₂,
    
    # 5. 回归系数 (如果有 exog)
    β₁, β₂, ...
]
```

### 4.3 update() 方法实现对比

| 步骤 | SARIMAX | VARMAX | UnobservedComponents | DynamicFactor |
|-----|---------|--------|---------------------|---------------|
| **参数分段** | 按类型分段 | 按类型分段 | 按分量分段 | 按层次分段 |
| **多项式构建** | AR/MA 多项式相乘 | 直接使用矩阵 | 无 | 无 |
| **过渡矩阵更新** | 第一列/行 | 分块填充 | 各分量独立 | 因子+误差 |
| **设计矩阵更新** | 通常固定 | 通常固定 | 分量载荷 | 因子载荷 |
| **选择矩阵更新** | MA 参数 | 通常固定 | 分量选择 | 分块 |
| **协方差更新** | 单元素 | 多元素/Cholesky | 多元素 | 多元素/Cholesky |
| **时变矩阵** | trend/exog) | trend | 无 | 无 |

---

## 5. 完整代码引用索引

### 5.1 类继承与职责

| 类 | 文件 | 行号范围 | 核心职责 |
|---|------|----------|---------|
| **Representation** | `representation.py` | 87-500+ | 状态空间矩阵存储与访问 |
| **KalmanFilter** | `kalman_filter.py` | 62-600+ | 卡尔曼滤波算法 |
| **SimulationSmoother** | `simulation_smoother.py` | - | 模拟平滑扩展 |
| **MLEModel** | `mlemodel.py` | 93-2000+ | MLE 估计框架 |
| **SARIMAX** | `sarimax.py` | 37-1800+ | 单变量 ARIMA/SARIMA |
| **VARMAX** | `varmax.py` | 36-1000+ | 多变量 VARMA |
| **UnobservedComponents** | `structural.py` | 47-1500+ | 结构时间序列 |
| **DynamicFactor** | `dynamic_factor.py` | 33-1200+ | 动态因子模型 |

### 5.2 参数更新链路关键方法

#### MLEModel 层

| 方法 | 文件:行号 | 职责 |
|-----|----------|------|
| `fit()` | `mlemodel.py:540-787 | 协调优化器与滤波 |
| `loglike()` | `mlemodel.py:987-1040` | 计算对数似然 |
| `filter()` | `mlemodel.py:848-919` | 执行卡尔曼滤波 |
| `handle_params()` | `mlemodel.py:1930-1963` | 参数预处理 |
| `update()` (基类) | `mlemodel.py:1965-1991` | 参数映射（需覆盖） |

#### 具体模型层

| 模型 | 方法 | 文件:行号 |
|-----|------|----------|
| SARIMAX | `transform_params()` | `sarimax.py:1299-1401` |
| SARIMAX | `untransform_params()` | `sarimax.py:1403-1505` |
| SARIMAX | `update()` | `sarimax.py:1530-1719` |
| VARMAX | `transform_params()` | `varmax.py:518-599` |
| VARMAX | `update()` | `varmax.py:708-850` |
| UnobservedComponents | `transform_params()` | `structural.py:1053-1093` |
| UnobservedComponents | `update()` | `structural.py:1151-1300` |
| DynamicFactor | `transform_params()` | `dynamic_factor.py:658-755` |
| DynamicFactor | `update()` | `dynamic_factor.py:879-1050` |

### 5.3 工具函数

| 函数 | 文件:行号 | 功能 |
|-----|----------|------|
| `constrain_stationary_univariate()` | `tools.py:481-515` | 单变量平稳性约束 |
| `unconstrain_stationary_univariate()` | `tools.py:518-552` | 单变量逆变换 |
| `constrain_stationary_multivariate()` | `tools.py:555-650` | 多变量平稳性约束 |
| `unconstrain_stationary_multivariate()` | `tools.py:653-750` | 多变量逆变换 |
| `companion_matrix()` | `tools.py:127-245` | 伴随矩阵创建 |

### 5.4 滤波算法实现

| 函数 | 文件 | 功能 |
|-----|------|------|
| `forecast_conventional()` | `_conventional.pyx.in:108-169` | 标准预测步 |
| `updating_conventional()` | `_conventional.pyx.in:200+` | 标准更新步 |
| `forecast_missing_conventional()` | `_conventional.pyx.in:54-79` | 缺失值预测步 |
| `updating_missing_conventional()` | `_conventional.pyx.in:80-87` | 缺失值更新步 |
| `loglikelihood_missing_conventional()` | `_conventional.pyx.in:95-96` | 缺失值似然 |
| 单变量滤波系列 | `_univariate.pyx.in` | 单变量逐次更新 |

---

## 附录：关键设计模式总结

### A.1 模板方法模式 (Template Method)

**应用场景**：`MLEModel.loglike()` 方法

```python
class MLEModel:
    def loglike(self, params, ...):
        # 固定算法骨架
        params = self.handle_params(params, ...)    # 固定步骤
        self.update(params, ...)                     # 可变步骤（钩子）
        return self.ssm.loglike(...)                 # 固定步骤
```

**子类覆盖钩子方法：
```python
class SARIMAX(MLEModel):
    def update(self, params, ...):
        # 实现特定的参数映射逻辑
        self.ssm["transition"] = ...
        self.ssm["selection"] = ...
```

### A.2 描述符模式 (Descriptor)

**应用场景**：`MatrixWrapper` 管理状态空间矩阵

```python
class MatrixWrapper:
    def __get__(self, obj, objtype):
        return getattr(obj, self._attribute)
    
    def __set__(self, obj, value):
        # 自动处理形状验证、内存布局
        value = np.asarray(value, order="F")
        setattr(obj, self._attribute, value)

class Representation:
    design = MatrixWrapper("design", "design")
    transition = MatrixWrapper("transition", "transition")
    # ...
```

### A.3 策略模式 (Strategy)

**应用场景**：多种滤波方法选择

```python
# 滤波方法作为策略
FILTER_CONVENTIONAL = 0x01
FILTER_UNIVARIATE = 0x10
FILTER_COLLAPSED = 0x20

# 运行时选择策略
if nmissing[t] > 0:
    # 使用单变量滤波策略
    use_filter = FILTER_UNIVARIATE
else:
    # 使用常规滤波策略
    use_filter = FILTER_CONVENTIONAL
```

### A.4 工厂模式 (Factory)

**应用场景**：状态空间类根据数据类型选择实现

```python
# 根据数据类型选择不同精度的实现
prefix_statespace_map = {
    "s": _representation.sStatespace,   # float32
    "d": _representation.dStatespace,   # float64
    "c": _representation.cStatespace,   # complex64
    "z": _representation.zStatespace,   # complex128
}

# 运行时创建实例
StatespaceClass = prefix_statespace_map[prefix]
model = StatespaceClass(...)
```

---

## 参考资料

1. Monahan, John F. (1984). "A Note on Enforcing Stationarity in Autoregressive-moving Average Models." Biometrika 71(2): 403-404.

2. Ansley, C. F., & Kohn, R. (1986). "A Note on Reparameterizing a Vector Autoregressive Model to Enforce Stationarity." Biometrika 73(3): 737-740.

3. Durbin, J., & Koopman, S. J. (2012). Time Series Analysis by State Space Methods (2nd ed.). Oxford University Press.
