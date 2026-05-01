# Statsmodels 状态空间框架实现分析（V2修订版）

## 目录
1. [架构概述](#1-架构概述)
2. [状态空间模型定义](#2-状态空间模型定义)
3. [状态转移方程实现](#3-状态转移方程实现)
4. [Kalman 滤波实现](#4-kalman-滤波实现)
5. [后向平滑实现](#5-后向平滑实现)
6. [缺失观测场景下的处理机制](#6-缺失观测场景下的处理机制)
7. [Diffuse 阶段与单变量回退处理](#7-diffuse-阶段与单变量回退处理)
8. [数据流转与同步机制](#8-数据流转与同步机制)
9. [滤波与平滑的闭环关系](#9-滤波与平滑的闭环关系)
10. [完整流程图](#10-完整流程图)

---

## 6. 缺失观测场景下的处理机制

### 6.1 缺失观测的三种分类

statsmodels 将缺失观测场景分为三类，每类有不同的处理逻辑：

| 类别 | 条件 | 处理函数 | 说明 |
|------|------|----------|------|
| **无缺失** | `nmissing = 0` | 标准函数 | 完整观测的标准处理 |
| **部分缺失** | `0 < nmissing < k_endog` | `_select_missing_partial_obs` | 部分观测缺失，部分完整 |
| **完全缺失** | `nmissing == k_endog` | `_select_missing_entire_obs` | 所有观测都缺失 |

**关键代码位置**: `_representation.pyx.in:615-645`

```python
cdef int select_missing(self, unsigned int t):
    # 设置当前迭代的缺失数量
    self._nmissing = self.nmissing[t]
    
    # 完全缺失
    if self._nmissing == self.k_endog:
        self._select_missing_entire_obs(t)
    # 部分缺失
    elif self._nmissing > 0:
        self._select_missing_partial_obs(t)
        k_endog = self.k_endog - self._nmissing
    
    return k_endog
```

### 6.2 完全缺失观测的处理

**处理函数**: `_select_missing_entire_obs` (`_representation.pyx.in:647-655`)

```python
cdef void _select_missing_entire_obs(self, unsigned int t):
    # 设计矩阵设为零矩阵
    for i in range(self.k_states):
        for j in range(self.k_endog):
            self.selected_design[j + i*self.k_endog] = 0.0
    self._design = &self.selected_design[0]
```

**数学含义**：
- 设计矩阵 `Z_t = 0`（零矩阵）
- 观测方程变为：`y_t = 0 * α_t + d_t + ε_t`（实际上观测不可用）
- 更新步骤被跳过，滤波状态 = 预测状态

**滤波时的处理** (`_filters/_conventional.pyx.in:54-99`):

```python
# 预测误差设为零
for i in range(kfilter.k_endog):
    kfilter._forecast[i] = 0
    kfilter._forecast_error[i] = 0

# 预测误差协方差设为零
for i in range(kfilter.k_endog):
    for j in range(kfilter.k_endog):
        kfilter._forecast_error_cov[j + i*kfilter.k_endog] = 0

# 更新步骤：直接复制预测状态到滤波状态
# α_{t|t} = α_{t|t-1}
# P_{t|t} = P_{t|t-1}
blas.{{prefix}}copy(&kfilter.k_states, kfilter._input_state, &inc, kfilter._filtered_state, &inc)
blas.{{prefix}}copy(&kfilter.k_states2, kfilter._input_state_cov, &inc, kfilter._filtered_state_cov, &inc)

# 对数似然设为零
loglikelihood_missing_conventional(kfilter, model, determinant):
    return 0.0
```

**平滑时的处理** (`_smoothers/_conventional.pyx.in:91-206`):

当检测到完全缺失时，函数指针会切换：

```python
# _kalman_smoother.pyx.in:784-792
if not diffuse and self._smooth_method & SMOOTH_CONVENTIONAL and self.model._nmissing == self.model.k_endog:
    # 切换到缺失观测处理函数
    self.smooth_estimators_measurement = {{prefix}}smoothed_estimators_missing_conventional
    self.smooth_disturbances = {{prefix}}smoothed_disturbances_missing_conventional
```

**完全缺失时的平滑递推公式**：

```python
# scaled_smoothed_estimators_missing_conventional
# r_{t-1} = T_t' r_t
blas.{{prefix}}gemv("T", &model._k_states, &model._k_states,
          &alpha, model._transition, &model._k_states,
                  smoother._input_scaled_smoothed_estimator, &inc,
          &beta, smoother._scaled_smoothed_estimator, &inc)

# N_{t-1} = T_t' N_t T_t
# (通过时间更新函数处理)
```

### 6.3 部分缺失观测的处理

**处理函数**: `_select_missing_partial_obs` (`_representation.pyx.in:657-687`)

这是最复杂的处理场景，需要选择性地复制非缺失的观测和矩阵行。

```python
cdef void _select_missing_partial_obs(self, unsigned int t):
    cdef:
        int i, j, k, l
        int inc = 1
        int k_endog = self.k_endog - self._nmissing
    
    k = 0  # selected_* 数组的索引
    for i in range(self.k_endog):
        if not self.missing[i, t]:  # 只处理非缺失的观测
            # 1. 复制观测值 y_t
            self.selected_obs[k] = self._obs[i]
            self.selected_obs_intercept[k] = self._obs_intercept[i]
            
            # 2. 复制设计矩阵 Z_t 的第 i 行
            # Z^*_t = W_t Z_t，其中 W_t 是选择矩阵
            blas.{{prefix}}copy(&self.k_states,
                  &self._design[i], &self.k_endog,
                  &self.selected_design[k], &k_endog)
            
            # 3. 复制观测协方差矩阵 H_t 的子矩阵
            # H^*_t = W_t H_t W_t'
            l = 0
            for j in range(self.k_endog):
                if not self.missing[j, t]:
                    self.selected_obs_cov[l + k*k_endog] = self._obs_cov[j + i*self.k_endog]
                    l += 1
            k += 1
    
    # 重定向指针到 selected_* 数组
    self._obs = &self.selected_obs[0]
    self._obs_intercept = &self.selected_obs_intercept[0]
    self._design = &self.selected_design[0]
    self._obs_cov = &self.selected_obs_cov[0]
```

**数学含义**：

使用**选择矩阵** `W_t`（`(k_endog^* × k_endog)` 矩阵，`k_endog^* = k_endog - nmissing`）来选择非缺失的观测：

```
W_t = [e_{i_1}, e_{i_2}, ..., e_{i_{k_endog^*}}]'

其中 e_j 是第 j 个标准基向量，i_1, ..., i_{k_endog^*} 是非缺失观测的索引
```

**变换后的观测方程**：

```
y^*_t = W_t y_t
Z^*_t = W_t Z_t
H^*_t = W_t H_t W_t'
d^*_t = W_t d_t
```

**滤波时的特殊处理**：

部分缺失时，`k_endog` 会动态变化：

```python
# select_missing 返回值
k_endog = self.k_endog - self._nmissing

# set_dimensions 更新 _k_endog
self.set_dimensions(k_endog, self.k_states, self.k_posdef)
```

**关键点**：
- 滤波和平滑的核心算法**不需要改变**
- 只需要在迭代前通过 `select_missing()` 准备好 `selected_*` 数组
- 指针重定向后，BLAS/LAPACK 调用会自动使用正确的维度

### 6.4 三种场景的处理对比

| 维度 | 无缺失 | 部分缺失 | 完全缺失 |
|------|--------|----------|----------|
| `_k_endog` | `k_endog` | `k_endog - nmissing` | `k_endog` |
| `_design` | 原始 `design` | `selected_design`（部分行） | `selected_design`（零矩阵） |
| `_obs` | 原始 `obs` | `selected_obs`（部分元素） | 原始 `obs`（但未使用） |
| `_obs_cov` | 原始 `obs_cov` | `selected_obs_cov`（子矩阵） | 原始 `obs_cov`（但未使用） |
| 滤波更新 | 标准更新 | 标准更新（使用 reduced 维度） | 跳过更新（`α_{t|t} = α_{t|t-1}`） |
| 平滑递推 | 标准递推 | 标准递推（使用滤波时的中间结果） | 特殊递推（`r_{t-1} = T_t' r_t`） |

### 6.5 缺失观测下的状态扰动协方差（修正版）

#### 6.5.1 正确的公式

根据 `_smoothers/_conventional.pyx.in:177-189` 的代码，**状态扰动协方差的公式在所有场景下都是相同的**：

```python
# Smoothed state disturbance covariance matrix
# $Var(\eta_t | Y_n) = Q_t - \#_0' N_t \#_0$
# $(r \times r) = (r \times r) - (r \times m) (m \times m) (m \times r)$

# 其中 $\#_0 = R_t Q_t$
blas.{{prefix}}gemm("N", "N", &model._k_states, &model._k_posdef, &model._k_posdef,
          &alpha, model._selection, &model._k_states,
                  model._state_cov, &model._k_posdef,
          &beta, smoother._tmp0, &kfilter.k_states)

# $Var(\eta_t | Y_n) = Q_t - (R_t Q_t)' N_t (R_t Q_t)$
# 等价于: $Q_t - Q_t R_t' N_t R_t Q_t$
blas.{{prefix}}gemm("N", "N", &model._k_states, &model._k_posdef, &model._k_states,
          &alpha, smoother._input_scaled_smoothed_estimator_cov, &kfilter.k_states,
                  smoother._tmp0, &kfilter.k_states,
          &beta, smoother._tmpL, &kfilter.k_states)
blas.{{prefix}}copy(&model._k_posdef2, model._state_cov, &inc, smoother._smoothed_state_disturbance_cov, &inc)
blas.{{prefix}}gemm("T", "N", &model._k_posdef, &model._k_posdef, &model._k_states,
          &gamma, smoother._tmp0, &kfilter.k_states,
                  smoother._tmpL, &kfilter.k_states,
          &alpha, smoother._smoothed_state_disturbance_cov, &kfilter.k_posdef)
```

#### 6.5.2 观测扰动协方差

观测扰动协方差在完全缺失时使用**无条件分布**：

```python
# _smoothers/_conventional.pyx.in:202-204
# Smoothed measurement and state disturbances have unconditional covariance
# matrix of $H_t, Q_t$, respectively
blas.{{prefix}}copy(&model._k_endog2, model._obs_cov, &inc, smoother._smoothed_measurement_disturbance_cov, &inc)
```

#### 6.5.3 修正后的对比表

| 扰动协方差 | 有观测时 | 完全缺失时 | 原因 |
|------------|----------|------------|------|
| `Var(ε_t \| Y_n)` | `H_t - H_t (F_t^{-1} - K_t' N_t K_t) H_t` | `H_t` | 无观测，无法修正，使用无条件分布 |
| `Var(η_t \| Y_n)` | `Q_t - Q_t R_t' N_t R_t Q_t` | **相同公式** | 状态扰动影响未来状态，`N_t` 仍包含未来信息 |

#### 6.5.4 关键解释

**为什么状态扰动协方差的公式不变？**

考虑状态方程：

```
α_{t+1} = T_t α_t + R_t η_t
```

状态扰动 `η_t` 会影响 `α_{t+1}, α_{t+2}, ..., α_n`，进而影响观测 `y_{t+1}, ..., y_n`。

因此，即使 `y_t` 缺失：
- `η_t` 仍然影响未来观测
- `N_t` 仍然包含未来观测的信息（通过后向递推）
- 公式 `Var(η_t | Y_n) = Q_t - Q_t R_t' N_t R_t Q_t` 仍然适用

**为什么观测扰动协方差需要特殊处理？**

观测扰动 `ε_t` 只影响 `y_t`：

```
y_t = Z_t α_t + d_t + ε_t
```

如果 `y_t` 缺失：
- 没有关于 `ε_t` 的任何信息
- 只能使用无条件分布 `Var(ε_t | Y_n) = H_t`
- 这就是为什么代码中要显式复制 `model._obs_cov` 到 `_smoothed_measurement_disturbance_cov`

### 6.6 完整的平滑扰动处理代码分析

让我们详细分析 `smoothed_disturbances_conventional` 和 `smoothed_disturbances_missing_conventional` 的差异：

#### 6.6.1 有观测时的扰动计算

```python
# _smoothers/_conventional.pyx.in:250-350
cdef int {{prefix}}smoothed_disturbances_conventional({{prefix}}KalmanSmoother smoother, {{prefix}}KalmanFilter kfilter, {{prefix}}Statespace model) except *:
    cdef:
        int inc = 1
        {{cython_type}} alpha = 1.0
        {{cython_type}} beta = 0.0
        {{cython_type}} gamma = -1.0
    
    # 临时矩阵计算
    # $\#_0 = R_t Q_t$
    if smoother.smoother_output & (SMOOTHER_DISTURBANCE | SMOOTHER_DISTURBANCE_COV):
        blas.{{prefix}}gemm("N", "N", &model._k_states, &model._k_posdef, &model._k_posdef,
                  &alpha, model._selection, &model._k_states,
                          model._state_cov, &model._k_posdef,
                  &beta, smoother._tmp0, &kfilter.k_states)
    
    # === 平滑观测扰动 ===
    # $\hat \varepsilon_t = H_t u_t$
    # $(p \times 1) = (p \times p) (p \times 1)$
    if smoother.smoother_output & SMOOTHER_DISTURBANCE:
        blas.{{prefix}}gemv("N", &model._k_endog, &model._k_endog,
                  &alpha, model._obs_cov, &model._k_endog,
                          smoother._smoothing_error, &inc,
                  &beta, smoother._smoothed_measurement_disturbance, &inc)
    
    # $Var(\varepsilon_t | Y_n) = H_t - H_t (F_t^{-1} - K_t' N_t K_t) H_t$
    # 或者更简单的形式 (代码中使用的):
    # $Var(\varepsilon_t | Y_n) = H_t - H_t u_t u_t' H_t / (v_t' F_t^{-1} v_t)$
    # 实际上代码中是通过 tmp3 = F_t^{-1} Z_t 等临时数组计算的
    if smoother.smoother_output & SMOOTHER_DISTURBANCE_COV:
        # 复杂的矩阵运算...
        # 最终结果存储在 _smoothed_measurement_disturbance_cov
    
    # === 平滑状态扰动 ===
    # $\hat \eta_t = Q_t R_t' r_t$
    # $(r \times 1) = (r \times r) (r \times m) (m \times 1)$
    if smoother.smoother_output & SMOOTHER_DISTURBANCE:
        blas.{{prefix}}gemv("T", &kfilter.k_states, &kfilter.k_posdef,
                      &alpha, smoother._tmp0, &kfilter.k_states,
                              smoother._input_scaled_smoothed_estimator, &inc,
                      &beta, smoother._smoothed_state_disturbance, &inc)
    
    # $Var(\eta_t | Y_n) = Q_t - (R_t Q_t)' N_t (R_t Q_t)$
    if smoother.smoother_output & SMOOTHER_DISTURBANCE_COV:
        # $\#_1 = N_t \#_0$
        blas.{{prefix}}gemm("N", "N", &model._k_states, &model._k_posdef, &model._k_states,
                  &alpha, smoother._input_scaled_smoothed_estimator_cov, &kfilter.k_states,
                          smoother._tmp0, &kfilter.k_states,
                  &beta, smoother._tmpL, &kfilter.k_states)
        # $Var(\eta_t | Y_n) = Q_t - \#_0' \#_1$
        blas.{{prefix}}copy(&model._k_posdef2, model._state_cov, &inc, smoother._smoothed_state_disturbance_cov, &inc)
        blas.{{prefix}}gemm("T", "N", &model._k_posdef, &model._k_posdef, &model._k_states,
                  &gamma, smoother._tmp0, &kfilter.k_states,
                          smoother._tmpL, &kfilter.k_states,
                  &alpha, smoother._smoothed_state_disturbance_cov, &kfilter.k_posdef)
```

#### 6.6.2 完全缺失时的扰动计算

```python
# _smoothers/_conventional.pyx.in:151-206
cdef int {{prefix}}smoothed_disturbances_missing_conventional({{prefix}}KalmanSmoother smoother, {{prefix}}KalmanFilter kfilter, {{prefix}}Statespace model):
    cdef:
        int inc = 1
        {{cython_type}} alpha = 1.0
        {{cython_type}} beta = 0.0
        {{cython_type}} gamma = -1.0
    
    # 临时矩阵计算（相同）
    # $\#_0 = R_t Q_t$
    if smoother.smoother_output & (SMOOTHER_DISTURBANCE | SMOOTHER_DISTURBANCE_COV):
        blas.{{prefix}}gemm("N", "N", &model._k_states, &model._k_posdef, &model._k_posdef,
                  &alpha, model._selection, &model._k_states,
                          model._state_cov, &model._k_posdef,
                  &beta, smoother._tmp0, &kfilter.k_states)
    
    # === 平滑状态扰动（相同公式！）===
    # $\hat \eta_t = Q_t R_t' r_t$
    if smoother.smoother_output & SMOOTHER_DISTURBANCE:
        blas.{{prefix}}gemv("T", &kfilter.k_states, &kfilter.k_posdef,
                      &alpha, smoother._tmp0, &kfilter.k_states,
                              smoother._input_scaled_smoothed_estimator, &inc,
                      &beta, smoother._smoothed_state_disturbance, &inc)
    
    # $Var(\eta_t | Y_n) = Q_t - (R_t Q_t)' N_t (R_t Q_t)$
    if smoother.smoother_output & SMOOTHER_DISTURBANCE_COV:
        blas.{{prefix}}gemm("N", "N", &model._k_states, &model._k_posdef, &model._k_states,
                  &alpha, smoother._input_scaled_smoothed_estimator_cov, &kfilter.k_states,
                          smoother._tmp0, &kfilter.k_states,
                  &beta, smoother._tmpL, &kfilter.k_states)
        blas.{{prefix}}copy(&model._k_posdef2, model._state_cov, &inc, smoother._smoothed_state_disturbance_cov, &inc)
        blas.{{prefix}}gemm("T", "N", &model._k_posdef, &model._k_posdef, &model._k_states,
                  &gamma, smoother._tmp0, &kfilter.k_states,
                          smoother._tmpL, &kfilter.k_states,
                  &alpha, smoother._smoothed_state_disturbance_cov, &kfilter.k_posdef)
    
    # === 观测扰动：无条件分布（不同！）===
    # "Just return the unconditional distribution for the measurement
    #  disturbances corresponding to a missing observation"
    
    # 平滑观测扰动：无条件期望为 0
    # 代码中什么都不做（_smoothed_measurement_disturbance 保持 0）
    
    # 平滑观测扰动协方差：使用 H_t（无条件分布）
    blas.{{prefix}}copy(&model._k_endog2, model._obs_cov, &inc, smoother._smoothed_measurement_disturbance_cov, &inc)
```

#### 6.6.3 关键差异总结

| 计算项 | 有观测 | 完全缺失 | 代码差异 |
|--------|--------|----------|----------|
| **平滑状态扰动均值** `η̂_t` | `Q_t R_t' r_t` | **相同** | 无差异 |
| **平滑状态扰动协方差** `Var(η_t\|Y_n)` | `Q_t - Q_t R_t' N_t R_t Q_t` | **相同** | 无差异 |
| **平滑观测扰动均值** `ε̂_t` | `H_t u_t` | `0`（无条件期望） | 缺失时不计算 `u_t` |
| **平滑观测扰动协方差** `Var(ε_t\|Y_n)` | `H_t - H_t F_t^{-1} H_t + ...` | `H_t`（无条件分布） | 缺失时直接复制 `H_t` |
| **缩放平滑估计量** `r_{t-1}` | `Z_t' u_t + L_t' r_t` | `T_t' r_t` | 缺失时无 `Z_t' u_t` 项 |
| **缩放平滑估计量协方差** `N_{t-1}` | `Z_t' F_t^{-1} Z_t + L_t' N_t L_t` | `T_t' N_t T_t` | 缺失时无 `Z_t' F_t^{-1} Z_t` 项 |

---

## 7. Diffuse 阶段与单变量回退处理

### 7.1 概述

在 Kalman 滤波和平滑过程中，有两种特殊情况需要特殊处理：

1. **Diffuse 初始化**：当初始状态协方差趋于无穷大时（`P_1 = κ P_{1,inf} + P_{1,*}`，`κ → ∞`）
2. **单变量回退**：当多变量滤波因 `F_t` 奇异而失败时，回退到逐元素的单变量滤波

### 7.2 Diffuse 初始化阶段

#### 7.2.1 数学背景

精确扩散初始化（Exact diffuse initialization）处理初始状态不确定性无穷大的情况。状态分解为：

```
α_1 = a_1 + P_{1,inf} δ + P_{1,*} η
```

其中 `δ ~ N(0, κI)`，`κ → ∞`，`η ~ N(0, I)`。

滤波需要跟踪两个协方差矩阵：
- `P_t = P_{t,*} + κ P_{t,inf}`
- `P_{t,inf}`：扩散部分（趋于无穷）
- `P_{t,*}`：常规部分（有限）

#### 7.2.2 关键数组和变量

```python
# _kalman_filter.pyx.in
cdef:
    int nobs_diffuse  # diffuse 阶段的观测数量
    
    # 扩散状态协方差
    self.predicted_diffuse_state_cov  # (k_states, k_states, nobs+1)
    self.forecast_error_diffuse_cov   # (k_endog, k_endog, nobs)
    
    # 临时数组
    self.M_inf  # (k_states, k_endog, nobs) - P_{t,inf} Z_t'
```

#### 7.2.3 Diffuse 阶段的检测

```python
# _filters/_univariate_diffuse.pyx.in
# 或通过检查 P_{t,inf} 是否为零来判断 diffuse 阶段是否结束

# 平滑时的函数指针切换
# _kalman_smoother.pyx.in:749-783
cdef void initialize_function_pointers(self) except *:
    cdef int diffuse = self.t < self.kfilter.nobs_diffuse
    
    # Diffuse 阶段使用专门的函数
    if diffuse:
        self.smooth_estimators_measurement = {{prefix}}smoothed_estimators_measurement_univariate_diffuse
        self.smooth_estimators_time = {{prefix}}smoothed_estimators_time_univariate_diffuse
        self.smooth_state = {{prefix}}smoothed_state_univariate_diffuse
        self.smooth_disturbances = {{prefix}}smoothed_disturbances_univariate_diffuse
    # ...
```

#### 7.2.4 Diffuse 阶段的数组结构

平滑器需要额外的数组来处理扩散初始化：

```python
# _kalman_smoother.pxd
cdef class {{prefix}}KalmanSmoother:
    # 扩散平滑估计量
    cdef readonly {{cython_type}} [::1,:] scaled_smoothed_diffuse_estimator           # r_{inf,t}
    cdef readonly {{cython_type}} [::1,:,:] scaled_smoothed_diffuse1_estimator_cov    # N_{1,t}
    cdef readonly {{cython_type}} [::1,:,:] scaled_smoothed_diffuse2_estimator_cov    # N_{2,t}
    
    # 指针
    cdef {{cython_type}} * _input_scaled_smoothed_diffuse_estimator
    cdef {{cython_type}} * _input_scaled_smoothed_diffuse1_estimator_cov
    cdef {{cython_type}} * _input_scaled_smoothed_diffuse2_estimator_cov
    cdef {{cython_type}} * _scaled_smoothed_diffuse_estimator
    cdef {{cython_type}} * _scaled_smoothed_diffuse1_estimator_cov
    cdef {{cython_type}} * _scaled_smoothed_diffuse2_estimator_cov
```

#### 7.2.5 指针初始化：Diffuse vs 常规

```python
# _kalman_smoother.pyx.in:710-747
cdef void initialize_smoother_object_pointers(self) except *:
    cdef:
        int t = self.t
        int diffuse = self.t < self.kfilter.nobs_diffuse
    
    # === 常规平滑估计量指针 ===
    if diffuse or self._smooth_method & (SMOOTH_CONVENTIONAL | SMOOTH_CLASSICAL | SMOOTH_UNIVARIATE):
        # Conventional/Classical/Univariate:
        #   _input_* -> r_{t+1}, N_{t+1}
        #   _scaled_* -> r_t, N_t
        self._input_scaled_smoothed_estimator = &self.scaled_smoothed_estimator[0, t+1]
        self._input_scaled_smoothed_estimator_cov = &self.scaled_smoothed_estimator_cov[0, 0, t+1]
        self._scaled_smoothed_estimator = &self.scaled_smoothed_estimator[0, t]
        self._scaled_smoothed_estimator_cov = &self.scaled_smoothed_estimator_cov[0, 0, t]
    else:  # SMOOTH_ALTERNATIVE
        # Alternative:
        #   _input_* -> r_t, N_t
        #   _scaled_* -> r_{t-1}, N_{t-1}
        self._input_scaled_smoothed_estimator = &self.scaled_smoothed_estimator[0, t]
        self._input_scaled_smoothed_estimator_cov = &self.scaled_smoothed_estimator_cov[0, 0, t]
        self._scaled_smoothed_estimator = &self.scaled_smoothed_estimator[0, t-1]
        self._scaled_smoothed_estimator_cov = &self.scaled_smoothed_estimator_cov[0, 0, t-1]
    
    # === Diffuse 平滑估计量指针（仅 Diffuse 阶段）===
    if diffuse:
        self._input_scaled_smoothed_diffuse_estimator = &self.scaled_smoothed_diffuse_estimator[0, t+1]
        self._input_scaled_smoothed_diffuse1_estimator_cov = &self.scaled_smoothed_diffuse1_estimator_cov[0, 0, t+1]
        self._input_scaled_smoothed_diffuse2_estimator_cov = &self.scaled_smoothed_diffuse2_estimator_cov[0, 0, t+1]
        self._scaled_smoothed_diffuse_estimator = &self.scaled_smoothed_diffuse_estimator[0, t]
        self._scaled_smoothed_diffuse1_estimator_cov = &self.scaled_smoothed_diffuse1_estimator_cov[0, 0, t]
        self._scaled_smoothed_diffuse2_estimator_cov = &self.scaled_smoothed_diffuse2_estimator_cov[0, 0, t]
```

### 7.3 单变量回退处理

#### 7.3.1 触发条件

当多变量滤波的 `F_t = Z_t P_{t|t-1} Z_t' + H_t` 奇异时：

```python
# _kalman_filter.pyx.in:957-977
try:
    self.determinant = self.inversion(self, self.model, self.determinant)
except np.linalg.LinAlgError:
    # 检查是否可以回退到单变量滤波
    if not ((self.inversion_method & INVERT_UNIVARIATE) or
            (self.inversion_method & SOLVE_CHOLESKY)):
        raise NotImplementedError(...)
    elif self.univariate_filter[self.t]:
        raise  # 已经是单变量，无法再回退
    else:
        # 切换到单变量滤波
        self.univariate_filter[self.t] = 1
        next(self)  # 重新执行当前迭代
        return
```

#### 7.3.2 关键数组：`univariate_filter`

```python
# _kalman_filter.pyx.in:524-525
# 标记每个时刻是否使用单变量滤波
self.univariate_filter = np.PyArray_ZEROS(1, dim1, np.NPY_INT32, FORTRAN)

# 默认值：多变量（0）或单变量（1）
if self.filter_method & FILTER_UNIVARIATE:
    self.univariate_filter[:] = 1
else:
    self.univariate_filter[:] = 0
```

#### 7.3.3 平滑时的函数指针切换

```python
# _kalman_smoother.pyx.in:749-783
cdef void initialize_function_pointers(self) except *:
    cdef int diffuse = self.t < self.kfilter.nobs_diffuse
    
    if diffuse:
        # Diffuse 阶段：使用单变量 diffuse 函数
        self.smooth_estimators_measurement = {{prefix}}smoothed_estimators_measurement_univariate_diffuse
        self.smooth_estimators_time = {{prefix}}smoothed_estimators_time_univariate_diffuse
        self.smooth_state = {{prefix}}smoothed_state_univariate_diffuse
        self.smooth_disturbances = {{prefix}}smoothed_disturbances_univariate_diffuse
    
    # 单变量滤波（或回退到单变量）
    elif (self._smooth_method & SMOOTH_UNIVARIATE) or self.kfilter.univariate_filter[self.t]:
        self.smooth_estimators_measurement = {{prefix}}smoothed_estimators_measurement_univariate
        self.smooth_estimators_time = {{prefix}}smoothed_estimators_time_univariate
        self.smooth_state = {{prefix}}smoothed_state_conventional
        self.smooth_disturbances = {{prefix}}smoothed_disturbances_univariate
    
    # 其他多变量方法...
    elif self._smooth_method & SMOOTH_ALTERNATIVE:
        # ...
    elif self._smooth_method & SMOOTH_CLASSICAL:
        # ...
    elif self._smooth_method & SMOOTH_CONVENTIONAL:
        # ...
```

#### 7.3.4 特殊情况：方法切换时的额外处理

当从多变量切换到单变量（或反之）时，需要额外的时间平滑步骤：

```python
# _kalman_smoother.pyx.in:602-641
def __next__(self):
    # ... 标准平滑步骤 ...
    
    # 检查是否需要额外的单变量时间平滑步骤
    # 两种情况：
    # 1. Diffuse 阶段结束 (nobs_diffuse > 0 且 t == nobs_diffuse)
    # 2. 前一时刻是单变量，当前是多变量 (univariate_filter[t] == 0 且 univariate_filter[t-1] == 1)
    if ((self.kfilter.nobs_diffuse > 0 and self.t == self.kfilter.nobs_diffuse) or
            (self.t > 0 and self.kfilter.univariate_filter[self.t] == 0 and self.kfilter.univariate_filter[self.t-1] == 1)):
        
        t = self.t
        
        if self._smooth_method & SMOOTH_CONVENTIONAL:
            # 执行额外的单变量时间平滑
            {{prefix}}smoothed_estimators_time_univariate(self, self.kfilter, self.model)
        
        if self._smooth_method & SMOOTH_ALTERNATIVE:
            # Alternative 方法需要调整指针
            self.t = self.t - 1
            self.initialize_statespace_object_pointers()
            self.initialize_filter_object_pointers()
            self.initialize_smoother_object_pointers()
            self.initialize_function_pointers()
            {{prefix}}smoothed_estimators_time_univariate(self, self.kfilter, self.model)
            self.t = t
        
        elif self._smooth_method & SMOOTH_CLASSICAL:
            # Classical 方法需要两个额外步骤
            self.t = t - 1
            self.initialize_statespace_object_pointers()
            self.initialize_filter_object_pointers()
            self.initialize_smoother_object_pointers()
            self.initialize_function_pointers()
            {{prefix}}smoothed_estimators_measurement_classical(self, self.kfilter, self.model)
            self.t = t
            self.initialize_statespace_object_pointers()
            self.initialize_filter_object_pointers()
            self.initialize_smoother_object_pointers()
            self.initialize_function_pointers()
            {{prefix}}smoothed_estimators_time_univariate(self, self.kfilter, self.model)
```

### 7.4 完整的指针切换与时序链路图

#### 7.4.1 滤波阶段的时序链路

```
时间轴: t=0 ── t=1 ── t=2 ── t=3 ── t=4 ── t=5
          │      │      │      │      │      │
          ▼      ▼      ▼      ▼      ▼      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                           前向滤波阶段                                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  univariate_filter 数组标记:                                              │
│  ├─ t=0: 未使用 (初始时刻)                                                │
│  ├─ t=1: 0 (多变量) 或 1 (单变量)                                        │
│  ├─ t=2: 0 (多变量) ← 假设这里 F_t 奇异，触发回退                        │
│  ├─ t=2: 1 (单变量) ← 重新执行，使用单变量                                │
│  ├─ t=3: 1 (单变量)                                                       │
│  ├─ t=4: 0 (多变量) ← F_t 非奇异，切回多变量                             │
│  └─ t=5: 0 (多变量)                                                       │
│                                                                          │
│  nobs_diffuse: 假设 = 2 (t=0,1 是 diffuse 阶段)                          │
│                                                                          │
│  每个时刻的函数选择 (initialize_function_pointers):                       │
│  ├─ t < nobs_diffuse: 使用 univariate_diffuse 函数                       │
│  ├─ t >= nobs_diffuse 且 univariate_filter[t]==1: 使用 univariate 函数   │
│  └─ t >= nobs_diffuse 且 univariate_filter[t]==0: 使用 conventional/...  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 7.4.2 平滑阶段的时序链路

```
时间轴: t=5 ── t=4 ── t=3 ── t=2 ── t=1 ── t=0
          │      │      │      │      │      │
          ▼      ▼      ▼      ▼      ▼      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                           后向平滑阶段                                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  初始化:                                                                  │
│  ├─ r_n = 0, N_n = 0 (对于常规)                                          │
│  ├─ r_{inf,n} = 0, N_{1,n} = 0, N_{2,n} = 0 (对于 diffuse)             │
│  └─ self.t = nobs - 1 = 5                                                │
│                                                                          │
│  迭代过程 (__next__):                                                     │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ t=5 (非 diffuse, 多变量):                                            │  │
│  │  ├─ initialize_smoother_object_pointers():                           │  │
│  │  │   ├─ _input_* -> r_6, N_6 (初值 0)                              │  │
│  │  │   └─ _scaled_* -> r_5, N_5 (待计算)                              │  │
│  │  ├─ initialize_function_pointers():                                  │  │
│  │  │   └─ 选择 conventional 函数 (t >= nobs_diffuse, univariate=0)    │  │
│  │  ├─ 执行: measurement, smooth_state, smooth_disturbances, time       │  │
│  │  └─ 检查额外步骤: t=4 时 univariate_filter[t-1]=1？ 否               │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              │                                             │
│                              ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ t=4 (非 diffuse, 多变量):                                            │  │
│  │  ├─ 注意: univariate_filter[4] = 0, univariate_filter[3] = 1        │  │
│  │  ├─ 标准步骤: measurement, smooth_state, smooth_disturbances, time    │  │
│  │  └─ 额外步骤 (关键!):                                                 │  │
│  │      ├─ 条件: t=4>0 且 univariate_filter[4]==0 且 univariate_filter[3]==1 │  │
│  │      └─ 执行: smoothed_estimators_time_univariate() 一次             │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              │                                             │
│                              ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ t=3 (非 diffuse, 单变量):                                            │  │
│  │  ├─ initialize_function_pointers():                                  │  │
│  │  │   └─ 选择 univariate 函数 (univariate_filter[3] = 1)             │  │
│  │  └─ 执行: univariate 版本的函数                                      │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              │                                             │
│                              ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ t=2 (非 diffuse, 单变量):                                            │  │
│  │  └─ 类似 t=3                                                         │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              │                                             │
│                              ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ t=1 (diffuse 阶段):                                                  │  │
│  │  ├─ 条件: self.t < self.kfilter.nobs_diffuse (1 < 2)               │  │
│  │  ├─ initialize_smoother_object_pointers():                           │  │
│  │  │   └─ 额外设置 diffuse 指针: r_{inf,t}, N_{1,t}, N_{2,t}         │  │
│  │  ├─ initialize_function_pointers():                                  │  │
│  │  │   └─ 选择 univariate_diffuse 函数                                 │  │
│  │  └─ 执行: diffuse 版本的函数                                         │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              │                                             │
│                              ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ t=0 (diffuse 阶段结束):                                              │  │
│  │  ├─ 检查额外步骤: t == nobs_diffuse (0 == 2？ 否)                    │  │
│  │  └─ 注意: 如果 t == nobs_diffuse，需要额外的 univariate time 步骤   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 7.4.3 不同平滑方法的时序差异

| 平滑方法 | `_input_*` 指向 | `_scaled_*` 指向 | 额外步骤处理 |
|----------|-----------------|------------------|--------------|
| **Conventional** | `r_{t+1}, N_{t+1}` | `r_t, N_t` | 直接执行 `time_univariate` |
| **Alternative** | `r_t, N_t` | `r_{t-1}, N_{t-1}` | 先 `t -= 1`，再 `time_univariate`，再 `t += 1` |
| **Classical** | `r_{t+1}, N_{t+1}` | `r_t, N_t` | 先 `t -= 1`，执行 `measurement_classical`，再 `t += 1`，执行 `time_univariate` |

**代码位置**: `_kalman_smoother.pyx.in:617-640`

```python
if self._smooth_method & SMOOTH_CONVENTIONAL:
    {{prefix}}smoothed_estimators_time_univariate(self, self.kfilter, self.model)

if self._smooth_method & SMOOTH_ALTERNATIVE:
    self.t = self.t - 1  # 调整指针
    self.initialize_statespace_object_pointers()
    self.initialize_filter_object_pointers()
    self.initialize_smoother_object_pointers()
    self.initialize_function_pointers()
    {{prefix}}smoothed_estimators_time_univariate(self, self.kfilter, self.model)
    self.t = t  # 恢复

elif self._smooth_method & SMOOTH_CLASSICAL:
    self.t = t - 1  # 先处理 measurement
    self.initialize_statespace_object_pointers()
    self.initialize_filter_object_pointers()
    self.initialize_smoother_object_pointers()
    self.initialize_function_pointers()
    {{prefix}}smoothed_estimators_measurement_classical(self, self.kfilter, self.model)
    self.t = t  # 再处理 time
    self.initialize_statespace_object_pointers()
    self.initialize_filter_object_pointers()
    self.initialize_smoother_object_pointers()
    self.initialize_function_pointers()
    {{prefix}}smoothed_estimators_time_univariate(self, self.kfilter, self.model)
```

### 7.5 总结：特殊情况的处理机制

| 特殊情况 | 触发条件 | 处理方式 | 关键代码位置 |
|----------|----------|----------|--------------|
| **Diffuse 初始化** | `initialization_type == 'diffuse'` | 额外数组 `r_{inf,t}, N_{1,t}, N_{2,t}`，专用函数 | `_filters/_univariate_diffuse.pyx.in`, `_smoothers/_univariate_diffuse.pyx.in` |
| **单变量滤波** | `filter_method & FILTER_UNIVARIATE` | 逐元素处理，避免矩阵求逆 | `_filters/_univariate.pyx.in` |
| **单变量回退** | `F_t` 奇异 + `inversion_method` 支持 | 动态设置 `univariate_filter[t] = 1`，重新执行 | `_kalman_filter.pyx.in:957-977` |
| **方法切换平滑** | `univariate_filter[t] != univariate_filter[t-1]` 或 `t == nobs_diffuse` | 额外的 `time_univariate` 步骤，指针调整 | `_kalman_smoother.pyx.in:602-641` |

---

## 附录

### A. 核心文件位置

| 文件 | 路径 | 说明 |
|------|------|------|
| `_representation.pyx.in` | `statsmodels/tsa/statespace/` | 缺失观测选择逻辑 (`select_missing`, `_select_missing_*`) |
| `_kalman_filter.pyx.in` | `statsmodels/tsa/statespace/` | 单变量回退触发逻辑 (`__next__` 中的异常处理) |
| `_kalman_smoother.pyx.in` | `statsmodels/tsa/statespace/` | 函数指针切换 (`initialize_function_pointers`)，特殊情况处理 (`__next__` 中的额外步骤) |
| `_conventional.pyx.in` | `statsmodels/tsa/statespace/_filters/` | 完全缺失的滤波函数 (`*_missing_conventional`) |
| `_conventional.pyx.in` | `statsmodels/tsa/statespace/_smoothers/` | 完全缺失的平滑函数，扰动协方差公式 |
| `_univariate.pyx.in` | `statsmodels/tsa/statespace/_filters/` | 单变量滤波实现 |
| `_univariate_diffuse.pyx.in` | `statsmodels/tsa/statespace/_filters/` | Diffuse 单变量滤波 |
| `_univariate.pyx.in` | `statsmodels/tsa/statespace/_smoothers/` | 单变量平滑实现 |
| `_univariate_diffuse.pyx.in` | `statsmodels/tsa/statespace/_smoothers/` | Diffuse 单变量平滑 |

### B. 关键数组速查

| 数组名 | 维度 | 说明 |
|--------|------|------|
| `missing` | `(k_endog, nobs)` | 布尔矩阵，标记哪些观测缺失 |
| `nmissing` | `(nobs,)` | 每个时刻缺失的观测数量 |
| `univariate_filter` | `(nobs,)` | 标记每个时刻是否使用单变量滤波 |
| `nobs_diffuse` | `int` | Diffuse 阶段的观测数量 |
| `scaled_smoothed_estimator` | `(k_states, nobs+1)` | `r_t` - 缩放平滑估计量 |
| `scaled_smoothed_estimator_cov` | `(k_states, k_states, nobs+1)` | `N_t` - 缩放平滑估计量协方差 |
| `scaled_smoothed_diffuse_estimator` | `(k_states, nobs+1)` | `r_{inf,t}` - Diffuse 版本 |
| `scaled_smoothed_diffuse1_estimator_cov` | `(k_states, k_states, nobs+1)` | `N_{1,t}` - Diffuse 版本 |
| `scaled_smoothed_diffuse2_estimator_cov` | `(k_states, k_states, nobs+1)` | `N_{2,t}` - Diffuse 版本 |

### C. 平滑方法类型

| 常量名 | 值 | 说明 |
|--------|---|------|
| `SMOOTHER_CONVENTIONAL` | `0x01` | Durbin-Koopman 常规方法 |
| `SMOOTHER_ALTERNATIVE` | `0x02` | 替代方法（指针方向不同） |
| `SMOOTHER_CLASSICAL` | `0x04` | Anderson-Moore 经典方法 |
| `SMOOTHER_UNIVARIATE` | `0x08` | 单变量方法 |

---

*报告生成时间: 2026-05-01*

*V2修订内容:*
- 新增第6章：缺失观测场景下的处理机制（部分缺失 vs 完全缺失）
- 修正第6.5节：缺失观测下的状态扰动协方差（状态扰动公式不变，观测扰动使用无条件分布）
- 新增第7章：Diffuse 阶段与单变量回退处理（指针切换、时序链路、方法差异）
- 详细分析了 `smoothed_disturbances_*` 函数的代码实现
- 补充了完整的指针切换与时序链路图