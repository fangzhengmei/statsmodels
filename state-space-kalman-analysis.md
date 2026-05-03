# Statsmodels 状态空间模型与卡尔曼滤波实现分析

## 目录
1. [状态空间模型基础](#1-状态空间模型基础)
2. [参数结构到状态空间矩阵的映射](#2-参数结构到状态空间矩阵的映射)
3. [最大似然框架驱动卡尔曼滤波](#3-最大似然框架驱动卡尔曼滤波)
4. [参数变换逻辑与复用机制](#4-参数变换逻辑与复用机制)
5. [缺失观测值的处理路径](#5-缺失观测值的处理路径)
6. [完整代码引用索引](#6-完整代码引用索引)

---

## 1. 状态空间模型基础

### 1.1 通用状态空间形式

Statsmodels 使用的状态空间模型遵循 Durbin & Koopman (2012) 的标准形式：

```
观测方程: y_t = Z_t α_t + d_t + ε_t,  ε_t ~ N(0, H_t)
状态方程: α_{t+1} = T_t α_t + c_t + R_t η_t,  η_t ~ N(0, Q_t)
```

其中各矩阵定义如下：

| 矩阵名 | Python 属性 | 维度 | 含义 |
|--------|-------------|------|------|
| Z_t | `design` | (k_endog, k_states) | 设计矩阵（观测矩阵） |
| d_t | `obs_intercept` | (k_endog,) | 观测截距 |
| H_t | `obs_cov` | (k_endog, k_endog) | 观测噪声协方差 |
| T_t | `transition` | (k_states, k_states) | 过渡矩阵（状态转移矩阵） |
| c_t | `state_intercept` | (k_states,) | 状态截距 |
| R_t | `selection` | (k_states, k_posdef) | 选择矩阵 |
| Q_t | `state_cov` | (k_posdef, k_posdef) | 状态噪声协方差 |

### 1.2 类继承结构

```
Representation (状态空间表示基类)
    └── KalmanFilter (卡尔曼滤波实现)
            └── SimulationSmoother (模拟平滑器)
                    └── MLEModel (最大似然估计基类)
                            └── SARIMAX (SARIMA/ARIMA 模型实现)
```

关键类的职责：
- **Representation**: 管理状态空间矩阵的存储和访问
- **KalmanFilter**: 实现卡尔曼滤波算法，处理缺失值
- **MLEModel**: 提供最大似然估计框架，连接优化器和滤波
- **SARIMAX**: 实现 ARMA/ARIMA/SARIMA 模型的具体参数映射

---

## 2. 参数结构到状态空间矩阵的映射

### 2.1 SARIMAX 模型初始化

在 `sarimax.py` 的 `__init__` 方法中完成状态空间的初始化：

**初始化流程：**
1. 计算状态维度 `k_states = d + s*D + max(p+P, q+Q) + k_exog`
2. 调用父类 `MLEModel.__init__` 创建 `SimulationSmoother` 实例
3. 设置初始状态空间矩阵

**关键代码位置：** `sarimax.py:519-546`

```python
# 初始化状态空间
super().__init__(
    endog, exog=exog, k_states=k_states, k_posdef=k_posdef, **kwargs
)

# 初始化状态空间模型的固定组件
self.ssm["design"] = self.initial_design
self.ssm["state_intercept"] = self.initial_state_intercept
self.ssm["transition"] = self.initial_transition
self.ssm["selection"] = self.initial_selection
```

### 2.2 设计矩阵 (Design Matrix) 初始化

**属性：** `initial_design` (`sarimax.py:689-716`)

设计矩阵 Z_t 连接观测值 y_t 和状态向量 α_t。

**结构说明：**
```
design = [1, 1, ..., 1,      # 差分部分 (d 个)
          0,0,...,1,          # 季节性差分部分 (s*D 个)
          1, 0, 0, ..., 0]    # ARMA 部分 (max(p+P, q+Q) 个)
```

**示例：AR(2) 模型**
- p=2, d=0, q=0
- `k_states = max(2, 0) = 2`
- `design = [1, 0]`

这表示：`y_t = 1*α_{1,t} + 0*α_{2,t}`

### 2.3 过渡矩阵 (Transition Matrix) 初始化

**属性：** `initial_transition` (`sarimax.py:729-791`)

过渡矩阵 T_t 决定状态如何从 t 时刻演化到 t+1 时刻。

**核心结构：伴随矩阵 (Companion Matrix)**

对于 AR(p) 模型，过渡矩阵使用伴随矩阵形式：

```
    [φ₁  1  0  ...  0]
    [φ₂  0  1  ...  0]
T = [φ₃  0  0  ...  0]
    [...            1]
    [φₚ  0  0  ...  0]
```

**伴随矩阵创建函数：** `companion_matrix` (`tools.py:127-245`)

```python
def companion_matrix(polynomial):
    """
    创建伴随矩阵
    
    对于多项式 c(L) = c₀ + c₁L + c₂L² + ... + cₚLᵖ
    伴随矩阵第一列为: φᵢ = -cᵢ / c₀
    
    AR(p) 模型: (1 - φ₁L - ... - φₚLᵖ)y_t = ε_t
    即: c(L) = 1 - φ₁L - ... - φₚLᵖ
    所以: c₀=1, c₁=-φ₁, ..., cₚ=-φₚ
    因此: φᵢ = -cᵢ / c₀ = -(-φᵢ) / 1 = φᵢ ✓
    """
```

**两种表示方式：**

| 表示方式 | 过渡矩阵结构 | AR 参数位置 | MA 参数位置 |
|---------|-------------|------------|------------|
| **Harvey (默认)** | 标准伴随矩阵 | 第一列 | selection 矩阵 |
| **Hamilton** | 转置伴随矩阵 | 第一行 | design 矩阵 |

**关键代码：** `sarimax.py:748-754`
```python
if self._k_order > 0:
    transition[start:end, start:end] = companion_matrix(self._k_order)
    if self.hamilton_representation:
        transition[start:end, start:end] = np.transpose(
            companion_matrix(self._k_order)
        )
```

### 2.4 参数更新机制 (update 方法)

**方法：** `update` (`sarimax.py:1530-1719`)

这是最核心的方法，负责将估计的参数映射到状态空间矩阵。

**参数提取顺序：**
```
params = [
    # 1. 趋势参数 (k_trend个)
    δ₀, δ₁, ..., δ_{k_trend-1},
    
    # 2. MLE 回归系数 (k_exog个, 如果 mle_regression=True)
    β₁, β₂, ..., β_{k_exog},
    
    # 3. AR 参数 (k_ar_params个)
    φ₁, φ₂, ..., φ_{k_ar_params},
    
    # 4. MA 参数 (k_ma_params个)
    θ₁, θ₂, ..., θ_{k_ma_params},
    
    # 5. 季节 AR 参数 (k_seasonal_ar_params个)
    Φ₁, Φ₂, ..., Φ_{k_seasonal_ar_params},
    
    # 6. 季节 MA 参数 (k_seasonal_ma_params个)
    Θ₁, Θ₂, ..., Θ_{k_seasonal_ma_params},
    
    # 7. 时变回归方差 (如果 time_varying_regression=True)
    σ_β₁², ...,
    
    # 8. 测量误差方差 (如果 measurement_error=True)
    σ_ε²,
    
    # 9. 状态噪声方差 (如果 state_error=True 且非集中尺度)
    σ_η²
]
```

**滞后多项式构建：**

AR 多项式：`φ(L) = 1 - φ₁L - φ₂L² - ... - φₚLᵖ`
```python
# sarimax.py:1599-1605
if self.k_ar > 0:
    # polynomial_ar 初始化为 [1, 0, 0, ..., 0]
    # 注意符号: -params_ar 表示多项式系数
    self._polynomial_ar[self._polynomial_ar_idx] = -params_ar
```

季节 AR 多项式：`Φ(L^s) = 1 - Φ₁L^s - ... - Φ_PL^{sP}`

**简化多项式计算：**

将常规和季节多项式相乘得到简化形式：

```python
# sarimax.py:1642-1653
if self.k_seasonal_ar > 0:
    # φ(L) * Φ(L^s) = (1 - φ₁L - ...)(1 - Φ₁L^s - ...)
    reduced_polynomial_ar = -np.polymul(
        self._polynomial_ar, self._polynomial_seasonal_ar
    )
else:
    reduced_polynomial_ar = -self._polynomial_ar
```

**状态空间矩阵更新：**

```python
# sarimax.py:1693-1717

# 1. 更新过渡矩阵 (AR 参数)
if self.k_ar > 0 or self.k_seasonal_ar > 0:
    # Harvey: transition[第一列] = reduced_polynomial_ar[1:]
    # Hamilton: transition[第一行] = reduced_polynomial_ar[1:]
    self.ssm[self.transition_ar_params_idx] = reduced_polynomial_ar[1:]

# 2. 更新选择矩阵或设计矩阵 (MA 参数)
if self.k_ma > 0 or self.k_seasonal_ma > 0:
    if not self.hamilton_representation:
        # Harvey 表示: MA 参数在 selection 矩阵
        self.ssm[self.selection_ma_params_idx] = reduced_polynomial_ma[1:]
    else:
        # Hamilton 表示: MA 参数在 design 矩阵
        self.ssm[self.design_ma_params_idx] = reduced_polynomial_ma[1:]

# 3. 更新状态协方差矩阵
if self.k_posdef > 0:
    if not self.concentrate_scale:
        self["state_cov", 0, 0] = params_variance
```

### 2.5 完整示例：AR(1) 模型的状态空间形式

**模型：** `y_t = φ₁ y_{t-1} + ε_t`

**状态向量：** `α_t = [y_t, ε_t]'` （实际为 `[y_t, 0]'` 用于扩展）

**状态空间矩阵：**
```
design = [1, 0]           # y_t = 1*α_{1,t} + 0*α_{2,t}

transition = [[φ₁, 1],    # α_{1,t+1} = φ₁*α_{1,t} + 1*α_{2,t}
              [0,  0]]    # α_{2,t+1} = 0

selection = [[1],          # 选择扰动影响哪些状态
             [0]]

state_cov = [[σ²]]        # 扰动方差
```

**解释：**
- 状态方程：`α_{t+1} = T α_t + R η_t`
- 即：`[y_{t+1}] = [φ₁ 1] [y_t] + [1] ε_t`
- `[  0   ] = [0  0] [0  ] + [0]`

---

## 3. 最大似然框架驱动卡尔曼滤波

### 3.1 MLE 估计流程概览

```
用户调用 model.fit()
        │
        ▼
┌─────────────────────┐
│  MLEModel.fit()     │
│  - 获取 start_params │
│  - 调用 scipy.optimize │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  优化器迭代          │
│  while not converged:│
│    - 计算 loglike(params) │
│    - 更新参数         │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  MLEModel.loglike() │
│  1. handle_params()  │
│  2. update() → 更新状态空间矩阵 │
│  3. ssm.loglike() → 卡尔曼滤波计算似然 │
└─────────────────────┘
```

### 3.2 fit 方法详解

**位置：** `mlemodel.py:540-787`

**关键步骤：**

1. **获取初始参数**
```python
if start_params is None:
    start_params = self.start_params
    transformed = True
    includes_fixed = True
```

2. **设置优化器参数**
```python
# L-BFGS-B 默认使用近似梯度
if optim_score is None and method == "lbfgs":
    kwargs.setdefault("approx_grad", True)
    kwargs.setdefault("epsilon", 1e-5)
```

3. **执行优化**
   - 调用 `scipy.optimize.fmin_l_bfgs_b` 或其他优化器
   - 优化器反复调用 `loglike()` 方法评估似然

### 3.3 loglike 方法详解

**位置：** `mlemodel.py:987-1040`

这是连接优化器和卡尔曼滤波的核心方法。

```python
def loglike(self, params, *args, **kwargs):
    """
    对数似然评估
    
    流程:
    1. 处理参数 (转换为约束形式)
    2. 更新状态空间矩阵
    3. 执行卡尔曼滤波计算似然
    """
    
    # 1. 处理参数
    params = self.handle_params(
        params, transformed=transformed, includes_fixed=includes_fixed
    )
    
    # 2. 更新状态空间矩阵 (关键: 子类实现)
    self.update(
        params, transformed=True, includes_fixed=True, complex_step=complex_step
    )
    
    # 3. 执行卡尔曼滤波并返回对数似然
    loglike = self.ssm.loglike(complex_step=complex_step, **kwargs)
    
    return loglike
```

### 3.4 filter 方法详解

**位置：** `mlemodel.py:848-919`

用于执行完整的卡尔曼滤波并返回滤波结果。

```python
def filter(self, params, ...):
    # 1. 处理参数
    params = self.handle_params(
        params, transformed=transformed, includes_fixed=includes_fixed
    )
    
    # 2. 更新状态空间矩阵
    self.update(
        params, transformed=True, includes_fixed=True, complex_step=complex_step
    )
    
    # 3. 执行卡尔曼滤波
    result = self.ssm.filter(complex_step=complex_step, **kwargs)
    
    # 4. 包装结果
    return self._wrap_results(params, result, ...)
```

### 3.5 卡尔曼滤波算法实现

**核心滤波迭代：** `_conventional.pyx.in:108-200+`

标准卡尔曼滤波的预测-更新循环：

```
预测步 (Prediction):
┌─────────────────────────────────────────────────────────┐
│ a_{t|t-1} = T_t a_{t-1|t-1} + c_t                      │
│ P_{t|t-1} = T_t P_{t-1|t-1} T_t' + R_t Q_t R_t'        │
│                                                          │
│ y_{t|t-1} = Z_t a_{t|t-1} + d_t     (预测值)            │
│ v_t = y_t - y_{t|t-1}              (预测误差)            │
│ F_t = Z_t P_{t|t-1} Z_t' + H_t    (预测误差协方差)      │
└─────────────────────────────────────────────────────────┘

更新步 (Updating):
┌─────────────────────────────────────────────────────────┐
│ K_t = P_{t|t-1} Z_t' F_t^{-1}        (卡尔曼增益)        │
│ a_{t|t} = a_{t|t-1} + K_t v_t        (滤波状态)          │
│ P_{t|t} = P_{t|t-1} - K_t F_t K_t'   (滤波协方差)        │
└─────────────────────────────────────────────────────────┘
```

**对数似然计算：**
```
log L = -0.5 * Σ [k_endog * log(2π) + log|F_t| + v_t' F_t^{-1} v_t]
```

### 3.6 完整数据流

```
优化器参数 θ (无约束)
        │
        ▼ transform_params()
┌─────────────────────┐
│ 约束参数 θ'          │
│ - AR: 平稳性约束     │
│ - MA: 可逆性约束     │
│ - 方差: 非负约束     │
└─────────────────────┘
        │
        ▼ update()
┌─────────────────────┐
│ 状态空间矩阵         │
│ - transition (T)    │
│ - design (Z)        │
│ - selection (R)     │
│ - state_cov (Q)     │
│ - obs_cov (H)       │
└─────────────────────┘
        │
        ▼ KalmanFilter.filter()
┌─────────────────────┐
│ 卡尔曼滤波迭代       │
│ for t = 1..nobs:    │
│   预测步             │
│   更新步 (处理缺失值) │
│   累积对数似然        │
└─────────────────────┘
        │
        ▼
返回 log L(θ|y) 给优化器
```

---

## 4. 参数变换逻辑与复用机制

### 4.1 为什么需要参数变换？

**问题：** 优化器在无约束空间搜索，但模型参数有约束：

| 参数类型 | 约束条件 | 原因 |
|---------|---------|------|
| AR 系数 | 所有根在单位圆内 | 保证过程平稳 |
| MA 系数 | 所有根在单位圆外 | 保证过程可逆 |
| 方差参数 | 非负 | 方差定义要求 |

**解决方案：** 在优化器的无约束空间和模型的约束空间之间建立双射变换。

```
无约束空间 (优化器)        约束空间 (模型)
─────────────────        ─────────────────
    θ ∈ ℝᵏ      ──────►    θ' ∈ Θ (约束集)
                  变换
    θ ∈ ℝᵏ      ◄──────    θ' ∈ Θ (约束集)
                 逆变换
```

### 4.2 平稳性约束变换

**函数：** `constrain_stationary_univariate` (`tools.py:481-515`)

**算法：** Monahan (1984) 方法

**核心思想：** 利用偏自相关系数 (PACF) 的性质
- 平稳 AR 过程的所有偏自相关系数绝对值 < 1
- 可以通过 Levinson-Durbin 递推从 PACF 恢复 AR 系数

**变换步骤：**

1. **将无约束参数映射到 (-1, 1) 区间：**
   ```python
   r = unconstrained / ((1 + unconstrained**2)**0.5)
   # 这是 tanh 函数的变体，保证 r ∈ (-1, 1)
   ```

2. **使用 Levinson-Durbin 递推构建 AR 系数：**
   ```python
   y = np.zeros((n, n), dtype=unconstrained.dtype)
   for k in range(n):
       for i in range(k):
           y[k, i] = y[k - 1, i] + r[k] * y[k - 1, k - i - 1]
       y[k, k] = r[k]
   ```

3. **返回最终 AR 系数：**
   ```python
   return -y[n - 1, :]
   # 负号是为了匹配多项式的符号约定
   ```

**示例：AR(2) 变换**

```
无约束参数: x = [x₁, x₂]  (任意实数)

步骤1: 映射到 (-1, 1)
r₁ = x₁ / √(1 + x₁²)
r₂ = x₂ / √(1 + x₂²)

步骤2: Levinson-Durbin 递推
k=0: y[0,0] = r₁
k=1: y[1,0] = y[0,0] + r₂*y[0,0] = r₁ + r₂*r₁
     y[1,1] = r₂

步骤3: 返回
φ₁ = -y[1,0] = -r₁(1 + r₂)
φ₂ = -y[1,1] = -r₂

验证平稳性: 特征方程 1 - φ₁z - φ₂z² = 0
因为 |r₁| < 1, |r₂| < 1，保证所有根在单位圆外
```

### 4.3 逆变换

**函数：** `unconstrain_stationary_univariate` (`tools.py:518-552`)

从约束的 AR 系数恢复无约束参数。

```python
def unconstrain_stationary_univariate(constrained):
    n = constrained.shape[0]
    y = np.zeros((n, n), dtype=constrained.dtype)
    y[n-1:] = -constrained  # 注意符号
    
    # 反向 Levinson-Durbin 递推
    for k in range(n-1, 0, -1):
        for i in range(k):
            y[k-1, i] = (y[k, i] - y[k, k]*y[k, k-i-1]) / (1 - y[k, k]**2)
    
    # 提取偏自相关系数 (对角线)
    r = y.diagonal()
    
    # 映射回无约束空间
    x = r / ((1 - r**2)**0.5)
    return x
```

### 4.4 SARIMAX 中的参数变换

**方法：** `transform_params` (`sarimax.py:1299-1401`)

按参数类型依次处理：

```python
def transform_params(self, unconstrained):
    constrained = np.zeros(unconstrained.shape, unconstrained.dtype)
    
    start = end = 0
    
    # 1. 趋势参数: 无约束
    if self._k_trend > 0:
        end += self._k_trend
        constrained[start:end] = unconstrained[start:end]
        start += self._k_trend
    
    # 2. MLE 回归系数: 无约束
    if self.mle_regression:
        end += self._k_exog
        constrained[start:end] = unconstrained[start:end]
        start += self._k_exog
    
    # 3. AR 参数: 平稳性约束
    if self.k_ar_params > 0:
        end += self.k_ar_params
        if self.enforce_stationarity:
            constrained[start:end] = constrain_stationary_univariate(
                unconstrained[start:end]
            )
        else:
            constrained[start:end] = unconstrained[start:end]
        start += self.k_ar_params
    
    # 4. MA 参数: 可逆性约束
    # 注意: MA 的变换是 AR 变换的负值
    # 原因: MA 多项式为 1 + θ₁L + ... + θ_qL^q
    #       可逆性要求根在单位圆外，相当于 -θ 是平稳 AR 系数
    if self.k_ma_params > 0:
        end += self.k_ma_params
        if self.enforce_invertibility:
            constrained[start:end] = -constrain_stationary_univariate(
                unconstrained[start:end]
            )
        else:
            constrained[start:end] = unconstrained[start:end]
        start += self.k_ma_params
    
    # 5. 季节 AR 参数: 同样使用平稳性约束
    if self.k_seasonal_ar > 0:
        end += self.k_seasonal_ar_params
        if self.enforce_stationarity:
            constrained[start:end] = constrain_stationary_univariate(
                unconstrained[start:end]
            )
        start += self.k_seasonal_ar_params
    
    # 6. 季节 MA 参数: 同样使用可逆性约束
    if self.k_seasonal_ma_params > 0:
        end += self.k_seasonal_ma_params
        if self.enforce_invertibility:
            constrained[start:end] = -constrain_stationary_univariate(
                unconstrained[start:end]
            )
        start += self.k_seasonal_ma_params
    
    # 7. 方差参数: 平方保证非负
    if self.state_regression and self.time_varying_regression:
        end += self._k_exog
        constrained[start:end] = unconstrained[start:end]**2
        start += self._k_exog
    if self.measurement_error:
        constrained[start] = unconstrained[start]**2
        start += 1
        end += 1
    if self.state_error and not self.concentrate_scale:
        constrained[start] = unconstrained[start]**2
    
    return constrained
```

### 4.5 复用机制

**核心设计理念：** 工具函数与模型分离

```
┌─────────────────────────────────────────────────────────┐
│                    tools.py                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │ constrain_stationary_univariate()               │    │
│  │   - 通用: 处理任意单变量 AR/MA 系数              │    │
│  │   - 被 SARIMAX, VARMAX, Structural 等模型复用   │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │ unconstrain_stationary_univariate()             │    │
│  │   - 逆变换                                        │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │ companion_matrix()                                │    │
│  │   - 创建伴随矩阵                                   │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
    ┌─────────┐      ┌─────────┐      ┌─────────────┐
    │ SARIMAX │      │ VARMAX  │      │ Structural  │
    │  (单变量)│      │ (多变量) │      │  (结构模型)  │
    └─────────┘      └─────────┘      └─────────────┘
```

**AR 和 MA 参数复用同一变换函数：**

| 参数类型 | 变换方式 | 原因 |
|---------|---------|------|
| AR | `constrain_stationary_univariate(x)` | AR 多项式: 1 - φ₁L - ... |
| MA | `-constrain_stationary_univariate(x)` | MA 多项式: 1 + θ₁L + ... = 1 - (-θ₁)L - ... |

**数学解释：**

MA 可逆性条件：MA(q) 过程
```
y_t = ε_t + θ₁ ε_{t-1} + ... + θ_q ε_{t-q}
```
其滞后多项式为：
```
θ(L) = 1 + θ₁ L + θ₂ L² + ... + θ_q L^q
```

可逆性要求：θ(z) = 0 的所有根都在单位圆外。

令 `ψ_i = -θ_i`，则多项式变为：
```
θ(L) = 1 - ψ₁ L - ψ₂ L² - ... - ψ_q L^q
```

这与 AR 多项式形式完全相同！因此：
- MA 的可逆性条件等价于 ψ 是平稳 AR 系数
- 可以复用 `constrain_stationary_univariate` 函数
- 最后取负号得到 θ

---

## 5. 缺失观测值的处理路径

### 5.1 缺失值检测

**数据绑定阶段：** `Representation.bind()` (`_representation.pyx.in`)

当数据被绑定到状态空间模型时，自动检测缺失值：

```python
# 伪代码表示
def bind(self, endog):
    # 检测 NaN 值
    self.missing = np.isnan(endog)  # shape: (k_endog, nobs)
    
    # 计算每个时间点的缺失数量
    self.nmissing = np.sum(self.missing, axis=0)  # shape: (nobs,)
```

**存储的变量：**
- `missing`: 布尔数组，`missing[i, t] = True` 表示第 i 个变量在 t 时刻缺失
- `nmissing`: 整数数组，`nmissing[t]` 表示 t 时刻缺失的变量数

### 5.2 三种缺失情况

| 情况 | 条件 | 处理方式 |
|-----|------|---------|
| **完全观测** | `nmissing[t] == 0` | 标准卡尔曼滤波 |
| **完全缺失** | `nmissing[t] == k_endog` | 跳过更新，状态保持预测值 |
| **部分缺失** | `0 < nmissing[t] < k_endog` | 单变量滤波，逐个处理非缺失变量 |

### 5.3 完全缺失的处理

**实现位置：** `_conventional.pyx.in:54-99`

当所有观测变量都缺失时，无法进行更新，只能进行预测。

#### 5.3.1 预测步处理

```cython
cdef int {{prefix}}forecast_missing_conventional({{prefix}}KalmanFilter kfilter, {{prefix}}Statespace model):
    cdef int i, j
    
    # 预测值和预测误差设为 0
    # 因为没有观测值，这些实际上不使用
    for i in range(kfilter.k_endog):
        kfilter._forecast[i] = 0
        kfilter._forecast_error[i] = 0
    
    # 预测误差协方差设为 0 矩阵
    for i in range(kfilter.k_endog):
        for j in range(kfilter.k_endog):
            kfilter._forecast_error_cov[j + i*kfilter.k_endog] = 0
```

#### 5.3.2 更新步处理

```cython
cdef int {{prefix}}updating_missing_conventional({{prefix}}KalmanFilter kfilter, {{prefix}}Statespace model):
    cdef int inc = 1
    
    # 关键: 直接将预测状态复制到滤波状态
    # a_{t|t} = a_{t|t-1}
    blas.{{prefix}}copy(&kfilter.k_states, 
          kfilter._input_state, &inc, 
          kfilter._filtered_state, &inc)
    
    # P_{t|t} = P_{t|t-1}
    blas.{{prefix}}copy(&kfilter.k_states2, 
          kfilter._input_state_cov, &inc, 
          kfilter._filtered_state_cov, &inc)
```

**核心思想：** 没有新信息，状态不更新

```
标准更新:    a_{t|t} = a_{t|t-1} + K_t v_t
缺失值更新:  a_{t|t} = a_{t|t-1}  (v_t 不存在)
```

#### 5.3.3 对数似然贡献

```cython
cdef {{cython_type}} {{prefix}}loglikelihood_missing_conventional(
    {{prefix}}KalmanFilter kfilter, 
    {{prefix}}Statespace model, 
    {{cython_type}} determinant):
    # 缺失值对似然没有贡献
    return 0.0
```

**数学解释：**
- 完全缺失的观测不提供任何信息
- 其对对数似然的贡献为 0
- 不影响参数估计

### 5.4 部分缺失的处理

**方法：** 单变量滤波 (Univariate Filtering)

**实现位置：** `_univariate.pyx.in`

**核心思想：** 将多变量观测分解为多个独立的单变量更新

```
多变量观测: y_t = [y_{1,t}, y_{2,t}, ..., y_{p,t}]'

如果 y_{2,t} 缺失:
    ┌─────────────────────────────────────────────────────┐
    │  1. 使用 y_{1,t} 执行单变量更新                      │
    │     a^{(1)}_{t|t} = a_{t|t-1} + K_{1,t} v_{1,t}   │
    │                                                     │
    │  2. 跳过 y_{2,t} (缺失)                             │
    │     a^{(2)}_{t|t} = a^{(1)}_{t|t}                  │
    │                                                     │
    │  3. 使用 y_{3,t} 执行单变量更新                      │
    │     a^{(3)}_{t|t} = a^{(2)}_{t|t} + K_{3,t} v_{3,t}│
    │                                                     │
    │  最终: a_{t|t} = a^{(p)}_{t|t}                      │
    └─────────────────────────────────────────────────────┘
```

**单变量滤波的优势：**
1. 可以逐个处理观测，自然跳过缺失值
2. 避免处理奇异的预测误差协方差矩阵
3. 数值稳定性更好

### 5.5 滤波路径选择逻辑

**伪代码表示：**

```python
def filter_iteration(t):
    if nmissing[t] == k_endog:
        # 路径1: 完全缺失
        forecast_missing_conventional()
        updating_missing_conventional()
        loglike_contribution = 0.0
        
    elif nmissing[t] == 0:
        # 路径2: 完全观测
        if filter_method == FILTER_UNIVARIATE:
            # 使用单变量滤波 (更稳定)
            forecast_univariate()
            for i in range(k_endog):
                updating_univariate(i)  # 逐个更新
        else:
            # 使用常规滤波
            forecast_conventional()
            updating_conventional()
            
    else:  # 0 < nmissing[t] < k_endog
        # 路径3: 部分缺失 → 强制使用单变量滤波
        # 将非缺失变量排序在前，缺失变量在后
        reorder_missing_matrix()
        reorder_missing_vector()
        
        # 只对非缺失变量执行更新
        forecast_univariate()
        for i in range(k_endog - nmissing[t]):
            updating_univariate(i)  # 只更新非缺失的
        # 缺失的变量被跳过
```

### 5.6 结果处理

**位置：** `kalman_filter.py:1683-1702`

滤波后，缺失位置的预测值会被恢复为 NaN：

```python
# 填充缺失位置的预测值为 NaN
for t in range(nobs):
    if self.nmissing[t] > 0:
        mask = ~self.missing[:, t].astype(bool)
        
        # 完全缺失: 所有预测值设为 NaN
        if self.nmissing[t] == self.k_endog:
            self.forecasts[:, t] = np.nan
            self.forecasts_error[:, t] = np.nan
        
        # 部分缺失: 只保留非缺失位置的结果
        else:
            # 非缺失位置已经有正确值
            # 缺失位置设为 NaN
            self.forecasts[~mask, t] = np.nan
            self.forecasts_error[~mask, t] = np.nan
```

### 5.7 完整处理流程图

```
                    ┌─────────────────┐
                    │  开始滤波迭代    │
                    │  for t = 1..T   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ 检查 nmissing[t] │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │  == k_endog │   │   == 0      │   │  0 < x < p  │
    │ (完全缺失)   │   │ (完全观测)   │   │ (部分缺失)   │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                  │                  │
           ▼                  ▼                  ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │forecast_    │   │forecast_    │   │重新排序变量  │
    │missing_     │   │conventional │   │缺失在后     │
    │conventional │   │或 univariate│   │             │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                  │                  │
           ▼                  ▼                  ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │updating_    │   │updating_    │   │单变量滤波    │
    │missing_     │   │conventional │   │只更新非缺失  │
    │conventional │   │或 univariate│   │             │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                  │                  │
           ▼                  ▼                  ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │loglike = 0  │   │计算似然贡献  │   │计算似然贡献  │
    │(无信息)     │   │             │   │(只非缺失部分)│
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  累积对数似然    │
                     │  t = t + 1      │
                     └─────────────────┘
```

---

## 6. 完整代码引用索引

### 6.1 文件位置总览

```
statsmodels/tsa/statespace/
├── sarimax.py              # SARIMAX/ARIMA 模型实现
├── mlemodel.py             # 最大似然估计基类
├── kalman_filter.py        # 卡尔曼滤波 Python 包装
├── representation.py       # 状态空间表示基类
├── tools.py                # 工具函数 (参数变换、伴随矩阵)
├── _filters/
│   ├── _conventional.pyx.in   # 常规卡尔曼滤波 Cython 实现
│   └── _univariate.pyx.in     # 单变量滤波 Cython 实现
└── _kalman_filter.pyx.in     # KalmanFilter 类 Cython 实现
```

### 6.2 详细代码索引

#### 参数到状态空间矩阵的映射

| 功能 | 文件 | 行号 | 关键函数/属性 |
|-----|------|------|--------------|
| SARIMAX 初始化 | `sarimax.py` | 519-546 | `__init__` |
| 设计矩阵初始化 | `sarimax.py` | 689-716 | `initial_design` |
| 过渡矩阵初始化 | `sarimax.py` | 729-791 | `initial_transition` |
| 选择矩阵初始化 | `sarimax.py` | 794-816 | `initial_selection` |
| 参数更新核心 | `sarimax.py` | 1530-1719 | `update` |
| 伴随矩阵创建 | `tools.py` | 127-245 | `companion_matrix` |

#### 最大似然框架

| 功能 | 文件 | 行号 | 关键函数/属性 |
|-----|------|------|--------------|
| MLE 模型拟合 | `mlemodel.py` | 540-787 | `fit` |
| 卡尔曼滤波执行 | `mlemodel.py` | 848-919 | `filter` |
| 对数似然计算 | `mlemodel.py` | 987-1040 | `loglike` |
| 常规滤波预测 | `_conventional.pyx.in` | 108-169 | `forecast_conventional` |
| 常规滤波更新 | `_conventional.pyx.in` | 200+ | `updating_conventional` |

#### 参数变换

| 功能 | 文件 | 行号 | 关键函数/属性 |
|-----|------|------|--------------|
| SARIMAX 参数变换 | `sarimax.py` | 1299-1401 | `transform_params` |
| SARIMAX 逆变换 | `sarimax.py` | 1403-1505 | `untransform_params` |
| 平稳性约束变换 | `tools.py` | 481-515 | `constrain_stationary_univariate` |
| 平稳性约束逆变换 | `tools.py` | 518-552 | `unconstrain_stationary_univariate` |

#### 缺失值处理

| 功能 | 文件 | 行号 | 关键函数/属性 |
|-----|------|------|--------------|
| 完全缺失预测 | `_conventional.pyx.in` | 54-79 | `forecast_missing_conventional` |
| 完全缺失更新 | `_conventional.pyx.in` | 80-87 | `updating_missing_conventional` |
| 完全缺失似然 | `_conventional.pyx.in` | 95-96 | `loglikelihood_missing_conventional` |
| 缺失值存储 | `kalman_filter.py` | 1390-1395 | `missing`, `nmissing` |
| 结果恢复 NaN | `kalman_filter.py` | 1683-1702 | 滤波后处理 |

### 6.3 关键数据结构

**状态空间矩阵索引约定：**

```python
# 矩阵访问方式
self.ssm["design"]           # 整个设计矩阵
self.ssm["transition", 0, :] # 过渡矩阵第一行
self.ssm["transition", :, 0] # 过渡矩阵第一列

# 预计算的索引 (SARIMAX 类)
self.transition_ar_params_idx  # AR 参数在过渡矩阵中的位置
self.selection_ma_params_idx   # MA 参数在选择矩阵中的位置
self.design_ma_params_idx      # MA 参数在设计矩阵中的位置 (Hamilton 表示)
```

**参数向量结构：**

```
SARIMAX(p,d,q)(P,D,Q,s) with exog 和 trend:

params = [
    # 趋势参数
    δ₀ (constant), δ₁ (linear), ...,
    
    # MLE 回归系数 (如果 mle_regression=True)
    β₁, β₂, ..., β_{k_exog},
    
    # AR 参数
    φ₁, φ₂, ..., φ_p,
    
    # MA 参数
    θ₁, θ₂, ..., θ_q,
    
    # 季节 AR 参数
    Φ₁, Φ₂, ..., Φ_P,
    
    # 季节 MA 参数
    Θ₁, Θ₂, ..., Θ_Q,
    
    # 方差参数
    σ² (状态噪声),
    σ_ε² (测量误差, 如果 measurement_error=True)
]
```

---

## 参考资料

1. Durbin, J., & Koopman, S. J. (2012). Time Series Analysis by State Space Methods (2nd ed.). Oxford University Press.

2. Monahan, J. F. (1984). A Note on Enforcing Stationarity in Autoregressive-moving Average Models. Biometrika, 71(2), 403-404.

3. Harvey, A. C. (1989). Forecasting, Structural Time Series Models and the Kalman Filter. Cambridge University Press.

4. Hamilton, J. D. (1994). Time Series Analysis. Princeton University Press.
