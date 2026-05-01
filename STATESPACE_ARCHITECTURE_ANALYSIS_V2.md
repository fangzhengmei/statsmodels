# Statsmodels 状态空间模型子系统架构深度分析（修正版 V2）

## 修正说明

本报告是在 V1 版本基础上的重要修正和深化，主要更正了以下关键理解：

1. **多精度实现**：不是"单精度统一升为双精度"，而是**BLAS 矩阵运算保持原精度**，**少数标量函数使用更高精度**
2. **参数更新机制**：基类 `update()` 只调用 `handle_params()`，子类需要显式调用或通过 `super()` 继承
3. **仿真平滑器**：CFA 算法**仅支持状态采样**，KFS 算法支持状态和扰动采样
4. **派生协方差**：补充 `selected_state_cov = R Q R'` 在预测递推和平稳初始化中的关键作用
5. **运行时链路**：补充从数据类型检测到 Cython 内核实例化的完整延迟加载链路

---

## 1. 引言

Statsmodels 的状态空间模型子系统是一个设计精妙、层次清晰的软件工程典范。本报告深入分析该子系统的四个核心架构层面，并针对 V1 报告中的不准确表述进行关键修正。

---

## 2. 多精度实现机制：精确区分同精度调用与合并计算路径

### 2.1 关键修正：不是"统一升为双精度"

**V1 报告的不准确表述**：
> "单精度统一升为双精度"

**修正后的精确理解**：

从 `_conventional.pyx.in` 模板代码分析：

```python
# 模板中的类型定义
TYPES = {
    "s": ("np.float32_t", "np.float32", "np.NPY_FLOAT32"),
    "d": ("np.float64_t", "float", "np.NPY_FLOAT64"),
    "c": ("np.complex64_t", "np.complex64", "np.NPY_COMPLEX64"),
    "z": ("np.complex128_t", "complex", "np.NPY_COMPLEX128"),
}

# 关键：只有少数函数使用 combined_prefix
combined_prefix = prefix
combined_cython_type = cython_type
if prefix == 'c':
    combined_prefix = 'z'      # complex64 → complex128
    combined_cython_type = 'np.complex128_t'
if prefix == 's':
    combined_prefix = 'd'      # float32 → float64
    combined_cython_type = 'np.float64_t'
```

### 2.2 两类调用路径的精确区分

#### 路径一：同精度 BLAS 调用（绝大多数）

```cython
# 所有 BLAS 矩阵运算都使用原始前缀 prefix
# 这些是计算量最大的部分

# 例如 forecast_conventional 中：
blas.{{prefix}}copy(...)      # 始终是 scopy, dcopy, ccopy, zcopy
blas.{{prefix}}gemv(...)      # 始终是 sgemv, dgemv, cgemv, zgemv
blas.{{prefix}}gemm(...)      # 始终是 sgemm, dgemm, cgemm, zgemm
blas.{{prefix}}axpy(...)      # 始终是 saxpy, daxpy, caxpy, zaxpy
blas.{{prefix}}scal(...)      # 始终是 sscal, dscal, cscal, zscal
```

#### 路径二：合并精度的少数函数（极小部分）

```cython
# 只有以下几处使用 combined_prefix

# 1. 对数函数（用于对数似然计算）
loglikelihood = -0.5*(model._k_endog*{{combined_prefix}}log(2*M_PI) + determinant)

# 2. 绝对值函数（用于对角线检查）
if not ({{combined_prefix}}abs(self.obs_cov[i, j, obs_cov_t]) < 1e-9):
    diagonal_obs_cov = 0

# 3. 条件分支
{{if combined_prefix == 'd'}}
    # s 和 d 都走此分支（因为 s 的 combined_prefix == 'd'）
    loglikelihood = loglikelihood - 0.5*blas.{{prefix}}dot(...)
{{else}}
    # c 和 z 走此分支
    blas.{{prefix}}gemv(...)
    loglikelihood = loglikelihood - 0.5 * kfilter._tmp0[0]
{{endif}}
```

### 2.3 四类精度的实际行为

| 数据类型 | BLAS 矩阵运算 | 对数/绝对值函数 | 条件分支 |
|----------|--------------|----------------|----------|
| **float32 (s)** | `scopy`, `sgemm`, `sgemv`, ... | `dlog`, `dabs` | 走 `combined_prefix == 'd'` 分支 |
| **float64 (d)** | `dcopy`, `dgemm`, `dgemv`, ... | `dlog`, `dabs` | 走 `combined_prefix == 'd'` 分支 |
| **complex64 (c)** | `ccopy`, `cgemm`, `cgemv`, ... | `zlog`, `zabs` | 走 else 分支 |
| **complex128 (z)** | `zcopy`, `zgemm`, `zgemv`, ... | `zlog`, `zabs` | 走 else 分支 |

### 2.4 设计意图

**为什么这样设计？**

1. **性能考虑**：BLAS 矩阵运算占计算量的 95%+，保持原精度可最大化缓存效率
2. **数值稳定性**：对数似然计算涉及小数字的乘积，使用更高精度可避免数值下溢
3. **代码简洁**：通过模板变量 `combined_prefix` 实现条件分支，无需重复代码

### 2.5 运行时精度选择机制

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
    # 自动检测最优 BLAS 类型
    return find_best_blas_type(arrays)[0]
```

**`find_best_blas_type` 规则**：
- 所有 float32 → 's'
- 有 float64 → 'd'（即使有 float32）
- 所有 complex64 → 'c'
- 有 complex128 → 'z'（即使有 complex64）
- 实数 + 复数 → 复数类型

---

## 3. 参数更新机制：基类默认行为与子类覆写关系

### 3.1 基类 MLEModel.update() 的实际行为

**V1 报告的模糊表述**：
> "子类必须覆写 update 方法"

**精确理解**：

```python
# mlemodel.py:1965-1991
def update(
    self, params, transformed=True, includes_fixed=False, complex_step=False
):
    """
    Update the parameters of the model
    
    Notes
    -----
    Since Model is a base class, this method should be overridden by
    subclasses to perform actual updating steps.
    """
    # 基类只做一件事：调用 handle_params
    return self.handle_params(
        params=params, transformed=transformed, includes_fixed=includes_fixed
    )
```

### 3.2 handle_params() 的职责

```python
# mlemodel.py:1920-1963
def handle_params(self, params, transformed=True, includes_fixed=False,
                  return_jacobian=False):
    """
    参数预处理函数，不涉及矩阵更新
    
    职责：
    1. 参数变换（如果 transformed=False）
    2. 注入固定参数
    3. 可选：返回变换的雅可比行列式
    """
    # 1. 参数变换（从无约束空间到约束空间）
    if not transformed:
        params = self.transform_params(params)
    
    # 2. 注入固定参数
    if not includes_fixed and self._has_fixed_params:
        params[self._fixed_params_index] = list(self._fixed_params.values())
    
    return (params, transform_score) if return_jacobian else params
```

### 3.3 子类 update() 的典型实现

#### 模式一：显式调用 handle_params（推荐）

```python
# sarimax.py:1530-...
def update(self, params, transformed=True, includes_fixed=False,
           complex_step=False):
    """
    SARIMAX 的 update 方法
    
    流程：
    1. 显式调用 handle_params 处理参数
    2. 从处理后的参数中提取各部分
    3. 更新状态空间矩阵
    """
    # 步骤1：参数预处理
    params = self.handle_params(params, transformed=transformed,
                                includes_fixed=includes_fixed)
    
    # 步骤2：提取参数
    if self.k_ar_params > 0:
        ar_params = params[:self.k_ar_params]
    if self.k_ma_params > 0:
        ma_params = params[self.k_ar_params:self.k_ar_params + self.k_ma_params]
    # ... 其他参数
    
    # 步骤3：更新矩阵
    if self.k_ar > 0:
        self.ssm['transition', 0, :self.k_ar] = ar_params
    
    if self.k_ma > 0:
        self.ssm['selection', :self.k_ma, 0] = ma_params
    
    # 状态协方差（创新方差 σ²）
    self.ssm['state_cov', 0, 0] = params[self.k_params - 1] ** 2
```

#### 模式二：通过 super() 调用基类 update

```python
# exponential_smoothing.py:528-...
def update(self, params, transformed=True, includes_fixed=False,
           complex_step=False):
    """
    指数平滑模型的 update 方法
    
    流程：
    1. 通过 super().update() 间接调用 handle_params
    2. 更新矩阵
    """
    # 步骤1：通过 super() 调用基类 update（内部调用 handle_params）
    params = self.handle_params(params, transformed=transformed,
                                includes_fixed=includes_fixed)
    
    # 步骤2：更新矩阵
    self.ssm["selection", 0, 0] = 1 - params[0]
    self.ssm["selection", 1, 0] = params[0]
    # ...
```

#### 模式三：测试用例中的极简实现

```python
# tests/test_models.py:46-50
def update(self, params, **kwargs):
    """
    测试用例中的 update 方法
    """
    # 调用父类（实际是 handle_params）
    params = super().update(params, **kwargs)
    
    # 简单的矩阵更新
    self["obs_intercept"] = params[:3]
    self["state_intercept"] = params[3:]
```

### 3.4 完整的参数更新调用链

```
用户调用 fit()
    │
    ▼
scipy.optimize.minimize(负对数似然)
    │
    ▼
每次迭代调用 loglike(params)  # mlemodel.py:987-1040
    │
    ├──► handle_params(params)      # 参数预处理
    │       ├──► transform_params()  # 可选：无约束 → 约束
    │       └──► 注入固定参数
    │
    ├──► update(params)              # 关键：子类覆写
    │       ├──► 调用 handle_params()  # 再次处理（或通过 super()）
    │       └──► 更新状态空间矩阵
    │               ├──► design/Z
    │               ├──► transition/T
    │               ├──► selection/R
    │               ├──► state_cov/Q
    │               └──► obs_cov/H
    │
    └──► ssm.loglike()                # 卡尔曼滤波计算似然
```

### 3.5 设计模式分析

这是典型的**模板方法模式**：

| 角色 | 方法 | 职责 |
|------|------|------|
| **抽象基类** | `MLEModel.update()` | 定义模板骨架，调用 `handle_params()` |
| **具体子类** | `SARIMAX.update()` | 实现具体的参数到矩阵映射 |
| **辅助类** | `handle_params()` | 参数预处理（变换、固定参数注入） |

**为什么基类 update() 只调用 handle_params()？**

1. **单一职责**：`handle_params()` 专门负责参数预处理
2. **灵活性**：子类可以选择何时何地调用 `handle_params()`
3. **复用性**：`handle_params()` 可以被其他方法复用（如 `loglike()` 内部）

---

## 4. 两类仿真平滑器的能力边界

### 4.1 关键修正：CFA 仅支持状态采样

**V1 报告的模糊表述**：
> "两类仿真路径"

**精确理解**：

Statsmodels 实现了**两种独立的仿真平滑算法**，能力边界有本质区别：

| 特性 | **KFS 仿真平滑器** | **CFA 仿真平滑器** |
|------|---------------------|---------------------|
| **算法基础** | Kalman Filter + Smoother | Cholesky Factor Algorithm |
| **状态采样** | ✅ 支持 | ✅ 支持 |
| **扰动采样** | ✅ 支持 | ❌ 不支持 |
| **退化分布** | ✅ 支持（半正定协方差） | ❌ 不支持（必须正定） |
| **漫延初始化** | ✅ 支持 | ❌ 不支持 |
| **缺失数据** | ✅ 支持 | ❌ 只能用于完整数据 |
| **高阶 AR 模型** | ✅ 支持（通过增广状态） | ❌ 不支持（退化创新） |
| **稀疏矩阵优化** | ❌ 无 | ✅ 有（带状 Cholesky） |

### 4.2 KFS 仿真平滑器（SimulationSmoother）

#### 输出能力

```python
# simulation_smoother.py 中的输出选项
SIMULATION_STATE = 0x01              # 状态采样：α ~ p(α|y)
SIMULATION_DISTURBANCE = 0x04        # 扰动采样：ε, η ~ p(ε,η|y)
SIMULATION_ALL = SIMULATION_STATE | SIMULATION_DISTURBANCE
```

#### 两类采样的区别

**状态采样**：
```python
# 从条件后验分布采样状态轨迹
# α ~ N(α̂_smooth, Var(α|Y))
simulated_state = sim.simulated_state
```

**扰动采样**（KFS 特有）：
```python
# 从条件后验分布采样扰动
# ε ~ N(ε̂_smooth, Var(ε|Y))
# η ~ N(η̂_smooth, Var(η|Y))
simulated_obs_disturbance = sim.simulated_observation_disturbance
simulated_state_disturbance = sim.simulated_state_disturbance
```

#### 算法原理（Durbin-Koopman 仿真平滑器）

```
1. 运行卡尔曼滤波和平滑，得到：
   - 滤波状态 a_{t|t}, P_{t|t}
   - 平滑状态 α̂_t, Var(α|Y)

2. 生成新的时间序列实现 y^+：
   - 从先验分布采样状态 α^+ ~ p(α)
   - 生成观测 y^+ = Z α^+ + ε^+, ε^+ ~ N(0,H)

3. 运行平滑器得到平滑状态 α̂^+ ~ p(α|y^+)

4. 条件后验采样：
   α_sim = α̂ + (α^+ - α̂^+)
   
   这满足：α_sim ~ p(α|y)
```

### 4.3 CFA 仿真平滑器（CFASimulationSmoother）

#### 核心限制

```python
# cfa_simulation_smoother.py:36-58
"""
**Important caveat**:

However, this simulation smoother cannot be used with all state space
models, including several of the most popular. In particular, the CFA
algorithm cannot support degenerate distributions (i.e. positive
semi-definite covariance matrices) for the initial state (which is the
prior for the first state) or the observation or state innovations.

One practical problem with this algorithm is that an autoregressive term
with order higher than one is typically put into state space form by
augmenting the states using identities. As identities, these augmenting
terms will not be subject to random innovations, and so the state
innovation will be degenerate.
"""
```

#### 退化分布的例子

**AR(2) 模型的状态空间表示**：
```
状态方程：
α_{t+1} = T α_t + R η_t

其中：
T = [[φ1, φ2],
     [1,  0]]   # 第二行是恒等式

R = [[1],
     [0]]        # 只有第一个状态有创新

Q = [[σ²]]       # 正定，但 selected_state_cov 是退化的

selected_state_cov = R Q R' = [[σ², 0],
                                [0,  0]]  # 退化！
```

**这就是为什么 AR(p>1) 不能用 CFA**。

#### 可用场景

```python
# cfa_simulation_smoother.py:56-58
"""
This simulation smoother has so-far found most of its use with dynamic
factor and stochastic volatility models, which satisfy the restrictions
described above.
"""
```

**适用模型**：
- 动态因子模型（Dynamic Factor Models）
- 随机波动率模型（Stochastic Volatility）
- 状态创新维度 = 状态维度的模型

#### 算法原理（Cholesky Factor Algorithm）

```
1. 构造联合后验分布：
   [α_1', α_2', ..., α_T']' | Y ~ N(μ, Σ)
   
   其中 Σ 是带状稀疏矩阵（状态空间 Markov 性质）

2. 计算稀疏 Cholesky 分解：
   L L' = Σ^{-1}
   
   L 是下三角带状矩阵

3. 采样：
   u ~ N(0, I)
   α_sim = μ + (L')^{-1} u
   
   这满足：α_sim ~ N(μ, Σ)
```

#### 优势：计算效率

```python
# cfa_simulation_smoother.py:25-34
"""
In particular, this simulation smoother computes the joint posterior mean
and covariance matrix for the unobserved state vector all at once, rather
than using the recursive computations of the Kalman filter and smoother. It
then uses these posterior moments to sample directly from this joint
posterior. For some models, it can be more computationally efficient than
the simulation smoother based on the Kalman filter and smoother.
"""
```

**CFA vs KFS 性能对比**：

| 场景 | CFA | KFS |
|------|-----|-----|
| 状态维度小 | 慢 | 快 |
| 状态维度大 | 快（稀疏优化） | 慢 |
| 单次采样 | 需要重新 Cholesky | 可以复用滤波结果 |
| 多次采样 | 可复用 Cholesky 因子 | 每次都要递归 |

### 4.4 输出对比

**KFS 仿真平滑器输出**：
```python
sim = model.simulation_smoother()

# 状态采样（两者都有）
sim.simulated_state          # (k_states, nobs)

# 扰动采样（仅 KFS）
sim.simulated_observation_disturbance  # (k_endog, nobs)
sim.simulated_state_disturbance        # (k_posdef, nobs)
```

**CFA 仿真平滑器输出**：
```python
cfa_sim = CFASimulationSmoother(model)
cfa_sim.simulate()

# 只有状态采样
cfa_sim.simulated_state       # (k_states, nobs)

# 额外的后验矩（CFA 特有）
cfa_sim.posterior_mean             # 联合后验均值
cfa_sim.posterior_cov_inv_chol     # 稀疏 Cholesky 因子
cfa_sim.posterior_cov               # 完整后验协方差（可能很大！）
```

---

## 5. selected_state_cov：状态噪声选择后的派生协方差

### 5.1 为什么需要派生协方差？

**状态空间模型的标准形式**：

```
观测方程: y_t = Z_t α_t + d_t + ε_t,   ε_t ~ N(0, H_t)
状态方程: α_{t+1} = T_t α_t + c_t + R_t η_t,  η_t ~ N(0, Q_t)
```

**关键问题**：状态方程中的噪声项是 `R_t η_t`，不是 `η_t`。

**状态噪声的实际分布**：
```
R_t η_t ~ N(0, R_t Q_t R_t')
```

这就是 `selected_state_cov` 的定义！

### 5.2 数学定义

```python
# _representation.pyx.in:1018-1056
cdef int {{prefix}}select_cov(int k, int k_posdef,
                              {{cython_type}} * tmp,
                              {{cython_type}} * selection,  # R_t
                              {{cython_type}} * cov,        # Q_t
                              {{cython_type}} * selected_cov):  # 输出 Q_t^*
    """
    计算选择后的状态协方差：
    Q_t^* = R_t Q_t R_t'
    
    这是状态方程中实际噪声项 R_t η_t 的协方差矩阵
    """
    # 第一步：tmp = R * Q
    # (m × r) = (m × r) × (r × r)
    blas.{{prefix}}gemm("N", "N", &k, &k_posdef, &k_posdef,
          &alpha, selection, &k,
                  cov, &k_posdef,
          &beta, tmp, &k)
    
    # 第二步：selected_cov = tmp * R' = R Q R'
    # (m × m) = (m × r) × (m × r)'
    blas.{{prefix}}gemm("N", "T", &k, &k, &k_posdef,
          &alpha, tmp, &k,
                  selection, &k,
          &beta, selected_cov, &k)
```

### 5.3 维度关系

| 矩阵 | 符号 | 维度 | 说明 |
|------|------|------|------|
| 状态维度 | `k_states` (m) | - | 状态向量 α_t 的维度 |
| 创新维度 | `k_posdef` (r) | - | 状态噪声 η_t 的维度 |
| 状态协方差 | `state_cov` (Q) | r × r | 原始创新协方差（正定） |
| 选择矩阵 | `selection` (R) | m × r | 将创新映射到状态空间 |
| 派生协方差 | `selected_state_cov` (Q*) | m × m | R Q R'（可能退化） |

### 5.4 在预测递推中的核心作用

**卡尔曼滤波的预测步骤**：

```
预测状态均值：
a_{t+1} = T_t a_{t|t} + c_t

预测状态协方差：
P_{t+1} = T_t P_{t|t} T_t' + Q_t^*

其中 Q_t^* = R_t Q_t R_t' = selected_state_cov
```

**代码实现**（`_conventional.pyx.in:338-382`）：

```cython
cdef int {{prefix}}prediction_conventional({{prefix}}KalmanFilter kfilter, {{prefix}}Statespace model):
    """
    卡尔曼滤波的预测步骤
    
    关键：使用 selected_state_cov 而非 state_cov
    """
    # ...
    
    # #### Predicted state covariance matrix for time t+1
    # $P_{t+1} = T_t P_{t|t} T_t' + Q_t^*$
    #
    # 注意：Q_t^* = R_t Q_t R_t' 就是 selected_state_cov
    
    if not kfilter.converged:
        # 第一步：用 selected_state_cov 初始化 predicted_state_cov
        # predicted_state_cov = Q_t^*
        blas.{{prefix}}copy(&model._k_states2, 
                            model._selected_state_cov, &inc, 
                            kfilter._predicted_state_cov, &inc)
        
        # 第二步：加上 T P_{t|t} T'
        if not model.identity_transition:
            # tmp0 = T P_{t|t}
            blas.{{prefix}}gemm("N", "N", &model._k_states, &model._k_states, &model._k_states,
                &alpha, model._transition, &model._k_states,
                        kfilter._filtered_state_cov, &kfilter.k_states,
                &beta, kfilter._tmp0, &kfilter.k_states)
            
            # predicted_state_cov += tmp0 T' = T P_{t|t} T'
            blas.{{prefix}}gemm("N", "T", &model._k_states, &model._k_states, &model._k_states,
                &alpha, kfilter._tmp0, &kfilter.k_states,
                        model._transition, &model._k_states,
                &alpha, kfilter._predicted_state_cov, &kfilter.k_states)
        else:
            # T = I，直接加上 P_{t|t}
            blas.{{prefix}}axpy(&model._k_states2, &alpha, 
                                kfilter._filtered_state_cov, &inc, 
                                kfilter._predicted_state_cov, &inc)
```

### 5.5 在平稳初始化中的作用

**平稳协方差的 Lyapunov 方程**：

对于平稳的状态空间模型，初始状态协方差满足：

```
P = T P T' + Q^*

其中 Q^* = R Q R' = selected_state_cov
```

这是离散 Lyapunov 方程。

**代码实现**（`_representation.pyx.in:445-516`）：

```cython
def initialize_stationary(self, complex_step=False):
    """
    平稳初始化：求解 Lyapunov 方程
    
    P = T P T' + Q^*
    """
    cdef:
        int k_states2 = self.k_states**2
        # ...
    
    # 第一步：计算 selected_state_cov = R Q R'
    {{prefix}}select_cov(self.k_states, self.k_posdef,
                               &self.tmp[0,0],
                               &self.selection[0,0,0],
                               &self.state_cov[0,0,0],
                               &self.selected_state_cov[0,0,0])
    
    # 第二步：如果有非零状态截距，求解 (I - T) a = c
    # 得到平稳状态均值
    if state_intercept非零:
        # 求解 (I - T) x = c
        # ...
    
    # 第三步：求解 Lyapunov 方程得到平稳协方差
    # P = T P T' + Q^*
    
    # 复制 selected_state_cov 作为初始值
    blas.{{prefix}}copy(&k_states2, &self.selected_state_cov[0,0,0], &inc,
                               &self.initial_state_cov[0,0], &inc)
    
    # 求解离散 Lyapunov 方程
    tools._{{prefix}}solve_discrete_lyapunov(
        &self.tmp[0,0],        # T 的副本
        &self.initial_state_cov[0,0],  # 输入：Q^*，输出：P
        self.k_states, 
        complex_step
    )
```

### 5.6 为什么不直接用 Q 代替 Q*？

**例子：AR(2) 模型**

```
状态空间形式（Harvey 表示）：

α_t = [y_t, y_{t-1}]'

状态方程：
α_{t+1} = [[φ1, φ2],  T
            [1,  0]] α_t + [[1],  R
                            [0]] η_t

η_t ~ N(0, σ²)  Q = [σ²]
```

**直接用 Q 的错误**：
```
如果直接用 Q = [σ²]，维度是 1×1
但 T P T' 的维度是 2×2
无法相加！
```

**正确做法**：
```
Q* = R Q R' = [[1],  × [σ²] × [1, 0]
               [0]]
           
   = [[σ², 0],
      [0,  0]]

维度：2×2，与 T P T' 匹配
```

### 5.7 设计优势

| 方面 | 说明 |
|------|------|
| **数学正确性** | 精确对应状态方程中实际噪声项的分布 |
| **维度统一** | Q* 始终是 m×m，与状态协方差 P 同维度 |
| **退化支持** | Q* 可以是退化的（允许恒等式增广状态） |
| **计算效率** | 预计算后可直接使用，避免每次预测都重新计算 |
| **时变支持** | 支持时变的 R_t 和 Q_t |

### 5.8 完整的调用时机

```
何时计算/更新 selected_state_cov？

1. 初始化阶段：initialize_stationary()
   - 计算 Q* = R Q R'
   - 求解 Lyapunov 方程

2. 每次滤波迭代：seek(t)
   - 调用 select_state_cov(t)
   - 计算当前时刻的 Q_t^*
   
3. 预测步骤：prediction_conventional()
   - 直接使用预先计算的 selected_state_cov
```

---

## 6. 运行时从数据类型到计算内核实例化链路

### 6.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Python 层（延迟加载）                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Representation                                                          │
│  ├── _representations: Dict[str, Dict[str, ndarray]]                    │
│  │       └── 'd': {'obs': ..., 'design': ..., 'transition': ...}      │
│  │       └── 's': {'obs': ..., 'design': ..., 'transition': ...}      │
│  │       └── ...                                                         │
│  │                                                                       │
│  ├── _statespaces: Dict[str, CythonClass]                              │
│  │       └── 'd': dStatespace 实例                                      │
│  │       └── 's': sStatespace 实例（按需创建）                          │
│  │       └── ...                                                         │
│  │                                                                       │
│  └── prefix 属性：动态检测当前最优精度                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ 延迟实例化
┌─────────────────────────────────────────────────────────────────────────┐
│                          Cython 层（编译时生成）                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  四套独立的类型安全内核：                                                │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ sStatespace │  │ dStatespace │  │ cStatespace │  │ zStatespace │ │
│  │ (float32)   │  │ (float64)   │  │ (complex64) │  │ (complex128)│ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │sKalmanFilter│  │dKalmanFilter│  │cKalmanFilter│  │zKalmanFilter│ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ sKalman...  │  │ dKalman...  │  │ cKalman...  │  │ zKalman...  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                                          │
│  由 Tempita 模板在构建时生成                                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 步骤一：数据类型检测

```python
# representation.py:731-747
@property
def prefix(self):
    """
    动态检测当前数据的最优 BLAS 前缀
    
    返回值：
    - 's': float32
    - 'd': float64
    - 'c': complex64
    - 'z': complex128
    """
    # 收集所有相关数组
    arrays = (
        self._design, self._obs_intercept, self._obs_cov,
        self._transition, self._state_intercept, 
        self._selection, self._state_cov,
    )
    if self.endog is not None:
        arrays = (self.endog,) + arrays
    
    # 调用 scipy.linalg.blas.find_best_blas_type
    # 自动确定最优精度
    return find_best_blas_type(arrays)[0]
```

**`find_best_blas_type` 的规则**：

| 输入数组类型 | 返回前缀 | 说明 |
|-------------|---------|------|
| 全部 float32 | 's' | 单精度实数 |
| 有 float64 | 'd' | 双精度实数（即使有 float32） |
| 全部 complex64 | 'c' | 单精度复数 |
| 有 complex128 | 'z' | 双精度复数（即使有 complex64） |
| 混合实数+复数 | 'c' 或 'z' | 统一为复数类型 |

### 6.3 步骤二：内核映射表

```python
# tools.py:28-118
# 前缀到 Cython 类的映射

# 状态表示
prefix_statespace_map = {
    "s": _representation.sStatespace,
    "d": _representation.dStatespace,
    "c": _representation.cStatespace,
    "z": _representation.zStatespace
}

# 卡尔曼滤波
prefix_kalman_filter_map = {
    "s": _kalman_filter.sKalmanFilter,
    "d": _kalman_filter.dKalmanFilter,
    "c": _kalman_filter.cKalmanFilter,
    "z": _kalman_filter.zKalmanFilter
}

# 卡尔曼平滑
prefix_kalman_smoother_map = {
    "s": _kalman_smoother.sKalmanSmoother,
    "d": _kalman_smoother.dKalmanSmoother,
    "c": _kalman_smoother.cKalmanSmoother,
    "z": _kalman_smoother.zKalmanSmoother
}

# 仿真平滑
prefix_simulation_smoother_map = {
    "s": _simulation_smoother.sSimulationSmoother,
    "d": _simulation_smoother.dSimulationSmoother,
    "c": _simulation_smoother.cSimulationSmoother,
    "z": _simulation_smoother.zSimulationSmoother
}

# CFA 仿真平滑
prefix_cfa_simulation_smoother_map = {
    "s": _cfa_simulation_smoother.sCFASimulationSmoother,
    "d": _cfa_simulation_smoother.dCFASimulationSmoother,
    "c": _cfa_simulation_smoother.cCFASimulationSmoother,
    "z": _cfa_simulation_smoother.zCFASimulationSmoother
}
```

### 6.4 步骤三：延迟实例化机制

```python
# representation.py:1042-1124
def _initialize_representation(self, prefix=None):
    """
    核心实例化函数：按需创建 Cython 内核
    
    设计模式：延迟加载 + 缓存
    """
    # 1. 如果未指定前缀，自动检测
    if prefix is None:
        prefix = self.prefix
    dtype = tools.prefix_dtype_map[prefix]  # np.float32, np.float64, 等
    
    # 2. 准备/转换矩阵数据到目标 dtype
    if prefix not in self._representations:
        self._representations[prefix] = {}
        
        # 遍历所有矩阵，转换到目标 dtype
        for matrix in ['obs', 'design', 'obs_intercept', 'obs_cov',
                       'transition', 'state_intercept', 'selection', 'state_cov']:
            # 获取原始矩阵
            original = getattr(self, f"_{matrix}")
            
            # 转换为目标 dtype，保持 Fortran 顺序
            new = original.astype(dtype, order="F")
            
            # 存储到缓存
            self._representations[prefix][matrix] = new
    
    # 3. 检查是否需要（重新）创建 Cython 内核
    if prefix in self._statespaces:
        ss = self._statespaces[prefix]
        # 检查是否有变化（维度、时变性等）
        create = (
            not ss.obs.shape[1] == self.endog.shape[1]
            or not ss.design.shape[2] == self.design.shape[2]
            or not ss.obs_intercept.shape[1] == self.obs_intercept.shape[1]
            # ... 更多检查
        )
    else:
        create = True
    
    # 4. 实例化 Cython 内核（如果需要）
    if create:
        # 删除旧实例（如果存在）
        if prefix in self._statespaces:
            del self._statespaces[prefix]
        
        # 获取对应的 Cython 类
        cls = self.prefix_statespace_map[prefix]
        
        # 实例化（传入所有矩阵）
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
    
    # 5. 初始化状态分布
    if isinstance(self.initialization, Initialization):
        self._statespaces[prefix].initialize(
            self.initialization, complex_step=complex_step
        )
    
    return prefix, dtype, create
```

### 6.5 步骤四：访问时自动选择

```python
# representation.py:776-780
@property
def _statespace(self):
    """
    获取当前最优精度的 Cython 状态空间实例
    
    每次访问都会：
    1. 检测当前数据类型
    2. 返回对应精度的实例
    """
    prefix = self.prefix
    if prefix in self._statespaces:
        return self._statespaces[prefix]
    return None
```

### 6.6 完整的调用链示例

**场景：用户调用 `model.filter()`**

```
用户调用：model.filter()
         │
         ▼
1. 检测数据类型
   prefix = model.prefix  # 例如 'd'
         │
         ▼
2. 检查/创建表示缓存
   if 'd' not in model._representations:
       # 转换所有矩阵到 float64
       model._representations['d'] = {
           'obs': model._obs.astype(np.float64, order='F'),
           'design': model._design.astype(np.float64, order='F'),
           ...
       }
         │
         ▼
3. 检查/创建 Cython 内核
   if 'd' not in model._statespaces or 需要重建:
       cls = tools.prefix_statespace_map['d']  # dStatespace
       model._statespaces['d'] = cls(
           model._representations['d']['obs'],
           model._representations['d']['design'],
           ...
       )
         │
         ▼
4. 调用内核方法
   model._statespaces['d'].filter(...)
         │
         ▼
5. 返回结果
   FilterResults(...)
```

### 6.7 设计优势

| 方面 | 说明 |
|------|------|
| **按需创建** | 只创建实际使用的精度内核，节省内存 |
| **类型安全** | Cython 静态类型，无运行时类型检查开销 |
| **自动优化** | 数据类型变化时自动重建内核 |
| **缓存复用** | 同一精度多次使用时直接复用 |
| **透明使用** | 用户无需关心精度，系统自动选择 |

### 6.8 多精度共存的场景

```python
# 示例：同一模型支持多种精度

model = SARIMAX(endog, order=(1,0,0))

# 第一次用 float64（默认）
res1 = model.filter()  # 创建 dStatespace 实例

# 切换到 float32
model.endog = model.endog.astype(np.float32)
res2 = model.filter()  # 创建 sStatespace 实例

# 现在 _statespaces 中有两个实例：
# model._statespaces['d'] = dStatespace(...)
# model._statespaces['s'] = sStatespace(...)
```

---

## 7. 架构设计总结（修正版）

### 7.1 多精度实现的精确理解

**关键点**：
- ❌ **不是**："单精度统一升为双精度"
- ✅ **而是**：BLAS 矩阵运算保持原精度，少数标量函数（log, abs）使用更高精度
- **条件分支**：`combined_prefix` 决定走哪个分支，`s` 和 `d` 走同一分支，`c` 和 `z` 走另一分支

### 7.2 参数更新机制的精确理解

**关键点**：
- 基类 `MLEModel.update()` **只调用 `handle_params()`**，不更新矩阵
- 子类 `update()` 需要：
  1. 调用 `handle_params()` 或 `super().update()`
  2. 实际更新状态空间矩阵
- `handle_params()` 职责：参数变换 + 固定参数注入

### 7.3 两类仿真平滑器的精确区分

| 特性 | KFS | CFA |
|------|-----|-----|
| **状态采样** | ✅ | ✅ |
| **扰动采样** | ✅ | ❌ |
| **退化分布** | ✅ | ❌ |
| **漫延初始化** | ✅ | ❌ |
| **适用模型** | 所有状态空间模型 | 仅动态因子、随机波动率等 |

### 7.4 selected_state_cov 的关键作用

**数学定义**：
```
Q_t^* = R_t Q_t R_t'
```

**两个关键用途**：
1. **预测递推**：$P_{t+1} = T P_{t|t} T' + Q^*$
2. **平稳初始化**：求解 Lyapunov 方程 $P = T P T' + Q^*$

**为什么不用 Q 代替 Q*？**
- Q 维度是 r×r，Q* 维度是 m×m
- 只有 Q* 与 P 同维度，可以相加
- Q* 允许退化（支持恒等式增广状态）

### 7.5 运行时实例化链路

```
数据输入 → 类型检测(prefix) → 矩阵缓存(_representations) → Cython内核(_statespaces)
     │              │                    │                            │
     │              │                    │                            │
     │         find_best_blas_type       │                      prefix_*_map
     │              │                    │                            │
     │              └────► astype(dtype) └────► 按需实例化 Cython类
```

---

## 8. 附录：核心文件索引

| 文件路径 | 主要内容 |
|----------|----------|
| `representation.py` | `Representation` 类，`prefix` 属性，`_initialize_representation()` |
| `_representation.pyx.in` | Cython 状态表示模板，`select_cov()` 函数，`initialize_stationary()` |
| `_conventional.pyx.in` | 卡尔曼滤波模板，**区分 `prefix` 和 `combined_prefix`** |
| `mlemodel.py` | `MLEModel` 类，`update()` 基类实现，`handle_params()` |
| `simulation_smoother.py` | KFS 仿真平滑器，**支持状态和扰动采样** |
| `cfa_simulation_smoother.py` | CFA 仿真平滑器，**仅支持状态采样**，限制说明 |
| `tools.py` | 前缀映射表 `prefix_*_map` |
| `sarimax.py` | SARIMAX 模型的 `update()` 实现示例 |

---

## 9. 修订历史

| 版本 | 日期 | 修订内容 |
|------|------|----------|
| V1 | 2026-05-01 | 初始版本 |
| **V2** | **2026-05-01** | **重要修正：**<br>1. 多精度实现：区分 `prefix`（BLAS 同精度）和 `combined_prefix`（少数合并）<br>2. 参数更新：明确基类 `update()` 只调用 `handle_params()`<br>3. 仿真平滑器：CFA 仅支持状态采样，KFS 支持状态和扰动采样<br>4. 派生协方差：补充 `selected_state_cov = R Q R'` 在预测和平稳初始化中的作用<br>5. 运行时链路：补充从数据类型检测到 Cython 内核实例化的完整延迟加载机制 |
