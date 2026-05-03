# ARIMA 与 SARIMAX 职责边界与调用链分析

## 目录
1. [类继承关系与核心差异](#1-类继承关系与核心差异)
2. [趋势项处理方式对比](#2-趋势项处理方式对比)
3. [ARIMA 的额外组件与职责](#3-arima-的额外组件与职责)
4. [完整调用链分析](#4-完整调用链分析)
5. [代码引用索引](#5-代码引用索引)

---

## 1. 类继承关系与核心差异

### 1.1 继承层次

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            继承关系图                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Representation                                                              │
│       │                                                                      │
│       ▼                                                                      │
│  KalmanFilter                                                               │
│       │                                                                      │
│       ▼                                                                      │
│  SimulationSmoother                                                         │
│       │                                                                      │
│       ▼                                                                      │
│  MLEModel                                                                   │
│       │                                                                      │
│       ▼                                                                      │
│  SARIMAX (statespace/sarimax.py)                                           │
│       │                                                                      │
│       ▼                                                                      │
│  ARIMA (arima/model.py) ← 新增: 规格验证 + 多种估计方法                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 类定义位置

| 类 | 文件 | 关键继承 |
|---|------|---------|
| **ARIMA** | `arima/model.py:26` | `class ARIMA(sarimax.SARIMAX):` |
| **SARIMAX** | `statespace/sarimax.py` | 继承自 `MLEModel` |

### 1.3 核心差异对比表

| 特性 | ARIMA | SARIMAX |
|-----|-------|---------|
| **趋势处理** | 趋势合并到 `exog`（回归与 ARIMA 误差） | 趋势在 `state_intercept` 或 `obs_intercept` |
| **trend 参数** | 默认: "c" (非集成) / "n" (集成) | 直接传递给状态空间 |
| **规格验证** | 通过 `SARIMAXSpecification` | 内置验证 |
| **参数管理** | 通过 `SARIMAXParams` | 直接管理 |
| **估计方法** | 6 种: statespace, innovations_mle, hannan_rissanen, burg, innovations, yule_walker | 仅状态空间 MLE |
| **GLS 估计** | 支持 (回归系数) | 不支持 |
| **手动差分** | 非状态空间方法需手动差分 | 内部处理 (状态空间或简单差分) |

---

## 2. 趋势项处理方式对比

### 2.1 核心差异：两种模型规格

**关键区别**：
- **SARIMAX**：趋势是状态空间模型的固有部分
- **ARIMA**：趋势是外生回归变量的一部分（"回归与 ARIMA 误差"）

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    SARIMAX 模型规格（趋势在状态空间）                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  状态空间形式:                                                                 │
│                                                                              │
│  观测方程: y_t = Z_t α_t + d_t + ε_t                                       │
│  状态方程: α_{t+1} = T_t α_t + c_t + R_t η_t                               │
│                                                                              │
│  趋势通过以下方式纳入:                                                         │
│  - state_intercept (c_t): 状态截距（Harvey 表示）                            │
│  - obs_intercept (d_t): 观测截距（Hamilton 表示）                            │
│                                                                              │
│  示例: trend='c' (常数项)                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ SARIMAX.update() (sarimax.py:1668-1687):                              │ │
│  │                                                                          │ │
│  │ if self._k_trend > 0:                                                   │ │
│  │     data = np.dot(self._trend_data, params_trend)                      │ │
│  │                                                                          │ │
│  │     if not self.hamilton_representation:                                 │ │
│  │         // Harvey 表示: 趋势进入状态截距                                 │ │
│  │         self.ssm["state_intercept", self._k_states_diff, :] = data     │ │
│  │     else:                                                                │ │
│  │         // Hamilton 表示: 趋势进入观测截距                               │ │
│  │         // 注意: 需要转换为均值形式                                        │ │
│  │         data /= np.sum(-reduced_polynomial_ar)                          │ │
│  │         self.ssm["obs_intercept"] = data[None, :]                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  _trend_data 是预计算的趋势矩阵:                                             │
│  - trend='c' → [1, 1, 1, ...] (nobs × 1)                                   │
│  - trend='t' → [1, 2, 3, ...] (nobs × 1)                                  │
│  - trend='ct' → [[1,1], [1,2], [1,3], ...] (nobs × 2)                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    ARIMA 模型规格（回归与 ARIMA 误差）                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  模型形式:                                                                    │
│                                                                              │
│  Y_t - δ₀ - δ₁t - ... - X_tβ = ε_t                                         │
│  (1-L)^d (1-L^s)^D Φ(L) Φ_s(L) ε_t = Θ(L) Θ_s(L) η_t                     │
│                                                                              │
│  其中: η_t ~ WN(0, σ²)                                                      │
│                                                                              │
│  关键点: 趋势项 (δ₀, δ₁, ...) 和回归系数 (β) 都被视为外生回归变量!          │
│                                                                              │
│  趋势数据合并到 exog 的过程 (specification.py:432-444):                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ // 添加趋势数据到 exog                                                   │ │
│  │ if self.trend_order is not None:                                        │ │
│  │     trend_data = self.construct_trend_data(nobs, trend_offset)         │ │
│  │                                                                          │ │
│  │     if exog is None:                                                     │ │
│  │         exog = trend_data                                                │ │
│  │     elif exog_is_pandas:                                                 │ │
│  │         // Pandas DataFrame: 按列拼接                                   │ │
│  │         trend_data = pd.DataFrame(...)                                   │ │
│  │         exog = pd.concat([trend_data, exog], axis=1)                   │ │
│  │     else:                                                                │ │
│  │         // NumPy array: 按列拼接                                         │ │
│  │         exog = np.c_[trend_data, exog]                                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  示例: trend='c' (常数项) + exog = [X₁, X₂]                               │
│                                                                              │
│  ARIMA 处理流程:                                                             │
│  1. 生成趋势数据: trend_data = [1, 1, 1, ...] (nobs × 1)                 │
│  2. 合并到 exog: exog = [trend_data, X₁, X₂] (nobs × 3)                 │
│  3. 传递给 SARIMAX 时: trend=None                                          │
│  4. SARIMAX 看到的 exog = [1, X₁, X₂]                                      │
│  5. SARIMAX 的 mle_regression=True 自动启用                                │
│                                                                              │
│  参数向量在 SARIMAX.update() 中的处理:                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ // SARIMAX.update() (sarimax.py:1566-1660)                             │ │
│  │                                                                          │ │
│  │ // 参数向量的第一个元素是趋势参数!                                       │ │
│  │ end += self._k_trend          // = 0 (因为 ARIMA 传入 trend=None)      │ │
│  │ params_trend = params[start:end]  // 空!                               │ │
│  │ start += self._k_trend                                                   │ │
│  │                                                                          │ │
│  │ // 然后是 exog 参数 (包含趋势!)                                           │ │
│  │ end += self._k_exog              // = 3 (常数 + X₁ + X₂)              │ │
│  │ params_exog = params[start:end]   // [δ₀, β₁, β₂]                     │ │
│  │ start += self._k_exog                                                    │ │
│  │                                                                          │ │
│  │ // 回归系数通过观测截距实现                                                │ │
│  │ if self.mle_regression:                                                   │ │
│  │     self.ssm["obs_intercept"] = np.dot(self.exog, params_exog)[None, :] │ │
│  │     // = δ₀*1 + β₁*X₁ + β₂*X₂                                          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 初始化时的关键差异

**ARIMA 初始化** (`arima/model.py:143-201`)：

```python
def __init__(self, endog, exog=None, order=(0, 0, 0),
             seasonal_order=(0, 0, 0, 0), trend=None,
             ...):
    
    # ========== 关键差异 1: 趋势默认值 ==========
    # SARIMAX 没有这个逻辑，ARIMA 自动设置
    integrated = order[1] > 0 or seasonal_order[1] > 0
    if trend is None and not integrated:
        trend = "c"       # 非集成模型默认有截距
    elif trend is None:
        trend = "n"       # 集成模型默认无趋势

    # ========== 关键差异 2: 创建规格对象 ==========
    # SARIMAX 没有这一步
    self._spec_arima = SARIMAXSpecification(
        endog, exog=exog, order=order, seasonal_order=seasonal_order,
        trend=trend, enforce_stationarity=None, enforce_invertibility=None,
        ...)
    # SARIMAXSpecification 内部:
    # - 将 trend 转换为数据列
    # - 拼接 trend_data + exog → 新的 exog
    exog = self._spec_arima._model.data.orig_exog  # 包含趋势!

    # ========== 关键差异 3: 区分原始 exog 和趋势 ==========
    # 保存用户原始输入的 exog（不包含趋势），用于 append 等操作
    input_exog = None
    if exog is not None:
        if _is_using_pandas(exog, None):
            input_exog = exog.iloc[:, self._spec_arima.k_trend:]
        else:
            input_exog = exog[:, self._spec_arima.k_trend:]
    
    # ========== 关键差异 4: 调用 SARIMAX 时传入 trend=None ==========
    # 因为趋势已经合并到 exog 中了!
    super().__init__(
        endog, exog, trend=None,  # ← 关键点!
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        ...)
    
    self.trend = trend  # 保存原始 trend 参数值
    self._input_exog = input_exog  # 保存用户原始 exog
```

### 2.3 两种规格的数学对比

| 规格 | 趋势表示 | 参数位置 |
|-----|---------|---------|
| **SARIMAX (状态空间趋势)** | `state_intercept` 或 `obs_intercept` | 参数向量最前面 (params_trend) |
| **ARIMA (回归与误差)** | `exog` 的列 | 参数向量的 exog 部分 (params_exog) |

**ARIMA 调用 SARIMAX 后的状态：**
- SARIMAX 看到的 `trend=None` → `_k_trend=0`
- SARIMAX 看到的 `exog` 包含趋势数据 → `mle_regression=True`
- 趋势参数在 `params_exog` 中处理

---

## 3. ARIMA 的额外组件与职责

### 3.1 组件架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         ARIMA 新增组件架构                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ARIMA (arima/model.py)                                                      │
│       │                                                                      │
│       ├──► SARIMAXSpecification (arima/specification.py)                   │
│       │         │                                                            │
│       │         ├── 验证模型规格 (validate_estimator, etc.)               │
│       │         ├── 处理趋势项 (trend_data → exog)                        │
│       │         ├── 管理滞后阶数 (ar_lags, ma_lags, etc.)                │
│       │         ├── 拆分参数 (split_params)                                │
│       │         └── 构建 TimeSeriesModel (_model)                          │
│       │                                                                      │
│       ├──► SARIMAXParams (arima/params.py)                                  │
│       │         │                                                            │
│       │         ├── 参数分段访问 (exog_params, ar_params, ma_params)     │
│       │         ├── 多项式表示 (ar_poly, ma_poly)                         │
│       │         └── 平稳性/可逆性验证 (is_stationary, is_invertible)      │
│       │                                                                      │
│       └──► 多种估计器 (arima/estimators/*.py)                              │
│                 │                                                            │
│                 ├── statespace.py → 委托给 SARIMAX.fit()                  │
│                 ├── gls.py → 广义最小二乘法                                │
│                 ├── hannan_rissanen.py → Hannan-Rissanen 方法            │
│                 ├── yule_walker.py → Yule-Walker 方法 (AR 专用)          │
│                 ├── burg.py → Burg 算法 (AR 专用)                         │
│                 ├── innovations.py → 创新算法 (MA 专用)                    │
│                 ├── innovations_mle.py → 创新极大似然                       │
│                 └── durbin_levinson.py → Durbin-Levinson 算法             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 SARIMAXSpecification 详解

**文件位置**：`arima/specification.py`

**核心职责**：

```python
class SARIMAXSpecification:
    """
    SARIMAX 模型规格验证与管理
    
    这是 ARIMA 新增的核心组件，SARIMAX 没有对应的类。
    """
    
    def __init__(self, endog, exog=None, order=(0, 0, 0),
                 seasonal_order=(0, 0, 0, 0), trend=None, ...):
        
        # ========== 1. 规格验证 ==========
        # 验证滞后阶数格式（整数或列表）
        # 例如: order=(2,1,1) 或 order=([1,3], 1, [1])
        
        # ========== 2. 趋势处理 (关键!) ==========
        self.trend = trend
        self.trend_poly, _ = prepare_trend_spec(trend)
        # trend_poly 示例:
        #   'n' → [0, 0, 0, ...] (无趋势)
        #   'c' → [1, 0, 0, ...] (常数, t^0)
        #   't' → [0, 1, 0, ...] (线性, t^1)
        #   'ct' → [1, 1, 0, ...] (常数+线性)
        #   [1,1,0,1] → 1 + t + t^3 (多项式)
        
        self.trend_terms = np.where(self.trend_poly == 1)[0]
        # 例如 'ct' → [0, 1] (t^0, t^1)
        
        self.k_trend = len(self.trend_terms)
        
        # ========== 3. 合并趋势到 exog ==========
        if self.trend_order is not None:
            # 构建趋势数据矩阵
            trend_data = self.construct_trend_data(nobs, trend_offset)
            # trend='c' → [[1], [1], [1], ...]
            # trend='t' → [[1], [2], [3], ...]
            # trend='ct' → [[1,1], [1,2], [1,3], ...]
            
            # 拼接到 exog (趋势在前，exog 在后)
            if exog is None:
                exog = trend_data
            else:
                exog = np.c_[trend_data, exog]  # NumPy
                # 或 pd.concat([trend_data, exog], axis=1)  # Pandas
        
        # ========== 4. 创建底层 TimeSeriesModel ==========
        # 处理 endog/exog 验证、数据类型转换等
        self._model = TimeSeriesModel(endog, exog=exog, ...)
        self.exog = self._model.exog  # 包含趋势!
```

**关键属性**：

| 属性 | 含义 | 示例 |
|-----|------|------|
| `trend` | 原始趋势参数 | `'c'`, `'t'`, `'ct'`, `[1,1,0,1]` |
| `trend_poly` | 趋势多项式指示 | `'ct'` → `[1, 1, 0, ...]` |
| `trend_terms` | 包含的趋势项指数 | `'ct'` → `[0, 1]` |
| `k_trend` | 趋势参数数量 | `'ct'` → `2` |
| `trend_order` | 趋势阶数（连续） | `'ct'` → `1`, `[1,1,0,1]` → `None` |
| `trend_degree` | 最高趋势阶数 | `'ct'` → `1`, `[1,1,0,1]` → `3` |
| `ar_lags` | AR 滞后列表 | `order=2` → `[1, 2]`, `order=[1,3]` → `[1, 3]` |
| `ma_lags` | MA 滞后列表 | 同上 |
| `k_params` | 总参数数 | `k_exog + k_ar + k_ma + ... + (1 if not concentrate_scale)` |

**关键方法**：

```python
def split_params(self, params, allow_infnan=False):
    """
    将参数向量拆分为各分量
    
    返回: dict 包含:
    - 'exog_params': 外生/趋势参数
    - 'ar_params': AR 参数
    - 'ma_params': MA 参数
    - 'seasonal_ar_params': 季节 AR 参数
    - 'seasonal_ma_params': 季节 MA 参数
    - 'variance': 方差参数
    """

def validate_estimator(self, method):
    """
    验证估计方法是否适用于当前规格
    
    例如:
    - 'yule_walker' 只适用于纯 AR 模型
    - 'burg' 只适用于纯 AR 模型
    - 'innovations' 只适用于纯 MA 模型
    """

def construct_trend_data(self, nobs, offset=1):
    """
    构建趋势数据矩阵
    
    offset=1 时:
    trend='c' → [[1], [1], [1], ...] (nobs × 1)
    trend='t' → [[1], [2], [3], ...] (nobs × 1)
    trend='ct' → [[1,1], [1,2], [1,3], ...] (nobs × 2)
    """
```

### 3.3 SARIMAXParams 详解

**文件位置**：`arima/params.py`

**核心职责**：提供参数的结构化访问

```python
class SARIMAXParams:
    """
    SARIMAX 参数管理类
    
    提供方便的属性访问，以及多项式表示和平稳性验证。
    """
    
    def __init__(self, spec):
        self.spec = spec  # SARIMAXSpecification 实例
        
        # 参数数量
        self.k_exog_params = spec.k_exog_params
        self.k_ar_params = spec.k_ar_params
        self.k_ma_params = spec.k_ma_params
        # ...
    
    @property
    def exog_params(self):
        """外生/趋势参数"""
        return self._params_split["exog_params"]
    
    @property
    def ar_params(self):
        """AR 参数"""
        return self._params_split["ar_params"]
    
    @property
    def ma_params(self):
        """MA 参数"""
        return self._params_split["ma_params"]
    
    @property
    def ar_poly(self):
        """
        AR 滞后多项式 (numpy.polynomial.Polynomial)
        
        对于 AR 参数 [φ₁, φ₂], 多项式为:
        1 - φ₁ L - φ₂ L²
        
        返回: Polynomial([1, -φ₁, -φ₂])
        """
        coef = np.zeros(self.spec.max_ar_order + 1)
        coef[0] = 1
        ix = self.spec.ar_lags
        coef[ix] = -self._params_split["ar_params"]  # 注意负号!
        return Polynomial(coef)
    
    @property
    def ma_poly(self):
        """
        MA 滞后多项式
        
        对于 MA 参数 [θ₁, θ₂], 多项式为:
        1 + θ₁ L + θ₂ L²
        
        返回: Polynomial([1, θ₁, θ₂])
        """
        coef = np.zeros(self.spec.max_ma_order + 1)
        coef[0] = 1
        ix = self.spec.ma_lags
        coef[ix] = self._params_split["ma_params"]  # 没有负号!
        return Polynomial(coef)
    
    @property
    def is_stationary(self):
        """
        AR 多项式是否平稳 (所有根在单位圆外)
        """
        return is_invertible(self.ar_poly)
    
    @property
    def is_invertible(self):
        """
        MA 多项式是否可逆 (所有根在单位圆外)
        """
        return is_invertible(self.ma_poly)
    
    @property
    def params(self):
        """
        完整参数向量
        
        顺序: [exog_params, ar_params, ma_params, 
               seasonal_ar_params, seasonal_ma_params, variance]
        """
        # ... 拼接各部分
```

### 3.4 多种估计方法

**ARIMA.fit()** 支持 6 种估计方法（`arima/model.py:228-497`）：

```python
def fit(self, start_params=None, transformed=True, includes_fixed=False,
        method=None, method_kwargs=None, gls=None, gls_kwargs=None,
        ...):
    """
    参数:
    - method: 估计方法
      - 'statespace': 状态空间 MLE (默认，委托给 SARIMAX.fit())
      - 'innovations_mle': 创新极大似然
      - 'hannan_rissanen': Hannan-Rissanen 方法
      - 'burg': Burg 算法 (仅 AR(p))
      - 'innovations': 创新算法 (仅 MA(q))
      - 'yule_walker': Yule-Walker 方法 (仅 AR(p))
    """
    
    # ========== 1. 选择估计方法 ==========
    if method is not None:
        self._spec_arima.validate_estimator(method)
    else:
        method = "statespace"  # 默认使用状态空间
    
    # ========== 2. 方法可用性检查 ==========
    methods_with_fixed_params = ["statespace", "hannan_rissanen"]
    if self._has_fixed_params and method not in methods_with_fixed_params:
        raise ValueError("固定参数仅支持 statespace 和 hannan_rissanen")
    
    # ========== 3. 准备 kwargs ==========
    if method == "statespace":
        # 传递模型级别的参数给 SARIMAX
        method_kwargs["enforce_stationarity"] = self.enforce_stationarity
        method_kwargs["enforce_invertibility"] = self.enforce_invertibility
        method_kwargs["concentrate_scale"] = self.concentrate_scale
    elif method == "innovations_mle":
        method_kwargs["enforce_invertibility"] = self.enforce_invertibility
    
    # ========== 4. 执行估计 ==========
    has_exog = self._spec_arima.exog is not None
    
    if has_exog or method == "statespace":
        # ========== 分支 A: 有 exog 或 statespace 方法 ==========
        
        # GLS 估计选择
        # - 有 exog 且 (显式请求 GLS 或方法非 statespace)
        if has_exog and (gls or (gls is None and method != "statespace")):
            # 广义最小二乘法
            # 先估计回归系数，再估计 ARMA 参数
            p, fit_details = estimate_gls(
                self.endog, exog=self.exog, order=self.order,
                seasonal_order=self.seasonal_order, include_constant=False,
                arma_estimator=method, arma_estimator_kwargs=method_kwargs,
                **gls_kwargs)
        
        elif method != "statespace":
            # 有 exog 但禁用 GLS 且方法非 statespace → 错误
            raise ValueError("有 exog 且禁用 GLS 时仅支持 method='statespace'")
        
        else:
            # ========== 关键: 委托给 SARIMAX.fit() ==========
            method_kwargs.setdefault("disp", 0)
            
            # 调用父类方法 (SARIMAX.fit())
            res = super().fit(
                return_params=return_params, low_memory=low_memory,
                cov_type=cov_type, cov_kwds=cov_kwds, **method_kwargs)
            
            if not return_params:
                res.fit_details = res.mlefit
    
    else:
        # ========== 分支 B: 无 exog 且方法非 statespace ==========
        # 这些方法不支持内置差分，需要手动处理
        
        endog = self.endog
        order = self._spec_arima.order
        seasonal_order = self._spec_arima.seasonal_order
        
        # 手动差分 (如果需要)
        if self._spec_arima.is_integrated:
            warnings.warn('已对数据进行差分以消除集成性')
            endog = diff(
                endog, k_diff=self._spec_arima.diff,
                k_seasonal_diff=self._spec_arima.seasonal_diff,
                seasonal_periods=self._spec_arima.seasonal_periods)
            # 调整阶数 (差分阶数设为 0)
            if order[1] > 0:
                order = (order[0], 0, order[2])
            if seasonal_order[1] > 0:
                seasonal_order = (seasonal_order[0], 0, seasonal_order[2],
                                  seasonal_order[3])
        
        # ========== 调用专门估计器 ==========
        if method == "yule_walker":
            # Yule-Walker: 仅 AR(p)
            p, fit_details = yule_walker(
                endog, ar_order=order[0], demean=False, **method_kwargs)
        
        elif method == "burg":
            # Burg 算法: 仅 AR(p)
            p, fit_details = burg(endog, ar_order=order[0],
                                  demean=False, **method_kwargs)
        
        elif method == "hannan_rissanen":
            # Hannan-Rissanen: ARMA
            p, fit_details = hannan_rissanen(
                endog, ar_order=order[0],
                ma_order=order[2], demean=False, **method_kwargs)
        
        elif method == "innovations":
            # 创新算法: 仅 MA(q)
            p, fit_details = innovations(
                endog, ma_order=order[2], demean=False,
                **method_kwargs)
            p = p[-1]  # 取指定阶数的估计
        
        elif method == "innovations_mle":
            # 创新极大似然: ARMA
            p, fit_details = innovations_mle(
                endog, order=order,
                seasonal_order=seasonal_order,
                demean=False, **method_kwargs)
    
    # ========== 5. 后处理 (非 statespace 方法) ==========
    if p is not None:
        # 验证估计结果是否满足平稳性/可逆性约束
        if (self.enforce_stationarity
                and self._spec_arima.max_reduced_ar_order > 0
                and not p.is_stationary):
            raise ValueError('估计的 AR 参数非平稳，考虑设置 enforce_stationarity=False')
        
        if (self.enforce_invertibility
                and self._spec_arima.max_reduced_ma_order > 0
                and not p.is_invertible):
            raise ValueError('估计的 MA 参数非可逆，考虑设置 enforce_invertibility=False')
        
        if return_params:
            res = p.params
        else:
            # 手动调用 filter() 或 smooth() 产生结果对象
            if low_memory:
                conserve_memory = self.ssm.conserve_memory
                self.ssm.set_conserve_memory(MEMORY_CONSERVE)
            
            # 选择滤波或平滑
            if (self.ssm.memory_no_predicted or self.ssm.memory_no_gain
                    or self.ssm.memory_no_smoothing):
                func = self.filter
            else:
                func = self.smooth
            
            # 执行滤波/平滑
            res = func(p.params, transformed=True, includes_fixed=True,
                       cov_type=cov_type, cov_kwds=cov_kwds)
            
            res.fit_details = fit_details
            
            if low_memory:
                self.ssm.set_conserve_memory(conserve_memory)
    
    return res
```

### 3.5 估计方法对比表

| 方法 | 适用模型 | 是否支持差分 | 是否支持 exog | 估计原理 |
|-----|---------|------------|--------------|---------|
| **statespace** | 所有 SARIMAX | 内置 | ✅ 支持 | 卡尔曼滤波 + MLE |
| **innovations_mle** | ARMA | 需手动 | ❌ 不支持 | 创新算法 + MLE |
| **hannan_rissanen** | ARMA | 需手动 | ❌ 不支持 | 多阶段最小二乘 |
| **yule_walker** | 仅 AR(p) | 需手动 | ❌ 不支持 | Yule-Walker 方程 |
| **burg** | 仅 AR(p) | 需手动 | ❌ 不支持 | Burg 算法（递推） |
| **innovations** | 仅 MA(q) | 需手动 | ❌ 不支持 | 创新算法 |

**注意**：
- 非 `statespace` 方法需要 **手动差分** 处理集成模型
- 非 `statespace` 方法不支持 `exog`（除非使用 GLS）
- 只有 `statespace` 和 `hannan_rissanen` 支持固定参数

### 3.6 GLS 估计详解

当有 `exog` 且使用非 `statespace` 方法时，ARIMA 使用 **GLS（广义最小二乘法）**：

**文件位置**：`arima/estimators/gls.py`

**基本思想**：
1. 先用 OLS 估计回归系数
2. 用残差估计 ARMA 参数
3. 用 ARMA 参数构造 GLS 变换
4. 用变换后的数据重新估计

这是 Cochrane-Orcutt 类型的迭代估计方法。

---

## 4. 完整调用链分析

### 4.1 场景 1: ARIMA 用 statespace 方法（默认）

**用户调用**：
```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(endog, order=(1, 0, 1), trend='c')
res = model.fit(method='statespace')  # 默认
```

**完整调用链**：

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    ARIMA 初始化阶段                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. ARIMA.__init__() (arima/model.py:138-222)                              │
│     │                                                                        │
│     ├──► 设置趋势默认值                                                      │
│     │    integrated = False (d=0)                                          │
│     │    trend = 'c' (默认)                                                 │
│     │                                                                        │
│     ├──► 创建 SARIMAXSpecification                                          │
│     │    spec = SARIMAXSpecification(endog, order=(1,0,1), trend='c')    │
│     │    │                                                                  │
│     │    └──► spec 内部:                                                   │
│     │         - trend_poly = [1, 0, 0, ...]  (常数项)                    │
│     │         - trend_terms = [0]                                          │
│     │         - k_trend = 1                                                 │
│     │         - 构建 trend_data = [[1], [1], [1], ...] (nobs × 1)         │
│     │         - exog = trend_data (因为用户没提供 exog)                      │
│     │         - 创建 _model = TimeSeriesModel(endog, exog=[[1],...])      │
│     │                                                                        │
│     ├──► 提取 input_exog                                                     │
│     │    input_exog = None (用户没提供 exog)                                 │
│     │                                                                        │
│     └──► 调用 SARIMAX.__init__()                                            │
│          super().__init__(                                                  │
│              endog, exog=[[1],...], trend=None,  # ← 关键: trend=None!   │
│              order=(1,0,1), ...                                             │
│          )                                                                   │
│          │                                                                  │
│          └──► SARIMAX 内部:                                                 │
│               - trend=None → _k_trend = 0                                  │
│               - exog 存在 → mle_regression = True                           │
│               - _k_exog = 1                                                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    ARIMA 拟合阶段 (method='statespace')                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  2. ARIMA.fit(method='statespace') (arima/model.py:228-497)                │
│     │                                                                        │
│     ├──► 验证方法有效性                                                      │
│     │    method = 'statespace' (默认)                                       │
│     │                                                                        │
│     ├──► 准备 method_kwargs                                                  │
│     │    method_kwargs['enforce_stationarity'] = True                      │
│     │    method_kwargs['enforce_invertibility'] = True                     │
│     │    method_kwargs['concentrate_scale'] = False                         │
│     │                                                                        │
│     └──► 调用 SARIMAX.fit()                                                 │
│          res = super().fit(**method_kwargs)                                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    SARIMAX → MLEModel 阶段                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  3. SARIMAX.fit() → MLEModel.fit() (mlemodel.py:540-787)                  │
│     │                                                                        │
│     ├──► 获取 start_params                                                   │
│     │    if start_params is None:                                           │
│     │        start_params = self.start_params (SARIMAX 提供)                │
│     │                                                                        │
│     │    参数向量结构 (k_params = 4):                                        │
│     │    [β₀ (常数项), φ₁ (AR), θ₁ (MA), σ² (方差)]                        │
│     │                                                                        │
│     └──► 调用 scipy.optimize.fmin_l_bfgs_b                                │
│          优化器反复调用 self.loglike()                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    对数似然计算阶段                                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  4. MLEModel.loglike(params) (mlemodel.py:987-1040)                        │
│     │                                                                        │
│     ├──► handle_params(params)                                               │
│     │    │                                                                  │
│     │    └──► 若未变换: transform_params(params)                           │
│     │         SARIMAX.transform_params():                                   │
│     │         - β₀: 不变换 (直接使用)                                        │
│     │         - φ₁: constrain_stationary_univariate()                       │
│     │         - θ₁: -constrain_stationary_univariate()                      │
│     │         - σ²: x**2                                                    │
│     │                                                                        │
│     ├──► SARIMAX.update(params) (sarimax.py:1530-1719)                    │
│     │    │                                                                  │
│     │    ├──► 参数分段提取                                                  │
│     │    │    start = end = 0                                              │
│     │    │    end += self._k_trend     # = 0 (trend=None)                 │
│     │    │    params_trend = params[0:0]  # 空!                            │
│     │    │    start += 0                                                    │
│     │    │                                                                  │
│     │    │    end += self._k_exog       # = 1 (常数项)                     │
│     │    │    params_exog = params[0:1]   # [β₀]                           │
│     │    │    start += 1                                                    │
│     │    │                                                                  │
│     │    │    end += self.k_ar_params    # = 1                              │
│     │    │    params_ar = params[1:2]      # [φ₁]                          │
│     │    │    start += 1                                                    │
│     │    │                                                                  │
│     │    │    end += self.k_ma_params    # = 1                              │
│     │    │    params_ma = params[2:3]      # [θ₁]                          │
│     │    │    start += 1                                                    │
│     │    │                                                                  │
│     │    │    params_variance = params[3]   # σ²                           │
│     │    │                                                                  │
│     │    ├──► 更新状态空间矩阵                                              │
│     │    │                                                                  │
│     │    │    // 1. 观测截距 (回归系数)                                    │
│     │    │    if self.mle_regression:                                      │
│     │    │        // 关键: 常数项通过观测截距实现                           │
│     │    │        // self.exog = [[1], [1], [1], ...] (nobs × 1)          │
│     │    │        // params_exog = [β₀]                                      │
│     │    │        self.ssm["obs_intercept"] = np.dot(self.exog, params_exog)│
│     │    │                                    // = [β₀*1, β₀*1, ...]        │
│     │    │                                    // = [β₀, β₀, ...]           │
│     │    │                                                                  │
│     │    │    // 2. 过渡矩阵 (AR 参数)                                      │
│     │    │    // 伴随矩阵第一列                                             │
│     │    │    self.ssm[self.transition_ar_params_idx] = φ₁                │
│     │    │                                                                  │
│     │    │    // 3. 选择矩阵 (MA 参数)                                      │
│     │    │    self.ssm[self.selection_ma_params_idx] = θ₁                 │
│     │    │                                                                  │
│     │    │    // 4. 状态协方差                                              │
│     │    │    self.ssm["state_cov", 0, 0] = σ²                            │
│     │    │                                                                  │
│     │    └──► 返回 params                                                   │
│     │                                                                        │
│     └──► 卡尔曼滤波计算似然                                                  │
│          loglike = self.ssm.loglike()                                       │
│          → KalmanFilter.filter()                                            │
│          → 预测-更新迭代，累积对数似然                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 场景 2: ARIMA 用非 statespace 方法

**用户调用**：
```python
model = ARIMA(endog, order=(1, 1, 1), trend='n')  # d=1, 无趋势
res = model.fit(method='innovations_mle')
```

**关键差异**：
1. 有集成 (`d=1`)，需要 **手动差分**
2. 没有 `exog`（`trend='n'`），走分支 B
3. 估计后手动调用 `filter()` 产生结果

**调用链简化**：

```
ARIMA.fit(method='innovations_mle')
    │
    ├──► 方法检查: innovations_mle 适用于 ARMA
    │
    ├──► 无 exog 且方法非 statespace → 分支 B
    │
    ├──► 手动差分 (因为 d=1)
    │    endog = diff(endog, k_diff=1)
    │    order = (1, 0, 1)  # d 设为 0
    │
    ├──► 调用专门估计器
    │    p, fit_details = innovations_mle(
    │        endog, order=(1,0,1), ...)
    │    │
    │    └──► 返回 SARIMAXParams 对象 p
    │         p.params = [φ₁, θ₁, σ²]
    │         p.is_stationary → 验证
    │         p.is_invertible → 验证
    │
    ├──► 验证平稳性/可逆性
    │    if enforce_stationarity and not p.is_stationary:
    │        raise ValueError
    │
    └──► 手动调用 filter() 产生结果
         res = self.filter(p.params, transformed=True, ...)
         res.fit_details = fit_details
```

### 4.3 ARIMA 与 SARIMAX 的参数索引对比

**相同模型规格，不同参数顺序**：

```
模型: ARIMA(1,0,1) with trend='c'
等价于: 带常数项的 ARMA(1,1)
```

**ARIMA 的处理**：
- `trend='c'` → 合并到 `exog` → `mle_regression=True`
- 参数向量: `[β₀, φ₁, θ₁, σ²]`
- `params_exog` = `[β₀]` (第一个元素)

**如果直接用 SARIMAX**：
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(endog, order=(1,0,1), trend='c')
```
- `trend='c'` → 走 `state_intercept` 或 `obs_intercept`
- 参数向量: `[δ₀, φ₁, θ₁, σ²]`
- `params_trend` = `[δ₀]` (第一个元素)
- `_k_trend = 1`

**看似相同，实则不同**：
- ARIMA: `β₀` 是 **回归系数**，通过 `obs_intercept = Xβ` 实现
- SARIMAX: `δ₀` 是 **趋势参数**，通过 `state_intercept` 或 `obs_intercept` 直接设置

**数值上等价，但实现路径不同**。

---

## 5. 代码引用索引

### 5.1 ARIMA 核心文件

| 文件 | 行号 | 功能 |
|-----|------|------|
| `arima/model.py:26` | 26 | `class ARIMA(sarimax.SARIMAX)` |
| `arima/model.py:138-222` | 138-222 | `ARIMA.__init__()` |
| `arima/model.py:143-152` | 143-152 | 趋势默认值设置 |
| `arima/model.py:159-164` | 159-164 | 创建 `SARIMAXSpecification` |
| `arima/model.py:195-201` | 195-201 | 调用 `SARIMAX.__init__()` 时 `trend=None` |
| `arima/model.py:228-497` | 228-497 | `ARIMA.fit()` 多种估计方法 |
| `arima/model.py:316-325` | 316-325 | 方法选择逻辑 |
| `arima/model.py:371-392` | 371-392 | GLS 估计分支 |
| `arima/model.py:393-399` | 393-399 | 委托给 `SARIMAX.fit()` |
| `arima/model.py:401-449` | 401-449 | 非 statespace 方法分支 |
| `arima/model.py:452-496` | 452-496 | 后处理 (filter/smooth) |

### 5.2 SARIMAXSpecification 文件

| 文件 | 行号 | 功能 |
|-----|------|------|
| `arima/specification.py:23` | 23 | `class SARIMAXSpecification` |
| `arima/specification.py:381-421` | 381-421 | 趋势处理 (`trend_poly`, `trend_terms`, `k_trend`) |
| `arima/specification.py:432-444` | 432-444 | 趋势数据合并到 `exog` |
| `arima/specification.py:449-452` | 449-452 | 创建 `TimeSeriesModel` |
| `arima/specification.py:568-600` | 568-600 | `valid_estimators` 属性 |
| `arima/specification.py:736-800` | 736-800 | `split_params()` 方法 |
| `arima/specification.py:1033-1050` | 1033-1050 | `construct_trend_data()` 方法 |

### 5.3 SARIMAXParams 文件

| 文件 | 行号 | 功能 |
|-----|------|------|
| `arima/params.py:15` | 15 | `class SARIMAXParams` |
| `arima/params.py:80-90` | 80-90 | `exog_params` 属性 |
| `arima/params.py:93-103` | 93-103 | `ar_params` 属性 |
| `arima/params.py:106-131` | 106-131 | `ar_poly` 属性 (多项式构建) |
| `arima/params.py:147-160` | 147-160 | `ma_poly` 属性 |
| `arima/params.py:170-180` | 170-180 | `is_stationary` 属性 |
| `arima/params.py:182-192` | 182-192 | `is_invertible` 属性 |

### 5.4 估计器文件

| 文件 | 功能 |
|-----|------|
| `arima/estimators/statespace.py` | 委托给 `SARIMAX` 的包装 |
| `arima/estimators/gls.py` | 广义最小二乘法 |
| `arima/estimators/hannan_rissanen.py` | Hannan-Rissanen 方法 |
| `arima/estimators/yule_walker.py` | Yule-Walker 方法 |
| `arima/estimators/burg.py` | Burg 算法 |
| `arima/estimators/innovations.py` | 创新算法 |
| `arima/estimators/innovations_mle.py` | 创新极大似然 |

### 5.5 SARIMAX 相关代码（对比用）

| 文件 | 行号 | 功能 |
|-----|------|------|
| `statespace/sarimax.py:389` | 389 | `self._k_trend = self.k_trend` |
| `statespace/sarimax.py:588-591` | 588-591 | 准备 `_trend_data` |
| `statespace/sarimax.py:1664-1687` | 1664-1687 | 趋势通过 `state_intercept` 或 `obs_intercept` |
| `statespace/sarimax.py:1566-1660` | 1566-1660 | `update()` 中的参数分段 |

---

## 附录：关键设计决策总结

### A.1 为什么 ARIMA 要把趋势合并到 exog？

**原因**：实现 **"回归与 ARIMA 误差" (regression with ARIMA errors)** 的经典规格：

```
Y_t = X_t β + ε_t
ε_t ~ ARIMA(p,d,q)
```

这与 SARIMAX 的规格不同：
- SARIMAX: 趋势是状态空间模型的固有部分
- ARIMA: 趋势是回归方程的一部分，误差项服从 ARIMA

**优势**：
1. 与经典计量经济学文献一致
2. 可以使用 GLS 等传统估计方法
3. 更灵活的趋势指定（多项式趋势等）

### A.2 为什么 ARIMA 支持多种估计方法？

**原因**：
1. **教学目的**：展示不同的时间序列估计方法
2. **历史原因**：状态空间方法是相对较新的
3. **特殊场景**：某些方法在特定情况下更高效（如 Burg 对于短序列 AR 估计）

**推荐**：实际应用中通常使用默认的 `method='statespace'`，因为它：
- 支持完整的 SARIMAX 规格
- 内置处理差分
- 支持 `exog`
- 支持缺失值

### A.3 ARIMA vs SARIMAX 选择指南

| 场景 | 推荐使用 | 原因 |
|-----|---------|------|
| 快速上手，默认配置 | **ARIMA** | 自动设置趋势默认值，更易用 |
| 需要传统估计方法 | **ARIMA** | 支持 6 种估计方法 |
| 需要状态空间高级选项 | **SARIMAX** | 支持 `measurement_error`, `time_varying_regression`, `hamilton_representation` 等 |
| 需要简单差分 | **SARIMAX** | 支持 `simple_differencing=True` |
| 需要控制趋势纳入方式 | **都可以** | ARIMA 用回归方式，SARIMAX 用状态空间方式 |

**注意**：
- `statsmodels.tsa.arima.model.ARIMA` 是 **推荐使用** 的新版 API
- 旧版 `statsmodels.tsa.arima_model.ARIMA` 已弃用
- `statsmodels.tsa.statespace.sarimax.SARIMAX` 也是稳定的 API
