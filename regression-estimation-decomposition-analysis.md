# Statsmodels 回归估计数值分解策略分析报告

## 1. 整体架构

Statsmodels 的回归估计系统采用了**分层继承 + 统一接口**的设计模式，确保不同的估计方法能够在相同的框架下协同工作。

### 1.1 类继承体系

```
base.LikelihoodModel
    └── RegressionModel (基类，定义统一接口)
            ├── GLS (广义最小二乘)
            │       └── GLSAR (带自回归误差的 GLS)
            └── WLS (加权最小二乘)
                    └── OLS (普通最小二乘)
```

## 2. 数值分解策略

Statsmodels 的线性回归支持两种主要的数值分解策略：

### 2.1 伪逆法 (Pseudoinverse)

**实现位置**：`statsmodels/regression/linear_model.py:349-365`

**核心原理**：
```python
if method == "pinv":
    if not (hasattr(self, "pinv_wexog") and ...):
        # 使用 SVD 计算 Moore-Penrose 伪逆
        self.pinv_wexog, singular_values = pinv_extended(self.wexog)
        # 归一化协方差参数 = pinv(X) @ pinv(X).T
        self.normalized_cov_params = np.dot(
            self.pinv_wexog, np.transpose(self.pinv_wexog)
        )
        # 计算矩阵秩
        self.rank = np.linalg.matrix_rank(np.diag(singular_values))
    
    # 计算回归系数: β = pinv(X) @ y
    beta = np.dot(self.pinv_wexog, self.wendog)
```

**伪逆计算详情**：`statsmodels/tools/tools.py:244-265`

```python
def pinv_extended(x, rcond=1e-15):
    x = np.asarray(x)
    x = x.conjugate()
    # SVD 分解: X = U @ diag(S) @ Vt
    u, s, vt = np.linalg.svd(x, False)
    s_orig = np.copy(s)
    
    # 处理小奇异值（数值稳定性）
    cutoff = rcond * np.maximum.reduce(s)
    for i in range(min(n, m)):
        if s[i] > cutoff:
            s[i] = 1./s[i]
        else:
            s[i] = 0.
    
    # 伪逆 = Vt.T @ diag(1/S) @ U.T
    res = np.dot(np.transpose(vt), np.multiply(s[:, np.newaxis],
                                               np.transpose(u)))
    return res, s_orig
```

### 2.2 QR 分解法 (QR Factorization)

**实现位置**：`statsmodels/regression/linear_model.py:367-388`

**核心原理**：
```python
elif method == "qr":
    if not (hasattr(self, "exog_Q") and hasattr(self, "exog_R") and ...):
        # QR 分解: X = Q @ R
        Q, R = np.linalg.qr(self.wexog)
        self.exog_Q, self.exog_R = Q, R
        
        # 归一化协方差参数 = inv(R.T @ R)
        self.normalized_cov_params = np.linalg.inv(np.dot(R.T, R))
        
        # 从 R 计算奇异值
        self.wexog_singular_values = np.linalg.svd(R, 0, 0)
        self.rank = np.linalg.matrix_rank(R)
    else:
        Q, R = self.exog_Q, self.exog_R
    
    # 仍然需要 pinv 用于某些协方差估计器
    self.pinv_wexog = np.linalg.pinv(self.wexog)
    
    # 计算回归系数:
    # effects = Q.T @ y (投影后的响应)
    self.effects = effects = np.dot(Q.T, self.wendog)
    # 解三角系统: R @ β = effects
    beta = np.linalg.solve(R, effects)
```

## 3. 统一接口机制

### 3.1 fit 方法的统一入口

**实现位置**：`statsmodels/regression/linear_model.py:284-415`

所有回归模型共享同一个 `fit()` 方法，通过 `method` 参数选择不同的数值分解策略：

```python
def fit(
    self,
    method: Literal["pinv", "qr"] = "pinv",  # 数值分解方法
    cov_type: Literal["nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC", ...] = "nonrobust",
    cov_kwds=None,
    use_t: bool | None = None,
    **kwargs,
) -> RegressionResults:
    # ... 矩阵计算逻辑 ...
    
    # 统一创建结果对象
    if isinstance(self, OLS):
        lfit = OLSResults(
            self, beta,
            normalized_cov_params=self.normalized_cov_params,
            cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t
        )
    else:
        lfit = RegressionResults(
            self, beta,
            normalized_cov_params=self.normalized_cov_params,
            cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t,
            **kwargs,
        )
    return RegressionResultsWrapper(lfit)
```

### 3.2 数据预处理的统一流程

无论使用哪种分解方法，数据都经过相同的预处理流程：

**初始化流程**：`statsmodels/regression/linear_model.py:225-234`

```python
def initialize(self):
    """Initialize model components."""
    # 1. 白化（whiten）设计矩阵 - 各模型自行实现
    self.wexog = self.whiten(self.exog)
    self.wendog = self.whiten(self.endog)
    
    # 2. 统一计算样本量
    self.nobs = float(self.wexog.shape[0])
    
    # 3. 初始化自由度和秩
    self._df_model = None
    self._df_resid = None
    self.rank = None
```

**白化方法的差异**：

| 模型 | whiten 方法实现 | 目的 |
|------|----------------|------|
| **OLS** | 直接返回输入 (`return x`) | 无需变换 |
| **WLS** | 乘以 `sqrt(weights)` | 处理异方差 |
| **GLS** | 乘以 `cholsigmainv` (Cholesky 逆) | 处理一般协方差结构 |
| **GLSAR** | 减去自回归项，丢弃前 p 个观测 | 处理自相关误差 |

## 4. 结果对象的统一生成

### 4.1 结果类继承体系

```
base.LikelihoodModelResults
    └── RegressionResults
            └── OLSResults (OLS 专用，增加特定功能)
```

### 4.2 结果对象初始化

**实现位置**：`statsmodels/regression/linear_model.py:1659-1757`

```python
class RegressionResults(base.LikelihoodModelResults):
    def __init__(
        self,
        model,                    # 模型实例
        params,                   # 估计的系数 β
        normalized_cov_params,   # 归一化协方差矩阵
        scale=1.0,               # 残差尺度
        cov_type="nonrobust",    # 协方差类型
        cov_kwds=None,           # 协方差额外参数
        use_t=None,              # 是否使用 t 分布
        **kwargs,
    ):
        super().__init__(model, params, normalized_cov_params, scale)
        
        # 继承模型的自由度
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        
        # 处理协方差类型
        if cov_type == "nonrobust":
            self.cov_type = "nonrobust"
            self.cov_kwds = {"description": "Standard Errors assume that..."}
            self.use_t = True if use_t is None else use_t
        else:
            # 调用稳健协方差估计方法
            self.get_robustcov_results(
                cov_type=cov_type, use_self=True, use_t=use_t, **cov_kwds
            )
```

### 4.3 协方差矩阵的统一处理

无论使用哪种分解方法，协方差矩阵的计算遵循统一的逻辑：

**基本公式**：
```
cov(β) = scale * normalized_cov_params

其中：
- scale = SSR / df_resid (残差均方)
- normalized_cov_params:
  * pinv 方法: pinv(X) @ pinv(X).T
  * QR 方法: inv(R.T @ R)
```

**稳健协方差的统一接口**：`get_robustcov_results()` 方法支持多种协方差类型：

| cov_type | 方法 | 适用场景 |
|----------|------|----------|
| `nonrobust` | 标准 OLS 协方差 | 同方差、无自相关 |
| `HC0-HC3` | 异方差稳健 (White 等) | 存在异方差 |
| `HAC` | 异方差-自相关稳健 (Newey-West) | 存在自相关 |
| `cluster` | 聚类稳健标准误 | 面板数据/聚类结构 |
| `hac-panel` | 面板 HAC | 面板数据 |
| `hac-groupsum` | Driscoll-Kraay | 面板数据 |

## 5. 两种分解策略的对比

### 5.1 数学等价性

从数学上讲，两种方法应该得到相同的结果：

**PINV 方法**：
```
β_pinv = pinv(X) @ y
      = (X.T @ X)^+ @ X.T @ y  (当 X 满秩时)
```

**QR 方法**：
```
X = Q @ R  (Q 正交, R 上三角)
β_qr = solve(R, Q.T @ y)

由于 Q 正交:
X.T @ X = R.T @ Q.T @ Q @ R = R.T @ R
X.T @ y = R.T @ Q.T @ y

所以:
β_qr = solve(R, Q.T @ y) = inv(R.T @ R) @ X.T @ y = (X.T @ X)^-1 @ X.T @ y
```

### 5.2 数值稳定性对比

| 特性 | PINV (SVD 基) | QR 分解 |
|------|---------------|---------|
| **对奇异值的处理** | 显式截断小奇异值，更稳定 | 依赖 `np.linalg.solve` 的精度 |
| **秩检测** | 基于奇异值，更准确 | 基于 R 的对角元 |
| **计算复杂度** | O(n×p²) | O(n×p²) |
| **内存使用** | 需要存储 U, S, Vt | 需要存储 Q, R |
| **适用场景** | 可能存在多重共线性 | 满秩或接近满秩矩阵 |

### 5.3 代码中的差异处理

```python
# QR 方法的额外处理
if method == "qr":
    # ... QR 分解 ...
    
    # 关键差异 1: 仍然计算 pinv 用于后续协方差估计
    # 见 GH #8157 - 某些协方差估计器需要 pinv
    self.pinv_wexog = np.linalg.pinv(self.wexog)
    
    # 关键差异 2: 存储 effects 用于 ANOVA
    self.effects = effects = np.dot(Q.T, self.wendog)
```

## 6. 完整的估计流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                     用户调用: model.fit(method="pinv/qr")        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段 1: 模型初始化 (所有模型共享)                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. 调用 model.initialize()                               │   │
│  │    - wexog = whiten(exog)  [各模型自定义]               │   │
│  │    - wendog = whiten(endog) [各模型自定义]              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段 2: 数值分解 (根据 method 分支)                              │
│                                                                  │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │ method == "pinv" │         │  method == "qr"  │             │
│  └──────────────────┘         └──────────────────┘             │
│           │                              │                       │
│           ▼                              ▼                       │
│  ┌─────────────────────┐    ┌─────────────────────┐             │
│  │ pinv_extended(wexog)│    │ np.linalg.qr(wexog)│             │
│  │ - SVD 分解          │    │ - QR 分解           │             │
│  │ - 截断小奇异值      │    │ - Q 正交, R 上三角  │             │
│  │ - 返回 pinv + 奇异值│    │ - 从 R 计算奇异值  │             │
│  └─────────────────────┘    └─────────────────────┘             │
│           │                              │                       │
│           ▼                              ▼                       │
│  ┌─────────────────────┐    ┌─────────────────────┐             │
│  │ β = pinv @ wendog   │    │ effects = Q.T @ y   │             │
│  │                     │    │ β = solve(R, effects)│             │
│  └─────────────────────┘    └─────────────────────┘             │
│           │                              │                       │
│           └──────────────┬───────────────┘                       │
│                          ▼                                           │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ 统一输出:                                                 │     │
│  │ - β: 回归系数                                             │     │
│  │ - normalized_cov_params: 归一化协方差                     │     │
│  │ - rank: 矩阵秩                                            │     │
│  │ - wexog_singular_values: 奇异值 (用于诊断)               │     │
│  └─────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段 3: 结果对象创建 (统一接口)                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. 计算自由度: df_model, df_resid                        │   │
│  │ 2. 创建结果对象:                                          │   │
│  │    - OLS → OLSResults                                    │   │
│  │    - 其他 → RegressionResults                            │   │
│  │ 3. 处理协方差类型:                                        │   │
│  │    - nonrobust: 使用 normalized_cov_params              │   │
│  │    - 其他: 调用 get_robustcov_results()                  │   │
│  │ 4. 包装结果: RegressionResultsWrapper                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     返回: RegressionResultsWrapper                │
│  包含:                                                             │
│  - params: 系数估计                                                │
│  - bse: 标准误                                                     │
│  - tvalues/pvalues: 推断统计                                       │
│  - rsquared/rsquared_adj: 拟合优度                                │
│  - resid/fittedvalues: 残差/拟合值                                │
│  - summary(): 汇总输出                                             │
└─────────────────────────────────────────────────────────────────┘
```

## 7. 关键设计模式总结

### 7.1 模板方法模式 (Template Method)

`RegressionModel.fit()` 定义了算法的骨架，将具体的数值分解步骤延迟到方法参数控制：

```python
def fit(self, method="pinv", ...):
    # 固定步骤 1: 检查缓存
    # 固定步骤 2: 根据 method 选择分解策略
    # 固定步骤 3: 计算 β
    # 固定步骤 4: 创建结果对象
```

### 7.2 策略模式 (Strategy Pattern)

通过 `method` 参数在运行时选择不同的数值分解策略：

- **策略接口**：产生相同的输出 (`β`, `normalized_cov_params`, `rank`)
- **具体策略**：`pinv` 和 `qr` 两种实现
- **上下文**：`RegressionModel.fit()` 方法

### 7.3 工厂方法模式 (Factory Method)

结果对象的创建使用了工厂方法思想：

```python
if isinstance(self, OLS):
    lfit = OLSResults(...)  # OLS 专用结果
else:
    lfit = RegressionResults(...)  # 通用结果
```

### 7.4 装饰器模式 (Decorator Pattern)

`RegressionResultsWrapper` 装饰原始结果对象，提供更友好的 API：

```python
return RegressionResultsWrapper(lfit)
```

## 8. 代码参考位置

| 功能 | 文件位置 | 行号 |
|------|----------|------|
| 基类定义 | `linear_model.py` | 213-282 |
| fit 方法 | `linear_model.py` | 284-415 |
| PINV 实现 | `linear_model.py` | 349-365 |
| QR 实现 | `linear_model.py` | 367-388 |
| pinv_extended | `tools.py` | 244-265 |
| RegressionResults | `linear_model.py` | 1659-2806 |
| 稳健协方差 | `linear_model.py` | 2497-2806 |

## 9. 总结

Statsmodels 的回归估计系统通过精心设计的接口实现了数值分解策略的统一性：

1. **统一入口**：所有模型共享同一个 `fit()` 方法，通过 `method` 参数选择分解策略
2. **统一预处理**：`whiten()` 方法处理数据变换，`initialize()` 方法标准化初始化流程
3. **统一输出**：无论使用哪种分解方法，都产生相同的中间结果（`β`, `normalized_cov_params`, `rank`）
4. **统一结果**：`RegressionResults` 类封装所有后处理逻辑，提供一致的推断 API

这种设计使得：
- **用户**：无需关心底层实现细节，只需选择 `method` 参数
- **开发者**：新增分解策略只需修改 `fit()` 方法的对应分支，不影响结果接口
- **可维护性**：数值计算与统计推断逻辑分离，便于独立优化
