# Targets 模块

## 📋 概述

`targets/` 模块负责目标变量（标签）的生成与转换，支持从原始价格数据到机器学习标签的完整流程。

## 📁 文件结构

```
targets/
├── __init__.py              # 模块导出
├── target_engineering.py    # 目标变量生成（未来收益率）
├── label_transformer.py     # 标签转换（残差收益、行业中性化）
├── ranking_labels.py        # 排序标签构造（Reg-on-Rank, LambdaRank）
└── README.md                # 本文档
```

## 🔗 模块职责划分

```
价格数据 (close)
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  TargetEngineer (target_engineering.py)                         │
│  生成未来收益率: future_return_1d, future_return_5d, ...        │
│  支持: 单标的 / 多标的 (MultiIndex)                              │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  LabelTransformer (label_transformer.py)                        │
│  金融层面转换:                                                   │
│  - 残差收益（vs 指数/行业）                                      │
│  - 排名标准化                                                    │
│  - 分位数标签                                                    │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  RankingLabelFactory (ranking_labels.py)                        │
│  ML训练标签:                                                     │
│  - regression: 原始收益（直通）                                  │
│  - regression_rank: GaussRank / ZScore（连续值）                 │
│  - lambdarank: 分箱等级（离散） + groups 向量                    │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 模块说明

### 1. TargetEngineer (`target_engineering.py`)

**目标变量生成器**，基于收盘价生成未来收益率。

```python
from targets import TargetEngineer

engineer = TargetEngineer(data_dir='ML output')

# 生成未来收益率
result = engineer.generate_future_returns(
    data=market_data,          # DataFrame，含 'close' 列
    periods=[1, 5, 10, 20],    # 未来 N 天收益
    price_col='close'
)
# 新增列: future_return_1d, future_return_5d, ...
```

**特点**：
- 支持单标的（DatetimeIndex）和多标的（MultiIndex [date, ticker]）
- 自动处理尾部 NaN（未来数据不可知）
- 防止时间序列数据泄漏

**生成的列**：
| 列名 | 计算公式 |
|------|---------|
| `future_return_1d` | (price[t+1] - price[t]) / price[t] |
| `future_return_5d` | (price[t+5] - price[t]) / price[t] |
| `future_return_10d` | (price[t+10] - price[t]) / price[t] |

### 2. LabelTransformer (`label_transformer.py`)

**标签转换器**，用于金融层面的收益调整。

```python
from targets import LabelTransformer

transformer = LabelTransformer()

# 1. 相对指数的残差收益
residual_returns = transformer.residualize_vs_index(
    returns=forward_returns,       # MultiIndex [date, ticker]
    index_returns=index_returns,   # 指数收益
    method='ols'                   # 'ols' 或 'demean'
)

# 2. 相对行业的残差收益
industry_residual = transformer.residualize_vs_industry(
    returns=forward_returns,
    industry_map=industry_map,     # ticker → industry 映射
    method='demean'
)
```

**残差收益公式**：
```
r_residual = r_stock - β * r_benchmark

其中 β 通过日内横截面回归估计：
r_stock ~ α + β * r_benchmark + ε
```

**适用场景**：
- 行业中性化策略
- 对冲指数收益后的超额收益预测

### 3. RankingLabelFactory (`ranking_labels.py`)

**排序标签工厂**，为 Learning-to-Rank 模型构造训练标签。

```python
from targets import RankingLabelFactory

factory = RankingLabelFactory(
    n_bins=5,              # LambdaRank 分箱数
    rank_method='zscore'   # 'zscore' / 'gauss' / 'uniform'
)

# 创建标签（三种任务类型）
result = factory.create_labels(
    forward_returns=forward_returns,   # MultiIndex [date, ticker]
    task_type='lambdarank',            # 'regression' / 'regression_rank' / 'lambdarank'
    target_col='ret_5d',
    min_samples=30
)

labels = result['labels']   # 标签 Series
groups = result['groups']   # LambdaRank 需要的 group 向量
```

**三种任务类型**：

| 任务类型 | 标签类型 | 说明 |
|---------|---------|------|
| `regression` | 连续值 | 原始收益，直通不转换 |
| `regression_rank` | 连续值 | 横截面 GaussRank/ZScore |
| `lambdarank` | 离散整数 (0~n_bins-1) | 分箱等级 + groups |

**GaussRank 公式**：
```python
# 1. 计算横截面排序百分位
rank_pct = (rank - 1) / (N - 1)  # [0, 1]

# 2. 裁剪到 (0, 1) 避免无穷大
rank_pct_clipped = clip(rank_pct, 1e-6, 1-1e-6)

# 3. 逆正态变换
gauss_rank = sqrt(2) * erfinv(2 * rank_pct_clipped - 1)
```

**辅助方法**：

```python
# 对齐特征与标签（去除 NaN）
X_aligned, y_aligned = factory.align_features_with_labels(features, labels)
```

## 🎯 使用场景

| 模块 | 使用场景 | 下游模块 |
|------|---------|---------|
| TargetEngineer | 数据准备阶段 | prepare_data.py |
| LabelTransformer | 因子评估（残差收益） | CrossSectionAnalyzer |
| RankingLabelFactory | 模型训练（三条线对比） | run_baseline_pipeline.py |

## 📊 与 Pipeline 的集成

```
prepare_data.py
└── TargetEngineer.generate_future_returns()
        ↓
    future_return_5d (原始收益)
        ↓
run_baseline_pipeline.py
└── RankingLabelFactory.create_labels()
    ├── task_type='regression'      → 原始收益
    ├── task_type='regression_rank' → GaussRank 标签
    └── task_type='lambdarank'      → 分箱等级 + groups
```

## 📝 配置示例

`configs/ml_baseline.yml` 中的相关配置：

```yaml
target:
  forward_periods: 5          # 未来 N 天收益
  return_type: simple         # 'simple' 或 'log'
  transform: none             # 'none', 'residual_index', 'residual_industry'

ranking:
  task_type: regression       # 默认任务类型
  regression_rank:
    rank_method: zscore       # 'zscore', 'gauss', 'uniform'
    min_samples_per_day: 30
  lambdarank:
    n_bins: 5                 # 分箱数
```
