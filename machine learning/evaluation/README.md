# Evaluation 模块

## 📋 概述

`evaluation/` 模块是因子与模型评估的核心框架，提供两套评估体系：
1. **横截面评估框架**（Alphalens 风格）：因子 IC、分位数收益、Spread、单调性等
2. **传统评估框架**：MSE、MAE、分桶分析、报告生成

## 📁 文件结构

```
evaluation/
├── __init__.py                  # 模块导出
├── README.md                    # 本文档
│
│ ══════════════════════════════════════════════════════════
│  横截面评估框架（Alphalens 风格）⭐核心
│ ══════════════════════════════════════════════════════════
├── cross_section_analyzer.py    # 分析器主类（统一入口）
├── cross_section_metrics.py     # 核心度量计算（IC/ICIR/Spread）
├── factor_preprocessing.py      # 因子预处理（Winsorize/标准化/中性化）
├── visualization.py             # 7种可视化图表
├── tearsheet.py                 # HTML报告 + CSV导出
├── drift_detector.py            # 漂移检测（PSI/KS）
├── cross_section_adapter.py     # 适配器（对接 DataLoader）
│
│ ══════════════════════════════════════════════════════════
│  传统评估框架
│ ══════════════════════════════════════════════════════════
├── metrics.py                   # 传统指标（MSE/MAE/IC）
├── bucketing.py                 # 分桶分析
├── reporting.py                 # 报告生成
│
│ ══════════════════════════════════════════════════════════
│  聚类评估
│ ══════════════════════════════════════════════════════════
└── cluster/
    ├── __init__.py
    └── cluster_evaluate.py      # KMeans 聚类收益评估
```

## 🔗 模块依赖关系

```
CrossSectionAnalyzer (统一入口)
├── cross_section_metrics       # IC/ICIR/Spread/单调性/换手率
├── factor_preprocessing        # Winsorize/标准化/中性化
├── visualization               # 7种图表
└── tearsheet                   # HTML报告

DriftDetector (漂移检测)
└── CrossSectionAnalyzer        # 复用分析逻辑

CrossSectionAdapter (适配器)
├── DataLoader                  # 数据加载
├── MarketDataLoader            # 市场数据
└── CrossSectionAnalyzer        # 评估分析
```

---

## 📦 横截面评估框架（核心）

### 1. CrossSectionAnalyzer (`cross_section_analyzer.py`)

**因子评估统一入口**，封装所有横截面评估逻辑。

```python
from evaluation import CrossSectionAnalyzer

analyzer = CrossSectionAnalyzer(
    factors=factors_df,              # MultiIndex [date, ticker]
    forward_returns=forward_returns, # 或提供 prices 自动计算
    prices=prices_df,                # 可选
    tradable_mask=tradable_mask,     # 可选，过滤不可交易样本
    market_cap=market_cap,           # 可选，用于中性化
    industry=industry                # 可选，用于中性化
)

# 运行分析
analyzer.analyze(
    periods=[5, 10, 20],             # 收益期
    n_quantiles=5,                   # 分位数
    preprocess=True,                 # 是否预处理
    winsorize_quantile=0.01,         # Winsorize 分位数
    check_quality=True               # 深度质量检查（IC衰减/PSI/KS）
)

# 获取结果
results = analyzer.get_results()
```

**输出结果包含**：
| 键 | 类型 | 说明 |
|---|------|------|
| `daily_ic` | DataFrame | 每日 IC 序列 |
| `ic_summary` | Dict | IC 统计摘要（mean/std/ICIR/t_stat） |
| `quantile_returns` | Dict | 分位数组合收益 |
| `cumulative_returns` | Dict | 累计净值 |
| `spreads` | Dict | Top-Mean / Top-Bottom Spread |
| `monotonicity` | Dict | 单调性指标（Kendall τ） |
| `turnover_stats` | Dict | 换手率统计 |
| `quality_checks` | Dict | 深度质量检查（IC衰减/PSI/KS） |

### 2. cross_section_metrics (`cross_section_metrics.py`)

**核心度量计算**，使用 Numba JIT 加速。

```python
from evaluation import (
    calculate_forward_returns,
    calculate_daily_ic,
    calculate_ic_summary,
    calculate_quantile_returns,
    calculate_spread,
    calculate_monotonicity,
    calculate_turnover
)

# 计算远期收益
forward_returns = calculate_forward_returns(
    prices=prices_df,
    periods=[1, 5, 10, 20],
    method='simple'  # 'simple' 或 'log'
)

# 计算每日 IC（Spearman）
daily_ic = calculate_daily_ic(
    factors=factors_df,
    forward_returns=forward_returns,
    method='spearman'
)

# IC 汇总统计
ic_summary = calculate_ic_summary(ic_series)
# 返回: mean, std, icir, icir_annual, t_stat, positive_ratio

# 分位数收益
quantile_returns = calculate_quantile_returns(
    factors=factors_df,
    forward_returns=forward_returns,
    n_quantiles=5
)

# Spread 计算
spread = calculate_spread(
    quantile_returns=quantile_returns,
    method='top_bottom'  # 'top_bottom' 或 'top_mean'
)

# 单调性检验
monotonicity = calculate_monotonicity(quantile_mean_returns)
# 返回 Kendall τ

# 换手率
turnover = calculate_turnover(
    factors=factors_df,
    n_quantiles=5
)
```

**性能优化**：
- 使用 `@numba.jit` 加速 Spearman 相关计算
- 向量化操作，避免 `groupby().apply()` 开销
- 支持 Numba 不可用时的 fallback

### 3. factor_preprocessing (`factor_preprocessing.py`)

**因子预处理管道**。

```python
from evaluation import (
    winsorize_factor,
    standardize_factor,
    neutralize_factor,
    preprocess_factor_pipeline
)

# Winsorize（极值处理）
factors_win = winsorize_factor(
    factors=factors_df,
    lower_quantile=0.01,
    upper_quantile=0.99,
    cross_section=True  # 按日横截面处理
)

# 标准化
factors_std = standardize_factor(
    factors=factors_df,
    method='z_score',    # 'z_score', 'min_max', 'rank'
    cross_section=True
)

# 中性化（行业/市值）
factors_neutral = neutralize_factor(
    factors=factors_df,
    market_cap=market_cap,  # 市值中性化
    industry=industry,      # 行业中性化
    method='ols'            # 'ols' 或 'demean'
)

# 完整管道
factors_processed = preprocess_factor_pipeline(
    factors=factors_df,
    winsorize=True,
    standardize=True,
    neutralize=True,
    market_cap=market_cap,
    industry=industry
)
```

### 4. visualization (`visualization.py`)

**7种可视化图表**。

```python
from evaluation import (
    plot_ic_time_series,
    plot_ic_distribution,
    plot_quantile_cumulative_returns,
    plot_quantile_mean_returns,
    plot_spread_cumulative_returns,
    plot_monthly_ic_heatmap,
    plot_turnover_time_series,
    create_factor_tearsheet_plots
)

# IC 时间序列（走廊图）
fig = plot_ic_time_series(
    ic_series=daily_ic['factor_name'],
    title='IC Time Series',
    save_path='ic_series.png'
)

# IC 分布直方图
fig = plot_ic_distribution(ic_series, save_path='ic_dist.png')

# 分位数累计收益
fig = plot_quantile_cumulative_returns(
    cumulative_returns=cum_ret_df,
    save_path='quantile_cumret.png'
)

# 分位数平均收益柱状图
fig = plot_quantile_mean_returns(
    quantile_mean=quantile_mean_df,
    save_path='quantile_meanret.png'
)

# Spread 累计收益
fig = plot_spread_cumulative_returns(
    spread_series=spread,
    save_path='spread_cumret.png'
)

# 月度 IC 热力图
fig = plot_monthly_ic_heatmap(
    ic_series=daily_ic,
    save_path='ic_heatmap.png'
)

# 一键生成所有图表
plot_paths = create_factor_tearsheet_plots(
    analyzer_results=results,
    factor_name='ROC_20',
    return_period='5d',
    output_dir='figures/'
)
```

### 5. tearsheet (`tearsheet.py`)

**HTML 报告 + CSV 导出**。

```python
from evaluation import (
    generate_html_tearsheet,
    generate_full_tearsheet,
    save_ic_to_csv,
    save_quantile_returns_to_csv
)

# 生成 HTML Tearsheet
generate_html_tearsheet(
    analyzer_results=results,
    factor_name='ROC_20',
    return_period='5d',
    output_path='tearsheet_ROC_20_5d.html',
    plot_paths=plot_paths  # 可选，嵌入图表
)

# 保存 IC 序列到 CSV
save_ic_to_csv(
    ic_series=daily_ic,
    output_path='ic_ROC_20_5d.csv'
)

# 保存分位数收益到 CSV
save_quantile_returns_to_csv(
    quantile_returns=quantile_ret,
    output_path='quantile_returns_ROC_20_5d.csv'
)
```

### 6. drift_detector (`drift_detector.py`)

**漂移检测（Train vs Valid vs Test）**。

```python
from evaluation import DriftDetector, compare_splits_with_analyzer

detector = DriftDetector(
    drift_threshold=0.2,       # 20% 差异阈值
    significance_level=0.05
)

# 比较 IC 汇总
comparison = detector.compare_ic_summaries(
    train_summary=train_ic_summary,
    valid_summary=valid_ic_summary,
    test_summary=test_ic_summary
)

# 计算 PSI（Population Stability Index）
psi = detector.calculate_psi(
    reference_data=train_features['factor'],
    current_data=test_features['factor']
)
# PSI < 0.1: 无漂移
# 0.1 ≤ PSI < 0.2: 轻微漂移
# PSI ≥ 0.2: 显著漂移

# KS 检验
ks_stat, p_value = detector.ks_test(
    reference_data=train_features,
    current_data=test_features
)

# 一键对比（与 CrossSectionAnalyzer 集成）
results = compare_splits_with_analyzer(
    factors=factors_df,
    forward_returns=forward_returns,
    train_idx=train_idx,
    valid_idx=valid_idx,
    test_idx=test_idx,
    output_dir='reports/cv/',
    drift_threshold=0.2
)
```

---

## 📦 传统评估框架

### 7. metrics (`metrics.py`)

**传统回归指标**。

```python
from evaluation import calculate_metrics, calculate_ic_by_date

# 计算评估指标
metrics = calculate_metrics(y_true, y_pred)
# 返回: mse, mae, rmse, r2, ic, rank_ic, n_samples

# 按日期计算 IC
daily_ic = calculate_ic_by_date(predictions_df)
# predictions_df 需要包含 y_true, y_pred 列
```

### 8. bucketing (`bucketing.py`)

**分桶分析**。

```python
from evaluation import bucket_predictions, analyze_bucket_performance

# 分桶
predictions_with_bucket = bucket_predictions(
    predictions_df=predictions_df,
    n_buckets=5,
    method='quantile',      # 'quantile' 或 'equal_width'
    cross_section=True      # 按日横截面分桶
)

# 分析分桶表现
bucket_perf = analyze_bucket_performance(predictions_with_bucket)
# 返回每个桶的平均收益、样本数等
```

### 9. reporting (`reporting.py`)

**报告生成**。

```python
from evaluation import generate_report

generate_report(
    results={
        'model_metrics': {...},
        'bucket_performance': bucket_perf_df,
        'predictions': predictions_df
    },
    output_dir='reports/evaluation/',
    bucket_performance_file='model_bucket_performance.csv',
    predictions_file='test_predictions.csv',
    summary_file='summary.json'
)
```

---

## 📦 聚类评估

### 10. cluster_evaluate (`cluster/cluster_evaluate.py`)

**KMeans 聚类收益评估**。

```python
from evaluation.cluster.cluster_evaluate import ClusterEvaluator

evaluator = ClusterEvaluator(reports_dir='ML output/reports')

# 加载数据
states_train, states_test, targets = evaluator.load_pca_states_and_targets(
    states_train_path='states/states_pca_train.npy',
    states_test_path='states/states_pca_test.npy',
    targets_path='datasets/with_targets.csv'
)

# 运行聚类分析（k=4,5,6）
evaluator.run_cluster_analysis(states_train, states_test, targets)

# 生成报告
evaluator.generate_report()

# 获取最佳聚类
best_cluster = evaluator.get_best_cluster()
```

---

## 🎯 使用场景

| 场景 | 推荐模块 |
|------|---------|
| 因子研究 | `CrossSectionAnalyzer` + `visualization` + `tearsheet` |
| 因子预处理 | `factor_preprocessing` |
| 模型评估（排序） | `CrossSectionAnalyzer`（评估预测分数的 IC） |
| 模型评估（传统） | `metrics` + `bucketing` + `reporting` |
| 漂移检测 | `DriftDetector` |
| 聚类策略 | `cluster/cluster_evaluate` |

---

## 📊 与 Pipeline 的集成

```
prepare_factors.py
└── CrossSectionAnalyzer.analyze()        # 因子评估
    ├── cross_section_metrics             # IC/ICIR/Spread
    ├── factor_preprocessing              # 预处理
    ├── visualization                     # 图表
    └── tearsheet                         # HTML报告

run_baseline_pipeline.py
├── DriftDetector.calculate_psi()         # 漂移检测
└── CrossSectionAnalyzer.analyze()        # 模型预测评估

run_cluster_analysis.py
└── ClusterEvaluator.run_cluster_analysis()  # 聚类评估
```

---

## 📁 输出文件

```
ML output/
├── reports/baseline_v1/
│   ├── factors/
│   │   ├── tearsheet_{factor}_5d.html    # HTML 因子报告
│   │   ├── ic_{factor}_5d.csv            # IC 时间序列
│   │   └── quantile_returns_{factor}_5d.csv
│   ├── evaluation/
│   │   ├── model_bucket_performance.csv  # 分桶表现
│   │   ├── test_predictions.csv          # 预测明细
│   │   └── summary.json                  # 评估摘要
│   ├── ranking/
│   │   └── drift_report.json             # 漂移检测报告
│   └── clustering/
│       ├── clustering_analysis_report.txt
│       └── cluster_comparison.csv
└── figures/baseline_v1/factors/{factor}/
    ├── ic_series_{factor}_5d.png         # IC 时间序列图
    ├── ic_dist_{factor}_5d.png           # IC 分布图
    ├── ic_heatmap_{factor}_5d.png        # 月度 IC 热力图
    ├── quantile_cumret_{factor}_5d.png   # 分位数累计收益
    ├── quantile_meanret_{factor}_5d.png  # 分位数平均收益
    └── spread_cumret_{factor}_5d.png     # Spread 累计收益
```

---

## 📝 配置参数

`configs/ml_baseline.yml` 相关配置：

```yaml
evaluation:
  # 横截面评估
  cross_section:
    periods: [5, 10, 20]        # 收益期
    n_quantiles: 5              # 分位数
    preprocess: true            # 是否预处理
    winsorize_quantile: 0.01    # Winsorize 分位数
  
  # 漂移检测
  drift:
    threshold: 0.2              # PSI 阈值
    significance_level: 0.05    # 统计显著性

  # 分桶分析
  bucketing:
    n_buckets: 5
    method: quantile            # 'quantile' 或 'equal_width'
```

---

## ⚡ 性能优化

1. **Numba JIT 加速**：`cross_section_metrics.py` 中的 Spearman 相关计算
2. **向量化操作**：避免 `groupby().apply()` 开销
3. **缓存机制**：`CrossSectionAnalyzer` 缓存计算结果
4. **按需计算**：`check_quality=False` 跳过深度检查
