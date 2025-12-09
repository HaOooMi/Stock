# Baseline 模型训练管道

## 📋 概述

`run_baseline_pipeline.py` 是 Learning-to-Rank 实验的主流程，实现三条线对比：
- **Baseline A**：回归原始收益（LGBMRegressor）
- **Baseline B**：Reg-on-Rank（LGBMRegressor + GaussRank 标签）
- **Baseline C (Sorting)**：LambdaRank（LGBMRanker）

## 🔗 模块集成关系

```
run_baseline_pipeline.py (主流程)
├── data.DataLoader              # 数据加载
│   └── load_features_and_targets()
├── data.TimeSeriesCV            # 时序切分（Purged + Embargo）
│   └── single_split()
├── evaluation.DriftDetector     # 漂移检测（Train vs Valid vs Test）
│   └── calculate_psi()
├── targets.RankingLabelFactory  # 标签构造
│   └── create_labels()
├── models.LightGBMModel         # 回归模型（Baseline A/B）
├── models.LightGBMRanker        # 排序模型（Baseline C）
└── evaluation.CrossSectionAnalyzer  # 横截面评估（IC/ICIR/Spread）
```

## 🎯 使用方法

```bash
# 运行默认任务（从配置读取）
python run_baseline_pipeline.py

# 指定单个任务类型
python run_baseline_pipeline.py --task_type regression
python run_baseline_pipeline.py --task_type regression_rank
python run_baseline_pipeline.py --task_type lambdarank

# 三条线对比（推荐）
python run_baseline_pipeline.py --compare_all

# 跳过漂移检测
python run_baseline_pipeline.py --compare_all --skip_drift

# 使用自定义配置
python run_baseline_pipeline.py --config configs/my_config.yml
```

## 📊 流程步骤

### 步骤 1: 数据加载

```python
features, forward_returns, _ = prepare_data(config)
```

使用 `DataLoader.load_features_and_targets()` 加载：
- 特征数据（MultiIndex [date, ticker]）
- 远期收益（作为评估基准）

### 步骤 2: 时序 CV 切分

```python
cv = TimeSeriesCV.from_config(config)
train_idx, valid_idx, test_idx = cv.single_split(features)
```

应用 Purged + Embargo 切分：
- **Purge Gap**：训练集与验证/测试集之间的间隔（避免标签泄漏）
- **Embargo**：验证/测试集之后的隔离期

### 步骤 3: 漂移检测

```python
run_drift_detection(features, train_idx, valid_idx, test_idx, output_dir)
```

使用 PSI (Population Stability Index) 检测特征分布漂移：
- Train vs Valid
- Train vs Test
- 输出漂移特征列表

### 步骤 4: 标签构造

```python
label_factory = RankingLabelFactory(n_bins=5, rank_method='zscore')
result = label_factory.create_labels(forward_returns, task_type, target_col)
labels = result['labels']
groups = result['groups']  # LambdaRank 需要
```

三种任务类型对应的标签：

| 任务类型 | 标签 | 模型 |
|---------|------|------|
| `regression` | 原始收益 | LightGBMModel |
| `regression_rank` | GaussRank（连续） | LightGBMModel |
| `lambdarank` | 分箱等级（0~4） | LightGBMRanker |

### 步骤 5: 模型训练

```python
# 回归任务
model = LightGBMModel(params=config['models']['lightgbm']['params'])
model.fit(X_train, y_train, X_valid, y_valid)

# 排序任务
model = LightGBMRanker(params=config['models']['lightgbm_ranker']['params'])
model.fit(X_train, y_train, X_valid, y_valid, 
          groups=train_groups, valid_groups=valid_groups)
```

### 步骤 6: 横截面评估

```python
analyzer = CrossSectionAnalyzer(
    factors=predictions.to_frame('model_score'),
    forward_returns=test_forward_returns
)
analyzer.analyze()
results = analyzer.get_results()
```

评估指标：
- **Rank IC / ICIR**：预测分数与实际收益的秩相关
- **Top-Mean Spread**：头部股票超额收益
- **IC 正比例**：正 IC 天数占比

### 步骤 7: 结果对比

```python
compare_results(all_results, output_dir)
```

输出对比表格：
```
任务类型                    Mean IC      ICIR        ICIR(年化)    Spread
--------------------------------------------------------------------------------
regression                 0.0674       0.3878       6.1569       0.0046
regression_rank            0.0543       0.3772       5.9878       0.0017
lambdarank                 0.0316       0.2101       3.3356       0.0035
--------------------------------------------------------------------------------

📈 相对回归基线的提升:
  regression_rank: IC 提升 -19.4%, ICIR 提升 -2.7%
  lambdarank: IC 提升 -53.0%, ICIR 提升 -45.8%
```

## 📁 输出文件

```
ML output/reports/baseline_v1/ranking/
├── drift_report.json              # 漂移检测报告
├── regression_results.json        # Baseline A 结果
├── regression_rank_results.json   # Baseline B 结果
├── lambdarank_results.json        # Baseline C 结果
├── model_comparison.json          # 三条线对比汇总
├── regression_predictions.parquet # Baseline A 预测
├── regression_rank_predictions.parquet
├── lambdarank_predictions.parquet
├── regression_model.pkl           # Baseline A 模型
├── regression_rank_model.pkl
└── lambdarank_model.pkl
```

## ⚙️ 配置示例

`configs/ml_baseline.yml` 相关配置：

```yaml
# 数据配置
data:
  symbol: ["000001", "000002", "000063", ...]
  start_date: "2018-01-01"
  end_date: "2024-12-31"

# 目标配置
target:
  forward_periods: 5

# 时序切分配置
split:
  train_ratio: 0.7
  valid_ratio: 0.15
  test_ratio: 0.15
  purge_days: 5
  embargo_days: 5
  drift_threshold: 0.2

# 排序配置
ranking:
  task_type: regression
  regression_rank:
    rank_method: zscore
    min_samples_per_day: 30
  lambdarank:
    n_bins: 5

# 模型配置
models:
  lightgbm:
    params:
      objective: regression
      n_estimators: 500
      learning_rate: 0.05
      num_leaves: 31
      max_depth: 8
  
  lightgbm_ranker:
    params:
      objective: lambdarank
      metric: ndcg
      ndcg_eval_at: [10, 30, 50]
      n_estimators: 500
      learning_rate: 0.05
      num_leaves: 31
      max_depth: 6
      min_data_in_leaf: 50
```

## 🔬 实验设计说明

### 为什么对比三条线？

| 任务 | 优化目标 | 假设 |
|------|---------|------|
| Baseline A | MSE(y_true, y_pred) | 收益率绝对值可预测 |
| Baseline B | MSE(rank_true, rank_pred) | 相对排序比绝对值更稳定 |
| Baseline C | NDCG | 只关心头部排序质量 |

### 预期结论

- 如果 **B > A**：说明排序标签比原始收益更稳定
- 如果 **C > B**：说明 LambdaRank 的 pairwise 优化有优势
- 如果 **A ≈ B ≈ C**：说明当前因子预测能力有限，模型选择不敏感

## 📝 注意事项

1. **数据要求**：必须有足够的历史数据（建议 ≥3 年）
2. **样本量**：每日至少 30 只股票（`min_samples_per_day`）
3. **LambdaRank**：训练数据必须按日期排序
4. **漂移检测**：PSI > 0.2 表示显著漂移，需警惕
