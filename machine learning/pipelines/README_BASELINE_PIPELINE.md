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

### 步骤 7: 组合回测（Open-to-Open）

```python
backtester = SimplePortfolioBacktester(
    predictions=predictions,
    prices=prices,
    top_n=20,
    rebalance_freq='1M',
    execution_mode='open_to_open'  # T+1 开盘执行
)
backtest_results = backtester.run_backtest()
backtester.plot_results(save_dir=output_dir)
```

**回测设计**：
- **选股规则**：每期选 Top-N 预测分数最高的股票
- **权重方案**：等权配置（1/N）
- **调仓频率**：月度调仓（月初第一个交易日）
- **执行假设**：
  - T 日收盘生成信号
  - T+1 日开盘执行交易（符合 A 股 T+1 制度）
  - 持有至下次调仓

**绩效指标**：
- **年化收益率**：几何平均收益
- **波动率**：收益率标准差（年化）
- **夏普比率**：超额收益 / 波动率
- **最大回撤**：峰谷跌幅
- **Alpha / Beta**：相对基准的超额收益与系统风险
- **胜率**：正收益期数占比

**输出图表**：

*A/B 对比模式*（默认）：
- 净值曲线对比（Close-to-Close vs Open-to-Open）
- 回撤对比
- 关键指标对比柱状图（年化收益、夏普比率）
- Alpha 衰减分析饼图

*单模式*：
- 净值曲线（可选基准对比）
- 回撤曲线
- 换手率柱状图

### 步骤 8: 结果对比

```python
compare_results(all_results, output_dir)
```

**横截面评估对比**：
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

**回测绩效对比**：
```
策略                年化收益    波动率     夏普比率    最大回撤    换手率
--------------------------------------------------------------------------------
regression         12.5%      18.3%      0.68       -22.1%     25.3%
regression_rank    10.8%      17.9%      0.60       -24.3%     23.8%
lambdarank         9.2%       19.2%      0.48       -26.8%     28.1%
--------------------------------------------------------------------------------
注：以上为 Open-to-Open 模式的示例结果
```

## 📁 输出文件

```
ML output/reports/baseline_v1/ranking/
├── drift_report.json                    # 特征漂移检测报告
├── prediction_drift_report.json         # 模型预测漂移报告
├── regression_results.json              # Baseline A 横截面评估
├── regression_rank_results.json         # Baseline B 横截面评估
├── lambdarank_results.json              # Baseline C 横截面评估
├── model_comparison.json                # 三条线对比汇总
├── regression_backtest_results.json     # Baseline A 回测结果
├── regression_rank_backtest_results.json
├── lambdarank_backtest_results.json
├── regression_predictions.parquet       # Baseline A 预测
├── regression_rank_predictions.parquet
├── lambdarank_predictions.parquet
├── regression_model.pkl                 # Baseline A 模型
├── regression_rank_model.pkl
├── lambdarank_model.pkl
├── regression_backtest_comparison.png   # A/B 对比图表
├── regression_rank_backtest_comparison.png
└── lambdarank_backtest_comparison.png
```

## ⚙️ 配置示例

`configs/ml_baseline.yml` 相关配置：

```yaml
# 数据配置
data:
  symbol: ["000001", "000002", "000063", ...]
  start_date: "2018-01-01"
  end_date: "2024-12-31"
  influxdb:
    enabled: true
    url: "http://localhost:8086"
    org: "stock"
    bucket: "stock_kdata"
    token: "YOUR_TOKEN_HERE"

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

# 回测配置
backtest:
  top_n: 20                      # 选股数量
  rebalance_freq: '1M'           # 调仓频率（1D/1W/1M/3M）
  execution_mode: 'open_to_open' # 执行模式（open_to_open/close_to_close）
  transaction_cost: 0.001        # 单边交易成本（0.1%）
  benchmark: '000300'            # 基准指数（沪深300）

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

## 🎯 回测执行模式详解

### Open-to-Open vs Close-to-Close

**Open-to-Open（推荐）**：
```
T 日收盘 → 生成信号 → T+1 日开盘买入 → T+N 日开盘卖出
```
- ✅ 符合 A 股 T+1 制度
- ✅ 信号产生与执行有充足时间差
- ✅ 避免收盘价竞价博弈
- ❌ 隔夜风险敞口

**Close-to-Close（理论测试）**：
```
T 日收盘 → 生成信号 → T 日收盘买入 → T+N 日收盘卖出
```
- ⚠️ 不符合实际交易（T 日收盘无法根据 T 日信息交易）
- ⚠️ 仅用于理论对比或 A/B Testing

**为什么选择 Open-to-Open？**

1. **真实可执行**：T 日收盘后有充足时间计算信号，T+1 日开盘挂单
2. **避免前视偏差**：使用 T 日收盘数据预测 T+1 日开盘后的收益
3. **行业标准**：多数量化机构采用此模式

## 📝 注意事项

1. **数据要求**：必须有足够的历史数据（建议 ≥3 年）
2. **样本量**：每日至少 30 只股票（`min_samples_per_day`）
3. **LambdaRank**：训练数据必须按日期排序
4. **漂移检测**：PSI > 0.2 表示显著漂移，需警惕
5. **价格数据**：回测需要加载开盘价和收盘价数据（从 InfluxDB）
6. **交易成本**：默认 0.1% 单边成本，可根据实际调整
7. **停牌处理**：停牌股票自动剔除选股池

## 🐛 常见问题

**Q: 回测净值曲线为什么不平滑？**

A: 检查调仓频率设置，月度调仓会产生阶梯状净值曲线。如需平滑可改为周度或日度。

**Q: 为什么回测收益与 IC 不成正比？**

A: IC 衡量预测能力，但回测受交易成本、流动性、极端行情影响。高 IC 不一定转化为高收益。

**Q: Alpha 为负是什么原因？**

A: 可能是：
1. 选股能力不足（IC 过低）
2. 调仓过于频繁（交易成本吞噬收益）
3. 市场风格不匹配（因子失效期）

**Q: 如何提高回测稳健性？**

A: 
1. 增加样本量（更多股票、更长历史）
2. 控制换手率（降低调仓频率）
3. 风险中性化（行业中性、市值中性）
4. 多期验证（滚动回测）
