# Machine Learning 目录结构说明

## 📁 目录结构概览

```
machine learning/
├── configs/                    # 配置文件
│   └── ml_baseline.yml        # 基线训练配置
│
├── data/                       # 数据加载模块
│   ├── __init__.py
│   └── data_loader.py         # 统一数据加载器
│
├── features/                   # 特征工程模块
│   ├── __init__.py
│   └── feature_engineering.py # 特征生成、选择、标准化
│
├── targets/                    # 目标工程模块
│   ├── __init__.py
│   └── target_engineering.py  # 目标变量生成
│
├── models/                     # 模型模块
│   ├── __init__.py
│   ├── base_model.py          # 基础模型接口
│   ├── ridge_model.py         # Ridge回归
│   ├── rf_model.py            # 随机森林
│   ├── lgbm_model.py          # LightGBM
│   └── transformers/          # 变换器（降维等）
│       ├── __init__.py
│       └── pca.py             # PCA降维
│
├── evaluation/                 # 评估模块
│   ├── __init__.py
│   ├── metrics.py             # 评估指标计算
│   ├── bucketing.py           # 分桶分析
│   ├── reporting.py           # 报告生成
│   └── cluster/               # 聚类评估
│       ├── __init__.py
│       └── cluster_evaluate.py
│
├── backtest/                   # 回测模块
│   ├── __init__.py
│   ├── cluster_stratepy_backtest.py    #聚类信号策略回测
│   └── top_bucket_backtest.py # Top桶策略回测
│
├── utils/                      # 工具模块
│   ├── __init__.py
│   ├── splitting.py           # 时间序列切分
│   ├── logger.py              # 日志设置
│   ├── windowing.py           # 滑动窗口工具
│   └── triage.py              # 快速质检工具
│
├── pipelines/                  # 运行脚本
│   ├── prepare_data.py        # 完整数据准备流程
│   ├── run_cluster_analysis.py # 聚类分析
│   ├── run_pca_state.py       # PCA降维
│   └──train_models.py         #机器学习基线训练主脚本
│
├── scripts/                    # 独立脚本
│   └── (可选的独立工具脚本)
│
├── ML output/                  # 输出目录
│   ├── reports/               # 评估报告
│   ├── models/                # 训练好的模型
│   ├── states/                # PCA等状态文件
│   ├── figures/               # 可视化图表
│   └── cache/                 # 缓存文件
│
├── train_models.py            # 主训练脚本
├── TRAIN_MODELS_README.md     # 训练流程说明
└── README_STRUCTURE.md        # 本文档
```

## 🔄 文件移动记录

### 已完成的移动

| 原文件 | 新位置 | 说明 |
|--------|--------|------|
| `feature_engineering.py` | `features/feature_engineering.py` | 特征工程模块 |
| `target_engineering.py` | `targets/target_engineering.py` | 目标工程模块 |
| `pca_state.py` | `models/transformers/pca.py` | PCA降维工具 |
| `cluster_evaluate.py` | `evaluation/cluster/cluster_evaluate.py` | 聚类评估 |
| `strategy_backtest.py` | `backtest/top_bucket_backtest.py` | 回测模块 |
| `sliding_window.py` | `utils/windowing.py` | 滑动窗口工具 |
| `quick_triage.py` | `utils/triage.py` | 快速质检工具 |

## 🚀 使用流程

### 1. 完整的数据准备流程

```bash
# 运行完整的数据准备（特征工程 + 目标工程）
python pipelines/prepare_data.py --config configs/ml_baseline.yml
```

这会执行：
- 加载原始OHLCV数据
- 生成技术特征
- 特征选择和标准化
- 生成目标变量
- 保存完整数据集

输出文件：
- `ML output/scaler_{symbol}.pkl` - 特征标准化器
- `ML output/scaler_{symbol}_scaled_features.csv` - 标准化特征
- `ML output/with_targets_{symbol}_complete_{timestamp}.csv` - 完整数据集

### 2. 模型训练与评估

```bash
# 运行基线模型训练
python train_models.py --config configs/ml_baseline.yml
```

这会执行：
- 加载特征和目标数据
- 时间序列切分（train/valid/test）
- 训练Ridge、RandomForest、LightGBM
- 测试集预测
- 分桶分析
- 生成评估报告

输出文件：
- `ML output/reports/model_bucket_performance.csv` - 分桶表现
- `ML output/reports/test_predictions.csv` - 预测明细
- `ML output/reports/summary.json` - 评估摘要
- `ML output/models/{model_name}_model.pkl` - 训练好的模型

### 3. PCA降维分析（可选）

```bash
# 运行PCA降维
python pipelines/run_pca_state.py
```

输出文件：
- `ML output/states/pca_metadata_{symbol}_{timestamp}.json` - PCA元数据
- `ML output/models/pca_{symbol}.pkl` - PCA模型

### 4. 聚类分析（可选）

```bash
# 运行聚类分析
python pipelines/run_cluster_analysis.py
```

输出文件：
- `ML output/reports/clustering_*.csv` - 聚类评估报告
- `ML output/reports/cluster_features_k*.csv` - 各簇特征统计

### 5. 快速数据质检

```bash
# 快速检查数据质量
python utils/triage.py
```

## 📊 数据形状规范

### MultiIndex 格式

所有横截面数据统一使用 `MultiIndex [date, ticker]`：

```python
# 特征数据
features_df.index = MultiIndex([
    ('2023-01-04', '000001'),
    ('2023-01-05', '000001'),
    ...
])

# 目标数据
targets_series.index = MultiIndex([...])  # 同上
```

### 预测数据格式

```python
predictions_df.columns = ['y_true', 'y_pred', 'model', 'bucket']
predictions_df.index = MultiIndex(['date', 'ticker'])
```

## 🔧 模块接口规范

### DataLoader (data/data_loader.py)

```python
from data.data_loader import DataLoader

loader = DataLoader(data_root="machine learning/ML output")

# 加载特征和目标
features, targets = loader.load_features_and_targets(
    symbol="000001",
    target_col="future_return_5d",
    use_scaled=True
)

# 加载可交易标的
universe = loader.load_universe(
    symbol="000001",
    min_volume=1000000,
    min_price=1.0
)
```

### FeatureEngineer (features/feature_engineering.py)

```python
from features.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()

# 加载原始数据
data = engineer.load_stock_data("000001", "2022-01-01", "2024-12-31")

# 生成特征
features_df = engineer.prepare_features(data, use_auto_features=False)

# 特征选择
results = engineer.select_features(features_df, final_k=20)

# 特征标准化
scaled = engineer.scale_features(results['final_features_df'])
```

### TargetEngineer (targets/target_engineering.py)

```python
from targets.target_engineering import TargetEngineer

target_engineer = TargetEngineer()

# 生成目标变量
complete_df = target_engineer.create_complete_dataset(
    features_df=scaled_features,
    periods=[1, 5, 10],
    include_labels=True
)

# 保存数据集
filepath = target_engineer.save_dataset(complete_df, symbol="000001")
```

### Models (models/*.py)

```python
from models.ridge_model import RidgeModel
from models.rf_model import RandomForestModel
from models.lgbm_model import LightGBMModel

# 训练模型
model = RidgeModel(params={'alpha': [0.1, 1.0, 10.0]})
results = model.fit(X_train, y_train, X_valid, y_valid)

# 预测
y_pred = model.predict(X_test)

# 特征重要性
importance = model.get_feature_importance()
```

### Evaluation (evaluation/*.py)

```python
from evaluation.metrics import calculate_metrics
from evaluation.bucketing import bucket_predictions, analyze_bucket_performance

# 计算指标
metrics = calculate_metrics(y_true, y_pred)

# 分桶分析
bucketed = bucket_predictions(predictions_df, n_buckets=5)
bucket_stats = analyze_bucket_performance(bucketed)
```

## ⚙️ 配置文件说明

`configs/ml_baseline.yml` 包含所有可配置参数：

- **paths**: 各类输出目录
- **data**: 数据源和过滤条件
- **features**: 特征工程参数
- **target**: 目标变量配置
- **split**: 时间序列切分比例
- **models**: 各模型的超参数
- **evaluation**: 评估和分桶配置
- **backtest**: 回测参数
- **pca**: PCA配置（可选）
- **clustering**: 聚类配置（可选）
- **runtime**: 运行时配置

## 📈 输出文件说明

### reports/ 目录

| 文件 | 说明 |
|------|------|
| `model_bucket_performance.csv` | 各模型各桶的表现统计 |
| `test_predictions.csv` | 测试集预测明细 |
| `summary.json` | 评估摘要和验收结果 |
| `evaluation_report.txt` | 可读的详细报告 |
| `strategy_analysis_*.txt` | 回测策略分析 |
| `clustering_*.csv` | 聚类评估报告 |

### models/ 目录

| 文件 | 说明 |
|------|------|
| `ridge_model.pkl` | Ridge回归模型 |
| `randomforest_model.pkl` | 随机森林模型 |
| `lightgbm_model.pkl` | LightGBM模型 |
| `pca_{symbol}.pkl` | PCA降维模型 |

### states/ 目录

| 文件 | 说明 |
|------|------|
| `pca_metadata_*.json` | PCA元数据 |
| `scaler_*_meta.json` | 特征标准化元数据 |

## 🔍 故障排除

### 导入错误

如果遇到模块导入错误，确保：

1. 在项目根目录运行脚本
2. 或者设置 PYTHONPATH：

```bash
# Windows PowerShell
$env:PYTHONPATH = "d:\vscode projects\stock"

# Linux/Mac
export PYTHONPATH="/path/to/stock"
```

### 配置路径问题

所有相对路径都基于项目根目录。如果遇到路径错误：

1. 检查 `configs/ml_baseline.yml` 中的路径配置
2. 确保在正确的目录运行脚本
3. 使用绝对路径（不推荐）

### 数据文件未找到

确保已运行数据准备流程：

```bash
python pipelines/prepare_data.py
```

## 📚 参考文档

- [TRAIN_MODELS_README.md](TRAIN_MODELS_README.md) - 训练流程详细说明
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - 项目总体说明
- [configs/ml_baseline.yml](configs/ml_baseline.yml) - 配置文件模板

## 🎯 最佳实践

1. **数据准备优先**: 始终先运行 `prepare_data.py`
2. **配置驱动**: 通过修改 `ml_baseline.yml` 调整参数
3. **模块化开发**: 各模块独立可测试
4. **统一接口**: 遵循 MultiIndex 数据格式
5. **版本控制**: 输出文件带时间戳，便于回溯

---

**更新日期**: 2025-10-20
**版本**: 2.0.0 (重构版)
