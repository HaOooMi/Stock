# Machine Learning 代码与输出文件对照

## � 核心脚本与输出文件

### 1. 数据准备流程

**脚本**: `pipelines/prepare_data.py`

**功能**: 完整的数据准备（特征工程 + 目标工程）

**输出文件**:
```
ML output/
├── datasets/baseline_v1/
│   ├── scaler_{symbol}.pkl                              # 特征标准化器
│   ├── scaler_{symbol}_scaled_features.csv              # 标准化后的特征
│   └── with_targets_{symbol}_complete_{timestamp}.csv   # 完整数据集（特征+目标）
└── scalers/baseline_v1/
    ├── scaler_{symbol}.pkl                              # 标准化器备份
    └── scaler_{symbol}_meta.json                        # 标准化元数据
```

**运行方式**:
```bash
# 单标的
python pipelines/prepare_data.py

# 多标的（命令行）
python pipelines/prepare_data.py --symbols 000001 600000 000858
```

---

### 2. 模型训练

**脚本**: `pipelines/train_models.py`

**功能**: 训练Ridge、RandomForest、LightGBM并评估

**输出文件**:
```
ML output/
├── models/baseline_v1/
│   ├── ridge_model.pkl                    # Ridge回归模型
│   ├── randomforest_model.pkl             # 随机森林模型
│   └── lightgbm_model.pkl                 # LightGBM模型
│
├── predictions/baseline_v1/
│   └── test_predictions.csv               # 测试集预测明细
│
└── reports/baseline_v1/
    ├── model_bucket_performance.csv       # 各模型分桶表现
    ├── summary.json                       # 评估摘要（IC、RankIC、准确率等）
    └── evaluation_report.txt              # 详细评估报告（可读文本）
```

**运行方式**:
```bash
python pipelines/train_models.py
```

---

### 3. PCA降维分析

**脚本**: `pipelines/run_pca_state.py`

**功能**: 对特征进行PCA降维，生成状态空间

**输出文件**:
```
ML output/
├── models/baseline_v1/
│   └── pca_{symbol}.pkl                   # PCA降维模型
│
└── states/baseline_v1/
    └── pca_metadata_{symbol}_{timestamp}.json  # PCA元数据（解释方差等）
```

**运行方式**:
```bash
python pipelines/run_pca_state.py
```

---

### 4. 聚类分析

**脚本**: `pipelines/run_cluster_analysis.py`

**功能**: 对PCA降维后的状态进行聚类分析

**输出文件**:
```
ML output/reports/baseline_v1/
├── clustering_evaluation_{timestamp}.csv           # 聚类质量评估
├── cluster_features_k{n}_{timestamp}.csv          # 各簇的特征统计
└── cluster_daily_distribution_{timestamp}.csv      # 每日聚类分布
```

**运行方式**:
```bash
python pipelines/run_cluster_analysis.py
```

---

### 5. 回测分析

#### 5.1 Top桶策略回测

**脚本**: `backtest/top_bucket_backtest.py`

**功能**: 基于预测分桶的多空策略回测

**输出文件**:
```
ML output/reports/baseline_v1/
└── strategy_analysis_{model_name}_{timestamp}.txt  # 策略表现分析
```

#### 5.2 聚类信号回测

**脚本**: `backtest/cluster_strategy_backtest.py`

**功能**: 基于聚类信号的策略回测

**输出文件**:
```
ML output/reports/baseline_v1/
└── cluster_strategy_analysis_{timestamp}.txt  # 聚类策略分析
```

---

## 🗂️ 输出目录结构

```
ML output/
├── datasets/baseline_v1/          # 数据集
│   ├── scaler_*.pkl
│   ├── scaler_*_scaled_features.csv
│   └── with_targets_*_complete_*.csv
│
├── models/baseline_v1/            # 模型文件
│   ├── ridge_model.pkl
│   ├── randomforest_model.pkl
│   ├── lightgbm_model.pkl
│   └── pca_*.pkl
│
├── scalers/baseline_v1/           # 标准化器
│   ├── scaler_*.pkl
│   └── scaler_*_meta.json
│
├── predictions/baseline_v1/       # 预测结果
│   └── test_predictions.csv
│
├── reports/baseline_v1/           # 评估报告
│   ├── model_bucket_performance.csv
│   ├── summary.json
│   ├── evaluation_report.txt
│   ├── clustering_*.csv
│   └── strategy_analysis_*.txt
│
├── states/baseline_v1/            # 状态文件
│   └── pca_metadata_*.json
│
└── figures/baseline_v1/           # 可视化图表（待扩展）
```

---

## � 关键输出文件说明

### 数据集文件

| 文件 | 说明 | 生成者 |
|------|------|--------|
| `with_targets_{symbol}_complete_{timestamp}.csv` | 完整数据集（特征+目标） | prepare_data.py |
| `scaler_{symbol}_scaled_features.csv` | 标准化后的特征 | prepare_data.py |

### 模型文件

| 文件 | 说明 | 生成者 |
|------|------|--------|
| `ridge_model.pkl` | Ridge回归模型 | train_models.py |
| `randomforest_model.pkl` | 随机森林模型 | train_models.py |
| `lightgbm_model.pkl` | LightGBM模型 | train_models.py |
| `pca_{symbol}.pkl` | PCA降维模型 | run_pca_state.py |

### 评估报告

| 文件 | 说明 | 生成者 |
|------|------|--------|
| `model_bucket_performance.csv` | 各模型各桶的表现统计 | train_models.py |
| `test_predictions.csv` | 测试集预测明细（含bucket） | train_models.py |
| `summary.json` | IC、RankIC、准确率等关键指标 | train_models.py |
| `evaluation_report.txt` | 可读的详细评估报告 | train_models.py |
| `clustering_evaluation_*.csv` | 聚类质量评估（轮廓系数等） | run_cluster_analysis.py |
| `cluster_features_k*.csv` | 各簇的特征统计 | run_cluster_analysis.py |
| `strategy_analysis_*.txt` | 回测策略表现 | top_bucket_backtest.py |

---

## 🚀 快速使用流程

### 完整流程

```bash
# 1. 数据准备
python pipelines/prepare_data.py

# 2. 模型训练
python pipelines/train_models.py

# 3. (可选) PCA分析
python pipelines/run_pca_state.py

# 4. (可选) 聚类分析
python pipelines/run_cluster_analysis.py
```

### 查看输出

```bash
# 查看模型表现
cat "ML output/reports/baseline_v1/summary.json"

# 查看分桶统计
cat "ML output/reports/baseline_v1/model_bucket_performance.csv"

# 查看详细报告
cat "ML output/reports/baseline_v1/evaluation_report.txt"
```

---

**更新日期**: 2025-11-07  
**版本**: 3.0.0 (精简版)
