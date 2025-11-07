# Machine Learning 代码文件与输出对应关系

## 📋 核心代码文件与输出文件映射表

### 1️⃣ 数据准备流程 (`pipelines/prepare_data.py`)

**功能**: 完整的特征工程 + 目标变量生成

**调用的核心模块**:
- `features/feature_engineering.py` - 特征生成、选择、标准化
- `targets/target_engineering.py` - 目标变量生成

**输出文件**:
```
ML output/
├── artifacts/baseline_v1/
│   └── final_feature_list.txt                      # 最终特征清单
├── scalers/baseline_v1/
│   ├── scaler_{symbol}.pkl                         # 特征标准化器模型
│   ├── scaler_{symbol}_meta.json                   # 标准化元数据
│   └── scaler_{symbol}_scaled_features.csv         # 标准化后的特征CSV
├── datasets/baseline_v1/
│   └── with_targets_{symbol}_complete_{timestamp}.csv  # 完整数据集（特征+目标）
└── reports/baseline_v1/
    └── pipeline_summary_{timestamp}.txt            # 流程摘要报告
```

---

### 2️⃣ 模型训练 (`pipelines/train_models.py`)

**功能**: Ridge/RF/LightGBM 训练与评估

**调用的核心模块**:
- `models/ridge_model.py` - Ridge回归
- `models/rf_model.py` - 随机森林
- `models/lgbm_model.py` - LightGBM
- `evaluation/metrics.py` - 评估指标计算
- `evaluation/bucketing.py` - 分桶分析
- `evaluation/reporting.py` - 报告生成

**输出文件**:
```
ML output/
├── models/baseline_v1/
│   ├── ridge/
│   │   └── ridge_model.pkl                         # Ridge回归模型
│   ├── random_forest/
│   │   └── randomforest_model.pkl                  # 随机森林模型
│   └── lightgbm/
│       └── lightgbm_model.pkl                      # LightGBM模型
└── reports/baseline_v1/evaluation/
    ├── model_bucket_performance.csv                # 各模型分桶表现
    ├── test_predictions.csv                        # 测试集预测明细
    ├── summary.json                                # 评估摘要JSON
    └── evaluation_report.txt                       # 评估详细报告
```

---

### 3️⃣ PCA降维 (`pipelines/run_pca_state.py`)

**功能**: PCA状态生成与降维

**调用的核心模块**:
- `models/transformers/pca.py` - PCA降维实现

**输出文件**:
```
ML output/
├── models/baseline_v1/pca/
│   ├── pca_{symbol}_{timestamp}.pkl                # PCA模型（包含PCA对象和元数据）
│   └── pca_metadata_{symbol}_{timestamp}.json      # PCA元数据JSON（解释方差、特征等）
├── states/baseline_v1/
│   ├── states_pca_train_{symbol}_{timestamp}.npy   # 训练集PCA状态（降维后）
│   └── states_pca_test_{symbol}_{timestamp}.npy    # 测试集PCA状态（降维后）
└── reports/baseline_v1/
    └── pipeline_summary_{timestamp}.txt             # PCA流程摘要（完整流程）
```

---

### 4️⃣ 聚类分析 (`pipelines/run_cluster_analysis.py`)

**功能**: KMeans聚类与收益评估

**调用的核心模块**:
- `evaluation/cluster/cluster_evaluate.py` - 聚类评估实现

**输出文件**:
```
ML output/reports/baseline_v1/clustering/
├── clustering_analysis_report.txt                  # 聚类综合报告
├── cluster_comparison.csv                          # 聚类比较表（全局排名）
├── cluster_features_k4.csv                         # k=4 聚类特征统计
├── cluster_features_k5.csv                         # k=5 聚类特征统计
├── cluster_features_k6.csv                         # k=6 聚类特征统计
├── clustering_validation_results.csv               # 验证结果汇总
├── clustering_summary_all_k.csv                    # 所有k值汇总
├── cluster_models.pkl                              # 聚类模型（用于回测）
└── pc_metadata.pkl                                 # 最佳PC元数据（用于回测）
```

---

### 5️⃣ 聚类策略回测 (`backtest/cluster_strategy_backtest.py`)

**功能**: 基于聚类信号的策略回测

**输出文件**:
```
ML output/reports/baseline_v1/clustering/
├── strategy_equity_{symbol}_{timestamp}.csv        # 权益曲线
└── strategy_analysis_{symbol}_{timestamp}.txt      # 回测分析报告
```

**报告内容**:
- 选中的最佳聚类信息
- 策略收益 vs 基准收益
- 年化收益、夏普比率、最大回撤
- 随机基准对比（100次模拟）
- 验收结果（3项检查）

---

### 6️⃣ Top桶策略回测 (`backtest/top_bucket_backtest.py`)

**功能**: 基于Top桶的策略回测

**输出文件**:
```
ML output/reports/baseline_v1/
├── strategy_equity_{symbol}_{timestamp}.csv        # 权益曲线
└── strategy_analysis_{symbol}_{timestamp}.txt      # 回测分析报告
```

---

### 7️⃣ 数据快照管理 (`data/data_snapshot.py`)

**功能**: 数据版本化管理与质量检查

**输出文件**:
```
ML output/datasets/baseline_v1/snapshots/{snapshot_id}/
├── {symbol}_data.parquet                           # Parquet格式数据（或CSV）
├── metadata.json                                   # 快照元数据
└── reports/data_quality/
    └── quality_report_{timestamp}.json             # 数据质量报告
```

---

### 8️⃣ 快速质检工具 (`utils/triage.py`)

**功能**: 快速数据质量检查

**输出文件**:
```
ML output/reports/baseline_v1/
└── triage_report_{timestamp}.txt                   # 质检报告
```

---

## 🔄 完整流程示意图

```
┌─────────────────────┐
│ prepare_data.py     │ → scalers/  + datasets/
│ (特征+目标生成)      │
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│ train_models.py     │ → models/  + reports/evaluation/
│ (模型训练与评估)     │
└─────────────────────┘

┌─────────────────────┐
│ run_pca_state.py    │ → models/pca/  + states/
│ (PCA降维)           │
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│run_cluster_analysis │ → reports/clustering/  (包含cluster_models.pkl和pc_metadata.pkl)
│ (聚类分析)          │
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│cluster_strategy_    │ → reports/clustering/  (策略权益曲线和分析报告)
│backtest.py          │
│ (聚类策略回测)      │
└─────────────────────┘
```

---

## 📊 关键输出文件说明

| 文件名 | 生成脚本/模块 | 用途 |
|--------|---------|------|
| **数据准备相关** |||
| `final_feature_list.txt` | feature_engineering.py | 最终选择的特征清单 |
| `scaler_{symbol}.pkl` | feature_engineering.py | 特征标准化器模型 |
| `scaler_{symbol}_meta.json` | feature_engineering.py | 标准化元数据（特征名、统计信息） |
| `scaler_{symbol}_scaled_features.csv` | feature_engineering.py | 标准化后的特征CSV文件 |
| `with_targets_{symbol}_complete_{timestamp}.csv` | target_engineering.py | 完整数据集（特征+目标变量） |
| `pipeline_summary_{timestamp}.txt` | prepare_data.py | 数据准备流程摘要 |
| **模型训练相关** |||
| `ridge_model.pkl` | ridge_model.py | Ridge回归模型 |
| `randomforest_model.pkl` | rf_model.py | 随机森林模型 |
| `lightgbm_model.pkl` | lgbm_model.py | LightGBM模型 |
| `model_bucket_performance.csv` | reporting.py | 各模型分桶表现对比 |
| `test_predictions.csv` | reporting.py | 测试集预测明细 |
| `summary.json` | reporting.py | 评估摘要JSON |
| `evaluation_report.txt` | reporting.py | 评估详细报告 |
| **PCA降维相关** |||
| `pca_{symbol}_{timestamp}.pkl` | pca.py | PCA模型 |
| `pca_metadata_{symbol}_{timestamp}.json` | pca.py | PCA元数据 |
| `states_pca_train_{symbol}_{timestamp}.npy` | pca.py | 训练集PCA状态 |
| `states_pca_test_{symbol}_{timestamp}.npy` | pca.py | 测试集PCA状态 |
| **聚类分析相关** |||
| `cluster_models.pkl` | cluster_evaluate.py | KMeans聚类模型（所有k值） |
| `pc_metadata.pkl` | cluster_evaluate.py | 最佳PC元数据（用于回测信号） |
| `cluster_comparison.csv` | cluster_evaluate.py | 聚类全局排名 |
| `clustering_analysis_report.txt` | cluster_evaluate.py | 聚类综合报告 |
| `cluster_features_k{n}.csv` | cluster_evaluate.py | 各k值的聚类特征统计 |
| `clustering_validation_results.csv` | cluster_evaluate.py | 聚类验证结果 |
| `clustering_summary_all_k.csv` | cluster_evaluate.py | 所有k值汇总 |
| **回测相关** |||
| `strategy_equity_{symbol}_{timestamp}.csv` | cluster_strategy_backtest.py / top_bucket_backtest.py | 策略权益曲线 |
| `strategy_analysis_{symbol}_{timestamp}.txt` | cluster_strategy_backtest.py / top_bucket_backtest.py | 回测分析报告 |
| **数据快照相关** |||
| `{symbol}_data.parquet` | data_snapshot.py | Parquet格式数据快照 |
| `metadata.json` | data_snapshot.py | 快照元数据 |
| `quality_report_{timestamp}.json` | data_snapshot.py | 数据质量报告 |
| **质检工具** |||
| `triage_report_{timestamp}.txt` | triage.py | 快速质检报告 |

---

## 💡 使用建议

1. **按顺序执行**: 严格按照流程图顺序运行脚本
2. **检查输出**: 每步完成后检查对应输出文件是否生成
3. **时间戳匹配**: 注意文件名中的时间戳，确保使用最新文件
4. **配置管理**: 所有路径在 `configs/ml_baseline.yml` 中统一配置

---

**更新日期**: 2025-11-07  
**版本**: 3.0.0 (精简版)


