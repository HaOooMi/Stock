# 机器学习基线训练系统

## 阶段 12：机器学习基线（回归/排序）

### 功能概述

本系统实现了完整的机器学习基线训练流程：

1. **数据加载**: 从ML output加载标准化特征和目标数据
2. **数据切分**: 时间序列切分（防泄漏，支持purge）
3. **模型训练**: Ridge、RandomForest、LightGBM（可选）
4. **预测评估**: 测试集预测，计算MSE、MAE、IC、Rank IC
5. **分桶分析**: 按日横截面分5桶，统计每桶真实收益
6. **策略回测**: Top桶策略收益对比
7. **报告生成**: CSV报告、JSON摘要、TXT详细报告

### 验收标准

- ✅ Top桶收益 > 全体均值
- ✅ Spread（Top - Bottom）> 0

### 目录结构

```
machine learning/
├── train_models.py          # 主训练脚本
├── configs/
│   └── ml_baseline.yml      # 配置文件
├── data/
│   ├── __init__.py
│   └── data_loader.py       # 数据加载器
├── models/
│   ├── __init__.py
│   ├── base_model.py        # 基础模型类
│   ├── ridge_model.py       # Ridge回归
│   ├── rf_model.py          # 随机森林
│   └── lgbm_model.py        # LightGBM
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py           # 评估指标
│   ├── bucketing.py         # 分桶分析
│   └── reporting.py         # 报告生成
├── utils/
│   ├── __init__.py
│   ├── splitting.py         # 时间序列切分
│   └── logger.py            # 日志设置
└── ML output/
    ├── scaler_*_scaled_features.csv  # 标准化特征
    ├── with_targets_*_complete_*.csv # 目标数据
    └── reports/                      # 输出报告
```

### 使用步骤

#### 1. 安装依赖

```bash
pip install pandas numpy scikit-learn scipy pyyaml
pip install lightgbm  # 可选，如果安装失败会自动跳过
```

#### 2. 准备数据

确保已运行前置步骤：
- `feature_engineering.py` - 生成特征
- `target_engineering.py` - 生成目标
- 确保存在标准化特征文件

#### 3. 配置参数

编辑 `configs/ml_baseline.yml`：

```yaml
data:
  symbol: "000001"            # 股票代码
  start_date: "2022-01-01"
  end_date: "2024-12-31"

target:
  name: "future_return_5d"    # 目标变量

split:
  train_ratio: 0.6            # 训练集60%
  valid_ratio: 0.2            # 验证集20%
  test_ratio: 0.2             # 测试集20%
  purge_days: 10              # 防泄漏清除天数

models:
  ridge:
    enabled: true
    params:
      alpha: [0.1, 1.0, 10.0, 100.0]  # CV选择最优alpha
  
  random_forest:
    enabled: true
    params:
      n_estimators: 500
      max_depth: 10
  
  lightgbm:
    enabled: true              # 如果未安装会自动跳过
    params:
      n_estimators: 500
      learning_rate: 0.05

evaluation:
  n_buckets: 5                 # 分5桶
  bucket_method: "quantile"    # 等分位分桶
```

#### 4. 运行训练

```bash
cd "d:\vscode projects\stock\machine learning"
python train_models.py --config configs/ml_baseline.yml
```

或使用默认配置：

```bash
python train_models.py
```

#### 5. 查看结果

训练完成后，检查输出目录 `ML output/reports/`：

- **model_bucket_performance.csv**: 各模型各桶的表现统计
- **test_predictions.csv**: 测试集预测明细
- **summary.json**: 评估摘要（包含验收结果）
- **evaluation_report.txt**: 详细的可读报告

### 输出示例

#### 控制台输出

```
======================================================================
🚀 机器学习基线训练
======================================================================

📋 加载配置...
   ✅ 配置加载完成
   🎲 随机种子: 42

📊 加载数据...
   ✅ 数据加载完成
      特征数: 20
      样本数: 580

📅 数据切分...
   ✅ 切分完成:
      训练集: 340 样本, 2022-01-04 ~ 2023-09-26
      验证集: 120 样本, 2023-10-17 ~ 2024-05-28
      测试集: 120 样本, 2024-06-11 ~ 2024-12-30

🤖 模型训练...

📌 训练Ridge模型
   🔧 训练 Ridge 模型...
      🔍 使用交叉验证选择最优alpha: [0.1, 1.0, 10.0, 100.0]
      ✅ 最优alpha: 1.0000
      ⏱️  训练时间: 0.05秒
      📊 训练集 MSE: 0.000123, MAE: 0.008456
      📊 验证集 MSE: 0.000145, MAE: 0.009123

🌲 训练RandomForest模型
   ...

🎯 测试集预测与评估...
   📊 评估 Ridge 模型
      MSE: 0.000134
      MAE: 0.008789
      IC: 0.2345 (p=0.0089)
      Rank IC: 0.2567 (p=0.0045)

📊 分桶分析...
   🪣 Ridge 分桶分析
   📊 按日横截面分5桶 (方法: quantile)
   ✅ 分桶完成: 120/120 样本

   📊 Top桶 (桶5): 平均收益 0.0156
   📊 Bottom桶 (桶1): 平均收益 -0.0089
   📈 Top-Bottom Spread: 0.0245

✅ 验收检查...
   Ridge:
      Top桶 > 全体均值: ✅ (0.015600 vs 0.003200)
      Spread > 0: ✅ (0.024500)
      验收结果: ✅ 通过

🎉 训练流程完成！
======================================================================

📊 模型数量: 3
📈 测试样本: 120

🏆 最佳模型: LightGBM
   Rank IC: 0.2891
   Top桶收益: 0.0178
   Spread: 0.0287
   验收: ✅ 通过

📁 报告目录: machine learning/ML output/reports
```

#### CSV报告示例 (model_bucket_performance.csv)

| model | bucket | n_obs | mean_y_true | std_y_true | mean_y_pred | Top-Bottom |
|-------|--------|-------|-------------|------------|-------------|------------|
| Ridge | 1 | 24 | -0.0089 | 0.0123 | -0.0156 | - |
| Ridge | 2 | 24 | -0.0012 | 0.0098 | -0.0045 | - |
| Ridge | 3 | 24 | 0.0034 | 0.0089 | 0.0023 | - |
| Ridge | 4 | 24 | 0.0098 | 0.0112 | 0.0089 | - |
| Ridge | 5 | 24 | 0.0156 | 0.0145 | 0.0198 | 0.0245 |

### 高级配置

#### 1. 调整交易成本

```yaml
backtest:
  transaction_cost:
    commission: 0.0003    # 双边佣金0.03%
    slippage: 0.0010      # 双边滑点0.1%
```

#### 2. 修改分桶方法

```yaml
evaluation:
  n_buckets: 10           # 改为10桶
  bucket_method: "equal_width"  # 等宽分桶
```

#### 3. 启用缓存

```yaml
runtime:
  use_cache: true
  cache_dir: "machine learning/ML output/cache"
```

### 故障排除

#### 问题1: LightGBM安装失败

**解决方案**:
```bash
# Windows
pip install --upgrade pip setuptools wheel
pip install lightgbm

# 或者直接禁用LightGBM
# 在配置文件中设置: lightgbm.enabled = false
```

#### 问题2: 找不到特征文件

**解决方案**:
1. 确保已运行 `feature_engineering.py`
2. 检查 `ML output/` 目录下是否存在 `scaler_*_scaled_features.csv`
3. 确认配置文件中的 `symbol` 与文件名匹配

#### 问题3: YAML模块未安装

**解决方案**:
```bash
pip install pyyaml
```

#### 问题4: 验收未通过

**原因分析**:
- 特征质量不足
- 目标变量噪声过大
- 模型参数不合适
- 样本量不足

**解决方案**:
1. 增加更多有效特征
2. 调整目标窗口（如改为10日）
3. 优化模型超参数
4. 扩大数据时间范围

### 下一步工作

1. **特征优化**: 添加更多技术指标、基本面特征
2. **模型集成**: Stacking、加权平均
3. **超参优化**: 使用Optuna等工具
4. **多标的**: 扩展到多只股票
5. **滚动训练**: 实现walk-forward验证
6. **风险控制**: 添加止损、仓位管理

### 参考文献

- [scikit-learn文档](https://scikit-learn.org/)
- [LightGBM文档](https://lightgbm.readthedocs.io/)
- 《Python量化交易》- 黄宇

### 作者与支持

如有问题，请检查：
1. 配置文件是否正确
2. 数据文件是否存在
3. Python环境是否完整

---

**版本**: 1.0.0  
**更新日期**: 2025-10-20
