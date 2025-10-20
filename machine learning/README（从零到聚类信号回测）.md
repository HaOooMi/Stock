# 股票量化交易系统

基于 PCA 降维和 K-Means 聚类的无监督量化系统，从原始数据到交易信号的完整流程。

## 核心特性

- **数据流**: InfluxDB → 特征工程 → PCA降维 → K-Means聚类 → 交易信号
- **防泄漏**: 8层防护（特征shift、时间切分、Purge gap、T+1执行）
- **样本外验证**: 训练/测试严格分离，双重验证机制

## 技术架构

```
OHLCV数据 → 54特征 → 20精选特征 → 6个PC → 状态聚类 → 交易信号
```

## 核心模块

### 1. 特征工程 (`feature_engineering.py`)
- **数据加载**: 从InfluxDB加载股票OHLCV历史数据
- **手工特征**: 生成54个技术特征（收益率4个、滚动统计16个、动量4个、波动率5个、成交量8个、价格范围8个、技术指标9个）
- **自动特征**: 可选TSFresh自动特征生成（窗口20天，最多30个）
- **特征选择**: 方差过滤(0.001) → 相关性去重(0.95) → 重要性排序 → 选择top 20
- **特征标准化**: RobustScaler（训练集fit，测试集transform，防泄漏）
- **全局shift(1)**: 所有特征整体滞后1期，防止当期泄漏

### 2. 目标工程 (`target_engineering.py`)
- **未来收益率**: 生成future_return_1d/5d/10d（使用shift(-period)）
- **分类标签**: 二分类（涨跌）和多分类（分位数）标签
- **尾部NaN验证**: 确保尾部有period个NaN，防止标签泄漏
- **完整数据集**: 合并特征和目标，保存为CSV

### 3. PCA状态生成 (`pca_state.py`)
- **时间切分**: 80%训练，20%测试
- **Purge机制**: 训练集尾部删除10天（≥max_target_period）
- **PCA训练**: 只在训练集fit，保留90%累计解释方差（通常6个成分）
- **状态生成**: 训练集和测试集分别transform生成PCA状态
- **质量验证**: 成分数应为原始特征的1/6到1/3，解释方差≥90%
- **完整流程**: 集成特征工程→目标工程→PCA三步骤

### 4. 聚类评估 (`cluster_evaluate.py`)
- **K-Means聚类**: k=4,5,6三个值，n_init=20，max_iter=500
- **簇收益排序**: 按future_return_5d对每个簇排序
- **最佳PC选择**: 基于训练集历史计算IC，选择|IC|最大的PC（保存到pc_metadata.pkl）
- **训练验证**: 最佳vs最差簇差异 > global_std × 0.4
- **测试验证**: 训练最佳簇在测试集排名 ≤ 50%
- **质量指标**: Silhouette Score和Calinski-Harabasz Score
- **综合报告**: 生成cluster_comparison.csv、clustering_analysis_report.txt等

### 5. 策略回测 (`strategy_backtest.py`)
- **加载结果**: 读取cluster_models.pkl和pc_metadata.pkl
- **簇筛选**: 验证通过 + 占比[10%,60%] + 测试收益>0
- **最佳簇选择**: 按全局排名选top N（默认3个）
- **信号生成**: 最佳簇=+1，其他=0（可使用PC门槛过滤）
- **T+1执行**: signal_t1 = roll(signal, 1)，今天信号决定明天仓位
- **性能计算**: 总收益、年化收益、夏普比率、最大回撤、胜率、盈亏比、换手率
- **随机基准**: 100次随机信号对比验证

### 6. 快速诊断 (`quick_triage.py`)
- **体检1**: 信号对齐与时间穿越检查（验证T+1对齐，IC范围0.02~0.15）
- **体检1A**: 破坏性对照实验（过去收益IC、随机标签IC、当期收益IC），智能区分动量vs泄漏
- **体检2**: 成本换手分析（按回合计费，成本侵蚀比例，敏感性分析）
- **体检3**: IC排名能力检查（T+1对齐，分层收益5分位，Spread计算）
- **体检4**: 状态过滤验证（多k值聚类，训练/测试一致性）
- **体检5**: 门槛持有期优化（网格搜索分位数×持有期，样本外最优）
- **诊断决策树**: IC≥0.02且成本侵蚀<50%→继续优化，否则回到特征工程

## 快速开始

```python
# 运行完整流程
python pca_state.py          # 特征工程 → 目标工程 → PCA
python cluster_evaluate.py   # 聚类分析
python strategy_backtest.py  # 策略回测
python quick_triage.py       # 快速诊断
```

## 防数据泄漏机制

| 层级 | 机制 | 代码位置 |
|------|------|----------|
| 1️⃣ | 特征shift(1) | `feature_engineering.py:296` |
| 2️⃣ | 标签尾部NaN | `target_engineering.py:119` |
| 3️⃣ | PCA时间切分 | `pca_state.py:146-162` |
| 4️⃣ | Purge gap | `pca_state.py:153` |
| 5️⃣ | Scaler fit-transform分离 | `feature_engineering.py:917-919` |
| 6️⃣ | PCA fit-transform分离 | `pca_state.py:202-206` |
| 7️⃣ | KMeans fit-predict分离 | `cluster_evaluate.py:135,217` |
| 8️⃣ | T+1执行对齐 | `strategy_backtest.py:578-590` |

## 性能示例（000001平安银行）

**特征**: 54 → 20个（压缩2.7x）  
**PCA**: 20 → 6个成分（解释方差91.2%）  
**聚类**: k=5，训练差异0.061，测试排名2/5 ✅  
**策略**: 年化17.2%，夏普1.23，最大回撤-4.21%，胜率56.8%

## 技术栈

Python 3.9+ | NumPy | Pandas | scikit-learn | InfluxDB | AkShare | TA-Lib（可选）

## 目录结构

```
machine learning/
├── feature_engineering.py   # 特征工程
├── target_engineering.py    # 目标工程
├── pca_state.py            # PCA状态生成
├── cluster_evaluate.py      # 聚类评估
├── strategy_backtest.py     # 策略回测
├── quick_triage.py         # 快速诊断
└── ML output/
    ├── scaler_*.pkl
    ├── with_targets_*.csv
    ├── models/
    ├── states/
    └── reports/
```

## 文档

- `README.md` - 本文档
- `PROJECT_SUMMARY.md` - 完整技术文档

---
作者: HaOooMi | 许可: MIT
