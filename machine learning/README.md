# 股票机器学习预处理流程

这个项目实现了完整的股票特征工程到PCA状态生成的机器学习预处理流程。

## 📁 项目结构

```
machine learning/
├── feature_engineering.py      # 特征工程核心模块
├── target_engineering.py       # 目标变量工程模块  
├── pca_state.py                # PCA状态生成模块
├── complete_pipeline.py        # 集成管道模块
├── run_complete_pipeline.py    # 完整流程运行脚本
├── run_pca_state.py            # PCA状态生成脚本
├── test_pca_state.py           # PCA模块测试脚本
└── ML output/                  # 机器学习输出目录
    ├── models/                 # 训练好的模型文件
    ├── states/                 # PCA状态表示文件
    ├── scaled_features.csv     # 标准化特征数据
    ├── final_feature_list.txt  # 最终特征列表
    ├── target_variables.csv    # 目标变量数据
    ├── target_labels_*.csv     # 分类标签数据
    └── pipeline_summary.txt    # 流程摘要报告
```

## 🚀 快速开始

### 方法1: 运行完整流程（推荐）

运行完整的预处理流程，包括特征工程、目标工程和PCA状态生成：

```bash
python run_complete_pipeline.py
```

### 方法2: 分步执行

如果需要分步执行或调试特定模块：

#### 步骤1: 特征工程
```bash
python feature_engineering.py
```

#### 步骤2: 目标变量工程
```bash
python target_engineering.py
```

#### 步骤3: PCA状态生成
```bash
python run_pca_state.py
```

### 方法3: 测试PCA模块

在使用真实数据前测试PCA功能：

```bash
python test_pca_state.py
```

## 📊 输出文件说明

### 特征文件
- `scaled_features.csv`: 标准化后的特征数据，用于机器学习训练
- `final_feature_list.txt`: 经过重要性筛选的最终特征列表

### 目标文件  
- `target_variables.csv`: 未来收益率等连续型目标变量
- `target_labels_binary.csv`: 二分类标签（涨/跌）
- `target_labels_quantile.csv`: 多分类标签（分位数）

### PCA文件
- `models/pca_model_*.pkl`: 训练好的PCA模型
- `states/pca_states_train_*.npy`: 训练集的PCA状态表示
- `states/pca_states_test_*.npy`: 测试集的PCA状态表示

### 日志和摘要
- `*.log`: 各模块的运行日志
- `pipeline_summary.txt`: 完整流程的摘要报告

## ⚙️ 配置参数

### 特征工程参数
- `max_horizon`: 最大预测窗口（默认10天）
- `importance_threshold`: 特征重要性阈值（默认0.01）
- `correlation_threshold`: 相关性去重阈值（默认0.95）

### 目标工程参数  
- `future_horizons`: 未来预测期间（默认[1, 3, 5, 10]天）
- `quantile_bins`: 分位数分类数量（默认5）

### PCA参数
- `n_components`: 目标解释方差（默认0.9）
- `train_ratio`: 训练数据比例（默认0.8）
- 主成分数量约束：原始特征数的1/6到1/3

## 🔧 环境要求

### Python版本
- Python 3.8+

### 必需包
```bash
pip install pandas numpy scikit-learn akshare influxdb-client
```

### 可选包（用于可视化和高级分析）
```bash
pip install matplotlib seaborn plotly
```

## 📈 使用流程

1. **数据准备**: 确保InfluxDB中有股票历史数据
2. **运行流程**: 执行 `python run_complete_pipeline.py`
3. **检查结果**: 查看 `ML output/` 目录下的输出文件
4. **开始建模**: 使用生成的特征和目标数据进行机器学习建模

## 🐛 故障排除

### 常见问题

1. **缺少输入数据**
   - 确保InfluxDB连接正常
   - 检查股票数据是否已下载

2. **内存不足**
   - 减少处理的股票数量
   - 降低特征维度
   - 分批处理数据

3. **PCA成分过少/过多**
   - 调整 `n_components` 参数
   - 检查原始特征质量
   - 考虑特征预筛选

### 调试技巧

- 查看日志文件了解详细错误信息
- 使用测试脚本验证单个模块功能
- 检查中间输出文件的数据质量

## 📝 开发说明

### 模块职责

- `feature_engineering.py`: 负责从原始数据生成和筛选特征
- `target_engineering.py`: 负责生成各种类型的目标变量
- `pca_state.py`: 负责PCA降维和状态表示生成
- `complete_pipeline.py`: 负责模块间的集成和流程控制

### 扩展建议

1. **添加新特征**: 在 `feature_engineering.py` 中扩展特征计算函数
2. **自定义目标**: 在 `target_engineering.py` 中添加新的目标变量类型
3. **改进PCA**: 考虑使用IncrementalPCA或KernelPCA
4. **并行处理**: 使用multiprocessing加速大数据集处理

## 📄 许可证

本项目仅供学习和研究使用。

## 👥 作者

Assistant - AI编程助手

---

💡 **提示**: 首次运行建议使用 `test_pca_state.py` 验证环境配置，然后再运行完整流程。