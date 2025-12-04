# Models 模块

## 📋 概述

`models/` 模块提供统一的机器学习模型接口，支持回归和排序两类任务。所有模型继承自 `BaseModel` 抽象类，确保接口一致性。

## 📁 文件结构

```
models/
├── __init__.py           # 模块导出
├── base_model.py         # 基础模型抽象类
├── ridge_model.py        # Ridge 回归
├── rf_model.py           # 随机森林回归
├── lgbm_model.py         # LightGBM 回归
├── lgbm_ranker.py        # LightGBM 排序（LambdaRank）
├── transformers/         # 转换器（PCA等）
└── README.md             # 本文档
```

## 🔗 模型继承关系

```
BaseModel (抽象类)
├── RidgeModel          # 线性回归 + L2正则化
├── RandomForestModel   # 随机森林回归
├── LightGBMModel       # LightGBM 回归（objective='regression'）
└── LightGBMRanker      # LightGBM 排序（objective='lambdarank'）
```

## 📦 模型说明

### 1. BaseModel (`base_model.py`)

**基础模型抽象类**，定义所有模型必须实现的接口：

| 方法 | 说明 |
|------|------|
| `fit(X, y, X_valid, y_valid)` | 训练模型 |
| `predict(X)` | 预测 |
| `save(filepath)` | 保存模型（pickle/joblib） |
| `load(filepath)` | 加载模型 |
| `get_feature_importance()` | 获取特征重要性 |

### 2. RidgeModel (`ridge_model.py`)

**Ridge 回归模型**，适用于线性关系建模。

```python
from models import RidgeModel

model = RidgeModel(params={
    'alpha': 1.0,           # 正则化强度
    'fit_intercept': True
})
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

**特点**：
- 训练速度极快
- 支持交叉验证选择 alpha（传入 `alpha=[0.1, 1, 10]`）
- 特征重要性 = 回归系数绝对值

### 3. RandomForestModel (`rf_model.py`)

**随机森林回归模型**，适用于非线性关系和特征交互。

```python
from models import RandomForestModel

model = RandomForestModel(params={
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_leaf': 5,
    'n_jobs': -1
})
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

**特点**：
- 可捕捉非线性关系
- 自动处理特征交互
- 提供基于 Gini 的特征重要性

### 4. LightGBMModel (`lgbm_model.py`)

**LightGBM 回归模型**，用于 Baseline A（回归原始收益）和 Baseline B（Reg-on-Rank）。

```python
from models import LightGBMModel

model = LightGBMModel(params={
    'objective': 'regression',
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 8,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1
})
result = model.fit(X_train, y_train, X_valid, y_valid)  # 支持早停
pred = model.predict(X_test)
```

**特点**：
- 训练速度快，内存占用小
- 支持验证集早停（`early_stopping_rounds=50`）
- 返回训练结果包含 MSE、MAE、训练时间

### 5. LightGBMRanker (`lgbm_ranker.py`)

**LightGBM 排序模型（LambdaRank）**，用于 Baseline C（Sorting/LambdaRank）。

```python
from models import LightGBMRanker

model = LightGBMRanker(params={
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [10, 30, 50],
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,          # 比回归更浅
    'min_data_in_leaf': 50   # 比回归更大
})

# 必须提供 groups（每日样本数）
train_groups = X_train.groupby(level='date').size().tolist()
valid_groups = X_valid.groupby(level='date').size().tolist()

result = model.fit(
    X_train, y_train, 
    X_valid, y_valid,
    groups=train_groups,
    valid_groups=valid_groups
)
pred = model.predict(X_test)
```

**关键要求**：
- 训练数据必须按日期（group）排序
- 标签必须是离散整数（0, 1, 2, ... n_bins-1）
- 必须提供 `groups` 向量

**LambdaRank 原理**：
- 优化目标：最大化 NDCG（头部排序质量）
- 损失函数：基于 pairwise 的梯度提升
- 适用场景：关注 Top-K 股票的排序准确性

## 🎯 使用场景

| 模型 | 适用场景 | 优势 |
|------|---------|------|
| RidgeModel | 因子线性组合、基线对比 | 快速、可解释 |
| RandomForestModel | 特征筛选、非线性建模 | 稳健、无需调参 |
| LightGBMModel | 收益预测（回归） | 效果好、速度快 |
| LightGBMRanker | 股票排序（选股） | 直接优化排序质量 |

## 📊 与 Pipeline 的集成

```
run_baseline_pipeline.py
├── task_type='regression'      → LightGBMModel
├── task_type='regression_rank' → LightGBMModel（标签是 GaussRank）
└── task_type='lambdarank'      → LightGBMRanker
```

## 📝 输出文件

模型保存位置：
```
ML output/
├── models/baseline_v1/
│   ├── ridge/ridge_model.pkl
│   ├── random_forest/randomforest_model.pkl
│   ├── lightgbm/lightgbm_model.pkl
│   └── lightgbm_ranker/lightgbm_ranker_model.pkl
└── reports/baseline_v1/ranking/
    ├── regression_model.pkl
    ├── regression_rank_model.pkl
    └── lambdarank_model.pkl
```
