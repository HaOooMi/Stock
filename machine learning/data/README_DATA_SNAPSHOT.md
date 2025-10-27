# 数据清洗与快照层 - 使用指南

## 📋 概述

数据清洗与快照层是符合研究宪章要求的核心数据处理模块，确保数据的无偏性和可追溯性。

### 核心功能

1. **数据快照管理** - 数据版本化和可追溯性
2. **交易可行性过滤** - 7层过滤确保可交易性
3. **Point-in-Time对齐** - 防止未来信息泄漏
4. **数据质量报告** - 自动质量检查和报告生成

---

## 🏗️ 架构设计

### 模块组成

```
data/
├── data_snapshot.py        # 数据快照管理器
├── tradability_filter.py   # 交易可行性过滤器
├── pit_aligner.py          # PIT数据对齐器
└── data_loader.py          # 增强版数据加载器（集成以上模块）
```

### 数据流程

```
原始数据
    ↓
[交易可行性过滤] (7层)
    ├─ 1. ST/退市
    ├─ 2. 停牌
    ├─ 3. 涨跌停
    ├─ 4. 上市龄
    ├─ 5. 成交量
    ├─ 6. 价格
    └─ 7. 换手率
    ↓
[PIT对齐验证]
    ├─ 尾部NaN保留
    ├─ 特征shift验证
    └─ 时间顺序检查
    ↓
[数据快照]
    ├─ 版本化存储
    ├─ 元数据记录
    └─ 质量报告生成
    ↓
可用于模型训练
```

---

## 🚀 快速开始

### 1. 基础使用

```python
from data.data_loader import DataLoader

# 初始化数据加载器（启用所有功能）
loader = DataLoader(
    data_root="machine learning/ML output/datasets/baseline_v1",
    enable_snapshot=True,
    enable_filtering=True,
    enable_pit_alignment=True,
    filter_config={
        'min_volume': 1000000,
        'min_amount': 50000000,
        'min_price': 1.0,
        'min_turnover': 0.002,
        'min_listing_days': 60,
        'exclude_st': True,
        'exclude_limit_moves': True
    }
)

# 加载数据并创建快照
features, targets, snapshot_id = loader.load_with_snapshot(
    symbol='000001',
    start_date='2022-01-01',
    end_date='2024-12-31',
    target_col='future_return_5d',
    use_scaled=True,
    random_seed=42
)

print(f"快照ID: {snapshot_id}")
print(f"特征形状: {features.shape}")
```

### 2. 使用集成脚本

```bash
# 运行数据准备脚本（带快照管理）
cd "d:\vscode projects\stock\machine learning"
python pipelines/prepare_data_with_snapshot.py
```

### 3. 从快照加载数据

```python
# 从已有快照加载
features, targets = loader.load_from_snapshot('ds_2025Q4_v1')
```

---

## 📊 数据结构规范

### MultiIndex格式

```python
# 索引: [date, ticker]
DatetimeIndex(['2022-01-04', '2022-01-05', ...], name='date')
Index(['000001', '000001', ...], name='ticker')

# 必需列
- ohlcv: open, high, low, close, volume, amount
- adj_factor: 后复权因子
- tradable_flag: 0/1 可交易标记
- industry_code: 行业代码（可选）
- mkt_cap_float: 流通市值（可选）
- shares_outstanding: 流通股本（可选）
```

### Parquet存储

```
snapshots/
└── ds_2025Q4_v1/
    ├── 000001_data.parquet  # 数据文件（快速、列存）
    └── metadata.json         # 元数据文件
```

---

## 🔍 交易可行性过滤

### 7层过滤顺序

| 层级 | 过滤项 | 条件 | 说明 |
|------|--------|------|------|
| 1 | ST/退市 | 股票名称包含ST | 排除特殊处理股票 |
| 2 | 停牌 | 成交量=0 | 无法交易 |
| 3 | 涨跌停 | 涨跌幅>±9.5% | 流动性受限 |
| 4 | 上市龄 | 上市<60天 | 新股波动大 |
| 5 | 成交量 | volume<100万 | 流动性不足 |
| 6 | 价格 | close<1元 | 低价股风险 |
| 7 | 换手率 | turnover<0.2% | 流动性不足 |

### 过滤日志

```csv
filter,removed,remaining,ratio
1_ST_退市,50,950,0.95
2_停牌,30,920,0.92
3_涨跌停,20,900,0.90
...
```

保存位置: `ML output/datasets/baseline_v1/filter_log_{symbol}.csv`

---

## 📅 Point-in-Time (PIT) 对齐

### 核心原则

1. **财务数据**: 严格按`公告日 + 滞后天数`生效
2. **历史成分**: 使用当时的实际成分（避免幸存者偏差）
3. **价格数据**: 后复权处理
4. **交易日对齐**: 统一到交易日历

### 验证检查

```python
pit_results = {
    'tail_nans_preserved': True,   # 尾部NaN保留
    'features_valid': True,         # 特征有效性
    'time_ordered': True,           # 时间顺序
    'overall_pass': True            # 总体通过
}
```

---

## 📸 数据快照管理

### 快照ID格式

```
ds_YYYYQQ_vN
```

示例:
- `ds_2025Q4_v1` - 2025年第4季度第1版
- `ds_2025Q4_v2` - 2025年第4季度第2版

### 元数据记录

```json
{
  "snapshot_id": "ds_2025Q4_v1",
  "created_at": "2025-10-24T10:30:00",
  "symbol": "000001",
  "start_date": "2022-01-01",
  "end_date": "2024-12-31",
  "n_samples": 1000,
  "n_features": 20,
  "filters": {
    "min_volume": 1000000,
    "min_price": 1.0,
    "exclude_st": true
  },
  "random_seed": 42,
  "data_hash": "abc123...",
  "quality_checks": {...}
}
```

### 快照列表

```python
snapshots = loader.snapshot_mgr.list_snapshots()
print(snapshots)

# 输出:
#   snapshot_id     created_at        symbol  n_samples  quality
#   ds_2025Q4_v1    2025-10-24...     000001  1000       PASS
#   ds_2025Q4_v2    2025-10-25...     000001  1050       PASS
```

---

## 📊 数据质量报告

### 检查项

| 检查项 | 红灯阈值 | 说明 |
|--------|----------|------|
| 缺失率 | >20% | 任何列缺失率过高 |
| 重复率 | >1% | 索引重复 |
| 可交易样本 | <70% | 可交易样本过少 |
| 时间连续性 | 间隔>10天 | 数据缺失 |

### 报告示例

```json
{
  "timestamp": "2025-10-24T10:30:00",
  "total_samples": 1000,
  "checks": {
    "missing_ratio": {
      "max": 0.05,
      "mean": 0.01,
      "red_flag": false
    },
    "duplicates": {
      "count": 0,
      "ratio": 0.0,
      "red_flag": false
    },
    "tradable_samples": {
      "count": 900,
      "ratio": 0.90,
      "red_flag": false
    }
  },
  "overall_quality": "PASS",
  "red_flags_count": 0
}
```

保存位置: `ML output/reports/data_quality/{snapshot_id}.json`

---

## ✅ 验收标准

### 数据清洗层验收

根据研究宪章要求:

| 检查项 | 标准 | 当前实现 |
|--------|------|----------|
| PIT对齐 | 通过验证 | ✅ 自动验证 |
| 历史成分 | 无幸存者偏差 | ✅ 支持历史成分 |
| 可交易样本 | ≥200/日 | ✅ 可配置阈值 |
| 数据质量 | 无红灯项 | ✅ 自动检查 |
| 版本化 | 快照ID记录 | ✅ 自动生成 |
| 元数据 | 完整记录 | ✅ JSON格式 |

### 运行验收脚本

```bash
python pipelines/prepare_data_with_snapshot.py
```

输出示例:
```
[步骤7] 数据验收检查
   ✅ 样本规模: 1000 (最低 200)
   ✅ PIT对齐验证
   ✅ 数据质量: PASS (0 个红灯)

======================================================================
✅ 验收通过
======================================================================

🎉 恭喜! 数据清洗与快照层验收通过
   快照ID: ds_2025Q4_v1
   可用于后续模型训练
```

---

## 🔧 配置说明

### 配置文件: `configs/ml_baseline.yml`

```yaml
data:
  # 数据快照
  snapshot:
    enabled: true               # 启用快照管理
    save_parquet: true         # 保存为Parquet格式
    auto_generate_id: true     # 自动生成快照ID
    
  # 交易可行性过滤
  universe:
    min_volume: 1000000        # 最小成交量
    min_amount: 50000000       # 最小成交额（5000万）
    min_price: 1.0             # 最小价格
    min_turnover: 0.002        # 最小换手率（0.2%）
    min_listing_days: 60       # 最小上市天数
    exclude_st: true           # 排除ST股票
    exclude_limit_moves: true  # 排除涨跌停
    limit_threshold: 0.095     # 涨跌停阈值（9.5%）
    
  # PIT对齐
  pit:
    enabled: true              # 启用PIT对齐
    financial_lag_days: 90     # 财务数据滞后天数
    use_adj_factor: true       # 使用后复权因子
    validate_alignment: true   # 验证PIT对齐
```

---

## 📁 输出文件

### 目录结构

```
ML output/
├── snapshots/                    # 数据快照
│   └── ds_2025Q4_v1/
│       ├── 000001_data.parquet   # 数据文件
│       └── metadata.json         # 元数据
│
├── reports/
│   └── data_quality/             # 数据质量报告
│       └── ds_2025Q4_v1.json
│
└── datasets/
    └── baseline_v1/
        └── filter_log_000001.csv  # 过滤日志
```

---

## 🎯 最佳实践

### 1. 每次实验前创建新快照

```python
# 固定快照ID到实验元数据
features, targets, snapshot_id = loader.load_with_snapshot(...)

# 记录到实验日志
experiment_metadata = {
    'exp_id': 'EXP-20251024-001',
    'snapshot_id': snapshot_id,
    'random_seed': 42,
    'created_at': datetime.now().isoformat()
}
```

### 2. 复现实验

```python
# 使用相同的快照ID
features, targets = loader.load_from_snapshot('ds_2025Q4_v1')

# 确保相同的随机种子
np.random.seed(42)
```

### 3. 版本控制

- 数据快照ID: `ds_2025Q4_v1`
- 实验ID: `EXP-20251024-001`
- 模型版本: `baseline_v1`

### 4. 审计追踪

```python
# 查看快照列表
snapshots = loader.snapshot_mgr.list_snapshots()

# 查看质量报告
quality_report_path = f"ML output/reports/data_quality/{snapshot_id}.json"

# 查看过滤日志
filter_log_path = f"ML output/datasets/baseline_v1/filter_log_{symbol}.csv"
```

---

## 🐛 常见问题

### Q1: Parquet保存失败

**问题**: `ModuleNotFoundError: No module named 'pyarrow'`

**解决**:
```bash
pip install pyarrow
```

### Q2: 过滤后样本数过少

**问题**: 过滤后可交易样本 < 200

**解决**:
1. 降低过滤阈值（如`min_volume`）
2. 扩展时间范围
3. 增加股票池

### Q3: PIT验证失败

**问题**: `tail_nans_preserved = False`

**解决**:
检查目标变量计算过程中是否正确保留尾部NaN

### Q4: 质量报告红灯

**问题**: 数据质量检查出现红灯项

**解决**:
1. 查看质量报告详情
2. 检查原始数据源
3. 调整过滤参数或数据范围

---

## 📚 相关文档

- [研究宪章 v1.0](../docs/research_charter_v1.md)
- [实验日志模板](../docs/experiments/EXPERIMENT_TEMPLATE.md)
- [数据加载器API文档](./data_loader.py)

---

## 🔄 更新日志

### v1.0 (2025-10-24)
- ✅ 初始版本
- ✅ 数据快照管理
- ✅ 7层交易可行性过滤
- ✅ PIT对齐验证
- ✅ 数据质量报告
- ✅ 符合研究宪章要求

---

## 👥 维护者

HaOooMi - 2025年10月24日

如有问题或建议,请参考研究宪章或联系维护者。
