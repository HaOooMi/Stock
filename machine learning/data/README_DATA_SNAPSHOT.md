# 数据加载与清洗 - 使用指南

## 📋 概述

符合研究宪章要求的数据处理模块，从 InfluxDB 加载实时市场数据，从 MySQL 加载财务数据，结合 ML 特征进行清洗过滤，生成可追溯的数据快照。

**核心功能：** 数据快照管理 | 7层交易可行性过滤 | PIT对齐验证 | 数据质量报告

---

## 🏗️ 数据流程

```
InfluxDB (原始市场数据: OHLCV, 换手率等)
    ↓
MySQL (财务数据: 净利润、ROE等)
    ↓
ML output (工程特征)
    ↓
合并数据
    ↓
7层过滤 (ST/停牌/涨跌停/上市龄/成交量/价格/换手率)
    ↓
PIT对齐验证
    ↓
数据快照 (版本化存储 + 元数据 + 质量报告)
    ↓
模型训练
```

---

## 🚀 快速开始

### 1. 配置 InfluxDB（`configs/ml_baseline.yml`）

```yaml
data:
  influxdb:
    enabled: true
    url: "http://localhost:8086"
    org: "stock"
    bucket: "stock_kdata"
    token: "your-token-from-config"
```

### 2. 运行数据准备脚本

```bash
cd "d:\vscode projects\stock\machine learning"
python pipelines/prepare_data_with_snapshot.py
```

### 3. 使用代码加载

```python
from data.data_loader import DataLoader

# 初始化（启用所有功能）
loader = DataLoader(
    data_root="ML output/datasets/baseline_v1",
    enable_influxdb=True,
    enable_filtering=True,
    enable_snapshot=True,
    enable_pit_alignment=True
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

# 或从已有快照加载
features, targets = loader.load_from_snapshot('ds_2025Q4_v1')
```

---

## 📊 7层交易可行性过滤

| 层级 | 过滤项 | 条件 | 数据来源 |
|------|--------|------|----------|
| 1 | ST/退市 | 股票名称包含ST | InfluxDB |
| 2 | 停牌 | 成交量=0 | InfluxDB |
| 3 | 涨跌停 | 涨跌幅>±9.5% | InfluxDB |
| 4 | 上市龄 | 上市<60天 | Database |
| 5 | 成交量 | <100万 | InfluxDB |
| 6 | 价格 | <1元 | InfluxDB |
| 7 | 换手率 | <0.2% | InfluxDB |

**过滤日志：** `ML output/datasets/baseline_v1/filter_log_{symbol}.csv`

---

## � 数据快照

### 快照ID格式
```
ds_2025Q4_v1  (年份_季度_版本号)
```

### 快照内容
```
ML output/snapshots/ds_2025Q4_v1/
├── 000001_data.parquet    # Parquet格式数据（需安装pyarrow）
└── metadata.json          # 元数据（参数、质量报告、hash）
```

### 元数据示例
```json
{
  "snapshot_id": "ds_2025Q4_v1",
  "symbol": "000001",
  "n_samples": 1000,
  "n_features": 20,
  "filters": {"min_volume": 1000000, "exclude_st": true},
  "random_seed": 42,
  "quality_checks": {"overall_quality": "PASS"}
}
```

---

## � 配置参数

```yaml
# configs/ml_baseline.yml
data:
  # InfluxDB 配置
  influxdb:
    enabled: true
    url: "http://localhost:8086"
    org: "stock"
    bucket: "stock_kdata"
    token: "aIX6s47Ymo..."
    
  # 交易可行性过滤
  universe:
    min_volume: 1000000          # 最小成交量
    min_amount: 50000000         # 最小成交额
    min_price: 1.0               # 最小价格
    min_turnover: 0.002          # 最小换手率（0.2%）
    min_listing_days: 60         # 最小上市天数
    exclude_st: true             # 排除ST股票
    exclude_limit_moves: true    # 排除涨跌停
    limit_threshold: 0.095       # 涨跌停阈值
    
  # PIT对齐
  pit:
    enabled: true
    financial_lag_days: 90       # 财务数据滞后天数
    validate_alignment: true     # 验证PIT对齐
    
  # 快照管理
  snapshot:
    enabled: true
    save_parquet: true           # 保存为Parquet（需pyarrow）
    auto_generate_id: true       # 自动生成快照ID
```

---

## 📁 输出文件

```
ML output/
├── snapshots/
│   └── ds_2025Q4_v1/
│       ├── 000001_data.parquet
│       └── metadata.json
├── reports/
│   └── data_quality/
│       └── ds_2025Q4_v1.json
└── datasets/
    └── baseline_v1/
        └── filter_log_000001.csv
```

---

## 🐛 常见问题

**Q1: InfluxDB 连接失败**
```bash
# 检查 InfluxDB 是否运行
cd "C:\Program Files\InfluxData"
.\influxd

# 测试连接
curl http://localhost:8086/ping
```

**Q2: Parquet 保存失败**
```bash
pip install pyarrow
```

**Q3: 过滤后样本过少**
- 降低过滤阈值（如 `min_volume`）
- 扩展时间范围
- 检查过滤日志找出主要原因

**Q4: 未找到市场数据**
```bash
# 运行数据采集
python get_stock_info/main.py
```

---

## ✅ 验收标准

| 检查项 | 标准 | 当前实现 |
|--------|------|----------|
| PIT对齐 | 通过验证 | ✅ 自动验证 |
| 可交易样本 | ≥200/日 | ✅ 可配置阈值 |
| 数据质量 | 无红灯项 | ✅ 自动检查 |
| 版本化 | 快照ID记录 | ✅ 自动生成 |

**运行验收：**
```bash
python pipelines/prepare_data_with_snapshot.py
```

---

## 🎯 最佳实践

1. **实验可重复性**
   ```python
   features, targets, snapshot_id = loader.load_with_snapshot(...)
   # 记录 snapshot_id 到实验日志
   ```

2. **版本控制**
   - 数据快照ID: `ds_2025Q4_v1`
   - 实验ID: `EXP-20251024-001`
   - 模型版本: `baseline_v1`

3. **审计追踪**
   ```python
   # 查看所有快照
   snapshots = loader.snapshot_mgr.list_snapshots()
   
   # 查看质量报告
   quality_report = f"ML output/reports/data_quality/{snapshot_id}.json"
   ```

---

**维护者：** HaOooMi | **文档版本：** v1.0 (2025-10-24)
