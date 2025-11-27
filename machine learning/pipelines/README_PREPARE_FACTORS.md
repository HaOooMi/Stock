# 因子准备管道 - 完整流程说明

## 📋 概述

`prepare_factors.py` 实现了因子工厂 v1 的完整流程，**充分利用了已有的横截面评估框架**。

## 🔗 模块集成关系

```
prepare_factors.py (主流程)
├── data.MarketDataLoader        # 步骤1-2: 批量加载多股票数据
│   └── load_market_data_batch() #   返回MultiIndex[date, ticker]
├── data.TradabilityFilter       # 步骤2.5: 7层交易可行性过滤
├── data.FinancialDataLoader     # 步骤2.6: PIT对齐财务数据 (可选)
├── data.DataSnapshot            # 步骤2.7: 数据快照管理 ✨新增
├── features.FactorFactory       # 步骤3: 生成因子
├── evaluation (横截面评估框架)   # 步骤4: 因子质量检查 ⭐核心
│   ├── CrossSectionAnalyzer     #   - 统一评估接口 + 深度质量检查
│   │                             #     * 标准分析: IC/ICIR/Spread/单调性/换手率
│   │                             #     * 深度检查: IC衰减/PSI/KS (check_quality=True)
│   ├── cross_section_metrics    #   - IC/ICIR/Spread/单调性/换手率计算 (Numba加速)
│   ├── factor_preprocessing     #   - Winsorize/标准化/中性化
│   ├── visualization            #   - 6种图表生成 ✨已集成
│   └── tearsheet                #   - HTML报告生成
├── features.FactorLibraryManager # 步骤5: 因子入库
└── 报告输出                      # 步骤6: Tearsheet报告 + 可视化图表
```
```

### 模块功能说明

**CrossSectionAnalyzer (统一评估接口):**

**标准分析模式** (`analyzer.analyze()`)
- ✅ Rank IC / ICIR（每日横截面Spearman）
- ✅ 分位数收益 & 单调性（Kendall τ）
- ✅ Top-Mean Spread
- ✅ 换手率统计
- ✅ 生成Tearsheet报告

**深度质量检查模式** (`analyzer.analyze(check_quality=True)`)
- ✅ IC半衰期与IC Decay曲线（时间衰减特性）
- ✅ PSI测试（分布稳定性，训练集vs测试集）
- ✅ KS测试（Kolmogorov-Smirnov分布差异检验）

**设计理念:**
- 日常使用标准模式即可（快速，覆盖核心指标）
- 因子入库前启用深度检查（确保稳定性与独特性）
- 统一接口，避免维护多套评估系统

## 🎯 工作流程

### 步骤1-2: 数据加载
```python
data_loader = DataLoader(...)
features_df, targets_df = data_loader.load_features_and_targets(
    start_date='2020-01-01',
    end_date='2024-12-31',
    enable_tradability_filter=True
)
```

**产出:**
- `features_df`: OHLCV + 市值/行业 (MultiIndex[date, ticker])
- `targets_df`: 未来收益标签

---

### 步骤3: 因子生成
```python
factory = FactorFactory()
all_factors_df = factory.generate_all_factors(features_df)
```

**产出:**
- 动量因子: ROC_5, ROC_10, ROC_20, ROC_60, ROC_120, ...
- 波动率因子: RealizedVol_20, RealizedVol_60, Parkinson, ...
- 量价因子: Turnover, VolumePriceCorr, VWAP_Dev, ...
- 技术指标: RSI, MACD, Bollinger Bands, ...

---

### 步骤4: 横截面质量检查 ⭐ **使用你的评估框架**

这是核心步骤！完全使用 `evaluation/` 下的模块。

```python
# 4.1 计算远期收益（使用你的metrics模块）
from evaluation.cross_section_metrics import calculate_forward_returns

forward_returns_df = calculate_forward_returns(
    prices=prices_df,
    periods=[1, 5, 10, 20],
    method='simple'
)

# 4.2 逐个因子评估（使用你的CrossSectionAnalyzer）
from evaluation.cross_section_analyzer import CrossSectionAnalyzer

for factor_name in all_factors_df.columns:
    # 构建分析器
    analyzer = CrossSectionAnalyzer(
        factors=all_factors_df[[factor_name]],
        forward_returns=forward_returns_df,
        tradable_mask=tradable_mask,
        market_cap=market_cap,
        industry=industry
    )
    
    # 预处理（使用你的预处理管道）
    analyzer.preprocess(
        winsorize=True,
        standardize=True,
        neutralize=True  # 可选
    )
    
    # 运行完整分析（标准模式）
    analyzer.analyze(
        n_quantiles=5,
        ic_method='spearman',
        spread_method='top_minus_mean'
    )
    
    # 或启用深度质量检查
    analyzer.analyze(
        n_quantiles=5,
        ic_method='spearman',
        spread_method='top_minus_mean',
        check_quality=True  # 额外计算PSI/KS/IC衰减
    )
    
    # 获取结果
    results = analyzer.get_results()
    
    # 提取关键指标
    key = (factor_name, 'ret_5d')
    ic_summary = results['ic_summary'][key]
    spread_summary = results['spread_summary'][key]
    monotonicity = results['monotonicity'][key]
    
    # 判断是否通过
    if (ic_summary['mean'] >= 0.02 and 
        ic_summary['icir_annual'] >= 0.5 and
        spread_summary['mean'] > 0):
        qualified_factors.append(factor_name)
```

**评估指标:**
- ✅ Rank IC ≥ 0.02 且统计显著 (p < 0.05)
- ✅ ICIR (年化) ≥ 0.5
- ✅ Top-Mean Spread > 0
- ✅ 单调性: Kendall τ 显著
- ✅ 与已有因子相关性 < 0.7

**产出:**
- `qualified_factors`: 通过检查的因子列表
- `quality_reports`: 每个因子的详细评估结果

---

### 步骤5: 因子入库
```python
manager = FactorLibraryManager()

for factor_name in qualified_factors:
    manager.add_factor(
        factor_name=factor_name,
        quality_report=quality_reports[factor_name],
        formula=factor_info['formula'],
        family=factor_info['family']
    )
```

**产出:**
- 因子库元数据
- 族别表现统计

---

### 步骤6: 生成Tearsheet报告 ⭐ **使用你的tearsheet模块**

为每个通过的因子生成完整的HTML报告。

```python
from evaluation.tearsheet import generate_html_tearsheet

for factor_name in qualified_factors:
    report = quality_reports[factor_name]
    full_results = report['full_results']
    
    # 生成HTML tearsheet
    generate_html_tearsheet(
        analyzer_results=full_results,
        factor_name=factor_name,
        return_period='ret_5d',
        output_path=f"reports/tearsheet_{factor_name}_5d.html",
        plot_paths=None
    )
    
    # 保存CSV数据
    full_results['ic_series'][key].to_csv(f"reports/ic_{factor_name}_5d.csv")
    full_results['quantile_returns'][key].to_csv(f"reports/quantile_returns_{factor_name}_5d.csv")
```

**产出文件结构:**
```
ML output/
├── snapshots/{snapshot_id}/                  # ✨新增: 数据快照
│   ├── {symbols}_data.parquet               # Parquet格式快照
│   ├── metadata.json                        # 快照元数据
│   └── reports/data_quality/
│       └── {snapshot_id}.json               # 数据质量报告
├── reports/baseline_v1/factors/
│   ├── tearsheet_ROC_20_5d.html          ⭐ HTML综合报告
│   ├── ic_ROC_20_5d.csv                  📊 IC时间序列
│   ├── quantile_returns_ROC_20_5d.csv    📊 分位数收益
│   ├── tearsheet_RealizedVol_60_5d.html
│   └── ...
├── figures/baseline_v1/factors/{factor}/    # ✨新增: 可视化图表
│   ├── ic_series_{factor}_5d.png           📈 IC走廊图
│   ├── ic_dist_{factor}_5d.png             📈 IC分布图
│   ├── ic_heatmap_{factor}_5d.png          📈 月度IC热力图
│   ├── quantile_cumret_{factor}_5d.png     📈 累计收益曲线
│   ├── quantile_meanret_{factor}_5d.png    📈 平均收益柱状图
│   └── spread_cumret_{factor}_5d.png       📈 Spread收益
└── datasets/baseline_v1/
    ├── qualified_factors_20250119.parquet  💾 通过的因子数据
    ├── qualified_factors_20250119.csv
    └── final_feature_list.txt              📝 因子清单
```

---

## ✅ 验收标准

### 验收1: 稳定因子数量
- **要求**: ≥10 个因子通过检查
- **实际**: 从质量报告中统计

### 验收2: 横截面 Rank IC 显著
- **要求**: 80%以上的通过因子满足 IC > 0.02 且 p < 0.05
- **来源**: `ic_summary['mean']` 和 `ic_summary['p_value']`

### 验收3: 组合IC提升
- **要求**: 所有通过因子的等权组合 IC > 0.03
- **计算**: 
  ```python
  combined_factor = qualified_factors_df.mean(axis=1)
  combined_ic = calculate_ic(combined_factor, targets)
  ```

---

## 🎨 与旧版本的对比

### ❌ 旧版本问题 (你提到的)
```python
# 旧版本使用了不存在的模块
from features.factor_quality_checker import FactorQualityChecker

checker = FactorQualityChecker(...)
report = checker.comprehensive_check(...)  # 自己实现了一套IC计算

# 问题:
# 1. 重复造轮子
# 2. 没有利用已有的横截面评估框架
# 3. 无法生成标准的tearsheet报告
```

### ✅ 新版本优势
```python
# 新版本完全使用evaluation模块
from evaluation.cross_section_analyzer import CrossSectionAnalyzer
from evaluation.cross_section_metrics import calculate_forward_returns
from evaluation.tearsheet import generate_html_tearsheet

# 优势:
# 1. ✅ 复用已有的成熟框架
# 2. ✅ IC/ICIR/Spread计算与手算一致
# 3. ✅ 自动生成完整的tearsheet报告
# 4. ✅ 输出目录结构符合宪章要求
# 5. ✅ 所有图表自动生成
```

---

## 🔧 配置参数

在 `configs/ml_baseline.yml` 中配置:

```yaml
data:
  influxdb:
    url: "http://localhost:8086"
    token: "your-token"
    org: "stock"
    bucket: "stock_kdata"
  
  tradability_filter:
    enabled: true
    min_volume: 1000000
    exclude_st: true
    exclude_limit: true

targets:
  type: 'forward_return'
  horizon: 5

factors:
  momentum:
    periods: [5, 10, 20, 60, 120]
  
  volatility:
    windows: [20, 60]
  
  volume_price:
    enabled: true
```

---

## 🚀 快速开始

```bash
cd "machine learning/pipelines"

# 测试模式（3只股票）
python prepare_factors.py

# 全市场模式
python prepare_factors.py --full-market

# 指定股票池
python prepare_factors.py --tickers 000001.SZ,000002.SZ,600000.SH
```

---

## 📖 相关文档

- `evaluation/README_CROSS_SECTION.md`: 横截面评估框架详细文档
- `features/README_FACTOR_FACTORY.md`: 因子工厂说明
- `data/README_DATA_SNAPSHOT.md`: 数据快照管理

---

**作者**: HaOooMi  
**版本**: v1.1 (集成数据快照 + 可视化图表)  
**更新**: 2025-01-27
