# 横截面评估框架（Alphalens风格）

## 📖 简介

本框架提供了完整的横截面因子评估工具，遵循Alphalens风格，专为量化投资中的因子研究设计。完全符合《机器学习量化研究宪章 v1.0》第2章要求。

### 🎯 实现概览

已成功构建完整的横截面因子评估框架，包含**6个核心模块**和**1个示例脚本**，共约2500行代码。

### ✨ 核心特性

✅ **完整的IC分析**
- 每日横截面Rank IC（Spearman相关）
- IC统计摘要（均值、标准差、ICIR、t检验、p-value）
- IC胜率（正IC比例）
- IC时间序列可视化

✅ **分位数组合分析**
- 5分位等频分桶（横截面）
- 计算各分位数的远期收益
- 累计净值曲线
- 单调性检验（Kendall τ）

✅ **Spread分析**
- Top-Mean Spread（实盘推荐）
- Top-Bottom Spread（学术常见）
- Spread累计收益与夏普比
- 年化夏普比计算

✅ **因子预处理**
- Winsorize（1%-99%极值处理）
- Z-score标准化（横截面）
- 市值中性化（回归残差法）
- 行业中性化（回归残差法）
- 综合中性化（市值+行业）

✅ **换手率分析**
- Top分位数换手率跟踪
- 换手率时间序列
- 持仓变化统计

✅ **完善的可视化**（7种专业图表）
- IC时间序列图（走廊图，含±1σ区间）
- IC分布直方图（含正态拟合）
- 分位数累计收益图（彩色净值曲线）
- 分位数平均收益柱状图
- Spread累计收益图
- 换手率时间序列图
- 月度IC热力图

✅ **深度质量检查**（`check_quality=True`启用）
- IC衰减曲线与半衰期计算
- PSI（Population Stability Index）分布稳定性检验
- KS（Kolmogorov-Smirnov）分布差异检验
- 训练集/测试集稳定性验证

✅ **专业报告输出**
- HTML格式Tearsheet报告（响应式设计）
- 自动因子评估（优秀/合格/弱）
- CSV数据导出
- 图表自动生成（300 DPI）

---

## 📦 模块架构

### 核心模块清单

| 模块 | 文件 | 功能 | 代码量 |
|------|------|------|--------|
| 核心度量 | `cross_section_metrics.py` | Forward Returns, IC, ICIR, Spread, 换手率等 | ~600行 |
| 因子预处理 | `factor_preprocessing.py` | Winsorize, 标准化, 中性化 | ~400行 |
| 分析器 | `cross_section_analyzer.py` | 统一评估接口, 流程编排, **深度质量检查** | ~750行 |
| 可视化 | `visualization.py` | 7种专业图表 | ~600行 |
| 报告生成 | `tearsheet.py` | HTML报告, CSV导出 | ~400行 |
| 示例脚本 | `run_cross_section_analysis.py` | 端到端示例 | ~300行 |

**注**：v1.1 版本将质量检查功能（PSI/KS/IC衰减）整合到 `CrossSectionAnalyzer` 中，通过 `check_quality=True` 参数启用。

### 输出目录结构

```
ML output/reports/baseline_vX/factors/
└── {factor_name}/
    ├── tearsheet_{factor_name}_{period}.html      # HTML综合报告
    ├── ic_{factor_name}_{period}.csv              # IC时间序列
    ├── quantile_returns_{factor_name}_{period}.csv # 分位数收益
    ├── ic_series_{factor_name}_{period}.png       # IC走廊图
    ├── ic_dist_{factor_name}_{period}.png         # IC分布
    ├── quantile_cumret_{factor_name}_{period}.png # 累计收益
    ├── quantile_meanret_{factor_name}_{period}.png # 平均收益
    ├── spread_cumret_{factor_name}_{period}.png   # Spread
    ├── ic_heatmap_{factor_name}_{period}.png      # IC热力图
    └── turnover_{factor_name}.png                  # 换手率
```

---

## 🚀 快速开始

```bash
cd "machine learning/pipelines"
python run_cross_section_analysis.py
```

---

## 📊 数据格式要求

### 输入数据格式

所有输入DataFrame都需要**MultiIndex[date, ticker]**格式：

```python
import pandas as pd

index = pd.MultiIndex.from_product(
    [dates, tickers],
    names=['date', 'ticker']
)

# 因子数据
factors = pd.DataFrame({
    'factor_1': [...],
    'factor_2': [...]
}, index=index)

# 价格数据
prices = pd.DataFrame({
    'close': [...]
}, index=index)

# 市值数据（可选）
market_cap = pd.DataFrame({
    'market_cap': [...]
}, index=index)

# 行业数据（可选）
industry = pd.DataFrame({
    'industry': ['金融', '科技', ...]
}, index=index)
```

---

## 📈 核心度量说明

### 1. Forward Returns（远期收益）

**公式**：
- **Simple Return**: $r_{t \rightarrow t+H} = \frac{P_{t+H}}{P_t} - 1$
- **Log Return**: $r_{t \rightarrow t+H} = \log(P_{t+H}) - \log(P_t)$

**实现要点**：使用 `groupby(level='ticker').shift(-period)` 保证按股票分组计算，避免跨股票错误

### 2. Rank IC（排序信息系数）

**定义**：每日横截面因子值与远期收益的Spearman秩相关系数

$$\text{IC}_t = \text{Spearman}(\text{factor}_t, \text{forward\_return}_{t \rightarrow t+H})$$

**宪章标准**：
- 目标值: |Rank IC| ≥ 0.02
- 统计显著性: p-value < 0.05

**实现要点**：按日期分组，每日独立计算相关系数，使用 `scipy.stats.spearmanr()` 并自动计算p-value

### 3. ICIR（IC信息比率）

**公式**：

$$\text{ICIR} = \frac{\text{Mean}(\text{IC})}{\text{Std}(\text{IC})}$$

年化：$\text{ICIR}_{\text{annual}} = \text{ICIR} \times \sqrt{252}$

**宪章标准**：
- 合格: ICIR ≥ 0.5
- 优秀: ICIR ≥ 1.0

### 4. IC胜率

**定义**：日度IC > 0的比例

**宪章标准**：
- 合格: ≥ 55%
- 优秀: ≥ 60%

### 5. 分位数收益

**实现流程**：
1. 每日横截面按因子值排序
2. 使用`pd.qcut()`等频分位分组（5档或10档）
3. 计算各组平均收益
4. 累计计算净值曲线

### 6. Spread

**公式**：
- **Top-Mean**: $\text{Spread} = R_{\text{top}} - \text{Mean}(R_{\text{all}})$ （实盘推荐，更稳健）
- **Top-Bottom**: $\text{Spread} = R_{\text{top}} - R_{\text{bottom}}$ （学术常见）

**宪章标准**：
- 测试集 Spread > 0（硬约束）
- Spread Sharpe(年化) > 1.0

### 7. 单调性（Kendall τ）

**定义**：检验分位数收益是否随因子值单调递增

**实现**：使用 `scipy.stats.kendalltau()` 计算秩相关及p-value

### 8. 换手率

**公式**：

$$\text{Turnover}_t = 1 - \frac{|\text{Holdings}_t \cap \text{Holdings}_{t-1}|}{|\text{Holdings}_t|}$$

**用途**：追踪Top分位数持仓变化，估算交易成本

---

## 🎯 因子预处理流程

### 标准流程（推荐）

#### 1. Winsorize（极值处理）

**方法**：按截面1%-99%分位数裁剪

```python
# 按日横截面处理
for date in dates:
    lower = quantile(0.01)
    upper = quantile(0.99)
    factor_winsorized = factor.clip(lower, upper)
```

#### 2. Z-score标准化（横截面）

**公式**：按日期截面标准化

$$z = \frac{x - \mu_{\text{cross}}}{\sigma_{\text{cross}}}$$

**实现**：
```python
# 按日横截面处理
for date in dates:
    mean = factor.mean()
    std = factor.std()
    factor_zscore = (factor - mean) / std
```

**其他方法**：
- `'min_max'`: Min-Max标准化到[0, 1]
- `'rank'`: 排名标准化到[0, 1]

#### 3. 中性化（回归残差法）

**方法**：截面回归取残差

$$\text{factor} \sim \alpha + \beta_1 \log(\text{mkt\_cap}) + \beta_2 \text{industry\_dummies}$$

**实现要点**：
- 按日横截面处理，每日独立回归
- OLS: $\beta = (X'X)^{-1} X'y$
- 取残差作为中性化后的因子

**支持模式**：
- 市值中性化
- 行业中性化
- 综合中性化（市值+行业）

**关键原则**：所有计算按**日横截面**独立进行，避免前视偏差（Look-ahead Bias）

---

## 📜 研究宪章合规性

符合《机器学习量化研究宪章 v1.0》第2章要求：

| 条款 | 要求 | 实现 |
|------|------|------|
| 横截面口径 | 按日独立计算 | ✅ 所有函数横截面循环 |
| IC标准 | Rank IC + 显著性检验 | ✅ Spearman + t-test |
| ICIR标准 | 含年化计算 | ✅ ICIR × √252 |
| 分位数分析 | 等频分桶 + 单调性 | ✅ pd.qcut + Kendall τ |
| Spread分析 | Top-Mean优先 | ✅ 双模式支持 |
| 因子预处理 | Winsorize/标准化/中性化 | ✅ 完整流水线 |
| 报告输出 | HTML+CSV+PNG | ✅ 自动生成 |

---

### 因子评估标准

**优秀因子**：
- Mean IC > 0.03
- ICIR(年化) > 1.5
- p-value < 0.01
- 正IC比例 > 60%
- Spread Sharpe(年化) > 1.0

**合格因子**：
- Mean IC > 0.01
- ICIR(年化) > 0.5
- p-value < 0.05
- 正IC比例 > 55%

**弱因子**：不满足合格标准

### 可交易性过滤

建议构建可交易性mask，过滤以下情况：
- ST股票
- 停牌股票
- 涨停板股票
- 跌停板股票

---

## 🐛 常见问题

### Q1: 为什么IC很低？

**可能原因**：
- 因子预测能力弱
- 未进行中性化处理
- 数据质量问题（停牌、涨跌停未过滤）
- 前瞻期选择不当

**解决方案**：
1. 开启完整预处理流程（Winsorize + 标准化 + 中性化）
2. 添加可交易性过滤
3. 尝试不同的前瞻期（1/5/10/20日）
4. 检查数据质量

### Q2: 为什么分位数收益不单调？

**可能原因**：
- 因子噪音较大
- 样本数不足
- 存在极端值
- 未进行横截面标准化

**解决方案**：
1. 开启Winsorize处理极端值
2. 确保每日样本数 ≥ 30
3. 使用横截面标准化
4. 考虑市值/行业中性化

### Q3: 换手率过高怎么办？

**策略**：
- 延长持仓周期（5日→10日→20日）
- 设置换手率上限
- 考虑交易成本影响
- 使用因子平滑技术

---

## 🎓 术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| 横截面 | Cross-Section | 同一时间点上对多个股票进行的分析 |
| Rank IC | Rank Information Coefficient | 因子值与未来收益的Spearman秩相关系数 |
| ICIR | IC Information Ratio | IC的均值/标准差，衡量IC的稳定性 |
| Winsorize | Winsorization | 极值处理，将超出分位数的值裁剪到分位数 |
| 中性化 | Neutralization | 通过回归残差法去除因子中的市值/行业效应 |
| Spread | Spread | 顶部分位数收益与底部/均值的差值 |
| Tearsheet | Tearsheet | 综合性评估报告，包含多维度分析结果 |
| Monotonicity | Monotonicity | 单调性，检验分位数收益是否随因子值单调递增 |
| Turnover | Turnover | 换手率，衡量持仓变化频率 |

---

*最后更新: 2024 | 文档版本: v1.0.0 | 框架版本: v1.0.0*
