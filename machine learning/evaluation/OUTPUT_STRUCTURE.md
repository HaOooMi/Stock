# 横截面评估框架 - 输出文件结构文档

> 更新日期: 2025-11-19  
> 版本: v1.0

---

## 目录

1. [输出目录概览](#输出目录概览)
2. [文件类型说明](#文件类型说明)
3. [输出文件命名规范](#输出文件命名规范)
4. [详细文件格式](#详细文件格式)
5. [使用示例](#使用示例)

---

## 1. 输出目录概览

### 1.1 标准输出目录结构

```
ML output/
└── reports/
    └── baseline_v{X}/
        └── factors/
            ├── {factor_name}/
            │   ├── tearsheet_{factor_name}_{period}.html      # HTML综合报告
            │   ├── ic_{factor_name}_{period}.csv              # IC时间序列数据
            │   ├── quantile_returns_{factor_name}_{period}.csv # 分位数收益数据
            │   ├── ic_series_{factor_name}_{period}.png       # IC走廊图
            │   ├── ic_dist_{factor_name}_{period}.png         # IC分布图
            │   ├── quantile_cumret_{factor_name}_{period}.png # 分位数累计收益
            │   ├── quantile_meanret_{factor_name}_{period}.png # 分位数平均收益
            │   ├── spread_cumret_{factor_name}_{period}.png   # Spread累计收益
            │   ├── ic_heatmap_{factor_name}_{period}.png      # IC月度热力图
            │   └── turnover_{factor_name}.png                 # 换手率时间序列
            │
            ├── {another_factor}/
            │   └── ... (同上)
            │
            └── multi_factor_comparison/
                ├── factor_ic_comparison.html                   # 多因子IC对比
                ├── factor_correlation_matrix.png               # 因子相关性矩阵
                └── factor_performance_summary.csv              # 因子表现汇总
```

### 1.2 实际示例

```
ML output/
└── reports/
    └── baseline_v1/
        └── factors/
            ├── momentum_20d/
            │   ├── tearsheet_momentum_20d_ret_5d.html
            │   ├── tearsheet_momentum_20d_ret_10d.html
            │   ├── ic_momentum_20d_ret_5d.csv
            │   ├── quantile_returns_momentum_20d_ret_5d.csv
            │   ├── ic_series_momentum_20d_ret_5d.png
            │   ├── ic_dist_momentum_20d_ret_5d.png
            │   ├── quantile_cumret_momentum_20d_ret_5d.png
            │   ├── quantile_meanret_momentum_20d_ret_5d.png
            │   ├── spread_cumret_momentum_20d_ret_5d.png
            │   ├── ic_heatmap_momentum_20d_ret_5d.png
            │   └── turnover_momentum_20d.png
            │
            ├── pe_ratio/
            │   ├── tearsheet_pe_ratio_ret_5d.html
            │   ├── ic_pe_ratio_ret_5d.csv
            │   └── ... (同上)
            │
            └── roe/
                ├── tearsheet_roe_ret_5d.html
                ├── ic_roe_ret_5d.csv
                └── ... (同上)
```

---

## 2. 文件类型说明

### 2.1 HTML报告 (Tearsheet)

**文件名格式**: `tearsheet_{factor_name}_{period}.html`

**用途**: 
- 因子综合评估报告
- 包含IC统计、分位数分析、Spread分析
- 嵌入所有可视化图表
- 自动因子质量评级（优秀/合格/弱）

**特点**:
- 响应式设计，支持移动端查看
- 包含完整的统计指标
- 图文并茂，易于理解
- 可独立分享

### 2.2 CSV数据文件

#### IC时间序列 (ic_*.csv)

**文件名格式**: `ic_{factor_name}_{period}.csv`

**用途**: IC值的逐日时间序列数据

**示例**:
```csv
date,ic
2023-01-04,0.0234
2023-01-05,0.0189
2023-01-06,0.0312
...
```

#### 分位数收益 (quantile_returns_*.csv)

**文件名格式**: `quantile_returns_{factor_name}_{period}.csv`

**用途**: 各分位数的逐日收益率

**示例**:
```csv
date,Q1,Q2,Q3,Q4,Q5
2023-01-04,-0.0012,0.0003,0.0008,0.0015,0.0025
2023-01-05,-0.0008,0.0005,0.0010,0.0018,0.0030
...
```

### 2.3 PNG图表文件

所有图表统一使用**300 DPI**高清输出，适合论文和报告使用。

#### 图表清单

| 图表类型 | 文件名 | 说明 |
|---------|--------|------|
| IC走廊图 | `ic_series_{factor}_{period}.png` | IC时间序列+均值线+±1σ区间 |
| IC分布图 | `ic_dist_{factor}_{period}.png` | IC直方图+正态拟合曲线+统计量 |
| 分位数累计收益 | `quantile_cumret_{factor}_{period}.png` | 各分位数净值曲线 |
| 分位数平均收益 | `quantile_meanret_{factor}_{period}.png` | 各分位数平均收益柱状图 |
| Spread累计收益 | `spread_cumret_{factor}_{period}.png` | Spread净值曲线 |
| IC月度热力图 | `ic_heatmap_{factor}_{period}.png` | 按月份统计的IC热力图 |
| 换手率时间序列 | `turnover_{factor}.png` | Top分位数换手率走势 |

---

## 3. 输出文件命名规范

### 3.1 因子名称规范

**规则**: 
- 使用小写字母+下划线
- 避免特殊字符和空格
- 保持简短有意义

**示例**:
- ✅ `momentum_20d`
- ✅ `pe_ratio`
- ✅ `roe_ttm`
- ❌ `Momentum-20D`
- ❌ `P/E Ratio`

### 3.2 收益期命名规范

**格式**: `ret_{N}d` 或 `ret_{N}w`

**示例**:
- `ret_1d` - 1日收益率
- `ret_5d` - 5日收益率
- `ret_10d` - 10日收益率
- `ret_20d` - 20日收益率
- `ret_1w` - 1周收益率

### 3.3 完整文件名示例

```
tearsheet_momentum_20d_ret_5d.html
ic_pe_ratio_ret_10d.csv
quantile_cumret_roe_ttm_ret_20d.png
turnover_value_factor.png
```

---

## 4. 详细文件格式

### 4.1 HTML报告详细结构

```html
<!DOCTYPE html>
<html>
<head>
    <title>因子评估报告 - {factor_name} @ {return_period}</title>
    <style>...</style>
</head>
<body>
    <div class="container">
        <!-- 1. 标题和基本信息 -->
        <h1>因子评估报告: {factor_name}</h1>
        <div class="info-box">
            <p>收益期: {return_period}</p>
            <p>生成时间: {timestamp}</p>
            <p>样本期间: {start_date} ~ {end_date}</p>
        </div>
        
        <!-- 2. IC统计摘要 -->
        <h2>📈 IC统计摘要</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">IC均值</div>
                <div class="metric-value">{ic_mean}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">ICIR</div>
                <div class="metric-value">{icir}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">ICIR(年化)</div>
                <div class="metric-value">{icir_annual}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">IC胜率</div>
                <div class="metric-value">{win_rate}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">t统计量</div>
                <div class="metric-value">{t_stat}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">p-value</div>
                <div class="metric-value">{p_value}</div>
            </div>
        </div>
        
        <!-- 3. 因子质量评级 -->
        <div class="quality-badge">
            {quality_rating}  <!-- 优秀因子/合格因子/弱因子 -->
        </div>
        
        <!-- 4. 分位数收益分析 -->
        <h2>📊 分位数收益分析</h2>
        <table>
            <thead>
                <tr>
                    <th>分位数</th>
                    <th>平均收益</th>
                    <th>累计收益</th>
                    <th>夏普比</th>
                </tr>
            </thead>
            <tbody>
                <!-- Q5 ~ Q1 数据 -->
            </tbody>
        </table>
        
        <!-- 5. Spread分析 -->
        <h2>📈 Spread分析</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Spread均值</div>
                <div class="metric-value">{spread_mean}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Spread夏普比(年化)</div>
                <div class="metric-value">{spread_sharpe}</div>
            </div>
        </div>
        
        <!-- 6. 单调性检验 -->
        <h2>📐 单调性检验</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Kendall τ</div>
                <div class="metric-value">{kendall_tau}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">单调顺序比例</div>
                <div class="metric-value">{correct_ratio}%</div>
            </div>
        </div>
        
        <!-- 7. 换手率统计 -->
        <h2>🔄 换手率统计</h2>
        <p>平均换手率: {mean_turnover}%</p>
        
        <!-- 8. 可视化图表 -->
        <h2>📊 可视化图表</h2>
        <img src="ic_series_{factor}_{period}.png" />
        <img src="ic_dist_{factor}_{period}.png" />
        <img src="quantile_cumret_{factor}_{period}.png" />
        <img src="quantile_meanret_{factor}_{period}.png" />
        <img src="spread_cumret_{factor}_{period}.png" />
        <img src="ic_heatmap_{factor}_{period}.png" />
        <img src="turnover_{factor}.png" />
        
        <!-- 9. 页脚 -->
        <div class="footer">
            <p>生成时间: {timestamp}</p>
            <p>框架版本: v1.0</p>
        </div>
    </div>
</body>
</html>
```

### 4.2 IC CSV格式详解

**文件**: `ic_{factor_name}_{period}.csv`

**格式**: 
```
列数: 2列 (date, ic)
索引: date (datetime)
编码: UTF-8
```

**完整示例**:
```csv
date,ic
2023-01-04,0.0234
2023-01-05,0.0189
2023-01-06,0.0312
2023-01-09,0.0278
2023-01-10,0.0301
...
2024-12-31,0.0267
```

**数据说明**:
- `date`: 交易日期（YYYY-MM-DD格式）
- `ic`: 当日横截面Spearman相关系数
- 包含所有交易日
- 缺失日期表示该日无有效数据

### 4.3 分位数收益CSV格式详解

**文件**: `quantile_returns_{factor_name}_{period}.csv`

**格式**: 
```
列数: 6列 (date, Q1, Q2, Q3, Q4, Q5) 或 11列 (date, Q1~Q10)
索引: date (datetime)
编码: UTF-8
```

**5分位示例**:
```csv
date,Q1,Q2,Q3,Q4,Q5
2023-01-04,-0.0012,0.0003,0.0008,0.0015,0.0025
2023-01-05,-0.0008,0.0005,0.0010,0.0018,0.0030
2023-01-06,-0.0015,-0.0002,0.0006,0.0012,0.0028
...
```

**10分位示例**:
```csv
date,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10
2023-01-04,-0.0018,-0.0012,-0.0006,0.0001,0.0007,0.0012,0.0016,0.0020,0.0024,0.0032
...
```

**数据说明**:
- `Q1`: 因子值最低分位数的平均收益率
- `Q5/Q10`: 因子值最高分位数的平均收益率
- 收益率为小数格式（0.0025 = 0.25%）
- 按日期升序排列

### 4.4 图表文件规格

**通用规格**:
- 格式: PNG
- 分辨率: 300 DPI
- 尺寸: 
  - IC走廊图: 14×5英寸
  - IC分布图: 10×6英寸
  - 累计收益图: 12×7英寸
  - 平均收益图: 10×6英寸
  - 热力图: 12×8英寸
- 字体: 支持中文显示
- 配色: 专业配色方案

**质量要求**:
- 线条清晰，无锯齿
- 文字可读，标签完整
- 图例位置合理
- 标题和坐标轴标签完整

---

## 5. 使用示例

### 5.1 生成单个因子完整报告

```python
from evaluation import CrossSectionAnalyzer
from evaluation.visualization import create_factor_tearsheet_plots
from evaluation.tearsheet import generate_full_tearsheet

# 1. 执行分析
analyzer = CrossSectionAnalyzer(
    factors=factors_df,
    prices=prices_df,
    forward_periods=[5],
    quantiles=5
)

results = analyzer.preprocess(
    winsorize=True,
    standardize=True,
    neutralize=True
).analyze()

# 2. 设置输出目录
output_dir = "ML output/reports/baseline_v1/factors/momentum_20d"

# 3. 生成图表
plot_paths = create_factor_tearsheet_plots(
    results,
    factor_name='momentum_20d',
    return_period='ret_5d',
    output_dir=output_dir
)

# 4. 生成完整报告（HTML + CSV）
generate_full_tearsheet(
    results,
    factor_name='momentum_20d',
    return_period='ret_5d',
    output_dir=output_dir,
    plot_paths=plot_paths
)
```

**生成的文件**:
```
ML output/reports/baseline_v1/factors/momentum_20d/
├── tearsheet_momentum_20d_ret_5d.html          ← HTML报告
├── ic_momentum_20d_ret_5d.csv                  ← IC数据
├── quantile_returns_momentum_20d_ret_5d.csv    ← 分位数收益
├── ic_series_momentum_20d_ret_5d.png           ← 7张图表
├── ic_dist_momentum_20d_ret_5d.png
├── quantile_cumret_momentum_20d_ret_5d.png
├── quantile_meanret_momentum_20d_ret_5d.png
├── spread_cumret_momentum_20d_ret_5d.png
├── ic_heatmap_momentum_20d_ret_5d.png
└── turnover_momentum_20d.png
```

### 5.2 批量生成多因子报告

```python
from evaluation import CrossSectionAnalyzer
from evaluation.tearsheet import generate_full_tearsheet
from evaluation.visualization import create_factor_tearsheet_plots

# 因子列表
factor_names = ['momentum_20d', 'pe_ratio', 'roe_ttm', 'revenue_yoy']

for factor_name in factor_names:
    print(f"\n{'='*60}")
    print(f"处理因子: {factor_name}")
    print(f"{'='*60}")
    
    # 提取单个因子
    factor_single = factors_df[[factor_name]].copy()
    
    # 创建分析器
    analyzer = CrossSectionAnalyzer(
        factors=factor_single,
        prices=prices_df,
        forward_periods=[5, 10],
        quantiles=5
    )
    
    # 执行分析
    results = analyzer.preprocess(
        winsorize=True,
        standardize=True,
        neutralize=True
    ).analyze()
    
    # 为每个收益期生成报告
    for period in ['ret_5d', 'ret_10d']:
        output_dir = f"ML output/reports/baseline_v1/factors/{factor_name}"
        
        # 生成图表
        plot_paths = create_factor_tearsheet_plots(
            results,
            factor_name=factor_name,
            return_period=period,
            output_dir=output_dir
        )
        
        # 生成完整报告
        generate_full_tearsheet(
            results,
            factor_name=factor_name,
            return_period=period,
            output_dir=output_dir,
            plot_paths=plot_paths
        )
```

### 5.3 读取已生成的数据

```python
import pandas as pd

# 读取IC数据
ic_data = pd.read_csv(
    'ML output/reports/baseline_v1/factors/momentum_20d/ic_momentum_20d_ret_5d.csv',
    index_col='date',
    parse_dates=True
)

print(f"IC均值: {ic_data['ic'].mean():.4f}")
print(f"IC标准差: {ic_data['ic'].std():.4f}")
print(f"ICIR: {ic_data['ic'].mean() / ic_data['ic'].std():.4f}")

# 读取分位数收益
quantile_rets = pd.read_csv(
    'ML output/reports/baseline_v1/factors/momentum_20d/quantile_returns_momentum_20d_ret_5d.csv',
    index_col='date',
    parse_dates=True
)

print(f"\nQ5平均收益: {quantile_rets['Q5'].mean():.4f}")
print(f"Q1平均收益: {quantile_rets['Q1'].mean():.4f}")
print(f"Spread: {(quantile_rets['Q5'] - quantile_rets['Q1']).mean():.4f}")
```

---

## 6. 输出管理最佳实践

### 6.1 版本管理

**建议**: 使用版本号区分不同实验

```
ML output/reports/
├── baseline_v1/          # 初始版本
├── baseline_v2/          # 添加新特征
├── baseline_v3/          # 优化参数
└── production_v1/        # 生产环境版本
```

### 6.2 文件清理

**定期清理策略**:
- 保留最近3个版本的完整输出
- 旧版本只保留HTML和CSV，删除PNG
- 使用压缩包归档历史版本

**清理脚本示例**:
```python
import os
import shutil
from datetime import datetime, timedelta

def cleanup_old_outputs(reports_dir, keep_versions=3):
    """清理旧版本输出"""
    versions = sorted([d for d in os.listdir(reports_dir) 
                      if d.startswith('baseline_v')],
                     reverse=True)
    
    for version in versions[keep_versions:]:
        version_path = os.path.join(reports_dir, version)
        print(f"清理旧版本: {version}")
        
        # 只删除PNG文件，保留HTML和CSV
        for root, dirs, files in os.walk(version_path):
            for file in files:
                if file.endswith('.png'):
                    os.remove(os.path.join(root, file))
```

### 6.3 输出验证

**生成后检查清单**:
- [ ] HTML文件可正常打开
- [ ] CSV文件编码正确（UTF-8）
- [ ] PNG图表清晰可见
- [ ] 所有文件命名规范
- [ ] 文件大小合理（HTML < 5MB, PNG < 1MB, CSV < 10MB）

---

## 7. 故障排查

### Q1: HTML报告打不开或显示异常？

**可能原因**:
- 图表路径错误
- 编码问题
- CSS样式缺失

**解决方案**:
```python
# 使用绝对路径或相对路径
plot_paths = {
    'ic_series': './ic_series_factor_ret_5d.png',  # 相对路径
    # 或
    'ic_series': os.path.abspath('ic_series_factor_ret_5d.png')  # 绝对路径
}
```

### Q2: CSV文件中文乱码？

**解决方案**:
```python
# 读取时指定编码
df = pd.read_csv('file.csv', encoding='utf-8')

# 保存时指定编码
df.to_csv('file.csv', encoding='utf-8-sig')  # 带BOM
```

### Q3: 图表显示不完整？

**可能原因**:
- 数据范围过大
- 坐标轴标签重叠
- 图例超出范围

**解决方案**:
```python
# 调整图表尺寸
fig, ax = plt.subplots(figsize=(14, 6))

# 调整布局
plt.tight_layout()

# 保存时使用bbox_inches
fig.savefig('plot.png', dpi=300, bbox_inches='tight')
```

---

## 8. 常见问题

### Q1: 为什么需要多个收益期的报告？

**A**: 不同收益期可能展现因子的不同特性：
- `ret_1d`: 短期反转/动量
- `ret_5d`: 周度趋势
- `ret_10d`: 双周趋势
- `ret_20d`: 月度趋势

建议至少评估2-3个收益期。

### Q2: 如何选择分位数？

**A**: 
- **5分位**: 标准选择，适合大部分场景
- **10分位**: 更精细，适合样本量大的情况
- **3分位**: 样本量小时使用

### Q3: 图表DPI为什么是300？

**A**: 
- 300 DPI是印刷质量标准
- 适合论文、报告使用
- 72 DPI仅适合屏幕显示

### Q4: CSV和HTML哪个更重要？

**A**: 
- **HTML**: 用于快速查看和分享
- **CSV**: 用于二次分析和验证
- 建议都保留，CSV文件不大

---

## 9. 更新日志

### v1.0 (2025-11-19)

**初始版本**:
- ✅ 定义标准输出目录结构
- ✅ 规范文件命名
- ✅ 详细文档HTML、CSV、PNG格式
- ✅ 提供使用示例和最佳实践
- ✅ 故障排查指南

---

## 附录

### A. 文件大小参考

| 文件类型 | 典型大小 | 说明 |
|---------|---------|------|
| HTML | 500KB - 2MB | 取决于嵌入图表数量 |
| IC CSV | 10KB - 100KB | 取决于样本数 |
| 分位数CSV | 10KB - 100KB | 取决于样本数和分位数 |
| PNG图表 | 100KB - 500KB | 300 DPI高清 |

### B. 推荐工具

**查看HTML**:
- Chrome / Edge浏览器
- Firefox浏览器

**处理CSV**:
- Excel / WPS
- Python pandas
- R

**查看PNG**:
- 系统默认图片查看器
- Adobe Acrobat（打印PDF时）

---

**文档维护者:** AI Assistant  
**最后更新:** 2025-11-19  
**版本:** v1.0
