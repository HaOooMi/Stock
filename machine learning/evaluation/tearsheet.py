#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tearsheet报表生成模块

生成HTML格式的综合评估报告和CSV数据输出
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional
from datetime import datetime
import json


def generate_html_tearsheet(analyzer_results: Dict,
                           factor_name: str,
                           return_period: str,
                           output_path: str,
                           plot_paths: Optional[Dict[str, str]] = None):
    """
    生成HTML格式的Tearsheet报告
    
    Parameters:
    -----------
    analyzer_results : dict
        CrossSectionAnalyzer.get_results()
    factor_name : str
        因子名称
    return_period : str
        收益期
    output_path : str
        输出HTML文件路径
    plot_paths : dict, optional
        图表路径字典
    """
    key = (factor_name, return_period)
    
    # HTML模板
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>因子评估报告 - {factor_name} @ {return_period}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                border-left: 4px solid #3498db;
                padding-left: 10px;
                margin-top: 30px;
            }}
            .summary-box {{
                background-color: #ecf0f1;
                border-left: 5px solid #3498db;
                padding: 15px;
                margin: 20px 0;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .metric-card {{
                background-color: #ffffff;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            .metric-label {{
                font-size: 0.9em;
                color: #7f8c8d;
                margin-bottom: 5px;
            }}
            .metric-value {{
                font-size: 1.5em;
                font-weight: bold;
                color: #2c3e50;
            }}
            .metric-value.positive {{
                color: #27ae60;
            }}
            .metric-value.negative {{
                color: #e74c3c;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .plot-container {{
                margin: 30px 0;
                text-align: center;
            }}
            .plot-container img {{
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .footer {{
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                text-align: center;
                color: #7f8c8d;
                font-size: 0.9em;
            }}
            .alert {{
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            .alert-success {{
                background-color: #d4edda;
                border-left: 5px solid #28a745;
                color: #155724;
            }}
            .alert-warning {{
                background-color: #fff3cd;
                border-left: 5px solid #ffc107;
                color: #856404;
            }}
            .alert-danger {{
                background-color: #f8d7da;
                border-left: 5px solid #dc3545;
                color: #721c24;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 因子评估报告 (Alphalens风格)</h1>
            
            <div class="summary-box">
                <p><strong>因子名称:</strong> {factor_name}</p>
                <p><strong>收益期:</strong> {return_period}</p>
                <p><strong>生成时间:</strong> {timestamp}</p>
            </div>
            
            <h2>1. IC统计 (Information Coefficient)</h2>
            <div class="metric-grid">
                {ic_metrics}
            </div>
            
            {ic_alert}
            
            <h2>2. Spread统计</h2>
            <div class="metric-grid">
                {spread_metrics}
            </div>
            
            {spread_alert}
            
            <h2>3. 单调性检验</h2>
            <div class="metric-grid">
                {monotonicity_metrics}
            </div>
            
            <h2>4. 分位数收益统计</h2>
            {quantile_table}
            
            <h2>5. 换手率统计</h2>
            {turnover_section}
            
            <h2>6. 可视化图表</h2>
            {plots_section}
            
            <div class="footer">
                <p>Powered by 横截面评估框架 (Alphalens风格) | 生成于 {timestamp}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 准备数据
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # IC指标
    ic_summary = analyzer_results.get('ic_summary', {}).get(key, {})
    ic_metrics_html = f"""
        <div class="metric-card">
            <div class="metric-label">Mean IC</div>
            <div class="metric-value {get_value_class(ic_summary.get('mean', 0))}">{ic_summary.get('mean', 0):.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">IC Standard Deviation</div>
            <div class="metric-value">{ic_summary.get('std', 0):.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">ICIR</div>
            <div class="metric-value {get_value_class(ic_summary.get('icir', 0))}">{ic_summary.get('icir', 0):.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">ICIR (年化)</div>
            <div class="metric-value {get_value_class(ic_summary.get('icir_annual', 0))}">{ic_summary.get('icir_annual', 0):.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">t-statistic</div>
            <div class="metric-value">{ic_summary.get('t_stat', 0):.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">p-value</div>
            <div class="metric-value">{ic_summary.get('p_value', 1):.6f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">正IC比例</div>
            <div class="metric-value">{ic_summary.get('positive_ratio', 0):.2%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">观测数</div>
            <div class="metric-value">{ic_summary.get('n_obs', 0)}</div>
        </div>
    """
    
    # IC评估
    ic_alert_html = generate_ic_alert(ic_summary)
    
    # Spread指标
    spread_summary = analyzer_results.get('spread_summaries', {}).get(key, {})
    spread_metrics_html = f"""
        <div class="metric-card">
            <div class="metric-label">Mean Spread</div>
            <div class="metric-value {get_value_class(spread_summary.get('mean', 0))}">{spread_summary.get('mean', 0):.6f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Spread Std</div>
            <div class="metric-value">{spread_summary.get('std', 0):.6f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value {get_value_class(spread_summary.get('sharpe', 0))}">{spread_summary.get('sharpe', 0):.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sharpe (年化)</div>
            <div class="metric-value {get_value_class(spread_summary.get('sharpe_annual', 0))}">{spread_summary.get('sharpe_annual', 0):.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">正Spread比例</div>
            <div class="metric-value">{spread_summary.get('positive_ratio', 0):.2%}</div>
        </div>
    """
    
    # Spread评估
    spread_alert_html = generate_spread_alert(spread_summary)
    
    # 单调性指标
    monotonicity = analyzer_results.get('monotonicities', {}).get(key, {})
    monotonicity_metrics_html = f"""
        <div class="metric-card">
            <div class="metric-label">Kendall τ</div>
            <div class="metric-value {get_value_class(monotonicity.get('kendall_tau', 0))}">{monotonicity.get('kendall_tau', 0):.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Kendall p-value</div>
            <div class="metric-value">{monotonicity.get('kendall_p_value', 1):.6f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">正确排序比例</div>
            <div class="metric-value">{monotonicity.get('correct_order_ratio', 0):.2%}</div>
        </div>
    """
    
    # 分位数收益表格
    quantile_table_html = generate_quantile_table(
        analyzer_results.get('quantile_returns', {}).get(key)
    )
    
    # 换手率
    turnover_html = generate_turnover_section(
        analyzer_results.get('turnover_stats', {}).get(factor_name)
    )
    
    # 图表部分
    plots_html = generate_plots_section(plot_paths)
    
    # 填充模板
    html_content = html_template.format(
        factor_name=factor_name,
        return_period=return_period,
        timestamp=timestamp,
        ic_metrics=ic_metrics_html,
        ic_alert=ic_alert_html,
        spread_metrics=spread_metrics_html,
        spread_alert=spread_alert_html,
        monotonicity_metrics=monotonicity_metrics_html,
        quantile_table=quantile_table_html,
        turnover_section=turnover_html,
        plots_section=plots_html
    )
    
    # 写入文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"   💾 保存HTML报告: {output_path}")


def get_value_class(value: float) -> str:
    """根据值返回CSS类"""
    if value > 0:
        return 'positive'
    elif value < 0:
        return 'negative'
    else:
        return ''


def generate_ic_alert(ic_summary: Dict) -> str:
    """生成IC评估提示"""
    mean_ic = ic_summary.get('mean', 0)
    icir_annual = ic_summary.get('icir_annual', 0)
    p_value = ic_summary.get('p_value', 1)
    
    if mean_ic > 0.03 and icir_annual > 1.5 and p_value < 0.01:
        return '<div class="alert alert-success">✅ <strong>优秀因子:</strong> IC显著为正，ICIR优秀，具有强预测能力</div>'
    elif mean_ic > 0.01 and icir_annual > 0.5 and p_value < 0.05:
        return '<div class="alert alert-warning">⚠️ <strong>合格因子:</strong> IC为正且显著，但ICIR偏低，建议组合使用</div>'
    else:
        return '<div class="alert alert-danger">❌ <strong>弱因子:</strong> IC不显著或为负，预测能力不足</div>'


def generate_spread_alert(spread_summary: Dict) -> str:
    """生成Spread评估提示"""
    mean_spread = spread_summary.get('mean', 0)
    sharpe_annual = spread_summary.get('sharpe_annual', 0)
    
    if mean_spread > 0 and sharpe_annual > 1.0:
        return '<div class="alert alert-success">✅ <strong>Spread显著:</strong> 多空策略有效，夏普比优秀</div>'
    elif mean_spread > 0 and sharpe_annual > 0.5:
        return '<div class="alert alert-warning">⚠️ <strong>Spread有效:</strong> 多空策略可用，但夏普比偏低</div>'
    else:
        return '<div class="alert alert-danger">❌ <strong>Spread无效:</strong> 多空策略无明显优势</div>'


def generate_quantile_table(quantile_returns: Optional[pd.DataFrame]) -> str:
    """生成分位数收益表格"""
    if quantile_returns is None:
        return '<p>暂无数据</p>'
    
    # 计算统计量
    mean_rets = quantile_returns.mean()
    std_rets = quantile_returns.std()
    sharpe_rets = mean_rets / std_rets
    
    table_html = '<table><thead><tr><th>分位数</th><th>平均收益</th><th>标准差</th><th>夏普比</th></tr></thead><tbody>'
    
    for q in quantile_returns.columns:
        table_html += f"""
        <tr>
            <td>{q}</td>
            <td>{mean_rets[q]:.6f}</td>
            <td>{std_rets[q]:.6f}</td>
            <td>{sharpe_rets[q]:.4f}</td>
        </tr>
        """
    
    table_html += '</tbody></table>'
    
    return table_html


def generate_turnover_section(turnover_stats: Optional[Dict]) -> str:
    """生成换手率部分"""
    if turnover_stats is None:
        return '<p>暂无换手率数据</p>'
    
    mean_turnover = turnover_stats.get('mean_turnover', 0)
    std_turnover = turnover_stats.get('std_turnover', 0)
    
    return f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">平均换手率</div>
            <div class="metric-value">{mean_turnover:.2%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">换手率标准差</div>
            <div class="metric-value">{std_turnover:.2%}</div>
        </div>
    </div>
    """


def generate_plots_section(plot_paths: Optional[Dict[str, str]]) -> str:
    """生成图表部分"""
    if not plot_paths:
        return '<p>暂无图表</p>'
    
    plots_html = ''
    
    plot_titles = {
        'ic_series': 'IC时间序列图',
        'ic_distribution': 'IC分布图',
        'cumulative_returns': '分位数累计收益图',
        'mean_returns': '分位数平均收益图',
        'spread_cumulative': 'Spread累计收益图',
        'ic_heatmap': '月度IC热力图',
        'turnover': '换手率时间序列图'
    }
    
    for key, path in plot_paths.items():
        title = plot_titles.get(key, key)
        # 使用相对路径
        rel_path = os.path.basename(path)
        plots_html += f"""
        <div class="plot-container">
            <h3>{title}</h3>
            <img src="{rel_path}" alt="{title}">
        </div>
        """
    
    return plots_html


def save_ic_to_csv(analyzer_results: Dict,
                   factor_name: str,
                   return_period: str,
                   output_path: str):
    """保存IC序列到CSV"""
    key = (factor_name, return_period)
    
    if 'daily_ic' not in analyzer_results or key not in analyzer_results['daily_ic'].columns:
        print(f"   ⚠️  未找到IC数据")
        return
    
    ic_series = analyzer_results['daily_ic'][key]
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ic_series.to_csv(output_path, header=['ic'])
    
    print(f"   💾 保存IC数据: {output_path}")


def save_quantile_returns_to_csv(analyzer_results: Dict,
                                 factor_name: str,
                                 return_period: str,
                                 output_path: str):
    """保存分位数收益到CSV"""
    key = (factor_name, return_period)
    
    if 'quantile_returns' not in analyzer_results or key not in analyzer_results['quantile_returns']:
        print(f"   ⚠️  未找到分位数收益数据")
        return
    
    quantile_rets = analyzer_results['quantile_returns'][key]
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    quantile_rets.to_csv(output_path)
    
    print(f"   💾 保存分位数收益: {output_path}")


def generate_full_tearsheet(analyzer_results: Dict,
                           factor_name: str,
                           return_period: str,
                           output_dir: str,
                           plot_paths: Optional[Dict[str, str]] = None):
    """
    生成完整的Tearsheet（HTML + CSV）
    
    Parameters:
    -----------
    analyzer_results : dict
        分析结果
    factor_name : str
        因子名称
    return_period : str
        收益期
    output_dir : str
        输出目录
    plot_paths : dict, optional
        图表路径
    """
    print(f"\n📄 生成Tearsheet报告: {factor_name} @ {return_period}")
    print("-" * 70)
    
    # HTML报告
    html_path = os.path.join(output_dir, f"tearsheet_{factor_name}_{return_period}.html")
    generate_html_tearsheet(
        analyzer_results,
        factor_name,
        return_period,
        html_path,
        plot_paths
    )
    
    # IC CSV
    ic_csv_path = os.path.join(output_dir, f"ic_{factor_name}_{return_period}.csv")
    save_ic_to_csv(analyzer_results, factor_name, return_period, ic_csv_path)
    
    # 分位数收益CSV
    quantile_csv_path = os.path.join(output_dir, f"quantile_returns_{factor_name}_{return_period}.csv")
    save_quantile_returns_to_csv(analyzer_results, factor_name, return_period, quantile_csv_path)
    
    print("✅ Tearsheet生成完成\n")


if __name__ == '__main__':
    """测试代码"""
    print("=" * 70)
    print("Tearsheet生成模块测试")
    print("=" * 70)
    
    # 模拟结果数据
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    mock_results = {
        'ic_summary': {
            ('factor_1', 'ret_5d'): {
                'mean': 0.045,
                'std': 0.12,
                'icir': 0.375,
                'icir_annual': 5.96,
                't_stat': 5.85,
                'p_value': 0.00001,
                'n_obs': 365,
                'positive_ratio': 0.65
            }
        },
        'spread_summaries': {
            ('factor_1', 'ret_5d'): {
                'mean': 0.0025,
                'std': 0.015,
                'sharpe': 0.167,
                'sharpe_annual': 2.65,
                'positive_ratio': 0.58
            }
        },
        'monotonicities': {
            ('factor_1', 'ret_5d'): {
                'kendall_tau': 0.85,
                'kendall_p_value': 0.001,
                'correct_order_ratio': 0.72
            }
        },
        'daily_ic': pd.DataFrame({
            ('factor_1', 'ret_5d'): np.random.randn(len(dates)) * 0.1 + 0.045
        }, index=dates),
        'quantile_returns': {
            ('factor_1', 'ret_5d'): pd.DataFrame({
                'Q1': np.random.randn(len(dates)) * 0.02 - 0.001,
                'Q2': np.random.randn(len(dates)) * 0.02,
                'Q3': np.random.randn(len(dates)) * 0.02 + 0.0005,
                'Q4': np.random.randn(len(dates)) * 0.02 + 0.001,
                'Q5': np.random.randn(len(dates)) * 0.02 + 0.002,
            }, index=dates)
        },
        'turnover_stats': {
            'factor_1': {
                'mean_turnover': 0.25,
                'std_turnover': 0.08
            }
        }
    }
    
    print("\n生成测试报告...")
    generate_full_tearsheet(
        mock_results,
        'factor_1',
        'ret_5d',
        './test_output'
    )
    
    print("\n✅ 测试完成！")
