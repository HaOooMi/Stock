#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
横截面评估可视化模块

核心图表：
1. IC时间序列图（IC走廊图）
2. IC分布直方图
3. 分位数累计收益图
4. 分位数平均收益柱状图
5. Spread累计收益图
6. 换手率时间序列图
7. 月度IC热力图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置风格
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


def plot_ic_time_series(ic_series: pd.Series,
                        title: str = "IC Time Series",
                        figsize: Tuple[int, int] = (14, 6),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制IC时间序列图（走廊图）
    
    Parameters:
    -----------
    ic_series : pd.Series
        IC时间序列，index为date
    title : str
        图表标题
    figsize : Tuple[int, int]
        图表大小
    save_path : str, optional
        保存路径
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制IC
    ic_series.plot(ax=ax, linewidth=1.5, alpha=0.8, color='steelblue')
    
    # 添加零线
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # 添加±1标准差区间
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    
    ax.axhline(y=mean_ic, color='red', linestyle='-', linewidth=1.5, 
               label=f'Mean IC: {mean_ic:.4f}')
    ax.axhline(y=mean_ic + std_ic, color='orange', linestyle='--', 
               linewidth=1, alpha=0.7, label=f'+1σ: {mean_ic + std_ic:.4f}')
    ax.axhline(y=mean_ic - std_ic, color='orange', linestyle='--', 
               linewidth=1, alpha=0.7, label=f'-1σ: {mean_ic - std_ic:.4f}')
    
    # 填充正负IC区域
    ax.fill_between(ic_series.index, 0, ic_series, 
                    where=(ic_series > 0), alpha=0.3, color='green', 
                    label='Positive IC')
    ax.fill_between(ic_series.index, 0, ic_series, 
                    where=(ic_series < 0), alpha=0.3, color='red', 
                    label='Negative IC')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('IC', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 保存图表: {save_path}")
    
    return fig


def plot_ic_distribution(ic_series: pd.Series,
                        title: str = "IC Distribution",
                        figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制IC分布直方图
    
    Parameters:
    -----------
    ic_series : pd.Series
        IC时间序列
    title : str
        图表标题
    figsize : Tuple[int, int]
        图表大小
    save_path : str, optional
        保存路径
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 直方图
    ic_series.hist(bins=50, ax=ax, color='steelblue', alpha=0.7, edgecolor='black')
    
    # 添加统计信息
    mean_ic = ic_series.mean()
    median_ic = ic_series.median()
    std_ic = ic_series.std()
    
    ax.axvline(x=mean_ic, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_ic:.4f}')
    ax.axvline(x=median_ic, color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {median_ic:.4f}')
    
    # 添加正态分布曲线
    from scipy import stats
    x = np.linspace(ic_series.min(), ic_series.max(), 100)
    pdf = stats.norm.pdf(x, mean_ic, std_ic)
    
    # 缩放PDF以匹配直方图
    ax2 = ax.twinx()
    ax2.plot(x, pdf, 'r-', linewidth=2, alpha=0.6, label='Normal Distribution')
    ax2.set_ylabel('Probability Density', fontsize=12)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('IC', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 保存图表: {save_path}")
    
    return fig


def plot_quantile_cumulative_returns(cumulative_returns: pd.DataFrame,
                                     title: str = "Quantile Cumulative Returns",
                                     figsize: Tuple[int, int] = (14, 8),
                                     save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制分位数累计收益图（净值曲线）
    
    Parameters:
    -----------
    cumulative_returns : pd.DataFrame
        累计收益，index=date, columns=[Q1, Q2, ..., Qn]
    title : str
        图表标题
    figsize : Tuple[int, int]
        图表大小
    save_path : str, optional
        保存路径
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 颜色映射（从红到绿）
    n_quantiles = len(cumulative_returns.columns)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_quantiles))
    
    # 绘制每个分位数
    for i, col in enumerate(cumulative_returns.columns):
        cumulative_returns[col].plot(
            ax=ax,
            linewidth=2,
            alpha=0.8,
            color=colors[i],
            label=col
        )
    
    # 突出Top和Bottom
    cumulative_returns.iloc[:, -1].plot(
        ax=ax, linewidth=3, color='darkgreen', 
        label=f'{cumulative_returns.columns[-1]} (Top)', linestyle='--'
    )
    cumulative_returns.iloc[:, 0].plot(
        ax=ax, linewidth=3, color='darkred', 
        label=f'{cumulative_returns.columns[0]} (Bottom)', linestyle='--'
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (Net Value)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 添加1.0基准线
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 保存图表: {save_path}")
    
    return fig


def plot_quantile_mean_returns(quantile_returns: pd.DataFrame,
                               title: str = "Quantile Mean Returns",
                               figsize: Tuple[int, int] = (10, 6),
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制分位数平均收益柱状图
    
    Parameters:
    -----------
    quantile_returns : pd.DataFrame
        分位数日收益，columns=[Q1, Q2, ..., Qn]
    title : str
        图表标题
    figsize : Tuple[int, int]
        图表大小
    save_path : str, optional
        保存路径
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 计算平均收益
    mean_returns = quantile_returns.mean()
    
    # 颜色（正为绿，负为红）
    colors = ['green' if x > 0 else 'red' for x in mean_returns]
    
    # 绘制柱状图
    bars = ax.bar(range(len(mean_returns)), mean_returns, 
                  color=colors, alpha=0.7, edgecolor='black')
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, mean_returns)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{val:.4f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Quantile', fontsize=12)
    ax.set_ylabel('Mean Return', fontsize=12)
    ax.set_xticks(range(len(mean_returns)))
    ax.set_xticklabels(mean_returns.index)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 保存图表: {save_path}")
    
    return fig


def plot_spread_cumulative_returns(spread_series: pd.Series,
                                   title: str = "Spread Cumulative Returns",
                                   figsize: Tuple[int, int] = (14, 6),
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制Spread累计收益图
    
    Parameters:
    -----------
    spread_series : pd.Series
        Spread日收益序列
    title : str
        图表标题
    figsize : Tuple[int, int]
        图表大小
    save_path : str, optional
        保存路径
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 计算累计收益
    cumulative_spread = (1 + spread_series).cumprod()
    
    # 绘制累计收益
    cumulative_spread.plot(ax=ax, linewidth=2, color='purple', alpha=0.8)
    
    # 添加统计信息
    final_return = cumulative_spread.iloc[-1] - 1
    sharpe = spread_series.mean() / spread_series.std() if spread_series.std() != 0 else 0
    sharpe_annual = sharpe * np.sqrt(252)
    
    stats_text = (
        f"Final Return: {final_return:.2%}\n"
        f"Sharpe: {sharpe:.4f}\n"
        f"Sharpe(Annual): {sharpe_annual:.4f}"
    )
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (Net Value)', fontsize=12)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 保存图表: {save_path}")
    
    return fig


def plot_turnover_time_series(turnover_series: pd.Series,
                              title: str = "Turnover Time Series",
                              figsize: Tuple[int, int] = (14, 6),
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制换手率时间序列图
    
    Parameters:
    -----------
    turnover_series : pd.Series
        换手率时间序列
    title : str
        图表标题
    figsize : Tuple[int, int]
        图表大小
    save_path : str, optional
        保存路径
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制换手率
    turnover_series.plot(ax=ax, linewidth=1.5, alpha=0.8, color='coral')
    
    # 添加平均线
    mean_turnover = turnover_series.mean()
    ax.axhline(y=mean_turnover, color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {mean_turnover:.2%}')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Turnover Rate', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 格式化y轴为百分比
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 保存图表: {save_path}")
    
    return fig


def plot_monthly_ic_heatmap(ic_series: pd.Series,
                            title: str = "Monthly IC Heatmap",
                            figsize: Tuple[int, int] = (14, 8),
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制月度IC热力图
    
    Parameters:
    -----------
    ic_series : pd.Series
        IC时间序列
    title : str
        图表标题
    figsize : Tuple[int, int]
        图表大小
    save_path : str, optional
        保存路径
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # 转换为月度平均
    monthly_ic = ic_series.resample('M').mean()
    
    # 提取年份和月份
    monthly_ic.index = pd.to_datetime(monthly_ic.index)
    monthly_ic_df = pd.DataFrame({
        'year': monthly_ic.index.year,
        'month': monthly_ic.index.month,
        'ic': monthly_ic.values
    })
    
    # 透视表
    pivot_table = monthly_ic_df.pivot(index='month', columns='year', values='ic')
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(pivot_table, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlGn',
                center=0,
                cbar_kws={'label': 'IC'},
                linewidths=0.5,
                ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Month', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 保存图表: {save_path}")
    
    return fig


def create_factor_tearsheet_plots(analyzer_results: Dict,
                                  factor_name: str,
                                  return_period: str,
                                  output_dir: str) -> Dict[str, str]:
    """
    为单个因子生成全套图表
    
    Parameters:
    -----------
    analyzer_results : dict
        CrossSectionAnalyzer.get_results()的返回值
    factor_name : str
        因子名称
    return_period : str
        收益期（如'ret_1d'）
    output_dir : str
        输出目录
        
    Returns:
    --------
    dict
        {图表名称: 文件路径}
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    key = (factor_name, return_period)
    plot_paths = {}
    
    print(f"\n📊 生成因子图表: {factor_name} @ {return_period}")
    print("-" * 70)
    
    # 1. IC时间序列
    if 'daily_ic' in analyzer_results and key in analyzer_results['daily_ic'].columns:
        ic_series = analyzer_results['daily_ic'][key]
        path = os.path.join(output_dir, f"ic_series_{factor_name}_{return_period}.png")
        plot_ic_time_series(
            ic_series,
            title=f"IC Time Series: {factor_name} @ {return_period}",
            save_path=path
        )
        plot_paths['ic_series'] = path
    
    # 2. IC分布
    if 'daily_ic' in analyzer_results and key in analyzer_results['daily_ic'].columns:
        ic_series = analyzer_results['daily_ic'][key]
        path = os.path.join(output_dir, f"ic_dist_{factor_name}_{return_period}.png")
        plot_ic_distribution(
            ic_series,
            title=f"IC Distribution: {factor_name} @ {return_period}",
            save_path=path
        )
        plot_paths['ic_distribution'] = path
    
    # 3. 分位数累计收益
    if 'cumulative_returns' in analyzer_results and key in analyzer_results['cumulative_returns']:
        cum_rets = analyzer_results['cumulative_returns'][key]
        path = os.path.join(output_dir, f"quantile_cumret_{factor_name}_{return_period}.png")
        plot_quantile_cumulative_returns(
            cum_rets,
            title=f"Quantile Cumulative Returns: {factor_name} @ {return_period}",
            save_path=path
        )
        plot_paths['cumulative_returns'] = path
    
    # 4. 分位数平均收益
    if 'quantile_returns' in analyzer_results and key in analyzer_results['quantile_returns']:
        q_rets = analyzer_results['quantile_returns'][key]
        path = os.path.join(output_dir, f"quantile_meanret_{factor_name}_{return_period}.png")
        plot_quantile_mean_returns(
            q_rets,
            title=f"Quantile Mean Returns: {factor_name} @ {return_period}",
            save_path=path
        )
        plot_paths['mean_returns'] = path
    
    # 5. Spread累计收益
    if 'spreads' in analyzer_results and key in analyzer_results['spreads']:
        spread = analyzer_results['spreads'][key]
        path = os.path.join(output_dir, f"spread_cumret_{factor_name}_{return_period}.png")
        plot_spread_cumulative_returns(
            spread,
            title=f"Spread Cumulative Returns: {factor_name} @ {return_period}",
            save_path=path
        )
        plot_paths['spread_cumulative'] = path
    
    # 6. 月度IC热力图
    if 'daily_ic' in analyzer_results and key in analyzer_results['daily_ic'].columns:
        ic_series = analyzer_results['daily_ic'][key]
        path = os.path.join(output_dir, f"ic_heatmap_{factor_name}_{return_period}.png")
        plot_monthly_ic_heatmap(
            ic_series,
            title=f"Monthly IC Heatmap: {factor_name} @ {return_period}",
            save_path=path
        )
        plot_paths['ic_heatmap'] = path
    
    # 7. 换手率（如果有单因子数据）
    if 'turnover_stats' in analyzer_results and factor_name in analyzer_results['turnover_stats']:
        turnover = analyzer_results['turnover_stats'][factor_name]['turnover_series']['turnover']
        path = os.path.join(output_dir, f"turnover_{factor_name}.png")
        plot_turnover_time_series(
            turnover,
            title=f"Turnover Time Series: {factor_name}",
            save_path=path
        )
        plot_paths['turnover'] = path
    
    print(f"✅ 生成{len(plot_paths)}个图表")
    
    return plot_paths


if __name__ == '__main__':
    """测试代码"""
    print("=" * 70)
    print("可视化模块测试")
    print("=" * 70)
    
    # 生成测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # IC序列
    ic_series = pd.Series(
        np.random.randn(len(dates)) * 0.05 + 0.02,
        index=dates
    )
    
    # 分位数收益
    quantile_returns = pd.DataFrame({
        'Q1': np.random.randn(len(dates)) * 0.02 - 0.001,
        'Q2': np.random.randn(len(dates)) * 0.02,
        'Q3': np.random.randn(len(dates)) * 0.02 + 0.0005,
        'Q4': np.random.randn(len(dates)) * 0.02 + 0.001,
        'Q5': np.random.randn(len(dates)) * 0.02 + 0.002,
    }, index=dates)
    
    cumulative_returns = (1 + quantile_returns).cumprod()
    
    # Spread
    spread = quantile_returns['Q5'] - quantile_returns.mean(axis=1)
    
    # 换手率
    turnover = pd.Series(
        np.random.rand(len(dates)) * 0.3 + 0.2,
        index=dates
    )
    
    print("\n测试各类图表...")
    
    print("\n1. IC时间序列图...")
    plot_ic_time_series(ic_series)
    
    print("\n2. IC分布图...")
    plot_ic_distribution(ic_series)
    
    print("\n3. 分位数累计收益图...")
    plot_quantile_cumulative_returns(cumulative_returns)
    
    print("\n4. 分位数平均收益图...")
    plot_quantile_mean_returns(quantile_returns)
    
    print("\n5. Spread累计收益图...")
    plot_spread_cumulative_returns(spread)
    
    print("\n6. 换手率时间序列图...")
    plot_turnover_time_series(turnover)
    
    print("\n7. 月度IC热力图...")
    plot_monthly_ic_heatmap(ic_series)
    
    plt.show()
    
    print("\n✅ 测试完成！")
