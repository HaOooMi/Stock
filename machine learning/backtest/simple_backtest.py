#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简易组合回测器 - Simple Portfolio Backtester

功能：
1. 接收模型预测分数 (Score)
2. 执行 Top-K 等权选股（支持按分数加权）
3. Open-to-Open 执行模式：T日收盘生成信号，T+1日开盘执行
4. 支持调仓频率：日度/周度/月度
5. 基准对比与超额收益分析
6. 计算扣费后的净值曲线与核心指标
7. 丰富的可视化：净值曲线、月度收益热力图、回撤分析

设计原则：
- 极简主义：基于向量化计算，无事件驱动
- 真实性：T+1 Execution Lag + 交易成本
- 可解释性：输出持仓、换手、成本等中间产物
- 接口清晰：输入 Predictions + Market Data -> 输出 Stats + Curve + Plots

关键改进（v2.0）：
- 移除双模式对比，专注 Open-to-Open
- 新增调仓频率控制（减少过度交易）
- 新增基准对比与 Alpha/Beta 分析
- 新增月度收益热力图
- 标准化输出落盘（weights/returns/stats）

创建: 2025-12-09 | 更新: 2025-12-19 | 版本: v2.0
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class SimplePortfolioBacktester:
    """
    简易组合回测器 (Vectorized Backtester)
    
    核心逻辑：
    1. 每日根据预测分数 (Score) 排序
    2. 选择 Top-K 股票构建组合
    3. 采用等权重 (Equal Weight) 分配
    4. 支持 T+0 (理想) 和 T+1 (现实) 两种执行模式
    5. 扣除双边交易成本
    
    用于验证模型在真实执行延迟下的有效性。
    """
    
    def __init__(self,
                 top_k: int = 30,
                 commission: float = 0.0003,      # 佣金 (万3)
                 stamp_duty: float = 0.001,       # 印花税 (千1, 卖出)
                 slippage: float = 0.001,         # 滑点 (千1)
                 rebalance_freq: str = '1D',      # 调仓频率 ('1D', '1W', '1M')
                 weighting: str = 'equal'):       # 权重方式 ('equal', 'score_weighted')
        """
        初始化回测器
        
        Parameters:
        -----------
        top_k : int
            每日持仓股票数量
        commission : float
            佣金费率（单边）
        stamp_duty : float
            印花税（卖出方）
        slippage : float
            滑点估计（双边合计）
        rebalance_freq : str
            调仓频率：
            - '1D': 每日调仓
            - '1W': 每周调仓（周一）
            - '1M': 每月调仓（月初第一个交易日）
        weighting : str
            权重方式：
            - 'equal': 等权
            - 'score_weighted': 按分数加权（softmax）
        """
        self.top_k = top_k
        self.commission = commission
        self.stamp_duty = stamp_duty
        self.slippage = slippage
        self.rebalance_freq = rebalance_freq
        self.weighting = weighting
        
        # 计算单次换仓成本（双边）
        # 买入: commission + slippage/2
        # 卖出: commission + stamp_duty + slippage/2
        self.cost_per_trade = 2 * commission + stamp_duty + slippage
        
        print(f"📊 SimplePortfolioBacktester 初始化")
        print(f"   Top-K: {top_k}")
        print(f"   执行模式: Open-to-Open (T+1)")
        print(f"   调仓频率: {rebalance_freq}")
        print(f"   权重方式: {weighting}")
        print(f"   单次换仓成本: {self.cost_per_trade:.4%}")
    
    def run(self,
            predictions: Union[pd.DataFrame, pd.Series],
            prices: pd.DataFrame,
            tradable_mask: Optional[pd.Series] = None,
            benchmark: Optional[pd.Series] = None,
            save_dir: Optional[str] = None) -> Dict:
        """
        运行回测
        
        Parameters:
        -----------
        predictions : pd.DataFrame or pd.Series
            模型预测分数，MultiIndex [date, ticker]
            如果是 DataFrame，使用第一列或 'score' 列
        prices : pd.DataFrame
            价格数据，MultiIndex [date, ticker]
            必须包含 'open' 列
        tradable_mask : pd.Series, optional
            可交易标记，True 为可交易
        benchmark : pd.Series, optional
            基准净值曲线，index 为 date
        save_dir : str, optional
            保存中间产物的目录（weights/returns/stats）
            
        Returns:
        --------
        Dict
            包含 'equity_curve', 'daily_returns', 'stats', 'weights', 'benchmark_curve' 等
        """
        print(f"\n{'='*60}")
        print(f"🚀 开始回测 (Open-to-Open, Top-K: {self.top_k}, Rebalance: {self.rebalance_freq})")
        print(f"{'='*60}")
        
        # ========== 1. 数据预处理 ==========
        # 统一预测值格式
        if isinstance(predictions, pd.DataFrame):
            if 'score' in predictions.columns:
                score_series = predictions['score']
            else:
                score_series = predictions.iloc[:, 0]
        else:
            score_series = predictions
        
        # 确保索引对齐
        common_idx = score_series.index.intersection(prices.index)
        if len(common_idx) == 0:
            raise ValueError("预测值与价格数据没有共同索引")
        
        scores = score_series.loc[common_idx].copy()
        mkt_data = prices.loc[common_idx].copy()
        
        # 应用可交易过滤
        if tradable_mask is not None:
            mask = tradable_mask.loc[common_idx]
            scores = scores.where(mask, -np.inf)
        
        print(f"   样本数: {len(scores):,}")
        print(f"   日期范围: {scores.index.get_level_values('date').min()} ~ {scores.index.get_level_values('date').max()}")
        
        # ========== 2. 计算每日排名与目标持仓 ==========
        # 按日期分组，计算排名
        ranks = scores.groupby(level='date').rank(ascending=False, method='first')
        
        # 目标持仓：排名 <= top_k 的股票
        target_holdings = (ranks <= self.top_k).astype(float)
        
        # ========== 2.5 应用调仓频率 ==========
        # 生成调仓日期标记
        dates = scores.index.get_level_values('date').unique().sort_values()
        rebalance_dates = self._get_rebalance_dates(dates)
        
        # 构建持仓矩阵（应用调仓频率）
        holdings_matrix = target_holdings.unstack(level='ticker').fillna(0)
        holdings_matrix = self._apply_rebalance_freq(holdings_matrix, rebalance_dates)
        
        # ========== 3. 计算个股收益率（Open-to-Open）==========
        # T+1 开盘买入 -> T+2 开盘卖出
        # 收益 = Open_{t+2} / Open_{t+1} - 1
        if 'open' not in mkt_data.columns:
            raise ValueError("prices 必须包含 'open' 列")
        
        exec_price = mkt_data['open']
        # 个股收益：Shift(-2) / Shift(-1) - 1
        # 这里的逻辑：T日算出信号，对应的收益是从 T+1 Open 到 T+2 Open
        grouped = exec_price.groupby(level='ticker')
        stock_returns = grouped.shift(-2) / grouped.shift(-1) - 1
        
        # 填充 NaN（退市/停牌导致无法计算）
        stock_returns = stock_returns.fillna(0)
        
        # ========== 4. 计算权重与组合收益 ==========
        # 重新 stack holdings_matrix 为 Series
        target_holdings_adj = holdings_matrix.stack()
        
        # 每日实际持仓数量
        daily_counts = target_holdings_adj.groupby(level='date').sum()
        
        # 计算权重
        if self.weighting == 'equal':
            # 等权
            weights = target_holdings_adj / daily_counts.reindex(
                target_holdings_adj.index.get_level_values('date')
            ).values
        elif self.weighting == 'score_weighted':
            # 按分数加权（softmax）
            selected_scores = scores.where(target_holdings_adj > 0, 0)
            score_exp = np.exp(selected_scores - selected_scores.groupby(level='date').max())
            score_sum = score_exp.groupby(level='date').sum()
            weights = score_exp / score_sum.reindex(
                score_exp.index.get_level_values('date')
            ).values
        else:
            raise ValueError(f"未知的权重方式: {self.weighting}")
        
        weights = weights.fillna(0)
        
        # 加权收益
        weighted_returns = weights * stock_returns
        
        # 日组合毛收益
        portfolio_gross_ret = weighted_returns.groupby(level='date').sum()
        
        # ========== 5. 计算换手率与成本 ==========
        # 持仓变化 (换手)
        holdings_diff = holdings_matrix.diff().abs()
        # 第一天全仓买入
        holdings_diff.iloc[0] = holdings_matrix.iloc[0]
        
        # 单边换手率 = 变化股票数 / 2 / 持仓数
        turnover = holdings_diff.sum(axis=1) / 2 / self.top_k
        turnover = turnover.fillna(0)
        
        # 交易成本（仅在调仓日发生）
        transaction_costs = turnover * self.cost_per_trade
        
        # ========== 6. 计算净收益 ==========
        portfolio_net_ret = portfolio_gross_ret - transaction_costs
        
        # ========== 7. 计算累计净值 ==========
        equity_curve = (1 + portfolio_net_ret).cumprod()
        
        # ========== 7.5 处理基准 ==========
        benchmark_curve = None
        excess_returns = None
        if benchmark is not None:
            # 对齐基准日期
            common_dates = equity_curve.index.intersection(benchmark.index)
            if len(common_dates) > 0:
                benchmark_curve = benchmark.loc[common_dates]
                # 计算超额收益
                benchmark_ret = benchmark_curve.pct_change().fillna(0)
                excess_returns = portfolio_net_ret.loc[common_dates] - benchmark_ret
            else:
                print("   ⚠️ 基准日期与策略不重叠，跳过基准对比")
        
        # ========== 8. 统计指标 ==========
        stats = self._calculate_stats(portfolio_net_ret, equity_curve, turnover, transaction_costs, 
                                       benchmark_curve, excess_returns)
        
        # ========== 9. 保存中间产物 ==========
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存权重
            weights_df = weights.unstack(level='ticker').fillna(0)
            weights_path = os.path.join(save_dir, 'portfolio_weights.parquet')
            weights_df.to_parquet(weights_path)
            print(f"   💾 权重已保存: {weights_path}")
            
            # 保存收益率
            returns_df = pd.DataFrame({
                'gross_return': portfolio_gross_ret,
                'net_return': portfolio_net_ret,
                'transaction_cost': transaction_costs,
                'turnover': turnover
            })
            returns_path = os.path.join(save_dir, 'daily_returns.parquet')
            returns_df.to_parquet(returns_path)
            print(f"   💾 收益率已保存: {returns_path}")
            
            # 保存统计指标
            import json
            stats_path = os.path.join(save_dir, 'backtest_stats.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
            print(f"   💾 统计指标已保存: {stats_path}")
        
        # ========== 10. 输出结果 ==========
        print("\n" + "-" * 50)
        print("📊 回测结果")
        print("-" * 50)
        for k, v in stats.items():
            if isinstance(v, float):
                if 'Return' in k or 'Drawdown' in k or 'Turnover' in k or 'Cost' in k or 'Alpha' in k or 'Beta' in k:
                    if 'Beta' in k:
                        print(f"   {k}: {v:.4f}")
                    else:
                        print(f"   {k}: {v:.2%}")
                else:
                    print(f"   {k}: {v:.4f}")
            else:
                print(f"   {k}: {v}")
        print("-" * 50)
        
        result = {
            'equity_curve': equity_curve,
            'daily_returns': portfolio_net_ret,
            'daily_gross_returns': portfolio_gross_ret,
            'turnover': turnover,
            'transaction_costs': transaction_costs,
            'weights': weights.unstack(level='ticker').fillna(0),
            'stats': stats
        }
        
        if benchmark_curve is not None:
            result['benchmark_curve'] = benchmark_curve
            result['excess_returns'] = excess_returns
        
        return result
    
    def _calculate_stats(self,
                         daily_returns: pd.Series,
                         equity_curve: pd.Series,
                         turnover: pd.Series,
                         costs: pd.Series,
                         benchmark_curve: Optional[pd.Series] = None,
                         excess_returns: Optional[pd.Series] = None) -> Dict:
        """计算统计指标"""
        
        # 基本统计
        total_days = len(daily_returns)
        total_return = equity_curve.iloc[-1] - 1
        
        # 年化收益
        ann_return = (1 + total_return) ** (252 / total_days) - 1 if total_days > 0 else 0
        
        # 年化波动率
        ann_volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio (假设无风险利率 2%)
        risk_free = 0.02
        sharpe = (ann_return - risk_free) / ann_volatility if ann_volatility > 0 else 0
        
        # Sortino Ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (ann_return - risk_free) / downside_std if downside_std > 0 else 0
        
        # 最大回撤
        running_max = equity_curve.cummax()
        drawdown = equity_curve / running_max - 1
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 胜率
        win_rate = (daily_returns > 0).sum() / total_days if total_days > 0 else 0
        
        # 盈亏比
        avg_win = daily_returns[daily_returns > 0].mean() if (daily_returns > 0).any() else 0
        avg_loss = abs(daily_returns[daily_returns < 0].mean()) if (daily_returns < 0).any() else 1
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        stats = {
            'Total Days': total_days,
            'Total Return': total_return,
            'Annualized Return': ann_return,
            'Annualized Volatility': ann_volatility,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Calmar Ratio': calmar,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Profit/Loss Ratio': profit_loss_ratio,
            'Avg Daily Turnover': turnover.mean(),
            'Total Cost': costs.sum()
        }
        
        # 基准对比指标
        if benchmark_curve is not None and excess_returns is not None:
            # 基准收益
            benchmark_total = benchmark_curve.iloc[-1] / benchmark_curve.iloc[0] - 1
            benchmark_ann = (1 + benchmark_total) ** (252 / len(benchmark_curve)) - 1
            
            # 超额收益
            excess_ann = ann_return - benchmark_ann
            
            # Alpha & Beta (CAPM)
            benchmark_ret = benchmark_curve.pct_change().fillna(0)
            aligned_strat = daily_returns.loc[benchmark_ret.index]
            
            if len(aligned_strat) > 0 and benchmark_ret.std() > 0:
                beta = aligned_strat.cov(benchmark_ret) / benchmark_ret.var()
                alpha_daily = aligned_strat.mean() - beta * benchmark_ret.mean()
                alpha_ann = alpha_daily * 252
            else:
                beta = 0
                alpha_ann = 0
            
            # 信息比率
            if excess_returns.std() > 0:
                information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            else:
                information_ratio = 0
            
            stats.update({
                'Benchmark Return': benchmark_ann,
                'Excess Return': excess_ann,
                'Alpha (Annual)': alpha_ann,
                'Beta': beta,
                'Information Ratio': information_ratio
            })
        
        return stats
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """生成调仓日期"""
        if self.rebalance_freq == '1D':
            return dates
        elif self.rebalance_freq == '1W':
            # 每周第一个交易日
            df_dates = pd.DataFrame({'date': dates})
            df_dates['week'] = dates.to_period('W')
            first_dates = df_dates.groupby('week')['date'].first()
            return pd.DatetimeIndex(first_dates.values)
        elif self.rebalance_freq == '1M':
            # 每月第一个交易日
            df_dates = pd.DataFrame({'date': dates})
            df_dates['month'] = dates.to_period('M')
            first_dates = df_dates.groupby('month')['date'].first()
            return pd.DatetimeIndex(first_dates.values)
        else:
            raise ValueError(f"未知的调仓频率: {self.rebalance_freq}")
    
    def _apply_rebalance_freq(self, holdings_matrix: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """应用调仓频率约束"""
        if self.rebalance_freq == '1D':
            return holdings_matrix
        
        # 非调仓日延续上一日持仓
        result = holdings_matrix.copy()
        for i in range(1, len(result)):
            if result.index[i] not in rebalance_dates:
                result.iloc[i] = result.iloc[i - 1]
        
        return result
    
    def _calculate_monthly_returns(self, daily_returns: pd.Series) -> pd.DataFrame:
        """计算月度收益矩阵（用于热力图）"""
        if len(daily_returns) == 0:
            return pd.DataFrame()
        
        # 确保索引是 DatetimeIndex
        if not isinstance(daily_returns.index, pd.DatetimeIndex):
            daily_returns.index = pd.to_datetime(daily_returns.index)
        
        # 按月聚合收益
        monthly = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0)
        
        if len(monthly) == 0:
            return pd.DataFrame()
        
        monthly_df = pd.DataFrame({
            'Year': monthly.index.year,
            'Month': monthly.index.month,
            'Return': monthly.values
        })
        
        # pivot 并填充缺失月份
        pivot_df = monthly_df.pivot(index='Year', columns='Month', values='Return')
        
        # 确保所有月份列都存在（1-12）
        for month in range(1, 13):
            if month not in pivot_df.columns:
                pivot_df[month] = np.nan
        
        # 按月份排序
        pivot_df = pivot_df[sorted(pivot_df.columns)]
        
        return pivot_df
    
    def plot(self, 
             result: Dict, 
             save_path: Optional[str] = None,
             title_suffix: str = ''):
        """
        绘制回测结果（增强版）
        
        Parameters:
        -----------
        result : Dict
            run() 方法返回的结果
        save_path : str, optional
            保存路径
        title_suffix : str
            标题后缀
        """
        equity = result['equity_curve']
        daily_ret = result['daily_returns']
        drawdown = equity / equity.cummax() - 1
        turnover = result['turnover']
        stats = result['stats']
        benchmark_curve = result.get('benchmark_curve')
        excess_returns = result.get('excess_returns')
        
        # 判断是否有基准
        has_benchmark = benchmark_curve is not None
        
        # 创建子图（始终显示 4 个，提供完整功能）
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1.5], hspace=0.3, wspace=0.25)
        
        # ===== 1. 净值曲线 =====
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(equity.index, equity.values, label='Strategy', color='blue', linewidth=1.8)
        
        if has_benchmark:
            # 归一化基准到相同起点
            benchmark_norm = benchmark_curve / benchmark_curve.iloc[0]
            ax1.plot(benchmark_norm.index, benchmark_norm.values, label='Benchmark', 
                    color='gray', linewidth=1.5, alpha=0.7, linestyle='--')
        
        # 添加统计信息
        if has_benchmark and 'Alpha (Annual)' in stats:
            info_text = (f"Sharpe: {stats['Sharpe Ratio']:.2f}  |  "
                        f"Ann.Ret: {stats['Annualized Return']:.1%}  |  "
                        f"Alpha: {stats['Alpha (Annual)']:.2%}  |  "
                        f"Beta: {stats['Beta']:.2f}  |  "
                        f"MDD: {stats['Max Drawdown']:.1%}")
        else:
            info_text = (f"Sharpe: {stats['Sharpe Ratio']:.2f}  |  "
                        f"Ann.Ret: {stats['Annualized Return']:.1%}  |  "
                        f"MDD: {stats['Max Drawdown']:.1%}  |  "
                        f"Turnover: {stats['Avg Daily Turnover']:.1%}")
        
        ax1.set_title(f'Portfolio Backtest (Open-to-Open, Top-{self.top_k}, {self.rebalance_freq}) {title_suffix}\n{info_text}', 
                     fontsize=12, fontweight='bold')
        ax1.set_ylabel('净值（Net Value）', fontsize=10)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # ===== 2. 回撤 =====
        ax2 = fig.add_subplot(gs[1, 0])
        
        ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.4, label='Strategy')
        
        if has_benchmark:
            benchmark_dd = benchmark_curve / benchmark_curve.cummax() - 1
            ax2.fill_between(benchmark_dd.index, benchmark_dd.values, 0, 
                            color='gray', alpha=0.3, label='Benchmark')
            min_dd = min(drawdown.min(), benchmark_dd.min())
        else:
            min_dd = drawdown.min()
        
        ax2.set_title('回撤曲线 (Drawdown)', fontsize=10)
        ax2.set_ylabel('回撤 (Drawdown)', fontsize=9)
        ax2.set_ylim([min_dd * 1.1, 0.05])
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # ===== 3. 月度收益热力图（始终显示）=====
        ax3 = fig.add_subplot(gs[1, 1])
        try:
            monthly_ret = self._calculate_monthly_returns(daily_ret)
            
            if len(monthly_ret) > 0:
                im = ax3.imshow(monthly_ret.values, cmap='RdYlGn', aspect='auto', 
                               vmin=-0.1, vmax=0.1, interpolation='nearest')
                ax3.set_title('月度收益热力图', fontsize=10)
                ax3.set_xlabel('月份', fontsize=9)
                ax3.set_ylabel('年份', fontsize=9)
                ax3.set_xticks(range(min(12, monthly_ret.shape[1])))
                ax3.set_xticklabels(range(1, min(13, monthly_ret.shape[1] + 1)), fontsize=8)
                ax3.set_yticks(range(len(monthly_ret)))
                ax3.set_yticklabels(monthly_ret.index, fontsize=8)
                
                # 添加颜色条
                cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
                cbar.set_label('收益率', fontsize=8)
                
                # 在格子中显示数值（限制显示数量避免过密）
                if len(monthly_ret) <= 10:
                    for i in range(len(monthly_ret)):
                        for j in range(min(12, monthly_ret.shape[1])):
                            if j < monthly_ret.shape[1] and not pd.isna(monthly_ret.iloc[i, j]):
                                text = ax3.text(j, i, f'{monthly_ret.iloc[i, j]:.1%}',
                                              ha="center", va="center", color="black", fontsize=6)
            else:
                ax3.text(0.5, 0.5, '数据不足，无法生成月度热力图', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=10)
                ax3.set_title('月度收益热力图', fontsize=10)
        except Exception as e:
            ax3.text(0.5, 0.5, f'月度热力图生成失败\n{str(e)}', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=9)
            ax3.set_title('月度收益热力图', fontsize=10)
            print(f"   ⚠️ 月度热力图生成失败: {e}")
        
        # ===== 4. 换手率与成本 =====
        ax4 = fig.add_subplot(gs[2, :])
        
        ax4_twin = ax4.twinx()
        ax4.bar(turnover.index, turnover.values, color='steelblue', alpha=0.6, 
               width=1, label='换手率')
        ax4_twin.plot(result['transaction_costs'].index, 
                     result['transaction_costs'].cumsum().values, 
                     color='red', linewidth=1.5, label='累计成本')
        
        ax4.set_title('换手率与交易成本', fontsize=10)
        ax4.set_ylabel('换手率 (Turnover)', fontsize=9, color='steelblue')
        ax4_twin.set_ylabel('累计成本 (Cumulative Cost)', fontsize=9, color='red')
        ax4.set_xlabel('日期', fontsize=9)
        ax4.tick_params(axis='y', labelcolor='steelblue')
        ax4_twin.tick_params(axis='y', labelcolor='red')
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # 合并图例
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📈 回测图表已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()


def run_backtest_from_predictions(predictions_path: str,
                                   prices_path: str,
                                   output_dir: str,
                                   top_k: int = 30,
                                   rebalance_freq: str = '1M',
                                   benchmark_path: Optional[str] = None) -> Dict:
    """
    便捷函数：从保存的预测文件运行回测
    
    Parameters:
    -----------
    predictions_path : str
        预测文件路径 (.parquet)
    prices_path : str
        价格文件路径 (.parquet)
    output_dir : str
        输出目录
    top_k : int
        Top-K 选股数量
    rebalance_freq : str
        调仓频率 ('1D', '1W', '1M')
    benchmark_path : str, optional
        基准净值文件路径 (.parquet)
        
    Returns:
    --------
    Dict
        回测结果
    """
    # 加载数据
    predictions = pd.read_parquet(predictions_path)
    prices = pd.read_parquet(prices_path)
    
    benchmark = None
    if benchmark_path and os.path.exists(benchmark_path):
        benchmark = pd.read_parquet(benchmark_path)
        if isinstance(benchmark, pd.DataFrame):
            benchmark = benchmark.iloc[:, 0]
    
    # 创建回测器
    backtester = SimplePortfolioBacktester(top_k=top_k, rebalance_freq=rebalance_freq)
    
    # 运行回测
    result = backtester.run(predictions, prices, benchmark=benchmark, save_dir=output_dir)
    
    # 绘制图表
    plot_path = os.path.join(output_dir, 'backtest_result.png')
    backtester.plot(result, save_path=plot_path)
    
    return result


if __name__ == "__main__":
    # 简单测试
    print("SimplePortfolioBacktester 模块加载成功 (v2.0)")
    print("\n使用方法:")
    print("  # 创建回测器")
    print("  backtester = SimplePortfolioBacktester(")
    print("      top_k=30,")
    print("      rebalance_freq='1M',  # '1D', '1W', '1M'")
    print("      weighting='equal'     # 'equal', 'score_weighted'")
    print("  )")
    print("\n  # 运行回测（带基准对比）")
    print("  result = backtester.run(predictions, prices, benchmark=benchmark_series, save_dir='output/')")
    print("\n  # 绘制图表")
    print("  backtester.plot(result, save_path='backtest.png')")
    print("\n主要改进：")
    print("  ✅ 固定 Open-to-Open 执行模式")
    print("  ✅ 支持调仓频率控制（降低换手）")
    print("  ✅ 基准对比与 Alpha/Beta 分析")
    print("  ✅ 月度收益热力图")
    print("  ✅ 自动保存 weights/returns/stats")
