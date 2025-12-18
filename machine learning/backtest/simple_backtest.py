#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简易组合回测器 - Simple Portfolio Backtester

功能：
1. 接收模型预测分数 (Score)
2. 执行 Top-K 等权选股
3. 支持两种执行模式：
   - close_to_close: T日收盘买入，T+1收盘卖出（理想情况，有前视偏差）
   - open_to_open: T+1日开盘买入，T+2开盘卖出（现实情况，无偏差）
4. 计算扣费后的净值曲线与核心指标

设计原则：
- 极简主义：基于向量化计算，无事件驱动
- 真实性：支持 T+1 Execution Lag 和交易成本
- 接口清晰：输入 Predictions + Market Data -> 输出 Stats + Curve

简历亮点：
- 支持多重执行假设（T+0/T+1）的回测引擎
- 消除了 Look-ahead Bias
- 模块化设计，可扩展至复杂组合优化

创建: 2025-12-09 | 版本: v1.0
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
                 execution_mode: str = 'open_to_open',
                 holding_period: int = 1):
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
        execution_mode : str
            执行模式：
            - 'close_to_close': T日收盘买入，T+1收盘卖出（理想情况）
            - 'open_to_open': T+1日开盘买入，T+2开盘卖出（现实情况）
        holding_period : int
            持有天数（默认1天）
        """
        self.top_k = top_k
        self.commission = commission
        self.stamp_duty = stamp_duty
        self.slippage = slippage
        self.execution_mode = execution_mode
        self.holding_period = holding_period
        
        # 计算单次换仓成本（双边）
        # 买入: commission + slippage/2
        # 卖出: commission + stamp_duty + slippage/2
        self.cost_per_trade = 2 * commission + stamp_duty + slippage
        
        print(f"📊 SimplePortfolioBacktester 初始化")
        print(f"   Top-K: {top_k}")
        print(f"   执行模式: {execution_mode}")
        print(f"   单次换仓成本: {self.cost_per_trade:.4%}")
    
    def run(self,
            predictions: Union[pd.DataFrame, pd.Series],
            prices: pd.DataFrame,
            tradable_mask: Optional[pd.Series] = None) -> Dict:
        """
        运行回测
        
        Parameters:
        -----------
        predictions : pd.DataFrame or pd.Series
            模型预测分数，MultiIndex [date, ticker]
            如果是 DataFrame，使用第一列或 'score' 列
        prices : pd.DataFrame
            价格数据，MultiIndex [date, ticker]
            必须包含 'open' 和 'close' 列
        tradable_mask : pd.Series, optional
            可交易标记，True 为可交易
            
        Returns:
        --------
        Dict
            包含 'equity_curve', 'daily_returns', 'stats', 'positions' 等
        """
        print(f"\n{'='*60}")
        print(f"🚀 开始回测 (Mode: {self.execution_mode}, Top-K: {self.top_k})")
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
        
        # ========== 2. 计算每日排名与持仓 ==========
        # 按日期分组，计算排名
        ranks = scores.groupby(level='date').rank(ascending=False, method='first')
        
        # 目标持仓：排名 <= top_k 的股票
        target_holdings = (ranks <= self.top_k).astype(float)
        
        # ========== 3. 计算个股收益率 ==========
        # 根据执行模式选择价格列
        if self.execution_mode == 'open_to_open':
            # T+1 开盘买入 -> T+2 开盘卖出
            # 收益 = Open_{t+2} / Open_{t+1} - 1
            if 'open' not in mkt_data.columns:
                raise ValueError("执行模式为 'open_to_open' 时，prices 必须包含 'open' 列")
            
            exec_price = mkt_data['open']
            # 个股收益：Shift(-2) / Shift(-1) - 1
            # 这里的逻辑：T日算出信号，对应的收益是从 T+1 Open 到 T+2 Open
            grouped = exec_price.groupby(level='ticker')
            stock_returns = grouped.shift(-2) / grouped.shift(-1) - 1
            
        elif self.execution_mode == 'close_to_close':
            # T 收盘买入 -> T+1 收盘卖出
            # 收益 = Close_{t+1} / Close_t - 1
            if 'close' not in mkt_data.columns:
                raise ValueError("prices 必须包含 'close' 列")
            
            exec_price = mkt_data['close']
            grouped = exec_price.groupby(level='ticker')
            stock_returns = grouped.shift(-1) / exec_price - 1
            
        else:
            raise ValueError(f"未知的执行模式: {self.execution_mode}")
        
        # 填充 NaN（退市/停牌导致无法计算）
        stock_returns = stock_returns.fillna(0)
        
        # ========== 4. 计算组合收益 ==========
        # 等权组合：每日持仓股票的平均收益
        # 注意：持仓是 T 日决定的，收益归属在 T+1 日（或 T+1~T+2）
        
        # 每日实际持仓数量
        daily_counts = target_holdings.groupby(level='date').sum()
        
        # 等权权重
        weights = target_holdings / daily_counts.reindex(
            target_holdings.index.get_level_values('date')
        ).values
        weights = weights.fillna(0)
        
        # 加权收益
        weighted_returns = weights * stock_returns
        
        # 日组合毛收益
        portfolio_gross_ret = weighted_returns.groupby(level='date').sum()
        
        # ========== 5. 计算换手率与成本 ==========
        # 将持仓展开为矩阵 [date x ticker]
        holdings_matrix = target_holdings.unstack(level='ticker').fillna(0)
        
        # 持仓变化 (换手)
        holdings_diff = holdings_matrix.diff().abs()
        # 第一天全仓买入
        holdings_diff.iloc[0] = holdings_matrix.iloc[0]
        
        # 单边换手率 = 变化股票数 / 2 / 持仓数
        turnover = holdings_diff.sum(axis=1) / 2 / self.top_k
        
        # 交易成本
        transaction_costs = turnover * self.cost_per_trade
        
        # ========== 6. 计算净收益 ==========
        portfolio_net_ret = portfolio_gross_ret - transaction_costs
        
        # ========== 7. 计算累计净值 ==========
        equity_curve = (1 + portfolio_net_ret).cumprod()
        
        # ========== 8. 统计指标 ==========
        stats = self._calculate_stats(portfolio_net_ret, equity_curve, turnover, transaction_costs)
        
        # ========== 9. 输出结果 ==========
        print("\n" + "-" * 50)
        print("📊 回测结果")
        print("-" * 50)
        for k, v in stats.items():
            if isinstance(v, float):
                if 'Return' in k or 'Drawdown' in k or 'Turnover' in k or 'Cost' in k:
                    print(f"   {k}: {v:.2%}")
                else:
                    print(f"   {k}: {v:.4f}")
            else:
                print(f"   {k}: {v}")
        print("-" * 50)
        
        return {
            'equity_curve': equity_curve,
            'daily_returns': portfolio_net_ret,
            'daily_gross_returns': portfolio_gross_ret,
            'turnover': turnover,
            'transaction_costs': transaction_costs,
            'stats': stats,
            'execution_mode': self.execution_mode
        }
    
    def _calculate_stats(self,
                         daily_returns: pd.Series,
                         equity_curve: pd.Series,
                         turnover: pd.Series,
                         costs: pd.Series) -> Dict:
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
        
        # 胜率
        win_rate = (daily_returns > 0).sum() / total_days if total_days > 0 else 0
        
        # 盈亏比
        avg_win = daily_returns[daily_returns > 0].mean() if (daily_returns > 0).any() else 0
        avg_loss = abs(daily_returns[daily_returns < 0].mean()) if (daily_returns < 0).any() else 1
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        return {
            'Total Days': total_days,
            'Total Return': total_return,
            'Annualized Return': ann_return,
            'Annualized Volatility': ann_volatility,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Profit/Loss Ratio': profit_loss_ratio,
            'Avg Daily Turnover': turnover.mean(),
            'Total Cost': costs.sum()
        }
    
    def plot(self, 
             result: Dict, 
             benchmark: Optional[pd.Series] = None,
             save_path: Optional[str] = None,
             title_suffix: str = ''):
        """
        绘制回测结果
        
        Parameters:
        -----------
        result : Dict
            run() 方法返回的结果
        benchmark : pd.Series, optional
            基准净值曲线
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
        mode = result['execution_mode']
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1.5, 1]})
        
        # ===== 1. 净值曲线 =====
        ax1 = axes[0]
        ax1.plot(equity.index, equity.values, label='Strategy', color='blue', linewidth=1.5)
        
        if benchmark is not None:
            ax1.plot(benchmark.index, benchmark.values, label='Benchmark', color='gray', linewidth=1, alpha=0.7)
        
        # 添加统计信息
        info_text = (f"Sharpe: {stats['Sharpe Ratio']:.2f}  |  "
                     f"Ann.Ret: {stats['Annualized Return']:.1%}  |  "
                     f"MDD: {stats['Max Drawdown']:.1%}  |  "
                     f"Turnover: {stats['Avg Daily Turnover']:.1%}")
        ax1.set_title(f'Portfolio Backtest ({mode}) {title_suffix}\n{info_text}', fontsize=12)
        ax1.set_ylabel('Net Value')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # ===== 2. 回撤 =====
        ax2 = axes[1]
        ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown')
        ax2.set_ylim([drawdown.min() * 1.1, 0.05])
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # ===== 3. 换手率 =====
        ax3 = axes[2]
        ax3.bar(turnover.index, turnover.values, color='gray', alpha=0.5, width=1)
        ax3.set_ylabel('Turnover')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📈 回测图表已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_modes(self,
                      predictions: Union[pd.DataFrame, pd.Series],
                      prices: pd.DataFrame,
                      tradable_mask: Optional[pd.Series] = None,
                      save_dir: Optional[str] = None) -> Dict:
        """
        对比两种执行模式的结果（A/B 测试）
        
        Parameters:
        -----------
        predictions : pd.DataFrame or pd.Series
            模型预测分数
        prices : pd.DataFrame
            价格数据
        tradable_mask : pd.Series, optional
            可交易标记
        save_dir : str, optional
            保存目录
            
        Returns:
        --------
        Dict
            包含两种模式的结果和对比
        """
        print("\n" + "=" * 70)
        print("🔬 A/B 测试：Close-to-Close vs Open-to-Open")
        print("=" * 70)
        
        results = {}
        
        # ===== 实验 A: Close-to-Close (理想情况) =====
        self.execution_mode = 'close_to_close'
        results['close_to_close'] = self.run(predictions, prices, tradable_mask)
        
        # ===== 实验 B: Open-to-Open (现实情况) =====
        self.execution_mode = 'open_to_open'
        results['open_to_open'] = self.run(predictions, prices, tradable_mask)
        
        # ===== 对比分析 =====
        print("\n" + "=" * 70)
        print("📊 A/B 测试对比结果")
        print("=" * 70)
        
        comparison = {}
        metrics_to_compare = ['Annualized Return', 'Sharpe Ratio', 'Max Drawdown', 'Avg Daily Turnover']
        
        print(f"\n{'指标':<25} {'Close-to-Close':>18} {'Open-to-Open':>18} {'差异':>15}")
        print("-" * 80)
        
        for metric in metrics_to_compare:
            val_cc = results['close_to_close']['stats'][metric]
            val_oo = results['open_to_open']['stats'][metric]
            
            if 'Return' in metric or 'Drawdown' in metric or 'Turnover' in metric:
                diff = val_oo - val_cc
                print(f"{metric:<25} {val_cc:>17.2%} {val_oo:>17.2%} {diff:>14.2%}")
            else:
                diff = val_oo - val_cc
                print(f"{metric:<25} {val_cc:>17.4f} {val_oo:>17.4f} {diff:>14.4f}")
            
            comparison[metric] = {
                'close_to_close': val_cc,
                'open_to_open': val_oo,
                'difference': diff
            }
        
        print("-" * 80)
        
        # 计算 Alpha 衰减
        ret_cc = results['close_to_close']['stats']['Annualized Return']
        ret_oo = results['open_to_open']['stats']['Annualized Return']
        alpha_decay = (ret_cc - ret_oo) / abs(ret_cc) if ret_cc != 0 else 0
        
        print(f"\n⚠️ Alpha 衰减: {alpha_decay:.1%}")
        if alpha_decay > 0.5:
            print("   警告：超过 50% 的 Alpha 来自不可执行的隔夜收益！")
        elif alpha_decay > 0.3:
            print("   注意：约 1/3 的 Alpha 来自隔夜收益，需要进一步验证。")
        else:
            print("   良好：大部分 Alpha 在 T+1 执行下仍然存在。")
        
        comparison['alpha_decay'] = alpha_decay
        
        # ===== 绘制对比图 =====
        if save_dir:
            try:
                self._plot_comparison(results, comparison, save_dir)
            except Exception as e:
                print(f"⚠️ 绘制对比图失败: {e}")
                import traceback
                traceback.print_exc()
        
        return {
            'close_to_close': results['close_to_close'],
            'open_to_open': results['open_to_open'],
            'comparison': comparison
        }
    
    def _plot_comparison(self, results: Dict, comparison: Dict, save_dir: str):
        """绘制对比图"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ===== 1. 净值曲线对比 =====
        ax1 = axes[0, 0]
        eq_cc = results['close_to_close']['equity_curve']
        eq_oo = results['open_to_open']['equity_curve']
        
        ax1.plot(eq_cc.index, eq_cc.values, label='Close-to-Close (理想)', color='blue', linewidth=1.5)
        ax1.plot(eq_oo.index, eq_oo.values, label='Open-to-Open (现实)', color='red', linewidth=1.5)
        ax1.set_title('净值曲线对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('Net Value')
        
        # ===== 2. 回撤对比 =====
        ax2 = axes[0, 1]
        dd_cc = eq_cc / eq_cc.cummax() - 1
        dd_oo = eq_oo / eq_oo.cummax() - 1
        
        ax2.plot(dd_cc.index, dd_cc.values, label='Close-to-Close', color='blue', alpha=0.7)
        ax2.plot(dd_oo.index, dd_oo.values, label='Open-to-Open', color='red', alpha=0.7)
        ax2.set_title('回撤对比')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylabel('Drawdown')
        
        # ===== 3. 指标对比柱状图 =====
        ax3 = axes[1, 0]
        metrics = ['Annualized Return', 'Sharpe Ratio']
        x = np.arange(len(metrics))
        width = 0.35
        
        vals_cc = [results['close_to_close']['stats'][m] for m in metrics]
        vals_oo = [results['open_to_open']['stats'][m] for m in metrics]
        
        ax3.bar(x - width/2, vals_cc, width, label='Close-to-Close', color='blue', alpha=0.7)
        ax3.bar(x + width/2, vals_oo, width, label='Open-to-Open', color='red', alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.set_title('关键指标对比')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # ===== 4. Alpha 衰减分析 =====
        ax4 = axes[1, 1]
        alpha_decay = comparison['alpha_decay']
        
        # 确保 sizes 为非负值
        if alpha_decay < 0:
            # 如果 Open-to-Open 收益更高（Alpha 衰减为负），显示 Alpha 增强
            labels = ['可执行 Alpha', 'Alpha 增强']
            sizes = [1.0, abs(alpha_decay)]
            colors = ['green', 'lightgreen']
            title_text = f'Alpha 分析 (增强: {abs(alpha_decay):.1%})'
        elif alpha_decay >= 1:
            # 如果衰减超过 100%，说明策略在 Open-to-Open 下失效
            labels = ['可执行 Alpha', '隔夜衰减']
            sizes = [0.01, 0.99]
            colors = ['green', 'red']
            title_text = f'Alpha 衰减分析 (衰减: {alpha_decay:.1%})'
        else:
            # 正常情况：0 <= alpha_decay < 1
            labels = ['可执行 Alpha', '隔夜衰减']
            sizes = [1 - alpha_decay, alpha_decay]
            colors = ['green', 'red']
            title_text = f'Alpha 衰减分析 (衰减: {alpha_decay:.1%})'
        
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title(title_text)
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'backtest_ab_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📈 A/B 对比图表已保存: {save_path}")
        
        plt.close()


def run_backtest_from_predictions(predictions_path: str,
                                   prices_path: str,
                                   output_dir: str,
                                   top_k: int = 30,
                                   compare_modes: bool = True) -> Dict:
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
    compare_modes : bool
        是否对比两种执行模式
        
    Returns:
    --------
    Dict
        回测结果
    """
    # 加载数据
    predictions = pd.read_parquet(predictions_path)
    prices = pd.read_parquet(prices_path)
    
    # 创建回测器
    backtester = SimplePortfolioBacktester(top_k=top_k)
    
    if compare_modes:
        return backtester.compare_modes(predictions, prices, save_dir=output_dir)
    else:
        result = backtester.run(predictions, prices)
        backtester.plot(result, save_path=os.path.join(output_dir, 'backtest_result.png'))
        return result


if __name__ == "__main__":
    # 简单测试
    print("SimplePortfolioBacktester 模块加载成功")
    print("使用方法:")
    print("  backtester = SimplePortfolioBacktester(top_k=30)")
    print("  result = backtester.run(predictions, prices)")
    print("  backtester.plot(result)")
