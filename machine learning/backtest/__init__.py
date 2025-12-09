"""
回测模块 - 组合回测功能

包含：
- SimplePortfolioBacktester: 简易组合回测器（Top-K 等权）
- StrategyBacktest: 聚类策略回测（历史遗留）
"""

from .simple_backtest import SimplePortfolioBacktester

__all__ = [
    'SimplePortfolioBacktester'
]
