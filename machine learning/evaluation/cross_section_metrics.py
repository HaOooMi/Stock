#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
横截面评估核心度量模块（Alphalens风格）

核心功能：
1. Forward Returns 计算（simple/log）
2. Rank IC（每日横截面 Spearman）
3. ICIR（IC的信息比率）
4. 分桶收益分析
5. Spread计算（Top-Mean, Top-Bottom）
6. 单调性检验（Kendall τ）
7. 因子换手率

性能优化：
- 使用numpy向量化运算
- 可选numba JIT加速
- 避免groupby().apply()开销
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 尝试导入numba进行JIT加速
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # 定义空装饰器
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# ============== Numba加速的核心计算函数 ==============

@jit(nopython=True, cache=True)
def _rank_array(arr: np.ndarray) -> np.ndarray:
    """快速计算排名（用于Spearman相关）"""
    n = len(arr)
    ranks = np.empty(n, dtype=np.float64)
    order = np.argsort(arr)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    return ranks


@jit(nopython=True, cache=True)
def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """快速Spearman相关系数"""
    n = len(x)
    if n < 3:
        return np.nan
    
    # 计算排名
    rx = _rank_array(x)
    ry = _rank_array(y)
    
    # Pearson相关
    mx = rx.mean()
    my = ry.mean()
    
    num = 0.0
    dx2 = 0.0
    dy2 = 0.0
    
    for i in range(n):
        dx = rx[i] - mx
        dy = ry[i] - my
        num += dx * dy
        dx2 += dx * dx
        dy2 += dy * dy
    
    denom = np.sqrt(dx2 * dy2)
    if denom == 0:
        return np.nan
    return num / denom


def calculate_forward_returns(prices: pd.DataFrame,
                              periods: List[int] = [1, 5, 10, 20],
                              method: str = 'simple',
                              price_col: str = 'close',
                              execution_lag: int = 0) -> pd.DataFrame:
    """
    计算远期收益率
    
    Parameters:
    -----------
    prices : pd.DataFrame
        价格数据，MultiIndex[date, ticker]
    periods : List[int]
        前瞻期数列表（如[1, 5, 10, 20]天）
    method : str
        'simple': (P_{t+H} - P_t) / P_t
        'log': log(P_{t+H}) - log(P_t)
    price_col : str
        使用的价格列名（默认'close'）。
        若要计算 T+1 Open 执行的收益，请传入 'open' 并设置 execution_lag=1
    execution_lag : int
        执行延迟天数（默认0）。
        0: 基于当期价格 P_t 计算 (Close_t -> Close_{t+H}) - 存在前视偏差
        1: 基于下一期价格 P_{t+1} 计算 (Open_{t+1} -> Open_{t+1+H}) - 模拟 T+1 执行
        
    Returns:
    --------
    pd.DataFrame
        MultiIndex[date, ticker]，列为['ret_1d', 'ret_5d', ...]
    """
    if isinstance(prices, pd.Series):
        prices_series = prices
    elif isinstance(prices, pd.DataFrame):
        if price_col in prices.columns:
            prices_series = prices[price_col]
        elif 'close' in prices.columns:
            prices_series = prices['close']
        else:
            prices_series = prices.iloc[:, 0]
    else:
        raise ValueError("prices必须是Series或DataFrame")
    
    forward_returns = pd.DataFrame(index=prices_series.index)
    
    # 按标的分组
    grouped = prices_series.groupby(level='ticker')
    
    for period in periods:
        col_name = f'ret_{period}d'
        
        # 计算逻辑：
        # execution_lag=0 (Close-to-Close): P_{t+H} / P_t - 1
        # execution_lag=1 (Open-to-Open):   P_{t+1+H} / P_{t+1} - 1
        
        # 分母：P_{t + lag}
        denom = grouped.shift(-execution_lag)
        # 分子：P_{t + lag + H}
        numer = grouped.shift(-(execution_lag + period))
        
        if method == 'simple':
            # Simple return: P_end / P_start - 1
            forward_returns[col_name] = numer / denom - 1
        elif method == 'log':
            # Log return: log(P_end) - log(P_start)
            forward_returns[col_name] = np.log(numer) - np.log(denom)
        else:
            raise ValueError(f"不支持的方法: {method}")
    
    return forward_returns


def calculate_daily_ic(factors: pd.DataFrame,
                      forward_returns: pd.DataFrame,
                      method: str = 'spearman') -> pd.DataFrame:
    """
    计算每日横截面IC - 高性能版本（使用numpy向量化）
    
    Parameters:
    -----------
    factors : pd.DataFrame
        因子值，MultiIndex[date, ticker]，列为因子名称
    forward_returns : pd.DataFrame
        远期收益率，MultiIndex[date, ticker]，列为['ret_1d', 'ret_5d', ...]
    method : str
        'spearman': Rank IC（推荐）
        'pearson': IC
        
    Returns:
    --------
    pd.DataFrame
        index=date，列为MultiIndex[(factor_name, return_period)]
    """
    # 合并数据
    data = factors.join(forward_returns, how='inner')
    
    factor_cols = factors.columns.tolist()
    return_cols = forward_returns.columns.tolist()
    col_tuples = [(f, r) for f in factor_cols for r in return_cols]
    
    # 按日期分组
    grouped = data.groupby(level='date')
    dates = []
    all_ics = []
    
    for date, group in grouped:
        if len(group) < 3:
            continue
        
        dates.append(date)
        ic_row = []
        
        for factor_col in factor_cols:
            for return_col in return_cols:
                f_vals = group[factor_col].values
                r_vals = group[return_col].values
                
                # 快速移除NaN
                mask = ~(np.isnan(f_vals) | np.isnan(r_vals))
                if mask.sum() < 3:
                    ic_row.append(np.nan)
                    continue
                
                f_clean = f_vals[mask]
                r_clean = r_vals[mask]
                
                # 使用numba加速的Spearman计算（如果可用）
                if method == 'spearman':
                    if HAS_NUMBA:
                        ic = _spearman_corr(f_clean, r_clean)
                    else:
                        # Rank转换
                        f_rank = np.argsort(np.argsort(f_clean)).astype(float)
                        r_rank = np.argsort(np.argsort(r_clean)).astype(float)
                        # Pearson on ranks = Spearman
                        n = len(f_rank)
                        f_mean, r_mean = f_rank.mean(), r_rank.mean()
                        cov = ((f_rank - f_mean) * (r_rank - r_mean)).sum()
                        f_std = np.sqrt(((f_rank - f_mean) ** 2).sum())
                        r_std = np.sqrt(((r_rank - r_mean) ** 2).sum())
                        ic = cov / (f_std * r_std) if f_std > 0 and r_std > 0 else np.nan
                else:
                    # Pearson
                    f_mean, r_mean = f_clean.mean(), r_clean.mean()
                    cov = ((f_clean - f_mean) * (r_clean - r_mean)).sum()
                    f_std = np.sqrt(((f_clean - f_mean) ** 2).sum())
                    r_std = np.sqrt(((r_clean - r_mean) ** 2).sum())
                    ic = cov / (f_std * r_std) if f_std > 0 and r_std > 0 else np.nan
                
                ic_row.append(ic)
        
        all_ics.append(ic_row)
    
    if not dates:
        return pd.DataFrame(columns=pd.MultiIndex.from_tuples(col_tuples))
    
    result = pd.DataFrame(all_ics, index=dates, columns=pd.MultiIndex.from_tuples(col_tuples))
    result.index.name = 'date'
    
    return result


def calculate_ic_summary(ic_series: pd.Series,
                        annualize: bool = True,
                        periods_per_year: int = 252) -> Dict:
    """
    计算IC汇总统计量
    
    Parameters:
    -----------
    ic_series : pd.Series
        IC时间序列
    annualize : bool
        是否年化ICIR
    periods_per_year : int
        一年的周期数（日频=252，周频=52）
        
    Returns:
    --------
    dict
        包含mean, std, ICIR, t_stat, p_value等
    """
    ic_clean = ic_series.dropna()
    
    if len(ic_clean) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'icir': np.nan,
            'icir_annual': np.nan,
            't_stat': np.nan,
            'p_value': np.nan,
            'n_obs': 0,
            'positive_ratio': np.nan
        }
    
    mean_ic = ic_clean.mean()
    std_ic = ic_clean.std()
    icir = mean_ic / std_ic if std_ic != 0 else np.nan
    
    # 年化ICIR
    icir_annual = icir * np.sqrt(periods_per_year) if annualize else icir
    
    # t检验（H0: IC=0）
    t_stat, p_value = stats.ttest_1samp(ic_clean, 0)
    
    # 正IC比例
    positive_ratio = (ic_clean > 0).sum() / len(ic_clean)
    
    return {
        'mean': mean_ic,
        'std': std_ic,
        'icir': icir,
        'icir_annual': icir_annual,
        't_stat': t_stat,
        'p_value': p_value,
        'n_obs': len(ic_clean),
        'positive_ratio': positive_ratio
    }


def calculate_quantile_returns(factors: pd.DataFrame,
                               forward_returns: pd.DataFrame,
                               n_quantiles: int = 5,
                               quantile_method: str = 'quantile') -> Dict[str, pd.DataFrame]:
    """
    计算分位数组合收益 - 高性能版本
    
    Parameters:
    -----------
    factors : pd.DataFrame
        因子值，MultiIndex[date, ticker]
    forward_returns : pd.DataFrame
        远期收益率，MultiIndex[date, ticker]
    n_quantiles : int
        分位数数量（5或10）
    quantile_method : str
        'quantile': 等分位
        'equal_width': 等宽
        
    Returns:
    --------
    dict
        {(factor_name, return_period): pd.DataFrame}
        DataFrame: index=date, columns=[Q1, Q2, ..., Qn]
    """
    data = factors.join(forward_returns, how='inner')
    
    factor_cols = factors.columns.tolist()
    return_cols = forward_returns.columns.tolist()
    
    quantile_returns = {}
    
    # 按日期分组
    grouped = data.groupby(level='date')
    
    for factor_col in factor_cols:
        for return_col in return_cols:
            dates = []
            q_returns = []
            
            for date, group in grouped:
                f_vals = group[factor_col].values
                r_vals = group[return_col].values
                
                # 移除NaN
                mask = ~(np.isnan(f_vals) | np.isnan(r_vals))
                if mask.sum() < n_quantiles:
                    continue
                
                f_clean = f_vals[mask]
                r_clean = r_vals[mask]
                
                # 快速分位数计算
                try:
                    if quantile_method == 'quantile':
                        # 使用numpy的percentile
                        edges = np.percentile(f_clean, np.linspace(0, 100, n_quantiles + 1))
                        edges[0] = -np.inf
                        edges[-1] = np.inf
                        q_labels = np.digitize(f_clean, edges[1:-1])
                    else:
                        edges = np.linspace(f_clean.min(), f_clean.max(), n_quantiles + 1)
                        q_labels = np.digitize(f_clean, edges[1:-1])
                    
                    # 计算每个分位的平均收益
                    q_ret = []
                    for q in range(n_quantiles):
                        q_mask = (q_labels == q)
                        if q_mask.sum() > 0:
                            q_ret.append(r_clean[q_mask].mean())
                        else:
                            q_ret.append(np.nan)
                    
                    dates.append(date)
                    q_returns.append(q_ret)
                except Exception:
                    continue
            
            if dates:
                cols = [f'Q{i+1}' for i in range(n_quantiles)]
                df = pd.DataFrame(q_returns, index=dates, columns=cols)
                df.index.name = 'date'
                quantile_returns[(factor_col, return_col)] = df
    
    return quantile_returns


def calculate_cumulative_returns(quantile_returns: pd.DataFrame) -> pd.DataFrame:
    """
    计算累计收益（净值曲线）
    
    Parameters:
    -----------
    quantile_returns : pd.DataFrame
        分位数组合的日收益率，columns=[Q1, Q2, ..., Qn]
        
    Returns:
    --------
    pd.DataFrame
        累计收益（净值），初始值为1
    """
    # 累计收益 = (1 + ret_1) * (1 + ret_2) * ... - 1
    cumulative = (1 + quantile_returns).cumprod()
    
    return cumulative


def calculate_spread(quantile_returns: pd.DataFrame,
                    method: str = 'top_minus_mean') -> pd.Series:
    """
    计算Spread（对冲收益）
    
    Parameters:
    -----------
    quantile_returns : pd.DataFrame
        分位数组合的日收益率，columns=[Q1, Q2, ..., Qn]
    method : str
        'top_minus_mean': Top分位数 - 全市场均值（更实盘）
        'top_minus_bottom': Top分位数 - Bottom分位数（经典学术）
        
    Returns:
    --------
    pd.Series
        每日Spread
    """
    if method == 'top_minus_mean':
        # Top - Mean
        top_quantile = quantile_returns.iloc[:, -1]
        mean_return = quantile_returns.mean(axis=1)
        spread = top_quantile - mean_return
        
    elif method == 'top_minus_bottom':
        # Top - Bottom
        top_quantile = quantile_returns.iloc[:, -1]
        bottom_quantile = quantile_returns.iloc[:, 0]
        spread = top_quantile - bottom_quantile
        
    else:
        raise ValueError(f"不支持的方法: {method}")
    
    return spread


def calculate_monotonicity(quantile_returns: pd.DataFrame) -> Dict:
    """
    计算单调性指标
    
    Parameters:
    -----------
    quantile_returns : pd.DataFrame
        分位数组合的日收益率，columns=[Q1, Q2, ..., Qn]
        
    Returns:
    --------
    dict
        包含kendall_tau, correct_order_ratio等
    """
    # 计算平均收益
    mean_returns = quantile_returns.mean()
    
    # Kendall τ（期望分位数与实际收益的相关性）
    expected_ranks = np.arange(len(mean_returns))
    actual_ranks = mean_returns.rank() - 1
    
    kendall_tau, p_value = stats.kendalltau(expected_ranks, actual_ranks)
    
    # 正确排序比例（按日）
    correct_orders = []
    
    for date in quantile_returns.index:
        daily_rets = quantile_returns.loc[date]
        
        # 检查是否单调递增
        is_monotonic = all(
            daily_rets.iloc[i] <= daily_rets.iloc[i+1]
            for i in range(len(daily_rets) - 1)
        )
        correct_orders.append(is_monotonic)
    
    correct_order_ratio = np.mean(correct_orders)
    
    return {
        'kendall_tau': kendall_tau,
        'kendall_p_value': p_value,
        'correct_order_ratio': correct_order_ratio,
        'mean_returns': mean_returns.to_dict()
    }


def calculate_turnover(factors: pd.DataFrame,
                      quantile: int = 4,  # Top分位（0-indexed）
                      n_quantiles: int = 5) -> pd.DataFrame:
    """
    计算因子换手率（Top分位数的持仓变化）
    
    Parameters:
    -----------
    factors : pd.DataFrame
        因子值，MultiIndex[date, ticker]
    quantile : int
        目标分位数（0-indexed，4表示Top分位）
    n_quantiles : int
        总分位数
        
    Returns:
    --------
    pd.DataFrame
        每日换手率（0-1之间）
    """
    if len(factors.columns) > 1:
        raise ValueError("换手率计算仅支持单因子，请传入单列DataFrame")
    
    factor_col = factors.columns[0]
    
    dates = factors.index.get_level_values('date').unique().sort_values()
    n_stocks = factors.index.get_level_values('ticker').nunique()
    
    # 检查股票数量是否足够
    if n_stocks < n_quantiles:
        warnings.warn(f"股票数量({n_stocks})少于分位数({n_quantiles})，无法计算换手率")
        return pd.DataFrame(columns=['turnover', 'n_current', 'n_prev', 'n_intersection'])
    
    turnovers = []
    prev_tickers = None
    
    for date in dates:
        date_data = factors.xs(date, level='date').copy()
        
        # 移除NaN
        valid_data = date_data[date_data[factor_col].notna()]
        
        if len(valid_data) < n_quantiles:
            continue
        
        # 分位数分组
        try:
            valid_data['quantile'] = pd.qcut(
                valid_data[factor_col],
                q=n_quantiles,
                labels=False,
                duplicates='drop'
            )
        except ValueError:
            # 无法分组（可能因为值太集中）
            continue
        
        # 获取目标分位数的股票列表
        current_tickers = set(
            valid_data[valid_data['quantile'] == quantile].index
        )
        
        if prev_tickers is not None and len(prev_tickers) > 0:
            # 计算交集比例
            intersection = len(current_tickers & prev_tickers)
            union = len(current_tickers | prev_tickers)
            
            # 换手率 = 1 - (交集 / 当期持仓数)
            turnover = 1 - (intersection / len(current_tickers)) if len(current_tickers) > 0 else np.nan
            
            turnovers.append({
                'date': date,
                'turnover': turnover,
                'n_current': len(current_tickers),
                'n_prev': len(prev_tickers),
                'n_intersection': intersection
            })
        
        prev_tickers = current_tickers
    
    if not turnovers:
        # 返回空DataFrame
        return pd.DataFrame(columns=['turnover', 'n_current', 'n_prev', 'n_intersection'])
    
    turnover_df = pd.DataFrame(turnovers).set_index('date')
    
    return turnover_df


def calculate_factor_autocorrelation(factors: pd.DataFrame,
                                     lags: List[int] = [1, 5, 10]) -> Dict[int, float]:
    """
    计算因子自相关性（按ticker）
    
    Parameters:
    -----------
    factors : pd.DataFrame
        因子值，MultiIndex[date, ticker]
    lags : List[int]
        滞后期列表
        
    Returns:
    --------
    dict
        {lag: autocorrelation}
    """
    if len(factors.columns) > 1:
        raise ValueError("自相关计算仅支持单因子")
    
    factor_col = factors.columns[0]
    
    autocorrs = {}
    
    for lag in lags:
        ticker_autocorrs = []
        
        tickers = factors.index.get_level_values('ticker').unique()
        
        for ticker in tickers:
            ticker_data = factors.xs(ticker, level='ticker')[factor_col].dropna()
            
            if len(ticker_data) < lag + 10:  # 至少需要足够的数据
                continue
            
            # 计算自相关
            if len(ticker_data) > lag:
                autocorr = ticker_data.autocorr(lag=lag)
                if not np.isnan(autocorr):
                    ticker_autocorrs.append(autocorr)
        
        # 平均自相关
        if ticker_autocorrs:
            autocorrs[lag] = np.mean(ticker_autocorrs)
        else:
            autocorrs[lag] = np.nan
    
    return autocorrs


def calculate_information_coefficient_weighted(
    factors: pd.DataFrame,
    forward_returns: pd.DataFrame,
    weights: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    计算加权IC（例如按市值加权）
    
    Parameters:
    -----------
    factors : pd.DataFrame
        因子值，MultiIndex[date, ticker]
    forward_returns : pd.DataFrame
        远期收益率
    weights : pd.DataFrame, optional
        权重，MultiIndex[date, ticker]
        
    Returns:
    --------
    pd.DataFrame
        加权IC时间序列
    """
    if weights is None:
        # 如果没有权重，等权处理
        return calculate_daily_ic(factors, forward_returns)
    
    # TODO: 实现加权IC计算
    # 这需要使用加权相关系数，实现较复杂
    raise NotImplementedError("加权IC计算尚未实现")


if __name__ == '__main__':
    """测试代码"""
    print("=" * 70)
    print("横截面评估核心度量模块测试")
    print("=" * 70)
    
    # 生成测试数据
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    tickers = [f'Stock_{i:03d}' for i in range(50)]
    
    # 创建MultiIndex
    index = pd.MultiIndex.from_product(
        [dates, tickers],
        names=['date', 'ticker']
    )
    
    # 模拟价格数据
    prices = pd.DataFrame({
        'close': np.random.randn(len(index)).cumsum() + 100
    }, index=index)
    
    # 模拟因子数据
    factors = pd.DataFrame({
        'factor_1': np.random.randn(len(index)),
        'factor_2': np.random.randn(len(index))
    }, index=index)
    
    print("\n1. 计算远期收益...")
    forward_rets = calculate_forward_returns(prices, periods=[1, 5, 10])
    print(f"   远期收益形状: {forward_rets.shape}")
    print(f"   列: {forward_rets.columns.tolist()}")
    
    print("\n2. 计算每日IC...")
    daily_ic = calculate_daily_ic(factors, forward_rets)
    print(f"   IC形状: {daily_ic.shape}")
    print(f"   前5行:\n{daily_ic.head()}")
    
    print("\n3. IC汇总统计...")
    for col in daily_ic.columns[:2]:  # 测试前2个组合
        summary = calculate_ic_summary(daily_ic[col])
        print(f"   {col}:")
        print(f"      Mean IC: {summary['mean']:.4f}")
        print(f"      ICIR: {summary['icir']:.4f}")
        print(f"      ICIR(年化): {summary['icir_annual']:.4f}")
        print(f"      p-value: {summary['p_value']:.4f}")
    
    print("\n4. 分位数收益...")
    quantile_rets = calculate_quantile_returns(factors, forward_rets, n_quantiles=5)
    first_key = list(quantile_rets.keys())[0]
    print(f"   第一个组合 {first_key}:")
    print(f"   形状: {quantile_rets[first_key].shape}")
    print(f"   前5行:\n{quantile_rets[first_key].head()}")
    
    print("\n5. 累计收益...")
    cum_rets = calculate_cumulative_returns(quantile_rets[first_key])
    print(f"   最终净值:\n{cum_rets.iloc[-1]}")
    
    print("\n6. Spread计算...")
    spread_tm = calculate_spread(quantile_rets[first_key], method='top_minus_mean')
    spread_tb = calculate_spread(quantile_rets[first_key], method='top_minus_bottom')
    print(f"   Top-Mean 平均: {spread_tm.mean():.6f}")
    print(f"   Top-Bottom 平均: {spread_tb.mean():.6f}")
    
    print("\n7. 单调性检验...")
    mono = calculate_monotonicity(quantile_rets[first_key])
    print(f"   Kendall τ: {mono['kendall_tau']:.4f}")
    print(f"   正确排序比例: {mono['correct_order_ratio']:.4f}")
    
    print("\n8. 换手率计算...")
    turnover = calculate_turnover(factors[['factor_1']], quantile=4, n_quantiles=5)
    print(f"   平均换手率: {turnover['turnover'].mean():.4f}")
    print(f"   前5天:\n{turnover.head()}")
    
    print("\n✅ 所有测试完成！")
