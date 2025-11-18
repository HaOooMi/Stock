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
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def calculate_forward_returns(prices: pd.DataFrame,
                              periods: List[int] = [1, 5, 10, 20],
                              method: str = 'simple') -> pd.DataFrame:
    """
    计算远期收益率
    
    Parameters:
    -----------
    prices : pd.DataFrame
        价格数据，MultiIndex[date, ticker]，列为'close'或直接Series
    periods : List[int]
        前瞻期数列表（如[1, 5, 10, 20]天）
    method : str
        'simple': (P_{t+H} - P_t) / P_t
        'log': log(P_{t+H}) - log(P_t)
        
    Returns:
    --------
    pd.DataFrame
        MultiIndex[date, ticker]，列为['ret_1d', 'ret_5d', ...]
    """
    if isinstance(prices, pd.Series):
        prices_series = prices
    elif isinstance(prices, pd.DataFrame):
        if 'close' in prices.columns:
            prices_series = prices['close']
        else:
            prices_series = prices.iloc[:, 0]
    else:
        raise ValueError("prices必须是Series或DataFrame")
    
    forward_returns = pd.DataFrame(index=prices_series.index)
    
    for period in periods:
        col_name = f'ret_{period}d'
        
        if method == 'simple':
            # Simple return: r = P_{t+H} / P_t - 1
            forward_returns[col_name] = (
                prices_series.groupby(level='ticker')
                .shift(-period) / prices_series - 1
            )
        elif method == 'log':
            # Log return: r = log(P_{t+H}) - log(P_t)
            forward_returns[col_name] = (
                np.log(prices_series.groupby(level='ticker').shift(-period)) -
                np.log(prices_series)
            )
        else:
            raise ValueError(f"不支持的方法: {method}")
    
    return forward_returns


def calculate_daily_ic(factors: pd.DataFrame,
                      forward_returns: pd.DataFrame,
                      method: str = 'spearman') -> pd.DataFrame:
    """
    计算每日横截面IC
    
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
        MultiIndex[date]，列为MultiIndex[(factor_name, return_period)]
        每个值是该日横截面的IC
    """
    # 合并数据
    data = factors.join(forward_returns, how='inner')
    
    # 提取因子列和收益列
    factor_cols = factors.columns.tolist()
    return_cols = forward_returns.columns.tolist()
    
    # 按日期分组计算IC
    daily_ics = []
    
    dates = data.index.get_level_values('date').unique()
    
    for date in dates:
        date_data = data.xs(date, level='date')
        
        if len(date_data) < 3:  # 至少需要3个样本
            continue
        
        ic_row = {'date': date}
        
        for factor_col in factor_cols:
            for return_col in return_cols:
                # 移除NaN
                valid_mask = (
                    date_data[factor_col].notna() & 
                    date_data[return_col].notna()
                )
                
                if valid_mask.sum() < 3:
                    ic_row[(factor_col, return_col)] = np.nan
                    continue
                
                factor_values = date_data.loc[valid_mask, factor_col]
                return_values = date_data.loc[valid_mask, return_col]
                
                # 计算IC
                if method == 'spearman':
                    ic, _ = stats.spearmanr(factor_values, return_values)
                elif method == 'pearson':
                    ic, _ = stats.pearsonr(factor_values, return_values)
                else:
                    raise ValueError(f"不支持的方法: {method}")
                
                ic_row[(factor_col, return_col)] = ic
        
        daily_ics.append(ic_row)
    
    # 构建DataFrame
    result = pd.DataFrame(daily_ics).set_index('date')
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    
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
    计算分位数组合收益
    
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
    # 合并数据
    data = factors.join(forward_returns, how='inner')
    
    factor_cols = factors.columns.tolist()
    return_cols = forward_returns.columns.tolist()
    
    quantile_returns = {}
    
    for factor_col in factor_cols:
        for return_col in return_cols:
            # 按日期分组
            daily_quantile_rets = []
            
            dates = data.index.get_level_values('date').unique()
            
            for date in dates:
                date_data = data.xs(date, level='date')
                
                # 移除NaN
                valid_mask = (
                    date_data[factor_col].notna() & 
                    date_data[return_col].notna()
                )
                
                if valid_mask.sum() < n_quantiles:
                    continue
                
                valid_data = date_data[valid_mask]
                
                # 分位数分组
                if quantile_method == 'quantile':
                    valid_data['quantile'] = pd.qcut(
                        valid_data[factor_col],
                        q=n_quantiles,
                        labels=False,
                        duplicates='drop'
                    )
                elif quantile_method == 'equal_width':
                    valid_data['quantile'] = pd.cut(
                        valid_data[factor_col],
                        bins=n_quantiles,
                        labels=False
                    )
                else:
                    raise ValueError(f"不支持的方法: {quantile_method}")
                
                # 计算每个分位数的平均收益
                quantile_ret = valid_data.groupby('quantile')[return_col].mean()
                
                ret_row = {'date': date}
                for q in range(n_quantiles):
                    ret_row[f'Q{q+1}'] = quantile_ret.get(q, np.nan)
                
                daily_quantile_rets.append(ret_row)
            
            # 构建DataFrame
            if daily_quantile_rets:
                df = pd.DataFrame(daily_quantile_rets).set_index('date')
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
                      n_quantiles: int = 5) -> pd.Series:
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
    pd.Series
        每日换手率（0-1之间）
    """
    if len(factors.columns) > 1:
        raise ValueError("换手率计算仅支持单因子，请传入单列DataFrame")
    
    factor_col = factors.columns[0]
    
    dates = factors.index.get_level_values('date').unique().sort_values()
    
    turnovers = []
    prev_tickers = None
    
    for date in dates:
        date_data = factors.xs(date, level='date')
        
        # 移除NaN
        valid_data = date_data[date_data[factor_col].notna()]
        
        if len(valid_data) < n_quantiles:
            continue
        
        # 分位数分组
        valid_data['quantile'] = pd.qcut(
            valid_data[factor_col],
            q=n_quantiles,
            labels=False,
            duplicates='drop'
        )
        
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
