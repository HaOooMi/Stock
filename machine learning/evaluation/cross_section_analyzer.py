#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
横截面评估框架主类（Alphalens风格）

统一接口，封装所有横截面评估逻辑
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

from .cross_section_metrics import (
    calculate_forward_returns,
    calculate_daily_ic,
    calculate_ic_summary,
    calculate_quantile_returns,
    calculate_cumulative_returns,
    calculate_spread,
    calculate_monotonicity,
    calculate_turnover,
    calculate_factor_autocorrelation
)

from scipy import stats
from scipy.stats import ks_2samp

from .factor_preprocessing import (
    preprocess_factor_pipeline
)

warnings.filterwarnings('ignore')


class CrossSectionAnalyzer:
    """
    横截面因子评估分析器
    
    契约：
    ------
    输入：
        - factors: DataFrame, MultiIndex[date, ticker], columns=因子名称
        - forward_returns: DataFrame, MultiIndex[date, ticker], columns=['ret_1d', 'ret_5d', ...]
          （可选，也可以传入prices自动计算）
        - prices: DataFrame, MultiIndex[date, ticker], columns=['close']
          （如果提供，会自动计算forward_returns）
        - tradable_mask: DataFrame, MultiIndex[date, ticker], bool值
          （可选，用于过滤不可交易的样本）
        - market_cap: DataFrame, MultiIndex[date, ticker]（可选，用于中性化）
        - industry: DataFrame, MultiIndex[date, ticker]（可选，用于中性化）
    
    输出：
        - dict 包含：
            - ic_series[H]: 每日IC序列
            - ic_summary[H]: IC统计摘要
            - quantile_returns[H]: 分位数组合收益
            - cumulative_returns[H]: 累计净值
            - spread[H]: Top-Mean或Top-Bottom
            - monotonicity[H]: 单调性指标
            - turnover_stats: 换手率统计
            - plots: 图表对象或路径
    """
    
    def __init__(self,
                 factors: pd.DataFrame,
                 forward_returns: Optional[pd.DataFrame] = None,
                 prices: Optional[pd.DataFrame] = None,
                 tradable_mask: Optional[pd.DataFrame] = None,
                 market_cap: Optional[pd.DataFrame] = None,
                 industry: Optional[pd.DataFrame] = None):
        """
        初始化分析器
        
        Parameters:
        -----------
        factors : pd.DataFrame
            因子值，MultiIndex[date, ticker]
        forward_returns : pd.DataFrame, optional
            预计算的远期收益
        prices : pd.DataFrame, optional
            价格数据（如果未提供forward_returns）
        tradable_mask : pd.DataFrame, optional
            可交易标记
        market_cap : pd.DataFrame, optional
            市值数据
        industry : pd.DataFrame, optional
            行业数据
        """
        self.factors_raw = factors.copy()
        self.forward_returns = forward_returns
        self.prices = prices
        self.tradable_mask = tradable_mask
        self.market_cap = market_cap
        self.industry = industry
        
        # 处理后的因子
        self.factors_processed = None
        
        # 结果缓存
        self.results = {}
        
        # 验证输入
        self._validate_inputs()
    
    def _validate_inputs(self):
        """验证输入数据"""
        # 检查索引格式
        if not isinstance(self.factors_raw.index, pd.MultiIndex):
            raise ValueError("factors必须有MultiIndex[date, ticker]")
        
        if self.factors_raw.index.names != ['date', 'ticker']:
            raise ValueError("factors索引必须命名为['date', 'ticker']")
        
        # 检查是否提供了收益或价格
        if self.forward_returns is None and self.prices is None:
            raise ValueError("必须提供forward_returns或prices之一")
        
        # 如果提供了forward_returns，验证格式
        if self.forward_returns is not None:
            if not isinstance(self.forward_returns.index, pd.MultiIndex):
                raise ValueError("forward_returns必须有MultiIndex[date, ticker]")
        
        print("✅ 输入数据验证通过")
    
    def preprocess(self,
                  winsorize: bool = True,
                  standardize: bool = True,
                  neutralize: bool = False,
                  **kwargs):
        """
        因子预处理
        
        Parameters:
        -----------
        winsorize : bool
            是否进行极值处理
        standardize : bool
            是否标准化
        neutralize : bool
            是否中性化（需要提供market_cap或industry）
        **kwargs : dict
            其他预处理参数
        """
        print("\n" + "=" * 70)
        print("因子预处理")
        print("=" * 70)
        
        self.factors_processed = preprocess_factor_pipeline(
            self.factors_raw,
            market_cap=self.market_cap if neutralize else None,
            industry=self.industry if neutralize else None,
            winsorize=winsorize,
            standardize=standardize,
            neutralize=neutralize,
            **kwargs
        )
        
        print("✅ 预处理完成\n")
        
        return self
    
    def calculate_returns(self,
                         periods: List[int] = [1, 5, 10, 20],
                         method: str = 'simple'):
        """
        计算远期收益率
        
        Parameters:
        -----------
        periods : List[int]
            前瞻期数
        method : str
            'simple'或'log'
        """
        if self.forward_returns is not None:
            print("⚠️  已提供forward_returns，跳过计算")
            return self
        
        if self.prices is None:
            raise ValueError("未提供prices，无法计算forward_returns")
        
        print("\n" + "=" * 70)
        print("计算远期收益率")
        print("=" * 70)
        
        self.forward_returns = calculate_forward_returns(
            self.prices,
            periods=periods,
            method=method
        )
        
        print(f"✅ 计算完成，期数: {periods}\n")
        
        return self
    
    def analyze(self,
               n_quantiles: int = 5,
               ic_method: str = 'spearman',
               spread_method: str = 'top_minus_mean',
               periods_per_year: int = 252,
               check_quality: bool = False):
        """
        执行完整的横截面分析
        
        Parameters:
        -----------
        n_quantiles : int
            分位数数量
        ic_method : str
            IC计算方法（'spearman'或'pearson'）
        spread_method : str
            Spread计算方法
        periods_per_year : int
            年化参数
        check_quality : bool
            是否执行深度质量检查（PSI/KS/IC衰减等）
        """
        print("\n" + "=" * 70)
        print("横截面分析")
        print("=" * 70)
        
        # 使用处理后的因子，如果没有预处理则使用原始因子
        factors = (
            self.factors_processed 
            if self.factors_processed is not None 
            else self.factors_raw
        )
        
        # 应用可交易性过滤
        if self.tradable_mask is not None:
            print("\n📊 应用可交易性过滤...")
            factors = self._apply_tradable_mask(factors)
            forward_returns = self._apply_tradable_mask(self.forward_returns)
        else:
            forward_returns = self.forward_returns
        
        # 1. 计算每日IC
        print("\n1️⃣  计算每日IC...")
        daily_ic = calculate_daily_ic(factors, forward_returns, method=ic_method)
        self.results['daily_ic'] = daily_ic
        print(f"   ✅ IC形状: {daily_ic.shape}")
        
        # 2. IC汇总统计
        print("\n2️⃣  IC汇总统计...")
        ic_summary = {}
        for col in daily_ic.columns:
            summary = calculate_ic_summary(
                daily_ic[col],
                annualize=True,
                periods_per_year=periods_per_year
            )
            ic_summary[col] = summary
            
            factor_name, return_period = col
            print(f"   {factor_name} @ {return_period}:")
            print(f"      Mean IC: {summary['mean']:.4f}")
            print(f"      ICIR: {summary['icir']:.4f}")
            print(f"      ICIR(年化): {summary['icir_annual']:.4f}")
        
        self.results['ic_summary'] = ic_summary
        
        # 3. 分位数组合收益
        print("\n3️⃣  计算分位数组合收益...")
        quantile_returns = calculate_quantile_returns(
            factors,
            forward_returns,
            n_quantiles=n_quantiles
        )
        self.results['quantile_returns'] = quantile_returns
        print(f"   ✅ 生成{len(quantile_returns)}个组合")
        
        # 4. 累计收益
        print("\n4️⃣  计算累计收益...")
        cumulative_returns = {}
        for key, qret in quantile_returns.items():
            cumulative_returns[key] = calculate_cumulative_returns(qret)
        self.results['cumulative_returns'] = cumulative_returns
        
        # 5. Spread
        print("\n5️⃣  计算Spread...")
        spreads = {}
        spread_summaries = {}
        
        for key, qret in quantile_returns.items():
            spread = calculate_spread(qret, method=spread_method)
            spreads[key] = spread
            
            # Spread统计
            spread_summaries[key] = {
                'mean': spread.mean(),
                'std': spread.std(),
                'sharpe': spread.mean() / spread.std() if spread.std() != 0 else 0,
                'sharpe_annual': (spread.mean() / spread.std()) * np.sqrt(periods_per_year) if spread.std() != 0 else 0,
                'positive_ratio': (spread > 0).sum() / len(spread)
            }
            
            factor_name, return_period = key
            print(f"   {factor_name} @ {return_period}:")
            print(f"      Mean Spread: {spread_summaries[key]['mean']:.6f}")
            print(f"      Sharpe(年化): {spread_summaries[key]['sharpe_annual']:.4f}")
        
        self.results['spreads'] = spreads
        self.results['spread_summaries'] = spread_summaries
        
        # 6. 单调性
        print("\n6️⃣  计算单调性...")
        monotonicities = {}
        for key, qret in quantile_returns.items():
            mono = calculate_monotonicity(qret)
            monotonicities[key] = mono
            
            factor_name, return_period = key
            print(f"   {factor_name} @ {return_period}:")
            print(f"      Kendall τ: {mono['kendall_tau']:.4f}")
            print(f"      正确排序比例: {mono['correct_order_ratio']:.4f}")
        
        self.results['monotonicities'] = monotonicities
        
        # 7. 换手率（仅针对单因子）
        print("\n7️⃣  计算换手率...")
        turnover_stats = {}
        
        for factor_col in factors.columns:
            factor_single = factors[[factor_col]]
            
            try:
                turnover = calculate_turnover(
                    factor_single,
                    quantile=n_quantiles - 1,  # Top分位
                    n_quantiles=n_quantiles
                )
                
                turnover_stats[factor_col] = {
                    'turnover_series': turnover,
                    'mean_turnover': turnover['turnover'].mean(),
                    'std_turnover': turnover['turnover'].std()
                }
                
                print(f"   {factor_col}:")
                print(f"      平均换手率: {turnover_stats[factor_col]['mean_turnover']:.4f}")
                
            except Exception as e:
                print(f"   ⚠️  {factor_col} 换手率计算失败: {e}")
        
        self.results['turnover_stats'] = turnover_stats
        
        # 8. 深度质量检查 (PSI/KS/IC衰减)
        if check_quality:
            print("\n8️⃣  执行深度质量检查 (PSI/KS/IC衰减)...")
            quality_reports = {}
            
            for factor_col in factors.columns:
                # 准备数据
                factor_series = factors[factor_col]
                # 默认使用第一个周期的收益率作为目标
                target_col = forward_returns.columns[0]
                target_series = forward_returns[target_col]
                
                # 计算PSI
                train_end_idx = int(len(factor_series) * 0.8)
                psi = self._calculate_psi(factor_series, train_end_idx)
                
                # 计算KS
                ks_stat, ks_p = self._calculate_ks(factor_series, train_end_idx)
                
                # 计算IC衰减 (如果有价格数据)
                ic_decay = None
                half_life = np.nan
                if self.prices is not None:
                    ic_decay, half_life = self._calculate_ic_decay(factor_series, self.prices)
                
                quality_reports[factor_col] = {
                    'psi': psi,
                    'ks_stat': ks_stat,
                    'ks_p': ks_p,
                    'ic_half_life': half_life,
                    'ic_decay': ic_decay
                }
                
                print(f"   {factor_col}:")
                print(f"      PSI: {psi:.4f}")
                print(f"      KS p-value: {ks_p:.4f}")
                if not np.isnan(half_life):
                    print(f"      IC半衰期: {half_life:.1f}天")
            
            self.results['quality_reports'] = quality_reports

        print("\n" + "=" * 70)
        print("✅ 分析完成")
        print("=" * 70 + "\n")
        
        return self

    def _calculate_psi(self, factor: pd.Series, train_end_idx: int, n_bins: int = 10) -> float:
        """内部方法: 计算PSI"""
        try:
            train_factor = factor.iloc[:train_end_idx].dropna()
            test_factor = factor.iloc[train_end_idx:].dropna()
            
            if len(train_factor) < 30 or len(test_factor) < 30:
                return np.nan
            
            _, bin_edges = pd.qcut(train_factor, q=n_bins, retbins=True, duplicates='drop')
            
            train_dist, _ = np.histogram(train_factor, bins=bin_edges)
            test_dist, _ = np.histogram(test_factor, bins=bin_edges)
            
            train_pct = train_dist / len(train_factor)
            test_pct = test_dist / len(test_factor)
            
            train_pct = np.where(train_pct == 0, 0.0001, train_pct)
            test_pct = np.where(test_pct == 0, 0.0001, test_pct)
            
            return np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))
        except:
            return np.nan

    def _calculate_ks(self, factor: pd.Series, train_end_idx: int) -> Tuple[float, float]:
        """内部方法: 计算KS统计量"""
        try:
            train_factor = factor.iloc[:train_end_idx].dropna()
            test_factor = factor.iloc[train_end_idx:].dropna()
            
            if len(train_factor) < 30 or len(test_factor) < 30:
                return np.nan, np.nan
            
            return ks_2samp(train_factor, test_factor)
        except:
            return np.nan, np.nan

    def _calculate_ic_decay(self, factor: pd.Series, prices: pd.DataFrame, max_period: int = 20) -> Tuple[pd.DataFrame, float]:
        """内部方法: 计算IC衰减"""
        try:
            ic_decay = []
            for period in range(1, max_period + 1):
                forward_return = prices['close'].pct_change(period).shift(-period)
                valid_mask = factor.notna() & forward_return.notna()
                if valid_mask.sum() < 30:
                    continue
                ic, _ = stats.spearmanr(factor[valid_mask], forward_return[valid_mask])
                ic_decay.append({'period': period, 'ic': ic, 'abs_ic': abs(ic)})
            
            decay_df = pd.DataFrame(ic_decay)
            if decay_df.empty:
                return pd.DataFrame(), np.nan
                
            initial_ic = abs(decay_df.iloc[0]['ic'])
            half_ic = initial_ic * 0.5
            below_half = decay_df[decay_df['abs_ic'] < half_ic]
            half_life = below_half.iloc[0]['period'] if not below_half.empty else max_period
            return decay_df, half_life
        except:
            return pd.DataFrame(), np.nan
    
    def _apply_tradable_mask(self, data: pd.DataFrame) -> pd.DataFrame:
        """应用可交易性过滤"""
        if self.tradable_mask is None:
            return data
        
        # 对齐索引
        mask_aligned = self.tradable_mask.reindex(data.index)
        
        if isinstance(mask_aligned, pd.DataFrame):
            mask_values = mask_aligned.iloc[:, 0]
        else:
            mask_values = mask_aligned
        
        # 过滤
        result = data.copy()
        result[~mask_values] = np.nan
        
        return result
    
    def get_results(self) -> Dict:
        """获取所有结果"""
        return self.results
    
    def summary(self):
        """打印摘要"""
        print("\n" + "=" * 70)
        print("横截面分析摘要")
        print("=" * 70)
        
        if 'ic_summary' not in self.results:
            print("⚠️  尚未执行分析，请先调用analyze()")
            return
        
        print("\n📊 IC统计:")
        print("-" * 70)
        
        for (factor_name, return_period), summary in self.results['ic_summary'].items():
            print(f"\n{factor_name} @ {return_period}:")
            print(f"  Mean IC:        {summary['mean']:>10.4f}")
            print(f"  ICIR:           {summary['icir']:>10.4f}")
            print(f"  ICIR(年化):     {summary['icir_annual']:>10.4f}")
            print(f"  t-stat:         {summary['t_stat']:>10.4f}")
            print(f"  p-value:        {summary['p_value']:>10.6f}")
            print(f"  正IC比例:       {summary['positive_ratio']:>10.2%}")
            print(f"  观测数:         {summary['n_obs']:>10d}")
        
        if 'spread_summaries' in self.results:
            print("\n📈 Spread统计:")
            print("-" * 70)
            
            for (factor_name, return_period), summary in self.results['spread_summaries'].items():
                print(f"\n{factor_name} @ {return_period}:")
                print(f"  Mean Spread:    {summary['mean']:>10.6f}")
                print(f"  Std Spread:     {summary['std']:>10.6f}")
                print(f"  Sharpe:         {summary['sharpe']:>10.4f}")
                print(f"  Sharpe(年化):   {summary['sharpe_annual']:>10.4f}")
                print(f"  正Spread比例:   {summary['positive_ratio']:>10.2%}")
        
        if 'turnover_stats' in self.results:
            print("\n🔄 换手率统计:")
            print("-" * 70)
            
            for factor_name, stats in self.results['turnover_stats'].items():
                print(f"\n{factor_name}:")
                print(f"  平均换手率:     {stats['mean_turnover']:>10.2%}")
                print(f"  换手率标准差:   {stats['std_turnover']:>10.2%}")
        
        print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    """测试代码"""
    print("=" * 70)
    print("横截面评估框架测试")
    print("=" * 70)
    
    # 生成测试数据
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    tickers = [f'Stock_{i:03d}' for i in range(100)]
    
    index = pd.MultiIndex.from_product(
        [dates, tickers],
        names=['date', 'ticker']
    )
    
    # 价格
    prices = pd.DataFrame({
        'close': 100 + np.random.randn(len(index)).cumsum() * 0.1
    }, index=index)
    
    # 因子（添加一些预测能力）
    returns_true = prices['close'].groupby(level='ticker').pct_change()
    factors = pd.DataFrame({
        'factor_1': returns_true.shift(1) + np.random.randn(len(index)) * 0.02,
        'factor_2': np.random.randn(len(index))
    }, index=index)
    
    # 市值
    market_cap = pd.DataFrame({
        'market_cap': np.random.lognormal(20, 2, len(index))
    }, index=index)
    
    # 行业
    industries = ['金融', '科技', '消费', '医药', '工业']
    industry = pd.DataFrame({
        'industry': np.random.choice(industries, len(index))
    }, index=index)
    
    print("\n创建分析器...")
    analyzer = CrossSectionAnalyzer(
        factors=factors,
        prices=prices,
        market_cap=market_cap,
        industry=industry
    )
    
    print("\n执行预处理...")
    analyzer.preprocess(
        winsorize=True,
        standardize=True,
        neutralize=False  # 测试时不中性化，避免过度处理
    )
    
    print("\n计算远期收益...")
    analyzer.calculate_returns(periods=[1, 5, 10])
    
    print("\n执行分析...")
    analyzer.analyze(
        n_quantiles=5,
        ic_method='spearman',
        spread_method='top_minus_mean'
    )
    
    print("\n打印摘要...")
    analyzer.summary()
    
    print("\n✅ 测试完成！")
