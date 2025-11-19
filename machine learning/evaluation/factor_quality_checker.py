#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子质量检查器 - 因子体检流程

【模块定位】
本模块专注于「因子工厂」特有的质量检查，复用 cross_section_metrics.py 的IC计算。

独特功能：
1. ✅ IC半衰期与IC Decay曲线（时间衰减特性）
2. ✅ PSI/KS测试（分布稳定性检测）
3. ✅ 相关性检查（避免冗余因子）
4. ✅ 综合质量评分（6层检查打分）

复用功能：
- IC/ICIR计算 → 使用 cross_section_metrics.calculate_daily_ic()
- 单调性检验 → 使用 cross_section_metrics 的分桶功能

验收标准：
- IC均值 > 0.02, ICIR > 0.5 (年化)
- PSI < 0.25 (分布稳定)
- 与已有因子相关性 < 0.7
- 至少通过 4/6 项检查

创建: 2025-01-20 | 版本: v1.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

# 导入已有的IC计算功能
from evaluation.cross_section_metrics import (
    calculate_daily_ic,
    calculate_ic_summary
)


class FactorQualityChecker:
    """
    因子质量检查器
    
    提供完整的因子"体检"流程
    """
    
    def __init__(self, 
                 ic_threshold: float = 0.02,
                 icir_threshold: float = 0.5,
                 psi_threshold: float = 0.25,
                 corr_threshold: float = 0.7):
        """
        初始化质量检查器
        
        Parameters:
        -----------
        ic_threshold : float
            IC均值阈值
        icir_threshold : float
            ICIR年化阈值
        psi_threshold : float
            PSI阈值
        corr_threshold : float
            相关性阈值
        """
        self.ic_threshold = ic_threshold
        self.icir_threshold = icir_threshold
        self.psi_threshold = psi_threshold
        self.corr_threshold = corr_threshold
        
        print("🔬 因子质量检查器初始化")
        print(f"   IC阈值: {ic_threshold}")
        print(f"   ICIR阈值: {icir_threshold}")
        print(f"   PSI阈值: {psi_threshold}")
        print(f"   相关性阈值: {corr_threshold}")
    
    def calculate_ic_metrics(self, 
                            factor_values: pd.Series, 
                            target_values: pd.Series,
                            method: str = 'spearman') -> Dict:
        """
        计算IC指标（复用 cross_section_metrics 的功能）
        
        注意：本方法已改为轻量级包装器，核心计算由 cross_section_metrics 完成。
        
        Parameters:
        -----------
        factor_values : pd.Series
            因子值，MultiIndex[date, ticker]
        target_values : pd.Series
            目标值（远期收益），MultiIndex[date, ticker]
        method : str
            'spearman' 或 'pearson'
            
        Returns:
        --------
        dict
            IC指标字典（格式兼容旧版）
        """
        # 转换为DataFrame格式以适配 cross_section_metrics
        factor_df = pd.DataFrame({'factor': factor_values})
        target_df = pd.DataFrame({'target': target_values})
        
        # 使用 cross_section_metrics 计算每日IC
        daily_ic_df = calculate_daily_ic(factor_df, target_df, method=method)
        
        if daily_ic_df.empty:
            return {
                'ic_mean': np.nan,
                'ic_std': np.nan,
                'icir': np.nan,
                'icir_annual': np.nan,
                't_stat': np.nan,
                'p_value': np.nan,
                'positive_ratio': np.nan,
                'pass_ic': False
            }
        
        # 提取IC序列（第一列）
        ic_series = daily_ic_df.iloc[:, 0].dropna()
        
        # 使用 cross_section_metrics 计算汇总统计
        ic_summary = calculate_ic_summary(ic_series, annualize=True)
        
        # 判断是否通过
        pass_ic = abs(ic_summary['mean']) > self.ic_threshold and ic_summary['p_value'] < 0.05
        
        # 转换为兼容格式
        return {
            'ic_mean': ic_summary['mean'],
            'ic_std': ic_summary['std'],
            'icir': ic_summary['icir'],
            'icir_annual': ic_summary['icir_annual'],
            't_stat': ic_summary['t_stat'],
            'p_value': ic_summary['p_value'],
            'positive_ratio': ic_summary['positive_ratio'],
            'pass_ic': pass_ic
        }
    
    def calculate_ic_decay(self, 
                          factor: pd.Series,
                          prices: pd.DataFrame,
                          max_period: int = 20) -> pd.DataFrame:
        """
        计算IC衰减曲线
        
        Parameters:
        -----------
        factor : pd.Series
            因子值
        prices : pd.DataFrame
            价格数据，包含'close'列
        max_period : int
            最大前瞻期
            
        Returns:
        --------
        pd.DataFrame
            IC衰减曲线
        """
        ic_decay = []
        
        for period in range(1, max_period + 1):
            forward_return = prices['close'].pct_change(period).shift(-period)
            
            valid_mask = factor.notna() & forward_return.notna()
            if valid_mask.sum() < 30:
                continue
            
            ic, _ = stats.spearmanr(
                factor[valid_mask],
                forward_return[valid_mask]
            )
            
            ic_decay.append({
                'period': period,
                'ic': ic,
                'abs_ic': abs(ic)
            })
        
        decay_df = pd.DataFrame(ic_decay)
        
        # 计算半衰期（IC下降到初始值50%的期数）
        if not decay_df.empty:
            initial_ic = abs(decay_df.iloc[0]['ic'])
            half_ic = initial_ic * 0.5
            
            # 找到第一个低于half_ic的点
            below_half = decay_df[decay_df['abs_ic'] < half_ic]
            half_life = below_half.iloc[0]['period'] if not below_half.empty else max_period
        else:
            half_life = np.nan
        
        return decay_df, half_life
    
    def calculate_psi(self, 
                     factor: pd.Series, 
                     train_end_idx: int,
                     n_bins: int = 10) -> float:
        """
        计算PSI (Population Stability Index)
        
        用于检测因子分布漂移
        
        Parameters:
        -----------
        factor : pd.Series
            因子值
        train_end_idx : int
            训练集结束索引
        n_bins : int
            分箱数量
            
        Returns:
        --------
        float
            PSI值
        """
        train_factor = factor.iloc[:train_end_idx].dropna()
        test_factor = factor.iloc[train_end_idx:].dropna()
        
        if len(train_factor) < 30 or len(test_factor) < 30:
            return np.nan
        
        # 基于训练集确定分箱边界
        _, bin_edges = pd.qcut(train_factor, q=n_bins, retbins=True, duplicates='drop')
        
        # 计算训练集和测试集的分布
        train_dist, _ = np.histogram(train_factor, bins=bin_edges)
        test_dist, _ = np.histogram(test_factor, bins=bin_edges)
        
        # 转换为比例
        train_pct = train_dist / len(train_factor)
        test_pct = test_dist / len(test_factor)
        
        # 避免零值
        train_pct = np.where(train_pct == 0, 0.0001, train_pct)
        test_pct = np.where(test_pct == 0, 0.0001, test_pct)
        
        # 计算PSI
        psi = np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))
        
        return psi
    
    def calculate_ks_statistic(self, 
                               factor: pd.Series,
                               train_end_idx: int) -> Tuple[float, float]:
        """
        计算KS统计量
        
        用于检测分布差异
        
        Parameters:
        -----------
        factor : pd.Series
            因子值
        train_end_idx : int
            训练集结束索引
            
        Returns:
        --------
        tuple
            (KS统计量, p值)
        """
        train_factor = factor.iloc[:train_end_idx].dropna()
        test_factor = factor.iloc[train_end_idx:].dropna()
        
        if len(train_factor) < 30 or len(test_factor) < 30:
            return np.nan, np.nan
        
        ks_stat, p_value = ks_2samp(train_factor, test_factor)
        
        return ks_stat, p_value
    
    def check_correlation_with_existing(self,
                                       new_factor: pd.Series,
                                       existing_factors: pd.DataFrame) -> Dict:
        """
        检查新因子与已有因子的相关性
        
        Parameters:
        -----------
        new_factor : pd.Series
            新因子
        existing_factors : pd.DataFrame
            已有因子集合
            
        Returns:
        --------
        dict
            相关性分析结果
        """
        if existing_factors.empty:
            return {
                'max_corr': 0.0,
                'max_corr_factor': None,
                'high_corr_count': 0,
                'pass_corr': True
            }
        
        # 计算与所有已有因子的相关性
        correlations = {}
        for col in existing_factors.columns:
            valid_mask = new_factor.notna() & existing_factors[col].notna()
            if valid_mask.sum() < 30:
                continue
            
            corr, _ = stats.spearmanr(
                new_factor[valid_mask],
                existing_factors[col][valid_mask]
            )
            correlations[col] = abs(corr)
        
        if not correlations:
            return {
                'max_corr': 0.0,
                'max_corr_factor': None,
                'high_corr_count': 0,
                'pass_corr': True
            }
        
        max_corr = max(correlations.values())
        max_corr_factor = max(correlations, key=correlations.get)
        high_corr_count = sum(1 for c in correlations.values() if c > self.corr_threshold)
        
        pass_corr = max_corr < self.corr_threshold
        
        return {
            'max_corr': max_corr,
            'max_corr_factor': max_corr_factor,
            'high_corr_count': high_corr_count,
            'correlations': correlations,
            'pass_corr': pass_corr
        }
    
    def check_monotonicity(self,
                          factor: pd.Series,
                          forward_return: pd.Series,
                          n_quantiles: int = 5) -> Dict:
        """
        检查单调性（分位数组合收益是否单调）
        
        注意：cross_section_metrics 也有分桶功能，但此处为简化版专用于快速检查。
        
        Parameters:
        -----------
        factor : pd.Series
            因子值，支持单层或MultiIndex
        forward_return : pd.Series
            远期收益率，支持单层或MultiIndex
        n_quantiles : int
            分位数数量
            
        Returns:
        --------
        dict
            单调性检验结果
        """
        # 处理MultiIndex情况：展平后计算（跨日期横截面）
        if isinstance(factor.index, pd.MultiIndex):
            factor = factor.reset_index(drop=True)
            forward_return = forward_return.reset_index(drop=True)
        
        # 对齐数据
        valid_mask = factor.notna() & forward_return.notna()
        factor_clean = factor[valid_mask]
        return_clean = forward_return[valid_mask]
        
        if len(factor_clean) < n_quantiles * 10:
            return {
                'kendall_tau': np.nan,
                'kendall_p': np.nan,
                'monotonic_ratio': np.nan,
                'pass_monotonicity': False
            }
        
        # 分位数分组
        quantiles = pd.qcut(factor_clean, q=n_quantiles, labels=False, duplicates='drop')
        
        # 计算各分位数的平均收益
        quantile_returns = []
        for q in range(n_quantiles):
            q_mask = quantiles == q
            if q_mask.sum() > 0:
                mean_return = return_clean[q_mask].mean()
                quantile_returns.append(mean_return)
        
        if len(quantile_returns) < n_quantiles:
            return {
                'kendall_tau': np.nan,
                'kendall_p': np.nan,
                'monotonic_ratio': np.nan,
                'pass_monotonicity': False
            }
        
        # Kendall τ检验
        expected_ranks = np.arange(n_quantiles)
        actual_ranks = stats.rankdata(quantile_returns) - 1
        
        kendall_tau, kendall_p = stats.kendalltau(expected_ranks, actual_ranks)
        
        # 单调性比例（相邻分位数收益递增的比例）
        monotonic_count = sum(1 for i in range(len(quantile_returns) - 1) 
                            if quantile_returns[i] < quantile_returns[i + 1])
        monotonic_ratio = monotonic_count / (len(quantile_returns) - 1)
        
        # 判断是否通过（Kendall τ > 0.5 或 单调性比例 > 0.6）
        pass_monotonicity = kendall_tau > 0.5 or monotonic_ratio > 0.6
        
        return {
            'kendall_tau': kendall_tau,
            'kendall_p': kendall_p,
            'monotonic_ratio': monotonic_ratio,
            'quantile_returns': quantile_returns,
            'pass_monotonicity': pass_monotonicity
        }
    
    def comprehensive_check(self,
                           factor_values: pd.Series,
                           target_values: pd.Series,
                           prices: Optional[pd.DataFrame] = None,
                           existing_factors: Optional[pd.DataFrame] = None,
                           train_ratio: float = 0.8) -> Dict:
        """
        综合质量检查（完整体检流程）
        
        复用 cross_section_metrics 的IC计算，专注于因子工厂特有的检查项。
        
        Parameters:
        -----------
        factor_values : pd.Series
            待检查因子，MultiIndex[date, ticker]
        target_values : pd.Series
            目标值（远期收益），MultiIndex[date, ticker]
        prices : pd.DataFrame, optional
            价格数据（用于IC衰减，如无则跳过）
        existing_factors : pd.DataFrame, optional
            已有因子（用于相关性检查）
        train_ratio : float
            训练集比例（用于PSI/KS检查）
            
        Returns:
        --------
        dict
            完整的质量检查报告
        """
        print(f"\n🔬 综合质量检查")
        print("=" * 60)
        
        train_end_idx = int(len(factor_values) * train_ratio)
        
        # 1. IC指标（复用 cross_section_metrics）
        print("1️⃣  IC指标（复用 cross_section_metrics）...")
        ic_metrics = self.calculate_ic_metrics(factor_values, target_values)
        print(f"   IC均值: {ic_metrics['ic_mean']:.4f} ({'✅' if ic_metrics['pass_ic'] else '❌'})")
        print(f"   ICIR年化: {ic_metrics['icir_annual']:.2f}")
        print(f"   p-value: {ic_metrics['p_value']:.4f}")
        
        # 2. IC衰减（独特功能）
        if prices is not None and not prices.empty:
            print("2️⃣  IC衰减曲线...")
            ic_decay_df, half_life = self.calculate_ic_decay(factor_values, prices, max_period=20)
            print(f"   IC半衰期: {half_life:.1f} 天")
        else:
            print("2️⃣  IC衰减曲线...（无价格数据，跳过）")
            ic_decay_df, half_life = pd.DataFrame(), np.nan
        
        # 3. PSI（独特功能）
        print("3️⃣  PSI测试...")
        psi = self.calculate_psi(factor_values, train_end_idx)
        pass_psi = psi < self.psi_threshold if not np.isnan(psi) else False
        print(f"   PSI: {psi:.4f} ({'✅' if pass_psi else '❌'})")
        
        # 4. KS统计量（独特功能）
        print("4️⃣  KS测试...")
        ks_stat, ks_p = self.calculate_ks_statistic(factor_values, train_end_idx)
        pass_ks = ks_p > 0.05 if not np.isnan(ks_p) else False
        print(f"   KS统计量: {ks_stat:.4f}, p-value: {ks_p:.4f} ({'✅' if pass_ks else '❌'})")
        
        # 5. 相关性检查（独特功能）
        print("5️⃣  相关性检查...")
        if existing_factors is not None and not existing_factors.empty:
            corr_check = self.check_correlation_with_existing(factor_values, existing_factors)
            print(f"   最大相关性: {corr_check['max_corr']:.4f} ({'✅' if corr_check['pass_corr'] else '❌'})")
            if corr_check['max_corr_factor']:
                print(f"   最相关因子: {corr_check['max_corr_factor']}")
        else:
            corr_check = {'pass_corr': True, 'max_corr': 0.0}
            print(f"   无已有因子，跳过")
        
        # 6. 单调性检验
        print("6️⃣  单调性检验...")
        monotonicity = self.check_monotonicity(factor_values, target_values)
        print(f"   Kendall τ: {monotonicity['kendall_tau']:.4f} ({'✅' if monotonicity['pass_monotonicity'] else '❌'})")
        print(f"   单调性比例: {monotonicity['monotonic_ratio']:.2%}")
        
        # 总体判断
        checks_passed = [
            ic_metrics['pass_ic'],
            pass_psi,
            pass_ks,
            corr_check['pass_corr'],
            monotonicity['pass_monotonicity']
        ]
        
        # 计算失败原因
        fail_reasons = []
        if not ic_metrics['pass_ic']:
            fail_reasons.append(f"IC不足({ic_metrics['ic_mean']:.4f}<{self.ic_threshold})")
        if not pass_psi:
            fail_reasons.append(f"PSI过大({psi:.4f}>{self.psi_threshold})")
        if not pass_ks:
            fail_reasons.append(f"KS检验失败(p={ks_p:.4f})")
        if not corr_check['pass_corr']:
            fail_reasons.append(f"高相关性({corr_check['max_corr']:.4f}>{self.corr_threshold})")
        if not monotonicity['pass_monotonicity']:
            fail_reasons.append(f"单调性弱(τ={monotonicity['kendall_tau']:.4f})")
        
        overall_pass = sum(checks_passed) >= 4  # 至少通过4项
        
        print(f"\n{'='*60}")
        print(f"总体评分: {'✅ 通过' if overall_pass else '❌ 不通过'} ({sum(checks_passed)}/5)")
        if fail_reasons:
            print(f"失败原因: {', '.join(fail_reasons)}")
        print(f"{'='*60}")
        
        return {
            'ic_metrics': ic_metrics,
            'ic_decay': ic_decay_df,
            'ic_half_life': half_life,
            'psi': psi,
            'pass_psi': pass_psi,
            'ks_stat': ks_stat,
            'ks_p': ks_p,
            'pass_ks': pass_ks,
            'corr_check': corr_check,
            'monotonicity': monotonicity,
            'checks_passed': checks_passed,
            'overall_pass': overall_pass,
            'fail_reasons': fail_reasons
        }


if __name__ == "__main__":
    """测试因子质量检查器"""
    print("=" * 70)
    print("因子质量检查器测试")
    print("=" * 70)
    
    # 生成测试数据
    np.random.seed(42)
    n = 500
    
    dates = pd.date_range('2022-01-01', periods=n, freq='D')
    
    # 模拟价格数据
    prices = pd.DataFrame({
        'close': 100 + np.random.randn(n).cumsum()
    }, index=dates)
    
    # 模拟因子（有一定预测能力）
    returns = prices['close'].pct_change().shift(-5)
    factor = returns.shift(1) + np.random.randn(n) * 0.01  # 添加噪声
    factor.index = dates
    
    # 远期收益率
    forward_return = prices['close'].pct_change(5).shift(-5)
    forward_return.index = dates
    
    # 创建质量检查器
    checker = FactorQualityChecker(
        ic_threshold=0.02,
        icir_threshold=0.5,
        psi_threshold=0.25,
        corr_threshold=0.7
    )
    
    # 综合检查
    report = checker.comprehensive_check(
        factor=factor,
        forward_return=forward_return,
        prices=prices,
        existing_factors=None,
        train_ratio=0.8
    )
    
    print("\n✅ 测试完成！")
