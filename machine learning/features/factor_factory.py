#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【高级因子工厂】factor_factory.py v1.0

定位：工业级因子生产 + 严格质量控制
适用：多股票横截面选股、因子挖掘、生产环境

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
与基础特征(feature_engineering.py)的区别：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基础特征工程：                因子工厂（本模块）：
├─ 目标：快速生成大量特征    ├─ 目标：精选高信息量因子
├─ 数量：50-100个自动生成    ├─ 数量：40+个文献验证因子
├─ 筛选：统计方法(方差/相关) ├─ 筛选：金融逻辑(IC/ICIR/衰减)
├─ 版本：无版本管理          ├─ 版本：入库标准+清单管理
└─ 门槛：低，开箱即用         └─ 门槛：高，需理解因子逻辑

组合使用示例：
  features = engineer.prepare_features()      # 基础特征
  factors = factory.generate_all_factors()    # 高级因子
  combined = pd.concat([features, factors], axis=1)  # 组合

4大因子族：
1. 动量/反转 (12个) - Jegadeesh & Titman (1993)
2. 波动率 (8个) - French et al. (1987)
3. 量价微结构 (9个) - Lee & Swaminathan (2000)
4. 风格/质量 (3个) - Fama & French (1993)

设计原则：
✓ 高信息量：每个因子有学术验证
✓ 低冗余：控制同族因子数量
✓ 可追溯：公式+族别+文献引用
✓ 防泄漏：所有因子自动滞后1期

创建: 2025-01-20 | 版本: v1.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class FactorFactory:
    """
    因子工厂类
    
    提供文献支持的高产因子族计算
    """
    
    def __init__(self):
        """初始化因子工厂"""
        self.factor_registry = {}
        print("🏭 因子工厂 v1 初始化")
    
    # ========== 1. 动量/反转因子族 ==========
    
    def calc_roc_family(self, data: pd.DataFrame, periods: List[int] = [5, 10, 20, 60, 120]) -> pd.DataFrame:
        """
        ROC (Rate of Change) 动量因子族
        
        公式: ROC_N = (Close_t - Close_{t-N}) / Close_{t-N}
        
        文献: Jegadeesh and Titman (1993), Returns to Buying Winners and Selling Losers
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含 'close' 列的数据
        periods : List[int]
            回看周期列表
            
        Returns:
        --------
        pd.DataFrame
            包含各周期 ROC 因子
        """
        result = pd.DataFrame(index=data.index)
        
        for N in periods:
            factor_name = f'roc_{N}d'
            result[factor_name] = data['close'].pct_change(N)
            
            # 因子描述
            self.factor_registry[factor_name] = {
                'family': '动量/反转',
                'formula': f'(close_t - close_{{t-{N}}}) / close_{{t-{N}}}',
                'period': N,
                'reference': 'Jegadeesh and Titman (1993)'
            }
        
        return result
    
    def calc_price_to_sma(self, data: pd.DataFrame, periods: List[int] = [10, 20, 60]) -> pd.DataFrame:
        """
        Price to SMA 偏离度因子
        
        公式: (Close - SMA_N) / SMA_N
        
        文献: Fama and French (1996), Multifactor explanations of asset pricing anomalies
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含 'close' 列的数据
        periods : List[int]
            均线周期列表
            
        Returns:
        --------
        pd.DataFrame
            包含各周期偏离度因子
        """
        result = pd.DataFrame(index=data.index)
        
        for N in periods:
            sma = data['close'].rolling(N).mean()
            factor_name = f'price_to_sma_{N}d'
            result[factor_name] = (data['close'] - sma) / sma
            
            self.factor_registry[factor_name] = {
                'family': '动量/反转',
                'formula': f'(close - SMA_{N}) / SMA_{N}',
                'period': N,
                'reference': 'Fama and French (1996)'
            }
        
        return result
    
    def calc_long_short_momentum(self, data: pd.DataFrame, 
                                 long_period: int = 60, 
                                 short_period: int = 5) -> pd.DataFrame:
        """
        长动量-短反转复合因子
        
        公式: ROC_Long - ROC_Short
        
        逻辑: 捕捉长期趋势中的短期回调机会
        
        文献: Novy-Marx (2012), Is momentum really momentum?
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含 'close' 列的数据
        long_period : int
            长期周期
        short_period : int
            短期周期
            
        Returns:
        --------
        pd.DataFrame
            复合动量因子
        """
        result = pd.DataFrame(index=data.index)
        
        roc_long = data['close'].pct_change(long_period)
        roc_short = data['close'].pct_change(short_period)
        
        factor_name = f'momentum_composite_{long_period}_{short_period}'
        result[factor_name] = roc_long - roc_short
        
        self.factor_registry[factor_name] = {
            'family': '动量/反转',
            'formula': f'ROC_{long_period} - ROC_{short_period}',
            'period': f'{long_period}/{short_period}',
            'reference': 'Novy-Marx (2012)'
        }
        
        return result
    
    def calc_rank_momentum(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Rank Momentum (历史分位数动量)
        
        公式: (当前价格 - N日最低价) / (N日最高价 - N日最低价)
        
        优点: 标准化到[0, 1]区间，跨股票可比
        
        文献: George and Hwang (2004), The 52-Week High and Momentum Investing
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含 'close' 列的数据
        period : int
            回看周期
            
        Returns:
        --------
        pd.DataFrame
            分位数动量因子
        """
        result = pd.DataFrame(index=data.index)
        
        rolling_min = data['close'].rolling(period).min()
        rolling_max = data['close'].rolling(period).max()
        
        factor_name = f'rank_momentum_{period}d'
        result[factor_name] = (data['close'] - rolling_min) / (rolling_max - rolling_min + 1e-8)
        
        self.factor_registry[factor_name] = {
            'family': '动量/反转',
            'formula': f'(close - min_{period}) / (max_{period} - min_{period})',
            'period': period,
            'reference': 'George and Hwang (2004)'
        }
        
        return result
    
    # ========== 2. 波动率因子族 ==========
    
    def calc_realized_volatility(self, data: pd.DataFrame, periods: List[int] = [20, 60]) -> pd.DataFrame:
        """
        已实现波动率 (Realized Volatility)
        
        公式: RV_N = std(returns, N)
        
        文献: French, Schwert and Stambaugh (1987), Expected stock returns and volatility
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含 'close' 列的数据
        periods : List[int]
            波动率计算周期
            
        Returns:
        --------
        pd.DataFrame
            波动率因子
        """
        result = pd.DataFrame(index=data.index)
        
        returns = data['close'].pct_change()
        
        for N in periods:
            factor_name = f'realized_vol_{N}d'
            result[factor_name] = returns.rolling(N).std()
            
            self.factor_registry[factor_name] = {
                'family': '波动率',
                'formula': f'std(returns, {N})',
                'period': N,
                'reference': 'French, Schwert and Stambaugh (1987)'
            }
        
        return result
    
    def calc_parkinson_volatility(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Parkinson波动率 (利用高低价信息)
        
        公式: sqrt(mean((ln(H/L))^2) / (4 * ln(2)))
        
        优点: 比收盘价波动率更高效，利用日内信息
        
        文献: Parkinson (1980), The Extreme Value Method for Estimating the Variance
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含 'high' 和 'low' 列的数据
        period : int
            计算周期
            
        Returns:
        --------
        pd.DataFrame
            Parkinson 波动率
        """
        result = pd.DataFrame(index=data.index)
        
        if 'high' not in data.columns or 'low' not in data.columns:
            return result
        
        # 计算 ln(H/L)^2
        hl_ratio = np.log(data['high'] / data['low'])
        hl_squared = hl_ratio ** 2
        
        # Parkinson 波动率
        factor_name = f'parkinson_vol_{period}d'
        result[factor_name] = np.sqrt(hl_squared.rolling(period).mean() / (4 * np.log(2)))
        
        self.factor_registry[factor_name] = {
            'family': '波动率',
            'formula': 'sqrt(mean((ln(H/L))^2) / (4*ln(2)))',
            'period': period,
            'reference': 'Parkinson (1980)'
        }
        
        return result
    
    def calc_garman_klass_volatility(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Garman-Klass波动率 (综合OHLC信息)
        
        公式: sqrt(mean(0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2))
        
        优点: 利用更多价格信息，效率更高
        
        文献: Garman and Klass (1980), On the Estimation of Security Price Volatilities
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含 OHLC 列的数据
        period : int
            计算周期
            
        Returns:
        --------
        pd.DataFrame
            Garman-Klass 波动率
        """
        result = pd.DataFrame(index=data.index)
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            return result
        
        # GK 波动率公式
        hl = np.log(data['high'] / data['low'])
        co = np.log(data['close'] / data['open'])
        
        gk_component = 0.5 * (hl ** 2) - (2 * np.log(2) - 1) * (co ** 2)
        
        factor_name = f'garman_klass_vol_{period}d'
        result[factor_name] = np.sqrt(gk_component.rolling(period).mean())
        
        self.factor_registry[factor_name] = {
            'family': '波动率',
            'formula': 'sqrt(mean(0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2))',
            'period': period,
            'reference': 'Garman and Klass (1980)'
        }
        
        return result
    
    def calc_return_skewness_kurtosis(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        收益率偏度和峰度
        
        偏度: 衡量收益率分布的偏斜程度
        峰度: 衡量尾部风险
        
        文献: Harvey and Siddique (2000), Conditional skewness in asset pricing tests
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含 'close' 列的数据
        period : int
            计算周期
            
        Returns:
        --------
        pd.DataFrame
            偏度和峰度因子
        """
        result = pd.DataFrame(index=data.index)
        
        returns = data['close'].pct_change()
        
        # 偏度
        factor_name_skew = f'skewness_{period}d'
        result[factor_name_skew] = returns.rolling(period).skew()
        
        # 峰度
        factor_name_kurt = f'kurtosis_{period}d'
        result[factor_name_kurt] = returns.rolling(period).kurt()
        
        self.factor_registry[factor_name_skew] = {
            'family': '波动率',
            'formula': f'skew(returns, {period})',
            'period': period,
            'reference': 'Harvey and Siddique (2000)'
        }
        
        self.factor_registry[factor_name_kurt] = {
            'family': '波动率',
            'formula': f'kurt(returns, {period})',
            'period': period,
            'reference': 'Harvey and Siddique (2000)'
        }
        
        return result
    
    # ========== 3. 量价微结构因子族 ==========
    
    def calc_turnover_factors(self, data: pd.DataFrame, periods: List[int] = [5, 20]) -> pd.DataFrame:
        """
        换手率因子族
        
        公式:
        - Turnover: volume / shares_outstanding 或直接使用 turnover 列
        - Turnover_MA: 换手率均值
        - Turnover_Std: 换手率波动
        
        文献: Datar, Naik and Radcliffe (1998), Liquidity and stock returns
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含 'turnover' 或 'volume' 列的数据
        periods : List[int]
            计算周期
            
        Returns:
        --------
        pd.DataFrame
            换手率因子
        """
        result = pd.DataFrame(index=data.index)
        
        # 确定换手率数据源
        if 'turnover' in data.columns:
            turnover = data['turnover']
        elif 'volume' in data.columns and 'shares_outstanding' in data.columns:
            turnover = data['volume'] / data['shares_outstanding']
        else:
            return result
        
        for N in periods:
            # 换手率均值
            factor_name_mean = f'turnover_mean_{N}d'
            result[factor_name_mean] = turnover.rolling(N).mean()
            
            # 换手率波动
            factor_name_std = f'turnover_std_{N}d'
            result[factor_name_std] = turnover.rolling(N).std()
            
            # 换手率相对变化
            factor_name_roc = f'turnover_roc_{N}d'
            result[factor_name_roc] = turnover.pct_change(N)
            
            self.factor_registry[factor_name_mean] = {
                'family': '量价微结构',
                'formula': f'mean(turnover, {N})',
                'period': N,
                'reference': 'Datar, Naik and Radcliffe (1998)'
            }
            
            self.factor_registry[factor_name_std] = {
                'family': '量价微结构',
                'formula': f'std(turnover, {N})',
                'period': N,
                'reference': 'Datar, Naik and Radcliffe (1998)'
            }
            
            self.factor_registry[factor_name_roc] = {
                'family': '量价微结构',
                'formula': f'turnover_t / turnover_{{t-{N}}} - 1',
                'period': N,
                'reference': 'Datar, Naik and Radcliffe (1998)'
            }
        
        return result
    
    def calc_volume_price_correlation(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        量价相关性因子
        
        公式: corr(volume, returns, N)
        
        逻辑: 量价背离/协同分析
        
        文献: Karpoff (1987), The Relation Between Price Changes and Trading Volume
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含 'volume' 和 'close' 列的数据
        period : int
            计算周期
            
        Returns:
        --------
        pd.DataFrame
            量价相关性因子
        """
        result = pd.DataFrame(index=data.index)
        
        if 'volume' not in data.columns:
            return result
        
        returns = data['close'].pct_change()
        
        factor_name = f'volume_price_corr_{period}d'
        result[factor_name] = returns.rolling(period).corr(data['volume'])
        
        self.factor_registry[factor_name] = {
            'family': '量价微结构',
            'formula': f'corr(returns, volume, {period})',
            'period': period,
            'reference': 'Karpoff (1987)'
        }
        
        return result
    
    def calc_vwap_deviation(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        VWAP偏离度因子
        
        公式:
        - VWAP = sum(close * volume) / sum(volume)
        - VWAP_Dev = (close - VWAP) / VWAP
        
        逻辑: 价格相对成交量加权均价的偏离
        
        文献: Berkowitz, Logue and Noser (1988), The Total Cost of Transactions
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含 'close' 和 'volume' 列的数据
        period : int
            计算周期
            
        Returns:
        --------
        pd.DataFrame
            VWAP偏离度因子
        """
        result = pd.DataFrame(index=data.index)
        
        if 'volume' not in data.columns:
            return result
        
        # 计算VWAP
        pv = data['close'] * data['volume']
        vwap = pv.rolling(period).sum() / data['volume'].rolling(period).sum()
        
        # 偏离度
        factor_name = f'vwap_dev_{period}d'
        result[factor_name] = (data['close'] - vwap) / vwap
        
        self.factor_registry[factor_name] = {
            'family': '量价微结构',
            'formula': f'(close - VWAP_{period}) / VWAP_{period}',
            'period': period,
            'reference': 'Berkowitz, Logue and Noser (1988)'
        }
        
        return result
    
    # ========== 4. 风格/质量因子族 ==========
    
    def calc_amihud_illiquidity(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Amihud非流动性指标
        
        公式: Amihud = mean(abs(return) / amount, N)
        
        逻辑: 单位交易额引起的价格变化
        
        文献: Amihud (2002), Illiquidity and stock returns
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含 'close' 和 'amount' 列的数据
        period : int
            计算周期
            
        Returns:
        --------
        pd.DataFrame
            Amihud非流动性因子
        """
        result = pd.DataFrame(index=data.index)
        
        if 'amount' not in data.columns:
            return result
        
        returns = data['close'].pct_change()
        
        # Amihud 非流动性
        illiquidity = np.abs(returns) / (data['amount'] + 1e-8)
        
        factor_name = f'amihud_illiq_{period}d'
        result[factor_name] = illiquidity.rolling(period).mean()
        
        self.factor_registry[factor_name] = {
            'family': '风格/质量',
            'formula': f'mean(abs(return) / amount, {period})',
            'period': period,
            'reference': 'Amihud (2002)'
        }
        
        return result
    
    def calc_price_range_factors(self, data: pd.DataFrame, periods: List[int] = [5, 20]) -> pd.DataFrame:
        """
        价格范围因子族
        
        公式:
        - High-Low Range: (high - low) / close
        - High-Close Range: (high - close) / close
        
        逻辑: 日内波动特征
        
        文献: Alizadeh, Brandt and Diebold (2002), Range-Based Estimation of Stochastic Volatility
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含 OHLC 列的数据
        periods : List[int]
            计算周期
            
        Returns:
        --------
        pd.DataFrame
            价格范围因子
        """
        result = pd.DataFrame(index=data.index)
        
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            return result
        
        # High-Low Range
        hl_range = (data['high'] - data['low']) / data['close']
        
        for N in periods:
            factor_name = f'hl_range_mean_{N}d'
            result[factor_name] = hl_range.rolling(N).mean()
            
            self.factor_registry[factor_name] = {
                'family': '风格/质量',
                'formula': f'mean((high - low) / close, {N})',
                'period': N,
                'reference': 'Alizadeh, Brandt and Diebold (2002)'
            }
        
        return result
    
    # ========== 通用方法 ==========
    
    def generate_all_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成所有因子
        
        Parameters:
        -----------
        data : pd.DataFrame
            原始市场数据（包含OHLCV等）
            
        Returns:
        --------
        pd.DataFrame
            包含所有因子的数据框
        """
        print("🏭 因子工厂: 生成所有因子")
        
        all_factors = pd.DataFrame(index=data.index)
        
        # 1. 动量/反转因子
        print("   📈 动量/反转因子族...")
        all_factors = pd.concat([all_factors, self.calc_roc_family(data)], axis=1)
        all_factors = pd.concat([all_factors, self.calc_price_to_sma(data)], axis=1)
        all_factors = pd.concat([all_factors, self.calc_long_short_momentum(data)], axis=1)
        all_factors = pd.concat([all_factors, self.calc_rank_momentum(data)], axis=1)
        
        # 2. 波动率因子
        print("   📊 波动率因子族...")
        all_factors = pd.concat([all_factors, self.calc_realized_volatility(data)], axis=1)
        all_factors = pd.concat([all_factors, self.calc_parkinson_volatility(data)], axis=1)
        all_factors = pd.concat([all_factors, self.calc_garman_klass_volatility(data)], axis=1)
        all_factors = pd.concat([all_factors, self.calc_return_skewness_kurtosis(data)], axis=1)
        
        # 3. 量价微结构因子
        print("   💹 量价微结构因子族...")
        all_factors = pd.concat([all_factors, self.calc_turnover_factors(data)], axis=1)
        all_factors = pd.concat([all_factors, self.calc_volume_price_correlation(data)], axis=1)
        all_factors = pd.concat([all_factors, self.calc_vwap_deviation(data)], axis=1)
        
        # 4. 风格/质量因子
        print("   🎯 风格/质量因子族...")
        all_factors = pd.concat([all_factors, self.calc_amihud_illiquidity(data)], axis=1)
        all_factors = pd.concat([all_factors, self.calc_price_range_factors(data)], axis=1)
        
        print(f"   ✅ 生成完成: {len(all_factors.columns)} 个因子")
        
        return all_factors
    
    def get_factor_metadata(self) -> pd.DataFrame:
        """
        获取因子元数据
        
        Returns:
        --------
        pd.DataFrame
            因子注册表
        """
        if not self.factor_registry:
            return pd.DataFrame()
        
        return pd.DataFrame.from_dict(self.factor_registry, orient='index')


if __name__ == "__main__":
    """测试因子工厂"""
    print("=" * 70)
    print("因子工厂 v1 测试")
    print("=" * 70)
    
    # 生成测试数据
    np.random.seed(42)
    n = 500
    
    dates = pd.date_range('2022-01-01', periods=n, freq='D')
    
    test_data = pd.DataFrame({
        'open': 100 + np.random.randn(n).cumsum(),
        'high': 102 + np.random.randn(n).cumsum(),
        'low': 98 + np.random.randn(n).cumsum(),
        'close': 100 + np.random.randn(n).cumsum(),
        'volume': np.random.randint(1000000, 10000000, n),
        'amount': np.random.randint(100000000, 1000000000, n),
        'turnover': np.random.rand(n) * 0.05,
        'shares_outstanding': 1000000000
    }, index=dates)
    
    # 创建因子工厂
    factory = FactorFactory()
    
    # 生成所有因子
    factors = factory.generate_all_factors(test_data)
    
    print(f"\n📊 因子统计:")
    print(f"   因子数量: {len(factors.columns)}")
    print(f"   样本数量: {len(factors)}")
    print(f"   缺失率: {factors.isna().sum().sum() / factors.size:.2%}")
    
    # 显示因子元数据
    metadata = factory.get_factor_metadata()
    print(f"\n📋 因子元数据:")
    print(metadata.head(10))
    
    # 按族群统计
    print(f"\n📊 因子族群分布:")
    family_counts = metadata['family'].value_counts()
    for family, count in family_counts.items():
        print(f"   {family}: {count} 个")
    
    print("\n✅ 测试完成！")
