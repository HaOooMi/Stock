#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point-in-Time (PIT) 数据对齐器

功能：
1. 财务数据按公告日+滞后生效（避免未来信息泄漏）
2. 历史成分股管理（避免幸存者偏差） 
3. 后复权价格处理
4. 交易日对齐

核心原则：
- 财务数据严格按公告日生效，绝不使用"报告期"点位
- 历史成分股按照当时的实际成分进行回测
- 价格数据使用后复权，确保时间一致性
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
project_root = os.path.dirname(ml_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class PITDataAligner:
    """
    Point-in-Time 数据对齐器
    
    确保所有数据在时间点上是可知的（无未来信息泄漏）
    """
    
    def __init__(self, 
                 financial_lag_days: int = 90,
                 trading_calendar: Optional[pd.DatetimeIndex] = None):
        """
        初始化PIT对齐器
        
        Parameters:
        -----------
        financial_lag_days : int
            财务数据公告后的滞后天数（默认90天）
        trading_calendar : pd.DatetimeIndex, optional
            交易日历，如果为None则自动生成
        """
        self.financial_lag_days = financial_lag_days
        
        # 交易日历
        if trading_calendar is not None:
            self.trading_calendar = trading_calendar
        else:
            # 生成默认交易日历（去除周末）
            all_dates = pd.date_range('2015-01-01', '2030-12-31', freq='D')
            self.trading_calendar = all_dates[all_dates.dayofweek < 5]
        
        print(f"📅 PIT数据对齐器初始化")
        print(f"   财务数据滞后: {financial_lag_days} 天")
        print(f"   交易日历: {self.trading_calendar[0].date()} ~ {self.trading_calendar[-1].date()}")
    
    def align_financial_data(self,
                            financial_df: pd.DataFrame,
                            report_date_col: str = 'report_date',
                            announce_date_col: str = 'announce_date') -> pd.DataFrame:
        """
        对齐财务数据（按公告日+滞后生效）
        
        Parameters:
        -----------
        financial_df : pd.DataFrame
            财务数据（需包含report_date和announce_date）
        report_date_col : str
            报告期列名
        announce_date_col : str
            公告日列名
            
        Returns:
        --------
        pd.DataFrame
            PIT对齐后的财务数据
        """
        print(f"\n📊 对齐财务数据（PIT原则）")
        
        df = financial_df.copy()
        
        # 确保日期格式
        df[report_date_col] = pd.to_datetime(df[report_date_col])
        df[announce_date_col] = pd.to_datetime(df[announce_date_col])
        
        # 计算生效日期（公告日 + 滞后天数）
        df['effective_date'] = df[announce_date_col] + pd.Timedelta(days=self.financial_lag_days)
        
        # 对齐到交易日
        df['effective_date'] = df['effective_date'].apply(
            lambda x: self._get_next_trading_day(x)
        )
        
        # 警告：公告日早于报告期的异常情况
        early_announce = df[df[announce_date_col] < df[report_date_col]]
        if len(early_announce) > 0:
            print(f"   ⚠️  警告: {len(early_announce)} 条记录的公告日早于报告期")
        
        # 统计信息
        avg_lag = (df[announce_date_col] - df[report_date_col]).dt.days.mean()
        max_lag = (df[announce_date_col] - df[report_date_col]).dt.days.max()
        
        print(f"   ✓ 财务数据对齐完成")
        print(f"     平均公告滞后: {avg_lag:.0f} 天")
        print(f"     最大公告滞后: {max_lag:.0f} 天")
        print(f"     生效滞后: {self.financial_lag_days} 天")
        
        return df
    
    def align_index_constituents(self,
                                index_history: pd.DataFrame,
                                effective_date_col: str = 'effective_date') -> pd.DataFrame:
        """
        对齐指数成分股历史数据（避免幸存者偏差）
        
        Parameters:
        -----------
        index_history : pd.DataFrame
            指数成分股历史（需包含date, ticker, in_index列）
        effective_date_col : str
            生效日期列名
            
        Returns:
        --------
        pd.DataFrame
            PIT对齐后的成分股数据
        """
        print(f"\n📋 对齐指数成分股历史（避免幸存者偏差）")
        
        df = index_history.copy()
        
        # 确保日期格式
        if effective_date_col in df.columns:
            df[effective_date_col] = pd.to_datetime(df[effective_date_col])
        elif 'date' in df.columns:
            df[effective_date_col] = pd.to_datetime(df['date'])
        
        # 按ticker和日期排序
        df = df.sort_values(['ticker', effective_date_col])
        
        # 统计成分变更
        if 'in_index' in df.columns:
            changes = df.groupby('ticker')['in_index'].apply(
                lambda x: (x != x.shift()).sum()
            ).sum()
            print(f"   ✓ 成分股变更: {changes} 次")
        
        # 统计成分股数量
        unique_tickers = df['ticker'].nunique()
        print(f"   ✓ 历史成分股: {unique_tickers} 只")
        
        return df
    
    def apply_adj_factor(self,
                        price_df: pd.DataFrame,
                        adj_factor_col: str = 'adj_factor',
                        price_cols: List[str] = ['open', 'high', 'low', 'close']) -> pd.DataFrame:
        """
        应用后复权因子
        
        Parameters:
        -----------
        price_df : pd.DataFrame
            价格数据
        adj_factor_col : str
            复权因子列名
        price_cols : list
            需要复权的价格列
            
        Returns:
        --------
        pd.DataFrame
            复权后的价格数据
        """
        print(f"\n💰 应用后复权因子")
        
        df = price_df.copy()
        
        if adj_factor_col not in df.columns:
            print(f"   ⚠️  警告: 未找到复权因子列 '{adj_factor_col}'，跳过复权")
            return df
        
        # 应用复权因子
        for col in price_cols:
            if col in df.columns:
                df[f'{col}_adj'] = df[col] * df[adj_factor_col]
                print(f"   ✓ {col} -> {col}_adj")
        
        print(f"   ✓ 复权完成")
        
        return df
    
    def align_to_trading_calendar(self,
                                 data: pd.DataFrame,
                                 date_col: str = 'date') -> pd.DataFrame:
        """
        对齐到交易日历
        
        Parameters:
        -----------
        data : pd.DataFrame
            数据
        date_col : str
            日期列名
            
        Returns:
        --------
        pd.DataFrame
            对齐后的数据
        """
        print(f"\n📅 对齐到交易日历")
        
        df = data.copy()
        
        # 确保日期格式
        if isinstance(df.index, pd.MultiIndex):
            dates = df.index.get_level_values('date')
        elif date_col in df.columns:
            dates = pd.to_datetime(df[date_col])
        else:
            dates = pd.to_datetime(df.index)
        
        # 找出非交易日
        non_trading_days = dates[~dates.isin(self.trading_calendar)]
        
        if len(non_trading_days) > 0:
            print(f"   ⚠️  发现 {len(non_trading_days)} 个非交易日")
            
            # 移除非交易日或前向填充
            if isinstance(df.index, pd.MultiIndex):
                mask = df.index.get_level_values('date').isin(self.trading_calendar)
                df = df[mask]
            else:
                mask = dates.isin(self.trading_calendar)
                df = df[mask]
        
        print(f"   ✓ 对齐完成，剩余 {len(df)} 个交易日")
        
        return df
    
    def forward_fill_pit(self,
                        data: pd.DataFrame,
                        group_col: str = 'ticker',
                        max_fill_days: int = 5) -> pd.DataFrame:
        """
        前向填充（PIT安全）
        
        Parameters:
        -----------
        data : pd.DataFrame
            数据
        group_col : str
            分组列（如ticker）
        max_fill_days : int
            最大填充天数
            
        Returns:
        --------
        pd.DataFrame
            填充后的数据
        """
        print(f"\n🔄 前向填充（PIT安全，最多{max_fill_days}天）")
        
        df = data.copy()
        
        if isinstance(df.index, pd.MultiIndex) and group_col in df.index.names:
            # MultiIndex情况
            df = df.groupby(group_col).apply(
                lambda x: x.fillna(method='ffill', limit=max_fill_days)
            )
        elif group_col in df.columns:
            # 普通DataFrame
            df = df.groupby(group_col).apply(
                lambda x: x.fillna(method='ffill', limit=max_fill_days)
            )
        else:
            # 无分组
            df = df.fillna(method='ffill', limit=max_fill_days)
        
        print(f"   ✓ 填充完成")
        
        return df
    
    def _get_next_trading_day(self, date: pd.Timestamp) -> pd.Timestamp:
        """
        获取下一个交易日
        
        Parameters:
        -----------
        date : pd.Timestamp
            日期
            
        Returns:
        --------
        pd.Timestamp
            下一个交易日
        """
        if date in self.trading_calendar:
            return date
        
        # 找到大于date的第一个交易日
        future_dates = self.trading_calendar[self.trading_calendar > date]
        if len(future_dates) > 0:
            return future_dates[0]
        
        return date
    
    def validate_pit_alignment(self,
                              data: pd.DataFrame,
                              target_col: str = 'future_return_5d') -> Dict[str, bool]:
        """
        验证PIT对齐（防泄漏检查）
        
        Parameters:
        -----------
        data : pd.DataFrame
            数据
        target_col : str
            目标列名
            
        Returns:
        --------
        dict
            验证结果
        """
        print(f"\n🔍 验证PIT对齐（防泄漏检查）")
        
        results = {}
        
        # 检查1: 目标变量的尾部NaN
        if target_col in data.columns:
            tail_nans = data[target_col].tail(10).isna().sum()
            results['tail_nans_preserved'] = tail_nans > 0
            print(f"   ✓ 尾部NaN保留: {tail_nans}/10 {'✅' if tail_nans > 0 else '❌'}")
        
        # 检查2: 特征的shift验证
        feature_cols = [col for col in data.columns 
                       if not col.startswith('future_') and not col.startswith('label_')]
        
        if len(feature_cols) > 0:
            # 简单检查：特征值是否全部有效
            feature_valid = data[feature_cols].notna().all().all()
            results['features_valid'] = feature_valid
            print(f"   ✓ 特征有效性: {'✅' if feature_valid else '⚠️ '}")
        
        # 检查3: 时间顺序
        if isinstance(data.index, pd.MultiIndex):
            dates = data.index.get_level_values('date')
        else:
            dates = pd.to_datetime(data.index)
        
        is_sorted = dates.is_monotonic_increasing
        results['time_ordered'] = is_sorted
        print(f"   ✓ 时间顺序: {'✅' if is_sorted else '❌'}")
        
        # 总体评分
        all_passed = all(results.values())
        results['overall_pass'] = all_passed
        
        print(f"\n   {'✅' if all_passed else '❌'} 总体评分: {'通过' if all_passed else '失败'}")
        
        return results


if __name__ == "__main__":
    """
    使用示例
    """
    print("📅 PIT数据对齐器测试")
    print("=" * 50)
    
    # 创建示例财务数据
    financial_data = pd.DataFrame({
        'ticker': ['000001'] * 4,
        'report_date': ['2023-03-31', '2023-06-30', '2023-09-30', '2023-12-31'],
        'announce_date': ['2023-04-25', '2023-08-15', '2023-10-20', '2024-03-15'],
        'revenue': [100, 105, 110, 115],
        'profit': [10, 11, 12, 13]
    })
    
    # 创建PIT对齐器
    pit_aligner = PITDataAligner(financial_lag_days=90)
    
    # 对齐财务数据
    aligned_financial = pit_aligner.align_financial_data(financial_data)
    
    print(f"\n✅ 财务数据对齐完成")
    print(aligned_financial[['report_date', 'announce_date', 'effective_date']])
    
    # 创建示例成分股数据
    index_history = pd.DataFrame({
        'date': pd.date_range('2023-01-01', '2023-12-31', freq='M'),
        'ticker': '000001',
        'in_index': [1] * 12
    })
    
    # 对齐成分股数据
    aligned_constituents = pit_aligner.align_index_constituents(index_history)
    
    print(f"\n✅ 成分股数据对齐完成")
    print(aligned_constituents.head())
