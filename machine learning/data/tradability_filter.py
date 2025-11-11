#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易可行性过滤器 - Tradability Filter

功能：
1. 停牌检测与过滤
2. 涨跌停检测与过滤
3. 流动性过滤（成交额、换手率、价格）
4. 上市龄过滤
5. ST/退市股票过滤

符合研究宪章要求的过滤顺序：
过滤1: ST/退市
过滤2: 停牌
过滤3: 涨跌停
过滤4: 上市龄
过滤5: 成交量
过滤6: 价格
过滤7: 换手率
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
project_root = os.path.dirname(ml_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TradabilityFilter:
    """
    交易可行性过滤器
    
    按照研究宪章规定的7层过滤顺序执行:
    1. ST/退市股票
    2. 停牌
    3. 涨跌停
    4. 上市龄
    5. 成交量
    6. 价格
    7. 换手率
    """
    
    def __init__(self,
                 min_volume: float = 1000000,
                 min_amount: float = 50000000,  # 5000万
                 min_price: float = 1.0,
                 min_turnover: float = 0.002,  # 0.2%
                 min_listing_days: int = 60,
                 exclude_st: bool = True,
                 exclude_limit_moves: bool = True,
                 limit_threshold: float = 0.095):  # 9.5%
        """
        初始化过滤器
        
        Parameters:
        -----------
        min_volume : float
            最小成交量（手，1手=100股）
            注意：InfluxDB中存储的成交量单位为"手"
        min_amount : float
            最小成交额（元）
        min_price : float
            最小价格（元）
        min_turnover : float
            最小换手率（注意：akshare返回的是百分比数值，0.2表示0.2%，不是0.002）
        min_listing_days : int
            最小上市天数
        exclude_st : bool
            是否排除ST股票
        exclude_limit_moves : bool
            是否排除涨跌停
        limit_threshold : float
            涨跌停阈值（普通股9.5%，ST股4.5%）
        """
        self.min_volume = min_volume
        self.min_amount = min_amount
        self.min_price = min_price
        self.min_turnover = min_turnover
        self.min_listing_days = min_listing_days
        self.exclude_st = exclude_st
        self.exclude_limit_moves = exclude_limit_moves
        self.limit_threshold = limit_threshold
        
        # 过滤统计
        self.filter_stats = {}
        
        print(f"🔧 交易可行性过滤器初始化")
        print(f"   最小成交量: {min_volume:,.0f}")
        print(f"   最小成交额: {min_amount:,.0f}")
        print(f"   最小价格: {min_price}")
        print(f"   最小换手率: {min_turnover:.2%}")
        print(f"   最小上市天数: {min_listing_days}")
        print(f"   排除ST: {exclude_st}")
        print(f"   排除涨跌停: {exclude_limit_moves}")
    
    def apply_filters(self, 
                     data: pd.DataFrame,
                     save_log: bool = True,
                     log_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        应用所有过滤器
        
        Parameters:
        -----------
        data : pd.DataFrame
            原始数据（需包含必要字段）
        save_log : bool
            是否保存过滤日志
        log_path : str, optional
            日志保存路径
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            (过滤后的数据, 过滤日志)
        """
        print(f"\n🔍 应用交易可行性过滤器")
        print(f"   初始样本数: {len(data):,}")
        
        # 初始化tradable_flag
        data = data.copy()
        data['tradable_flag'] = 1
        
        # 记录每一层过滤的结果
        filter_log = []
        initial_count = len(data)
        
        # 过滤1: ST/退市
        if self.exclude_st:
            data, removed = self._filter_st(data)
            filter_log.append({
                'filter': '1_ST_退市',
                'removed': removed,
                'remaining': len(data[data['tradable_flag'] == 1]),
                'ratio': len(data[data['tradable_flag'] == 1]) / initial_count
            })
            print(f"   ✓ 过滤1 (ST/退市): 剔除 {removed:,}, 剩余 {filter_log[-1]['remaining']:,} ({filter_log[-1]['ratio']:.1%})")
        
        # 过滤2: 停牌
        data, removed = self._filter_suspended(data)
        filter_log.append({
            'filter': '2_停牌',
            'removed': removed,
            'remaining': len(data[data['tradable_flag'] == 1]),
            'ratio': len(data[data['tradable_flag'] == 1]) / initial_count
        })
        print(f"   ✓ 过滤2 (停牌): 剔除 {removed:,}, 剩余 {filter_log[-1]['remaining']:,} ({filter_log[-1]['ratio']:.1%})")
        
        # 过滤3: 涨跌停
        if self.exclude_limit_moves:
            data, removed = self._filter_limit_moves(data)
            filter_log.append({
                'filter': '3_涨跌停',
                'removed': removed,
                'remaining': len(data[data['tradable_flag'] == 1]),
                'ratio': len(data[data['tradable_flag'] == 1]) / initial_count
            })
            print(f"   ✓ 过滤3 (涨跌停): 剔除 {removed:,}, 剩余 {filter_log[-1]['remaining']:,} ({filter_log[-1]['ratio']:.1%})")
        
        # 过滤4: 上市龄
        data, removed = self._filter_listing_days(data)
        filter_log.append({
            'filter': '4_上市龄',
            'removed': removed,
            'remaining': len(data[data['tradable_flag'] == 1]),
            'ratio': len(data[data['tradable_flag'] == 1]) / initial_count
        })
        print(f"   ✓ 过滤4 (上市龄): 剔除 {removed:,}, 剩余 {filter_log[-1]['remaining']:,} ({filter_log[-1]['ratio']:.1%})")
        
        # 过滤5: 成交量
        data, removed = self._filter_volume(data)
        filter_log.append({
            'filter': '5_成交量',
            'removed': removed,
            'remaining': len(data[data['tradable_flag'] == 1]),
            'ratio': len(data[data['tradable_flag'] == 1]) / initial_count
        })
        print(f"   ✓ 过滤5 (成交量): 剔除 {removed:,}, 剩余 {filter_log[-1]['remaining']:,} ({filter_log[-1]['ratio']:.1%})")
        
        # 过滤6: 价格
        data, removed = self._filter_price(data)
        filter_log.append({
            'filter': '6_价格',
            'removed': removed,
            'remaining': len(data[data['tradable_flag'] == 1]),
            'ratio': len(data[data['tradable_flag'] == 1]) / initial_count
        })
        print(f"   ✓ 过滤6 (价格): 剔除 {removed:,}, 剩余 {filter_log[-1]['remaining']:,} ({filter_log[-1]['ratio']:.1%})")
        
        # 过滤7: 换手率
        data, removed = self._filter_turnover(data)
        filter_log.append({
            'filter': '7_换手率',
            'removed': removed,
            'remaining': len(data[data['tradable_flag'] == 1]),
            'ratio': len(data[data['tradable_flag'] == 1]) / initial_count
        })
        print(f"   ✓ 过滤7 (换手率): 剔除 {removed:,}, 剩余 {filter_log[-1]['remaining']:,} ({filter_log[-1]['ratio']:.1%})")
        
        # 创建过滤日志DataFrame
        filter_log_df = pd.DataFrame(filter_log)
        
        # 总结
        final_count = len(data[data['tradable_flag'] == 1])
        total_removed = initial_count - final_count
        
        print(f"\n   ✅ 过滤完成:")
        print(f"      初始样本: {initial_count:,}")
        print(f"      最终样本: {final_count:,}")
        print(f"      剔除样本: {total_removed:,} ({total_removed/initial_count:.1%})")
        
        # 保存过滤日志
        if save_log and log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            filter_log_df.to_csv(log_path, index=False)
            print(f"      📝 过滤日志已保存: {log_path}")
        
        return data, filter_log_df
    
    def _filter_st(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        过滤1: ST/退市股票
        
        检测方式：
        - 股票名称包含'ST'或'*ST'
        - 如果有st_flag列，直接使用
        """
        before_count = len(data[data['tradable_flag'] == 1])
        
        if 'st_flag' in data.columns:
            # 使用st_flag列
            mask = data['st_flag'] == 1
        elif 'name' in data.columns:
            # 检查股票名称
            mask = data['name'].str.contains('ST', na=False)
        else:
            # 无法检测，跳过
            return data, 0
        
        # 标记为不可交易
        data.loc[mask, 'tradable_flag'] = 0
        
        after_count = len(data[data['tradable_flag'] == 1])
        removed = before_count - after_count
        
        return data, removed
    
    def _filter_suspended(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        过滤2: 停牌
        
        检测方式：
        - 成交量为0
        - 成交额为0
        """
        before_count = len(data[data['tradable_flag'] == 1])
        
        mask = data['tradable_flag'] == 1
        
        if 'volume' in data.columns:
            mask = mask & (data['volume'] == 0)
        
        if 'amount' in data.columns:
            mask = mask & (data['amount'] == 0)
        
        # 标记为不可交易
        data.loc[mask, 'tradable_flag'] = 0
        
        after_count = len(data[data['tradable_flag'] == 1])
        removed = before_count - after_count
        
        return data, removed
    
    def _filter_limit_moves(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        过滤3: 涨跌停
        
        检测方式：
        - 动态阈值：根据股票板块和ST状态选择合适阈值
          * 主板非ST: 9.8% (0.098)
          * ST股票: 4.8% (0.048)
          * 创业板/科创板: 19.8% (0.198)
        - 使用涨跌幅绝对值判断，移除"接近涨跌停"的模糊判断
        """
        before_count = len(data[data['tradable_flag'] == 1])
        
        # 计算涨跌幅
        if 'pct_change' in data.columns:
            pct_change = data['pct_change']
        elif 'close' in data.columns:
            if isinstance(data.index, pd.MultiIndex):
                pct_change = data.groupby('ticker')['close'].pct_change()
            else:
                pct_change = data['close'].pct_change()
        else:
            return data, 0
        
        # 判断是否为ST股
        is_st = pd.Series(False, index=data.index)
        if 'st_flag' in data.columns:
            is_st = data['st_flag'] == 1
        elif 'name' in data.columns:
            is_st = data['name'].str.contains('ST', na=False)
        
        # 判断是否为创业板/科创板（股票代码前缀：300xxx创业板，688xxx科创板）
        is_cy_kc = pd.Series(False, index=data.index)
        if isinstance(data.index, pd.MultiIndex):
            tickers = data.index.get_level_values('ticker')
        else:
            # 如果有ticker列
            if 'ticker' in data.columns:
                tickers = data['ticker']
            else:
                tickers = pd.Series(['000000'] * len(data), index=data.index)
        
        # 创业板(300)和科创板(688)判断
        is_cy_kc = tickers.astype(str).str.startswith(('300', '688'))
        
        # 动态设置涨跌停阈值
        limit_threshold = pd.Series(self.limit_threshold, index=data.index)
        
        # ST股票：4.8% (接近实际5%涨跌停)
        limit_threshold[is_st] = 0.048
        
        # 创业板/科创板：19.8% (接近实际20%涨跌停)
        limit_threshold[is_cy_kc & ~is_st] = 0.198
        
        # 主板非ST：使用配置的阈值(推荐0.098，接近实际10%涨跌停)
        # 已在初始化时设置
        
        # 检测涨跌停（移除"接近涨跌停"的模糊判断）
        mask = data['tradable_flag'] == 1
        mask = mask & (pct_change.abs() > limit_threshold)
        
        # 标记为不可交易
        data.loc[mask, 'tradable_flag'] = 0
        
        after_count = len(data[data['tradable_flag'] == 1])
        removed = before_count - after_count
        
        return data, removed
    
    def _filter_listing_days(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        过滤4: 上市龄
        
        检测方式：
        - 上市天数 < min_listing_days
        """
        before_count = len(data[data['tradable_flag'] == 1])
        
        if 'list_date' not in data.columns:
            # 无法检测，跳过
            return data, 0
        
        # 计算上市天数
        if isinstance(data.index, pd.MultiIndex):
            current_date = data.index.get_level_values('date')
        else:
            current_date = pd.to_datetime(data.index)
        
        # 确保日期都是 tz-naive（移除时区信息）
        list_date = pd.to_datetime(data['list_date']).dt.tz_localize(None)
        if hasattr(current_date, 'tz') and current_date.tz is not None:
            current_date = current_date.tz_localize(None)
        
        listing_days = (current_date - list_date).dt.days
        
        # 过滤上市天数不足的
        mask = (data['tradable_flag'] == 1) & (listing_days < self.min_listing_days)
        
        # 标记为不可交易
        data.loc[mask, 'tradable_flag'] = 0
        
        after_count = len(data[data['tradable_flag'] == 1])
        removed = before_count - after_count
        
        return data, removed
    
    def _filter_volume(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        过滤5: 成交量
        """
        before_count = len(data[data['tradable_flag'] == 1])
        
        if 'volume' not in data.columns:
            return data, 0
        
        mask = (data['tradable_flag'] == 1) & (data['volume'] < self.min_volume)
        data.loc[mask, 'tradable_flag'] = 0
        
        after_count = len(data[data['tradable_flag'] == 1])
        removed = before_count - after_count
        
        return data, removed
    
    def _filter_price(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        过滤6: 价格
        """
        before_count = len(data[data['tradable_flag'] == 1])
        
        if 'close' not in data.columns:
            return data, 0
        
        mask = (data['tradable_flag'] == 1) & (data['close'] < self.min_price)
        data.loc[mask, 'tradable_flag'] = 0
        
        after_count = len(data[data['tradable_flag'] == 1])
        removed = before_count - after_count
        
        return data, removed
    
    def _filter_turnover(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        过滤7: 换手率
        
        注意：akshare返回的换手率是百分比数值（如0.5表示0.5%）
        """
        before_count = len(data[data['tradable_flag'] == 1])
        
        # 计算换手率（优先使用从InfluxDB加载的turnover列）
        if 'turnover' in data.columns:
            turnover = data['turnover']
        elif 'turnover_rate' in data.columns:
            turnover = data['turnover_rate']
        elif 'volume' in data.columns and 'shares_outstanding' in data.columns:
            turnover = data['volume'] / data['shares_outstanding']
        else:
            return data, 0
        
        mask = (data['tradable_flag'] == 1) & (turnover < self.min_turnover)
        data.loc[mask, 'tradable_flag'] = 0
        
        after_count = len(data[data['tradable_flag'] == 1])
        removed = before_count - after_count
        
        return data, removed


if __name__ == "__main__":
    """
    使用示例
    """
    print("🔧 交易可行性过滤器测试")
    print("=" * 50)
    
    # 创建示例数据
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
    tickers = ['000001'] * len(dates)
    
    data = pd.DataFrame({
        'close': np.random.randn(len(dates)) * 10 + 100,
        'volume': np.random.randint(0, 10000000, len(dates)),
        'amount': np.random.randint(0, 1000000000, len(dates)),
        'pct_change': np.random.randn(len(dates)) * 0.03,
        'turnover_rate': np.random.rand(len(dates)) * 0.01,
        'st_flag': np.random.choice([0, 1], len(dates), p=[0.95, 0.05]),
        'list_date': '2015-01-01'
    }, index=pd.MultiIndex.from_arrays([dates, tickers], names=['date', 'ticker']))
    
    # 创建过滤器
    filter_engine = TradabilityFilter(
        min_volume=1000000,
        min_amount=50000000,
        min_price=1.0,
        min_turnover=0.002,
        min_listing_days=60,
        exclude_st=True,
        exclude_limit_moves=True
    )
    
    # 应用过滤
    filtered_data, filter_log = filter_engine.apply_filters(
        data,
        save_log=False
    )
    
    print(f"\n✅ 过滤完成")
    print(f"\n过滤日志:")
    print(filter_log)
