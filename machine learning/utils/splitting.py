#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间序列数据切分模块
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict


def time_series_split(data: pd.DataFrame,
                     train_ratio: float = 0.6,
                     valid_ratio: float = 0.2,
                     test_ratio: float = 0.2,
                     purge_days: int = 0) -> Dict[str, pd.DataFrame]:
    """
    时间序列切分（防泄漏）
    
    Parameters:
    -----------
    data : pd.DataFrame
        数据，索引为时间或MultiIndex[date, ticker]
    train_ratio : float
        训练集比例
    valid_ratio : float
        验证集比例
    test_ratio : float
        测试集比例
    purge_days : int
        在切分点前后清除的天数（防泄漏）
        
    Returns:
    --------
    dict
        {'train': train_df, 'valid': valid_df, 'test': test_df}
    """
    print(f"📅 时间序列数据切分...")
    print(f"   比例: train={train_ratio:.1%}, valid={valid_ratio:.1%}, test={test_ratio:.1%}")
    print(f"   Purge天数: {purge_days}")
    
    # 检查比例
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + valid_ratio + test_ratio 必须等于1.0")
    
    # 获取日期索引
    if isinstance(data.index, pd.MultiIndex):
        dates = data.index.get_level_values('date').unique().sort_values()
    else:
        dates = data.index.unique().sort_values()
    
    n_dates = len(dates)
    
    # 计算切分点
    train_end_idx = int(n_dates * train_ratio)
    valid_end_idx = int(n_dates * (train_ratio + valid_ratio))
    
    # 应用purge
    if purge_days > 0:
        train_end_idx = max(0, train_end_idx - purge_days)
        valid_start_idx = train_end_idx + purge_days
        valid_end_idx = min(n_dates, valid_end_idx)
        test_start_idx = valid_end_idx + purge_days
    else:
        valid_start_idx = train_end_idx
        test_start_idx = valid_end_idx
    
    # 切分日期
    train_dates = dates[:train_end_idx]
    valid_dates = dates[valid_start_idx:valid_end_idx]
    test_dates = dates[test_start_idx:]
    
    # 切分数据
    if isinstance(data.index, pd.MultiIndex):
        train_df = data[data.index.get_level_values('date').isin(train_dates)]
        valid_df = data[data.index.get_level_values('date').isin(valid_dates)]
        test_df = data[data.index.get_level_values('date').isin(test_dates)]
    else:
        train_df = data[data.index.isin(train_dates)]
        valid_df = data[data.index.isin(valid_dates)]
        test_df = data[data.index.isin(test_dates)]
    
    print(f"\n   ✅ 切分完成:")
    print(f"      训练集: {len(train_df)} 样本, {train_dates.min().date()} ~ {train_dates.max().date()}")
    print(f"      验证集: {len(valid_df)} 样本, {valid_dates.min().date() if len(valid_dates)>0 else 'N/A'} ~ {valid_dates.max().date() if len(valid_dates)>0 else 'N/A'}")
    print(f"      测试集: {len(test_df)} 样本, {test_dates.min().date() if len(test_dates)>0 else 'N/A'} ~ {test_dates.max().date() if len(test_dates)>0 else 'N/A'}")
    
    return {
        'train': train_df,
        'valid': valid_df,
        'test': test_df
    }
