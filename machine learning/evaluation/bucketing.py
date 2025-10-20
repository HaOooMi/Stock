#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分桶模块 - 按日横截面分桶
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def bucket_predictions(predictions_df: pd.DataFrame,
                      n_buckets: int = 5,
                      method: str = 'quantile') -> pd.DataFrame:
    """
    按日横截面对预测进行分桶
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        预测数据，包含y_pred列，索引为MultiIndex[date, ticker]
    n_buckets : int
        分桶数量
    method : str
        分桶方法: 'quantile'(等分位) 或 'equal_width'(等宽)
        
    Returns:
    --------
    pd.DataFrame
        包含bucket列的预测数据
    """
    print(f"📊 按日横截面分{n_buckets}桶 (方法: {method})")
    
    result_df = predictions_df.copy()
    result_df['bucket'] = np.nan
    
    dates = result_df.index.get_level_values('date').unique()
    
    for date in dates:
        # 获取当日数据
        date_mask = result_df.index.get_level_values('date') == date
        date_predictions = result_df.loc[date_mask, 'y_pred']
        
        if len(date_predictions) < n_buckets:
            # 样本数不足，跳过
            continue
        
        # 分桶
        if method == 'quantile':
            # 等分位分桶
            buckets = pd.qcut(date_predictions, q=n_buckets, labels=False, duplicates='drop')
        elif method == 'equal_width':
            # 等宽分桶
            buckets = pd.cut(date_predictions, bins=n_buckets, labels=False)
        else:
            raise ValueError(f"不支持的分桶方法: {method}")
        
        result_df.loc[date_mask, 'bucket'] = buckets
    
    # 统计分桶结果
    valid_buckets = result_df['bucket'].notna().sum()
    total_samples = len(result_df)
    
    print(f"   ✅ 分桶完成: {valid_buckets}/{total_samples} 样本")
    
    # 显示每个桶的样本数
    bucket_counts = result_df['bucket'].value_counts().sort_index()
    print(f"   📊 各桶样本数:")
    for bucket, count in bucket_counts.items():
        print(f"      桶{int(bucket)+1}: {count} 样本")
    
    return result_df


def analyze_bucket_performance(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    分析各桶的表现
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        包含y_true, y_pred, bucket的数据
        
    Returns:
    --------
    pd.DataFrame
        各桶的统计数据
    """
    print("📊 分析各桶表现...")
    
    # 过滤有效分桶数据
    valid_df = predictions_df.dropna(subset=['bucket'])
    
    # 按桶分组统计
    bucket_stats = []
    
    for bucket in sorted(valid_df['bucket'].unique()):
        bucket_data = valid_df[valid_df['bucket'] == bucket]
        
        stats = {
            'bucket': int(bucket) + 1,  # 从1开始编号
            'n_obs': len(bucket_data),
            'mean_y_true': bucket_data['y_true'].mean(),
            'std_y_true': bucket_data['y_true'].std(),
            'mean_y_pred': bucket_data['y_pred'].mean(),
            'std_y_pred': bucket_data['y_pred'].std(),
            'min_y_true': bucket_data['y_true'].min(),
            'max_y_true': bucket_data['y_true'].max()
        }
        
        bucket_stats.append(stats)
    
    result_df = pd.DataFrame(bucket_stats)
    
    # 计算Top-Bottom spread
    if len(result_df) >= 2:
        top_bucket = result_df.iloc[-1]
        bottom_bucket = result_df.iloc[0]
        spread = top_bucket['mean_y_true'] - bottom_bucket['mean_y_true']
        
        print(f"\n   📊 Top桶 (桶{top_bucket['bucket']}): 平均收益 {top_bucket['mean_y_true']:.4f}")
        print(f"   📊 Bottom桶 (桶{bottom_bucket['bucket']}): 平均收益 {bottom_bucket['mean_y_true']:.4f}")
        print(f"   📈 Top-Bottom Spread: {spread:.4f}")
        
        # 计算全体平均
        overall_mean = predictions_df['y_true'].mean()
        print(f"   📊 全体平均收益: {overall_mean:.4f}")
        
        # 验收检查
        top_vs_mean = top_bucket['mean_y_true'] > overall_mean
        spread_positive = spread > 0
        
        print(f"\n   ✅ 验收结果:")
        print(f"      Top桶 > 全体均值: {'✅ 通过' if top_vs_mean else '❌ 未通过'}")
        print(f"      Spread > 0: {'✅ 通过' if spread_positive else '❌ 未通过'}")
    
    return result_df
