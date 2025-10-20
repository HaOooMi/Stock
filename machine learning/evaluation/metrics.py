#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标计算模块
"""

import pandas as pd
import numpy as np
from typing import Dict
from scipy import stats


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict:
    """
    计算评估指标
    
    Parameters:
    -----------
    y_true : pd.Series
        真实值
    y_pred : np.ndarray
        预测值
        
    Returns:
    --------
    dict
        评估指标字典
    """
    # 基本回归指标
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    # 信息系数 (IC)
    ic, ic_pvalue = stats.pearsonr(y_true, y_pred)
    
    # 排序信息系数 (Rank IC)
    rank_ic, rank_ic_pvalue = stats.spearmanr(y_true, y_pred)
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'ic': ic,
        'ic_pvalue': ic_pvalue,
        'rank_ic': rank_ic,
        'rank_ic_pvalue': rank_ic_pvalue,
        'n_samples': len(y_true)
    }


def calculate_ic_by_date(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    按日期计算IC
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        包含y_true和y_pred的预测数据，索引为MultiIndex[date, ticker]
        
    Returns:
    --------
    pd.DataFrame
        每日IC统计
    """
    daily_ic = []
    
    dates = predictions_df.index.get_level_values('date').unique()
    
    for date in dates:
        date_data = predictions_df.xs(date, level='date')
        
        if len(date_data) >= 3:  # 至少需要3个样本
            ic, pval = stats.spearmanr(date_data['y_true'], date_data['y_pred'])
            daily_ic.append({
                'date': date,
                'ic': ic,
                'pvalue': pval,
                'n_samples': len(date_data)
            })
    
    return pd.DataFrame(daily_ic).set_index('date')
