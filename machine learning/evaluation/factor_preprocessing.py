#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子预处理模块

核心功能：
1. Winsorize（极值处理）
2. Z-score标准化（横截面）
3. 行业中性化
4. 市值中性化
5. 综合中性化（市值+行业）
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Union
import warnings

warnings.filterwarnings('ignore')


def winsorize_factor(factors: pd.DataFrame,
                    lower_quantile: float = 0.01,
                    upper_quantile: float = 0.99,
                    cross_section: bool = True) -> pd.DataFrame:
    """
    极值处理（Winsorize）
    
    Parameters:
    -----------
    factors : pd.DataFrame
        因子值，MultiIndex[date, ticker]
    lower_quantile : float
        下分位数（如0.01表示1%）
    upper_quantile : float
        上分位数（如0.99表示99%）
    cross_section : bool
        True: 按日横截面处理（推荐）
        False: 全局处理
        
    Returns:
    --------
    pd.DataFrame
        处理后的因子值
    """
    result = factors.copy()
    
    if cross_section:
        # 按日横截面处理
        dates = factors.index.get_level_values('date').unique()
        
        for col in factors.columns:
            for date in dates:
                date_mask = factors.index.get_level_values('date') == date
                date_values = factors.loc[date_mask, col]
                
                if date_values.notna().sum() < 3:
                    continue
                
                # 计算分位数
                lower_bound = date_values.quantile(lower_quantile)
                upper_bound = date_values.quantile(upper_quantile)
                
                # 裁剪
                result.loc[date_mask, col] = date_values.clip(
                    lower=lower_bound,
                    upper=upper_bound
                )
    else:
        # 全局处理
        for col in factors.columns:
            values = factors[col].dropna()
            
            if len(values) < 3:
                continue
            
            lower_bound = values.quantile(lower_quantile)
            upper_bound = values.quantile(upper_quantile)
            
            result[col] = factors[col].clip(
                lower=lower_bound,
                upper=upper_bound
            )
    
    return result


def standardize_factor(factors: pd.DataFrame,
                       method: str = 'z_score',
                       cross_section: bool = True) -> pd.DataFrame:
    """
    因子标准化
    
    Parameters:
    -----------
    factors : pd.DataFrame
        因子值，MultiIndex[date, ticker]
    method : str
        'z_score': (x - μ) / σ
        'min_max': (x - min) / (max - min)
        'rank': 转换为排名（0-1之间）
    cross_section : bool
        True: 按日横截面标准化（推荐）
        False: 全局标准化
        
    Returns:
    --------
    pd.DataFrame
        标准化后的因子值
    """
    result = factors.copy()
    
    if cross_section:
        # 按日横截面标准化
        dates = factors.index.get_level_values('date').unique()
        
        for col in factors.columns:
            for date in dates:
                date_mask = factors.index.get_level_values('date') == date
                date_values = factors.loc[date_mask, col]
                
                if date_values.notna().sum() < 3:
                    continue
                
                if method == 'z_score':
                    # Z-score标准化
                    mean = date_values.mean()
                    std = date_values.std()
                    
                    if std != 0 and not np.isnan(std):
                        result.loc[date_mask, col] = (date_values - mean) / std
                    else:
                        result.loc[date_mask, col] = 0
                
                elif method == 'min_max':
                    # Min-Max标准化
                    min_val = date_values.min()
                    max_val = date_values.max()
                    
                    if max_val != min_val:
                        result.loc[date_mask, col] = (
                            (date_values - min_val) / (max_val - min_val)
                        )
                    else:
                        result.loc[date_mask, col] = 0.5
                
                elif method == 'rank':
                    # 排名标准化（0-1之间）
                    n = date_values.notna().sum()
                    if n > 0:
                        result.loc[date_mask, col] = (
                            date_values.rank() - 1
                        ) / (n - 1) if n > 1 else 0.5
                
                else:
                    raise ValueError(f"不支持的标准化方法: {method}")
    
    else:
        # 全局标准化
        for col in factors.columns:
            values = factors[col].dropna()
            
            if len(values) < 3:
                continue
            
            if method == 'z_score':
                mean = values.mean()
                std = values.std()
                
                if std != 0 and not np.isnan(std):
                    result[col] = (factors[col] - mean) / std
                else:
                    result[col] = 0
            
            elif method == 'min_max':
                min_val = values.min()
                max_val = values.max()
                
                if max_val != min_val:
                    result[col] = (factors[col] - min_val) / (max_val - min_val)
                else:
                    result[col] = 0.5
            
            elif method == 'rank':
                result[col] = factors[col].rank() / len(values)
    
    return result


def neutralize_factor(factors: pd.DataFrame,
                     market_cap: Optional[pd.DataFrame] = None,
                     industry: Optional[pd.DataFrame] = None,
                     add_constant: bool = True) -> pd.DataFrame:
    """
    因子中性化（回归残差法）
    
    对每个截面日期，回归：
    factor ~ α + β1 * log(market_cap) + β2 * industry_dummies
    
    取残差作为中性化后的因子值
    
    Parameters:
    -----------
    factors : pd.DataFrame
        因子值，MultiIndex[date, ticker]
    market_cap : pd.DataFrame, optional
        市值，MultiIndex[date, ticker]，列为'market_cap'
    industry : pd.DataFrame, optional
        行业，MultiIndex[date, ticker]，列为'industry'（字符串或代码）
    add_constant : bool
        是否添加截距项
        
    Returns:
    --------
    pd.DataFrame
        中性化后的因子值（残差）
    """
    if market_cap is None and industry is None:
        raise ValueError("至少需要提供market_cap或industry之一")
    
    result = factors.copy()
    
    dates = factors.index.get_level_values('date').unique()
    
    for col in factors.columns:
        for date in dates:
            # 获取当日数据
            date_mask = factors.index.get_level_values('date') == date
            date_factors = factors.loc[date_mask, col]
            
            if date_factors.notna().sum() < 3:
                continue
            
            # 构建回归数据
            reg_data = pd.DataFrame({'factor': date_factors})
            
            # 添加市值
            if market_cap is not None:
                date_cap = market_cap.loc[date_mask]
                
                if isinstance(date_cap, pd.DataFrame):
                    cap_col = date_cap.columns[0]
                    reg_data['log_cap'] = np.log(date_cap[cap_col])
                else:
                    reg_data['log_cap'] = np.log(date_cap)
            
            # 添加行业哑变量
            if industry is not None:
                date_industry = industry.loc[date_mask]
                
                if isinstance(date_industry, pd.DataFrame):
                    ind_col = date_industry.columns[0]
                    ind_values = date_industry[ind_col]
                else:
                    ind_values = date_industry
                
                # 创建哑变量（删除第一个类别避免共线性）
                ind_dummies = pd.get_dummies(
                    ind_values,
                    prefix='ind',
                    drop_first=True
                )
                
                reg_data = reg_data.join(ind_dummies)
            
            # 移除缺失值
            reg_data = reg_data.dropna()
            
            if len(reg_data) < 3:
                continue
            
            # 构建自变量矩阵
            X_cols = [c for c in reg_data.columns if c != 'factor']
            
            if len(X_cols) == 0:
                continue
            
            X = reg_data[X_cols].values
            y = reg_data['factor'].values
            
            # 添加截距项
            if add_constant:
                X = np.column_stack([np.ones(len(X)), X])
            
            # OLS回归（使用最小二乘）
            try:
                # β = (X'X)^(-1) X'y
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                
                # 计算残差
                y_pred = X @ beta
                residuals = y - y_pred
                
                # 更新结果
                result.loc[reg_data.index, col] = residuals
                
            except np.linalg.LinAlgError:
                # 矩阵奇异，跳过
                continue
    
    return result


def neutralize_factor_simple(factors: pd.DataFrame,
                             neutralizer: pd.DataFrame,
                             neutralizer_name: str = 'neutralizer') -> pd.DataFrame:
    """
    简单中性化（单变量回归残差）
    
    Parameters:
    -----------
    factors : pd.DataFrame
        因子值，MultiIndex[date, ticker]
    neutralizer : pd.DataFrame
        中性化变量（如市值），MultiIndex[date, ticker]
    neutralizer_name : str
        中性化变量名称（用于日志）
        
    Returns:
    --------
    pd.DataFrame
        中性化后的因子值
    """
    result = factors.copy()
    
    dates = factors.index.get_level_values('date').unique()
    
    for col in factors.columns:
        for date in dates:
            date_mask = factors.index.get_level_values('date') == date
            
            date_factors = factors.loc[date_mask, col]
            date_neutralizer = neutralizer.loc[date_mask]
            
            if isinstance(date_neutralizer, pd.DataFrame):
                date_neutralizer = date_neutralizer.iloc[:, 0]
            
            # 合并数据
            reg_data = pd.DataFrame({
                'factor': date_factors,
                'neutralizer': date_neutralizer
            }).dropna()
            
            if len(reg_data) < 3:
                continue
            
            # 线性回归
            X = reg_data['neutralizer'].values.reshape(-1, 1)
            y = reg_data['factor'].values
            
            # 添加截距
            X = np.column_stack([np.ones(len(X)), X])
            
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                y_pred = X @ beta
                residuals = y - y_pred
                
                result.loc[reg_data.index, col] = residuals
                
            except np.linalg.LinAlgError:
                continue
    
    return result


def preprocess_factor_pipeline(factors: pd.DataFrame,
                               market_cap: Optional[pd.DataFrame] = None,
                               industry: Optional[pd.DataFrame] = None,
                               winsorize: bool = True,
                               standardize: bool = True,
                               neutralize: bool = True,
                               winsorize_params: dict = None,
                               standardize_params: dict = None) -> pd.DataFrame:
    """
    因子预处理流水线
    
    标准流程：
    1. Winsorize（极值处理）
    2. 标准化（Z-score）
    3. 中性化（市值+行业）
    
    Parameters:
    -----------
    factors : pd.DataFrame
        原始因子值
    market_cap : pd.DataFrame, optional
        市值数据
    industry : pd.DataFrame, optional
        行业数据
    winsorize : bool
        是否进行极值处理
    standardize : bool
        是否标准化
    neutralize : bool
        是否中性化
    winsorize_params : dict
        极值处理参数
    standardize_params : dict
        标准化参数
        
    Returns:
    --------
    pd.DataFrame
        预处理后的因子值
    """
    result = factors.copy()
    
    print("📊 因子预处理流水线...")
    
    # 1. Winsorize
    if winsorize:
        params = winsorize_params or {}
        result = winsorize_factor(result, **params)
        print("   ✅ 极值处理完成")
    
    # 2. 标准化
    if standardize:
        params = standardize_params or {}
        result = standardize_factor(result, **params)
        print("   ✅ 标准化完成")
    
    # 3. 中性化
    if neutralize:
        if market_cap is not None or industry is not None:
            result = neutralize_factor(
                result,
                market_cap=market_cap,
                industry=industry
            )
            print("   ✅ 中性化完成")
        else:
            print("   ⚠️  跳过中性化（未提供市值或行业数据）")
    
    print("✅ 预处理完成")
    
    return result


if __name__ == '__main__':
    """测试代码"""
    print("=" * 70)
    print("因子预处理模块测试")
    print("=" * 70)
    
    # 生成测试数据
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
    tickers = [f'Stock_{i:03d}' for i in range(50)]
    
    index = pd.MultiIndex.from_product(
        [dates, tickers],
        names=['date', 'ticker']
    )
    
    # 模拟因子（包含极值）
    factors = pd.DataFrame({
        'factor_1': np.random.randn(len(index)) * 10,
    }, index=index)
    
    # 添加一些极值
    factors.iloc[0] = 1000
    factors.iloc[100] = -1000
    
    # 模拟市值
    market_cap = pd.DataFrame({
        'market_cap': np.random.lognormal(20, 2, len(index))
    }, index=index)
    
    # 模拟行业
    industries = ['金融', '科技', '消费', '医药', '工业']
    industry = pd.DataFrame({
        'industry': np.random.choice(industries, len(index))
    }, index=index)
    
    print("\n1. 原始因子统计...")
    print(f"   均值: {factors['factor_1'].mean():.4f}")
    print(f"   标准差: {factors['factor_1'].std():.4f}")
    print(f"   最小值: {factors['factor_1'].min():.4f}")
    print(f"   最大值: {factors['factor_1'].max():.4f}")
    
    print("\n2. Winsorize测试...")
    winsorized = winsorize_factor(factors, lower_quantile=0.01, upper_quantile=0.99)
    print(f"   处理后最小值: {winsorized['factor_1'].min():.4f}")
    print(f"   处理后最大值: {winsorized['factor_1'].max():.4f}")
    
    print("\n3. 标准化测试...")
    standardized = standardize_factor(winsorized, method='z_score')
    print(f"   Z-score均值: {standardized['factor_1'].mean():.4f}")
    print(f"   Z-score标准差: {standardized['factor_1'].std():.4f}")
    
    print("\n4. 中性化测试（市值）...")
    neutralized_cap = neutralize_factor_simple(
        standardized,
        np.log(market_cap),
        neutralizer_name='log_market_cap'
    )
    print(f"   中性化后均值: {neutralized_cap['factor_1'].mean():.4f}")
    print(f"   中性化后标准差: {neutralized_cap['factor_1'].std():.4f}")
    
    print("\n5. 综合中性化测试（市值+行业）...")
    neutralized_full = neutralize_factor(
        standardized,
        market_cap=market_cap,
        industry=industry
    )
    print(f"   中性化后均值: {neutralized_full['factor_1'].mean():.4f}")
    print(f"   中性化后标准差: {neutralized_full['factor_1'].std():.4f}")
    
    print("\n6. 完整流水线测试...")
    processed = preprocess_factor_pipeline(
        factors,
        market_cap=market_cap,
        industry=industry,
        winsorize=True,
        standardize=True,
        neutralize=True
    )
    print(f"   最终因子均值: {processed['factor_1'].mean():.4f}")
    print(f"   最终因子标准差: {processed['factor_1'].std():.4f}")
    
    print("\n✅ 所有测试完成！")
