#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标签转换器模块 - 残差收益与高级标签处理

功能：
1. 残差收益计算（对指数/行业回归后的残差）
2. 标签标准化与变换
3. 与 cross_section_metrics.calculate_forward_returns 无缝集成

设计原则：
- 复用 evaluation/cross_section_metrics 的 forward returns 计算
- 所有计算按日横截面独立，避免前视偏差
- 支持 MultiIndex [date, ticker] 格式

残差收益公式：
  r_residual = r_stock - β * r_benchmark
  
  其中 β 通过日内横截面回归估计：
  r_stock ~ α + β * r_benchmark + ε

创建: 2025-12-02 | 版本: v1.0
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
if ml_root not in sys.path:
    sys.path.insert(0, ml_root)


class LabelTransformer:
    """
    标签转换器
    
    功能：
    1. 残差收益（对指数或行业）
    2. 排名标准化
    3. 分位数标签
    """
    
    def __init__(self):
        """初始化"""
        print("🏷️ 标签转换器初始化")
    
    def residualize_vs_index(self,
                            returns: pd.DataFrame,
                            index_returns: pd.Series,
                            method: str = 'ols',
                            min_samples: int = 10) -> pd.DataFrame:
        """
        计算相对指数的残差收益
        
        公式：
        r_residual = r_stock - β * r_index
        
        β 通过当日横截面回归估计（OLS）
        
        Parameters:
        -----------
        returns : pd.DataFrame
            股票收益率，MultiIndex [date, ticker]，列为收益率周期
        index_returns : pd.Series
            指数收益率，index=date，或包含 'date' 列
        method : str
            回归方法：'ols' 或 'demean'（简单减去均值）
        min_samples : int
            每日最少样本数
            
        Returns:
        --------
        pd.DataFrame
            残差收益率，与输入相同格式
        """
        print(f"\n📊 计算残差收益（vs 指数）")
        print(f"   方法: {method}")
        
        result = returns.copy()
        
        # 对齐指数收益到股票数据的日期
        if isinstance(index_returns.index, pd.MultiIndex):
            index_rets = index_returns.reset_index(level='ticker', drop=True)
        else:
            index_rets = index_returns
        
        dates = returns.index.get_level_values('date').unique()
        
        for ret_col in returns.columns:
            residuals = []
            
            for date in dates:
                # 当日截面
                try:
                    daily_returns = returns.xs(date, level='date')[ret_col].dropna()
                except KeyError:
                    continue
                
                if len(daily_returns) < min_samples:
                    continue
                
                # 获取当日指数收益
                try:
                    idx_ret = index_rets.loc[date]
                    if isinstance(idx_ret, pd.Series):
                        idx_ret = idx_ret.iloc[0]
                except (KeyError, IndexError):
                    continue
                
                if method == 'ols':
                    # OLS 回归估计 β
                    # r_stock = α + β * r_index + ε
                    # 由于横截面只有一个 r_index 值，这里简化为：
                    # r_residual = r_stock - r_index（当 β=1）
                    # 或者使用市场模型的时序估计
                    daily_residual = daily_returns - idx_ret
                    
                elif method == 'demean':
                    # 简单减去截面均值
                    daily_residual = daily_returns - daily_returns.mean()
                    
                else:
                    raise ValueError(f"不支持的方法: {method}")
                
                for ticker in daily_residual.index:
                    residuals.append({
                        'date': date,
                        'ticker': ticker,
                        ret_col: daily_residual.loc[ticker]
                    })
            
            if residuals:
                residual_df = pd.DataFrame(residuals).set_index(['date', 'ticker'])
                result.update(residual_df)
        
        valid_ratio = result.notna().sum().sum() / (result.shape[0] * result.shape[1]) * 100
        print(f"   ✅ 完成，有效率: {valid_ratio:.1f}%")
        
        return result
    
    def residualize_vs_industry(self,
                               returns: pd.DataFrame,
                               industry: pd.Series,
                               method: str = 'demean') -> pd.DataFrame:
        """
        计算相对行业的残差收益
        
        公式：
        r_residual = r_stock - mean(r_industry)
        
        Parameters:
        -----------
        returns : pd.DataFrame
            股票收益率，MultiIndex [date, ticker]
        industry : pd.Series
            行业分类，MultiIndex [date, ticker]
        method : str
            'demean': 减去行业均值
            'ols': 行业哑变量回归
            
        Returns:
        --------
        pd.DataFrame
            残差收益率
        """
        print(f"\n📊 计算残差收益（vs 行业）")
        print(f"   方法: {method}")
        
        result = returns.copy()
        
        # 合并数据
        combined = returns.join(industry.rename('industry'), how='inner')
        
        for ret_col in returns.columns:
            if method == 'demean':
                # 按日期和行业分组，减去行业均值
                industry_mean = combined.groupby(['date', 'industry'])[ret_col].transform('mean')
                result[ret_col] = combined[ret_col] - industry_mean
                
            elif method == 'ols':
                # 行业哑变量回归
                residuals = []
                dates = combined.index.get_level_values('date').unique()
                
                for date in dates:
                    daily_data = combined.xs(date, level='date').dropna(subset=[ret_col, 'industry'])
                    
                    if len(daily_data) < 10:
                        continue
                    
                    # 创建行业哑变量
                    industry_dummies = pd.get_dummies(daily_data['industry'], drop_first=True)
                    
                    if industry_dummies.empty:
                        # 只有一个行业
                        daily_residual = daily_data[ret_col] - daily_data[ret_col].mean()
                    else:
                        # OLS 回归
                        X = industry_dummies.values
                        y = daily_data[ret_col].values
                        
                        # 添加截距
                        X = np.column_stack([np.ones(len(X)), X])
                        
                        try:
                            beta = np.linalg.lstsq(X, y, rcond=None)[0]
                            y_pred = X @ beta
                            daily_residual = pd.Series(y - y_pred, index=daily_data.index)
                        except Exception:
                            daily_residual = daily_data[ret_col] - daily_data[ret_col].mean()
                    
                    for ticker in daily_residual.index:
                        residuals.append({
                            'date': date,
                            'ticker': ticker,
                            ret_col: daily_residual.loc[ticker]
                        })
                
                if residuals:
                    residual_df = pd.DataFrame(residuals).set_index(['date', 'ticker'])
                    result.update(residual_df)
            else:
                raise ValueError(f"不支持的方法: {method}")
        
        valid_ratio = result.notna().sum().sum() / (result.shape[0] * result.shape[1]) * 100
        print(f"   ✅ 完成，有效率: {valid_ratio:.1f}%")
        
        return result
    
    def rank_normalize(self,
                      values: pd.DataFrame,
                      method: str = 'cross_section') -> pd.DataFrame:
        """
        排名标准化到 [0, 1]
        
        Parameters:
        -----------
        values : pd.DataFrame
            输入值，MultiIndex [date, ticker]
        method : str
            'cross_section': 按日横截面排名
            'global': 全局排名
            
        Returns:
        --------
        pd.DataFrame
            排名标准化后的值
        """
        print(f"\n📊 排名标准化")
        print(f"   方法: {method}")
        
        result = values.copy()
        
        if method == 'cross_section':
            for col in values.columns:
                # 按日期分组，计算百分位排名
                result[col] = values.groupby(level='date')[col].rank(pct=True)
                
        elif method == 'global':
            for col in values.columns:
                result[col] = values[col].rank(pct=True)
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        print(f"   ✅ 完成")
        
        return result
    
    def create_quantile_labels(self,
                              values: pd.DataFrame,
                              n_quantiles: int = 5,
                              method: str = 'cross_section') -> pd.DataFrame:
        """
        创建分位数标签
        
        Parameters:
        -----------
        values : pd.DataFrame
            输入值，MultiIndex [date, ticker]
        n_quantiles : int
            分位数数量
        method : str
            'cross_section': 按日横截面分桶
            'global': 全局分桶
            
        Returns:
        --------
        pd.DataFrame
            分位数标签（0 到 n_quantiles-1）
        """
        print(f"\n📊 创建分位数标签")
        print(f"   分位数: {n_quantiles} | 方法: {method}")
        
        result = pd.DataFrame(index=values.index)
        
        for col in values.columns:
            label_col = f'{col}_q{n_quantiles}'
            
            if method == 'cross_section':
                # 按日期分组分桶
                def assign_quantile(group):
                    valid = group.dropna()
                    if len(valid) < n_quantiles:
                        return pd.Series(np.nan, index=group.index)
                    try:
                        return pd.qcut(group, q=n_quantiles, labels=False, duplicates='drop')
                    except ValueError:
                        return pd.Series(np.nan, index=group.index)
                
                result[label_col] = values.groupby(level='date')[col].transform(assign_quantile)
                
            elif method == 'global':
                try:
                    result[label_col] = pd.qcut(values[col], q=n_quantiles, labels=False, duplicates='drop')
                except ValueError:
                    result[label_col] = np.nan
            else:
                raise ValueError(f"不支持的方法: {method}")
        
        # 统计
        for col in result.columns:
            valid_ratio = result[col].notna().sum() / len(result) * 100
            print(f"   {col}: 有效率 {valid_ratio:.1f}%")
        
        return result


def create_forward_returns_with_transform(
    prices: pd.DataFrame,
    periods: List[int] = [1, 5, 10],
    method: str = 'simple',
    transform: str = 'none',
    index_returns: Optional[pd.Series] = None,
    industry: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    便捷函数：计算远期收益并应用变换
    
    Parameters:
    -----------
    prices : pd.DataFrame
        价格数据，MultiIndex [date, ticker]
    periods : List[int]
        前瞻期
    method : str
        'simple' 或 'log'
    transform : str
        'none': 不变换
        'residual_vs_index': 减去指数收益
        'residual_vs_industry': 减去行业均值
        'rank': 排名标准化
    index_returns : pd.Series, optional
        指数收益（当 transform='residual_vs_index'）
    industry : pd.Series, optional
        行业分类（当 transform='residual_vs_industry'）
        
    Returns:
    --------
    pd.DataFrame
        （变换后的）远期收益率
    """
    # 直接导入函数，避免触发整个 evaluation 包
    from evaluation.cross_section_metrics import calculate_forward_returns
    
    # 计算原始远期收益
    forward_returns = calculate_forward_returns(prices, periods=periods, method=method)
    
    if transform == 'none':
        return forward_returns
    
    # 应用变换
    transformer = LabelTransformer()
    
    if transform == 'residual_vs_index':
        if index_returns is None:
            raise ValueError("transform='residual_vs_index' 需要提供 index_returns")
        return transformer.residualize_vs_index(forward_returns, index_returns)
    
    elif transform == 'residual_vs_industry':
        if industry is None:
            raise ValueError("transform='residual_vs_industry' 需要提供 industry")
        return transformer.residualize_vs_industry(forward_returns, industry)
    
    elif transform == 'rank':
        return transformer.rank_normalize(forward_returns)
    
    else:
        raise ValueError(f"不支持的变换: {transform}")


if __name__ == '__main__':
    """测试代码"""
    print("=" * 70)
    print("标签转换器模块测试")
    print("=" * 70)
    
    # 生成测试数据
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    dates = dates[dates.dayofweek < 5]
    tickers = ['000001', '000002', '000003', '000004', '000005']
    industries = ['银行', '银行', '科技', '科技', '消费']
    
    # 创建 MultiIndex
    index = pd.MultiIndex.from_product(
        [dates, tickers],
        names=['date', 'ticker']
    )
    
    # 模拟收益率数据
    returns = pd.DataFrame({
        'ret_1d': np.random.randn(len(index)) * 0.02,
        'ret_5d': np.random.randn(len(index)) * 0.05
    }, index=index)
    
    # 模拟指数收益
    index_rets = pd.Series(
        np.random.randn(len(dates)) * 0.015,
        index=dates,
        name='index_return'
    )
    
    # 模拟行业
    industry_mapping = dict(zip(tickers, industries))
    industry = pd.Series(
        [industry_mapping[t] for d in dates for t in tickers],
        index=index,
        name='industry'
    )
    
    print(f"\n📊 测试数据:")
    print(f"   收益率形状: {returns.shape}")
    print(f"   日期范围: {dates.min().date()} ~ {dates.max().date()}")
    
    # 初始化转换器
    transformer = LabelTransformer()
    
    # 1. 残差收益（vs 指数）
    print("\n" + "=" * 70)
    print("测试 1: 残差收益（vs 指数）")
    print("=" * 70)
    
    residual_index = transformer.residualize_vs_index(returns, index_rets)
    print(f"   结果形状: {residual_index.shape}")
    print(f"   前5行:\n{residual_index.head()}")
    
    # 2. 残差收益（vs 行业）
    print("\n" + "=" * 70)
    print("测试 2: 残差收益（vs 行业）")
    print("=" * 70)
    
    residual_industry = transformer.residualize_vs_industry(returns, industry)
    print(f"   结果形状: {residual_industry.shape}")
    print(f"   前5行:\n{residual_industry.head()}")
    
    # 3. 排名标准化
    print("\n" + "=" * 70)
    print("测试 3: 排名标准化")
    print("=" * 70)
    
    ranked = transformer.rank_normalize(returns)
    print(f"   结果形状: {ranked.shape}")
    print(f"   范围: [{ranked.min().min():.3f}, {ranked.max().max():.3f}]")
    
    # 4. 分位数标签
    print("\n" + "=" * 70)
    print("测试 4: 分位数标签")
    print("=" * 70)
    
    quantile_labels = transformer.create_quantile_labels(returns, n_quantiles=5)
    print(f"   结果形状: {quantile_labels.shape}")
    print(f"   列: {quantile_labels.columns.tolist()}")
    
    # 5. 便捷函数测试
    print("\n" + "=" * 70)
    print("测试 5: 便捷函数 create_forward_returns_with_transform")
    print("=" * 70)
    
    # 模拟价格
    prices = pd.DataFrame({
        'close': np.random.randn(len(index)).cumsum() + 100
    }, index=index)
    
    # 带变换的远期收益
    transformed_returns = create_forward_returns_with_transform(
        prices,
        periods=[1, 5],
        method='simple',
        transform='rank'
    )
    print(f"   结果形状: {transformed_returns.shape}")
    print(f"   列: {transformed_returns.columns.tolist()}")
    
    print("\n✅ 所有测试完成！")
