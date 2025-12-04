#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
排序标签构造模块 - Ranking Labels

功能：
1. Reg-on-Rank 标签：将收益率转为横截面排序值（连续）
2. LambdaRank 标签：将收益率转为分箱等级（离散）+ group 向量
3. 与现有 LabelTransformer 接口兼容

设计原则：
- 所有计算按日横截面独立，避免前视偏差
- 支持 MultiIndex [date, ticker] 格式
- 标签构造与模型解耦

核心公式：
- Reg-on-Rank: rank_pct = (rank - 1) / (N - 1)，再做 zscore/GaussRank
- LambdaRank: label = floor(rank_pct * n_bins)，得到 0~(n_bins-1) 的整数

创建: 2025-12-04 | 版本: v1.0
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.special import erfinv
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
if ml_root not in sys.path:
    sys.path.insert(0, ml_root)


class RankingLabelFactory:
    """
    排序标签工厂
    
    提供三种任务模式的标签构造：
    1. regression: 原始收益（直通）
    2. regression_rank: 排序后的收益（连续值）
    3. lambdarank: 分箱等级 + group 向量
    """
    
    def __init__(self, n_bins: int = 5, rank_method: str = 'zscore'):
        """
        初始化
        
        Parameters:
        -----------
        n_bins : int
            LambdaRank 分箱数（默认5档）
        rank_method : str
            Reg-on-Rank 的标准化方法：
            - 'zscore': 横截面 z-score
            - 'gauss': GaussRank（均匀分布→正态分布）
            - 'uniform': 保持 [0,1] 均匀分布
        """
        self.n_bins = n_bins
        self.rank_method = rank_method
        print(f"🏷️ 排序标签工厂初始化")
        print(f"   分箱数: {n_bins}")
        print(f"   Rank 方法: {rank_method}")
    
    # ==================== Reg-on-Rank 标签 ====================
    
    def make_regression_rank_labels(self,
                                    forward_returns: pd.DataFrame,
                                    target_col: str = 'ret_5d',
                                    min_samples: int = 30) -> pd.Series:
        """
        构造 Reg-on-Rank 标签
        
        将每日横截面的 forward return 转为排序值（连续），用于回归模型
        
        Parameters:
        -----------
        forward_returns : pd.DataFrame
            远期收益，MultiIndex [date, ticker]
        target_col : str
            目标收益列名
        min_samples : int
            每日最少样本数
            
        Returns:
        --------
        pd.Series
            排序后的标签，MultiIndex [date, ticker]
        """
        print(f"\n📊 构造 Reg-on-Rank 标签")
        print(f"   目标列: {target_col}")
        print(f"   方法: {self.rank_method}")
        
        if target_col not in forward_returns.columns:
            raise ValueError(f"目标列 '{target_col}' 不在 forward_returns 中")
        
        returns = forward_returns[target_col].copy()
        
        # 按日期分组计算
        dates = returns.index.get_level_values('date').unique()
        rank_labels = []
        
        for date in dates:
            try:
                daily = returns.xs(date, level='date').dropna()
            except KeyError:
                continue
            
            if len(daily) < min_samples:
                continue
            
            # 计算横截面排序
            n = len(daily)
            ranks = daily.rank(method='average')
            
            # 转为百分位 [0, 1]
            rank_pct = (ranks - 1) / (n - 1)
            
            # 应用变换
            if self.rank_method == 'zscore':
                # 横截面 z-score
                transformed = (rank_pct - rank_pct.mean()) / rank_pct.std()
            elif self.rank_method == 'gauss':
                # GaussRank: 均匀分布 → 正态分布
                # 避免边界值导致无穷大
                rank_pct_clipped = rank_pct.clip(0.001, 0.999)
                transformed = pd.Series(
                    np.sqrt(2) * erfinv(2 * rank_pct_clipped - 1),
                    index=rank_pct.index
                )
            elif self.rank_method == 'uniform':
                # 保持 [0, 1] 均匀分布
                transformed = rank_pct
            else:
                raise ValueError(f"不支持的 rank_method: {self.rank_method}")
            
            # 重建 MultiIndex
            for ticker in transformed.index:
                rank_labels.append({
                    'date': date,
                    'ticker': ticker,
                    'label': transformed.loc[ticker]
                })
        
        result = pd.DataFrame(rank_labels).set_index(['date', 'ticker'])['label']
        result.name = f'{target_col}_rank'
        
        print(f"   ✅ 完成，样本数: {len(result):,}")
        print(f"   标签分布: mean={result.mean():.4f}, std={result.std():.4f}")
        
        return result
    
    # ==================== LambdaRank 标签 + Group ====================
    
    def make_lambdarank_labels_and_groups(self,
                                          forward_returns: pd.DataFrame,
                                          target_col: str = 'ret_5d',
                                          min_samples: int = 30) -> Tuple[pd.Series, List[int]]:
        """
        构造 LambdaRank 标签和 Group 向量
        
        将每日横截面的 forward return 分箱为等级（整数），同时返回 group 向量
        
        Parameters:
        -----------
        forward_returns : pd.DataFrame
            远期收益，MultiIndex [date, ticker]
        target_col : str
            目标收益列名
        min_samples : int
            每日最少样本数
            
        Returns:
        --------
        Tuple[pd.Series, List[int]]
            - labels: 分箱标签（0 ~ n_bins-1），MultiIndex [date, ticker]
            - groups: 每个日期的样本数列表
        """
        print(f"\n📊 构造 LambdaRank 标签 + Group")
        print(f"   目标列: {target_col}")
        print(f"   分箱数: {self.n_bins}")
        
        if target_col not in forward_returns.columns:
            raise ValueError(f"目标列 '{target_col}' 不在 forward_returns 中")
        
        returns = forward_returns[target_col].copy()
        
        # 按日期分组计算
        dates = sorted(returns.index.get_level_values('date').unique())
        
        bin_labels = []
        groups = []
        
        for date in dates:
            try:
                daily = returns.xs(date, level='date').dropna()
            except KeyError:
                continue
            
            if len(daily) < min_samples:
                continue
            
            n = len(daily)
            
            # 计算横截面排序
            ranks = daily.rank(method='average')
            rank_pct = (ranks - 1) / (n - 1)
            
            # 分箱：[0, 1/n_bins) → 0, [1/n_bins, 2/n_bins) → 1, ...
            # 注意：最高的要归到 n_bins-1，不能是 n_bins
            bins = (rank_pct * self.n_bins).astype(int).clip(upper=self.n_bins - 1)
            
            # 记录 group 大小
            groups.append(n)
            
            # 重建 MultiIndex
            for ticker in bins.index:
                bin_labels.append({
                    'date': date,
                    'ticker': ticker,
                    'label': bins.loc[ticker]
                })
        
        result = pd.DataFrame(bin_labels).set_index(['date', 'ticker'])['label']
        result.name = f'{target_col}_bin'
        
        # 标签分布统计
        label_dist = result.value_counts().sort_index()
        
        print(f"   ✅ 完成")
        print(f"   样本数: {len(result):,}")
        print(f"   日期数（groups）: {len(groups)}")
        print(f"   标签分布:\n{label_dist.to_string()}")
        
        return result, groups
    
    # ==================== 统一接口 ====================
    
    def create_labels(self,
                      forward_returns: pd.DataFrame,
                      task_type: str = 'regression',
                      target_col: str = 'ret_5d',
                      min_samples: int = 30) -> Dict:
        """
        统一标签构造接口
        
        Parameters:
        -----------
        forward_returns : pd.DataFrame
            远期收益，MultiIndex [date, ticker]
        task_type : str
            任务类型：'regression', 'regression_rank', 'lambdarank'
        target_col : str
            目标收益列名
        min_samples : int
            每日最少样本数
            
        Returns:
        --------
        dict
            {
                'labels': pd.Series,  # 标签
                'groups': List[int] | None,  # group 向量（仅 lambdarank）
                'task_type': str,
                'target_col': str
            }
        """
        print(f"\n{'='*60}")
        print(f"创建标签 - 任务类型: {task_type}")
        print(f"{'='*60}")
        
        result = {
            'task_type': task_type,
            'target_col': target_col,
            'groups': None
        }
        
        if task_type == 'regression':
            # 直接使用原始收益
            labels = forward_returns[target_col].copy()
            labels.name = target_col
            result['labels'] = labels
            print(f"✅ 使用原始收益作为标签")
            print(f"   样本数: {len(labels):,}")
            
        elif task_type == 'regression_rank':
            # Reg-on-Rank
            result['labels'] = self.make_regression_rank_labels(
                forward_returns, target_col, min_samples
            )
            
        elif task_type == 'lambdarank':
            # LambdaRank + Groups
            labels, groups = self.make_lambdarank_labels_and_groups(
                forward_returns, target_col, min_samples
            )
            result['labels'] = labels
            result['groups'] = groups
            
        else:
            raise ValueError(f"不支持的 task_type: {task_type}")
        
        return result
    
    # ==================== 工具函数 ====================
    
    def align_features_with_labels(self,
                                   features: pd.DataFrame,
                                   labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        对齐特征和标签（取交集）
        
        Parameters:
        -----------
        features : pd.DataFrame
            特征，MultiIndex [date, ticker]
        labels : pd.Series
            标签，MultiIndex [date, ticker]
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            对齐后的 (features, labels)
        """
        common_idx = features.index.intersection(labels.index)
        
        aligned_features = features.loc[common_idx]
        aligned_labels = labels.loc[common_idx]
        
        print(f"   📊 对齐后样本数: {len(common_idx):,}")
        
        return aligned_features, aligned_labels
    
    def compute_groups_from_index(self,
                                  data: pd.DataFrame,
                                  sort_by_date: bool = True) -> Tuple[pd.DataFrame, List[int]]:
        """
        从 MultiIndex [date, ticker] 计算 group 向量
        
        LightGBM 排序模型要求：
        1. 数据按 date 排序
        2. group[i] = 第 i 个日期的样本数
        
        Parameters:
        -----------
        data : pd.DataFrame
            MultiIndex [date, ticker] 的数据
        sort_by_date : bool
            是否按日期排序（必须为 True）
            
        Returns:
        --------
        Tuple[pd.DataFrame, List[int]]
            - 排序后的数据
            - group 向量
        """
        if sort_by_date:
            data = data.sort_index(level='date')
        
        groups = data.groupby(level='date').size().tolist()
        
        return data, groups


# ==================== 便捷函数 ====================

def create_ranking_labels(forward_returns: pd.DataFrame,
                          task_type: str = 'regression',
                          target_col: str = 'ret_5d',
                          n_bins: int = 5,
                          rank_method: str = 'zscore',
                          min_samples: int = 30) -> Dict:
    """
    便捷函数：创建排序标签
    
    Parameters:
    -----------
    forward_returns : pd.DataFrame
        远期收益，MultiIndex [date, ticker]
    task_type : str
        任务类型：'regression', 'regression_rank', 'lambdarank'
    target_col : str
        目标收益列名
    n_bins : int
        LambdaRank 分箱数
    rank_method : str
        Reg-on-Rank 方法：'zscore', 'gauss', 'uniform'
    min_samples : int
        每日最少样本数
        
    Returns:
    --------
    dict
        标签信息字典
    """
    factory = RankingLabelFactory(n_bins=n_bins, rank_method=rank_method)
    return factory.create_labels(forward_returns, task_type, target_col, min_samples)


# ==================== 测试 ====================

if __name__ == "__main__":
    print("🧪 测试排序标签构造模块")
    print("=" * 60)
    
    # 构造模拟数据
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=20, freq='D')
    tickers = [f'{i:06d}' for i in range(1, 51)]  # 50只股票
    
    # 创建 MultiIndex
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
    
    # 模拟 forward returns
    forward_returns = pd.DataFrame({
        'ret_1d': np.random.randn(len(index)) * 0.02,
        'ret_5d': np.random.randn(len(index)) * 0.05,
        'ret_10d': np.random.randn(len(index)) * 0.08
    }, index=index)
    
    print(f"模拟数据形状: {forward_returns.shape}")
    print(f"日期数: {len(dates)}, 股票数: {len(tickers)}")
    
    # 测试三种模式
    factory = RankingLabelFactory(n_bins=5, rank_method='zscore')
    
    # 1. 原始回归
    print("\n" + "=" * 60)
    print("测试 1: regression（原始收益）")
    result1 = factory.create_labels(forward_returns, 'regression', 'ret_5d')
    print(f"标签形状: {result1['labels'].shape}")
    
    # 2. Reg-on-Rank
    print("\n" + "=" * 60)
    print("测试 2: regression_rank")
    result2 = factory.create_labels(forward_returns, 'regression_rank', 'ret_5d')
    print(f"标签形状: {result2['labels'].shape}")
    
    # 3. LambdaRank
    print("\n" + "=" * 60)
    print("测试 3: lambdarank")
    result3 = factory.create_labels(forward_returns, 'lambdarank', 'ret_5d')
    print(f"标签形状: {result3['labels'].shape}")
    print(f"Groups 长度: {len(result3['groups'])}")
    print(f"Groups 样本: {result3['groups'][:5]}")
    
    print("\n✅ 所有测试通过")
