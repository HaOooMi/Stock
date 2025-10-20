#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载器 - 统一数据接口

功能：
1. 从ML output加载标准化特征数据
2. 加载目标变量数据
3. 统一返回MultiIndex [date, ticker]格式
4. 数据对齐与清洗
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


class DataLoader:
    """
    数据加载器类
    
    功能：
    1. 加载特征和目标数据
    2. 数据对齐与清洗
    3. 统一格式为MultiIndex
    """
    
    def __init__(self, data_root: str = "machine learning/ML output"):
        """
        初始化数据加载器
        
        Parameters:
        -----------
        data_root : str
            数据根目录
        """
        if os.path.isabs(data_root):
            self.data_root = data_root
        else:
            self.data_root = os.path.join(project_root, data_root)
        
        print(f"📁 数据加载器初始化")
        print(f"   数据根目录: {self.data_root}")
    
    def load_features_and_targets(self, 
                                  symbol: str,
                                  target_col: str = 'future_return_5d',
                                  use_scaled: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        加载特征和目标数据（从ML output目录）
        
        Parameters:
        -----------
        symbol : str
            股票代码
        target_col : str
            目标列名
        use_scaled : bool
            是否使用标准化后的特征
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            (特征数据, 目标数据)，索引为MultiIndex [date, ticker]
        """
        print(f"📊 加载数据: {symbol}")
        
        # 1. 加载特征数据
        if use_scaled:
            # 从scaled_features.csv加载
            feature_pattern = f"scaler_{symbol}_scaled_features.csv"
            feature_files = [f for f in os.listdir(self.data_root) if f.startswith(f"scaler_{symbol}")]
            
            if not feature_files:
                raise FileNotFoundError(f"未找到标准化特征文件: {feature_pattern}")
            
            feature_file = os.path.join(self.data_root, feature_files[0])
            print(f"   📈 加载标准化特征: {feature_files[0]}")
        else:
            # 从with_targets文件加载
            target_pattern = f"with_targets_{symbol}_complete_*.csv"
            target_files = [f for f in os.listdir(self.data_root) if f.startswith(f"with_targets_{symbol}")]
            
            if not target_files:
                raise FileNotFoundError(f"未找到目标文件: {target_pattern}")
            
            # 使用最新的文件
            target_files.sort(reverse=True)
            feature_file = os.path.join(self.data_root, target_files[0])
            print(f"   📈 加载特征: {target_files[0]}")
        
        features_df = pd.read_csv(feature_file, index_col=0, parse_dates=True)
        
        # 2. 加载目标数据（从with_targets文件）
        target_pattern = f"with_targets_{symbol}_complete_*.csv"
        target_files = [f for f in os.listdir(self.data_root) if f.startswith(f"with_targets_{symbol}")]
        
        if not target_files:
            raise FileNotFoundError(f"未找到目标文件: {target_pattern}")
        
        # 使用最新的文件
        target_files.sort(reverse=True)
        target_file = os.path.join(self.data_root, target_files[0])
        print(f"   🎯 加载目标: {target_files[0]}")
        
        targets_df = pd.read_csv(target_file, index_col=0, parse_dates=True)
        
        # 3. 检查目标列是否存在
        if target_col not in targets_df.columns:
            available_targets = [col for col in targets_df.columns if col.startswith('future_return_')]
            raise ValueError(f"目标列 '{target_col}' 不存在。可用目标: {available_targets}")
        
        # 4. 提取特征列（排除close和目标列）
        exclude_cols = ['close'] + [col for col in features_df.columns if col.startswith('future_return_') or col.startswith('label_')]
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # 5. 对齐索引
        common_index = features_df.index.intersection(targets_df.index)
        features_aligned = features_df.loc[common_index, feature_cols]
        targets_aligned = targets_df.loc[common_index, target_col]
        
        # 6. 转换为MultiIndex格式 [date, ticker]
        # 为单个股票创建MultiIndex
        dates = features_aligned.index
        tickers = [symbol] * len(dates)
        multi_index = pd.MultiIndex.from_arrays([dates, tickers], names=['date', 'ticker'])
        
        features_multi = pd.DataFrame(features_aligned.values, index=multi_index, columns=feature_cols)
        targets_multi = pd.Series(targets_aligned.values, index=multi_index, name=target_col)
        
        # 7. 清洗数据（删除NaN）
        valid_mask = ~(features_multi.isna().any(axis=1) | targets_multi.isna())
        features_clean = features_multi[valid_mask]
        targets_clean = targets_multi[valid_mask]
        
        print(f"   ✅ 数据加载完成:")
        print(f"      特征数量: {len(feature_cols)}")
        print(f"      有效样本: {len(features_clean)} / {len(features_aligned)}")
        print(f"      时间范围: {features_clean.index.get_level_values('date').min().date()} ~ "
              f"{features_clean.index.get_level_values('date').max().date()}")
        
        return features_clean, targets_clean
    
    def load_universe(self, 
                     symbol: str,
                     min_volume: Optional[float] = None,
                     min_price: Optional[float] = None) -> pd.DataFrame:
        """
        加载可交易标的列表（过滤低流动性）
        
        Parameters:
        -----------
        symbol : str
            股票代码
        min_volume : float, optional
            最小成交量
        min_price : float, optional
            最小价格
            
        Returns:
        --------
        pd.DataFrame
            可交易标的的MultiIndex数据
        """
        print(f"🔍 加载可交易标的列表")
        
        # 加载原始数据（包含volume和close）
        target_files = [f for f in os.listdir(self.data_root) if f.startswith(f"with_targets_{symbol}")]
        if not target_files:
            raise FileNotFoundError(f"未找到数据文件")
        
        target_files.sort(reverse=True)
        data_file = os.path.join(self.data_root, target_files[0])
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        
        # 初始化过滤mask
        valid_mask = pd.Series(True, index=df.index)
        
        # 应用过滤条件
        if min_volume is not None and 'volume' in df.columns:
            volume_mask = df['volume'] >= min_volume
            valid_mask &= volume_mask
            print(f"   📊 成交量过滤: {volume_mask.sum()} / {len(df)} 样本")
        
        if min_price is not None and 'close' in df.columns:
            price_mask = df['close'] >= min_price
            valid_mask &= price_mask
            print(f"   💰 价格过滤: {price_mask.sum()} / {len(df)} 样本")
        
        # 转换为MultiIndex格式
        dates = df[valid_mask].index
        tickers = [symbol] * len(dates)
        multi_index = pd.MultiIndex.from_arrays([dates, tickers], names=['date', 'ticker'])
        
        universe_df = pd.DataFrame({
            'tradable': True,
            'close': df.loc[valid_mask, 'close'].values if 'close' in df.columns else np.nan,
            'volume': df.loc[valid_mask, 'volume'].values if 'volume' in df.columns else np.nan
        }, index=multi_index)
        
        print(f"   ✅ 可交易标的: {len(universe_df)} 个时间点")
        
        return universe_df
    
    def get_feature_list(self, symbol: str, use_scaled: bool = True) -> List[str]:
        """
        获取特征列表
        
        Parameters:
        -----------
        symbol : str
            股票代码
        use_scaled : bool
            是否使用标准化特征
            
        Returns:
        --------
        List[str]
            特征名称列表
        """
        # 优先从final_feature_list.txt读取
        feature_list_file = os.path.join(self.data_root, "final_feature_list.txt")
        if os.path.exists(feature_list_file):
            with open(feature_list_file, 'r', encoding='utf-8') as f:
                features = [line.strip() for line in f if line.strip()]
            print(f"   📋 从文件加载特征列表: {len(features)} 个特征")
            return features
        
        # 否则从数据文件中提取
        if use_scaled:
            feature_files = [f for f in os.listdir(self.data_root) if f.startswith(f"scaler_{symbol}_scaled_features.csv")]
        else:
            feature_files = [f for f in os.listdir(self.data_root) if f.startswith(f"with_targets_{symbol}_complete_")]
        
        if not feature_files:
            raise FileNotFoundError("未找到特征文件")
        
        feature_file = os.path.join(self.data_root, feature_files[0])
        df = pd.read_csv(feature_file, index_col=0, nrows=1)
        
        exclude_cols = ['close'] + [col for col in df.columns if col.startswith('future_return_') or col.startswith('label_')]
        features = [col for col in df.columns if col not in exclude_cols]
        
        print(f"   📋 从数据文件提取特征列表: {len(features)} 个特征")
        return features


if __name__ == "__main__":
    """
    使用示例
    """
    print("📊 数据加载器测试")
    print("=" * 50)
    
    try:
        # 初始化加载器
        loader = DataLoader()
        
        # 加载特征和目标
        symbol = "000001"
        features, targets = loader.load_features_and_targets(
            symbol=symbol,
            target_col='future_return_5d',
            use_scaled=True
        )
        
        print(f"\n✅ 数据加载成功:")
        print(f"   特征形状: {features.shape}")
        print(f"   目标形状: {targets.shape}")
        print(f"   索引类型: {type(features.index)}")
        print(f"   索引级别: {features.index.names}")
        
        # 加载可交易标的
        universe = loader.load_universe(
            symbol=symbol,
            min_volume=1000000,
            min_price=1.0
        )
        
        print(f"\n✅ 可交易标的加载成功:")
        print(f"   形状: {universe.shape}")
        
        # 获取特征列表
        feature_list = loader.get_feature_list(symbol=symbol)
        print(f"\n✅ 特征列表获取成功:")
        print(f"   特征数量: {len(feature_list)}")
        print(f"   前10个特征: {feature_list[:10]}")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
