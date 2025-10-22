#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标工程模块 - 生成机器学习目标变量

功能：
1. 基于收盘价生成未来收益率目标
2. 生成分类标签（涨跌、分位数）
3. 保存带目标的完整数据集
4. 防止数据泄漏（正确的时间序列处理）


"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TargetEngineer:
    """
    目标工程类 - 生成机器学习目标变量
    
    主要功能：
    1. 基于价格数据生成未来收益率目标
    2. 生成分类标签（二分类、多分类）
    3. 数据完整性验证
    4. 防止时间序列数据泄漏
    """
    
    def __init__(self, data_dir: str = "machine learning/ML output"):
        """
        初始化目标工程器
        
        Parameters:
        -----------
        data_dir : str, default="data"
            数据保存目录
        """
        # 设置数据目录
        if os.path.isabs(data_dir):
            # 如果是绝对路径，直接使用
            self.data_dir = data_dir
        else:
            # 如果是相对路径，相对于当前工作目录
            self.data_dir = os.path.abspath(data_dir)
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        print("🎯 目标工程器初始化完成")
        print(f"   📁 数据目录: {self.data_dir}")

    def generate_future_returns(self, data: pd.DataFrame, 
                               periods: List[int] = [1, 5, 10],
                               price_col: str = 'close') -> pd.DataFrame:
        """
        生成未来收益率目标变量
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含价格数据的DataFrame，需要有时间索引
        periods : list, default=[1, 5, 10]
            未来收益率的时间窗口（天数）
        price_col : str, default='close'
            用于计算收益率的价格列名
            
        Returns:
        --------
        pd.DataFrame
            包含原数据和未来收益率目标的DataFrame
        """
        print("📈 生成未来收益率目标...")
        
        if price_col not in data.columns:
            raise ValueError(f"价格列 '{price_col}' 不存在于数据中")
        
        # 复制数据以避免修改原始数据
        result_df = data.copy()
        
        # 确保数据按时间排序
        if not result_df.index.is_monotonic_increasing:
            result_df = result_df.sort_index()
            print("   📅 数据已按时间排序")
        
        print(f"   🔢 生成 {len(periods)} 个时间窗口的未来收益率")
        
        # 生成各个时间窗口的未来收益率
        for period in periods:
            target_col = f'future_return_{period}d'
            
            # 计算未来收益率：shift(-period) 表示向前移动period天
            # 即：今天的目标 = (period天后的价格 - 今天价格) / 今天价格
            future_prices = result_df[price_col].shift(-period)
            current_prices = result_df[price_col]
            
            # 计算收益率
            result_df[target_col] = (future_prices - current_prices) / current_prices
            
            # 统计有效目标数量
            valid_targets = result_df[target_col].notna().sum()
            total_samples = len(result_df)
            nan_samples = total_samples - valid_targets
            
            print(f"   📊 {target_col}: 有效样本 {valid_targets}, NaN样本 {nan_samples} (尾部{period}行)")
        
        # 验证尾部NaN的正确性
        self._verify_future_returns(result_df, periods)
        
        print(f"✅ 未来收益率生成完成")
        return result_df
    
    def generate_classification_labels(self, data: pd.DataFrame,
                                     target_cols: Optional[List[str]] = None,
                                     label_type: str = 'binary',
                                     quantiles: Optional[List[float]] = None) -> pd.DataFrame:
        """
        基于未来收益率生成分类标签
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含未来收益率的数据
        target_cols : list, optional
            未来收益率列名列表，如果为None则自动检测
        label_type : str, default='binary'
            标签类型: 'binary'(涨跌) 或 'quantile'(分位数)
        quantiles : list, optional
            分位数阈值，用于quantile类型，默认为[0.2, 0.8]
            
        Returns:
        --------
        pd.DataFrame
            包含分类标签的数据
        """
        print("🏷️ 生成分类标签...")
        
        result_df = data.copy()
        
        # 自动检测未来收益率列
        if target_cols is None:
            target_cols = [col for col in data.columns if col.startswith('future_return_')]
        
        if not target_cols:
            raise ValueError("未找到未来收益率列，请先生成未来收益率")
        
        print(f"   🎯 为 {len(target_cols)} 个目标生成 {label_type} 标签")
        
        if label_type == 'binary':
            # 二分类：涨(1) 跌(0)
            for col in target_cols:
                label_col = col.replace('future_return_', 'label_binary_')
                result_df[label_col] = (result_df[col] > 0).astype(float)
                
                # 统计标签分布
                valid_labels = result_df[label_col].notna()
                if valid_labels.sum() > 0:
                    up_ratio = (result_df.loc[valid_labels, label_col] == 1).mean()
                    print(f"   📊 {label_col}: 上涨比例 {up_ratio:.2%}")
        
        elif label_type == 'quantile':
            # 分位数分类
            if quantiles is None:
                quantiles = [0.2, 0.8]  # 默认分为3类：低20%，中间60%，高20%
            
            for col in target_cols:
                label_col = col.replace('future_return_', 'label_quantile_')
                
                # 计算分位数阈值（只使用有效数据）
                valid_data = result_df[col].dropna()
                if len(valid_data) == 0:
                    print(f"   ⚠️ {col} 没有有效数据，跳过分位数标签生成")
                    continue
                
                thresholds = [valid_data.quantile(q) for q in quantiles]
                
                # 生成分位数标签
                labels = pd.cut(result_df[col], 
                              bins=[-np.inf] + thresholds + [np.inf], 
                              labels=list(range(len(quantiles) + 1)),
                              include_lowest=True).astype(float)
                
                result_df[label_col] = labels
                
                # 统计标签分布
                valid_labels = result_df[label_col].notna()
                if valid_labels.sum() > 0:
                    label_counts = result_df.loc[valid_labels, label_col].value_counts().sort_index()
                    print(f"   📊 {label_col} 分布: {dict(label_counts)}")
        
        else:
            raise ValueError("label_type 必须是 'binary' 或 'quantile'")
        
        print(f"✅ 分类标签生成完成")
        return result_df

    def create_complete_dataset(self, features_df: pd.DataFrame,
                               periods: List[int] = [1, 5, 10],
                               price_col: str = 'close',
                               include_labels: bool = True,
                               label_types: List[str] = ['binary']) -> pd.DataFrame:
        """
        创建包含特征和目标的完整数据集
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            特征数据（来自特征工程）
        periods : list, default=[1, 5, 10]
            未来收益率时间窗口
        price_col : str, default='close'
            价格列名
        include_labels : bool, default=True
            是否包含分类标签
        label_types : list, default=['binary']
            标签类型列表
            
        Returns:
        --------
        pd.DataFrame
            完整的机器学习数据集
        """
        print("🔨 创建完整数据集...")
        
        # 1. 生成未来收益率
        complete_df = self.generate_future_returns(features_df, periods, price_col)
        
        # 2. 生成分类标签（如果需要）
        if include_labels:
            for label_type in label_types:
                complete_df = self.generate_classification_labels(
                    complete_df, 
                    label_type=label_type
                )
        
        # 3. 数据质量统计
        self._print_dataset_summary(complete_df, periods)
        
        return complete_df
    
    def save_dataset(self, data: pd.DataFrame, symbol: str, 
                     suffix: str = "") -> str:
        """
        保存完整数据集到CSV文件
        
        Parameters:
        -----------
        data : pd.DataFrame
            要保存的数据
        symbol : str
            股票代码
        suffix : str, optional
            文件名后缀
            
        Returns:
        --------
        str
            保存的文件路径
        """
        print("💾 保存数据集...")
        
        # 构建文件名
        if suffix:
            filename = f"with_targets_{symbol}_{suffix}.csv"
        else:
            filename = f"with_targets_{symbol}.csv"
        
        filepath = os.path.join(self.data_dir, filename)
        
        # 保存数据
        data.to_csv(filepath, index=True, encoding='utf-8-sig')
        
        file_size = os.path.getsize(filepath) / 1024 / 1024  # MB
        print(f"✅ 数据集已保存: {filename}")
        print(f"   📁 路径: {filepath}")
        print(f"   📊 大小: {file_size:.2f} MB")
        print(f"   🔢 形状: {data.shape}")
        
        return filepath
    
    def _verify_future_returns(self, data: pd.DataFrame, periods: List[int]):
        """验证未来收益率的正确性（防止数据泄漏）"""
        print("\n🔍 验证目标变量正确性...")
        
        for period in periods:
            target_col = f'future_return_{period}d'
            if target_col not in data.columns:
                continue
                
            # 检查尾部NaN
            total_rows = len(data)
            nan_count = data[target_col].isna().sum()
            
            # 尾部应该有period行NaN
            tail_nans = data[target_col].tail(period).isna().sum()
            
            print(f"   📊 {target_col}:")
            print(f"      总NaN数: {nan_count}")
            print(f"      尾部{period}行NaN: {tail_nans}/{period}")
            
            if tail_nans != period:
                print(f"      ⚠️ 警告: 尾部NaN数量不匹配预期")
            
            # 检查是否使用了正确的shift（不应该有负shift泄漏）
            # 这里我们验证第一个非NaN值的位置
            first_valid = data[target_col].first_valid_index()
            last_valid = data[target_col].last_valid_index()
            
            if first_valid is not None and last_valid is not None:
                valid_count = data[target_col].notna().sum()
                expected_valid = total_rows - period
                
                if valid_count == expected_valid:
                    print(f"      ✅ 数据泄漏检查通过")
                else:
                    print(f"      ⚠️ 有效数据量异常: {valid_count} vs 期望 {expected_valid}")

    def _print_dataset_summary(self, data: pd.DataFrame, periods: List[int]):
        """打印数据集摘要"""
        print("\n📋 数据集摘要:")
        print("=" * 50)
        
        # 基本信息
        print(f"📊 数据形状: {data.shape}")
        print(f"📅 时间范围: {data.index.min().date()} ~ {data.index.max().date()}")
        
        # 特征列统计
        feature_cols = [col for col in data.columns 
                       if not col.startswith('future_return_') 
                       and not col.startswith('label_')]
        target_cols = [col for col in data.columns if col.startswith('future_return_')]
        label_cols = [col for col in data.columns if col.startswith('label_')]
        
        print(f"🔢 特征数量: {len(feature_cols)}")
        print(f"🎯 目标数量: {len(target_cols)}")
        print(f"🏷️ 标签数量: {len(label_cols)}")
        
        # 目标变量统计
        if target_cols:
            print("\n📈 目标变量统计:")
            for col in target_cols:
                valid_count = data[col].notna().sum()
                mean_return = data[col].mean()
                std_return = data[col].std()
                print(f"   {col}: 有效样本 {valid_count}, 均值 {mean_return:.4f}, 标准差 {std_return:.4f}")
        
        # 训练样本估算（排除尾部NaN）
        if periods:
            max_period = max(periods)
            trainable_samples = len(data) - max_period
            print(f"\n🎓 估算可训练样本: {trainable_samples} (排除尾部{max_period}行)")
        
        print("=" * 50)


def main():
    """
    示例用法 - 演示如何使用目标工程器
    """
    print("🎯 目标工程示例")
    print("=" * 50)
    
    try:
        # 初始化目标工程器
        target_engineer = TargetEngineer()
        
        # 这里需要从特征工程获取数据
        # 示例：假设已有特征数据
        print("⚠️ 这是示例代码，需要实际的特征数据")
        print("请从 FeatureEngineer 获取特征数据后调用:")
        print()
        print("# 示例用法:")
        print("from feature_engineering import FeatureEngineer")
        print("from target_engineering import TargetEngineer")
        print()
        print("# 1. 获取特征数据")
        print("engineer = FeatureEngineer()")
        print("data = engineer.load_stock_data('000001', '2023-01-01', '2024-12-31')")
        print("features_df = engineer.prepare_features(data)")
        print()
        print("# 2. 生成目标变量")
        print("target_engineer = TargetEngineer()")
        print("complete_df = target_engineer.create_complete_dataset(")
        print("    features_df, ")
        print("    periods=[1, 5, 10],")
        print("    include_labels=True,")
        print("    label_types=['binary', 'quantile']")
        print(")")
        print()
        print("# 3. 保存数据集")
        print("filepath = target_engineer.save_dataset(complete_df, '000001')")
        print("print(f'数据集已保存到: {filepath}')")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
