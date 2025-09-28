#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整数据管道 - 特征工程 + 目标工程集成

功能：
1. 从InfluxDB加载股票数据
2. 生成技术特征
3. 特征选择和标准化
4. 生成目标变量和标签
5. 保存完整数据集

"""

import os
import sys
from datetime import datetime

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from feature_engineering import FeatureEngineer
from target_engineering import TargetEngineer


def create_complete_dataset(symbol: str, start_date: str, end_date: str,
                           use_auto_features: bool = False,
                           final_k_features: int = 20,
                           target_periods: list = [1, 5, 10],
                           include_scaling: bool = True):
    """
    创建完整的机器学习数据集
    
    Parameters:
    -----------
    symbol : str
        股票代码
    start_date : str  
        开始日期
    end_date : str
        结束日期
    use_auto_features : bool, default=False
        是否使用自动特征生成
    final_k_features : int, default=20
        最终保留的特征数量
    target_periods : list, default=[1, 5, 10]
        目标变量的时间窗口
    include_scaling : bool, default=True
        是否包含特征标准化
    
    Returns:
    --------
    tuple: (完整数据集DataFrame, 特征工程结果, 目标工程器, 保存路径)
    """
    
    # 开始数据管道
    print("=" * 60)
    
    # ===== 阶段1: 特征工程 =====
    print("📊 阶段1: 特征工程")
    print("-" * 30)
    
    # 初始化特征工程器
    feature_engineer = FeatureEngineer(use_talib=True, use_tsfresh=use_auto_features)
    
    # 加载数据
    print(f"📈 加载股票数据: {symbol} ({start_date} ~ {end_date})")
    raw_data = feature_engineer.load_stock_data(symbol, start_date, end_date)
    
    if len(raw_data) < 100:
        raise ValueError(f"数据量太少({len(raw_data)}行)，建议至少100行数据")
    
    # 生成特征
    print("🏭 生成技术特征...")
    features_df = feature_engineer.prepare_features(
        raw_data,
        use_auto_features=use_auto_features,
        window_size=20,
        max_auto_features=30
    )
    
    # 特征选择
    # 执行特征选择
    selection_results = feature_engineer.select_features(
        features_df,
        final_k=final_k_features,
        variance_threshold=0.01,
        correlation_threshold=0.9,
        train_ratio=0.8
    )
    
    final_features_df = selection_results['final_features_df']
    
    # 特征标准化（可选）
    if include_scaling:
        # 执行特征标准化
        scale_results = feature_engineer.scale_features(
            final_features_df,
            scaler_type='robust',
            train_ratio=0.8,
            save_path=f'machine learning/ML output/scaler_{symbol}.pkl'
        )
        scaled_features_df = scale_results['scaled_df']
        print(f"   ✅ 缩放器已保存: {scale_results['scaler_path']}")
        if scale_results.get('csv_path'):
            print(f"   📊 标准化特征已保存: {scale_results['csv_path']}")
    else:
        scaled_features_df = final_features_df
    
    # 特征分析
    print("📊 分析特征质量...")
    analysis_results = feature_engineer.analyze_features(scaled_features_df)
    
    # ===== 阶段2: 目标工程 =====
    # 阶段2: 目标工程  
    print("-" * 30)
    
    # 初始化目标工程器
    target_engineer = TargetEngineer()
    
    # 创建完整数据集（特征 + 目标）
    print("🔨 创建完整数据集...")
    complete_dataset = target_engineer.create_complete_dataset(
        scaled_features_df,
        periods=target_periods,
        price_col='close',
        include_labels=True,
        label_types=['binary', 'quantile']
    )
    
    # ===== 阶段3: 保存数据 =====
    print("\n💾 阶段3: 保存数据")
    print("-" * 30)
    
    # 添加时间戳后缀
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"features{final_k_features}_{timestamp}"
    
    # 保存完整数据集
    save_path = target_engineer.save_dataset(complete_dataset, symbol, suffix)
    
    # ===== 最终总结 =====
    print("\n" + "=" * 60)
    print("Data pipeline completed")
    print(f"   📊 原始数据: {len(raw_data)} 行")
    print(f"   🏭 生成特征: {len(features_df.columns)-1} 个")
    print(f"   Features selected: {len(selection_results['final_features'])}")
    print(f"   Scaling: {'Yes' if include_scaling else 'No'}")
    print(f"   Target window: {target_periods} days")
    print(f"   💾 保存路径: {save_path}")
    
    # 数据可用性检查
    max_period = max(target_periods)
    total_samples = len(complete_dataset)
    trainable_samples = total_samples - max_period
    print(f"\n📊 数据可用性:")
    print(f"   🔢 总样本数: {total_samples}")
    print(f"   🎓 可训练样本: {trainable_samples} (排除尾部{max_period}行NaN)")
    print(f"   ⚠️ 注意: 尾部{max_period}行目标为NaN，不参与训练")
    
    return complete_dataset, selection_results, target_engineer, save_path


def main():
    """
    主函数 - 执行完整的数据管道
    """
    print("Stock ML Data Pipeline")
    print("=" * 60)
    
    try:
        # 配置参数
        config = {
            'symbol': '000001',  # 平安银行
            'start_date': '2023-01-01',
            'end_date': '2024-12-31', 
            'use_auto_features': False,  # 是否使用自动特征生成
            'final_k_features': 15,      # 最终特征数量
            'target_periods': [1, 5, 10], # 目标时间窗口
            'include_scaling': True       # 是否标准化
        }
        
        print("Configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        print()
        
        # 执行完整管道
        dataset, features_info, target_eng, save_path = create_complete_dataset(**config)
        
        # 显示最终结果预览
        print("\n📄 最终数据集预览:")
        print("-" * 40)
        
        # 显示列信息
        feature_cols = [col for col in dataset.columns 
                       if not col.startswith(('future_return_', 'label_'))]
        target_cols = [col for col in dataset.columns if col.startswith('future_return_')]
        label_cols = [col for col in dataset.columns if col.startswith('label_')]
        
        print(f"特征列 ({len(feature_cols)}个): {feature_cols[:5]}...")
        print(f"目标列 ({len(target_cols)}个): {target_cols}")
        print(f"标签列 ({len(label_cols)}个): {label_cols}")
        
        # 显示数据样本
        print("\n前5行数据样本:")
        print(dataset.head()[['close'] + target_cols[:2]].round(4))
        
        print("\n尾部5行数据样本 (验证NaN):")
        print(dataset.tail()[['close'] + target_cols[:2]].round(4))
        
        print(f"\n✅ 数据管道执行成功！")
        print(f"💾 完整数据集已保存到: {save_path}")
        print(f"🎯 可用于机器学习模型训练和验证")
        
        return dataset, save_path
        
    except Exception as e:
        print(f"❌ 数据管道执行失败: {str(e)}")
        raise


if __name__ == "__main__":
    # 执行完整数据管道
    main()