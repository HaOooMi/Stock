#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的特征+目标生成流程

功能：
1. 加载原始数据
2. 生成技术特征
3. 特征选择和标准化
4. 生成目标变量
5. 保存完整数据集
"""

import os
import sys
import yaml
import argparse
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
if ml_root not in sys.path:
    sys.path.insert(0, ml_root)

# 导入特征和目标工程模块
from features.feature_engineering import FeatureEngineer
from targets.target_engineering import TargetEngineer


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    # 如果是相对路径，转换为基于ml_root的绝对路径
    if not os.path.isabs(config_path):
        config_path = os.path.join(ml_root, config_path.replace("machine learning/", ""))
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path: str = "configs/ml_baseline.yml"):
    """
    完整的数据准备流程
    
    Parameters:
    -----------
    config_path : str
        配置文件路径
    """
    print("=" * 70)
    print("🔨 数据准备流程")
    print("=" * 70)
    
    # 1. 加载配置
    print("\n📋 加载配置...")
    config = load_config(config_path)
    
    # 显示项目信息
    project_info = config.get('project', {})
    if project_info:
        print(f"   📦 项目: {project_info.get('name', 'N/A')}")
        print(f"   📝 描述: {project_info.get('description', 'N/A')}")
    
    # 创建输出目录（转换为绝对路径）
    datasets_dir = os.path.join(ml_root, config['paths'].get('datasets_dir', 'ML output/datasets/baseline_v1'))
    scalers_dir = os.path.join(ml_root, config['paths'].get('scalers_dir', 'ML output/scalers/baseline_v1'))
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(scalers_dir, exist_ok=True)
    
    symbol = config['data']['symbol']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    print(f"   股票代码: {symbol}")
    print(f"   时间范围: {start_date} ~ {end_date}")
    
    # 2. 特征工程
    print("\n🔧 特征工程...")
    feature_engineer = FeatureEngineer(use_talib=True, use_tsfresh=False)
    
    # 加载原始数据
    raw_data = feature_engineer.load_stock_data(symbol, start_date, end_date)
    
    # 生成特征
    features_df = feature_engineer.prepare_features(
        raw_data,
        use_auto_features=False
    )
    
    # 特征选择
    selection_results = feature_engineer.select_features(
        features_df,
        final_k=20,
        variance_threshold=0.01,
        correlation_threshold=0.9,
        train_ratio=0.8
    )
    
    selected_features = selection_results['final_features_df']
    
    # 特征标准化
    scaler_path = os.path.join(scalers_dir, f"scaler_{symbol}.pkl")
    scale_results = feature_engineer.scale_features(
        selected_features,
        scaler_type='robust',
        train_ratio=0.8,
        save_path=scaler_path
    )
    
    scaled_features = scale_results['scaled_df']
    
    print(f"   ✅ 特征工程完成: {len(selection_results['final_features'])} 个特征")
    print(f"   💾 标准化器保存到: {scaler_path}")
    
    # 3. 目标工程
    print("\n🎯 目标工程...")
    target_engineer = TargetEngineer(data_dir=datasets_dir)
    
    # 生成目标变量
    complete_df = target_engineer.create_complete_dataset(
        features_df=scaled_features,
        periods=[1, 5, 10],
        price_col='close',
        include_labels=True,
        label_types=['binary']
    )
    
    # 保存数据集
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = target_engineer.save_dataset(
        complete_df,
        symbol=symbol,
        suffix=f"complete_{timestamp}"
    )
    
    print(f"   ✅ 目标工程完成")
    
    # 4. 总结
    print("\n" + "=" * 70)
    print("✅ 数据准备完成！")
    print("=" * 70)
    print(f"\n📊 输出文件:")
    print(f"   特征标准化: {scale_results['scaler_path']}")
    print(f"   完整数据集: {filepath}")
    print(f"\n📈 数据统计:")
    print(f"   特征数量: {len(selection_results['final_features'])}")
    print(f"   样本数量: {len(complete_df)}")
    print(f"   目标变量: future_return_1d, future_return_5d, future_return_10d")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='数据准备流程')
    parser.add_argument('--config', type=str, 
                       default='configs/ml_baseline.yml',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    try:
        main(args.config)
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
