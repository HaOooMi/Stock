#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的特征+目标生成流程（统一支持单标的和多标的）

功能：
1. 智能检测单标的/多标的模式（根据配置文件中的 symbol 类型）
2. 加载原始数据（单标的或批量）
3. 生成技术特征
4. 特征选择和标准化
5. 生成目标变量
6. 保存完整数据集

使用方式：
- 单标的：配置文件中 symbol: "000001"
- 多标的：配置文件中 symbol: ["000001", "600000", "000858"]
  或命令行：python prepare_data.py --symbols 000001 600000 000858
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


def main(config_path: str = "configs/ml_baseline.yml", symbols: list = None):
    """
    完整的数据准备流程（统一支持单标的和多标的）
    
    Parameters:
    -----------
    config_path : str
        配置文件路径（默认使用 ml_baseline.yml）
    symbols : list, optional
        股票代码列表（覆盖配置文件中的设置）
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
    datasets_dir_cfg = config['paths'].get('datasets_dir', 'ML output/datasets/baseline_v1')
    scalers_dir_cfg = config['paths'].get('scalers_dir', 'ML output/scalers/baseline_v1')

    datasets_dir = datasets_dir_cfg if os.path.isabs(datasets_dir_cfg) else os.path.join(ml_root, datasets_dir_cfg)
    scalers_dir = scalers_dir_cfg if os.path.isabs(scalers_dir_cfg) else os.path.join(ml_root, scalers_dir_cfg)
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(scalers_dir, exist_ok=True)
    
    # 2. 智能检测单标的/多标的模式
    if symbols is None:
        config_symbol = config['data']['symbol']
        # 智能检测：如果是列表则用列表，如果是字符串则转为单元素列表
        if isinstance(config_symbol, list):
            symbols = config_symbol
            is_multi = True
        else:
            symbols = [config_symbol]
            is_multi = False
    else:
        is_multi = len(symbols) > 1
    
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    mode_name = "多标的" if is_multi else "单标的"
    print(f"   模式: {mode_name}")
    print(f"   股票代码: {symbols if is_multi else symbols[0]}")
    print(f"   股票数量: {len(symbols)}")
    print(f"   时间范围: {start_date} ~ {end_date}")
    
    # 3. 特征工程
    print(f"\n🔧 特征工程（{mode_name}模式）...")
    feature_engineer = FeatureEngineer(use_talib=True, use_tsfresh=False)
    
    if is_multi:
        # 多标的模式：批量加载和生成特征
        features_df = feature_engineer.prepare_features_batch(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            use_auto_features=False,
            keep_base_columns=True
        )
        # 统计信息
        feature_cols = [c for c in features_df.columns if c not in ['close', 'volume', 'amount', 'pct_change', 'turnover']]
        print(f"   ✅ 批量特征生成完成")
        print(f"      总样本: {len(features_df):,}")
        print(f"      特征数: {len(feature_cols)}")
        print(f"      股票数: {features_df.index.get_level_values('ticker').nunique()}")
    else:
        # 单标的模式：传统流程
        raw_data = feature_engineer.load_stock_data(symbols[0], start_date, end_date)
        features_df = feature_engineer.prepare_features(
            raw_data,
            use_auto_features=False,
            keep_base_columns=True
        )
        print(f"   ✅ 特征生成完成: {features_df.shape[1]-1} 个特征")
    
    # 4. 特征选择（可选，配置文件可控制是否跳过）
    skip_selection = config.get('features', {}).get('skip_selection', False)  # 默认不跳过
    
    if skip_selection:
        print(f"\n⏭️  跳过特征选择（配置文件设置）")
        selected_features = features_df
        final_feature_count = len([c for c in features_df.columns if c not in ['close', 'volume', 'amount', 'pct_change', 'turnover']])
    else:
        print(f"\n🎯 特征选择...")
        
        # 从配置文件读取特征选择参数
        final_k = config.get('features', {}).get('final_k', 20)
        variance_threshold = config.get('features', {}).get('variance_threshold', 0.01)
        correlation_threshold = config.get('features', {}).get('correlation_threshold', 0.9)
        
        selection_results = feature_engineer.select_features(
            features_df,
            final_k=final_k,
            variance_threshold=variance_threshold,
            correlation_threshold=correlation_threshold,
            train_ratio=0.8
        )
        selected_features = selection_results['final_features_df']
        final_feature_count = len(selection_results['final_features'])
        print(f"   ✅ 特征选择完成: {final_feature_count} 个特征")
    
    # 5. 特征标准化
    print(f"\n📏 特征标准化...")
    scaler_suffix = "multi" if is_multi else symbols[0]
    scaler_path = os.path.join(scalers_dir, f"scaler_{scaler_suffix}.pkl")
    
    scale_results = feature_engineer.scale_features(
        selected_features,
        scaler_type='robust',
        train_ratio=0.8,
        save_path=scaler_path
    )
    
    scaled_features = scale_results['scaled_df']
    print(f"   ✅ 特征标准化完成")
    print(f"   💾 标准化器: {scaler_path}")
    
    # 6. 目标工程
    print(f"\n🎯 目标工程（{mode_name}模式）...")
    target_engineer = TargetEngineer(data_dir=datasets_dir)
    
    # 生成目标变量
    complete_df = target_engineer.create_complete_dataset(
        features_df=scaled_features,
        periods=[1, 5, 10],
        price_col='close',
        include_labels=True,
        label_types=['binary']
    )
    
    # 7. 保存数据集
    print(f"\n💾 保存数据集...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if is_multi:
        # 多标的：保存为 complete_multi_timestamp.csv
        filepath = target_engineer.save_dataset(
            complete_df,
            symbol="multi",
            suffix=f"complete_{timestamp}"
        )
    else:
        # 单标的：保存为 complete_SYMBOL_timestamp.csv
        filepath = target_engineer.save_dataset(
            complete_df,
            symbol=symbols[0],
            suffix=f"complete_{timestamp}"
        )
    
    print(f"   ✅ 数据集已保存: {filepath}")
    
    # 8. 总结
    print("\n" + "=" * 70)
    print(f"✅ 数据准备完成！（{mode_name}模式）")
    print("=" * 70)
    print(f"\n📊 输出文件:")
    print(f"   特征标准化器: {scale_results['scaler_path']}")
    print(f"   完整数据集: {filepath}")
    print(f"\n📈 数据统计:")
    print(f"   模式: {mode_name}")
    print(f"   股票数量: {len(symbols)}")
    print(f"   特征数量: {final_feature_count}")
    print(f"   样本数量: {len(complete_df):,}")
    if is_multi:
        print(f"   索引格式: MultiIndex [date, ticker]")
        print(f"   唯一股票数: {complete_df.index.get_level_values('ticker').nunique()}")
    else:
        print(f"   索引格式: DatetimeIndex")
    print(f"   目标变量: future_return_1d, future_return_5d, future_return_10d")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='数据准备流程（统一支持单标的和多标的）')
    parser.add_argument('--config', type=str, 
                       default='configs/ml_baseline.yml',
                       help='配置文件路径')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='股票代码列表（覆盖配置文件），例如：--symbols 000001 600000 000858')
    
    args = parser.parse_args()
    
    try:
        main(args.config, symbols=args.symbols)
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
