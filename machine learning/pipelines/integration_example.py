#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成示例：在训练流程中使用横截面评估

功能：
1. 展示如何在 train_models.py 中集成横截面评估
2. 在特征工程后评估特征质量
3. 在模型训练后评估预测质量
"""

import os
import sys
import pandas as pd
import numpy as np

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
if ml_root not in sys.path:
    sys.path.insert(0, ml_root)

# 导入模块
from data.data_loader import DataLoader
from evaluation import CrossSectionAdapter


def example_1_evaluate_single_feature():
    """
    示例1：评估单个特征
    
    适用场景：
    - 检验某个新特征是否有预测能力
    - 调试特征工程
    """
    print("=" * 60)
    print("示例1：评估单个特征")
    print("=" * 60)
    
    # 1. 初始化 DataLoader
    data_loader = DataLoader(
        data_root="ML output/datasets/baseline_v1",
        enable_snapshot=False,
        enable_filtering=False,
        enable_pit_alignment=False,
        enable_influxdb=False
    )
    
    # 2. 加载数据
    symbol = "000001"
    features, targets = data_loader.load_features_and_targets(
        symbol=symbol,
        target_col='future_return_5d',
        use_scaled=True  # 使用标准化后的特征
    )
    
    print(f"\n✅ 数据加载完成:")
    print(f"   特征数: {features.shape[1]}")
    print(f"   样本数: {len(features)}")
    
    # 3. 创建适配器
    adapter = CrossSectionAdapter(
        data_loader=data_loader,
        market_data_loader=None,  # 可选
        enable_neutralization=False  # 单股票不需要中性化
    )
    
    # 4. 评估单个特征（选择第一个特征作为示例）
    feature_to_test = features.columns[0]
    
    dates = features.index.get_level_values('date')
    start_date = dates.min().strftime('%Y-%m-%d')
    end_date = dates.max().strftime('%Y-%m-%d')
    
    results = adapter.evaluate_feature(
        features=features,
        targets=targets,
        feature_col=feature_to_test,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        forward_periods=[5],
        quantiles=5,
        output_dir="ML output/reports/baseline_v1/factors"
    )
    
    # 5. 解读结果
    ic_summary = results['ic_summary_5']
    
    print(f"\n📊 评估结果:")
    print(f"   特征名称: {feature_to_test}")
    print(f"   IC均值: {ic_summary['ic_mean']:.4f}")
    print(f"   ICIR: {ic_summary['ic_ir']:.2f}")
    print(f"   IC胜率: {ic_summary['ic_win_rate']:.2%}")
    print(f"   p-value: {ic_summary['p_value']:.4f}")
    
    if ic_summary['ic_ir'] > 1.0:
        print(f"   ✅ 优秀特征")
    elif ic_summary['ic_ir'] > 0.5:
        print(f"   ⚠️  合格特征")
    else:
        print(f"   ❌ 弱特征")


def example_2_batch_evaluate_features():
    """
    示例2：批量评估所有特征
    
    适用场景：
    - 特征筛选（保留高质量特征）
    - 特征工程后的验证
    """
    print("\n" + "=" * 60)
    print("示例2：批量评估所有特征")
    print("=" * 60)
    
    # 1. 初始化
    data_loader = DataLoader(
        data_root="ML output/datasets/baseline_v1",
        enable_snapshot=False,
        enable_filtering=False,
        enable_pit_alignment=False,
        enable_influxdb=False
    )
    
    adapter = CrossSectionAdapter(
        data_loader=data_loader,
        market_data_loader=None,
        enable_neutralization=False
    )
    
    # 2. 加载数据
    symbol = "000001"
    features, targets = data_loader.load_features_and_targets(
        symbol=symbol,
        target_col='future_return_5d',
        use_scaled=True
    )
    
    # 3. 批量评估（示例：仅评估前10个特征以节省时间）
    dates = features.index.get_level_values('date')
    start_date = dates.min().strftime('%Y-%m-%d')
    end_date = dates.max().strftime('%Y-%m-%d')
    
    summary_df = adapter.evaluate_all_features(
        features=features,
        targets=targets,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        output_dir="ML output/reports/baseline_v1/factors",
        top_k=10  # 仅评估前10个特征
    )
    
    # 4. 特征筛选
    qualified_features = summary_df[summary_df['qualified']]['feature'].tolist()
    
    print(f"\n📊 特征筛选结果:")
    print(f"   总特征数: {len(summary_df)}")
    print(f"   合格特征数: {len(qualified_features)}")
    print(f"   合格率: {len(qualified_features) / len(summary_df):.2%}")
    
    if qualified_features:
        print(f"\n   ✅ 合格特征列表:")
        for feat in qualified_features:
            ic = summary_df[summary_df['feature'] == feat]['ic_mean'].values[0]
            icir = summary_df[summary_df['feature'] == feat]['icir'].values[0]
            print(f"      - {feat}: IC={ic:.4f}, ICIR={icir:.2f}")


def example_3_integrate_with_training():
    """
    示例3：集成到训练流程（伪代码）
    
    展示如何在 train_models.py 中集成
    """
    print("\n" + "=" * 60)
    print("示例3：集成到训练流程（伪代码）")
    print("=" * 60)
    
    code_example = '''
# 在 train_models.py 中集成横截面评估

def train_with_feature_evaluation(config):
    """训练流程 + 特征评估"""
    
    # 1. 加载数据
    data_loader = DataLoader(...)
    features, targets = data_loader.load_features_and_targets(...)
    
    # ===== 新增：特征评估阶段 =====
    from evaluation import CrossSectionAdapter
    
    adapter = CrossSectionAdapter(
        data_loader=data_loader,
        market_data_loader=None,
        enable_neutralization=False
    )
    
    # 批量评估特征
    summary_df = adapter.evaluate_all_features(
        features=features,
        targets=targets,
        symbol=config['data']['symbol'],
        start_date=...,
        end_date=...,
        output_dir="ML output/reports/baseline_v1/factors"
    )
    
    # 筛选合格特征
    qualified_features = summary_df[summary_df['qualified']]['feature'].tolist()
    
    # 仅使用合格特征训练
    features_filtered = features[qualified_features]
    
    print(f"特征筛选: {len(features.columns)} -> {len(qualified_features)}")
    # ===== 特征评估阶段结束 =====
    
    # 2. 继续后续训练流程...
    X_train, X_test, y_train, y_test = time_series_split(...)
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # 3. 模型评估...
    '''
    
    print(code_example)


def example_4_quick_api():
    """
    示例4：使用快捷API
    
    适用场景：
    - 快速验证某个特征
    - Jupyter Notebook 交互式分析
    """
    print("\n" + "=" * 60)
    print("示例4：使用快捷API")
    print("=" * 60)
    
    from evaluation import quick_evaluate
    
    # 一行代码评估特征
    results = quick_evaluate(
        symbol="000001",
        feature_col="volume",  # 替换为实际特征名
        data_root="ML output/datasets/baseline_v1",
        target_col='future_return_5d',
        use_scaled=True,
        output_dir="ML output/reports/baseline_v1/factors"
    )
    
    print(f"\n✅ 快速评估完成:")
    print(f"   IC均值: {results['ic_summary_5']['ic_mean']:.4f}")
    print(f"   ICIR: {results['ic_summary_5']['ic_ir']:.2f}")


def main():
    """
    运行所有示例
    """
    print("\n" + "🚀" * 30)
    print("横截面评估集成示例")
    print("🚀" * 30 + "\n")
    
    try:
        # 示例1：评估单个特征
        example_1_evaluate_single_feature()
        
        # 示例2：批量评估（可选，比较耗时）
        # example_2_batch_evaluate_features()
        
        # 示例3：集成伪代码
        example_3_integrate_with_training()
        
        # 示例4：快捷API（可选）
        # example_4_quick_api()
        
        print("\n" + "=" * 60)
        print("✅ 所有示例运行完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
