#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习基线训练主脚本

阶段12：机器学习基线（回归/排序）
- 回归目标：future_return_5d
- 模型：Ridge, RandomForestRegressor, LightGBM
- 预测测试段 → 分5桶 → 统计每桶真实收益
- 输出：reports/model_bucket_performance.csv
- 生成基于"Top桶"策略收益对比

验收：
- Top桶收益 > 全体均值
- Spread（Top - Bottom）为正
"""

import os
import sys
import yaml
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入模块
from data.data_loader import DataLoader
from models.ridge_model import RidgeModel
from models.rf_model import RandomForestModel
try:
    from models.lgbm_model import LightGBMModel
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️ LightGBM未安装，将跳过该模型")

from evaluation.metrics import calculate_metrics
from evaluation.bucketing import bucket_predictions, analyze_bucket_performance
from evaluation.reporting import generate_report
from utils.splitting import time_series_split
from utils.logger import setup_logger


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path: str = "machine learning/configs/ml_baseline.yml"):
    """
    主训练流程
    
    Parameters:
    -----------
    config_path : str
        配置文件路径
    """
    print("=" * 70)
    print("🚀 机器学习基线训练")
    print("=" * 70)
    
    # 1. 加载配置
    print("\n📋 加载配置...")
    config = load_config(config_path)
    print(f"   ✅ 配置加载完成")
    
    # 设置随机种子
    random_seed = config['runtime']['random_seed']
    np.random.seed(random_seed)
    print(f"   🎲 随机种子: {random_seed}")
    
    # 2. 加载数据
    print("\n📊 加载数据...")
    data_loader = DataLoader(config['paths']['data_root'])
    
    features, targets = data_loader.load_features_and_targets(
        symbol=config['data']['symbol'],
        target_col=config['target']['name'],
        use_scaled=config['features']['use_scaled_features']
    )
    
    print(f"   ✅ 数据加载完成")
    print(f"      特征数: {features.shape[1]}")
    print(f"      样本数: {len(features)}")
    
    # 3. 数据切分
    print("\n📅 数据切分...")
    # 合并特征和目标以便切分
    full_data = features.copy()
    full_data['target'] = targets
    
    splits = time_series_split(
        full_data,
        train_ratio=config['split']['train_ratio'],
        valid_ratio=config['split']['valid_ratio'],
        test_ratio=config['split']['test_ratio'],
        purge_days=config['split']['purge_days']
    )
    
    # 分离特征和目标
    X_train = splits['train'].drop('target', axis=1)
    y_train = splits['train']['target']
    
    X_valid = splits['valid'].drop('target', axis=1) if len(splits['valid']) > 0 else None
    y_valid = splits['valid']['target'] if len(splits['valid']) > 0 else None
    
    X_test = splits['test'].drop('target', axis=1)
    y_test = splits['test']['target']
    
    print(f"   ✅ 数据切分完成")
    
    # 4. 模型训练
    print("\n🤖 模型训练...")
    models_config = config['models']
    models = {}
    training_results = {}
    
    # Ridge
    if models_config['ridge']['enabled']:
        print("\n📌 训练Ridge模型")
        ridge_model = RidgeModel(params=models_config['ridge']['params'])
        ridge_results = ridge_model.fit(X_train, y_train, X_valid, y_valid)
        models['Ridge'] = ridge_model
        training_results['Ridge'] = ridge_results
    
    # RandomForest
    if models_config['random_forest']['enabled']:
        print("\n🌲 训练RandomForest模型")
        rf_model = RandomForestModel(params=models_config['random_forest']['params'])
        rf_results = rf_model.fit(X_train, y_train, X_valid, y_valid)
        models['RandomForest'] = rf_model
        training_results['RandomForest'] = rf_results
    
    # LightGBM
    if models_config['lightgbm']['enabled'] and HAS_LIGHTGBM:
        print("\n💡 训练LightGBM模型")
        lgbm_model = LightGBMModel(params=models_config['lightgbm']['params'])
        lgbm_results = lgbm_model.fit(X_train, y_train, X_valid, y_valid)
        models['LightGBM'] = lgbm_model
        training_results['LightGBM'] = lgbm_results
    
    print(f"\n✅ 所有模型训练完成，共{len(models)}个模型")
    
    # 5. 测试集预测与评估
    print("\n🎯 测试集预测与评估...")
    
    all_predictions = []
    model_metrics = {}
    
    for model_name, model in models.items():
        print(f"\n   📊 评估 {model_name} 模型")
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        metrics = calculate_metrics(y_test, y_pred)
        model_metrics[model_name] = metrics
        
        print(f"      MSE: {metrics['mse']:.6f}")
        print(f"      MAE: {metrics['mae']:.6f}")
        print(f"      IC: {metrics['ic']:.4f} (p={metrics['ic_pvalue']:.4f})")
        print(f"      Rank IC: {metrics['rank_ic']:.4f} (p={metrics['rank_ic_pvalue']:.4f})")
        
        # 保存预测
        pred_df = pd.DataFrame({
            'y_true': y_test,
            'y_pred': y_pred,
            'model': model_name
        }, index=y_test.index)
        
        all_predictions.append(pred_df)
    
    # 6. 分桶分析
    print("\n📊 分桶分析...")
    
    bucket_results = []
    
    for model_name, model in models.items():
        print(f"\n   🪣 {model_name} 分桶分析")
        
        # 获取该模型的预测
        model_pred = [p for p in all_predictions if p['model'].iloc[0] == model_name][0]
        
        # 分桶
        bucketed = bucket_predictions(
            model_pred,
            n_buckets=config['evaluation']['n_buckets'],
            method=config['evaluation']['bucket_method']
        )
        
        # 分析桶表现
        bucket_stats = analyze_bucket_performance(bucketed)
        bucket_stats['model'] = model_name
        
        bucket_results.append(bucket_stats)
    
    # 合并所有模型的分桶结果
    all_bucket_stats = pd.concat(bucket_results, ignore_index=True)
    
    # 7. 验收检查
    print("\n✅ 验收检查...")
    
    validation_results = {}
    
    for model_name in models.keys():
        model_buckets = all_bucket_stats[all_bucket_stats['model'] == model_name]
        
        if len(model_buckets) >= 2:
            top_bucket = model_buckets.iloc[-1]
            bottom_bucket = model_buckets.iloc[0]
            
            top_mean = top_bucket['mean_y_true']
            bottom_mean = bottom_bucket['mean_y_true']
            spread = top_mean - bottom_mean
            
            overall_mean = y_test.mean()
            
            top_vs_mean = top_mean > overall_mean
            spread_positive = spread > 0
            
            validation_results[model_name] = {
                'top_mean': float(top_mean),
                'bottom_mean': float(bottom_mean),
                'overall_mean': float(overall_mean),
                'spread': float(spread),
                'top_vs_mean': bool(top_vs_mean),
                'spread_positive': bool(spread_positive),
                'pass': bool(top_vs_mean and spread_positive)
            }
            
            print(f"\n   {model_name}:")
            print(f"      Top桶 > 全体均值: {'✅' if top_vs_mean else '❌'} ({top_mean:.6f} vs {overall_mean:.6f})")
            print(f"      Spread > 0: {'✅' if spread_positive else '❌'} ({spread:.6f})")
            print(f"      验收结果: {'✅ 通过' if (top_vs_mean and spread_positive) else '❌ 未通过'}")
    
    # 8. 保存模型
    if config['output']['save_models']:
        print("\n💾 保存模型...")
        models_dir = config['paths']['models_dir']
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in models.items():
            model_file = os.path.join(models_dir, f"{model_name.lower()}_model.pkl")
            model.save(model_file, format=config['output']['model_format'])
    
    # 9. 生成报告
    print("\n📝 生成报告...")
    
    # 合并所有预测
    all_predictions_df = pd.concat(all_predictions, ignore_index=False)
    
    # 准备报告数据
    report_data = {
        'model_metrics': model_metrics,
        'bucket_performance': all_bucket_stats,
        'predictions': all_predictions_df,
        'validation': validation_results,
        'training_results': training_results
    }
    
    # 生成报告
    generate_report(
        report_data,
        config['paths']['reports_dir'],
        config['output']['bucket_performance'],
        config['output']['predictions_file'],
        config['output']['summary_file']
    )
    
    # 10. 总结
    print("\n" + "=" * 70)
    print("🎉 训练流程完成！")
    print("=" * 70)
    
    print(f"\n📊 模型数量: {len(models)}")
    print(f"📈 测试样本: {len(y_test)}")
    
    # 显示最佳模型
    best_model = None
    best_ic = -999
    
    for model_name, metrics in model_metrics.items():
        if metrics['rank_ic'] > best_ic:
            best_ic = metrics['rank_ic']
            best_model = model_name
    
    if best_model:
        print(f"\n🏆 最佳模型: {best_model}")
        print(f"   Rank IC: {best_ic:.4f}")
        
        if best_model in validation_results:
            val = validation_results[best_model]
            print(f"   Top桶收益: {val['top_mean']:.4f}")
            print(f"   Spread: {val['spread']:.4f}")
            print(f"   验收: {'✅ 通过' if val['pass'] else '❌ 未通过'}")
    
    print(f"\n📁 报告目录: {config['paths']['reports_dir']}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='机器学习基线训练')
    parser.add_argument('--config', type=str, 
                       default='machine learning/configs/ml_baseline.yml',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    try:
        main(args.config)
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
