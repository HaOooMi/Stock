#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
排序模型训练管道 - 三条线对比

功能：
1. Baseline A：回归原始收益（LGBMRegressor）
2. Baseline B：Reg-on-Rank（LGBMRegressor + 排序标签）
3. Sorting：LambdaRank（LGBMRanker）

统一使用 CrossSectionAnalyzer 评估，对比三条线的：
- Rank IC / ICIR
- Top-Mean / Top-Bottom Spread
- 稳定性 / 漂移

使用方法：
    python run_ranking_pipeline.py
    python run_ranking_pipeline.py --task_type lambdarank
    python run_ranking_pipeline.py --compare_all  # 运行三条线对比

输出：
    /ML output/reports/baseline_v1/ranking/
    ├── model_comparison.json
    ├── regression_results.json
    ├── regression_rank_results.json
    ├── lambdarank_results.json
    └── comparison_tearsheet.html

创建: 2025-12-04 | 版本: v1.0
"""

import os
import sys
import yaml
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
if ml_root not in sys.path:
    sys.path.insert(0, ml_root)

# 导入模块
from data.data_loader import DataLoader
from data.time_series_cv import TimeSeriesCV
from targets.ranking_labels import RankingLabelFactory, create_ranking_labels
from models.lgbm_model import LightGBMModel
from models.lgbm_ranker import LightGBMRanker, prepare_ranking_data
from evaluation.cross_section_analyzer import CrossSectionAnalyzer
from evaluation.cross_section_metrics import calculate_forward_returns


def load_config(config_path: str = "configs/ml_baseline.yml") -> dict:
    """加载配置文件"""
    if not os.path.isabs(config_path):
        config_path = os.path.join(ml_root, config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def prepare_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    准备数据：特征、远期收益、价格
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (features, forward_returns, prices)
    """
    print("\n" + "=" * 70)
    print("准备数据")
    print("=" * 70)
    
    # 获取配置
    data_config = config['data']
    target_config = config['target']
    symbols = data_config['symbol']
    if isinstance(symbols, str):
        symbols = [symbols]
    
    start_date = data_config['start_date']
    end_date = data_config['end_date']
    forward_periods = target_config['forward_periods']
    target_col = f"future_return_{forward_periods}d"
    
    # 初始化数据加载器
    # 注意：influxdb_config 需要移除 'enabled' 字段
    influxdb_config = data_config.get('influxdb', {}).copy()
    influxdb_config.pop('enabled', None)  # 移除 enabled 字段
    
    loader = DataLoader(
        data_root=config['paths']['data_root'],
        enable_snapshot=data_config['snapshot']['enabled'],
        enable_filtering=True,
        enable_influxdb=data_config['influxdb']['enabled'],
        influxdb_config=influxdb_config,
        filter_config=data_config['universe']
    )
    
    # 加载多个股票的数据
    all_features = []
    all_targets = []
    
    for symbol in symbols:
        try:
            features, targets = loader.load_features_and_targets(
                symbol=symbol,
                target_col=target_col,
                use_scaled=config['features']['use_scaled_features']
            )
            all_features.append(features)
            all_targets.append(targets)
            print(f"   ✅ {symbol}: {len(features)} 样本")
        except Exception as e:
            print(f"   ⚠️ {symbol} 加载失败: {e}")
            continue
    
    if not all_features:
        raise ValueError("没有成功加载任何股票数据")
    
    # 合并所有股票
    features = pd.concat(all_features, axis=0)
    targets = pd.concat(all_targets, axis=0)
    
    print(f"✅ 特征加载完成: {features.shape}")
    print(f"✅ 目标加载完成: {len(targets)}")
    
    # 构造 forward_returns DataFrame（评估需要）
    forward_returns = targets.to_frame(f'ret_{forward_periods}d')
    
    # prices 暂时设为 None（如果需要可以从 InfluxDB 加载）
    prices = None
    
    print(f"✅ 样本总数: {len(features):,}")
    
    return features, forward_returns, prices


def run_single_task(task_type: str,
                    config: dict,
                    features: pd.DataFrame,
                    forward_returns: pd.DataFrame,
                    train_idx: pd.Index,
                    valid_idx: pd.Index,
                    test_idx: pd.Index,
                    output_dir: str) -> Dict:
    """
    运行单个任务类型
    
    Parameters:
    -----------
    task_type : str
        任务类型：'regression', 'regression_rank', 'lambdarank'
    config : dict
        配置字典
    features : pd.DataFrame
        特征数据
    forward_returns : pd.DataFrame
        远期收益
    train_idx, valid_idx, test_idx : pd.Index
        切分索引
    output_dir : str
        输出目录
        
    Returns:
    --------
    dict
        结果汇总
    """
    print(f"\n{'='*70}")
    print(f"任务类型: {task_type}")
    print(f"{'='*70}")
    
    # 获取目标列名
    target_col = f"ret_{config['target']['forward_periods']}d"
    
    # 排序配置
    ranking_config = config.get('ranking', {})
    
    # 创建标签
    label_factory = RankingLabelFactory(
        n_bins=ranking_config.get('lambdarank', {}).get('n_bins', 5),
        rank_method=ranking_config.get('regression_rank', {}).get('rank_method', 'zscore')
    )
    
    min_samples = ranking_config.get('regression_rank', {}).get('min_samples_per_day', 30)
    label_result = label_factory.create_labels(
        forward_returns, task_type, target_col, min_samples
    )
    
    labels = label_result['labels']
    groups = label_result['groups']
    
    # 对齐特征与标签
    X_aligned, y_aligned = label_factory.align_features_with_labels(features, labels)
    
    # 按切分索引获取训练/验证/测试集
    train_common = train_idx.intersection(X_aligned.index)
    valid_common = valid_idx.intersection(X_aligned.index)
    test_common = test_idx.intersection(X_aligned.index)
    
    X_train = X_aligned.loc[train_common].sort_index(level='date')
    y_train = y_aligned.loc[train_common].sort_index(level='date')
    X_valid = X_aligned.loc[valid_common].sort_index(level='date')
    y_valid = y_aligned.loc[valid_common].sort_index(level='date')
    X_test = X_aligned.loc[test_common].sort_index(level='date')
    y_test = y_aligned.loc[test_common].sort_index(level='date')
    
    print(f"训练集: {len(X_train):,} 样本")
    print(f"验证集: {len(X_valid):,} 样本")
    print(f"测试集: {len(X_test):,} 样本")
    
    # 根据任务类型选择模型
    if task_type == 'lambdarank':
        # LambdaRank 需要 group
        train_groups = X_train.groupby(level='date').size().tolist()
        valid_groups = X_valid.groupby(level='date').size().tolist()
        
        model_config = config['models'].get('lightgbm_ranker', {}).get('params', {})
        model = LightGBMRanker(params=model_config)
        
        train_result = model.fit(
            X_train, y_train,
            X_valid, y_valid,
            groups=train_groups,
            valid_groups=valid_groups
        )
    else:
        # 回归模型（regression 或 regression_rank）
        model_config = config['models'].get('lightgbm', {}).get('params', {})
        model = LightGBMModel(params=model_config)
        
        train_result = model.fit(X_train, y_train, X_valid, y_valid)
    
    # 预测
    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)
    pred_test = model.predict(X_test)
    
    # 将预测值转为 Series（保持 MultiIndex）
    pred_train_series = pd.Series(pred_train, index=X_train.index, name='score')
    pred_valid_series = pd.Series(pred_valid, index=X_valid.index, name='score')
    pred_test_series = pd.Series(pred_test, index=X_test.index, name='score')
    
    # 合并所有预测
    all_predictions = pd.concat([pred_train_series, pred_valid_series, pred_test_series])
    all_predictions = all_predictions.to_frame('score')
    
    # 使用 CrossSectionAnalyzer 评估
    # 注意：评估时统一使用原始收益作为 forward_returns
    test_forward_returns = forward_returns.loc[test_common]
    
    print("\n📊 测试集横截面评估...")
    
    analyzer = CrossSectionAnalyzer(
        factors=pred_test_series.to_frame('model_score'),
        forward_returns=test_forward_returns
    )
    analyzer.analyze()
    
    results = analyzer.get_results()
    
    # 提取关键指标
    ic_summary = results.get('ic_summary', {})
    spreads = results.get('spreads', {})
    
    # 构建结果汇总
    summary = {
        'task_type': task_type,
        'train_samples': len(X_train),
        'valid_samples': len(X_valid),
        'test_samples': len(X_test),
        'training_result': train_result,
        'ic_summary': {},
        'spreads': {}
    }
    
    # 转换 IC 统计
    for key, value in ic_summary.items():
        if isinstance(value, dict):
            summary['ic_summary'][str(key)] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in value.items()
            }
    
    # 转换 Spread
    for key, value in spreads.items():
        if hasattr(value, 'mean'):
            summary['spreads'][str(key)] = {
                'mean': float(value.mean()),
                'std': float(value.std()),
                'sharpe': float(value.mean() / value.std()) if value.std() != 0 else 0
            }
    
    # 保存结果
    result_path = os.path.join(output_dir, f'{task_type}_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ 结果已保存: {result_path}")
    
    # 保存预测
    pred_path = os.path.join(output_dir, f'{task_type}_predictions.parquet')
    all_predictions.to_parquet(pred_path)
    print(f"✅ 预测已保存: {pred_path}")
    
    # 保存模型
    model_path = os.path.join(output_dir, f'{task_type}_model.pkl')
    model.save(model_path)
    print(f"✅ 模型已保存: {model_path}")
    
    return summary


def compare_results(results: Dict[str, Dict], output_dir: str) -> Dict:
    """
    对比三条线的结果
    
    Parameters:
    -----------
    results : Dict[str, Dict]
        各任务类型的结果
    output_dir : str
        输出目录
        
    Returns:
    --------
    dict
        对比汇总
    """
    print("\n" + "=" * 70)
    print("三条线对比")
    print("=" * 70)
    
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'tasks': list(results.keys()),
        'metrics': {}
    }
    
    # 收集各任务的关键指标
    for task_type, result in results.items():
        ic_summary = result.get('ic_summary', {})
        spreads = result.get('spreads', {})
        
        # 提取第一个因子的 IC
        first_ic_key = list(ic_summary.keys())[0] if ic_summary else None
        if first_ic_key:
            ic_stats = ic_summary[first_ic_key]
            comparison['metrics'][task_type] = {
                'mean_ic': ic_stats.get('mean', 0),
                'icir': ic_stats.get('icir', 0),
                'icir_annual': ic_stats.get('icir_annual', 0),
                't_stat': ic_stats.get('t_stat', 0),
                'ic_positive_ratio': ic_stats.get('positive_ratio', 0)
            }
        
        # 提取 Spread
        first_spread_key = list(spreads.keys())[0] if spreads else None
        if first_spread_key:
            spread_stats = spreads[first_spread_key]
            comparison['metrics'][task_type]['spread_mean'] = spread_stats.get('mean', 0)
            comparison['metrics'][task_type]['spread_sharpe'] = spread_stats.get('sharpe', 0)
    
    # 打印对比表格
    print("\n📊 关键指标对比:")
    print("-" * 80)
    print(f"{'任务类型':<20} {'Mean IC':>12} {'ICIR':>12} {'ICIR(年化)':>12} {'Spread':>12}")
    print("-" * 80)
    
    for task_type, metrics in comparison['metrics'].items():
        print(f"{task_type:<20} "
              f"{metrics.get('mean_ic', 0):>12.4f} "
              f"{metrics.get('icir', 0):>12.4f} "
              f"{metrics.get('icir_annual', 0):>12.4f} "
              f"{metrics.get('spread_mean', 0):>12.4f}")
    
    print("-" * 80)
    
    # 计算提升比例
    if 'regression' in comparison['metrics'] and len(comparison['metrics']) > 1:
        baseline_ic = comparison['metrics']['regression'].get('mean_ic', 0)
        baseline_icir = comparison['metrics']['regression'].get('icir', 0)
        
        print("\n📈 相对回归基线的提升:")
        for task_type, metrics in comparison['metrics'].items():
            if task_type == 'regression':
                continue
            
            ic_improvement = (abs(metrics.get('mean_ic', 0)) - abs(baseline_ic)) / abs(baseline_ic) * 100 if baseline_ic != 0 else 0
            icir_improvement = (abs(metrics.get('icir', 0)) - abs(baseline_icir)) / abs(baseline_icir) * 100 if baseline_icir != 0 else 0
            
            print(f"  {task_type}: IC 提升 {ic_improvement:+.1f}%, ICIR 提升 {icir_improvement:+.1f}%")
            
            comparison['metrics'][task_type]['ic_improvement_vs_baseline'] = ic_improvement
            comparison['metrics'][task_type]['icir_improvement_vs_baseline'] = icir_improvement
    
    # 保存对比结果
    comparison_path = os.path.join(output_dir, 'model_comparison.json')
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 对比结果已保存: {comparison_path}")
    
    return comparison


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='排序模型训练管道')
    parser.add_argument('--config', type=str, default='configs/ml_baseline.yml',
                        help='配置文件路径')
    parser.add_argument('--task_type', type=str, default=None,
                        choices=['regression', 'regression_rank', 'lambdarank'],
                        help='任务类型（默认从配置读取）')
    parser.add_argument('--compare_all', action='store_true',
                        help='运行三条线对比')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("排序模型训练管道")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载配置
    config = load_config(args.config)
    print(f"✅ 配置加载完成: {args.config}")
    
    # 确定任务类型
    if args.compare_all:
        task_types = ['regression', 'regression_rank', 'lambdarank']
    elif args.task_type:
        task_types = [args.task_type]
    else:
        task_types = [config.get('ranking', {}).get('task_type', 'regression')]
    
    print(f"📋 任务类型: {task_types}")
    
    # 创建输出目录
    output_dir = os.path.join(ml_root, config['paths']['reports_dir'], 'ranking')
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")
    
    # 准备数据
    try:
        features, forward_returns, prices = prepare_data(config)
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        print("尝试使用模拟数据进行测试...")
        
        # 模拟数据（用于测试）
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        tickers = [f'{i:06d}' for i in range(1, 51)]
        index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
        
        features = pd.DataFrame(
            np.random.randn(len(index), 20),
            columns=[f'feature_{i}' for i in range(20)],
            index=index
        )
        
        forward_returns = pd.DataFrame({
            'ret_5d': np.random.randn(len(index)) * 0.05
        }, index=index)
        
        prices = None
        print(f"✅ 模拟数据生成完成: {features.shape}")
    
    # 时序切分
    cv = TimeSeriesCV.from_config(config)
    train_idx, valid_idx, test_idx = cv.single_split(features)
    
    print(f"\n📊 时序切分:")
    print(f"   训练集: {len(train_idx):,}")
    print(f"   验证集: {len(valid_idx):,}")
    print(f"   测试集: {len(test_idx):,}")
    
    # 运行各任务
    all_results = {}
    
    for task_type in task_types:
        try:
            result = run_single_task(
                task_type=task_type,
                config=config,
                features=features,
                forward_returns=forward_returns,
                train_idx=train_idx,
                valid_idx=valid_idx,
                test_idx=test_idx,
                output_dir=output_dir
            )
            all_results[task_type] = result
        except Exception as e:
            print(f"❌ 任务 {task_type} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 对比结果
    if len(all_results) > 1:
        compare_results(all_results, output_dir)
    
    print("\n" + "=" * 70)
    print("✅ 排序模型训练完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
