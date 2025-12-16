#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline 模型训练管道 - Learning-to-Rank 三条线对比

功能：
1. Baseline A：回归原始收益（LGBMRegressor）
2. Baseline B：Reg-on-Rank（LGBMRegressor + GaussRank 标签）
3. Sorting：LambdaRank（LGBMRanker）

流程：
1. 数据加载（复用 DataLoader）
2. 时序 CV 切分（Purged + Embargo）
3. 特征分布漂移检测（PSI）- 训练前
4. 三条线模型训练
5. 横截面评估（CrossSectionAnalyzer）
6. 模型预测漂移检测（IC/Spread）- 训练后
7. 结果对比与报告

使用方法：
    python run_baseline_pipeline.py
    python run_baseline_pipeline.py --task_type lambdarank
    python run_baseline_pipeline.py --compare_all  # 运行三条线对比
    python run_baseline_pipeline.py --skip_drift   # 跳过漂移检测

输出：
    /ML output/reports/baseline_v1/ranking/
    ├── model_comparison.json           # 三条线对比结果
    ├── feature_drift_report.json       # 特征分布漂移检测（PSI）
    ├── prediction_drift_report.json    # 模型预测漂移检测（IC/Spread）
    ├── regression_results.json
    ├── regression_rank_results.json
    ├── lambdarank_results.json
    ├── {task_type}_predictions.parquet
    └── {task_type}_model.pkl

创建: 2025-12-04 | 版本: v1.2
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
from evaluation.drift_detector import DriftDetector
from backtest.simple_backtest import SimplePortfolioBacktester


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
    
    forward_periods = target_config['forward_periods']
    target_col = f"future_return_{forward_periods}d"
    
    # 检查是否使用中性化因子
    use_neutralized = config['features'].get('use_neutralized_features', False)
    
    if use_neutralized:
        # ===== 直接加载中性化因子文件（已经是多股票合并的） =====
        print("📂 加载中性化因子...")
        
        datasets_dir = os.path.join(ml_root, "ML output/datasets/baseline_v1")
        
        # 查找中性化因子文件
        neutral_files = [f for f in os.listdir(datasets_dir) 
                        if f.startswith('qualified_factors_neutralized_') and f.endswith('.parquet')]
        
        if not neutral_files:
            print("   ⚠️ 未找到中性化因子文件，降级使用原始因子")
            use_neutralized = False
        else:
            neutral_files.sort(reverse=True)
            neutral_file = os.path.join(datasets_dir, neutral_files[0])
            print(f"   📈 加载: {neutral_files[0]}")
            
            features = pd.read_parquet(neutral_file)
            print(f"   ✅ 特征形状: {features.shape}")
            
            # 加载对应的目标数据（从 with_targets 文件）
            # 中性化因子的索引应该是 MultiIndex [date, ticker]
            if isinstance(features.index, pd.MultiIndex):
                # 从索引中提取股票列表
                available_tickers = features.index.get_level_values('ticker').unique().tolist()
                print(f"   📋 包含股票: {available_tickers[:5]}{'...' if len(available_tickers) > 5 else ''}")
                
                # 加载目标数据
                all_targets = []
                for ticker in available_tickers:
                    # 查找匹配的 with_targets 文件（格式: with_targets_{ticker}_complete_YYYYMMDD_HHMMSS.csv）
                    target_files = [f for f in os.listdir(datasets_dir) 
                                   if f.startswith(f"with_targets_{ticker}_complete_") and f.endswith('.csv')]
                    
                    if target_files:
                        # 使用最新的文件
                        target_files.sort(reverse=True)
                        target_file = os.path.join(datasets_dir, target_files[0])
                        
                        df = pd.read_csv(target_file, index_col=0, parse_dates=True)
                        if target_col in df.columns:
                            targets = df[[target_col]].copy()
                            targets['ticker'] = ticker
                            targets = targets.reset_index()
                            targets = targets.rename(columns={'index': 'date'})
                            all_targets.append(targets)
                
                if all_targets:
                    targets_df = pd.concat(all_targets, ignore_index=True)
                    
                    # 确保日期格式统一（去除时区信息）
                    targets_df['date'] = pd.to_datetime(targets_df['date']).dt.tz_localize(None)
                    
                    targets_df = targets_df.set_index(['date', 'ticker'])
                    forward_returns = targets_df[[target_col]].rename(columns={target_col: f'ret_{forward_periods}d'})
                    
                    # 确保特征索引也无时区
                    if features.index.get_level_values('date').tz is not None:
                        features = features.reset_index()
                        features['date'] = features['date'].dt.tz_localize(None)
                        features = features.set_index(['date', 'ticker'])
                    
                    # 对齐特征和目标
                    common_idx = features.index.intersection(forward_returns.index)
                    print(f"   � 共同索引数: {len(common_idx)}")
                    
                    if len(common_idx) == 0:
                        raise ValueError("特征和目标没有共同索引，请检查日期格式")
                    
                    features = features.loc[common_idx]
                    forward_returns = forward_returns.loc[common_idx]
                else:
                    raise FileNotFoundError("无法加载目标数据")
            else:
                raise ValueError("中性化因子文件索引格式错误，应为 MultiIndex [date, ticker]")
            
            print(f"✅ 特征加载完成: {features.shape}")
            print(f"✅ 目标加载完成: {len(forward_returns)}")
            
            # ===== 加载价格数据用于回测（中性化因子模式） =====
            print("\n📂 加载价格数据用于回测...")
            prices = None
            
            # 准备 InfluxDB 配置
            influxdb_config = data_config.get('influxdb', {})
            
            # 初始化 MarketDataLoader
            if influxdb_config.get('enabled', False):
                try:
                    from data.market_data_loader import MarketDataLoader
                    
                    market_loader = MarketDataLoader(
                        url=influxdb_config['url'],
                        token=influxdb_config['token'],
                        org=influxdb_config['org'],
                        bucket=influxdb_config['bucket']
                    )
                    
                    all_prices = []
                    for ticker in available_tickers:
                        try:
                            price_df = market_loader.load_market_data(
                                symbol=ticker,
                                start_date=str(data_config['start_date']),
                                end_date=str(data_config['end_date'])
                            )
                            if not price_df.empty:
                                price_df['ticker'] = ticker
                                price_df = price_df.reset_index()
                                price_df = price_df.rename(columns={'index': 'date'})
                                price_df['date'] = pd.to_datetime(price_df['date']).dt.tz_localize(None)
                                price_df = price_df.set_index(['date', 'ticker'])
                                all_prices.append(price_df)
                                print(f"   ✅ {ticker}: {len(price_df)} 条价格记录")
                        except Exception as e:
                            print(f"   ⚠️ {ticker} 价格加载失败: {e}")
                            continue
                    
                    if all_prices:
                        prices = pd.concat(all_prices)
                        required_cols = ['open', 'close']
                        missing_cols = [col for col in required_cols if col not in prices.columns]
                        if missing_cols:
                            print(f"   ⚠️ 价格数据缺少列: {missing_cols}，回测将无法运行")
                            prices = None
                        else:
                            print(f"   ✅ 价格数据加载完成: {len(prices)} 条记录")
                except Exception as e:
                    print(f"   ⚠️ 价格数据加载失败: {e}")
                    prices = None
            else:
                print("   ⚠️ InfluxDB 未启用，无法加载价格数据")
            
            print(f"✅ 样本总数: {len(features):,}")
            
            return features, forward_returns, prices
    
    # ===== 原有逻辑：按单股票加载 =====
    # 初始化数据加载器
    influxdb_config = data_config.get('influxdb', {}).copy()
    influxdb_config.pop('enabled', None)
    
    # 数据集目录（with_targets 文件在这里）
    datasets_dir = os.path.join(ml_root, config['paths'].get('datasets_dir', 'ML output/datasets/baseline_v1'))
    
    loader = DataLoader(
        data_root=datasets_dir,  # 使用 datasets 目录
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
                use_scaled=config['features']['use_scaled_features'],
                use_neutralized=False  # 这里不再使用中性化，因为上面已经处理
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
    
    # ===== 加载价格数据用于回测 =====
    print("\n📂 加载价格数据用于回测...")
    prices = None
    
    if loader.market_data_loader is not None:
        # 从 InfluxDB 加载价格数据
        try:
            all_prices = []
            for symbol in symbols:
                try:
                    price_df = loader.market_data_loader.load_market_data(
                        symbol=symbol,
                        start_date=str(data_config['start_date']),
                        end_date=str(data_config['end_date'])
                    )
                    if not price_df.empty:
                        # 添加 ticker 列并设置 MultiIndex
                        price_df['ticker'] = symbol
                        price_df = price_df.reset_index()
                        price_df = price_df.rename(columns={'index': 'date'})
                        price_df['date'] = pd.to_datetime(price_df['date'])
                        price_df = price_df.set_index(['date', 'ticker'])
                        all_prices.append(price_df)
                        print(f"   ✅ {symbol}: {len(price_df)} 条价格记录")
                except Exception as e:
                    print(f"   ⚠️ {symbol} 价格加载失败: {e}")
                    continue
            
            if all_prices:
                prices = pd.concat(all_prices)
                # 确保包含 open 和 close 列
                required_cols = ['open', 'close']
                missing_cols = [col for col in required_cols if col not in prices.columns]
                if missing_cols:
                    print(f"   ⚠️ 价格数据缺少列: {missing_cols}，回测将无法运行")
                    prices = None
                else:
                    print(f"   ✅ 价格数据加载完成: {len(prices)} 条记录")
        except Exception as e:
            print(f"   ⚠️ 价格数据加载失败: {e}")
            prices = None
    else:
        print("   ⚠️ MarketDataLoader 未初始化，无法加载价格数据")
    
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
    # 注意：groups 在切分后会重新计算，这里不需要保留
    
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
    # y_test 用于未来计算测试集排序损失（如 NDCG），当前评估用原始收益
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


def run_feature_drift_detection(features: pd.DataFrame,
                                 train_idx: pd.Index,
                                 valid_idx: pd.Index,
                                 test_idx: pd.Index,
                                 output_dir: str,
                                 drift_threshold: float = 0.2,
                                 max_features: Optional[int] = None) -> Dict:
    """
    运行特征分布漂移检测（训练前）
    
    使用 PSI 检测特征分布是否发生变化，用于：
    - 发现数据质量问题
    - 检测市场环境变化
    - 决定是否需要重新训练模型
    
    Parameters:
    -----------
    features : pd.DataFrame
        特征数据
    train_idx, valid_idx, test_idx : pd.Index
        切分索引
    output_dir : str
        输出目录
    drift_threshold : float
        漂移阈值（PSI >= 0.2 表示显著漂移）
    max_features : int, optional
        最多检测的特征数量，None 表示检测所有特征
        
    Returns:
    --------
    dict
        漂移检测结果
    """
    print("\n" + "=" * 70)
    print("特征分布漂移检测 (PSI)")
    print("=" * 70)
    
    # 使用 DriftDetector 模块
    detector = DriftDetector(drift_threshold=drift_threshold)
    
    # 按索引切分特征
    train_features = features.loc[train_idx]
    valid_features = features.loc[valid_idx]
    test_features = features.loc[test_idx]
    
    # 检测特征分布漂移
    drift_results = detector.detect_feature_drift(
        train_features=train_features,
        valid_features=valid_features,
        test_features=test_features,
        max_features=max_features
    )
    
    # 打印摘要
    print(f"   检测特征数: {drift_results['n_checked']}")
    print(f"   漂移特征数: {drift_results['n_drifted']}")
    drifted = drift_results['drifted_features']
    if drifted:
        print(f"   漂移特征: {drifted[:5]}{'...' if len(drifted) > 5 else ''}")
    
    # 保存结果
    drift_path = os.path.join(output_dir, 'feature_drift_report.json')
    with open(drift_path, 'w', encoding='utf-8') as f:
        json.dump(drift_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 特征漂移报告已保存: {drift_path}")
    
    return drift_results


def run_prediction_drift_detection(predictions: Dict[str, pd.Series],
                                    forward_returns: pd.DataFrame,
                                    train_idx: pd.Index,
                                    valid_idx: pd.Index,
                                    test_idx: pd.Index,
                                    output_dir: str,
                                    drift_threshold: float = 0.2) -> Dict:
    """
    运行模型预测漂移检测（训练后）
    
    比较 Train/Valid/Test 的 IC 和 Spread 差异，用于：
    - 验证模型泛化能力
    - 检测过拟合
    - 满足研究宪章验收标准（Valid vs Test 差异 < 20%）
    
    Parameters:
    -----------
    predictions : Dict[str, pd.Series]
        各任务类型的预测结果 {task_type: pred_series}
    forward_returns : pd.DataFrame
        远期收益
    train_idx, valid_idx, test_idx : pd.Index
        切分索引
    output_dir : str
        输出目录
    drift_threshold : float
        漂移阈值（默认 20%）
        
    Returns:
    --------
    dict
        预测漂移检测结果
    """
    print("\n" + "=" * 70)
    print("模型预测漂移检测 (IC/Spread)")
    print("=" * 70)
    
    detector = DriftDetector(drift_threshold=drift_threshold)
    
    all_drift_reports = {}
    
    for task_type, pred_series in predictions.items():
        print(f"\n📊 检测 {task_type}...")
        
        # 获取各分割的预测和收益
        train_common = train_idx.intersection(pred_series.index)
        valid_common = valid_idx.intersection(pred_series.index)
        test_common = test_idx.intersection(pred_series.index)
        
        if len(train_common) == 0 or len(valid_common) == 0 or len(test_common) == 0:
            print(f"   ⚠️ {task_type} 数据不足，跳过")
            continue
        
        # 构建因子 DataFrame
        factors = pred_series.to_frame('model_score')
        
        # 分别分析各分割
        train_analyzer = CrossSectionAnalyzer(
            factors=factors.loc[train_common],
            forward_returns=forward_returns.loc[train_common]
        )
        train_analyzer.analyze()
        train_results = train_analyzer.get_results()
        
        valid_analyzer = CrossSectionAnalyzer(
            factors=factors.loc[valid_common],
            forward_returns=forward_returns.loc[valid_common]
        )
        valid_analyzer.analyze()
        valid_results = valid_analyzer.get_results()
        
        test_analyzer = CrossSectionAnalyzer(
            factors=factors.loc[test_common],
            forward_returns=forward_returns.loc[test_common]
        )
        test_analyzer.analyze()
        test_results = test_analyzer.get_results()
        
        # 使用 DriftDetector 的 detect_drift 方法
        ret_col = list(forward_returns.columns)[0]
        period = ret_col.replace('ret_', '')
        
        drift_report = detector.detect_drift(
            train_results=train_results,
            valid_results=valid_results,
            test_results=test_results,
            factor_name='model_score',
            period=period
        )
        
        all_drift_reports[task_type] = drift_report
    
    # 保存结果
    drift_path = os.path.join(output_dir, 'prediction_drift_report.json')
    
    # 转换为可序列化格式
    def convert_to_native(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        return obj
    
    with open(drift_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_native(all_drift_reports), f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 预测漂移报告已保存: {drift_path}")
    
    # 汇总结果
    print("\n📊 预测漂移检测汇总:")
    print("-" * 60)
    for task_type, report in all_drift_reports.items():
        status = "✅ 通过" if report.get('overall_pass', False) else "❌ 未通过"
        print(f"   {task_type}: {status}")
    
    return all_drift_reports


def run_portfolio_backtest(predictions: Dict[str, pd.Series],
                           prices: pd.DataFrame,
                           output_dir: str,
                           top_k: int = 30,
                           compare_modes: bool = True) -> Dict:
    """
    运行组合回测（阶段二：闭环回测）
    
    支持 A/B 测试：
    - Close-to-Close (理想情况，有前视偏差)
    - Open-to-Open (现实情况，T+1 执行)
    
    Parameters:
    -----------
    predictions : Dict[str, pd.Series]
        各任务类型的预测结果 {task_type: pred_series}
    prices : pd.DataFrame
        价格数据，MultiIndex [date, ticker]，必须包含 'open' 和 'close' 列
    output_dir : str
        输出目录
    top_k : int
        Top-K 选股数量
    compare_modes : bool
        是否对比两种执行模式
        
    Returns:
    --------
    Dict
        各任务类型的回测结果
    """
    print("\n" + "=" * 70)
    print("组合回测 (Simple Portfolio Backtest)")
    print("=" * 70)
    
    if prices is None:
        print("⚠️ 价格数据为空，跳过回测")
        print("   提示：请确保 DataLoader 加载了包含 'open' 和 'close' 列的价格数据")
        return {}
    
    # 检查价格数据是否包含必要列
    required_cols = ['open', 'close']
    missing_cols = [col for col in required_cols if col not in prices.columns]
    if missing_cols:
        print(f"⚠️ 价格数据缺少列: {missing_cols}，跳过回测")
        return {}
    
    all_backtest_results = {}
    
    for task_type, pred_series in predictions.items():
        print(f"\n📊 回测 {task_type}...")
        
        try:
            backtester = SimplePortfolioBacktester(top_k=top_k)
            
            if compare_modes:
                # A/B 测试：对比两种执行模式
                result = backtester.compare_modes(
                    predictions=pred_series,
                    prices=prices,
                    save_dir=output_dir
                )
                all_backtest_results[task_type] = result
                
                # 保存统计结果
                stats_path = os.path.join(output_dir, f'{task_type}_backtest_stats.json')
                stats_to_save = {
                    'close_to_close': result['close_to_close']['stats'],
                    'open_to_open': result['open_to_open']['stats'],
                    'comparison': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                                   for k, v in result['comparison'].items() 
                                   if not isinstance(v, dict)}
                }
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(stats_to_save, f, indent=2, ensure_ascii=False, default=str)
                print(f"   ✅ 回测统计已保存: {stats_path}")
                
            else:
                # 单模式回测
                result = backtester.run(pred_series, prices)
                all_backtest_results[task_type] = result
                
                # 绘制并保存图表
                plot_path = os.path.join(output_dir, f'{task_type}_backtest.png')
                backtester.plot(result, save_path=plot_path)
                
        except Exception as e:
            print(f"   ❌ {task_type} 回测失败: {e}")
            import traceback
            traceback.print_exc()
    
    return all_backtest_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Baseline 模型训练管道')
    parser.add_argument('--config', type=str, default='configs/ml_baseline.yml',
                        help='配置文件路径')
    parser.add_argument('--task_type', type=str, default=None,
                        choices=['regression', 'regression_rank', 'lambdarank'],
                        help='任务类型（默认从配置读取）')
    parser.add_argument('--compare_all', action='store_true',
                        help='运行三条线对比')
    parser.add_argument('--skip_drift', action='store_true',
                        help='跳过漂移检测')
    parser.add_argument('--skip_backtest', action='store_true',
                        help='跳过组合回测')
    parser.add_argument('--backtest_top_k', type=int, default=30,
                        help='回测 Top-K 选股数量')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Baseline 模型训练管道 (Learning-to-Rank)")
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
    features, forward_returns, prices = prepare_data(config)
    
    # 时序切分
    cv = TimeSeriesCV.from_config(config)
    train_idx, valid_idx, test_idx = cv.single_split(features)
    
    print(f"\n📊 时序切分 (Purged + Embargo):")
    print(f"   训练集: {len(train_idx):,}")
    print(f"   验证集: {len(valid_idx):,}")
    print(f"   测试集: {len(test_idx):,}")
    
    # 特征分布漂移检测（训练前）
    if not args.skip_drift:
        drift_threshold = config.get('split', {}).get('drift_threshold', 0.2)
        run_feature_drift_detection(
            features=features,
            train_idx=train_idx,
            valid_idx=valid_idx,
            test_idx=test_idx,
            output_dir=output_dir,
            drift_threshold=drift_threshold
        )
    
    # 运行各任务
    all_results = {}
    all_predictions = {}  # 收集预测结果用于漂移检测
    
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
            
            # 加载预测结果用于漂移检测
            pred_path = os.path.join(output_dir, f'{task_type}_predictions.parquet')
            if os.path.exists(pred_path):
                pred_df = pd.read_parquet(pred_path)
                all_predictions[task_type] = pred_df['score']
                
        except Exception as e:
            print(f"❌ 任务 {task_type} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 模型预测漂移检测（训练后）
    if not args.skip_drift and all_predictions:
        drift_threshold = config.get('split', {}).get('drift_threshold', 0.2)
        run_prediction_drift_detection(
            predictions=all_predictions,
            forward_returns=forward_returns,
            train_idx=train_idx,
            valid_idx=valid_idx,
            test_idx=test_idx,
            output_dir=output_dir,
            drift_threshold=drift_threshold
        )
    
    # 对比结果
    if len(all_results) > 1:
        compare_results(all_results, output_dir)
    
    # ========== 组合回测 (阶段二) ==========
    if not args.skip_backtest and all_predictions:
        # 尝试加载价格数据用于回测
        # 如果 prepare_data 没有返回 prices，尝试从 InfluxDB 或文件加载
        if prices is None:
            print("\n📂 尝试加载价格数据用于回测...")
            try:
                # 尝试从 DataLoader 加载价格数据
                data_config = config['data']
                influxdb_config = data_config.get('influxdb', {}).copy()
                influxdb_config.pop('enabled', None)
                
                loader = DataLoader(
                    enable_influxdb=data_config['influxdb']['enabled'],
                    influxdb_config=influxdb_config
                )
                
                # 获取所有股票代码
                all_tickers = list(set(
                    idx[1] for pred in all_predictions.values() 
                    for idx in pred.index
                ))
                
                # 加载价格数据
                prices_list = []
                for ticker in all_tickers[:10]:  # 限制数量避免过慢
                    try:
                        price_df = loader.load_market_data(ticker)
                        if price_df is not None and 'open' in price_df.columns:
                            prices_list.append(price_df)
                    except:
                        continue
                
                if prices_list:
                    prices = pd.concat(prices_list)
                    print(f"   ✅ 加载价格数据: {len(prices)} 条记录")
                else:
                    print("   ⚠️ 无法加载价格数据，跳过回测")
                    prices = None
                    
            except Exception as e:
                print(f"   ⚠️ 加载价格数据失败: {e}")
                prices = None
        
        if prices is not None:
            backtest_results = run_portfolio_backtest(
                predictions=all_predictions,
                prices=prices,
                output_dir=output_dir,
                top_k=args.backtest_top_k,
                compare_modes=True  # 默认进行 A/B 测试
            )
    
    print("\n" + "=" * 70)
    print("✅ Baseline 模型训练完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
