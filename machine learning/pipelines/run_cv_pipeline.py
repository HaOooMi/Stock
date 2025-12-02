#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时序交叉验证管道 - 统一的 Purged+Embargo CV 流程

功能：
1. 加载配置和数据
2. 应用 Purged + Embargo 时间切分
3. 可选：Walk-Forward 多折验证
4. 因子横截面评估（各分割独立）
5. 漂移检测与报告生成

使用方法：
    python run_cv_pipeline.py
    python run_cv_pipeline.py --config configs/ml_baseline.yml
    python run_cv_pipeline.py --wfa  # 强制使用 Walk-Forward

输出：
    /ML output/reports/baseline_v1/cv/
    ├── drift_report.json
    ├── drift_tearsheet.html
    ├── split_comparison.csv
    └── fold_X_results.json (WFA 模式)

创建: 2025-12-02 | 版本: v1.0
"""

import os
import pandas as pd
import sys
import yaml
import argparse
import json
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
if ml_root not in sys.path:
    sys.path.insert(0, ml_root)

from data.data_loader import DataLoader
from data.time_series_cv import TimeSeriesCV, create_cv_from_config
from data.market_data_loader import MarketDataLoader
from targets.label_transformer import create_forward_returns_with_transform
from evaluation.cross_section_analyzer import CrossSectionAnalyzer
from evaluation.cross_section_metrics import calculate_forward_returns
from evaluation.drift_detector import DriftDetector, compare_splits_with_analyzer
from features.factor_factory import FactorFactory


def load_config(config_path: str = "configs/ml_baseline.yml") -> dict:
    """加载配置文件"""
    if not os.path.isabs(config_path):
        config_path = os.path.join(ml_root, config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def run_single_split_cv(config: dict,
                        factors: 'pd.DataFrame',
                        forward_returns: 'pd.DataFrame',
                        output_dir: str) -> dict:
    """
    单次时间切分 + 漂移检测
    
    Parameters:
    -----------
    config : dict
        配置字典
    factors : pd.DataFrame
        因子数据
    forward_returns : pd.DataFrame
        远期收益
    output_dir : str
        输出目录
        
    Returns:
    --------
    dict
        结果汇总
    """
    print("\n" + "=" * 80)
    print("单次时间切分 CV（Purged + Embargo）")
    print("=" * 80)
    
    # 创建 CV 实例
    cv = TimeSeriesCV.from_config(config)
    
    # 获取切分索引
    train_idx, valid_idx, test_idx = cv.single_split(factors)
    
    # 验证无泄漏
    target_horizon = config.get('target', {}).get('forward_periods', 5)
    cv.validate_no_leakage(train_idx, valid_idx, test_idx, target_horizon)
    
    # 漂移检测与分析
    results = compare_splits_with_analyzer(
        factors=factors,
        forward_returns=forward_returns,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        output_dir=output_dir,
        drift_threshold=config.get('split', {}).get('drift_threshold', 0.2)
    )
    
    # 保存切分元数据
    meta_path = os.path.join(output_dir, 'cv_meta.json')
    meta = cv.get_split_meta()
    meta['train_samples'] = len(train_idx)
    meta['valid_samples'] = len(valid_idx)
    meta['test_samples'] = len(test_idx)
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 切分元数据已保存: {meta_path}")
    
    return results


def run_walk_forward_cv(config: dict,
                        factors: 'pd.DataFrame',
                        forward_returns: 'pd.DataFrame',
                        output_dir: str) -> dict:
    """
    Walk-Forward 验证
    
    Parameters:
    -----------
    config : dict
        配置字典
    factors : pd.DataFrame
        因子数据
    forward_returns : pd.DataFrame
        远期收益
    output_dir : str
        输出目录
        
    Returns:
    --------
    dict
        结果汇总
    """
    print("\n" + "=" * 80)
    print("Walk-Forward 验证（Purged + Embargo）")
    print("=" * 80)
    
    # 创建 CV 实例
    cv = TimeSeriesCV.from_config(config)
    
    # WFA 配置
    wfa_config = config.get('split', {}).get('walk_forward', {})
    n_splits = wfa_config.get('n_splits', 5)
    min_train_days = wfa_config.get('min_train_days', 252)
    expanding = wfa_config.get('expanding', True)
    
    # 收集所有折的结果
    all_fold_results = []
    all_oos_ic = []
    all_oos_spread = []
    
    detector = DriftDetector(
        drift_threshold=config.get('split', {}).get('drift_threshold', 0.2)
    )
    
    for fold, train_idx, valid_idx, test_idx in cv.walk_forward_split(
        factors, n_splits=n_splits, min_train_days=min_train_days, expanding=expanding
    ):
        fold_dir = os.path.join(output_dir, f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # 各分割分析
        try:
            # Train
            train_analyzer = CrossSectionAnalyzer(
                factors=factors.loc[train_idx],
                forward_returns=forward_returns.loc[train_idx]
            )
            train_analyzer.analyze()
            
            # Valid
            valid_analyzer = CrossSectionAnalyzer(
                factors=factors.loc[valid_idx],
                forward_returns=forward_returns.loc[valid_idx]
            )
            valid_analyzer.analyze()
            
            # Test
            test_analyzer = CrossSectionAnalyzer(
                factors=factors.loc[test_idx],
                forward_returns=forward_returns.loc[test_idx]
            )
            test_analyzer.analyze()
            
            # 收集 OOS 结果（Valid + Test）
            oos_ic = valid_analyzer.results.get('daily_ic', None)
            if oos_ic is not None:
                all_oos_ic.append(oos_ic)
            
            oos_ic_test = test_analyzer.results.get('daily_ic', None)
            if oos_ic_test is not None:
                all_oos_ic.append(oos_ic_test)
            
            # 收集 OOS Spread 结果
            oos_spread = valid_analyzer.results.get('spreads', None)
            if oos_spread is not None:
                all_oos_spread.append(oos_spread)
            
            oos_spread_test = test_analyzer.results.get('spreads', None)
            if oos_spread_test is not None:
                all_oos_spread.append(oos_spread_test)
            
            fold_results = {
                'fold': fold + 1,
                'train_samples': len(train_idx),
                'valid_samples': len(valid_idx),
                'test_samples': len(test_idx),
                'train_ic_summary': train_analyzer.results.get('ic_summary', {}),
                'valid_ic_summary': valid_analyzer.results.get('ic_summary', {}),
                'test_ic_summary': test_analyzer.results.get('ic_summary', {})
            }
            
            all_fold_results.append(fold_results)
            
        except Exception as e:
            print(f"   ⚠️  Fold {fold+1} 分析失败: {e}")
            continue
    
    # 合并 OOS 结果
    print("\n" + "=" * 70)
    print("合并 OOS 结果")
    print("=" * 70)
    
    if all_oos_ic:
        import pandas as pd
        combined_oos_ic = pd.concat(all_oos_ic).groupby(level=0).mean()
        
        # 计算合并后的 IC 统计
        from evaluation.cross_section_metrics import calculate_ic_summary
        
        combined_summary = {}
        for col in combined_oos_ic.columns:
            combined_summary[col] = calculate_ic_summary(combined_oos_ic[col])
        
        print(f"\n📊 合并 OOS IC 统计:")
        for key, summary in list(combined_summary.items())[:3]:
            print(f"   {key}:")
            print(f"      Mean IC: {summary['mean']:.4f}")
            print(f"      ICIR: {summary['icir']:.4f}")
            print(f"      ICIR(年化): {summary['icir_annual']:.4f}")
    
    # 合并 OOS Spread 结果
    if all_oos_spread:
        # spreads 是字典 {(factor, period): Series}，需要按 key 合并
        combined_spread_stats = {}
        for spread_dict in all_oos_spread:
            for key, spread_series in spread_dict.items():
                if key not in combined_spread_stats:
                    combined_spread_stats[key] = []
                combined_spread_stats[key].append(spread_series)
        
        print(f"\n📊 合并 OOS Spread 统计:")
        for key, series_list in list(combined_spread_stats.items())[:3]:
            combined = pd.concat(series_list)
            mean_spread = combined.mean()
            std_spread = combined.std()
            sharpe = mean_spread / std_spread if std_spread != 0 else 0
            print(f"   {key}: Mean={mean_spread:.4f}, Std={std_spread:.4f}, Sharpe={sharpe:.4f}")
    
    # 保存 WFA 结果
    wfa_results = {
        'mode': 'walk_forward',
        'n_folds': len(all_fold_results),
        'config': {
            'n_splits': n_splits,
            'min_train_days': min_train_days,
            'expanding': expanding
        },
        'folds': all_fold_results
    }
    
    results_path = os.path.join(output_dir, 'wfa_results.json')
    
    # 转换 numpy 类型
    def convert_to_native(obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_native(i) for i in obj)
        return obj
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_native(wfa_results), f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 WFA 结果已保存: {results_path}")
    
    return wfa_results


def main(config_path: str = "configs/ml_baseline.yml",
         force_wfa: bool = False):
    """
    主函数
    
    Parameters:
    -----------
    config_path : str
        配置文件路径
    force_wfa : bool
        强制使用 Walk-Forward 模式
    """
    print("=" * 80)
    print("时序交叉验证管道")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载配置
    print("\n" + "=" * 80)
    print("步骤 1: 加载配置")
    print("=" * 80)
    
    config = load_config(config_path)
    
    project_name = config.get('project', {}).get('name', 'baseline_v1')
    cv_mode = config.get('split', {}).get('cv_mode', 'single_split')
    
    if force_wfa:
        cv_mode = 'walk_forward'
    
    print(f"   项目: {project_name}")
    print(f"   CV 模式: {cv_mode}")
    
    # 2. 加载数据
    print("\n" + "=" * 80)
    print("步骤 2: 加载数据")
    print("=" * 80)
    
    influxdb_config = config['data']['influxdb']
    tickers = config['data'].get('symbol', ['000001'])
    if isinstance(tickers, str):
        tickers = [tickers]
    
    start_date = config['data'].get('start_date', '2018-01-01')
    end_date = config['data'].get('end_date', '2024-12-31')
    
    market_loader = MarketDataLoader(
        url=influxdb_config['url'],
        token=influxdb_config['token'],
        org=influxdb_config['org'],
        bucket=influxdb_config['bucket']
    )
    
    market_data = market_loader.load_market_data_batch(
        symbols=tickers,
        start_date=start_date,
        end_date=end_date
    )
    
    if market_data.empty:
        raise ValueError("未加载到市场数据")
    
    print(f"   数据形状: {market_data.shape}")
    print(f"   日期范围: {market_data.index.get_level_values('date').min()} ~ {market_data.index.get_level_values('date').max()}")
    
    # 3. 生成因子
    print("\n" + "=" * 80)
    print("步骤 3: 生成因子")
    print("=" * 80)
    
    factory = FactorFactory()
    
    # 生成一些基础因子
    factors_list = []
    
    # 动量因子
    roc_factors = factory.calc_roc_family(market_data, periods=[5, 20, 60])
    factors_list.append(roc_factors)
    
    # 波动率因子
    vol_factors = factory.calc_realized_volatility(market_data, periods=[20])
    factors_list.append(vol_factors)
    
    import pandas as pd
    factors = pd.concat(factors_list, axis=1).dropna()
    
    print(f"   因子数量: {factors.shape[1]}")
    print(f"   样本数量: {len(factors)}")
    
    # 4. 计算远期收益
    print("\n" + "=" * 80)
    print("步骤 4: 计算远期收益")
    print("=" * 80)
    
    target_config = config.get('target', {})
    periods = [target_config.get('forward_periods', 5)]
    method = target_config.get('return_type', 'simple')
    transform = target_config.get('transform', 'none')
    
    forward_returns = calculate_forward_returns(
        market_data[['close']],
        periods=periods,
        method=method
    )
    
    # 对齐因子和收益
    common_idx = factors.index.intersection(forward_returns.index)
    factors = factors.loc[common_idx]
    forward_returns = forward_returns.loc[common_idx]
    
    print(f"   收益周期: {periods}")
    print(f"   收益类型: {method}")
    print(f"   对齐后样本: {len(factors)}")
    
    # 5. 运行 CV
    output_dir = os.path.join(ml_root, 'ML output', 'reports', project_name, 'cv')
    os.makedirs(output_dir, exist_ok=True)
    
    if cv_mode == 'walk_forward':
        results = run_walk_forward_cv(config, factors, forward_returns, output_dir)
    else:
        results = run_single_split_cv(config, factors, forward_returns, output_dir)
    
    # 6. 完成
    print("\n" + "=" * 80)
    print("完成")
    print("=" * 80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出目录: {output_dir}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='时序交叉验证管道')
    parser.add_argument('--config', type=str, default='configs/ml_baseline.yml',
                       help='配置文件路径')
    parser.add_argument('--wfa', action='store_true',
                       help='强制使用 Walk-Forward 模式')
    
    args = parser.parse_args()
    
    try:
        main(config_path=args.config, force_wfa=args.wfa)
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
