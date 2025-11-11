#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗与快照层 - 集成示例

展示如何使用新的数据清洗和快照功能

功能流程：
1. 加载原始数据
2. 应用交易可行性过滤（7层）
3. PIT对齐验证
4. 创建数据快照
5. 生成数据质量报告
"""

import os
import sys
import pandas as pd
import yaml
from datetime import datetime

# 添加项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
project_root = os.path.dirname(ml_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 添加 machine learning 目录到路径
if ml_root not in sys.path:
    sys.path.insert(0, ml_root)

from data.data_loader import DataLoader
from features.feature_engineering import FeatureEngineer
from targets.target_engineering import TargetEngineer


def load_config(config_path: str = "configs/ml_baseline.yml"):
    """加载配置文件"""
    config_file = os.path.join(ml_root, config_path)
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主函数"""
    print("=" * 70)
    print("数据清洗与快照层 - 批量处理")
    print("=" * 70)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 1. 加载配置
    print("\n[步骤1] 加载配置")
    config = load_config()
    
    # 支持单标的和多标的
    config_symbol = config['data']['symbol']
    if isinstance(config_symbol, list):
        symbols = config_symbol
        is_multi = True
    else:
        symbols = [config_symbol]
        is_multi = False
    
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    target_col = config['target']['name']
    random_seed = config['runtime']['random_seed']
    
    mode_name = "多标的" if is_multi else "单标的"
    print(f"   模式: {mode_name}")
    print(f"   股票代码: {symbols}")
    print(f"   股票数量: {len(symbols)}")
    print(f"   时间范围: {start_date} ~ {end_date}")
    print(f"   目标变量: {target_col}")
    print(f"   随机种子: {random_seed}")
    
    # 2. 初始化数据加载器（启用所有功能）
    print("\n[步骤2] 初始化增强版数据加载器")
    
    # 提取过滤器配置
    filter_config = {
        'min_volume': config['data']['universe']['min_volume'],
        'min_amount': config['data']['universe']['min_amount'],
        'min_price': config['data']['universe']['min_price'],
        'min_turnover': config['data']['universe']['min_turnover'],
        'min_listing_days': config['data']['universe']['min_listing_days'],
        'exclude_st': config['data']['universe']['exclude_st'],
        'exclude_limit_moves': config['data']['universe']['exclude_limit_moves'],
        'limit_threshold': config['data']['universe']['limit_threshold']
    }
    
    # 从配置文件读取 InfluxDB 配置
    influxdb_enabled = config['data'].get('influxdb', {}).get('enabled', True)
    influxdb_config = None
    if influxdb_enabled:
        influxdb_config = {
            'url': config['data']['influxdb']['url'],
            'org': config['data']['influxdb']['org'],
            'bucket': config['data']['influxdb']['bucket'],
            'token': config['data']['influxdb']['token']
        }
    
    loader = DataLoader(
        data_root=os.path.join(ml_root, "ML output/datasets/baseline_v1"),
        enable_snapshot=config['data']['snapshot']['enabled'],
        enable_filtering=True,
        enable_pit_alignment=config['data']['pit']['enabled'],
        enable_influxdb=influxdb_enabled,
        influxdb_config=influxdb_config,
        filter_config=filter_config
    )
    
    # 3. 批量处理所有股票
    print(f"\n[步骤3] 批量处理 {len(symbols)} 只股票")
    print("=" * 70)
    
    success_count = 0
    failed_symbols = []
    
    for idx, symbol in enumerate(symbols, 1):
        print(f"\n{'='*70}")
        print(f"处理股票 [{idx}/{len(symbols)}]: {symbol}")
        print(f"{'='*70}")
        
        try:
            # 步骤3.1: 特征工程
            print(f"\n[步骤3.1] 特征工程")
            feature_engineer = FeatureEngineer(use_talib=True, use_tsfresh=False)
            
            # 加载原始数据
            raw_data = feature_engineer.load_stock_data(symbol, start_date, end_date)
            if raw_data is None or len(raw_data) == 0:
                print(f"   ⚠️ 跳过 {symbol}：无原始数据")
                failed_symbols.append(symbol)
                continue
            
            print(f"   原始数据: {raw_data.shape}")
            
            # 生成技术特征
            features_df = feature_engineer.prepare_features(
                raw_data,
                use_auto_features=False,
                keep_base_columns=True
            )
            print(f"   ✅ 特征生成完成: {features_df.shape[1]} 个特征, {features_df.shape[0]} 个样本")
            
            # 步骤3.1.1: 应用交易可行性过滤（在标准化之前）
            print(f"\n[步骤3.1.1] 应用交易可行性过滤器（基于原始值）")
            if loader.filter_engine:
                filtered_features, filter_log = loader.filter_engine.apply_filters(
                    features_df,
                    save_log=True,
                    log_path=os.path.join(
                        config['paths'].get('datasets_dir', 'ML output/datasets/baseline_v1'),
                        f"filter_log_{symbol}.csv"
                    ) if not os.path.isabs(config['paths'].get('datasets_dir', 'ML output/datasets/baseline_v1')) 
                      else os.path.join(ml_root, config['paths'].get('datasets_dir', 'ML output/datasets/baseline_v1'), f"filter_log_{symbol}.csv")
                )
                
                # 只保留可交易的样本
                features_df = filtered_features[filtered_features['tradable_flag'] == 1].copy()
                features_df = features_df.drop(columns=['tradable_flag'], errors='ignore')
                
                print(f"\n   ✅ 交易过滤完成:")
                print(f"      过滤前: {len(filtered_features)} 个样本")
                print(f"      过滤后: {len(features_df)} 个可交易样本")
                print(f"      剔除: {len(filtered_features) - len(features_df)} 个样本 ({(len(filtered_features) - len(features_df))/len(filtered_features):.1%})")
                
                if len(features_df) == 0:
                    print(f"\n      ⚠️  警告: 所有样本均被过滤，跳过 {symbol}")
                    failed_symbols.append(symbol)
                    continue
            else:
                print(f"   ⏭️  跳过交易过滤（未启用）")
            
            # 步骤3.2: 特征选择（可选）
            skip_selection = config.get('features', {}).get('skip_selection', False)
            
            if skip_selection:
                print(f"\n[步骤3.2] ⏭️  跳过特征选择（配置文件设置）")
                selected_features = features_df
                final_feature_count = len([c for c in features_df.columns 
                                          if c not in ['close', 'volume', 'amount', 'pct_change', 'turnover']])
            else:
                print(f"\n[步骤3.2] 特征选择")
                
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
            
            # 步骤3.3: 特征标准化
            print(f"\n[步骤3.3] 特征标准化")
            scalers_dir = config['paths'].get('scalers_dir', 'ML output/scalers/baseline_v1')
            if not os.path.isabs(scalers_dir):
                scalers_dir = os.path.join(ml_root, scalers_dir)
            os.makedirs(scalers_dir, exist_ok=True)
            
            scaler_path = os.path.join(scalers_dir, f"scaler_{symbol}.pkl")
            scale_results = feature_engineer.scale_features(
                selected_features,
                scaler_type='robust',
                train_ratio=0.8,
                save_path=scaler_path
            )
            scaled_features = scale_results['scaled_df']
            print(f"   ✅ 特征标准化完成")
            print(f"   💾 标准化器: {scaler_path}")
            
            # 步骤3.4: 目标工程
            print(f"\n[步骤3.4] 目标工程")
            datasets_dir = config['paths'].get('datasets_dir', 'ML output/datasets/baseline_v1')
            if not os.path.isabs(datasets_dir):
                datasets_dir = os.path.join(ml_root, datasets_dir)
            os.makedirs(datasets_dir, exist_ok=True)
            
            target_engineer = TargetEngineer(data_dir=datasets_dir)
            complete_df = target_engineer.create_complete_dataset(
                features_df=scaled_features,
                periods=[1, 5, 10],
                price_col='close',
                include_labels=True,
                label_types=['binary']
            )
            
            print(f"   ✅ 目标变量生成完成: {complete_df.shape}")
            
            # 临时保存完整数据集（用于后续的数据质量检查）
            temp_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_csv = f"with_targets_{symbol}_complete_{temp_timestamp}.csv"
            temp_csv_path = os.path.join(datasets_dir, temp_csv)
            complete_df.to_csv(temp_csv_path)
            print(f"   💾 临时保存: {temp_csv}")
            
            # 步骤3.4: 数据质量检查和快照
            print(f"\n[步骤3.4] 数据质量检查和快照")
            # 注意：过滤已在步骤3.1.1完成，这里不再重复过滤
            features, targets, snapshot_id = loader.load_with_snapshot(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                target_col=target_col,
                use_scaled=config['features']['use_scaled_features'],
                filters=None,  # 不再应用过滤器（已在标准化前完成）
                random_seed=random_seed,
                save_parquet=config['data']['snapshot']['save_parquet']
            )
            
            print(f"\n✅ 数据加载成功!")
            print(f"   快照ID: {snapshot_id}")
            print(f"   特征形状: {features.shape}")
            print(f"   目标形状: {targets.shape}")
            
            # 4. 展示数据质量统计
            print("\n[步骤4] 数据质量统计")
            print(f"   特征缺失率: {features.isna().sum().sum() / features.size:.2%}")
            print(f"   目标缺失率: {targets.isna().sum() / len(targets):.2%}")
            print(f"   时间范围: {features.index.get_level_values('date').min().date()} ~ "
                  f"{features.index.get_level_values('date').max().date()}")
            
            # 5. 快照信息
            if snapshot_id and loader.snapshot_mgr:
                # 质量报告位置
                quality_report_path = os.path.join(
                    loader.snapshot_mgr.quality_reports_dir,
                    f"{snapshot_id}.json"
                )
            
            # 6. 验收检查
            print("\n[步骤5] 数据验收检查")
            
            # 检查1: 可交易样本规模
            n_samples = len(features)
            min_samples = 200  # 根据宪章要求
            sample_check = n_samples >= min_samples
            print(f"   {'✅' if sample_check else '❌'} 样本规模: {n_samples} (最低 {min_samples})")
            
            # 检查2: PIT对齐
            if loader.pit_aligner:
                combined = features.copy()
                combined[target_col] = targets
                pit_results = loader.pit_aligner.validate_pit_alignment(combined, target_col)
                pit_check = pit_results.get('overall_pass', False)
                print(f"   {'✅' if pit_check else '❌'} PIT对齐验证")
            else:
                pit_check = True
                print(f"   ⚠️  PIT对齐验证（未启用）")
            
            # 检查3: 数据质量
            if snapshot_id and loader.snapshot_mgr:
                import json
                with open(quality_report_path, 'r', encoding='utf-8') as f:
                    quality_report = json.load(f)
                quality_check = quality_report.get('overall_quality') == 'PASS'
                red_flags = quality_report.get('red_flags_count', 0)
                print(f"   {'✅' if quality_check else '❌'} 数据质量: {quality_report.get('overall_quality')} ({red_flags} 个红灯)")
            else:
                quality_check = True
                print(f"   ⚠️  数据质量检查（未启用快照）")
            
            # 总体验收
            all_passed = sample_check and pit_check and quality_check
            
            if all_passed:
                # 7. 保存CSV格式数据集（用于后续 train_models.py）
                print("\n[步骤6] 保存CSV格式数据集（用于模型训练）")
                datasets_dir = config['paths'].get('datasets_dir', 'ML output/datasets/baseline_v1')
                if not os.path.isabs(datasets_dir):
                    datasets_dir = os.path.join(ml_root, datasets_dir)
                os.makedirs(datasets_dir, exist_ok=True)
                
                # 合并特征和目标
                complete_df = features.copy()
                complete_df[target_col] = targets
                
                # 保存为CSV
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = f"with_targets_{symbol}_complete_{timestamp}.csv"
                csv_path = os.path.join(datasets_dir, csv_filename)
                complete_df.to_csv(csv_path)
                
                print(f"   ✅ CSV数据集已保存: {csv_path}")
                print(f"   📊 形状: {complete_df.shape}")
                
                success_count += 1
                print(f"\n✅ {symbol} 处理成功！")
            else:
                print(f"\n⚠️ {symbol} 验收失败，跳过保存")
                failed_symbols.append(symbol)
            
        except Exception as e:
            print(f"\n❌ {symbol} 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_symbols.append(symbol)
            continue
    
    # 4. 最终统计
    print("\n" + "=" * 70)
    print("批量处理完成")
    print("=" * 70)
    print(f"✅ 成功: {success_count}/{len(symbols)}")
    if failed_symbols:
        print(f"❌ 失败: {len(failed_symbols)}/{len(symbols)}")
        print(f"   失败股票: {failed_symbols}")
    print("=" * 70)
    
    return 0 if success_count == len(symbols) else 1


if __name__ == "__main__":
    exit(main())
