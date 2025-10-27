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

from data.data_loader import DataLoader


def load_config(config_path: str = "configs/ml_baseline.yml"):
    """加载配置文件"""
    config_file = os.path.join(ml_root, config_path)
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主函数"""
    print("=" * 70)
    print("数据清洗与快照层 - 集成演示")
    print("=" * 70)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 1. 加载配置
    print("\n[步骤1] 加载配置")
    config = load_config()
    
    symbol = config['data']['symbol']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    target_col = config['target']['name']
    random_seed = config['runtime']['random_seed']
    
    print(f"   股票代码: {symbol}")
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
    
    loader = DataLoader(
        data_root=os.path.join(ml_root, "ML output/datasets/baseline_v1"),
        enable_snapshot=config['data']['snapshot']['enabled'],
        enable_filtering=True,
        enable_pit_alignment=config['data']['pit']['enabled'],
        filter_config=filter_config
    )
    
    # 3. 加载数据并创建快照
    print("\n[步骤3] 加载数据并创建快照")
    
    try:
        features, targets, snapshot_id = loader.load_with_snapshot(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            target_col=target_col,
            use_scaled=config['features']['use_scaled_features'],
            filters=filter_config,
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
        
        # 5. 展示特征列表
        print("\n[步骤5] 特征列表（前10个）")
        for i, col in enumerate(features.columns[:10], 1):
            print(f"   {i:2d}. {col}")
        print(f"   ... 共 {len(features.columns)} 个特征")
        
        # 6. 快照信息
        if snapshot_id and loader.snapshot_mgr:
            print("\n[步骤6] 快照信息")
            snapshots = loader.snapshot_mgr.list_snapshots()
            print("\n现有快照:")
            print(snapshots.to_string(index=False))
            
            # 质量报告位置
            quality_report_path = os.path.join(
                loader.snapshot_mgr.quality_reports_dir,
                f"{snapshot_id}.json"
            )
            if os.path.exists(quality_report_path):
                print(f"\n   📊 数据质量报告: {quality_report_path}")
        
        # 7. 验收检查
        print("\n[步骤7] 数据验收检查")
        
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
        
        print("\n" + "=" * 70)
        print(f"{'✅ 验收通过' if all_passed else '❌ 验收失败'}")
        print("=" * 70)
        
        if all_passed:
            print(f"\n🎉 恭喜! 数据清洗与快照层验收通过")
            print(f"   快照ID: {snapshot_id}")
            print(f"   可用于后续模型训练")
        else:
            print(f"\n⚠️  警告: 部分验收项未通过，请检查数据质量")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
