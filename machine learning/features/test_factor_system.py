#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子工厂系统测试脚本

快速测试因子工厂的各个组件
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
if ml_root not in sys.path:
    sys.path.insert(0, ml_root)

from features.factor_factory import FactorFactory
from features.factor_quality_checker import FactorQualityChecker
from features.factor_library_manager import FactorLibraryManager


def generate_mock_data(n_days=500, n_stocks=10):
    """生成模拟市场数据"""
    print("📊 生成模拟数据...")
    
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    tickers = [f"00000{i}.SZ" for i in range(1, n_stocks + 1)]
    
    # 创建MultiIndex
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
    
    # 模拟OHLCV数据
    np.random.seed(42)
    n_rows = len(index)
    
    data = {
        'open': np.random.uniform(10, 50, n_rows),
        'high': np.random.uniform(10, 50, n_rows),
        'low': np.random.uniform(10, 50, n_rows),
        'close': np.random.uniform(10, 50, n_rows),
        'volume': np.random.uniform(1000000, 10000000, n_rows),
        'amount': np.random.uniform(10000000, 100000000, n_rows),
        'turnover': np.random.uniform(0.5, 5.0, n_rows)
    }
    
    # 确保 high >= low, close在high-low之间
    for i in range(n_rows):
        data['high'][i] = max(data['high'][i], data['low'][i], data['close'][i])
        data['low'][i] = min(data['low'][i], data['close'][i])
    
    df = pd.DataFrame(data, index=index)
    
    # 生成模拟目标（未来5日收益）
    df['future_return_5d'] = np.random.uniform(-0.1, 0.1, n_rows)
    
    print(f"   数据形状: {df.shape}")
    print(f"   日期范围: {df.index.get_level_values('date').min()} ~ {df.index.get_level_values('date').max()}")
    print(f"   股票数量: {df.index.get_level_values('ticker').nunique()}")
    
    return df


def test_factor_factory():
    """测试因子工厂"""
    print("\n" + "=" * 80)
    print("测试 1: 因子工厂 (FactorFactory)")
    print("=" * 80)
    
    # 生成模拟数据
    mock_data = generate_mock_data(n_days=500, n_stocks=10)
    
    # 创建因子工厂
    factory = FactorFactory()
    
    # 生成所有因子
    print("\n🏭 生成因子...")
    factors_df = factory.generate_all_factors(mock_data)
    
    print(f"\n✅ 因子生成成功")
    print(f"   生成因子数: {factors_df.shape[1]}")
    print(f"   因子形状: {factors_df.shape}")
    print(f"\n前5个因子:")
    print(f"   {list(factors_df.columns[:5])}")
    
    # 获取因子注册信息
    registry = factory.get_factor_registry()
    
    # 统计各族因子
    family_counts = {}
    for factor_info in registry.values():
        family = factor_info['family']
        family_counts[family] = family_counts.get(family, 0) + 1
    
    print(f"\n因子族统计:")
    for family, count in family_counts.items():
        print(f"   {family}: {count} 个")
    
    return mock_data, factors_df, registry


def test_quality_checker(mock_data, factors_df):
    """测试质量检查器"""
    print("\n" + "=" * 80)
    print("测试 2: 质量检查器 (FactorQualityChecker)")
    print("=" * 80)
    
    checker = FactorQualityChecker(
        ic_threshold=0.01,  # 模拟数据用较低阈值
        icir_threshold=0.3,
        psi_threshold=0.3,
        corr_threshold=0.7
    )
    
    # 选取前3个因子测试
    test_factors = factors_df.columns[:3]
    
    print(f"\n🔍 测试 {len(test_factors)} 个因子...")
    
    qualified = []
    reports = {}
    
    for factor_name in test_factors:
        print(f"\n检查因子: {factor_name}")
        
        factor_values = factors_df[factor_name]
        target_values = mock_data['future_return_5d']
        
        # 综合检查
        report = checker.comprehensive_check(
            factor_values=factor_values,
            target_values=target_values
        )
        
        reports[factor_name] = report
        
        if report['overall_pass']:
            qualified.append(factor_name)
            print(f"   ✅ 通过")
        else:
            print(f"   ❌ 拒绝 | 原因: {', '.join(report['fail_reasons'])}")
        
        # 显示关键指标
        print(f"   IC: {report['ic_metrics']['ic_mean']:.4f}")
        print(f"   ICIR: {report['ic_metrics']['icir_annual']:.2f}")
        if 'psi' in report:
            print(f"   PSI: {report['psi']:.4f}")
    
    print(f"\n✅ 质量检查完成")
    print(f"   通过因子: {len(qualified)} / {len(test_factors)}")
    
    return qualified, reports


def test_library_manager(qualified_factors, reports, registry):
    """测试库管理器"""
    print("\n" + "=" * 80)
    print("测试 3: 库管理器 (FactorLibraryManager)")
    print("=" * 80)
    
    # 创建测试目录
    test_artifacts_dir = os.path.join(ml_root, "ML output/artifacts/test")
    test_reports_dir = os.path.join(ml_root, "ML output/reports/test/factors")
    
    manager = FactorLibraryManager(
        artifacts_dir=test_artifacts_dir,
        reports_dir=test_reports_dir
    )
    
    # 添加通过的因子
    print(f"\n📥 添加 {len(qualified_factors)} 个因子...")
    
    for factor_name in qualified_factors:
        factor_info = registry.get(factor_name, {})
        quality_report = reports[factor_name]
        
        success = manager.add_factor(
            factor_name=factor_name,
            quality_report=quality_report,
            formula=factor_info.get('formula', ''),
            family=factor_info.get('family', ''),
            reference=factor_info.get('reference', '')
        )
    
    # 列出因子
    print(f"\n📋 当前因子库:")
    for factor in manager.list_factors():
        print(f"   - {factor}")
    
    # 生成报告
    print(f"\n📊 生成报告...")
    report_df = manager.generate_factor_report()
    
    print(f"\n因子报告:")
    print(report_df.to_string(index=False))
    
    print(f"\n✅ 库管理器测试完成")
    
    return manager


def main():
    """主测试流程"""
    print("=" * 80)
    print("因子工厂系统测试")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 测试1: 因子工厂
        mock_data, factors_df, registry = test_factor_factory()
        
        # 测试2: 质量检查
        qualified, reports = test_quality_checker(mock_data, factors_df)
        
        # 测试3: 库管理
        if qualified:
            manager = test_library_manager(qualified, reports, registry)
        else:
            print("\n⚠️  没有合格因子，跳过库管理测试")
        
        # 总结
        print("\n" + "=" * 80)
        print("测试总结")
        print("=" * 80)
        print(f"✅ 所有测试通过")
        print(f"   生成因子数: {factors_df.shape[1]}")
        print(f"   合格因子数: {len(qualified)}")
        print(f"   通过率: {len(qualified) / factors_df.shape[1] * 100:.1f}%")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
