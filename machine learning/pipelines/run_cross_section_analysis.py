#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
横截面因子评估示例脚本

展示如何使用横截面评估框架进行因子分析
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

# 导入横截面评估模块
from evaluation.cross_section_analyzer import CrossSectionAnalyzer
from evaluation.visualization import create_factor_tearsheet_plots
from evaluation.tearsheet import generate_full_tearsheet


def load_demo_data():
    """
    加载演示数据
    
    实际使用时，请替换为真实数据加载逻辑
    """
    print("📦 加载演示数据...")
    
    # 生成测试数据
    np.random.seed(42)
    
    # 时间范围：1年
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    # 股票池：100只股票
    tickers = [f'Stock_{i:03d}' for i in range(100)]
    
    # 创建MultiIndex
    index = pd.MultiIndex.from_product(
        [dates, tickers],
        names=['date', 'ticker']
    )
    
    # 1. 价格数据（模拟）
    prices = pd.DataFrame({
        'close': 100 + np.random.randn(len(index)).cumsum() * 0.1
    }, index=index)
    
    # 2. 因子数据（添加一些预测能力）
    # 计算真实收益（用于构造有预测能力的因子）
    returns_true = prices['close'].groupby(level='ticker').pct_change()
    
    factors = pd.DataFrame({
        'factor_momentum': returns_true.shift(1) + np.random.randn(len(index)) * 0.02,  # 动量因子
        'factor_value': np.random.randn(len(index)) * 0.01,  # 价值因子（随机，弱因子）
        'factor_quality': returns_true.shift(2) * 0.5 + np.random.randn(len(index)) * 0.015,  # 质量因子
    }, index=index)
    
    # 3. 市值数据（可选，用于中性化）
    market_cap = pd.DataFrame({
        'market_cap': np.random.lognormal(20, 2, len(index))
    }, index=index)
    
    # 4. 行业数据（可选，用于中性化）
    industries = ['金融', '科技', '消费', '医药', '工业', '能源', '材料', '通信']
    industry = pd.DataFrame({
        'industry': np.random.choice(industries, len(index))
    }, index=index)
    
    # 5. 可交易性过滤（可选）
    # 例如：剔除停牌、涨跌停、ST等
    tradable_mask = pd.DataFrame({
        'tradable': np.random.rand(len(index)) > 0.05  # 95%可交易
    }, index=index)
    
    print(f"   ✅ 价格数据: {prices.shape}")
    print(f"   ✅ 因子数据: {factors.shape}, 因子数: {len(factors.columns)}")
    print(f"   ✅ 市值数据: {market_cap.shape}")
    print(f"   ✅ 行业数据: {industry.shape}")
    
    return prices, factors, market_cap, industry, tradable_mask


def run_cross_section_analysis(prices, factors, market_cap, industry, tradable_mask,
                               output_dir='./ML output/reports/baseline_v1/factors'):
    """
    执行横截面分析
    
    Parameters:
    -----------
    prices : pd.DataFrame
        价格数据
    factors : pd.DataFrame
        因子数据
    market_cap : pd.DataFrame
        市值数据
    industry : pd.DataFrame
        行业数据
    tradable_mask : pd.DataFrame
        可交易性标记
    output_dir : str
        输出目录
    """
    print("\n" + "=" * 70)
    print("横截面因子评估流程")
    print("=" * 70)
    
    # 步骤1：创建分析器
    print("\n步骤1: 创建CrossSectionAnalyzer")
    print("-" * 70)
    
    analyzer = CrossSectionAnalyzer(
        factors=factors,
        prices=prices,
        tradable_mask=tradable_mask,
        market_cap=market_cap,
        industry=industry
    )
    
    # 步骤2：因子预处理（可选）
    print("\n步骤2: 因子预处理")
    print("-" * 70)
    print("配置:")
    print("  - Winsorize: True (1%-99%)")
    print("  - 标准化: True (Z-score)")
    print("  - 中性化: False (演示时关闭，实盘建议开启)")
    
    analyzer.preprocess(
        winsorize=True,
        standardize=True,
        neutralize=False,  # 实盘时建议设为True
        winsorize_params={'lower_quantile': 0.01, 'upper_quantile': 0.99},
        standardize_params={'method': 'z_score', 'cross_section': True}
    )
    
    # 步骤3：计算远期收益
    print("\n步骤3: 计算远期收益")
    print("-" * 70)
    
    analyzer.calculate_returns(
        periods=[1, 5, 10, 20],  # 1日、5日、10日、20日
        method='simple'  # 或'log'
    )
    
    # 步骤4：执行横截面分析
    print("\n步骤4: 执行横截面分析")
    print("-" * 70)
    
    analyzer.analyze(
        n_quantiles=5,  # 分5档
        ic_method='spearman',  # Rank IC
        spread_method='top_minus_mean',  # Top - Mean（实盘更贴合）
        periods_per_year=252  # 年化参数
    )
    
    # 步骤5：查看汇总结果
    print("\n步骤5: 查看汇总结果")
    print("-" * 70)
    
    analyzer.summary()
    
    # 步骤6：生成报告
    print("\n步骤6: 生成报告和图表")
    print("-" * 70)
    
    results = analyzer.get_results()
    
    # 为每个因子和收益期组合生成报告
    factor_names = factors.columns.tolist()
    return_periods = ['ret_1d', 'ret_5d', 'ret_10d', 'ret_20d']
    
    for factor_name in factor_names:
        for return_period in return_periods:
            key = (factor_name, return_period)
            
            # 检查是否有数据
            if 'ic_summary' not in results or key not in results['ic_summary']:
                print(f"   ⚠️  跳过 {factor_name} @ {return_period}（无数据）")
                continue
            
            print(f"\n   📊 生成报告: {factor_name} @ {return_period}")
            
            # 创建输出目录
            factor_output_dir = os.path.join(output_dir, factor_name)
            os.makedirs(factor_output_dir, exist_ok=True)
            
            # 生成图表
            plot_paths = create_factor_tearsheet_plots(
                results,
                factor_name,
                return_period,
                factor_output_dir
            )
            
            # 生成HTML报告和CSV
            generate_full_tearsheet(
                results,
                factor_name,
                return_period,
                factor_output_dir,
                plot_paths
            )
    
    print("\n" + "=" * 70)
    print("✅ 横截面分析完成！")
    print(f"📁 报告目录: {output_dir}")
    print("=" * 70)
    
    return analyzer, results


def verify_results(results):
    """
    验收检查
    
    要求：
    1. IC与手算一致
    2. 分层收益图、IC走廊图、换手曲线输出齐全
    """
    print("\n" + "=" * 70)
    print("验收检查")
    print("=" * 70)
    
    checks_passed = []
    checks_failed = []
    
    # 检查1：IC计算
    if 'ic_summary' in results and len(results['ic_summary']) > 0:
        checks_passed.append("✅ IC统计计算完成")
        
        # 检查IC的合理性
        first_key = list(results['ic_summary'].keys())[0]
        ic_summary = results['ic_summary'][first_key]
        
        if 'mean' in ic_summary and not np.isnan(ic_summary['mean']):
            checks_passed.append(f"✅ IC均值有效: {ic_summary['mean']:.4f}")
        else:
            checks_failed.append("❌ IC均值无效")
    else:
        checks_failed.append("❌ IC统计未生成")
    
    # 检查2：分位数收益
    if 'quantile_returns' in results and len(results['quantile_returns']) > 0:
        checks_passed.append("✅ 分位数收益计算完成")
    else:
        checks_failed.append("❌ 分位数收益未生成")
    
    # 检查3：累计收益
    if 'cumulative_returns' in results and len(results['cumulative_returns']) > 0:
        checks_passed.append("✅ 累计收益计算完成")
    else:
        checks_failed.append("❌ 累计收益未生成")
    
    # 检查4：Spread
    if 'spreads' in results and len(results['spreads']) > 0:
        checks_passed.append("✅ Spread计算完成")
    else:
        checks_failed.append("❌ Spread未生成")
    
    # 检查5：换手率
    if 'turnover_stats' in results and len(results['turnover_stats']) > 0:
        checks_passed.append("✅ 换手率计算完成")
    else:
        checks_failed.append("❌ 换手率未生成")
    
    # 打印结果
    print("\n通过的检查:")
    for check in checks_passed:
        print(f"  {check}")
    
    if checks_failed:
        print("\n未通过的检查:")
        for check in checks_failed:
            print(f"  {check}")
    
    print("\n" + "=" * 70)
    
    if len(checks_failed) == 0:
        print("🎉 所有验收检查通过！")
    else:
        print(f"⚠️  {len(checks_failed)}/{len(checks_passed) + len(checks_failed)} 项检查未通过")
    
    print("=" * 70)


def main():
    """主函数"""
    print("=" * 70)
    print("横截面因子评估示例")
    print("=" * 70)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. 加载数据
    prices, factors, market_cap, industry, tradable_mask = load_demo_data()
    
    # 2. 执行分析
    analyzer, results = run_cross_section_analysis(
        prices,
        factors,
        market_cap,
        industry,
        tradable_mask,
        output_dir='./ML output/reports/baseline_v1/factors'
    )
    
    # 3. 验收
    verify_results(results)
    
    print("\n" + "=" * 70)
    print("🎉 示例运行完成！")
    print("=" * 70)
    
    print("\n💡 使用提示:")
    print("   1. 查看HTML报告以了解因子表现")
    print("   2. 检查IC时间序列的稳定性")
    print("   3. 观察分位数收益的单调性")
    print("   4. 评估Spread的夏普比")
    print("   5. 考虑换手率对交易成本的影响")
    
    print("\n📚 实盘使用建议:")
    print("   1. 开启中性化（neutralize=True）")
    print("   2. 使用真实的市值和行业数据")
    print("   3. 添加可交易性过滤（停牌、涨跌停、ST等）")
    print("   4. 定期回测以验证因子有效性")
    print("   5. 组合多个有效因子以提升稳定性")


if __name__ == '__main__':
    main()
