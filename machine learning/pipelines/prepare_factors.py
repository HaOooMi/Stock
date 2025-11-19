#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子准备管道 - 因子工厂完整流程

本管道完全基于已有的横截面评估框架 (evaluation/) 构建，
实现从因子生成到质量检查、报告输出的全流程自动化。

流程：
1. 加载市场数据（DataLoader）
   - 从InfluxDB加载OHLCV数据
   - 应用可交易性过滤
   
2. 生成因子（FactorFactory）
   - 动量族: ROC_N, Price_to_SMA, RankMomentum
   - 波动率族: RealizedVol, Parkinson, Skewness
   - 量价族: Turnover, VolumePriceCorr, VWAP_Dev
   
3. 横截面质量检查 ⭐核心步骤
   使用 evaluation/CrossSectionAnalyzer 进行完整评估：
   - calculate_forward_returns: 计算远期收益
   - preprocess: Winsorize + 标准化 + 中性化
   - analyze: 计算IC/ICIR/Spread/单调性
   - 评估标准: IC≥0.02, ICIR≥0.5, Spread>0
   
4. 因子入库（FactorLibraryManager）
   - 只有通过横截面检查的因子才入库
   - 保存质量报告和元数据
   
5. 生成报告 ⭐输出标准化
   使用 evaluation/tearsheet 生成报告：
   - HTML tearsheet: tearsheet_{factor}_{period}.html
   - IC序列CSV: ic_{factor}_{period}.csv
   - 分位数收益CSV: quantile_returns_{factor}_{period}.csv
   - 自动生成所有图表 (IC走廊图、累计收益、Spread等)

输出目录结构:
/ML output/reports/baseline_v1/factors/
  ├── tearsheet_ROC_20_5d.html
  ├── ic_ROC_20_5d.csv
  └── quantile_returns_ROC_20_5d.csv
/ML output/figures/baseline_v1/factors/
  ├── ic_series_ROC_20_5d.png
  ├── quantile_cumret_ROC_20_5d.png
  └── spread_cumret_ROC_20_5d.png
/ML output/datasets/baseline_v1/
  └── qualified_factors_YYYYMMDD.parquet

验收标准:
- ≥10个稳定因子通过检查
- 横截面Rank IC显著 (IC>0.02, p<0.05)
- 合入后组合IC有实质提升 (>0.03)

详细文档: pipelines/README_PREPARE_FACTORS.md
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
if ml_root not in sys.path:
    sys.path.insert(0, ml_root)

from features.factor_factory import FactorFactory
from features.factor_library_manager import FactorLibraryManager
from data.data_loader import DataLoader
# 使用你已有的横截面评估框架！
from evaluation.cross_section_analyzer import CrossSectionAnalyzer
from evaluation.cross_section_metrics import calculate_forward_returns
from evaluation.factor_preprocessing import preprocess_factor_pipeline
from evaluation.tearsheet import generate_html_tearsheet
# 使用factor_quality_checker进行补充检查（IC半衰期、PSI/KS）
from evaluation.factor_quality_checker import FactorQualityChecker


def load_config(config_path: str = "configs/ml_baseline.yml") -> dict:
    """加载配置文件"""
    if not os.path.isabs(config_path):
        config_path = os.path.join(ml_root, config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def prepare_factors(config_path: str = "configs/ml_baseline.yml",
                   start_date: str = "2020-01-01",
                   end_date: str = "2024-12-31",
                   tickers: list = None):
    """
    因子准备完整流程
    
    Parameters:
    -----------
    config_path : str
        配置文件路径
    start_date : str
        开始日期
    end_date : str
        结束日期
    tickers : list, optional
        股票列表（None表示全市场）
    """
    print("=" * 80)
    print("因子工厂 v1 - 完整流程")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据区间: {start_date} ~ {end_date}")
    print(f"股票范围: {tickers if tickers else '全市场'}")
    print()
    
    # 1. 加载配置
    print("\n" + "=" * 80)
    print("步骤 1: 加载配置")
    print("=" * 80)
    config = load_config(config_path)
    
    # 从配置中提取参数
    influxdb_config = config['data']['influxdb']
    target_config = config['targets']
    tradability_config = config['data'].get('tradability_filter', {})
    
    print(f"✅ 配置加载完成")
    print(f"   InfluxDB: {influxdb_config['url']}")
    print(f"   预测目标: {target_config['type']} ({target_config['horizon']}日)")
    
    # 2. 加载数据（使用MarketDataLoader批量加载）
    print("\n" + "=" * 80)
    print("步骤 2: 加载市场数据")
    print("=" * 80)
    
    # 使用MarketDataLoader批量加载多股票数据
    from data.market_data_loader import MarketDataLoader
    
    market_loader = MarketDataLoader(
        url=influxdb_config['url'],
        token=influxdb_config['token'],
        org=influxdb_config['org'],
        bucket=influxdb_config['bucket']
    )
    
    # 如果未指定tickers，可以从配置或数据库获取股票池
    if not tickers:
        # TODO: 从配置文件或数据库获取股票池
        print(f"\n⚠️  未指定股票列表，请在配置中设置或传入tickers参数")
        raise ValueError("必须提供tickers参数")
    
    # 批量加载市场数据（返回MultiIndex[date, ticker]格式）
    features_df = market_loader.load_market_data_batch(
        symbols=tickers,
        start_date=start_date,
        end_date=end_date
    )
    
    if features_df.empty:
        raise ValueError(f"未加载到任何数据，请检查InfluxDB连接和股票代码")
    
    # 计算目标变量（远期收益）
    from evaluation.cross_section_metrics import calculate_forward_returns
    
    prices_df = features_df[['close']]
    targets_df = calculate_forward_returns(
        prices=prices_df,
        periods=[1, 5, 10, 20],
        method='simple'
    )
    
    print(f"\n✅ 数据加载完成")
    print(f"   特征数据形状: {features_df.shape}")
    print(f"   目标数据形状: {targets_df.shape}")
    print(f"   日期范围: {features_df.index.get_level_values('date').min()} ~ {features_df.index.get_level_values('date').max()}")
    print(f"   股票数量: {features_df.index.get_level_values('ticker').nunique()}")
    print(f"   股票列表: {', '.join(features_df.index.get_level_values('ticker').unique()[:5])}..." if len(tickers) > 5 else f"   股票列表: {', '.join(tickers)}")
    
    # 3. 生成因子
    print("\n" + "=" * 80)
    print("步骤 3: 生成因子")
    print("=" * 80)
    
    factory = FactorFactory()
    
    # 生成所有因子族
    print("\n🏭 生成因子...")
    all_factors_df = factory.generate_all_factors(features_df)
    
    print(f"\n✅ 因子生成完成")
    print(f"   生成因子数: {all_factors_df.shape[1]}")
    print(f"   因子族统计:")
    
    # 统计各族因子
    factor_families = factory.get_factor_registry()
    family_counts = {}
    for factor_info in factor_families.values():
        family = factor_info['family']
        family_counts[family] = family_counts.get(family, 0) + 1
    
    for family, count in family_counts.items():
        print(f"   - {family}: {count} 个")
    
    # 4. 横截面质量检查（使用你已有的评估框架！）
    print("\n" + "=" * 80)
    print("步骤 4: 横截面因子评估（Alphalens风格）")
    print("=" * 80)
    
    # 准备价格数据用于计算forward returns
    prices_df = features_df[['close']] if 'close' in features_df.columns else None
    
    # 计算远期收益（使用你的cross_section_metrics）
    print(f"\n📊 计算远期收益...")
    forward_horizons = [1, 5, 10, 20]
    forward_returns_df = calculate_forward_returns(
        prices=prices_df,
        periods=forward_horizons,
        method='simple'
    )
    print(f"   ✅ 远期收益计算完成: {forward_returns_df.shape}")
    
    # 准备可交易性mask（可选）
    # 如果features_df中有tradable列，使用它；否则为None
    tradable_mask = None
    if 'tradable' in features_df.columns:
        tradable_mask = features_df[['tradable']]
        print(f"   ✅ 使用可交易性mask")
    else:
        print(f"   ⚠️  未提供可交易性mask，将使用全部样本")
    
    # 准备市值和行业数据（用于中性化）
    market_cap = features_df[['market_cap']] if 'market_cap' in features_df.columns else None
    industry = features_df[['industry']] if 'industry' in features_df.columns else None
    
    # 质量检查阈值
    IC_THRESHOLD = 0.02
    ICIR_THRESHOLD = 0.5
    SPREAD_THRESHOLD = 0.0
    CORR_THRESHOLD = 0.7
    
    # 逐个因子评估
    qualified_factors = []
    quality_reports = {}
    
    print(f"\n🔍 开始横截面评估 (共 {all_factors_df.shape[1]} 个因子)...\n")
    
    # 预处理配置
    preprocess_config = {
        'winsorize': True,
        'winsorize_limits': (0.01, 0.99),
        'standardize': 'zscore',
        'neutralize': None  # 可选: ['market_cap', 'industry']
    }
    
    for i, factor_name in enumerate(all_factors_df.columns, 1):
        print(f"[{i}/{all_factors_df.shape[1]}] 评估因子: {factor_name}")
        
        try:
            # 构建单因子DataFrame
            single_factor_df = all_factors_df[[factor_name]]
            
            # 使用你的CrossSectionAnalyzer！
            analyzer = CrossSectionAnalyzer(
                factors=single_factor_df,
                forward_returns=forward_returns_df,
                tradable_mask=tradable_mask,
                market_cap=market_cap,
                industry=industry
            )
            
            # 预处理（使用你的预处理管道）
            if preprocess_config.get('winsorize') or preprocess_config.get('standardize'):
                analyzer.preprocess(
                    winsorize=preprocess_config.get('winsorize', True),
                    standardize=preprocess_config.get('standardize') is not None,
                    neutralize=preprocess_config.get('neutralize') is not None,
                    winsorize_limits=preprocess_config.get('winsorize_limits', (0.01, 0.99)),
                    standardize_method=preprocess_config.get('standardize', 'zscore'),
                    neutralize_factors=preprocess_config.get('neutralize')
                )
            
            # 运行完整分析
            analyzer.analyze(
                n_quantiles=5,
                ic_method='spearman',
                spread_method='top_minus_mean',  # 实盘更稳健
                periods_per_year=252
            )
            
            # 获取结果
            results = analyzer.get_results()
            
            # 提取关键指标（key为(factor_name, 'ret_5d')）
            key_5d = (factor_name, 'ret_5d')
            ic_summary = results['ic_summary'][key_5d]
            spread_summary = results['spread_summary'][key_5d]
            monotonicity = results['monotonicity'][key_5d]
            
            # 判断是否通过（横截面评估的核心指标）
            pass_ic = ic_summary['mean'] >= IC_THRESHOLD and ic_summary['p_value'] < 0.05
            pass_icir = ic_summary['icir_annual'] >= ICIR_THRESHOLD
            pass_spread = spread_summary['mean'] > SPREAD_THRESHOLD
            pass_mono = monotonicity['kendall_tau'] > 0 and monotonicity['p_value'] < 0.05
            
            # 使用factor_quality_checker进行补充检查（IC半衰期、PSI/KS、相关性）
            quality_checker = FactorQualityChecker(
                ic_threshold=IC_THRESHOLD,
                icir_threshold=ICIR_THRESHOLD,
                psi_threshold=0.25,
                corr_threshold=CORR_THRESHOLD
            )
            
            # 提取5日远期收益
            forward_return_5d = forward_returns_df['ret_5d']
            
            # 补充质量检查
            extra_checks = quality_checker.comprehensive_check(
                factor_values=single_factor_df[factor_name],
                target_values=forward_return_5d,
                prices=prices_df if 'close' in features_df.columns else None,
                existing_factors=all_factors_df[qualified_factors] if qualified_factors else None,
                train_ratio=0.8
            )
            
            # 综合判断（横截面指标 + 补充检查）
            pass_corr = extra_checks['corr_check']['pass_corr']
            pass_psi = extra_checks['pass_psi']
            
            # 核心指标必须通过，补充指标可降权
            overall_pass = pass_ic and pass_icir and pass_spread and pass_corr
            
            # 保存报告
            quality_reports[factor_name] = {
                'ic_mean': ic_summary['mean'],
                'icir_annual': ic_summary['icir_annual'],
                'ic_pvalue': ic_summary['p_value'],
                'spread': spread_summary['mean'],
                'monotonicity_tau': monotonicity['kendall_tau'],
                'max_correlation': extra_checks['corr_check']['max_corr'],
                'ic_half_life': extra_checks.get('ic_half_life', np.nan),
                'psi': extra_checks.get('psi', np.nan),
                'pass_ic': pass_ic,
                'pass_icir': pass_icir,
                'pass_spread': pass_spread,
                'pass_correlation': pass_corr,
                'pass_psi': pass_psi,
                'overall_pass': overall_pass,
                'full_results': results,  # 横截面完整结果
                'extra_checks': extra_checks  # 补充检查结果
            }
            
            if overall_pass:
                qualified_factors.append(factor_name)
                print(f"   ✅ 通过")
                print(f"      IC={ic_summary['mean']:.4f} (ICIR={ic_summary['icir_annual']:.2f})")
                print(f"      Spread={spread_summary['mean']:.4f}, τ={monotonicity['kendall_tau']:.3f}")
            else:
                fail_reasons = []
                if not pass_ic: fail_reasons.append("IC不显著")
                if not pass_icir: fail_reasons.append("ICIR过低")
                if not pass_spread: fail_reasons.append("Spread≤0")
                if not pass_corr: fail_reasons.append("与已有因子高度相关")
                
                print(f"   ❌ 拒绝 | {', '.join(fail_reasons)}")
        
        except Exception as e:
            print(f"   ⚠️  评估失败: {str(e)}")
            quality_reports[factor_name] = {
                'overall_pass': False,
                'error': str(e)
            }
        
        print()
    
    print(f"✅ 横截面评估完成")
    print(f"   通过因子数: {len(qualified_factors)} / {all_factors_df.shape[1]}")
    print(f"   通过率: {len(qualified_factors) / all_factors_df.shape[1] * 100:.1f}%")
    
    # 5. 因子入库
    print("\n" + "=" * 80)
    print("步骤 5: 因子入库管理")
    print("=" * 80)
    
    manager = FactorLibraryManager()
    
    # 添加通过的因子
    print(f"\n📥 将 {len(qualified_factors)} 个通过的因子加入库中...\n")
    
    factor_registry = factory.get_factor_registry()
    
    for factor_name in qualified_factors:
        factor_info = factor_registry.get(factor_name, {})
        quality_report = quality_reports[factor_name]
        
        manager.add_factor(
            factor_name=factor_name,
            quality_report=quality_report,
            formula=factor_info.get('formula', ''),
            family=factor_info.get('family', ''),
            reference=factor_info.get('reference', '')
        )
    
    print(f"\n✅ 因子入库完成")
    
    # 6. 生成完整报告（使用你的tearsheet！）
    print("\n" + "=" * 80)
    print("步骤 6: 生成Tearsheet报告")
    print("=" * 80)
    
    # 为每个通过的因子生成完整的tearsheet报告
    reports_dir = os.path.join(ml_root, "ML output/reports/baseline_v1/factors")
    figures_dir = os.path.join(ml_root, "ML output/figures/baseline_v1/factors")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"\n📝 生成 {len(qualified_factors)} 个因子的详细报告...\n")
    
    for i, factor_name in enumerate(qualified_factors, 1):
        print(f"[{i}/{len(qualified_factors)}] 生成报告: {factor_name}")
        
        try:
            report = quality_reports[factor_name]
            full_results = report['full_results']
            
            # 生成HTML tearsheet（使用你的evaluation模块！）
            tearsheet_path = os.path.join(reports_dir, f"tearsheet_{factor_name}_5d.html")
            
            # 使用正确的tearsheet函数
            from evaluation.tearsheet import generate_html_tearsheet
            generate_html_tearsheet(
                analyzer_results=full_results,
                factor_name=factor_name,
                return_period='ret_5d',
                output_path=tearsheet_path,
                plot_paths=None  # 图表会自动生成在figures目录
            )
            
            # 保存IC时间序列CSV
            ic_series_path = os.path.join(reports_dir, f"ic_{factor_name}_5d.csv")
            full_results['ic_series'][5].to_csv(ic_series_path)
            
            # 保存分位数收益CSV
            quantile_returns_path = os.path.join(reports_dir, f"quantile_returns_{factor_name}_5d.csv")
            full_results['quantile_returns'][5].to_csv(quantile_returns_path)
            
            print(f"   ✅ 报告生成完成")
            print(f"      HTML: {tearsheet_path}")
        
        except Exception as e:
            print(f"   ⚠️  报告生成失败: {str(e)}")
        
        print()
    
    # 因子清单报告
    print(f"\n📊 生成因子库汇总报告...")
    report_df = manager.generate_factor_report()
    
    # 族别表现报告
    family_df = manager.analyze_factor_family_performance()
    
    print(f"\n族别表现汇总:")
    print(family_df.to_string(index=False))
    
    # 保存通过的因子数据
    print(f"\n💾 保存因子数据...")
    
    qualified_factors_df = all_factors_df[qualified_factors]
    
    # 保存路径
    datasets_dir = os.path.join(ml_root, "ML output/datasets/baseline_v1")
    os.makedirs(datasets_dir, exist_ok=True)
    
    # 保存Parquet格式
    output_path = os.path.join(datasets_dir, f"qualified_factors_{datetime.now().strftime('%Y%m%d')}.parquet")
    qualified_factors_df.to_parquet(output_path)
    print(f"   ✅ 因子数据 (Parquet): {output_path}")
    
    # 同时保存CSV格式（兼容性）
    csv_path = os.path.join(datasets_dir, f"qualified_factors_{datetime.now().strftime('%Y%m%d')}.csv")
    qualified_factors_df.to_csv(csv_path)
    print(f"   ✅ 因子数据 (CSV): {csv_path}")
    
    # 保存final_feature_list.txt
    feature_list_path = os.path.join(ml_root, "ML output/final_feature_list.txt")
    with open(feature_list_path, 'w', encoding='utf-8') as f:
        f.write("# 因子工厂 v1 - 合格因子清单\n")
        f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 数据区间: {start_date} ~ {end_date}\n")
        f.write(f"# 通过因子数: {len(qualified_factors)}\n")
        f.write("\n")
        for factor_name in qualified_factors:
            report = quality_reports[factor_name]
            f.write(f"{factor_name}\t")
            f.write(f"IC={report['ic_mean']:.4f}\t")
            f.write(f"ICIR={report['icir_annual']:.2f}\t")
            f.write(f"Spread={report['spread']:.4f}\n")
    
    print(f"   ✅ 因子清单: {feature_list_path}")
    
    # 7. 验收检查
    print("\n" + "=" * 80)
    print("验收检查")
    print("=" * 80)
    
    acceptance_passed = True
    
    # 检查1: 至少10个稳定因子过检
    print(f"\n✓ 检查1: 稳定因子数量")
    print(f"   要求: ≥10 个")
    print(f"   实际: {len(qualified_factors)} 个")
    
    if len(qualified_factors) < 10:
        print(f"   ❌ 未通过")
        acceptance_passed = False
    else:
        print(f"   ✅ 通过")
    
    # 检查2: 横截面 Rank IC 显著
    print(f"\n✓ 检查2: Rank IC 显著性")
    
    significant_factors = []
    for factor_name in qualified_factors:
        report = quality_reports[factor_name]
        if report['ic_metrics']['pass_ic']:
            significant_factors.append(factor_name)
    
    print(f"   要求: IC > 0.02 且统计显著")
    print(f"   实际: {len(significant_factors)} / {len(qualified_factors)} 个因子显著")
    
    if len(significant_factors) < len(qualified_factors) * 0.8:
        print(f"   ❌ 未通过 (显著因子比例过低)")
        acceptance_passed = False
    else:
        print(f"   ✅ 通过")
    
    # 检查3: 合入后组合 IC 有实质提升
    print(f"\n✓ 检查3: 组合 IC 提升")
    print(f"   基准特征数: {features_df.shape[1]}")
    print(f"   新增因子数: {len(qualified_factors)}")
    
    # 简单组合测试：所有因子等权平均
    combined_factor = qualified_factors_df.mean(axis=1)
    target_values = targets_df[target_config['type']]
    
    # 计算组合IC
    aligned_df = pd.DataFrame({
        'factor': combined_factor,
        'target': target_values
    }).dropna()
    
    grouped = aligned_df.groupby(level='date')
    ic_series = grouped.apply(lambda x: x['factor'].corr(x['target'], method='spearman'))
    
    combined_ic = ic_series.mean()
    combined_icir = ic_series.mean() / ic_series.std() * np.sqrt(252)
    
    print(f"   组合IC: {combined_ic:.4f}")
    print(f"   组合ICIR: {combined_icir:.2f}")
    
    if combined_ic < 0.03:
        print(f"   ⚠️  组合IC偏低，建议进一步优化")
    else:
        print(f"   ✅ 组合IC显著")
    
    # 最终验收结果
    print("\n" + "=" * 80)
    if acceptance_passed:
        print("🎉 验收通过！因子工厂 v1 构建成功")
    else:
        print("⚠️  部分验收指标未达标，需要进一步优化")
    print("=" * 80)
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return qualified_factors_df, quality_reports, manager


if __name__ == "__main__":
    """运行因子准备流程"""
    
    # 测试参数
    tickers = ['000001.SZ', '000002.SZ', '000063.SZ']  # 测试用股票
    
    qualified_factors_df, quality_reports, manager = prepare_factors(
        config_path="configs/ml_baseline.yml",
        start_date="2020-01-01",
        end_date="2024-12-31",
        tickers=tickers
    )
    
    print("\n✅ 流程执行完成！")
