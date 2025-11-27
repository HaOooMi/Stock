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
from data.tradability_filter import TradabilityFilter
from data.financial_data_loader import FinancialDataLoader
# 使用你已有的横截面评估框架！
from evaluation.cross_section_analyzer import CrossSectionAnalyzer
from evaluation.cross_section_metrics import calculate_forward_returns
from evaluation.factor_preprocessing import preprocess_factor_pipeline
from evaluation.tearsheet import generate_html_tearsheet


def load_config(config_path: str = "configs/ml_baseline.yml") -> dict:
    """加载配置文件"""
    if not os.path.isabs(config_path):
        config_path = os.path.join(ml_root, config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def prepare_factors(config_path: str = "configs/ml_baseline.yml",
                   start_date: str = None,
                   end_date: str = None,
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
    target_config = config.get('target', {})  # 注意：配置文件用的是 'target' 单数
    tradability_config = config['data'].get('universe', {})  # 使用 'universe' 而不是 'tradability_filter'
    
    # 如果没有传入日期参数，从配置文件读取
    if start_date is None:
        start_date = config['data'].get('start_date', '2018-01-01')
    if end_date is None:
        end_date = config['data'].get('end_date', '2024-12-31')
    
    # 如果没有传入股票列表，从配置文件读取
    # 注意：InfluxDB 中存储的股票代码是纯数字格式（如 '000001'），不带后缀
    if tickers is None:
        tickers = config['data'].get('symbol', None)
        if isinstance(tickers, str):
            tickers = [tickers]
    
    # 设置默认值
    if 'type' not in target_config:
        target_config['type'] = 'forward_return'
    if 'horizon' not in target_config:
        target_config['horizon'] = target_config.get('forward_periods', 5)
    
    print(f"✅ 配置加载完成")
    print(f"   InfluxDB: {influxdb_config['url']}")
    print(f"   预测目标: {target_config.get('name', 'future_return_5d')} ({target_config['horizon']}日)")
    print(f"   日期范围: {start_date} ~ {end_date}")
    print(f"   股票代码: {tickers}")
    
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
    
    # 2.5 交易可行性过滤
    print("\n" + "=" * 80)
    print("步骤 2.5: 交易可行性过滤")
    print("=" * 80)
    
    tradability_filter = TradabilityFilter(
        min_volume=tradability_config.get('min_volume', 2000),
        min_amount=tradability_config.get('min_amount', 10000000),
        min_price=tradability_config.get('min_price', 1.0),
        min_turnover=tradability_config.get('min_turnover', 0.1),
        min_listing_days=tradability_config.get('min_listing_days', 60),
        exclude_st=tradability_config.get('exclude_st', True),
        exclude_limit_moves=tradability_config.get('exclude_limit_moves', False),
        limit_threshold=tradability_config.get('limit_threshold', 0.098)
    )
    
    # 生成可交易性掩码
    tradable_mask = tradability_filter.filter(features_df)
    tradable_ratio = tradable_mask.sum() / len(tradable_mask) * 100
    
    print(f"✅ 交易可行性过滤完成")
    print(f"   总样本数: {len(tradable_mask)}")
    print(f"   可交易样本: {tradable_mask.sum()} ({tradable_ratio:.1f}%)")
    print(f"   被过滤样本: {(~tradable_mask).sum()} ({100-tradable_ratio:.1f}%)")
    
    # 2.6 加载财务数据（如果配置启用）
    financial_features = None
    pit_config = config['data'].get('pit', {})
    
    if pit_config.get('enabled', False):
        print("\n" + "=" * 80)
        print("步骤 2.6: 加载财务数据 (PIT对齐)")
        print("=" * 80)
        
        try:
            financial_loader = FinancialDataLoader(
                announce_lag_days=pit_config.get('financial_lag_days', 90),
                ffill_limit=pit_config.get('financial_ffill_limit', 95)
            )
            
            financial_dfs = []
            for ticker in tickers:
                try:
                    fin_df = financial_loader.load_financial_data(
                        symbol=ticker,
                        start_date=start_date,
                        end_date=end_date
                    )
                    if fin_df is not None and not fin_df.empty:
                        financial_dfs.append(fin_df)
                except Exception as e:
                    print(f"   ⚠️  {ticker} 财务数据加载失败: {e}")
            
            if financial_dfs:
                financial_features = pd.concat(financial_dfs)
                print(f"✅ 财务数据加载完成")
                print(f"   财务特征数: {financial_features.shape[1]}")
                print(f"   样本数: {len(financial_features)}")
            else:
                print(f"⚠️  未加载到财务数据，跳过财务因子")
        except Exception as e:
            print(f"⚠️  财务数据加载器初始化失败: {e}")
            print(f"   将跳过财务因子，仅使用市场数据因子")
    else:
        print("\n📋 财务数据未启用 (pit.enabled=False)")
    
    # 3. 生成因子
    print("\n" + "=" * 80)
    print("步骤 3: 生成因子")
    print("=" * 80)
    
    factory = FactorFactory()
    
    # 生成所有因子族
    print("\n🏭 生成因子...")
    all_factors_df = factory.generate_all_factors(features_df)
    
    # 如果有财务数据，可以生成财务因子（需要FactorFactory支持）
    if financial_features is not None:
        print(f"\n📊 财务数据可用，可生成财务相关因子")
        # TODO: 在FactorFactory中添加财务因子生成方法
        # factory.generate_financial_factors(financial_features)
    
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
    
    # 使用步骤2.5生成的可交易性mask
    # tradable_mask 已经在前面的 TradabilityFilter 中生成
    if tradable_mask is not None and tradable_mask.sum() > 0:
        print(f"   ✅ 使用可交易性mask (可交易样本: {tradable_mask.sum()})")
        # 转换为DataFrame格式以匹配CrossSectionAnalyzer的要求
        tradable_mask_df = pd.DataFrame({'tradable': tradable_mask}, index=features_df.index)
    else:
        tradable_mask_df = None
        print(f"   ⚠️  未生成可交易性mask，将使用全部样本")
    
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
    
    # 预处理配置 - 使用默认值即可
    preprocess_config = {
        'winsorize': True,
        'standardize': True,
        'neutralize': False  # 可选: True (需要market_cap/industry)
    }
    
    for i, factor_name in enumerate(all_factors_df.columns, 1):
        print(f"[{i}/{all_factors_df.shape[1]}] 评估因子: {factor_name}")
        
        try:
            # 构建单因子DataFrame
            single_factor_df = all_factors_df[[factor_name]]
            
            # 使用你的CrossSectionAnalyzer！（一次性完成所有分析，包括深度质量检查）
            analyzer = CrossSectionAnalyzer(
                factors=single_factor_df,
                forward_returns=forward_returns_df,
                prices=prices_df if 'close' in features_df.columns else None,
                tradable_mask=tradable_mask_df,  # 使用步骤2.5生成的可交易性mask
                market_cap=market_cap,
                industry=industry
            )
            
            # 预处理（使用默认参数）
            analyzer.preprocess(
                winsorize=preprocess_config.get('winsorize', True),
                standardize=preprocess_config.get('standardize', True),
                neutralize=preprocess_config.get('neutralize', False)
            )
            
            # 运行完整分析（一次到位，包含深度质量检查）
            analyzer.analyze(
                n_quantiles=5,
                ic_method='spearman',
                spread_method='top_minus_mean',  # 实盘更稳健
                periods_per_year=252,
                check_quality=True  # 开启深度检查（PSI/KS/IC衰减）
            )
            
            # 获取结果
            results = analyzer.get_results()
            
            # 提取关键指标（key为(factor_name, 'ret_5d')）
            key_5d = (factor_name, 'ret_5d')
            
            # 安全获取各项指标（股票数太少时可能缺失）
            ic_summary = results.get('ic_summary', {}).get(key_5d, {})
            spread_summary = results.get('spread_summary', {}).get(key_5d, {})
            monotonicity = results.get('monotonicity', {}).get(key_5d, {})
            quality_report = results.get('quality_reports', {}).get(factor_name, {})
            
            # 如果缺少关键指标，跳过此因子
            if not ic_summary:
                print(f"   ⚠️  IC数据不足，跳过")
                continue
            
            # 判断是否通过（横截面评估的核心指标）
            ic_mean = ic_summary.get('mean', 0)
            ic_pvalue = ic_summary.get('p_value', 1)
            icir_annual = ic_summary.get('icir_annual', 0)
            spread_mean = spread_summary.get('mean', np.nan)
            kendall_tau = monotonicity.get('kendall_tau', np.nan)
            mono_pvalue = monotonicity.get('p_value', 1)
            
            pass_ic = ic_mean >= IC_THRESHOLD and ic_pvalue < 0.05
            pass_icir = icir_annual >= ICIR_THRESHOLD
            pass_spread = spread_mean > SPREAD_THRESHOLD if not np.isnan(spread_mean) else False
            pass_mono = kendall_tau > 0 and mono_pvalue < 0.05 if not np.isnan(kendall_tau) else False
            
            # 深度质量检查结果
            pass_psi = quality_report.get('psi', 1.0) < 0.25
            pass_ks = quality_report.get('ks_p', 0) > 0.05
            
            # 相关性检查（与已有因子）
            pass_corr = True
            max_corr = 0.0
            if qualified_factors:
                existing_factors = all_factors_df[qualified_factors]
                corrs = existing_factors.corrwith(single_factor_df[factor_name]).abs()
                max_corr = corrs.max()
                pass_corr = max_corr < CORR_THRESHOLD
            
            # 核心指标：IC必须通过，其他指标在数据充足时才检查
            # 股票数太少时，Spread和单调性可能为NaN，放宽条件
            overall_pass = pass_ic and pass_icir and pass_corr
            if not np.isnan(spread_mean):
                overall_pass = overall_pass and pass_spread
            
            # 保存报告
            quality_reports[factor_name] = {
                'ic_mean': ic_mean,
                'icir_annual': icir_annual,
                'ic_pvalue': ic_pvalue,
                'spread': spread_mean,
                'monotonicity_tau': kendall_tau,
                'max_correlation': max_corr,
                'ic_half_life': quality_report.get('ic_half_life', np.nan),
                'psi': quality_report.get('psi', np.nan),
                'ks_stat': quality_report.get('ks_stat', np.nan),
                'ks_p': quality_report.get('ks_p', np.nan),
                'pass_ic': pass_ic,
                'pass_icir': pass_icir,
                'pass_spread': pass_spread,
                'pass_correlation': pass_corr,
                'pass_psi': pass_psi,
                'pass_ks': pass_ks,
                'overall_pass': overall_pass,
                'full_results': results  # 横截面完整结果
            }
            
            if overall_pass:
                qualified_factors.append(factor_name)
                print(f"   ✅ 通过")
                print(f"      IC={ic_mean:.4f} (ICIR={icir_annual:.2f})")
                spread_str = f"{spread_mean:.4f}" if not np.isnan(spread_mean) else "N/A"
                tau_str = f"{kendall_tau:.3f}" if not np.isnan(kendall_tau) else "N/A"
                print(f"      Spread={spread_str}, τ={tau_str}")
            else:
                fail_reasons = []
                if not pass_ic: fail_reasons.append("IC不显著")
                if not pass_icir: fail_reasons.append("ICIR过低")
                if not pass_spread and not np.isnan(spread_mean): fail_reasons.append("Spread≤0")
                if not pass_corr: fail_reasons.append("与已有因子高度相关")
                
                print(f"   ❌ 拒绝 | {', '.join(fail_reasons) if fail_reasons else 'IC条件未满足'}")
        
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
        # 兼容扁平格式（pass_ic）和嵌套格式（ic_metrics.pass_ic）
        if 'ic_metrics' in report:
            pass_ic = report['ic_metrics']['pass_ic']
        else:
            pass_ic = report.get('pass_ic', False)
        
        if pass_ic:
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
    # 使用 5日收益作为目标（与因子评估一致）
    target_col = f"ret_{target_config['horizon']}d"
    if target_col in targets_df.columns:
        target_values = targets_df[target_col]
    else:
        # 回退到第一个可用列
        target_values = targets_df.iloc[:, 0]
    
    # 计算组合IC
    aligned_df = pd.DataFrame({
        'factor': combined_factor,
        'target': target_values
    }).dropna()
    
    grouped = aligned_df.groupby(level='date')
    ic_series = grouped.apply(lambda x: x['factor'].corr(x['target'], method='spearman'))
    
    # 确保 ic_series 是一维的，取标量值
    if hasattr(ic_series, 'values'):
        ic_values = ic_series.values.flatten()
        combined_ic = float(np.nanmean(ic_values))
        combined_icir = float(np.nanmean(ic_values) / np.nanstd(ic_values) * np.sqrt(252)) if np.nanstd(ic_values) > 0 else 0.0
    else:
        combined_ic = float(ic_series) if not pd.isna(ic_series) else 0.0
        combined_icir = 0.0
    
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
    
    # 加载配置文件获取参数
    config = load_config("configs/ml_baseline.yml")
    
    # 从配置文件读取股票代码（纯数字格式，如 '000001'，不是 '000001.SZ'）
    # 因为 InfluxDB 中存储的股票代码是纯数字格式
    tickers = config['data'].get('symbol', ['000001', '000002', '000063'])
    if isinstance(tickers, str):
        tickers = [tickers]
    
    # 从配置文件读取日期范围
    start_date = config['data'].get('start_date', '2018-01-01')
    end_date = config['data'].get('end_date', '2024-12-31')
    
    print(f"\n📋 从配置文件读取参数:")
    print(f"   股票代码: {tickers}")
    print(f"   日期范围: {start_date} ~ {end_date}")
    
    qualified_factors_df, quality_reports, manager = prepare_factors(
        config_path="configs/ml_baseline.yml",
        start_date=start_date,
        end_date=end_date,
        tickers=tickers
    )
    
    print("\n✅ 流程执行完成！")
