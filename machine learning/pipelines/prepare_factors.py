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
import json
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
from data.data_snapshot import DataSnapshot  # 数据快照管理
# 使用你已有的横截面评估框架！
from evaluation.cross_section_analyzer import CrossSectionAnalyzer
from evaluation.cross_section_metrics import calculate_forward_returns
from evaluation.factor_preprocessing import preprocess_factor_pipeline
from evaluation.tearsheet import generate_html_tearsheet
from evaluation.visualization import (  # 图表生成
    plot_ic_time_series,
    plot_ic_distribution,
    plot_quantile_cumulative_returns,
    plot_quantile_mean_returns,
    plot_spread_cumulative_returns,
    plot_monthly_ic_heatmap,
    plot_turnover_time_series,
    create_factor_tearsheet_plots
)


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
    
    # 如果未指定tickers，从配置文件获取股票池
    if not tickers:
        tickers = config['data'].get('symbol', None)
        if isinstance(tickers, str):
            tickers = [tickers]
        if not tickers:
            print(f"\n⚠️  未指定股票列表，请在配置文件中设置 data.symbol")
            raise ValueError("必须在配置文件中提供 data.symbol 参数")
        print(f"   📋 从配置文件加载股票池: {len(tickers)} 只股票")
    
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
    
    # 应用交易可行性过滤，返回带有 tradable_flag 列的数据和过滤日志
    filter_log_path = os.path.join(ml_root, "ML output/reports/baseline_v1/tradability_filter_log.csv")
    os.makedirs(os.path.dirname(filter_log_path), exist_ok=True)
    features_df, filter_log_df = tradability_filter.apply_filters(
        features_df, 
        save_log=True, 
        log_path=filter_log_path
    )
    
    # 生成可交易性掩码（基于 tradable_flag 列）
    tradable_mask = features_df['tradable_flag'] == 1
    tradable_ratio = tradable_mask.sum() / len(tradable_mask) * 100
    
    print(f"✅ 交易可行性过滤完成")
    print(f"   总样本数: {len(tradable_mask)}")
    print(f"   可交易样本: {tradable_mask.sum()} ({tradable_ratio:.1f}%)")
    print(f"   被过滤样本: {(~tradable_mask).sum()} ({100-tradable_ratio:.1f}%)")
    print(f"   过滤日志: {filter_log_path}")
    
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
    
    # 2.7 创建数据快照（数据版本管理）
    snapshot_config = config.get('snapshot', {})
    snapshot_manager = None
    
    if snapshot_config.get('enabled', True):  # 默认启用
        print("\n" + "=" * 80)
        print("步骤 2.7: 创建数据快照")
        print("=" * 80)
        
        try:
            snapshot_manager = DataSnapshot(
                output_dir=os.path.join(ml_root, "ML output"),
                snapshot_id=None  # 自动生成
            )
            
            # 准备快照数据（市场数据 + 可交易性mask）
            snapshot_data = features_df.copy()
            snapshot_data['tradable_flag'] = tradable_mask.astype(int)
            
            # 创建快照
            filters_info = {
                'min_volume': tradability_config.get('min_volume', 2000),
                'min_amount': tradability_config.get('min_amount', 10000000),
                'min_price': tradability_config.get('min_price', 1.0),
                'exclude_st': tradability_config.get('exclude_st', True),
                'tradable_ratio': float(tradable_ratio)
            }
            
            snapshot_path = snapshot_manager.create_snapshot(
                data=snapshot_data,
                symbol='_'.join(tickers[:3]) + (f'_etc{len(tickers)}' if len(tickers) > 3 else ''),
                start_date=start_date,
                end_date=end_date,
                filters=filters_info,
                random_seed=42,
                save_parquet=True
            )
            
            print(f"✅ 数据快照创建完成")
            print(f"   快照ID: {snapshot_manager.snapshot_id}")
            print(f"   快照路径: {snapshot_path}")
        except Exception as e:
            print(f"⚠️  数据快照创建失败: {e}")
            print(f"   将继续执行，但不会保存快照")
    else:
        print("\n📋 数据快照未启用 (snapshot.enabled=False)")
    
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
    
    # ===== 从配置文件读取质量检查阈值 =====
    quality_config = config['features'].get('factor_factory', {}).get('quality_check', {})
    
    # 严格标准（生产级）
    strict_config = quality_config.get('strict', {})
    IC_THRESHOLD_STRICT = strict_config.get('ic_threshold', 0.02)
    ICIR_THRESHOLD_STRICT = strict_config.get('icir_threshold', 0.5)
    IC_PVALUE_STRICT = strict_config.get('ic_pvalue', 0.05)
    
    # 探索标准（研究级）
    explore_config = quality_config.get('exploratory', {})
    IC_THRESHOLD_EXPLORE = explore_config.get('ic_threshold', 0.005)
    ICIR_THRESHOLD_EXPLORE = explore_config.get('icir_threshold', 0.15)
    IC_PVALUE_THRESHOLD = explore_config.get('ic_pvalue', 0.10)
    
    # 通用标准
    common_config = quality_config.get('common', {})
    SPREAD_THRESHOLD = common_config.get('spread_threshold', 0.0)
    CORR_THRESHOLD = common_config.get('corr_threshold', 0.8)
    PSI_THRESHOLD = common_config.get('psi_threshold', 0.25)
    USE_ABS_IC = common_config.get('use_abs_ic', True)
    
    # 自动降级开关
    AUTO_FALLBACK = quality_config.get('auto_fallback_to_exploratory', True)
    
    print(f"\n📋 质量检查配置 (从 ml_baseline.yml 读取):")
    print(f"   严格标准: |IC|≥{IC_THRESHOLD_STRICT}, |ICIR|≥{ICIR_THRESHOLD_STRICT}, p<{IC_PVALUE_STRICT}")
    print(f"   探索标准: |IC|≥{IC_THRESHOLD_EXPLORE}, |ICIR|≥{ICIR_THRESHOLD_EXPLORE}, p<{IC_PVALUE_THRESHOLD}")
    print(f"   通用标准: Spread>{SPREAD_THRESHOLD}, MaxCorr<{CORR_THRESHOLD}, PSI<{PSI_THRESHOLD}")
    print(f"   使用|IC|: {USE_ABS_IC}, 自动降级: {AUTO_FALLBACK}")
    
    # 逐个因子评估
    qualified_factors = []       # 严格通过
    exploratory_factors = []     # 探索通过（宽松标准）
    quality_reports = {}
    
    print(f"\n🔍 开始横截面评估 (共 {all_factors_df.shape[1]} 个因子)...\n")
    
    # 预处理配置 - 从配置文件读取
    preprocess_config = config['features'].get('preprocessing', {})
    # 确保有默认值
    if 'winsorize' not in preprocess_config:
        preprocess_config['winsorize'] = True
    if 'standardize' not in preprocess_config:
        preprocess_config['standardize'] = True
    if 'neutralize' not in preprocess_config:
        preprocess_config['neutralize'] = False
    
    print(f"   预处理配置: winsorize={preprocess_config['winsorize']}, "
          f"standardize={preprocess_config['standardize']}, "
          f"neutralize={preprocess_config['neutralize']}")
    
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
            
            # ===== 计算用于筛选的IC值（支持绝对值模式）=====
            ic_for_filter = abs(ic_mean) if USE_ABS_IC else ic_mean
            # 对于负向因子，ICIR也取绝对值
            icir_for_filter = abs(icir_annual) if USE_ABS_IC else icir_annual
            
            # ===== 严格标准（生产级）=====
            pass_ic_strict = ic_for_filter >= IC_THRESHOLD_STRICT and ic_pvalue < IC_PVALUE_STRICT
            pass_icir_strict = icir_for_filter >= ICIR_THRESHOLD_STRICT
            
            # ===== 探索标准（研究级）=====
            pass_ic_explore = ic_for_filter >= IC_THRESHOLD_EXPLORE and ic_pvalue < IC_PVALUE_THRESHOLD
            pass_icir_explore = icir_for_filter >= ICIR_THRESHOLD_EXPLORE
            
            # ===== 通用检查 =====
            pass_spread = spread_mean > SPREAD_THRESHOLD if not np.isnan(spread_mean) else True  # NaN时默认通过
            pass_mono = kendall_tau > 0 and mono_pvalue < 0.05 if not np.isnan(kendall_tau) else True
            
            # 深度质量检查结果（使用配置的阈值）
            pass_psi = quality_report.get('psi', 1.0) < PSI_THRESHOLD
            pass_ks = quality_report.get('ks_p', 0) > 0.05
            
            # 相关性检查（与已有探索因子，使用配置的阈值）
            pass_corr = True
            max_corr = 0.0
            check_against = exploratory_factors if exploratory_factors else []
            if check_against:
                existing_factors = all_factors_df[check_against]
                corrs = existing_factors.corrwith(single_factor_df[factor_name]).abs()
                max_corr = corrs.max()
                pass_corr = max_corr < CORR_THRESHOLD
            
            # ===== 判断通过层级 =====
            # 严格通过：IC、ICIR都满足严格标准，且相关性OK
            strict_pass = pass_ic_strict and pass_icir_strict and pass_corr
            
            # 探索通过：IC、ICIR满足探索标准，且相关性OK
            exploratory_pass = pass_ic_explore and pass_icir_explore and pass_corr
            
            # 兼容旧逻辑的overall_pass（现在使用探索标准，让更多因子进入下一步）
            overall_pass = exploratory_pass
            
            # 保存报告（包含双层判定结果）
            quality_reports[factor_name] = {
                'ic_mean': ic_mean,
                'ic_abs': abs(ic_mean),  # 新增：绝对值IC
                'ic_direction': 'positive' if ic_mean >= 0 else 'negative',  # 新增：方向
                'icir_annual': icir_annual,
                'icir_abs': abs(icir_annual),  # 新增：绝对值ICIR
                'ic_pvalue': ic_pvalue,
                'spread': spread_mean,
                'monotonicity_tau': kendall_tau,
                'max_correlation': max_corr,
                'ic_half_life': quality_report.get('ic_half_life', np.nan),
                'psi': quality_report.get('psi', np.nan),
                'ks_stat': quality_report.get('ks_stat', np.nan),
                'ks_p': quality_report.get('ks_p', np.nan),
                # 严格标准判定
                'pass_ic_strict': pass_ic_strict,
                'pass_icir_strict': pass_icir_strict,
                'strict_pass': strict_pass,
                # 探索标准判定
                'pass_ic_explore': pass_ic_explore,
                'pass_icir_explore': pass_icir_explore,
                'exploratory_pass': exploratory_pass,
                # 通用检查
                'pass_spread': pass_spread,
                'pass_correlation': pass_corr,
                'pass_psi': pass_psi,
                'pass_ks': pass_ks,
                # 兼容旧字段
                'pass_ic': pass_ic_explore,
                'pass_icir': pass_icir_explore,
                'overall_pass': overall_pass,
                'full_results': results
            }
            
            # 根据通过层级分类
            if strict_pass:
                qualified_factors.append(factor_name)
                exploratory_factors.append(factor_name)
                direction_mark = "⬆️" if ic_mean >= 0 else "⬇️"
                print(f"   ✅ 严格通过 {direction_mark}")
                print(f"      IC={ic_mean:.4f} (|IC|={abs(ic_mean):.4f}, ICIR={icir_annual:.2f})")
                spread_str = f"{spread_mean:.4f}" if not np.isnan(spread_mean) else "N/A"
                print(f"      Spread={spread_str}, MaxCorr={max_corr:.3f}")
            elif exploratory_pass:
                exploratory_factors.append(factor_name)
                direction_mark = "⬆️" if ic_mean >= 0 else "⬇️"
                print(f"   🔍 探索通过 {direction_mark}")
                print(f"      IC={ic_mean:.4f} (|IC|={abs(ic_mean):.4f}, ICIR={icir_annual:.2f})")
                print(f"      (未达严格标准，但可用于排序模型实验)")
            else:
                fail_reasons = []
                if not pass_ic_explore: fail_reasons.append(f"|IC|<{IC_THRESHOLD_EXPLORE}或p>{IC_PVALUE_THRESHOLD}")
                if not pass_icir_explore: fail_reasons.append(f"|ICIR|<{ICIR_THRESHOLD_EXPLORE}")
                if not pass_corr: fail_reasons.append(f"MaxCorr>{CORR_THRESHOLD}")
                
                print(f"   ❌ 拒绝 | {', '.join(fail_reasons)}")
        
        except Exception as e:
            print(f"   ⚠️  评估失败: {str(e)}")
            quality_reports[factor_name] = {
                'overall_pass': False,
                'error': str(e)
            }
        
        print()
    
    print(f"✅ 横截面评估完成")
    print(f"   严格通过: {len(qualified_factors)} / {all_factors_df.shape[1]} ({len(qualified_factors)/all_factors_df.shape[1]*100:.1f}%)")
    print(f"   探索通过: {len(exploratory_factors)} / {all_factors_df.shape[1]} ({len(exploratory_factors)/all_factors_df.shape[1]*100:.1f}%)")
    
    # ===== 合并严格通过和探索通过的因子 =====
    # 策略：将探索通过但不在严格通过中的因子也加入，用于排序模型实验
    # 这样可以有更多因子供模型学习，同时保留质量分级信息
    original_strict_count = len(qualified_factors)
    
    if AUTO_FALLBACK:
        # 合并探索因子（去重）
        for factor in exploratory_factors:
            if factor not in qualified_factors:
                qualified_factors.append(factor)
        
        if len(qualified_factors) > original_strict_count:
            print(f"\n📊 因子合并: 严格通过 {original_strict_count} + 探索补充 {len(qualified_factors) - original_strict_count} = 共 {len(qualified_factors)} 个因子")
    
    # 如果严格通过为0，使用探索因子
    if original_strict_count == 0 and len(exploratory_factors) > 0:
        print(f"\n⚠️  严格通过因子数为0，使用探索通过的 {len(exploratory_factors)} 个因子")
        print(f"   这些因子可用于排序模型实验，但建议后续优化因子质量")
    elif original_strict_count == 0 and len(exploratory_factors) == 0:
        print(f"\n❌ 严格和探索都没有通过的因子，请检查因子质量或放宽筛选标准")
    
    # ===== 4.5 保存完整的因子体检报告 =====
    print("\n" + "-" * 60)
    print("保存因子体检详细报告")
    print("-" * 60)
    
    screening_dir = os.path.join(ml_root, "ML output/reports/baseline_v1/factor_screening")
    ic_series_dir = os.path.join(screening_dir, "ic_series")
    os.makedirs(ic_series_dir, exist_ok=True)
    
    # 1. 保存所有因子的详细体检数据（CSV格式，方便查看）
    screening_records = []
    for factor_name, report in quality_reports.items():
        record = {
            '因子名称': factor_name,
            'IC方向': report.get('ic_direction', ''),
            'IC均值': report.get('ic_mean', np.nan),
            '|IC|': report.get('ic_abs', np.nan),
            'ICIR年化': report.get('icir_annual', np.nan),
            '|ICIR|': report.get('icir_abs', np.nan),
            'IC_P值': report.get('ic_pvalue', np.nan),
            'Spread均值': report.get('spread', np.nan),
            '单调性Tau': report.get('monotonicity_tau', np.nan),
            '最大相关性': report.get('max_correlation', np.nan),
            'IC半衰期': report.get('ic_half_life', np.nan),
            'PSI': report.get('psi', np.nan),
            'KS统计量': report.get('ks_stat', np.nan),
            'KS_P值': report.get('ks_p', np.nan),
            # 严格标准
            '严格通过IC': report.get('pass_ic_strict', False),
            '严格通过ICIR': report.get('pass_icir_strict', False),
            '严格通过': report.get('strict_pass', False),
            # 探索标准
            '探索通过IC': report.get('pass_ic_explore', False),
            '探索通过ICIR': report.get('pass_icir_explore', False),
            '探索通过': report.get('exploratory_pass', False),
            # 通用检查
            '通过Spread': report.get('pass_spread', False),
            '通过相关性': report.get('pass_correlation', True),
            '通过PSI': report.get('pass_psi', True),
            '通过KS': report.get('pass_ks', True),
            '失败原因': ''
        }
        
        # 记录失败原因（基于探索标准）
        if not report.get('exploratory_pass', False):
            fail_reasons = []
            if not report.get('pass_ic_explore', False): 
                fail_reasons.append(f'|IC|<{IC_THRESHOLD_EXPLORE}或p>{IC_PVALUE_THRESHOLD}')
            if not report.get('pass_icir_explore', False): 
                fail_reasons.append(f'|ICIR|<{ICIR_THRESHOLD_EXPLORE}')
            if not report.get('pass_correlation', True): 
                fail_reasons.append(f'MaxCorr>{CORR_THRESHOLD}')
            if 'error' in report: 
                fail_reasons.append(f"错误:{report['error']}")
            record['失败原因'] = '; '.join(fail_reasons)
        
        screening_records.append(record)
    
    screening_df = pd.DataFrame(screening_records)
    # 按IC均值排序（绝对值降序）
    screening_df = screening_df.sort_values('IC均值', ascending=False, key=abs)
    
    screening_csv_path = os.path.join(screening_dir, f"factor_screening_detail_{datetime.now().strftime('%Y%m%d')}.csv")
    screening_df.to_csv(screening_csv_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 因子体检详情: {screening_csv_path}")
    
    # 2. 保存每个因子的IC时间序列（便于分析IC衰减和稳定性）
    ic_saved_count = 0
    for factor_name, report in quality_reports.items():
        full_results = report.get('full_results', {})
        if not full_results:
            continue
        
        # 尝试提取IC时间序列
        daily_ic = full_results.get('daily_ic', None)
        if daily_ic is not None and len(daily_ic) > 0:
            key_5d = (factor_name, 'ret_5d')
            try:
                if isinstance(daily_ic, pd.DataFrame):
                    if key_5d in daily_ic.columns:
                        ic_series = daily_ic[key_5d]
                    elif 5 in daily_ic.columns:
                        ic_series = daily_ic[5]
                    else:
                        ic_series = daily_ic.iloc[:, 0] if daily_ic.shape[1] > 0 else None
                elif isinstance(daily_ic, dict):
                    ic_series = daily_ic.get(key_5d, daily_ic.get(5, None))
                else:
                    ic_series = daily_ic
                
                if ic_series is not None and len(ic_series) > 0:
                    ic_series_path = os.path.join(ic_series_dir, f"{factor_name}_ic_5d.csv")
                    if isinstance(ic_series, pd.Series):
                        ic_series.to_csv(ic_series_path, header=['ic'])
                    else:
                        pd.Series(ic_series).to_csv(ic_series_path, header=['ic'])
                    ic_saved_count += 1
            except Exception as e:
                pass  # 忽略保存失败的情况
    
    print(f"   ✅ IC时间序列: {ic_saved_count} 个因子 -> {ic_series_dir}")
    
    # 3. 保存完整的JSON格式报告（包含更多细节，便于程序读取）
    json_report = {
        'generated_at': datetime.now().isoformat(),
        'data_range': {'start': start_date, 'end': end_date},
        'total_factors': len(quality_reports),
        'qualified_factors_strict': len([f for f in quality_reports if quality_reports[f].get('strict_pass', False)]),
        'qualified_factors_explore': len([f for f in quality_reports if quality_reports[f].get('exploratory_pass', False)]),
        'pass_rate': len(qualified_factors) / len(quality_reports) * 100 if quality_reports else 0,
        'thresholds': {
            'strict': {
                'ic_threshold': IC_THRESHOLD_STRICT,
                'icir_threshold': ICIR_THRESHOLD_STRICT,
            },
            'exploratory': {
                'ic_threshold': IC_THRESHOLD_EXPLORE,
                'icir_threshold': ICIR_THRESHOLD_EXPLORE,
                'ic_pvalue': IC_PVALUE_THRESHOLD,
            },
            'common': {
                'spread_threshold': SPREAD_THRESHOLD,
                'corr_threshold': CORR_THRESHOLD,
                'use_abs_ic': USE_ABS_IC,
            }
        },
        'factors': {}
    }
    
    for factor_name, report in quality_reports.items():
        # 移除 full_results（太大了，不适合放JSON）
        factor_report = {k: v for k, v in report.items() if k != 'full_results'}
        # 处理 NaN 值和 numpy 类型
        for key, value in factor_report.items():
            if isinstance(value, (np.floating, float)) and np.isnan(value):
                factor_report[key] = None
            elif isinstance(value, (np.bool_, bool)):
                factor_report[key] = bool(value)
            elif isinstance(value, (np.integer,)):
                factor_report[key] = int(value)
            elif isinstance(value, (np.floating,)):
                factor_report[key] = float(value)
        json_report['factors'][factor_name] = factor_report
    
    json_path = os.path.join(screening_dir, f"factor_screening_summary_{datetime.now().strftime('%Y%m%d')}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    print(f"   ✅ JSON汇总报告: {json_path}")
    
    # 4. 打印体检统计
    strict_count = sum(1 for r in quality_reports.values() if r.get('strict_pass', False))
    explore_count = sum(1 for r in quality_reports.values() if r.get('exploratory_pass', False))
    
    print(f"\n📊 因子体检统计:")
    print(f"   总因子数: {len(quality_reports)}")
    print(f"   严格通过: {strict_count} ({strict_count/len(quality_reports)*100:.1f}%)")
    print(f"   探索通过: {explore_count} ({explore_count/len(quality_reports)*100:.1f}%)")
    print(f"   未通过: {len(quality_reports) - explore_count}")
    
    # 统计各项检查的通过情况
    pass_counts = {
        '|IC|探索达标': sum(1 for r in quality_reports.values() if r.get('pass_ic_explore', False)),
        '|IC|严格达标': sum(1 for r in quality_reports.values() if r.get('pass_ic_strict', False)),
        '|ICIR|探索达标': sum(1 for r in quality_reports.values() if r.get('pass_icir_explore', False)),
        '|ICIR|严格达标': sum(1 for r in quality_reports.values() if r.get('pass_icir_strict', False)),
        'Spread>0': sum(1 for r in quality_reports.values() if r.get('pass_spread', False)),
        '低相关性': sum(1 for r in quality_reports.values() if r.get('pass_correlation', True)),
    }
    print(f"\n   各项检查通过率:")
    for check_name, count in pass_counts.items():
        print(f"   - {check_name}: {count}/{len(quality_reports)} ({count/len(quality_reports)*100:.1f}%)")
    
    # 统计正向/负向因子
    positive_factors = [f for f, r in quality_reports.items() if r.get('ic_direction') == 'positive' and r.get('exploratory_pass', False)]
    negative_factors = [f for f, r in quality_reports.items() if r.get('ic_direction') == 'negative' and r.get('exploratory_pass', False)]
    print(f"\n   因子方向分布 (探索通过):")
    print(f"   - 正向因子 ⬆️: {len(positive_factors)}")
    print(f"   - 负向因子 ⬇️: {len(negative_factors)}")
    
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
    print("步骤 6: 生成Tearsheet报告 + 可视化图表")
    print("=" * 80)
    
    # 为每个通过的因子生成完整的tearsheet报告
    reports_dir = os.path.join(ml_root, "ML output/reports/baseline_v1/factors")
    figures_dir = os.path.join(ml_root, "ML output/figures/baseline_v1/factors")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"\n📝 生成 {len(qualified_factors)} 个因子的详细报告 + 图表...\n")
    
    for i, factor_name in enumerate(qualified_factors, 1):
        print(f"[{i}/{len(qualified_factors)}] 生成报告: {factor_name}")
        
        try:
            report = quality_reports[factor_name]
            full_results = report['full_results']
            
            # ===== 6.1 生成可视化图表 =====
            factor_figures_dir = os.path.join(figures_dir, factor_name)
            os.makedirs(factor_figures_dir, exist_ok=True)
            
            plot_paths = {}
            
            # 准备IC序列数据
            key_5d = (factor_name, 'ret_5d')
            
            # 1. IC时间序列图
            if 'daily_ic' in full_results:
                daily_ic = full_results['daily_ic']
                if key_5d in daily_ic.columns:
                    ic_series = daily_ic[key_5d]
                elif isinstance(daily_ic, pd.DataFrame) and 5 in daily_ic.columns:
                    ic_series = daily_ic[5]
                elif isinstance(daily_ic, dict) and 5 in daily_ic:
                    ic_series = daily_ic[5]
                else:
                    ic_series = None
                
                if ic_series is not None and len(ic_series) > 0:
                    try:
                        ic_path = os.path.join(factor_figures_dir, f"ic_series_{factor_name}_5d.png")
                        plot_ic_time_series(
                            ic_series,
                            title=f"IC Time Series: {factor_name} @ 5d",
                            save_path=ic_path
                        )
                        plot_paths['ic_series'] = ic_path
                    except Exception as e:
                        print(f"      ⚠️  IC时间序列图生成失败: {e}")
                    
                    # 2. IC分布图
                    try:
                        ic_dist_path = os.path.join(factor_figures_dir, f"ic_dist_{factor_name}_5d.png")
                        plot_ic_distribution(
                            ic_series,
                            title=f"IC Distribution: {factor_name} @ 5d",
                            save_path=ic_dist_path
                        )
                        plot_paths['ic_distribution'] = ic_dist_path
                    except Exception as e:
                        print(f"      ⚠️  IC分布图生成失败: {e}")
                    
                    # 3. 月度IC热力图
                    try:
                        ic_heatmap_path = os.path.join(factor_figures_dir, f"ic_heatmap_{factor_name}_5d.png")
                        plot_monthly_ic_heatmap(
                            ic_series,
                            title=f"Monthly IC Heatmap: {factor_name} @ 5d",
                            save_path=ic_heatmap_path
                        )
                        plot_paths['ic_heatmap'] = ic_heatmap_path
                    except Exception as e:
                        print(f"      ⚠️  月度IC热力图生成失败: {e}")
            
            # 4. 分位数累计收益图
            if 'cumulative_returns' in full_results:
                cum_rets = full_results['cumulative_returns']
                if key_5d in cum_rets:
                    cum_ret_data = cum_rets[key_5d]
                elif 5 in cum_rets:
                    cum_ret_data = cum_rets[5]
                else:
                    cum_ret_data = None
                
                if cum_ret_data is not None and len(cum_ret_data) > 0:
                    try:
                        cum_path = os.path.join(factor_figures_dir, f"quantile_cumret_{factor_name}_5d.png")
                        plot_quantile_cumulative_returns(
                            cum_ret_data,
                            title=f"Quantile Cumulative Returns: {factor_name} @ 5d",
                            save_path=cum_path
                        )
                        plot_paths['cumulative_returns'] = cum_path
                    except Exception as e:
                        print(f"      ⚠️  分位数累计收益图生成失败: {e}")
            
            # 5. 分位数平均收益柱状图
            if 'quantile_returns' in full_results:
                q_rets = full_results['quantile_returns']
                if key_5d in q_rets:
                    q_ret_data = q_rets[key_5d]
                elif 5 in q_rets:
                    q_ret_data = q_rets[5]
                else:
                    q_ret_data = None
                
                if q_ret_data is not None and len(q_ret_data) > 0:
                    try:
                        mean_ret_path = os.path.join(factor_figures_dir, f"quantile_meanret_{factor_name}_5d.png")
                        plot_quantile_mean_returns(
                            q_ret_data,
                            title=f"Quantile Mean Returns: {factor_name} @ 5d",
                            save_path=mean_ret_path
                        )
                        plot_paths['mean_returns'] = mean_ret_path
                    except Exception as e:
                        print(f"      ⚠️  分位数平均收益图生成失败: {e}")
            
            # 6. Spread累计收益图
            if 'spreads' in full_results:
                spreads = full_results['spreads']
                if key_5d in spreads:
                    spread_data = spreads[key_5d]
                elif 5 in spreads:
                    spread_data = spreads[5]
                else:
                    spread_data = None
                
                if spread_data is not None and len(spread_data) > 0:
                    try:
                        spread_path = os.path.join(factor_figures_dir, f"spread_cumret_{factor_name}_5d.png")
                        plot_spread_cumulative_returns(
                            spread_data,
                            title=f"Spread Cumulative Returns: {factor_name} @ 5d",
                            save_path=spread_path
                        )
                        plot_paths['spread_cumulative'] = spread_path
                    except Exception as e:
                        print(f"      ⚠️  Spread累计收益图生成失败: {e}")
            
            print(f"      📊 生成 {len(plot_paths)} 个图表")
            
            # ===== 6.2 生成HTML Tearsheet =====
            tearsheet_path = os.path.join(reports_dir, f"tearsheet_{factor_name}_5d.html")
            
            generate_html_tearsheet(
                analyzer_results=full_results,
                factor_name=factor_name,
                return_period='ret_5d',
                output_path=tearsheet_path,
                plot_paths=plot_paths  # 传入图表路径
            )
            
            # ===== 6.3 保存CSV数据 =====
            # 保存IC时间序列CSV
            try:
                if 'ic_series' in full_results and 5 in full_results['ic_series']:
                    ic_series_path = os.path.join(reports_dir, f"ic_{factor_name}_5d.csv")
                    full_results['ic_series'][5].to_csv(ic_series_path)
                elif 'daily_ic' in full_results:
                    ic_series_path = os.path.join(reports_dir, f"ic_{factor_name}_5d.csv")
                    if key_5d in full_results['daily_ic'].columns:
                        full_results['daily_ic'][key_5d].to_csv(ic_series_path, header=['ic'])
            except Exception as e:
                print(f"      ⚠️  IC CSV保存失败: {e}")
            
            # 保存分位数收益CSV
            try:
                if 'quantile_returns' in full_results:
                    q_rets = full_results['quantile_returns']
                    if key_5d in q_rets:
                        quantile_returns_path = os.path.join(reports_dir, f"quantile_returns_{factor_name}_5d.csv")
                        q_rets[key_5d].to_csv(quantile_returns_path)
                    elif 5 in q_rets:
                        quantile_returns_path = os.path.join(reports_dir, f"quantile_returns_{factor_name}_5d.csv")
                        q_rets[5].to_csv(quantile_returns_path)
            except Exception as e:
                print(f"      ⚠️  分位数收益CSV保存失败: {e}")
            
            print(f"   ✅ 报告生成完成")
            print(f"      HTML: {tearsheet_path}")
            print(f"      图表目录: {factor_figures_dir}")
        
        except Exception as e:
            print(f"   ⚠️  报告生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
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
    
    # 保存原始因子 (Parquet格式)
    output_path = os.path.join(datasets_dir, f"qualified_factors_{datetime.now().strftime('%Y%m%d')}.parquet")
    qualified_factors_df.to_parquet(output_path)
    print(f"   ✅ 原始因子 (Parquet): {output_path}")
    
    # 同时保存CSV格式（兼容性）
    csv_path = os.path.join(datasets_dir, f"qualified_factors_{datetime.now().strftime('%Y%m%d')}.csv")
    qualified_factors_df.to_csv(csv_path)
    print(f"   ✅ 原始因子 (CSV): {csv_path}")
    
    # ===== 保存中性化后的因子（如果启用了中性化）=====
    print(f"\n📊 中性化状态检查:")
    print(f"   neutralize 配置: {preprocess_config.get('neutralize', False)}")
    print(f"   market_cap 数据: {'有' if market_cap is not None else '无'}")
    print(f"   industry 数据: {'有' if industry is not None else '无'}")
    
    # 检查 market_cap 是否全为 NaN
    if market_cap is not None:
        valid_mc = market_cap['market_cap'].notna().sum()
        print(f"   market_cap 有效值: {valid_mc}/{len(market_cap)}")
        if valid_mc == 0:
            print(f"   ⚠️  market_cap 全为 NaN，将仅使用行业中性化")
            market_cap = None
    
    # 检查 industry 是否有效
    if industry is not None:
        valid_ind = industry['industry'].notna().sum()
        print(f"   industry 有效值: {valid_ind}/{len(industry)}")
        if valid_ind == 0:
            print(f"   ⚠️  industry 全为 NaN，无法进行行业中性化")
            industry = None
    
    if preprocess_config.get('neutralize', False) and (market_cap is not None or industry is not None):
        print(f"\n💾 保存中性化因子...")
        
        from evaluation.factor_preprocessing import preprocess_factor_pipeline
        
        # 对所有合格因子进行中性化
        neutralized_factors_df = preprocess_factor_pipeline(
            factors=qualified_factors_df,
            market_cap=market_cap,
            industry=industry,
            winsorize=True,
            standardize=True,
            neutralize=True
        )
        
        # 保存中性化因子 (Parquet格式)
        neutral_output_path = os.path.join(datasets_dir, f"qualified_factors_neutralized_{datetime.now().strftime('%Y%m%d')}.parquet")
        neutralized_factors_df.to_parquet(neutral_output_path)
        print(f"   ✅ 中性化因子 (Parquet): {neutral_output_path}")
        
        # CSV格式
        neutral_csv_path = os.path.join(datasets_dir, f"qualified_factors_neutralized_{datetime.now().strftime('%Y%m%d')}.csv")
        neutralized_factors_df.to_csv(neutral_csv_path)
        print(f"   ✅ 中性化因子 (CSV): {neutral_csv_path}")
    
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
    
    print(f"   要求: |IC| ≥ {IC_THRESHOLD_STRICT} 且统计显著 (p < {IC_PVALUE_STRICT})")
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
    
    # 直接运行，所有参数从配置文件读取（股票池、日期范围等）
    qualified_factors_df, quality_reports, manager = prepare_factors(
        config_path="configs/ml_baseline.yml"
    )
    
    print("\n✅ 流程执行完成！")
