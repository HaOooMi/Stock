#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载器 - 统一数据接口

功能：
1. 从ML output加载标准化特征数据
2. 加载目标变量数据
3. 统一返回MultiIndex [date, ticker]格式
4. 数据对齐与清洗
5. 集成数据快照管理
6. 集成交易可行性过滤
7. 集成PIT数据对齐
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
project_root = os.path.dirname(ml_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入新模块
try:
    from data.financial_data_loader import FinancialDataLoader
    from data.data_snapshot import DataSnapshot
    from data.tradability_filter import TradabilityFilter
    from data.pit_aligner import PITDataAligner
    from data.market_data_loader import MarketDataLoader
except ImportError:
    # 如果模块未找到，尝试相对导入
    try:
        from data_snapshot import DataSnapshot
        from tradability_filter import TradabilityFilter
        from pit_aligner import PITDataAligner
        from market_data_loader import MarketDataLoader
    except ImportError:
        print("⚠️  警告: 无法导入数据清洗模块，部分功能可能不可用")
        DataSnapshot = None
        TradabilityFilter = None
        PITDataAligner = None
        MarketDataLoader = None


class DataLoader:
    """
    数据加载器类（增强版）
    
    功能：
    1. 加载特征和目标数据
    2. 数据对齐与清洗
    3. 统一格式为MultiIndex
    4. 数据快照管理
    5. 交易可行性过滤
    6. PIT数据对齐
    """
    
    def __init__(self, 
                 data_root: str = "ML output",
                 enable_snapshot: bool = True,
                 enable_filtering: bool = True,
                 enable_pit_alignment: bool = True,
                 enable_influxdb: bool = True,
                 influxdb_config: Optional[Dict[str, str]] = None,
                 filter_config: Optional[Dict[str, Any]] = None,
                 pit_config: Optional[Dict[str, Any]] = None):
        """
        初始化数据加载器
        
        Parameters:
        -----------
        data_root : str
            数据根目录
        enable_snapshot : bool
            是否启用快照管理
        enable_filtering : bool
            是否启用交易可行性过滤
        enable_pit_alignment : bool
            是否启用PIT对齐
        enable_influxdb : bool
            是否启用InfluxDB数据加载
        influxdb_config : dict, optional
            InfluxDB配置
        filter_config : dict, optional
            过滤器配置
        pit_config : dict, optional
            PIT对齐配置，包含:
            - financial_lag_days: 财务数据滞后天数（默认90）
            - financial_ffill_limit: 前向填充上限（默认95）
        """
        if os.path.isabs(data_root):
            self.data_root = data_root
        else:
            self.data_root = os.path.join(ml_root, data_root)
        
        # 功能开关
        self.enable_snapshot = enable_snapshot
        self.enable_filtering = enable_filtering
        self.enable_pit_alignment = enable_pit_alignment
        self.enable_influxdb = enable_influxdb
        
        # PIT配置
        self.pit_config = pit_config or {}
        
        # 初始化市场数据加载器（InfluxDB + MySQL）
        if enable_influxdb and MarketDataLoader is not None:
            influxdb_config = influxdb_config or {}
            try:
                self.market_data_loader = MarketDataLoader(**influxdb_config)
            except Exception as e:
                print(f"   ⚠️  市场数据加载器初始化失败: {e}")
                self.market_data_loader = None
        else:
            self.market_data_loader = None
        
        # 初始化子模块
        if enable_snapshot and DataSnapshot is not None:
            # DataSnapshot 应该使用 ML output 根目录,而不是 datasets_dir
            # 避免在 datasets/baseline_v1 下创建 snapshots/ 和 reports/ 子目录
            if 'ML output' in self.data_root:
                snapshot_root = self.data_root.split('ML output')[0] + 'ML output'
            else:
                snapshot_root = self.data_root
            self.snapshot_mgr = DataSnapshot(output_dir=snapshot_root)
        else:
            self.snapshot_mgr = None
        
        if enable_filtering and TradabilityFilter is not None:
            filter_config = filter_config or {}
            self.filter_engine = TradabilityFilter(**filter_config)
        else:
            self.filter_engine = None
        
        if enable_pit_alignment and PITDataAligner is not None:
            self.pit_aligner = PITDataAligner()
        else:
            self.pit_aligner = None
        
        print(f"📁 数据加载器初始化（增强版）")
        print(f"   数据根目录: {self.data_root}")
        print(f"   快照管理: {'✅' if enable_snapshot else '❌'}")
        print(f"   交易过滤: {'✅' if enable_filtering else '❌'}")
        print(f"   PIT对齐: {'✅' if enable_pit_alignment else '❌'}")
        print(f"   市场数据: {'✅' if self.market_data_loader is not None else '❌'}")
    
    def load_features_and_targets(self, 
                                  symbol: str,
                                  target_col: str = 'future_return_5d',
                                  use_scaled: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        加载特征和目标数据（从ML output目录）
        
        Parameters:
        -----------
        symbol : str
            股票代码
        target_col : str
            目标列名
        use_scaled : bool
            是否使用标准化后的特征
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            (特征数据, 目标数据)，索引为MultiIndex [date, ticker]
        """
        print(f"📊 加载数据: {symbol}")
        
        # 确定根目录（ML output）
        if 'ML output' in self.data_root:
            ml_output_root = self.data_root.split('ML output')[0] + 'ML output'
        else:
            ml_output_root = self.data_root
        
        # 1. 加载特征数据
        if use_scaled:
            # 标准化特征在 scalers/baseline_v1 目录
            scalers_dir = os.path.join(ml_output_root, 'scalers', 'baseline_v1')
            feature_pattern = f"scaler_{symbol}_scaled_features.csv"
            
            if not os.path.exists(scalers_dir):
                raise FileNotFoundError(f"标准化器目录不存在: {scalers_dir}")
            
            feature_files = [f for f in os.listdir(scalers_dir) 
                           if f == feature_pattern]
            
            if not feature_files:
                raise FileNotFoundError(f"未找到标准化特征文件: {feature_pattern} (目录: {scalers_dir})")
            
            feature_file = os.path.join(scalers_dir, feature_files[0])
            print(f"   📈 加载标准化特征: {feature_files[0]}")
        else:
            # 从with_targets文件加载
            target_pattern = f"with_targets_{symbol}_complete_*.csv"
            target_files = [f for f in os.listdir(self.data_root) if f.startswith(f"with_targets_{symbol}")]
            
            if not target_files:
                raise FileNotFoundError(f"未找到目标文件: {target_pattern}")
            
            # 使用最新的文件
            target_files.sort(reverse=True)
            feature_file = os.path.join(self.data_root, target_files[0])
            print(f"   📈 加载特征: {target_files[0]}")
        
        # 加载特征数据
        features_df = pd.read_csv(feature_file, index_col=0, parse_dates=True, encoding='utf-8')
        
        # 2. 加载目标数据（从 datasets 目录）
        target_pattern = f"with_targets_{symbol}_complete_*.csv"
        
        target_files = [f for f in os.listdir(self.data_root) 
                       if f.startswith(f"with_targets_{symbol}_complete_") and f.endswith('.csv')]
        
        if not target_files:
            raise FileNotFoundError(f"未找到目标文件: {target_pattern} (目录: {self.data_root})")
        
        # 使用最新的文件
        target_files.sort(reverse=True)
        target_file = os.path.join(self.data_root, target_files[0])
        print(f"   🎯 加载目标: {target_files[0]}")
        
        # 加载目标数据
        targets_df = pd.read_csv(target_file, index_col=0, parse_dates=True, encoding='utf-8')
        
        # 3. 检查目标列是否存在
        if target_col not in targets_df.columns:
            available_targets = [col for col in targets_df.columns if col.startswith('future_return_')]
            raise ValueError(f"目标列 '{target_col}' 不存在。可用目标: {available_targets}")
        
        # 4. 提取特征列（排除close和目标列）
        exclude_cols = ['close'] + [col for col in features_df.columns if col.startswith('future_return_') or col.startswith('label_')]
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # 5. 对齐索引
        common_index = features_df.index.intersection(targets_df.index)
        features_aligned = features_df.loc[common_index, feature_cols]
        targets_aligned = targets_df.loc[common_index, target_col]
        
        # 6. 转换为MultiIndex格式 [date, ticker]
        # 为单个股票创建MultiIndex
        dates = features_aligned.index
        tickers = [symbol] * len(dates)
        multi_index = pd.MultiIndex.from_arrays([dates, tickers], names=['date', 'ticker'])
        
        features_multi = pd.DataFrame(features_aligned.values, index=multi_index, columns=feature_cols)
        targets_multi = pd.Series(targets_aligned.values, index=multi_index, name=target_col)
        
        # 7. 清洗数据（仅删除特征行的NaN，保留目标尾部NaN以通过PIT验证）
        # 检查目标尾部NaN（用于诊断）
        tail_target_nans = targets_multi.tail(10).isna().sum()
        
        # 统计特征缺失情况
        feature_nan_counts = features_multi.isna().sum()
        high_nan_cols = feature_nan_counts[feature_nan_counts > 0].sort_values(ascending=False)
        
        print(f"\n   🔍 数据质量诊断:")
        print(f"      目标尾部10行NaN: {tail_target_nans}/10 (PIT要求保留)")
        if len(high_nan_cols) > 0:
            print(f"      存在缺失的特征列: {len(high_nan_cols)} 个")
            if len(high_nan_cols) <= 5:
                print(f"      缺失列详情: {dict(high_nan_cols)}")
            else:
                print(f"      缺失TOP5: {dict(high_nan_cols.head(5))}")
        
        # 修改过滤逻辑：仅删除特征行有NaN的样本，保留目标NaN（特别是尾部）
        valid_mask = ~features_multi.isna().any(axis=1)
        features_clean = features_multi[valid_mask]
        targets_clean = targets_multi[valid_mask]
        
        # 再次检查目标尾部NaN（确认保留）
        tail_target_nans_after = targets_clean.tail(10).isna().sum()
        
        print(f"\n   ✅ 数据加载完成:")
        print(f"      特征数量: {len(feature_cols)}")
        print(f"      初始样本: {len(features_aligned)}")
        print(f"      有效样本: {len(features_clean)} (过滤掉 {len(features_aligned) - len(features_clean)} 个特征缺失样本)")
        print(f"      目标尾部NaN保留: {tail_target_nans_after}/10 ({'✅' if tail_target_nans_after > 0 else '⚠️'})")
        if len(features_clean) > 0:
            print(f"      时间范围: {features_clean.index.get_level_values('date').min().date()} ~ "
                  f"{features_clean.index.get_level_values('date').max().date()}")
        else:
            print(f"      ⚠️  警告: 过滤后样本数为0，请检查特征缺失率或放宽过滤条件")
        
        return features_clean, targets_clean
    
    def _load_and_merge_market_data(self,
                                    features: pd.DataFrame,
                                    symbol: str,
                                    start_date: str,
                                    end_date: str) -> pd.DataFrame:
        """
        加载市场数据并合并到特征（InfluxDB + MySQL）
        
        注：调用 MarketDataLoader.load_market_data()，该方法会：
        1. 从 InfluxDB 加载 OHLCV 数据
        2. 从 MySQL 加载元数据（ST状态、上市日期、总股本等）
        
        Parameters:
        -----------
        features : pd.DataFrame
            现有特征数据，MultiIndex [date, ticker]
        symbol : str
            股票代码
        start_date : str
            开始日期
        end_date : str
            结束日期
            
        Returns:
        --------
        pd.DataFrame
            合并后的特征数据
        """
        if self.market_data_loader is None:
            print(f"   ⚠️  市场数据加载器未启用，跳过市场数据加载")
            return features
        
        try:
            # 加载市场数据（InfluxDB + MySQL）
            market_df = self.market_data_loader.load_market_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if market_df.empty:
                print(f"   ⚠️  未找到市场数据")
                return features
            
            # 提取特征数据的日期索引（去重）
            unique_dates = features.index.get_level_values('date').unique()
            
            # 对齐市场数据到特征数据的日期
            market_aligned = market_df.reindex(unique_dates)
            
            # 添加需要的列到特征数据（考虑 MultiIndex）
            # 注意：不覆盖已存在的列
            added_cols = []
            for col in market_aligned.columns:
                if col not in features.columns:
                    # 从 MultiIndex 中提取日期，然后从 market_aligned 中获取对应值
                    feature_dates = features.index.get_level_values('date')
                    features[col] = market_aligned[col].reindex(feature_dates).values
                    added_cols.append(col)
            
            print(f"   ✅ 市场数据合并完成: 添加 {len(added_cols)} 列")
            if added_cols:
                print(f"      新增列: {', '.join(added_cols[:5])}{'...' if len(added_cols) > 5 else ''}")
            
        except Exception as e:
            print(f"   ⚠️  市场数据加载失败: {e}")
        
        return features
    
    def _load_and_merge_financial_data(self,
                                       features: pd.DataFrame,
                                       targets: pd.Series,
                                       symbol: str,
                                       start_date: str,
                                       end_date: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        加载财务数据并合并到特征（PIT对齐）
        
        Parameters:
        -----------
        features : pd.DataFrame
            现有特征数据
        targets : pd.Series
            目标数据
        symbol : str
            股票代码
        start_date : str
            开始日期
        end_date : str
            结束日期
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            合并后的特征和目标
        """
        try:            
            # 初始化财务数据加载器（从配置读取参数）
            announce_lag_days = self.pit_config.get('financial_lag_days', 90)
            ffill_limit = self.pit_config.get('financial_ffill_limit', 95)
            
            financial_loader = FinancialDataLoader(
                announce_lag_days=announce_lag_days,
                ffill_limit=ffill_limit
            )
            
            # 加载财务数据
            financial_df = financial_loader.load_financial_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if financial_df.empty:
                print(f"   ⚠️  未找到财务数据")
                return features, targets
            
            # 计算财务特征
            financial_df = financial_loader.calculate_financial_features(financial_df)
            
            # 对齐到交易日
            dates = features.index.get_level_values('date').unique()
            aligned_financial = financial_loader.align_to_trading_dates(
                financial_df, 
                pd.DatetimeIndex(dates)
            )
            
            # 合并到特征数据
            fin_cols_added = []
            for col in aligned_financial.columns:
                if col not in ['symbol', 'report_date', 'announce_date', 'effective_date']:
                    # 添加前缀避免列名冲突
                    new_col = f'fin_{col}'
                    if new_col not in features.columns:
                        # 直接使用 aligned_financial 的值，因为它已经和交易日对齐过了
                        # 长度相同则直接赋值，否则需要 reindex
                        if len(features) == len(aligned_financial):
                            features[new_col] = aligned_financial[col].values
                        else:
                            # 长度不同时需要重新对齐
                            feature_dates = features.index.get_level_values('date')
                            aligned_values = aligned_financial[col].reindex(feature_dates)
                            features[new_col] = aligned_values.values
                        
                        fin_cols_added.append(new_col)
            
            # 财务数据的前向填充已在 FinancialDataLoader.align_to_trading_dates() 中完成
            # 无需在此重复填充，避免覆盖已有的填充逻辑
            if fin_cols_added:
                # 计算财务列缺失率（仅统计，不再填充）
                fin_nans = features[fin_cols_added].isna().sum().sum()
                fin_total = len(features) * len(fin_cols_added)
                fin_missing_rate = (fin_nans / fin_total * 100) if fin_total > 0 else 0
                
                print(f"   ✓ 财务特征合并完成: 添加 {len(fin_cols_added)} 个财务特征")
                print(f"      财务数据缺失率: {fin_missing_rate:.2f}%")
                if fin_nans > 0:
                    print(f"      缺失样本数: {fin_nans} / {fin_total}")
            else:
                print(f"   ✓ 财务特征合并完成: 未添加新列（可能已存在或无可用数据）")
            
        except Exception as e:
            print(f"   ⚠️  财务数据加载失败: {e}")
        
        return features, targets
    
    def load_universe(self, 
                     symbol: str,
                     min_volume: Optional[float] = None,
                     min_price: Optional[float] = None) -> pd.DataFrame:
        """
        加载可交易标的列表（过滤低流动性）
        
        Parameters:
        -----------
        symbol : str
            股票代码
        min_volume : float, optional
            最小成交量
        min_price : float, optional
            最小价格
            
        Returns:
        --------
        pd.DataFrame
            可交易标的的MultiIndex数据
        """
        print(f"🔍 加载可交易标的列表")
        
        # 加载原始数据（包含volume和close）
        target_files = [f for f in os.listdir(self.data_root) if f.startswith(f"with_targets_{symbol}")]
        if not target_files:
            raise FileNotFoundError(f"未找到数据文件")
        
        target_files.sort(reverse=True)
        data_file = os.path.join(self.data_root, target_files[0])
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        
        # 初始化过滤mask
        valid_mask = pd.Series(True, index=df.index)
        
        # 应用过滤条件
        if min_volume is not None and 'volume' in df.columns:
            volume_mask = df['volume'] >= min_volume
            valid_mask &= volume_mask
            print(f"   📊 成交量过滤: {volume_mask.sum()} / {len(df)} 样本")
        
        if min_price is not None and 'close' in df.columns:
            price_mask = df['close'] >= min_price
            valid_mask &= price_mask
            print(f"   💰 价格过滤: {price_mask.sum()} / {len(df)} 样本")
        
        # 转换为MultiIndex格式
        dates = df[valid_mask].index
        tickers = [symbol] * len(dates)
        multi_index = pd.MultiIndex.from_arrays([dates, tickers], names=['date', 'ticker'])
        
        universe_df = pd.DataFrame({
            'tradable': True,
            'close': df.loc[valid_mask, 'close'].values if 'close' in df.columns else np.nan,
            'volume': df.loc[valid_mask, 'volume'].values if 'volume' in df.columns else np.nan
        }, index=multi_index)
        
        print(f"   ✅ 可交易标的: {len(universe_df)} 个时间点")
        
        return universe_df
    
    def get_feature_list(self, symbol: str, use_scaled: bool = True) -> List[str]:
        """
        获取特征列表
        
        Parameters:
        -----------
        symbol : str
            股票代码
        use_scaled : bool
            是否使用标准化特征
            
        Returns:
        --------
        List[str]
            特征名称列表
        """
        # 优先从final_feature_list.txt读取
        feature_list_file = os.path.join(self.data_root, "final_feature_list.txt")
        if os.path.exists(feature_list_file):
            with open(feature_list_file, 'r', encoding='utf-8') as f:
                features = [line.strip() for line in f if line.strip()]
            print(f"   📋 从文件加载特征列表: {len(features)} 个特征")
            return features
        
        # 否则从数据文件中提取
        if use_scaled:
            feature_files = [f for f in os.listdir(self.data_root) if f.startswith(f"scaler_{symbol}_scaled_features.csv")]
        else:
            feature_files = [f for f in os.listdir(self.data_root) if f.startswith(f"with_targets_{symbol}_complete_")]
        
        if not feature_files:
            raise FileNotFoundError("未找到特征文件")
        
        feature_file = os.path.join(self.data_root, feature_files[0])
        df = pd.read_csv(feature_file, index_col=0, nrows=1)
        
        exclude_cols = ['close'] + [col for col in df.columns if col.startswith('future_return_') or col.startswith('label_')]
        features = [col for col in df.columns if col not in exclude_cols]
        
        print(f"   📋 从数据文件提取特征列表: {len(features)} 个特征")
        return features
    
    def load_with_snapshot(self,
                          symbol: str,
                          start_date: str,
                          end_date: str,
                          target_col: str = 'future_return_5d',
                          use_scaled: bool = True,
                          filters: Optional[Dict[str, Any]] = None,
                          random_seed: int = 42,
                          save_parquet: bool = True) -> Tuple[pd.DataFrame, pd.Series, str]:
        """
        加载数据并创建快照（推荐使用）
        
        Parameters:
        -----------
        symbol : str
            股票代码
        start_date : str
            开始日期
        end_date : str
            结束日期
        target_col : str
            目标列名
        use_scaled : bool
            是否使用标准化特征
        filters : dict, optional
            过滤参数
        random_seed : int
            随机种子
        save_parquet : bool
            是否保存为Parquet
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series, str]
            (特征数据, 目标数据, 快照ID)
        """
        print(f"\n{'='*60}")
        print(f"📊 加载数据并创建快照")
        print(f"{'='*60}")
        
        # 1. 加载原始数据
        features, targets = self.load_features_and_targets(
            symbol=symbol,
            target_col=target_col,
            use_scaled=use_scaled
        )
        
        # 2. 加载并合并市场数据（InfluxDB OHLCV + MySQL 元数据）
        if self.enable_influxdb and self.market_data_loader is not None:
            print(f"\n[市场数据] 加载 InfluxDB + MySQL 数据")
            features = self._load_and_merge_market_data(
                features=features,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
        
        # 3. 加载并合并财务数据（PIT对齐）
        if self.enable_pit_alignment:
            print(f"\n[财务数据] 加载并对齐财务数据")
            features, targets = self._load_and_merge_financial_data(
                features, targets, symbol, start_date, end_date
            )
        
        # 4. 应用交易可行性过滤
        # 注意：如果调用方传入 filters=None，表示已在之前的步骤完成过滤（通常在标准化前）
        #       此时跳过重复过滤，避免用原始阈值对比标准化后的值
        initial_sample_count = len(features)
        if filters is not None and self.enable_filtering and self.filter_engine is not None:
            # 合并特征和目标以便过滤
            combined_data = features.copy()
            combined_data[target_col] = targets
            
            # 应用过滤
            filter_log_path = os.path.join(
                self.data_root,
                f'filter_log_{symbol}.csv'
            )
            
            filtered_data, filter_log_df = self.filter_engine.apply_filters(
                combined_data,
                save_log=True,
                log_path=filter_log_path
            )
            
            # 提取可交易样本
            tradable_mask = filtered_data['tradable_flag'] == 1
            features = filtered_data[tradable_mask][features.columns]
            targets = filtered_data[tradable_mask][target_col]
            
            removed_by_filter = initial_sample_count - len(features)
            print(f"\n   ✅ 交易过滤完成:")
            print(f"      过滤前: {initial_sample_count} 个样本")
            print(f"      过滤后: {len(features)} 个可交易样本")
            print(f"      剔除: {removed_by_filter} 个样本 ({removed_by_filter/initial_sample_count:.1%})")
            
            if len(features) == 0:
                print(f"\n      ⚠️  警告: 所有样本均被过滤，建议:")
                print(f"         1. 降低 min_volume (当前: {self.filter_engine.min_volume:,.0f})")
                print(f"         2. 降低 min_turnover (当前: {self.filter_engine.min_turnover:.2%})")
                print(f"         3. 降低 min_listing_days (当前: {self.filter_engine.min_listing_days})")
                print(f"         4. 检查过滤日志: {filter_log_path}")
        elif filters is None:
            print(f"\n   ⏭️  跳过过滤器（已在标准化前完成）")
        
        # 4.5 PIT对齐前的统计（用于诊断）
        print(f"\n   📊 PIT验证前数据状态:")
        print(f"      样本数: {len(features)}")
        if len(features) > 0:
            tail_nans = targets.tail(10).isna().sum()
            print(f"      目标尾部10行NaN: {tail_nans}/10")
            feature_nan_ratio = features.isna().sum().sum() / (len(features) * len(features.columns))
            print(f"      特征总体缺失率: {feature_nan_ratio:.2%}")
        
        # 5. PIT对齐验证（在过滤之后）
        if self.enable_pit_alignment and self.pit_aligner is not None:
            combined_data = features.copy()
            combined_data[target_col] = targets
            
            pit_results = self.pit_aligner.validate_pit_alignment(
                combined_data,
                target_col=target_col
            )
            
            if not pit_results.get('overall_pass', False):
                print(f"\n   ⚠️  PIT对齐验证未通过，详细结果:")
                for check_name, result in pit_results.items():
                    if check_name != 'overall_pass':
                        status = '✅' if result else '❌'
                        print(f"      {status} {check_name}: {result}")
        
        # 5. 创建数据快照
        snapshot_id = None
        if self.enable_snapshot and self.snapshot_mgr is not None:
            # 准备快照数据
            snapshot_data = features.copy()
            snapshot_data[target_col] = targets
            
            # 过滤参数
            filters = filters or {
                'min_volume': getattr(self.filter_engine, 'min_volume', None) if self.filter_engine else None,
                'min_price': getattr(self.filter_engine, 'min_price', None) if self.filter_engine else None,
                'exclude_st': getattr(self.filter_engine, 'exclude_st', None) if self.filter_engine else None
            }
            
            # 创建快照
            _ = self.snapshot_mgr.create_snapshot(
                data=snapshot_data,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                filters=filters,
                random_seed=random_seed,
                save_parquet=save_parquet
            )
            
            snapshot_id = self.snapshot_mgr.snapshot_id
            print(f"\n   ✅ 数据快照创建完成: {snapshot_id}")
        
        print(f"\n{'='*60}")
        print(f"✅ 数据加载完成")
        print(f"{'='*60}")
        print(f"   特征数量: {len(features.columns)}")
        print(f"   样本数量: {len(features)}")
        print(f"   快照ID: {snapshot_id or 'N/A'}")
        print(f"{'='*60}\n")
        
        return features, targets, snapshot_id
    
    def load_from_snapshot(self, snapshot_id: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        从快照加载数据
        
        Parameters:
        -----------
        snapshot_id : str
            快照ID
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            (特征数据, 目标数据)
        """
        if self.snapshot_mgr is None:
            raise RuntimeError("快照管理器未启用")
        
        # 加载快照
        data, metadata = self.snapshot_mgr.load_snapshot(snapshot_id)
        
        # 分离特征和目标
        target_col = metadata.get('target_col', 'future_return_5d')
        
        # 如果目标列在数据中
        if target_col in data.columns:
            targets = data[target_col]
            features = data.drop(columns=[target_col])
        else:
            # 否则假设所有列都是特征
            features = data
            targets = pd.Series(index=data.index, dtype=float)
        
        print(f"✅ 从快照加载数据: {snapshot_id}")
        print(f"   特征数量: {len(features.columns)}")
        print(f"   样本数量: {len(features)}")
        
        return features, targets


if __name__ == "__main__":
    """
    使用示例
    """
    print("📊 数据加载器测试")
    print("=" * 50)
    
    try:
        # 初始化加载器
        loader = DataLoader()
        
        # 加载特征和目标
        symbol = "000001"
        features, targets = loader.load_features_and_targets(
            symbol=symbol,
            target_col='future_return_5d',
            use_scaled=True
        )
        
        print(f"\n✅ 数据加载成功:")
        print(f"   特征形状: {features.shape}")
        print(f"   目标形状: {targets.shape}")
        print(f"   索引类型: {type(features.index)}")
        print(f"   索引级别: {features.index.names}")
        
        # 加载可交易标的
        universe = loader.load_universe(
            symbol=symbol,
            min_volume=1000000,
            min_price=1.0
        )
        
        print(f"\n✅ 可交易标的加载成功:")
        print(f"   形状: {universe.shape}")
        
        # 获取特征列表
        feature_list = loader.get_feature_list(symbol=symbol)
        print(f"\n✅ 特征列表获取成功:")
        print(f"   特征数量: {len(feature_list)}")
        print(f"   前10个特征: {feature_list[:10]}")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
