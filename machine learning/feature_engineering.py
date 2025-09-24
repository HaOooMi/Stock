#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票特征工程模块 - 使用真实InfluxDB历史数据

功能:
1. 从InfluxDB加载真实股票历史数据
2. 生成技术分析特征 
3. 特征选择和优化
4. 特征质量分析

作者: Your Name
日期: 2024
"""

import pandas as pd
import numpy as np
import sys
import os
import pickle
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# InfluxDB相关
sys.path.append(r'd:\vscode projects\stock\stock_info')
from utils import get_influxdb_client
from stock_market_data_akshare import get_history_data

# 机器学习相关 
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 尝试导入可选库
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("⚠️ talib 未安装，部分技术指标将使用pandas实现")

try:
    from tsfresh import extract_features
    HAS_TSFRESH = True
except ImportError:
    HAS_TSFRESH = False
    print("⚠️ tsfresh 未安装，自动特征生成不可用")

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ xgboost 未安装，将使用RandomForest进行特征重要性评估")




class FeatureEngineer:
    """
    股票特征工程类 - 使用真实InfluxDB数据
    
    功能:
    1. 从InfluxDB加载真实历史股票数据
    2. 生成技术分析特征
    3. 特征选择和优化 
    4. 特征质量分析
    """
    
    def __init__(self, use_talib: bool = True, use_tsfresh: bool = False):
        """
        初始化特征工程器
        
        Parameters:
        -----------
        use_talib : bool, default=True
            是否使用TA-Lib进行技术指标计算
        use_tsfresh : bool, default=False  
            是否使用TSFresh进行自动特征生成
        """
        self.use_talib = use_talib and HAS_TALIB
        self.use_tsfresh = use_tsfresh and HAS_TSFRESH
        self.use_xgboost = HAS_XGBOOST
        
        # 初始化InfluxDB客户端
        self.influx_client = get_influxdb_client()
        if not self.influx_client:
            raise ConnectionError("无法连接到InfluxDB，请检查服务状态和配置")
        self.query_api = self.influx_client.query_api()
        
        print("🔧 特征工程器初始化完成")
        print(f"   📊 TA-Lib: {'✅' if self.use_talib else '❌'}")
        print(f"   🤖 TSFresh: {'✅' if self.use_tsfresh else '❌'}")
        print(f"   💾 InfluxDB: ✅")
    
    def load_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从InfluxDB加载真实股票历史数据
        
        Parameters:
        -----------
        symbol : str
            股票代码，如 '000001' 
        start_date : str
            开始日期，格式 'YYYY-MM-DD'
        end_date : str  
            结束日期，格式 'YYYY-MM-DD'
        
        Returns:
        --------
        pd.DataFrame
            包含OHLCV数据的DataFrame
        """
        print(f"📊 从InfluxDB加载股票数据: {symbol} ({start_date} 到 {end_date})")
        
        try:
            # 转换日期格式为InfluxDB Flux查询格式
            start_flux = pd.to_datetime(start_date).strftime('%Y-%m-%dT00:00:00Z')
            end_flux = pd.to_datetime(end_date).strftime('%Y-%m-%dT23:59:59Z')
            
            # 从InfluxDB获取数据
            df = get_history_data(self.query_api, symbol, start_flux, end_flux)
            
            if df.empty:
                raise ValueError(f"未找到股票 {symbol} 在指定时间范围内的数据")
            
            # 标准化列名 
            column_mapping = {
                '日期': 'timestamp',
                '开盘': 'open', 
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low', 
                '成交量': 'volume',
                '成交额': 'turnover'
            }
            
            # 重命名存在的列
            existing_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df.rename(columns=existing_cols)
            
            # 确保时间列正确处理
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            elif '日期' in df.columns:
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.set_index('日期')
            
            # 确保数据类型正确
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 删除缺失值
            df = df.dropna()
            
            # 按时间排序
            df = df.sort_index()
            
            print(f"✅ 成功加载 {len(df)} 条数据记录")
            print(f"📅 数据时间范围: {df.index.min().date()} 到 {df.index.max().date()}")
            
            return df
            
        except Exception as e:
            print(f"❌ 加载股票数据时出错: {str(e)}")
            raise

    def prepare_features(self, data: pd.DataFrame, use_auto_features: bool = True, 
                        window_size: int = 20, max_auto_features: int = 50) -> pd.DataFrame:
        """
        统一特征生成方法 - 支持手工特征和可选的自动特征
        
        Parameters:
        -----------
        data : pd.DataFrame
            原始股票数据，包含OHLCV列
        use_auto_features : bool, default=False
            是否使用TSFresh自动生成特征
        window_size : int, default=20
            自动特征生成的滑动窗口大小
        max_auto_features : int, default=50
            自动特征的最大数量
        
        Returns:
        --------
        pd.DataFrame
            包含所有特征的数据框
        """
        print("🔨 开始特征生成...")
        
        # === 1. 手工特征生成 ===
        print("📊 生成手工特征...")
        data = data.copy()
        
        # 收益率特征
        print("   📈 计算收益率特征...")
        data['return_1d'] = data['close'].pct_change()
        data['return_5d'] = data['close'].pct_change(5)
        data['return_10d'] = data['close'].pct_change(10)
        data['return_20d'] = data['close'].pct_change(20)
        
        # 滚动统计特征
        print("   📊 计算滚动统计特征...")
        for window in [5, 10, 20, 30]:
            data[f'rolling_mean_{window}d'] = data['close'].rolling(window).mean()
            data[f'rolling_std_{window}d'] = data['close'].rolling(window).std()
            data[f'rolling_median_{window}d'] = data['close'].rolling(window).median()
            data[f'price_position_{window}d'] = (data['close'] - data['close'].rolling(window).min()) / (
                data['close'].rolling(window).max() - data['close'].rolling(window).min() + 1e-8) * 2 - 1
        
        # 动量特征
        print("   🚀 计算动量特征...")
        data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
        data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
        data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
        data['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
        
        # 波动率特征  
        print("   📊 计算波动率特征...")
        if self.use_talib:
            data['atr_14'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        else:
            data['high_low'] = data['high'] - data['low']
            data['high_close'] = np.abs(data['high'] - data['close'].shift())
            data['low_close'] = np.abs(data['low'] - data['close'].shift())
            data['atr_14'] = pd.concat([data['high_low'], data['high_close'], data['low_close']], axis=1).max(axis=1).rolling(14).mean()
            data.drop(['high_low', 'high_close', 'low_close'], axis=1, inplace=True)
        
        data['volatility_5d'] = data['return_1d'].rolling(5).std()
        data['volatility_20d'] = data['return_1d'].rolling(20).std()
        data['skewness_20d'] = data['return_1d'].rolling(20).skew()
        data['kurtosis_20d'] = data['return_1d'].rolling(20).kurt()
        
        # 成交量特征
        print("   💰 计算成交量特征...")
        data['volume_change'] = data['volume'].pct_change()
        data['volume_ma_5'] = data['volume'].rolling(5).mean()
        data['volume_ma_20'] = data['volume'].rolling(20).mean()
        data['volume_ratio_5d'] = data['volume'] / data['volume_ma_5']
        data['volume_ratio_20d'] = data['volume'] / data['volume_ma_20']
        data['volume_roc_3d'] = data['volume'].pct_change(3)
        data['volume_roc_5d'] = data['volume'].pct_change(5)
        data['volume_roc_10d'] = data['volume'].pct_change(10)
        
        # 价格范围特征
        print("   📏 计算价格范围特征...")
        data['high_low_ratio'] = data['high'] / data['low']
        data['high_close_ratio'] = data['high'] / data['close']
        data['low_close_ratio'] = data['low'] / data['close']
        data['open_close_ratio'] = data['open'] / data['close']
        data['price_range'] = data['high'] - data['low']
        data['price_range_pct'] = (data['high'] - data['low']) / data['close']
        data['open_close_range'] = np.abs(data['open'] - data['close'])
        data['open_close_range_pct'] = np.abs(data['open'] - data['close']) / data['close']
        
        # 技术指标
        print("   🔍 计算技术指标特征...")
        if self.use_talib:
            data['rsi_14'] = talib.RSI(data['close'], timeperiod=14)
            data['bb_upper'], data['bb_middle'], data['bb_lower'] = talib.BBANDS(data['close'])
            data['macd'], data['macd_signal'], data['macd_hist'] = talib.MACD(data['close'])
        else:
            # RSI的pandas实现
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            data['rsi_14'] = 100 - (100 / (1 + rs))
            
            # 布林带的pandas实现
            data['bb_middle'] = data['close'].rolling(20).mean()
            bb_std = data['close'].rolling(20).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
            
            # MACD的pandas实现
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            data['macd'] = ema_12 - ema_26
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # 布林带位置和宽度
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'] + 1e-8)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # 清理无用列
        feature_columns = [col for col in data.columns 
                          if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'open_interest']]
        
        # 创建手工特征结果，保持时间索引一致性
        manual_result = data[['close'] + feature_columns].dropna()
        # 为了后续匹配，添加一个数值索引列作为时间戳
        manual_result['time_idx'] = range(len(manual_result))
        print(f"   ✅ 手工特征生成完成，特征数量: {len(feature_columns)}")
        
        # === 2. 自动特征生成（可选）===
        if use_auto_features and self.use_tsfresh:
            print("\n🤖 生成自动特征...")
            print(f"   📊 窗口大小: {window_size}")
            print(f"   🔢 最大特征数: {max_auto_features}")
            
            try:
                # 准备tsfresh格式的数据 - 使用清理后的数据
                clean_data = data.dropna()  # 使用与manual_result相同的清理逻辑
                tsfresh_data = []
                
                for i in range(window_size, len(clean_data)):
                    window_data = clean_data.iloc[i-window_size:i]
                    for col in ['close', 'volume']:
                        if col in window_data.columns:
                            for j, value in enumerate(window_data[col]):
                                tsfresh_data.append({
                                    'id': i,
                                    'time': j,
                                    'value': value,
                                    'variable': col
                                })
                
                if tsfresh_data:
                    tsfresh_df = pd.DataFrame(tsfresh_data)
                    
                    # 提取特征（区分不同变量类型）
                    from tsfresh.feature_extraction import MinimalFCParameters
                    extracted_features = extract_features(
                        tsfresh_df,
                        column_id='id',
                        column_sort='time',
                        column_value='value',
                        column_kind='variable',  # 关键：区分close和volume变量
                        default_fc_parameters=MinimalFCParameters(),
                        disable_progressbar=True,  # 禁用进度条
                        n_jobs=1  # 单线程避免潜在问题
                    )
                    
                    # 重命名特征，保持语义清晰（close__variance -> auto_close__variance）
                    renamed_features = {}
                    for col in extracted_features.columns:
                        renamed_features[col] = f'auto_{col}'
                    extracted_features = extracted_features.rename(columns=renamed_features)
                    
                    # 选择最重要的特征（基于方差）
                    auto_cols_renamed = [col for col in extracted_features.columns if col.startswith('auto_')]
                    if len(auto_cols_renamed) > max_auto_features:
                        # 基于方差选择top-k特征
                        feature_vars = extracted_features[auto_cols_renamed].var()
                        selected_features = feature_vars.nlargest(max_auto_features).index.tolist()
                        extracted_features = extracted_features[selected_features]
                        print(f"   🎯 基于方差选择了 {len(selected_features)} 个最优特征")
                    
                    # 创建临时数据框
                    temp_auto_df = pd.DataFrame(extracted_features)
                    
                    # 创建自动特征结果数据框
                    result_indices = range(window_size, len(clean_data))
                    auto_result = pd.DataFrame({
                        'close': clean_data.iloc[result_indices]['close'].values,
                        'time_idx': result_indices  # 使用相同的时间索引
                    })
                    
                    # 添加自动特征（清洗将在select_features中进行）
                    final_auto_cols = [col for col in temp_auto_df.columns if col.startswith('auto_')]
                    for col in final_auto_cols:
                        auto_result[col] = temp_auto_df[col].values
                    
                    auto_result = auto_result.dropna()
                    final_auto_count = len([col for col in auto_result.columns if col.startswith('auto_')])
                    print(f"   ✅ 自动特征生成完成，最终特征数量: {final_auto_count}")
                    
                    # 合并手工特征和自动特征 - 使用time_idx进行匹配
                    manual_times = set(manual_result['time_idx'])
                    auto_times = set(auto_result['time_idx'])
                    common_times = manual_times.intersection(auto_times)
                    
                    if common_times and len(common_times) > 0:
                        # 按time_idx匹配数据
                        manual_filtered = manual_result[manual_result['time_idx'].isin(common_times)].copy()
                        auto_filtered = auto_result[auto_result['time_idx'].isin(common_times)].copy()
                        
                        # 按time_idx排序确保对应关系正确
                        manual_filtered = manual_filtered.sort_values('time_idx').reset_index(drop=True)
                        auto_filtered = auto_filtered.sort_values('time_idx').reset_index(drop=True)
                        
                        # 合并特征（保留手工特征的所有列，添加自动特征）
                        auto_features_only = auto_filtered.drop(['close', 'time_idx'], axis=1, errors='ignore')
                        combined_result = pd.concat([manual_filtered, auto_features_only], axis=1)
                        
                        # 添加真实的时间索引
                        if not combined_result.empty:
                            time_indices = combined_result['time_idx'].values
                            combined_result.index = clean_data.index[time_indices]
                            combined_result = combined_result.drop('time_idx', axis=1)  # 删除临时索引列
                        
                        # 计算合并后的特征统计
                        total_features = len(combined_result.columns) - 1  # 排除close列
                        manual_features = len([col for col in combined_result.columns if not col.startswith('auto_') and col != 'close'])
                        auto_features = len([col for col in combined_result.columns if col.startswith('auto_')])
                        
                        print(f"✅ 特征合并完成:")
                        print(f"   📈 共享样本: {len(common_times)}")
                        print(f"   🔢 总特征数: {total_features} (手工:{manual_features} + 自动:{auto_features})")
                        
                        return combined_result
                    else:
                        print(f"⚠️ 手工特征和自动特征时间不匹配")
                        print(f"   手工特征时间范围: {len(manual_times)} 个样本")
                        print(f"   自动特征时间范围: {len(auto_times)} 个样本")
                        print(f"   重叠样本: {len(common_times)} 个")
                        
            except Exception as e:
                print(f"⚠️ 自动特征生成失败: {e}")
                print("   继续使用手工特征...")
        
        # 清理临时索引列并返回手工特征
        if 'time_idx' in manual_result.columns:
            manual_result = manual_result.drop('time_idx', axis=1)
        
        feature_count = len(manual_result.columns) - 1  # 排除close列
        print(f"✅ 特征生成完成，手工特征数量: {feature_count}")
        
        return manual_result

    def select_features(self, features_df: pd.DataFrame, final_k: int = 20,
                       variance_threshold: float = 0.01, correlation_threshold: float = 0.95,
                       importance_method: str = 'random_forest', train_ratio: float = 0.8) -> Dict:
        """
        统一的特征选择管道 - 整合了方差过滤、相关性去除和重要性选择
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            特征数据框，应包含目标列'close'
        final_k : int, default=20
            最终保留的特征数量
        variance_threshold : float, default=0.01
            方差阈值，低于此值的特征将被删除
        correlation_threshold : float, default=0.95
            相关性阈值，高于此值的特征对中保留一个
        importance_method : str, default='random_forest'
            重要性评估方法 ('random_forest' 或 'xgboost')
            
        Returns:
        --------
        dict
            包含各步骤结果的综合信息
        """
        print("🚀 开始综合特征选择管道...")
        # 动态计算原始特征数
        datetime_col = 'datetime' if 'datetime' in features_df.columns else None
        exclude_cols = ['close']
        if datetime_col:
            exclude_cols.append(datetime_col)
        original_feature_count = len(features_df.columns) - len(exclude_cols)
        
        print(f"   🎯 目标: 从 {original_feature_count} 个特征中选择 {final_k} 个")
        print("=" * 60)
        
        results = {
            'original_features': original_feature_count,
            'final_k': final_k,
            'pipeline_steps': []
        }
        
        current_df = features_df.copy()
        
        # 步骤0: 自动特征清洗（如果存在）
        auto_cols = [col for col in current_df.columns if col.startswith('auto_')]
        if auto_cols:
            print("🧽 步骤0: 自动特征清洗")
            original_auto_count = len(auto_cols)
            removed_features = []
            removal_reasons = {}
            
            for col in auto_cols:
                data = current_df[col]
                should_remove = False
                reason = ""
                
                # 1. 检查无穷值和NaN
                if not np.isfinite(data).all():
                    should_remove = True
                    reason = "包含无穷值或NaN"
                elif data.std() < 1e-10:
                    # 2. 检查常数特征
                    should_remove = True
                    reason = "常数特征（方差接近于0）"
                elif abs(data.mean()) > 1e10 or data.std() > 1e10:
                    # 3. 检查数值范围异常
                    should_remove = True
                    reason = f"数值范围异常（均值:{abs(data.mean()):.2e}, 标准差:{data.std():.2e}）"
                elif col.endswith('__sum_values'):
                    # 4. 移除结构性冗余特征（窗口求和）
                    should_remove = True
                    reason = "结构性冗余（窗口求和，与均值等价）"
                elif col.endswith('__variance') and any(c.endswith('__standard_deviation') and c.replace('__standard_deviation', '') == col.replace('__variance', '') for c in current_df.columns):
                    # 5. 如果同时存在variance和standard_deviation，保留后者
                    should_remove = True
                    reason = "已存在对应的standard_deviation特征"
                else:
                    # 6. 检查极端分布
                    try:
                        skew_val = data.skew()
                        kurt_val = data.kurtosis()
                        if abs(skew_val) > 15 or abs(kurt_val) > 200:
                            should_remove = True
                            reason = f"极端分布（偏度:{skew_val:.2f}, 峰度:{kurt_val:.2f}）"
                    except:
                        pass
                        
                if should_remove:
                    removed_features.append(col)
                    removal_reasons[col] = reason
            
            # 执行移除
            if removed_features:
                current_df = current_df.drop(columns=removed_features)
                remaining_auto_cols = [col for col in current_df.columns if col.startswith('auto_')]
                
                print(f"   ❌ 移除异常自动特征: {len(removed_features)}/{original_auto_count}")
                # 显示前3个被移除特征的原因
                for i, feature in enumerate(removed_features[:3]):
                    reason = removal_reasons.get(feature, "未知原因")
                    print(f"      {i+1}. {feature}: {reason}")
                if len(removed_features) > 3:
                    print(f"      ... 还有 {len(removed_features) - 3} 个")
                    
                print(f"   ✅ 保留有效自动特征: {len(remaining_auto_cols)}个")
                
                results['pipeline_steps'].append({
                    'step': 'auto_feature_cleaning',
                    'original_auto_features': original_auto_count,
                    'removed_features': removed_features,
                    'remaining_auto_features': len(remaining_auto_cols),
                    'removal_reasons': removal_reasons
                })
            else:
                print("   ✅ 所有自动特征都通过了质量检查")
                results['pipeline_steps'].append({
                    'step': 'auto_feature_cleaning',
                    'original_auto_features': original_auto_count,
                    'removed_features': [],
                    'remaining_auto_features': original_auto_count,
                    'removal_reasons': {}
                })
        
        # 步骤1: 方差阈值过滤
        print("\n🔸 步骤1: 方差阈值过滤")
        # 动态检查datetime列
        datetime_col = 'datetime' if 'datetime' in current_df.columns else None
        exclude_cols = ['close']
        if datetime_col:
            exclude_cols.append(datetime_col)
        feature_cols = [col for col in current_df.columns if col not in exclude_cols]
        
        if feature_cols:
            # 提取数值特征
            features_only = current_df[feature_cols].select_dtypes(include=[np.number])
            
            if not features_only.empty:
                # 应用方差阈值过滤
                selector = VarianceThreshold(threshold=variance_threshold)
                selector.fit(features_only.fillna(0))
                
                # 获取保留的特征
                selected_mask = selector.get_support()
                removed_features = [col for col, keep in zip(features_only.columns, selected_mask) if not keep]
                kept_features = [col for col, keep in zip(features_only.columns, selected_mask) if keep]
                
                # 构建结果DataFrame
                result_columns = ['close'] + kept_features
                if datetime_col:
                    result_columns = [datetime_col] + result_columns
                current_df = current_df[result_columns].copy()
                
                print(f"   📊 原始特征数: {len(feature_cols)}")
                print(f"   ❌ 删除低方差特征: {len(removed_features)}")
                print(f"   ✅ 保留特征数: {len(kept_features)}")
                
                results['pipeline_steps'].append({
                    'step': 'variance_filter',
                    'removed_features': removed_features,
                    'remaining_features': len(kept_features)
                })
            else:
                print("   ⚠️ 没有数值型特征，跳过方差过滤")
        else:
            print("   ⚠️ 没有特征列，跳过方差过滤")
        
        # 步骤2: 高共线性去除
        print("\n🔸 步骤2: 高共线性特征去除")
        # 动态检查datetime列
        datetime_col = 'datetime' if 'datetime' in current_df.columns else None
        exclude_cols = ['close']
        if datetime_col:
            exclude_cols.append(datetime_col)
        feature_cols = [col for col in current_df.columns if col not in exclude_cols]
        
        if len(feature_cols) >= 2:
            # 提取数值特征并计算相关矩阵
            features_only = current_df[feature_cols].select_dtypes(include=[np.number])
            
            if not features_only.empty and len(features_only.columns) >= 2:
                corr_matrix = features_only.corr().fillna(0)
                
                # 找出需要删除的高相关特征
                removed_features = []
                remaining_features = list(features_only.columns)
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if (corr_matrix.columns[i] in remaining_features and 
                            corr_matrix.columns[j] in remaining_features):
                            if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                                # 删除方差较小的特征
                                var_i = features_only[corr_matrix.columns[i]].var()
                                var_j = features_only[corr_matrix.columns[j]].var()
                                
                                if var_i < var_j:
                                    removed_features.append(corr_matrix.columns[i])
                                    remaining_features.remove(corr_matrix.columns[i])
                                else:
                                    removed_features.append(corr_matrix.columns[j])
                                    remaining_features.remove(corr_matrix.columns[j])
                
                result_columns = ['close'] + remaining_features
                if datetime_col:
                    result_columns = [datetime_col] + result_columns
                current_df = current_df[result_columns].copy()
                
                print(f"   📊 输入特征数: {len(feature_cols)}")
                print(f"   ❌ 删除高相关特征: {len(removed_features)}")
                print(f"   ✅ 保留特征数: {len(remaining_features)}")
                
                results['pipeline_steps'].append({
                    'step': 'correlation_filter',
                    'removed_features': removed_features,
                    'remaining_features': len(remaining_features)
                })
            else:
                print("   ⚠️ 数值特征不足，跳过共线性检查")
        else:
            print("   ⚠️ 特征数不足2个，跳过共线性检查")
        
        # 步骤3: 基于重要性的特征选择
        # 计算剩余特征数（排除close和可能的datetime列）
        datetime_col = 'datetime' if 'datetime' in current_df.columns else None
        exclude_cols = ['close']
        if datetime_col:
            exclude_cols.append(datetime_col)
        remaining_features = len(current_df.columns) - len(exclude_cols)
        
        if remaining_features > final_k:
            print(f"\n🔸 步骤3: 基于重要性选择Top-{final_k}特征 (防泄漏: train_ratio={train_ratio:.1%})")
            
            feature_cols = [col for col in current_df.columns if col not in exclude_cols]
            features_data = current_df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
            
            if not features_data.empty:
                # ========== 时间序列切分防止数据泄漏 ==========
                n_samples = len(features_data)
                split_idx = int(n_samples * train_ratio)
                
                # 确保训练集有足够样本
                if split_idx < 50:
                    print(f"   ⚠️ 训练样本过少({split_idx}<50)，跳过重要性选择")
                    feature_importance_dict = {}
                else:
                    print(f"   📊 时间切分: 训练集 {split_idx}/{n_samples} ({train_ratio:.1%})")
                    
                    # 只使用训练集计算特征重要性
                    train_features = features_data.iloc[:split_idx].copy()
                    train_close = current_df['close'].iloc[:split_idx]
                    
                    # 生成目标变量（只在训练集内）
                    importance_results = {}
                    combined_importance = pd.Series(0.0, index=train_features.columns)
                    
                    targets = {
                        'return_1d': train_close.pct_change().shift(-1),
                        'return_5d': train_close.pct_change(5).shift(-5), 
                        'return_10d': train_close.pct_change(10).shift(-10)
                    }
                    
                    valid_targets = 0
                    for target_name, target_values in targets.items():
                        try:
                            # 准备训练数据（确保目标值有效且不使用未来数据）
                            valid_mask = ~(target_values.isna() | train_features.isna().any(axis=1))
                            valid_count = valid_mask.sum()
                            
                            if valid_count < 30:  # 每个目标至少30个样本
                                print(f"     ⚠️ {target_name}: 有效样本不足({valid_count}<30)")
                                continue
                                
                            X_train = train_features[valid_mask]
                            y_train = target_values[valid_mask]
                            
                            # 选择模型
                            if importance_method == 'xgboost' and self.use_xgboost:
                                model = XGBRegressor(
                                    n_estimators=100, 
                                    max_depth=6,
                                    learning_rate=0.1,
                                    random_state=42, 
                                    verbosity=0
                                )
                            else:
                                model = RandomForestRegressor(
                                    n_estimators=100, 
                                    max_depth=10,
                                    min_samples_leaf=5,
                                    random_state=42, 
                                    n_jobs=-1
                                )
                            
                            # 训练模型并获取特征重要性
                            model.fit(X_train, y_train)
                            feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
                            
                            # 标准化重要性分数避免偏置
                            if feature_importance.sum() > 0:
                                feature_importance = feature_importance / feature_importance.sum()
                                importance_results[target_name] = feature_importance
                                combined_importance += feature_importance
                                valid_targets += 1
                                print(f"     ✅ {target_name}: {valid_count}样本, Top特征: {feature_importance.nlargest(3).index.tolist()}")
                            
                        except Exception as e:
                            print(f"     ❌ {target_name}: 计算失败 - {str(e)}")
                            continue
                
                    if valid_targets > 0 and combined_importance.sum() > 0:
                        # 选择top-k特征（基于无泄漏的重要性分数）
                        top_features = combined_importance.nlargest(final_k).index.tolist()
                        
                        # 构建结果DataFrame（应用到全量数据但不重新训练）
                        result_columns = ['close'] + top_features
                        if datetime_col:
                            result_columns = [datetime_col] + result_columns
                        current_df = current_df[result_columns].copy()
                        
                        print(f"   📊 输入特征数: {remaining_features}")
                        print(f"   ✅ 选择特征数: {len(top_features)}")
                        print(f"   🎯 有效目标数: {valid_targets}/3")
                        print(f"   🏆 Top-5特征: {top_features[:5]}")
                        
                        # 保存特征重要性用于返回
                        feature_importance_dict = dict(combined_importance.nlargest(final_k))
                        
                        results['pipeline_steps'].append({
                            'step': 'importance_selection',
                            'method': importance_method,
                            'train_ratio': train_ratio,
                            'train_samples': split_idx,
                            'valid_targets': valid_targets,
                            'selected_features': top_features,
                            'feature_importance': feature_importance_dict
                        })
                    else:
                        print("   ⚠️ 无有效目标或重要性计算失败，保持当前特征")
                        feature_importance_dict = {}
            else:
                print("   ⚠️ 没有有效的数值特征")
        else:
            print(f"\n✅ 当前特征数({remaining_features})已满足目标，跳过重要性选择")
            feature_importance_dict = {}
            results['pipeline_steps'].append({
                'step': 'importance_selection',
                'skipped': True,
                'reason': f'features_count({remaining_features}) <= target({final_k})',
                'remaining_features': remaining_features
            })
        
        # 最终结果
        datetime_col = 'datetime' if 'datetime' in current_df.columns else None
        exclude_cols = ['close']
        if datetime_col:
            exclude_cols.append(datetime_col)
        final_features = [col for col in current_df.columns if col not in exclude_cols]
        results.update({
            'final_features_df': current_df,
            'final_features': final_features,
            'final_features_count': len(final_features),
            'feature_importance': feature_importance_dict,
            'reduction_ratio': (results['original_features'] - len(final_features)) / results['original_features']
        })
        
        print("\n" + "=" * 60)
        print("🎉 综合特征选择管道完成!")
        print(f"   📊 原始特征数: {results['original_features']}")
        print(f"   ✅ 最终特征数: {len(final_features)}")
        print(f"   📉 特征削减率: {results['reduction_ratio']:.1%}")
        
        # 显示管道步骤统计
        if results['pipeline_steps']:
            auto_clean_step = next((s for s in results['pipeline_steps'] if s['step'] == 'auto_feature_cleaning'), None)
            if auto_clean_step and auto_clean_step['removed_features']:
                print(f"   🧽 自动特征清洗: 移除 {len(auto_clean_step['removed_features'])} 个")
        
        if final_features:
            print(f"   🏆 最终Top-10特征: {final_features[:10]}")
        
        return results

    def scale_features(self, features_df: pd.DataFrame, 
                       scaler_type: str = 'robust',
                       train_ratio: float = 0.8,
                       save_path: str = 'scaler.pkl',
                       exclude_cols: Optional[List[str]] = None) -> Dict:
        """
        对特征做尺度标准化（时间序列防泄漏：仅用训练段 fit，其余段 transform）
        
        Parameters
        ----------
        features_df : pd.DataFrame
            已完成特征选择的特征数据，包含 'close'
        scaler_type : str, default 'robust'
            缩放方式: 'robust' | 'standard' | 'minmax'
        train_ratio : float, default 0.8
            训练集比例（时间切分）
        save_path : str
            持久化缩放器路径（pickle）
        exclude_cols : list
            不参与缩放的列（默认: ['close'] + datetime）
        
        Returns
        -------
        dict:
            {
              'scaled_df': 缩放后的数据（保持原索引与列顺序）
              'scaler': 已拟合缩放器对象
              'train_index': 训练区间索引
              'valid_index': 验证/未来区间索引
              'feature_cols': 实际缩放的特征列
              'scaler_path': 保存路径
            }
        """
        print("📏 开始特征标准化...")
        
        df = features_df.copy()
        if df.empty:
            raise ValueError("scale_features: 输入的特征数据为空")
        
        # 识别排除列
        datetime_col = 'datetime' if 'datetime' in df.columns else None
        if exclude_cols is None:
            exclude_cols = ['close']
            if datetime_col:
                exclude_cols.append(datetime_col)
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        if not feature_cols:
            raise ValueError("scale_features: 没有可缩放的特征列")
        
        # 时间切分（保持与特征选择一致的逻辑）
        n_samples = len(df)
        split_idx = int(n_samples * train_ratio)
        if split_idx < 30:
            raise ValueError(f"scale_features: 训练集样本过少({split_idx})，无法拟合缩放器")
        
        train_index = df.index[:split_idx]
        valid_index = df.index[split_idx:]
        
        print(f"   📊 时间切分: 训练集 {split_idx}/{n_samples} ({train_ratio:.1%})")
        print(f"   📅 训练段: {train_index.min().date()} ~ {train_index.max().date()}")
        if len(valid_index) > 0:
            print(f"   📅 验证段: {valid_index.min().date()} ~ {valid_index.max().date()}")
        
        train_X = df.loc[train_index, feature_cols]
        valid_X = df.loc[valid_index, feature_cols] if len(valid_index) > 0 else None
        
        # 选择缩放器
        if scaler_type == 'robust':
            scaler = RobustScaler()
            print(f"   🔧 使用 RobustScaler (中位数-IQR标准化，适合金融数据)")
        elif scaler_type == 'standard':
            scaler = StandardScaler()
            print(f"   🔧 使用 StandardScaler (均值-标准差标准化)")
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
            print(f"   🔧 使用 MinMaxScaler (最小-最大值标准化)")
        else:
            raise ValueError("scaler_type 必须是 'robust' | 'standard' | 'minmax'")
        
        # 拟合 + 变换（只在训练集上拟合）
        print(f"   🎯 在训练集上拟合缩放器...")
        scaler.fit(train_X.fillna(0))  # 处理可能的缺失值
        scaled_train = scaler.transform(train_X.fillna(0))
        
        if valid_X is not None:
            print(f"   🔄 对验证集进行变换...")
            scaled_valid = scaler.transform(valid_X.fillna(0))
        
        # 回填结果（保持原有的列结构和索引）
        scaled_df = df.copy()
        scaled_df.loc[train_index, feature_cols] = scaled_train
        if valid_X is not None:
            scaled_df.loc[valid_index, feature_cols] = scaled_valid
        
        # 持久化缩放器和元数据
        try:
            # 保存缩放器和元数据
            scaler_data = {
                'scaler': scaler,
                'feature_cols': feature_cols,
                'scaler_type': scaler_type,
                'train_ratio': train_ratio,
                'train_samples': split_idx,
                'total_samples': n_samples,
                'train_range': (str(train_index.min().date()), str(train_index.max().date())),
                'valid_range': (str(valid_index.min().date()), str(valid_index.max().date())) if len(valid_index) > 0 else None,
                'fit_timestamp': datetime.now().isoformat(),
                'feature_count': len(feature_cols)
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(scaler_data, f)
            
            # 另外保存一个可读的元数据文件
            meta_path = save_path.replace('.pkl', '_meta.json')
            readable_meta = {k: v for k, v in scaler_data.items() if k != 'scaler'}  # 排除不能JSON序列化的scaler对象
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(readable_meta, f, indent=2, ensure_ascii=False)
                
            print(f"   ✅ 缩放器已保存: {save_path}")
            print(f"   📋 元数据已保存: {meta_path}")
            
        except Exception as e:
            print(f"   ⚠️ 缩放器保存失败: {e}")
        
        # 计算缩放前后的统计信息
        original_stats = train_X.describe()
        scaled_stats = pd.DataFrame(scaled_train, columns=feature_cols).describe()
        
        print(f"\n📊 缩放效果统计:")
        print(f"   🔢 缩放特征数: {len(feature_cols)}")
        print(f"   📈 原始数据范围: 均值 [{original_stats.loc['mean'].min():.4f}, {original_stats.loc['mean'].max():.4f}]")
        print(f"   📉 缩放后范围: 均值 [{scaled_stats.loc['mean'].min():.4f}, {scaled_stats.loc['mean'].max():.4f}]")
        print(f"   📊 原始标准差: [{original_stats.loc['std'].min():.4f}, {original_stats.loc['std'].max():.4f}]")
        print(f"   📊 缩放后标准差: [{scaled_stats.loc['std'].min():.4f}, {scaled_stats.loc['std'].max():.4f}]")
        
        # 显示缩放最剧烈的特征
        original_ranges = original_stats.loc['max'] - original_stats.loc['min']
        scaled_ranges = scaled_stats.loc['max'] - scaled_stats.loc['min']
        scale_ratios = original_ranges / (scaled_ranges + 1e-8)
        top_scaled_features = scale_ratios.nlargest(5)
        
        print(f"\n🎯 缩放效果最明显的特征:")
        for i, (feature, ratio) in enumerate(top_scaled_features.items(), 1):
            orig_range = original_ranges[feature]
            scaled_range = scaled_ranges[feature]
            print(f"   {i}. {feature}: {orig_range:.4f} → {scaled_range:.4f} (压缩 {ratio:.1f}x)")
        
        return {
            'scaled_df': scaled_df,
            'scaler': scaler,
            'train_index': train_index,
            'valid_index': valid_index,
            'feature_cols': feature_cols,
            'scaler_path': save_path,
            'meta_path': meta_path,
            'scaler_type': scaler_type,
            'train_samples': split_idx,
            'feature_count': len(feature_cols)
        }

    def analyze_features(self, features_df: pd.DataFrame) -> Dict:
        """
        分析特征分布和质量（应在特征选择之后使用）
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            特征数据（已经过特征选择的数据）
            
        Returns:
        --------
        dict
            特征分析结果
        """
        print("📊 开始特征分析...")
        
        # 首先展示特征数据预览
        self._display_feature_preview(features_df, "分析特征")
        
        # 动态检测datetime列
        datetime_col = 'datetime' if 'datetime' in features_df.columns else None
        exclude_cols = ['close']
        if datetime_col:
            exclude_cols.append(datetime_col)
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        analysis = {
            'total_features': len(feature_cols),
            'missing_values': {},
            'extreme_values': {},
            'distributions': {}
        }
        
        # 缺失值分析
        for col in feature_cols:
            missing_count = features_df[col].isnull().sum()
            missing_pct = missing_count / len(features_df) * 100
            if missing_count > 0:
                analysis['missing_values'][col] = {
                    'count': missing_count,
                    'percentage': missing_pct
                }
        
        # 极值分析
        for col in feature_cols:
            if features_df[col].dtype in ['float64', 'int64']:
                q1 = features_df[col].quantile(0.25)
                q3 = features_df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = ((features_df[col] < lower_bound) | (features_df[col] > upper_bound)).sum()
                if outliers > 0:
                    analysis['extreme_values'][col] = {
                        'count': outliers,
                        'percentage': outliers / len(features_df) * 100,
                        'bounds': (lower_bound, upper_bound)
                    }
        
        # 分布统计
        numeric_features = features_df[feature_cols].select_dtypes(include=[np.number])
        analysis['distributions'] = {
            'mean': numeric_features.mean().to_dict(),
            'std': numeric_features.std().to_dict(),
            'skewness': numeric_features.skew().to_dict(),
            'kurtosis': numeric_features.kurtosis().to_dict()
        }
        
        # 输出分析结果
        print("\n📈 特征分析结果:")
        print(f"   🔢 总特征数: {analysis['total_features']}")
        print(f"   ❌ 缺失值特征: {len(analysis['missing_values'])}")
        print(f"   ⚠️ 异常值特征: {len(analysis['extreme_values'])}")
        
        if analysis['missing_values']:
            print("   📋 缺失值详情:")
            for col, info in analysis['missing_values'].items():
                print(f"      {col}: {info['count']} ({info['percentage']:.1f}%)")
        
        if analysis['extreme_values']:
            print("   📋 异常值详情:")
            for col, info in list(analysis['extreme_values'].items())[:5]:  # 显示前5个
                print(f"      {col}: {info['count']} ({info['percentage']:.1f}%)")
        

        
        # 添加更多统计信息
        print(f"\n📊 整体数据质量评估:")
        total_cells = len(features_df) * len(feature_cols)
        missing_cells = sum(analysis['missing_values'][col]['count'] for col in analysis['missing_values'])
        data_completeness = (total_cells - missing_cells) / total_cells * 100 if total_cells > 0 else 100
        print(f"   📊 数据完整性: {data_completeness:.1f}%")
        print(f"   📈 数据范围: {features_df.index.min().date()} ~ {features_df.index.max().date()}")
        print(f"   📅 数据点数: {len(features_df)} 个时间点")
        
        return analysis
    
    def _display_feature_preview(self, features_df: pd.DataFrame, data_type: str = "特征"):
        """
        智能展示特征数据预览
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            特征数据框
        data_type : str
            数据类型描述
        """
        print(f"\n📊 {data_type}数据预览:")
        print("=" * 50)
        
        # 基本信息
        print(f"📈 数据形状: {features_df.shape[0]} 行 × {features_df.shape[1]} 列")
        print(f"📅 时间范围: {features_df.index.min().date()} 到 {features_df.index.max().date()}")
        
        # 特征列（排除close列）
        feature_cols = [col for col in features_df.columns if col != 'close']
        print(f"🔢 特征数量: {len(feature_cols)}")
        
        # 数据质量概览
        missing_info = features_df.isnull().sum()
        missing_features = missing_info[missing_info > 0]
        print(f"❌ 缺失值特征: {len(missing_features)} 个")
        
        # 展示策略：根据特征数量决定展示详细程度
        if len(feature_cols) <= 10:
            # 特征较少：展示所有特征的详细信息
            print("\n📋 所有特征详情:")
            for i, col in enumerate(feature_cols, 1):
                stats = features_df[col].describe()
                missing_pct = (features_df[col].isnull().sum() / len(features_df)) * 100
                print(f"  {i:2d}. {col:<25} | 均值:{stats['mean']:8.4f} | 标准差:{stats['std']:8.4f} | 缺失:{missing_pct:5.1f}%")
                
        elif len(feature_cols) <= 30:
            # 特征适中：展示前10个特征的详细信息 + 统计概览
            print("\n📋 前10个特征详情:")
            for i, col in enumerate(feature_cols[:10], 1):
                stats = features_df[col].describe()
                missing_pct = (features_df[col].isnull().sum() / len(features_df)) * 100
                print(f"  {i:2d}. {col:<25} | 均值:{stats['mean']:8.4f} | 标准差:{stats['std']:8.4f} | 缺失:{missing_pct:5.1f}%")
            
            if len(feature_cols) > 10:
                print(f"  ... 还有 {len(feature_cols) - 10} 个特征")
                
        else:
            # 特征很多：只展示统计概览和特征名称分类
            print("\n📋 特征统计概览:")
            numeric_features = features_df[feature_cols].select_dtypes(include=[np.number])
            overall_stats = numeric_features.describe().T
            
            print(f"  平均值范围: {overall_stats['mean'].min():.4f} ~ {overall_stats['mean'].max():.4f}")
            print(f"  标准差范围: {overall_stats['std'].min():.4f} ~ {overall_stats['std'].max():.4f}")
            print(f"  最小值范围: {overall_stats['min'].min():.4f} ~ {overall_stats['min'].max():.4f}")
            print(f"  最大值范围: {overall_stats['max'].min():.4f} ~ {overall_stats['max'].max():.4f}")
            
            # 特征分类展示
            print("\n🏷️  特征分类:")
            
            categories = {
                '收益率特征': [col for col in feature_cols if 'return' in col],
                '动量特征': [col for col in feature_cols if 'momentum' in col],
                '滚动统计': [col for col in feature_cols if 'rolling' in col],
                '波动率特征': [col for col in feature_cols if any(x in col for x in ['volatility', 'atr', 'skewness', 'kurtosis'])],
                '成交量特征': [col for col in feature_cols if 'volume' in col],
                '价格特征': [col for col in feature_cols if any(x in col for x in ['price', 'high', 'low', 'open', 'ratio'])],
                '技术指标': [col for col in feature_cols if any(x in col for x in ['rsi', 'bb', 'macd'])],
                '自动特征': [col for col in feature_cols if col.startswith('auto_')],
                '其他特征': [col for col in feature_cols if not any([
                    'return' in col, 'momentum' in col, 'rolling' in col,
                    any(x in col for x in ['volatility', 'atr', 'skewness', 'kurtosis']),
                    'volume' in col, any(x in col for x in ['price', 'high', 'low', 'open', 'ratio']),
                    any(x in col for x in ['rsi', 'bb', 'macd']), col.startswith('auto_')
                ])]
            }
            
            for category, features in categories.items():
                if features:
                    print(f"  📌 {category} ({len(features)}个): {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")
        
        # 数据样本预览（前5行，重要列）
        print("\n📄 数据样本预览（前5行）:")
        preview_cols = ['close'] + feature_cols[:min(6, len(feature_cols))]  # close + 最多6个特征
        preview_data = features_df[preview_cols].head()
        
        # 格式化显示
        pd.set_option('display.max_columns', 10)
        pd.set_option('display.width', 120)
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(preview_data)
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.float_format')
        
        # 保存到文件提示
        if len(features_df) > 100 or len(feature_cols) > 20:
            print("\n💾 提示: 数据较大，可使用 features_df.to_csv('features.csv') 保存完整数据")
            
        print("=" * 50)
    



if __name__ == "__main__":
    """
    使用示例 - 仅用于演示真实数据加载和特征工程
    """
    print("🎯 股票特征工程 - 真实数据版本")
    print("=" * 50)
    
    try:
        # 初始化特征工程器
        engineer = FeatureEngineer(use_talib=True, use_tsfresh=True)
        
        # 示例：加载平安银行(000001)最近1年的数据
        symbol = "000001"
        start_date = "2023-01-01" 
        end_date = "2024-12-31"
        
        # 加载真实数据
        data = engineer.load_stock_data(symbol, start_date, end_date)
        
        if len(data) < 100:
            print("⚠️ 数据量不足100天，建议扩大时间范围")
        
        # 生成特征
        print("\n🏭 生成技术特征...")
        features_df = engineer.prepare_features(
            data, 
            use_auto_features=True,  # 启用自动特征生成
            window_size=20,
            max_auto_features=30  # 限制自动特征数量
        )
        total_features = features_df.shape[1] - 1
        manual_count = len([col for col in features_df.columns if not col.startswith('auto_') and col != 'close'])
        auto_count = len([col for col in features_df.columns if col.startswith('auto_')])
        print(f"✅ 成功生成 {total_features} 个特征 (手工:{manual_count} + 自动:{auto_count})")
        
        # 特征选择
        print("\n🎯 执行特征选择...")
        selection_results = engineer.select_features(
            features_df,
            final_k=20,
            variance_threshold=0.01,
            correlation_threshold=0.9,
            train_ratio=0.8  # 只用80%的历史数据计算特征重要性，防止数据泄漏
        )
        
        final_features = selection_results['final_features']
        print(f"✅ 最终选择 {len(final_features)} 个重要特征")
        
        # 特征标准化（新增步骤）
        print("\n� 执行特征标准化...")
        scale_results = engineer.scale_features(
            selection_results['final_features_df'],
            scaler_type='robust',  # 金融数据推荐使用RobustScaler
            train_ratio=0.8,       # 与特征选择保持一致的时间切分
            save_path='feature_scaler.pkl'
        )
        scaled_df = scale_results['scaled_df']
        print(f"✅ 特征标准化完成，缩放器已保存到 {scale_results['scaler_path']}")
        
        # 特征分析（使用标准化后的数据）
        print("\n📊 分析标准化后的特征质量...")
        analysis = engineer.analyze_features(scaled_df)
        
        print(f"\n📋 处理完成！")
        print(f"   🔢 原始数据: {len(data)} 天")
        print(f"   🏭 生成特征: {features_df.shape[1]-1} 个")
        print(f"   🎯 最终特征: {len(final_features)} 个")
        print(f"   � 标准化特征: {scale_results['feature_count']} 个")
        print(f"   �📊 特征质量: {analysis['total_features'] - len(analysis['missing_values'])} 个无缺失值")
        
        print("\n💡 使用说明:")
        print("   1. engineer.load_stock_data() - 加载真实股票数据")
        print("   2. engineer.prepare_features() - 生成技术特征")
        print("   3. engineer.select_features() - 执行特征选择")
        print("   4. engineer.scale_features() - 特征标准化（防泄漏）")
        print("   5. engineer.analyze_features() - 分析特征质量")
        
        print(f"\n💾 输出文件:")
        print(f"   📦 特征缩放器: {scale_results['scaler_path']}")
        print(f"   📋 缩放元数据: {scale_results['meta_path']}")
        print("   📊 可用 scaled_df.to_csv('scaled_features.csv') 保存标准化特征")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("请检查:")
        print("  1. InfluxDB服务是否运行")
        print("  2. 网络连接是否正常")
        print("  3. 股票代码和日期范围是否正确")