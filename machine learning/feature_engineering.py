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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class FeatureEngineer:
    """
    股票特征工程类 - 使用真实InfluxDB数据
    
    功能:
    1. 从InfluxDB加载真实历史股票数据
    2. 生成技术分析特征
    3. 特征选择和优化 
    4. 特征质量分析
    """
    
    def __init__(self, use_talib: bool = True, use_tsfresh: bool = True):
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
                    
                    # 提取特征
                    from tsfresh.feature_extraction import MinimalFCParameters
                    extracted_features = extract_features(
                        tsfresh_df,
                        column_id='id',
                        column_sort='time',
                        column_value='value',
                        default_fc_parameters=MinimalFCParameters()
                    )
                    
                    # 选择最重要的特征
                    if len(extracted_features.columns) > max_auto_features:
                        feature_vars = extracted_features.var()
                        selected_features = feature_vars.nlargest(max_auto_features).index
                        extracted_features = extracted_features[selected_features]
                    
                    # 创建自动特征结果数据框
                    result_indices = range(window_size, len(clean_data))
                    auto_result = pd.DataFrame({
                        'close': clean_data.iloc[result_indices]['close'].values,
                        'time_idx': result_indices  # 使用相同的时间索引
                    })
                    
                    # 添加自动特征
                    for col in extracted_features.columns:
                        auto_result[f'auto_{col}'] = extracted_features[col].values
                    
                    auto_result = auto_result.dropna()
                    print(f"   ✅ 自动特征生成完成，特征数量: {len(extracted_features.columns)}")
                    
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
                        
                        feature_count = len(combined_result.columns) - 1  # 排除close列
                        print(f"✅ 特征合并完成，共享样本: {len(common_times)}, 总特征数量: {feature_count}")
                        
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
        print(f"✅ 手工特征生成完成，特征数量: {feature_count}")
        
        return manual_result

    def select_features(self, features_df: pd.DataFrame, final_k: int = 20,
                       variance_threshold: float = 0.01, correlation_threshold: float = 0.95,
                       importance_method: str = 'random_forest') -> Dict:
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
        
        # 步骤1: 方差阈值过滤
        print("🔸 步骤1: 方差阈值过滤")
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
            print(f"\n🔸 步骤3: 基于重要性选择Top-{final_k}特征")
            
            feature_cols = [col for col in current_df.columns if col not in exclude_cols]
            features_data = current_df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
            
            if not features_data.empty:
                # 生成多个预测目标（不同时间跨度的收益率）
                importance_results = {}
                combined_importance = pd.Series(0.0, index=features_data.columns)
                
                # 为不同的预测目标计算特征重要性
                targets = {
                    'return_1d': current_df['close'].pct_change().shift(-1),
                    'return_5d': current_df['close'].pct_change(5).shift(-5),
                    'return_10d': current_df['close'].pct_change(10).shift(-10)
                }
                
                for target_name, target_values in targets.items():
                    try:
                        # 准备训练数据
                        valid_indices = ~(target_values.isna() | features_data.isna().any(axis=1))
                        if valid_indices.sum() < 50:  # 至少需要50个样本
                            continue
                            
                        X = features_data[valid_indices]
                        y = target_values[valid_indices]
                        
                        # 选择模型
                        if importance_method == 'xgboost' and self.use_xgboost:
                            model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
                        else:
                            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                        
                        # 训练模型并获取特征重要性
                        model.fit(X, y)
                        feature_importance = pd.Series(model.feature_importances_, index=X.columns)
                        importance_results[target_name] = feature_importance
                        
                        # 累加重要性（用于综合排名）
                        combined_importance += feature_importance
                        
                    except Exception as e:
                        continue
                
                if importance_results:
                    # 选择top-k特征
                    top_features = combined_importance.nlargest(final_k).index.tolist()
                    
                    # 构建结果DataFrame
                    result_columns = ['close'] + top_features
                    if datetime_col:
                        result_columns = [datetime_col] + result_columns
                    current_df = current_df[result_columns].copy()
                    
                    print(f"   📊 输入特征数: {remaining_features}")
                    print(f"   ✅ 选择特征数: {final_k}")
                    print(f"   🏆 Top-5特征: {top_features[:5]}")
                    
                    # 保存特征重要性用于返回
                    feature_importance_dict = dict(combined_importance.nlargest(final_k))
                    
                    results['pipeline_steps'].append({
                        'step': 'importance_selection',
                        'method': importance_method,
                        'selected_features': top_features,
                        'feature_importance': feature_importance_dict
                    })
                else:
                    print("   ⚠️ 重要性计算失败，保持当前特征")
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
        if final_features:
            print(f"   🏆 最终Top-10特征: {final_features[:10]}")
        
        return results

    def analyze_features(self, features_df: pd.DataFrame, plot: bool = True) -> Dict:
        """
        分析特征分布和质量（应在特征选择之后使用）
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            特征数据（已经过特征选择的数据）
        plot : bool, default=True
            是否绘制分析图表
            
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
        
        # 绘图分析
        if plot and len(numeric_features.columns) > 0:
            try:
                self._plot_feature_analysis(numeric_features)
            except Exception as e:
                print(f"⚠️ 绘图功能不可用: {str(e)}")
                print("推荐使用 features_df.to_csv('特征数据.csv') 保存数据后在Excel中查看")
        
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
            self._categorize_features(feature_cols)
        
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
    
    def _categorize_features(self, feature_cols: list):
        """
        将特征按类型分类展示
        """
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
    
    def _plot_feature_analysis(self, features_df: pd.DataFrame, max_plots: int = 12):
        """绘制特征分析图表"""
        if not HAS_MATPLOTLIB:
            print("⚠️ matplotlib 未安装，无法显示图表")
            print("💾 建议使用: pip install matplotlib")
            return
            
        try:
            import matplotlib.pyplot as plt
            
            # 设置中文显示
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            n_features = min(len(features_df.columns), max_plots)
            if n_features == 0:
                print("⚠️ 没有数值特征可以绘制")
                return
                
            # 计算子图布局
            rows = (n_features + 3) // 4  # 每行4个子图
            cols = min(4, n_features)
            
            fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes
            else:
                axes = axes.ravel()
            
            print(f"📈 正在生成 {n_features} 个特征的分布图...")
            
            for i, col in enumerate(features_df.columns[:n_features]):
                try:
                    # 计算有效数据
                    valid_data = features_df[col].dropna()
                    if len(valid_data) == 0:
                        axes[i].text(0.5, 0.5, f'{col}\n无有效数据', 
                                   ha='center', va='center', transform=axes[i].transAxes)
                        continue
                    
                    # 绘制直方图
                    n_bins = min(30, max(10, len(valid_data) // 10))
                    axes[i].hist(valid_data, bins=n_bins, alpha=0.7, edgecolor='black', color='skyblue')
                    axes[i].set_title(f'{col}\n均值:{valid_data.mean():.3f}, 标准差:{valid_data.std():.3f}', fontsize=10)
                    axes[i].tick_params(labelsize=8)
                    axes[i].grid(True, alpha=0.3)
                    
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'{col}\n绘图失败: {str(e)[:20]}', 
                               ha='center', va='center', transform=axes[i].transAxes)
            
            # 隐藏多余的子图
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.suptitle(f'特征分布分析 (Top-{n_features})', fontsize=14, y=0.98)
            
            # 显示图表
            print("📈 正在显示特征分布图...")
            plt.show()
            
        except ImportError:
            print("⚠️ matplotlib 导入失败")
        except Exception as e:
            print(f"⚠️ 绘图过程中出现错误: {str(e)}")
            print("💾 建议保存数据到CSV文件后在其他工具中查看")


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
        features_df = engineer.prepare_features(data, use_auto_features=True)
        print(f"✅ 成功生成 {features_df.shape[1]-1} 个特征")
        
        # 特征选择
        print("\n🎯 执行特征选择...")
        selection_results = engineer.select_features(
            features_df,
            final_k=20,
            variance_threshold=0.01,
            correlation_threshold=0.9
        )
        
        final_features = selection_results['final_features']
        print(f"✅ 最终选择 {len(final_features)} 个重要特征")
        
        # 特征分析
        print("\n📊 分析特征质量...")
        analysis = engineer.analyze_features(selection_results['final_features_df'], plot=True)
        
        print(f"\n📋 处理完成！")
        print(f"   🔢 原始数据: {len(data)} 天")
        print(f"   🏭 生成特征: {features_df.shape[1]-1} 个")
        print(f"   🎯 最终特征: {len(final_features)} 个")
        print(f"   📊 特征质量: {analysis['total_features'] - len(analysis['missing_values'])} 个无缺失值")
        
        print("\n💡 使用说明:")
        print("   1. engineer.load_stock_data() - 加载真实股票数据")
        print("   2. engineer.prepare_features() - 生成技术特征")
        print("   3. engineer.select_features() - 执行特征选择")
        print("   4. engineer.analyze_features() - 分析特征质量")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("请检查:")
        print("  1. InfluxDB服务是否运行")
        print("  2. 网络连接是否正常")
        print("  3. 股票代码和日期范围是否正确")