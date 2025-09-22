"""
特征工程模块 - 股票数据特征生成
包含手工特征和自动特征生成功能
"""
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 可选依赖导入
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ talib 未安装，部分技术指标将使用pandas实现")

try:
    import tsfresh
    from tsfresh import extract_features
    from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters
    from tsfresh.utilities.dataframe_functions import impute
    TSFRESH_AVAILABLE = True
except ImportError:
    TSFRESH_AVAILABLE = False
    print("⚠️ tsfresh 未安装，自动特征生成不可用")

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    特征工程类 - 股票数据特征生成
    支持手工特征和自动特征生成
    """
    
    def __init__(self, use_talib: bool = True, use_tsfresh: bool = True):
        """
        初始化特征工程器
        
        Parameters:
        -----------
        use_talib : bool, default=True
            是否使用talib库计算技术指标
        use_tsfresh : bool, default=True
            是否启用tsfresh自动特征生成
        """
        self.use_talib = use_talib and TALIB_AVAILABLE
        self.use_tsfresh = use_tsfresh and TSFRESH_AVAILABLE
        self.scaler = None
        
        print(f"🔧 特征工程器初始化完成")
        print(f"   📊 TA-Lib: {'✅' if self.use_talib else '❌'}")
        print(f"   🤖 TSFresh: {'✅' if self.use_tsfresh else '❌'}")
    
    def prepare_manual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成手工基础特征（可解释特征）
        
        Parameters:
        -----------
        df : pd.DataFrame
            包含OHLCV数据的DataFrame
            
        Returns:
        --------
        pd.DataFrame
            包含手工特征的DataFrame
        """
        print("🔨 开始生成手工基础特征...")
        data = df.copy()
        
        # 1. 收益率特征 (Returns)
        print("   📈 计算收益率特征...")
        data['return_1d'] = data['close'].pct_change()
        data['return_5d'] = data['close'].pct_change(5)
        data['return_10d'] = data['close'].pct_change(10)
        data['return_20d'] = data['close'].pct_change(20)
        
        # 2. 滚动统计特征 (Rolling Statistics)
        print("   📊 计算滚动统计特征...")
        windows = [5, 10, 20, 30]
        for window in windows:
            # 滚动均值
            data[f'rolling_mean_{window}d'] = data['close'].rolling(window).mean()
            # 滚动标准差
            data[f'rolling_std_{window}d'] = data['close'].rolling(window).std()
            # 滚动中位数
            data[f'rolling_median_{window}d'] = data['close'].rolling(window).median()
            # 价格相对位置
            data[f'price_position_{window}d'] = (data['close'] - data[f'rolling_mean_{window}d']) / data[f'rolling_std_{window}d']
        
        # 3. 动量特征 (Momentum)
        print("   🚀 计算动量特征...")
        momentum_periods = [3, 5, 10, 20]
        for period in momentum_periods:
            data[f'momentum_{period}d'] = (data['close'] / data['close'].shift(period)) - 1
        
        # 4. ATR和波动率特征 (Volatility)
        print("   📊 计算波动率特征...")
        if self.use_talib:
            data['atr_14'] = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
        else:
            # 手工计算ATR
            data['tr1'] = data['high'] - data['low']
            data['tr2'] = abs(data['high'] - data['close'].shift(1))
            data['tr3'] = abs(data['low'] - data['close'].shift(1))
            data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
            data['atr_14'] = data['true_range'].rolling(14).mean()
            data.drop(['tr1', 'tr2', 'tr3', 'true_range'], axis=1, inplace=True)
        
        # 价格波动率
        data['volatility_5d'] = data['return_1d'].rolling(5).std()
        data['volatility_20d'] = data['return_1d'].rolling(20).std()
        
        # 偏度和峰度
        data['skewness_20d'] = data['return_1d'].rolling(20).skew()
        data['kurtosis_20d'] = data['return_1d'].rolling(20).kurt()
        
        # 5. 成交量特征 (Volume Features)
        print("   💰 计算成交量特征...")
        data['volume_change'] = data['volume'].pct_change()
        data['volume_ma_5'] = data['volume'].rolling(5).mean()
        data['volume_ma_20'] = data['volume'].rolling(20).mean()
        data['volume_ratio_5d'] = data['volume'] / data['volume_ma_5']
        data['volume_ratio_20d'] = data['volume'] / data['volume_ma_20']
        
        # 成交量变化率
        volume_periods = [3, 5, 10]
        for period in volume_periods:
            data[f'volume_roc_{period}d'] = (data['volume'] / data['volume'].shift(period)) - 1
        
        # 6. 价格范围特征 (Price Range)
        print("   📏 计算价格范围特征...")
        data['high_low_ratio'] = data['high'] / data['low']
        data['high_close_ratio'] = data['high'] / data['close']
        data['low_close_ratio'] = data['low'] / data['close']
        data['open_close_ratio'] = data['close'] / data['open']
        
        # 价格范围相对化
        data['price_range'] = data['high'] - data['low']
        data['price_range_pct'] = data['price_range'] / data['close']
        data['open_close_range'] = abs(data['close'] - data['open'])
        data['open_close_range_pct'] = data['open_close_range'] / data['close']
        
        # 7. 技术指标特征 (Technical Indicators)
        print("   🔍 计算技术指标特征...")
        
        # RSI
        if self.use_talib:
            data['rsi_14'] = talib.RSI(data['close'].values, timeperiod=14)
        else:
            data['rsi_14'] = self._calculate_rsi(data['close'], window=14)
        
        # 布林带
        if self.use_talib:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(data['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
            data['bb_upper'] = bb_upper
            data['bb_middle'] = bb_middle
            data['bb_lower'] = bb_lower
        else:
            bb_window = 20
            bb_std = 2
            bb_ma = data['close'].rolling(bb_window).mean()
            bb_std_val = data['close'].rolling(bb_window).std()
            data['bb_upper'] = bb_ma + (bb_std_val * bb_std)
            data['bb_middle'] = bb_ma
            data['bb_lower'] = bb_ma - (bb_std_val * bb_std)
        
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # MACD
        if self.use_talib:
            macd, macd_signal, macd_hist = talib.MACD(data['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
            data['macd'] = macd
            data['macd_signal'] = macd_signal
            data['macd_hist'] = macd_hist
        else:
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            data['macd'] = ema_12 - ema_26
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # 清理无用列并去除缺失值
        feature_columns = [col for col in data.columns 
                          if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'open_interest']]
        
        result = data[['datetime', 'close'] + feature_columns].dropna()
        
        print(f"✅ 手工特征生成完成，特征数量: {len(feature_columns)}")
        return result
    
    def prepare_auto_features(self, df: pd.DataFrame, window_size: int = 30, 
                            max_features: int = 100, n_jobs: int = 1) -> pd.DataFrame:
        """
        使用tsfresh自动生成特征
        
        Parameters:
        -----------
        df : pd.DataFrame
            包含OHLCV数据的DataFrame
        window_size : int, default=30
            特征提取的窗口大小
        max_features : int, default=100
            最大特征数量（用于控制维度爆炸）
        n_jobs : int, default=1
            并行处理的作业数
            
        Returns:
        --------
        pd.DataFrame
            包含自动生成特征的DataFrame
        """
        if not self.use_tsfresh:
            print("❌ tsfresh不可用，跳过自动特征生成")
            return df[['datetime', 'close']].copy()
        
        print("🤖 开始自动特征生成...")
        print(f"   📊 窗口大小: {window_size}")
        print(f"   🔢 最大特征数: {max_features}")
        
        data = df.copy()
        
        # 准备tsfresh格式的数据
        tsfresh_data = []
        
        # 为每个窗口创建时序数据
        for i in range(window_size, len(data)):
            window_data = data.iloc[i-window_size:i].copy()
            window_id = i - window_size
            
            # 为tsfresh格式添加id和sort列
            window_data['id'] = window_id
            window_data['time'] = range(len(window_data))
            
            tsfresh_data.append(window_data)
        
        if not tsfresh_data:
            print("❌ 数据量不足以生成自动特征")
            return df[['datetime', 'close']].copy()
        
        # 合并所有窗口数据
        combined_data = pd.concat(tsfresh_data, ignore_index=True)
        
        # 选择要提取特征的列
        value_columns = ['close', 'volume', 'high', 'low', 'open']
        value_columns = [col for col in value_columns if col in combined_data.columns]
        
        try:
            print("   🔄 提取特征中...")
            
            # 使用最小特征集以控制维度
            if max_features <= 50:
                fc_parameters = MinimalFCParameters()
            else:
                fc_parameters = ComprehensiveFCParameters()
            
            # 提取特征
            extracted_features = extract_features(
                combined_data[['id', 'time'] + value_columns],
                column_id='id',
                column_sort='time',
                default_fc_parameters=fc_parameters,
                n_jobs=n_jobs
            )
            
            # 处理缺失值
            impute(extracted_features)
            
            # 特征选择（控制维度）
            if len(extracted_features.columns) > max_features:
                print(f"   ✂️ 特征降维: {len(extracted_features.columns)} -> {max_features}")
                
                # 简单的方差筛选
                feature_vars = extracted_features.var()
                selected_features = feature_vars.nlargest(max_features).index
                extracted_features = extracted_features[selected_features]
            
            # 添加时间戳和价格信息
            result_indices = list(range(window_size, len(data)))
            result_data = data.iloc[result_indices][['datetime', 'close']].reset_index(drop=True)
            
            # 合并特征
            extracted_features.reset_index(drop=True, inplace=True)
            result = pd.concat([result_data, extracted_features], axis=1)
            
            print(f"✅ 自动特征生成完成，特征数量: {len(extracted_features.columns)}")
            return result
            
        except Exception as e:
            print(f"❌ 自动特征生成失败: {str(e)}")
            return df[['datetime', 'close']].copy()
    
    def prepare_combined_features(self, df: pd.DataFrame, window_size: int = 30,
                                auto_features: bool = True, max_auto_features: int = 50) -> pd.DataFrame:
        """
        生成组合特征（手工 + 自动）
        
        Parameters:
        -----------
        df : pd.DataFrame
            原始OHLCV数据
        window_size : int, default=30
            自动特征提取的窗口大小
        auto_features : bool, default=True
            是否包含自动特征
        max_auto_features : int, default=50
            最大自动特征数量
            
        Returns:
        --------
        pd.DataFrame
            包含所有特征的DataFrame
        """
        print("🔧 开始生成组合特征...")
        
        # 生成手工特征
        manual_features = self.prepare_manual_features(df)
        
        if not auto_features or not self.use_tsfresh:
            print("✅ 仅使用手工特征")
            return manual_features
        
        # 生成自动特征
        auto_features_df = self.prepare_auto_features(df, window_size, max_auto_features)
        
        # 合并特征（以手工特征为主）
        if len(auto_features_df) > 0 and len(auto_features_df.columns) > 2:
            # 找到重叠的时间范围
            manual_times = set(manual_features['datetime'])
            auto_times = set(auto_features_df['datetime'])
            common_times = manual_times.intersection(auto_times)
            
            if common_times:
                # 筛选共同时间段的数据
                manual_filtered = manual_features[manual_features['datetime'].isin(common_times)].copy()
                auto_filtered = auto_features_df[auto_features_df['datetime'].isin(common_times)].copy()
                
                # 按时间排序
                manual_filtered = manual_filtered.sort_values('datetime').reset_index(drop=True)
                auto_filtered = auto_filtered.sort_values('datetime').reset_index(drop=True)
                
                # 合并特征（去除重复的datetime和close列）
                auto_features_only = auto_filtered.drop(['datetime', 'close'], axis=1, errors='ignore')
                combined = pd.concat([manual_filtered, auto_features_only], axis=1)
                
                print(f"✅ 组合特征生成完成")
                print(f"   📊 手工特征: {len(manual_filtered.columns) - 2}")
                print(f"   🤖 自动特征: {len(auto_features_only.columns)}")
                print(f"   🎯 总特征数: {len(combined.columns) - 2}")
                
                return combined
        
        print("⚠️ 自动特征合并失败，仅返回手工特征")
        return manual_features
    
    def analyze_features(self, features_df: pd.DataFrame, plot: bool = True) -> Dict:
        """
        分析特征分布和质量
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            特征数据
        plot : bool, default=True
            是否绘制分析图表
            
        Returns:
        --------
        Dict
            特征分析结果
        """
        print("📊 开始特征分析...")
        
        feature_cols = [col for col in features_df.columns if col not in ['datetime', 'close']]
        
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
            'min': numeric_features.min().to_dict(),
            'max': numeric_features.max().to_dict()
        }
        
        # 打印分析结果
        print(f"📈 特征分析结果:")
        print(f"   🔢 总特征数: {analysis['total_features']}")
        print(f"   ❌ 缺失值特征: {len(analysis['missing_values'])}")
        print(f"   ⚠️ 异常值特征: {len(analysis['extreme_values'])}")
        
        if analysis['missing_values']:
            print("   📋 缺失值详情:")
            for col, info in list(analysis['missing_values'].items())[:5]:  # 只显示前5个
                print(f"      {col}: {info['count']} ({info['percentage']:.1f}%)")
        
        if analysis['extreme_values']:
            print("   📋 异常值详情:")
            for col, info in list(analysis['extreme_values'].items())[:5]:  # 只显示前5个
                print(f"      {col}: {info['count']} ({info['percentage']:.1f}%)")
        
        # 绘图分析
        if plot and len(numeric_features.columns) > 0:
            self._plot_feature_analysis(numeric_features)
        
        return analysis
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """手工计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _plot_feature_analysis(self, features_df: pd.DataFrame, max_plots: int = 12):
        """绘制特征分析图表"""
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            n_features = min(len(features_df.columns), max_plots)
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            axes = axes.ravel()
            
            for i, col in enumerate(features_df.columns[:n_features]):
                # 分布直方图
                axes[i].hist(features_df[col].dropna(), bins=50, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{col}', fontsize=10)
                axes[i].tick_params(labelsize=8)
            
            # 隐藏多余的子图
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.suptitle('特征分布分析', fontsize=14, y=0.98)
            plt.show()
            
        except Exception as e:
            print(f"⚠️ 绘图失败: {str(e)}")


# 测试函数
def test_feature_engineering():
    """测试特征工程功能"""
    print("🧪 测试特征工程功能")
    
    # 创建测试数据
    dates = pd.date_range('2022-01-01', periods=200, freq='D')
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        'datetime': dates,
        'open': 100 + np.cumsum(np.random.randn(200) * 0.5),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(1000, 10000, 200)
    })
    
    # 生成OHLC数据
    test_data['close'] = test_data['open'] + np.random.randn(200) * 0.3
    test_data['high'] = np.maximum(test_data['open'], test_data['close']) + np.random.rand(200) * 0.5
    test_data['low'] = np.minimum(test_data['open'], test_data['close']) - np.random.rand(200) * 0.5
    
    # 测试特征工程
    engineer = FeatureEngineer()
    
    # 测试手工特征
    manual_features = engineer.prepare_manual_features(test_data)
    print(f"📊 手工特征测试完成，特征数量: {len(manual_features.columns) - 2}")
    
    # 分析特征
    analysis = engineer.analyze_features(manual_features, plot=False)
    
    return manual_features, analysis


if __name__ == "__main__":
    # 运行测试
    features, analysis = test_feature_engineering()
    print("✅ 特征工程测试完成")