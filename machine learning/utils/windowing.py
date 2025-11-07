"""
股票滑窗数据生成器 - 重构版
用于时序预测的样本生成，使用独立的特征工程模块

核心概念：
1. 窗口长度(window_size)：用多少历史数据作为输入特征
2. 预测目标(target_type)：预测什么（价格、收益率、涨跌等）
3. 预测步长(prediction_steps)：预测未来第几天
4. 滑动步长(stride)：窗口每次移动多少步
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List, Optional, Dict, Any
import warnings
import sys
import os

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
if ml_root not in sys.path:
    sys.path.insert(0, ml_root)

# 导入特征工程模块
from features.feature_engineering import FeatureEngineer

# 添加上层目录以导入stock_info模块
project_root = os.path.dirname(os.path.dirname(ml_root))  # stock/
get_stock_info_path = os.path.join(project_root, "get_stock_info")
if get_stock_info_path not in sys.path:
    sys.path.insert(0, get_stock_info_path)

try:
    import utils
    from stock_market_data_akshare import get_history_data
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    print("⚠️ InfluxDB相关模块导入失败，将回退到CSV数据")

warnings.filterwarnings('ignore')


def load_real_stock_data(symbol: str = "000001", start_date: str = "2022-01-01", end_date: str = "2024-12-31") -> pd.DataFrame:
    """
    从InfluxDB加载真实股票数据
    
    Parameters:
    -----------
    symbol : str, default="000001"
        股票代码
    start_date : str, default="2022-01-01"
        开始日期
    end_date : str, default="2024-12-31"
        结束日期
        
    Returns:
    --------
    pd.DataFrame
        包含OHLCV数据的DataFrame，如果加载失败则返回None
    """
    if not INFLUXDB_AVAILABLE:
        print("ERROR: InfluxDB modules not available, cannot load real data")
        return None
    
    try:
        print(f"Loading {symbol} data from InfluxDB...")
        
        # 获取InfluxDB客户端
        client = utils.get_influxdb_client()
        if client is None:
            print("ERROR: Cannot connect to InfluxDB")
            return None
        
        query_api = client.query_api()
        
        # 转换日期格式
        start_str_rfc = f"{start_date}T00:00:00Z"
        end_str_rfc = f"{end_date}T23:59:59Z"
        
        # 获取历史数据
        df = get_history_data(query_api, symbol, start_str_rfc, end_str_rfc)
        
        if df.empty:
            print(f"ERROR: No data found for {symbol} in InfluxDB")
            client.close()
            return None
        
        # 标准化列名（完整映射InfluxDB字段）
        column_mapping = {
            '日期': 'datetime',
            '开盘': 'open',
            '最高': 'high', 
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
            '成交额': 'amount',      # 成交额是amount
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover',    # 换手率才是turnover
            '是否停牌': 'is_suspended'
        }
        
        df = df.rename(columns=column_mapping)
        
        # 确保datetime列是正确的时间格式
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # 添加缺失的列（如果原始数据没有）
        if 'amount' not in df.columns:
            df['amount'] = 0.0
        if 'turnover' not in df.columns:
            df['turnover'] = 0.0
        
        print(f"SUCCESS: Loaded {len(df)} records for {symbol} from InfluxDB")
        print(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
        
        client.close()
        return df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover']]
        
    except Exception as e:
        print(f"ERROR: Failed to load data from InfluxDB: {str(e)}")
        return None


class SlidingWindowGenerator:
    """
    滑窗数据生成器 - 重构版
    使用独立的特征工程模块处理特征生成
    """
    
    def __init__(self, 
                 window_size: int = 30,
                 prediction_steps: int = 1,
                 stride: int = 1,
                 target_type: str = 'return',
                 scaler_type: str = 'standard',
                 feature_type: str = 'manual',  # 'manual', 'auto', 'combined'
                 max_auto_features: int = 50):
        """
        初始化滑窗生成器
        
        Parameters:
        -----------
        window_size : int, default=30
            滑动窗口大小（历史数据长度）
        prediction_steps : int, default=1
            预测步长（预测未来第几步）
        stride : int, default=1
            滑动步长（窗口每次移动的步数）
        target_type : str, default='return'
            目标类型：'price', 'return', 'return_multi', 'direction', 'high_low'
        scaler_type : str, default='standard'
            特征缩放类型：'standard', 'minmax', None
        feature_type : str, default='manual'
            特征类型：'manual'(手工), 'auto'(自动), 'combined'(组合)
        max_auto_features : int, default=50
            最大自动特征数量
        """
        self.window_size = window_size
        self.prediction_steps = prediction_steps
        self.stride = stride
        self.target_type = target_type
        self.scaler_type = scaler_type
        self.feature_type = feature_type
        self.max_auto_features = max_auto_features
        self.scaler = None
        
        # 初始化特征工程器
        self.feature_engineer = FeatureEngineer()
        
        print(f"🔧 滑窗生成器配置:")
        print(f"   📊 窗口大小: {self.window_size}")
        print(f"   🎯 预测步长: {self.prediction_steps} 步后")
        print(f"   ⚡ 滑动步长: {self.stride}")
        print(f"   📈 目标类型: {self.target_type}")
        print(f"   📏 缩放方式: {self.scaler_type}")
        print(f"   🔧 特征类型: {self.feature_type}")
        if self.feature_type in ['auto', 'combined']:
            print(f"   🤖 最大自动特征: {self.max_auto_features}")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用特征工程模块生成特征
        
        Parameters:
        -----------
        df : pd.DataFrame
            原始OHLCV数据
            
        Returns:
        --------
        pd.DataFrame
            包含特征的DataFrame
        """
        if self.feature_type == 'manual':
            return self.feature_engineer.prepare_manual_features(df)
        elif self.feature_type == 'auto':
            return self.feature_engineer.prepare_auto_features(
                df, window_size=self.window_size, max_features=self.max_auto_features
            )
        elif self.feature_type == 'combined':
            return self.feature_engineer.prepare_combined_features(
                df, window_size=self.window_size, 
                auto_features=True, max_auto_features=self.max_auto_features
            )
        else:
            raise ValueError(f"不支持的特征类型: {self.feature_type}")

    def create_target(self, df: pd.DataFrame) -> np.ndarray:
        """
        根据target_type创建预测目标
        
        注意：已修复np.roll导致的末尾循环问题
        超出范围的位置填充np.nan，避免使用错误的循环数据
        """
        close_prices = df['close'].values
        
        if self.target_type == 'price':
            # 预测未来价格
            target = np.full_like(close_prices, np.nan, dtype=float)
            if self.prediction_steps < len(close_prices):
                target[:-self.prediction_steps] = close_prices[self.prediction_steps:]
            
        elif self.target_type == 'return':
            # 预测未来收益率
            target = np.full_like(close_prices, np.nan, dtype=float)
            if self.prediction_steps < len(close_prices):
                future_prices = close_prices[self.prediction_steps:]
                current_prices = close_prices[:-self.prediction_steps]
                target[:-self.prediction_steps] = (future_prices - current_prices) / current_prices
            
        elif self.target_type == 'return_multi':
            # 预测未来N天累计收益率
            target = []
            for i in range(len(close_prices)):
                if i + self.prediction_steps < len(close_prices):
                    future_returns = []
                    for j in range(1, self.prediction_steps + 1):
                        if i + j < len(close_prices):
                            daily_return = (close_prices[i + j] - close_prices[i + j - 1]) / close_prices[i + j - 1]
                            future_returns.append(daily_return)
                    
                    if future_returns:
                        # 累计收益率
                        cumulative_return = np.prod(1 + np.array(future_returns)) - 1
                        target.append(cumulative_return)
                    else:
                        target.append(np.nan)
                else:
                    target.append(np.nan)
            target = np.array(target)
            
        elif self.target_type == 'direction':
            # 预测涨跌方向（分类任务）
            target = np.full(len(close_prices), np.nan, dtype=float)
            if self.prediction_steps < len(close_prices):
                future_prices = close_prices[self.prediction_steps:]
                current_prices = close_prices[:-self.prediction_steps]
                target[:-self.prediction_steps] = (future_prices > current_prices).astype(int)
            
        elif self.target_type == 'high_low':
            # 预测未来N天的最高价和最低价
            high_prices = df['high'].values
            low_prices = df['low'].values
            target = []
            
            for i in range(len(close_prices)):
                if i + self.prediction_steps < len(close_prices):
                    future_high = np.max(high_prices[i+1:i+1+self.prediction_steps])
                    future_low = np.min(low_prices[i+1:i+1+self.prediction_steps])
                    # 转换为相对当前价格的比例
                    high_ratio = future_high / close_prices[i] - 1
                    low_ratio = future_low / close_prices[i] - 1
                    target.append([high_ratio, low_ratio])
                else:
                    target.append([np.nan, np.nan])
            target = np.array(target)
        
        else:
            raise ValueError(f"不支持的target_type: {self.target_type}")
        
        return target

    def generate_samples(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        生成滑窗样本
        
        Returns:
        --------
        X : np.ndarray, shape (n_samples, window_size, n_features)
            输入特征数组
        y : np.ndarray, shape (n_samples, target_dim)
            目标数组
        metadata : pd.DataFrame
            样本元数据（时间戳等）
        """
        print(f"🔄 开始生成滑窗样本...")
        
        # 使用特征工程模块生成特征
        data = self.prepare_features(df)
        print(f"📊 特征工程完成，特征数量: {len(data.columns) - 2}")  # 减去datetime和close
        
        # 创建目标值
        target = self.create_target(data)
        
        # 准备特征数组
        feature_columns = [col for col in data.columns if col not in ['datetime', 'close']]
        features = data[feature_columns].values
        
        # 特征缩放
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
            features = self.scaler.fit_transform(features)
        
        # 初始化样本存储
        X, y, metadata = [], [], []
        
        # 计算最大起始索引
        max_start_idx = len(features) - self.window_size - self.prediction_steps
        
        # 滑窗生成样本
        for i in range(0, max_start_idx, self.stride):
            # 提取窗口特征
            x_window = features[i:i+self.window_size]
            
            # 目标索引
            target_idx = i + self.window_size - 1  # 窗口最后一天的索引
            
            # 检查目标值有效性
            if target_idx < len(target) and not np.any(np.isnan(target[target_idx])):
                X.append(x_window)
                y.append(target[target_idx])
                
                # 保存元数据
                window_start = data.iloc[i]['datetime']
                window_end = data.iloc[i+self.window_size-1]['datetime']
                prediction_date = data.iloc[min(target_idx + self.prediction_steps, len(data)-1)]['datetime']
                
                metadata.append({
                    'window_start': window_start,
                    'window_end': window_end,
                    'prediction_date': prediction_date,
                    'current_price': data.iloc[target_idx]['close']
                })
        
        # 转换为numpy数组
        X = np.array(X)
        y = np.array(y)
        metadata = pd.DataFrame(metadata)
        
        print(f"✅ 样本生成完成:")
        print(f"   📦 样本数量: {len(X)}")
        print(f"   📏 输入形状: {X.shape}")
        print(f"   🎯 输出形状: {y.shape}")
        
        return X, y, metadata

    def analyze_samples(self, X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame):
        """
        分析生成的样本
        """
        print(f"\n📊 样本分析报告")
        print("=" * 50)
        
        # 基本信息
        print(f"数据概览:")
        print(f"  - 总样本数: {len(X)}")
        print(f"  - 输入维度: {X.shape}")
        print(f"  - 输出维度: {y.shape}")
        print(f"  - 时间跨度: {metadata['window_start'].min()} 到 {metadata['prediction_date'].max()}")
        
        # 目标值分析
        print(f"\n目标值分析:")
        if self.target_type == 'direction':
            # 分类任务分析
            unique, counts = np.unique(y, return_counts=True)
            for val, count in zip(unique, counts):
                label = "上涨" if val == 1 else "下跌"
                print(f"  - {label}: {count} 个样本 ({count/len(y)*100:.1f}%)")
        else:
            # 回归任务分析
            if len(y.shape) > 1 and y.shape[1] > 1:
                # 多维目标
                for i in range(y.shape[1]):
                    print(f"  - 维度{i} - 均值: {np.mean(y[:, i]):.4f}, 标准差: {np.std(y[:, i]):.4f}")
            else:
                # 单维目标
                print(f"  - 均值: {np.mean(y):.4f}")
                print(f"  - 标准差: {np.std(y):.4f}")
                print(f"  - 最小值: {np.min(y):.4f}")
                print(f"  - 最大值: {np.max(y):.4f}")
        
        # 缺失值检查
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            print(f"\n⚠️  发现缺失值:")
            print(f"  - X中缺失值: {np.sum(np.isnan(X))}")
            print(f"  - y中缺失值: {np.sum(np.isnan(y))}")
        else:
            print(f"\n✅ 无缺失值")

    def visualize_samples(self, X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame, n_samples: int = 3):
        """
        可视化样本分析
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 子图1: 样本时间分布
        axes[0,0].hist(metadata['window_end'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('样本时间分布')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 子图2: 目标值分布
        if self.target_type == 'direction':
            unique, counts = np.unique(y, return_counts=True)
            labels = ['下跌' if x == 0 else '上涨' for x in unique]
            axes[0,1].bar(labels, counts, color=['red', 'green'], alpha=0.7)
            axes[0,1].set_title('涨跌分布')
        else:
            axes[0,1].hist(y.flatten() if len(y.shape) > 1 else y, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0,1].set_title('目标值分布')
        
        # 子图3: 特征趋势（展示第0个特征的平均曲线）
        if len(X) > 0:
            mean_feature_0 = np.mean(X[:, :, 0], axis=0)
            axes[1,0].plot(range(self.window_size), mean_feature_0, marker='o', linewidth=2)
            axes[1,0].set_title('特征0的平均趋势')
            axes[1,0].set_xlabel('时间步')
            axes[1,0].grid(True, alpha=0.3)
        
        # 子图4: 随机样本展示
        if len(X) >= n_samples:
            sample_indices = np.random.choice(len(X), n_samples, replace=False)
            for idx in sample_indices:
                axes[1,1].plot(range(self.window_size), X[idx, :, 0], alpha=0.7, 
                             label=f'样本{idx} (目标: {y[idx]:.3f})')
            axes[1,1].set_title(f'随机{n_samples}个样本的特征0')
            axes[1,1].set_xlabel('时间步')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def demo_new_sliding_window():
    """
    演示新的滑窗生成器功能
    """
    print("🚀 新版滑窗生成器演示")
    print("=" * 60)
    
    # 优先尝试从InfluxDB加载真实数据
    print("Loading data...")
    df = load_real_stock_data("000001", "2022-01-01", "2024-12-31")
    
    data_source = "Real stock data (InfluxDB)"
    
    # 如果真实数据太多，取最近的数据以提高演示速度
    if len(df) > 500:
        df = df.tail(500).reset_index(drop=True)
        print(f"Using latest 500 records for demo performance")
    
    print(f"SUCCESS: Data loaded ({data_source}): {len(df)} records")
    print(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
    
    # 测试配置
    test_configs = [
        {
            'name': '手工特征 - 短期预测',
            'window_size': 30,
            'prediction_steps': 1,
            'target_type': 'return',
            'feature_type': 'manual',
            'stride': 1
        },
        {
            'name': '手工特征 - 中期预测', 
            'window_size': 60,
            'prediction_steps': 5,
            'target_type': 'return',
            'feature_type': 'manual',
            'stride': 5
        },
        {
            'name': '组合特征 - 分类预测',
            'window_size': 20,
            'prediction_steps': 3,
            'target_type': 'direction',
            'feature_type': 'manual',  # 暂时只用手工特征，避免tsfresh问题
            'stride': 1,
            'max_auto_features': 30
        }
    ]
    
    results = {}
    
    for i, config in enumerate(test_configs):
        print(f"\n{'='*60}")
        print(f"🔬 配置 {i+1}: {config['name']}")
        print(f"{'='*60}")
        
        # 创建生成器
        generator = SlidingWindowGenerator(
            window_size=config['window_size'],
            prediction_steps=config['prediction_steps'],
            target_type=config['target_type'],
            feature_type=config['feature_type'],
            stride=config['stride'],
            max_auto_features=config.get('max_auto_features', 50)
        )
        
        # 生成样本
        X, y, metadata = generator.generate_samples(df)
        
        # 分析样本
        generator.analyze_samples(X, y, metadata)
        
        # 保存结果
        results[config['name']] = {
            'generator': generator,
            'X': X,
            'y': y,
            'metadata': metadata
        }
        
        # 第一个配置展示可视化
        if i == 0:
            generator.visualize_samples(X, y, metadata)
    
    return results


def practical_examples():
    """
    展示最佳实践建议
    """
    print("\n" + "=" * 60)
    print("💡 滑窗设计最佳实践")
    print("=" * 60)
    
    practices = {
        "短线交易 (1-3天)": {
            "窗口大小": "5-20天",
            "预测步长": "1-3天",
            "特征类型": "手工特征 + 短期技术指标",
            "适用场景": "日内交易、短期波动捕捉"
        },
        "中线交易 (5-20天)": {
            "窗口大小": "30-60天", 
            "预测步长": "5-10天",
            "特征类型": "组合特征",
            "适用场景": "趋势跟踪、波段操作"
        },
        "长线交易 (30天+)": {
            "窗口大小": "60-120天",
            "预测步长": "20-30天", 
            "特征类型": "自动特征 + 宏观指标",
            "适用场景": "价值投资、长期趋势"
        }
    }
    
    for strategy, params in practices.items():
        print(f"\n📈 {strategy}:")
        for key, value in params.items():
            print(f"   {key}: {value}")
    
    print(f"\n⚠️  常见踩坑点:")
    print(f"   1. 窗口过小: 信息不足，模式学习困难")
    print(f"   2. 窗口过大: 包含过时信息，计算量增加")
    print(f"   3. stride过大: 样本数量不足，模型难以训练") 
    print(f"   4. 未来数据泄露: 特征中包含目标值信息")
    print(f"   5. 数据不平衡: 分类任务中正负样本比例失衡")
    print(f"   6. 自动特征爆炸: tsfresh生成过多特征，需要筛选")
    
    print(f"\n✅ 选择建议:")
    print(f"   • 根据持仓周期选择prediction_steps")
    print(f"   • 训练时stride=1，预测时可适当增加")
    print(f"   • 手工特征优先，自动特征作为补充")
    print(f"   • 组合特征时控制总特征数量(<100)")


def test_boundary_fix():
    """
    测试边界修复效果 - 使用InfluxDB真实股票数据
    """
    print("🧪 测试create_target边界修复效果")
    print("=" * 50)
    
    # 使用InfluxDB真实股票数据进行测试
    print("📊 加载真实股票数据进行边界测试...")
    test_data = load_real_stock_data("000001", "2024-01-01", "2024-01-31")
    
    # 如果InfluxDB不可用，使用CSV数据作为备用
    if test_data is None or len(test_data) < 10:
        print("⚠️ InfluxDB数据不可用")
    else:
        # 如果数据太多，只取前15天进行快速测试
        if len(test_data) > 15:
            test_data = test_data.head(15).copy()
        print(f"✅ 使用InfluxDB真实股票数据进行测试: {len(test_data)} 条记录")
    
    print(f"� 数据时间范围: {test_data['datetime'].min().date()} 到 {test_data['datetime'].max().date()}")
    print("收盘价前5个:", test_data['close'].head().tolist())
    
    # 测试不同target_type的边界处理
    test_configs = [
        ('price', 3),
        ('return', 2), 
        ('direction', 4)
    ]
    
    for target_type, prediction_steps in test_configs:
        print(f"\n🔍 测试 {target_type}, prediction_steps={prediction_steps}")
        
        generator = SlidingWindowGenerator(
            target_type=target_type,
            prediction_steps=prediction_steps,
            feature_type='manual'
        )
        
        target = generator.create_target(test_data)
        
        print(f"目标数组长度: {len(target)}")
        print(f"NaN数量: {np.sum(np.isnan(target))}")
        print(f"有效值数量: {np.sum(~np.isnan(target))}")
        print(f"目标值: {target}")
        
        # 验证末尾是否正确填充了NaN
        expected_nan_count = prediction_steps
        actual_nan_count = np.sum(np.isnan(target))
        
        if actual_nan_count >= expected_nan_count:
            print("✅ 边界处理正确")
        else:
            print("❌ 边界处理可能有问题")
    
    print(f"\n✅ 边界修复测试完成")


if __name__ == "__main__":
    # 先测试边界修复
    test_boundary_fix()
    
    print("\n" + "="*60)
    
    # 运行新版演示
    results = demo_new_sliding_window()
    
    # 展示最佳实践
    practical_examples()