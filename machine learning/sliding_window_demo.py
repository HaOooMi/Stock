"""
股票滑窗数据生成器
用于时序预测的样本生成和特征工程

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
warnings.filterwarnings('ignore')

class SlidingWindowGenerator:
    """
    股票时序数据滑窗生成器
    """
    
    def __init__(self, 
                 window_size: int = 60,
                 prediction_steps: int = 1, 
                 stride: int = 1,
                 target_type: str = 'return',
                 scaler_type: str = 'standard'):
        """
        初始化滑窗生成器
        
        Parameters:
        -----------
        window_size : int, default=60
            历史数据窗口大小（多少天的数据作为输入）
        prediction_steps : int, default=1
            预测未来第几天（1=明天，5=未来5天后）
        stride : int, default=1
            滑动步长（每次移动多少天）
        target_type : str, default='return'
            预测目标类型：
            - 'price': 预测未来价格
            - 'return': 预测未来收益率
            - 'return_multi': 预测未来N天累计收益率  
            - 'direction': 预测涨跌方向（分类）
            - 'high_low': 预测未来N天最高价和最低价
        scaler_type : str, default='standard'
            特征缩放方法：'standard', 'minmax', None
        """
        self.window_size = window_size
        self.prediction_steps = prediction_steps
        self.stride = stride
        self.target_type = target_type
        self.scaler_type = scaler_type
        self.scaler = None
        
        print(f"🔧 滑窗配置:")
        print(f"   📏 窗口大小: {window_size} 天")
        print(f"   🎯 预测目标: {target_type}")
        print(f"   📍 预测步长: {prediction_steps} 天后")
        print(f"   👣 滑动步长: {stride} 天")
        print(f"   📊 缩放方式: {scaler_type}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备技术指标特征
        """
        data = df.copy()
        
        # 基础价格特征
        data['price_change'] = data['close'].pct_change()
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        
        # 移动平均线
        for window in [5, 10, 20, 30]:
            data[f'ma_{window}'] = data['close'].rolling(window).mean()
            data[f'price_ma_{window}_ratio'] = data['close'] / data[f'ma_{window}']
        
        # 波动性指标
        data['volatility_5'] = data['price_change'].rolling(5).std()
        data['volatility_20'] = data['price_change'].rolling(20).std()
        
        # 成交量特征
        data['volume_ma_5'] = data['volume'].rolling(5).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma_5']
        
        # RSI 相对强弱指数
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        data['rsi'] = calculate_rsi(data['close'])
        
        # 布林带
        ma_20 = data['close'].rolling(20).mean()
        std_20 = data['close'].rolling(20).std()
        data['bollinger_upper'] = ma_20 + (std_20 * 2)
        data['bollinger_lower'] = ma_20 - (std_20 * 2)
        data['bollinger_position'] = (data['close'] - data['bollinger_lower']) / (data['bollinger_upper'] - data['bollinger_lower'])
        
        # 删除原始OHLCV，只保留技术指标
        feature_columns = [col for col in data.columns if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'open_interest']]
        
        return data[['datetime', 'close'] + feature_columns].dropna()
    
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
        
        # 准备特征
        data = self.prepare_features(df)
        print(f"📊 特征工程完成，特征数量: {len(data.columns) - 2}")  # 减去datetime和close
        
        # 创建目标
        target = self.create_target(data)
        
        # 准备特征数组（除了datetime和close）
        feature_columns = [col for col in data.columns if col not in ['datetime', 'close']]
        features = data[feature_columns].values
        
        # 特征缩放
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
            features = self.scaler.fit_transform(features)
        
        # 生成滑窗样本
        X, y, metadata = [], [], []
        
        max_start_idx = len(features) - self.window_size - self.prediction_steps
        
        for i in range(0, max_start_idx, self.stride):
            # 输入特征窗口
            x_window = features[i:i+self.window_size]
            
            # 目标值
            target_idx = i + self.window_size - 1  # 窗口最后一天的目标
            if target_idx < len(target) and not np.any(np.isnan(target[target_idx])):
                X.append(x_window)
                y.append(target[target_idx])
                
                # 记录元数据
                window_start = data.iloc[i]['datetime']
                window_end = data.iloc[i+self.window_size-1]['datetime']
                prediction_date = data.iloc[min(target_idx + self.prediction_steps, len(data)-1)]['datetime']
                
                metadata.append({
                    'window_start': window_start,
                    'window_end': window_end,
                    'prediction_date': prediction_date,
                    'current_price': data.iloc[target_idx]['close']
                })
        
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
        print(f"\n📊 样本分析报告:")
        print(f"=" * 50)
        
        # 基本统计
        print(f"数据概览:")
        print(f"  - 总样本数: {len(X)}")
        print(f"  - 输入维度: {X.shape}")
        print(f"  - 输出维度: {y.shape}")
        print(f"  - 时间跨度: {metadata['window_start'].min()} 到 {metadata['prediction_date'].max()}")
        
        # 目标分析
        if self.target_type == 'direction':
            print(f"\n分类目标分析:")
            unique, counts = np.unique(y, return_counts=True)
            for val, count in zip(unique, counts):
                label = "上涨" if val == 1 else "下跌"
                print(f"  - {label}: {count} 个样本 ({count/len(y)*100:.1f}%)")
        else:
            print(f"\n回归目标分析:")
            print(f"  - 均值: {np.mean(y):.4f}")
            print(f"  - 标准差: {np.std(y):.4f}")
            print(f"  - 最小值: {np.min(y):.4f}")
            print(f"  - 最大值: {np.max(y):.4f}")
            
            if len(y.shape) > 1 and y.shape[1] > 1:
                for i in range(y.shape[1]):
                    print(f"  - 维度{i} - 均值: {np.mean(y[:, i]):.4f}, 标准差: {np.std(y[:, i]):.4f}")
        
        # 缺失值检查
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            print(f"\n⚠️  发现缺失值:")
            print(f"  - X中缺失值: {np.sum(np.isnan(X))}")
            print(f"  - y中缺失值: {np.sum(np.isnan(y))}")
        else:
            print(f"\n✅ 无缺失值")
    
    def visualize_samples(self, X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame, n_samples: int = 3):
        """
        可视化几个样本
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'滑窗样本可视化 (窗口={self.window_size}, 预测={self.target_type})', fontsize=16)
        
        # 1. 样本时间分布
        axes[0, 0].hist(pd.to_datetime(metadata['window_end']), bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('样本时间分布')
        axes[0, 0].set_xlabel('时间')
        axes[0, 0].set_ylabel('样本数量')
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # 2. 目标值分布
        if self.target_type == 'direction':
            unique, counts = np.unique(y, return_counts=True)
            labels = ['下跌' if x == 0 else '上涨' for x in unique]
            axes[0, 1].bar(labels, counts, alpha=0.7, color=['red', 'green'])
            axes[0, 1].set_title('涨跌分布')
        else:
            axes[0, 1].hist(y.flatten() if len(y.shape) > 1 else y, bins=50, alpha=0.7, color='green')
            axes[0, 1].set_title('目标值分布')
            axes[0, 1].set_xlabel('目标值')
            axes[0, 1].set_ylabel('频次')
        
        # 3. 特征重要性（显示第一个特征的变化）
        if X.shape[2] > 0:
            feature_mean = np.mean(X[:, :, 0], axis=0)  # 第一个特征在所有样本上的均值
            axes[1, 0].plot(feature_mean)
            axes[1, 0].set_title('特征趋势 (特征0的窗口内平均)')
            axes[1, 0].set_xlabel('窗口内位置')
            axes[1, 0].set_ylabel('特征值')
        
        # 4. 样本展示
        axes[1, 1].set_title(f'随机展示 {min(n_samples, len(X))} 个样本')
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        sample_indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            if X.shape[2] > 0:
                # 显示第一个特征
                axes[1, 1].plot(X[idx, :, 0], 
                               color=colors[i % len(colors)], 
                               alpha=0.7,
                               label=f'样本{idx} (目标: {y[idx]:.3f})')
        
        axes[1, 1].legend()
        axes[1, 1].set_xlabel('时间步')
        axes[1, 1].set_ylabel('特征值')
        
        plt.tight_layout()
        plt.show()


def demo_sliding_window():
    """
    演示不同滑窗配置的效果
    """
    print("🚀 股票滑窗数据生成器演示")
    print("=" * 60)
    
    # 加载数据
    data_path = r"d:\vscode projects\stock\csv_data\000001.SZSE_d_2022-01-01_2024-12-31.csv"
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"📊 加载数据: {len(df)} 条记录")
    print(f"📅 时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    
    # 配置不同的滑窗参数进行演示
    configs = [
        {
            'name': '短期价格预测',
            'window_size': 30,
            'prediction_steps': 1,
            'target_type': 'return',
            'stride': 1
        },
        {
            'name': '中期趋势预测',
            'window_size': 60,
            'prediction_steps': 5,
            'target_type': 'return_multi',
            'stride': 5
        },
        {
            'name': '涨跌方向分类',
            'window_size': 20,
            'prediction_steps': 3,
            'target_type': 'direction',
            'stride': 1
        },
        {
            'name': '高低点预测',
            'window_size': 40,
            'prediction_steps': 10,
            'target_type': 'high_low',
            'stride': 3
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n" + "="*50)
        print(f"🔧 测试配置: {config['name']}")
        print(f"="*50)
        
        # 创建生成器
        generator = SlidingWindowGenerator(
            window_size=config['window_size'],
            prediction_steps=config['prediction_steps'],
            target_type=config['target_type'],
            stride=config['stride']
        )
        
        # 生成样本
        X, y, metadata = generator.generate_samples(df)
        
        # 分析样本
        generator.analyze_samples(X, y, metadata)
        
        # 保存结果
        results[config['name']] = {
            'X': X,
            'y': y, 
            'metadata': metadata,
            'generator': generator
        }
        
        # 可视化（只展示第一个配置）
        if config == configs[0]:
            generator.visualize_samples(X, y, metadata)
    
    return results


def practical_examples():
    """
    实际应用的最佳实践示例
    """
    print("\n🎯 滑窗设计最佳实践")
    print("=" * 60)
    
    print("1. 短线交易（日内/短线）:")
    print("   - 窗口: 5-20天")
    print("   - 预测: 1-3天")
    print("   - 目标: 涨跌方向或短期收益")
    print("   - 特点: 反应快，噪音多")
    
    print("\n2. 中线交易（波段）:")
    print("   - 窗口: 30-60天")
    print("   - 预测: 5-10天")
    print("   - 目标: 累计收益或趋势方向")
    print("   - 特点: 平衡性好，适合大多数场景")
    
    print("\n3. 长线交易（趋势）:")
    print("   - 窗口: 60-120天")
    print("   - 预测: 20-30天")
    print("   - 目标: 长期收益或重要拐点")
    print("   - 特点: 稳定性高，反应慢")
    
    print("\n⚠️  常见踩坑点:")
    print("1. 窗口太小 → 学不到模式，噪音大")
    print("2. 窗口太大 → 训练慢，可能过时")
    print("3. stride太大 → 样本少，信息丢失")
    print("4. 未来信息泄露 → 不小心用了未来数据")
    print("5. 数据不平衡 → 涨跌样本比例悬殊")
    
    print("\n💡 选择建议:")
    print("- 根据交易周期选窗口大小")
    print("- 预测步长 = 你的持仓周期")
    print("- stride=1 获得最多样本")
    print("- 考虑计算资源和训练时间")
    print("- 先简单后复杂，逐步优化")


if __name__ == "__main__":
    # 运行演示
    results = demo_sliding_window()
    
    # 展示最佳实践
    practical_examples()