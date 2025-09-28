#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略信号与回测模块（使用独立InfluxDB数据+已训练PCA模型，避免数据泄漏）

功能：
1. 基于聚类分析结果选择最佳簇
2. 从InfluxDB获取完全独立的新数据进行交易信号生成（signal=1/0）
3. 使用已训练的PCA模型对新数据进行降维
4. 计算策略收益 vs 基准
5. 策略性能评估和可视化
6. 随机基准对比验证

数据流：InfluxDB新数据 → PCA完整流程 → 聚类预测 → 策略回测
确保测试数据完全独立，符合实际应用场景
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入项目模块
try:
    from pca_state import PCAStateGenerator
    # InfluxDB加载器可能需要根据实际项目结构调整
    # from data_integration.influxdb_loader import InfluxDBLoader
    print("✅ 成功导入PCA模块")
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保项目结构正确且依赖模块存在")
    # 如果导入失败，继续运行但功能受限
    PCAStateGenerator = None

# 配置日志
import logging
logger = logging.getLogger(__name__)

class StrategyBacktestClean:
    """
    策略回测模块 - 使用PCA完整流程和独立数据源
    """
    
    def __init__(self, project_root: str = None):
        """
        初始化策略回测器
        
        Parameters:
        -----------
        project_root : str
            项目根目录路径
        """
        self.project_root = project_root or os.path.dirname(os.path.dirname(__file__))
        # self.influxdb_loader = InfluxDBLoader()  # 暂时注释，根据实际情况调整
        self.cluster_models = {}
        self.best_clusters = {}
        
        print("🎯 策略回测模块初始化完成")
        print(f"📁 项目根目录: {self.project_root}")

    def load_cluster_analysis_results(self, symbol: str = "000001") -> Dict:
        """
        加载聚类分析的结果，用于策略回测
        
        Parameters:
        -----------
        symbol : str
            股票代码
            
        Returns:
        --------
        dict
            聚类分析结果
        """
        print(f"📊 加载聚类分析结果: {symbol}")
        
        # 查找聚类模型文件
        cluster_dir = os.path.join(self.project_root, "machine learning/ML output/models")
        
        if not os.path.exists(cluster_dir):
            raise FileNotFoundError(f"聚类模型目录不存在: {cluster_dir}")
        
        # 加载聚类模型
        cluster_files = [f for f in os.listdir(cluster_dir) 
                        if f.startswith(f'kmeans_{symbol}_') and f.endswith('.pkl')]
        
        if not cluster_files:
            raise FileNotFoundError(f"未找到股票 {symbol} 的聚类模型文件")
        
        for cluster_file in cluster_files:
            # 提取k值
            k_value = int(cluster_file.split('_k')[1].split('_')[0])
            model_path = os.path.join(cluster_dir, cluster_file)
            
            with open(model_path, 'rb') as f:
                cluster_data = pickle.load(f)
                self.cluster_models[k_value] = cluster_data
        
        # 从cluster_evaluate.py的结果中选择最佳k值
        evaluate_results_path = os.path.join(self.project_root, "machine learning/ML output/cluster_evaluation_results.csv")
        
        if os.path.exists(evaluate_results_path):
            eval_results = pd.read_csv(evaluate_results_path)
            # 假设按综合评分选择最佳k（这里需要根据实际评估标准调整）
            best_k = eval_results.loc[eval_results['silhouette_score'].idxmax(), 'k']
            self.best_clusters[symbol] = int(best_k)
        else:
            # 如果没有评估结果，使用默认的最佳k值
            self.best_clusters[symbol] = min(self.cluster_models.keys())
        
        print(f"   ✅ 加载完成，找到 {len(self.cluster_models)} 个聚类模型")
        print(f"   🎯 最佳k值: {self.best_clusters[symbol]}")
        
        return {
            'models': self.cluster_models,
            'best_clusters': self.best_clusters
        }

    def load_independent_test_data(self, symbol: str = "000001", 
                                  test_days: int = 60) -> Tuple[pd.DataFrame, Dict]:
        """
        使用PCA模块完整流程生成独立测试数据
        
        Parameters:
        -----------
        symbol : str
            股票代码
        test_days : int
            测试数据天数（从当前日期往前推）
            
        Returns:
        --------
        tuple
            (降维后的测试数据DataFrame, PCA相关信息字典)
        """
        print(f"🌐 使用PCA模块完整流程加载{test_days}天独立测试数据...")
        
        try:
            # 使用PCA模块的完整流程生成测试数据
            pca_generator = PCAStateGenerator()
            
            # 1. 运行完整的特征工程流程
            print(f"   🔧 运行完整特征工程流程...")
            features_df = pca_generator.run_complete_feature_pipeline(
                symbol=symbol,
                data_source="influxdb_new",  # 使用独立的InfluxDB新数据
                sample_days=test_days + 100  # 多取一些数据确保有足够的样本
            )
            
            if features_df is None or features_df.empty:
                raise ValueError(f"PCA特征工程流程失败：{symbol}")
            
            print(f"   📊 特征工程完成: {features_df.shape}")
            
            # 2. 运行完整的目标变量流程
            print(f"   🎯 运行完整目标变量流程...")
            targets_df = pca_generator.run_complete_target_pipeline(
                features_df=features_df,
                symbol=symbol
            )
            
            if targets_df is None or targets_df.empty:
                raise ValueError(f"PCA目标变量流程失败：{symbol}")
            
            print(f"   📈 目标变量生成完成: {targets_df.shape}")
            
            # 3. 取最近的test_days天数据作为测试集
            test_data = targets_df.tail(test_days).copy()
            
            if len(test_data) < test_days:
                print(f"   ⚠️ 实际测试数据只有 {len(test_data)} 天，少于要求的 {test_days} 天")
            
            # 4. 构建PCA信息
            pca_info = {
                'data_source': 'influxdb_new_via_pca_module',
                'sample_days': test_days + 100,
                'actual_test_days': len(test_data),
                'features_shape': features_df.shape,
                'targets_shape': targets_df.shape,
                'pca_pipeline': 'complete'
            }
            
            print(f"   ✅ PCA模块完整流程执行成功:")
            print(f"   📊 测试样本: {len(test_data)}")
            print(f"   🎯 特征列数: {test_data.shape[1]}")
            print(f"   🔗 数据来源: InfluxDB新数据 (via PCA模块)")
            
            return test_data, pca_info
            
        except Exception as e:
            print(f"❌ PCA模块完整流程执行失败: {e}")
            raise

    def generate_signals_from_pca(self, selected_k: int, 
                                   symbol: str = "000001") -> pd.DataFrame:
        """
        基于PCA测试数据生成交易信号
        
        Parameters:
        -----------
        selected_k : int
            选择的k值
        symbol : str
            股票代码
            
        Returns:
        --------
        pd.DataFrame
            包含交易信号的数据框
        """
        print(f"📡 使用k={selected_k}生成交易信号...")
        
        # 加载测试数据
        test_data, pca_info = self.load_independent_test_data(symbol)
        
        if test_data is None or test_data.empty:
            raise ValueError("测试数据为空")
        
        # 获取PCA特征列（排除价格和目标变量）
        pca_columns = [col for col in test_data.columns 
                      if col.startswith('PC') and col[2:].isdigit()]
        
        if not pca_columns:
            raise ValueError("未找到PCA特征列")
        
        # 使用对应的聚类模型进行预测
        if selected_k not in self.cluster_models:
            raise ValueError(f"未找到k={selected_k}的聚类模型")
        
        cluster_model = self.cluster_models[selected_k]['model']
        X_pca = test_data[pca_columns].values
        
        # 生成聚类预测
        cluster_labels = cluster_model.predict(X_pca)
        
        # 基于聚类结果生成交易信号
        # 策略逻辑：特定簇标记为买入信号（这里需要根据实际分析调整）
        target_clusters = self._identify_profitable_clusters(selected_k, test_data)
        signals = np.where(np.isin(cluster_labels, target_clusters), 1, 0)
        
        # 创建结果DataFrame
        result_df = test_data.copy()
        result_df['cluster'] = cluster_labels
        result_df['signal'] = signals
        result_df['strategy_position'] = signals.astype(int)
        
        signal_stats = {
            'total_signals': len(signals),
            'buy_signals': np.sum(signals),
            'signal_rate': np.mean(signals),
            'target_clusters': target_clusters
        }
        
        print(f"   ✅ 信号生成完成:")
        print(f"   📊 总样本: {signal_stats['total_signals']}")
        print(f"   📈 买入信号: {signal_stats['buy_signals']}")
        print(f"   📉 信号率: {signal_stats['signal_rate']:.2%}")
        print(f"   🎯 目标簇: {target_clusters}")
        
        return result_df
    
    def _identify_profitable_clusters(self, k: int, test_data: pd.DataFrame) -> List[int]:
        """
        识别盈利的聚类簇（简化版本，实际需要基于历史数据分析）
        
        Parameters:
        -----------
        k : int
            聚类数量
        test_data : pd.DataFrame
            测试数据
            
        Returns:
        --------
        List[int]
            盈利簇的标签列表
        """
        # 这里是简化逻辑，实际应该基于历史回测结果
        # 假设选择聚类中心较高的簇作为买入信号
        if k == 4:
            return [1, 2]  # 假设簇1和簇2表现较好
        elif k == 5:
            return [1, 3]  # 假设簇1和簇3表现较好
        elif k == 6:
            return [2, 4]  # 假设簇2和簇4表现较好
        else:
            return [1]  # 默认选择簇1

    def calculate_strategy_returns(self, signal_data: pd.DataFrame) -> Dict:
        """
        计算策略收益率
        
        Parameters:
        -----------
        signal_data : pd.DataFrame
            包含信号的数据
            
        Returns:
        --------
        Dict
            策略收益统计
        """
        print("💰 计算策略收益...")
        
        # 假设有收盘价列
        if 'close' not in signal_data.columns:
            raise ValueError("数据中缺少收盘价信息")
        
        # 计算日收益率
        signal_data['daily_return'] = signal_data['close'].pct_change()
        
        # 计算策略收益（持仓时获得收益，否则收益为0）
        signal_data['strategy_return'] = signal_data['daily_return'] * signal_data['strategy_position'].shift(1)
        
        # 计算累计收益
        signal_data['cumulative_return'] = (1 + signal_data['daily_return']).cumprod() - 1
        signal_data['strategy_cumulative'] = (1 + signal_data['strategy_return']).cumprod() - 1
        
        # 计算性能指标
        total_return = signal_data['strategy_cumulative'].iloc[-1]
        benchmark_return = signal_data['cumulative_return'].iloc[-1]
        
        # 计算夏普比率（简化版）
        strategy_std = signal_data['strategy_return'].std()
        sharpe_ratio = signal_data['strategy_return'].mean() / strategy_std if strategy_std > 0 else 0
        
        # 最大回撤
        rolling_max = signal_data['strategy_cumulative'].expanding().max()
        drawdown = (signal_data['strategy_cumulative'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        performance = {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'excess_return': total_return - benchmark_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': len(signal_data[signal_data['strategy_return'] > 0]) / len(signal_data[signal_data['strategy_return'] != 0])
        }
        
        print(f"   ✅ 策略收益计算完成:")
        print(f"   📈 策略总收益: {performance['total_return']:.2%}")
        print(f"   📊 基准收益: {performance['benchmark_return']:.2%}")
        print(f"   🎯 超额收益: {performance['excess_return']:.2%}")
        print(f"   📉 最大回撤: {performance['max_drawdown']:.2%}")
        print(f"   🎲 胜率: {performance['win_rate']:.2%}")
        
        return performance

    def run_strategy_backtest(self, symbol: str = "000001", test_days: int = 60) -> Dict:
        """
        运行完整的策略回测
        
        Parameters:
        -----------
        symbol : str
            股票代码
        test_days : int
            测试天数
            
        Returns:
        --------
        Dict
            完整的回测结果
        """
        print(f"🚀 开始策略回测: {symbol}")
        print("=" * 60)
        
        try:
            # 1. 加载聚类分析结果
            cluster_results = self.load_cluster_analysis_results(symbol)
            
            # 2. 获取最佳k值
            best_k = self.best_clusters[symbol]
            
            # 3. 生成交易信号
            signal_data = self.generate_signals_from_pca(best_k, symbol)
            
            # 4. 计算策略收益
            performance = self.calculate_strategy_returns(signal_data)
            
            # 5. 整合结果
            backtest_results = {
                'symbol': symbol,
                'test_period': test_days,
                'best_k': best_k,
                'signal_data': signal_data,
                'performance': performance,
                'cluster_results': cluster_results
            }
            
            print("=" * 60)
            print(f"✅ 策略回测完成: {symbol}")
            print(f"📊 最佳聚类数: k={best_k}")
            print(f"💰 策略表现: {performance['total_return']:.2%} vs 基准 {performance['benchmark_return']:.2%}")
            
            return backtest_results
            
        except Exception as e:
            print(f"❌ 策略回测失败: {e}")
            raise

    def compare_random_baseline(self, signal_data: pd.DataFrame, n_simulations: int = 100) -> Dict:
        """
        与随机基准进行对比
        
        Parameters:
        -----------
        signal_data : pd.DataFrame
            策略信号数据
        n_simulations : int
            随机模拟次数
            
        Returns:
        --------
        Dict
            随机基准对比结果
        """
        print(f"🎲 进行随机基准对比 (模拟{n_simulations}次)...")
        
        daily_returns = signal_data['daily_return'].dropna()
        signal_count = signal_data['signal'].sum()
        
        random_returns = []
        
        for i in range(n_simulations):
            # 生成随机信号（保持相同的信号密度）
            random_signals = np.random.choice([0, 1], size=len(signal_data), 
                                            p=[1-signal_count/len(signal_data), signal_count/len(signal_data)])
            
            # 计算随机策略收益
            random_strategy_return = daily_returns * pd.Series(random_signals[1:])  # shift(1)的效果
            random_cumulative = (1 + random_strategy_return).cumprod().iloc[-1] - 1
            random_returns.append(random_cumulative)
        
        random_returns = np.array(random_returns)
        strategy_return = signal_data['strategy_cumulative'].iloc[-1]
        
        baseline_stats = {
            'random_mean': np.mean(random_returns),
            'random_std': np.std(random_returns),
            'strategy_return': strategy_return,
            'percentile': (random_returns < strategy_return).mean() * 100,
            'outperform_prob': (random_returns < strategy_return).mean()
        }
        
        print(f"   ✅ 随机基准对比完成:")
        print(f"   🎲 随机策略平均收益: {baseline_stats['random_mean']:.2%}")
        print(f"   📊 我们的策略收益: {baseline_stats['strategy_return']:.2%}")
        print(f"   🏆 超越随机基准概率: {baseline_stats['outperform_prob']:.1%}")
        print(f"   📈 收益百分位: {baseline_stats['percentile']:.1f}%")
        
        return baseline_stats


def main():
    """
    主函数 - 运行策略回测示例
    """
    print("🎯 策略回测模块测试")
    print("=" * 60)
    
    try:
        # 初始化回测器
        backtest = StrategyBacktestClean()
        
        # 运行策略回测
        results = backtest.run_strategy_backtest(
            symbol="000001",
            test_days=60
        )
        
        # 随机基准对比
        baseline_comparison = backtest.compare_random_baseline(
            results['signal_data'],
            n_simulations=100
        )
        
        print("\n🎉 回测完成！")
        
    except Exception as e:
        print(f"❌ 回测失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()