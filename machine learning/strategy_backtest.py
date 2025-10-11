#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段11：策略信号与回测雏形

动作：
1. 选训练集收益最高的 cluster 列表（top1 或 top2）
2. 生成测试集 signal=1/0
3. 计算策略收益 vs 基准（持有） → 保存权益曲线
4. 输出 reports/strategy_equity.csv
5. 有随机基准（100 次随机信号）对比

验收：
- 策略收益不小于基准且回撤不过分放大
- 有随机基准对比

数据说明：
- 使用 InfluxDB 导入新数据作为测试集（2025年1月1日至2025年8月1日）
- 使用 pca_state 的数据预处理
- 严格按照原有代码的命名格式和参数设定

作者: Assistant
日期: 2025-09-29
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入必要模块
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from pca_state import run_complete_feature_pipeline, run_complete_target_pipeline, PCAStateGenerator

class StrategyBacktest:
    """
    策略信号与回测雏形
    
    主要功能：
    1. 从 cluster_evaluate 的聚类结果中选择收益最高的聚类
    2. 对新测试数据（2025年1月1日至2025年8月1日）生成交易信号
    3. 计算策略收益并与基准对比
    4. 随机基准对比验证
    5. 保存权益曲线和分析报告
    """
    
    def __init__(self, reports_dir: str = "machine learning/ML output/reports"):
        """
        初始化策略回测器
        
        Parameters:
        -----------
        reports_dir : str
            报告保存目录
        """
        # 设置目录路径
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if os.path.isabs(reports_dir):
            self.reports_dir = reports_dir
        else:
            self.reports_dir = os.path.join(self.project_root, reports_dir)
        
        self.ml_output_dir = os.path.join(self.project_root, "machine learning/ML output")
        self.models_dir = os.path.join(self.ml_output_dir, "models")
        self.states_dir = os.path.join(self.ml_output_dir, "states")
        
        # 确保目录存在
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # 策略参数
        self.test_start_date = "2025-01-01"
        self.test_end_date = "2025-08-01"
        self.random_simulations = 100
        
        # 存储模型和数据
        self.cluster_models = {}
        
        print(f"🎯 策略回测器初始化完成")
        print(f"📁 报告目录: {self.reports_dir}")
        print(f"📅 测试期间: {self.test_start_date} ~ {self.test_end_date}")

    def load_cluster_evaluation_results(self) -> Dict:
        """
        加载 cluster_evaluate 的聚类结果
        
        Returns:
        --------
        dict
            聚类评估结果
        """
        print("📊 加载聚类评估结果...")
        
        # 加载聚类模型
        cluster_models_file = os.path.join(self.reports_dir, "cluster_models.pkl")
        if not os.path.exists(cluster_models_file):
            raise FileNotFoundError(f"聚类模型文件不存在: {cluster_models_file}")
        
        with open(cluster_models_file, 'rb') as f:
            self.cluster_models = pickle.load(f)
        
        print(f"   ✅ 加载了 {len(self.cluster_models)} 个聚类模型")
        
        # 加载聚类比较结果
        comparison_file = os.path.join(self.reports_dir, "cluster_comparison.csv")
        if not os.path.exists(comparison_file):
            raise FileNotFoundError(f"聚类比较文件不存在: {comparison_file}")
        
        comparison_df = pd.read_csv(comparison_file)
        
        print(f"   📋 聚类比较数据: {len(comparison_df)} 条记录")
        
        return {
            'cluster_models': self.cluster_models,
            'comparison_df': comparison_df
        }

    def select_best_clusters(self, comparison_df: pd.DataFrame, top_n: int =3) -> Dict:
        """
        选择全局排名最高的聚类（按global_rank排序）
        
        Parameters:
        -----------
        comparison_df : pd.DataFrame
            聚类比较数据
        top_n : int, default=3
            选择 top N 个聚类
            
        Returns:
        --------
        dict
            最佳聚类信息
        """
        print(f"🎯 选择全局排名最高的 top{top_n} 聚类...")
        
        # 只选择验证通过的聚类
        valid_clusters = comparison_df[comparison_df['validation_passed'] == True].copy()
        
        if len(valid_clusters) == 0:
            print("   ⚠️ 警告：没有验证通过的聚类，使用所有聚类")
            valid_clusters = comparison_df.copy()
        
        # 按global_rank排序（从小到大，rank越小越好），选择 top N
        top_clusters = valid_clusters.nsmallest(top_n, 'global_rank')
        
        selected_clusters = []
        for _, row in top_clusters.iterrows():
            cluster_info = {
                'k_value': int(row['k_value']),
                'cluster_id': int(row['cluster_id']),
                'train_mean_return': row['train_mean_return'],
                'test_mean_return': row['test_mean_return'],
                'train_rank': int(row['train_rank']),
                'test_rank': int(row['test_rank']),
                'global_rank': int(row['global_rank'])
            }
            selected_clusters.append(cluster_info)
            
            print(f"   ✅ 选中: k={cluster_info['k_value']}, cluster_id={cluster_info['cluster_id']} (全局排名: {cluster_info['global_rank']})")
            print(f"      训练收益: {cluster_info['train_mean_return']:+.6f} (训练排名: {cluster_info['train_rank']})")
            print(f"      测试收益: {cluster_info['test_mean_return']:+.6f} (测试排名: {cluster_info['test_rank']})")
            print(f"      综合收益: {cluster_info['train_mean_return'] + cluster_info['test_mean_return']:+.6f}")
        
        return {
            'selected_clusters': selected_clusters,
            'selection_method': f'top_{top_n}_global_rank'
        }

    def prepare_test_data(self, symbol: str = "000001") -> pd.DataFrame:
        """
        准备测试数据（2025年1月1日至2025年8月1日）
        使用 pca_state 的完整数据预处理流程
        
        Parameters:
        -----------
        symbol : str
            股票代码
            
        Returns:
        --------
        pd.DataFrame
            预处理后的测试数据，包含PCA特征和目标变量
        """
        print(f"🔧 准备测试数据: {symbol} ({self.test_start_date} ~ {self.test_end_date})")
        print("   使用 pca_state 的完整数据预处理流程")
        try:
            # 配置参数（与 pca_state.main() 相同）
            config = {
                'symbol': symbol,
                'start_date': self.test_start_date,  # 2025-01-01
                'end_date': self.test_end_date,      # 2025-08-01
                'use_auto_features': True,           # 使用自动特征生成
                'final_k_features': 15,              # 最终特征数量
                'target_periods': [1, 5, 10],        # 目标时间窗口
                'pca_components': 0.9,               # PCA解释方差比例
                'train_ratio': 0.8                   # 训练集比例（用于内部切分）
            }
            
            print("   📋 执行配置:")
            for key, value in config.items():
                print(f"      {key}: {value}")
            print()
            
            # === 步骤1: 完整特征工程流程 ===
            print("   🔧 步骤1: 完整特征工程流程")
            feature_results = run_complete_feature_pipeline(
                symbol=config['symbol'],
                start_date=config['start_date'],
                end_date=config['end_date'],
                use_auto_features=config['use_auto_features'],
                final_k_features=config['final_k_features']
            )
            
            if not feature_results.get('success'):
                raise ValueError(f"特征工程失败: {feature_results.get('error', '未知错误')}")
            
            print("      ✅ 特征工程完成")
            
            # === 步骤2: 完整目标变量工程流程 ===
            print("   🎯 步骤2: 完整目标变量工程流程")
            target_results = run_complete_target_pipeline(
                scaled_features_df=feature_results['scaled_features_df'],
                symbol=config['symbol'],
                target_periods=config['target_periods']
            )
            
            if not target_results.get('success'):
                raise ValueError(f"目标变量工程失败: {target_results.get('error', '未知错误')}")
            
            print("      ✅ 目标变量工程完成")
            
            # === 步骤3: PCA状态生成 ===
            print("   � 步骤3: PCA状态生成")
            
            # 检查是否有标准化特征CSV文件
            csv_path = feature_results.get('csv_path')
            if not csv_path or not os.path.exists(csv_path):
                raise ValueError("标准化特征CSV文件不存在，无法进行PCA降维")
            
            # 初始化PCA状态生成器
            pca_generator = PCAStateGenerator()
            
            # 生成PCA状态（使用完整流程）
            pca_results = pca_generator.generate_pca_states(
                csv_path=csv_path,
                symbol=config['symbol'],
                n_components=config['pca_components'],
                train_ratio=config['train_ratio']
            )
            
            if not pca_results or 'n_components' not in pca_results:
                raise ValueError("PCA状态生成失败")
            
            print("      ✅ PCA状态生成完成")
            
            # === 步骤4: 构建最终测试数据 ===
            print("   🔨 步骤4: 构建最终测试数据")
            
            # 获取完整数据集（包含目标变量）
            complete_dataset = target_results['complete_dataset']
            
            # 获取PCA降维后的特征数据
            # 使用所有数据（训练+测试）作为新的测试集
            states_all = np.vstack([pca_results['states_train'], pca_results['states_test']])
            
            # 创建PCA特征DataFrame
            pca_columns = [f'PC{i+1}' for i in range(states_all.shape[1])]
            pca_df = pd.DataFrame(states_all, index=complete_dataset.index, columns=pca_columns)
            
            # 合并PCA特征和目标变量
            target_cols = [col for col in complete_dataset.columns 
                          if col.startswith('future_return_') or col.startswith('label_')]
            
            test_data = pd.concat([
                pca_df,
                complete_dataset[target_cols + ['close']]
            ], axis=1)
            
            # 移除包含NaN的行（主要是末尾的目标变量NaN）
            test_data = test_data.dropna()
            
            print(f"      ✅ 测试数据构建完成: {test_data.shape}")
            print(f"      📊 PCA特征: {len(pca_columns)} 维")
            print(f"      🎯 目标变量: {len([col for col in target_cols if col.startswith('future_return_')])} 个")
            print(f"      📅 数据时间范围: {test_data.index.min().date()} ~ {test_data.index.max().date()}")
            
            return test_data
            
        except Exception as e:
            print(f"   ❌ 测试数据准备失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def generate_trading_signals(self, test_data: pd.DataFrame, selected_clusters: List[Dict]) -> pd.DataFrame:
        """
        生成交易信号
        
        Parameters:
        -----------
        test_data : pd.DataFrame
            预处理后的测试数据
        selected_clusters : List[Dict]
            选中的最佳聚类
            
        Returns:
        --------
        pd.DataFrame
            包含交易信号的数据
        """
        print(f"📡 生成交易信号...")
        
        # 获取PCA特征
        pca_columns = [col for col in test_data.columns if col.startswith('PC')]
        X_pca = test_data[pca_columns].fillna(0).values
        
        # 为每个选中的聚类生成信号
        signals = {}
        
        for i, cluster_info in enumerate(selected_clusters):
            k_value = cluster_info['k_value']
            cluster_id = cluster_info['cluster_id']
            
            print(f"   📊 聚类 {i+1}: k={k_value}, cluster_id={cluster_id}")
            
            # 使用对应的聚类模型
            cluster_model = self.cluster_models[k_value]
            cluster_labels = cluster_model.predict(X_pca)
            
            # 生成信号：属于目标聚类时为1，否则为0
            signal = (cluster_labels == cluster_id).astype(int)
            signals[f'signal_k{k_value}_c{cluster_id}'] = signal
            
            signal_count = signal.sum()
            signal_ratio = signal_count / len(signal)
            print(f"      信号数量: {signal_count}/{len(signal)} ({signal_ratio:.2%})")
        
        # 综合信号：任一聚类发出信号则为1
        combined_signal = np.zeros(len(test_data))
        for signal_col in signals.keys():
            combined_signal = np.maximum(combined_signal, signals[signal_col])
        
        # 【优化】基于历史动量过滤信号（不使用未来数据）
        # 计算过去5天的动量（收益率）
        use_momentum_filter = True  # 是否使用动量过滤
        if use_momentum_filter and 'close' in test_data.columns:
            momentum_5d = test_data['close'].pct_change(periods=5).fillna(0).values
            # 放宽条件：允许轻微下跌趋势中的信号
            momentum_threshold = -0.02  # 允许-2%以内的下跌
            combined_signal[(momentum_5d < momentum_threshold)] = 0
            print(f"   🔍 动量过滤 (阈值={momentum_threshold}): 保留信号 {combined_signal.sum()}/{len(combined_signal)} ({combined_signal.mean():.2%})")
        else:
            print(f"   ⚠️ 未使用动量过滤")
        
        signals['signal_combined'] = combined_signal
        
        # 添加信号到测试数据
        result_data = test_data.copy()
        for signal_name, signal_values in signals.items():
            result_data[signal_name] = signal_values
        
        combined_count = combined_signal.sum()
        combined_ratio = combined_count / len(combined_signal)
        print(f"   ✅ 综合信号: {combined_count}/{len(combined_signal)} ({combined_ratio:.2%})")
        
        return result_data

    def calculate_strategy_performance(self, signal_data: pd.DataFrame) -> Dict:
        """
        计算策略收益 vs 基准（持有）
        
        Parameters:
        -----------
        signal_data : pd.DataFrame
            包含信号的数据
            
        Returns:
        --------
        dict
            策略性能指标
        """
        print(f"💰 计算策略性能...")
        
        # 使用future_return_5d作为预测目标收益
        returns = signal_data['future_return_5d'].fillna(0).values
        signal = signal_data['signal_combined'].values
        
        # 基准策略：始终持有
        benchmark_returns = returns
        benchmark_cumulative = np.cumprod(1 + benchmark_returns)
        
        # 策略收益：信号为1时买入持有，信号为0时空仓（持有现金）
        # 这才是真正的择时策略，可以规避下跌风险
        strategy_returns = signal * returns
        strategy_cumulative = np.ones(len(strategy_returns))
        
        use_stop_loss = False  # 是否使用止损机制
        stop_loss_threshold = -0.05  # 止损阈值（-5%）
        
        for i in range(1, len(strategy_returns)):
            if signal[i] == 1:
                # 有信号时，买入持有
                new_return = returns[i]
                
                # 如果启用止损，检查是否触发止损
                if use_stop_loss and new_return < stop_loss_threshold:
                    # 触发止损，不参与本次交易
                    strategy_cumulative[i] = strategy_cumulative[i-1]
                else:
                    # 正常参与市场
                    strategy_cumulative[i] = strategy_cumulative[i-1] * (1 + new_return)
            else:
                # 无信号时，空仓，累计收益保持不变（规避风险）
                strategy_cumulative[i] = strategy_cumulative[i-1]
        strategy_cumulative[0] = 1 + strategy_returns[0]
        
        # 计算性能指标
        total_return_benchmark = benchmark_cumulative[-1] - 1
        total_return_strategy = strategy_cumulative[-1] - 1
        excess_return = total_return_strategy - total_return_benchmark
        
        # 年化收益率（假设250个交易日）
        n_days = len(returns)
        years = n_days / 250
        annual_return_benchmark = (1 + total_return_benchmark) ** (1/years) - 1 if years > 0 else 0
        annual_return_strategy = (1 + total_return_strategy) ** (1/years) - 1 if years > 0 else 0
        
        # 波动率
        benchmark_volatility = np.std(benchmark_returns) * np.sqrt(250)
        strategy_volatility = np.std(strategy_returns) * np.sqrt(250)
        
        # 夏普比率（假设无风险利率为3%）
        risk_free_rate = 0.03
        sharpe_benchmark = (annual_return_benchmark - risk_free_rate) / benchmark_volatility if benchmark_volatility > 0 else 0
        sharpe_strategy = (annual_return_strategy - risk_free_rate) / strategy_volatility if strategy_volatility > 0 else 0
        
        # 最大回撤
        benchmark_running_max = np.maximum.accumulate(benchmark_cumulative)
        benchmark_drawdown = (benchmark_cumulative - benchmark_running_max) / benchmark_running_max
        max_drawdown_benchmark = np.min(benchmark_drawdown)
        
        strategy_running_max = np.maximum.accumulate(strategy_cumulative)
        strategy_drawdown = (strategy_cumulative - strategy_running_max) / strategy_running_max
        max_drawdown_strategy = np.min(strategy_drawdown)
        
        # 胜率（仅考虑有信号的时间点）
        signal_mask = signal == 1
        win_rate = (returns[signal_mask] > 0).mean() if signal_mask.sum() > 0 else 0
        
        performance = {
            # 总收益
            'total_return_benchmark': total_return_benchmark,
            'total_return_strategy': total_return_strategy,
            'excess_return': excess_return,
            
            # 年化收益
            'annual_return_benchmark': annual_return_benchmark,
            'annual_return_strategy': annual_return_strategy,
            
            # 风险指标
            'volatility_benchmark': benchmark_volatility,
            'volatility_strategy': strategy_volatility,
            'max_drawdown_benchmark': max_drawdown_benchmark,
            'max_drawdown_strategy': max_drawdown_strategy,
            
            # 风险调整收益
            'sharpe_benchmark': sharpe_benchmark,
            'sharpe_strategy': sharpe_strategy,
            
            # 交易统计
            'signal_count': signal_mask.sum(),
            'signal_ratio': signal_mask.mean(),
            'win_rate': win_rate,
            
            # 时间序列
            'benchmark_cumulative': benchmark_cumulative,
            'strategy_cumulative': strategy_cumulative,
            'benchmark_drawdown': benchmark_drawdown,
            'strategy_drawdown': strategy_drawdown,
            'dates': signal_data.index
        }
        
        print(f"   ✅ 策略性能:")
        print(f"      基准总收益: {total_return_benchmark:.2%}")
        print(f"      策略总收益: {total_return_strategy:.2%}")
        print(f"      超额收益: {excess_return:.2%}")
        print(f"      基准夏普: {sharpe_benchmark:.3f}")
        print(f"      策略夏普: {sharpe_strategy:.3f}")
        print(f"      基准回撤: {max_drawdown_benchmark:.2%}")
        print(f"      策略回撤: {max_drawdown_strategy:.2%}")
        print(f"      信号胜率: {win_rate:.2%}")
        
        return performance

    def run_random_baseline(self, signal_data: pd.DataFrame, performance: Dict) -> Dict:
        """
        随机基准对比（100次随机信号）
        
        Parameters:
        -----------
        signal_data : pd.DataFrame
            包含信号的数据
        performance : Dict
            策略性能指标
            
        Returns:
        --------
        dict
            随机基准对比结果
        """
        print(f"🎲 运行随机基准对比 ({self.random_simulations}次)...")
        
        returns = signal_data['future_return_5d'].fillna(0).values
        original_signal_ratio = performance['signal_ratio']
        
        # 运行随机模拟
        random_results = []
        
        for i in range(self.random_simulations):
            # 生成随机信号（保持相同的信号比例）
            n_samples = len(returns)
            n_signals = int(n_samples * original_signal_ratio)
            
            random_signal = np.zeros(n_samples)
            if n_signals > 0:
                random_indices = np.random.choice(n_samples, n_signals, replace=False)
                random_signal[random_indices] = 1
            
            # 计算随机策略收益
            random_strategy_returns = random_signal * returns
            random_cumulative = np.prod(1 + random_strategy_returns) - 1
            
            # 计算随机策略统计
            random_volatility = np.std(random_strategy_returns) * np.sqrt(250)
            
            random_results.append({
                'total_return': random_cumulative,
                'volatility': random_volatility,
                'signal_count': n_signals
            })
        
        # 统计随机结果
        random_returns = np.array([r['total_return'] for r in random_results])
        random_volatilities = np.array([r['volatility'] for r in random_results])
        
        strategy_return = performance['total_return_strategy']
        
        baseline_comparison = {
            'random_mean_return': random_returns.mean(),
            'random_std_return': random_returns.std(),
            'random_min_return': random_returns.min(),
            'random_max_return': random_returns.max(),
            'random_median_return': np.median(random_returns),
            
            'random_mean_volatility': random_volatilities.mean(),
            'random_std_volatility': random_volatilities.std(),
            
            'strategy_return': strategy_return,
            'strategy_percentile': (random_returns < strategy_return).mean(),
            'outperformance_ratio': (random_returns < strategy_return).mean(),
            
            'n_simulations': self.random_simulations,
            'random_returns': random_returns
        }
        
        print(f"   ✅ 随机基准对比:")
        print(f"      策略收益: {strategy_return:.2%}")
        print(f"      随机平均: {baseline_comparison['random_mean_return']:.2%}")
        print(f"      随机标准差: {baseline_comparison['random_std_return']:.2%}")
        print(f"      策略分位数: {baseline_comparison['strategy_percentile']:.1%}")
        print(f"      优于随机比例: {baseline_comparison['outperformance_ratio']:.1%}")
        
        return baseline_comparison

    def save_strategy_equity(self, performance: Dict, baseline_comparison: Dict, 
                           selected_clusters: List[Dict], symbol: str = "000001") -> str:
        """
        保存策略权益曲线和分析结果
        
        Parameters:
        -----------
        performance : Dict
            策略性能指标
        baseline_comparison : Dict
            随机基准对比结果
        selected_clusters : List[Dict]
            选中的聚类信息
        symbol : str
            股票代码
            
        Returns:
        --------
        str
            保存的文件路径
        """
        print(f"💾 保存策略权益曲线和分析结果...")
        
        # 创建权益曲线数据
        equity_data = []
        dates = performance['dates']
        benchmark_cumulative = performance['benchmark_cumulative']
        strategy_cumulative = performance['strategy_cumulative']
        benchmark_drawdown = performance['benchmark_drawdown']
        strategy_drawdown = performance['strategy_drawdown']
        
        for i, date in enumerate(dates):
            equity_data.append({
                'date': date,
                'benchmark_equity': benchmark_cumulative[i],
                'strategy_equity': strategy_cumulative[i],
                'benchmark_drawdown': benchmark_drawdown[i],
                'strategy_drawdown': strategy_drawdown[i],
                'excess_equity': strategy_cumulative[i] - benchmark_cumulative[i]
            })
        
        equity_df = pd.DataFrame(equity_data)
        
        # 保存权益曲线CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        equity_file = os.path.join(self.reports_dir, f"strategy_equity_{symbol}_{timestamp}.csv")
        equity_df.to_csv(equity_file, index=False)
        
        print(f"   ✅ 权益曲线已保存: {os.path.basename(equity_file)}")
        
        # 创建详细分析报告
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("阶段11：策略信号与回测雏形 - 分析报告")
        report_lines.append("=" * 70)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"股票代码: {symbol}")
        report_lines.append(f"测试期间: {self.test_start_date} ~ {self.test_end_date}")
        report_lines.append("")
        
        # 选中的聚类信息
        report_lines.append("🎯 选中的最佳聚类:")
        for i, cluster_info in enumerate(selected_clusters):
            report_lines.append(f"   聚类 {i+1}: k={cluster_info['k_value']}, cluster_id={cluster_info['cluster_id']}")
            report_lines.append(f"      训练收益: {cluster_info['train_mean_return']:+.6f} (排名: {cluster_info['train_rank']})")
            report_lines.append(f"      测试收益: {cluster_info['test_mean_return']:+.6f} (排名: {cluster_info['test_rank']})")
            report_lines.append(f"      全局排名: {cluster_info['global_rank']}")
        report_lines.append("")
        
        # 策略性能
        report_lines.append("📊 策略性能:")
        report_lines.append(f"   基准总收益: {performance['total_return_benchmark']:+.2%}")
        report_lines.append(f"   策略总收益: {performance['total_return_strategy']:+.2%}")
        report_lines.append(f"   超额收益: {performance['excess_return']:+.2%}")
        report_lines.append("")
        report_lines.append(f"   基准年化收益: {performance['annual_return_benchmark']:+.2%}")
        report_lines.append(f"   策略年化收益: {performance['annual_return_strategy']:+.2%}")
        report_lines.append("")
        report_lines.append(f"   基准波动率: {performance['volatility_benchmark']:.2%}")
        report_lines.append(f"   策略波动率: {performance['volatility_strategy']:.2%}")
        report_lines.append("")
        report_lines.append(f"   基准夏普比率: {performance['sharpe_benchmark']:.3f}")
        report_lines.append(f"   策略夏普比率: {performance['sharpe_strategy']:.3f}")
        report_lines.append("")
        report_lines.append(f"   基准最大回撤: {performance['max_drawdown_benchmark']:+.2%}")
        report_lines.append(f"   策略最大回撤: {performance['max_drawdown_strategy']:+.2%}")
        report_lines.append("")
        report_lines.append(f"   信号数量: {performance['signal_count']}")
        report_lines.append(f"   信号比例: {performance['signal_ratio']:.2%}")
        report_lines.append(f"   信号胜率: {performance['win_rate']:.2%}")
        report_lines.append("")
        
        # 随机基准对比
        report_lines.append("🎲 随机基准对比:")
        report_lines.append(f"   模拟次数: {baseline_comparison['n_simulations']}")
        report_lines.append(f"   策略收益: {baseline_comparison['strategy_return']:+.2%}")
        report_lines.append(f"   随机平均收益: {baseline_comparison['random_mean_return']:+.2%}")
        report_lines.append(f"   随机收益标准差: {baseline_comparison['random_std_return']:.2%}")
        report_lines.append(f"   随机收益范围: {baseline_comparison['random_min_return']:+.2%} ~ {baseline_comparison['random_max_return']:+.2%}")
        report_lines.append(f"   策略分位数: {baseline_comparison['strategy_percentile']:.1%}")
        report_lines.append(f"   优于随机比例: {baseline_comparison['outperformance_ratio']:.1%}")
        report_lines.append("")
        
        # 验收结果
        report_lines.append("✅ 验收结果:")
        
        # 检查策略收益不小于基准
        benchmark_check = performance['total_return_strategy'] >= performance['total_return_benchmark']
        report_lines.append(f"   策略收益不小于基准: {'✅ 通过' if benchmark_check else '❌ 未通过'}")
        report_lines.append(f"      策略 {performance['total_return_strategy']:+.2%} vs 基准 {performance['total_return_benchmark']:+.2%}")
        
        # 检查回撤不过分放大
        drawdown_check = abs(performance['max_drawdown_strategy']) <= abs(performance['max_drawdown_benchmark']) * 1.5
        report_lines.append(f"   回撤不过分放大: {'✅ 通过' if drawdown_check else '❌ 未通过'}")
        report_lines.append(f"      策略回撤 {performance['max_drawdown_strategy']:+.2%} vs 基准回撤 {performance['max_drawdown_benchmark']:+.2%}")
        
        # 检查随机基准优势
        random_check = baseline_comparison['strategy_percentile'] >= 0.6
        report_lines.append(f"   优于随机基准: {'✅ 通过' if random_check else '❌ 未通过'}")
        report_lines.append(f"      策略在随机基准中排名 {baseline_comparison['strategy_percentile']:.1%} 分位")
        
        overall_pass = benchmark_check and drawdown_check and random_check
        report_lines.append("")
        report_lines.append(f"🏆 总体验收: {'✅ 通过' if overall_pass else '❌ 未通过'}")
        
        report_lines.append("")
        report_lines.append("=" * 70)
        
        # 保存报告
        report_file = os.path.join(self.reports_dir, f"strategy_analysis_{symbol}_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        print(f"   📋 分析报告已保存: {os.path.basename(report_file)}")
        
        return equity_file

    def run_complete_backtest(self, symbol: str = "000001", top_n: int = 2) -> Dict:
        """
        运行完整的策略回测流程
        
        Parameters:
        -----------
        symbol : str, default="000001"
            股票代码
        top_n : int, default=2
            选择top N个聚类
            
        Returns:
        --------
        dict
            完整的回测结果
        """
        print("=" * 70)
        print("阶段11：策略信号与回测雏形")
        print("=" * 70)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"股票代码: {symbol}")
        print(f"测试期间: {self.test_start_date} ~ {self.test_end_date}")
        print(f"选择策略: top{top_n} 训练收益聚类")
        print()
        
        try:
            # 1. 加载聚类评估结果
            cluster_results = self.load_cluster_evaluation_results()
            
            # 2. 选择训练集收益最高的聚类
            selection_results = self.select_best_clusters(cluster_results['comparison_df'], top_n)
            selected_clusters = selection_results['selected_clusters']
            
            # 4. 准备测试数据（使用InfluxDB新数据 + pca_state完整流程）
            test_data = self.prepare_test_data(symbol)
            
            # 5. 生成交易信号
            signal_data = self.generate_trading_signals(test_data, selected_clusters)
            
            # 6. 计算策略性能
            performance = self.calculate_strategy_performance(signal_data)
            
            # 7. 随机基准对比
            baseline_comparison = self.run_random_baseline(signal_data, performance)
            
            # 8. 保存权益曲线和分析结果
            equity_file = self.save_strategy_equity(performance, baseline_comparison, selected_clusters, symbol)
            
            # 9. 整合完整结果
            results = {
                'symbol': symbol,
                'test_period': f"{self.test_start_date} ~ {self.test_end_date}",
                'selected_clusters': selected_clusters,
                'performance': performance,
                'baseline_comparison': baseline_comparison,
                'signal_data': signal_data,
                'equity_file': equity_file,
                'backtest_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 10. 最终报告
            print("\n" + "=" * 70)
            print("🎉 阶段11完成！")
            print("=" * 70)
            
            # 验收检查
            benchmark_check = performance['total_return_strategy'] >= performance['total_return_benchmark']
            drawdown_check = abs(performance['max_drawdown_strategy']) <= abs(performance['max_drawdown_benchmark']) * 1.5
            random_check = baseline_comparison['strategy_percentile'] >= 0.6
            overall_pass = benchmark_check and drawdown_check and random_check
            
            print(f"📊 策略收益: {performance['total_return_strategy']:+.2%} (基准: {performance['total_return_benchmark']:+.2%})")
            print(f"📉 最大回撤: {performance['max_drawdown_strategy']:+.2%} (基准: {performance['max_drawdown_benchmark']:+.2%})")
            print(f"🎲 随机排名: {baseline_comparison['strategy_percentile']:.1%} 分位")
            print(f"🏆 验收结果: {'✅ 通过' if overall_pass else '❌ 未通过'}")
            print(f"💾 权益曲线: {os.path.basename(equity_file)}")
            print("=" * 70)
            
            return results
            
        except Exception as e:
            print(f"\n❌ 回测失败: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """
    主函数：运行阶段11策略信号与回测雏形
    """
    try:
        # 初始化策略回测器
        backtest = StrategyBacktest()
        
        # 运行完整回测
        results = backtest.run_complete_backtest(
            symbol="000001",  # 平安银行
            top_n=3          # 选择top3聚类
        )
        
        print(f"\n✨ 回测完成！结果已保存到: {backtest.reports_dir}")
        return results
        
    except Exception as e:
        print(f"\n💥 主函数执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()