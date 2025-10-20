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
        self.test_start_date = "2023-01-01"
        self.test_end_date = "2024-12-01"
        self.random_simulations = 100
        
        # 存储模型和数据
        self.cluster_models = {}
        
        print(f"🎯 策略回测器初始化完成")
        print(f"📁 报告目录: {self.reports_dir}")
        print(f"📅 测试期间: {self.test_start_date} ~ {self.test_end_date}")
        
        # 存储训练阶段选择的最佳PC信息
        self.best_pc = None
        self.pc_direction = None
        self.pc_threshold = None
        self.pc_threshold_quantile = None

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
        
        # 加载PC元数据（从cluster_evaluate的训练阶段保存）
        pc_metadata_file = os.path.join(self.reports_dir, "pc_metadata.pkl")
        if os.path.exists(pc_metadata_file):
            with open(pc_metadata_file, 'rb') as f:
                pc_metadata = pickle.load(f)
            
            self.best_pc = pc_metadata.get('best_pc', 'PC1')
            self.pc_direction = pc_metadata.get('pc_direction', 1.0)
            self.pc_threshold = pc_metadata.get('pc_threshold', 0.0)
            self.pc_threshold_quantile = pc_metadata.get('threshold_quantile', 0.6)
            ic_value = pc_metadata.get('ic_value', 0.0)
            
            print(f"   ✅ 加载PC元数据: {self.best_pc} (IC={ic_value:+.4f}, 方向={self.pc_direction:+.1f}, 门槛={self.pc_threshold:.4f})")
        else:
            print(f"   ⚠️ 未找到PC元数据文件，将使用默认PC1")
            self.best_pc = 'PC1'
            self.pc_direction = 1.0
            self.pc_threshold = 0.0
            self.pc_threshold_quantile = 0.6
        
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

    def select_best_clusters(self, comparison_df: pd.DataFrame, top_n: int = 3, 
                           min_cluster_pct: float = 0.10, max_cluster_pct: float = 0.60) -> Dict:
        """
        选择全局排名最高的聚类（按global_rank排序）
        
        约束条件（避免极端簇）：
        1. 簇占比必须在 [min_cluster_pct, max_cluster_pct] 区间内
        2. 样本外收益（test_mean_return）必须为正
        3. 必须通过验证（validation_passed=True）
        
        Parameters:
        -----------
        comparison_df : pd.DataFrame
            聚类比较数据，必须包含train_samples列
        top_n : int, default=3
            选择 top N 个聚类
        min_cluster_pct : float, default=0.10
            最小簇占比（10%），低于此值的簇会被过滤
        max_cluster_pct : float, default=0.60
            最大簇占比（60%），高于此值的簇会被过滤
            
        Returns:
        --------
        dict
            最佳聚类信息
        """
        print(f"🎯 选择全局排名最高的 top{top_n} 聚类...")
        print(f"   � 簇占比约束: [{min_cluster_pct:.0%}, {max_cluster_pct:.0%}]")
        print(f"   📈 样本外收益约束: 必须 > 0")
        
        # === 步骤1: 只选择验证通过的聚类 ===
        valid_clusters = comparison_df[comparison_df['validation_passed'] == True].copy()
        
        if len(valid_clusters) == 0:
            print("   ⚠️ 警告：没有验证通过的聚类，使用所有聚类")
            valid_clusters = comparison_df.copy()
        
        print(f"   ✅ 验证通过: {len(valid_clusters)}/{len(comparison_df)} 个聚类")
        
        # === 步骤2: 过滤簇占比异常的簇 ===
        if 'train_samples' in valid_clusters.columns:
            group_totals = valid_clusters.groupby('k_value')['train_samples'].transform('sum')
            # 避免除以0
            group_totals = group_totals.replace(0, np.nan)
            valid_clusters['cluster_pct'] = valid_clusters['train_samples'] / group_totals
            
            # 过滤掉占比过小/过大的簇
            before_count = len(valid_clusters)
            valid_clusters = valid_clusters[
                (valid_clusters['cluster_pct'].notna()) &
                (valid_clusters['cluster_pct'] >= min_cluster_pct) & 
                (valid_clusters['cluster_pct'] <= max_cluster_pct)
            ].copy()
            after_count = len(valid_clusters)
            
            if before_count > after_count:
                filtered = before_count - after_count
                print(f"   🗑️  过滤占比异常簇: {filtered} 个 (占比不在[{min_cluster_pct:.0%}, {max_cluster_pct:.0%}])")
            
            if len(valid_clusters) == 0:
                print("   ⚠️ 警告: 所有簇都被过滤，放宽占比要求")
                valid_clusters = comparison_df[comparison_df['validation_passed'] == True].copy()
                if 'train_samples' in valid_clusters.columns:
                    group_totals = valid_clusters.groupby('k_value')['train_samples'].transform('sum').replace(0, np.nan)
                    valid_clusters['cluster_pct'] = valid_clusters['train_samples'] / group_totals
        else:
            print("   ⚠️ 警告: 数据中无train_samples列，跳过占比过滤")
        
        # === 步骤3: 过滤样本外收益为负的簇 ===
        if 'test_mean_return' in valid_clusters.columns:
            before_count = len(valid_clusters)
            valid_clusters = valid_clusters[valid_clusters['test_mean_return'] > 0].copy()
            after_count = len(valid_clusters)
            
            if before_count > after_count:
                filtered = before_count - after_count
                print(f"   🗑️  过滤样本外负收益簇: {filtered} 个")
            
            if len(valid_clusters) == 0:
                print("   ⚠️ 警告: 所有簇样本外收益都为负，退化选择")
                valid_clusters = comparison_df[comparison_df['validation_passed'] == True].copy()
                if 'train_samples' in valid_clusters.columns:
                    total_train_samples = valid_clusters['train_samples'].sum()
                    valid_clusters['cluster_pct'] = valid_clusters['train_samples'] / total_train_samples
        
        # === 步骤4: 按global_rank排序（从小到大，rank越小越好），选择 top N ===
        top_clusters = valid_clusters.nsmallest(top_n, 'global_rank')
        
        selected_clusters = []
        for _, row in top_clusters.iterrows():
            cluster_pct = row.get('cluster_pct', None)
            
            cluster_info = {
                'k_value': int(row['k_value']),
                'cluster_id': int(row['cluster_id']),
                'train_mean_return': row['train_mean_return'],
                'test_mean_return': row['test_mean_return'],
                'train_rank': int(row['train_rank']),
                'test_rank': int(row['test_rank']),
                'global_rank': int(row['global_rank']),
                'train_samples': int(row['train_samples']) if 'train_samples' in row else None,
                'cluster_pct': float(cluster_pct) if cluster_pct is not None else None
            }
            selected_clusters.append(cluster_info)
            
            pct_info = f"占比: {cluster_pct:.1%}" if cluster_pct is not None else ""
            print(f"   ✅ 选中: k={cluster_info['k_value']}, cluster_id={cluster_info['cluster_id']} "
                  f"(全局排名: {cluster_info['global_rank']}) {pct_info}")
            print(f"      训练收益: {cluster_info['train_mean_return']:+.6f} (训练排名: {cluster_info['train_rank']})")
            print(f"      测试收益: {cluster_info['test_mean_return']:+.6f} (测试排名: {cluster_info['test_rank']})")
        
        return {
            'selected_clusters': selected_clusters,
            'selection_method': f'top_{top_n}_global_rank_with_constraints'
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
            
            # 【关键修复】PCA由于purge导致训练集减少了purge_periods行
            # 需要根据PCA的train_index和test_index来对齐数据
            train_index = pca_results['train_index']
            test_index = pca_results['test_index']
            
            # 合并train和test的索引（注意：train_index已经被purge过了）
            # 所以我们需要使用PCA实际使用的索引
            pca_used_indices = train_index.union(test_index)
            
            # 从complete_dataset中提取对应的数据
            complete_dataset_aligned = complete_dataset.loc[pca_used_indices]
            
            print(f"      📊 完整数据集: {len(complete_dataset)} 行")
            print(f"      📊 PCA使用数据: {len(pca_used_indices)} 行 (训练:{len(train_index)} + 测试:{len(test_index)})")
            print(f"      🚫 Purge gap: {len(complete_dataset) - len(pca_used_indices)} 行")
            
            # 获取PCA降维后的特征数据
            states_all = np.vstack([pca_results['states_train'], pca_results['states_test']])
            
            # 创建PCA特征DataFrame（使用对齐后的索引）
            pca_columns = [f'PC{i+1}' for i in range(states_all.shape[1])]
            pca_df = pd.DataFrame(states_all, index=complete_dataset_aligned.index, columns=pca_columns)
            
            # 合并PCA特征和目标变量（使用对齐后的数据）
            target_cols = [col for col in complete_dataset_aligned.columns 
                          if col.startswith('future_return_') or col.startswith('label_')]
            
            test_data = pd.concat([
                pca_df,
                complete_dataset_aligned[target_cols + ['close']]
            ], axis=1)
            
            # 移除包含NaN的行（主要是末尾的目标变量NaN）
            test_data = test_data.dropna()
            
            print(f"      ✅ 测试数据构建完成: {test_data.shape}")
            print(f"      📊 PCA特征: {len(pca_columns)} 维")
            print(f"      🎯 目标变量: {len([col for col in target_cols if col.startswith('future_return_')])} 个")
            print(f"      📅 数据时间范围: {test_data.index.min().date()} ~ {test_data.index.max().date()}")
            
            # === 步骤5: PC信息已从cluster_evaluate的训练阶段加载 ===
            print("   ℹ️ 步骤5: 使用已加载的PC元数据")
            print(f"      📌 最佳PC: {self.best_pc} (方向: {self.pc_direction:+.1f})")
            if self.pc_threshold is not None:
                quantile = self.pc_threshold_quantile if self.pc_threshold_quantile is not None else 0.6
                print(f"      🎯 强度门槛: {self.pc_threshold:.4f} (q={quantile:.2f})")
            print("      💡 PC信息来源: cluster_evaluate训练阶段（历史数据）")
            
            return test_data
            
        except Exception as e:
            print(f"   ❌ 测试数据准备失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def generate_trading_signals(self, test_data: pd.DataFrame, selected_clusters: List[Dict]) -> pd.DataFrame:
        """
        生成交易信号（改进版：避免前视偏差）
        
        策略逻辑：
        1. 聚类状态过滤：属于选中簇时候选
        2. PC强度门槛：使用训练阶段选择的最佳PC、方向和门槛值（避免前视偏差）
        3. 持有期：信号触发后持有3期
        
        关键改进：
        - 最佳PC的选择、方向统一、门槛计算均在训练阶段完成
        - 测试阶段仅应用训练阶段确定的规则，不再重新选择或计算
        - 彻底避免测试数据参与规则选择的前视偏差
        
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
        print(f"📡 生成交易信号 (聚类状态 + PC强度门槛 + 持有期)...")
        
        # 获取PCA特征
        pca_columns = [col for col in test_data.columns if col.startswith('PC')]
        X_pca = test_data[pca_columns].fillna(0).values
        
        # === 步骤1: 聚类状态过滤 ===
        print(f"   步骤1: 聚类状态过滤")
        signals = {}
        
        for i, cluster_info in enumerate(selected_clusters):
            k_value = cluster_info['k_value']
            cluster_id = cluster_info['cluster_id']
            
            print(f"      聚类 {i+1}: k={k_value}, cluster_id={cluster_id}")
            
            # 使用对应的聚类模型
            cluster_model = self.cluster_models[k_value]
            cluster_labels = cluster_model.predict(X_pca)
            
            # 生成信号：属于目标聚类时为1，否则为0
            signal = (cluster_labels == cluster_id).astype(int)
            signals[f'signal_k{k_value}_c{cluster_id}'] = signal
            
            signal_count = signal.sum()
            signal_ratio = signal_count / len(signal)
            print(f"         状态信号: {signal_count}/{len(signal)} ({signal_ratio:.2%})")
        
        # 综合聚类状态信号：任一聚类发出信号则为1
        state_signal = np.zeros(len(test_data), dtype=int)
        for signal_col in signals.keys():
            state_signal = np.maximum(state_signal, signals[signal_col])
        
        print(f"      ✅ 综合状态信号: {state_signal.sum()}/{len(state_signal)} ({state_signal.mean():.2%})")
        
        # === 步骤2: 使用训练阶段选择的最佳PC（避免前视偏差） ===
        print(f"   步骤2: 应用训练阶段选择的最佳PC")
        
        if self.best_pc is None or self.pc_direction is None or self.pc_threshold is None:
            print(f"      ⚠️ 警告: 未找到训练阶段的PC选择结果，跳过PC门槛过滤")
            combined_signal = state_signal
        else:
            # 使用训练阶段选择的最佳PC和方向
            best_col = self.best_pc
            orient = self.pc_direction
            thr = self.pc_threshold
            
            print(f"      最佳PC: {best_col} (训练阶段选择)")
            print(f"      方向: {'正向' if orient > 0 else '反向'} (统一为IC>0)")
            threshold_q = self.pc_threshold_quantile if self.pc_threshold_quantile is not None else 0.6
            print(f"      门槛值: {thr:.4f} (训练阶段q={threshold_q:.2f})")
            
            # 计算整个测试数据的PC强度
            strength = test_data[best_col].fillna(0).values * orient
            
            # === 步骤3: 应用强度门槛（使用训练阶段的门槛） ===
            print(f"   步骤3: 应用强度门槛")
            
            # 应用门槛：状态信号 & 强度超过门槛
            gated = (state_signal == 1) & (strength > thr)
            print(f"      门槛后信号: {gated.sum()}/{len(gated)} ({gated.mean():.2%})")
            
            # === 步骤4: 持有期（hold=3） ===
            print(f"   步骤4: 持有期 (hold=3)")
            hold_n = 3
            n = len(test_data)  # 数据长度
            final_signal = np.zeros_like(gated, dtype=int)
            i = 0
            while i < n:
                if gated[i]:
                    final_signal[i:i+hold_n] = 1
                    i += hold_n
                else:
                    i += 1
            
            combined_signal = final_signal
            print(f"      ✅ 最终信号: {combined_signal.sum()}/{len(combined_signal)} ({combined_signal.mean():.2%})")
            
            # 保存元信息
            signals['signal_strength_pc'] = best_col
            signals['signal_strength_ic'] = f"训练阶段选择"
            signals['signal_threshold_q'] = threshold_q
            signals['signal_threshold_value'] = thr
            signals['signal_hold_n'] = hold_n
        
        signals['signal_combined'] = combined_signal
        
        # 添加信号到测试数据
        result_data = test_data.copy()
        for signal_name, signal_values in signals.items():
            if isinstance(signal_values, np.ndarray):
                result_data[signal_name] = signal_values
            else:
                result_data[signal_name] = signal_values
        
        return result_data

    def calculate_strategy_performance(self, signal_data: pd.DataFrame,
                                       transaction_cost: float = 0.002,
                                       slippage: float = 0.001) -> Dict:
        """
        计算策略收益 vs 基准（持有）
        
        改进：
        1. 严格T+1执行（今天信号→明天仓位）
        2. 按回合计费（开+平为一个回合）
        3. 统一胜率、收益等统计口径
        
        Parameters:
        -----------
        signal_data : pd.DataFrame
            包含信号的数据
        transaction_cost : float, default=0.002
            交易成本（单边）
        slippage : float, default=0.001
            滑点（单边）
            
        Returns:
        --------
        dict
            策略性能指标
        """
        print(f"💰 计算策略性能 (T+1执行 + 回合计费)...")
        
        # 使用future_return_5d作为预测目标收益
        returns = signal_data['future_return_5d'].fillna(0).values
        signal = signal_data['signal_combined'].values
        
        # === 关键修复1: T+1执行 ===
        # 今天的信号决定明天的仓位，避免look-ahead bias
        signal_t_plus_1 = np.roll(signal, 1)
        signal_t_plus_1[0] = 0  # 第一天无信号
        
        print(f"   🔄 T+1执行对齐:")
        print(f"      原始信号: {signal.sum()}/{len(signal)} ({signal.mean():.2%})")
        print(f"      对齐信号: {signal_t_plus_1.sum()}/{len(signal_t_plus_1)} ({signal_t_plus_1.mean():.2%})")
        
        # === 基准策略：始终持有 ===
        benchmark_returns = returns
        benchmark_cumulative = np.cumprod(1 + benchmark_returns)
        
        # === 策略收益：使用T+1对齐的信号 ===
        strategy_returns = signal_t_plus_1 * returns
        strategy_cumulative = np.ones(len(strategy_returns))
        
        for i in range(1, len(strategy_returns)):
            if signal_t_plus_1[i] == 1:
                # 有信号时，买入持有
                strategy_cumulative[i] = strategy_cumulative[i-1] * (1 + returns[i])
            else:
                # 无信号时，空仓，累计收益保持不变
                strategy_cumulative[i] = strategy_cumulative[i-1]
        strategy_cumulative[0] = 1 + strategy_returns[0]
        
        # === 关键修复2: 按回合计费 ===
        # 换手统计：计算信号翻转次数
        signal_changes = np.abs(np.diff(signal, prepend=signal[0]))
        flips = signal_changes.sum()
        roundtrips = flips / 2.0  # 每两个翻转构成一次完整回合（开+平）
        turnover_rate = roundtrips / len(signal)
        
        # 交易成本：按回合计费（双边）
        per_roundtrip_cost = (transaction_cost + slippage) * 2
        total_transaction_cost = roundtrips * per_roundtrip_cost
        
        print(f"   💸 交易成本:")
        print(f"      回合数: {roundtrips:.1f}")
        print(f"      换手率: {turnover_rate:.2%}")
        print(f"      单回合成本: {per_roundtrip_cost:.4f}")
        print(f"      总交易成本: {total_transaction_cost:.4f}")
        
        # 计算性能指标
        gross_return = strategy_cumulative[-1] - 1
        net_return = gross_return - total_transaction_cost
        total_return_benchmark = benchmark_cumulative[-1] - 1
        total_return_strategy = net_return  # 使用扣除成本后的净收益
        excess_return = total_return_strategy - total_return_benchmark
        
        print(f"   📊 收益拆解:")
        print(f"      毛收益: {gross_return:+.4f}")
        print(f"      交易成本: {total_transaction_cost:.4f}")
        print(f"      净收益: {net_return:+.4f}")
        print(f"      成本侵蚀比例: {(total_transaction_cost/abs(gross_return)*100):.1f}%" if gross_return != 0 else "      成本侵蚀比例: N/A")
        
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
        
        # === 关键修复3: 胜率统计（使用T+1对齐的信号） ===
        signal_mask = signal_t_plus_1 == 1
        win_rate = (returns[signal_mask] > 0).mean() if signal_mask.sum() > 0 else 0
        
        performance = {
            # 总收益
            'total_return_benchmark': total_return_benchmark,
            'total_return_strategy': total_return_strategy,
            'excess_return': excess_return,
            'gross_return': gross_return,
            
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
            'roundtrips': roundtrips,
            'turnover_rate': turnover_rate,
            'transaction_cost': total_transaction_cost,
            
            # 时间序列
            'benchmark_cumulative': benchmark_cumulative,
            'strategy_cumulative': strategy_cumulative,
            'benchmark_drawdown': benchmark_drawdown,
            'strategy_drawdown': strategy_drawdown,
            'dates': signal_data.index
        }
        
        print(f"   ✅ 策略性能:")
        print(f"      基准总收益: {total_return_benchmark:.2%}")
        print(f"      策略净收益: {total_return_strategy:.2%}")
        print(f"      超额收益: {excess_return:.2%}")
        print(f"      基准夏普: {sharpe_benchmark:.3f}")
        print(f"      策略夏普: {sharpe_strategy:.3f}")
        print(f"      基准回撤: {max_drawdown_benchmark:.2%}")
        print(f"      策略回撤: {max_drawdown_strategy:.2%}")
        print(f"      信号胜率: {win_rate:.2%} (T+1对齐)")
        
        return performance

    def run_random_baseline(self, signal_data: pd.DataFrame, performance: Dict) -> Dict:
        """
        随机基准对比（100次随机信号）
        
        改进：
        1. T+1对齐随机信号，保持公平对比
        2. 匹配相同的信号比例和持有期
        
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
        
        print(f"   📊 匹配策略信号比例: {original_signal_ratio:.2%}")
        print(f"   🔄 使用T+1执行（与策略一致）")
        
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
            
            # T+1对齐（与策略保持一致）
            random_signal_t1 = np.roll(random_signal, 1)
            random_signal_t1[0] = 0
            
            # 计算随机策略收益
            random_strategy_returns = random_signal_t1 * returns
            random_cumulative_returns = []
            cumulative = 1.0
            
            for j in range(len(random_strategy_returns)):
                if random_signal_t1[j] == 1:
                    cumulative *= (1 + returns[j])
                random_cumulative_returns.append(cumulative)
            
            total_random_return = cumulative - 1
            
            # 计算随机策略统计
            random_volatility = np.std(random_strategy_returns) * np.sqrt(250)
            
            random_results.append({
                'total_return': total_random_return,
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
        report_lines.append(f"   策略毛收益: {performance['gross_return']:+.2%}")
        report_lines.append(f"   交易成本: {performance['transaction_cost']:.4f}")
        report_lines.append(f"   策略净收益: {performance['total_return_strategy']:+.2%}")
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
        report_lines.append(f"   信号数量: {performance['signal_count']} (T+1对齐)")
        report_lines.append(f"   信号比例: {performance['signal_ratio']:.2%}")
        report_lines.append(f"   信号胜率: {performance['win_rate']:.2%} (T+1对齐)")
        report_lines.append(f"   回合数: {performance['roundtrips']:.1f}")
        report_lines.append(f"   换手率: {performance['turnover_rate']:.2%}")
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