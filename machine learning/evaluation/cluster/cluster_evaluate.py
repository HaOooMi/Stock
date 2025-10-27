#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聚类与状态收益评估模块

功能：
1. 基于PCA状态进行KMeans聚类分析
2. 评估不同聚类的未来收益表现
3. 生成训练集和测试集的聚类性能报告
4. 验证聚类效果的显著性


"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# evaluation/cluster/ -> evaluation/ -> machine learning/
ml_root = os.path.dirname(os.path.dirname(current_dir))
project_root = os.path.dirname(ml_root)
if ml_root not in sys.path:
    sys.path.insert(0, ml_root)


class ClusterEvaluator:
    """
    聚类收益评估器
    
    主要功能：
    1. 对PCA状态进行KMeans聚类
    2. 分析每个聚类的未来收益表现
    3. 生成聚类性能报告
    4. 验证聚类的预测能力
    """
    
    def __init__(self, reports_dir: str = "ML output/reports"):
        """
        初始化聚类评估器
        
        Parameters:
        -----------
        reports_dir : str
            报告保存目录
        """
        # 设置报告目录
        if os.path.isabs(reports_dir):
            self.reports_dir = reports_dir
        else:
            self.reports_dir = os.path.join(ml_root, reports_dir)
        
        # 确保目录存在
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # 聚类配置
        self.k_values = [4, 5, 6]
        self.random_state = 42
        
        # 存储聚类结果
        self.cluster_models = {}
        self.train_results = {}
        self.test_results = {}

    def load_pca_states_and_targets(self, states_train_path: str, states_test_path: str,
                                   targets_path: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        加载PCA状态和目标收益数据
        
        Parameters:
        -----------
        states_train_path : str
            训练集PCA状态文件路径
        states_test_path : str
            测试集PCA状态文件路径
        targets_path : str
            目标收益数据文件路径
            
        Returns:
        --------
        tuple
            (训练状态, 测试状态, 目标数据)
        """
        print("Loading PCA states and targets...")
        
        # 加载PCA状态
        states_train = np.load(states_train_path)
        states_test = np.load(states_test_path)
        
        # 加载目标收益数据
        targets_df = pd.read_csv(targets_path, index_col=0, parse_dates=True)
        
        print(f"Training states: {states_train.shape}")
        print(f"Test states: {states_test.shape}")
        print(f"Targets: {targets_df.shape}")
        
        return states_train, states_test, targets_df



    def perform_kmeans_clustering(self, states_train: np.ndarray, k: int) -> KMeans:
        """
        执行KMeans聚类
        
        Parameters:
        -----------
        states_train : np.ndarray
            训练集PCA状态
        k : int
            聚类数量
            
        Returns:
        --------
        KMeans
            训练好的KMeans模型
        """
        kmeans = KMeans(
            n_clusters=k,
            random_state=self.random_state,
            n_init=20,
            max_iter=500
        )
        
        kmeans.fit(states_train)
        
        print(f"K={k} clustering completed")
        
        return kmeans

    def evaluate_cluster_returns(self, states: np.ndarray, targets_df: pd.DataFrame,
                                kmeans: KMeans, phase: str = "train") -> pd.DataFrame:
        """
        评估聚类的收益表现
        
        Parameters:
        -----------
        states : np.ndarray
            PCA状态数据
        targets_df : pd.DataFrame
            目标收益数据
        kmeans : KMeans
            聚类模型
        phase : str
            数据阶段 ("train" 或 "test")
            
        Returns:
        --------
        pd.DataFrame
            聚类性能评估结果
        """
        # 获取聚类标签
        cluster_labels = kmeans.predict(states)
        
        # 确保索引对齐
        if phase == "train":
            # 训练集使用前70%的目标数据
            n_train = len(states)
            target_data = targets_df.iloc[:n_train]
        else:
            # 测试集使用后30%的目标数据
            n_train = len(states)
            target_data = targets_df.iloc[-n_train:]
        
        # 重置索引确保对齐
        if len(target_data) != len(cluster_labels):
            print(f"Warning: Length mismatch - targets: {len(target_data)}, clusters: {len(cluster_labels)}")
            min_len = min(len(target_data), len(cluster_labels))
            target_data = target_data.iloc[:min_len]
            cluster_labels = cluster_labels[:min_len]
        
        # 计算每个聚类的未来收益统计
        results = []
        
        for cluster_id in range(kmeans.n_clusters):
            mask = cluster_labels == cluster_id
            cluster_returns = target_data.loc[mask, 'future_return_5d']
            cluster_states = states[mask]
            
            if len(cluster_returns) > 0 and len(cluster_states) > 0:
                # 基本收益统计
                stats = {
                    'cluster_id': cluster_id,
                    'count': len(cluster_returns),
                    'mean_return': cluster_returns.mean(),
                    'std_return': cluster_returns.std(),
                    'median_return': cluster_returns.median(),
                    'min_return': cluster_returns.min(),
                    'max_return': cluster_returns.max(),
                    'positive_ratio': (cluster_returns > 0).mean(),
                    'phase': phase,
                    'k_value': kmeans.n_clusters
                }
                
                # 添加特征统计信息
                for pc_idx in range(cluster_states.shape[1]):
                    pc_values = cluster_states[:, pc_idx]
                    stats[f'PC{pc_idx+1}_mean'] = pc_values.mean()
                    stats[f'PC{pc_idx+1}_std'] = pc_values.std()
                    stats[f'PC{pc_idx+1}_min'] = pc_values.min()
                    stats[f'PC{pc_idx+1}_max'] = pc_values.max()
                
                results.append(stats)
        
        results_df = pd.DataFrame(results)
        
        # 按平均收益排序
        results_df = results_df.sort_values('mean_return', ascending=False)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        return results_df

    def validate_cluster_performance(self, train_results: pd.DataFrame, 
                                   test_results: pd.DataFrame,
                                   global_std: float) -> Dict:
        """
        验证聚类性能
        
        Parameters:
        -----------
        train_results : pd.DataFrame
            训练集聚类结果
        test_results : pd.DataFrame
            测试集聚类结果
        global_std : float
            全局收益标准差
            
        Returns:
        --------
        dict
            验证结果
        """
        validation = {}
        
        # 训练集验证：最佳vs最差聚类差异 > 全局std * 0.4
        best_train = train_results.iloc[0]['mean_return']
        worst_train = train_results.iloc[-1]['mean_return']
        train_diff = best_train - worst_train
        threshold = global_std * 0.4
        
        validation['train_best_return'] = best_train
        validation['train_worst_return'] = worst_train
        validation['train_difference'] = train_diff
        validation['threshold'] = threshold
        validation['train_significant'] = train_diff > threshold
        
        # 测试集验证：最佳聚类仍在前50%
        best_train_cluster = train_results.iloc[0]['cluster_id']
        test_cluster_match = test_results[test_results['cluster_id'] == best_train_cluster]
        
        if len(test_cluster_match) > 0:
            test_best_rank = test_cluster_match['rank'].iloc[0]
            total_clusters = len(test_results)
            validation['test_best_cluster_rank'] = test_best_rank
            validation['total_clusters'] = total_clusters
            validation['test_top_50_percent'] = test_best_rank <= (total_clusters * 0.5)
        else:
            # 训练集最佳簇在测试集中不存在
            print(f"   ⚠️ 警告: 训练集最佳簇 {best_train_cluster} 在测试集中无样本")
            validation['test_best_cluster_rank'] = None
            validation['total_clusters'] = len(test_results)
            validation['test_top_50_percent'] = False
        
        return validation
    
    
    def generate_comprehensive_report(self, all_train_results: List, all_test_results: List, 
                                    validations: Dict, global_std: float):
        """
        生成综合聚类分析报告 - 合并所有报告生成功能
        
        Parameters:
        -----------
        all_train_results : List
            所有k值的训练结果
        all_test_results : List
            所有k值的测试结果
        validations : dict
            所有k值的验证结果
        global_std : float
            全局标准差
        """
        print("\n📊 Generating comprehensive clustering reports...")
        
        # === 1. 总结报告数据 ===
        summary = {
            'global_std': global_std,
            'threshold': global_std * 0.4,
            'total_k_values': len(self.k_values),
            'passed_train_validation': 0,
            'passed_test_validation': 0,
            'passed_both_validation': 0,
            'best_k': None,
            'best_performance': None
        }
        
        best_score = -1
        
        for k, validation in validations.items():
            # 统计通过验收的k值
            if validation['train_significant']:
                summary['passed_train_validation'] += 1
            
            if validation['test_top_50_percent']:
                summary['passed_test_validation'] += 1
            
            if validation['train_significant'] and validation['test_top_50_percent']:
                summary['passed_both_validation'] += 1
                
                # 计算综合分数 (训练差异 + 测试排名权重)
                score = validation['train_difference'] - (validation['test_best_cluster_rank'] - 1) * 0.01
                if score > best_score:
                    best_score = score
                    summary['best_k'] = k
                    summary['best_performance'] = validation
        
        # 计算成功率
        summary['train_success_rate'] = summary['passed_train_validation'] / summary['total_k_values']
        summary['test_success_rate'] = summary['passed_test_validation'] / summary['total_k_values']
        summary['overall_success_rate'] = summary['passed_both_validation'] / summary['total_k_values']
        
        # === 2. 生成主报告文件 ===
        main_report_lines = []
        main_report_lines.append("K-Means聚类分析综合报告")
        main_report_lines.append("=" * 60)
        main_report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        main_report_lines.append(f"全局收益标准差: {global_std:.6f}")
        main_report_lines.append(f"训练显著性阈值: {global_std * 0.4:.6f}")
        main_report_lines.append("")
        
        # 验证统计
        main_report_lines.append("� 验证统计:")
        main_report_lines.append(f"  测试k值总数: {summary['total_k_values']}")
        main_report_lines.append(f"  训练显著性通过: {summary['passed_train_validation']}/{summary['total_k_values']} ({summary['train_success_rate']:.1%})")
        main_report_lines.append(f"  测试前50%通过: {summary['passed_test_validation']}/{summary['total_k_values']} ({summary['test_success_rate']:.1%})")
        main_report_lines.append(f"  双重验证通过: {summary['passed_both_validation']}/{summary['total_k_values']} ({summary['overall_success_rate']:.1%})")
        
        if summary['best_k']:
            main_report_lines.append(f"\n🏆 最佳k值: {summary['best_k']}")
            best_perf = summary['best_performance']
            main_report_lines.append(f"  训练差异: {best_perf['train_difference']:+.6f}")
            main_report_lines.append(f"  测试最佳排名: {best_perf['test_best_cluster_rank']}")
        
        main_report_lines.append("\n" + "=" * 60)
        
        # === 3. 为每个k值生成详细报告 ===
        all_summary_data = []
        
        for k in self.k_values:
            validation = validations[k]
            
            # 获取该k值的结果
            train_k = pd.concat([df for df in all_train_results if df['k_value'].iloc[0] == k], ignore_index=True)
            test_k = pd.concat([df for df in all_test_results if df['k_value'].iloc[0] == k], ignore_index=True)
            kmeans = self.cluster_models[k]
            
            # 添加到主报告
            main_report_lines.append(f"\n🔍 K={k} 详细分析 {'✅' if validation['train_significant'] and validation['test_top_50_percent'] else '❌'}")
            main_report_lines.append("-" * 40)
            main_report_lines.append(f"验证: 训练显著性={validation['train_significant']}, 测试前50%={validation['test_top_50_percent']}")
            
            # 只在有质量指标时显示
            if 'silhouette_score' in validation and 'calinski_score' in validation:
                main_report_lines.append(f"质量分数: Silhouette={validation['silhouette_score']:.4f}, Calinski-Harabasz={validation['calinski_score']:.2f}")
            else:
                main_report_lines.append("质量分数: 未计算")
            
            # 详细聚类信息
            detailed_csv_data = []
            
            for cluster_id in range(k):
                train_row = train_k[train_k['cluster_id'] == cluster_id]
                test_row = test_k[test_k['cluster_id'] == cluster_id]
                
                if len(train_row) > 0 and len(test_row) > 0:
                    train_data = train_row.iloc[0]
                    test_data = test_row.iloc[0]
                    cluster_center = kmeans.cluster_centers_[cluster_id]
                    
                    # 添加到主报告
                    main_report_lines.append(f"  聚类{cluster_id}: 训练收益={train_data['mean_return']:+.6f}(排名{train_data['rank']}), 测试收益={test_data['mean_return']:+.6f}(排名{test_data['rank']})")
                    
                    # 准备CSV数据
                    row_data = {
                        'k_value': k,
                        'cluster_id': cluster_id,
                        'train_mean_return': train_data['mean_return'],
                        'test_mean_return': test_data['mean_return'],
                        'train_rank': train_data['rank'],
                        'test_rank': test_data['rank'],
                        'train_samples': train_data['count'],
                        'test_samples': test_data['count'],
                        'train_positive_ratio': train_data['positive_ratio'],
                        'test_positive_ratio': test_data['positive_ratio']
                    }
                    
                    # 添加聚类中心特征
                    for i, center_val in enumerate(cluster_center):
                        row_data[f'center_PC{i+1}'] = center_val
                    
                    # 添加训练集特征统计
                    feature_cols = [col for col in train_data.index if col.startswith('PC')]
                    for col in feature_cols:
                        row_data[f'train_{col}'] = train_data[col]
                    
                    detailed_csv_data.append(row_data)
                elif len(train_row) > 0:
                    # 只在训练集中存在
                    train_data = train_row.iloc[0]
                    main_report_lines.append(f"  聚类{cluster_id}: 训练收益={train_data['mean_return']:+.6f}(排名{train_data['rank']}), 测试集无样本")
                elif len(test_row) > 0:
                    # 只在测试集中存在
                    test_data = test_row.iloc[0]
                    main_report_lines.append(f"  聚类{cluster_id}: 训练集无样本, 测试收益={test_data['mean_return']:+.6f}(排名{test_data['rank']})")
                else:
                    # 训练集和测试集都没有样本（理论上不应该发生）
                    main_report_lines.append(f"  聚类{cluster_id}: 训练集和测试集都无样本")
            
            # 保存单独的k值详细CSV
            if detailed_csv_data:
                detailed_csv = pd.DataFrame(detailed_csv_data)
                csv_file = os.path.join(self.reports_dir, f"cluster_features_k{k}.csv")
                detailed_csv.to_csv(csv_file, index=False)
                
                # 添加到汇总数据
                all_summary_data.extend(detailed_csv_data)
        
        # === 4. 保存主报告文件 ===
        main_report_file = os.path.join(self.reports_dir, "clustering_analysis_report.txt")
        with open(main_report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(main_report_lines))
        
        # === 5. 生成聚类比较表 ===
        comparison_data = []
        for k in self.k_values:
            train_k = pd.concat([df for df in all_train_results if df['k_value'].iloc[0] == k], ignore_index=True)
            test_k = pd.concat([df for df in all_test_results if df['k_value'].iloc[0] == k], ignore_index=True)
            validation = validations[k]
            
            for cluster_id in range(k):
                train_row = train_k[train_k['cluster_id'] == cluster_id]
                test_row = test_k[test_k['cluster_id'] == cluster_id]
                
                if len(train_row) > 0 and len(test_row) > 0:
                    train_data = train_row.iloc[0]
                    test_data = test_row.iloc[0]
                    
                    comparison_data.append({
                        'k_value': k,
                        'cluster_id': cluster_id,
                        'train_samples': train_data['count'],
                        'test_samples': test_data['count'],
                        'train_mean_return': train_data['mean_return'],
                        'test_mean_return': test_data['mean_return'],
                        'train_rank': train_data['rank'],
                        'test_rank': test_data['rank'],
                        'overall_return': train_data['mean_return'] + test_data['mean_return'],
                        'validation_passed': validation['train_significant'] and validation['test_top_50_percent'],
                        'is_best_in_k': train_data['rank'] == 1,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                elif len(train_row) > 0:
                    # 只在训练集中存在，无法进行样本外验证
                    train_data = train_row.iloc[0]
                    comparison_data.append({
                        'k_value': k,
                        'cluster_id': cluster_id,
                        'train_samples': train_data['count'],
                        'test_samples': 0,
                        'train_mean_return': train_data['mean_return'],
                        'test_mean_return': np.nan,
                        'train_rank': train_data['rank'],
                        'test_rank': np.nan,
                        'overall_return': train_data['mean_return'],
                        'validation_passed': False,  # 无测试集样本视为未通过验证
                        'is_best_in_k': train_data['rank'] == 1,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                # 如果只在测试集中存在或都不存在，则跳过（这种情况不应该发生在正常聚类中）
        
        # 保存聚类比较表
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values(['validation_passed', 'overall_return'], ascending=[False, False])
            comparison_df['global_rank'] = range(1, len(comparison_df) + 1)
            comparison_csv_file = os.path.join(self.reports_dir, "cluster_comparison.csv")
            comparison_df.to_csv(comparison_csv_file, index=False)
        
        # === 6. 保存汇总CSV文件 ===
        if all_summary_data:
            all_summary_csv = pd.DataFrame(all_summary_data)
            summary_csv_file = os.path.join(self.reports_dir, "clustering_summary_all_k.csv")
            all_summary_csv.to_csv(summary_csv_file, index=False)
        
        # === 7. 保存验证结果CSV ===
        validation_data = []
        for k, validation in validations.items():
            validation_data.append({
                'k_value': k,
                'silhouette_score': validation['silhouette_score'],
                'calinski_score': validation['calinski_score'],
                'train_difference': validation['train_difference'],
                'train_significant': validation['train_significant'],
                'test_best_cluster_rank': validation['test_best_cluster_rank'],
                'test_top_50_percent': validation['test_top_50_percent'],
                'overall_valid': validation['train_significant'] and validation['test_top_50_percent']
            })
        
        validation_csv = pd.DataFrame(validation_data)
        validation_csv_file = os.path.join(self.reports_dir, "clustering_validation_results.csv")
        validation_csv.to_csv(validation_csv_file, index=False)
        
        # === 8. 输出生成的文件列表 ===
        print(f"📄 主报告: {main_report_file}")
        print(f"🏆 聚类比较: {comparison_csv_file}")
        print(f"📊 汇总数据: {summary_csv_file}")
        print(f"✅ 验证结果: {validation_csv_file}")
        print(f"📁 详细特征: {len(self.k_values)}个k值的单独CSV文件")
        print(f"\n报告生成完成！所有文件保存在: {self.reports_dir}")
        
        return summary

    def run_clustering_analysis(self, states_train_path: str, states_test_path: str,
                              targets_path: str) -> Dict:
        """
        运行完整的聚类分析
        
        Parameters:
        -----------
        states_train_path : str
            训练集PCA状态文件路径
        states_test_path : str
            测试集PCA状态文件路径
        targets_path : str
            目标收益数据文件路径
            
        Returns:
        --------
        dict
            分析结果摘要
        """
        print("Starting clustering analysis...")
        
        # 加载数据
        states_train, states_test, targets_df = self.load_pca_states_and_targets(
            states_train_path, states_test_path, targets_path
        )
        
        # 计算全局收益标准差
        global_std = targets_df['future_return_5d'].std()
        print(f"Global return std: {global_std:.4f}")
        
        all_train_results = []
        all_test_results = []
        all_validations = {}
        
        # 对每个k值进行聚类分析
        for k in self.k_values:
            print(f"\n--- K-Means with k={k} ---")
            
            # 训练KMeans模型
            kmeans = self.perform_kmeans_clustering(states_train, k)
            self.cluster_models[k] = kmeans
            
            # 评估训练集性能
            train_results = self.evaluate_cluster_returns(
                states_train, targets_df, kmeans, phase="train"
            )
            train_results['timestamp'] = datetime.now()
            all_train_results.append(train_results)
            
            # 评估测试集性能
            test_results = self.evaluate_cluster_returns(
                states_test, targets_df, kmeans, phase="test"
            )
            test_results['timestamp'] = datetime.now()
            all_test_results.append(test_results)
            
            # 验证聚类性能
            validation = self.validate_cluster_performance(
                train_results, test_results, global_std
            )
            validation['k_value'] = k
            
            # 添加聚类质量指标
            silhouette_avg = silhouette_score(states_train, kmeans.labels_)
            calinski_score_val = calinski_harabasz_score(states_train, kmeans.labels_)
            validation['silhouette_score'] = silhouette_avg
            validation['calinski_score'] = calinski_score_val
            
            all_validations[k] = validation
            
            print(f"Training: Best={train_results.iloc[0]['mean_return']:.4f}, "
                  f"Worst={train_results.iloc[-1]['mean_return']:.4f}")
            print(f"Validation: Train significant={validation['train_significant']}, "
                  f"Test top 50%={validation['test_top_50_percent']}")
        
        # 合并所有结果
        train_combined = pd.concat(all_train_results, ignore_index=True)
        test_combined = pd.concat(all_test_results, ignore_index=True)
        
        # === 计算并保存最佳PC信息（基于训练集历史数据） ===
        print("\n📊 计算最佳PC（基于训练集）...")
        from scipy import stats

        pc_metadata = {
            'best_pc': None,
            'best_pc_index': None,
            'pc_direction': 1.0,
            'pc_threshold': 0.0,
            'threshold_quantile': 0.6,
            'ic_value': 0.0,
            'all_ic_values': [],
            'calculated_time': datetime.now().isoformat()
        }

        if 'future_return_5d' in targets_df.columns and len(states_train) > 10:
            ret_values = targets_df.iloc[:len(states_train)]['future_return_5d'].fillna(0).values
            ic_list = []

            for idx in range(states_train.shape[1]):
                pc_values = states_train[:, idx]
                pc_t1 = np.roll(pc_values, 1)
                pc_t1[0] = 0  # 第一个位置无T+1

                ic, _ = stats.spearmanr(pc_t1, ret_values)
                ic_list.append(0.0 if np.isnan(ic) else float(ic))

            pc_metadata['all_ic_values'] = ic_list

            if ic_list:
                abs_ic = np.abs(ic_list)
                best_pc_idx = int(np.argmax(abs_ic))
                best_ic = ic_list[best_pc_idx]
                direction = 1.0 if best_ic >= 0 else -1.0

                strength = states_train[:, best_pc_idx] * direction
                threshold = float(np.quantile(strength, pc_metadata['threshold_quantile']))

                pc_metadata.update({
                    'best_pc': f'PC{best_pc_idx + 1}',
                    'best_pc_index': best_pc_idx,
                    'pc_direction': direction,
                    'pc_threshold': threshold,
                    'ic_value': best_ic
                })

                print(f"   ✅ 最佳PC: {pc_metadata['best_pc']} (IC={best_ic:.4f}, 门槛={threshold:.4f})")
            else:
                print("   ⚠️ 未能计算PC的IC，使用默认PC1")
        else:
            print("   ⚠️ 训练数据缺少future_return_5d或样本不足，使用默认PC1")

        # 若未成功确定最佳PC，使用默认配置
        if pc_metadata['best_pc'] is None:
            pc_metadata.update({
                'best_pc': 'PC1',
                'best_pc_index': 0,
                'pc_direction': 1.0,
                'pc_threshold': 0.0,
                'ic_value': 0.0
            })

        # 保存PC元数据
        pc_metadata_file = os.path.join(self.reports_dir, "pc_metadata.pkl")
        with open(pc_metadata_file, 'wb') as f:
            pickle.dump(pc_metadata, f)
        print(f"   💾 PC元数据已保存: {pc_metadata_file}")
        
        # 保存聚类模型
        models_file = os.path.join(self.reports_dir, "cluster_models.pkl")
        with open(models_file, 'wb') as f:
            pickle.dump(self.cluster_models, f)
        print(f"💾 聚类模型已保存: {models_file}")
        
        # 生成综合报告（合并所有报告生成功能）
        summary = self.generate_comprehensive_report(all_train_results, all_test_results, 
                                                   all_validations, global_std)
        
        return {
            'train_results': train_combined,
            'test_results': test_combined,
            'validations': all_validations,
            'summary': summary
        }


    def print_analysis_summary(self, results: Dict):
        """
        打印分析摘要
        
        Parameters:
        -----------
        results : dict
            分析结果
        """
        summary = results['summary']
        
        print("\n" + "="*60)
        print("CLUSTERING ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Global return std: {summary['global_std']:.4f}")
        print(f"Significance threshold: {summary['threshold']:.4f}")
        print(f"K values tested: {self.k_values}")
        
        print(f"\nValidation Results:")
        print(f"  Training significance: {summary['passed_train_validation']}/{summary['total_k_values']} "
              f"({summary['train_success_rate']:.1%})")
        print(f"  Test top 50%: {summary['passed_test_validation']}/{summary['total_k_values']} "
              f"({summary['test_success_rate']:.1%})")
        print(f"  Both criteria: {summary['passed_both_validation']}/{summary['total_k_values']} "
              f"({summary['overall_success_rate']:.1%})")
        
        if summary['best_k']:
            best = summary['best_performance']
            print(f"\nBest performing k={summary['best_k']}:")
            print(f"  Training difference: {best['train_difference']:.4f} > {best['threshold']:.4f} ✓")
            print(f"  Test rank: {best['test_best_cluster_rank']}/{best['total_clusters']} "
                  f"(top {best['test_best_cluster_rank']/best['total_clusters']:.1%}) ✓")
        else:
            print(f"\nNo k value passed both validation criteria")
        
        print("="*60)



def find_latest_files(evaluator, states_dir: str = None, targets_dir: str = None):
    """
    查找最新的PCA状态和目标文件
    
    Parameters:
    -----------
    evaluator : ClusterEvaluator
        评估器对象
    states_dir : str, optional
        状态文件目录，如果为None则使用默认路径
    targets_dir : str, optional
        目标文件目录，如果为None则使用默认路径
    """
    if states_dir is None:
        states_dir = os.path.join(ml_root, "ML output/states/baseline_v1")
    if targets_dir is None:
        targets_dir = os.path.join(ml_root, "ML output/datasets/baseline_v1")
    
    if not (os.path.exists(states_dir) and os.path.exists(targets_dir)):
        return None, None, None
    
    # 查找状态文件
    state_files = [f for f in os.listdir(states_dir) if f.startswith('states_pca_train_') and f.endswith('.npy')]
    target_files = [f for f in os.listdir(targets_dir) if f.startswith('with_targets_') and f.endswith('.csv')]
    
    if not (state_files and target_files):
        return None, None, None
    
    # 获取最新文件
    latest_state_train = max(state_files, key=lambda x: os.path.getctime(os.path.join(states_dir, x)))
    latest_state_test = latest_state_train.replace('states_pca_train_', 'states_pca_test_')
    latest_target = max(target_files, key=lambda x: os.path.getctime(os.path.join(targets_dir, x)))
    
    train_path = os.path.join(states_dir, latest_state_train)
    test_path = os.path.join(states_dir, latest_state_test)
    targets_path = os.path.join(targets_dir, latest_target)
    
    # 检查文件完整性
    if all(os.path.exists(p) for p in [train_path, test_path, targets_path]):
        return train_path, test_path, targets_path
    else:
        return None, None, None


def main(config: dict = None):
    """
    主函数：聚类+状态收益评估
    
    Parameters:
    -----------
    config : dict, optional
        配置字典，如果为None则使用默认配置
    """
    print("="*60)
    print("聚类 + 状态收益评估")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 如果没有提供配置，使用默认配置
    if config is None:
        config = {
            'paths': {
                'reports_clustering': 'ML output/reports',
                'states_dir': 'ML output/states'
            },
            'clustering': {
                'k_range': [4, 5, 6],
                'method': 'kmeans',
                'random_state': 42
            }
        }
    
    # 从配置中提取参数
    reports_dir = config.get('paths', {}).get('reports_clustering', 'ML output/reports/baseline_v1/clustering')
    states_dir = config.get('paths', {}).get('states_dir', 'ML output/states/baseline_v1')
    k_range = config.get('clustering', {}).get('k_range', [4, 5, 6])
    random_state = config.get('clustering', {}).get('random_state', 42)
    
    # 转换相对路径为绝对路径
    if not os.path.isabs(reports_dir):
        reports_dir = os.path.join(ml_root, reports_dir)
    if not os.path.isabs(states_dir):
        states_dir = os.path.join(ml_root, states_dir)
    
    print(f"\n📋 配置:")
    print(f"   reports_dir: {reports_dir}")
    print(f"   states_dir: {states_dir}")
    print(f"   k_range: {k_range}")
    
    # 初始化评估器（使用配置的目录）
    evaluator = ClusterEvaluator(reports_dir=reports_dir)
    evaluator.k_values = k_range
    evaluator.random_state = random_state
    
    # 查找实际数据文件（传入配置的目录）
    train_path, test_path, targets_path = find_latest_files(
        evaluator, 
        states_dir=states_dir,
        targets_dir=os.path.join(ml_root, "ML output/datasets/baseline_v1")
    )
    
    if train_path is None:
        print("\n❌ 未找到所需的数据文件！")
        print("请确保以下目录存在相应文件：")
        print(f"  - {states_dir} (states_pca_train_*.npy)")
        print(f"  - {os.path.join(ml_root, 'ML output/datasets/baseline_v1')} (with_targets_*.csv)")
        return
    else:
        print(f"\n✅ 找到数据文件:")
        print(f"  训练状态: {os.path.basename(train_path)}")
        print(f"  测试状态: {os.path.basename(test_path)}")
        print(f"  目标数据: {os.path.basename(targets_path)}")
    
    try:
        print(f"\n开始聚类分析...")
        print("-" * 40)
        
        # 运行聚类分析
        results = evaluator.run_clustering_analysis(
            train_path, test_path, targets_path
        )
        
        # 打印摘要
        evaluator.print_analysis_summary(results)
        
        # 验收结果检查
        print("\n" + "="*60)
        print("验收标准检查")
        print("="*60)
        
        summary = results['summary']
        
        print("1. KMeans聚类 (k=4,5,6) → future_return_5d排序: ✅")
        print("2. 生成训练集和测试集性能报告: ✅")
        
        if summary['passed_train_validation'] > 0:
            print(f"3. 训练集验收 (最佳vs最差 > 全局std*0.4): ✅ ({summary['passed_train_validation']}/{summary['total_k_values']})")
        else:
            print(f"3. 训练集验收 (最佳vs最差 > 全局std*0.4): ❌ (0/{summary['total_k_values']})")
        
        if summary['passed_test_validation'] > 0:
            print(f"4. 测试集验收 (最佳聚类保持前50%): ✅ ({summary['passed_test_validation']}/{summary['total_k_values']})")
        else:
            print(f"4. 测试集验收 (最佳聚类保持前50%): ❌ (0/{summary['total_k_values']})")
        
        overall_success = summary['overall_success_rate'] > 0
        print(f"\n总体评价: {'✅ 通过' if overall_success else '❌ 未通过'} ({summary['passed_both_validation']}/{summary['total_k_values']} 满足全部条件)")
        
        print("\n" + "="*60)
        print("阶段10完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 聚类分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()