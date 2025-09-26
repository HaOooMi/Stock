#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA状态表示模块 - 降维和状态生成

功能：
1. 基于时间切分的PCA训练（防止数据泄漏）
2. 生成低维状态表示
3. 保存PCA模型和状态数据
4. 累计解释方差验证

作者: Assistant
日期: 2025年9月25日
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入特征工程和目标工程
from feature_engineering import FeatureEngineer
from target_engineering import TargetEngineer


class PCAStateGenerator:
    """
    PCA状态生成器 - 基于时间序列的降维表示
    
    主要功能：
    1. 时间切分防止数据泄漏的PCA训练
    2. 生成训练集和测试集的PCA状态
    3. 模型持久化和状态保存
    4. 解释方差验证
    """
    
    def __init__(self, models_dir: str = "machine learning/ML output/models",
                 states_dir: str = "machine learning/ML output/states"):
        """
        初始化PCA状态生成器
        
        Parameters:
        -----------
        models_dir : str
            模型保存目录
        states_dir : str
            状态数据保存目录
        """
        # 设置保存目录
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if os.path.isabs(models_dir):
            self.models_dir = models_dir
        else:
            self.models_dir = os.path.join(self.project_root, models_dir)
            
        if os.path.isabs(states_dir):
            self.states_dir = states_dir
        else:
            self.states_dir = os.path.join(self.project_root, states_dir)
        
        # 确保目录存在
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.states_dir, exist_ok=True)
        
        print("🎯 PCA状态生成器初始化完成")
        print(f"   📁 模型目录: {self.models_dir}")
        print(f"   📁 状态目录: {self.states_dir}")

    def load_scaled_features(self, csv_path: str) -> pd.DataFrame:
        """
        加载已标准化的特征数据
        
        Parameters:
        -----------
        csv_path : str
            标准化特征CSV文件路径
            
        Returns:
        --------
        pd.DataFrame
            标准化后的特征数据
        """
        print("📊 加载标准化特征数据...")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"标准化特征文件不存在: {csv_path}")
        
        # 读取CSV文件
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        # 移除目标列和标签列，只保留特征列
        feature_cols = [col for col in df.columns 
                       if not col.startswith(('future_return_', 'label_', 'close'))]
        
        features_df = df[feature_cols].copy()
        
        print(f"   ✅ 成功加载特征数据: {features_df.shape}")
        print(f"   📅 时间范围: {features_df.index.min().date()} ~ {features_df.index.max().date()}")
        print(f"   🔢 特征数量: {len(feature_cols)}")
        
        return features_df

    def fit_pca_with_time_split(self, features_df: pd.DataFrame,
                               n_components: float = 0.9,
                               train_ratio: float = 0.8) -> Dict:
        """
        基于时间切分拟合PCA（防止数据泄漏）
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            标准化后的特征数据
        n_components : float, default=0.9
            目标解释方差比例
        train_ratio : float, default=0.8
            训练集比例
            
        Returns:
        --------
        dict
            包含PCA模型和相关信息的字典
        """
        print("🔧 开始PCA训练...")
        print(f"   🎯 目标解释方差: {n_components:.1%}")
        print(f"   📊 时间切分比例: {train_ratio:.1%}")
        
        # 时间切分
        n_samples = len(features_df)
        split_idx = int(n_samples * train_ratio)
        
        if split_idx < 50:
            raise ValueError(f"训练样本过少({split_idx})，无法进行PCA训练")
        
        train_index = features_df.index[:split_idx]
        test_index = features_df.index[split_idx:]
        
        print(f"   📈 训练集: {split_idx} 样本 ({train_index.min().date()} ~ {train_index.max().date()})")
        print(f"   📉 测试集: {len(test_index)} 样本 ({test_index.min().date()} ~ {test_index.max().date()})")
        
        # 提取训练和测试特征
        X_train = features_df.iloc[:split_idx].fillna(0)  # 填充可能的缺失值
        X_test = features_df.iloc[split_idx:].fillna(0)
        
        original_features = X_train.shape[1]
        print(f"   🔢 原始特征维度: {original_features}")
        
        # 初始化PCA（先用较大的成分数量）
        pca_init = PCA(n_components=min(original_features, split_idx-1))
        pca_init.fit(X_train)
        
        # 计算累计解释方差
        cumsum_variance = np.cumsum(pca_init.explained_variance_ratio_)
        
        # 找到满足目标解释方差的成分数量
        n_components_needed = np.argmax(cumsum_variance >= n_components) + 1
        final_variance = cumsum_variance[n_components_needed - 1]
        
        # 验证成分数量范围（应为原始特征数的1/6到1/3）
        min_components = max(2, original_features // 6)
        max_components = original_features // 3
        
        if n_components_needed < min_components:
            print(f"   ⚠️ 成分数量过少({n_components_needed})，调整为最小值 {min_components}")
            n_components_needed = min_components
        elif n_components_needed > max_components:
            print(f"   ⚠️ 成分数量过多({n_components_needed})，调整为最大值 {max_components}")
            n_components_needed = max_components
        
        final_variance = cumsum_variance[n_components_needed - 1]
        
        print(f"   🎯 最终成分数量: {n_components_needed}")
        print(f"   📊 累计解释方差: {final_variance:.3f} ({final_variance:.1%})")
        print(f"   📉 特征压缩率: {original_features}/{n_components_needed} = {original_features/n_components_needed:.1f}x")
        
        # 训练最终PCA模型
        pca_final = PCA(n_components=n_components_needed)
        pca_final.fit(X_train)
        
        # 生成训练集和测试集的PCA状态
        states_train = pca_final.transform(X_train)
        states_test = pca_final.transform(X_test)
        
        print(f"   ✅ PCA训练完成")
        print(f"   📊 训练状态形状: {states_train.shape}")
        print(f"   📊 测试状态形状: {states_test.shape}")
        
        # 验收检查
        if final_variance >= 0.9:
            print(f"   ✅ 验收通过: 累计解释方差 {final_variance:.3f} ≥ 0.9")
        else:
            print(f"   ⚠️ 验收警告: 累计解释方差 {final_variance:.3f} < 0.9")
        
        component_ratio = n_components_needed / original_features
        if 1/6 <= component_ratio <= 1/3:
            print(f"   ✅ 验收通过: 成分比例 {component_ratio:.3f} 在合理范围内")
        else:
            print(f"   ⚠️ 验收警告: 成分比例 {component_ratio:.3f} 超出建议范围 [1/6, 1/3]")
        
        return {
            'pca_model': pca_final,
            'states_train': states_train,
            'states_test': states_test,
            'train_index': train_index,
            'test_index': test_index,
            'n_components': n_components_needed,
            'explained_variance_ratio': pca_final.explained_variance_ratio_,
            'cumulative_variance': final_variance,
            'original_features': original_features,
            'compression_ratio': original_features / n_components_needed,
            'feature_names': list(X_train.columns)
        }

    def save_pca_results(self, pca_results: Dict, symbol: str = "stock") -> Dict[str, str]:
        """
        保存PCA模型和状态数据
        
        Parameters:
        -----------
        pca_results : dict
            PCA训练结果
        symbol : str
            股票代码，用于文件命名
            
        Returns:
        --------
        dict
            保存的文件路径
        """
        print("💾 保存PCA结果...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存PCA模型
        pca_filename = f"pca_{symbol}_{timestamp}.pkl"
        pca_path = os.path.join(self.models_dir, pca_filename)
        
        pca_data = {
            'pca_model': pca_results['pca_model'],
            'n_components': pca_results['n_components'],
            'explained_variance_ratio': pca_results['explained_variance_ratio'],
            'cumulative_variance': pca_results['cumulative_variance'],
            'original_features': pca_results['original_features'],
            'compression_ratio': pca_results['compression_ratio'],
            'feature_names': pca_results['feature_names'],
            'train_samples': len(pca_results['states_train']),
            'test_samples': len(pca_results['states_test']),
            'created_time': datetime.now().isoformat()
        }
        
        with open(pca_path, 'wb') as f:
            pickle.dump(pca_data, f)
        
        pca_size = os.path.getsize(pca_path) / 1024  # KB
        print(f"   ✅ PCA模型已保存: {pca_filename} ({pca_size:.1f} KB)")
        
        # 保存训练状态
        states_train_filename = f"states_pca_train_{symbol}_{timestamp}.npy"
        states_train_path = os.path.join(self.states_dir, states_train_filename)
        
        np.save(states_train_path, pca_results['states_train'])
        train_size = os.path.getsize(states_train_path) / 1024  # KB
        print(f"   ✅ 训练状态已保存: {states_train_filename} ({train_size:.1f} KB)")
        
        # 保存测试状态
        states_test_filename = f"states_pca_test_{symbol}_{timestamp}.npy"
        states_test_path = os.path.join(self.states_dir, states_test_filename)
        
        np.save(states_test_path, pca_results['states_test'])
        test_size = os.path.getsize(states_test_path) / 1024  # KB
        print(f"   ✅ 测试状态已保存: {states_test_filename} ({test_size:.1f} KB)")
        
        # 保存元数据文件
        metadata = {
            'symbol': symbol,
            'created_time': datetime.now().isoformat(),
            'original_features': pca_results['original_features'],
            'n_components': pca_results['n_components'],
            'cumulative_variance': float(pca_results['cumulative_variance']),
            'compression_ratio': float(pca_results['compression_ratio']),
            'train_samples': len(pca_results['states_train']),
            'test_samples': len(pca_results['states_test']),
            'train_period': f"{pca_results['train_index'].min().date()} ~ {pca_results['train_index'].max().date()}",
            'test_period': f"{pca_results['test_index'].min().date()} ~ {pca_results['test_index'].max().date()}",
            'files': {
                'pca_model': pca_filename,
                'states_train': states_train_filename,
                'states_test': states_test_filename
            }
        }
        
        metadata_filename = f"pca_metadata_{symbol}_{timestamp}.json"
        metadata_path = os.path.join(self.models_dir, metadata_filename)
        
        import json
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"   📋 元数据已保存: {metadata_filename}")
        
        return {
            'pca_model_path': pca_path,
            'states_train_path': states_train_path,
            'states_test_path': states_test_path,
            'metadata_path': metadata_path
        }

    def generate_pca_states(self, csv_path: str, symbol: str = "stock",
                           n_components: float = 0.9, train_ratio: float = 0.8) -> Dict:
        """
        完整的PCA状态生成流程
        
        Parameters:
        -----------
        csv_path : str
            标准化特征CSV文件路径
        symbol : str
            股票代码
        n_components : float
            目标解释方差比例
        train_ratio : float
            训练集比例
            
        Returns:
        --------
        dict
            包含PCA结果和保存路径的完整信息
        """
        print("🚀 开始完整PCA状态生成流程...")
        print("=" * 60)
        
        # 1. 加载特征数据
        features_df = self.load_scaled_features(csv_path)
        
        # 2. 拟合PCA
        pca_results = self.fit_pca_with_time_split(
            features_df, 
            n_components=n_components,
            train_ratio=train_ratio
        )
        
        # 3. 保存结果
        save_paths = self.save_pca_results(pca_results, symbol)
        
        # 4. 最终报告
        print("\n" + "=" * 60)
        print("🎉 PCA状态生成完成!")
        print(f"   📊 原始特征: {pca_results['original_features']} 维")
        print(f"   🎯 PCA成分: {pca_results['n_components']} 维")
        print(f"   📈 解释方差: {pca_results['cumulative_variance']:.3f} ({pca_results['cumulative_variance']:.1%})")
        print(f"   📉 压缩比率: {pca_results['compression_ratio']:.1f}x")
        print(f"   🏋️ 训练样本: {len(pca_results['states_train'])}")
        print(f"   🧪 测试样本: {len(pca_results['states_test'])}")
        
        # 合并结果
        final_results = {**pca_results, **save_paths}
        
        return final_results

    def load_pca_model(self, pca_path: str) -> Dict:
        """
        加载已保存的PCA模型
        
        Parameters:
        -----------
        pca_path : str
            PCA模型文件路径
            
        Returns:
        --------
        dict
            PCA模型数据
        """
        print(f"📖 加载PCA模型: {os.path.basename(pca_path)}")
        
        with open(pca_path, 'rb') as f:
            pca_data = pickle.load(f)
        
        print(f"   ✅ 模型加载成功")
        print(f"   🎯 成分数量: {pca_data['n_components']}")
        print(f"   📊 解释方差: {pca_data['cumulative_variance']:.3f}")
        
        return pca_data


def run_complete_feature_pipeline(symbol: str = '000001',
                                  start_date: str = '2023-01-01', 
                                  end_date: str = '2024-12-31',
                                  use_auto_features: bool = False,
                                  final_k_features: int = 15) -> Dict:
    """
    运行完整的特征工程管道
    
    Parameters:
    -----------
    symbol : str
        股票代码
    start_date : str
        开始日期
    end_date : str
        结束日期
    use_auto_features : bool
        是否使用自动特征生成
    final_k_features : int
        最终保留的特征数量
        
    Returns:
    --------
    dict
        特征工程结果
    """
    print("🔧 步骤1: 完整特征工程流程")
    print("-" * 50)
    
    try:
        # 初始化特征工程器
        feature_engineer = FeatureEngineer(use_talib=True, use_tsfresh=use_auto_features)
        
        # 加载数据
        print(f"📈 加载股票数据: {symbol} ({start_date} ~ {end_date})")
        raw_data = feature_engineer.load_stock_data(symbol, start_date, end_date)
        
        if len(raw_data) < 100:
            raise ValueError(f"数据量太少({len(raw_data)}行)，建议至少100行数据")
        
        # 生成特征
        print("🏭 生成技术特征...")
        features_df = feature_engineer.prepare_features(
            raw_data,
            use_auto_features=use_auto_features,
            window_size=20,
            max_auto_features=30
        )
        
        # 特征选择
        print("🎯 执行特征选择...")
        selection_results = feature_engineer.select_features(
            features_df,
            final_k=final_k_features,
            variance_threshold=0.01,
            correlation_threshold=0.9,
            train_ratio=0.8
        )
        
        final_features_df = selection_results['final_features_df']
        
        # 特征标准化
        print("📏 执行特征标准化...")
        scale_results = feature_engineer.scale_features(
            final_features_df,
            scaler_type='robust',
            train_ratio=0.8,
            save_path=f'machine learning/ML output/scaler_{symbol}.pkl'
        )
        
        scaled_features_df = scale_results['scaled_df']
        print(f"   ✅ 缩放器已保存: {scale_results['scaler_path']}")
        if scale_results.get('csv_path'):
            print(f"   📊 标准化特征已保存: {scale_results['csv_path']}")
        
        # 特征分析
        print("📊 分析特征质量...")
        analysis_results = feature_engineer.analyze_features(scaled_features_df)
        
        return {
            'success': True,
            'scaled_features_df': scaled_features_df,
            'csv_path': scale_results.get('csv_path'),
            'scaler_path': scale_results['scaler_path'],
            'final_feature_count': len(selection_results['final_features']),
            'sample_count': len(scaled_features_df),
            'selection_results': selection_results,
            'scale_results': scale_results,
            'analysis_results': analysis_results
        }
        
    except Exception as e:
        print(f"❌ 特征工程失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def run_complete_target_pipeline(scaled_features_df: pd.DataFrame, 
                                symbol: str = 'stock',
                                target_periods: list = [1, 5, 10]) -> Dict:
    """
    运行完整的目标变量工程管道
    
    Parameters:
    -----------
    scaled_features_df : pd.DataFrame
        标准化后的特征数据
    symbol : str
        股票代码
    target_periods : list
        目标时间窗口
        
    Returns:
    --------
    dict
        目标工程结果
    """
    print("\n🎯 步骤2: 完整目标变量工程流程")
    print("-" * 50)
    
    try:
        # 初始化目标工程器
        target_engineer = TargetEngineer()
        
        # 创建完整数据集（特征 + 目标）
        print("🔨 创建完整数据集...")
        complete_dataset = target_engineer.create_complete_dataset(
            scaled_features_df,
            periods=target_periods,
            price_col='close',
            include_labels=True,
            label_types=['binary', 'quantile']
        )
        
        # 保存数据集
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"complete_{timestamp}"
        save_path = target_engineer.save_dataset(complete_dataset, symbol, suffix)
        
        # 计算目标统计
        target_cols = [col for col in complete_dataset.columns if col.startswith('future_return_')]
        max_period = max(target_periods) if target_periods else 0
        trainable_samples = len(complete_dataset) - max_period
        
        return {
            'success': True,
            'complete_dataset': complete_dataset,
            'save_path': save_path,
            'target_sample_count': len(complete_dataset),
            'trainable_samples': trainable_samples,
            'target_cols': target_cols,
            'max_period': max_period
        }
        
    except Exception as e:
        print(f"❌ 目标变量工程失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def generate_final_summary(feature_results: Dict, target_results: Dict, pca_results: Dict = None) -> bool:
    """
    生成最终流程摘要报告
    
    Parameters:
    -----------
    feature_results : dict
        特征工程结果
    target_results : dict
        目标工程结果
    pca_results : dict, optional
        PCA结果
        
    Returns:
    --------
    bool
        是否成功生成摘要
    """
    print("\n📋 生成最终摘要报告")
    print("-" * 50)
    
    try:
        # 创建摘要内容
        summary_lines = []
        summary_lines.append("=" * 70)
        summary_lines.append("股票机器学习完整流程摘要报告")
        summary_lines.append("=" * 70)
        summary_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")
        
        # 特征工程摘要
        summary_lines.append("🔧 特征工程结果:")
        if feature_results.get('success'):
            summary_lines.append(f"   ✅ 状态: 成功完成")
            summary_lines.append(f"   📊 最终特征数: {feature_results['final_feature_count']}")
            summary_lines.append(f"   🔢 样本数量: {feature_results['sample_count']}")
            summary_lines.append(f"   💾 缩放器: {os.path.basename(feature_results['scaler_path'])}")
            if feature_results.get('csv_path'):
                summary_lines.append(f"   📊 特征文件: {os.path.basename(feature_results['csv_path'])}")
        else:
            summary_lines.append(f"   ❌ 状态: 失败 - {feature_results.get('error', '未知错误')}")
        summary_lines.append("")
        
        # 目标工程摘要
        summary_lines.append("🎯 目标变量工程结果:")
        if target_results.get('success'):
            summary_lines.append(f"   ✅ 状态: 成功完成")
            summary_lines.append(f"   📊 目标样本数: {target_results['target_sample_count']}")
            summary_lines.append(f"   🎓 可训练样本: {target_results['trainable_samples']}")
            summary_lines.append(f"   🎯 目标变量数: {len(target_results['target_cols'])}")
            summary_lines.append(f"   💾 数据集文件: {os.path.basename(target_results['save_path'])}")
        else:
            summary_lines.append(f"   ❌ 状态: 失败 - {target_results.get('error', '未知错误')}")
        summary_lines.append("")
        
        # PCA摘要（如果存在）
        if pca_results:
            summary_lines.append("🔍 PCA状态生成结果:")
            if 'n_components' in pca_results:
                summary_lines.append(f"   ✅ 状态: 成功完成")
                summary_lines.append(f"   📊 原始特征数: {pca_results['original_features']}")
                summary_lines.append(f"   🎯 PCA成分数: {pca_results['n_components']}")
                summary_lines.append(f"   📈 解释方差: {pca_results['cumulative_variance']:.3f} ({pca_results['cumulative_variance']:.1%})")
                summary_lines.append(f"   📉 压缩比率: {pca_results['compression_ratio']:.1f}x")
                summary_lines.append(f"   🏋️ 训练样本: {len(pca_results['states_train'])}")
                summary_lines.append(f"   🧪 测试样本: {len(pca_results['states_test'])}")
                if 'pca_model_path' in pca_results:
                    summary_lines.append(f"   💾 PCA模型: {os.path.basename(pca_results['pca_model_path'])}")
            else:
                summary_lines.append(f"   ❌ 状态: 失败")
            summary_lines.append("")
        
        # 文件统计
        ml_output_dir = os.path.join("machine learning", "ML output")
        summary_lines.append("📁 生成文件统计:")
        
        try:
            file_count = 0
            total_size = 0
            
            # 扫描ML output目录及子目录
            for root, dirs, files in os.walk(ml_output_dir):
                for file in files:
                    if file.endswith(('.csv', '.pkl', '.npy', '.json', '.txt')):
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            size = os.path.getsize(file_path) / 1024  # KB
                            rel_path = os.path.relpath(file_path, ml_output_dir)
                            summary_lines.append(f"   📄 {rel_path} ({size:.1f} KB)")
                            file_count += 1
                            total_size += size
            
            summary_lines.append("")
            summary_lines.append(f"   📊 总文件数: {file_count}")
            summary_lines.append(f"   💾 总大小: {total_size:.1f} KB ({total_size/1024:.2f} MB)")
            
        except Exception as e:
            summary_lines.append(f"   ⚠️ 文件扫描失败: {str(e)}")
        
        summary_lines.append("")
        summary_lines.append("=" * 70)
        
        # 根据结果判断最终状态
        all_success = (feature_results.get('success', False) and 
                      target_results.get('success', False))
        
        if all_success:
            summary_lines.append("🎊 完整流程成功完成！")
            summary_lines.append("✨ 现在可以开始机器学习建模了")
        else:
            summary_lines.append("⚠️ 流程部分完成，请检查失败的步骤")
        
        summary_lines.append("=" * 70)
        
        # 显示摘要
        summary_text = "\n".join(summary_lines)
        print(summary_text)
        
        # 保存摘要到文件
        os.makedirs(ml_output_dir, exist_ok=True)
        summary_path = os.path.join(ml_output_dir, f'pipeline_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print(f"\n💾 摘要已保存: {os.path.basename(summary_path)}")
        return True
        
    except Exception as e:
        print(f"❌ 摘要生成失败: {str(e)}")
        return False

def main():
    """
    主函数 - 完整的股票机器学习预处理流程
    包含：特征工程 → 目标工程 → PCA状态生成
    """
    print("🚀 股票机器学习完整预处理流程")
    print("=" * 70)
    print("包含: 特征工程 → 目标工程 → PCA状态生成")
    print("=" * 70)
    
    start_time = datetime.now()
    
    try:
        # 配置参数
        config = {
            'symbol': '000001',  # 平安银行
            'start_date': '2023-01-01',
            'end_date': '2024-12-31',
            'use_auto_features': True,  # 是否使用自动特征生成
            'final_k_features': 15,      # 最终特征数量
            'target_periods': [1, 5, 10], # 目标时间窗口
            'pca_components': 0.9,       # PCA解释方差比例
            'train_ratio': 0.8           # 训练集比例
        }
        
        print("📋 执行配置:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        print()
        
        # 确保输出目录存在
        os.makedirs("machine learning/ML output/models", exist_ok=True)
        os.makedirs("machine learning/ML output/states", exist_ok=True)
        
        success_steps = 0
        total_steps = 3
        
        # === 步骤1: 特征工程 ===
        feature_results = run_complete_feature_pipeline(
            symbol=config['symbol'],
            start_date=config['start_date'],
            end_date=config['end_date'],
            use_auto_features=config['use_auto_features'],
            final_k_features=config['final_k_features']
        )
        
        if feature_results.get('success'):
            success_steps += 1
            print("✅ 特征工程完成")
        else:
            print("❌ 特征工程失败，流程终止")
            return False
        
        # === 步骤2: 目标变量工程 ===
        target_results = run_complete_target_pipeline(
            scaled_features_df=feature_results['scaled_features_df'],
            symbol=config['symbol'],
            target_periods=config['target_periods']
        )
        
        if target_results.get('success'):
            success_steps += 1
            print("✅ 目标变量工程完成")
        else:
            print("❌ 目标变量工程失败，流程终止")
            return False
        
        # === 步骤3: PCA状态生成 ===
        print(f"\n🔍 步骤3: PCA状态生成")
        print("-" * 50)
        
        pca_results = None
        try:
            # 检查是否有CSV文件可用
            csv_path = feature_results.get('csv_path')
            if not csv_path or not os.path.exists(csv_path):
                print("⚠️ 标准化特征CSV文件不存在，跳过PCA步骤")
            else:
                # 初始化PCA状态生成器
                pca_generator = PCAStateGenerator()
                
                # 生成PCA状态
                pca_results = pca_generator.generate_pca_states(
                    csv_path=csv_path,
                    symbol=config['symbol'],
                    n_components=config['pca_components'],
                    train_ratio=config['train_ratio']
                )
                
                if pca_results and 'n_components' in pca_results:
                    success_steps += 1
                    print("✅ PCA状态生成完成")
                else:
                    print("⚠️ PCA状态生成失败，但前续步骤已完成")
        
        except Exception as e:
            print(f"⚠️ PCA状态生成异常: {str(e)}")
            print("   前续步骤已完成，可以继续后续分析")
        
        # === 生成最终摘要 ===
        generate_final_summary(feature_results, target_results, pca_results)
        
        # 计算总耗时
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n⏱️ 流程总耗时: {duration.total_seconds():.1f} 秒")
        print(f"📊 成功步骤: {success_steps}/{total_steps}")
        
        if success_steps >= 2:  # 至少特征工程和目标工程成功
            print("\n🎊 核心流程成功完成！")
            print("📁 所有结果文件已保存到: machine learning/ML output/")
            print("✨ 现在可以开始机器学习建模了")
            return True
        else:
            print(f"\n⚠️ 流程未能成功完成")
            return False
        
    except Exception as e:
        print(f"\n💥 流程异常终止: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 执行PCA状态生成
    main()
