#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM 排序模型 - LGBMRanker

功能：
1. 封装 LightGBM 排序模型（LambdaRank / Pairwise）
2. 支持 group 分组（按日期）
3. 与 BaseModel 接口兼容
4. 提供 NDCG 评估与早停

设计原则：
- 复用 BaseModel 的 save/load/get_feature_importance 接口
- 参数预设偏向稳定（防止排序模型过拟合）
- 训练数据必须按日期排序，group 与样本顺序一致

LambdaRank 核心：
- 优化目标：最大化 NDCG（头部排序质量）
- 损失函数：基于 pairwise 的梯度提升
- 适用场景：关注 Top-K 股票的排序准确性

创建: 2025-12-04 | 版本: v1.0
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from .base_model import BaseModel

# 尝试导入 LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️ LightGBM 未安装，排序模型将不可用")


class LightGBMRanker(BaseModel):
    """
    LightGBM 排序模型
    
    特点：
    - 使用 LambdaRank 目标函数，直接优化排序质量
    - 支持 group 分组，适合横截面股票排序
    - 参数预设偏向稳定，减少过拟合风险
    - 评估指标：NDCG@K
    
    使用要求：
    - 训练数据必须按日期（group）排序
    - 标签必须是离散整数（0, 1, 2, ... n_bins-1）
    - 必须提供 group 向量
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化 LightGBM 排序模型
        
        Parameters:
        -----------
        params : dict, optional
            模型参数，支持的关键参数：
            - objective: 排序目标（默认 'lambdarank'）
            - metric: 评估指标（默认 'ndcg'）
            - ndcg_eval_at: NDCG@K 的 K 值列表
            - label_gain: 各等级的增益权重
            - n_estimators: 迭代次数
            - learning_rate: 学习率
            - num_leaves: 叶子数
            - max_depth: 最大深度
            - feature_fraction: 特征采样比例
            - bagging_fraction: 样本采样比例
            - min_data_in_leaf: 叶子最小样本数
            - lambda_l1/l2: L1/L2 正则化
        """
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM 未安装，请运行: pip install lightgbm")
        
        super().__init__(name='LightGBMRanker', params=params)
        
        # 默认参数（偏向稳定）
        default_params = {
            # 排序目标
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [10, 30, 50],  # 关注头部
            'label_gain': None,  # 自动计算：[0, 1, 3, 7, 15, ...]
            
            # 树结构（保守设置）
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,  # 比回归更浅
            'min_data_in_leaf': 50,  # 比回归更大
            
            # 采样（强制正则化）
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            
            # 正则化
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            
            # 其他
            'random_state': 42,
            'verbose': -1,
            'force_row_wise': True  # 排序模型推荐
        }
        
        # 用户参数覆盖默认值
        default_params.update(self.params)
        self.params = default_params
        
        # 记录 group 信息
        self._train_groups = None
        self._valid_groups = None
    
    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            X_valid: Optional[pd.DataFrame] = None,
            y_valid: Optional[pd.Series] = None,
            groups: Optional[List[int]] = None,
            valid_groups: Optional[List[int]] = None,
            early_stopping_rounds: int = 50) -> Dict:
        """
        训练 LightGBM 排序模型
        
        Parameters:
        -----------
        X : pd.DataFrame
            训练特征（必须按日期排序）
        y : pd.Series
            训练标签（离散整数 0 ~ n_bins-1）
        X_valid : pd.DataFrame, optional
            验证特征（用于早停）
        y_valid : pd.Series, optional
            验证标签
        groups : List[int]
            训练集 group 向量，groups[i] = 第 i 个日期的样本数
        valid_groups : List[int], optional
            验证集 group 向量
        early_stopping_rounds : int
            早停轮数
            
        Returns:
        --------
        dict
            训练结果
        """
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM 未安装")
        
        # 校验 group
        if groups is None:
            raise ValueError("排序模型必须提供 groups 参数")
        
        if sum(groups) != len(X):
            raise ValueError(f"groups 总和 ({sum(groups)}) 与样本数 ({len(X)}) 不匹配")
        
        print(f"   🎯 训练 {self.name} 模型...")
        print(f"      📊 迭代次数: {self.params['n_estimators']}, 学习率: {self.params['learning_rate']}")
        print(f"      📊 日期数(groups): {len(groups)}, 样本数: {len(X)}")
        print(f"      📊 标签分布: {dict(y.value_counts().sort_index())}")
        
        start_time = time.time()
        
        # 保存特征名称
        self.feature_names = list(X.columns)
        self._train_groups = groups
        
        # 准备参数（移除非 LightGBM 原生参数）
        train_params = self.params.copy()
        n_estimators = train_params.pop('n_estimators', 500)
        
        # 自动计算 label_gain（如果未指定）
        if train_params.get('label_gain') is None:
            n_labels = int(y.max()) + 1
            # 指数增益：等级越高，增益越大
            train_params['label_gain'] = [2**i - 1 for i in range(n_labels)]
            print(f"      📊 自动 label_gain: {train_params['label_gain']}")
        
        # 创建训练数据集
        train_data = lgb.Dataset(
            X.values if isinstance(X, pd.DataFrame) else X,
            label=y.values if isinstance(y, pd.Series) else y,
            group=groups,
            feature_name=self.feature_names
        )
        
        # 准备验证集
        valid_sets = [train_data]
        valid_names = ['train']
        
        callbacks = []
        
        if X_valid is not None and y_valid is not None:
            if valid_groups is None:
                raise ValueError("提供验证集时必须同时提供 valid_groups")
            
            if sum(valid_groups) != len(X_valid):
                raise ValueError(f"valid_groups 总和 ({sum(valid_groups)}) 与验证样本数 ({len(X_valid)}) 不匹配")
            
            self._valid_groups = valid_groups
            
            valid_data = lgb.Dataset(
                X_valid.values if isinstance(X_valid, pd.DataFrame) else X_valid,
                label=y_valid.values if isinstance(y_valid, pd.Series) else y_valid,
                group=valid_groups,
                reference=train_data
            )
            valid_sets.append(valid_data)
            valid_names.append('valid')
            
            # 早停
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
            callbacks.append(lgb.log_evaluation(period=100))  # 每 100 轮打印
            
            print(f"      📊 验证集: {len(X_valid)} 样本, {len(valid_groups)} 日期")
        
        # 训练模型
        self.model = lgb.train(
            train_params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        self.is_fitted = True
        
        training_time = time.time() - start_time
        
        # 构建结果
        results = {
            'model_name': self.name,
            'training_time': training_time,
            'train_samples': len(X),
            'train_groups': len(groups),
            'n_features': len(self.feature_names),
            'n_estimators': self.model.num_trees(),
            'best_iteration': self.model.best_iteration
        }
        
        # 验证集结果
        if X_valid is not None and y_valid is not None:
            results['valid_samples'] = len(X_valid)
            results['valid_groups'] = len(valid_groups)
            
            # 获取最佳 NDCG
            if hasattr(self.model, 'best_score') and self.model.best_score:
                best_scores = self.model.best_score.get('valid', {})
                for metric_name, score in best_scores.items():
                    results[f'valid_{metric_name}'] = score
                    print(f"      📊 验证集 {metric_name}: {score:.6f}")
            
            print(f"      🎯 最佳迭代: {self.model.best_iteration}")
        
        print(f"      ⏱️  训练时间: {training_time:.2f}秒")
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测排序分数
        
        Parameters:
        -----------
        X : pd.DataFrame
            特征数据
            
        Returns:
        --------
        np.ndarray
            预测分数（越高表示预期排名越靠前）
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，无法预测")
        
        # 确保特征顺序一致
        if self.feature_names is not None:
            if isinstance(X, pd.DataFrame):
                X = X[self.feature_names]
        
        return self.model.predict(
            X.values if isinstance(X, pd.DataFrame) else X
        )
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Optional[pd.Series]:
        """
        获取特征重要性
        
        Parameters:
        -----------
        importance_type : str
            重要性类型: 'gain' 或 'split'
            
        Returns:
        --------
        pd.Series
            特征重要性
        """
        if not self.is_fitted:
            return None
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
    
    def get_group_info(self) -> Dict:
        """
        获取 group 信息
        
        Returns:
        --------
        dict
            包含 train_groups, valid_groups 等信息
        """
        return {
            'train_groups': self._train_groups,
            'valid_groups': self._valid_groups,
            'train_n_groups': len(self._train_groups) if self._train_groups else 0,
            'valid_n_groups': len(self._valid_groups) if self._valid_groups else 0
        }


# ==================== 便捷函数 ====================

def prepare_ranking_data(features: pd.DataFrame,
                         labels: pd.Series,
                         groups: Optional[List[int]] = None) -> tuple:
    """
    准备排序模型训练数据
    
    确保：
    1. 数据按日期排序
    2. 特征与标签对齐
    3. 如果未提供 groups，自动计算
    
    Parameters:
    -----------
    features : pd.DataFrame
        特征，MultiIndex [date, ticker]
    labels : pd.Series
        标签，MultiIndex [date, ticker]
    groups : List[int], optional
        group 向量
        
    Returns:
    --------
    tuple
        (X, y, groups)
    """
    # 对齐
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx].sort_index(level='date')
    y = labels.loc[common_idx].sort_index(level='date')
    
    # 计算 groups
    if groups is None:
        groups = X.groupby(level='date').size().tolist()
    
    return X, y, groups


# ==================== 测试 ====================

if __name__ == "__main__":
    print("🧪 测试 LightGBM 排序模型")
    print("=" * 60)
    
    if not HAS_LIGHTGBM:
        print("❌ LightGBM 未安装，跳过测试")
        print("   安装命令: pip install lightgbm")
    else:
        # 构造模拟数据
        np.random.seed(42)
        
        n_dates = 20
        n_stocks = 50
        n_features = 10
        n_bins = 5
        
        # 创建 MultiIndex
        dates = pd.date_range('2023-01-01', periods=n_dates, freq='D')
        tickers = [f'{i:06d}' for i in range(1, n_stocks + 1)]
        index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
        
        # 模拟特征
        X = pd.DataFrame(
            np.random.randn(len(index), n_features),
            columns=[f'feature_{i}' for i in range(n_features)],
            index=index
        )
        
        # 模拟标签（离散等级 0-4）
        y = pd.Series(
            np.random.randint(0, n_bins, len(index)),
            index=index,
            name='label'
        )
        
        # 按日期排序
        X = X.sort_index(level='date')
        y = y.sort_index(level='date')
        
        # 计算 groups
        groups = X.groupby(level='date').size().tolist()
        
        print(f"模拟数据: {len(index)} 样本, {n_dates} 日期, {n_stocks} 股票")
        print(f"Groups: {groups[:5]}... (共 {len(groups)} 个)")
        
        # 划分训练集和验证集（按日期）
        split_date = dates[int(n_dates * 0.7)]
        train_mask = X.index.get_level_values('date') < split_date
        valid_mask = ~train_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_valid, y_valid = X[valid_mask], y[valid_mask]
        
        train_groups = X_train.groupby(level='date').size().tolist()
        valid_groups = X_valid.groupby(level='date').size().tolist()
        
        print(f"\n训练集: {len(X_train)} 样本, {len(train_groups)} 日期")
        print(f"验证集: {len(X_valid)} 样本, {len(valid_groups)} 日期")
        
        # 训练模型
        print("\n💡 训练 LightGBM 排序模型")
        model = LightGBMRanker(params={
            'n_estimators': 100,
            'learning_rate': 0.1,
            'num_leaves': 15
        })
        
        results = model.fit(
            X_train, y_train,
            X_valid, y_valid,
            groups=train_groups,
            valid_groups=valid_groups
        )
        
        print(f"\n✅ 训练结果: {results}")
        
        # 测试预测
        print("\n🎯 测试预测")
        scores = model.predict(X_valid)
        print(f"预测形状: {scores.shape}")
        print(f"预测分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # 测试特征重要性
        print("\n📊 特征重要性 (gain)")
        importance = model.get_feature_importance('gain')
        print(f"Top 5:\n{importance.head()}")
        
        # Group 信息
        print("\n📊 Group 信息")
        print(model.get_group_info())
        
        print("\n✅ 所有测试通过")
