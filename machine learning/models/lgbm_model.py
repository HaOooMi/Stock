#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM回归模型
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .base_model import BaseModel

# 尝试导入LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️ LightGBM未安装，该模型将不可用")


class LightGBMModel(BaseModel):
    """
    LightGBM回归模型封装
    
    特点：
    - 梯度提升树模型，表达能力强
    - 训练速度快，内存占用小
    - 支持早停和验证集评估
    - 提供特征重要性
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化LightGBM模型
        
        Parameters:
        -----------
        params : dict, optional
            模型参数
        """
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM未安装，请运行: pip install lightgbm")
        
        super().__init__(name='LightGBM', params=params)
        
        # 默认参数
        default_params = {
            'objective': 'regression',
            'metric': 'mse',
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 8,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        default_params.update(self.params)
        self.params = default_params
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_valid: Optional[pd.DataFrame] = None,
            y_valid: Optional[pd.Series] = None) -> Dict:
        """
        训练LightGBM模型
        
        Parameters:
        -----------
        X : pd.DataFrame
            训练特征
        y : pd.Series
            训练目标
        X_valid : pd.DataFrame, optional
            验证特征（用于早停）
        y_valid : pd.Series, optional
            验证目标
            
        Returns:
        --------
        dict
            训练结果
        """
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM未安装")
        
        print(f"   💡 训练 {self.name} 模型...")
        print(f"      📊 迭代次数: {self.params['n_estimators']}, 学习率: {self.params['learning_rate']}")
        
        start_time = time.time()
        
        # 保存特征名称
        self.feature_names = list(X.columns)
        
        # 创建数据集
        train_data = lgb.Dataset(X, label=y)
        
        # 准备参数（移除n_estimators，因为在training中单独指定）
        train_params = self.params.copy()
        n_estimators = train_params.pop('n_estimators', 500)
        
        # 准备训练参数
        callbacks = []
        valid_sets = [train_data]
        valid_names = ['train']
        
        # 如果有验证集，使用早停
        if X_valid is not None and y_valid is not None:
            valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append('valid')
            
            # 早停
            callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))
            callbacks.append(lgb.log_evaluation(period=0))  # 不打印日志
        
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
        
        # 评估训练集
        y_pred_train = self.model.predict(X)
        train_mse = mean_squared_error(y, y_pred_train)
        train_mae = mean_absolute_error(y, y_pred_train)
        
        results = {
            'model_name': self.name,
            'training_time': training_time,
            'train_samples': len(X),
            'train_mse': train_mse,
            'train_mae': train_mae,
            'n_features': len(self.feature_names),
            'n_estimators': self.model.num_trees(),
            'best_iteration': self.model.best_iteration
        }
        
        # 评估验证集
        if X_valid is not None and y_valid is not None:
            y_pred_valid = self.model.predict(X_valid)
            valid_mse = mean_squared_error(y_valid, y_pred_valid)
            valid_mae = mean_absolute_error(y_valid, y_pred_valid)
            
            results['valid_samples'] = len(X_valid)
            results['valid_mse'] = valid_mse
            results['valid_mae'] = valid_mae
            
            print(f"      📊 验证集 MSE: {valid_mse:.6f}, MAE: {valid_mae:.6f}")
            print(f"      🎯 最佳迭代: {self.model.best_iteration}")
        
        print(f"      ⏱️  训练时间: {training_time:.2f}秒")
        print(f"      📊 训练集 MSE: {train_mse:.6f}, MAE: {train_mae:.6f}")
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测
        
        Parameters:
        -----------
        X : pd.DataFrame
            特征数据
            
        Returns:
        --------
        np.ndarray
            预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，无法预测")
        
        # 确保特征顺序一致
        if self.feature_names is not None:
            X = X[self.feature_names]
        
        return self.model.predict(X)
    
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


if __name__ == "__main__":
    """
    测试LightGBM模型
    """
    if not HAS_LIGHTGBM:
        print("❌ LightGBM未安装，跳过测试")
        print("   安装命令: pip install lightgbm")
    else:
        print("🧪 测试LightGBM模型")
        print("=" * 50)
        
        # 生成模拟数据
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        # 添加非线性关系
        y = pd.Series(X['feature_0']**2 + X['feature_1'] * X['feature_2'] + np.random.randn(n_samples) * 0.1)
        
        # 划分训练集和验证集
        split_idx = int(n_samples * 0.8)
        X_train, X_valid = X[:split_idx], X[split_idx:]
        y_train, y_valid = y[:split_idx], y[split_idx:]
        
        # 训练模型
        print("\n💡 训练LightGBM模型")
        model = LightGBMModel(params={
            'n_estimators': 100,  # 测试用较少的迭代
            'learning_rate': 0.1,
            'num_leaves': 15
        })
        results = model.fit(X_train, y_train, X_valid, y_valid)
        print(f"✅ 训练结果: {results}")
        
        # 测试预测
        print("\n🎯 测试预测")
        y_pred = model.predict(X_valid)
        print(f"预测形状: {y_pred.shape}")
        print(f"预测前5个值: {y_pred[:5]}")
        
        # 测试特征重要性
        print("\n📊 特征重要性 (gain)")
        importance_gain = model.get_feature_importance('gain')
        print(f"Top 5 特征:\n{importance_gain.head()}")
        
        print("\n📊 特征重要性 (split)")
        importance_split = model.get_feature_importance('split')
        print(f"Top 5 特征:\n{importance_split.head()}")
        
        print("\n✅ 所有测试通过")
