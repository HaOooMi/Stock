#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机森林回归模型
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .base_model import BaseModel


class RandomForestModel(BaseModel):
    """
    随机森林回归模型封装
    
    特点：
    - 非线性模型，表达能力强
    - 可以处理特征交互
    - 提供特征重要性
    - 训练和预测相对较慢
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化随机森林模型
        
        Parameters:
        -----------
        params : dict, optional
            模型参数，包括:
            - n_estimators: 树的数量
            - max_depth: 最大深度
            - min_samples_leaf: 叶节点最小样本数
            - max_features: 最大特征数
            - n_jobs: 并行数
            - random_state: 随机种子
        """
        super().__init__(name='RandomForest', params=params)
        
        # 默认参数
        default_params = {
            'n_estimators': 500,
            'max_depth': 10,
            'min_samples_leaf': 5,
            'min_samples_split': 10,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(self.params)
        self.params = default_params
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_valid: Optional[pd.DataFrame] = None,
            y_valid: Optional[pd.Series] = None) -> Dict:
        """
        训练随机森林模型
        
        Parameters:
        -----------
        X : pd.DataFrame
            训练特征
        y : pd.Series
            训练目标
        X_valid : pd.DataFrame, optional
            验证特征
        y_valid : pd.Series, optional
            验证目标
            
        Returns:
        --------
        dict
            训练结果
        """
        print(f"   🌲 训练 {self.name} 模型...")
        print(f"      📊 树数量: {self.params['n_estimators']}, 最大深度: {self.params['max_depth']}")
        
        start_time = time.time()
        
        # 保存特征名称
        self.feature_names = list(X.columns)
        
        # 创建模型
        self.model = RandomForestRegressor(
            n_estimators=self.params['n_estimators'],
            max_depth=self.params['max_depth'],
            min_samples_leaf=self.params['min_samples_leaf'],
            min_samples_split=self.params['min_samples_split'],
            max_features=self.params['max_features'],
            random_state=self.params['random_state'],
            n_jobs=self.params['n_jobs'],
            verbose=0
        )
        
        # 训练模型
        self.model.fit(X, y)
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
            'n_estimators': self.params['n_estimators']
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
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        获取特征重要性
        
        Returns:
        --------
        pd.Series
            特征重要性
        """
        if not self.is_fitted:
            return None
        
        importance = self.model.feature_importances_
        
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)


if __name__ == "__main__":
    """
    测试随机森林模型
    """
    print("🧪 测试随机森林模型")
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
    print("\n🌲 训练随机森林模型")
    model = RandomForestModel(params={
        'n_estimators': 100,  # 测试用较少的树
        'max_depth': 5,
        'n_jobs': 2
    })
    results = model.fit(X_train, y_train, X_valid, y_valid)
    print(f"✅ 训练结果: {results}")
    
    # 测试预测
    print("\n🎯 测试预测")
    y_pred = model.predict(X_valid)
    print(f"预测形状: {y_pred.shape}")
    print(f"预测前5个值: {y_pred[:5]}")
    
    # 测试特征重要性
    print("\n📊 特征重要性")
    importance = model.get_feature_importance()
    print(f"Top 5 特征:\n{importance.head()}")
    
    print("\n✅ 所有测试通过")
