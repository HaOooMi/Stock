#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ridge回归模型
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .base_model import BaseModel


class RidgeModel(BaseModel):
    """
    Ridge回归模型封装
    
    特点：
    - 线性模型，训练快速
    - L2正则化，防止过拟合
    - 支持交叉验证选择alpha
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化Ridge模型
        
        Parameters:
        -----------
        params : dict, optional
            模型参数，包括:
            - alpha: 正则化强度（或alpha列表用于CV）
            - fit_intercept: 是否拟合截距
            - random_state: 随机种子
        """
        super().__init__(name='Ridge', params=params)
        
        # 默认参数
        default_params = {
            'alpha': 1.0,
            'fit_intercept': True,
            'random_state': 42
        }
        default_params.update(self.params)
        self.params = default_params
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_valid: Optional[pd.DataFrame] = None,
            y_valid: Optional[pd.Series] = None) -> Dict:
        """
        训练Ridge模型
        
        Parameters:
        -----------
        X : pd.DataFrame
            训练特征
        y : pd.Series
            训练目标
        X_valid : pd.DataFrame, optional
            验证特征（用于评估）
        y_valid : pd.Series, optional
            验证目标
            
        Returns:
        --------
        dict
            训练结果
        """
        print(f"   🔧 训练 {self.name} 模型...")
        
        start_time = time.time()
        
        # 保存特征名称
        self.feature_names = list(X.columns)
        
        # 判断是否使用交叉验证
        alphas = self.params.get('alpha')
        if isinstance(alphas, (list, tuple)) and len(alphas) > 1:
            # 使用RidgeCV进行交叉验证
            print(f"      🔍 使用交叉验证选择最优alpha: {alphas}")
            self.model = RidgeCV(
                alphas=alphas,
                fit_intercept=self.params.get('fit_intercept', True),
                cv=5  # 5折交叉验证
            )
        else:
            # 使用固定alpha
            alpha = alphas if isinstance(alphas, (int, float)) else alphas[0]
            print(f"      📌 使用固定alpha: {alpha}")
            self.model = Ridge(
                alpha=alpha,
                fit_intercept=self.params.get('fit_intercept', True),
                random_state=self.params.get('random_state', 42)
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
            'n_features': len(self.feature_names)
        }
        
        # 如果使用了RidgeCV，记录最优alpha
        if isinstance(self.model, RidgeCV):
            results['best_alpha'] = self.model.alpha_
            print(f"      ✅ 最优alpha: {self.model.alpha_:.4f}")
        
        # 评估验证集（如果提供）
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
        获取特征重要性（线性模型系数的绝对值）
        
        Returns:
        --------
        pd.Series
            特征重要性
        """
        if not self.is_fitted:
            return None
        
        # 使用系数的绝对值作为重要性
        importance = np.abs(self.model.coef_)
        
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)


if __name__ == "__main__":
    """
    测试Ridge模型
    """
    print("🧪 测试Ridge模型")
    print("=" * 50)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(X.sum(axis=1) + np.random.randn(n_samples) * 0.1)
    
    # 划分训练集和验证集
    split_idx = int(n_samples * 0.8)
    X_train, X_valid = X[:split_idx], X[split_idx:]
    y_train, y_valid = y[:split_idx], y[split_idx:]
    
    # 测试1: 固定alpha
    print("\n📌 测试1: 固定alpha")
    model1 = RidgeModel(params={'alpha': 1.0})
    results1 = model1.fit(X_train, y_train, X_valid, y_valid)
    print(f"✅ 训练结果: {results1}")
    
    # 测试2: 交叉验证选择alpha
    print("\n🔍 测试2: 交叉验证")
    model2 = RidgeModel(params={'alpha': [0.1, 1.0, 10.0, 100.0]})
    results2 = model2.fit(X_train, y_train, X_valid, y_valid)
    print(f"✅ 训练结果: {results2}")
    
    # 测试预测
    print("\n🎯 测试预测")
    y_pred = model2.predict(X_valid)
    print(f"预测形状: {y_pred.shape}")
    print(f"预测前5个值: {y_pred[:5]}")
    
    # 测试特征重要性
    print("\n📊 特征重要性")
    importance = model2.get_feature_importance()
    print(f"Top 5 特征:\n{importance.head()}")
    
    print("\n✅ 所有测试通过")
