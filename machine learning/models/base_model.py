#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础模型类 - 定义统一接口
"""

from abc import ABC, abstractmethod
import pickle
import joblib
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class BaseModel(ABC):
    """
    基础模型抽象类
    
    所有模型必须实现的接口：
    - fit: 训练模型
    - predict: 预测
    - save: 保存模型
    - load: 加载模型
    - get_feature_importance: 获取特征重要性
    """
    
    def __init__(self, name: str, params: Optional[Dict] = None):
        """
        初始化模型
        
        Parameters:
        -----------
        name : str
            模型名称
        params : dict, optional
            模型参数
        """
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        self.feature_names = None
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            X_valid: Optional[pd.DataFrame] = None,
            y_valid: Optional[pd.Series] = None) -> Dict:
        """
        训练模型
        
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
            训练结果（包括训练时间、样本数等）
        """
        pass
    
    @abstractmethod
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
        pass
    
    def save(self, filepath: str, format: str = 'pickle'):
        """
        保存模型
        
        Parameters:
        -----------
        filepath : str
            保存路径
        format : str
            保存格式 ('pickle' 或 'joblib')
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，无法保存")
        
        model_data = {
            'name': self.name,
            'params': self.params,
            'model': self.model,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
        elif format == 'joblib':
            joblib.dump(model_data, filepath)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        print(f"   💾 模型已保存: {filepath}")
    
    def load(self, filepath: str, format: str = 'pickle'):
        """
        加载模型
        
        Parameters:
        -----------
        filepath : str
            模型文件路径
        format : str
            文件格式 ('pickle' 或 'joblib')
        """
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
        elif format == 'joblib':
            model_data = joblib.load(filepath)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        self.name = model_data['name']
        self.params = model_data['params']
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']
        
        print(f"   📂 模型已加载: {filepath}")
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        获取特征重要性
        
        Returns:
        --------
        pd.Series or None
            特征重要性（如果模型支持）
        """
        return None
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"
