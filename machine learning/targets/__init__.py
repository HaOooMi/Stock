"""
目标工程模块

包含：
1. TargetEngineer - 目标变量生成
2. LabelTransformer - 标签转换（残差收益、排名等）
"""

from .target_engineering import TargetEngineer
from .label_transformer import (
    LabelTransformer,
    create_forward_returns_with_transform
)

__all__ = [
    'TargetEngineer',
    'LabelTransformer',
    'create_forward_returns_with_transform'
]
