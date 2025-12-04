"""
目标工程模块

包含：
1. TargetEngineer - 目标变量生成
2. LabelTransformer - 标签转换（残差收益、排名等）
3. RankingLabelFactory - 排序标签构造（Reg-on-Rank, LambdaRank）
"""

from .target_engineering import TargetEngineer
from .label_transformer import (
    LabelTransformer,
    create_forward_returns_with_transform
)
from .ranking_labels import (
    RankingLabelFactory,
    create_ranking_labels
)

__all__ = [
    'TargetEngineer',
    'LabelTransformer',
    'create_forward_returns_with_transform',
    'RankingLabelFactory',
    'create_ranking_labels'
]
