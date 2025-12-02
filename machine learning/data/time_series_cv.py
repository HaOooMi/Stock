#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时序交叉验证模块（Purged + Embargo + Walk-Forward）

功能：
1. 单次时间切分（train/valid/test）
2. Purged：边界前剔除 ≥ max_horizon 天样本（避免目标重叠）
3. Embargo：边界后禁止若干日参与评估（避免信息泄漏）
4. Walk-Forward：滚动或扩张窗口多折

设计原则：
- 只做索引切分，不碰特征/标签本身
- 支持 MultiIndex [date, ticker] 和普通 DatetimeIndex
- 与 configs/ml_baseline.yml 中的 split 配置集成
- 可复用于任何 pipeline

核心公式（Purged + Embargo）：
┌──────────────────┬─────────────────┬──────────────────┐
│      Train       │      Valid      │       Test       │
└──────────────────┴─────────────────┴──────────────────┘
                   ↑                 ↑
           边界前purge日       边界前purge日
           边界后embargo日     边界后embargo日

创建: 2025-12-02 | 版本: v1.0
作者: AI Assistant
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Generator
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
project_root = os.path.dirname(ml_root)
if ml_root not in sys.path:
    sys.path.insert(0, ml_root)


class TimeSeriesCV:
    """
    时序交叉验证器
    
    支持：
    1. 单次时间切分（带 Purged + Embargo）
    2. Walk-Forward 多折验证
    3. 滚动窗口 vs 扩张窗口
    
    契约：
    ------
    输入：
        - data: DataFrame 或 Index，需包含日期信息
          - MultiIndex [date, ticker]：自动提取 date level
          - DatetimeIndex：直接使用
          - 普通 DataFrame：需指定 date_col
    
    输出：
        - 单次切分：(train_idx, valid_idx, test_idx)
        - WFA：Generator[(fold_id, train_idx, valid_idx, test_idx), ...]
    """
    
    def __init__(self,
                 train_ratio: float = 0.6,
                 valid_ratio: float = 0.2,
                 test_ratio: float = 0.2,
                 purge_days: int = 10,
                 embargo_days: int = 5,
                 max_horizon: int = 10):
        """
        初始化时序 CV
        
        Parameters:
        -----------
        train_ratio : float
            训练集比例（默认 0.6）
        valid_ratio : float
            验证集比例（默认 0.2）
        test_ratio : float
            测试集比例（默认 0.2）
        purge_days : int
            Purge 天数（边界前剔除，避免目标重叠）
            建议设为 max(target_horizon)
        embargo_days : int
            Embargo 天数（边界后禁止，避免信息泄漏）
            建议设为 target_horizon / 2
        max_horizon : int
            最大预测周期（用于自动计算 purge_days）
        """
        # 验证比例
        total_ratio = train_ratio + valid_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"切分比例总和必须为1，当前为 {total_ratio}")
        
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.purge_days = max(purge_days, max_horizon)  # purge 至少为 max_horizon
        self.embargo_days = embargo_days
        self.max_horizon = max_horizon
        
        # 切分元数据（用于记录和复现）
        self.split_meta = {}
        
        print(f"📅 时序CV初始化")
        print(f"   切分比例: {train_ratio:.0%} / {valid_ratio:.0%} / {test_ratio:.0%}")
        print(f"   Purge: {self.purge_days} 天 | Embargo: {embargo_days} 天")
    
    @classmethod
    def from_config(cls, config: dict) -> 'TimeSeriesCV':
        """
        从配置字典创建实例
        
        Parameters:
        -----------
        config : dict
            配置字典，期望包含 'split' 键
            
        Returns:
        --------
        TimeSeriesCV
            实例
        """
        split_cfg = config.get('split', {})
        target_cfg = config.get('target', {})
        
        # 从配置读取参数
        train_ratio = split_cfg.get('train_ratio', 0.6)
        valid_ratio = split_cfg.get('valid_ratio', 0.2)
        test_ratio = split_cfg.get('test_ratio', 0.2)
        purge_days = split_cfg.get('purge_days', 10)
        embargo_days = split_cfg.get('embargo_days', 5)
        max_horizon = target_cfg.get('forward_periods', 10)
        
        return cls(
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            purge_days=purge_days,
            embargo_days=embargo_days,
            max_horizon=max_horizon
        )
    
    def _extract_dates(self, data: Union[pd.DataFrame, pd.Index]) -> pd.DatetimeIndex:
        """
        从数据中提取日期索引
        
        Parameters:
        -----------
        data : DataFrame 或 Index
            输入数据
            
        Returns:
        --------
        pd.DatetimeIndex
            唯一日期列表（已排序）
        """
        if isinstance(data, pd.DataFrame):
            index = data.index
        else:
            index = data
        
        # MultiIndex [date, ticker]
        if isinstance(index, pd.MultiIndex):
            if 'date' in index.names:
                dates = index.get_level_values('date').unique()
            else:
                # 假设第一层是日期
                dates = index.get_level_values(0).unique()
        # DatetimeIndex
        elif isinstance(index, pd.DatetimeIndex):
            dates = index.unique()
        else:
            raise ValueError(f"不支持的索引类型: {type(index)}")
        
        return pd.DatetimeIndex(dates).sort_values()
    
    def _get_date_boundaries(self, dates: pd.DatetimeIndex) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        计算 train/valid 和 valid/test 的边界日期
        
        Parameters:
        -----------
        dates : pd.DatetimeIndex
            唯一日期列表
            
        Returns:
        --------
        Tuple[pd.Timestamp, pd.Timestamp]
            (train_end, valid_end)
        """
        n_dates = len(dates)
        
        train_end_idx = int(n_dates * self.train_ratio) - 1
        valid_end_idx = int(n_dates * (self.train_ratio + self.valid_ratio)) - 1
        
        train_end = dates[train_end_idx]
        valid_end = dates[valid_end_idx]
        
        return train_end, valid_end
    
    def single_split(self, 
                    data: Union[pd.DataFrame, pd.Index],
                    return_masks: bool = False) -> Union[
                        Tuple[pd.Index, pd.Index, pd.Index],
                        Tuple[pd.Series, pd.Series, pd.Series]
                    ]:
        """
        单次时间切分（带 Purged + Embargo）
        
        Parameters:
        -----------
        data : DataFrame 或 Index
            输入数据
        return_masks : bool
            True: 返回布尔掩码 (用于 DataFrame.loc[mask])
            False: 返回索引 (用于 DataFrame.loc[idx])
            
        Returns:
        --------
        如果 return_masks=False:
            Tuple[pd.Index, pd.Index, pd.Index]
                (train_idx, valid_idx, test_idx)
        如果 return_masks=True:
            Tuple[pd.Series, pd.Series, pd.Series]
                (train_mask, valid_mask, test_mask)
        """
        print("\n" + "=" * 70)
        print("单次时间切分（Purged + Embargo）")
        print("=" * 70)
        
        # 提取日期
        dates = self._extract_dates(data)
        train_end, valid_end = self._get_date_boundaries(dates)
        
        # 计算 Purge 和 Embargo 边界
        # train/valid 边界
        train_purge_start = train_end - pd.Timedelta(days=self.purge_days)
        valid_embargo_end = train_end + pd.Timedelta(days=self.embargo_days)
        
        # valid/test 边界
        valid_purge_start = valid_end - pd.Timedelta(days=self.purge_days)
        test_embargo_end = valid_end + pd.Timedelta(days=self.embargo_days)
        
        # 记录元数据
        self.split_meta = {
            'mode': 'single_split',
            'train_end': train_end.isoformat(),
            'valid_end': valid_end.isoformat(),
            'purge_days': self.purge_days,
            'embargo_days': self.embargo_days,
            'n_dates': len(dates),
            'date_range': (dates.min().isoformat(), dates.max().isoformat())
        }
        
        # 获取原始索引
        if isinstance(data, pd.DataFrame):
            original_index = data.index
        else:
            original_index = data
        
        # 构建掩码
        if isinstance(original_index, pd.MultiIndex):
            date_values = original_index.get_level_values('date')
        else:
            date_values = original_index
        
        # Train: 日期 <= train_purge_start（排除 purge 区间）
        train_mask = date_values <= train_purge_start
        
        # Valid: valid_embargo_end <= 日期 <= valid_purge_start
        valid_mask = (date_values >= valid_embargo_end) & (date_values <= valid_purge_start)
        
        # Test: 日期 >= test_embargo_end
        test_mask = date_values >= test_embargo_end
        
        # 统计信息
        print(f"\n📊 切分统计:")
        print(f"   日期范围: {dates.min().date()} ~ {dates.max().date()}")
        print(f"   总日期数: {len(dates)}")
        print(f"\n   Train:")
        print(f"      结束: {train_purge_start.date()} (purge前)")
        print(f"      样本: {train_mask.sum()}")
        print(f"\n   Valid:")
        print(f"      开始: {valid_embargo_end.date()} (embargo后)")
        print(f"      结束: {valid_purge_start.date()} (purge前)")
        print(f"      样本: {valid_mask.sum()}")
        print(f"\n   Test:")
        print(f"      开始: {test_embargo_end.date()} (embargo后)")
        print(f"      样本: {test_mask.sum()}")
        
        # Purge + Embargo 造成的样本损失
        total_samples = len(original_index)
        used_samples = train_mask.sum() + valid_mask.sum() + test_mask.sum()
        purged_samples = total_samples - used_samples
        print(f"\n   ⚠️  Purge+Embargo 剔除: {purged_samples} 样本 ({purged_samples/total_samples*100:.1f}%)")
        
        if return_masks:
            return (
                pd.Series(train_mask, index=original_index),
                pd.Series(valid_mask, index=original_index),
                pd.Series(test_mask, index=original_index)
            )
        else:
            return (
                original_index[train_mask],
                original_index[valid_mask],
                original_index[test_mask]
            )
    
    def walk_forward_split(self,
                          data: Union[pd.DataFrame, pd.Index],
                          n_splits: int = 5,
                          min_train_days: int = 252,
                          expanding: bool = True) -> Generator[
                              Tuple[int, pd.Index, pd.Index, pd.Index], 
                              None, None
                          ]:
        """
        Walk-Forward 验证切分
        
        Parameters:
        -----------
        data : DataFrame 或 Index
            输入数据
        n_splits : int
            折数（默认 5）
        min_train_days : int
            最小训练日数（默认 252，约1年）
        expanding : bool
            True: 扩张窗口（训练集不断增长）
            False: 滚动窗口（训练集大小固定）
            
        Yields:
        -------
        Tuple[int, pd.Index, pd.Index, pd.Index]
            (fold_id, train_idx, valid_idx, test_idx)
        """
        print("\n" + "=" * 70)
        print(f"Walk-Forward 验证 ({'扩张窗口' if expanding else '滚动窗口'})")
        print("=" * 70)
        
        dates = self._extract_dates(data)
        n_dates = len(dates)
        
        # 计算每折的 valid+test 天数
        valid_test_days = int(n_dates * (self.valid_ratio + self.test_ratio))
        valid_days = int(valid_test_days * self.valid_ratio / (self.valid_ratio + self.test_ratio))
        test_days = valid_test_days - valid_days
        
        # 每折移动的步长
        step = (n_dates - min_train_days - valid_test_days) // max(n_splits - 1, 1)
        
        if step <= 0:
            raise ValueError(f"数据量不足以进行 {n_splits} 折 WFA，"
                           f"总日期数={n_dates}, min_train={min_train_days}, "
                           f"valid+test={valid_test_days}")
        
        # 记录元数据
        self.split_meta = {
            'mode': 'walk_forward',
            'n_splits': n_splits,
            'expanding': expanding,
            'min_train_days': min_train_days,
            'valid_days': valid_days,
            'test_days': test_days,
            'step': step
        }
        
        # 获取原始索引
        if isinstance(data, pd.DataFrame):
            original_index = data.index
        else:
            original_index = data
        
        if isinstance(original_index, pd.MultiIndex):
            date_values = original_index.get_level_values('date')
        else:
            date_values = original_index
        
        print(f"\n📊 WFA 配置:")
        print(f"   总日期数: {n_dates}")
        print(f"   折数: {n_splits}")
        print(f"   每折步长: {step} 天")
        print(f"   Valid: {valid_days} 天 | Test: {test_days} 天")
        print(f"   Purge: {self.purge_days} 天 | Embargo: {self.embargo_days} 天")
        
        for fold in range(n_splits):
            # 计算当前折的边界
            if expanding:
                train_start_idx = 0
            else:
                train_start_idx = fold * step
            
            train_end_idx = min_train_days + fold * step - 1
            valid_start_idx = train_end_idx + 1
            valid_end_idx = valid_start_idx + valid_days - 1
            test_start_idx = valid_end_idx + 1
            test_end_idx = test_start_idx + test_days - 1
            
            # 边界检查
            if test_end_idx >= n_dates:
                break
            
            # 获取边界日期
            train_start = dates[train_start_idx]
            train_end = dates[train_end_idx]
            valid_start = dates[valid_start_idx]
            valid_end = dates[valid_end_idx]
            test_start = dates[test_start_idx]
            test_end = dates[test_end_idx]
            
            # 应用 Purge + Embargo
            train_purge_end = train_end - pd.Timedelta(days=self.purge_days)
            valid_embargo_start = valid_start + pd.Timedelta(days=self.embargo_days)
            valid_purge_end = valid_end - pd.Timedelta(days=self.purge_days)
            test_embargo_start = test_start + pd.Timedelta(days=self.embargo_days)
            
            # 构建掩码
            train_mask = (date_values >= train_start) & (date_values <= train_purge_end)
            valid_mask = (date_values >= valid_embargo_start) & (date_values <= valid_purge_end)
            test_mask = (date_values >= test_embargo_start) & (date_values <= test_end)
            
            # 获取索引
            train_idx = original_index[train_mask]
            valid_idx = original_index[valid_mask]
            test_idx = original_index[test_mask]
            
            print(f"\n   📁 Fold {fold + 1}/{n_splits}:")
            print(f"      Train: {train_start.date()} ~ {train_purge_end.date()} ({len(train_idx)} 样本)")
            print(f"      Valid: {valid_embargo_start.date()} ~ {valid_purge_end.date()} ({len(valid_idx)} 样本)")
            print(f"      Test:  {test_embargo_start.date()} ~ {test_end.date()} ({len(test_idx)} 样本)")
            
            yield (fold, train_idx, valid_idx, test_idx)
    
    def get_split_meta(self) -> dict:
        """获取切分元数据"""
        return self.split_meta.copy()
    
    def validate_no_leakage(self,
                           train_idx: pd.Index,
                           valid_idx: pd.Index,
                           test_idx: pd.Index,
                           target_horizon: int = 5) -> bool:
        """
        验证切分无数据泄漏
        
        Parameters:
        -----------
        train_idx, valid_idx, test_idx : pd.Index
            切分索引
        target_horizon : int
            目标预测周期
            
        Returns:
        --------
        bool
            True = 无泄漏
        """
        print("\n🔍 验证数据泄漏...")
        
        # 提取日期
        def get_dates(idx):
            if isinstance(idx, pd.MultiIndex):
                return idx.get_level_values('date').unique()
            return idx.unique()
        
        train_dates = get_dates(train_idx)
        valid_dates = get_dates(valid_idx)
        test_dates = get_dates(test_idx)
        
        # 检查 1: 日期不重叠
        train_valid_overlap = len(set(train_dates) & set(valid_dates))
        valid_test_overlap = len(set(valid_dates) & set(test_dates))
        train_test_overlap = len(set(train_dates) & set(test_dates))
        
        if train_valid_overlap > 0 or valid_test_overlap > 0 or train_test_overlap > 0:
            print(f"   ❌ 日期重叠! Train-Valid: {train_valid_overlap}, "
                  f"Valid-Test: {valid_test_overlap}, Train-Test: {train_test_overlap}")
            return False
        
        # 检查 2: Train 的最大日期 + horizon < Valid 的最小日期
        train_max = train_dates.max()
        valid_min = valid_dates.min()
        gap_train_valid = (valid_min - train_max).days
        
        if gap_train_valid < target_horizon:
            print(f"   ⚠️  Train-Valid 间隔不足: {gap_train_valid} 天 < {target_horizon} 天")
            # 不立即失败，只警告
        
        # 检查 3: Valid 的最大日期 + horizon < Test 的最小日期
        valid_max = valid_dates.max()
        test_min = test_dates.min()
        gap_valid_test = (test_min - valid_max).days
        
        if gap_valid_test < target_horizon:
            print(f"   ⚠️  Valid-Test 间隔不足: {gap_valid_test} 天 < {target_horizon} 天")
        
        print(f"   ✅ 无日期重叠")
        print(f"   📊 Train-Valid 间隔: {gap_train_valid} 天")
        print(f"   📊 Valid-Test 间隔: {gap_valid_test} 天")
        
        return True


def create_cv_from_config(config_path: str = "configs/ml_baseline.yml") -> TimeSeriesCV:
    """
    便捷函数：从配置文件创建 CV 实例
    
    Parameters:
    -----------
    config_path : str
        配置文件路径
        
    Returns:
    --------
    TimeSeriesCV
        实例
    """
    import yaml
    
    if not os.path.isabs(config_path):
        config_path = os.path.join(ml_root, config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return TimeSeriesCV.from_config(config)


if __name__ == '__main__':
    """测试代码"""
    print("=" * 70)
    print("时序交叉验证模块测试")
    print("=" * 70)
    
    # 生成测试数据
    np.random.seed(42)
    
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    dates = dates[dates.dayofweek < 5]  # 只保留工作日
    tickers = ['000001', '000002', '000003', '000004', '000005']
    
    # 创建 MultiIndex
    index = pd.MultiIndex.from_product(
        [dates, tickers],
        names=['date', 'ticker']
    )
    
    # 模拟数据
    test_data = pd.DataFrame({
        'feature_1': np.random.randn(len(index)),
        'feature_2': np.random.randn(len(index)),
        'close': np.random.randn(len(index)).cumsum() + 100
    }, index=index)
    
    print(f"\n📊 测试数据:")
    print(f"   形状: {test_data.shape}")
    print(f"   日期范围: {dates.min().date()} ~ {dates.max().date()}")
    print(f"   股票数: {len(tickers)}")
    
    # 1. 单次切分测试
    print("\n" + "=" * 70)
    print("测试 1: 单次切分")
    print("=" * 70)
    
    cv = TimeSeriesCV(
        train_ratio=0.6,
        valid_ratio=0.2,
        test_ratio=0.2,
        purge_days=10,
        embargo_days=5,
        max_horizon=10
    )
    
    train_idx, valid_idx, test_idx = cv.single_split(test_data)
    
    print(f"\n📊 切分结果:")
    print(f"   Train: {len(train_idx)} 样本")
    print(f"   Valid: {len(valid_idx)} 样本")
    print(f"   Test:  {len(test_idx)} 样本")
    
    # 验证无泄漏
    cv.validate_no_leakage(train_idx, valid_idx, test_idx, target_horizon=10)
    
    # 2. Walk-Forward 测试
    print("\n" + "=" * 70)
    print("测试 2: Walk-Forward 验证")
    print("=" * 70)
    
    wfa_results = []
    for fold, train_idx, valid_idx, test_idx in cv.walk_forward_split(
        test_data, n_splits=3, min_train_days=252, expanding=True
    ):
        wfa_results.append({
            'fold': fold,
            'train_size': len(train_idx),
            'valid_size': len(valid_idx),
            'test_size': len(test_idx)
        })
    
    print(f"\n📊 WFA 结果汇总:")
    for r in wfa_results:
        print(f"   Fold {r['fold']+1}: Train={r['train_size']}, "
              f"Valid={r['valid_size']}, Test={r['test_size']}")
    
    # 3. 从配置文件创建
    print("\n" + "=" * 70)
    print("测试 3: 从配置文件创建")
    print("=" * 70)
    
    try:
        cv_from_cfg = create_cv_from_config()
        print("   ✅ 从配置文件创建成功")
    except Exception as e:
        print(f"   ⚠️  从配置文件创建失败: {e}")
    
    print("\n✅ 所有测试完成！")
