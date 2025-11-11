#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据快照管理器 - Data Snapshot Manager

功能：
1. 数据版本化与快照管理
2. Point-in-Time (PIT) 数据确保无前视偏差
3. 数据质量检查与报告
4. 实验元数据记录

符合研究宪章要求：
- 历史成分管理（避免幸存者偏差）
- 财务数据PIT对齐
- 交易可行性过滤
- 数据质量报表
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import hashlib
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
project_root = os.path.dirname(ml_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class DataSnapshot:
    """
    数据快照管理器
    
    职责：
    1. 创建数据快照并分配唯一ID
    2. 记录快照元数据（样本期、股票池、过滤阈值等）
    3. 验证数据质量
    4. 生成数据质量报告
    """
    
    def __init__(self, 
                 output_dir: str = "ML output",
                 snapshot_id: Optional[str] = None):
        """
        初始化数据快照管理器
        
        Parameters:
        -----------
        output_dir : str
            输出目录
        snapshot_id : str, optional
            快照ID，如果为None则自动生成
        """
        if os.path.isabs(output_dir):
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(ml_root, output_dir)
        
        # 创建必要的目录
        self.snapshots_dir = os.path.join(self.output_dir, 'snapshots')
        self.quality_reports_dir = os.path.join(self.output_dir, 'reports', 'data_quality')
        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.quality_reports_dir, exist_ok=True)
        
        # 生成或使用指定的快照ID
        if snapshot_id is None:
            self.snapshot_id = self._generate_snapshot_id()
        else:
            self.snapshot_id = snapshot_id
        
        # 快照元数据
        self.metadata = {
            'snapshot_id': self.snapshot_id,
            'created_at': datetime.now().isoformat(),
            'version': '1.0',
            'data_sources': [],
            'filters': {},
            'quality_checks': {},
            'statistics': {}
        }
        
        print(f"📸 数据快照管理器初始化")
        print(f"   快照ID: {self.snapshot_id}")
        print(f"   快照目录: {self.snapshots_dir}")
    
    def _generate_snapshot_id(self) -> str:
        """
        生成快照ID
        
        格式: ds_YYYYQQ_vN (如 ds_2025Q4_v1)
        
        Returns:
        --------
        str
            快照ID
        """
        now = datetime.now()
        year = now.year
        quarter = (now.month - 1) // 3 + 1
        
        # 查找当前季度的最大版本号
        prefix = f"ds_{year}Q{quarter}"
        existing_snapshots = [d for d in os.listdir(self.snapshots_dir) 
                            if d.startswith(prefix)]
        
        if existing_snapshots:
            versions = [int(s.split('_v')[-1]) for s in existing_snapshots 
                       if '_v' in s]
            next_version = max(versions) + 1 if versions else 1
        else:
            next_version = 1
        
        return f"{prefix}_v{next_version}"
    
    def create_snapshot(self,
                       data: pd.DataFrame,
                       symbol: str,
                       start_date: str,
                       end_date: str,
                       filters: Dict[str, Any],
                       random_seed: int = 42,
                       save_parquet: bool = True) -> str:
        """
        创建数据快照
        
        Parameters:
        -----------
        data : pd.DataFrame
            原始数据（MultiIndex [date, ticker]）
        symbol : str
            股票代码
        start_date : str
            开始日期
        end_date : str
            结束日期
        filters : dict
            过滤参数
        random_seed : int
            随机种子
        save_parquet : bool
            是否保存为Parquet格式
            
        Returns:
        --------
        str
            快照路径
        """
        print(f"\n📸 创建数据快照: {self.snapshot_id}")
        
        # 更新元数据
        self.metadata['symbol'] = symbol
        self.metadata['start_date'] = start_date
        self.metadata['end_date'] = end_date
        self.metadata['filters'] = filters
        self.metadata['random_seed'] = random_seed
        self.metadata['n_samples'] = len(data)
        self.metadata['n_features'] = len(data.columns)
        
        # 数据质量检查
        quality_report = self.check_data_quality(data)
        self.metadata['quality_checks'] = quality_report
        
        # 计算数据哈希（用于验证数据完整性）
        data_hash = self._calculate_data_hash(data)
        self.metadata['data_hash'] = data_hash
        
        # 创建快照目录
        snapshot_path = os.path.join(self.snapshots_dir, self.snapshot_id)
        os.makedirs(snapshot_path, exist_ok=True)
        
        # 保存数据
        if save_parquet:
            # Parquet格式（推荐，快速且列存）
            data_file = os.path.join(snapshot_path, f'{symbol}_data.parquet')
            data.to_parquet(data_file, engine='pyarrow', compression='snappy')
            self.metadata['data_file'] = data_file
            self.metadata['data_format'] = 'parquet'
            print(f"   💾 数据已保存: {data_file}")
        else:
            # CSV格式（备选）
            data_file = os.path.join(snapshot_path, f'{symbol}_data.csv')
            data.to_csv(data_file)
            self.metadata['data_file'] = data_file
            self.metadata['data_format'] = 'csv'
            print(f"   💾 数据已保存: {data_file}")
        
        # 保存元数据
        metadata_file = os.path.join(snapshot_path, 'metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False, default=str)
        print(f"   📝 元数据已保存: {metadata_file}")
        
        # 生成数据质量报告
        self.save_quality_report(quality_report)
        
        print(f"   ✅ 快照创建完成: {snapshot_path}")
        
        return snapshot_path
    
    def check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        数据质量检查
        
        检查项：
        1. 缺失率
        2. 极值比例
        3. 重复数据
        4. 停牌/涨跌停比例
        5. 可交易样本数
        
        Parameters:
        -----------
        data : pd.DataFrame
            数据
            
        Returns:
        --------
        dict
            质量报告
        """
        print(f"\n🔍 执行数据质量检查...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(data),
            'checks': {}
        }
        
        # 1. 缺失率检查
        missing_ratio = data.isna().sum() / len(data)
        report['checks']['missing_ratio'] = {
            'max': float(missing_ratio.max()),
            'mean': float(missing_ratio.mean()),
            'columns_with_missing': missing_ratio[missing_ratio > 0].to_dict()
        }
        
        # 红灯：任何列缺失率 > 20%
        red_flag_missing = missing_ratio.max() > 0.20
        report['checks']['missing_ratio']['red_flag'] = bool(red_flag_missing)
        
        print(f"   ✓ 缺失率: 最大 {missing_ratio.max():.2%}, 平均 {missing_ratio.mean():.2%}")
        if red_flag_missing:
            print(f"     ⚠️  红灯: 存在列缺失率 > 20%")
        
        # 2. 极值检查（仅数值列）
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        extreme_ratios = {}
        for col in numeric_cols:
            q1, q99 = data[col].quantile([0.01, 0.99])
            extreme_count = ((data[col] < q1) | (data[col] > q99)).sum()
            extreme_ratios[col] = float(extreme_count / len(data))
        
        report['checks']['extreme_values'] = {
            'max': max(extreme_ratios.values()) if extreme_ratios else 0,
            'mean': np.mean(list(extreme_ratios.values())) if extreme_ratios else 0
        }
        
        print(f"   ✓ 极值比例: 最大 {report['checks']['extreme_values']['max']:.2%}")
        
        # 3. 重复数据检查
        if isinstance(data.index, pd.MultiIndex):
            duplicates = data.index.duplicated().sum()
        else:
            duplicates = data.index.duplicated().sum()
        
        report['checks']['duplicates'] = {
            'count': int(duplicates),
            'ratio': float(duplicates / len(data))
        }
        
        # 红灯：重复率 > 1%
        red_flag_duplicates = (duplicates / len(data)) > 0.01
        report['checks']['duplicates']['red_flag'] = bool(red_flag_duplicates)
        
        print(f"   ✓ 重复数据: {duplicates} ({duplicates/len(data):.2%})")
        if red_flag_duplicates:
            print(f"     ⚠️  红灯: 重复率 > 1%")
        
        # 4. 可交易样本检查
        if 'tradable_flag' in data.columns:
            tradable_count = data['tradable_flag'].sum()
            tradable_ratio = tradable_count / len(data)
            report['checks']['tradable_samples'] = {
                'count': int(tradable_count),
                'ratio': float(tradable_ratio)
            }
            
            # 红灯：可交易样本 < 70%
            red_flag_tradable = tradable_ratio < 0.70
            report['checks']['tradable_samples']['red_flag'] = bool(red_flag_tradable)
            
            print(f"   ✓ 可交易样本: {tradable_count} ({tradable_ratio:.2%})")
            if red_flag_tradable:
                print(f"     ⚠️  红灯: 可交易样本 < 70%")
        
        # 5. 停牌/涨跌停检查
        if 'volume' in data.columns:
            suspended = (data['volume'] == 0).sum()
            suspended_ratio = suspended / len(data)
            report['checks']['suspended'] = {
                'count': int(suspended),
                'ratio': float(suspended_ratio)
            }
            print(f"   ✓ 停牌: {suspended} ({suspended_ratio:.2%})")
        
        if 'pct_change' in data.columns or 'close' in data.columns:
            # 计算涨跌幅
            if 'pct_change' in data.columns:
                pct_change = data['pct_change']
            else:
                pct_change = data['close'].pct_change()
            
            limit_up = (pct_change > 0.095).sum()
            limit_down = (pct_change < -0.095).sum()
            limit_ratio = (limit_up + limit_down) / len(data)
            
            report['checks']['limit_moves'] = {
                'limit_up': int(limit_up),
                'limit_down': int(limit_down),
                'total': int(limit_up + limit_down),
                'ratio': float(limit_ratio)
            }
            print(f"   ✓ 涨跌停: 上 {limit_up}, 下 {limit_down} (总计 {limit_ratio:.2%})")
        
        # 6. 时间连续性检查
        if isinstance(data.index, pd.MultiIndex):
            dates = data.index.get_level_values('date').unique()
        else:
            dates = pd.to_datetime(data.index).unique()
        
        dates = pd.Series(dates).sort_values()
        date_gaps = dates.diff().dt.days
        max_gap = date_gaps.max()
        
        report['checks']['time_continuity'] = {
            'n_dates': int(len(dates)),
            'max_gap_days': int(max_gap) if not pd.isna(max_gap) else 0,
            'date_range': f"{dates.min().date()} ~ {dates.max().date()}"
        }
        
        # 红灯：最大间隔 > 10天（可能存在数据缺失）
        red_flag_gap = max_gap > 10 if not pd.isna(max_gap) else False
        report['checks']['time_continuity']['red_flag'] = bool(red_flag_gap)
        
        print(f"   ✓ 时间连续性: {len(dates)} 个交易日, 最大间隔 {max_gap} 天")
        if red_flag_gap:
            print(f"     ⚠️  红灯: 最大间隔 > 10天")
        
        # 总体评分
        red_flags = sum([
            report['checks'].get('missing_ratio', {}).get('red_flag', False),
            report['checks'].get('duplicates', {}).get('red_flag', False),
            report['checks'].get('tradable_samples', {}).get('red_flag', False),
            report['checks'].get('time_continuity', {}).get('red_flag', False)
        ])
        
        report['overall_quality'] = 'PASS' if red_flags == 0 else 'WARNING'
        report['red_flags_count'] = int(red_flags)
        
        print(f"\n   {'✅' if red_flags == 0 else '⚠️ '} 总体评分: {report['overall_quality']} ({red_flags} 个红灯)")
        
        return report
    
    def save_quality_report(self, report: Dict[str, Any]):
        """
        保存数据质量报告
        
        Parameters:
        -----------
        report : dict
            质量报告
        """
        # 按日期命名报告文件
        report_file = os.path.join(
            self.quality_reports_dir,
            f"{self.snapshot_id}.json"
        )
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"   📊 质量报告已保存: {report_file}")
    
    def load_snapshot(self, snapshot_id: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        加载快照数据
        
        Parameters:
        -----------
        snapshot_id : str, optional
            快照ID，如果为None则使用当前快照ID
            
        Returns:
        --------
        Tuple[pd.DataFrame, dict]
            (数据, 元数据)
        """
        if snapshot_id is None:
            snapshot_id = self.snapshot_id
        
        snapshot_path = os.path.join(self.snapshots_dir, snapshot_id)
        
        if not os.path.exists(snapshot_path):
            raise FileNotFoundError(f"快照不存在: {snapshot_path}")
        
        # 加载元数据
        metadata_file = os.path.join(snapshot_path, 'metadata.json')
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 加载数据
        data_format = metadata.get('data_format', 'parquet')
        symbol = metadata.get('symbol', '000001')
        
        if data_format == 'parquet':
            data_file = os.path.join(snapshot_path, f'{symbol}_data.parquet')
            data = pd.read_parquet(data_file)
        else:
            data_file = os.path.join(snapshot_path, f'{symbol}_data.csv')
            data = pd.read_csv(data_file, index_col=[0, 1], parse_dates=[0])
        
        # 验证数据完整性
        data_hash = self._calculate_data_hash(data)
        if data_hash != metadata.get('data_hash'):
            print(f"⚠️  警告: 数据哈希不匹配，数据可能已被修改")
        
        print(f"✅ 快照加载成功: {snapshot_id}")
        print(f"   样本数: {len(data)}")
        print(f"   特征数: {len(data.columns)}")
        
        return data, metadata
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """
        计算数据哈希值
        
        Parameters:
        -----------
        data : pd.DataFrame
            数据
            
        Returns:
        --------
        str
            MD5哈希值
        """
        # 使用数据的形状和前10行作为哈希输入
        hash_input = f"{data.shape}_{data.head(10).to_json()}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def list_snapshots(self) -> pd.DataFrame:
        """
        列出所有快照
        
        Returns:
        --------
        pd.DataFrame
            快照列表
        """
        snapshots = []
        
        for snapshot_id in os.listdir(self.snapshots_dir):
            snapshot_path = os.path.join(self.snapshots_dir, snapshot_id)
            metadata_file = os.path.join(snapshot_path, 'metadata.json')
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                snapshots.append({
                    'snapshot_id': snapshot_id,
                    'created_at': metadata.get('created_at'),
                    'symbol': metadata.get('symbol'),
                    'start_date': metadata.get('start_date'),
                    'end_date': metadata.get('end_date'),
                    'n_samples': metadata.get('n_samples'),
                    'quality': metadata.get('quality_checks', {}).get('overall_quality', 'UNKNOWN')
                })
        
        return pd.DataFrame(snapshots)


if __name__ == "__main__":
    """
    使用示例
    """
    print("📸 数据快照管理器测试")
    print("=" * 50)
    
    # 创建示例数据
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
    tickers = ['000001'] * len(dates)
    
    data = pd.DataFrame({
        'close': np.random.randn(len(dates)) * 10 + 100,
        'volume': np.random.randint(1000000, 10000000, len(dates)),
        'feature_1': np.random.randn(len(dates)),
        'feature_2': np.random.randn(len(dates)),
        'tradable_flag': np.random.choice([0, 1], len(dates), p=[0.1, 0.9])
    }, index=pd.MultiIndex.from_arrays([dates, tickers], names=['date', 'ticker']))
    
    # 创建快照管理器
    snapshot_mgr = DataSnapshot()
    
    # 创建快照
    filters = {
        'min_volume': 1000000,
        'min_price': 1.0,
        'exclude_st': True
    }
    
    snapshot_path = snapshot_mgr.create_snapshot(
        data=data,
        symbol='000001',
        start_date='2022-01-01',
        end_date='2024-12-31',
        filters=filters,
        random_seed=42
    )
    
    print(f"\n✅ 快照创建成功: {snapshot_path}")
    
    # 列出所有快照
    print("\n📋 现有快照:")
    snapshots_df = snapshot_mgr.list_snapshots()
    print(snapshots_df)
