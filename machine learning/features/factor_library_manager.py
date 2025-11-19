#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子库管理器 - 因子清单、版本控制、入库标准

功能：
1. 管理因子清单（final_feature_list.txt）
2. 因子版本控制
3. 因子入库/退库
4. 因子质量追踪
5. 生成因子报告
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
if ml_root not in sys.path:
    sys.path.insert(0, ml_root)


class FactorLibraryManager:
    """
    因子库管理器
    
    管理因子清单、版本、质量追踪
    """
    
    def __init__(self, 
                 artifacts_dir: str = "ML output/artifacts/baseline_v1",
                 reports_dir: str = "ML output/reports/baseline_v1/factors"):
        """
        初始化因子库管理器
        
        Parameters:
        -----------
        artifacts_dir : str
            制品目录（存放因子清单）
        reports_dir : str
            因子报告目录
        """
        # 规范化路径
        self.artifacts_dir = artifacts_dir if os.path.isabs(artifacts_dir) else os.path.join(ml_root, artifacts_dir)
        self.reports_dir = reports_dir if os.path.isabs(reports_dir) else os.path.join(ml_root, reports_dir)
        
        # 创建目录
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # 因子清单文件
        self.feature_list_path = os.path.join(self.artifacts_dir, "final_feature_list.txt")
        self.feature_metadata_path = os.path.join(self.artifacts_dir, "factor_metadata.json")
        self.quality_history_path = os.path.join(self.artifacts_dir, "quality_history.csv")
        
        # 加载现有清单
        self.factor_list = self._load_factor_list()
        self.factor_metadata = self._load_metadata()
        
        print("📚 因子库管理器初始化")
        print(f"   制品目录: {self.artifacts_dir}")
        print(f"   报告目录: {self.reports_dir}")
        print(f"   当前因子数: {len(self.factor_list)}")
    
    def _load_factor_list(self) -> List[str]:
        """加载因子清单"""
        if os.path.exists(self.feature_list_path):
            with open(self.feature_list_path, 'r', encoding='utf-8') as f:
                factors = [line.strip() for line in f if line.strip()]
            return factors
        return []
    
    def _save_factor_list(self):
        """保存因子清单"""
        with open(self.feature_list_path, 'w', encoding='utf-8') as f:
            for factor in self.factor_list:
                f.write(f"{factor}\n")
        print(f"   💾 因子清单已保存: {self.feature_list_path}")
    
    def _load_metadata(self) -> Dict:
        """加载因子元数据"""
        if os.path.exists(self.feature_metadata_path):
            with open(self.feature_metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """保存因子元数据"""
        with open(self.feature_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.factor_metadata, f, indent=2, ensure_ascii=False)
        print(f"   💾 元数据已保存: {self.feature_metadata_path}")
    
    def add_factor(self, 
                  factor_name: str,
                  quality_report: Dict,
                  formula: str = "",
                  family: str = "",
                  reference: str = "") -> bool:
        """
        添加因子到库中
        
        Parameters:
        -----------
        factor_name : str
            因子名称
        quality_report : dict
            质量检查报告
        formula : str
            因子公式
        family : str
            因子族
        reference : str
            文献引用
            
        Returns:
        --------
        bool
            是否添加成功
        """
        print(f"\n📥 添加因子: {factor_name}")
        
        # 检查是否已存在
        if factor_name in self.factor_list:
            print(f"   ⚠️  因子已存在")
            return False
        
        # 检查质量是否通过
        if not quality_report.get('overall_pass', False):
            print(f"   ❌ 质量检查未通过，拒绝入库")
            return False
        
        # 添加到清单
        self.factor_list.append(factor_name)
        
        # 保存元数据
        self.factor_metadata[factor_name] = {
            'formula': formula,
            'family': family,
            'reference': reference,
            'added_date': datetime.now().isoformat(),
            'quality_report': {
                'ic_mean': quality_report['ic_metrics']['ic_mean'],
                'icir_annual': quality_report['ic_metrics']['icir_annual'],
                'psi': quality_report.get('psi', np.nan),
                'ic_half_life': quality_report.get('ic_half_life', np.nan),
                'max_corr': quality_report.get('corr_check', {}).get('max_corr', 0.0)
            },
            'status': 'active',
            'version': 1
        }
        
        # 保存
        self._save_factor_list()
        self._save_metadata()
        
        # 记录质量历史
        self._record_quality_history(factor_name, quality_report)
        
        print(f"   ✅ 因子已添加到库中")
        return True
    
    def remove_factor(self, factor_name: str, reason: str = "") -> bool:
        """
        从库中移除因子
        
        Parameters:
        -----------
        factor_name : str
            因子名称
        reason : str
            移除原因
            
        Returns:
        --------
        bool
            是否移除成功
        """
        print(f"\n📤 移除因子: {factor_name}")
        
        if factor_name not in self.factor_list:
            print(f"   ⚠️  因子不存在")
            return False
        
        # 从清单中移除
        self.factor_list.remove(factor_name)
        
        # 更新元数据状态
        if factor_name in self.factor_metadata:
            self.factor_metadata[factor_name]['status'] = 'removed'
            self.factor_metadata[factor_name]['removed_date'] = datetime.now().isoformat()
            self.factor_metadata[factor_name]['removal_reason'] = reason
        
        # 保存
        self._save_factor_list()
        self._save_metadata()
        
        print(f"   ✅ 因子已移除")
        return True
    
    def update_factor_quality(self, factor_name: str, quality_report: Dict) -> bool:
        """
        更新因子质量指标
        
        Parameters:
        -----------
        factor_name : str
            因子名称
        quality_report : dict
            新的质量检查报告
            
        Returns:
        --------
        bool
            是否更新成功
        """
        if factor_name not in self.factor_metadata:
            print(f"   ⚠️  因子 {factor_name} 不存在")
            return False
        
        # 更新质量指标
        self.factor_metadata[factor_name]['quality_report'] = {
            'ic_mean': quality_report['ic_metrics']['ic_mean'],
            'icir_annual': quality_report['ic_metrics']['icir_annual'],
            'psi': quality_report.get('psi', np.nan),
            'ic_half_life': quality_report.get('ic_half_life', np.nan),
            'max_corr': quality_report.get('corr_check', {}).get('max_corr', 0.0)
        }
        self.factor_metadata[factor_name]['last_updated'] = datetime.now().isoformat()
        
        # 保存
        self._save_metadata()
        
        # 记录质量历史
        self._record_quality_history(factor_name, quality_report)
        
        print(f"   ✅ 因子质量指标已更新")
        return True
    
    def _record_quality_history(self, factor_name: str, quality_report: Dict):
        """记录质量历史"""
        history_record = {
            'factor_name': factor_name,
            'timestamp': datetime.now().isoformat(),
            'ic_mean': quality_report['ic_metrics']['ic_mean'],
            'icir_annual': quality_report['ic_metrics']['icir_annual'],
            'psi': quality_report.get('psi', np.nan),
            'ic_half_life': quality_report.get('ic_half_life', np.nan),
            'overall_pass': quality_report.get('overall_pass', False)
        }
        
        # 追加到历史文件
        history_df = pd.DataFrame([history_record])
        
        if os.path.exists(self.quality_history_path):
            existing_history = pd.read_csv(self.quality_history_path)
            history_df = pd.concat([existing_history, history_df], ignore_index=True)
        
        history_df.to_csv(self.quality_history_path, index=False)
    
    def get_factor_info(self, factor_name: str) -> Optional[Dict]:
        """
        获取因子信息
        
        Parameters:
        -----------
        factor_name : str
            因子名称
            
        Returns:
        --------
        dict or None
            因子元数据
        """
        return self.factor_metadata.get(factor_name)
    
    def list_factors(self, status: str = 'active') -> List[str]:
        """
        列出因子
        
        Parameters:
        -----------
        status : str
            'active' 或 'all'
            
        Returns:
        --------
        List[str]
            因子列表
        """
        if status == 'active':
            return self.factor_list
        else:
            return list(self.factor_metadata.keys())
    
    def generate_factor_report(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        生成因子报告
        
        Parameters:
        -----------
        output_path : str, optional
            输出路径
            
        Returns:
        --------
        pd.DataFrame
            因子报告表
        """
        print("\n📊 生成因子报告...")
        
        report_data = []
        
        for factor_name in self.factor_list:
            metadata = self.factor_metadata.get(factor_name, {})
            quality = metadata.get('quality_report', {})
            
            report_data.append({
                '因子名称': factor_name,
                '因子族': metadata.get('family', ''),
                'IC均值': quality.get('ic_mean', np.nan),
                'ICIR年化': quality.get('icir_annual', np.nan),
                'PSI': quality.get('psi', np.nan),
                'IC半衰期': quality.get('ic_half_life', np.nan),
                '最大相关性': quality.get('max_corr', np.nan),
                '添加日期': metadata.get('added_date', ''),
                '状态': metadata.get('status', '')
            })
        
        report_df = pd.DataFrame(report_data)
        
        # 排序（按IC均值降序）
        if not report_df.empty:
            report_df = report_df.sort_values('IC均值', ascending=False, key=abs)
        
        # 保存
        if output_path is None:
            output_path = os.path.join(self.reports_dir, f"factor_report_{datetime.now().strftime('%Y%m%d')}.csv")
        
        report_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"   ✅ 报告已保存: {output_path}")
        print(f"\n📋 因子统计:")
        print(f"   总因子数: {len(report_df)}")
        
        # 按族统计
        if not report_df.empty and '因子族' in report_df.columns:
            family_counts = report_df['因子族'].value_counts()
            for family, count in family_counts.items():
                print(f"   {family}: {count} 个")
        
        return report_df
    
    def analyze_factor_family_performance(self) -> pd.DataFrame:
        """
        分析因子族表现
        
        Returns:
        --------
        pd.DataFrame
            族别汇总统计
        """
        report_data = []
        
        # 按族分组
        families = {}
        for factor_name in self.factor_list:
            metadata = self.factor_metadata.get(factor_name, {})
            family = metadata.get('family', '未分类')
            
            if family not in families:
                families[family] = []
            families[family].append(metadata.get('quality_report', {}))
        
        # 汇总统计
        for family, quality_reports in families.items():
            ic_values = [q.get('ic_mean', np.nan) for q in quality_reports]
            icir_values = [q.get('icir_annual', np.nan) for q in quality_reports]
            
            report_data.append({
                '因子族': family,
                '因子数量': len(quality_reports),
                '平均IC': np.nanmean(ic_values),
                '平均ICIR': np.nanmean(icir_values),
                'IC标准差': np.nanstd(ic_values)
            })
        
        family_df = pd.DataFrame(report_data)
        
        if not family_df.empty:
            family_df = family_df.sort_values('平均IC', ascending=False, key=abs)
        
        return family_df


if __name__ == "__main__":
    """测试因子库管理器"""
    print("=" * 70)
    print("因子库管理器测试")
    print("=" * 70)
    
    # 创建管理器
    manager = FactorLibraryManager()
    
    # 模拟质量报告
    mock_quality_report = {
        'ic_metrics': {
            'ic_mean': 0.05,
            'icir_annual': 1.2,
            'pass_ic': True
        },
        'psi': 0.15,
        'ic_half_life': 8.5,
        'corr_check': {
            'max_corr': 0.45
        },
        'overall_pass': True
    }
    
    # 添加因子
    manager.add_factor(
        factor_name='roc_20d',
        quality_report=mock_quality_report,
        formula='(close_t - close_{t-20}) / close_{t-20}',
        family='动量/反转',
        reference='Jegadeesh and Titman (1993)'
    )
    
    manager.add_factor(
        factor_name='realized_vol_20d',
        quality_report=mock_quality_report,
        formula='std(returns, 20)',
        family='波动率',
        reference='French, Schwert and Stambaugh (1987)'
    )
    
    # 列出因子
    print(f"\n📋 当前因子清单:")
    for factor in manager.list_factors():
        print(f"   - {factor}")
    
    # 生成报告
    report_df = manager.generate_factor_report()
    print(f"\n📊 因子报告:")
    print(report_df)
    
    # 族别分析
    family_df = manager.analyze_factor_family_performance()
    print(f"\n📈 族别表现:")
    print(family_df)
    
    print("\n✅ 测试完成！")
