#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
漂移检测与分割对比模块

功能：
1. 比较 Train / Valid / Test 集的 IC 和 Spread
2. 检测漂移（验证 vs 测试差异 < 20%）
3. 生成漂移报告（JSON + HTML）
4. 与 CrossSectionAnalyzer 无缝集成

验收标准（来自研究宪章）：
- 验证 vs 测试：Rank IC、ICIR、分层收益差异 < 20%
- 统计检验与图形化（分布、时序）

输出目录：
/ML output/reports/baseline_vX/cv/
├── drift_report.json        # 漂移检测结果
├── drift_tearsheet.html     # 可视化报告
└── split_comparison.csv     # 分割对比详情

创建: 2025-12-02 | 版本: v1.0
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
if ml_root not in sys.path:
    sys.path.insert(0, ml_root)


class DriftDetector:
    """
    漂移检测器
    
    比较不同数据集（Train/Valid/Test）的因子表现差异
    """
    
    def __init__(self, 
                 drift_threshold: float = 0.2,
                 significance_level: float = 0.05):
        """
        初始化漂移检测器
        
        Parameters:
        -----------
        drift_threshold : float
            漂移阈值（默认 0.2，即 20%）
        significance_level : float
            统计显著性水平（默认 0.05）
        """
        self.drift_threshold = drift_threshold
        self.significance_level = significance_level
        
        print(f"🔍 漂移检测器初始化")
        print(f"   漂移阈值: {drift_threshold:.0%}")
        print(f"   显著性水平: {significance_level}")
    
    def compare_ic_summaries(self,
                            train_summary: Dict,
                            valid_summary: Dict,
                            test_summary: Dict) -> Dict:
        """
        比较 IC 汇总统计
        
        Parameters:
        -----------
        train_summary, valid_summary, test_summary : Dict
            IC 汇总统计（来自 calculate_ic_summary）
            
        Returns:
        --------
        Dict
            比较结果
        """
        comparison = {}
        
        metrics = ['mean', 'std', 'icir', 'icir_annual', 'positive_ratio']
        
        for metric in metrics:
            train_val = train_summary.get(metric, np.nan)
            valid_val = valid_summary.get(metric, np.nan)
            test_val = test_summary.get(metric, np.nan)
            
            # 计算相对差异
            if train_val != 0 and not np.isnan(train_val):
                valid_vs_train = abs(valid_val - train_val) / abs(train_val)
                test_vs_train = abs(test_val - train_val) / abs(train_val)
                valid_vs_test = abs(valid_val - test_val) / abs(train_val)
            else:
                valid_vs_train = np.nan
                test_vs_train = np.nan
                valid_vs_test = np.nan
            
            comparison[metric] = {
                'train': train_val,
                'valid': valid_val,
                'test': test_val,
                'valid_vs_train_pct': valid_vs_train,
                'test_vs_train_pct': test_vs_train,
                'valid_vs_test_pct': valid_vs_test,
                'drift_detected': valid_vs_test > self.drift_threshold if not np.isnan(valid_vs_test) else None
            }
        
        return comparison
    
    def compare_spreads(self,
                       train_spread: pd.Series,
                       valid_spread: pd.Series,
                       test_spread: pd.Series) -> Dict:
        """
        比较 Spread
        
        Parameters:
        -----------
        train_spread, valid_spread, test_spread : pd.Series
            Spread 时间序列
            
        Returns:
        --------
        Dict
            比较结果
        """
        def calc_stats(s):
            s = s.dropna()
            if len(s) == 0:
                return {'mean': np.nan, 'std': np.nan, 'sharpe': np.nan, 'positive_ratio': np.nan}
            return {
                'mean': s.mean(),
                'std': s.std(),
                'sharpe': s.mean() / s.std() if s.std() > 0 else np.nan,
                'positive_ratio': (s > 0).mean()
            }
        
        train_stats = calc_stats(train_spread)
        valid_stats = calc_stats(valid_spread)
        test_stats = calc_stats(test_spread)
        
        comparison = {}
        
        for metric in ['mean', 'sharpe', 'positive_ratio']:
            train_val = train_stats[metric]
            valid_val = valid_stats[metric]
            test_val = test_stats[metric]
            
            if train_val != 0 and not np.isnan(train_val):
                valid_vs_test = abs(valid_val - test_val) / abs(train_val)
            else:
                valid_vs_test = np.nan
            
            comparison[f'spread_{metric}'] = {
                'train': train_val,
                'valid': valid_val,
                'test': test_val,
                'valid_vs_test_pct': valid_vs_test,
                'drift_detected': valid_vs_test > self.drift_threshold if not np.isnan(valid_vs_test) else None
            }
        
        return comparison
    
    def statistical_test_ic(self,
                           ic_series_1: pd.Series,
                           ic_series_2: pd.Series,
                           test_type: str = 'mannwhitneyu') -> Dict:
        """
        IC 分布统计检验
        
        Parameters:
        -----------
        ic_series_1, ic_series_2 : pd.Series
            两个 IC 序列
        test_type : str
            检验类型：'ttest', 'mannwhitneyu', 'ks'
            
        Returns:
        --------
        Dict
            检验结果
        """
        s1 = ic_series_1.dropna()
        s2 = ic_series_2.dropna()
        
        if len(s1) < 5 or len(s2) < 5:
            return {'test': test_type, 'statistic': np.nan, 'p_value': np.nan, 'significant': None}
        
        if test_type == 'ttest':
            stat, p = stats.ttest_ind(s1, s2)
        elif test_type == 'mannwhitneyu':
            stat, p = stats.mannwhitneyu(s1, s2, alternative='two-sided')
        elif test_type == 'ks':
            stat, p = stats.ks_2samp(s1, s2)
        else:
            raise ValueError(f"不支持的检验类型: {test_type}")
        
        return {
            'test': test_type,
            'statistic': stat,
            'p_value': p,
            'significant': p < self.significance_level
        }
    
    def detect_drift(self,
                    train_results: Dict,
                    valid_results: Dict,
                    test_results: Dict,
                    factor_name: str = 'factor',
                    period: str = '5d') -> Dict:
        """
        综合漂移检测
        
        Parameters:
        -----------
        train_results, valid_results, test_results : Dict
            CrossSectionAnalyzer 的分析结果
        factor_name : str
            因子名称
        period : str
            收益周期
            
        Returns:
        --------
        Dict
            漂移检测报告
        """
        print(f"\n🔍 漂移检测: {factor_name} @ {period}")
        
        report = {
            'factor': factor_name,
            'period': period,
            'timestamp': datetime.now().isoformat(),
            'threshold': self.drift_threshold,
            'checks': {},
            'overall_pass': True
        }
        
        # 1. IC 比较
        key = (factor_name, f'ret_{period}')
        
        train_ic_summary = train_results.get('ic_summary', {}).get(key, {})
        valid_ic_summary = valid_results.get('ic_summary', {}).get(key, {})
        test_ic_summary = test_results.get('ic_summary', {}).get(key, {})
        
        ic_comparison = self.compare_ic_summaries(train_ic_summary, valid_ic_summary, test_ic_summary)
        report['checks']['ic_comparison'] = ic_comparison
        
        # 检查 IC 漂移
        ic_drift = ic_comparison.get('mean', {}).get('drift_detected', False)
        icir_drift = ic_comparison.get('icir', {}).get('drift_detected', False)
        
        if ic_drift:
            print(f"   ⚠️  IC 均值漂移: Valid vs Test > {self.drift_threshold:.0%}")
            report['overall_pass'] = False
        
        if icir_drift:
            print(f"   ⚠️  ICIR 漂移: Valid vs Test > {self.drift_threshold:.0%}")
            report['overall_pass'] = False
        
        # 2. Spread 比较
        train_spread = train_results.get('spreads', {}).get(key, pd.Series())
        valid_spread = valid_results.get('spreads', {}).get(key, pd.Series())
        test_spread = test_results.get('spreads', {}).get(key, pd.Series())
        
        spread_comparison = self.compare_spreads(train_spread, valid_spread, test_spread)
        report['checks']['spread_comparison'] = spread_comparison
        
        # 检查 Spread 漂移
        spread_drift = spread_comparison.get('spread_mean', {}).get('drift_detected', False)
        
        if spread_drift:
            print(f"   ⚠️  Spread 均值漂移: Valid vs Test > {self.drift_threshold:.0%}")
            report['overall_pass'] = False
        
        # 3. 统计检验
        train_ic_series = train_results.get('daily_ic', pd.DataFrame())
        valid_ic_series = valid_results.get('daily_ic', pd.DataFrame())
        test_ic_series = test_results.get('daily_ic', pd.DataFrame())
        
        if key in train_ic_series.columns and key in valid_ic_series.columns and key in test_ic_series.columns:
            valid_test_test = self.statistical_test_ic(
                valid_ic_series[key], 
                test_ic_series[key],
                test_type='ks'
            )
            report['checks']['statistical_test'] = valid_test_test
            
            if valid_test_test.get('significant', False):
                print(f"   ⚠️  IC 分布显著不同 (KS p={valid_test_test['p_value']:.4f})")
        
        # 总结
        if report['overall_pass']:
            print(f"   ✅ 漂移检测通过")
        else:
            print(f"   ❌ 漂移检测未通过")
        
        return report
    
    def generate_drift_report(self,
                             drift_reports: List[Dict],
                             output_dir: str) -> str:
        """
        生成漂移报告
        
        Parameters:
        -----------
        drift_reports : List[Dict]
            漂移检测结果列表
        output_dir : str
            输出目录
            
        Returns:
        --------
        str
            报告文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存 JSON 报告
        json_path = os.path.join(output_dir, 'drift_report.json')
        
        # 转换 numpy 类型为 Python 原生类型
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(i) for i in obj]
            return obj
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(convert_to_native(drift_reports), f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 漂移报告已保存: {json_path}")
        
        # 2. 生成汇总 CSV
        summary_rows = []
        for report in drift_reports:
            row = {
                'factor': report['factor'],
                'period': report['period'],
                'overall_pass': report['overall_pass']
            }
            
            # IC 比较
            ic_comp = report.get('checks', {}).get('ic_comparison', {})
            for metric in ['mean', 'icir']:
                m = ic_comp.get(metric, {})
                row[f'ic_{metric}_train'] = m.get('train')
                row[f'ic_{metric}_valid'] = m.get('valid')
                row[f'ic_{metric}_test'] = m.get('test')
                row[f'ic_{metric}_drift_pct'] = m.get('valid_vs_test_pct')
            
            # Spread 比较
            spread_comp = report.get('checks', {}).get('spread_comparison', {})
            for metric in ['spread_mean', 'spread_sharpe']:
                m = spread_comp.get(metric, {})
                row[f'{metric}_train'] = m.get('train')
                row[f'{metric}_valid'] = m.get('valid')
                row[f'{metric}_test'] = m.get('test')
                row[f'{metric}_drift_pct'] = m.get('valid_vs_test_pct')
            
            summary_rows.append(row)
        
        csv_path = os.path.join(output_dir, 'split_comparison.csv')
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
        print(f"📊 分割对比已保存: {csv_path}")
        
        # 3. 生成 HTML 报告
        html_path = self._generate_html_report(drift_reports, output_dir)
        
        return json_path
    
    def _generate_html_report(self, drift_reports: List[Dict], output_dir: str) -> str:
        """生成 HTML 漂移报告"""
        html_path = os.path.join(output_dir, 'drift_tearsheet.html')
        
        # 构建 HTML
        html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>漂移检测报告</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #007bff; color: white; }
        tr:hover { background: #f1f1f1; }
        .pass { color: #28a745; font-weight: bold; }
        .fail { color: #dc3545; font-weight: bold; }
        .warning { color: #ffc107; font-weight: bold; }
        .metric-card { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .summary { display: flex; gap: 20px; flex-wrap: wrap; }
        .summary-item { flex: 1; min-width: 200px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }
        .summary-item h3 { margin: 0; font-size: 2em; }
        .summary-item p { margin: 5px 0 0; opacity: 0.9; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 漂移检测报告</h1>
        <p>生成时间: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        <p>漂移阈值: """ + f"{self.drift_threshold:.0%}" + """</p>
        
        <div class="summary">
            <div class="summary-item">
                <h3>""" + str(len(drift_reports)) + """</h3>
                <p>检测因子数</p>
            </div>
            <div class="summary-item">
                <h3>""" + str(sum(1 for r in drift_reports if r['overall_pass'])) + """</h3>
                <p>通过数</p>
            </div>
            <div class="summary-item">
                <h3>""" + str(sum(1 for r in drift_reports if not r['overall_pass'])) + """</h3>
                <p>未通过数</p>
            </div>
        </div>
        
        <h2>详细结果</h2>
        <table>
            <tr>
                <th>因子</th>
                <th>周期</th>
                <th>IC均值 (Train/Valid/Test)</th>
                <th>ICIR (Train/Valid/Test)</th>
                <th>Valid vs Test 差异</th>
                <th>状态</th>
            </tr>
"""
        
        for report in drift_reports:
            ic_comp = report.get('checks', {}).get('ic_comparison', {})
            ic_mean = ic_comp.get('mean', {})
            icir = ic_comp.get('icir', {})
            
            status_class = 'pass' if report['overall_pass'] else 'fail'
            status_text = '✅ 通过' if report['overall_pass'] else '❌ 未通过'
            
            drift_pct = ic_mean.get('valid_vs_test_pct', 0)
            drift_text = f"{drift_pct*100:.1f}%" if drift_pct and not np.isnan(drift_pct) else 'N/A'
            
            html += f"""
            <tr>
                <td>{report['factor']}</td>
                <td>{report['period']}</td>
                <td>{ic_mean.get('train', 'N/A'):.4f} / {ic_mean.get('valid', 'N/A'):.4f} / {ic_mean.get('test', 'N/A'):.4f}</td>
                <td>{icir.get('train', 'N/A'):.4f} / {icir.get('valid', 'N/A'):.4f} / {icir.get('test', 'N/A'):.4f}</td>
                <td>{drift_text}</td>
                <td class="{status_class}">{status_text}</td>
            </tr>
"""
        
        html += """
        </table>
        
        <h2>说明</h2>
        <div class="metric-card">
            <p><strong>漂移检测标准：</strong></p>
            <ul>
                <li>Valid vs Test 的 IC/ICIR/Spread 差异 < 20%</li>
                <li>IC 分布统计检验无显著差异 (p > 0.05)</li>
            </ul>
            <p><strong>红线标准（触发回滚）：</strong></p>
            <ul>
                <li>测试集 Spread ≤ 0</li>
                <li>ICIR 显著回落 > 50%</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"📊 HTML 报告已保存: {html_path}")
        
        return html_path


def compare_splits_with_analyzer(
    factors: pd.DataFrame,
    forward_returns: pd.DataFrame,
    train_idx: pd.Index,
    valid_idx: pd.Index,
    test_idx: pd.Index,
    output_dir: str,
    drift_threshold: float = 0.2
) -> Dict:
    """
    便捷函数：使用 CrossSectionAnalyzer 分析各个分割并比较
    
    Parameters:
    -----------
    factors : pd.DataFrame
        因子数据，MultiIndex [date, ticker]
    forward_returns : pd.DataFrame
        远期收益率
    train_idx, valid_idx, test_idx : pd.Index
        分割索引
    output_dir : str
        输出目录
    drift_threshold : float
        漂移阈值
        
    Returns:
    --------
    Dict
        漂移检测结果
    """
    from evaluation.cross_section_analyzer import CrossSectionAnalyzer
    
    # 分别创建各分割的分析器
    print("\n" + "=" * 70)
    print("分割对比分析")
    print("=" * 70)
    
    # Train
    print("\n📊 分析 Train 集...")
    train_analyzer = CrossSectionAnalyzer(
        factors=factors.loc[train_idx],
        forward_returns=forward_returns.loc[train_idx]
    )
    train_analyzer.analyze()
    train_results = train_analyzer.results
    
    # Valid
    print("\n📊 分析 Valid 集...")
    valid_analyzer = CrossSectionAnalyzer(
        factors=factors.loc[valid_idx],
        forward_returns=forward_returns.loc[valid_idx]
    )
    valid_analyzer.analyze()
    valid_results = valid_analyzer.results
    
    # Test
    print("\n📊 分析 Test 集...")
    test_analyzer = CrossSectionAnalyzer(
        factors=factors.loc[test_idx],
        forward_returns=forward_returns.loc[test_idx]
    )
    test_analyzer.analyze()
    test_results = test_analyzer.results
    
    # 漂移检测
    detector = DriftDetector(drift_threshold=drift_threshold)
    
    drift_reports = []
    for factor_col in factors.columns:
        for ret_col in forward_returns.columns:
            period = ret_col.replace('ret_', '')
            report = detector.detect_drift(
                train_results, valid_results, test_results,
                factor_name=factor_col,
                period=period
            )
            drift_reports.append(report)
    
    # 生成报告
    detector.generate_drift_report(drift_reports, output_dir)
    
    return {
        'train_results': train_results,
        'valid_results': valid_results,
        'test_results': test_results,
        'drift_reports': drift_reports
    }


if __name__ == '__main__':
    """测试代码"""
    print("=" * 70)
    print("漂移检测模块测试")
    print("=" * 70)
    
    # 生成测试数据
    np.random.seed(42)
    
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    dates = dates[dates.dayofweek < 5]
    tickers = ['000001', '000002', '000003', '000004', '000005']
    
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
    
    # 模拟因子和收益
    factors = pd.DataFrame({
        'factor_1': np.random.randn(len(index))
    }, index=index)
    
    forward_returns = pd.DataFrame({
        'ret_5d': np.random.randn(len(index)) * 0.05
    }, index=index)
    
    # 模拟分割
    n = len(dates)
    train_end = int(n * 0.6)
    valid_end = int(n * 0.8)
    
    train_dates = dates[:train_end]
    valid_dates = dates[train_end:valid_end]
    test_dates = dates[valid_end:]
    
    train_idx = factors.index[factors.index.get_level_values('date').isin(train_dates)]
    valid_idx = factors.index[factors.index.get_level_values('date').isin(valid_dates)]
    test_idx = factors.index[factors.index.get_level_values('date').isin(test_dates)]
    
    print(f"\n📊 测试数据:")
    print(f"   Train: {len(train_idx)} 样本")
    print(f"   Valid: {len(valid_idx)} 样本")
    print(f"   Test: {len(test_idx)} 样本")
    
    # 测试漂移检测
    output_dir = os.path.join(ml_root, 'ML output', 'reports', 'test_drift')
    
    try:
        results = compare_splits_with_analyzer(
            factors, forward_returns,
            train_idx, valid_idx, test_idx,
            output_dir
        )
        print("\n✅ 漂移检测测试完成！")
    except Exception as e:
        print(f"\n⚠️  测试失败: {e}")
        import traceback
        traceback.print_exc()
