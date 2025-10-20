#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成模块
"""

import os
import json
import pandas as pd
from typing import Dict, List
from datetime import datetime


def generate_report(results: Dict,
                   output_dir: str,
                   bucket_performance_file: str = "model_bucket_performance.csv",
                   predictions_file: str = "test_predictions.csv",
                   summary_file: str = "summary.json"):
    """
    生成评估报告
    
    Parameters:
    -----------
    results : dict
        评估结果字典
    output_dir : str
        输出目录
    bucket_performance_file : str
        分桶表现文件名
    predictions_file : str
        预测明细文件名
    summary_file : str
        摘要文件名
    """
    print("📝 生成评估报告...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存分桶表现
    if 'bucket_performance' in results:
        bucket_path = os.path.join(output_dir, bucket_performance_file)
        results['bucket_performance'].to_csv(bucket_path, index=False, encoding='utf-8-sig')
        print(f"   ✅ 分桶表现已保存: {bucket_path}")
    
    # 2. 保存预测明细
    if 'predictions' in results:
        pred_path = os.path.join(output_dir, predictions_file)
        results['predictions'].to_csv(pred_path, encoding='utf-8-sig')
        print(f"   ✅ 预测明细已保存: {pred_path}")
    
    # 3. 保存摘要
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models': list(results.get('model_metrics', {}).keys()),
        'metrics': results.get('model_metrics', {}),
        'validation': results.get('validation', {})
    }
    
    summary_path = os.path.join(output_dir, summary_file)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"   ✅ 摘要已保存: {summary_path}")
    
    # 4. 生成可读的文本报告
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("机器学习模型评估报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"生成时间: {summary['timestamp']}\n\n")
        
        # 模型指标
        if 'model_metrics' in results:
            f.write("## 模型评估指标\n\n")
            for model_name, metrics in results['model_metrics'].items():
                f.write(f"### {model_name}\n")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"  - {metric}: {value:.6f}\n")
                    else:
                        f.write(f"  - {metric}: {value}\n")
                f.write("\n")
        
        # 验收结果
        if 'validation' in results:
            f.write("## 验收结果\n\n")
            validation = results['validation']
            f.write(f"  ✅ Top桶 > 全体均值: {validation.get('top_vs_mean', False)}\n")
            f.write(f"  ✅ Spread > 0: {validation.get('spread_positive', False)}\n")
            f.write(f"  📊 Top桶平均收益: {validation.get('top_mean', 0):.6f}\n")
            f.write(f"  📊 全体平均收益: {validation.get('overall_mean', 0):.6f}\n")
            f.write(f"  📈 Top-Bottom Spread: {validation.get('spread', 0):.6f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"   ✅ 详细报告已保存: {report_path}")
    print(f"\n✅ 所有报告已生成到: {output_dir}")
