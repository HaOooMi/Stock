#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML 量化交易框架 - 统一入口

串联已有管道：
1. prepare_data_with_snapshot.py  - 数据准备 + 快照
2. prepare_factors.py             - 因子工程 + 评估  
3. run_baseline_pipeline.py       - 模型训练 + 横截面评估 + 回测

使用方法：
    # 完整流程
    python main.py
    
    # 快速模式（跳过漂移检测）
    python main.py --skip_drift
    
    # 仅运行特定步骤
    python main.py --only prepare_data
    python main.py --only prepare_factors
    python main.py --only train
    
    # 三条线对比
    python main.py --compare_all

创建: 2025-12-18
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

# 添加项目路径
ML_ROOT = os.path.dirname(os.path.abspath(__file__))
if ML_ROOT not in sys.path:
    sys.path.insert(0, ML_ROOT)


class MLPipeline:
    """ML 量化交易完整流程管道 - 复用已有脚本"""
    
    def __init__(self, config_path: str = "configs/ml_baseline.yml"):
        """
        初始化管道
        
        Parameters:
        -----------
        config_path : str
            配置文件路径
        """
        self.config_path = config_path
        self.pipelines_dir = os.path.join(ML_ROOT, "pipelines")
        
        print("\n" + "=" * 70)
        print("ML 量化交易框架 - Pipeline 初始化")
        print("=" * 70)
        print(f"配置文件: {config_path}")
        print(f"管道目录: {self.pipelines_dir}")
        print("=" * 70 + "\n")
    
    def _run_script(self, script_name: str, args: list = None) -> int:
        """
        运行 Python 脚本
        
        Parameters:
        -----------
        script_name : str
            脚本文件名（如 'prepare_data_with_snapshot.py'）
        args : list
            命令行参数
        
        Returns:
        --------
        int
            返回码（0 表示成功）
        """
        script_path = os.path.join(self.pipelines_dir, script_name)
        
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
        
        print(f"\n执行: {' '.join(cmd)}")
        print("-" * 70)
        
        result = subprocess.run(cmd, cwd=ML_ROOT)
        
        if result.returncode != 0:
            print(f"\n[错误] 脚本执行失败: {script_name}")
            return result.returncode
        
        print(f"\n✓ {script_name} 完成")
        return 0
    
    def step1_prepare_data(self, use_snapshot: bool = True) -> int:
        """
        步骤 1：数据准备
        
        调用: pipelines/prepare_data_with_snapshot.py
        
        Parameters:
        -----------
        use_snapshot : bool
            是否使用数据快照
        
        Returns:
        --------
        int
            返回码（0 表示成功）
        """
        print("\n" + "=" * 70)
        print("步骤 1/3：数据准备")
        print("=" * 70)
        
        script = 'prepare_data_with_snapshot.py' if use_snapshot else 'prepare_data.py'
        args = ['--config', self.config_path]
        
        return self._run_script(script, args)
    
    def step2_factor_engineering(self) -> int:
        """
        步骤 2：因子工程
        
        调用: pipelines/prepare_factors.py
        
        Returns:
        --------
        int
            返回码（0 表示成功）
        """
        print("\n" + "=" * 70)
        print("步骤 2/3：因子工程")
        print("=" * 70)
        
        args = ['--config', self.config_path]
        
        return self._run_script('prepare_factors.py', args)
    
    def step3_train_and_backtest(self, 
                                 skip_drift: bool = False,
                                 compare_all: bool = False,
                                 task_type: str = 'regression') -> int:
        """
        步骤 3：模型训练 + 横截面评估 + 回测
        
        调用: pipelines/run_baseline_pipeline.py
        
        Parameters:
        -----------
        skip_drift : bool
            是否跳过漂移检测
        compare_all : bool
            是否运行三条线对比
        task_type : str
            任务类型 (regression/regression_rank/lambdarank)
        
        Returns:
        --------
        int
            返回码（0 表示成功）
        """
        print("\n" + "=" * 70)
        print("步骤 3/3：模型训练 + 评估 + 回测")
        print("=" * 70)
        
        args = ['--config', self.config_path]
        
        if skip_drift:
            args.append('--skip_drift')
        
        if compare_all:
            args.append('--compare_all')
        else:
            args.extend(['--task_type', task_type])
        
        return self._run_script('run_baseline_pipeline.py', args)
    
    def run_full_pipeline(self, 
                         skip_drift: bool = False,
                         compare_all: bool = False,
                         use_snapshot: bool = True,
                         skip_factors: bool = False) -> int:
        """
        运行完整流程（串联已有管道）
        
        Parameters:
        -----------
        skip_drift : bool
            是否跳过漂移检测
        compare_all : bool
            是否运行三条线对比
        use_snapshot : bool
            是否使用数据快照
        skip_factors : bool
            是否跳过因子准备步骤（直接用原始数据）
        
        Returns:
        --------
        int
            返回码（0 表示成功）
        """
        print("\n" + "=" * 80)
        print(" " * 20 + "ML 量化交易框架 - 完整流程")
        print("=" * 80)
        
        start_time = datetime.now()
        
        # 步骤 1：数据准备
        ret = self.step1_prepare_data(use_snapshot=use_snapshot)
        if ret != 0:
            return ret
        
        # 步骤 2：因子工程（可选）
        if not skip_factors:
            ret = self.step2_factor_engineering()
            if ret != 0:
                return ret
        
        # 步骤 3：模型训练 + 评估 + 回测
        ret = self.step3_train_and_backtest(
            skip_drift=skip_drift,
            compare_all=compare_all
        )
        if ret != 0:
            return ret
        
        # 总结
        elapsed = datetime.now() - start_time
        print("\n" + "=" * 80)
        print(" " * 30 + "流程完成！")
        print("=" * 80)
        print(f"总耗时: {elapsed}")
        print(f"输出目录: ML output/")
        print("=" * 80 + "\n")
        
        return 0


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='ML 量化交易框架统一入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程
  python main.py
  
  # 快速模式（跳过漂移检测）
  python main.py --skip_drift
  
  # 三条线对比
  python main.py --compare_all
  
  # 仅运行数据准备
  python main.py --only prepare_data
  
  # 仅运行因子工程
  python main.py --only prepare_factors
  
  # 仅运行训练（跳过数据和因子步骤）
  python main.py --only train
        """
    )
    
    parser.add_argument('--config', type=str, default='configs/ml_baseline.yml',
                       help='配置文件路径')
    parser.add_argument('--only', type=str, 
                       choices=['prepare_data', 'prepare_factors', 'train'],
                       help='仅运行指定步骤')
    parser.add_argument('--skip_drift', action='store_true',
                       help='跳过漂移检测（加速训练）')
    parser.add_argument('--skip_factors', action='store_true',
                       help='跳过因子准备步骤')
    parser.add_argument('--compare_all', action='store_true',
                       help='运行三条线对比（regression + reg-on-rank + lambdarank）')
    parser.add_argument('--no_snapshot', action='store_true',
                       help='不使用数据快照')
    parser.add_argument('--task_type', type=str, 
                       choices=['regression', 'regression_rank', 'lambdarank'],
                       default='regression',
                       help='任务类型（不使用 --compare_all 时有效）')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 初始化管道
    pipeline = MLPipeline(config_path=args.config)
    
    # 根据参数运行
    if args.only:
        # 运行单个步骤
        if args.only == 'prepare_data':
            ret = pipeline.step1_prepare_data(use_snapshot=not args.no_snapshot)
        elif args.only == 'prepare_factors':
            ret = pipeline.step2_factor_engineering()
        elif args.only == 'train':
            ret = pipeline.step3_train_and_backtest(
                skip_drift=args.skip_drift,
                compare_all=args.compare_all,
                task_type=args.task_type
            )
        
        sys.exit(ret)
    else:
        # 运行完整流程
        ret = pipeline.run_full_pipeline(
            skip_drift=args.skip_drift,
            compare_all=args.compare_all,
            use_snapshot=not args.no_snapshot,
            skip_factors=args.skip_factors
        )
        sys.exit(ret)


if __name__ == '__main__':
    main()
