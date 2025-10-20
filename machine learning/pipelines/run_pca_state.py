#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA降维分析运行脚本

功能：
1. 加载标准化特征数据
2. 执行PCA降维
3. 保存PCA模型和元数据
4. 输出降维后的特征
"""

import os
import sys
import yaml
import argparse

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
if ml_root not in sys.path:
    sys.path.insert(0, ml_root)

# 导入PCA模块
sys.path.insert(0, os.path.join(ml_root, 'models', 'transformers'))
from pca import main as pca_main


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path: str = None):
    """
    PCA降维主流程
    
    Parameters:
    -----------
    config_path : str, optional
        配置文件路径
    """
    print("=" * 70)
    print("🔄 PCA降维分析运行")
    print("=" * 70)
    
    # 加载配置（如果提供）
    if config_path:
        config = load_config(config_path)
        print(f"\n📋 使用配置文件: {config_path}")
    else:
        print(f"\n📋 使用默认配置")
    
    # 调用PCA主函数
    print("\n🚀 开始PCA降维...")
    pca_main()
    
    print("\n" + "=" * 70)
    print("✅ PCA降维完成！")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCA降维运行脚本')
    parser.add_argument('--config', type=str, 
                       default=None,
                       help='配置文件路径（可选）')
    
    args = parser.parse_args()
    
    try:
        main(args.config)
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
