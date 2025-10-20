#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志设置模块
"""

import logging
import sys
from datetime import datetime


def setup_logger(name: str = 'ml_baseline',
                level: str = 'INFO',
                log_file: str = None) -> logging.Logger:
    """
    设置日志器
    
    Parameters:
    -----------
    name : str
        日志器名称
    level : str
        日志级别
    log_file : str, optional
        日志文件路径
        
    Returns:
    --------
    logging.Logger
        配置好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除已有的处理器
    logger.handlers = []
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger
