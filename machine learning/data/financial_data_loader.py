#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
财务数据加载器

功能：
1. 从 MySQL 加载财务数据
2. PIT 对齐（按报告期和公告日）
3. 财务指标计算和特征工程
4. 复用 get_stock_info 中的查询函数
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
project_root = os.path.dirname(ml_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 尝试导入 get_stock_info 中的工具函数
try:
    sys.path.insert(0, os.path.join(project_root, 'get_stock_info'))
    from utils import get_mysql_engine
    from stock_meta_akshare import get_financial_info_mysql
    HAVE_GET_STOCK_INFO = True
except ImportError:
    HAVE_GET_STOCK_INFO = False
    print("⚠️  无法导入 get_stock_info 模块")


class FinancialDataLoader:
    """
    财务数据加载器
    
    从 MySQL 加载财务报表数据，并进行 PIT 对齐
    """
    
    def __init__(self, announce_lag_days: int = 90):
        """
        初始化财务数据加载器
        
        Parameters:
        -----------
        announce_lag_days : int
            公告日后的滞后天数（默认90天），用于 PIT 对齐
        """
        self.announce_lag_days = announce_lag_days
        
        if not HAVE_GET_STOCK_INFO:
            raise ImportError("无法导入 get_stock_info 模块，请检查路径")
        
        print(f"📊 财务数据加载器初始化")
        print(f"   公告滞后天数: {announce_lag_days}")
    
    def load_financial_data(self,
                           symbol: str,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        从 MySQL 加载财务数据
        
        Parameters:
        -----------
        symbol : str
            股票代码
        start_date : str, optional
            开始日期 (YYYY-MM-DD)
        end_date : str, optional
            结束日期 (YYYY-MM-DD)
            
        Returns:
        --------
        pd.DataFrame
            财务数据，包含报告期、公告日、生效日等
            列包括：report_date, announce_date, effective_date, 各种财务指标
        """
        print(f"📊 从 MySQL 加载财务数据: {symbol}")
        
        try:
            engine = get_mysql_engine()
            with engine.connect() as conn:
                df = get_financial_info_mysql(conn, symbol)
                
                if df.empty:
                    print(f"   ⚠️  未找到财务数据")
                    return pd.DataFrame()
                
                # 重命名列为英文
                df = self._rename_columns(df)
                
                # 确保日期格式
                df['report_date'] = pd.to_datetime(df['report_date'])
                
                # 添加公告日（假设为报告期后 45 天，实际应从数据库获取）
                # TODO: 需要在 MySQL 添加公告日字段
                df['announce_date'] = df['report_date'] + pd.Timedelta(days=45)
                
                # 计算生效日期（公告日 + 滞后天数）
                df['effective_date'] = df['announce_date'] + pd.Timedelta(days=self.announce_lag_days)
                
                # 过滤日期范围
                if start_date:
                    df = df[df['effective_date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['effective_date'] <= pd.to_datetime(end_date)]
                
                # 按报告期排序
                df = df.sort_values('report_date')
                
                print(f"   ✅ 加载完成: {len(df)} 条财务记录")
                print(f"   时间范围: {df['report_date'].min()} ~ {df['report_date'].max()}")
                
                return df
                
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
            return pd.DataFrame()
    
    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        重命名列为英文
        
        Parameters:
        -----------
        df : pd.DataFrame
            原始财务数据
            
        Returns:
        --------
        pd.DataFrame
            重命名后的数据
        """
        column_map = {
            '股票代码': 'symbol',
            '报告期': 'report_date',
            '净利润': 'net_profit',
            '净利润同比增长率': 'net_profit_yoy',
            '扣非净利润': 'net_profit_deducted',
            '扣非净利润同比增长率': 'net_profit_deducted_yoy',
            '营业总收入': 'revenue',
            '营业总收入同比增长率': 'revenue_yoy',
            '基本每股收益': 'eps',
            '每股净资产': 'bps',
            '每股资本公积金': 'capital_reserve_ps',
            '每股未分配利润': 'undistributed_profit_ps',
            '每股经营现金流': 'ocf_ps',
            '销售净利率': 'net_profit_margin',
            '净资产收益率': 'roe',
            '净资产收益率_摊薄': 'roe_diluted',
            '营业周期': 'operating_cycle',
            '应收账款周转天数': 'receivable_turnover_days',
            '流动比率': 'current_ratio',
            '速动比率': 'quick_ratio',
            '保守速动比率': 'conservative_quick_ratio',
            '产权比率': 'debt_to_equity',
            '资产负债率': 'debt_to_assets'
        }
        
        # 只重命名存在的列
        existing_cols = {k: v for k, v in column_map.items() if k in df.columns}
        df = df.rename(columns=existing_cols)
        
        return df
    
    def calculate_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算财务特征
        
        Parameters:
        -----------
        df : pd.DataFrame
            原始财务数据
            
        Returns:
        --------
        pd.DataFrame
            添加了新特征的数据
        """
        print(f"🔢 计算财务特征")
        
        df = df.copy()
        
        # 增长率相关
        if 'net_profit_yoy' in df.columns and 'revenue_yoy' in df.columns:
            # 盈利质量：净利润增长率 / 营收增长率
            df['profit_quality'] = df['net_profit_yoy'] / (df['revenue_yoy'] + 1e-6)
        
        # ROE 相关
        if 'roe' in df.columns:
            # ROE 变化
            df['roe_change'] = df['roe'].diff()
            # ROE 季度移动平均
            df['roe_ma4'] = df['roe'].rolling(window=4, min_periods=1).mean()
        
        # 杜邦分析
        if all(col in df.columns for col in ['net_profit_margin', 'debt_to_assets']):
            # 资产周转率估算 = 营收 / 资产（简化版）
            # 杠杆系数 = 1 / (1 - 资产负债率)
            df['leverage_ratio'] = 1 / (1 - df['debt_to_assets'] + 1e-6)
        
        # 现金流相关
        if 'ocf_ps' in df.columns and 'eps' in df.columns:
            # 现金流盈利质量 = 每股经营现金流 / 每股收益
            df['cash_earning_quality'] = df['ocf_ps'] / (df['eps'] + 1e-6)
        
        # 营运效率
        if 'receivable_turnover_days' in df.columns and 'operating_cycle' in df.columns:
            # 存货周转天数 = 营业周期 - 应收账款周转天数
            df['inventory_turnover_days'] = df['operating_cycle'] - df['receivable_turnover_days']
        
        print(f"   ✅ 特征计算完成")
        
        return df
    
    def align_to_trading_dates(self,
                               financial_df: pd.DataFrame,
                               trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        将财务数据对齐到交易日（前向填充）
        
        Parameters:
        -----------
        financial_df : pd.DataFrame
            财务数据（包含 effective_date）
        trading_dates : pd.DatetimeIndex
            交易日序列
            
        Returns:
        --------
        pd.DataFrame
            对齐后的财务数据（每个交易日一行）
        """
        print(f"📅 对齐财务数据到交易日")
        
        # 设置 effective_date 为索引
        df = financial_df.set_index('effective_date').sort_index()
        
        # 创建交易日 DataFrame
        aligned_df = pd.DataFrame(index=trading_dates)
        
        # 前向填充（使用最近一次公告的财务数据）
        for col in df.columns:
            if col not in ['report_date', 'announce_date', 'symbol']:
                aligned_df[col] = df[col].reindex(trading_dates, method='ffill')
        
        # 只保留有财务数据的日期（第一个财务数据生效后）
        first_date = df.index.min()
        aligned_df = aligned_df[aligned_df.index >= first_date]
        
        print(f"   ✅ 对齐完成: {len(aligned_df)} 个交易日")
        
        return aligned_df
    
    def get_latest_financial_data(self, symbol: str, as_of_date: str) -> Dict:
        """
        获取指定日期的最新财务数据（PIT 原则）
        
        Parameters:
        -----------
        symbol : str
            股票代码
        as_of_date : str
            查询日期
            
        Returns:
        --------
        dict
            最新的财务数据
        """
        df = self.load_financial_data(symbol)
        
        if df.empty:
            return {}
        
        as_of_date = pd.to_datetime(as_of_date)
        
        # 找到生效日期 <= as_of_date 的最新财务数据
        valid_data = df[df['effective_date'] <= as_of_date]
        
        if valid_data.empty:
            return {}
        
        # 返回最新一期
        latest = valid_data.sort_values('effective_date', ascending=False).iloc[0]
        
        return latest.to_dict()


if __name__ == "__main__":
    """
    使用示例
    """
    print("📊 财务数据加载器测试")
    print("=" * 50)
    
    try:
        # 初始化加载器
        loader = FinancialDataLoader(announce_lag_days=90)
        
        # 加载财务数据
        symbol = "000001"
        df = loader.load_financial_data(
            symbol=symbol,
            start_date="2022-01-01",
            end_date="2024-12-31"
        )
        
        if not df.empty:
            print(f"\n✅ 财务数据加载成功:")
            print(f"   形状: {df.shape}")
            print(f"   列: {df.columns.tolist()}")
            print(f"\n前5行:")
            print(df.head())
            
            # 计算财务特征
            df_with_features = loader.calculate_financial_features(df)
            print(f"\n✅ 特征计算完成:")
            print(f"   新增特征: {set(df_with_features.columns) - set(df.columns)}")
            
            # 获取最新财务数据
            latest = loader.get_latest_financial_data(symbol, "2024-06-30")
            if latest:
                print(f"\n✅ 最新财务数据 (截至 2024-06-30):")
                print(f"   报告期: {latest.get('report_date')}")
                print(f"   ROE: {latest.get('roe')}")
                print(f"   净利润同比: {latest.get('net_profit_yoy')}")
        else:
            print(f"\n⚠️  未找到财务数据，请先运行数据采集:")
            print(f"   python get_stock_info/main.py")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
