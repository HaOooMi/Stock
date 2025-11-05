#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场数据加载器

功能：
1. 从 InfluxDB 加载原始市场数据（OHLCV、换手率等）
2. 从 MySQL 加载股票元数据（上市时间、ST状态等）
3. 为交易可行性过滤提供必要的字段
4. 复用 get_stock_info 中已有的查询函数
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Optional, List
from datetime import datetime
from influxdb_client import InfluxDBClient
from influxdb_client.client.query_api import QueryApi

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
project_root = os.path.dirname(ml_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 尝试导入 get_stock_info 中的工具函数
try:
    sys.path.insert(0, os.path.join(project_root, 'get_stock_info'))
    from utils import get_influxdb_client, get_mysql_engine
    from stock_market_data_akshare import get_history_data
    from stock_meta_akshare import get_basic_info_mysql
    HAVE_GET_STOCK_INFO = True
except ImportError:
    HAVE_GET_STOCK_INFO = False
    print("⚠️  无法导入 get_stock_info 模块")


class MarketDataLoader:
    """
    市场数据加载器
    
    从 InfluxDB 加载原始市场数据（OHLCV、换手率等）
    从 MySQL 加载股票元数据（上市时间、ST状态等）
    复用 get_stock_info 中的现有代码
    """
    
    def __init__(self,
                 url: str = "http://localhost:8086",
                 token: str = None,
                 org: str = "stock",
                 bucket: str = "stock_kdata"):
        """
        初始化 InfluxDB 连接
        
        Parameters:
        -----------
        url : str
            InfluxDB 服务地址
        token : str
            访问令牌（必须从配置文件传入）
        org : str
            组织名称
        bucket : str
            bucket 名称
        """
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        
        if not self.token:
            raise ValueError("token 是必需的，请在配置文件中设置 data.influxdb.token")
        
        # 优先使用 get_stock_info 中的客户端创建方法
        if HAVE_GET_STOCK_INFO:
            try:
                self.client = get_influxdb_client()
                if self.client:
                    self.query_api = self.client.query_api()
                    print(f"✅ InfluxDB 连接成功 (使用 get_stock_info 配置)")
                else:
                    self.client = None
                    self.query_api = None
            except Exception as e:
                print(f"⚠️  InfluxDB 连接失败: {e}")
                self.client = None
                self.query_api = None
        else:
            # 备用方案：直接创建客户端
            try:
                self.client = InfluxDBClient(url=url, token=self.token, org=org)
                self.query_api = self.client.query_api()
                print(f"✅ InfluxDB 连接成功: {url}")
            except Exception as e:
                print(f"⚠️  InfluxDB 连接失败: {e}")
                self.client = None
                self.query_api = None
    
    def load_market_data(self,
                        symbol: str,
                        start_date: str,
                        end_date: str) -> pd.DataFrame:
        """
        从 InfluxDB 加载历史市场数据
        
        Parameters:
        -----------
        symbol : str
            股票代码
        start_date : str
            开始日期 (YYYY-MM-DD)
        end_date : str
            结束日期 (YYYY-MM-DD)
            
        Returns:
        --------
        pd.DataFrame
            包含 OHLCV 和其他市场数据的 DataFrame
            索引为日期，列包括：open, high, low, close, volume, amount, 
            pct_change, turnover, amplitude 等
        """
        if self.query_api is None:
            raise RuntimeError("InfluxDB 未连接")
        
        print(f"📊 从 InfluxDB 加载市场数据: {symbol}")
        print(f"   时间范围: {start_date} ~ {end_date}")
        
        # 优先使用 get_stock_info 中的查询函数
        if HAVE_GET_STOCK_INFO:
            try:
                df = get_history_data(self.query_api, symbol, start_date, end_date)
                
                if df.empty:
                    print(f"   ⚠️  未找到数据")
                    return pd.DataFrame()
                
                # 重命名列为英文（保持与你的代码一致）
                column_map = {
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '振幅': 'amplitude',
                    '涨跌幅': 'pct_change',
                    '涨跌额': 'change',
                    '换手率': 'turnover',
                    '是否停牌': 'is_suspended'
                }
                
                # 重命名存在的列
                existing_cols = {k: v for k, v in column_map.items() if k in df.columns}
                df = df.rename(columns=existing_cols)
                
                # 删除 InfluxDB 的元数据列
                meta_cols = ['result', 'table', '_start', '_stop', '_measurement', '股票代码', '股票名称']
                df = df.drop(columns=[col for col in meta_cols if col in df.columns], errors='ignore')
                
                # 处理时间格式
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                
                # 排序
                df = df.sort_index()
                
                # 添加元数据字段（从MySQL获取）
                df = self._enrich_with_metadata(df, symbol)
                
                print(f"   ✅ 加载完成: {len(df)} 条记录")
                
                return df
                
            except Exception as e:
                print(f"   ❌ 加载失败: {e}")
                return pd.DataFrame()
        else:
            # 备用方案：直接查询
            return self._load_market_data_direct(symbol, start_date, end_date)
    
    def _load_market_data_direct(self,
                                 symbol: str,
                                 start_date: str,
                                 end_date: str) -> pd.DataFrame:
        """
        直接从 InfluxDB 查询市场数据（备用方案）
        """
        flux_query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: {start_date}T00:00:00Z, stop: {end_date}T23:59:59Z)
              |> filter(fn: (r) => r._measurement == "history_kdata")
              |> filter(fn: (r) => r.股票代码 == "{symbol}")
              |> pivot(
                  rowKey:["_time"],
                  columnKey: ["_field"],
                  valueColumn: "_value"
              )
              |> keep(columns: ["_time", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率", "是否停牌"])
        '''
        
        try:
            df = self.query_api.query_data_frame(query=flux_query)
            
            if df.empty:
                return pd.DataFrame()
            
            # 重命名列
            column_map = {
                '_time': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change',
                '换手率': 'turnover',
                '是否停牌': 'is_suspended'
            }
            
            df = df.rename(columns=column_map)
            
            # 删除不需要的列
            cols_to_keep = ['date', 'open', 'high', 'low', 'close', 'volume', 
                          'amount', 'pct_change', 'turnover', 'amplitude']
            df = df[[col for col in cols_to_keep if col in df.columns]]
            
            # 处理时间格式
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            print(f"   ❌ 查询失败: {e}")
            return pd.DataFrame()
    
    def _enrich_with_metadata(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        从MySQL添加元数据字段（ST状态、上市时间、总股本等）
        
        Parameters:
        -----------
        df : pd.DataFrame
            市场数据
        symbol : str
            股票代码
            
        Returns:
        --------
        pd.DataFrame
            添加了元数据的市场数据
        """
        if not HAVE_GET_STOCK_INFO:
            return df
        
        try:
            engine = get_mysql_engine()
            with engine.connect() as conn:
                info_dict = get_basic_info_mysql(conn, (symbol,))
                if symbol in info_dict:
                    stock_info = info_dict[symbol]
                    
                    # 添加股票名称
                    df['name'] = stock_info.get('股票简称', '')
                    
                    # 添加ST标志
                    stock_name = stock_info.get('股票简称', '')
                    df['st_flag'] = 1 if ('ST' in stock_name or 'st' in stock_name) else 0
                    
                    # 添加上市日期
                    list_date = stock_info.get('上市时间')
                    if list_date:
                        df['list_date'] = pd.to_datetime(list_date)
                    
                    # 添加总股本（用于计算换手率）
                    shares_outstanding = stock_info.get('总股本')
                    if shares_outstanding:
                        df['shares_outstanding'] = float(shares_outstanding)
                        
                        # 计算换手率（如果还没有）
                        if 'turnover_rate' not in df.columns and 'volume' in df.columns:
                            df['turnover_rate'] = df['volume'] / df['shares_outstanding']
                    
                    print(f"   ✓ 元数据添加完成")
                    
        except Exception as e:
            print(f"   ⚠️  添加元数据失败: {e}")
        
        return df
    
    def is_st_stock(self, symbol: str) -> bool:
        """
        判断是否为 ST 股票
        
        通过 MySQL 查询股票名称判断
        
        Parameters:
        -----------
        symbol : str
            股票代码
            
        Returns:
        --------
        bool
            是否为 ST 股票
        """
        if not HAVE_GET_STOCK_INFO:
            return False
        
        try:
            engine = get_mysql_engine()
            with engine.connect() as conn:
                info_dict = get_basic_info_mysql(conn, (symbol,))
                if symbol in info_dict:
                    stock_name = info_dict[symbol].get('股票简称', '')
                    # 判断股票名称是否包含 ST
                    return 'ST' in stock_name or 'st' in stock_name
        except Exception as e:
            print(f"   ⚠️  查询 ST 信息失败: {e}")
        
        return False
    
    def get_suspend_info(self,
                        symbol: str,
                        start_date: str,
                        end_date: str) -> pd.DataFrame:
        """
        获取停牌信息
        
        Parameters:
        -----------
        symbol : str
            股票代码
        start_date : str
            开始日期
        end_date : str
            结束日期
            
        Returns:
        --------
        pd.DataFrame
            包含停牌信息的 DataFrame
            索引为日期，列为 is_suspended (bool)
        """
        df = self.load_market_data(symbol, start_date, end_date)
        
        if df.empty:
            return pd.DataFrame()
        
        # 判断停牌：成交量为 0 或 NaN
        df['is_suspended'] = (df['volume'] == 0) | df['volume'].isna()
        
        return df[['is_suspended']]
    
    def get_listing_date(self, symbol: str) -> Optional[str]:
        """
        获取股票上市日期
        
        从 MySQL stock_individual_info 表查询
        
        Parameters:
        -----------
        symbol : str
            股票代码
            
        Returns:
        --------
        str or None
            上市日期 (YYYY-MM-DD)
        """
        if not HAVE_GET_STOCK_INFO:
            return None
        
        try:
            engine = get_mysql_engine()
            with engine.connect() as conn:
                info_dict = get_basic_info_mysql(conn, (symbol,))
                if symbol in info_dict:
                    listing_date = info_dict[symbol].get('上市时间')
                    if listing_date:
                        return str(listing_date)
        except Exception as e:
            print(f"   ⚠️  查询上市时间失败: {e}")
        
        return None
    
    def close(self):
        """关闭 InfluxDB 连接"""
        if self.client:
            self.client.close()
            print("✅ InfluxDB 连接已关闭")


if __name__ == "__main__":
    """
    使用示例
    """
    print("📊 市场数据加载器测试")
    print("=" * 50)
    
    try:
        # 初始化加载器
        loader = MarketDataLoader()
        
        if loader.query_api is None:
            print("❌ InfluxDB 未连接，请先启动 InfluxDB")
            print("   启动方法: cd 'C:\\Program Files\\InfluxData'; .\\influxd")
            exit(1)
        
        # 加载市场数据
        symbol = "000001"
        start_date = "2024-01-01"
        end_date = "2024-12-31"
        
        df = loader.load_market_data(symbol, start_date, end_date)
        
        if not df.empty:
            print(f"\n✅ 数据加载成功:")
            print(f"   形状: {df.shape}")
            print(f"   列: {df.columns.tolist()}")
            print(f"\n前5行:")
            print(df.head())
            
            # 检查数据质量
            print(f"\n数据质量:")
            print(f"   缺失值: {df.isna().sum().sum()}")
            print(f"   日期范围: {df.index.min()} ~ {df.index.max()}")
            
            # 获取停牌信息
            suspend_df = loader.get_suspend_info(symbol, start_date, end_date)
            if not suspend_df.empty:
                print(f"\n✅ 停牌信息:")
                print(f"   停牌天数: {suspend_df['is_suspended'].sum()}")
        else:
            print(f"\n⚠️  未找到数据，请检查:")
            print(f"   1. InfluxDB 是否正在运行")
            print(f"   2. 是否已导入股票 {symbol} 的数据")
            print(f"   3. 时间范围 {start_date} ~ {end_date} 是否有数据")
        
        # 关闭连接
        loader.close()
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
