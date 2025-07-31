import time
import akshare as ak
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

import utils as u

INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "aIX6s47YmoJ-OY-rjRbLFl6AHFSYcv000g3vJp3f6l6hkbmvuj-AMtgfkjz0ESF7r536jqasqxzL9NhohGMrwA=="  
INFLUX_ORG = "stock"              
INFLUX_BUCKET = "stock_kdata"



def fetch_market_data(client,measurement_name):
    try:
        stock_list_df = ak.stock_info_a_code_name()
        print(f"成功获取 {len(stock_list_df)} 只A股股票列表。")
    except Exception as e:
        print(f"获取A股列表失败: {e}")
        return

    for index, row in stock_list_df.iterrows():
        stock_code = row["code"]
        stock_name = row["name"]
        print(f"--- 正在处理股票历史行情数据: {stock_name} ({stock_code}) ---")

        try:
            kline_df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq")
            
            if kline_df.empty:
                print(f"  -> 未找到 {stock_name} ({stock_code}) 的K线数据，跳过。")
                continue

            points = []
            for _, kline_row in kline_df.iterrows():
                p = Point(measurement_name) \
                    .tag("股票代码", stock_code) \
                    .tag("股票名称", stock_name) \
                    .field("开盘", float(kline_row["开盘"])) \
                    .field("收盘", float(kline_row["收盘"])) \
                    .field("最高", float(kline_row["最高"])) \
                    .field("最低", float(kline_row["最低"])) \
                    .field("成交量", int(kline_row["成交量"])) \
                    .field("成交额", float(kline_row["成交额"])) \
                    .field("振幅", float(kline_row["振幅"])) \
                    .field("涨跌幅", float(kline_row["涨跌幅"])) \
                    .field("涨跌额", float(kline_row["涨跌额"])) \
                    .field("换手率", float(kline_row["换手率"])) \
                    .time(pd.to_datetime(kline_row["日期"]))
                
                points.append(p)

            write_api = client.write_api(
                write_options=WriteOptions(
                    batch_size=5000,  
                    flush_interval=10_000, 
                    jitter_interval=2_000, 
                    retry_interval=5_000,  
            )
            )
            write_api.write(bucket="stock_kdata", org="stock", record=points)
            print(f"  -> 成功写入 {len(points)} 条 {stock_name} ({stock_code}) 的历史行情数据。")
            
            time.sleep(0.2)

        except Exception as e:
            print(f"  -> 处理 {stock_name} ({stock_code}) 时发生错误: {e}")
            continue