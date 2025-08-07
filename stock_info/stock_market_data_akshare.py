import time
import akshare as ak
import pandas as pd
import random
from influxdb_client import Point, WriteOptions
from influxdb_client.client.query_api import QueryApi
from typing import List
from datetime import timezone, timedelta

from utils import parse_unit_value

# 需要先用powershell运行：cd -Path 'C:\Program Files\InfluxData'
#                         ./influxd
# 打开influxdb后，在浏览器访问 http://localhost:8086



def fetch_history_market_data(client,measurement_name):
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
            kline_df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="hfq")
            
            if kline_df.empty:
                print(f"  -> 未找到 {stock_name} ({stock_code}) 的历史行情数据,跳过。")
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

        except KeyboardInterrupt:
            print("\n程序被用户中断。正在关闭...")
            break
        except Exception as e:
            print(f"  -> 处理 {stock_name} ({stock_code}) 时发生错误: {e}")
            continue


def fetch_now_market_data(client,measurement_name): 
    while True:
        DELAY_BETWEEN_BATCHES_SECONDS = (5.0, 10.0)
        delay = random.uniform(DELAY_BETWEEN_BATCHES_SECONDS[0], DELAY_BETWEEN_BATCHES_SECONDS[1])
        try:
            print(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} 正在获取所有沪深京 A 股上市公司实时行情 ---")
            realtime_df = ak.stock_zh_a_spot_em()
            if realtime_df.empty:
                print("  -> 未获取到实时行情数据，稍后重试。")
                time.sleep(delay)
                continue
            points = []
            current_time = pd.to_datetime('now', utc=True)
            for _, row in realtime_df.iterrows():
                if pd.isna(row['最新价']):
                    continue
                p = Point(measurement_name) \
                    .tag("股票代码", row['代码']) \
                    .tag("股票名称", row['名称']) \
                    .field("最新价", parse_unit_value(row['最新价'])) \
                    .field("涨跌幅(%)", float(row['涨跌幅'])) \
                    .field("涨跌额", parse_unit_value(row['涨跌额'])) \
                    .field("成交量(手)", parse_unit_value(row['成交量'])) \
                    .field("成交额(元)", parse_unit_value(row['成交额'])) \
                    .field("振幅(%)", float(row['振幅'])) \
                    .field("最高", parse_unit_value(row['最高'])) \
                    .field("最低", parse_unit_value(row['最低'])) \
                    .field("今开", parse_unit_value(row['今开'])) \
                    .field("昨收", parse_unit_value(row['昨收'])) \
                    .field("量比", parse_unit_value(row['量比'])) \
                    .field("换手率(%)", float(row['换手率'])) \
                    .field("市盈率-动态", parse_unit_value(row['市盈率-动态'])) \
                    .field("市净率", parse_unit_value(row['市净率'])) \
                    .field("总市值(元)", parse_unit_value(row['总市值'])) \
                    .field("流通市值(元)", parse_unit_value(row['流通市值'])) \
                    .field("涨速", parse_unit_value(row['涨速'])) \
                    .field("5分钟涨跌(%)", float(row['5分钟涨跌'])) \
                    .field("60日涨跌幅(%)", float(row['60日涨跌幅'])) \
                    .field("年初至今涨跌幅(%)", float(row['年初至今涨跌幅'])) \
                    .time(current_time)
                points.append(p)
            if points:
                write_api = client.write_api()
                write_api.write(bucket="stock_kdata", org="stock", record=points)
                print(f"  -> 成功写入 {len(points)} 条实时行情数据。")
            else:
                print("  -> 没有有效数据点可以写入。")
            time.sleep(delay)

        except KeyboardInterrupt:
            print("\n程序被用户中断。正在关闭...")
            break
        except Exception as e:
            print(f"  -> 处理实时行情时发生错误: {e},将在10秒后重试。")
            time.sleep(delay)


def get_history_data(query_api: QueryApi, symbol: str, start: str, stop: str) -> pd.DataFrame:
    flux_query = f'''
        from(bucket: "stock_kdata")
          |> range(start: {start}, stop: {stop})
          |> filter(fn: (r) => r._measurement == "history_kdata")
          |> filter(fn: (r) => r.股票代码 == "{symbol}")
          |> pivot(
              rowKey:["_time"],
              columnKey: ["_field"],
              valueColumn: "_value"
          )
          |> keep(columns: ["_time", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"])
          |> rename(columns: {{_time: "日期"}})
    '''
    try:
        df = query_api.query_data_frame(query=flux_query)
        return df if not df.empty else pd.DataFrame()
    except Exception as e:
        print(f"InfluxDB historical query failed for {symbol}: {e}")
        return pd.DataFrame()


def get_now_data(query_api: QueryApi, codes: List[str]) -> pd.DataFrame:

    codes_str = '["' + '", "'.join(codes) + '"]'
    
    flux_query = f'''
        from(bucket: "stock_kdata")
          |> range(start: -5m) 
          |> filter(fn: (r) => r._measurement == "now_kdata")
          |> filter(fn: (r) => contains(value: r.股票代码, set: {codes_str}))
          |> last() 
          |> pivot(
              rowKey:["_time"],
              columnKey: ["_field"],
              valueColumn: "_value"
          )
          |> keep(columns: ["_time", "股票代码", "股票名称","最新价", "涨跌幅(%)", "涨跌额", "成交量(手)", "成交额(元)", "振幅(%)", "最高", "最低", "今开", "昨收", "量比", "换手率(%)", "市盈率-动态", "市净率", "总市值(元)", "流通市值(元)", "涨速", "5分钟涨跌(%)", "60日涨跌幅(%)", "年初至今涨跌幅(%)"]) 
          |> rename(columns: {{_time: "时间"}})
    '''
    try:
        df = query_api.query_data_frame(query=flux_query)
        if not df.empty:
            if pd.api.types.is_datetime64_any_dtype(df['时间']):
                df['时间'] = df['时间'].dt.tz_localize('UTC')
                cst_timezone = timezone(timedelta(hours=8))
                df['时间'] = df['时间'].dt.tz_convert(cst_timezone)
                df['时间'] = df['时间'].dt.tz_localize(None)
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"InfluxDB realtime query failed: {e}")
        return pd.DataFrame()
