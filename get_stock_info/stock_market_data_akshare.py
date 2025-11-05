import time
import os
import akshare as ak
import pandas as pd
import random
from influxdb_client import Point, WriteOptions
from influxdb_client.client.query_api import QueryApi
from typing import List
from datetime import timezone, timedelta
from requests.exceptions import ProxyError as RequestsProxyError, ConnectionError as RequestsConnectionError
from http.client import RemoteDisconnected

try:
    from get_stock_info.utils import parse_unit_value
except ImportError:
    from utils import parse_unit_value

# 彻底禁用外部代理，避免 VPN 或系统代理影响 EastMoney 请求
for env_key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    if env_key in os.environ:
        os.environ.pop(env_key)

if hasattr(ak, "proxies"):
    ak.proxies = None

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

    proxies_disabled = False  # 全局标记：是否已关闭 AkShare 代理

    for index, row in stock_list_df.iterrows():
        stock_code = row["code"]
        stock_name = row["name"]
        print(f"--- 正在处理股票历史行情数据: {stock_name} ({stock_code}) ---")

        # 重试机制：最多重试3次
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                kline_df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="hfq")
                
                if kline_df.empty:
                    print(f"  -> 未找到 {stock_name} ({stock_code}) 的历史行情数据,跳过。")
                    break

                points = []
                for _, kline_row in kline_df.iterrows():
                    # 判断是否停牌：成交量=0 且 成交额=0
                    volume = int(kline_row["成交量"])
                    amount = float(kline_row["成交额"])
                    is_suspended = 1 if (volume == 0 and amount == 0) else 0
                    
                    p = Point(measurement_name) \
                        .tag("股票代码", stock_code) \
                        .tag("股票名称", stock_name) \
                        .field("开盘", float(kline_row["开盘"])) \
                        .field("收盘", float(kline_row["收盘"])) \
                        .field("最高", float(kline_row["最高"])) \
                        .field("最低", float(kline_row["最低"])) \
                        .field("成交量", volume) \
                        .field("成交额", amount) \
                        .field("振幅", float(kline_row["振幅"])) \
                        .field("涨跌幅", float(kline_row["涨跌幅"])) \
                        .field("涨跌额", float(kline_row["涨跌额"])) \
                    .field("换手率", float(kline_row["换手率"])) \
                    .field("是否停牌", is_suspended) \
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
                
                success = True  # 成功，退出重试循环
                time.sleep(0.3)  # 稍微增加延迟，避免请求过快

            except RequestsProxyError as proxy_err:
                retry_count += 1
                proxy_detail = getattr(proxy_err, "__cause__", None)
                proxy_msg = str(proxy_detail) if proxy_detail else str(proxy_err)

                if not proxies_disabled and hasattr(ak, "proxies") and ak.proxies:
                    print("  -> 检测到通过代理请求行情数据失败，正在临时关闭 AkShare 代理设置后重试...")
                    ak.proxies = None
                    proxies_disabled = True
                    time.sleep(2)
                    continue

                wait_time = retry_count * 2
                print(f"  -> 处理 {stock_name} ({stock_code}) 时发生代理错误: {proxy_msg}")
                if retry_count < max_retries:
                    print(f"  -> 等待 {wait_time} 秒后重试 (第 {retry_count}/{max_retries} 次)...")
                    time.sleep(wait_time)
                else:
                    print(f"  -> 处理 {stock_name} ({stock_code}) 失败，代理多次连接失败，跳过。")

            except RequestsConnectionError as conn_err:
                retry_count += 1
                cause = getattr(conn_err, "__cause__", None)
                if isinstance(cause, RemoteDisconnected):
                    detail = "对端主动断开连接"
                else:
                    detail = str(conn_err)

                wait_time = min(10, 2 * retry_count)
                print(f"  -> 拉取 {stock_name} ({stock_code}) 行情时网络连接异常: {detail}")
                if retry_count < max_retries:
                    print(f"  -> 等待 {wait_time} 秒后重试 (第 {retry_count}/{max_retries} 次)...")
                    time.sleep(wait_time)
                else:
                    print(f"  -> 处理 {stock_name} ({stock_code}) 失败，网络多次连接失败，跳过。")
                    print("     建议确认 VPN/网络未拦截 eastmoney 域名，或稍后再试。")

            except KeyboardInterrupt:
                print("\n程序被用户中断。正在关闭...")
                return  # 直接退出函数
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = retry_count * 2  # 递增等待时间：2秒、4秒、6秒
                    print(f"  -> 处理 {stock_name} ({stock_code}) 时发生错误: {e}")
                    print(f"  -> 等待 {wait_time} 秒后重试 (第 {retry_count}/{max_retries} 次)...")
                    time.sleep(wait_time)
                else:
                    print(f"  -> 处理 {stock_name} ({stock_code}) 失败，已达最大重试次数，跳过。")
                    print(f"     错误信息: {e}")


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
                
                # 判断是否停牌：成交量=0 或 最新价=昨收（无交易）
                volume = parse_unit_value(row['成交量'])
                amount = parse_unit_value(row['成交额'])
                is_suspended = 1 if (volume == 0 or amount == 0) else 0
                
                p = Point(measurement_name) \
                    .tag("股票代码", row['代码']) \
                    .tag("股票名称", row['名称']) \
                    .field("最新价", parse_unit_value(row['最新价'])) \
                    .field("涨跌幅(%)", float(row['涨跌幅'])) \
                    .field("涨跌额", parse_unit_value(row['涨跌额'])) \
                    .field("成交量(手)", volume) \
                    .field("成交额(元)", amount) \
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
                    .field("是否停牌", is_suspended) \
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
          |> keep(columns: ["_time", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率", "是否停牌"])
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
            column_map = {
                "_time": "时间",
                "股票代码": "股票代码",
                "股票名称": "股票名称",
                "最新价": "最新价",
                "涨跌幅(%)": "涨跌幅",
                "涨跌额": "涨跌额",
                "成交量(手)": "成交量",
                "成交额(元)": "成交额",
                "振幅(%)": "振幅",
                "最高": "最高",
                "最低": "最低",
                "今开": "今开",
                "昨收": "昨收",
                "量比": "量比",
                "换手率(%)": "换手率",
                "市盈率-动态": "市盈率",
                "市净率": "市净率",
                "总市值(元)": "总市值",
                "流通市值(元)": "流通市值",
                "涨速": "涨速",
                "5分钟涨跌(%)": "五分钟涨跌幅",
                "60日涨跌幅(%)": "六十日涨跌幅",
                "年初至今涨跌幅(%)": "年初至今涨跌幅"
            }
            df.rename(columns=column_map, inplace=True)
            if pd.api.types.is_datetime64_any_dtype(df['时间']):
                cst_timezone = timezone(timedelta(hours=8))
                if df['时间'].dt.tz is None:
                    df['时间'] = df['时间'].dt.tz_localize('UTC').dt.tz_convert(cst_timezone)
                else:
                    df['时间'] = df['时间'].dt.tz_convert(cst_timezone)
                df['时间'] = df['时间'].dt.tz_localize(None)

            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"InfluxDB realtime query failed: {e}")
        return pd.DataFrame()