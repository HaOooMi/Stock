import time
import pandas as pd
import akshare as ak
from sqlalchemy import text
from sqlalchemy.engine import Connection
from typing import Tuple, Dict


import utils as u



def fetch_stock_basic_data(engine, table_name):
    try:
        stock_list_df = ak.stock_info_a_code_name()
        print(f"成功获取 {len(stock_list_df)} 只A股股票列表。")
    except Exception as e:
        print(f"获取A股列表失败: {e}")
        return
    for index, row in stock_list_df.iterrows():
        stock_code = row["code"]
        stock_name = row["name"]
        print(f"--- 正在处理股票基础信息: {stock_name} ({stock_code}) ---")

        try:
            individual_info_df = ak.stock_individual_info_em(symbol=stock_code)
            info_dict = dict(zip(individual_info_df['item'], individual_info_df['value']))
            data_to_write = {
                '股票代码': stock_code,
                '股票简称': stock_name,
                '总股本': u.parse_unit_value(info_dict.get('总股本')),
                '流通股': u.parse_unit_value(info_dict.get('流通股')),
                '总市值': u.parse_unit_value(info_dict.get('总市值')),
                '流通市值': u.parse_unit_value(info_dict.get('流通市值')),
                '所属行业': info_dict.get('行业'),
                '上市时间': pd.to_datetime(info_dict.get('上市时间'),format='%Y%m%d', errors='coerce'),
            }
            with engine.begin() as conn:
                exists = conn.execute(
                    text(f"SELECT 1 FROM {table_name} WHERE `股票代码` = :code"),
                    {'code': stock_code}
                ).fetchone()
                if exists:
                    conn.execute(
                        text(f"""
                        UPDATE {table_name}
                        SET
                            `股票简称` = :股票简称,
                            `总股本` = :总股本,
                            `流通股` = :流通股,
                            `总市值` = :总市值,
                            `流通市值` = :流通市值,
                            `所属行业` = :所属行业,
                            `上市时间` = :上市时间
                        WHERE `股票代码` = :股票代码
                        """),
                        data_to_write
                    )
                    print(f"  -> 已更新 {stock_name} ({stock_code}) 的数据。")
                else:
                    df_to_write = pd.DataFrame([data_to_write])
                    df_to_write.to_sql(
                        table_name,
                        engine,
                        if_exists='append',
                        index=False,
                    )
                    print(f"  -> 成功写入 {stock_name} ({stock_code}) 的数据。")

            time.sleep(0.4)
        except Exception as e:
            print(f"  -> 处理 {stock_name} ({stock_code}) 时发生错误: {e}")
            continue



def fetch_stock_financial_data(engine, table_name):
    """批量获取财务数据并写入数据库"""
    stock_list_df = ak.stock_info_a_code_name()
    total_stocks = len(stock_list_df)
    
    # 预加载所有常见报告期的公告日期（一次性获取，大幅减少网络请求）
    common_dates = pd.date_range(start='2010-01-01', end=pd.Timestamp.now(), freq='QE')
    announcement_cache = {}
    total_dates = len(common_dates)
    print(f"开始预加载公告日期数据，共 {total_dates} 个季度...")
    for idx, date in enumerate(common_dates, 1):
        date_str = date.strftime('%Y%m%d')
        print(f"  [{idx}/{total_dates}] 正在获取 {date.strftime('%Y-%m-%d')} 的公告数据...", end='')
        try:
            perf_df = ak.stock_yjbb_em(date=date_str)
            if perf_df is not None and not perf_df.empty and '股票代码' in perf_df.columns:
                announcement_cache[date_str] = perf_df
                print(f" ✓ 获取到 {len(perf_df)} 条记录")
                time.sleep(0.15)
            else:
                print(" - 无数据")
        except Exception as e:
            print(f" ✗ 失败: {e}")
    print(f"\n预加载完成！成功缓存 {len(announcement_cache)} 个报告期的数据\n")
    
    batch_size = 50  # 每50只股票批量写入一次
    all_records = []
    total_written = 0
    
    for index, row in stock_list_df.iterrows():
        stock_code = row["code"]
        stock_name = row["name"]
        print(f"[{index+1}/{total_stocks}] {stock_name} ({stock_code})")

        financial_df = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按报告期")
        if financial_df.empty:
            continue
        
        # 获取该股票的公告日期映射
        market_symbol = f"{'sh' if stock_code.startswith(('6','9')) else 'sz' if stock_code.startswith(('0','3')) else 'bj'}{stock_code}"
        announce_lookup = {}
        try:
            indicator_df = ak.stock_financial_analysis_indicator(symbol=market_symbol)
            if not indicator_df.empty:
                report_col = next((c for c in ['报告期', '报告日期', '日期'] if c in indicator_df.columns), None)
                announce_col = next((c for c in ['公告日期', '首次公告日期', '最新公告日期', '披露日期'] if c in indicator_df.columns), None)
                if report_col and announce_col:
                    temp_df = indicator_df[[report_col, announce_col]].dropna()
                    announce_lookup = {pd.to_datetime(r[report_col]).normalize(): pd.to_datetime(r[announce_col]).normalize() 
                                     for _, r in temp_df.iterrows() if pd.notna(r[announce_col])}
        except:
            pass

        for _, report_row in financial_df.iterrows():
            report_date = pd.to_datetime(report_row.get('报告期'))
            if pd.isna(report_date):
                continue
            report_date = report_date.normalize()
            report_date_str = report_date.strftime('%Y%m%d')
            
            # 查找公告日期
            announce_date = None
            for col in ['公告日期', '首次公告日期', '最新公告日期', '披露日期']:
                val = report_row.get(col)
                if pd.notna(val):
                    announce_date = pd.to_datetime(val)
                    break
            
            if not announce_date and report_date in announce_lookup:
                announce_date = announce_lookup[report_date]
            
            if not announce_date and report_date_str in announcement_cache:
                perf_df = announcement_cache[report_date_str]
                stock_data = perf_df[perf_df['股票代码'] == stock_code]
                if not stock_data.empty:
                    for col in ['公告日期', '首次公告日期', '最新公告日期', '披露日期']:
                        if col in stock_data.columns:
                            val = stock_data.iloc[0][col]
                            if pd.notna(val):
                                announce_date = pd.to_datetime(val)
                                break
            
            if not announce_date or pd.isna(announce_date):
                continue
            
            def safe_parse(val, parser):
                if pd.isna(val):
                    return None
                try:
                    if hasattr(val, 'string'):
                        val = val.string if val.string else str(val)
                    result = parser(str(val).strip())
                    if result is not None and abs(float(result)) > 1e9:
                        return None
                    return result
                except:
                    return None
            
            all_records.append({
                '股票代码': stock_code,
                '报告期': report_date.date(),
                '公告日期': announce_date.normalize().date(),
                '净利润': safe_parse(report_row.get('净利润'), u.parse_unit_value),
                '净利润同比增长率': safe_parse(report_row.get('净利润同比增长率'), u.parse_percentage),
                '扣非净利润': safe_parse(report_row.get('扣非净利润'), u.parse_unit_value),
                '扣非净利润同比增长率': safe_parse(report_row.get('扣非净利润同比增长率'), u.parse_percentage),
                '营业总收入': safe_parse(report_row.get('营业总收入'), u.parse_unit_value),
                '营业总收入同比增长率': safe_parse(report_row.get('营业总收入同比增长率'), u.parse_percentage),
                '基本每股收益': safe_parse(report_row.get('基本每股收益'), u.clean_value),
                '每股净资产': safe_parse(report_row.get('每股净资产'), u.clean_value),
                '每股资本公积金': safe_parse(report_row.get('每股资本公积金'), u.clean_value),
                '每股未分配利润': safe_parse(report_row.get('每股未分配利润'), u.clean_value),
                '每股经营现金流': safe_parse(report_row.get('每股经营现金流'), u.clean_value),
                '销售净利率': safe_parse(report_row.get('销售净利率'), u.parse_percentage),
                '净资产收益率': safe_parse(report_row.get('净资产收益率'), u.parse_percentage),
                '净资产收益率_摊薄': safe_parse(report_row.get('净资产收益率-摊薄'), u.parse_percentage),
                '营业周期': safe_parse(report_row.get('营业周期'), u.parse_unit_value),
                '应收账款周转天数': safe_parse(report_row.get('应收账款周转天数'), u.parse_unit_value),
                '流动比率': safe_parse(report_row.get('流动比率'), u.clean_value),
                '速动比率': safe_parse(report_row.get('速动比率'), u.clean_value),
                '保守速动比率': safe_parse(report_row.get('保守速动比率'), u.clean_value),
                '产权比率': safe_parse(report_row.get('产权比率'), u.clean_value),
                '资产负债率': safe_parse(report_row.get('资产负债率'), u.parse_percentage),
            })
        
        time.sleep(0.3)
        
        # 每处理batch_size只股票，批量写入一次
        if (index + 1) % batch_size == 0 and all_records:
            print(f"\n>>> 批量写入第 {total_written + 1}-{total_written + len(all_records)} 条记录...")
            df_batch = pd.DataFrame(all_records)
            df_batch.to_sql(table_name, engine, if_exists='append', index=False, method='multi', chunksize=1000)
            total_written += len(all_records)
            print(f">>> 已累计写入 {total_written} 条记录\n")
            all_records = []
    
    # 写入剩余数据
    if all_records:
        print(f"\n>>> 批量写入最后 {len(all_records)} 条记录...")
        df_batch = pd.DataFrame(all_records)
        df_batch.to_sql(table_name, engine, if_exists='append', index=False, method='multi', chunksize=1000)
        total_written += len(all_records)
        print(f">>> 总共写入 {total_written} 条记录")
    else:
        print(f"\n>>> 总共写入 {total_written} 条记录")


def get_basic_info_mysql(db: Connection, code: Tuple) -> Dict:
    if not code:
        return {}
    query = text("SELECT `股票代码`, `股票简称`, `总股本`, `流通股`,`总市值`,`流通市值`,`所属行业`, `上市时间` FROM `stock_individual_info` WHERE `股票代码` IN :codes")
    try:
        df = pd.read_sql(query, db, params={'codes': code})
        if '上市时间' in df.columns:
            df['上市时间'] = df['上市时间'].astype(str)
        return {row['股票代码']: row for _, row in df.iterrows()}
    except Exception as e:
        print(f"查询股票基本信息失败 (股票代码: {code}): {e}")
        return pd.DataFrame()
    
def get_financial_info_mysql(db: Connection, code: str) -> pd.DataFrame:
    if not code:
        return pd.DataFrame()
    query = text("SELECT * FROM `stock_financial_data` WHERE `股票代码` = :code ORDER BY `报告期` DESC")
    try:
        df = pd.read_sql(query, db, params={'code': code})
        if df.empty:
            return pd.DataFrame()
        
        float_columns = [
            '净利润', '净利润同比增长率', '扣非净利润', '扣非净利润同比增长率',
            '营业总收入', '营业总收入同比增长率', '基本每股收益', '每股净资产',
            '每股资本公积金', '每股未分配利润', '每股经营现金流', '销售净利率',
            '净资产收益率', '净资产收益率_摊薄', '营业周期', '应收账款周转天数',
            '流动比率', '速动比率', '保守速动比率', '产权比率', '资产负债率'
        ]
        # 强制转换为float，无效值转为NaN
        for col in float_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        print(f"查询财务数据失败 (股票代码: {code}): {e}")
        return pd.DataFrame()