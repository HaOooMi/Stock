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
    """
    从 AkShare 获取财务数据（包含公告日期）
    优先从东方财富获取真实公告日期，无法获取时使用监管规则估算
    """
    stock_list_df = ak.stock_info_a_code_name()
    total_stocks = len(stock_list_df)
    
    # 预先获取所有报告期的业绩报表数据（避免重复请求）
    print("\n正在预加载业绩报表数据（获取真实公告日期）...")
    announcement_cache = {}  # {报告期: DataFrame}
    
    # 常见报告期：从2018年到现在的所有季度
    current_year = pd.Timestamp.now().year
    report_dates = []
    for year in range(2018, current_year + 1):
        for month in [3, 6, 9, 12]:
            if month == 3:
                day = 31
            elif month == 6:
                day = 30
            elif month == 9:
                day = 30
            else:  # 12
                day = 31
            report_dates.append(f"{year}{month:02d}{day:02d}")
    
    # 批量获取业绩报表
    for report_date in report_dates:
        try:
            performance_df = ak.stock_yjbb_em(date=report_date)
            if performance_df is not None and not performance_df.empty:
                announcement_cache[report_date] = performance_df
                print(f"  ✓ 加载 {report_date} 报告期数据: {len(performance_df)} 条")
            time.sleep(0.2)  # 避免请求过快
        except Exception as e:
            print(f"  ✗ 无法获取 {report_date} 报告期数据")
            continue
    
    print(f"业绩报表缓存完成，共 {len(announcement_cache)} 个报告期\n")
    
    # 开始处理每只股票
    for index, row in stock_list_df.iterrows():
        stock_code = row["code"]
        stock_name = row["name"]
        print(f"[{index+1}/{total_stocks}] 正在处理: {stock_name} ({stock_code})")

        try:
            # 获取同花顺财务数据（包含详细财务指标）
            financial_df = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按报告期")
            
            if financial_df.empty:
                print(f"  -> 无财务数据")
                continue
            
            processed_count = 0
            for _, report_row in financial_df.iterrows():
                report_date = pd.to_datetime(report_row.get('报告期'))
                report_date_str = report_date.strftime('%Y%m%d')
                
                # 从缓存中查找真实公告日期
                announce_date = None
                if report_date_str in announcement_cache:
                    performance_df = announcement_cache[report_date_str]
                    stock_data = performance_df[performance_df['股票代码'] == stock_code]
                    if not stock_data.empty and '首次公告日期' in stock_data.columns:
                        first_announce = stock_data.iloc[0]['首次公告日期']
                        if pd.notna(first_announce):
                            announce_date = pd.to_datetime(first_announce)
                
                # 如果无法获取真实公告日期，使用监管规则估算
                # 一季报(3/31)→4/30前, 半年报(6/30)→8/31前, 三季报(9/30)→10/31前, 年报(12/31)→次年4/30前
                if announce_date is None:
                    month = report_date.month
                    if month == 3:  # 一季报
                        announce_date = pd.Timestamp(year=report_date.year, month=4, day=30)
                    elif month == 6:  # 半年报
                        announce_date = pd.Timestamp(year=report_date.year, month=8, day=31)
                    elif month == 9:  # 三季报
                        announce_date = pd.Timestamp(year=report_date.year, month=10, day=31)
                    elif month == 12:  # 年报
                        announce_date = pd.Timestamp(year=report_date.year + 1, month=4, day=30)
                    else:
                        announce_date = report_date + pd.DateOffset(months=2)
                
                data_to_write = {
                    '股票代码': stock_code,
                    '报告期': report_date.date(),
                    '公告日期': announce_date.date(),
                    '净利润': u.parse_unit_value(report_row.get('净利润')),
                    '净利润同比增长率': u.parse_percentage(report_row.get('净利润同比增长率')),
                    '扣非净利润': u.parse_unit_value(report_row.get('扣非净利润')),
                    '扣非净利润同比增长率': u.parse_percentage(report_row.get('扣非净利润同比增长率')),
                    '营业总收入': u.parse_unit_value(report_row.get('营业总收入')),
                    '营业总收入同比增长率': u.parse_percentage(report_row.get('营业总收入同比增长率')),
                    '基本每股收益': u.clean_value(report_row.get('基本每股收益')),
                    '每股净资产': u.clean_value(report_row.get('每股净资产')),
                    '每股资本公积金': u.clean_value(report_row.get('每股资本公积金')),
                    '每股未分配利润': u.clean_value(report_row.get('每股未分配利润')),
                    '每股经营现金流': u.clean_value(report_row.get('每股经营现金流')),
                    '销售净利率': u.parse_percentage(report_row.get('销售净利率')),
                    '净资产收益率': u.parse_percentage(report_row.get('净资产收益率')),
                    '净资产收益率_摊薄': u.parse_percentage(report_row.get('净资产收益率-摊薄')),
                    '营业周期': u.parse_unit_value(report_row.get('营业周期')),
                    '应收账款周转天数': u.parse_unit_value(report_row.get('应收账款周转天数')),
                    '流动比率': u.clean_value(report_row.get('流动比率')),
                    '速动比率': u.clean_value(report_row.get('速动比率')),
                    '保守速动比率': u.clean_value(report_row.get('保守速动比率')),
                    '产权比率': u.clean_value(report_row.get('产权比率')),
                    '资产负债率': u.parse_percentage(report_row.get('资产负债率')),
                }
                
                with engine.begin() as conn:
                    exists = conn.execute(
                        text(f"SELECT 1 FROM `{table_name}` "
                             "WHERE `股票代码` = :code AND `报告期` = :date"),
                        {'code': stock_code, 'date': report_date.date()}
                    ).fetchone()

                    if exists:
                        # 更新所有字段（包括公告日期）
                        update_sql = f"""
                        UPDATE `{table_name}` 
                        SET `公告日期` = :公告日期,
                            `净利润` = :净利润,
                            `净利润同比增长率` = :净利润同比增长率,
                            `扣非净利润` = :扣非净利润,
                            `扣非净利润同比增长率` = :扣非净利润同比增长率,
                            `营业总收入` = :营业总收入,
                            `营业总收入同比增长率` = :营业总收入同比增长率,
                            `基本每股收益` = :基本每股收益,
                            `每股净资产` = :每股净资产,
                            `每股资本公积金` = :每股资本公积金,
                            `每股未分配利润` = :每股未分配利润,
                            `每股经营现金流` = :每股经营现金流,
                            `销售净利率` = :销售净利率,
                            `净资产收益率` = :净资产收益率,
                            `净资产收益率_摊薄` = :净资产收益率_摊薄,
                            `营业周期` = :营业周期,
                            `应收账款周转天数` = :应收账款周转天数,
                            `流动比率` = :流动比率,
                            `速动比率` = :速动比率,
                            `保守速动比率` = :保守速动比率,
                            `产权比率` = :产权比率,
                            `资产负债率` = :资产负债率
                        WHERE `股票代码` = :股票代码 AND `报告期` = :报告期
                        """
                        conn.execute(text(update_sql), data_to_write)
                    else:
                        df_to_write = pd.DataFrame([data_to_write])
                        df_to_write.to_sql(
                            table_name,
                            conn,
                            if_exists='append',
                            index=False,
                        )
                    processed_count += 1
            
            print(f"  -> 完成 {processed_count} 条财务报告处理")
            time.sleep(0.4)
        except Exception as e:
            print(f"  -> 处理失败: {e}")
            continue


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