import time
import pandas as pd
import akshare as ak
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

import utils as u

DB_CONFIG = {
    "user": "root",
    "password": "123456",
    "host": "localhost",
    "port": 3306,
    "database": "stock_meta",
}


def get_engine():

    url = (
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        "?charset=utf8mb4"
    )
    return create_engine(url, echo=False, pool_pre_ping=True)



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
    stock_list_df = ak.stock_info_a_code_name()
    for index, row in stock_list_df.iterrows():
        stock_code = row["code"]
        stock_name = row["name"]
        print(f"--- 正在处理股票财务指标: {stock_name} ({stock_code}) ---")

        try:
            financial_df = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按报告期")
            for _, report_row in financial_df.iterrows():
                report_date = pd.to_datetime(report_row.get('报告期')).date()
                data_to_write = {
                    '股票代码': stock_code,
                    '报告期': report_date,
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
                    '净资产收益率-摊薄': u.parse_percentage(report_row.get('净资产收益率-摊薄')),
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
                        {'code': stock_code, 'date': report_date}
                    ).fetchone()

                    if exists:
                        continue
                    else:
                        df_to_write = pd.DataFrame([data_to_write])
                        df_to_write.to_sql(
                            table_name,
                            conn,
                            if_exists='append',
                            index=False,
                        )
            print(f"  -> 完成 {stock_name} ({stock_code}) 的 {len(financial_df)} 条财务报告处理。")             
            time.sleep(0.4)
        except Exception as e:
            print(f"  -> 处理 {stock_name} ({stock_code}) 财务数据时发生错误: {e}")
            continue


