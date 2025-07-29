import time
import pandas as pd
import akshare as ak
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


DB_CONFIG = {
    "user": "root",
    "password": "123456",
    "host": "localhost",
    "port": 3306,
    "database": "stock_meta",
}


TABLE_NAME = "a_stock_individual_info"


def get_engine():
    url = (
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        "?charset=utf8mb4"
    )
    return create_engine(url, echo=False, pool_pre_ping=True)


def create_table(engine, table_name: str):
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                `股票代码` VARCHAR(10) PRIMARY KEY,
                `股票简称` VARCHAR(50),
                `总股本` BIGINT,
                `流通股` BIGINT,
                `总市值` DECIMAL(20,2),
                `流通市值` DECIMAL(20,2),
                `所属行业` VARCHAR(50),
                `上市时间` DATE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """))

    print(f"表 `{table_name}` 已创建")

def parse_unit_value(value_str):

    if pd.isna(value_str) or value_str is None:
        return None

    value_str = str(value_str)

    try:
        if '亿' in value_str:
            num = float(value_str.replace('亿', ''))
            return num * 100000000
        elif '万' in value_str:
            num = float(value_str.replace('万', ''))
            return num * 10000
        else:
            return pd.to_numeric(value_str, errors='coerce')
    except (ValueError, TypeError):
        return None

def fetch_and_write_stock_data(engine, table_name):
    stock_list_df = ak.stock_info_a_code_name()
    for index, row in stock_list_df.iterrows():
        stock_code = row["code"]
        stock_name = row["name"]
        print(f"--- 正在处理: {stock_name} ({stock_code}) ---")

        try:
            individual_info_df = ak.stock_individual_info_em(symbol=stock_code)
            info_dict = dict(zip(individual_info_df['item'], individual_info_df['value']))
            data_to_write = {
                '股票代码': stock_code,
                '股票简称': stock_name,
                '总股本': parse_unit_value(info_dict.get('总股本')),
                '流通股': parse_unit_value(info_dict.get('流通股')),
                '总市值': parse_unit_value(info_dict.get('总市值')),
                '流通市值': parse_unit_value(info_dict.get('流通市值')),
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

            time.sleep(0.3)
        except Exception as e:
            print(f"  -> 处理 {stock_name} ({stock_code}) 时发生错误: {e}")
            continue

if __name__ == '__main__':
    engine = get_engine()
    create_table(engine, TABLE_NAME)
    fetch_and_write_stock_data(engine, TABLE_NAME)
    print("\n所有股票数据处理完毕！")





