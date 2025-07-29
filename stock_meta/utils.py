from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

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