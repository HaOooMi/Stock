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


def get_engine():

    url = (
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        "?charset=utf8mb4"
    )
    return create_engine(url, echo=False, pool_pre_ping=True)

def create_basic_table(engine, table_name: str):
    """创建股票基础信息表"""
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

def create_financial_table(engine, table_name: str):
    """创建股票财务指标表"""
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                `股票代码` VARCHAR(10) NOT NULL,
                `报告期` DATE NOT NULL,
                `净利润` DECIMAL(20,2),
                `净利润同比增长率` DECIMAL(10,2),
                `扣非净利润` DECIMAL(20,2),
                `扣非净利润同比增长率` DECIMAL(10,2),
                `营业总收入` DECIMAL(20,2),
                `营业总收入同比增长率` DECIMAL(10,2),
                `基本每股收益` DECIMAL(10,4),
                `每股净资产` DECIMAL(10,4),
                `每股经营现金流` DECIMAL(10,4),
                `净资产收益率` DECIMAL(10,2),
                `总资产报酬率` DECIMAL(10,2),
                `毛利率` DECIMAL(10,2),
                `净利率` DECIMAL(10,2),
                `负债权益比` DECIMAL(10,2),
                `流动比率` DECIMAL(10,2),
                `速动比率` DECIMAL(10,2),
                `保守速动比率` DECIMAL(10,2),
                `产权比率` DECIMAL(10,2),
                `资产负债率` DECIMAL(10,2),
                PRIMARY KEY (`股票代码`, `报告期`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """))
    print(f"表 `{table_name}` 已创建")

def parse_unit_value(value_str):
    """解析包含'亿'或'万'单位的字符串，转换为数字"""
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

def parse_percentage(value_str):
    """解析百分比字符串，如 '-9.37%'，转换为数字 -9.37"""
    if pd.isna(value_str) or value_str is None or value_str == 'False':
        return None
    try:
        return float(str(value_str).replace('%', ''))
    except (ValueError, TypeError):
        return None