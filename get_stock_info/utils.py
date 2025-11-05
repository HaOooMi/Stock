from sqlalchemy import create_engine, text
import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
import math



def get_mysql_engine():
    DB_CONFIG = {
    "user": "root",
    "password": "123456",
    "host": "localhost",
    "port": 3306,
    "database": "stock_meta",
    }
    url = (
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        "?charset=utf8mb4"
    )
    return create_engine(url, echo=False, pool_pre_ping=True)



def get_influxdb_client():
    INFLUX_URL = "http://localhost:8086"
    INFLUX_TOKEN = "aIX6s47YmoJ-OY-rjRbLFl6AHFSYcv000g3vJp3f6l6hkbmvuj-AMtgfkjz0ESF7r536jqasqxzL9NhohGMrwA=="  
    INFLUX_ORG = "stock"              
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        if client.ping():
            print("InfluxDB 连接成功！")
            return client
        else:
            print("InfluxDB 连接失败，请检查 URL, Token 或服务状态。")
            return None
    except Exception as e:
        print(f"创建 InfluxDB 客户端时发生错误: {e}")
        return None
    

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
    """
    创建股票财务指标表（优化后的结构）
    - 所有百分比字段使用 DECIMAL(10,6) 存储小数形式（如 0.15 表示 15%）
    - 金额字段使用 DECIMAL(20,2) 
    - 比率字段使用 DECIMAL(10,4)
    - 添加索引优化查询性能
    """
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                `股票代码` VARCHAR(10) NOT NULL COMMENT '股票代码',
                `报告期` DATE NOT NULL COMMENT '财报报告期',
                `公告日期` DATE COMMENT '财报公告日期（实际或估算）',
                
                -- 利润相关指标
                `净利润` DOUBLE COMMENT '净利润（元）',
                `净利润同比增长率` DOUBLE COMMENT '净利润同比增长率（小数形式）',
                `扣非净利润` DOUBLE COMMENT '扣除非经常性损益后净利润（元）',
                `扣非净利润同比增长率` DOUBLE COMMENT '扣非净利润同比增长率（小数形式）',
                
                -- 营收相关指标
                `营业总收入` DOUBLE COMMENT '营业总收入（元）',
                `营业总收入同比增长率` DOUBLE COMMENT '营业总收入同比增长率（小数形式）',
                
                -- 每股指标
                `基本每股收益` DOUBLE COMMENT '基本每股收益（元/股）',
                `每股净资产` DOUBLE COMMENT '每股净资产（元/股）',
                `每股资本公积金` DOUBLE COMMENT '每股资本公积金（元/股）',
                `每股未分配利润` DOUBLE COMMENT '每股未分配利润（元/股）',
                `每股经营现金流` DOUBLE COMMENT '每股经营现金流（元/股）',
                
                -- 盈利能力指标
                `销售净利率` DOUBLE COMMENT '销售净利率（小数形式）',
                `净资产收益率` DOUBLE COMMENT 'ROE 净资产收益率（小数形式）',
                `净资产收益率_摊薄` DOUBLE COMMENT 'ROE 摊薄（小数形式）',
                
                -- 运营能力指标
                `营业周期` DOUBLE COMMENT '营业周期（天）',
                `应收账款周转天数` DOUBLE COMMENT '应收账款周转天数（天）',
                
                -- 偿债能力指标
                `流动比率` DOUBLE COMMENT '流动比率',
                `速动比率` DOUBLE COMMENT '速动比率',
                `保守速动比率` DOUBLE COMMENT '保守速动比率',
                `产权比率` DOUBLE COMMENT '产权比率',
                `资产负债率` DOUBLE COMMENT '资产负债率（小数形式）',
                
                PRIMARY KEY (`股票代码`, `报告期`),
                INDEX `idx_announcement_date` (`公告日期`),
                INDEX `idx_stock_code` (`股票代码`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票财务指标表';
        """))
    print(f"表 `{table_name}` 已创建")

import pandas as pd

def parse_unit_value(value_str):
    """
    解析包含数字单位的字符串，并转换为数字。
    """
    if pd.isna(value_str) or value_str in [None, '', 'None', 'nan', 'NaN', 'N/A', '--', 'False', 'false']:
        return None

    value_str = str(value_str).strip().lower()

    unit_multipliers = {
        'k': 1_000,
        'm': 1_000_000,
        'g': 1_000_000_000,
        'b': 1_000_000_000,
        '万': 10_000,
        '亿': 100_000_000,
    }

    try:
        for unit, multiplier in unit_multipliers.items():
            if value_str.endswith(unit):
                num_str = value_str.removesuffix(unit)
                return float(num_str) * multiplier

        numeric_value = pd.to_numeric(value_str, errors='coerce')
        if pd.isna(numeric_value):
            return None
        return numeric_value

    except (ValueError, TypeError):

        return None

def parse_percentage(value):
    """将百分比转换为浮点数"""
    if value is None or pd.isna(value) or value is False or value == 'False':
        return None
    try:
        value = str(value).strip()
        float_value = float(value.replace('%', ''))
        return float_value / 100
    except (ValueError, AttributeError):
        return None
    
def clean_value(value):
    """处理 inf, -inf, 和 nan，并将它们全部转换成 None"""
    if value is None:
        return None
    try:
        f_val = float(value)
        if not math.isfinite(f_val):
            return None
        return f_val
    except (ValueError, TypeError):
        return None