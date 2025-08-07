from sqlalchemy import create_engine, text
import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS



def get_influxdb_client():
    INFLUX_URL = "http://localhost:8086"
    INFLUX_TOKEN = "aIX6s47YmoJ-OY-rjRbLFl6AHFSYcv000g3vJp3f6l6hkbmvuj-AMtgfkjz0ESF7r536jqasqxzL9NhohGMrwA=="  
    INFLUX_ORG = "stock"              
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        if client.ping():
            print("InfluxDB 连接成功！")
            return client.query_api()
        else:
            print("InfluxDB 连接失败，请检查 URL, Token 或服务状态。")
            return None
    except Exception as e:
        print(f"创建 InfluxDB 客户端时发生错误: {e}")
        return None
    
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
