import stock_market_data_akshare


from utils import create_basic_table, create_financial_table, get_engine, get_influxdb_client
from stock_meta_akshare import fetch_stock_basic_data, fetch_stock_financial_data
from stock_market_data_akshare import fetch_history_market_data


if __name__ == '__main__':
    BASIC_TABLE_NAME = "stock_individual_info"
    FINANCIAL_TABLE_NAME = "stock_financial_data"
    HISTORY_KDATA_MEASUREMENT_NAME = "history_kdata"  
    NOW_KDATA_MEASUREMENT_NAME = "now_kdata"
    engine = get_engine()
    influx_client = get_influxdb_client()
    create_basic_table(engine, BASIC_TABLE_NAME)
    create_financial_table(engine, FINANCIAL_TABLE_NAME)
    fetch_stock_basic_data(engine, BASIC_TABLE_NAME)
    fetch_stock_financial_data(engine, FINANCIAL_TABLE_NAME)
    fetch_history_market_data(influx_client, HISTORY_KDATA_MEASUREMENT_NAME)
    stock_market_data_akshare.fetch_now_market_data(influx_client, NOW_KDATA_MEASUREMENT_NAME)

    print("\n所有股票数据处理完毕！")