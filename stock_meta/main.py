import stock_meta_akshare 
import stock_market_data_akshare
import utils as u

if __name__ == '__main__':
    BASIC_TABLE_NAME = "stock_individual_info"
    FINANCIAL_TABLE_NAME = "stock_financial_data"
    MEASUREMENT_NAME = "market_data"  
    engine = u.get_engine()
    influx_client = u.get_influxdb_client()
    u.create_basic_table(engine, BASIC_TABLE_NAME)
    u.create_financial_table(engine, FINANCIAL_TABLE_NAME)
    stock_meta_akshare.fetch_stock_basic_data(engine, BASIC_TABLE_NAME)
    stock_meta_akshare.fetch_stock_financial_data(engine, FINANCIAL_TABLE_NAME)
    stock_market_data_akshare.fetch_history_market_data(influx_client, MEASUREMENT_NAME)

    print("\n所有股票数据处理完毕！")