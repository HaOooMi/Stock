import stock_meta_akshare as source1
import utils as u

if __name__ == '__main__':
    BASIC_TABLE_NAME = "stock_individual_info"
    FINANCIAL_TABLE_NAME = "stock_financial_data"
    engine = u.get_engine()
    # u.create_basic_table(engine, BASIC_TABLE_NAME)
    # u.create_financial_table(engine, FINANCIAL_TABLE_NAME)
    # source1.fetch_stock_basic_data(engine, BASIC_TABLE_NAME)
    source1.fetch_stock_financial_data(engine, FINANCIAL_TABLE_NAME)
    print("\n所有股票数据处理完毕！")