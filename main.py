if __name__ == '__main__':
    engine = get_engine()
    create_table(engine, TABLE_NAME)
    fetch_and_write_stock_data(engine, TABLE_NAME)
    print("\n所有股票数据处理完毕！")