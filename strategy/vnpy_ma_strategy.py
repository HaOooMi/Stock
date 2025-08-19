"""
基于vnpy的简单双均线策略回测
使用InfluxDB数据源
"""
import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 添加stock_info目录到Python路径
stock_info_dir = os.path.join(project_root, 'stock_info')
sys.path.insert(0, stock_info_dir)

from vnpy_ctastrategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager
)
from vnpy.trader.object import BarData, TickData, OrderData, TradeData
from vnpy.trader.constant import Direction, Offset, Interval, Exchange
from vnpy.trader.optimize import OptimizationSetting
from vnpy_ctastrategy.backtesting import BacktestingEngine
from datetime import datetime
import pandas as pd

# 导入你的数据模块
from stock_info import utils
from stock_info.stock_market_data_akshare import get_history_data


class SimpleMAStrategy(CtaTemplate):
    """简单双均线策略"""
    
    author = "HaOooMi"
    
    # 策略参数
    fast_window = 5     # 快线周期
    slow_window = 20    # 慢线周期
    
    # 策略变量
    fast_ma = 0.0       # 快线数值
    slow_ma = 0.0       # 慢线数值
    
    parameters = ["fast_window", "slow_window"]
    variables = ["fast_ma", "slow_ma"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        
        self.bg = BarGenerator(self.on_bar, 1, self.on_day_bar)  # 使用日线
        self.am = ArrayManager()
        
    def on_init(self):
        """策略初始化"""
        self.write_log("策略初始化")
        self.load_bar(30)  # 加载30天历史数据
        
    def on_start(self):
        """策略启动"""
        self.write_log("策略启动")
        
    def on_stop(self):
        """策略停止"""
        self.write_log("策略停止")
        
    def on_tick(self, tick: TickData):
        """收到行情TICK推送"""
        self.bg.update_tick(tick)
        
    def on_bar(self, bar: BarData):
        """收到Bar推送"""
        self.bg.update_bar(bar)
        
    def on_day_bar(self, bar: BarData):
        """收到日线Bar推送"""
        self.cancel_all()
        
        # 更新K线到技术指标缓存
        self.am.update_bar(bar)
        if not self.am.inited:
            self.write_log(f"数据未初始化完成，当前数据量: {self.am.count}")
            return
            
        # 计算技术指标
        fast_ma_array = self.am.sma(self.fast_window, array=True)
        slow_ma_array = self.am.sma(self.slow_window, array=True)
        
        self.fast_ma = fast_ma_array[-1]
        self.slow_ma = slow_ma_array[-1]
        
        # 获取前一个值用于判断金叉死叉
        if len(fast_ma_array) >= 2 and len(slow_ma_array) >= 2:
            fast_ma_prev = fast_ma_array[-2]
            slow_ma_prev = slow_ma_array[-2]
            
            # 判断交易信号
            # 金叉：快线上穿慢线，买入开仓
            if (self.fast_ma > self.slow_ma and fast_ma_prev <= slow_ma_prev and self.pos == 0):
                self.buy(bar.close_price * 1.01, 1)
                self.write_log(f"金叉买入信号: 快线{self.fast_ma:.2f} > 慢线{self.slow_ma:.2f}")
                
            # 死叉：快线下穿慢线，卖出平仓
            elif (self.fast_ma < self.slow_ma and fast_ma_prev >= slow_ma_prev and self.pos > 0):
                self.sell(bar.close_price * 0.99, abs(self.pos))
                self.write_log(f"死叉卖出信号: 快线{self.fast_ma:.2f} < 慢线{self.slow_ma:.2f}")
            
        # 更新图形界面
        self.put_event()
        
    def on_order(self, order: OrderData):
        """收到委托变化推送"""
        pass
        
    def on_trade(self, trade: TradeData):
        """收到成交推送"""
        self.write_log(f"成交回报: {trade.direction.value} {trade.volume}股 @{trade.price}")
        self.put_event()
        
    def on_stop_order(self, stop_order: StopOrder):
        """收到停止单推送"""
        pass


def load_data_from_influxdb(engine: BacktestingEngine, stock_code: str, start_date: str, end_date: str):
    """从InfluxDB加载数据到vnpy回测引擎"""
    try:
        # 获取InfluxDB客户端
        client = utils.get_influxdb_client()
        query_api = client.query_api()
        
        # 转换日期格式为RFC3339
        start_str_rfc = start_date + "T00:00:00Z"
        end_str_rfc = end_date + "T23:59:59Z"
        
        # 使用你的get_history_data函数
        df = get_history_data(query_api, stock_code, start_str_rfc, end_str_rfc)
        
        if df.empty:
            print(f"未找到股票 {stock_code} 的历史数据")
            return False
        
        # 确保日期列是datetime类型并排序
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values('日期').reset_index(drop=True)
        
        print(f"成功从InfluxDB获取 {stock_code} 的 {len(df)} 条历史数据")
        print(f"数据时间范围: {df['日期'].min()} 到 {df['日期'].max()}")
        
        # 转换为vnpy的BarData格式
        bars = []
        for _, row in df.iterrows():
            bar = BarData(
                symbol=stock_code,
                exchange=Exchange.SZSE if stock_code.startswith('00') else Exchange.SSE,
                datetime=row['日期'],
                interval=Interval.DAILY,
                volume=int(row.get('成交量', 0)),
                turnover=float(row.get('成交额', 0)),
                open_price=float(row['开盘']),
                high_price=float(row['最高']),
                low_price=float(row['最低']),
                close_price=float(row['收盘']),
                gateway_name="influxdb"
            )
            bars.append(bar)
        
        # 将数据加载到回测引擎
        engine.clear_data()  # 清除之前的数据
        for bar in bars:
            engine.new_bar(bar)
            
        print(f"成功将 {len(bars)} 条数据加载到vnpy回测引擎")
        return True
        
    except Exception as e:
        print(f"从InfluxDB加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_backtesting_with_influxdb(stock_code='000001', start_date='2023-01-01', end_date='2023-12-31'):
    """运行基于InfluxDB数据的回测"""
    
    # 创建回测引擎
    engine = BacktestingEngine()
    
    # 设置回测参数
    vt_symbol = f"{stock_code}.SZSE" if stock_code.startswith('00') else f"{stock_code}.SSE"
    
    engine.set_parameters(
        vt_symbol=vt_symbol,
        interval=Interval.DAILY,
        start=datetime.strptime(start_date, '%Y-%m-%d'),
        end=datetime.strptime(end_date, '%Y-%m-%d'),
        rate=0.0003,              # 手续费
        slippage=0.2,             # 滑点
        size=100,                 # 合约乘数
        pricetick=0.01,           # 最小价格变动
        capital=100000,           # 起始资金
    )
    
    # 添加策略
    engine.add_strategy(SimpleMAStrategy, {
        "fast_window": 5,
        "slow_window": 20
    })
    
    # 从InfluxDB加载数据
    print(f"正在从InfluxDB加载 {stock_code} 的历史数据...")
    if not load_data_from_influxdb(engine, stock_code, start_date, end_date):
        print("数据加载失败，回测终止")
        return None, None, None
    
    # 运行回测
    print("开始运行回测...")
    engine.run_backtesting()
    
    # 计算回测结果
    df = engine.calculate_result()
    print("\n=== 回测结果 ===")
    if not df.empty:
        print(df)
    else:
        print("无交易记录")
    
    # 计算统计数据
    statistics = engine.calculate_statistics()
    print("\n=== 统计数据 ===")
    for key, value in statistics.items():
        print(f"{key}: {value}")
    
    # 显示图表
    try:
        engine.show_chart()
    except Exception as e:
        print(f"显示图表失败: {e}")
    
    return engine, df, statistics


def run_optimization_with_influxdb(stock_code='000001', start_date='2022-01-01', end_date='2023-12-31'):
    """运行基于InfluxDB数据的参数优化"""
    
    # 创建回测引擎
    engine = BacktestingEngine()
    
    # 设置回测参数
    vt_symbol = f"{stock_code}.SZSE" if stock_code.startswith('00') else f"{stock_code}.SSE"
    
    engine.set_parameters(
        vt_symbol=vt_symbol,
        interval=Interval.DAILY,
        start=datetime.strptime(start_date, '%Y-%m-%d'),
        end=datetime.strptime(end_date, '%Y-%m-%d'),
        rate=0.0003,
        slippage=0.2,
        size=100,
        pricetick=0.01,
        capital=100000,
    )
    
    # 从InfluxDB加载数据
    print(f"正在从InfluxDB加载 {stock_code} 的历史数据...")
    if not load_data_from_influxdb(engine, stock_code, start_date, end_date):
        print("数据加载失败，优化终止")
        return None
    
    # 设置优化参数
    setting = OptimizationSetting()
    setting.set_target("sharpe_ratio")
    setting.add_parameter("fast_window", 3, 10, 1)
    setting.add_parameter("slow_window", 15, 30, 5)
    
    # 运行优化
    print("开始参数优化...")
    results = engine.run_optimization(SimpleMAStrategy, setting)
    
    print("\n=== 优化结果 ===")
    print(results)
    
    return results


if __name__ == "__main__":
    # 运行回测
    print("开始运行均线策略回测...")
    try:
        engine, df, statistics = run_backtesting_with_influxdb(
            stock_code='000001',      # 平安银行
            start_date='2022-01-01',  # 开始日期
            end_date='2023-12-31'     # 结束日期
        )
        
        if engine:
            print("回测完成!")
            
            # 使用vnpy框架显示图表
            try:
                print("\n正在显示vnpy回测图表...")
                engine.show_chart()
            except Exception as e:
                print(f"无法显示图表: {e}")
                print("提示：在VeighNa Studio中运行可以显示完整的图表界面")
            
            # 如果需要参数优化，取消下面的注释
            # print("\n开始参数优化...")
            # optimization_results = run_optimization_with_influxdb(
            #     stock_code='000001',
            #     start_date='2023-01-01',
            #     end_date='2023-12-31'
            # )
            # print("优化完成!")
        else:
            print("回测失败，请检查InfluxDB数据源")
            
    except Exception as e:
        print(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
