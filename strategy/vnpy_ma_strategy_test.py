from datetime import datetime
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
from vnpy.trader.constant import Interval, Direction, Offset
from vnpy.trader.optimize import OptimizationSetting
from vnpy_ctastrategy.backtesting import BacktestingEngine

class SimpleMAStrategy(CtaTemplate):
    """简单双均线策略（新版）"""

    author = "HaOooMi"

    fast_window = 5
    slow_window = 20

    fast_ma = 0.0
    slow_ma = 0.0

    parameters = ["fast_window", "slow_window"]
    variables = ["fast_ma", "slow_ma"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.bg = BarGenerator(self.on_bar, 15, self.on_15min_bar)
        self.am = ArrayManager()

    def on_init(self):
        self.write_log("策略初始化")
        self.load_bar(10)

    def on_start(self):
        self.write_log("策略启动")

    def on_stop(self):
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        self.bg.update_bar(bar)

    def on_15min_bar(self, bar: BarData):
        self.cancel_all()
        self.am.update_bar(bar)
        if not self.am.inited:
            return
        self.fast_ma = self.am.sma(self.fast_window, array=False)
        self.slow_ma = self.am.sma(self.slow_window, array=False)

        if self.fast_ma > self.slow_ma and self.pos == 0:
            self.buy(bar.close_price * 1.01, 1)
        elif self.fast_ma < self.slow_ma and self.pos > 0:
            self.sell(bar.close_price * 0.99, abs(self.pos))
        self.put_event()

    def on_order(self, order: OrderData):
        pass

    def on_trade(self, trade: TradeData):
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        pass


def run_backtesting():
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol="000001.SZSE",  # 平安银行
        interval=Interval.MINUTE,
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31),
        rate=0.0003,
        slippage=0.2,
        size=100,
        pricetick=0.01,
        capital=100000,
    )
    engine.add_strategy(SimpleMAStrategy, {"fast_window": 5, "slow_window": 20})
    engine.load_data()
    engine.run_backtesting()
    df = engine.calculate_result()
    print("\n=== 回测结果 ===")
    print(df)
    stats = engine.calculate_statistics()
    print("\n=== 统计数据 ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    engine.show_chart()


def run_optimization():
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol="000001.SZSE",
        interval=Interval.MINUTE,
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31),
        rate=0.0003,
        slippage=0.2,
        size=100,
        pricetick=0.01,
        capital=100000,
    )
    setting = OptimizationSetting()
    setting.set_target("sharpe_ratio")
    setting.add_parameter("fast_window", 3, 10, 1)
    setting.add_parameter("slow_window", 15, 30, 5)
    results = engine.run_optimization(SimpleMAStrategy, setting)
    print("\n=== 优化结果 ===")
    print(results)
    return results


if __name__ == "__main__":
    print("开始运行均线策略回测…")
    run_backtesting()
    # 若需要优化，取消下面两行的注释
    # print("开始参数优化…")
    # run_optimization()
