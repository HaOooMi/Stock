from vnpy.app.cta_strategy import CtaTemplate, BarGenerator, ArrayManager, TickData, BarData, TradeData, OrderData
from vnpy.app.cta_backtesting import CtaBacktestingEngine, OptimizationSetting

class MeanReversionStrategy(CtaTemplate):
    author = "AI"
    
    entry_threshold = 0.02
    exit_threshold = 0.02

    parameters = ["entry_threshold", "exit_threshold"]
    variables = []

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager()

    def on_bar(self, bar: BarData):
        self.am.update_bar(bar)
        if not self.am.inited:
            return

        if self.am.sma(20) < self.am.close[-1] * (1 - self.entry_threshold):
            self.buy(bar.close_price, 1)
        elif self.am.sma(20) > self.am.close[-1] * (1 + self.exit_threshold):
            self.sell(bar.close_price, 1)

# 创建回测引擎
engine = CtaBacktestingEngine()
engine.set_parameters vt_symbol="EURUSD", interval="1m", start_date="2021-01-01 00:00:00", end_date="2022-01-01 00:00:00"
engine.add_strategy(MeanReversionStrategy, {"entry_threshold": 0.02, "exit_threshold": 0.02})

# 进行回测
engine.run_backtesting()

