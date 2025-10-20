"""
简单MACD策略 - VeighNa Studio版本
基于ma策略模版，适配VeighNa Studio环境
"""
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

class SimpleMacdStrategy(CtaTemplate):
    """简单MACD策略 - VeighNa Studio版本"""
    author = "HaOooMi"

    # 策略参数
    fast_period = 12      # 快线周期
    slow_period = 26      # 慢线周期
    signal_period = 9     # 信号线周期
    trade_size = 1        # 交易数量

    # 策略变量
    dif = 0.0             # DIF值
    dea = 0.0             # DEA值
    macd = 0.0            # MACD柱值

    parameters = ["fast_period", "slow_period", "signal_period", "trade_size"]
    variables = ["dif", "dea", "macd"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.bg = BarGenerator(self.on_bar, 1, self.on_day_bar)
        self.am = ArrayManager()

    def on_init(self):
        self.write_log("策略初始化")
        self.load_bar(30)

    def on_start(self):
        self.write_log("策略启动")

    def on_stop(self):
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        self.bg.update_bar(bar)

    def on_day_bar(self, bar: BarData):
        self.cancel_all()
        self.am.update_bar(bar)
        if not self.am.inited:
            self.write_log(f"数据未初始化完成，当前数据量: {self.am.count}")
            return
        # 计算MACD指标 - 获取数组形式
        dif_array ,dea_array,macd_array= self.am.macd(self.fast_period, self.slow_period, self.signal_period, array=True)
        
        # 获取当前值
        self.dif = dif_array[-1]
        self.dea = dea_array[-1] 
        self.macd = macd_array[-1]
        
        # 判断交易信号
        # MACD金叉：DIF上穿DEA，买入
        # MACD死叉：DIF下穿DEA，卖出
        if len(dif_array) >= 2 and len(dea_array) >= 2:
            dif_prev = dif_array[-2]
            dea_prev = dea_array[-2]
            is_golden_cross = self.dif > self.dea and dif_prev <= dea_prev
            is_death_cross = self.dif < self.dea and dif_prev >= dea_prev
            if self.pos == 0:
                self.write_log(f"判断MACD金叉: (DIF > DEA): {self.dif > self.dea}, (前DIF <= 前DEA): {dif_prev <= dea_prev}. 最终结果: {is_golden_cross}")
            elif self.pos > 0:
                self.write_log(f"判断MACD死叉: (DIF < DEA): {self.dif < self.dea}, (前DIF >= 前DEA): {dif_prev >= dea_prev}. 最终结果: {is_death_cross}")
            if is_golden_cross and self.pos == 0:
                self.buy(bar.close_price * 1.01, self.trade_size)
                self.write_log(f"MACD金叉买入信号: DIF{self.dif:.2f} > DEA{self.dea:.2f}")
            elif is_death_cross and self.pos > 0:
                self.sell(bar.close_price * 0.99, abs(self.pos))
                self.write_log(f"MACD死叉卖出信号: DIF{self.dif:.2f} < DEA{self.dea:.2f}")
        self.put_event()

    def on_order(self, order: OrderData):
        pass

    def on_trade(self, trade: TradeData):
        self.write_log(f"成交回报: {trade.direction.value} {trade.volume}股 @{trade.price}")
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        pass
