"""
MACD+KD综合策略 - VeighNa Studio版本
结合MACD趋势判断和KD超买超卖信号
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

class MacdKdStrategy(CtaTemplate):
    """MACD+KD综合策略 - VeighNa Studio版本"""
    author = "HaOooMi"

    # MACD策略参数
    macd_fast_period = 12     # MACD快线周期
    macd_slow_period = 26     # MACD慢线周期
    macd_signal_period = 9    # MACD信号线周期
    
    # KD策略参数
    kd_window = 9             # KD计算周期
    kd_d_window = 3           # D值平滑周期
    oversold_level = 20       # 超卖阈值
    overbought_level = 80     # 超买阈值
    
    # 交易参数
    trade_size = 1            # 交易数量

    # 策略变量
    dif = 0.0                 # MACD DIF值
    dea = 0.0                 # MACD DEA值
    macd = 0.0                # MACD柱值
    k_value = 0.0             # KD K值
    d_value = 0.0             # KD D值
    
    # 状态记录变量
    kd_golden_cross_in_oversold = False    # 记录KD是否在超卖区域发生过金叉
    kd_death_cross_in_overbought = False   # 记录KD是否在超买区域发生过死叉

    parameters = [
        "macd_fast_period", "macd_slow_period", "macd_signal_period",
        "kd_window", "kd_d_window", "oversold_level", "overbought_level", 
        "trade_size"
    ]
    variables = ["dif", "dea", "macd", "k_value", "d_value", 
                 "kd_golden_cross_in_oversold", "kd_death_cross_in_overbought"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.bg = BarGenerator(self.on_bar, 1, self.on_day_bar)
        self.am = ArrayManager()

    def on_init(self):
        self.write_log("MACD+KD策略初始化")
        self.load_bar(30)

    def on_start(self):
        self.write_log("MACD+KD策略启动")

    def on_stop(self):
        self.write_log("MACD+KD策略停止")

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

        # 计算MACD指标
        dif_array, dea_array, macd_array = self.am.macd(
            self.macd_fast_period, 
            self.macd_slow_period, 
            self.macd_signal_period, 
            array=True
        )
        
        # 计算KD指标
        k_array, d_array = self.am.stoch(
            fastk_period=self.kd_window,
            slowk_period=self.kd_d_window,
            slowk_matype=0,
            slowd_period=self.kd_d_window,
            slowd_matype=0,
            array=True
        )

        # 获取当前值
        self.dif = dif_array[-1]
        self.dea = dea_array[-1]
        self.macd = macd_array[-1]
        self.k_value = k_array[-1]
        self.d_value = d_array[-1]

        # 判断交易信号
        if (len(dif_array) >= 2 and len(dea_array) >= 2 and 
            len(k_array) >= 2 and len(d_array) >= 2):
            
            # MACD趋势判断（基于零轴位置）
            macd_bullish_trend = self.dif > 0  # DIF在零轴以上为多头趋势
            macd_bearish_trend = self.dif < 0  # DIF在零轴以下为空头趋势
            
            # KD金叉死叉判断
            k_prev = k_array[-2]
            d_prev = d_array[-2]
            kd_golden_cross = self.k_value > self.d_value and k_prev <= d_prev
            kd_death_cross = self.k_value < self.d_value and k_prev >= d_prev
            
            # 检查KD是否在超卖区域发生金叉（状态记录）
            if kd_golden_cross and (self.k_value <= self.oversold_level or self.d_value <= self.oversold_level):
                self.kd_golden_cross_in_oversold = True
                self.write_log(f"KD在超卖区域金叉！K{self.k_value:.2f} > D{self.d_value:.2f}，等待上穿{self.oversold_level}")
            
            # 检查KD是否在超买区域发生死叉（状态记录）
            if kd_death_cross and (self.k_value >= self.overbought_level or self.d_value >= self.overbought_level):
                self.kd_death_cross_in_overbought = True
                self.write_log(f"KD在超买区域死叉！K{self.k_value:.2f} < D{self.d_value:.2f}，等待下穿{self.overbought_level}")
            
            # KD上穿下穿关键位判断
            k_cross_up_oversold = self.k_value > self.oversold_level and k_prev <= self.oversold_level
            k_cross_down_overbought = self.k_value < self.overbought_level and k_prev >= self.overbought_level
            
            # 做多信号：MACD在零轴以上 + 之前KD在超卖区域金叉过 + 现在K上穿20轴
            buy_signal = (macd_bullish_trend and self.kd_golden_cross_in_oversold and k_cross_up_oversold)
            
            # 做空信号：MACD在零轴以下 + 之前KD在超买区域死叉过 + 现在K下穿80轴  
            sell_signal = (macd_bearish_trend and self.kd_death_cross_in_overbought and k_cross_down_overbought)
            
            # 重置状态：如果信号触发或者KD重新进入相反区域
            if buy_signal:
                self.kd_golden_cross_in_oversold = False  # 买入后重置
            if sell_signal:
                self.kd_death_cross_in_overbought = False  # 卖出后重置
            
            # 如果KD重新进入超买区域，重置超卖金叉状态
            if self.k_value >= self.overbought_level:
                self.kd_golden_cross_in_oversold = False
            
            # 如果KD重新进入超卖区域，重置超买死叉状态  
            if self.k_value <= self.oversold_level:
                self.kd_death_cross_in_overbought = False

            # 输出详细判断信息
            if self.pos == 0:
                self.write_log(f"多头判断 - MACD零轴以上:{macd_bullish_trend}(DIF:{self.dif:.4f}), 超卖金叉状态:{self.kd_golden_cross_in_oversold}, K上穿20:{k_cross_up_oversold}")
                self.write_log(f"KD值: K{self.k_value:.2f}, D{self.d_value:.2f}, 前值: K{k_prev:.2f}, D{d_prev:.2f}")
                self.write_log(f"做多信号: {buy_signal}")
            elif self.pos > 0:
                self.write_log(f"空头判断 - MACD零轴以下:{macd_bearish_trend}(DIF:{self.dif:.4f}), 超买死叉状态:{self.kd_death_cross_in_overbought}, K下穿80:{k_cross_down_overbought}")
                self.write_log(f"KD值: K{self.k_value:.2f}, D{self.d_value:.2f}, 前值: K{k_prev:.2f}, D{d_prev:.2f}")
                self.write_log(f"做空信号: {sell_signal}")

            # 执行交易
            if buy_signal and self.pos == 0:
                self.buy(bar.close_price * 1.01, self.trade_size)
                self.write_log(f"   两步做多信号触发！")
                self.write_log(f"   步骤1已完成: KD在超卖区域金叉 ✓")
                self.write_log(f"   步骤2触发: K{k_prev:.2f}→{self.k_value:.2f} 上穿{self.oversold_level}")
                self.write_log(f"   MACD多头趋势: DIF{self.dif:.4f} > 0")
                
            elif sell_signal and self.pos > 0:
                self.sell(bar.close_price * 0.99, abs(self.pos))
                self.write_log(f"   两步做空信号触发！")
                self.write_log(f"   步骤1已完成: KD在超买区域死叉 ✓")
                self.write_log(f"   步骤2触发: K{k_prev:.2f}→{self.k_value:.2f} 下穿{self.overbought_level}")
                self.write_log(f"   MACD空头趋势: DIF{self.dif:.4f} < 0")

            # 显示当前状态
            if self.pos == 0:
                trend_str = "多头" if macd_bullish_trend else "空头" if macd_bearish_trend else "震荡"
                self.write_log(f"当前状态: MACD {trend_str}趋势, KD({self.k_value:.1f},{self.d_value:.1f})")

        self.put_event()

    def on_order(self, order: OrderData):
        pass

    def on_trade(self, trade: TradeData):
        self.write_log(f"✅ 成交回报: {trade.direction.value} {trade.volume}股 @{trade.price}")
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        pass
