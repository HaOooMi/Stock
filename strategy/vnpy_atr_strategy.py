"""
纯ATR突破策略 - VeighNa Studio版本
基于ATR（平均真实波幅）的价格突破策略

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


class AtrStrategy(CtaTemplate):
    """纯ATR突破策略 - VeighNa Studio版本"""
    
    author = "HaOooMi"
    
    # 策略参数
    atr_window = 14              # ATR计算周期
    atr_multiplier = 2.0         # ATR突破倍数
    stop_multiplier = 1.5        # 止损倍数
    lookback_period = 20         # 基准价格回望周期
    trade_size = 1               # 交易数量
    
    # 策略变量
    atr_value = 0.0              # ATR值
    base_price = 0.0             # 基准价格（N日收盘价均值）
    upper_band = 0.0             # 上轨：基准价 + ATR*倍数
    lower_band = 0.0             # 下轨：基准价 - ATR*倍数
    entry_price = 0.0            # 入场价格
    stop_loss_price = 0.0        # 止损价格
    
    parameters = ["atr_window", "atr_multiplier", "stop_multiplier", 
                  "lookback_period", "trade_size"]
    variables = ["atr_value", "base_price", "upper_band", "lower_band", 
                 "entry_price", "stop_loss_price"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        
        self.bg = BarGenerator(self.on_bar, 1, self.on_day_bar)  # 使用日线
        self.am = ArrayManager()
        
    def on_init(self):
        """策略初始化"""
        self.write_log("纯ATR突破策略初始化")
        self.load_bar(50)  # 加载50天历史数据
        
    def on_start(self):
        """策略启动"""
        self.write_log("纯ATR突破策略启动")
        
    def on_stop(self):
        """策略停止"""
        self.write_log("纯ATR突破策略停止")
        
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
        atr_array = self.am.atr(self.atr_window, array=True)
        close_array = self.am.close_array
        
        self.atr_value = atr_array[-1]
        
        # 计算基准价格（N日收盘价均值）
        if len(close_array) >= self.lookback_period:
            self.base_price = close_array[-self.lookback_period:].mean()
        else:
            self.base_price = close_array.mean()
        
        # 计算上下轨
        self.upper_band = self.base_price + self.atr_value * self.atr_multiplier
        self.lower_band = self.base_price - self.atr_value * self.atr_multiplier
        
        # 获取当前价格
        current_price = bar.close_price
        
        # 判断交易信号
        if len(atr_array) >= 2:
            
            # 突破信号
            long_breakout = current_price > self.upper_band
            # 删除空头突破信号
            
            # 回归信号（价格回到通道内，只针对多头持仓）
            price_return_to_band = (current_price <= self.base_price and self.pos > 0)
            
            # 下跌破位信号（价格跌破下轨，多头止损）
            downside_break = current_price < self.lower_band
            
            # 止损信号
            if self.pos > 0:  # 多头持仓
                self.stop_loss_price = self.entry_price - self.atr_value * self.stop_multiplier
                stop_loss_signal = current_price <= self.stop_loss_price
            else:
                stop_loss_signal = False

            # 输出详细判断信息
            if self.pos == 0:
                self.write_log(f"空仓状态 - 当前价:{current_price:.2f}, 基准价:{self.base_price:.2f}")
                self.write_log(f"上轨:{self.upper_band:.2f}, 下轨:{self.lower_band:.2f}, ATR:{self.atr_value:.2f}")
                self.write_log(f"上突破:{long_breakout}, 下跌破位:{downside_break}")
                
                # 显示距离突破点的距离
                distance_to_upper = self.upper_band - current_price
                distance_to_lower = current_price - self.lower_band
                self.write_log(f"距上轨:{distance_to_upper:.2f}, 距下轨:{distance_to_lower:.2f}")
                
            elif self.pos > 0:
                self.write_log(f"多头持仓 - 入场价:{self.entry_price:.2f}, 当前价:{current_price:.2f}")
                self.write_log(f"止损价:{self.stop_loss_price:.2f}, 止损信号:{stop_loss_signal}")
                self.write_log(f"回归信号:{price_return_to_band}, 破位信号:{downside_break}")
                profit = current_price - self.entry_price
                self.write_log(f"浮动盈亏:{profit:.2f}")

            # 执行交易逻辑
            if long_breakout and self.pos == 0:
                # 向上突破做多
                self.buy(current_price * 1.01, self.trade_size)
                self.entry_price = current_price
                self.write_log(f"🚀 向上突破做多!")
                self.write_log(f"   突破价:{current_price:.2f} > 上轨:{self.upper_band:.2f}")
                self.write_log(f"   ATR止损位:{current_price - self.atr_value * self.stop_multiplier:.2f}")
                
            elif (price_return_to_band or downside_break) and self.pos > 0:
                # 价格回归基准线或跌破下轨，平多仓
                self.sell(current_price * 0.99, abs(self.pos))
                profit = (current_price - self.entry_price) * abs(self.pos)
                
                if price_return_to_band:
                    self.write_log(f"💰 价格回归基准线，多头获利平仓!")
                else:
                    self.write_log(f"� 价格跌破下轨，多头止损平仓!")
                
                self.write_log(f"   平仓价:{current_price:.2f}, 盈亏:{profit:.2f}")
                self.entry_price = 0.0
                
            elif stop_loss_signal and self.pos > 0:
                # ATR止损
                self.sell(current_price * 0.99, abs(self.pos))
                loss = (current_price - self.entry_price) * abs(self.pos)
                self.write_log(f"⚠️ 多头ATR止损!")
                self.write_log(f"   止损价:{current_price:.2f}, 盈亏:{loss:.2f}")
                self.entry_price = 0.0

            # 显示当前通道状态
            if self.pos == 0:
                if current_price > self.base_price:
                    position_str = f"基准价上方{current_price - self.base_price:.2f}"
                elif current_price < self.lower_band:
                    position_str = f"下轨下方{self.lower_band - current_price:.2f}，等待反弹机会"
                else:
                    position_str = f"基准价下方{self.base_price - current_price:.2f}"
                    
                channel_width = self.upper_band - self.lower_band
                self.write_log(f"通道状态: {position_str}, 通道宽度:{channel_width:.2f}")
        
        # 更新图形界面
        self.put_event()
        
    def on_order(self, order: OrderData):
        """收到委托变化推送"""
        pass
        
    def on_trade(self, trade: TradeData):
        """收到成交推送"""
        self.write_log(f"✅ 成交回报: {trade.direction.value} {trade.volume}股 @{trade.price}")
        self.put_event()
        
    def on_stop_order(self, stop_order: StopOrder):
        """收到停止单推送"""
        pass
