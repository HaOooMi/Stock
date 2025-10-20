"""
MA+ATR综合策略 - VeighNa Studio版本
基于MA策略的趋势方向，使用ATR控制加仓和止损
适配VeighNa Studio环境
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


class MaAtrStrategy(CtaTemplate):
    """MA+ATR综合策略 - VeighNa Studio版本"""
    
    author = "HaOooMi"
    
    # 策略参数
    fast_window = 5         # 快线周期
    slow_window = 20        # 慢线周期
    atr_window = 14         # ATR计算周期
    atr_stop_multiplier = 2.0    # ATR止损倍数
    atr_add_multiplier = 1.0     # ATR加仓倍数
    max_positions = 3       # 最大持仓层数
    trade_size = 1          # 单次交易数量
    
    # 策略变量
    fast_ma = 0.0           # 快线数值
    slow_ma = 0.0           # 慢线数值
    atr_value = 0.0         # ATR值
    entry_price = 0.0       # 初始入场价格
    stop_loss_price = 0.0   # 止损价格
    position_count = 0      # 当前持仓层数
    last_add_price = 0.0    # 上次加仓价格
    
    parameters = ["fast_window", "slow_window", "atr_window", "atr_stop_multiplier", 
                  "atr_add_multiplier", "max_positions", "trade_size"]
    variables = ["fast_ma", "slow_ma", "atr_value", "entry_price", "stop_loss_price", 
                 "position_count", "last_add_price"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        
        self.bg = BarGenerator(self.on_bar, 1, self.on_day_bar)  # 使用日线
        self.am = ArrayManager()
        
    def on_init(self):
        """策略初始化"""
        self.write_log("MA+ATR综合策略初始化")
        self.load_bar(50)  # 加载50天历史数据
        
    def on_start(self):
        """策略启动"""
        self.write_log("MA+ATR综合策略启动")
        
    def on_stop(self):
        """策略停止"""
        self.write_log("MA+ATR综合策略停止")
        
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
        atr_array = self.am.atr(self.atr_window, array=True)
        
        self.fast_ma = fast_ma_array[-1]
        self.slow_ma = slow_ma_array[-1]
        self.atr_value = atr_array[-1]
        
        # 获取当前价格
        current_price = bar.close_price
        
        # 判断MA趋势信号
        if len(fast_ma_array) >= 2 and len(slow_ma_array) >= 2:
            fast_ma_prev = fast_ma_array[-2]
            slow_ma_prev = slow_ma_array[-2]
            
            # MA金叉死叉判断
            is_golden_cross = self.fast_ma > self.slow_ma and fast_ma_prev <= slow_ma_prev
            is_death_cross = self.fast_ma < self.slow_ma and fast_ma_prev >= slow_ma_prev
            
            # 趋势方向判断
            is_uptrend = self.fast_ma > self.slow_ma
            is_downtrend = self.fast_ma < self.slow_ma

            # ATR止损和加仓计算
            if self.pos > 0:  # 多头持仓
                # 计算止损价格（入场价 - ATR*倍数）
                self.stop_loss_price = self.entry_price - self.atr_value * self.atr_stop_multiplier
                
                # 加仓条件：价格上涨ATR*加仓倍数，且未达到最大持仓层数
                add_condition = (current_price >= self.last_add_price + self.atr_value * self.atr_add_multiplier and
                                self.position_count < self.max_positions and is_uptrend)
                
                # ATR止损条件 
                stop_condition = current_price <= self.stop_loss_price or is_death_cross
            else:
                add_condition = False
                stop_condition = False

            # 输出详细判断信息
            if self.pos == 0:
                self.write_log(f"空仓状态 - 快线:{self.fast_ma:.2f}, 慢线:{self.slow_ma:.2f}")
                self.write_log(f"趋势判断: 金叉{is_golden_cross}, 死叉{is_death_cross}")
                self.write_log(f"ATR:{self.atr_value:.2f}, 当前价:{current_price:.2f}")
            elif self.pos > 0:
                self.write_log(f"多头持仓{self.position_count}层 - 入场价:{self.entry_price:.2f}")
                self.write_log(f"止损价:{self.stop_loss_price:.2f}, 当前价:{current_price:.2f}")
                self.write_log(f"加仓条件:{add_condition}, 止损条件:{stop_condition}")
                distance_to_add = self.last_add_price + self.atr_value * self.atr_add_multiplier - current_price
                self.write_log(f"距离加仓点还需上涨:{distance_to_add:.2f}")

            # 执行交易逻辑
            if is_golden_cross and self.pos == 0:
                # MA金叉开多仓
                self.buy(current_price * 1.01, self.trade_size)
                self.entry_price = current_price
                self.last_add_price = current_price
                self.position_count = 1
                self.write_log(f" MA金叉开多仓!")
                self.write_log(f"   快线{self.fast_ma:.2f} > 慢线{self.slow_ma:.2f}")
                self.write_log(f"   入场价:{current_price:.2f}, ATR止损位:{current_price - self.atr_value * self.atr_stop_multiplier:.2f}")
                
            elif is_death_cross and self.pos > 0:
                # MA死叉平多仓（不开空仓）
                self.sell(current_price * 0.99, abs(self.pos))
                pnl = (current_price - self.entry_price) * abs(self.pos)
                self.write_log(f" MA死叉平多仓!")
                self.write_log(f"   快线{self.fast_ma:.2f} < 慢线{self.slow_ma:.2f}")
                self.write_log(f"   平仓价:{current_price:.2f}, 总盈亏:{pnl:.2f}")
                self.write_log(f"   等待下次金叉机会")
                self._reset_position_info()
                
            elif add_condition and self.pos > 0:
                # ATR多头加仓
                self.buy(current_price * 1.01, self.trade_size)
                self.last_add_price = current_price
                self.position_count += 1
                self.write_log(f" ATR多头加仓!")
                self.write_log(f"   第{self.position_count}层加仓, 价格:{current_price:.2f}")
                self.write_log(f"   上涨幅度达到ATR×{self.atr_add_multiplier} = {self.atr_value * self.atr_add_multiplier:.2f}")
                
            elif stop_condition and self.pos > 0:
                # 多头止损
                self.sell(current_price * 0.99, abs(self.pos))
                pnl = (current_price - self.entry_price) * abs(self.pos)
                self.write_log(f"⚠️ 多头止损平仓!")
                self.write_log(f"   止损原因: ATR止损 或 MA死叉")
                self.write_log(f"   平仓价:{current_price:.2f}, 总盈亏:{pnl:.2f}")
                self._reset_position_info()

            # 显示当前状态
            if self.pos == 0:
                trend_str = "上升" if is_uptrend else "下降" if is_downtrend else "震荡"
                if is_downtrend:
                    self.write_log(f"当前状态: {trend_str}趋势, ATR波动:{self.atr_value:.2f} - 等待金叉做多机会")
                else:
                    self.write_log(f"当前状态: {trend_str}趋势, ATR波动:{self.atr_value:.2f}")
        
        # 更新图形界面
        self.put_event()
    
    def _reset_position_info(self):
        """重置持仓信息"""
        self.entry_price = 0.0
        self.last_add_price = 0.0
        self.position_count = 0
        self.stop_loss_price = 0.0
        
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
