"""
简单布林带策略 - VeighNa Studio版本
基于布林带指标的突破和回归策略
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


class SimpleBollingerStrategy(CtaTemplate):
    """简单布林带策略 - VeighNa Studio版本"""
    
    author = "HaOooMi"
    
    # 策略参数
    boll_window = 20        # 布林带周期
    boll_dev = 2.0          # 布林带标准差倍数
    trade_size = 1          # 交易数量
    stop_loss_ratio = 0.05  # 止损比例（5%）
    profit_target_ratio = 0.10  # 止盈比例（10%）
    
    # 策略变量
    boll_mid = 0.0          # 布林带中轨（均线）
    boll_upper = 0.0        # 布林带上轨
    boll_lower = 0.0        # 布林带下轨
    entry_price = 0.0       # 入场价格
    
    parameters = ["boll_window", "boll_dev", "trade_size", "stop_loss_ratio", "profit_target_ratio"]
    variables = ["boll_mid", "boll_upper", "boll_lower", "entry_price"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        
        self.bg = BarGenerator(self.on_bar, 1, self.on_day_bar)  # 使用日线
        self.am = ArrayManager()
        
    def on_init(self):
        """策略初始化"""
        self.write_log("布林带策略初始化")
        self.load_bar(50)  # 加载50天历史数据
        
    def on_start(self):
        """策略启动"""
        self.write_log("布林带策略启动")
        
    def on_stop(self):
        """策略停止"""
        self.write_log("布林带策略停止")
        
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
            
        # 计算布林带指标
        boll_upper_array, boll_lower_array = self.am.boll(
            self.boll_window, 
            self.boll_dev, 
            array=True
        )
        
        # 计算中轨（均线）
        boll_mid_array = self.am.sma(self.boll_window, array=True)
        
        self.boll_upper = boll_upper_array[-1]
        self.boll_mid = boll_mid_array[-1]
        self.boll_lower = boll_lower_array[-1]
        
        # 获取当前价格
        current_price = bar.close_price
        
        # 获取前一个值用于判断突破
        if len(boll_upper_array) >= 2 and len(boll_lower_array) >= 2:
            boll_upper_prev = boll_upper_array[-2]
            boll_lower_prev = boll_lower_array[-2]
            boll_mid_prev = boll_mid_array[-2]
            current_price_prev = self.am.close_array[-2]
            
            # 检查突破条件
            # 上轨突破：当前价格 > 上轨 且 前一价格 <= 前一上轨
            upper_breakout = current_price > self.boll_upper and current_price_prev <= boll_upper_prev
            
            # 下轨突破：当前价格 < 下轨 且 前一价格 >= 前一下轨  
            lower_breakout = current_price < self.boll_lower and current_price_prev >= boll_lower_prev
            
            # 回归中轨条件
            return_to_mid = False
            if self.pos > 0:  # 多头持仓时，价格回落至中轨
                return_to_mid = current_price <= self.boll_mid

            # 计算布林带宽度（衡量波动性）
            band_width = (self.boll_upper - self.boll_lower) / self.boll_mid * 100
            
            # 价格在布林带中的位置（%B指标）
            price_position = (current_price - self.boll_lower) / (self.boll_upper - self.boll_lower) * 100

            # 输出详细判断信息
            if self.pos == 0:
                self.write_log(f"空仓状态 - 当前价:{current_price:.2f}")
                self.write_log(f"布林带: 上轨{self.boll_upper:.2f}, 中轨{self.boll_mid:.2f}, 下轨{self.boll_lower:.2f}")
                self.write_log(f"突破判断: 上突破{upper_breakout}, 下突破{lower_breakout}")
                self.write_log(f"带宽:{band_width:.2f}%, 价格位置:{price_position:.1f}%")
                
                # 显示距离轨道的距离
                distance_to_upper = self.boll_upper - current_price
                distance_to_lower = current_price - self.boll_lower
                self.write_log(f"距上轨:{distance_to_upper:.2f}, 距下轨:{distance_to_lower:.2f}")
                
            elif self.pos > 0:
                self.write_log(f"多头持仓 - 入场价:{self.entry_price:.2f}, 当前价:{current_price:.2f}")
                self.write_log(f"中轨:{self.boll_mid:.2f}, 回归信号:{return_to_mid}")
                profit = current_price - self.entry_price
                self.write_log(f"浮动盈亏:{profit:.2f}, 价格位置:{price_position:.1f}%")

            # 执行交易逻辑 - A股仅做多
            if upper_breakout and self.pos == 0:
                # 上轨突破做多
                self.buy(current_price * 1.01, self.trade_size)
                self.entry_price = current_price
                self.write_log(f"🚀 突破上轨做多!")
                self.write_log(f"   突破价:{current_price:.2f} > 上轨:{self.boll_upper:.2f}")
                self.write_log(f"   目标位:中轨{self.boll_mid:.2f}")
                
            elif lower_breakout and self.pos == 0:
                # 下轨反弹做多（A股逢低买入策略）
                self.buy(current_price * 1.01, self.trade_size)
                self.entry_price = current_price
                self.write_log(f"� 跌破下轨逢低买入!")
                self.write_log(f"   买入价:{current_price:.2f} < 下轨:{self.boll_lower:.2f}")
                self.write_log(f"   目标位:中轨{self.boll_mid:.2f}")
                
            elif return_to_mid and self.pos > 0:
                # 多头回归中轨平仓
                self.sell(current_price * 0.99, abs(self.pos))
                profit = (current_price - self.entry_price) * abs(self.pos)
                self.write_log(f"💰 多头回归中轨平仓!")
                self.write_log(f"   平仓价:{current_price:.2f}, 盈亏:{profit:.2f}")
                self.entry_price = 0.0

            # A股风控：止损止盈检查
            if self.pos > 0 and self.entry_price > 0:
                # 止损检查
                if current_price <= self.entry_price * (1 - self.stop_loss_ratio):
                    self.sell(current_price * 0.99, abs(self.pos))
                    loss = (self.entry_price - current_price) * abs(self.pos)
                    self.write_log(f"🛑 触发止损平仓!")
                    self.write_log(f"   止损价:{current_price:.2f}, 损失:{loss:.2f}")
                    self.entry_price = 0.0
                    
                # 止盈检查
                elif current_price >= self.entry_price * (1 + self.profit_target_ratio):
                    self.sell(current_price * 0.99, abs(self.pos))
                    profit = (current_price - self.entry_price) * abs(self.pos)
                    self.write_log(f"🎯 触发止盈平仓!")
                    self.write_log(f"   止盈价:{current_price:.2f}, 盈利:{profit:.2f}")
                    self.entry_price = 0.0

            # 显示当前布林带状态
            if self.pos == 0:
                if band_width < 10:
                    market_state = "低波动，等待突破"
                elif band_width > 25:
                    market_state = "高波动，谨慎交易"
                else:
                    market_state = "正常波动"
                    
                if price_position > 80:
                    position_desc = "接近上轨，可能回调"
                elif price_position < 20:
                    position_desc = "接近下轨，可能反弹"
                else:
                    position_desc = "在通道中部"
                    
                self.write_log(f"市场状态: {market_state}, 价格状态: {position_desc}")
        
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
