"""
简单双均线策略 - VeighNa Studio版本
基于vnpy_ma_strategy.py，适配VeighNa Studio环境
使用标准VeighNa数据加载机制
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


class SimpleMaStrategy(CtaTemplate):
    """简单双均线策略 - VeighNa Studio版本"""
    
    author = "HaOooMi"
    
    # 策略参数
    fast_window = 5     # 快线周期
    slow_window = 20    # 慢线周期
    trade_size = 1      # 交易数量
    
    # 策略变量
    fast_ma = 0.0       # 快线数值
    slow_ma = 0.0       # 慢线数值
    
    parameters = ["fast_window", "slow_window", "trade_size"]
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
            
            # 检查金叉条件
            is_golden_cross = self.fast_ma > self.slow_ma and fast_ma_prev <= slow_ma_prev
            # 检查死叉条件
            is_death_cross = self.fast_ma < self.slow_ma and fast_ma_prev >= slow_ma_prev

            # 打印判断条件，无论是否满足
            if self.pos == 0:
                self.write_log(f"判断金叉: (快线 > 慢线): {self.fast_ma > self.slow_ma}, (前快线 <= 前慢线): {fast_ma_prev <= slow_ma_prev}. 最终结果: {is_golden_cross}")
            elif self.pos > 0:
                self.write_log(f"判断死叉: (快线 < 慢线): {self.fast_ma < self.slow_ma}, (前快线 >= 前慢线): {fast_ma_prev >= slow_ma_prev}. 最终结果: {is_death_cross}")

            # 判断交易信号
            # 金叉：快线上穿慢线，买入开仓
            if is_golden_cross and self.pos == 0:
                self.buy(bar.close_price * 1.01, self.trade_size)
                self.write_log(f"金叉买入信号: 快线{self.fast_ma:.2f} > 慢线{self.slow_ma:.2f}")
                
            # 死叉：快线下穿慢线，卖出平仓
            elif is_death_cross and self.pos > 0:
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
