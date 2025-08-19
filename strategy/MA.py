from influxdb_client import InfluxDBClient
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import numpy as np

def get_history_data(stock_code, start_date, end_date) :
    INFLUX_URL = "http://localhost:8086"
    INFLUX_TOKEN = "aIX6s47YmoJ-OY-rjRbLFl6AHFSYcv000g3vJp3f6l6hkbmvuj-AMtgfkjz0ESF7r536jqasqxzL9NhohGMrwA=="  
    INFLUX_ORG = "stock"
    flux_query = f'''
        from(bucket: "stock_kdata")
          |> range(start: {start_date}, stop: {end_date})
          |> filter(fn: (r) => r._measurement == "history_kdata")
          |> filter(fn: (r) => r.股票代码 == "{stock_code}")
          |> pivot(
              rowKey:["_time"],
              columnKey: ["_field"],
              valueColumn: "_value"
          )
          |> keep(columns: ["_time", "开盘", "收盘", "最高", "最低", "成交量"])
    '''
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()
        df = query_api.query_data_frame(query=flux_query)
        required_columns = ["_time", "开盘", "收盘", "最高", "最低", "成交量"]
        df = df[required_columns]
        df = df.rename(columns={"_time": "date"})
        df.set_index('date', inplace=True)
        df = df.astype(float)
        return df if not df.empty else pd.DataFrame()
    except Exception as e:
        print(f"InfluxDB historical query failed for {stock_code}: {e}")
        return pd.DataFrame()
    

def generate_signals(df, short_window=5, long_window=20):
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['收盘']
    signals['short_ma'] = df['收盘'].rolling(window=short_window, min_periods=1).mean()
    signals['long_ma'] = df['收盘'].rolling(window=long_window, min_periods=1).mean()
    
    # 修复1：添加仓位状态管理，防止连续生成信号
    signals['position'] = 0  # 0表示空仓，1表示持有多单
    signals['signal'] = 0    # 实际交易信号
    
    # 修复2：正确识别开仓和平仓信号
    for i in range(1, len(signals)):
        # 当短期均线上穿长期均线 -> 买入信号
        if signals['short_ma'].iloc[i] > signals['long_ma'].iloc[i] and \
           signals['short_ma'].iloc[i-1] <= signals['long_ma'].iloc[i-1] and \
           signals['position'].iloc[i-1] == 0:  # 只在空仓状态下买入
            signals['signal'].iloc[i] = 1
            signals['position'].iloc[i:] = 1    # 开仓后保持持仓状态
            
        # 当短期均线下穿长期均线 -> 卖出信号
        elif signals['short_ma'].iloc[i] < signals['long_ma'].iloc[i] and \
             signals['short_ma'].iloc[i-1] >= signals['long_ma'].iloc[i-1] and \
             signals['position'].iloc[i-1] == 1:  # 只在持仓状态下卖出
            signals['signal'].iloc[i] = -1
            signals['position'].iloc[i:] = 0     # 平仓后保持空仓状态
            
    # 标记买卖点
    signals['buy'] = np.where(signals['signal'] == 1, signals['price'], np.nan)
    signals['sell'] = np.where(signals['signal'] == -1, signals['price'], np.nan)
    
    return signals

def calculate_returns(signals):
    positions = signals['signal'].replace(0, float('nan')).ffill().fillna(0)
    daily_returns = signals['price'].pct_change()
    strategy_returns = positions.shift(1) * daily_returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    total_return = cumulative_returns.iloc[-1]
    print(f"\n--- 策略收益计算结果 ---")
    print(f"策略周期内总收益率: {total_return:.2%}")
    return cumulative_returns

def plot_results(df, signals, cumulative_returns):
    """可视化K线、买卖点和收益曲线（加大K线区域）"""
    # 创建大尺寸图表
    fig = plt.figure(figsize=(18, 16), dpi=120)
    fig.suptitle('Simple Moving Average Strategy with Enhanced Candlestick', fontsize=20)
    
    # 使用正确的GridSpec配置
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1])  # 2行1列，高度比例3:1
    
    # --- 上区域：大尺寸K线图（占75%高度） ---
    ax1 = fig.add_subplot(gs[0])  # 第一行用于K线
    ax1.set_ylabel('Price', fontsize=16)
    
    # 准备K线数据
    ohlc = df[['开盘', '最高', '最低', '收盘']].copy()
    ohlc.reset_index(inplace=True)
    ohlc['date'] = ohlc['date'].map(mdates.date2num)
    
    # 计算K线宽度（动态调整）
    num_days = len(ohlc)
    width = max(0.8, min(1.2, 0.9 * (num_days/100)))  # 更宽的K线
    
    # 绘制大尺寸K线图（更清晰的蜡烛）
    from mplfinance.original_flavor import candlestick_ohlc
    candlestick_ohlc(ax1, ohlc.values, width=width, 
                    colorup='green', colordown='red', alpha=0.9)
    
    # 绘制均线（加粗）
    ax1.plot(df.index, signals['short_ma'], 
             label=f'MA{5}', color='blue', linestyle='-', linewidth=2.5)
    ax1.plot(df.index, signals['long_ma'], 
             label=f'MA{20}', color='orange', linestyle='-', linewidth=2.5)
    
    # 加大买卖信号标记
    buy_signals = signals[signals['buy'].notna()]
    ax1.plot(buy_signals.index, buy_signals['price'],
             '^', markersize=10, color='lime', markeredgecolor='darkgreen', 
             label='Buy Signal', zorder=10)
    
    sell_signals = signals[signals['sell'].notna()]
    ax1.plot(sell_signals.index, sell_signals['price'],
             'v', markersize=10, color='red', markeredgecolor='darkred', 
             label='Sell Signal', zorder=10)
    
    # 增大图例和字体
    ax1.legend(fontsize=14, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.xaxis_date()
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # 设置Y轴范围（根据您的图片数据）
    ax1.set_ylim(1300, 2100)
    ax1.set_yticks(range(1300, 2101, 100))
    
    # 日期格式优化
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    
    # --- 下区域：收益曲线（占25%高度） ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # 第二行用于收益
    ax2.set_ylabel('Cumulative Returns', fontsize=16)
    
    # 收益曲线（更粗的线条）
    cumulative_returns.plot(ax=ax2, label='Strategy Returns', linewidth=2.5)
    
    # 基准线
    benchmark_returns = (1 + df['收盘'].pct_change()).cumprod()
    benchmark_returns.plot(ax=ax2, label='Benchmark (Buy & Hold)', 
                          color='gray', linestyle='--', linewidth=2.5)
    
    # 收益区域设置
    ax2.legend(fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_ylim(0.7, 1.1)
    
    # 添加最终收益率标注（放大字体）
    final_return = cumulative_returns.iloc[-1] - 1
    ax2.text(0.95, 0.05, f'Strategy Final Return: {final_return:.2%}', 
             transform=ax2.transAxes, ha='right', va='bottom', 
             fontsize=14, bbox=dict(facecolor='white', alpha=0.9))

    # 最终调整
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.05)  # 减少子图间距
    
    # 保存高清大图
    plt.savefig('Large_Candlestick_Strategy.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    df_stock = get_history_data("000001", "2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z")
    if df_stock is not None:
        # 2. 生成交易信号
        ma_signals = generate_signals(df_stock, short_window=5, long_window=20)
        
        # 3. 计算收益
        returns_curve = calculate_returns(ma_signals)
        
        # 4. 可视化结果
        plot_results(df_stock, ma_signals, returns_curve)
    