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
    """可视化K线、买卖点和收益曲线，添加均线显示"""
    # 设置大尺寸图表
    fig = plt.figure(figsize=(16, 12), dpi=120)
    fig.suptitle('Simple Moving Average Strategy', fontsize=18)

    # --- 上图：价格线、均线和买卖点 ---
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_ylabel('Price', fontsize=14)
    
    # 绘制价格线（收盘价）
    ax1.plot(df.index, df['收盘'], 
             label='Price Line', color='black', linewidth=1.5, alpha=0.7)
    
    # 添加均线 - 按图片风格
    ax1.plot(df.index, signals['short_ma'], 
             label=f'MA{5}', color='blue', linestyle='-', linewidth=1.5)
    ax1.plot(df.index, signals['long_ma'], 
             label=f'MA{20}', color='orange', linestyle='-', linewidth=1.5)
    
    # 绘制买卖信号（与图片一致）
    buy_signals = signals[signals['buy'].notna()]
    ax1.plot(buy_signals.index, buy_signals['price'],
            '^', markersize=5, color='green', 
            label='Buy Signal', markeredgecolor='darkgreen')
    
    sell_signals = signals[signals['sell'].notna()]
    ax1.plot(sell_signals.index, sell_signals['price'],
            'v', markersize=5, color='red', 
            label='Sell Signal', markeredgecolor='darkred')
    
    # 图表优化
    ax1.legend(fontsize=11, loc='upper left', ncol=2)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 设置坐标范围（参照图片）
    ax1.set_ylim(1300, 2100)
    ax1.set_yticks(range(1300, 2100, 100))
    
    # --- 下图：策略收益曲线 ---
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    ax2.set_ylabel('Cumulative Returns', fontsize=14)
    
    # 策略收益线（蓝色实线）
    ax2.plot(cumulative_returns.index, cumulative_returns, 
             label='Strategy Returns', color='blue', linewidth=1.5)
    
    # 基准收益线（灰色虚线）
    benchmark_returns = (1 + df['收盘'].pct_change()).cumprod()
    ax2.plot(benchmark_returns.index, benchmark_returns, 
             label='Benchmark (Buy & Hold)', color='gray', 
             linestyle='--', linewidth=1.5)
    
    # 图表优化
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 设置收益坐标范围（约0.7-1.0）
    ax2.set_ylim(0.7, 1.1)
    
    # 设置X轴为日期格式
    date_format = mdates.DateFormatter('%m-%d')  # 月-日格式
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(date_format)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.get_xticklabels(), fontsize=10)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.1)
    
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
    