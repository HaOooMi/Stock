"""
从InfluxDB导出股票数据到CSV文件
用于VeighNa手动导入
"""
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入InfluxDB数据模块
try:
    import utils
    from stock_market_data_akshare import get_history_data
    INFLUXDB_AVAILABLE = True
    print("✅ InfluxDB模块导入成功")
except ImportError as e:
    INFLUXDB_AVAILABLE = False
    print(f"❌ InfluxDB模块导入失败: {e}")


def export_stock_data_to_csv(symbol: str, start_date: str = "2022-01-01", end_date: str = "2023-12-31", output_dir: str = "csv_data"):
    """
    从InfluxDB导出股票数据到CSV文件
    
    Args:
        symbol: 股票代码，如 '000001'
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date: 结束日期，格式 'YYYY-MM-DD'
        output_dir: 输出目录
    """
    if not INFLUXDB_AVAILABLE:
        print("❌ InfluxDB模块不可用")
        return False
    
    try:
        print(f"🚀 开始导出股票 {symbol} 数据...")
        
        # 获取InfluxDB客户端
        client = utils.get_influxdb_client()
        if client is None:
            print("❌ 无法连接到InfluxDB")
            return False
        
        query_api = client.query_api()
        
        # 转换日期格式
        start_str_rfc = f"{start_date}T00:00:00Z"
        end_str_rfc = f"{end_date}T23:59:59Z"
        
        # 获取历史数据
        df = get_history_data(query_api, symbol, start_str_rfc, end_str_rfc)
        
        if df.empty:
            print(f"❌ InfluxDB中未找到 {symbol} 的数据")
            return False
        
        # 确保日期列是datetime类型并排序
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values('日期').reset_index(drop=True)
        
        print(f"✅ 成功获取 {len(df)} 条 {symbol} 数据")
        print(f"📅 数据时间范围: {df['日期'].min().date()} 到 {df['日期'].max().date()}")
        
        # 创建输出目录（在当前文件夹内）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path_full = os.path.join(script_dir, output_dir)
        os.makedirs(output_path_full, exist_ok=True)
        
        # 准备VeighNa标准CSV格式数据
        vnpy_df = df.copy()
        
        # 重命名列以符合VeighNa标准
        column_mapping = {
            '日期': 'datetime',
            '开盘': 'open',
            '最高': 'high', 
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
            '成交额': 'turnover'
        }
        
        vnpy_df = vnpy_df.rename(columns=column_mapping)
        
        # 确保必要的列存在
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in vnpy_df.columns:
                print(f"❌ 缺少必要列: {col}")
                return False
        
        # 添加其他VeighNa需要的列
        if 'turnover' not in vnpy_df.columns:
            vnpy_df['turnover'] = 0.0
        
        vnpy_df['open_interest'] = 0  # 股票没有持仓量，设为0
        
        # 格式化日期时间
        vnpy_df['datetime'] = vnpy_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 选择和排序列
        final_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'open_interest']
        vnpy_df = vnpy_df[final_columns]
        
        # 生成文件名
        exchange = 'SZSE' if symbol.startswith('00') or symbol.startswith('30') else 'SSE'
        filename = f"{symbol}.{exchange}_d_{start_date}_{end_date}.csv"
        csv_output_path = os.path.join(output_path_full, filename)
        
        # 保存CSV文件
        vnpy_df.to_csv(csv_output_path, index=False, encoding='utf-8')
        
        print(f"✅ 数据已导出到: {csv_output_path}")
        print(f"📊 导出了 {len(vnpy_df)} 条记录")
        
        # 显示数据预览
        print("\n📋 数据预览（前5行）:")
        print(vnpy_df.head().to_string(index=False))
        
        # 显示VeighNa导入说明
        print(f"\n💡 VeighNa导入配置：")
        print(f"📁 选择文件: {csv_output_path}")
        print(f"🏷️ 代码: {symbol}")
        print(f"🏢 交易所: {exchange}")
        print(f"📅 周期: DAILY (改成DAILY，不是MINUTE)")
        print(f"🌏 时区: Asia/Shanghai")
        print(f"⏰ 时间格式: %Y-%m-%d %H:%M:%S")
        print(f"📋 字段映射: 保持默认")
        
        return True
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_export_stocks(symbols: list, start_date: str = "2022-01-01", end_date: str = "2023-12-31", output_dir: str = "csv_data"):
    """批量导出多只股票数据"""
    print(f"\n🔄 开始批量导出 {len(symbols)} 只股票数据...")
    print(f"📅 时间范围: {start_date} 到 {end_date}")
    print(f"📁 输出目录: {output_dir}")
    
    success_count = 0
    failed_symbols = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] 处理股票: {symbol}")
        
        if export_stock_data_to_csv(symbol, start_date, end_date, output_dir):
            success_count += 1
        else:
            failed_symbols.append(symbol)
    
    print(f"\n{'='*60}")
    print(f"\n📊 批量导出结果：")
    print(f"✅ 成功: {success_count}/{len(symbols)}")
    print(f"❌ 失败: {len(failed_symbols)}")
    
    if failed_symbols:
        print(f"失败的股票: {', '.join(failed_symbols)}")
    
    # 显示完整路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_path = os.path.join(script_dir, output_dir)
    print(f"📁 所有CSV文件保存在: {full_output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("🚀 InfluxDB股票数据CSV导出工具")
    print("=" * 50)
    
    # 检查环境
    if not INFLUXDB_AVAILABLE:
        print("❌ InfluxDB模块不可用，请检查安装")
        exit(1)
    
    # 单只股票导出示例
    print("\n📋 单只股票导出示例：")
    symbol = "000001"  # 平安银行
    success = export_stock_data_to_csv(
        symbol=symbol,
        start_date="2022-01-01",
        end_date="2024-12-31",
        output_dir="csv_data"
    )
    
    if success:
        print(f"\n🎉 {symbol} 数据导出完成！")
        print(f"\n🚀 使用步骤：")
        print(f"1. 运行导出脚本 ✅")
        print(f"2. 在VeighNa中导入CSV数据:")
        print(f"   📁 选择文件: 生成的CSV文件")
        print(f"   🏷️ 代码: {symbol}")
        print(f"   🏢 交易所: SZSE (深交所)")
        print(f"   📅 周期: DAILY (重要：改成DAILY)")
        print(f"   🌏 时区: Asia/Shanghai")
        print(f"   📋 字段映射: 保持默认")
        print(f"   ⏰ 时间格式: %Y-%m-%d %H:%M:%S")
        print(f"3. 导入完成后在CtaBacktester回测:")
        print(f"   🎯 策略选择: SimpleMAStrategy")
        print(f"   🏷️ 本地代码: {symbol}.SZSE")
        print(f"   📈 K线周期: 1d (日线)")
        print(f"4. 优势:")
        print(f"   ✅ 绕过数据服务配置问题")
        print(f"   ✅ 使用您自己的InfluxDB数据")
        print(f"   ✅ 完整的VeighNa回测图表功能")
        print(f"   ✅ 数据可控，随时更新")
    
    # 批量导出示例（取消注释以使用）
    # print("\n📋 批量导出示例：")
    # symbols = ["000001", "000002", "600000", "600036"]  # 多只股票
    # batch_export_stocks(
    #     symbols=symbols,
    #     start_date="2022-01-01", 
    #     end_date="2023-12-31",
    #     output_dir="csv_data"
    # )
    
    print("\n✨ 现在可以在VeighNa中手动导入这些CSV文件了！")
