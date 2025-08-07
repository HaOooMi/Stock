from fastapi import APIRouter,Depends,HTTPException, Query, Path   
from pydantic import BaseModel
from typing import List
from datetime import date, timedelta, datetime
from influxdb_client.client.query_api import QueryApi

from utils import get_influxdb_client
import sys
sys.path.append('d:/vscode projects/stock')
from stock_info.stock_market_data_akshare import get_history_data


class DailyPriceData(BaseModel):
    日期: datetime 
    开盘: float
    收盘: float
    最高: float
    最低: float
    成交量: int
    成交额: float
    振幅: float
    涨跌幅: float
    涨跌额: float
    换手率: float

class HistoricalDataResponse(BaseModel):
    股票代码: str
    data: List[DailyPriceData]


history=APIRouter()

@history.get("/historical-data/{stock_code}", response_model=HistoricalDataResponse, tags=["股票历史行情数据"])
def get_historical_data(
    stock_code: str = Path(..., description="单个股票代码", example="000001"),
    start_date: date = Query(None, description="开始日期 (YYYY-MM-DD)"),
    end_date: date = Query(date.today(), description="结束日期 (YYYY-MM-DD)"),
    query_api: QueryApi = Depends(get_influxdb_client) 
):
    if start_date is None:
        start_date = end_date - timedelta(days=365)
    
    start_str_rfc = start_date.isoformat() + "T00:00:00Z"
    stop_str_rfc = end_date.isoformat() + "T23:59:59Z"

    history_df = get_history_data(query_api, stock_code, start_str_rfc, stop_str_rfc)
    
    if history_df.empty:
        raise HTTPException(status_code=404, detail=f"未找到 {stock_code} 的历史数据。")
    
    return HistoricalDataResponse(
        股票代码=stock_code,
        data=history_df.to_dict('records')
    )
