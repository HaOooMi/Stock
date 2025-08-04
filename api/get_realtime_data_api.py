from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
from influxdb_client.client.query_api import QueryApi
from pydantic import BaseModel, Field
from typing import List, Optional, Dict


from stock_info.utils import get_influxdb_client
from stock_info.stock_market_data_akshare import get_now_data

class RealtimeQuote(BaseModel):
    时间: datetime
    股票代码: str
    股票名称: str
    最新价: Optional[float] = None
    涨跌幅: Optional[float] = None
    涨跌额: Optional[float] = None
    成交量: Optional[int] = None
    成交额: Optional[float] = None
    振幅: Optional[float] = None
    最高: Optional[float] = None
    最低: Optional[float] = None
    今开: Optional[float] = None
    昨收: Optional[float] = None
    量比: Optional[float] = None
    换手率: Optional[float] = None
    市盈率: Optional[int] = None
    市净率: Optional[float] = None
    总市值: Optional[float] = None
    流通市值: Optional[float] = None
    涨速: Optional[float] = None 
    五分钟涨跌幅: Optional[float] = None
    六十日涨跌幅: Optional[float] = None
    年初至今涨跌幅: Optional[float] = None

class RealtimeQuotesResponse(BaseModel):
    data: Dict[str, Optional[RealtimeQuote]]

class StockCodesRequest(BaseModel):
    stock_codes: List[str] = Field(..., description="股票代码列表", example=["000001", "000002"])

now=APIRouter()
    
@now.post("/realtime-quotes", response_model=RealtimeQuotesResponse, tags=["股票实时行情数据"])
def get_realtime_quotes(
    request: StockCodesRequest,
    query_api: QueryApi = Depends(get_influxdb_client)
):
    if not request.stock_codes:
        raise HTTPException(status_code=400, detail="股票代码列表不能为空。")
        
    unique_codes = list(set(request.stock_codes))
    quotes_df = get_now_data(query_api, unique_codes)
    raw_data = {row['股票代码']: row for _, row in quotes_df.iterrows()}
    result = {code: RealtimeQuote(**raw_data[code]) if code in raw_data else None for code in unique_codes}
    
    return RealtimeQuotesResponse(data=result)