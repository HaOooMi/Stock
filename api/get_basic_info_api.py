from fastapi import APIRouter,Depends,Path
from pydantic import BaseModel
from typing import Optional
from datetime import date
from sqlalchemy.engine import Connection

import sys
sys.path.append('d:/vscode projects/stock')
from stock_info.utils import get_mysql_engine
from stock_info.stock_meta_akshare import get_basic_info_mysql


class StockBasicInfo(BaseModel):
    股票代码: Optional[str] = None
    股票简称: Optional[str] = None
    总股本: Optional[float] = None
    流通股: Optional[float] = None
    总市值: Optional[float] = None
    流通市值: Optional[float] = None
    所属行业: Optional[str] = None
    上市时间: Optional[date] = None
        


basic=APIRouter()

@basic.get("/basical-info/{stock_code}", response_model=Optional[StockBasicInfo], tags=["股票基本数据"])
def get_basic_info(
    stock_code: str= Path(..., description="单个股票代码", example="000001"), 
    db: Connection = Depends(get_mysql_engine)):
    raw_data = get_base_info(db, (stock_code,))
    if stock_code in raw_data:
        return StockBasicInfo(**raw_data[stock_code])
    else:
        return None