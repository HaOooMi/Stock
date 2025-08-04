from fastapi import APIRouter,Depends
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date
from sqlalchemy.engine import Connection

from stock_info.utils import get_mysql_engine
from stock_info.stock_meta_akshare import get_base_info


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
def get_basic_info(stock_code: str, db: Connection = Depends(get_mysql_engine)):
    raw_data = get_base_info(db, (stock_code,))
    if stock_code in raw_data:
        return StockBasicInfo(**raw_data[stock_code])
    else:
        return None