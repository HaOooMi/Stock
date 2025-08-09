from fastapi import APIRouter,Depends,Path, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date
from sqlalchemy.engine import Connection
import numpy as np
import pandas as pd

from utils import get_mysql_engine
import sys
sys.path.append('d:/vscode projects/stock')
from stock_info.stock_meta_akshare import get_financial_info_mysql

class FinancialReport(BaseModel):

    股票代码: str
    报告期: date
    净利润: Optional[float] = None
    净利润同比增长率: Optional[float] = None
    扣非净利润: Optional[float] = None
    扣非净利润同比增长率: Optional[float] = None
    营业总收入: Optional[float] = None
    营业总收入同比增长率: Optional[float] = None
    基本每股收益: Optional[float] = None
    每股净资产: Optional[float] = None
    每股资本公积金: Optional[float] = None
    每股未分配利润: Optional[float] = None
    每股经营现金流: Optional[float] = None
    销售净利率: Optional[float] = None
    净资产收益率: Optional[float] = None
    净资产收益率_摊薄: Optional[float] 
    营业周期: Optional[float] = None
    应收账款周转天数: Optional[float] = None
    流动比率: Optional[float] = None
    速动比率: Optional[float] = None
    保守速动比率: Optional[float] = None
    产权比率: Optional[float] = None
    资产负债率: Optional[float] = None

class FinancialReportsResponse(BaseModel):

    stock_code: str
    data: List[FinancialReport]

financial=APIRouter()

@financial.get("/financial-reports/{stock_code}", response_model=FinancialReportsResponse, tags=["股票财务数据"])
def get_financial_reports(
    stock_code: str = Path(..., description="单个股票代码", example="000001"),
    db: Connection = Depends(get_mysql_engine)
    ):
 
    reports_df = get_financial_info_mysql(db, stock_code)
    if reports_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"在数据库中未找到股票 {stock_code} 的财务报告数据。"
        )
    float_columns = [
        '净利润', '净利润同比增长率', '扣非净利润', '扣非净利润同比增长率',
        '营业总收入', '营业总收入同比增长率', '基本每股收益', '每股净资产',
        '每股资本公积金', '每股未分配利润', '每股经营现金流', '销售净利率',
        '净资产收益率', '净资产收益率_摊薄', '营业周期', '应收账款周转天数',
        '流动比率', '速动比率', '保守速动比率', '产权比率', '资产负债率'
    ]

    # 清洗：替换 inf 和 NaN 为 None
    for col in float_columns:
        if col in reports_df.columns:
            reports_df[col] = reports_df[col].replace([np.inf, -np.inf], None)
            reports_df[col] = np.where(reports_df[col].isna(), None, reports_df[col])

    # 调试：打印清洗后数据
    print(f"清洗后数据: {reports_df}")
    return FinancialReportsResponse(
        stock_code=stock_code,
        data=reports_df.to_dict('records')
    )