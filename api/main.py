from fastapi import FastAPI
import uvicorn

from get_basic_info_api import basic
from get_financial_info_api import financial
from get_historical_data_api import history
from get_realtime_data_api import now

stock=FastAPI()

stock.include_router(basic)
stock.include_router(financial)
stock.include_router(history)
stock.include_router(now)

if __name__ == '__main__':
    uvicorn.run("main:stock", host="127.0.0.1", port=8000,reload=True)