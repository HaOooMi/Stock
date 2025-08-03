from fastapi import FastAPI
import uvicorn

app=FastAPI()

app.include_router()