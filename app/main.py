from fastapi import FastAPI, HTTPException, Body
from contextlib import asynccontextmanager
import logging
from .logging_config import setup_logging
import os

setup_logging()
logger = logging.getLogger("app")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    if not os.path.exists("store/model.joblib"):
        logger.info('Model train')
    yield
    
    # Shutdown logic
    print("App is shutting down")

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
def predict(data: object = Body(...)):
    print(data)
    return {"prediction": False}

@app.post("/retrain")
def retrain_model():
    print('retrain')
