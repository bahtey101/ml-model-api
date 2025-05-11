from fastapi import FastAPI, HTTPException, Body
from contextlib import asynccontextmanager
import logging
import os

from app.logging_config import setup_logging
from app.model_utils import predict, train_and_save_model, retrain, load_model
from pydantic import BaseModel

setup_logging()
logger = logging.getLogger("app")

class Features(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: float
    NumberOfTime30_59DaysPastDueNotWorse: float
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: float
    NumberRealEstateLoansOrLines: float
    NumberOfDependents: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    model_path = os.path.join(os.getcwd(), "store", "rf_model.joblib")
    if not os.path.exists(model_path):
        logger.info("Model not found, starting training")
        try:
            train_and_save_model()
        except Exception as e:
            logger.error("Error when training the model: %s", e)
            raise
    else:
        logger.info("Loading an existing model")
        try:
            load_model()
        except Exception as e:
            logger.error("Error loading the model: %s", e)
            raise
    yield
    # Shutdown logic
    logger.info("The application is shutting down")


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
def api_predict(data: Features):
    try:
        # Преобразуем объект в список фичей в нужном порядке
        feature_order = [
            data.RevolvingUtilizationOfUnsecuredLines,
            data.age,
            data.NumberOfTime30_59DaysPastDueNotWorse,
            data.DebtRatio,
            data.MonthlyIncome,
            data.NumberOfOpenCreditLinesAndLoans,
            data.NumberRealEstateLoansOrLines,
            data.NumberOfDependents,
        ]

        result = predict(feature_order)
        logger.info("Prediction for %s: %s", feature_order, result)
        return {"prediction": result}
    except Exception as e:
        logger.error("Error when prediction: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
def api_retrain():
    try:
        report = retrain()
        logger.info("Retrain report: \n%s", report)
        return {"classification_report": report}
    except Exception as e:
        logger.error("Error when training: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
