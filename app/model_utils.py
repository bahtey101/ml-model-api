import os
import joblib
import logging
import pandas as pd
from .logging_config import setup_logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from app.preprocessing import preprocess_training_data, preprocess_input

setup_logging()
logger = logging.getLogger("app")

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'credit_scoring_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'store', 'rf_model.joblib')
PARAMS_PATH = os.path.splitext(MODEL_PATH)[0] + '_params.pkl'

_model = None
_params = None

def train_and_save_model():
    """
    Загружает данные, выполняет предобработку, обучает модель,
    выводит отчёт классификации и сохраняет модель на диск.
    """
    global _model
    logger.info("Loading training data from %s", DATA_PATH)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    processed = preprocess_training_data(df=data, model_path=MODEL_PATH)

    # Разделение на признаки и цель
    X = processed.drop(columns=["SeriousDlqin2yrs"])
    y = processed["SeriousDlqin2yrs"]

    # Тренировочный и тестовый сплит
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Обучение модели
    logger.info("RandomForestClassifier training")
    _model = RandomForestClassifier(n_estimators=100, random_state=42)
    _model.fit(X_train, y_train)

    # Предсказание и отчёт
    logger.info("Prediction based on a test sample and generation of a classification report")
    y_pred = _model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    logger.info("\n%s", report)

    # Сохранение модели
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(_model, MODEL_PATH)
    logger.info("The model is saved along the way: %s", MODEL_PATH)

    return report


def load_model():
    """
    Возвращает модель из кеша или загружает её из файла.
    """
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        logger.info("Loading model from %s", MODEL_PATH)
        _model = joblib.load(MODEL_PATH)
    return _model

def load_preprocessing_params():
    if not os.path.exists(PARAMS_PATH):
        raise FileNotFoundError(f"No preprocessing parameters found: {PARAMS_PATH}")
    return joblib.load(PARAMS_PATH)


def predict(features: list[float]) -> int:
    """Предсказание класса на основе загруженной модели"""
    model = load_model()
    params = joblib.load('store/rf_model_params.pkl')
    iqr_limits = params['iqr_limits']

    feature_names = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
        "NumberOfDependents"
    ]

    df = pd.DataFrame([features], columns=feature_names)
    df = preprocess_input(df, iqr_limits)

    return int(model.predict(df)[0])

def retrain():
    """Переобучение модели, обновление кеша и возвращение отчёта классификации"""
    global _model
    logger.info("Retraining of the model has begun")
    report = train_and_save_model()
    return report