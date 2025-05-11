import pandas as pd
import numpy as np
from sklearn.utils import resample
import logging
from .logging_config import setup_logging
import joblib
import os

setup_logging()
logger = logging.getLogger("app")

iqr_cols = [
    'RevolvingUtilizationOfUnsecuredLines',
    'DebtRatio',
    'MonthlyIncome',
    'NumberRealEstateLoansOrLines'
]

def mark_iqr_outliers(series: pd.Series, k: float = 1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    return (series < lower) | (series > upper), lower, upper

def preprocess_training_data(
    df: pd.DataFrame,
    target_col: str = 'SeriousDlqin2yrs',
    target_ratio: float = 0.4,
    model_path: str | None = None
) -> pd.DataFrame:
    df_cleaned = df.copy()
    df_cleaned.fillna(0.0, inplace=True)

    logger.info("IQR Outlier treatment")
    iqr_limits = {}
    for col in iqr_cols:
        mask, lo, hi = mark_iqr_outliers(df_cleaned[col])
        df_cleaned[col] = df_cleaned[col].clip(lower=lo, upper=hi)
        iqr_limits[col] = (lo, hi)
        logger.info(f" - {col}: emissions {mask.sum()} ({mask.mean() * 100:.1f}%)")

    logger.info("Balancing classes")
    data_majority = df_cleaned[df_cleaned[target_col] == 0]
    data_minority = df_cleaned[df_cleaned[target_col] == 1]

    n_minority = len(data_minority)
    total_needed = int(n_minority / target_ratio)
    n_majority = total_needed - n_minority

    data_majority_downsampled = resample(
        data_majority,
        replace=False,
        n_samples=n_majority,
        random_state=42
    )

    data_balanced = pd.concat([data_majority_downsampled, data_minority], axis=0).sample(frac=1, random_state=42)

    logger.info("Removing highly correlated features")
    corr_matrix = data_balanced.drop(columns=[target_col]).corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column].abs() > 0.9)]
    logger.info(f"Features removed: {to_drop}")

    data_balanced.drop(columns=to_drop, inplace=True)

    feature_names = [col for col in data_balanced.columns if col != target_col]

    if model_path:
        params_path = os.path.splitext(model_path)[0] + '_params.pkl'
        joblib.dump({
            'iqr_limits': iqr_limits,
            'feature_names': feature_names,
            'dropped_cols': to_drop
        }, params_path)
        logger.info("The preprocessing parameters are saved in %s", params_path)

    return data_balanced

def preprocess_input(
    df: pd.DataFrame,
    iqr_limits: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    df = df.copy()
    df.fillna(0.0, inplace=True)

    for col, (lo, hi) in iqr_limits.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)

    return df