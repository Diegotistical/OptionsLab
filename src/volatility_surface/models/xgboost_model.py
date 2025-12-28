# src/volatility_surface/models/xgb_model.py

import logging
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from xgboost import XGBRegressor, callback

from ..base import VolatilityModelBase
from ..utils.feature_engineering import engineer_features

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "moneyness",
    "log_moneyness",
    "time_to_maturity",
    "ttm_squared",
    "risk_free_rate",
    "historical_volatility",
    "volatility_skew",
]


class XGBVolatilityModel(VolatilityModelBase[XGBRegressor]):
    def __init__(
        self,
        feature_columns: Optional[list] = None,
        scaler_type: str = "standard",
        xgb_params: Optional[Dict] = None,
        callbacks: Optional[List[callback.TrainingCallback]] = None,
        enable_benchmark: bool = False,
    ):
        super().__init__(
            feature_columns=feature_columns or FEATURE_COLUMNS,
            enable_benchmark=enable_benchmark,
        )

        self.scaler_type = scaler_type
        self._initialize_scaler()
        self.callbacks = callbacks or []

        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="reg:squarederror",
            tree_method="auto",
            random_state=42,
            **(xgb_params or {}),
        )

    def _initialize_scaler(self):
        scalers = {
            "standard": StandardScaler(),
            "robust": RobustScaler(),
            "minmax": MinMaxScaler(),
        }
        if self.scaler_type not in scalers:
            raise ValueError(f"Unsupported scaler type: {self.scaler_type}")
        self.scaler = scalers[self.scaler_type]

    def _prepare_features(
        self, df: pd.DataFrame, fit_scaler: bool = False
    ) -> np.ndarray:
        features = engineer_features(df[self.feature_columns])
        if fit_scaler:
            scaled = self.scaler.fit_transform(features)
        else:
            scaled = self.scaler.transform(features)
        return scaled

    def _train_impl(self, df: pd.DataFrame, val_split: float = 0.2) -> Dict[str, float]:
        if "implied_volatility" not in df.columns:
            raise ValueError("'implied_volatility' column is required for training.")

        # Split dataset
        train_df, val_df = train_test_split(df, test_size=val_split, random_state=42)
        X_train = self._prepare_features(train_df, fit_scaler=True)
        y_train = train_df["implied_volatility"].values.astype(np.float64)
        X_val = self._prepare_features(val_df)
        y_val = val_df["implied_volatility"].values.astype(np.float64)

        # Fit model with callbacks
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            callbacks=self.callbacks,
        )

        # Predict validation
        y_val_pred = self.model.predict(X_val)

        from sklearn.metrics import (
            mean_absolute_error,
            mean_absolute_percentage_error,
            mean_squared_error,
            r2_score,
        )

        metrics = {
            "val_rmse": float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
            "val_mae": float(mean_absolute_error(y_val, y_val_pred)),
            "val_r2": float(r2_score(y_val, y_val_pred)),
            "val_mape": float(mean_absolute_percentage_error(y_val, y_val_pred)),
        }

        self.trained = True
        return metrics

    def _predict_impl(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare_features(df)
        return self.model.predict(X)

    def _save_model_impl(self, model_path: str, scaler_path: str) -> None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        logger.info(
            f"XGB model saved to {model_path} and scaler saved to {scaler_path}"
        )

    def _load_model_impl(self, model_path: str, scaler_path: str) -> None:
        if not os.path.isfile(model_path) or not os.path.isfile(scaler_path):
            raise FileNotFoundError("Model or scaler file not found.")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.trained = True
        logger.info(
            f"XGB model loaded from {model_path} and scaler loaded from {scaler_path}"
        )
