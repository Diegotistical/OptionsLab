# src/volatility_surface/models/random_forest.py

import logging
import os
import threading
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ..base import VolatilityModelBase
from ..utils.feature_engineering import engineer_features

logger = logging.getLogger(__name__)


class RandomForestVolatilityModel(VolatilityModelBase[RandomForestRegressor]):
    def __init__(
        self,
        feature_columns: Optional[list] = None,
        rf_params: Optional[Dict] = None,
        scaler_type: str = "standard",
        enable_benchmark: bool = False,
    ):
        super().__init__(
            feature_columns=feature_columns, enable_benchmark=enable_benchmark
        )
        self._lock = threading.RLock()
        self.model = RandomForestRegressor(
            **(
                rf_params
                or {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "n_jobs": -1,
                    "random_state": 42,
                }
            )
        )
        self.scaler_type = scaler_type
        self._initialize_scaler()

    def _initialize_scaler(self):
        scalers = {
            "standard": StandardScaler(),
            "robust": RobustScaler(),
            "minmax": MinMaxScaler(),
        }
        if self.scaler_type not in scalers:
            raise ValueError(f"Unsupported scaler type: {self.scaler_type}")
        self.scaler = scalers[self.scaler_type]

    def _prepare_features(self, df: pd.DataFrame, fit_scaler=False):
        df = df.copy()
        df["time_to_maturity"] = np.maximum(df["time_to_maturity"], 1e-5)
        features = engineer_features(df)
        X = features.values
        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        return X

    def _train_impl(self, df: pd.DataFrame, val_split: float = 0.2) -> Dict[str, float]:
        if df is None:
            raise ValueError("DataFrame `df` must be provided for training.")
        if "implied_volatility" not in df.columns:
            raise ValueError("'implied_volatility' column is required for training.")

        with self._lock:
            self._on_train_start(df)

            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            n_val = int(len(df) * val_split)
            train_df, val_df = df.iloc[:-n_val], df.iloc[-n_val:]

            X_train = self._prepare_features(train_df, fit_scaler=True)
            y_train = train_df["implied_volatility"].values.astype(np.float64)

            X_val = self._prepare_features(val_df)
            y_val = val_df["implied_volatility"].values.astype(np.float64)

            self.model.fit(X_train, y_train)

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
            self._on_train_end(metrics)
            return metrics

    def _predict_impl(self, df: pd.DataFrame) -> np.ndarray:
        if df is None:
            raise ValueError("DataFrame `df` must be provided for prediction.")
        with self._lock:
            X = self._prepare_features(df)
            return self.model.predict(X)

    def _save_model_impl(self, model_path: str, scaler_path: str) -> None:
        with self._lock:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            logger.info(
                f"RandomForest model saved to {model_path} and scaler saved to {scaler_path}"
            )

    def _load_model_impl(self, model_path: str, scaler_path: str) -> None:
        with self._lock:
            if not os.path.isfile(model_path) or not os.path.isfile(scaler_path):
                raise FileNotFoundError("Model or scaler file not found.")
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.trained = True
            logger.info(
                f"RandomForest model loaded from {model_path} and scaler loaded from {scaler_path}"
            )
