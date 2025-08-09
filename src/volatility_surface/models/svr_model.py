# src / volatility_surface / models / svr_model.py

import threading
import os
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from ..base import VolatilityModelBase
from ..utils.feature_engineering import engineer_features
from ..utils.arbitrage_utils import validate_domain


class SVRModel(VolatilityModelBase):
    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        epsilon: float = 0.1,
        gamma: str = 'scale',
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(feature_columns=None, enable_benchmark=True)
        self._lock = threading.RLock()

        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, gamma=self.gamma)
        self.trained = False

    def _train_impl(self, df: pd.DataFrame, val_split: float = 0.2) -> Dict[str, float]:
        with self._lock:
            self._on_train_start(df)
            self.validate_input(df)  # will raise if features missing

            train_df, val_df = train_test_split(df, test_size=val_split, random_state=self.random_state)
            X_train = engineer_features(train_df)
            y_train = train_df['implied_volatility'].values
            X_val = engineer_features(val_df)
            y_val = val_df['implied_volatility'].values

            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            self.model.fit(X_train_scaled, y_train)
            self.trained = True

            train_preds = self.model.predict(X_train_scaled)
            val_preds = self.model.predict(X_val_scaled)

            metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_preds)),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_preds)),
                'train_r2': r2_score(y_train, train_preds),
                'val_r2': r2_score(y_val, val_preds),
                'validity': validate_domain(X_val, X_train)
            }
            self._on_train_end(metrics)
            return metrics

    def predict_volatility(self, df: pd.DataFrame) -> np.ndarray:
        with self._lock:
            self._on_predict_start(df)
            if not self.trained:
                raise RuntimeError("Model must be trained before prediction")
            self.validate_input(df)

            X = engineer_features(df)
            X_scaled = self.scaler.transform(X)
            preds = self.model.predict(X_scaled)
            self._on_predict_end(preds)
            return preds

    def _save_model_impl(self, model_path: str, scaler_path: str) -> None:
        with self._lock:
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)

    def _load_model_impl(self, model_path: str, scaler_path: str) -> None:
        with self._lock:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.trained = True
