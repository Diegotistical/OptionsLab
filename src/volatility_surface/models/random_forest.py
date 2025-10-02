# src/volatility_surface/models/random_forest.py

from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import logging
from ..base import VolatilityModelBase
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class RandomForestVolatilityModel(VolatilityModelBase):
    def __init__(self,
                 feature_columns: Optional[list] = None,
                 rf_params: Optional[Dict] = None,
                 enable_benchmark: bool = False):
        super().__init__(feature_columns=feature_columns, enable_benchmark=enable_benchmark)
        self.model = RandomForestRegressor(**(rf_params or {
            'n_estimators': 100,
            'max_depth': 10,
            'n_jobs': -1,
            'random_state': 42
        }))
        self.scaler = StandardScaler()  # ✅ ensure scaler exists

    def _train_impl(self, df: pd.DataFrame, val_split: float) -> Dict[str, float]:
        if 'implied_volatility' not in df.columns:
            raise ValueError("'implied_volatility' column is required for training.")

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n_val = int(len(df) * val_split)
        train_df = df.iloc[:-n_val]
        val_df = df.iloc[-n_val:]

        X_train = self._prepare_features(train_df, fit_scaler=True)
        y_train = train_df['implied_volatility'].values.astype(np.float64)

        X_val = self._prepare_features(val_df)
        y_val = val_df['implied_volatility'].values.astype(np.float64)

        self.model.fit(X_train, y_train)
        self.trained = True  # ✅ mark trained

        y_val_pred = self.model.predict(X_val)

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

        metrics = {
            'val_rmse': float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
            'val_mae': float(mean_absolute_error(y_val, y_val_pred)),
            'val_r2': float(r2_score(y_val, y_val_pred)),
            'val_mape': float(mean_absolute_percentage_error(y_val, y_val_pred))
        }
        return metrics

    def _predict_impl(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare_features(df)
        return self.model.predict(X)

    def _save_model_impl(self, model_path: str, scaler_path: str) -> None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Model saved to {model_path} and scaler saved to {scaler_path}")

    def _load_model_impl(self, model_path: str, scaler_path: str) -> None:
        if not os.path.isfile(model_path) or not os.path.isfile(scaler_path):
            raise FileNotFoundError("Model or scaler file not found.")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.trained = True
        logger.info(f"Model loaded from {model_path} and scaler loaded from {scaler_path}")
