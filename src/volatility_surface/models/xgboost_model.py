# src/volatility_surface/models/xgboost_model.py

import os
import threading
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
from xgboost import XGBRegressor, callback
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging

from ..utils.feature_engineering import engineer_features
from ..common.validation import check_required_columns


logger = logging.getLogger(__name__)


class ModelNotTrainedError(RuntimeError):
    pass


class XGBoostModel:
    """
    Robust XGBoost regression model for volatility surface fitting.
    Thread-safe, with input validation and detailed metrics.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: int = 42,
        early_stopping_rounds: int = 10,
        eval_metric: str = "rmse",
        model_dir: str = "models/saved_models",
        verbose: bool = False,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.model_dir = model_dir
        self.verbose = verbose

        self.scaler = StandardScaler()
        self.model: Optional[XGBRegressor] = None
        self.trained = False

        self._lock = threading.RLock()

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        required_cols = ['implied_volatility']
        check_required_columns(df, required_cols)

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        # Engineer and scale features
        features = engineer_features(df)
        return self.scaler.transform(features)

    def train(self, df: pd.DataFrame, val_split: float = 0.2) -> Dict[str, Any]:
        with self._lock:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Input df must be a pandas DataFrame")

            self._validate_dataframe(df)

            train_df, val_df = train_test_split(df, test_size=val_split, random_state=self.random_state)

            X_train = engineer_features(train_df)
            y_train = train_df['implied_volatility'].values

            X_val = engineer_features(val_df)
            y_val = val_df['implied_volatility'].values

            # Fit scaler on train only
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Instantiate model
            self.model = XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                verbosity=1 if self.verbose else 0,
                n_jobs=-1,
            )

            # Setup callbacks
            callbacks = []
            if self.early_stopping_rounds > 0:
                callbacks.append(
                    callback.EarlyStopping(rounds=self.early_stopping_rounds, save_best=True)
                )

            logger.info("Starting XGBoost training...")
            self.model.fit(
                X_train_scaled,
                y_train,
                eval_set=[(X_val_scaled, y_val)],
                eval_metric=self.eval_metric,
                callbacks=callbacks,
                verbose=self.verbose,
            )
            logger.info("Training completed.")

            self.trained = True

            # Predictions
            train_preds = self.model.predict(X_train_scaled)
            val_preds = self.model.predict(X_val_scaled)

            metrics = {
                "train_rmse": np.sqrt(mean_squared_error(y_train, train_preds)),
                "val_rmse": np.sqrt(mean_squared_error(y_val, val_preds)),
                "train_r2": r2_score(y_train, train_preds),
                "val_r2": r2_score(y_val, val_preds),
            }
            return metrics

    def predict_volatility(self, df: pd.DataFrame) -> np.ndarray:
        with self._lock:
            if not self.trained or self.model is None:
                raise ModelNotTrainedError("Model must be trained before prediction")

            if not isinstance(df, pd.DataFrame):
                raise TypeError("Input df must be a pandas DataFrame")

            features_scaled = self._prepare_features(df)
            return self.model.predict(features_scaled)

    def save_model(self, model_dir: Optional[str] = None) -> Dict[str, str]:
        with self._lock:
            model_dir = model_dir or self.model_dir
            os.makedirs(model_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            model_path = os.path.join(model_dir, f"xgb_model_{ts}.joblib")
            scaler_path = os.path.join(model_dir, f"xgb_scaler_{ts}.joblib")

            if self.model is None:
                raise ModelNotTrainedError("No trained model to save")

            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)

            logger.info(f"Saved model to {model_path} and scaler to {scaler_path}")

            return {"model": model_path, "scaler": scaler_path}

    def load_model(self, model_path: str, scaler_path: str) -> None:
        with self._lock:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.trained = True

            logger.info(f"Loaded model from {model_path} and scaler from {scaler_path}")
