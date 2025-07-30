# src/volatility_surface/models/svr_model.py

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from ..utils import engineer_features, validate_domain
from ..base import VolatilityModelBase


class SVRModel(VolatilityModelBase):
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale', random_state=42):
        super().__init__()
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        self.trained = False

    def train(self, df: pd.DataFrame, val_split: float = 0.2) -> dict:
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

        return {
            'train_rmse': np.sqrt(mean_squared_error(y_train, self.model.predict(X_train_scaled))),
            'val_rmse': np.sqrt(mean_squared_error(y_val, self.model.predict(X_val_scaled))),
            'train_r2': r2_score(y_train, self.model.predict(X_train_scaled)),
            'val_r2': r2_score(y_val, self.model.predict(X_val_scaled)),
            'validity': validate_domain(X_val, X_train)
        }

    def predict_volatility(self, df: pd.DataFrame) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")

        X = engineer_features(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save_model(self, model_dir: str = 'models/saved_models') -> dict:
        os.makedirs(model_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = os.path.join(model_dir, f'svr_model_{ts}.joblib')
        scaler_path = os.path.join(model_dir, f'svr_scaler_{ts}.joblib')

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        return {'model': model_path, 'scaler': scaler_path}

    def load_model(self, model_path: str, scaler_path: str) -> None:
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.trained = True
