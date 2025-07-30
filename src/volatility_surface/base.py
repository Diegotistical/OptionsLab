# src/volatility_surface/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import os
import logging
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

class VolatilityModelBase(ABC):
    """
    Abstract base class for volatility surface models.
    """

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.trained = False
        self.feature_columns = feature_columns or [
            'moneyness', 'log_moneyness', 'time_to_maturity',
            'ttm_squared', 'risk_free_rate',
            'historical_volatility', 'volatility_skew'
        ]
        self.scaler = StandardScaler()
        self.model: Optional[Any] = None # Placeholder for the model

    @abstractmethod
    def train(self, df: pd.DataFrame, val_split: float = 0.2) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict_volatility(self, df: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def save_model(self, model_dir: str = 'models/saved_models') -> Dict[str, str]:
        pass

    @abstractmethod
    def load_model(self, model_path: str, scaler_path: str) -> None:
        pass

    def validate_input(self, df: pd.DataFrame) -> None:
        missing = set(self.feature_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model on raw DataFrame. Requires 'implied_volatility' as ground truth.
        """
        self.validate_input(df)
        if 'implied_volatility' not in df.columns:
            raise ValueError("Ground truth 'implied_volatility' column is missing.")

        y = df['implied_volatility'].values
        y_pred = self.predict_volatility(df)  # << usa el método abstracto

        return {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mape': mean_absolute_percentage_error(y, y_pred)
        }

    def _mark_trained(self):
        self.trained = True

    def _check_model_initialized(self):
        if self.model is None:
            raise RuntimeError("Underlying model is not initialized.")

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        self.validate_input(df)
        X = df[self.feature_columns].values
        logger.debug(f"Preparing features with shape: {X.shape}")
        return self.scaler.transform(X) if self.trained else self.scaler.fit_transform(X)

    def __repr__(self): # logging or debugging purposes
        return f"<{self.__class__.__name__}(trained={self.trained}, features={self.feature_columns})>"
