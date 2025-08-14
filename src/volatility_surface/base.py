# src/volatility_surface/base.py

import threading
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, TypeVar, Generic
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

ModelType = TypeVar("ModelType")

# Configure logging
logger = logging.getLogger(__name__)


def benchmark_method(enabled_attr: str):
    """
    Decorator to benchmark execution time of methods if enabled.

    Args:
        enabled_attr: Name of the boolean attribute on self to enable benchmarking.

    Usage:
        @benchmark_method("_enable_benchmark")
        def train(self, ...):
            ...
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if getattr(self, enabled_attr, False):
                start = time.perf_counter()
                result = func(self, *args, **kwargs)
                elapsed = time.perf_counter() - start
                self._benchmark_timings[func.__name__] = elapsed
                logger.info(f"Benchmark: {func.__name__} took {elapsed:.4f} seconds")
                return result
            else:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


class VolatilityModelBase(ABC, Generic[ModelType]):
    """
    Thread-safe, hook-enabled, benchmarkable base class for volatility surface models.

    Features:
        - Thread-safe training, prediction, evaluation, save/load
        - Benchmarking support per method
        - Lifecycle hooks for subclass extensions
        - Feature validation and scaling
        - Abstract API to implement train, predict, save, and load logic
    """

    def __init__(self, feature_columns: Optional[List[str]] = None, enable_benchmark: bool = False) -> None:
        """
        Initialize the base volatility model.

        Args:
            feature_columns: List of column names to be used as model features.
            enable_benchmark: Enable timing benchmarks for methods decorated with @benchmark_method.
        """
        self._lock = threading.RLock()
        self.feature_columns: List[str] = feature_columns or [
            'moneyness',
            'log_moneyness',
            'time_to_maturity',
            'ttm_squared',
            'risk_free_rate',
            'historical_volatility',
            'volatility_skew'
        ]
        self.scaler: StandardScaler = StandardScaler()
        self.model: Optional[ModelType] = None
        self.trained: bool = False
        self._benchmark_timings: Dict[str, float] = {}
        self._enable_benchmark = enable_benchmark

    #  Lifecycle hooks (no-op) 
    def _on_train_start(self, df: pd.DataFrame) -> None: ...
    def _on_train_end(self, metrics: Dict[str, float]) -> None: ...
    def _on_predict_start(self, df: pd.DataFrame) -> None: ...
    def _on_predict_end(self, predictions: np.ndarray) -> None: ...
    def _on_evaluate_start(self, df: pd.DataFrame) -> None: ...
    def _on_evaluate_end(self, metrics: Dict[str, float]) -> None: ...
    def _on_save_model_start(self, model_path: str, scaler_path: str) -> None: ...
    def _on_save_model_end(self, model_path: str, scaler_path: str) -> None: ...
    def _on_load_model_start(self, model_path: str, scaler_path: str) -> None: ...
    def _on_load_model_end(self, model_path: str, scaler_path: str) -> None: ...

    #  Public API 
    @benchmark_method("_enable_benchmark")
    def train(self, df: pd.DataFrame, val_split: float = 0.2) -> Dict[str, float]:
        """
        Train the model using the provided DataFrame.

        Args:
            df: DataFrame containing feature columns and optionally target values.
            val_split: Fraction of data to use as validation set.

        Returns:
            Dictionary of training metrics (e.g., loss, RMSE).

        Raises:
            RuntimeError: If training fails in subclass implementation.
        """
        with self._lock:
            self._on_train_start(df)
            metrics = self._train_impl(df, val_split)
            self.trained = True
            self._on_train_end(metrics)
            return metrics

    @abstractmethod
    def _train_impl(self, df: pd.DataFrame, val_split: float) -> Dict[str, float]:
        """
        Subclass must implement actual training logic.

        Args:
            df: DataFrame containing all feature columns required by `self.feature_columns`.
                May also include target/ground truth columns (e.g., implied volatility).
            val_split: Fraction of data to use as a validation set (between 0 and 1).

        Returns:
            Dictionary of training metrics, such as:
                - 'loss': float
                - 'rmse': float
                - 'mae': float
            Keys and metrics can vary depending on the model, but should be meaningful.

        Raises:
            RuntimeError: If training fails or encounters numerical issues.
        """

    @benchmark_method("_enable_benchmark")
    def predict_volatility(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict volatility for input data.

        Args:
            df: DataFrame containing required feature columns.

        Returns:
            NumPy array of predicted volatilities.

        Raises:
            RuntimeError: If model has not been trained.
        """
        with self._lock:
            self._on_predict_start(df)
            self._assert_trained()
            predictions = self._predict_impl(df)
            self._on_predict_end(predictions)
            return predictions

    @abstractmethod
    def _predict_impl(self, df: pd.DataFrame) -> np.ndarray:
        """
        Subclass must implement actual prediction logic.

        Args:
            df: DataFrame containing all feature columns required by `self.feature_columns`.

        Returns:
            NumPy array of predicted volatilities, shape (n_samples,).

        Raises:
            RuntimeError: If model is uninitialized or encounters prediction errors.
        """

    @benchmark_method("_enable_benchmark")
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model predictions against ground truth.

        Args:
            df: DataFrame containing feature columns and 'implied_volatility' column.

        Returns:
            Dictionary of evaluation metrics: rmse, mae, r2, mape.

        Raises:
            RuntimeError: If model is not trained.
            ValueError: If required columns are missing or predictions mismatch shape.
        """
        with self._lock:
            self._on_evaluate_start(df)
            if not self.trained:
                raise RuntimeError("Model must be trained before evaluation.")

            self._validate_features(df)
            if 'implied_volatility' not in df.columns:
                raise ValueError("Ground truth 'implied_volatility' column is missing.")

            y_true = df['implied_volatility'].values.astype(np.float64)
            y_pred = self.predict_volatility(df)

            if y_pred.shape != y_true.shape:
                raise ValueError(f"Prediction shape {y_pred.shape} does not match ground truth shape {y_true.shape}.")

            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

            metrics = {
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred)),
                'mape': float(mean_absolute_percentage_error(y_true, y_pred))
            }
            self._on_evaluate_end(metrics)
            return metrics

    def save_model(self, model_path: str, scaler_path: str) -> None:
        """
        Save the model and scaler to disk.

        Args:
            model_path: File path to save the model.
            scaler_path: File path to save the scaler.
        """
        with self._lock:
            self._on_save_model_start(model_path, scaler_path)
            self._save_model_impl(model_path, scaler_path)
            self._on_save_model_end(model_path, scaler_path)

    @abstractmethod
    def _save_model_impl(self, model_path: str, scaler_path: str) -> None:
        """
        Subclass must implement logic to persist the model and scaler to disk.

        Args:
            model_path: Path to save model parameters or serialized model.
            scaler_path: Path to save fitted scaler parameters.

        Raises:
            IOError: If saving fails.
        """

    def load_model(self, model_path: str, scaler_path: str) -> None:
        """
        Load a trained model and scaler from disk.

        Args:
            model_path: File path to load the model from.
            scaler_path: File path to load the scaler from.
        """
        with self._lock:
            self._on_load_model_start(model_path, scaler_path)
            self._load_model_impl(model_path, scaler_path)
            self.trained = True
            self._on_load_model_end(model_path, scaler_path)

    @abstractmethod
    def _load_model_impl(self, model_path: str, scaler_path: str) -> None:
        """
        Subclass must implement logic to load the model and scaler from disk.

        Args:
            model_path: Path to load model parameters or serialized model.
            scaler_path: Path to load scaler parameters.

        Raises:
            IOError: If loading fails or files are corrupted.
        """

    #  Internal utilities
    def _validate_features(self, df: pd.DataFrame) -> None:
        """Ensure all required features exist, are numeric, and contain finite values."""
        missing = set(self.feature_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required feature columns: {sorted(missing)}")

        non_numeric = [
            col for col in self.feature_columns if not pd.api.types.is_numeric_dtype(df[col])
        ]
        if non_numeric:
            raise ValueError(f"Non-numeric feature columns detected: {sorted(non_numeric)}")

        if df[self.feature_columns].isnull().values.any():
            raise ValueError("NaN values detected in feature columns.")

        if not np.isfinite(df[self.feature_columns].values).all():
            raise ValueError("Infinite or NaN values detected in feature columns.")

    def _prepare_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """
        Validate and scale features.

        Args:
            df: DataFrame with feature columns.
            fit_scaler: Fit scaler if True, else transform existing scaler.

        Returns:
            Scaled NumPy array of features.
        """
        self._validate_features(df)
        X = df[self.feature_columns].values.astype(np.float64)

        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            if not self.trained:
                raise RuntimeError("Scaler has not been fitted yet. Train model first.")
            X_scaled = self.scaler.transform(X)

        return X_scaled

    def _assert_trained(self) -> None:
        """Raise if model is not trained or initialized."""
        if not self.trained or self.model is None:
            raise RuntimeError("Model is not trained or initialized.")

    def get_benchmark_timings(self) -> Dict[str, float]:
        """
        Return a dictionary of method names and last benchmark timings (seconds).
        """
        return dict(self._benchmark_timings)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(trained={self.trained}, features={self.feature_columns})>"
