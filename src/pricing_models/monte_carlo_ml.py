# src/pricing_models/monte_carlo_ml.py
"""
Machine Learning surrogate model for Monte Carlo option pricing.

This module provides a fast ML-based surrogate that learns to predict
option prices and Greeks. Training uses Black-Scholes formula for speed,
while predictions approximate what Monte Carlo would produce.

Features:
    - LightGBM for fast training and inference
    - **Vectorized training data generation using Black-Scholes**
    - Feature engineering (moneyness, normalized time, etc.)
    - Model persistence (save/load trained models)
    - Support for both calls and puts

Example:
    >>> from src.pricing_models.monte_carlo_ml import MonteCarloMLSurrogate
    >>> surrogate = MonteCarloMLSurrogate()
    >>> surrogate.fit(n_samples=5000)  # <2 seconds with BS-based training
    >>> predictions = surrogate.predict(test_df)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

# Try to import LightGBM, fall back to sklearn if not available
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Try to import joblib for model persistence
try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

__all__ = ["MonteCarloMLSurrogate", "LIGHTGBM_AVAILABLE"]

# Configure logger
logger = logging.getLogger(__name__)


# =============================================================================
# Vectorized Black-Scholes for ultra-fast training data
# =============================================================================


def _bs_d1_d2(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Black-Scholes d1 and d2 (vectorized)."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def _bs_price_vectorized(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray,
    is_call: bool = True,
) -> np.ndarray:
    """
    Vectorized Black-Scholes pricing.

    Runs in O(n) with pure NumPy operations - no loops.
    """
    d1, d2 = _bs_d1_d2(S, K, T, r, sigma, q)

    if is_call:
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return np.maximum(price, 0.0)


def _bs_delta_vectorized(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray,
    is_call: bool = True,
) -> np.ndarray:
    """Vectorized Black-Scholes delta."""
    d1, _ = _bs_d1_d2(S, K, T, r, sigma, q)
    discount = np.exp(-q * T)

    if is_call:
        return discount * norm.cdf(d1)
    else:
        return discount * (norm.cdf(d1) - 1)


def _bs_gamma_vectorized(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Vectorized Black-Scholes gamma (same for calls/puts)."""
    d1, _ = _bs_d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def _bs_vega_vectorized(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Vectorized Black-Scholes vega (same for calls/puts)."""
    d1, _ = _bs_d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


class MonteCarloMLSurrogate:
    """
    Machine Learning surrogate for fast option pricing and Greeks.

    This class trains an ML model to approximate option pricing. Training
    uses vectorized Black-Scholes for speed (O(n) vs O(n×sims)), while the
    model learns patterns that generalize well.

    Attributes:
        model: The trained ML model (LightGBM or sklearn fallback).
        trained (bool): Whether the model has been trained.
        feature_names (List[str]): Names of input features.
        target_names (List[str]): Names of prediction targets.

    Example:
        >>> surrogate = MonteCarloMLSurrogate()
        >>> surrogate.fit(n_samples=5000)  # <2 seconds!
        >>> result = surrogate.predict_single(100, 100, 1.0, 0.05, 0.2, 0.0)
        >>> print(f"Price: {result['price']:.4f}")
    """

    # Feature columns
    FEATURE_COLUMNS = ["S", "K", "T", "r", "sigma", "q"]
    ENGINEERED_FEATURES = ["moneyness", "log_moneyness", "sqrt_T", "T_sigma"]
    TARGET_COLUMNS = ["price", "delta", "gamma"]

    def __init__(
        self,
        num_simulations: int = 50000,
        num_steps: int = 100,
        seed: Optional[int] = 42,
        use_numba: bool = True,
        model_type: Literal["lightgbm", "sklearn"] = "lightgbm",
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_jobs: int = -1,
    ) -> None:
        """
        Initialize the ML surrogate.

        Args:
            num_simulations: (Legacy) Not used with BS-based training.
            num_steps: (Legacy) Not used with BS-based training.
            seed: Random seed for reproducibility.
            use_numba: (Legacy) Not used with BS-based training.
            model_type: ML model type - 'lightgbm' (fast) or 'sklearn'.
            n_estimators: Number of boosting rounds / trees.
            max_depth: Maximum tree depth.
            learning_rate: Learning rate for gradient boosting.
            n_jobs: Number of parallel jobs (-1 for all cores).
        """
        self.seed = seed
        self.n_jobs = n_jobs

        # Build ML model
        self.model_type = model_type if LIGHTGBM_AVAILABLE else "sklearn"
        self.model = self._build_model(n_estimators, max_depth, learning_rate)
        self.trained = False

        # Metadata
        self.feature_names = self.FEATURE_COLUMNS + self.ENGINEERED_FEATURES
        self.target_names = self.TARGET_COLUMNS
        self._training_stats: Dict[str, Any] = {}

        logger.info(
            f"Initialized MonteCarloMLSurrogate with model_type='{self.model_type}'"
        )

    def _build_model(
        self,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
    ) -> Any:
        """Build the ML model pipeline."""
        if self.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            base_model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=self.seed,
                n_jobs=self.n_jobs,
                verbose=-1,
                force_col_wise=True,
            )
        else:
            base_model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=self.seed,
            )

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", MultiOutputRegressor(base_model, n_jobs=self.n_jobs)),
            ]
        )

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to improve model accuracy."""
        result = df.copy()
        result["moneyness"] = result["S"] / result["K"]
        result["log_moneyness"] = np.log(result["moneyness"])
        result["sqrt_T"] = np.sqrt(result["T"])
        result["T_sigma"] = result["T"] * result["sigma"]
        return result

    def _generate_random_params(
        self,
        n_samples: int,
        S_range: Tuple[float, float] = (50.0, 200.0),
        K_range: Tuple[float, float] = (50.0, 200.0),
        T_range: Tuple[float, float] = (0.05, 2.0),
        r_range: Tuple[float, float] = (0.01, 0.10),
        sigma_range: Tuple[float, float] = (0.05, 0.60),
        q_range: Tuple[float, float] = (0.0, 0.05),
    ) -> pd.DataFrame:
        """Generate random option parameters for training."""
        rng = np.random.default_rng(self.seed)

        return pd.DataFrame(
            {
                "S": rng.uniform(S_range[0], S_range[1], n_samples),
                "K": rng.uniform(K_range[0], K_range[1], n_samples),
                "T": rng.uniform(T_range[0], T_range[1], n_samples),
                "r": rng.uniform(r_range[0], r_range[1], n_samples),
                "sigma": rng.uniform(sigma_range[0], sigma_range[1], n_samples),
                "q": rng.uniform(q_range[0], q_range[1], n_samples),
            }
        )

    def generate_training_data(
        self,
        n_samples: int = 5000,
        option_type: Literal["call", "put"] = "call",
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data using VECTORIZED Black-Scholes.

        This is O(n) and takes <1 second for 10,000 samples.

        Args:
            n_samples: Number of training samples to generate.
            option_type: Option type for training ('call' or 'put').
            param_ranges: Optional dict with custom parameter ranges.
            verbose: Whether to print progress updates.

        Returns:
            Tuple of (X, y) where X is features and y is targets.
        """
        if verbose:
            logger.info(f"Generating {n_samples} training samples (vectorized BS)...")

        # Generate random parameters
        if param_ranges:
            df = self._generate_random_params(n_samples, **param_ranges)
        else:
            df = self._generate_random_params(n_samples)

        # Extract arrays for vectorized operations
        S = df["S"].values
        K = df["K"].values
        T = df["T"].values
        r = df["r"].values
        sigma = df["sigma"].values
        q = df["q"].values
        is_call = option_type == "call"

        # FULLY VECTORIZED - O(n) with pure NumPy
        prices = _bs_price_vectorized(S, K, T, r, sigma, q, is_call)
        deltas = _bs_delta_vectorized(S, K, T, r, sigma, q, is_call)
        gammas = _bs_gamma_vectorized(S, K, T, r, sigma, q)

        # Add small noise to simulate MC variance (optional, improves generalization)
        rng = np.random.default_rng(self.seed)
        prices += rng.normal(0, 0.01 * prices.clip(min=0.1), n_samples)

        # Engineer features
        df_features = self._engineer_features(df)
        X = df_features[self.feature_names].values
        y = np.column_stack([prices, deltas, gammas])

        if verbose:
            logger.info(f"Training data generation complete: X{X.shape}, y{y.shape}")

        return X, y

    def fit(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[np.ndarray] = None,
        n_samples: int = 5000,
        option_type: Literal["call", "put"] = "call",
        verbose: bool = True,
    ) -> "MonteCarloMLSurrogate":
        """
        Train the ML surrogate model.

        If X and y are not provided, training data is automatically
        generated using vectorized Black-Scholes (very fast).

        Args:
            X: Feature matrix (optional).
            y: Target matrix (optional).
            n_samples: Number of samples to generate if X/y not provided.
            option_type: Option type for auto-generated data.
            verbose: Whether to print progress updates.

        Returns:
            Self for method chaining.
        """
        if verbose:
            logger.info("Starting ML surrogate training...")

        # Generate training data if not provided
        if X is None or y is None:
            X, y = self.generate_training_data(
                n_samples=n_samples,
                option_type=option_type,
                verbose=verbose,
            )
        elif isinstance(X, pd.DataFrame):
            X_df = self._engineer_features(X)
            X = X_df[self.feature_names].values

        # Train the model
        if verbose:
            logger.info(f"Training {self.model_type} model on {X.shape[0]} samples...")

        self.model.fit(X, y)
        self.trained = True

        self._training_stats = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_targets": y.shape[1],
            "option_type": option_type,
        }

        if verbose:
            logger.info("Training complete!")

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Predict option prices and Greeks using the trained surrogate.

        Args:
            X: Feature data.

        Returns:
            DataFrame with columns 'price', 'delta', 'gamma'.
        """
        if not self.trained:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X_df = self._engineer_features(X)
            X_array = X_df[self.feature_names].values
        else:
            X_array = X

        predictions = self.model.predict(X_array)
        return pd.DataFrame(predictions, columns=self.target_names)

    def predict_single(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> Dict[str, float]:
        """Predict for a single option."""
        df = pd.DataFrame(
            {
                "S": [S],
                "K": [K],
                "T": [T],
                "r": [r],
                "sigma": [sigma],
                "q": [q],
            }
        )
        result = self.predict(df)
        return {
            "price": float(result["price"].iloc[0]),
            "delta": float(result["delta"].iloc[0]),
            "gamma": float(result["gamma"].iloc[0]),
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save the trained model to disk."""
        if not self.trained:
            raise RuntimeError("Cannot save untrained model.")
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib required: pip install joblib")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": self.model,
                "model_type": self.model_type,
                "feature_names": self.feature_names,
                "target_names": self.target_names,
                "training_stats": self._training_stats,
                "seed": self.seed,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MonteCarloMLSurrogate":
        """Load a trained model from disk."""
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib required: pip install joblib")

        save_dict = joblib.load(path)
        instance = cls.__new__(cls)
        instance.model = save_dict["model"]
        instance.model_type = save_dict["model_type"]
        instance.feature_names = save_dict["feature_names"]
        instance.target_names = save_dict["target_names"]
        instance._training_stats = save_dict["training_stats"]
        instance.seed = save_dict["seed"]
        instance.trained = True

        logger.info(f"Model loaded from {path}")
        return instance

    def score(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_true: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        y_pred = self.predict(X).values

        scores = {}
        for i, name in enumerate(self.target_names):
            ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
            ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean((y_true[:, i] - y_pred[:, i]) ** 2))
            scores[f"{name}_r2"] = r2
            scores[f"{name}_rmse"] = rmse

        return scores


# Backward compatibility alias
MonteCarloML = MonteCarloMLSurrogate
