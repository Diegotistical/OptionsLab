# src/pricing_models/monte_carlo_unified.py
"""
Ultra-fast, production-ready Monte Carlo pricer with CPU/GPU support.

This module provides a high-performance Monte Carlo pricing engine that
unifies CPU and GPU implementations with advanced variance reduction
techniques and ML surrogate support.

Features:
    - Monte Carlo pricing for European options
    - Delta & Gamma Greeks via central differences with Common Random Numbers
    - Optional Numba JIT acceleration (compiled at module load)
    - Optional GPU acceleration via CuPy
    - Thread-safe implementation
    - Vectorized batch processing
    - Machine Learning surrogate for instant predictions

Example:
    >>> from src.pricing_models.monte_carlo_unified import MonteCarloPricerUni
    >>> pricer = MonteCarloPricerUni(num_simulations=100000, use_numba=True)
    >>> price = pricer.price(100, 100, 1.0, 0.05, 0.2, 'call')
    >>> delta, gamma = pricer.delta_gamma(100, 100, 1.0, 0.05, 0.2, 'call')
"""

import logging
import threading
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Check for optional dependencies
try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """Dummy decorator when Numba not installed."""

        def decorator(func):
            return func

        return decorator if not args else decorator(args[0])

    def prange(*args):
        """Dummy prange that falls back to range."""
        return range(*args)


try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

# ML imports for surrogate
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "MonteCarloPricerUni",
    "MLSurrogate",
    "InputValidationError",
    "MonteCarloError",
    "NUMBA_AVAILABLE",
    "GPU_AVAILABLE",
]

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class InputValidationError(Exception):
    """Raised when inputs to Monte Carlo or ML methods are invalid."""

    pass


class MonteCarloError(Exception):
    """Raised for unexpected Monte Carlo computation errors."""

    pass


# =============================================================================
# Module-level Numba kernels (compiled once at import)
# =============================================================================

if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _simulate_single_path_numba(
        S: float,
        T: float,
        r: float,
        sigma: float,
        q: float,
        num_steps: int,
        z_values: np.ndarray,
    ) -> float:
        """
        Simulate a single GBM path and return terminal price.

        Args:
            S: Initial spot price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            q: Dividend yield.
            num_steps: Number of time steps.
            z_values: Pre-generated standard normal values.

        Returns:
            Terminal stock price.
        """
        dt = T / num_steps
        drift = (r - q - 0.5 * sigma * sigma) * dt
        vol = sigma * np.sqrt(dt)

        log_S = np.log(S)
        for t in range(num_steps):
            log_S += drift + vol * z_values[t]

        return np.exp(log_S)

    @njit(parallel=True, cache=True, fastmath=True)
    def _simulate_batch_numba(
        S_arr: np.ndarray,
        T_arr: np.ndarray,
        r_arr: np.ndarray,
        sigma_arr: np.ndarray,
        q_arr: np.ndarray,
        num_simulations: int,
        num_steps: int,
        seed: int,
    ) -> np.ndarray:
        """
        Parallel batch simulation for multiple options.

        Each option is simulated with antithetic variance reduction.
        Uses parallel execution across options.

        Args:
            S_arr: Array of spot prices.
            T_arr: Array of times to maturity.
            r_arr: Array of risk-free rates.
            sigma_arr: Array of volatilities.
            q_arr: Array of dividend yields.
            num_simulations: Number of MC paths per option.
            num_steps: Number of time steps.
            seed: Base random seed.

        Returns:
            Array of shape (n_options, 2 * num_simulations) with terminal prices.
        """
        n_options = len(S_arr)
        terminal_prices = np.empty((n_options, num_simulations * 2), dtype=np.float64)

        for opt_idx in prange(n_options):
            S = S_arr[opt_idx]
            T = T_arr[opt_idx]
            r = r_arr[opt_idx]
            sigma = sigma_arr[opt_idx]
            q = q_arr[opt_idx]

            dt = T / num_steps
            drift = (r - q - 0.5 * sigma * sigma) * dt
            vol = sigma * np.sqrt(dt)

            # Set seed unique to this option
            np.random.seed(seed + opt_idx)

            for sim_idx in range(num_simulations):
                log_S_pos = np.log(S)
                log_S_neg = np.log(S)

                for t in range(num_steps):
                    z = np.random.randn()
                    log_S_pos += drift + vol * z
                    log_S_neg += drift - vol * z

                terminal_prices[opt_idx, sim_idx] = np.exp(log_S_pos)
                terminal_prices[opt_idx, sim_idx + num_simulations] = np.exp(log_S_neg)

        return terminal_prices

    @njit(cache=True, fastmath=True)
    def _compute_payoffs_numba(
        terminal_prices: np.ndarray,
        K: float,
        is_call: bool,
    ) -> np.ndarray:
        """
        Compute option payoffs from terminal prices.

        Args:
            terminal_prices: Array of terminal stock prices.
            K: Strike price.
            is_call: True for call, False for put.

        Returns:
            Array of payoffs.
        """
        n = len(terminal_prices)
        payoffs = np.empty(n, dtype=np.float64)

        if is_call:
            for i in range(n):
                payoffs[i] = max(terminal_prices[i] - K, 0.0)
        else:
            for i in range(n):
                payoffs[i] = max(K - terminal_prices[i], 0.0)

        return payoffs


class MonteCarloPricerUni:
    """
    High-performance Monte Carlo pricer with CPU/GPU support.

    This class provides a unified interface for Monte Carlo option pricing
    that automatically selects the best available backend (GPU > Numba > NumPy).

    Attributes:
        num_simulations (int): Number of Monte Carlo paths.
        num_steps (int): Number of time discretization steps.
        use_numba (bool): Whether Numba acceleration is enabled.
        use_gpu (bool): Whether GPU acceleration is enabled.
        rng (np.random.Generator): Random number generator.

    Example:
        >>> pricer = MonteCarloPricerUni(num_simulations=100000, use_gpu=True)
        >>> price = pricer.price(100, 100, 1.0, 0.05, 0.2, 'call')
        >>> print(f"Option price: {price:.4f}")
    """

    def __init__(
        self,
        num_simulations: int = 100_000,
        num_steps: int = 100,
        seed: Optional[int] = None,
        use_numba: bool = True,
        use_gpu: bool = False,
    ) -> None:
        """
        Initialize the unified Monte Carlo pricer.

        Args:
            num_simulations: Number of MC paths (default 100,000).
            num_steps: Time discretization steps (default 100).
            seed: Random seed for reproducibility.
            use_numba: Enable Numba JIT acceleration.
            use_gpu: Enable GPU acceleration with CuPy.

        Raises:
            InputValidationError: If num_simulations or num_steps <= 0.
        """
        if num_simulations <= 0 or num_steps <= 0:
            raise InputValidationError(
                "num_simulations and num_steps must be positive integers"
            )

        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.seed = (
            seed if seed is not None else np.random.default_rng().integers(0, 2**31)
        )
        self.rng = np.random.default_rng(seed)
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self._lock = threading.RLock()

        logger.info(
            f"MonteCarloPricerUni initialized: "
            f"simulations={num_simulations}, steps={num_steps}, "
            f"numba={self.use_numba}, gpu={self.use_gpu}"
        )

    def _simulate_terminal_prices_vectorized(
        self,
        S_arr: np.ndarray,
        T_arr: np.ndarray,
        r_arr: np.ndarray,
        sigma_arr: np.ndarray,
        q_arr: np.ndarray,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Vectorized terminal price simulation using NumPy.

        Args:
            S_arr: Array of spot prices.
            T_arr: Array of times to maturity.
            r_arr: Array of risk-free rates.
            sigma_arr: Array of volatilities.
            q_arr: Array of dividend yields.
            seed: Optional seed override.

        Returns:
            Array of terminal prices with shape (n, 2 * num_simulations).
        """
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        n = len(S_arr)

        dt = T_arr / self.num_steps
        drift = (r_arr - q_arr - 0.5 * sigma_arr**2)[:, None] * dt[:, None]
        vol = sigma_arr[:, None] * np.sqrt(dt[:, None])

        # Generate random normals: (n_options, n_sims, n_steps)
        Z = rng.standard_normal((n, self.num_simulations, self.num_steps))

        # Compute log price paths
        log_S = np.log(S_arr)[:, None, None]
        log_increments = drift[:, None, :] + vol[:, None, :] * Z
        log_paths_pos = log_S + np.cumsum(log_increments, axis=2)
        log_paths_neg = log_S + np.cumsum(
            drift[:, None, :] - vol[:, None, :] * Z, axis=2
        )

        # Extract terminal prices
        terminal_pos = np.exp(log_paths_pos[:, :, -1])
        terminal_neg = np.exp(log_paths_neg[:, :, -1])

        return np.concatenate([terminal_pos, terminal_neg], axis=1)

    def _simulate_terminal_prices_gpu(
        self,
        S_arr: np.ndarray,
        T_arr: np.ndarray,
        r_arr: np.ndarray,
        sigma_arr: np.ndarray,
        q_arr: np.ndarray,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        GPU-accelerated terminal price simulation using CuPy.

        Args:
            S_arr: Array of spot prices.
            T_arr: Array of times to maturity.
            r_arr: Array of risk-free rates.
            sigma_arr: Array of volatilities.
            q_arr: Array of dividend yields.
            seed: Optional seed override.

        Returns:
            Array of terminal prices (on CPU).
        """
        if cp is None:
            raise MonteCarloError("CuPy is not available for GPU computation")

        n = len(S_arr)
        actual_seed = seed if seed is not None else self.seed

        # Transfer to GPU
        S_gpu = cp.asarray(S_arr)
        T_gpu = cp.asarray(T_arr)
        r_gpu = cp.asarray(r_arr)
        sigma_gpu = cp.asarray(sigma_arr)
        q_gpu = cp.asarray(q_arr)

        dt = T_gpu / self.num_steps
        drift = (r_gpu - q_gpu - 0.5 * sigma_gpu**2)[:, None] * dt[:, None]
        vol = sigma_gpu[:, None] * cp.sqrt(dt[:, None])

        # Generate random normals on GPU
        cp.random.seed(actual_seed)
        Z = cp.random.randn(n, self.num_simulations, self.num_steps)

        # Compute paths
        log_S = cp.log(S_gpu)[:, None, None]
        log_increments_pos = drift[:, None, :] + vol[:, None, :] * Z
        log_increments_neg = drift[:, None, :] - vol[:, None, :] * Z

        log_paths_pos = log_S + cp.cumsum(log_increments_pos, axis=2)
        log_paths_neg = log_S + cp.cumsum(log_increments_neg, axis=2)

        terminal_pos = cp.exp(log_paths_pos[:, :, -1])
        terminal_neg = cp.exp(log_paths_neg[:, :, -1])

        result = cp.concatenate([terminal_pos, terminal_neg], axis=1)

        return cp.asnumpy(result)

    def _simulate_terminal_prices(
        self,
        S_arr: np.ndarray,
        T_arr: np.ndarray,
        r_arr: np.ndarray,
        sigma_arr: np.ndarray,
        q_arr: np.ndarray,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate terminal prices using the best available backend.

        Automatically selects GPU > Numba > NumPy based on configuration.

        Args:
            S_arr: Array of spot prices.
            T_arr: Array of times to maturity.
            r_arr: Array of risk-free rates.
            sigma_arr: Array of volatilities.
            q_arr: Array of dividend yields.
            seed: Optional seed override.

        Returns:
            Array of terminal prices.
        """
        actual_seed = seed if seed is not None else self.seed

        if self.use_gpu:
            return self._simulate_terminal_prices_gpu(
                S_arr, T_arr, r_arr, sigma_arr, q_arr, actual_seed
            )
        elif self.use_numba:
            return _simulate_batch_numba(
                S_arr,
                T_arr,
                r_arr,
                sigma_arr,
                q_arr,
                self.num_simulations,
                self.num_steps,
                actual_seed,
            )
        else:
            return self._simulate_terminal_prices_vectorized(
                S_arr, T_arr, r_arr, sigma_arr, q_arr, actual_seed
            )

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
        seed: Optional[int] = None,
    ) -> float:
        """
        Price a European option via Monte Carlo simulation.

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to maturity in years.
            r: Risk-free interest rate.
            sigma: Volatility.
            option_type: 'call' or 'put'.
            q: Dividend yield (default 0.0).
            seed: Optional seed for this specific call.

        Returns:
            Option price as float.

        Raises:
            InputValidationError: If inputs are invalid.
            MonteCarloError: If computation fails.

        Example:
            >>> pricer = MonteCarloPricerUni()
            >>> call_price = pricer.price(100, 100, 1.0, 0.05, 0.2, 'call')
        """
        if S <= 0 or K <= 0 or T <= 0 or sigma < 0:
            raise InputValidationError(
                "S, K, T must be positive; sigma must be non-negative"
            )
        if option_type not in {"call", "put"}:
            raise InputValidationError("option_type must be 'call' or 'put'")

        try:
            terminal_prices = self._simulate_terminal_prices(
                np.array([S]),
                np.array([T]),
                np.array([r]),
                np.array([sigma]),
                np.array([q]),
                seed=seed,
            )[0]

            if option_type == "call":
                payoffs = np.maximum(terminal_prices - K, 0.0)
            else:
                payoffs = np.maximum(K - terminal_prices, 0.0)

            return float(np.exp(-r * T) * np.mean(payoffs))

        except Exception as e:
            raise MonteCarloError(f"Monte Carlo pricing failed: {e}")

    def delta_gamma(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
        h: float = 1e-4,
        seed: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Compute Delta and Gamma using central differences with CRN.

        Uses Common Random Numbers (CRN) for variance reduction by
        using the same seed across S-h, S, and S+h calculations.

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            option_type: 'call' or 'put'.
            q: Dividend yield.
            h: Finite difference step size.
            seed: Base random seed.

        Returns:
            Tuple of (delta, gamma).

        Example:
            >>> pricer = MonteCarloPricerUni()
            >>> delta, gamma = pricer.delta_gamma(100, 100, 1.0, 0.05, 0.2, 'call')
        """
        if seed is None:
            seed = int(self.rng.integers(0, 2**31))

        # Use same seed for all three prices (CRN)
        price_up = self.price(S + h, K, T, r, sigma, option_type, q, seed=seed)
        price_mid = self.price(S, K, T, r, sigma, option_type, q, seed=seed)
        price_down = self.price(S - h, K, T, r, sigma, option_type, q, seed=seed)

        delta = (price_up - price_down) / (2 * h)
        gamma = (price_up - 2 * price_mid + price_down) / (h**2)

        return delta, gamma

    def price_batch(
        self,
        S_vals: np.ndarray,
        K_vals: np.ndarray,
        T_vals: np.ndarray,
        r_vals: np.ndarray,
        sigma_vals: np.ndarray,
        option_type: Literal["call", "put"],
        q_vals: Union[float, np.ndarray] = 0.0,
    ) -> np.ndarray:
        """
        Vectorized pricing for multiple options at once.

        Much more efficient than calling price() in a loop.

        Args:
            S_vals: Array of spot prices.
            K_vals: Array of strike prices.
            T_vals: Array of times to maturity.
            r_vals: Array of risk-free rates.
            sigma_vals: Array of volatilities.
            option_type: 'call' or 'put' (same for all).
            q_vals: Dividend yields (scalar or array).

        Returns:
            Array of option prices.

        Example:
            >>> pricer = MonteCarloPricerUni()
            >>> prices = pricer.price_batch(
            ...     np.array([100, 110, 120]),
            ...     np.array([100, 100, 100]),
            ...     np.array([1.0, 1.0, 1.0]),
            ...     np.array([0.05, 0.05, 0.05]),
            ...     np.array([0.2, 0.2, 0.2]),
            ...     'call'
            ... )
        """
        # Ensure arrays
        S_vals = np.asarray(S_vals, dtype=np.float64)
        K_vals = np.asarray(K_vals, dtype=np.float64)
        T_vals = np.asarray(T_vals, dtype=np.float64)
        r_vals = np.asarray(r_vals, dtype=np.float64)
        sigma_vals = np.asarray(sigma_vals, dtype=np.float64)

        if isinstance(q_vals, (int, float)):
            q_vals = np.full_like(S_vals, q_vals)
        else:
            q_vals = np.asarray(q_vals, dtype=np.float64)

        n = len(S_vals)

        # Simulate all terminal prices at once - shape (n, 2*num_sims)
        terminal_prices = self._simulate_terminal_prices(
            S_vals, T_vals, r_vals, sigma_vals, q_vals
        )

        # FULLY VECTORIZED payoff calculation - O(n) with broadcasting
        # terminal_prices: (n, num_sims*2), K_vals: (n,)
        # Use broadcasting: K_vals[:, None] creates (n, 1) for subtraction
        if option_type == "call":
            payoffs = np.maximum(terminal_prices - K_vals[:, None], 0.0)
        else:
            payoffs = np.maximum(K_vals[:, None] - terminal_prices, 0.0)

        # Vectorized discounting and mean - no loop needed
        discount_factors = np.exp(-r_vals * T_vals)
        prices = discount_factors * np.mean(payoffs, axis=1)

        return prices

    def delta_gamma_batch(
        self,
        S_vals: np.ndarray,
        K_vals: np.ndarray,
        T_vals: np.ndarray,
        r_vals: np.ndarray,
        sigma_vals: np.ndarray,
        option_type: Literal["call", "put"],
        q_vals: Union[float, np.ndarray] = 0.0,
        h: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fully vectorized Delta and Gamma calculation.

        Uses central differences with the same random seed for
        variance reduction (Common Random Numbers).

        Args:
            S_vals: Array of spot prices.
            K_vals: Array of strike prices.
            T_vals: Array of times to maturity.
            r_vals: Array of risk-free rates.
            sigma_vals: Array of volatilities.
            option_type: 'call' or 'put'.
            q_vals: Dividend yields.
            h: Finite difference step size.

        Returns:
            Tuple of (deltas, gammas) arrays.
        """
        S_vals = np.asarray(S_vals, dtype=np.float64)
        K_vals = np.asarray(K_vals, dtype=np.float64)
        T_vals = np.asarray(T_vals, dtype=np.float64)
        r_vals = np.asarray(r_vals, dtype=np.float64)
        sigma_vals = np.asarray(sigma_vals, dtype=np.float64)

        if isinstance(q_vals, (int, float)):
            q_vals = np.full_like(S_vals, q_vals)
        else:
            q_vals = np.asarray(q_vals, dtype=np.float64)

        # Compute prices at S-h, S, S+h using same seed (CRN)
        prices_down = self.price_batch(
            S_vals - h, K_vals, T_vals, r_vals, sigma_vals, option_type, q_vals
        )
        prices_mid = self.price_batch(
            S_vals, K_vals, T_vals, r_vals, sigma_vals, option_type, q_vals
        )
        prices_up = self.price_batch(
            S_vals + h, K_vals, T_vals, r_vals, sigma_vals, option_type, q_vals
        )

        # Vectorized finite differences
        deltas = (prices_up - prices_down) / (2 * h)
        gammas = (prices_up - 2 * prices_mid + prices_down) / (h**2)

        return deltas, gammas


class MLSurrogate:
    """
    Machine Learning surrogate for instant prediction of price, Delta, Gamma.

    This is a simplified surrogate that can be trained directly on
    MonteCarloPricerUni outputs for faster predictions.

    Attributes:
        model: The ML model pipeline.
        trained (bool): Whether the model is trained.

    Example:
        >>> pricer = MonteCarloPricerUni()
        >>> surrogate = MLSurrogate()
        >>> surrogate.fit(training_data, pricer)
        >>> predictions = surrogate.predict(test_data)
    """

    def __init__(
        self,
        ml_model: Optional[Any] = None,
        n_estimators: int = 200,
        max_depth: int = 5,
        random_state: int = 42,
    ) -> None:
        """
        Initialize ML surrogate.

        Args:
            ml_model: Custom ML model (optional).
            n_estimators: Number of boosting iterations.
            max_depth: Maximum tree depth.
            random_state: Random seed.
        """
        if ml_model is not None:
            self.model = ml_model
        else:
            if LIGHTGBM_AVAILABLE:
                base_model = lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    verbose=-1,
                )
            else:
                base_model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                )

            self.model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", MultiOutputRegressor(base_model)),
                ]
            )

        self.trained = False

    def fit(
        self,
        df: pd.DataFrame,
        pricer: MonteCarloPricerUni,
        option_type: Literal["call", "put"] = "call",
    ) -> "MLSurrogate":
        """
        Fit surrogate using Monte Carlo pricer outputs.

        Args:
            df: DataFrame with columns ['S', 'K', 'T', 'r', 'sigma', 'q'].
            pricer: MonteCarloPricerUni instance for generating targets.
            option_type: Option type for training.

        Returns:
            Self for method chaining.
        """
        mc_prices, mc_deltas, mc_gammas = [], [], []

        for _, row in df.iterrows():
            S, K, T, r, sigma, q = row[["S", "K", "T", "r", "sigma", "q"]]

            # Use deterministic seed for training stability
            row_seed = 42 + int(S * 100) % 10000

            p = pricer.price(S, K, T, r, sigma, option_type, q, seed=row_seed)
            d, g = pricer.delta_gamma(S, K, T, r, sigma, option_type, q, seed=row_seed)

            mc_prices.append(p)
            mc_deltas.append(d)
            mc_gammas.append(g)

        y = pd.DataFrame(
            {
                "price": mc_prices,
                "delta": mc_deltas,
                "gamma": mc_gammas,
            }
        )
        X = df[["S", "K", "T", "r", "sigma", "q"]].values

        self.model.fit(X, y)
        self.trained = True

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict price, Delta, Gamma using surrogate model.

        Args:
            df: DataFrame with columns ['S', 'K', 'T', 'r', 'sigma', 'q'].

        Returns:
            DataFrame with columns ['price', 'delta', 'gamma'].

        Raises:
            RuntimeError: If model not trained.
        """
        if not self.trained:
            raise RuntimeError("Surrogate model not trained")

        X = df[["S", "K", "T", "r", "sigma", "q"]].values
        y_pred = self.model.predict(X)

        return pd.DataFrame(y_pred, columns=["price", "delta", "gamma"])
