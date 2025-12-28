# src/risk_analysis/var.py

"""
Value at Risk (VaR) and Conditional VaR (Expected Shortfall).

Design goals:
- Explicit conventions: returns / pnl are PnL (positive = profit). VaR/ES are returned
  as **positive numbers representing expected loss** (so larger => worse).
- Deterministic randomness via numpy Generator (seedable).
- Defensive input validation with clear exceptions.
- Thread-safe for multi-threaded scenario runs.
- Accepts vectorized pricer functions for option-aware VaR for performance; falls back to loop with warning.
- Small built-in benchmarking hooks and structured logging.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

# Prefer your project's validation helpers if present
try:
    from ..common.validation import check_required_columns  # type: ignore
except Exception:
    # Fallback minimal validator
    def check_required_columns(df: pd.DataFrame, cols: List[str]) -> None:
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


logger = logging.getLogger(__name__)


#  Exceptions
class VaRError(RuntimeError):
    """Base exception type for VaR module."""


class InputValidationError(VaRError):
    """Input validation error."""


#  Utilities
def _timeit(func: Callable) -> Callable:
    """Simple timing decorator that logs duration at INFO level."""

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info("%s took %.4f s", func.__name__, end - start)
        return res

    return wrapper


def _as_array(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float).ravel()
    if arr.size == 0:
        raise InputValidationError("Input sequence is empty")
    return arr


def _validate_confidence(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise InputValidationError("confidence_level must be in (0,1)")


#  Main class
class VaRAnalyzer:
    """
    Production-ready VaR analyzer.

    Conventions:
      - All `returns` / `pnl` inputs are PnL values: positive = profit, negative = loss.
      - VaR and ES (CVaR) are returned as **positive** values representing expected loss.
        (So `var=0.05` means expected loss of 0.05 units per unit notional.)
    """

    def __init__(self, confidence_level: float = 0.95, time_horizon_days: int = 1):
        _validate_confidence(confidence_level)
        if time_horizon_days <= 0:
            raise InputValidationError("time_horizon_days must be >= 1")

        self.confidence_level = float(confidence_level)
        self.time_horizon_days = int(time_horizon_days)
        self.horizon_frac = float(self.time_horizon_days) / 365.0
        # left-tail z (quantile for returns' left tail)
        self._z_left = norm.ppf(1.0 - self.confidence_level)
        self._lock = threading.RLock()

    # Low-level helpers

    @staticmethod
    def _empirical_var_es_from_pnl(
        pnl: Iterable[float], alpha: float
    ) -> Tuple[float, float]:
        """Return (var_loss, es_loss) where both are positive numbers representing loss."""
        _validate_confidence(alpha)
        arr = _as_array(pnl)
        cutoff = np.quantile(arr, 1.0 - alpha)
        var_loss = -float(cutoff)  # positive value representing loss at VaR
        tail = arr[arr <= cutoff]
        if tail.size == 0:
            # defensive fallback: worst observed loss
            es_loss = -float(np.min(arr))
        else:
            es_loss = -float(np.mean(tail))
        return var_loss, es_loss

    # Historical (non-parametric)

    @_timeit
    def historical_var(
        self, pnl_series: Iterable[float], scale: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Historical VaR and ES computed directly from PnL series.

        Parameters:
        pnl_series : Iterable[float]
            PnL time series (same periodicity as horizon or daily; we do not rescale here).
        scale : Optional[float]
            If given, results are multiplied by scale (e.g., portfolio value).

        Returns:
        dict with 'var', 'cvar', 'confidence_level', 'samples'
        """
        arr = _as_array(pnl_series)
        if arr.size < 30:
            logger.warning(
                "historical_var: small sample size (n=%d). Results may be unstable.",
                arr.size,
            )

        var_loss, es_loss = self._empirical_var_es_from_pnl(arr, self.confidence_level)
        s = float(scale) if scale is not None else 1.0
        return {
            "var": var_loss * s,
            "cvar": es_loss * s,
            "confidence_level": self.confidence_level,
            "samples": arr.size,
        }

    # Parametric

    @_timeit
    def parametric_var(
        self,
        mu: float,
        sigma: float,
        scale: Optional[float] = None,
        assume_log_returns: bool = True,
    ) -> Dict[str, Any]:
        """
        Parametric VaR under normal or log-normal assumptions.

        Parameters:
        mu, sigma : float
            Annualized mean and volatility of returns (same units).
        scale : Optional[float]
            Multiply relative loss by scale to get monetary VaR.
        assume_log_returns : bool
            If True, assume log-returns (GBM); otherwise normal returns.

        Returns:
        dict with 'var' and 'cvar' (positive losses).
        """
        mu = float(mu)
        sigma = float(sigma)
        if sigma <= 0.0:
            raise InputValidationError("sigma must be positive")
        mu_h = mu * self.horizon_frac
        sigma_h = sigma * math.sqrt(self.horizon_frac)

        if assume_log_returns:
            # left quantile for log-returns
            q = norm.ppf(1.0 - self.confidence_level, loc=mu_h, scale=sigma_h)
            var_rel = max(0.0, 1.0 - math.exp(q))  # positive relative loss at VaR
            # approximate ES for log-normal left tail using conditional expectation formula
            # handle numerical edge cases defensively
            try:
                denom = norm.cdf((q - mu_h) / sigma_h)
                if denom <= 0 or math.isnan(denom):
                    es_rel = var_rel
                else:
                    numer = norm.cdf((q - mu_h - sigma_h**2) / sigma_h)
                    e_exp = math.exp(mu_h + 0.5 * sigma_h**2) * (numer / denom)
                    es_rel = max(0.0, 1.0 - e_exp)
            except Exception:
                es_rel = var_rel
            var_loss, es_loss = var_rel, es_rel
        else:
            # Normal returns
            # VaR (loss) = -(mu_h - z_left * sigma_h)
            var_loss = max(0.0, -(mu_h - self._z_left * sigma_h))
            z = self._z_left
            phi_z = norm.pdf(z)
            tail_prob = 1.0 - self.confidence_level
            es_loss = max(0.0, -mu_h + sigma_h * (phi_z / tail_prob))

        s = float(scale) if scale is not None else 1.0
        return {
            "var": var_loss * s,
            "cvar": es_loss * s,
            "confidence_level": self.confidence_level,
        }

    # Monte Carlo on underlying price (vectorized)

    @_timeit
    def monte_carlo_var(
        self,
        initial_price: float,
        mu: float,
        sigma: float,
        num_simulations: int = 10000,
        scale: Optional[float] = None,
        use_log_returns: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Monte Carlo VaR by simulating terminal underlying and computing PnL per unit notional.

        Returns positive losses scaled by `scale` if provided.
        """
        if initial_price <= 0:
            raise InputValidationError("initial_price must be positive")
        if sigma <= 0:
            raise InputValidationError("sigma must be positive")
        if num_simulations <= 0:
            raise InputValidationError("num_simulations must be > 0")

        rng = np.random.default_rng(seed)
        mu_h = mu * self.horizon_frac
        sigma_h = sigma * math.sqrt(self.horizon_frac)

        if use_log_returns:
            sim_r = rng.normal(loc=mu_h, scale=sigma_h, size=num_simulations)
            final_prices = initial_price * np.exp(sim_r)
            pnl = final_prices - initial_price  # profit per unit notional
        else:
            sim_r = rng.normal(loc=mu_h, scale=sigma_h, size=num_simulations)
            pnl = initial_price * sim_r

        var_loss, es_loss = self._empirical_var_es_from_pnl(pnl, self.confidence_level)
        s = float(scale) if scale is not None else 1.0
        return {
            "var": var_loss * s,
            "cvar": es_loss * s,
            "n_sims": num_simulations,
            "confidence_level": self.confidence_level,
        }

    # Portfolio delta-normal VaR (multi-asset)

    @_timeit
    def portfolio_var(
        self,
        weights: List[float],
        expected_returns: List[float],
        cov_matrix: np.ndarray,
        portfolio_value: float,
    ) -> Dict[str, Any]:
        """
        Delta-normal portfolio VaR (normal returns assumption).

        weights: array-like of weights (same order as expected_returns and cov_matrix rows).
        expected_returns: annualized returns (same order).
        cov_matrix: annualized covariance matrix.
        portfolio_value: scale to convert relative to monetary units.
        """
        if portfolio_value <= 0:
            raise InputValidationError("portfolio_value must be positive")

        w = np.asarray(weights, dtype=float)
        mu = np.asarray(expected_returns, dtype=float)
        cov = np.asarray(cov_matrix, dtype=float)

        if w.ndim != 1 or mu.ndim != 1:
            raise InputValidationError(
                "weights and expected_returns must be 1-D arrays"
            )
        if (
            w.shape[0] != mu.shape[0]
            or cov.shape[0] != w.shape[0]
            or cov.shape[1] != w.shape[0]
        ):
            raise InputValidationError(
                "dimension mismatch among weights, returns and covariance matrix"
            )

        mu_h = float(w @ mu) * self.horizon_frac
        sigma_port = math.sqrt(float(w.T @ cov @ w)) * math.sqrt(self.horizon_frac)

        var_loss = max(0.0, -(mu_h - self._z_left * sigma_port))
        z = self._z_left
        phi_z = norm.pdf(z)
        tail_prob = 1.0 - self.confidence_level
        es_loss = max(0.0, -mu_h + sigma_port * (phi_z / tail_prob))

        return {
            "var": var_loss * portfolio_value,
            "cvar": es_loss * portfolio_value,
            "confidence_level": self.confidence_level,
        }

    # Option-aware VaR (vectorized pricer recommended)

    @_timeit
    def option_var(
        self,
        S: float,
        pricer_fn: Callable[[np.ndarray, Dict[str, Any]], np.ndarray],
        pricer_params: Dict[str, Any],
        num_simulations: int = 10000,
        scale: Optional[float] = None,
        seed: Optional[int] = None,
        vectorized: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute VaR by simulating terminal underlying prices and pricing option(s).
        pricer_fn should accept (prices: np.ndarray, pricer_params: dict) and return np.ndarray of option prices
        for each input price. If pricer_fn is not vectorized (i.e., expects a scalar price), set vectorized=False
        and the function will fall back to a Python loop (slower).
        pricer_params should include keys like K, T, r, sigma, option_type, etc., used by your pricer.
        """
        if S <= 0:
            raise InputValidationError("S (spot) must be positive")
        if num_simulations <= 0:
            raise InputValidationError("num_simulations must be >= 1")
        if not callable(pricer_fn):
            raise InputValidationError("pricer_fn must be callable")

        rng = np.random.default_rng(seed)
        sigma = float(pricer_params.get("sigma", 0.2))
        mu = float(pricer_params.get("r", 0.0))
        sigma_h = sigma * math.sqrt(self.horizon_frac)
        mu_h = mu * self.horizon_frac

        sim_r = rng.normal(loc=mu_h, scale=sigma_h, size=num_simulations)
        final_prices = S * np.exp(sim_r)

        # Price the option at S (baseline)
        try:
            baseline_price = pricer_fn(np.array([S]), pricer_params)[0]
        except Exception:
            # If pricer expects scalar signature, try single-call
            baseline_price = float(pricer_fn(S, pricer_params))

        # Vectorized path
        if vectorized:
            try:
                sim_prices = pricer_fn(final_prices, pricer_params)
                pnl = sim_prices - baseline_price  # PnL per unit notional
            except Exception:
                logger.warning(
                    "pricer_fn vectorized call failed; falling back to Python loop",
                    exc_info=True,
                )
                vectorized = False

        if not vectorized:
            # Fallback loop (slow)
            pnls = []
            for p in final_prices:
                price = pricer_fn(p, pricer_params)
                pnls.append(float(price) - float(baseline_price))
            pnl = np.asarray(pnls, dtype=float)

        var_loss, es_loss = self._empirical_var_es_from_pnl(pnl, self.confidence_level)
        s = float(scale) if scale is not None else 1.0
        return {
            "var": var_loss * s,
            "cvar": es_loss * s,
            "n_sims": num_simulations,
            "confidence_level": self.confidence_level,
        }

    # Stress test helpers

    @_timeit
    def stress_test_var_from_returns(
        self, pnl_series: Iterable[float], shift: float, scale: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Simple stress test by adding a constant shift to the PnL series (e.g., shift = -0.1 for -10%).
        Returns VaR/ES computed on shocked series.
        """
        arr = _as_array(pnl_series)
        shocked = arr + float(shift)
        var_loss, es_loss = self._empirical_var_es_from_pnl(
            shocked, self.confidence_level
        )
        s = float(scale) if scale is not None else 1.0
        return {
            "var": var_loss * s,
            "cvar": es_loss * s,
            "confidence_level": self.confidence_level,
        }

    @_timeit
    def batch_stress_test(
        self,
        pnl_series: Iterable[float],
        shifts: Iterable[float],
        scale: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Run multiple additive stress shifts and return a DataFrame indexed by shift value
        with columns var and cvar (monetary units if scale given).
        """
        arr = _as_array(pnl_series)
        rows = []
        for shift in shifts:
            shocked = arr + float(shift)
            var_loss, es_loss = self._empirical_var_es_from_pnl(
                shocked, self.confidence_level
            )
            rows.append(
                {
                    "shift": shift,
                    "var": var_loss * (scale or 1.0),
                    "cvar": es_loss * (scale or 1.0),
                }
            )
        return pd.DataFrame(rows).set_index("shift")
