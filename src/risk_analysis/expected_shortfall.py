# src/risk_analysis/expected_shortfall.py

from typing import Iterable, Optional
import numpy as np
import logging
from scipy.stats import norm

logger = logging.getLogger(__name__)


class ExpectedShortfall:
    """Robust Expected Shortfall (ES / CVaR) utilities.

    Convention: `returns` are PnL (positive = gain). ES is returned as a positive expected loss.
    """

    @staticmethod
    def _as_array(returns: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(returns), dtype=float).ravel()
        if arr.size == 0:
            raise ValueError("returns array is empty")
        return arr

    @staticmethod
    def historical_es(returns: Iterable[float], alpha: float = 0.95) -> float:
        """
        Empirical Expected Shortfall (CVaR).

        Parameters
        returns : Iterable[float]
            Series of PnL (positive = profit). ES measures expected loss beyond VaR.
        alpha : float
            Confidence level in (0,1), e.g. 0.95.

        Returns:
        float
            Expected shortfall as a positive number representing expected loss.
        """
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0,1)")
        r = ExpectedShortfall._as_array(returns)
        # left tail cutoff (1-alpha quantile)
        cutoff = np.quantile(r, 1.0 - alpha)
        tail = r[r <= cutoff]
        if tail.size == 0:
            # no observations in tail; return worst loss
            logger.warning("historical_es: no tail observations at given alpha; returning worst loss")
            worst = -float(np.min(r))
            return worst
        es = -float(np.mean(tail))
        return es

    @staticmethod
    def parametric_es_gaussian(mu: float, sigma: float, alpha: float = 0.95) -> float:
        """
        Parametric ES under Gaussian returns assumption.
        Returns expected loss (positive).
        Formula (left tail): ES = -mu + sigma * phi(z) / (1-alpha), where z = Phi^{-1}(1-alpha) (left tail).
        """
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0,1)")
        z = norm.ppf(1.0 - alpha)
        phi_z = norm.pdf(z)
        es = -mu + sigma * (phi_z / (1.0 - alpha))
        return float(es)

    @staticmethod
    def monte_carlo_es(simulated_returns: Iterable[float], alpha: float = 0.95) -> float:
        """
        ES estimated from simulated return scenarios.
        Accepts any iterable of simulated returns (PnL).
        """
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0,1)")
        arr = ExpectedShortfall._as_array(simulated_returns)
        cutoff = np.quantile(arr, 1.0 - alpha)
        tail = arr[arr <= cutoff]
        if tail.size == 0:
            logger.warning("monte_carlo_es: no tail observations; returning worst loss")
            return -float(np.min(arr))
        es = -float(np.mean(tail))
        return es
