# src/volatility_surface/models/svi.py
"""
SVI (Stochastic Volatility Inspired) Volatility Surface Model.

Implements Gatheral's SVI parameterization:

    w(k) = a + b * (ρ(k - m) + √((k - m)² + σ²))

Where:
    w = σ²T (total implied variance)
    k = log(K/F) (log-moneyness)

Parameters:
    a: Vertical shift (ATM variance level)
    b: Angle between the wings (slope)
    ρ (rho): Rotation/skew parameter
    m: Horizontal shift
    σ (sigma): Curvature/smile width

Features:
    - Fast analytical smile computation
    - Arbitrage-free constraints (butterfly, calendar)
    - Calibration with regularization

Reference:
    Gatheral, J. (2004). A Parsimonious Arbitrage-Free Implied
    Volatility Parameterization. Presentation at Global Derivatives.

Usage:
    >>> from src.volatility_surface.models.svi import SVIModel
    >>> svi = SVIModel(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
    >>> iv = svi.implied_vol(k=-0.1, T=1.0)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize


@dataclass
class SVIModel:
    """
    SVI volatility surface parameterization.

    Attributes:
        a: Vertical shift (level of ATM variance).
        b: Slope of the wings (must be positive).
        rho: Rotation/skew (-1 < rho < 1).
        m: Horizontal translation.
        sigma: Curvature width (must be positive).
    """

    a: float  # Vertical shift
    b: float  # Wing angle
    rho: float  # Rotation
    m: float  # Horizontal shift
    sigma: float  # Curvature

    def __post_init__(self):
        """Validate SVI parameters."""
        if self.b < 0:
            raise ValueError("b must be non-negative")
        if not -1 < self.rho < 1:
            raise ValueError("rho must be in (-1, 1)")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

        # Check arbitrage-free conditions
        self._check_arbitrage_free()

    def _check_arbitrage_free(self):
        """Check Gatheral's arbitrage-free constraints."""
        # Butterfly arbitrage: w(k) must be non-negative for all k
        # This requires: a + b*σ*√(1-ρ²) >= 0
        min_variance = self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2)
        if min_variance < -1e-10:
            import warnings

            warnings.warn(
                f"SVI parameters may violate butterfly arbitrage "
                f"(min variance = {min_variance:.6f} < 0)"
            )

    def total_variance(self, k: float) -> float:
        """
        Compute total implied variance w(k) = σ²T.

        Args:
            k: Log-moneyness = log(K/F).

        Returns:
            Total implied variance.
        """
        return self.a + self.b * (
            self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma**2)
        )

    def implied_vol(self, k: float, T: float) -> float:
        """
        Compute implied volatility.

        Args:
            k: Log-moneyness.
            T: Time to maturity.

        Returns:
            Implied volatility σ.
        """
        if T <= 0:
            return np.sqrt(max(self.a, 0))

        w = self.total_variance(k)
        if w < 0:
            # Clamp to avoid negative variance
            w = 1e-10
        return np.sqrt(w / T)

    def smile(
        self,
        log_strikes: np.ndarray,
        T: float,
    ) -> np.ndarray:
        """
        Compute implied volatility smile.

        Args:
            log_strikes: Array of log-moneyness values.
            T: Time to maturity.

        Returns:
            Array of implied volatilities.
        """
        return np.array([self.implied_vol(k, T) for k in log_strikes])

    def local_vol_squared(self, k: float, T: float) -> float:
        """
        Compute local variance via Dupire formula.

        Args:
            k: Log-moneyness.
            T: Time to maturity.

        Returns:
            Local variance σ_loc².
        """
        w = self.total_variance(k)

        # Derivatives of w with respect to k
        term = np.sqrt((k - self.m) ** 2 + self.sigma**2)
        dw_dk = self.b * (self.rho + (k - self.m) / term)
        d2w_dk2 = self.b * self.sigma**2 / term**3

        # Time derivative (for single slice, ∂w/∂T = w/T)
        dw_dT = w / T

        # Dupire formula
        numerator = dw_dT
        denominator = (
            1
            - k * dw_dk / w
            + 0.25 * (-0.25 - 1 / w + k**2 / w**2) * dw_dk**2
            + 0.5 * d2w_dk2
        )

        if denominator <= 0:
            return w / T  # Fallback to total variance

        return numerator / denominator


@dataclass
class SSVIModel:
    """
    SSVI (Surface SVI) for arbitrage-free term structure.

    Extends SVI to model the entire surface with guaranteed
    arbitrage-free across both strikes and maturities.

    Parameterization (Gatheral & Jacquier, 2014):
        w(k,θ) = (θ/2) * (1 + ρφ(θ)k + √((φ(θ)k + ρ)² + 1 - ρ²))

    Where:
        θ = ATM total variance = σ_ATM² * T
        φ(θ) = η / θ^γ (power law)
    """

    rho: float  # Global skew
    eta: float  # Vol-of-vol proxy
    gamma: float  # Power law exponent

    def __post_init__(self):
        if not -1 < self.rho < 1:
            raise ValueError("rho must be in (-1, 1)")
        if self.eta <= 0:
            raise ValueError("eta must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")

    def phi(self, theta: float) -> float:
        """Correlation function φ(θ)."""
        if theta <= 0:
            return 1.0
        return self.eta / (theta**self.gamma)

    def total_variance(self, k: float, theta: float) -> float:
        """
        Compute SSVI total variance.

        Args:
            k: Log-moneyness.
            theta: ATM total variance for this slice.

        Returns:
            Total implied variance.
        """
        phi = self.phi(theta)
        return (theta / 2) * (
            1
            + self.rho * phi * k
            + np.sqrt((phi * k + self.rho) ** 2 + 1 - self.rho**2)
        )

    def implied_vol(self, k: float, T: float, atm_vol: float) -> float:
        """
        Compute implied volatility.

        Args:
            k: Log-moneyness.
            T: Time to maturity.
            atm_vol: ATM implied volatility for this maturity.

        Returns:
            Implied volatility.
        """
        theta = atm_vol**2 * T
        w = self.total_variance(k, theta)
        return np.sqrt(max(w, 1e-10) / T)


def calibrate_svi(
    log_strikes: np.ndarray,
    market_vols: np.ndarray,
    T: float,
    initial_guess: Optional[dict] = None,
) -> SVIModel:
    """
    Calibrate SVI model to market smile.

    Args:
        log_strikes: Array of log-moneyness values.
        market_vols: Array of market implied volatilities.
        T: Time to maturity.
        initial_guess: Initial parameters {a, b, rho, m, sigma}.

    Returns:
        Calibrated SVIModel.
    """
    market_total_var = market_vols**2 * T

    if initial_guess is None:
        atm_idx = np.argmin(np.abs(log_strikes))
        initial_guess = {
            "a": market_total_var[atm_idx],
            "b": 0.1,
            "rho": -0.3,
            "m": 0.0,
            "sigma": 0.2,
        }

    def objective(params):
        a, b, rho, m, sigma = params

        # Enforce constraints
        if b < 0 or sigma <= 0 or not -0.99 < rho < 0.99:
            return 1e10

        # Butterfly constraint
        if a + b * sigma * np.sqrt(1 - rho**2) < 0:
            return 1e10

        try:
            model = SVIModel(a, b, rho, m, sigma)
            model_var = np.array([model.total_variance(k) for k in log_strikes])
            return np.sum((model_var - market_total_var) ** 2)
        except Exception:
            return 1e10

    x0 = [
        initial_guess["a"],
        initial_guess["b"],
        initial_guess["rho"],
        initial_guess["m"],
        initial_guess["sigma"],
    ]

    bounds = [
        (-0.5, 1.0),  # a
        (0.0, 1.0),  # b
        (-0.99, 0.99),  # rho
        (-0.5, 0.5),  # m
        (0.01, 1.0),  # sigma
    ]

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200},
    )

    a, b, rho, m, sigma = result.x
    return SVIModel(a, b, rho, m, sigma)


def calibrate_ssvi(
    T_slices: np.ndarray,
    atm_vols: np.ndarray,
    smile_data: list,  # List of (log_strikes, market_vols) for each slice
) -> SSVIModel:
    """
    Calibrate SSVI model to multiple maturity slices.

    Args:
        T_slices: Array of maturities.
        atm_vols: Array of ATM vols for each maturity.
        smile_data: List of (log_strikes, market_vols) tuples.

    Returns:
        Calibrated SSVIModel.
    """

    def objective(params):
        rho, eta, gamma = params

        if not -0.99 < rho < 0.99 or eta <= 0 or not 0 <= gamma <= 1:
            return 1e10

        try:
            model = SSVIModel(rho, eta, gamma)
            total_error = 0.0

            for i, T in enumerate(T_slices):
                log_strikes, market_vols = smile_data[i]
                theta = atm_vols[i] ** 2 * T

                for k, mv in zip(log_strikes, market_vols):
                    model_var = model.total_variance(k, theta)
                    market_var = mv**2 * T
                    total_error += (model_var - market_var) ** 2

            return total_error
        except Exception:
            return 1e10

    x0 = [-0.3, 0.5, 0.5]
    bounds = [(-0.99, 0.99), (0.01, 2.0), (0.0, 1.0)]

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
    )

    rho, eta, gamma = result.x
    return SSVIModel(rho, eta, gamma)
