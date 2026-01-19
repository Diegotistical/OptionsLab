# src/pricing_models/sabr.py
"""
SABR Stochastic Alpha Beta Rho Model.

Implements the SABR model (Hagan et al., 2002) for implied volatility:

    dF = σ F^β dW₁
    dσ = α σ dW₂
    dW₁ dW₂ = ρ dt

Parameters:
    α (alpha): Initial volatility (ATM vol proxy)
    β (beta): CEV exponent (0 = normal, 1 = lognormal)
    ρ (rho): Correlation between forward and volatility
    ν (nu): Volatility of volatility

Features:
    - Hagan et al. (2002) implied volatility approximation
    - Obloj (2008) correction for ATM
    - Calibration to market smile

Reference:
    Hagan, P. S., et al. (2002). Managing Smile Risk.
    Wilmott Magazine, September 2002.

Usage:
    >>> from src.pricing_models.sabr import SABRModel
    >>> sabr = SABRModel(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
    >>> iv = sabr.implied_vol(F=100, K=105, T=1.0)
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class SABRModel:
    """
    SABR model for implied volatility smile.

    Attributes:
        alpha: Initial volatility level.
        beta: CEV exponent (0 = normal, 1 = lognormal).
        rho: Correlation between forward and volatility.
        nu: Volatility of volatility.
    """

    alpha: float  # Initial vol
    beta: float  # CEV exponent
    rho: float  # Correlation
    nu: float  # Vol of vol

    def __post_init__(self):
        """Validate SABR parameters."""
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0 <= self.beta <= 1:
            raise ValueError("beta must be in [0, 1]")
        if not -1 <= self.rho <= 1:
            raise ValueError("rho must be in [-1, 1]")
        if self.nu < 0:
            raise ValueError("nu must be non-negative")

    def implied_vol(
        self,
        F: float,
        K: float,
        T: float,
    ) -> float:
        """
        Compute SABR implied volatility using Hagan approximation.

        Args:
            F: Forward price.
            K: Strike price.
            T: Time to maturity.

        Returns:
            Implied volatility (lognormal).
        """
        if T <= 0:
            return self.alpha

        alpha, beta, rho, nu = self.alpha, self.beta, self.rho, self.nu

        # Handle ATM case
        if abs(K - F) < 1e-10:
            return self._atm_vol(F, T)

        # Log-moneyness
        log_FK = np.log(F / K)
        FK_mid = (F * K) ** ((1 - beta) / 2)

        # z parameter
        z = (nu / alpha) * FK_mid * log_FK

        # x(z) function
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

        # Handle z close to 0
        if abs(z) < 1e-7:
            zeta = 1.0
        else:
            zeta = z / x_z

        # Prefactor
        prefactor = alpha / (
            FK_mid
            * (
                1
                + ((1 - beta) ** 2 / 24) * log_FK**2
                + ((1 - beta) ** 4 / 1920) * log_FK**4
            )
        )

        # Correction terms
        term1 = ((1 - beta) ** 2 / 24) * alpha**2 / (FK_mid**2)
        term2 = 0.25 * rho * beta * nu * alpha / FK_mid
        term3 = (2 - 3 * rho**2) * nu**2 / 24

        correction = 1 + (term1 + term2 + term3) * T

        return prefactor * zeta * correction

    def _atm_vol(self, F: float, T: float) -> float:
        """ATM implied volatility with Obloj correction."""
        alpha, beta, rho, nu = self.alpha, self.beta, self.rho, self.nu

        F_beta = F ** (1 - beta)

        term1 = ((1 - beta) ** 2 / 24) * alpha**2 / (F_beta**2)
        term2 = 0.25 * rho * beta * nu * alpha / F_beta
        term3 = (2 - 3 * rho**2) * nu**2 / 24

        return (alpha / F_beta) * (1 + (term1 + term2 + term3) * T)

    def smile(
        self,
        F: float,
        T: float,
        strikes: np.ndarray,
    ) -> np.ndarray:
        """
        Compute implied volatility smile for array of strikes.

        Args:
            F: Forward price.
            T: Time to maturity.
            strikes: Array of strike prices.

        Returns:
            Array of implied volatilities.
        """
        return np.array([self.implied_vol(F, K, T) for K in strikes])

    def price(
        self,
        F: float,
        K: float,
        T: float,
        r: float,
        option_type: Literal["call", "put"] = "call",
    ) -> float:
        """
        Price option using SABR implied volatility in Black formula.

        Args:
            F: Forward price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate (for discounting).
            option_type: 'call' or 'put'.

        Returns:
            Option price.
        """
        sigma = self.implied_vol(F, K, T)
        return self._black_price(F, K, T, r, sigma, option_type)

    @staticmethod
    def _black_price(
        F: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
    ) -> float:
        """Black (1976) formula for forward pricing."""
        if T <= 0:
            intrinsic = max(F - K, 0) if option_type == "call" else max(K - F, 0)
            return np.exp(-r * T) * intrinsic

        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def calibrate_sabr(
    F: float,
    T: float,
    strikes: np.ndarray,
    market_vols: np.ndarray,
    beta: float = 0.5,
    initial_guess: Optional[dict] = None,
) -> SABRModel:
    """
    Calibrate SABR model to market implied volatilities.

    Args:
        F: Forward price.
        T: Time to maturity.
        strikes: Array of market strikes.
        market_vols: Array of market implied volatilities.
        beta: Fixed beta (typically 0, 0.5, or 1).
        initial_guess: Initial guess for (alpha, rho, nu).

    Returns:
        Calibrated SABRModel.
    """
    if initial_guess is None:
        # Start with ATM vol as alpha estimate
        atm_idx = np.argmin(np.abs(strikes - F))
        initial_guess = {
            "alpha": market_vols[atm_idx] * (F ** (1 - beta)),
            "rho": -0.3,
            "nu": 0.4,
        }

    def objective(params):
        alpha, rho, nu = params

        if alpha <= 0 or not -0.99 <= rho <= 0.99 or nu < 0:
            return 1e10

        try:
            model = SABRModel(alpha, beta, rho, nu)
            model_vols = model.smile(F, T, strikes)
            return np.sum((model_vols - market_vols) ** 2)
        except Exception:
            return 1e10

    x0 = [initial_guess["alpha"], initial_guess["rho"], initial_guess["nu"]]
    bounds = [(0.001, 2.0), (-0.99, 0.99), (0.001, 2.0)]

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200},
    )

    alpha, rho, nu = result.x
    return SABRModel(alpha, beta, rho, nu)
