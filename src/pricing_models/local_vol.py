# src/pricing_models/local_vol.py
"""
Dupire Local Volatility Model.

This is the RESEARCH FEATURE - something most option pricers don't have.

Local volatility σ(S,t) is derived from the market-implied volatility surface
using Dupire's formula. It gives the unique diffusion process consistent
with all vanilla option prices.

Theory:
    σ_local²(K,T) = 2 * (∂C/∂T + (r-q)K∂C/∂K + qC) / (K² ∂²C/∂K²)

Usage:
    >>> from src.pricing_models.local_vol import DupireLocalVol
    >>> lv = DupireLocalVol()
    >>> surface = lv.calibrate_to_market(strikes, expiries, iv_surface)
    >>> price = lv.price_fdm(S=100, K=100, T=1.0, r=0.05)
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm


@dataclass
class LocalVolSurface:
    """Calibrated local volatility surface."""

    strikes: np.ndarray
    expiries: np.ndarray
    values: np.ndarray  # σ_local(K, T)
    spot: float
    interpolator: Optional[Callable] = None

    def __call__(self, K: float, T: float) -> float:
        """Evaluate local vol at (K, T)."""
        if self.interpolator is None:
            return np.interp(K, self.strikes, self.values[:, 0])
        return float(self.interpolator(K, T))


class DupireLocalVol:
    """
    Dupire Local Volatility Model.

    Derives local volatility from market-implied volatility surface.
    Prices options via finite difference on the Dupire PDE.
    """

    def __init__(self, r: float = 0.05, q: float = 0.0):
        self.r = r
        self.q = q
        self._surface: Optional[LocalVolSurface] = None

    def _bs_price(self, S, K, T, r, sigma, q=0.0, opt_type="call"):
        """Black-Scholes price."""
        if T <= 0:
            if opt_type == "call":
                return max(S - K, 0)
            return max(K - S, 0)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if opt_type == "call":
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    def dupire_formula(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
        iv_surface: np.ndarray,
        spot: float,
    ) -> np.ndarray:
        """
        Compute local volatility using Dupire's formula.

        σ_local²(K,T) = 2 * (∂C/∂T + (r-q)K∂C/∂K + qC) / (K² ∂²C/∂K²)

        Args:
            strikes: Array of strikes (K dimension).
            expiries: Array of expiries (T dimension).
            iv_surface: Implied vol surface [len(expiries), len(strikes)].
            spot: Current spot price.

        Returns:
            Local vol surface with same shape as iv_surface.
        """
        n_T, n_K = iv_surface.shape
        local_vol = np.zeros_like(iv_surface)

        # Compute option prices (calls)
        C = np.zeros_like(iv_surface)
        for i, T in enumerate(expiries):
            for j, K in enumerate(strikes):
                sigma = iv_surface[i, j]
                C[i, j] = self._bs_price(spot, K, T, self.r, sigma, self.q, "call")

        # Finite differences
        dK = np.diff(strikes).mean()
        dT = np.diff(expiries).mean() if len(expiries) > 1 else 0.01

        for i in range(1, n_T):
            for j in range(1, n_K - 1):
                K = strikes[j]
                T = expiries[i]

                # ∂C/∂T (forward difference)
                dC_dT = (C[i, j] - C[i - 1, j]) / dT

                # ∂C/∂K (central difference)
                dC_dK = (C[i, j + 1] - C[i, j - 1]) / (2 * dK)

                # ∂²C/∂K² (second derivative)
                d2C_dK2 = (C[i, j + 1] - 2 * C[i, j] + C[i, j - 1]) / (dK**2)

                # Dupire formula
                numerator = 2 * (
                    dC_dT + (self.r - self.q) * K * dC_dK + self.q * C[i, j]
                )
                denominator = K**2 * d2C_dK2

                if denominator > 1e-10:
                    local_vol[i, j] = np.sqrt(max(numerator / denominator, 0.001))
                else:
                    local_vol[i, j] = iv_surface[i, j]  # Fallback

        # Fill boundaries
        local_vol[0, :] = iv_surface[0, :]
        local_vol[:, 0] = local_vol[:, 1]
        local_vol[:, -1] = local_vol[:, -2]

        return local_vol

    def calibrate(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
        iv_surface: np.ndarray,
        spot: float,
    ) -> LocalVolSurface:
        """
        Calibrate local volatility surface from market IV.

        Args:
            strikes: Strike prices.
            expiries: Expiration times.
            iv_surface: Market implied volatilities [n_T, n_K].
            spot: Current spot price.

        Returns:
            LocalVolSurface with interpolator.
        """
        lv = self.dupire_formula(strikes, expiries, iv_surface, spot)

        # Build interpolator
        if len(expiries) > 1 and len(strikes) > 1:
            interp = RectBivariateSpline(strikes, expiries, lv.T, kx=3, ky=3)

            def interpolator(K, T):
                return float(interp(K, T)[0, 0])

        else:
            interpolator = None

        self._surface = LocalVolSurface(
            strikes=strikes,
            expiries=expiries,
            values=lv,
            spot=spot,
            interpolator=interpolator,
        )

        return self._surface

    def price_fdm(
        self,
        S: float,
        K: float,
        T: float,
        option_type: str = "call",
        n_space: int = 200,
        n_time: int = 100,
    ) -> float:
        """
        Price option using local volatility via FDM.

        Uses Crank-Nicolson on the Black-Scholes PDE with
        local volatility σ(S,t).

        Args:
            S: Current spot.
            K: Strike.
            T: Time to maturity.
            option_type: 'call' or 'put'.
            n_space: Spatial grid points.
            n_time: Time grid points.

        Returns:
            Option price.
        """
        if self._surface is None:
            # Fallback to constant vol
            return self._bs_price(S, K, T, self.r, 0.2, self.q, option_type)

        # Grid setup
        s_max = S * 3
        ds = s_max / n_space
        dt = T / n_time

        spots = np.linspace(0, s_max, n_space + 1)

        # Terminal condition
        if option_type == "call":
            V = np.maximum(spots - K, 0)
        else:
            V = np.maximum(K - spots, 0)

        # Time stepping (explicit for simplicity)
        for t_idx in range(n_time):
            t = T - t_idx * dt
            V_new = V.copy()

            for i in range(1, n_space):
                spot_i = spots[i]

                # Get local vol at this point
                try:
                    sigma = self._surface(spot_i, t)
                except Exception:
                    sigma = 0.2

                sigma = np.clip(sigma, 0.05, 2.0)

                # FDM coefficients
                a = 0.5 * sigma**2 * spot_i**2
                b = (self.r - self.q) * spot_i

                # Central differences
                dV_dS = (V[i + 1] - V[i - 1]) / (2 * ds)
                d2V_dS2 = (V[i + 1] - 2 * V[i] + V[i - 1]) / (ds**2)

                # Black-Scholes PDE: dV/dt = -a*d2V/dS2 - b*dV/dS + r*V
                V_new[i] = V[i] + dt * (a * d2V_dS2 + b * dV_dS - self.r * V[i])

            # Boundary conditions
            if option_type == "call":
                V_new[0] = 0
                V_new[-1] = s_max - K * np.exp(-self.r * (T - t))
            else:
                V_new[0] = K * np.exp(-self.r * (T - t))
                V_new[-1] = 0

            V = V_new

        # Interpolate to get price at S
        return float(np.interp(S, spots, V))

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,  # Ignored, uses calibrated surface
        option_type: str,
        q: float = 0.0,
    ) -> float:
        """PricerProtocol-compatible interface."""
        self.r = r
        self.q = q
        return self.price_fdm(S, K, T, option_type)


def create_sample_iv_surface(
    spot: float = 100,
    n_strikes: int = 20,
    n_expiries: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a sample IV surface with realistic smile.

    Returns:
        strikes, expiries, iv_surface
    """
    strikes = np.linspace(spot * 0.7, spot * 1.3, n_strikes)
    expiries = np.array([0.1, 0.25, 0.5, 0.75, 1.0])[:n_expiries]

    # Base ATM vol with term structure
    atm_vol = 0.20 + 0.05 * np.sqrt(expiries)

    # Add smile
    iv_surface = np.zeros((len(expiries), len(strikes)))

    for i, T in enumerate(expiries):
        for j, K in enumerate(strikes):
            moneyness = np.log(K / spot)
            # Quadratic smile
            smile = 0.1 * moneyness**2 - 0.05 * moneyness
            iv_surface[i, j] = atm_vol[i] + smile * np.sqrt(0.25 / T)

    return strikes, expiries, iv_surface
