# src/greeks/vol_surface_pricer.py
"""
VolSurfacePricer — PricerProtocol adapter for volatility surface models.

Bridges vol surface models (CINN, SVI, MLP, etc.) to the unified Greeks
computation framework. Given a trained surface model, this adapter:
1. Queries the surface for sigma(K, T) at the requested strike/maturity
2. Prices the option via Black-Scholes using the surface-implied vol
3. Returns the price, enabling unified Greeks computation via finite differences

Usage:
    >>> from src.greeks.vol_surface_pricer import VolSurfacePricer
    >>> from src.greeks.unified_greeks import compute_greeks_unified
    >>> pricer = VolSurfacePricer(trained_surface_model, S_ref=100.0)
    >>> greeks = compute_greeks_unified(pricer, S=100, K=105, T=0.25, r=0.05, sigma=0.2)
"""

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm


class VolSurfacePricer:
    """
    Adapter that makes a vol surface model conform to PricerProtocol.

    The sigma parameter in price() is ignored — instead, the model's
    predicted implied vol at (K/S, T) is used.
    """

    def __init__(self, vol_surface_model, S_ref: float = 100.0):
        """
        Args:
            vol_surface_model: Trained model with predict_volatility(df) method.
                Expected to accept DataFrame with columns [log_moneyness, T].
            S_ref: Reference spot price used during model training.
                Used to compute log-moneyness = log(K / S_ref).
        """
        self.model = vol_surface_model
        self.S_ref = S_ref

    def _get_surface_vol(self, S: float, K: float, T: float) -> float:
        """Query the vol surface model for implied vol at (K, T)."""
        log_moneyness = np.log(K / S)
        df = pd.DataFrame({"log_moneyness": [log_moneyness], "T": [max(T, 1e-6)]})
        vol = self.model.predict_volatility(df)[0]
        return float(np.clip(vol, 0.01, 5.0))

    def _bs_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        q: float = 0.0,
    ) -> float:
        """Black-Scholes price."""
        if T <= 1e-10:
            if option_type == "call":
                return max(S - K, 0.0)
            else:
                return max(K - S, 0.0)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(
                d2
            )
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(
                -q * T
            ) * norm.cdf(-d1)

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call",
        q: float = 0.0,
        **kwargs,
    ) -> float:
        """
        Price an option using vol-surface-derived implied volatility.

        The `sigma` parameter is IGNORED. Instead, the surface model
        is queried for sigma(log(K/S), T).
        """
        surface_vol = self._get_surface_vol(S, K, T)
        return self._bs_price(S, K, T, r, surface_vol, option_type, q)
