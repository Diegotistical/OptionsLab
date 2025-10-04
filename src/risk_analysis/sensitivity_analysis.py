# src/risk_analysis/sensitivity_analysis.py

import logging
from typing import Callable, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SensitivityAnalysis:
    """
    Finite-difference sensitivities (Delta, Gamma, Vega) computed in a model-agnostic way.

    The pricing function must accept a pandas.DataFrame (market state) and return a 1D array-like
    of prices ordered the same as the DataFrame rows.
    """

    @staticmethod
    def _validate_market_df(market_df: pd.DataFrame) -> None:
        if not isinstance(market_df, pd.DataFrame):
            raise TypeError("market_df must be a pandas.DataFrame")

    def compute_delta(
        self,
        market_df: pd.DataFrame,
        price_fn: Callable[[pd.DataFrame], np.ndarray],
        bump: float = 1e-2,
        bump_type: str = "relative",
    ) -> np.ndarray:
        """
        Central finite-difference delta: (P(S*(1+h)) - P(S*(1-h))) / (S_up - S_down)
        """
        self._validate_market_df(market_df)
        if "underlying_price" not in market_df.columns:
            raise ValueError("market_df must contain 'underlying_price' column")

        base_prices = np.asarray(price_fn(market_df), dtype=float)
        up = market_df.copy()
        down = market_df.copy()

        if bump_type == "relative":
            up["underlying_price"] = up["underlying_price"] * (1.0 + bump)
            down["underlying_price"] = down["underlying_price"] * (1.0 - bump)
            denom = up["underlying_price"].values - down["underlying_price"].values
        else:
            up["underlying_price"] = up["underlying_price"] + bump
            down["underlying_price"] = down["underlying_price"] - bump
            denom = up["underlying_price"].values - down["underlying_price"].values

        p_up = np.asarray(price_fn(up), dtype=float)
        p_down = np.asarray(price_fn(down), dtype=float)

        delta = (p_up - p_down) / denom
        return delta

    def compute_gamma(
        self,
        market_df: pd.DataFrame,
        price_fn: Callable[[pd.DataFrame], np.ndarray],
        bump: float = 1e-2,
        bump_type: str = "relative",
    ) -> np.ndarray:
        """
        Central finite-difference gamma: (P(S+h) - 2P(S) + P(S-h)) / h^2
        """
        self._validate_market_df(market_df)
        if "underlying_price" not in market_df.columns:
            raise ValueError("market_df must contain 'underlying_price' column")

        S = market_df["underlying_price"].values
        if bump_type == "relative":
            h_up = S * (1.0 + bump)
            h_down = S * (1.0 - bump)
        else:
            h_up = S + bump
            h_down = S - bump

        up = market_df.copy()
        mid = market_df.copy()
        down = market_df.copy()
        up["underlying_price"] = h_up
        down["underlying_price"] = h_down

        p_up = np.asarray(price_fn(up), dtype=float)
        p_mid = np.asarray(price_fn(mid), dtype=float)
        p_down = np.asarray(price_fn(down), dtype=float)

        # h^2 term: use average spacing squared for stability
        h = (h_up - h_down) / 2.0
        denom = (2.0 * h) ** 2 / 4.0  # simplifies to h^2
        gamma = (p_up - 2.0 * p_mid + p_down) / (h**2)
        return gamma

    def compute_vega(
        self,
        market_df: pd.DataFrame,
        price_fn: Callable[[pd.DataFrame], np.ndarray],
        bump: float = 1e-4,
    ) -> np.ndarray:
        """
        Vega via central finite difference on implied volatility.
        Expects column 'implied_volatility' (or 'implied_vol').
        """
        self._validate_market_df(market_df)
        vol_col = (
            "implied_volatility"
            if "implied_volatility" in market_df.columns
            else "implied_vol"
        )
        if vol_col not in market_df.columns:
            raise ValueError(
                "market_df must contain 'implied_volatility' or 'implied_vol' column"
            )

        up = market_df.copy()
        down = market_df.copy()
        up[vol_col] = up[vol_col] + bump
        down[vol_col] = down[vol_col] - bump

        p_up = np.asarray(price_fn(up), dtype=float)
        p_down = np.asarray(price_fn(down), dtype=float)

        vega = (p_up - p_down) / (2.0 * bump)
        return vega

    def compute_all(
        self,
        market_df: pd.DataFrame,
        price_fn: Callable[[pd.DataFrame], np.ndarray],
        delta_bump: float = 1e-2,
        vega_bump: float = 1e-4,
    ) -> Dict[str, np.ndarray]:
        return {
            "delta": self.compute_delta(market_df, price_fn, bump=delta_bump),
            "gamma": self.compute_gamma(market_df, price_fn, bump=delta_bump),
            "vega": self.compute_vega(market_df, price_fn, bump=vega_bump),
        }
