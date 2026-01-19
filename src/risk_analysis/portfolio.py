# src/risk_analysis/portfolio.py
"""
Portfolio-Level Greeks and Risk Aggregation.

Provides tools for managing portfolios of options and computing
aggregate risk metrics across positions.

Features:
    - Portfolio construction with multiple positions
    - Aggregate Greeks (Delta, Gamma, Vega, Theta, Rho)
    - Greeks attribution by underlying
    - Scenario-based P&L analysis
    - Position-level and portfolio-level reports

Usage:
    >>> from src.risk_analysis.portfolio import OptionsPortfolio
    >>> portfolio = OptionsPortfolio()
    >>> portfolio.add_position("AAPL_C100", pricer, 10, S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    >>> greeks = portfolio.aggregate_greeks()
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Position:
    """A single option position in the portfolio."""

    position_id: str
    pricer: Any  # Option pricer with price() and Greeks methods
    quantity: float  # Positive = long, negative = short
    params: Dict[str, Any]  # Pricing parameters (S, K, T, r, sigma, etc.)
    underlying: str = "default"  # Underlying identifier for grouping

    def price(self) -> float:
        """Compute position value."""
        unit_price = self.pricer.price(**self.params)
        return self.quantity * unit_price

    def greeks(self) -> Dict[str, float]:
        """Compute position Greeks."""
        result = {}

        # Try different Greek methods
        for greek in ["delta", "gamma", "vega", "theta", "rho"]:
            try:
                method = getattr(self.pricer, greek, None)
                if method is not None:
                    result[greek] = self.quantity * method(**self.params)
                else:
                    result[greek] = np.nan
            except Exception:
                result[greek] = np.nan

        return result


class OptionsPortfolio:
    """
    Portfolio of option positions with aggregate risk management.

    Supports multiple pricers, underlyings, and position types.
    """

    def __init__(self):
        self.positions: Dict[str, Position] = {}

    def add_position(
        self,
        position_id: str,
        pricer: Any,
        quantity: float,
        underlying: str = "default",
        **pricing_params,
    ) -> None:
        """
        Add a position to the portfolio.

        Args:
            position_id: Unique identifier for this position.
            pricer: Option pricer instance with price() method.
            quantity: Number of contracts (positive=long, negative=short).
            underlying: Identifier for the underlying asset.
            **pricing_params: Parameters passed to pricer (S, K, T, r, sigma, etc.)
        """
        self.positions[position_id] = Position(
            position_id=position_id,
            pricer=pricer,
            quantity=quantity,
            params=pricing_params,
            underlying=underlying,
        )

    def remove_position(self, position_id: str) -> None:
        """Remove a position from the portfolio."""
        if position_id in self.positions:
            del self.positions[position_id]

    def total_value(self) -> float:
        """Compute total portfolio value."""
        return sum(pos.price() for pos in self.positions.values())

    def aggregate_greeks(self) -> Dict[str, float]:
        """
        Compute aggregate portfolio Greeks.

        Returns:
            Dict with total delta, gamma, vega, theta, rho.
        """
        totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

        for pos in self.positions.values():
            greeks = pos.greeks()
            for greek in totals:
                if not np.isnan(greeks.get(greek, np.nan)):
                    totals[greek] += greeks[greek]

        return totals

    def greeks_by_underlying(self) -> pd.DataFrame:
        """
        Compute Greeks grouped by underlying.

        Returns:
            DataFrame with Greeks for each underlying.
        """
        data = {}

        for pos in self.positions.values():
            if pos.underlying not in data:
                data[pos.underlying] = {
                    "delta": 0.0,
                    "gamma": 0.0,
                    "vega": 0.0,
                    "theta": 0.0,
                    "rho": 0.0,
                    "value": 0.0,
                }

            greeks = pos.greeks()
            for greek in ["delta", "gamma", "vega", "theta", "rho"]:
                if not np.isnan(greeks.get(greek, np.nan)):
                    data[pos.underlying][greek] += greeks[greek]

            data[pos.underlying]["value"] += pos.price()

        return pd.DataFrame(data).T

    def position_report(self) -> pd.DataFrame:
        """
        Generate detailed position-level report.

        Returns:
            DataFrame with all positions and their Greeks.
        """
        rows = []

        for pos_id, pos in self.positions.items():
            greeks = pos.greeks()
            row = {
                "position_id": pos_id,
                "underlying": pos.underlying,
                "quantity": pos.quantity,
                "value": pos.price(),
                **greeks,
            }

            # Add key parameters
            for key in ["K", "T", "option_type"]:
                if key in pos.params:
                    row[key] = pos.params[key]

            rows.append(row)

        return pd.DataFrame(rows)

    def scenario_pnl(
        self,
        spot_shocks: List[float],
        vol_shocks: List[float],
    ) -> pd.DataFrame:
        """
        Compute P&L under spot and volatility scenarios.

        Args:
            spot_shocks: List of spot price multipliers (e.g., [0.9, 1.0, 1.1]).
            vol_shocks: List of volatility shocks (e.g., [-0.05, 0, 0.05]).

        Returns:
            DataFrame with P&L for each scenario.
        """
        base_value = self.total_value()
        results = []

        for spot_mult in spot_shocks:
            for vol_shock in vol_shocks:
                scenario_value = 0.0

                for pos in self.positions.values():
                    # Apply shocks
                    params = pos.params.copy()
                    if "S" in params:
                        params["S"] = params["S"] * spot_mult
                    if "sigma" in params:
                        params["sigma"] = max(0.01, params["sigma"] + vol_shock)

                    try:
                        scenario_value += pos.quantity * pos.pricer.price(**params)
                    except Exception:
                        scenario_value += pos.price()  # Fallback to base

                pnl = scenario_value - base_value

                results.append(
                    {
                        "spot_shock": f"{(spot_mult - 1) * 100:+.0f}%",
                        "vol_shock": f"{vol_shock * 100:+.0f}%",
                        "scenario_value": scenario_value,
                        "pnl": pnl,
                        "pnl_pct": pnl / base_value * 100 if base_value != 0 else 0,
                    }
                )

        return pd.DataFrame(results)

    def delta_hedge_ratio(self, underlying: str = None) -> float:
        """
        Compute shares needed to delta-hedge the portfolio.

        Args:
            underlying: Specific underlying to hedge (None = total).

        Returns:
            Number of shares to short (positive) or buy (negative).
        """
        if underlying is None:
            greeks = self.aggregate_greeks()
            return -greeks["delta"]

        total_delta = 0.0
        for pos in self.positions.values():
            if pos.underlying == underlying:
                greeks = pos.greeks()
                if not np.isnan(greeks.get("delta", np.nan)):
                    total_delta += greeks["delta"]

        return -total_delta

    def vega_exposure(self) -> Dict[str, float]:
        """
        Compute vega exposure by maturity bucket.

        Returns:
            Dict with vega by time bucket (0-3m, 3-6m, 6-12m, 1y+).
        """
        buckets = {
            "0-3m": 0.0,
            "3-6m": 0.0,
            "6-12m": 0.0,
            "1y+": 0.0,
        }

        for pos in self.positions.values():
            T = pos.params.get("T", 1.0)
            greeks = pos.greeks()
            vega = greeks.get("vega", 0.0)

            if np.isnan(vega):
                continue

            if T <= 0.25:
                buckets["0-3m"] += vega
            elif T <= 0.5:
                buckets["3-6m"] += vega
            elif T <= 1.0:
                buckets["6-12m"] += vega
            else:
                buckets["1y+"] += vega

        return buckets
