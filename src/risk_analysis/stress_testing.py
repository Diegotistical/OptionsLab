# src/risk_analysis/stress_testing.py

import logging
import threading
from typing import Callable, Iterable, List

import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


class StressScenario:
    """Container for a single stress scenario."""

    def __init__(
        self, name: str, field: str, magnitude: float, shock_type: str = "relative"
    ):
        if shock_type not in ("relative", "absolute"):
            raise ValueError("shock_type must be 'relative' or 'absolute'")
        self.name = name
        self.field = field
        self.magnitude = magnitude
        self.shock_type = shock_type

    def __repr__(self) -> str:
        return f"StressScenario(name={self.name}, field={self.field}, mag={self.magnitude}, type={self.shock_type})"


class StressTester:
    """
    Apply stress scenarios to a market DataFrame and compute shocked valuations and P&L.

    price_fn: Callable that accepts a market_df and returns 1D array of prices in row order.
    Thread-safe for multiple scenario runs.
    """

    def __init__(self, price_fn: Callable[[pd.DataFrame], np.ndarray]):
        if not callable(price_fn):
            raise TypeError("price_fn must be callable")
        self.price_fn = price_fn
        self._lock = threading.RLock()

    def _apply_scenario(
        self, market_df: pd.DataFrame, scenario: StressScenario
    ) -> pd.DataFrame:
        df = market_df.copy()
        if scenario.field not in df.columns:
            raise ValueError(f"Market DataFrame has no column '{scenario.field}'")
        if scenario.shock_type == "relative":
            df[scenario.field] = df[scenario.field] * (1.0 + scenario.magnitude)
        else:
            df[scenario.field] = df[scenario.field] + scenario.magnitude
        return df

    def run_scenarios(
        self, market_df: pd.DataFrame, scenarios: Iterable[StressScenario]
    ) -> pd.DataFrame:
        """
        Run scenarios and return a DataFrame summarizing P&L impact per scenario.
        Columns: total_pnl, mean_pnl, median_pnl, worst_pnl, es_95
        """
        if not isinstance(market_df, pd.DataFrame):
            raise TypeError("market_df must be a pandas.DataFrame")

        with self._lock:
            baseline = np.asarray(self.price_fn(market_df), dtype=float)
            baseline_total = baseline.sum()

            rows = []
            for s in scenarios:
                shocked = self._apply_scenario(market_df, s)
                shocked_prices = np.asarray(self.price_fn(shocked), dtype=float)
                pnl = shocked_prices - baseline  # per-instrument pnl
                total_pnl = shocked_prices.sum() - baseline_total
                es_95 = None
                try:
                    # ES across instruments: expected loss in worst 5% of instruments
                    cutoff = np.quantile(pnl, 0.05)
                    tail = pnl[pnl <= cutoff]
                    es_95 = (
                        -float(np.mean(tail)) if tail.size > 0 else float(np.min(pnl))
                    )
                except Exception:
                    es_95 = float(np.nan)
                rows.append(
                    {
                        "scenario": s.name,
                        "total_pnl": float(total_pnl),
                        "mean_pnl": float(np.mean(pnl)),
                        "median_pnl": float(np.median(pnl)),
                        "worst_pnl": float(np.min(pnl)),
                        "es_95": es_95,
                    }
                )

            report = pd.DataFrame(rows).set_index("scenario")
            return report
