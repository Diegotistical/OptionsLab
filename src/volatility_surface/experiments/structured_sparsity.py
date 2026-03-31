# src/volatility_surface/experiments/structured_sparsity.py
"""
Structured sparsity experiment for volatility surface models.

Replaces uniform random dropout with liquidity-weighted dropout that
mimics real-world missing data patterns:
  - ATM strikes are almost always present
  - Wings thin out with distance from ATM
  - Short-dated near-expiry options disappear
  - Illiquid names have wider gaps

This addresses the paper's most significant methodological limitation:
"uniform dropout preserves ATM information that structured sparsity
would remove, potentially flattering interpolation-based methods."

Usage:
    >>> from src.volatility_surface.experiments.structured_sparsity import (
    ...     LiquidityDropout,
    ...     run_structured_sparsity_experiment,
    ... )
    >>> dropout = LiquidityDropout(base_rate=0.4, wing_steepness=3.0)
    >>> mask = dropout.generate_mask(log_moneyness, ttm)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class LiquidityDropout:
    """
    Liquidity-aware dropout model for option strike grids.

    P(drop | k, T) = base_rate * liquidity_weight(k, T)

    where liquidity_weight increases with:
      - |k| (distance from ATM)
      - 1/T (proximity to expiry)
      - spread (bid-ask width, if available)

    The model is calibrated so that the total dropout rate matches
    the target base_rate on average, but the spatial distribution
    of missing strikes follows real-world liquidity patterns.

    Parameters:
        base_rate: Target fraction of strikes to drop (0.0 to 1.0)
        wing_steepness: How sharply dropout increases in the wings.
            Higher values = wings thin out faster. Default 3.0.
        expiry_decay: How much more likely short-dated options are to
            be missing. Default 2.0.
        atm_protection: Minimum probability of retaining ATM strikes.
            Default 0.95 (5% chance of dropping even ATM).
        seed: Random seed for reproducibility.
    """

    base_rate: float = 0.4
    wing_steepness: float = 3.0
    expiry_decay: float = 2.0
    atm_protection: float = 0.95
    seed: Optional[int] = None

    def _compute_drop_probability(
        self,
        log_moneyness: np.ndarray,
        ttm: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-strike dropout probability.

        P(drop | k, T) ∝ exp(steepness * |k|) * exp(-decay * T) * (1 - atm_protection * exp(-|k|/0.05))

        Normalized so mean(P) = base_rate.
        """
        abs_k = np.abs(log_moneyness)

        # Wing effect: exponential increase with distance from ATM
        wing_factor = np.exp(self.wing_steepness * abs_k)

        # Expiry effect: short-dated options more likely to be missing
        min_ttm = max(ttm.min(), 7 / 365)  # Floor at 1 week
        expiry_factor = np.exp(-self.expiry_decay * (ttm - min_ttm))

        # ATM protection: reduce dropout probability near ATM
        atm_factor = 1.0 - self.atm_protection * np.exp(-abs_k / 0.05)

        # Raw (unnormalized) dropout probability
        raw_prob = wing_factor * expiry_factor * atm_factor

        # Normalize so mean(prob) = base_rate
        raw_prob = raw_prob / raw_prob.mean() * self.base_rate

        # Clip to [0, 0.99] — never drop everything
        return np.clip(raw_prob, 0.0, 0.99)

    def generate_mask(
        self,
        log_moneyness: np.ndarray,
        ttm: np.ndarray,
    ) -> np.ndarray:
        """
        Generate a boolean mask where True = keep, False = drop.

        Returns:
            mask: Boolean array, same shape as log_moneyness
        """
        rng = np.random.RandomState(self.seed)
        drop_prob = self._compute_drop_probability(log_moneyness, ttm)
        uniform = rng.uniform(size=len(log_moneyness))
        return uniform > drop_prob  # True = keep

    def describe(self, log_moneyness: np.ndarray, ttm: np.ndarray) -> Dict[str, float]:
        """Return statistics about the dropout pattern."""
        drop_prob = self._compute_drop_probability(log_moneyness, ttm)
        return {
            "mean_drop_rate": float(drop_prob.mean()),
            "atm_drop_rate": float(drop_prob[np.abs(log_moneyness) < 0.02].mean()),
            "wing_drop_rate": float(drop_prob[np.abs(log_moneyness) > 0.15].mean()),
            "short_dated_drop_rate": float(drop_prob[ttm < 30 / 365].mean()),
            "long_dated_drop_rate": float(drop_prob[ttm > 0.5].mean()),
            "min_drop_prob": float(drop_prob.min()),
            "max_drop_prob": float(drop_prob.max()),
        }


def apply_structured_dropout(
    df: pd.DataFrame,
    base_rate: float = 0.4,
    wing_steepness: float = 3.0,
    expiry_decay: float = 2.0,
    seed: Optional[int] = None,
    moneyness_col: str = "log_moneyness",
    ttm_col: str = "T",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Apply structured dropout to an options DataFrame.

    Returns:
        train_df: Surviving quotes (for model calibration)
        held_out_df: Dropped quotes (for RMSE evaluation)
        stats: Dropout statistics
    """
    dropout = LiquidityDropout(
        base_rate=base_rate,
        wing_steepness=wing_steepness,
        expiry_decay=expiry_decay,
        seed=seed,
    )

    mask = dropout.generate_mask(
        df[moneyness_col].values,
        df[ttm_col].values,
    )

    stats = dropout.describe(
        df[moneyness_col].values,
        df[ttm_col].values,
    )
    stats["actual_drop_rate"] = 1.0 - mask.mean()
    stats["n_kept"] = int(mask.sum())
    stats["n_dropped"] = int((~mask).sum())

    return df[mask].copy(), df[~mask].copy(), stats


def compare_dropout_methods(
    df: pd.DataFrame,
    base_rate: float = 0.4,
    n_trials: int = 10,
    moneyness_col: str = "log_moneyness",
    ttm_col: str = "T",
) -> pd.DataFrame:
    """
    Compare uniform vs structured dropout statistics.

    Returns a DataFrame summarizing the spatial distribution of
    missing strikes under both methods.
    """
    results = []

    for trial in range(n_trials):
        seed = 42 + trial

        # Uniform dropout
        rng = np.random.RandomState(seed)
        uniform_mask = rng.uniform(size=len(df)) > base_rate
        uniform_stats = {
            "method": "uniform",
            "trial": trial,
            "drop_rate": 1.0 - uniform_mask.mean(),
            "atm_kept_pct": uniform_mask[np.abs(df[moneyness_col]) < 0.02].mean(),
            "wing_kept_pct": uniform_mask[np.abs(df[moneyness_col]) > 0.15].mean(),
        }

        # Structured dropout
        _, _, struct_stats = apply_structured_dropout(
            df, base_rate=base_rate, seed=seed,
            moneyness_col=moneyness_col, ttm_col=ttm_col,
        )
        struct_mask = np.zeros(len(df), dtype=bool)
        struct_mask[:struct_stats["n_kept"]] = True  # Approximate

        structured_stats = {
            "method": "structured",
            "trial": trial,
            "drop_rate": struct_stats["actual_drop_rate"],
            "atm_kept_pct": 1.0 - struct_stats["atm_drop_rate"],
            "wing_kept_pct": 1.0 - struct_stats["wing_drop_rate"],
        }

        results.append(uniform_stats)
        results.append(structured_stats)

    return pd.DataFrame(results)
