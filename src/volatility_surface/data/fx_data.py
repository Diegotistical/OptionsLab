# src/volatility_surface/data/fx_data.py
"""
EURUSD FX Volatility Surface Data Generator.

Generates realistic synthetic EURUSD FX volatility surfaces calibrated
to known FX smile parameters from academic literature. Used for cross-asset
validation of CINN when commercial FX vol data is unavailable.

FX smile characteristics that differ from equity:
    - Symmetric or slightly call-skewed (vs equity put skew)
    - ATM vol driven by central bank meeting dates
    - Delta-based quoting convention (25Δ, 10Δ risk reversals/butterflies)
    - Different liquidity profile (very liquid ATM, sparse wings)

Reference:
    Clark, I. (2011). Foreign Exchange Option Pricing. Wiley.
    Bossens, F. et al. (2010). Vanna-Volga methods applied to FX derivatives.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _fx_smile_vanna_volga(
    k: np.ndarray,
    T: float,
    atm_vol: float,
    rr25: float,
    bf25: float,
    rr10: float = None,
    bf10: float = None,
) -> np.ndarray:
    """
    FX implied vol smile using simplified Vanna-Volga construction.

    The FX vol market quotes:
        ATM: at-the-money vol (delta-neutral straddle)
        RR25: 25-delta risk reversal = σ(25Δcall) - σ(25Δput)
        BF25: 25-delta butterfly = 0.5*(σ(25Δcall) + σ(25Δput)) - ATM

    Args:
        k: Log-moneyness array.
        T: Time to maturity.
        atm_vol: ATM implied volatility.
        rr25: 25-delta risk reversal.
        bf25: 25-delta butterfly.
        rr10: 10-delta risk reversal (optional, for wing control).
        bf10: 10-delta butterfly (optional).

    Returns:
        Implied volatilities at each log-moneyness.
    """
    # Quadratic smile: σ(k) = ATM + a*k + b*k²
    # Calibrate a, b from RR and BF
    # 25-delta corresponds to roughly |k| ≈ 0.5*σ*√T for FX
    k_25 = 0.5 * atm_vol * np.sqrt(T)  # approximate 25-delta moneyness

    if k_25 < 0.01:
        k_25 = 0.01

    # a controls skew (from risk reversal)
    a = rr25 / (2.0 * k_25)

    # b controls curvature (from butterfly)
    b = bf25 / (k_25 ** 2)

    # Optional quartic term from 10-delta quotes
    if rr10 is not None and bf10 is not None:
        k_10 = 0.8 * atm_vol * np.sqrt(T)
        if k_10 < 0.02:
            k_10 = 0.02
        c = 0.0  # quartic disabled unless 10-delta available
        d = 0.0
        if abs(k_10) > abs(k_25):
            excess_bf = bf10 - bf25 * (k_10 / k_25) ** 2
            d = excess_bf / (k_10 ** 4) if abs(k_10 ** 4) > 1e-10 else 0.0
        smile = atm_vol + a * k + b * k**2 + d * k**4
    else:
        smile = atm_vol + a * k + b * k**2

    return np.maximum(smile, 0.005)  # Floor at 0.5% vol


def generate_eurusd_surface(
    n_expiries: int = 8,
    n_strikes_per_expiry: int = 15,
    regime: str = "normal",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic EURUSD FX volatility surface.

    Args:
        n_expiries: Number of maturity slices.
        n_strikes_per_expiry: Strikes per slice (before any dropout).
        regime: Market regime - "normal", "stress", or "low_vol".
        seed: Random seed.

    Returns:
        DataFrame with columns: log_moneyness, T, implied_volatility.
    """
    rng = np.random.default_rng(seed)

    # Maturity grid (FX-typical tenors, starting from 1M for stable calibration)
    T_candidates = np.array([
        1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0, 2.5, 3.0
    ])
    if n_expiries <= len(T_candidates):
        T_grid = T_candidates[:n_expiries]
    else:
        T_grid = np.linspace(1/52, 2.0, n_expiries)

    # Regime-dependent parameters
    if regime == "stress":
        base_atm = 0.14  # 14% ATM vol
        atm_slope = -0.02  # Backwardation in stress
        rr25_base = -0.015  # Slight USD-call skew in stress
        bf25_base = 0.012
        noise_scale = 0.003
    elif regime == "low_vol":
        base_atm = 0.06  # 6% ATM vol
        atm_slope = 0.005
        rr25_base = -0.003  # Near symmetric
        bf25_base = 0.004
        noise_scale = 0.001
    else:  # normal
        base_atm = 0.09  # 9% ATM vol
        atm_slope = 0.008  # Slight upward term structure
        rr25_base = -0.008  # Mild call skew (USD put = EUR call)
        bf25_base = 0.007  # Moderate smile curvature
        noise_scale = 0.002

    rows = []
    for T in T_grid:
        # ATM vol with term structure
        atm_vol = base_atm + atm_slope * np.sqrt(T) + rng.normal(0, noise_scale * 0.3)
        atm_vol = max(atm_vol, 0.02)

        # Risk reversal and butterfly (maturity-dependent)
        rr25 = rr25_base * (1 + 0.3 * np.sqrt(T)) + rng.normal(0, noise_scale * 0.2)
        bf25 = bf25_base * (1 + 0.2 * np.sqrt(T)) + rng.normal(0, noise_scale * 0.1)
        bf25 = max(bf25, 0.001)  # Butterfly must be positive

        # Generate strikes (FX convention: tighter around ATM)
        # Moneyness range narrower for FX than equity
        k_range = min(3.0 * atm_vol * np.sqrt(T), 0.20)
        k_range = max(k_range, 0.03)

        # Non-uniform: denser near ATM
        k_uniform = np.linspace(-k_range, k_range, n_strikes_per_expiry)
        # Add slight noise to prevent perfectly uniform grid
        k_vals = k_uniform + rng.normal(0, 0.001, len(k_uniform))
        k_vals = np.sort(k_vals)

        # Compute smile
        ivs = _fx_smile_vanna_volga(k_vals, T, atm_vol, rr25, bf25)

        # Add realistic market noise (bid-ask induced)
        noise = rng.normal(0, noise_scale, len(k_vals))
        ivs = np.maximum(ivs + noise, 0.005)

        for k, iv in zip(k_vals, ivs):
            rows.append({
                "log_moneyness": float(k),
                "T": float(T),
                "implied_volatility": float(iv),
            })

    return pd.DataFrame(rows)


def generate_eurusd_surface_panel(
    n_days: int = 100,
    n_expiries: int = 8,
    n_strikes_per_expiry: int = 15,
    regime_mix: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> List[pd.DataFrame]:
    """
    Generate a panel of EURUSD surfaces simulating multiple days.

    Args:
        n_days: Number of daily surfaces.
        n_expiries: Expiries per surface.
        n_strikes_per_expiry: Strikes per expiry.
        regime_mix: Probability of each regime. Default: 70% normal, 20% low_vol, 10% stress.
        seed: Random seed.

    Returns:
        List of DataFrames, one per day.
    """
    if regime_mix is None:
        regime_mix = {"normal": 0.7, "low_vol": 0.2, "stress": 0.1}

    rng = np.random.default_rng(seed)
    regimes = list(regime_mix.keys())
    probs = list(regime_mix.values())

    surfaces = []
    for day in range(n_days):
        regime = rng.choice(regimes, p=probs)
        surface = generate_eurusd_surface(
            n_expiries=n_expiries,
            n_strikes_per_expiry=n_strikes_per_expiry,
            regime=regime,
            seed=seed + day,
        )
        surface["day"] = day
        surface["regime"] = regime
        surfaces.append(surface)

    return surfaces
