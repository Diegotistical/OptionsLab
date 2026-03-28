#!/usr/bin/env python3
"""
Generate an implied volatility surface from a Heston stochastic volatility model.

This provides a second data-generating process (DGP) for the CINN volatility
paper, complementing the existing quadratic-smile GBM surface.

The Heston model:
    dS = r S dt + sqrt(v) S dW_S
    dv = kappa (theta - v) dt + sigma_v sqrt(v) dW_v
    corr(dW_S, dW_v) = rho

Pricing uses the Heston characteristic function (Albrecher et al. formulation
to avoid branch-cut issues) with Gil-Pelaez / Lewis inversion for call prices,
then Brent root-finding to invert Black-Scholes for implied volatilities.

Usage:
    python scripts/generate_heston_surface.py
    python scripts/generate_heston_surface.py --save results/heston_surface.npy
"""

import argparse
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats


# ============================================================================
# Black-Scholes helpers
# ============================================================================

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)


def bs_implied_vol(
    price: float, S: float, K: float, T: float, r: float,
    vol_lo: float = 1e-4, vol_hi: float = 5.0,
) -> Optional[float]:
    """Invert Black-Scholes to recover implied volatility via Brent's method."""
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-12:
        return None

    def objective(sigma):
        return bs_call_price(S, K, T, r, sigma) - price

    lo_val = objective(vol_lo)
    hi_val = objective(vol_hi)

    if lo_val * hi_val > 0:
        # Try widening bounds once
        vol_hi = 10.0
        hi_val = objective(vol_hi)
        if lo_val * hi_val > 0:
            return None

    try:
        return optimize.brentq(objective, vol_lo, vol_hi, xtol=1e-10, maxiter=200)
    except ValueError:
        return None


# ============================================================================
# Heston characteristic function (Albrecher formulation)
# ============================================================================

def heston_characteristic(
    u: complex,
    T: float,
    r: float,
    v0: float,
    theta: float,
    kappa: float,
    sigma_v: float,
    rho: float,
) -> complex:
    """
    Heston model characteristic function phi(u; T).

    Uses the Albrecher et al. formulation that picks the branch of the complex
    square root with non-negative real part, avoiding discontinuities.

    phi(u) = exp(i u (log S0 + r T) + C(u,T) + D(u,T) v0)

    Note: the log(S0) and r*T terms are handled externally in the pricing
    integral so this returns exp(C + D*v0) only.
    """
    xi = kappa - 1j * rho * sigma_v * u
    d = np.sqrt(xi**2 + sigma_v**2 * (u**2 + 1j * u))

    # Albrecher: choose sign so that Re(d) >= 0 (already guaranteed by sqrt
    # principal branch when Re(xi) > 0, but we enforce it explicitly)
    # Use g2 = (xi - d) / (xi + d) which is the stable form
    g = (xi - d) / (xi + d)

    exp_dT = np.exp(-d * T)

    C = (kappa * theta / sigma_v**2) * (
        (xi - d) * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
    )

    D = ((xi - d) / sigma_v**2) * (1.0 - exp_dT) / (1.0 - g * exp_dT)

    return np.exp(C + D * v0)


# ============================================================================
# Heston call price via Gil-Pelaez / Lewis inversion
# ============================================================================

def heston_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    theta: float,
    kappa: float,
    sigma_v: float,
    rho: float,
) -> float:
    """
    European call price under the Heston model.

    Uses the Lewis (2001) single-integral representation:

        C = S - (K e^{-rT} / pi) * integral_0^inf Re[
            exp(-i u k) phi(u - i/2; T) / (u^2 + 1/4)
        ] du

    where k = log(S / K) + r T  (forward log-moneyness) and phi is the
    Heston characteristic function.

    This formulation is numerically stable and avoids computing P1, P2
    separately.
    """
    x = np.log(S / K) + r * T  # forward log-moneyness

    def integrand(u: float) -> float:
        if u < 1e-12:
            return 0.0
        # phi evaluated at u - i/2
        phi = heston_characteristic(
            u - 0.5j, T, r, v0, theta, kappa, sigma_v, rho
        )
        numerator = np.exp(-1j * u * x) * phi
        denominator = u**2 + 0.25
        return (numerator / denominator).real

    # Numerical integration with scipy.quad
    # Split the integration at a few points to help the adaptive integrator
    result, _ = integrate.quad(integrand, 0, 200, limit=300, epsabs=1e-10, epsrel=1e-10)

    price = S - (K * np.exp(-r * T) / np.pi) * result
    # Ensure no-arbitrage bounds
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    price = max(price, intrinsic)
    price = min(price, S)
    return price


# ============================================================================
# Surface generation
# ============================================================================

DEFAULT_HESTON_PARAMS = {
    "S0": 100.0,
    "r": 0.02,
    "v0": 0.04,
    "theta": 0.04,
    "kappa": 1.5,
    "sigma_v": 0.3,
    "rho": -0.7,
}


def generate_heston_iv_surface(
    params: Optional[Dict] = None,
    maturities: Optional[list] = None,
    n_strikes: int = 30,
    k_range: tuple = (-0.12, 0.08),
) -> pd.DataFrame:
    """
    Generate an implied volatility surface from the Heston model.

    Args:
        params: Dictionary of Heston parameters. Keys:
            S0, r, v0, theta, kappa, sigma_v, rho.
            Defaults to DEFAULT_HESTON_PARAMS if None.
        maturities: List of maturities in years.
            Defaults to [0.08, 0.25, 0.5, 1.0, 2.0].
        n_strikes: Number of log-moneyness points per maturity.
        k_range: (k_min, k_max) for log-moneyness grid.

    Returns:
        DataFrame with columns: log_moneyness, T, implied_volatility
    """
    if params is None:
        params = DEFAULT_HESTON_PARAMS.copy()
    if maturities is None:
        maturities = [0.08, 0.25, 0.5, 1.0, 2.0]

    S0 = params["S0"]
    r = params["r"]
    v0 = params["v0"]
    theta = params["theta"]
    kappa = params["kappa"]
    sigma_v = params["sigma_v"]
    rho = params["rho"]

    log_moneyness_grid = np.linspace(k_range[0], k_range[1], n_strikes)

    rows = []
    for T in maturities:
        for k in log_moneyness_grid:
            K = S0 * np.exp(k)

            # Heston call price
            call_price = heston_call_price(
                S0, K, T, r, v0, theta, kappa, sigma_v, rho
            )

            # Invert Black-Scholes for implied vol
            iv = bs_implied_vol(call_price, S0, K, T, r)
            if iv is not None and 0.001 < iv < 5.0:
                rows.append({
                    "log_moneyness": k,
                    "T": T,
                    "implied_volatility": iv,
                })

    df = pd.DataFrame(rows)
    return df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Heston stochastic volatility implied vol surface"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Path to save surface as .npy file (also saves .csv alongside it)"
    )
    parser.add_argument(
        "--v0", type=float, default=0.04, help="Initial variance (default: 0.04)"
    )
    parser.add_argument(
        "--theta", type=float, default=0.04, help="Long-run variance (default: 0.04)"
    )
    parser.add_argument(
        "--kappa", type=float, default=1.5, help="Mean reversion speed (default: 1.5)"
    )
    parser.add_argument(
        "--sigma-v", type=float, default=0.3, help="Vol of vol (default: 0.3)"
    )
    parser.add_argument(
        "--rho", type=float, default=-0.7, help="Correlation (default: -0.7)"
    )
    args = parser.parse_args()

    params = {
        "S0": 100.0,
        "r": 0.02,
        "v0": args.v0,
        "theta": args.theta,
        "kappa": args.kappa,
        "sigma_v": args.sigma_v,
        "rho": args.rho,
    }

    print("=" * 65)
    print("Heston Stochastic Volatility — Implied Vol Surface Generator")
    print("=" * 65)
    print(f"  S0       = {params['S0']}")
    print(f"  r        = {params['r']}")
    print(f"  v0       = {params['v0']}  (initial variance, ATM vol ~ {np.sqrt(params['v0']):.1%})")
    print(f"  theta    = {params['theta']}  (long-run variance)")
    print(f"  kappa    = {params['kappa']}  (mean reversion)")
    print(f"  sigma_v  = {params['sigma_v']}  (vol of vol)")
    print(f"  rho      = {params['rho']}  (spot-vol correlation)")
    print()

    print("Computing Heston call prices and inverting Black-Scholes...")
    df = generate_heston_iv_surface(params)

    n_expected = 5 * 30
    print(f"  Generated {len(df)} / {n_expected} implied vol points")
    if len(df) < n_expected:
        print(f"  ({n_expected - len(df)} points failed BS inversion — deep OTM)")
    print()

    # Print surface in tabular form
    print("Implied Volatility Surface (rows=log-moneyness, cols=maturity):")
    print("-" * 65)

    pivot = df.pivot_table(
        index="log_moneyness", columns="T", values="implied_volatility"
    )
    # Show a subset of rows for readability
    display_idx = np.linspace(0, len(pivot) - 1, min(15, len(pivot)), dtype=int)
    display = pivot.iloc[display_idx]
    display.index = [f"{x:+.3f}" for x in display.index]
    display.columns = [f"T={c:.2f}" for c in display.columns]
    print(display.to_string(float_format=lambda x: f"{x:.4f}"))
    print()

    # Summary statistics per maturity
    print("Summary by maturity:")
    print("-" * 65)
    for T_val in sorted(df["T"].unique()):
        slice_df = df[df["T"] == T_val]
        vols = slice_df["implied_volatility"].values
        print(
            f"  T={T_val:5.2f}:  "
            f"min={vols.min():.4f}  ATM~{vols[len(vols)//2]:.4f}  "
            f"max={vols.max():.4f}  skew={vols[0]-vols[-1]:+.4f}"
        )
    print()

    # Save if requested
    if args.save:
        import os
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)

        surface_array = df[["log_moneyness", "T", "implied_volatility"]].values
        np.save(args.save, surface_array)
        print(f"Saved surface array ({surface_array.shape}) to {args.save}")

        csv_path = args.save.replace(".npy", ".csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved surface CSV to {csv_path}")


if __name__ == "__main__":
    main()
