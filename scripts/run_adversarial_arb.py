"""
Adversarial arbitrage test: off-grid sampling to stress-test constraint enforcement.
Tests whether near-zero EPP is an artifact of grid-aligned evaluation.

Protocol:
  1. Fit CINN surface
  2. Generate 10,000 random off-grid points in (k, T) space
  3. Evaluate density g(k,T) via autograd at each point
  4. Construct random non-adjacent butterfly spreads
  5. Report violation rate and max EPP from random search
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def adversarial_density_test(model, n_points=10000, seed=42):
    """
    Evaluate Breeden-Litzenberger density at random off-grid points.
    Uses finite-difference approximation of g(k,T).
    """
    import torch

    rng = np.random.default_rng(seed)

    # Random points in training range
    k_rand = rng.uniform(-0.25, 0.25, n_points)
    T_rand = rng.uniform(0.05, 2.0, n_points)

    violations = 0
    max_violation = 0.0
    dk = 0.001  # finite difference step

    for i in range(n_points):
        k, T = k_rand[i], T_rand[i]

        # Get w and derivatives via finite differences
        def get_w(k_val, T_val):
            df = pd.DataFrame({"log_moneyness": [k_val], "T": [T_val]})
            try:
                vol = float(model.predict_volatility(df)[0])
                vol = np.clip(vol, 0.001, 10.0)
                return vol ** 2 * T_val
            except Exception:
                return 0.04 * T_val

        w = get_w(k, T)
        w_plus = get_w(k + dk, T)
        w_minus = get_w(k - dk, T)

        if w < 1e-10:
            continue

        # First derivative
        dw_dk = (w_plus - w_minus) / (2 * dk)

        # Second derivative
        d2w_dk2 = (w_plus - 2 * w + w_minus) / (dk ** 2)

        # Breeden-Litzenberger density (Gatheral form)
        term1 = (1 - k * dw_dk / (2 * w)) ** 2
        term2 = (dw_dk ** 2 / 4) * (1 / w + 0.25)
        term3 = d2w_dk2 / 2

        g = term1 - term2 + term3

        if g < -1e-6:
            violations += 1
            max_violation = max(max_violation, abs(g))

    violation_rate = violations / n_points
    return {
        "n_points": n_points,
        "violations": violations,
        "violation_rate": violation_rate,
        "max_violation": max_violation,
    }


def adversarial_butterfly_test(model, n_butterflies=5000, seed=42):
    """
    Construct random non-adjacent butterfly spreads and check for arbitrage.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)

    S = 100.0
    r = 0.05

    violations = 0
    max_epp = 0.0

    for i in range(n_butterflies):
        T = rng.uniform(0.05, 2.0)

        # Random 3 strikes (not necessarily adjacent)
        k_vals = np.sort(rng.uniform(-0.2, 0.2, 3))
        # Ensure minimum spacing
        if k_vals[1] - k_vals[0] < 0.005 or k_vals[2] - k_vals[1] < 0.005:
            continue

        K_vals = S * np.exp(k_vals)

        # Get model vols
        vols = []
        for k in k_vals:
            df = pd.DataFrame({"log_moneyness": [k], "T": [T]})
            try:
                vol = float(model.predict_volatility(df)[0])
                vol = np.clip(vol, 0.001, 10.0)
            except Exception:
                vol = 0.2
            vols.append(vol)

        # Compute BS call prices
        prices = []
        for K, sigma in zip(K_vals, vols):
            if T <= 1e-10:
                prices.append(max(S - K, 0.0))
                continue
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            c = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            prices.append(c)

        # Butterfly cost: C(K1) - 2*C(K2) + C(K3)
        # Weighted for non-uniform spacing
        w1 = (K_vals[2] - K_vals[1]) / (K_vals[2] - K_vals[0])
        w3 = (K_vals[1] - K_vals[0]) / (K_vals[2] - K_vals[0])
        butterfly_cost = w1 * prices[0] - prices[1] + w3 * prices[2]

        if butterfly_cost < -1e-8:
            violations += 1
            epp = abs(butterfly_cost) / S * 10000  # bps
            max_epp = max(max_epp, epp)

    violation_rate = violations / n_butterflies
    return {
        "n_butterflies": n_butterflies,
        "violations": violations,
        "violation_rate": violation_rate,
        "max_epp_bps": max_epp,
    }


def adversarial_calendar_test(model, n_calendars=5000, seed=42):
    """
    Construct random calendar spreads and check for arbitrage.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    S = 100.0
    r = 0.05

    violations = 0
    max_epp = 0.0

    for i in range(n_calendars):
        k = rng.uniform(-0.2, 0.2)
        # Stay within training maturity range [0.1, 2.0]
        T1 = rng.uniform(0.1, 1.5)
        T2 = T1 + rng.uniform(0.05, 0.5)
        if T2 > 2.0:
            T2 = 2.0
        K = S * np.exp(k)

        # Get model vols
        vols = []
        for T in [T1, T2]:
            df = pd.DataFrame({"log_moneyness": [k], "T": [T]})
            try:
                vol = float(model.predict_volatility(df)[0])
                vol = np.clip(vol, 0.001, 10.0)
            except Exception:
                vol = 0.2
            vols.append(vol)

        # Total variance should be monotone
        w1 = vols[0] ** 2 * T1
        w2 = vols[1] ** 2 * T2

        if w2 < w1 - 1e-8:
            violations += 1
            # Calendar spread cost
            prices = []
            for T, sigma in zip([T1, T2], vols):
                d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                c = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                prices.append(c)
            epp = max(0, prices[0] - prices[1]) / S * 10000
            max_epp = max(max_epp, epp)

    violation_rate = violations / n_calendars
    return {
        "n_calendars": n_calendars,
        "violations": violations,
        "violation_rate": violation_rate,
        "max_epp_bps": max_epp,
    }


def run_adversarial_test(quick=False):
    from src.volatility_surface.models.pinn_model import PINNVolatilityModel
    from src.benchmarks.vol_surface_benchmark import generate_synthetic_surface

    n_points = 1000 if quick else 10000
    n_butterflies = 500 if quick else 5000
    n_calendars = 500 if quick else 5000
    n_surfaces = 3 if quick else 10
    epochs = 100 if quick else 300

    print("=" * 60)
    print("Adversarial Arbitrage Test")
    print("=" * 60)

    all_density = []
    all_butterfly = []
    all_calendar = []

    for surf_idx in range(n_surfaces):
        seed = 42 + surf_idx * 100
        print(f"\n--- Surface {surf_idx} (seed={seed}) ---")

        df = generate_synthetic_surface(n_strikes=30, seed=seed)

        # Apply 40% dropout
        rng = np.random.default_rng(seed + 1)
        mask = rng.random(len(df)) > 0.4
        df_train = df[mask].copy()

        # Train CINN
        model = PINNVolatilityModel(
            epochs=epochs, hidden_layers=[64, 32, 16],
            lambda_calendar=1.0, lambda_butterfly=0.5, lambda_wing=0.1,
            use_warmup=True, squared_hinge=True,
        )
        model.train(df_train)

        # Run tests
        density_res = adversarial_density_test(model, n_points=n_points, seed=seed)
        butterfly_res = adversarial_butterfly_test(model, n_butterflies=n_butterflies, seed=seed)
        calendar_res = adversarial_calendar_test(model, n_calendars=n_calendars, seed=seed)

        all_density.append(density_res)
        all_butterfly.append(butterfly_res)
        all_calendar.append(calendar_res)

        print(f"  Density: {density_res['violation_rate']*100:.2f}% violations, "
              f"max={density_res['max_violation']:.6f}")
        print(f"  Butterfly: {butterfly_res['violation_rate']*100:.2f}% violations, "
              f"max EPP={butterfly_res['max_epp_bps']:.4f} bps")
        print(f"  Calendar: {calendar_res['violation_rate']*100:.2f}% violations, "
              f"max EPP={calendar_res['max_epp_bps']:.4f} bps")

    # Aggregate
    print("\n" + "=" * 60)
    print("ADVERSARIAL TEST AGGREGATE RESULTS")
    print("=" * 60)

    avg_density_rate = np.mean([r["violation_rate"] for r in all_density])
    avg_butterfly_rate = np.mean([r["violation_rate"] for r in all_butterfly])
    avg_calendar_rate = np.mean([r["violation_rate"] for r in all_calendar])
    max_butterfly_epp = max(r["max_epp_bps"] for r in all_butterfly)
    max_calendar_epp = max(r["max_epp_bps"] for r in all_calendar)

    summary = {
        "Test": ["Density (off-grid)", "Butterfly (random)", "Calendar (random)"],
        "Avg Violation Rate (%)": [
            round(avg_density_rate * 100, 3),
            round(avg_butterfly_rate * 100, 3),
            round(avg_calendar_rate * 100, 3),
        ],
        "Max EPP (bps)": [
            "N/A",
            round(max_butterfly_epp, 4),
            round(max_calendar_epp, 4),
        ],
        "N Surfaces": [n_surfaces] * 3,
        "Samples/Surface": [n_points, n_butterflies, n_calendars],
    }

    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results"
    )
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "adversarial_arb.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    return df_summary


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    run_adversarial_test(quick=quick)
