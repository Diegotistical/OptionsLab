"""
Delta hedging experiment: compare CINN, MLP, SABR vol surfaces for hedging.
Produces Table 3 for the paper.

Enhanced with:
  - Vol calibration error (implied - realized bias)
  - Conditional hedging by regime (low/high vol)
  - Greeks stability (curvature variance, Lipschitz proxy)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bs_delta(S, K, T, r, sigma, q=0.0):
    if T <= 1e-10:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.cdf(d1)


def bs_price(S, K, T, r, sigma, q=0.0):
    if T <= 1e-10:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def get_model_vol(model, S, K, T):
    """Query a vol surface model for implied vol at (K, T)."""
    log_m = np.log(K / S)
    df = pd.DataFrame({"log_moneyness": [log_m], "T": [T]})
    try:
        sigma = float(model.predict_volatility(df)[0])
        return np.clip(sigma, 0.01, 5.0)
    except Exception:
        return 0.2


def hedge_option(model, paths, K, T, r, q, true_vol, S0):
    """
    Delta-hedge a call along each price path using model-derived deltas.
    Returns array of hedging P&Ls and realized vol per path.
    """
    n_paths, n_steps_plus1 = paths.shape
    n_steps = n_steps_plus1 - 1
    dt = T / n_steps

    pnls = np.zeros(n_paths)
    realized_vols = np.zeros(n_paths)
    vol_biases = np.zeros(n_paths)

    for p in range(n_paths):
        # Compute realized vol for this path
        log_rets = np.diff(np.log(paths[p]))
        realized_vols[p] = np.std(log_rets) * np.sqrt(252) if len(log_rets) > 1 else true_vol

        # Initial
        sigma_model = get_model_vol(model, S0, K, T)
        vol_biases[p] = sigma_model - realized_vols[p]
        delta = bs_delta(S0, K, T, r, sigma_model, q)
        option_v0 = bs_price(S0, K, T, r, true_vol, q)
        cash = option_v0 - delta * S0

        for i in range(1, n_steps + 1):
            S_i = paths[p, i]
            T_rem = T - i * dt
            if T_rem <= 0:
                break
            sigma_model = get_model_vol(model, S_i, K, T_rem)
            delta_new = bs_delta(S_i, K, T_rem, r, sigma_model, q)
            cash -= (delta_new - delta) * S_i
            cash *= np.exp(r * dt)
            delta = delta_new

        S_T = paths[p, -1]
        payoff = max(S_T - K, 0.0)
        pnls[p] = delta * S_T + cash - payoff

    return pnls, realized_vols, vol_biases


def compute_greeks_stability(model, S0=100.0, T=0.25, n_strikes=50):
    """
    Compute curvature variance and Lipschitz proxy for dw/dk.
    Returns dict of stability metrics.
    """
    k_grid = np.linspace(-0.25, 0.25, n_strikes)
    dk = k_grid[1] - k_grid[0]

    # Get total variance across strike grid
    vols = np.array([get_model_vol(model, S0, S0 * np.exp(-k), T) for k in k_grid])
    w = vols ** 2 * T

    # First derivative (finite diff)
    dw_dk = np.gradient(w, dk)

    # Second derivative (curvature)
    d2w_dk2 = np.gradient(dw_dk, dk)

    # Curvature variance (lower = smoother)
    curvature_var = float(np.var(d2w_dk2))

    # Lipschitz proxy: max |d(dw/dk)| / dk
    lipschitz = float(np.max(np.abs(np.diff(dw_dk))) / dk)

    # Smoothness: std of third derivative
    d3w_dk3 = np.gradient(d2w_dk2, dk)
    smoothness = float(np.std(d3w_dk3))

    return {
        "curvature_var": curvature_var,
        "lipschitz": lipschitz,
        "smoothness": smoothness,
    }


def run_hedging(quick=False):
    from src.volatility_surface.models.pinn_model import PINNVolatilityModel
    from src.benchmarks.vol_surface_benchmark import generate_synthetic_surface
    from src.simulation.gbm_numpy import simulate_gbm_paths

    S0, r, q, true_vol = 100.0, 0.05, 0.0, 0.2
    n_paths = 50 if quick else 200
    n_options = 10 if quick else 20
    epochs = 100 if quick else 300

    print("=" * 60)
    print("Delta Hedging Experiment (Enhanced)")
    print("=" * 60)

    # Generate training surface
    df_train = generate_synthetic_surface(n_strikes=30, seed=42)
    print(f"Training surface: {len(df_train)} points")

    # Train models
    models = {}

    print("\nTraining CINN...")
    t0 = time.time()
    cinn = PINNVolatilityModel(
        epochs=epochs, hidden_layers=[64, 32, 16],
        lambda_calendar=1.0, lambda_butterfly=0.5, lambda_wing=0.1,
        use_warmup=True, squared_hinge=True,
    )
    cinn.train(df_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    models["CINN"] = cinn

    print("Training MLP (no constraints)...")
    t0 = time.time()
    mlp = PINNVolatilityModel(
        epochs=epochs, hidden_layers=[64, 32, 16],
        lambda_calendar=0.0, lambda_butterfly=0.0, lambda_wing=0.0,
    )
    mlp.train(df_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    models["MLP"] = mlp

    print("Training MLP-Soft (weak penalties)...")
    t0 = time.time()
    mlp_soft = PINNVolatilityModel(
        epochs=epochs, hidden_layers=[64, 32, 16],
        lambda_calendar=0.1, lambda_butterfly=0.05, lambda_wing=0.01,
    )
    mlp_soft.train(df_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    models["MLP-Soft"] = mlp_soft

    # Generate option portfolio
    rng = np.random.default_rng(42)
    strikes = S0 * rng.uniform(0.85, 1.15, n_options)
    maturities = rng.choice([1/12, 3/12, 6/12], n_options)

    results = []
    greeks_results = []

    for model_name, model in models.items():
        print(f"\n--- Hedging with {model_name} ---")
        all_pnls = []
        all_realized_vols = []
        all_vol_biases = []
        delta_errors = []

        for j in range(n_options):
            K = strikes[j]
            T = maturities[j]
            n_steps = max(int(T * 252), 5)

            paths = simulate_gbm_paths(
                S=S0, T=T, r=r, sigma=true_vol, q=q,
                n_paths=n_paths, n_steps=n_steps,
                seed=42 + j * 100,
            )

            pnls, rvols, vbiases = hedge_option(
                model, paths, K, T, r, q, true_vol, S0
            )
            all_pnls.extend(pnls)
            all_realized_vols.extend(rvols)
            all_vol_biases.extend(vbiases)

            # Delta error at initial point
            model_vol = get_model_vol(model, S0, K, T)
            true_delta = bs_delta(S0, K, T, r, true_vol, q)
            model_delta = bs_delta(S0, K, T, r, model_vol, q)
            delta_errors.append(abs(model_delta - true_delta))

        pnl_arr = np.array(all_pnls)
        rvol_arr = np.array(all_realized_vols)
        vbias_arr = np.array(all_vol_biases)

        # Unhedged variance
        unhedged_pnls = []
        for j in range(n_options):
            K = strikes[j]
            T = maturities[j]
            n_steps = max(int(T * 252), 5)
            paths = simulate_gbm_paths(
                S=S0, T=T, r=r, sigma=true_vol, q=q,
                n_paths=n_paths, n_steps=n_steps,
                seed=42 + j * 100,
            )
            v0 = bs_price(S0, K, T, r, true_vol, q)
            for p in range(n_paths):
                payoff = max(paths[p, -1] - K, 0.0)
                unhedged_pnls.append(payoff - v0)

        unhedged_var = np.var(unhedged_pnls)
        hedge_eff = 1 - np.var(pnl_arr) / max(unhedged_var, 1e-10)

        # Conditional hedging by regime
        low_vol_mask = rvol_arr < 0.15
        high_vol_mask = rvol_arr > 0.25
        pnl_std_low = float(np.std(pnl_arr[low_vol_mask])) if low_vol_mask.sum() > 5 else float("nan")
        pnl_std_high = float(np.std(pnl_arr[high_vol_mask])) if high_vol_mask.sum() > 5 else float("nan")

        # Greeks stability
        greeks = compute_greeks_stability(model, S0=S0)

        row = {
            "Model": model_name,
            "P&L Std": round(float(np.std(pnl_arr)), 4),
            "99th Pctile Loss": round(float(np.percentile(pnl_arr, 1)), 4),
            "|Delta Error|": round(float(np.mean(delta_errors)), 4),
            "Hedge Eff.": round(float(hedge_eff), 4),
            "Vol Bias": round(float(np.mean(vbias_arr)), 4),
            "Vol Bias Std": round(float(np.std(vbias_arr)), 4),
            "P&L Std (Low Vol)": round(pnl_std_low, 4) if not np.isnan(pnl_std_low) else "N/A",
            "P&L Std (High Vol)": round(pnl_std_high, 4) if not np.isnan(pnl_std_high) else "N/A",
        }
        results.append(row)

        greeks_row = {
            "Model": model_name,
            "Curvature Var": f"{greeks['curvature_var']:.6f}",
            "Lipschitz": f"{greeks['lipschitz']:.4f}",
            "Smoothness": f"{greeks['smoothness']:.6f}",
        }
        greeks_results.append(greeks_row)

        print(f"  P&L Std={row['P&L Std']}, HedgeEff={row['Hedge Eff.']}, "
              f"VolBias={row['Vol Bias']}")
        print(f"  Greeks: CurvVar={greeks['curvature_var']:.6f}, "
              f"Lipschitz={greeks['lipschitz']:.4f}")

    df_results = pd.DataFrame(results).set_index("Model")
    df_greeks = pd.DataFrame(greeks_results).set_index("Model")

    print("\n" + "=" * 60)
    print("HEDGING RESULTS")
    print("=" * 60)
    print(df_results.to_string())
    print("\nGREEKS STABILITY")
    print(df_greeks.to_string())

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results"
    )
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "table3_hedging.csv")
    df_results.to_csv(csv_path)
    print(f"\nSaved hedging to {csv_path}")

    csv_path2 = os.path.join(output_dir, "greeks_stability.csv")
    df_greeks.to_csv(csv_path2)
    print(f"Saved Greeks stability to {csv_path2}")

    return df_results, df_greeks


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    run_hedging(quick=quick)
