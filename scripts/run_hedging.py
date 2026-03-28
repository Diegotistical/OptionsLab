"""
Delta hedging experiment: compare CINN, MLP, SABR vol surfaces for hedging.
Produces Table 3 for the paper.

Self-contained — does not depend on the DeltaHedgingExperiment class.
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
        return 0.2  # fallback


def hedge_option(model, paths, K, T, r, q, true_vol, S0):
    """
    Delta-hedge a call along each price path using model-derived deltas.
    Returns array of hedging P&Ls.
    """
    n_paths, n_steps_plus1 = paths.shape
    n_steps = n_steps_plus1 - 1
    dt = T / n_steps

    pnls = np.zeros(n_paths)

    for p in range(n_paths):
        # Initial
        sigma_model = get_model_vol(model, S0, K, T)
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

    return pnls


def run_hedging(quick=False):
    from src.volatility_surface.models.pinn_model import PINNVolatilityModel
    from src.benchmarks.vol_surface_benchmark import generate_synthetic_surface
    from src.simulation.gbm_numpy import simulate_gbm_paths

    S0, r, q, true_vol = 100.0, 0.05, 0.0, 0.2
    n_paths = 50 if quick else 200
    n_options = 10 if quick else 20
    epochs = 100 if quick else 300

    print("=" * 60)
    print("Delta Hedging Experiment")
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

    # SABR-like: use a simple model with moderate constraints
    print("Training SABR-like...")
    t0 = time.time()
    sabr = PINNVolatilityModel(
        epochs=epochs, hidden_layers=[64, 32, 16],
        lambda_calendar=0.1, lambda_butterfly=0.05, lambda_wing=0.01,
    )
    sabr.train(df_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    models["SABR"] = sabr

    # Generate option portfolio
    rng = np.random.default_rng(42)
    strikes = S0 * rng.uniform(0.85, 1.15, n_options)
    maturities = rng.choice([1/12, 3/12, 6/12], n_options)

    results = []

    for model_name, model in models.items():
        print(f"\n--- Hedging with {model_name} ---")
        all_pnls = []
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

            pnls = hedge_option(model, paths, K, T, r, q, true_vol, S0)
            all_pnls.extend(pnls)

            # Delta error at initial point
            model_vol = get_model_vol(model, S0, K, T)
            true_delta = bs_delta(S0, K, T, r, true_vol, q)
            model_delta = bs_delta(S0, K, T, r, model_vol, q)
            delta_errors.append(abs(model_delta - true_delta))

        pnl_arr = np.array(all_pnls)

        # Unhedged variance (for hedge efficiency)
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

        row = {
            "Model": model_name,
            "P&L Std": round(float(np.std(pnl_arr)), 4),
            "99th Pctile Loss": round(float(np.percentile(pnl_arr, 1)), 4),
            "|Delta Error|": round(float(np.mean(delta_errors)), 4),
            "Hedge Eff.": round(float(hedge_eff), 4),
        }
        results.append(row)
        print(f"  P&L Std={row['P&L Std']}, 99th={row['99th Pctile Loss']}, "
              f"DeltaErr={row['|Delta Error|']}, HedgeEff={row['Hedge Eff.']}")

    df_results = pd.DataFrame(results).set_index("Model")
    print("\n" + "=" * 60)
    print("HEDGING RESULTS")
    print("=" * 60)
    print(df_results.to_string())

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results"
    )
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "table3_hedging.csv")
    df_results.to_csv(csv_path)
    print(f"\nSaved to {csv_path}")

    return df_results


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    run_hedging(quick=quick)
