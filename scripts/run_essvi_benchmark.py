"""
eSSVI benchmark: evaluate extended SSVI under sparse strike conditions.
Produces eSSVI rows for Tables 1 and 6 in the paper.

Usage:
    python scripts/run_essvi_benchmark.py
    python scripts/run_essvi_benchmark.py --quick
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.vol_surface_benchmark import generate_synthetic_surface
from src.volatility_surface.models.svi import (
    ESSVIModel, calibrate_essvi, calibrate_svi, SVIModel
)
from src.volatility_surface.utils.epp import compute_butterfly_epp, compute_calendar_epp


def evaluate_essvi_on_surface(df, dropout_rate=0.0, n_trials=10, seed=42):
    """Evaluate eSSVI on a single surface with optional dropout."""
    # Restrict to realistic moneyness range (matches paper: 0.8 <= K/S <= 1.2)
    df = df[(df["log_moneyness"] >= -0.22) & (df["log_moneyness"] <= 0.2)].copy()
    rng = np.random.default_rng(seed)
    T_vals = sorted(df["T"].unique())

    rmse_list = []
    epp_list = []

    for trial in range(n_trials):
        # Apply dropout
        if dropout_rate > 0:
            mask = rng.random(len(df)) > dropout_rate
            if mask.sum() < len(T_vals) * 3:  # minimum 3 points per slice
                mask[:len(T_vals) * 3] = True
            df_train = df[mask].copy()
            df_test = df[~mask].copy()
        else:
            df_train = df.copy()
            df_test = df.copy()

        # Build smile data per slice
        T_slices = []
        smile_data = []
        for T in T_vals:
            slice_df = df_train[df_train["T"] == T]
            if len(slice_df) < 2:
                continue
            T_slices.append(T)
            smile_data.append((
                slice_df["log_moneyness"].values,
                slice_df["implied_volatility"].values,
            ))

        if len(T_slices) < 2:
            continue

        T_slices = np.array(T_slices)

        # Calibrate eSSVI
        try:
            model = calibrate_essvi(T_slices, smile_data, n_restarts=10)
        except (ValueError, Exception) as e:
            print(f"  Trial {trial}: calibration failed: {e}")
            continue

        # Compute RMSE on test set
        errors = []
        for _, row in df_test.iterrows():
            k = row["log_moneyness"]
            T = row["T"]
            pred_vol = model.implied_vol(k, T)
            errors.append((pred_vol - row["implied_volatility"]) ** 2)

        if errors:
            rmse = np.sqrt(np.mean(errors)) * 10000  # bps
            rmse_list.append(rmse)

        # Compute EPP on full evaluation grid
        epp_total = 0.0
        for T in T_vals:
            n_k = 40
            k_grid = np.linspace(-0.25, 0.25, n_k)
            vols = model.smile(k_grid, T)
            strikes = 100.0 * np.exp(k_grid)

            but_epp = compute_butterfly_epp(strikes, vols, T, S=100.0, r=0.05)
            epp_total = max(epp_total, but_epp)

        # Calendar EPP
        for k_val in np.linspace(-0.2, 0.2, 20):
            mats = np.array(T_vals)
            vols_cal = np.array([model.implied_vol(k_val, T) for T in mats])
            cal_epp = compute_calendar_epp(
                100.0 * np.exp(k_val), mats, vols_cal, S=100.0, r=0.05
            )
            epp_total = max(epp_total, cal_epp)

        epp_list.append(epp_total * 10000)  # bps

    return {
        "RMSE_mean": np.mean(rmse_list) if rmse_list else float("nan"),
        "RMSE_std": np.std(rmse_list) if rmse_list else float("nan"),
        "EPP_mean": np.mean(epp_list) if epp_list else float("nan"),
        "EPP_std": np.std(epp_list) if epp_list else float("nan"),
        "n_successful": len(rmse_list),
    }


def run_essvi_benchmark(quick=False):
    n_trials = 3 if quick else 10
    n_surfaces = 5 if quick else 20

    print("=" * 60)
    print("eSSVI Benchmark")
    print("=" * 60)

    results = []

    for dropout in [0.0, 0.2, 0.4, 0.6]:
        print(f"\n--- Dropout: {dropout*100:.0f}% ---")

        all_rmse = []
        all_epp = []

        for surf_idx in range(n_surfaces):
            seed = 42 + surf_idx * 100
            df = generate_synthetic_surface(n_strikes=30, seed=seed)

            res = evaluate_essvi_on_surface(
                df, dropout_rate=dropout, n_trials=n_trials, seed=seed
            )
            if not np.isnan(res["RMSE_mean"]):
                all_rmse.append(res["RMSE_mean"])
                all_epp.append(res["EPP_mean"])
                print(f"  Surface {surf_idx}: RMSE={res['RMSE_mean']:.1f} bps, "
                      f"EPP={res['EPP_mean']:.4f} bps")

        if all_rmse:
            row = {
                "Dropout": f"{dropout*100:.0f}%",
                "RMSE (bps)": round(np.mean(all_rmse), 1),
                "EPP (bps)": round(np.mean(all_epp), 4),
                "EPP_std": round(np.std(all_epp), 4),
                "N_surfaces": len(all_rmse),
            }
            results.append(row)
            print(f"  Avg: RMSE={row['RMSE (bps)']}, EPP={row['EPP (bps)']}")

    df_results = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("eSSVI BENCHMARK RESULTS")
    print("=" * 60)
    print(df_results.to_string(index=False))

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results"
    )
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "essvi_benchmark.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    return df_results


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    run_essvi_benchmark(quick=quick)
