#!/usr/bin/env python3
"""
Benchmark CINN vs MLP vs SABR on Heston stochastic volatility surface.

Tests whether CINN results generalize beyond the GBM DGP.
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")


def main(quick: bool = False):
    from scripts.generate_heston_surface import generate_heston_iv_surface
    from src.volatility_surface.models.pinn_model import PINNVolatilityModel

    epochs = 200 if quick else 500
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    # Generate Heston surface
    print("=" * 60)
    print("Heston Stochastic Volatility Benchmark")
    print("=" * 60)

    df = generate_heston_iv_surface()
    n_points = len(df)
    print(f"  Surface: {n_points} valid IV points across {df['T'].nunique()} maturities")

    # Prepare data
    surface_data = df.copy()

    # Hold out 20% for evaluation
    np.random.seed(42)
    n = len(surface_data)
    idx = np.random.permutation(n)
    n_train = int(0.8 * n)
    train_df = surface_data.iloc[idx[:n_train]].reset_index(drop=True)
    test_df = surface_data.iloc[idx[n_train:]].reset_index(drop=True)

    results = []

    def compute_arb_free(model_obj, data_df):
        """Compute arb-free % using per-slice calendar check."""
        import torch
        if model_obj.model is None:
            return 0.0
        maturities = sorted(data_df['T'].unique())
        n_slices = len(maturities)
        arb_free_count = 0
        model_obj.model.eval()
        for T_val in maturities:
            slice_df = data_df[data_df['T'] == T_val]
            if len(slice_df) < 3:
                arb_free_count += 1
                continue
            X = slice_df[model_obj.feature_columns].values.astype(np.float32)
            X_scaled = X.copy()
            X_scaled[:, 1] = np.sqrt(np.abs(X_scaled[:, 1]))
            X_t = torch.tensor(X_scaled, device=model_obj.device, requires_grad=True)
            w = model_obj.model(X_t)
            grad_w = torch.autograd.grad(outputs=w.sum(), inputs=X_t, create_graph=False)[0]
            dw_dsqrtT = grad_w[:, 1]
            sqrt_T = X_t[:, 1]
            dw_dT = dw_dsqrtT / (2 * sqrt_T + 1e-8)
            if (dw_dT < -1e-6).any().item():
                continue
            arb_free_count += 1
        return (arb_free_count / n_slices) * 100 if n_slices > 0 else 0.0

    def train_and_eval(name, model_obj, train_data, test_data, full_data):
        t0 = time.time()
        model_obj.train(train_data)
        elapsed = time.time() - t0
        preds = model_obj.predict_volatility(test_data)
        true_v = test_data["implied_volatility"].values
        rmse = np.sqrt(np.mean((preds - true_v) ** 2))
        mae = np.mean(np.abs(preds - true_v))
        arb = compute_arb_free(model_obj, full_data)
        print(f"  RMSE={rmse:.4f}, MAE={mae:.4f}, Arb-Free={arb:.0f}%, Time={elapsed:.1f}s")
        return {
            "Model": name,
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "Calib. Time": f"{elapsed*1000:.0f} ms",
            "Arb-Free (%)": f"{arb:.0f}%",
        }

    # --- CINN ---
    print(f"\nTraining CINN ({epochs} epochs)...")
    cinn = PINNVolatilityModel(
        hidden_layers=[64, 32, 16],
        lambda_calendar=1.0, lambda_butterfly=0.5, lambda_wing=0.1,
        epochs=epochs, batch_size=64, dropout=0.1,
    )
    results.append(train_and_eval("CINN", cinn, train_df, test_df, surface_data))

    # --- MLP (unconstrained) ---
    print(f"\nTraining MLP ({epochs} epochs)...")
    mlp = PINNVolatilityModel(
        hidden_layers=[64, 32, 16],
        lambda_calendar=0.0, lambda_butterfly=0.0, lambda_wing=0.0,
        epochs=epochs, batch_size=64, dropout=0.1,
    )
    results.append(train_and_eval("MLP", mlp, train_df, test_df, surface_data))

    # --- ReLU MLP (no constraints, no dropout, ReLU activation) ---
    print(f"\nTraining ReLU MLP ({epochs} epochs)...")
    relu_mlp = PINNVolatilityModel(
        hidden_layers=[64, 32, 16],
        lambda_calendar=0.0, lambda_butterfly=0.0, lambda_wing=0.0,
        epochs=epochs, batch_size=64, dropout=0.0, activation="relu",
    )
    results.append(train_and_eval("ReLU MLP", relu_mlp, train_df, test_df, surface_data))

    # Summary
    print("\n" + "=" * 60)
    print("HESTON BENCHMARK RESULTS")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    results_df.to_csv(output_dir / "table_heston.csv", index=False)
    print(f"\nSaved to {output_dir / 'table_heston.csv'}")

    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick run (200 epochs)")
    args = parser.parse_args()
    main(quick=args.quick)
