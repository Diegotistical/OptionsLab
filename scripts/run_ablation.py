"""
Ablation study: test the effect of removing individual training components.
Produces Table 5 for the paper.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_arb_free_pct(model, df):
    """Compute % of expiry slices that are arbitrage-free (no calendar violations)."""
    import torch

    if model.model is None:
        return 0.0

    maturities = sorted(df['T'].unique())
    n_slices = len(maturities)
    arb_free_count = 0

    model.model.eval()
    for T_val in maturities:
        slice_df = df[df['T'] == T_val]
        if len(slice_df) < 3:
            arb_free_count += 1
            continue

        X = slice_df[model.feature_columns].values.astype(np.float32)
        X_scaled = X.copy()
        X_scaled[:, 1] = np.sqrt(np.abs(X_scaled[:, 1]))

        X_t = torch.tensor(X_scaled, device=model.device, requires_grad=True)
        w = model.model(X_t)

        # Check calendar: dw/dT >= 0
        grad_w = torch.autograd.grad(
            outputs=w.sum(), inputs=X_t, create_graph=False
        )[0]
        dw_dsqrtT = grad_w[:, 1]
        sqrt_T = X_t[:, 1]
        dw_dT = dw_dsqrtT / (2 * sqrt_T + 1e-8)

        if (dw_dT < -1e-6).any().item():
            continue  # Not arb-free

        arb_free_count += 1

    return (arb_free_count / n_slices) * 100 if n_slices > 0 else 0.0


def run_ablation(quick: bool = False):
    """Run ablation study with each training component removed."""
    from src.volatility_surface.models.pinn_model import PINNVolatilityModel
    from src.benchmarks.vol_surface_benchmark import generate_synthetic_surface

    epochs = 200 if quick else 500

    print("Generating synthetic surface...")
    df = generate_synthetic_surface(n_strikes=30, seed=42)
    print(f"  {len(df)} points, {df['T'].nunique()} maturities")

    variants = {
        "CINN (full)": dict(
            use_residual=True, activation="gelu",
            use_warmup=True, use_ema_norm=True, squared_hinge=True,
        ),
        "No warmup": dict(
            use_residual=True, activation="gelu",
            use_warmup=False, use_ema_norm=True, squared_hinge=True,
        ),
        "No EMA norm": dict(
            use_residual=True, activation="gelu",
            use_warmup=True, use_ema_norm=False, squared_hinge=True,
        ),
        "Linear hinge": dict(
            use_residual=True, activation="gelu",
            use_warmup=True, use_ema_norm=True, squared_hinge=False,
        ),
        "No residuals": dict(
            use_residual=False, activation="gelu",
            use_warmup=True, use_ema_norm=True, squared_hinge=True,
        ),
        "ReLU activation": dict(
            use_residual=True, activation="relu",
            use_warmup=True, use_ema_norm=True, squared_hinge=True,
        ),
    }

    results = []

    for name, flags in variants.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")

        t0 = time.time()

        model = PINNVolatilityModel(
            epochs=epochs,
            hidden_layers=[64, 32, 16],
            lambda_calendar=1.0,
            lambda_butterfly=0.5,
            lambda_wing=0.1,
            **flags,
        )

        try:
            model.train(df)
            train_time = time.time() - t0

            # Predict on training data to compute RMSE
            preds = model.predict_volatility(df)
            true_iv = df['implied_volatility'].values
            rmse = np.sqrt(np.mean((preds - true_iv)**2))

            # Arb-free check
            arb_free = compute_arb_free_pct(model, df)

            # EPP
            try:
                from src.volatility_surface.utils.epp import compute_epp
                epp_result = compute_epp(
                    model, S=100.0, r=0.05,
                    strike_range=(80, 120), n_strikes=20,
                    maturities=[0.08, 0.25, 0.5, 1.0, 2.0],
                )
                epp = epp_result.get("epp_total", 0.0)
            except Exception as e:
                print(f"  EPP failed: {e}")
                epp = float("nan")

            print(f"  RMSE={rmse:.4f}, Arb-Free={arb_free:.1f}%, EPP={epp:.4f}, Time={train_time:.1f}s")

            results.append({
                "Variant": name,
                "RMSE": round(rmse, 4),
                "EPP": round(epp, 4) if not np.isnan(epp) else "N/A",
                "Arb-Free (%)": round(arb_free, 1),
                "Train Time (s)": round(train_time, 1),
            })

        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results.append({
                "Variant": name,
                "RMSE": "FAILED",
                "EPP": "FAILED",
                "Arb-Free (%)": 0.0,
                "Train Time (s)": round(time.time() - t0, 1),
            })

    df_results = pd.DataFrame(results).set_index("Variant")
    print("\n" + "=" * 60)
    print("ABLATION RESULTS")
    print("=" * 60)
    print(df_results.to_string())

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results"
    )
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "table5_ablation.csv")
    df_results.to_csv(csv_path)
    print(f"\nSaved to {csv_path}")

    return df_results


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    run_ablation(quick=quick)
