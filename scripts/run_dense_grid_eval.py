"""
Dense-grid constraint evaluation for CINN volatility surface model.

The paper notes a limitation: arbitrage constraints are only evaluated at
training grid points, not between them. This script quantifies inter-grid-point
violations by evaluating the trained model on a much finer grid (5000 points)
than the 150-point training set.

Outputs:
    results/dense_grid_eval.csv          -- per-point violation data
    results/dense_grid_eval_summary.txt  -- aggregate statistics
"""

import os
import sys
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch

from src.volatility_surface.models.pinn_model import PINNVolatilityModel
from src.benchmarks.vol_surface_benchmark import generate_synthetic_surface


# ---------------------------------------------------------------------------
# Dense-grid constraint evaluation (autograd, mirrors training losses)
# ---------------------------------------------------------------------------

def evaluate_constraints_dense(model, k_grid, T_grid):
    """
    Evaluate calendar and butterfly constraints on a dense (k, T) grid
    using the same autograd approach as CalendarLoss / ButterflyLoss.

    Parameters
    ----------
    model : PINNVolatilityModel
        Trained model (must have .model and .device attributes).
    k_grid : np.ndarray, shape (N,)
        Log-moneyness values.
    T_grid : np.ndarray, shape (N,)
        Maturity values.

    Returns
    -------
    pd.DataFrame with columns:
        k, T, w, implied_vol,
        dw_dT, calendar_violation, calendar_violation_mag,
        g_k, butterfly_violation, butterfly_violation_mag
    """
    net = model.model
    device = model.device
    eps = 1e-6

    # Prepare input: [log_moneyness, sqrt(T)]
    sqrt_T = np.sqrt(T_grid).astype(np.float32)
    k = k_grid.astype(np.float32)
    X_np = np.stack([k, sqrt_T], axis=1)

    X = torch.tensor(X_np, device=device, requires_grad=True)

    net.eval()

    # Forward pass -- total variance w
    w = net(X)  # (N, 1)

    # ---- Calendar: dw/dT ----
    grad_w = torch.autograd.grad(
        outputs=w,
        inputs=X,
        grad_outputs=torch.ones_like(w),
        create_graph=True,
        retain_graph=True,
    )[0]

    dw_dsqrtT = grad_w[:, 1:2]
    sqrt_T_t = X[:, 1:2]
    dw_dT = dw_dsqrtT / (2 * sqrt_T_t + eps)  # chain rule

    # ---- Butterfly: Gatheral density g(k) ----
    dw_dk = grad_w[:, 0:1]

    # Second derivative d2w/dk2
    grad2_w = torch.autograd.grad(
        outputs=dw_dk,
        inputs=X,
        grad_outputs=torch.ones_like(dw_dk),
        create_graph=False,
        retain_graph=False,
    )[0]
    d2w_dk2 = grad2_w[:, 0:1]

    k_t = X[:, 0:1]
    term1 = (1 - k_t * dw_dk / (2 * w + eps)) ** 2
    term2 = (dw_dk ** 2) / 4 * (1 / (w + eps) + 0.25)
    term3 = d2w_dk2 / 2
    g_k = term1 - term2 + term3

    # ---- Convert to numpy ----
    w_np = w.detach().cpu().numpy().flatten()
    dw_dT_np = dw_dT.detach().cpu().numpy().flatten()
    g_k_np = g_k.detach().cpu().numpy().flatten()
    T_tensor = sqrt_T_t.detach().cpu().numpy().flatten() ** 2 + 1e-10
    iv_np = np.sqrt(np.maximum(w_np / T_tensor, 0.0))

    tol = 1e-6
    cal_violated = dw_dT_np < -tol
    cal_mag = np.where(cal_violated, -dw_dT_np, 0.0)

    but_violated = g_k_np < -tol
    but_mag = np.where(but_violated, -g_k_np, 0.0)

    return pd.DataFrame({
        "k": k_grid,
        "T": T_grid,
        "w": w_np,
        "implied_vol": iv_np,
        "dw_dT": dw_dT_np,
        "calendar_violation": cal_violated.astype(int),
        "calendar_violation_mag": cal_mag,
        "g_k": g_k_np,
        "butterfly_violation": but_violated.astype(int),
        "butterfly_violation_mag": but_mag,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # 1. Train CINN on standard 150-point surface
    # ------------------------------------------------------------------
    print("=" * 70)
    print("DENSE-GRID CONSTRAINT EVALUATION")
    print("=" * 70)

    print("\n[1/4] Generating synthetic surface (5 maturities x 30 strikes)...")
    df_train = generate_synthetic_surface(n_strikes=30, seed=42)
    print(f"  Training points: {len(df_train)}")
    print(f"  Maturities: {sorted(df_train['T'].unique())}")
    print(f"  Log-moneyness range: [{df_train['log_moneyness'].min():.3f}, "
          f"{df_train['log_moneyness'].max():.3f}]")

    print("\n[2/4] Training CINN model (500 epochs)...")
    t0 = time.time()
    model = PINNVolatilityModel(
        epochs=500,
        hidden_layers=[64, 32, 16],
        lambda_calendar=1.0,
        lambda_butterfly=0.5,
        lambda_wing=0.1,
        use_residual=True,
        activation="gelu",
        use_warmup=True,
        use_ema_norm=True,
        squared_hinge=True,
    )
    model.train(df_train)
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s")

    # Quick sanity: RMSE on training data
    preds = model.predict_volatility(df_train)
    rmse = np.sqrt(np.mean((preds - df_train["implied_volatility"].values) ** 2))
    print(f"  Training RMSE: {rmse:.6f}")

    # ------------------------------------------------------------------
    # 2. Build dense evaluation grid
    # ------------------------------------------------------------------
    print("\n[3/4] Evaluating constraints on dense grid...")
    n_T = 50
    n_k = 100
    T_dense = np.linspace(0.08, 2.0, n_T)
    k_dense = np.linspace(-0.3, 0.3, n_k)

    # Cartesian product
    T_mesh, k_mesh = np.meshgrid(T_dense, k_dense, indexing="ij")
    T_flat = T_mesh.flatten()
    k_flat = k_mesh.flatten()
    total_pts = len(T_flat)
    print(f"  Dense grid: {n_T} maturities x {n_k} strikes = {total_pts} points")

    t0 = time.time()
    df_eval = evaluate_constraints_dense(model, k_flat, T_flat)
    eval_time = time.time() - t0
    print(f"  Evaluation completed in {eval_time:.1f}s")

    # ------------------------------------------------------------------
    # 3. Also evaluate on training grid for comparison
    # ------------------------------------------------------------------
    print("\n  Evaluating constraints on training grid for comparison...")
    df_train_eval = evaluate_constraints_dense(
        model,
        df_train["log_moneyness"].values,
        df_train["T"].values,
    )

    # ------------------------------------------------------------------
    # 4. Report results
    # ------------------------------------------------------------------
    print("\n[4/4] Results")
    print("=" * 70)

    def summarise(label, df, n_pts):
        n_cal = df["calendar_violation"].sum()
        n_but = df["butterfly_violation"].sum()
        n_any = ((df["calendar_violation"] == 1) | (df["butterfly_violation"] == 1)).sum()
        pct_free = 100.0 * (1 - n_any / n_pts)
        max_cal = df["calendar_violation_mag"].max()
        max_but = df["butterfly_violation_mag"].max()
        mean_cal = df.loc[df["calendar_violation"] == 1, "calendar_violation_mag"].mean() \
            if n_cal > 0 else 0.0
        mean_but = df.loc[df["butterfly_violation"] == 1, "butterfly_violation_mag"].mean() \
            if n_but > 0 else 0.0

        print(f"\n  {label} ({n_pts} points)")
        print(f"  {'':->50}")
        print(f"    Arb-free points:         {n_pts - n_any}/{n_pts} ({pct_free:.2f}%)")
        print(f"    Calendar violations:     {n_cal} ({100*n_cal/n_pts:.2f}%)")
        print(f"      Max magnitude:         {max_cal:.6e}")
        print(f"      Mean magnitude:        {mean_cal:.6e}")
        print(f"    Butterfly violations:    {n_but} ({100*n_but/n_pts:.2f}%)")
        print(f"      Max magnitude:         {max_but:.6e}")
        print(f"      Mean magnitude:        {mean_but:.6e}")

        return {
            "grid": label,
            "n_points": n_pts,
            "arb_free_pct": round(pct_free, 4),
            "n_calendar_violations": int(n_cal),
            "max_calendar_mag": float(max_cal),
            "mean_calendar_mag": float(mean_cal),
            "n_butterfly_violations": int(n_but),
            "max_butterfly_mag": float(max_but),
            "mean_butterfly_mag": float(mean_but),
        }

    s_train = summarise("Training grid", df_train_eval, len(df_train_eval))
    s_dense = summarise("Dense grid", df_eval, total_pts)

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    output_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)

    # Per-point CSV (dense grid)
    csv_path = os.path.join(output_dir, "dense_grid_eval.csv")
    df_eval.to_csv(csv_path, index=False)
    print(f"\n  Saved per-point results to {csv_path}")

    # Summary text
    summary_path = os.path.join(output_dir, "dense_grid_eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Dense-Grid Constraint Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        for s in [s_train, s_dense]:
            f.write(f"Grid: {s['grid']} ({s['n_points']} points)\n")
            f.write(f"  Arb-free: {s['arb_free_pct']}%\n")
            f.write(f"  Calendar violations: {s['n_calendar_violations']}  "
                    f"(max={s['max_calendar_mag']:.6e}, mean={s['mean_calendar_mag']:.6e})\n")
            f.write(f"  Butterfly violations: {s['n_butterfly_violations']}  "
                    f"(max={s['max_butterfly_mag']:.6e}, mean={s['mean_butterfly_mag']:.6e})\n\n")
    print(f"  Saved summary to {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
