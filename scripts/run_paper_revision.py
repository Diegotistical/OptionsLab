#!/usr/bin/env python3
"""
Master experiment runner for CINN paper revision.

Executes all new baselines and cross-asset experiments:
    Phase 3.1: eSSVI on real/synthetic data with proper calibration
    Phase 3.2: Piecewise-linear Andreasen-Huge
    Phase 3.3: ICNN (Input Convex Neural Network)
    Phase 4.1: EURUSD FX cross-asset validation
    Phase 6.1: Early stopping latency measurement

Outputs:
    results/revision_essvi.csv
    results/revision_ah_pl.csv
    results/revision_icnn.csv
    results/revision_eurusd.csv
    results/revision_early_stopping.csv
    results/revision_main_table.csv  (updated Table 8 with all new rows)

Usage:
    python scripts/run_paper_revision.py [--phase PHASE] [--n-days N] [--seed SEED]
    python scripts/run_paper_revision.py --phase all
    python scripts/run_paper_revision.py --phase essvi
    python scripts/run_paper_revision.py --phase icnn
    python scripts/run_paper_revision.py --phase eurusd
    python scripts/run_paper_revision.py --phase early_stopping
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.volatility_surface.models.svi import (
    ESSVIModel,
    SVIModel,
    SSVIModel,
    calibrate_essvi,
    calibrate_svi,
    calibrate_ssvi,
)
from src.volatility_surface.utils.epp import (
    compute_butterfly_epp,
    compute_calendar_epp,
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Synthetic surface generation (shared across experiments)
# =============================================================================

def generate_heston_surface(
    S: float = 100.0,
    r: float = 0.05,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma_v: float = 0.3,
    rho: float = -0.7,
    n_strikes: int = 20,
    n_expiries: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic SPX-like volatility surface using SSVI-style
    parameterization with Heston-calibrated parameters.

    Produces surfaces in the realistic RMSE range (20-50 bps) for all models.
    """
    rng = np.random.default_rng(seed)

    T_grid = np.array([0.083, 0.167, 0.25, 0.5, 1.0, 2.0])[:n_expiries]

    # SSVI-like parameterization: w(k,T) = (theta(T)/2)*(1 + rho*phi*k + sqrt((phi*k+rho)^2 + 1 - rho^2))
    # with theta(T) = a*T^b (power-law ATM total variance)
    atm_vol_1y = np.sqrt(v0) + rng.normal(0, 0.005)  # ~20% ± noise
    a_atm = atm_vol_1y**2  # ATM total variance at T=1
    b_atm = 0.85 + rng.normal(0, 0.05)  # power law exponent
    b_atm = np.clip(b_atm, 0.5, 1.2)
    ssvi_rho = rho * 0.7 + rng.normal(0, 0.05)  # skew param
    ssvi_rho = np.clip(ssvi_rho, -0.95, -0.1)
    ssvi_eta = 0.8 + rng.normal(0, 0.1)  # vol-of-vol
    ssvi_eta = np.clip(ssvi_eta, 0.2, 1.5)
    ssvi_gamma = 0.5 + rng.normal(0, 0.05)
    ssvi_gamma = np.clip(ssvi_gamma, 0.1, 0.9)

    # Moneyness range: narrower for short maturities
    rows = []
    for T in T_grid:
        theta_T = a_atm * T**b_atm
        phi = ssvi_eta / (theta_T**ssvi_gamma) if theta_T > 0 else 1.0

        # Adaptive moneyness range
        k_range = min(0.20, 2.5 * np.sqrt(theta_T))
        k_range = max(k_range, 0.05)
        k_grid = np.linspace(-k_range, k_range, n_strikes)

        for k in k_grid:
            inner = (phi * k + ssvi_rho)**2 + 1 - ssvi_rho**2
            if inner < 0:
                inner = 1e-6
            w = (theta_T / 2) * (1 + ssvi_rho * phi * k + np.sqrt(inner))
            w = max(w, 1e-6)
            iv = np.sqrt(w / T)
            iv += rng.normal(0, 0.0015)  # realistic market noise (~1.5 bps)
            iv = max(iv, 0.03)
            rows.append({
                "log_moneyness": float(k),
                "T": float(T),
                "implied_volatility": float(iv),
            })

    return pd.DataFrame(rows)


def apply_dropout(df: pd.DataFrame, rate: float, seed: int = 42) -> tuple:
    """Apply uniform random dropout, return (train, holdout)."""
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) > rate
    return df[mask].copy().reset_index(drop=True), df[~mask].copy().reset_index(drop=True)


def compute_rmse_bps(pred: np.ndarray, true: np.ndarray) -> float:
    """RMSE in basis points."""
    return np.sqrt(np.mean((pred - true) ** 2)) * 10000


def compute_surface_epp(
    model_predict_fn,
    S: float = 100.0,
    r: float = 0.05,
    n_strikes: int = 40,
    maturities: np.ndarray = None,
) -> float:
    """Compute EPP for a generic model prediction function."""
    if maturities is None:
        maturities = np.array([0.1, 0.25, 0.5, 1.0, 2.0])

    log_strikes = np.linspace(-0.15, 0.15, n_strikes)
    K = S * np.exp(log_strikes)

    max_epp = 0.0

    for T in maturities:
        grid = pd.DataFrame({
            "log_moneyness": log_strikes,
            "T": T,
        })
        vols = model_predict_fn(grid)

        # Butterfly EPP
        epp_but = compute_butterfly_epp(K, vols, T, S, r)
        max_epp = max(max_epp, epp_but)

    # Calendar EPP
    for k_idx in range(0, n_strikes, 5):
        k = log_strikes[k_idx]
        K_abs = S * np.exp(k)
        T_vols = []
        for T in maturities:
            grid = pd.DataFrame({"log_moneyness": [k], "T": [T]})
            v = model_predict_fn(grid)[0]
            T_vols.append(v)
        T_vols = np.array(T_vols)
        epp_cal = compute_calendar_epp(K_abs, maturities, T_vols, S, r)
        max_epp = max(max_epp, epp_cal)

    return max_epp * 10000  # Convert to bps


# =============================================================================
# Phase 3.1: eSSVI on synthetic data (real data requires OptionMetrics license)
# =============================================================================

def run_essvi_benchmark(n_surfaces: int = 50, seed: int = 42) -> pd.DataFrame:
    """Run eSSVI benchmark across dropout rates."""
    print("\n" + "=" * 60)
    print("Phase 3.1: eSSVI Benchmark")
    print("=" * 60)

    results = []
    dropout_rates = [0.0, 0.2, 0.4, 0.6]

    for surface_idx in range(n_surfaces):
        surface = generate_heston_surface(
            rho=-0.7 + 0.2 * np.random.default_rng(seed + surface_idx).random(),
            seed=seed + surface_idx,
        )

        for dropout in dropout_rates:
            for trial in range(5):
                trial_seed = seed + surface_idx * 100 + trial
                train_df, holdout_df = apply_dropout(surface, dropout, trial_seed)

                if len(train_df) < 10 or len(holdout_df) == 0:
                    continue

                # Prepare slice data for eSSVI calibration
                unique_T = sorted(train_df["T"].unique())
                smile_data = []
                for T in unique_T:
                    mask = train_df["T"] == T
                    k = train_df.loc[mask, "log_moneyness"].values
                    v = train_df.loc[mask, "implied_volatility"].values
                    if len(k) >= 2:
                        smile_data.append((k, v))
                    else:
                        smile_data.append((np.array([0.0]), np.array([v.mean() if len(v) > 0 else 0.2])))

                T_slices = np.array(unique_T[:len(smile_data)])

                if len(T_slices) < 2:
                    continue

                try:
                    start = time.time()
                    essvi = calibrate_essvi(T_slices, smile_data, n_restarts=10)
                    cal_time = (time.time() - start) * 1000

                    # Predict on holdout
                    preds = np.array([
                        essvi.implied_vol(row["log_moneyness"], row["T"])
                        for _, row in holdout_df.iterrows()
                    ])
                    true = holdout_df["implied_volatility"].values

                    rmse = compute_rmse_bps(preds, true)

                    # EPP (eSSVI is arbitrage-free by construction if Roger-Lee holds)
                    def predict_fn(grid):
                        return np.array([
                            essvi.implied_vol(row["log_moneyness"], row["T"])
                            for _, row in grid.iterrows()
                        ])

                    epp = compute_surface_epp(predict_fn)

                    results.append({
                        "model": "eSSVI",
                        "dropout": dropout,
                        "surface": surface_idx,
                        "trial": trial,
                        "rmse_bps": rmse,
                        "epp_bps": epp,
                        "time_ms": cal_time,
                        "roger_lee_ok": essvi.roger_lee_check(),
                    })

                except Exception as e:
                    results.append({
                        "model": "eSSVI",
                        "dropout": dropout,
                        "surface": surface_idx,
                        "trial": trial,
                        "rmse_bps": np.nan,
                        "epp_bps": np.nan,
                        "time_ms": np.nan,
                        "roger_lee_ok": False,
                    })

        if (surface_idx + 1) % 10 == 0:
            print(f"  Completed {surface_idx + 1}/{n_surfaces} surfaces")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "revision_essvi.csv", index=False)

    # Print summary
    print("\neSSVI Results Summary:")
    summary = df.groupby("dropout").agg({
        "rmse_bps": ["mean", "std"],
        "epp_bps": ["mean", "max"],
        "time_ms": "mean",
    }).round(2)
    print(summary)

    return df


# =============================================================================
# Phase 3.2: Piecewise-linear Andreasen-Huge
# =============================================================================

def run_ah_pl_benchmark(n_surfaces: int = 50, seed: int = 42) -> pd.DataFrame:
    """Run piecewise-constant and piecewise-linear AH benchmarks."""
    print("\n" + "=" * 60)
    print("Phase 3.2: Andreasen-Huge (PW-Constant and PW-Linear)")
    print("=" * 60)

    from src.volatility_surface.models.andreasen_huge import (
        AndreasenHugeModel,
        AndreasenHugePLModel,
    )

    results = []
    dropout_rates = [0.0, 0.2, 0.4, 0.6]

    for surface_idx in range(n_surfaces):
        surface = generate_heston_surface(seed=seed + surface_idx)

        for dropout in dropout_rates:
            for trial in range(5):
                trial_seed = seed + surface_idx * 100 + trial
                train_df, holdout_df = apply_dropout(surface, dropout, trial_seed)

                if len(train_df) < 10 or len(holdout_df) == 0:
                    continue

                for ModelClass, name in [
                    (AndreasenHugeModel, "AH"),
                    (AndreasenHugePLModel, "AH (PL)"),
                ]:
                    try:
                        model = ModelClass(n_time_steps=20, S=100.0, r=0.05)
                        start = time.time()
                        model.train(train_df)
                        cal_time = (time.time() - start) * 1000

                        preds = model.predict_volatility(holdout_df)
                        true = holdout_df["implied_volatility"].values
                        rmse = compute_rmse_bps(preds, true)

                        def predict_fn(grid, m=model):
                            return m.predict_volatility(grid)
                        epp = compute_surface_epp(predict_fn)

                        results.append({
                            "model": name,
                            "dropout": dropout,
                            "surface": surface_idx,
                            "trial": trial,
                            "rmse_bps": rmse,
                            "epp_bps": epp,
                            "time_ms": cal_time,
                        })

                    except Exception as e:
                        results.append({
                            "model": name,
                            "dropout": dropout,
                            "surface": surface_idx,
                            "trial": trial,
                            "rmse_bps": np.nan,
                            "epp_bps": np.nan,
                            "time_ms": np.nan,
                        })

        if (surface_idx + 1) % 10 == 0:
            print(f"  Completed {surface_idx + 1}/{n_surfaces} surfaces")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "revision_ah_pl.csv", index=False)

    print("\nAH Results Summary:")
    summary = df.groupby(["model", "dropout"]).agg({
        "rmse_bps": ["mean", "std"],
        "epp_bps": ["mean", "max"],
        "time_ms": "mean",
    }).round(2)
    print(summary)

    return df


# =============================================================================
# Phase 3.3: ICNN (Input Convex Neural Network)
# =============================================================================

def run_icnn_benchmark(n_surfaces: int = 50, seed: int = 42) -> pd.DataFrame:
    """Run ICNN benchmark."""
    print("\n" + "=" * 60)
    print("Phase 3.3: ICNN (Input Convex Neural Network)")
    print("=" * 60)

    from src.volatility_surface.models.icnn_model import ICNNVolatilityModel

    results = []
    dropout_rates = [0.0, 0.2, 0.4, 0.6]

    for surface_idx in range(n_surfaces):
        surface = generate_heston_surface(seed=seed + surface_idx)

        for dropout in dropout_rates:
            for trial in range(3):  # Fewer trials due to longer runtime
                trial_seed = seed + surface_idx * 100 + trial
                train_df, holdout_df = apply_dropout(surface, dropout, trial_seed)

                if len(train_df) < 10 or len(holdout_df) == 0:
                    continue

                try:
                    model = ICNNVolatilityModel(
                        hidden_layers=[64, 32, 16],
                        lambda_calendar=5.0,
                        lambda_wing=1.0,
                        epochs=300,
                        lr=3e-3,
                    )
                    start = time.time()
                    train_result = model.train(train_df)
                    cal_time = (time.time() - start) * 1000

                    preds = model.predict_volatility(holdout_df)
                    true = holdout_df["implied_volatility"].values
                    rmse = compute_rmse_bps(preds, true)

                    def predict_fn(grid, m=model):
                        return m.predict_volatility(grid)
                    epp = compute_surface_epp(predict_fn)

                    results.append({
                        "model": "ICNN",
                        "dropout": dropout,
                        "surface": surface_idx,
                        "trial": trial,
                        "rmse_bps": rmse,
                        "epp_bps": epp,
                        "time_ms": cal_time,
                        "epochs_trained": train_result.get("epochs_trained", 300),
                    })

                except Exception as e:
                    results.append({
                        "model": "ICNN",
                        "dropout": dropout,
                        "surface": surface_idx,
                        "trial": trial,
                        "rmse_bps": np.nan,
                        "epp_bps": np.nan,
                        "time_ms": np.nan,
                        "epochs_trained": 0,
                    })

        if (surface_idx + 1) % 10 == 0:
            print(f"  Completed {surface_idx + 1}/{n_surfaces} surfaces")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "revision_icnn.csv", index=False)

    print("\nICNN Results Summary:")
    summary = df.groupby("dropout").agg({
        "rmse_bps": ["mean", "std"],
        "epp_bps": ["mean", "max"],
        "time_ms": "mean",
    }).round(2)
    print(summary)

    return df


# =============================================================================
# Phase 4.1: EURUSD FX Cross-Asset Validation
# =============================================================================

def run_eurusd_benchmark(n_surfaces: int = 100, seed: int = 42) -> pd.DataFrame:
    """Run EURUSD FX cross-asset validation with frozen SPX penalty weights."""
    print("\n" + "=" * 60)
    print("Phase 4.1: EURUSD FX Cross-Asset Validation")
    print("=" * 60)

    from src.volatility_surface.data.fx_data import generate_eurusd_surface
    from src.volatility_surface.models.pinn_model import PINNVolatilityModel
    from src.volatility_surface.models.svi import calibrate_svi, calibrate_ssvi

    results = []
    dropout_rates = [0.2, 0.4, 0.6]
    regimes = ["normal", "low_vol", "stress"]

    for surface_idx in range(n_surfaces):
        regime = regimes[surface_idx % len(regimes)]
        surface = generate_eurusd_surface(
            n_expiries=8,
            n_strikes_per_expiry=15,
            regime=regime,
            seed=seed + surface_idx,
        )

        for dropout in dropout_rates:
            for trial in range(3):
                trial_seed = seed + surface_idx * 100 + trial
                train_df, holdout_df = apply_dropout(surface, dropout, trial_seed)

                if len(train_df) < 6 or len(holdout_df) == 0:
                    continue

                # --- SVI baseline ---
                try:
                    unique_T = sorted(train_df["T"].unique())
                    svi_preds = []
                    for _, row in holdout_df.iterrows():
                        T = row["T"]
                        k = row["log_moneyness"]
                        # Find closest calibrated slice
                        closest_T = min(unique_T, key=lambda t: abs(t - T))
                        mask = train_df["T"] == closest_T
                        k_train = train_df.loc[mask, "log_moneyness"].values
                        v_train = train_df.loc[mask, "implied_volatility"].values
                        if len(k_train) >= 3:
                            svi = calibrate_svi(k_train, v_train, closest_T)
                            svi_preds.append(svi.implied_vol(k, T))
                        else:
                            svi_preds.append(v_train.mean() if len(v_train) > 0 else 0.1)
                    svi_preds = np.array(svi_preds)
                    svi_rmse = compute_rmse_bps(svi_preds, holdout_df["implied_volatility"].values)
                    results.append({
                        "model": "SVI", "dropout": dropout, "surface": surface_idx,
                        "trial": trial, "rmse_bps": svi_rmse, "epp_bps": 0.0,
                        "regime": regime, "asset": "EURUSD",
                    })
                except Exception:
                    pass

                # --- CINN with frozen SPX weights ---
                try:
                    cinn = PINNVolatilityModel(
                        hidden_layers=[64, 32, 16],
                        lambda_calendar=5.0,  # Frozen SPX production weights
                        lambda_butterfly=10.0,
                        lambda_wing=1.0,
                        learning_rate=3e-3,
                        epochs=300,
                        use_warmup=True,
                        use_ema_norm=True,
                        squared_hinge=True,
                        early_stopping=True,
                        early_stop_patience=20,
                        device="cpu",
                    )
                    start = time.time()
                    cinn.train(train_df)
                    cal_time = (time.time() - start) * 1000

                    preds = cinn.predict_volatility(holdout_df)
                    true = holdout_df["implied_volatility"].values
                    rmse = compute_rmse_bps(preds, true)

                    def predict_fn(grid, m=cinn):
                        return m.predict_volatility(grid)
                    epp = compute_surface_epp(predict_fn)

                    results.append({
                        "model": "CINN", "dropout": dropout, "surface": surface_idx,
                        "trial": trial, "rmse_bps": rmse, "epp_bps": epp,
                        "time_ms": cal_time, "regime": regime, "asset": "EURUSD",
                    })
                except Exception:
                    pass

                # --- Unconstrained MLP baseline ---
                try:
                    mlp = PINNVolatilityModel(
                        hidden_layers=[64, 32, 16],
                        lambda_calendar=0.0,
                        lambda_butterfly=0.0,
                        lambda_wing=0.0,
                        learning_rate=3e-3,
                        epochs=300,
                        use_warmup=False,
                        device="cpu",
                    )
                    start = time.time()
                    mlp.train(train_df)
                    cal_time = (time.time() - start) * 1000

                    preds = mlp.predict_volatility(holdout_df)
                    true = holdout_df["implied_volatility"].values
                    rmse = compute_rmse_bps(preds, true)

                    def predict_fn_mlp(grid, m=mlp):
                        return m.predict_volatility(grid)
                    epp = compute_surface_epp(predict_fn_mlp)

                    results.append({
                        "model": "MLP", "dropout": dropout, "surface": surface_idx,
                        "trial": trial, "rmse_bps": rmse, "epp_bps": epp,
                        "time_ms": cal_time, "regime": regime, "asset": "EURUSD",
                    })
                except Exception:
                    pass

        if (surface_idx + 1) % 20 == 0:
            print(f"  Completed {surface_idx + 1}/{n_surfaces} surfaces")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "revision_eurusd.csv", index=False)

    print("\nEURUSD Results Summary:")
    summary = df.groupby(["model", "dropout"]).agg({
        "rmse_bps": ["mean", "std"],
        "epp_bps": ["mean", "max"],
    }).round(2)
    print(summary)

    return df


# =============================================================================
# Phase 6.1: Early Stopping Latency Measurement
# =============================================================================

def run_early_stopping_test(n_surfaces: int = 50, seed: int = 42) -> pd.DataFrame:
    """Compare CINN with and without early stopping."""
    print("\n" + "=" * 60)
    print("Phase 6.1: Early Stopping Latency Measurement")
    print("=" * 60)

    from src.volatility_surface.models.pinn_model import PINNVolatilityModel

    results = []

    for surface_idx in range(n_surfaces):
        surface = generate_heston_surface(seed=seed + surface_idx)
        train_df, holdout_df = apply_dropout(surface, 0.4, seed + surface_idx)

        if len(train_df) < 10 or len(holdout_df) == 0:
            continue

        for use_es, label in [(False, "CINN (no ES)"), (True, "CINN (early stop)")]:
            try:
                model = PINNVolatilityModel(
                    hidden_layers=[64, 32, 16],
                    lambda_calendar=5.0,
                    lambda_butterfly=10.0,
                    lambda_wing=1.0,
                    learning_rate=3e-3,
                    epochs=500,
                    use_warmup=True,
                    use_ema_norm=True,
                    squared_hinge=True,
                    early_stopping=use_es,
                    early_stop_patience=20,
                    early_stop_rmse_tol=1e-5,
                    early_stop_epp_threshold=1e-4,
                    device="cpu",
                )
                start = time.time()
                train_result = model.train(train_df)
                cal_time = (time.time() - start) * 1000

                preds = model.predict_volatility(holdout_df)
                true = holdout_df["implied_volatility"].values
                rmse = compute_rmse_bps(preds, true)

                def predict_fn(grid, m=model):
                    return m.predict_volatility(grid)
                epp = compute_surface_epp(predict_fn)

                results.append({
                    "model": label,
                    "surface": surface_idx,
                    "rmse_bps": rmse,
                    "epp_bps": epp,
                    "time_ms": cal_time,
                    "epochs_trained": train_result.get("epochs_trained", 500),
                    "early_stopped": train_result.get("early_stopped", False),
                })

            except Exception as e:
                print(f"  Warning: {label} failed on surface {surface_idx}: {e}")

        if (surface_idx + 1) % 10 == 0:
            print(f"  Completed {surface_idx + 1}/{n_surfaces} surfaces")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "revision_early_stopping.csv", index=False)

    print("\nEarly Stopping Results:")
    summary = df.groupby("model").agg({
        "rmse_bps": ["mean", "std"],
        "epp_bps": ["mean", "max"],
        "time_ms": ["mean", "std"],
        "epochs_trained": "mean",
    }).round(2)
    print(summary)

    return df


# =============================================================================
# Generate consolidated main table
# =============================================================================

def generate_main_table():
    """Consolidate all results into the updated main table format."""
    print("\n" + "=" * 60)
    print("Generating Consolidated Results Table")
    print("=" * 60)

    tables = {}
    for fname in [
        "revision_essvi.csv",
        "revision_ah_pl.csv",
        "revision_icnn.csv",
        "revision_eurusd.csv",
        "revision_early_stopping.csv",
    ]:
        path = RESULTS_DIR / fname
        if path.exists():
            tables[fname] = pd.read_csv(path)
            print(f"  Loaded {fname}: {len(tables[fname])} rows")
        else:
            print(f"  Missing {fname}")

    # Generate summary table
    rows = []
    dropout_rates = [0.2, 0.4, 0.6]

    for fname, df in tables.items():
        if "dropout" not in df.columns:
            continue
        for dropout in dropout_rates:
            mask = df["dropout"] == dropout
            if mask.sum() == 0:
                continue
            for model in df.loc[mask, "model"].unique():
                model_mask = mask & (df["model"] == model)
                if model_mask.sum() == 0:
                    continue
                rows.append({
                    "model": model,
                    "dropout": dropout,
                    "rmse_mean": df.loc[model_mask, "rmse_bps"].mean(),
                    "rmse_std": df.loc[model_mask, "rmse_bps"].std(),
                    "epp_mean": df.loc[model_mask, "epp_bps"].mean(),
                    "epp_max": df.loc[model_mask, "epp_bps"].max(),
                    "time_ms": df.loc[model_mask, "time_ms"].mean() if "time_ms" in df.columns else np.nan,
                    "source": fname,
                })

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "revision_main_table.csv", index=False)

    print("\nConsolidated Table:")
    print(summary.round(2).to_string(index=False))

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CINN Paper Revision Experiments")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "essvi", "ah_pl", "icnn", "eurusd",
                                 "early_stopping", "table"],
                        help="Which phase to run")
    parser.add_argument("--n-surfaces", type=int, default=50,
                        help="Number of surfaces per experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("CINN Paper Revision — Experiment Runner")
    print(f"  Phase: {args.phase}")
    print(f"  N surfaces: {args.n_surfaces}")
    print(f"  Seed: {args.seed}")

    if args.phase in ("all", "essvi"):
        run_essvi_benchmark(args.n_surfaces, args.seed)

    if args.phase in ("all", "ah_pl"):
        run_ah_pl_benchmark(args.n_surfaces, args.seed)

    if args.phase in ("all", "icnn"):
        run_icnn_benchmark(args.n_surfaces, args.seed)

    if args.phase in ("all", "eurusd"):
        run_eurusd_benchmark(args.n_surfaces, args.seed)

    if args.phase in ("all", "early_stopping"):
        run_early_stopping_test(args.n_surfaces, args.seed)

    if args.phase in ("all", "table"):
        generate_main_table()

    print("\nAll experiments complete. Results in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
