#!/usr/bin/env python3
"""
Reproduce all benchmark tables and figures from the CINN volatility paper.

Generates:
  - Table 1: Sparse strike stress test (EPP + RMSE across models)
  - Table 2: Full data benchmark
  - Table 3: Delta hedging comparison
  - Table 4: Market making simulation comparison
  - Figure 1: Vol surface comparison (3D plots)
  - Figure 2: EPP vs model type
  - Figure 3: Uncertainty bands (MC dropout)

Usage:
    python scripts/reproduce_benchmark.py --tables --figures --output results/
    python scripts/reproduce_benchmark.py --quick  # Fast smoke test
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")


def train_models(surface_data: pd.DataFrame, quick: bool = False):
    """Train all vol surface models on the same data."""
    from src.volatility_surface.models.pinn_model import PINNVolatilityModel
    from src.benchmarks.vol_surface_benchmark import (
        MLPWrapper,
        SVIWrapper,
        SABRWrapper,
    )

    epochs = 100 if quick else 500
    models = {}

    # CINN (our model)
    print("  Training CINN...")
    cinn = PINNVolatilityModel(
        hidden_layers=[64, 32, 16],
        lambda_calendar=1.0,
        lambda_butterfly=0.5,
        lambda_wing=0.1,
        epochs=epochs,
        batch_size=64,
        dropout=0.1,
    )
    cinn.train(surface_data)
    models["CINN"] = cinn

    # Unconstrained MLP (ablation)
    print("  Training MLP (unconstrained)...")
    mlp_unconstrained = PINNVolatilityModel(
        hidden_layers=[64, 32, 16],
        lambda_calendar=0.0,
        lambda_butterfly=0.0,
        lambda_wing=0.0,
        epochs=epochs,
        batch_size=64,
        dropout=0.1,
    )
    mlp_unconstrained.train(surface_data)
    models["MLP"] = mlp_unconstrained

    # SVI (parametric baseline) — use wrapper interface
    print("  Training SVI...")
    models["SVI_wrapper"] = SVIWrapper()

    # SABR (parametric baseline) — use wrapper interface
    print("  Training SABR...")
    models["SABR_wrapper"] = SABRWrapper()

    return models


def generate_table_1_sparse(models, output_dir: str, quick: bool = False):
    """Table 1: Sparse strike stress test with EPP metric."""
    from src.benchmarks.vol_surface_benchmark import generate_synthetic_surface
    from src.volatility_surface.utils.epp import compute_epp

    print("\n=== Table 1: Sparse Strike Stress Test ===")

    sparsity_levels = [5, 10, 20, 30] if not quick else [10, 20]
    results = []

    for n_strikes in sparsity_levels:
        print(f"  n_strikes = {n_strikes}")
        surface = generate_synthetic_surface(n_strikes=n_strikes, seed=42)

        for name, model in models.items():
            if name.endswith("_wrapper"):
                continue  # Wrappers need per-slice calibration

            # Retrain on sparse data
            model.train(surface)
            preds = model.predict_volatility(surface)
            true_vols = surface["implied_volatility"].values

            rmse = np.sqrt(np.mean((preds - true_vols) ** 2))
            mae = np.mean(np.abs(preds - true_vols))

            # EPP metric
            epp_result = compute_epp(model, S=100.0, r=0.05, n_strikes=30)

            results.append({
                "Model": name,
                "n_strikes": n_strikes,
                "RMSE": rmse,
                "MAE": mae,
                "EPP": epp_result["epp_total"],
                "Butterfly Violations": epp_result["butterfly_violations"],
                "Calendar Violations": epp_result["calendar_violations"],
            })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(output_dir, "table1_sparse.csv"), index=False)
    return df


def generate_table_2_fulldata(models, output_dir: str, quick: bool = False):
    """Table 2: Full data benchmark."""
    from src.benchmarks.vol_surface_benchmark import (
        VolSurfaceBenchmark,
        generate_synthetic_surface,
    )

    print("\n=== Table 2: Full Data Benchmark ===")

    n_trials = 3 if quick else 10
    surface = generate_synthetic_surface(n_strikes=40, seed=42)

    model_names = ["svi", "sabr", "mlp", "pinn"]
    benchmark = VolSurfaceBenchmark(models=model_names, verbose=True)
    results = benchmark.run(surface, n_trials=n_trials)

    df = results.to_dataframe()
    print(df.to_string())
    df.to_csv(os.path.join(output_dir, "table2_fulldata.csv"))
    return df


def generate_table_3_hedging(models, output_dir: str, quick: bool = False):
    """Table 3: Delta hedging comparison."""
    from src.experiments.delta_hedging import DeltaHedgingExperiment

    print("\n=== Table 3: Delta Hedging Comparison ===")

    # Filter to trainable models only
    vol_models = {k: v for k, v in models.items() if not k.endswith("_wrapper")}

    n_paths = 500 if quick else 5000
    n_options = 10 if quick else 50

    all_results = []

    for freq in (["daily"] if quick else ["daily", "4h"]):
        exp = DeltaHedgingExperiment(
            vol_models=vol_models,
            S0=100.0,
            r=0.05,
            true_vol=0.2,
        )

        results = exp.run(
            n_paths=n_paths,
            n_options=n_options,
            rebalance_freq=freq,
            seed=42,
        )

        df = results.to_dataframe()
        df["Rebalance"] = freq
        all_results.append(df)
        print(f"\n  Frequency: {freq}")
        print(df.to_string())

    combined = pd.concat(all_results)
    combined.to_csv(os.path.join(output_dir, "table3_hedging.csv"))
    return combined


def generate_table_4_mm(models, output_dir: str, quick: bool = False):
    """Table 4: Market making simulation comparison."""
    from src.experiments.mm_simulation import run_mm_comparison

    print("\n=== Table 4: Market Making Simulation ===")

    # Filter to trainable models + baseline
    vol_models = {"Baseline (Rolling sigma)": None}
    for k, v in models.items():
        if not k.endswith("_wrapper"):
            vol_models[k] = v

    n_runs = 5 if quick else 50
    n_steps = 2000 if quick else 10000

    results = run_mm_comparison(
        vol_models=vol_models,
        n_runs=n_runs,
        n_steps=n_steps,
        base_seed=42,
    )

    df = results.to_dataframe()
    print(df.to_string())
    df.to_csv(os.path.join(output_dir, "table4_mm.csv"))
    return df


def generate_figures(models, output_dir: str, quick: bool = False):
    """Generate all figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("  matplotlib not available, skipping figures")
        return

    from src.benchmarks.vol_surface_benchmark import generate_synthetic_surface

    print("\n=== Generating Figures ===")

    surface = generate_synthetic_surface(n_strikes=30, seed=42)

    # Figure 1: Vol surface comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": "3d"})

    for ax, (name, model) in zip(axes, [(k, v) for k, v in models.items() if not k.endswith("_wrapper")][:2]):
        preds = model.predict_volatility(surface)
        k = surface["log_moneyness"].values
        T = surface["T"].values

        ax.scatter(k, T, surface["implied_volatility"].values, c="blue", alpha=0.3, s=10, label="Market")
        ax.scatter(k, T, preds, c="red", alpha=0.3, s=10, label=name)
        ax.set_xlabel("Log-Moneyness")
        ax.set_ylabel("Maturity")
        ax.set_zlabel("Implied Vol")
        ax.set_title(name)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_surface_comparison.png"), dpi=150)
    plt.close()
    print("  Saved fig1_surface_comparison.png")

    # Figure 2: Uncertainty bands (MC dropout)
    if "CINN" in models:
        cinn = models["CINN"]
        if hasattr(cinn, "predict_with_uncertainty"):
            fig, ax = plt.subplots(figsize=(10, 6))

            # Query at single maturity
            T_fixed = 0.5
            k_range = np.linspace(-0.4, 0.4, 100)
            query_df = pd.DataFrame({"log_moneyness": k_range, "T": T_fixed})

            mean_vol, std_vol = cinn.predict_with_uncertainty(query_df, n_samples=50)

            ax.plot(k_range, mean_vol, "b-", lw=2, label="CINN Mean")
            ax.fill_between(k_range, mean_vol - 2 * std_vol, mean_vol + 2 * std_vol,
                           alpha=0.3, color="blue", label="±2σ (MC Dropout)")

            # Add market points
            mask = np.isclose(surface["T"].values, T_fixed)
            if mask.sum() > 0:
                ax.scatter(surface.loc[mask, "log_moneyness"],
                          surface.loc[mask, "implied_volatility"],
                          c="red", s=30, zorder=5, label="Market")

            ax.set_xlabel("Log-Moneyness")
            ax.set_ylabel("Implied Volatility")
            ax.set_title(f"CINN Uncertainty (T={T_fixed})")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "fig2_uncertainty.png"), dpi=150)
            plt.close()
            print("  Saved fig2_uncertainty.png")

    print("  Figures complete.")


def main():
    parser = argparse.ArgumentParser(description="Reproduce paper results")
    parser.add_argument("--tables", action="store_true", help="Generate tables")
    parser.add_argument("--figures", action="store_true", help="Generate figures")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Generate everything")
    args = parser.parse_args()

    if args.all:
        args.tables = True
        args.figures = True

    if not args.tables and not args.figures:
        args.tables = True
        args.figures = True

    output_dir = os.path.join(str(PROJECT_ROOT), args.output)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("CINN Volatility Paper — Reproduction Script")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Generate training data
    from src.benchmarks.vol_surface_benchmark import generate_synthetic_surface

    print("\nGenerating synthetic surface...")
    surface = generate_synthetic_surface(n_strikes=30, seed=42)
    print(f"  {len(surface)} points, {surface['T'].nunique()} maturities")

    # Train models
    print("\nTraining models...")
    start = time.time()
    models = train_models(surface, quick=args.quick)
    print(f"  Training took {time.time() - start:.1f}s")

    if args.tables:
        generate_table_1_sparse(models, output_dir, args.quick)
        generate_table_2_fulldata(models, output_dir, args.quick)
        generate_table_3_hedging(models, output_dir, args.quick)
        generate_table_4_mm(models, output_dir, args.quick)

    if args.figures:
        generate_figures(models, output_dir, args.quick)

    print("\n" + "=" * 60)
    print("DONE. All results saved to:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
