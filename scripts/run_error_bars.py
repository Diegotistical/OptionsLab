"""
Error bars and significance tests: run main benchmark across multiple seeds.
Produces confidence intervals and Wilcoxon signed-rank tests for Table 1/2.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, bootstrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_single_seed(seed, epochs=300, n_strikes=30):
    """Run CINN and MLP on a single seed, return RMSE and arb-free %."""
    from src.volatility_surface.models.pinn_model import PINNVolatilityModel
    from src.benchmarks.vol_surface_benchmark import generate_synthetic_surface

    df = generate_synthetic_surface(n_strikes=n_strikes, seed=seed)

    # CINN
    cinn = PINNVolatilityModel(
        epochs=epochs, hidden_layers=[64, 32, 16],
        lambda_calendar=1.0, lambda_butterfly=0.5, lambda_wing=0.1,
        use_warmup=True, squared_hinge=True,
    )
    cinn.train(df)
    cinn_preds = cinn.predict_volatility(df)
    cinn_rmse = np.sqrt(np.mean((cinn_preds - df['implied_volatility'].values)**2))

    # MLP (unconstrained)
    mlp = PINNVolatilityModel(
        epochs=epochs, hidden_layers=[64, 32, 16],
        lambda_calendar=0.0, lambda_butterfly=0.0, lambda_wing=0.0,
    )
    mlp.train(df)
    mlp_preds = mlp.predict_volatility(df)
    mlp_rmse = np.sqrt(np.mean((mlp_preds - df['implied_volatility'].values)**2))

    return {
        'seed': seed,
        'cinn_rmse': float(cinn_rmse),
        'mlp_rmse': float(mlp_rmse),
    }


def run_error_bars(quick=False):
    n_seeds = 3 if quick else 10
    epochs = 100 if quick else 300
    seeds = list(range(42, 42 + n_seeds))

    print("=" * 60)
    print(f"Error Bars: {n_seeds} seeds x {epochs} epochs")
    print("=" * 60)

    results = []
    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i+1}/{n_seeds}) ---")
        t0 = time.time()
        row = run_single_seed(seed, epochs=epochs)
        elapsed = time.time() - t0
        print(f"  CINN RMSE={row['cinn_rmse']:.4f}, MLP RMSE={row['mlp_rmse']:.4f} ({elapsed:.1f}s)")
        results.append(row)

    df = pd.DataFrame(results)

    # Summary stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model in ['cinn', 'mlp']:
        vals = df[f'{model}_rmse'].values
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)
        ci_lo = mean - 1.96 * std / np.sqrt(len(vals))
        ci_hi = mean + 1.96 * std / np.sqrt(len(vals))
        print(f"  {model.upper()}: RMSE = {mean:.4f} +/- {std:.4f} (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")

    # Wilcoxon signed-rank test (paired)
    cinn_rmses = df['cinn_rmse'].values
    mlp_rmses = df['mlp_rmse'].values
    diffs = mlp_rmses - cinn_rmses

    if len(diffs) >= 5:
        try:
            stat, pval = wilcoxon(diffs, alternative='greater')
            print(f"\n  Wilcoxon signed-rank test (MLP > CINN):")
            print(f"    Statistic = {stat:.1f}, p-value = {pval:.4f}")
            if pval < 0.05:
                print(f"    CINN significantly better than MLP at alpha=0.05")
            else:
                print(f"    Not significant at alpha=0.05")
        except Exception as e:
            print(f"  Wilcoxon test failed: {e}")
    else:
        print(f"\n  (Need >= 5 seeds for Wilcoxon test, have {len(diffs)})")

    # Save
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results"
    )
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "error_bars.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    return df


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    run_error_bars(quick=quick)
