#!/usr/bin/env python3
"""
Verify the penalty-EPP theoretical bound on synthetic surfaces.

For each surface: train CINN, compute butterfly loss, theoretical EPP bound,
and actual EPP. Report bound validity and tightness.

Usage:
    python scripts/run_bound_verification.py
    python scripts/run_bound_verification.py --n-surfaces 20
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-surfaces", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from scripts.run_paper_revision import generate_heston_surface, apply_dropout
    from src.volatility_surface.models.pinn_model import PINNVolatilityModel
    from src.volatility_surface.utils.theoretical_bounds import verify_bound

    import torch

    print("=" * 60)
    print("Penalty-EPP Bound Verification")
    print(f"  Surfaces: {args.n_surfaces}")
    print("=" * 60)

    results = []
    for i in range(args.n_surfaces):
        seed = args.seed + i
        surface = generate_heston_surface(seed=seed)
        train_df, holdout_df = apply_dropout(surface, 0.4, seed)

        if len(train_df) < 10:
            continue

        # Train CINN with production settings
        model = PINNVolatilityModel(
            hidden_layers=[64, 32, 16],
            lambda_calendar=5.0,
            lambda_butterfly=10.0,
            lambda_wing=1.0,
            epochs=300,
            learning_rate=3e-3,
        )
        model.train(train_df)

        # Get the underlying network
        network = model.model
        if network is None:
            continue

        try:
            result = verify_bound(network, train_df)
            results.append(result)

            status = "HOLDS" if result["bound_holds"] else "FAILS"
            print(
                f"  Surface {i+1}/{args.n_surfaces}: "
                f"L_but={result['butterfly_loss']:.2e}, "
                f"EPP_bound={result['epp_bound_bps']:.2f}bps, "
                f"EPP_actual={result['epp_actual_bps']:.4f}bps, "
                f"tightness={result['tightness_ratio']:.0f}x [{status}]"
            )
        except Exception as e:
            print(f"  Surface {i+1}/{args.n_surfaces}: ERROR - {e}")

    if not results:
        print("No results!")
        return

    df = pd.DataFrame(results)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "bound_verification.csv", index=False)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Surfaces tested: {len(df)}")
    print(f"Bound holds: {df['bound_holds'].sum()}/{len(df)} ({100*df['bound_holds'].mean():.0f}%)")
    print(f"Butterfly loss: {df['butterfly_loss'].mean():.2e} (mean)")
    print(f"EPP bound: {df['epp_bound_bps'].mean():.2f} bps (mean)")
    print(f"EPP actual: {df['epp_actual_bps'].mean():.4f} bps (mean)")
    print(f"Tightness ratio: {df['tightness_ratio'].median():.0f}x (median)")
    print(f"L_g (density Lipschitz): {df['lipschitz_g'].mean():.1f} (mean)")


if __name__ == "__main__":
    main()
