#!/usr/bin/env python3
"""
Train and benchmark the amortized CINN model.

Usage:
    python scripts/train_and_benchmark_amortized.py
    python scripts/train_and_benchmark_amortized.py --n-surfaces 200 --epochs 50
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
MODEL_DIR = Path("models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-surfaces", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--embed-dim", type=int, default=64)
    args = parser.parse_args()

    from src.volatility_surface.models.amortized_model import (
        train_amortized,
        benchmark_amortized,
    )
    from scripts.run_paper_revision import generate_heston_surface, apply_dropout, compute_rmse_bps

    print("=" * 60)
    print("Amortized CINN Training + Benchmark")
    print(f"  Training surfaces: {args.n_surfaces}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Embedding dim: {args.embed_dim}")
    print("=" * 60)

    # Train
    t0 = time.time()
    model, history = train_amortized(
        n_surfaces=args.n_surfaces,
        embed_dim=args.embed_dim,
        epochs=args.epochs,
    )
    train_time = time.time() - t0
    print(f"\nTraining complete in {train_time:.1f}s")
    print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.6f}")

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    import torch
    torch.save(model.state_dict(), MODEL_DIR / "amortized_cinn.pt")
    print(f"  Model saved to {MODEL_DIR / 'amortized_cinn.pt'}")

    # Benchmark
    print(f"\nBenchmarking on {args.n_test} test surfaces (40% dropout)...")
    results = benchmark_amortized(model, n_test=args.n_test)

    amortized = results["amortized"]
    if amortized:
        rmses = [r["rmse_bps"] for r in amortized]
        times = [r["time_ms"] for r in amortized]
        print(f"\n{'='*60}")
        print("Amortized CINN Results")
        print(f"{'='*60}")
        print(f"  RMSE: {np.mean(rmses):.1f} +/- {np.std(rmses):.1f} bps")
        print(f"  Inference time: {np.mean(times):.1f} +/- {np.std(times):.1f} ms")
        print(f"  Speedup vs CINN (~3000ms): {3000/np.mean(times):.0f}x")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    import pandas as pd
    pd.DataFrame(amortized).to_csv(RESULTS_DIR / "amortized_benchmark.csv", index=False)


if __name__ == "__main__":
    main()
