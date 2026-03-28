"""
Master reproduction script: runs all experiments and generates all paper tables.

Usage:
    python scripts/reproduce_results.py          # Full run (~2 hours)
    python scripts/reproduce_results.py --quick   # Quick validation (~5 min)

Output: results/ directory with all CSV tables matching the paper.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    quick = "--quick" in sys.argv
    mode = "QUICK" if quick else "FULL"

    print("=" * 70)
    print(f"  REPRODUCE ALL PAPER RESULTS ({mode} MODE)")
    print("=" * 70)

    t_start = time.time()

    # Table 1 & 2: Main benchmark (sparse + full)
    print("\n[1/5] Running main benchmark (Tables 1-2)...")
    from scripts.reproduce_benchmark import main as run_benchmark
    run_benchmark()

    # Table 3: Delta hedging
    print("\n[2/5] Running delta hedging experiment (Table 3)...")
    from scripts.run_hedging import run_hedging
    run_hedging(quick=quick)

    # Table 4: Market making
    print("\n[3/5] Running market making simulation (Table 4)...")
    from scripts.run_mm_sim import run_mm_comparison
    run_mm_comparison(quick=quick)

    # Table 5: Ablation study
    print("\n[4/5] Running ablation study (Table 5)...")
    from scripts.run_ablation import run_ablation
    run_ablation(quick=quick)

    # UQ validation
    print("\n[5/5] Running UQ validation...")
    from scripts.run_uq_validation import run_uq_validation
    run_uq_validation(quick=quick)

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  ALL EXPERIMENTS COMPLETE ({elapsed/60:.1f} minutes)")
    print("=" * 70)

    # List output files
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results"
    )
    print(f"\nOutput files in {results_dir}:")
    for f in sorted(os.listdir(results_dir)):
        if f.endswith('.csv'):
            fpath = os.path.join(results_dir, f)
            size = os.path.getsize(fpath)
            print(f"  {f:30s} ({size:,d} bytes)")


if __name__ == "__main__":
    main()
