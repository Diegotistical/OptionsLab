#!/usr/bin/env python3
"""
Real-data validation: SPY options via yfinance.

Fetches live SPY options, converts to CINN format, and runs the same
dropout benchmark as the synthetic experiments. Caches fetched data
to avoid repeated API calls.

Usage:
    python scripts/run_spy_validation.py
    python scripts/run_spy_validation.py --use-cache  # reuse cached data
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

CACHE_DIR = Path("data/spy_surfaces")
RESULTS_DIR = Path("results")


def fetch_spy_surface(n_expiries: int = 10) -> pd.DataFrame:
    """Fetch SPY IV surface directly via yfinance and convert to CINN format."""
    import time as _time
    import yfinance as yf

    _time.sleep(3)  # Respect rate limits
    ticker = yf.Ticker("SPY")

    try:
        S = ticker.fast_info.last_price
    except Exception:
        hist = ticker.history(period="1d")
        S = hist["Close"].iloc[-1] if len(hist) > 0 else 550.0
    print(f"SPY spot: ${S:.2f}")

    _time.sleep(2)
    expiries = list(ticker.options)[:n_expiries]
    print(f"Fetching {len(expiries)} expiries: {expiries[:5]}...")

    rows = []
    from datetime import datetime
    today = datetime.now()

    for exp in expiries:
        _time.sleep(2)  # Rate limit between expiry fetches
        try:
            chain = ticker.option_chain(exp)
        except Exception as e:
            print(f"  Skipping {exp}: {e}")
            continue

        calls = chain.calls.copy()
        if len(calls) == 0:
            continue

        # Data quality filters
        calls = calls[calls["volume"] > 10]
        calls = calls[calls["impliedVolatility"] > 0.01]
        calls = calls[calls["impliedVolatility"] < 3.0]

        # Bid-ask spread filter
        mid = (calls["bid"] + calls["ask"]) / 2
        spread = (calls["ask"] - calls["bid"]) / mid.clip(lower=0.01)
        calls = calls[spread < 0.5]

        # Compute DTE and moneyness
        exp_date = datetime.strptime(exp, "%Y-%m-%d")
        dte = (exp_date - today).days
        T_years = dte / 365.0

        moneyness = calls["strike"] / S
        calls = calls[(moneyness > 0.8) & (moneyness < 1.2)]

        # Filter short-dated (< 7 days)
        if T_years < 7 / 365.0 or len(calls) < 3:
            continue

        log_m = np.log(calls["strike"].values / S)

        for lm, iv in zip(log_m, calls["impliedVolatility"].values):
            rows.append({
                "log_moneyness": float(lm),
                "T": float(T_years),
                "implied_volatility": float(iv),
            })
        print(f"  {exp}: {len(calls)} calls, T={T_years:.3f}yr")

    df = pd.DataFrame(rows)

    # Remove outliers: IV > 3 std from slice median
    if len(df) > 0:
        for T_val in df["T"].unique():
            mask = df["T"] == T_val
            slice_iv = df.loc[mask, "implied_volatility"]
            median = slice_iv.median()
            std = slice_iv.std()
            if std > 0:
                outlier = (slice_iv - median).abs() > 3 * std
                df = df[~(mask & outlier)]

    # Require >= 5 strikes per slice
    slice_counts = df.groupby("T").size()
    valid_slices = slice_counts[slice_counts >= 5].index
    df = df[df["T"].isin(valid_slices)]

    print(f"Surface: {len(df)} points across {df['T'].nunique()} slices")
    print(f"  Log-moneyness range: [{df['log_moneyness'].min():.3f}, {df['log_moneyness'].max():.3f}]")
    print(f"  T range: [{df['T'].min():.3f}, {df['T'].max():.3f}]")
    print(f"  IV range: [{df['implied_volatility'].min():.4f}, {df['implied_volatility'].max():.4f}]")

    return df


def cache_surface(df: pd.DataFrame, label: str = "latest") -> Path:
    """Save surface to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"spy_surface_{label}.csv"
    df.to_csv(path, index=False)
    print(f"Cached surface to {path}")
    return path


def load_cached_surface(label: str = "latest") -> pd.DataFrame:
    """Load cached surface."""
    path = CACHE_DIR / f"spy_surface_{label}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No cached surface at {path}. Run without --use-cache first.")
    df = pd.read_csv(path)
    print(f"Loaded cached surface from {path}: {len(df)} points")
    return df


def apply_dropout(df: pd.DataFrame, rate: float, seed: int) -> tuple:
    """Randomly drop strikes, return (train, holdout)."""
    rng = np.random.default_rng(seed)
    n = len(df)
    mask = rng.random(n) > rate
    # Ensure at least 5 points per slice in train
    for T_val in df["T"].unique():
        slice_mask = df["T"] == T_val
        slice_train = mask & slice_mask.values
        if slice_train.sum() < 5:
            # Keep all points in this slice
            mask[slice_mask.values] = True
    return df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)


def compute_rmse_bps(pred: np.ndarray, true: np.ndarray) -> float:
    """RMSE in basis points."""
    return np.sqrt(np.mean((pred - true) ** 2)) * 10000


def compute_surface_epp(predict_fn, S=100.0, r=0.05, n_strikes=40):
    """Compute EPP on a regular grid."""
    from src.volatility_surface.utils.epp import compute_butterfly_epp, compute_calendar_epp

    log_strikes = np.linspace(-0.15, 0.15, n_strikes)
    maturities = np.array([0.1, 0.25, 0.5, 1.0, 2.0])

    rows = []
    for T in maturities:
        for k in log_strikes:
            rows.append({"log_moneyness": k, "T": T, "implied_volatility": 0.0})
    grid = pd.DataFrame(rows)
    grid["implied_volatility"] = predict_fn(grid)

    max_epp = 0.0
    for T in maturities:
        mask = grid["T"] == T
        strikes = S * np.exp(grid.loc[mask, "log_moneyness"].values)
        ivs = grid.loc[mask, "implied_volatility"].values
        try:
            epp = compute_butterfly_epp(strikes, ivs, T, S, r)
            max_epp = max(max_epp, epp)
        except Exception:
            pass

    for k in log_strikes[::5]:
        mask = grid["log_moneyness"] == k
        if mask.sum() < 2:
            continue
        K = S * np.exp(k)
        Ts = grid.loc[mask, "T"].values
        ivs = grid.loc[mask, "implied_volatility"].values
        try:
            epp = compute_calendar_epp(K, Ts, ivs, S, r)
            max_epp = max(max_epp, epp)
        except Exception:
            pass

    return max_epp * 10000  # in bps


def run_benchmark(surface: pd.DataFrame, n_trials: int = 5) -> pd.DataFrame:
    """Run dropout benchmark on a real surface."""
    from src.volatility_surface.models.pinn_model import PINNVolatilityModel

    dropout_rates = [0.0, 0.2, 0.4, 0.6]
    results = []

    for dropout in dropout_rates:
        for trial in range(n_trials):
            seed = 42 + trial
            train_df, holdout_df = apply_dropout(surface, dropout, seed)

            if len(train_df) < 10 or len(holdout_df) == 0:
                continue

            for model_name, ModelClass, kwargs in [
                ("CINN", PINNVolatilityModel, dict(
                    hidden_layers=[64, 32, 16],
                    lambda_calendar=5.0,
                    lambda_butterfly=10.0,
                    lambda_wing=1.0,
                    epochs=300,
                    learning_rate=3e-3,
                )),
                ("MLP", PINNVolatilityModel, dict(
                    hidden_layers=[64, 32, 16],
                    lambda_calendar=0.0,
                    lambda_butterfly=0.0,
                    lambda_wing=0.0,
                    epochs=300,
                    learning_rate=3e-3,
                )),
            ]:
                try:
                    model = ModelClass(**kwargs)
                    t0 = time.time()
                    model.train(train_df)
                    cal_time = (time.time() - t0) * 1000

                    preds = model.predict_volatility(holdout_df)
                    true = holdout_df["implied_volatility"].values
                    rmse = compute_rmse_bps(preds, true)

                    def predict_fn(grid, m=model):
                        return m.predict_volatility(grid)
                    epp = compute_surface_epp(predict_fn)

                    results.append({
                        "model": model_name,
                        "dropout": dropout,
                        "trial": trial,
                        "rmse_bps": rmse,
                        "epp_bps": epp,
                        "time_ms": cal_time,
                    })
                except Exception as e:
                    print(f"  {model_name} failed at {dropout} dropout, trial {trial}: {e}")

        print(f"  Dropout {dropout:.0%} done")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cache", action="store_true", help="Use cached surface")
    parser.add_argument("--n-expiries", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("Real-Data Validation: SPY Options")
    print("=" * 60)

    if args.use_cache:
        surface = load_cached_surface()
    else:
        surface = fetch_spy_surface(args.n_expiries)
        cache_surface(surface)

    if len(surface) < 20:
        print("ERROR: Insufficient data points. Try later or check yfinance.")
        return

    print(f"\nRunning benchmark with {args.n_trials} trials per dropout level...")
    results = run_benchmark(surface, args.n_trials)

    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "spy_validation.csv", index=False)

    print("\n" + "=" * 60)
    print("SPY Real-Data Validation Results")
    print("=" * 60)
    summary = results.groupby(["model", "dropout"]).agg({
        "rmse_bps": ["mean", "std"],
        "epp_bps": ["mean", "max"],
        "time_ms": "mean",
    }).round(2)
    print(summary)


if __name__ == "__main__":
    main()
