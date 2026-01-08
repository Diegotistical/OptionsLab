# tests/test_benchmarks2.py

"""
Benchmark Monte Carlo pricer vs. Black-Scholes and Binomial Tree
with Delta/Gamma, CPU & GPU comparison.
"""

import os
import sys
import time

import numpy as np
from scipy.stats import norm

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.pricing_models.binomial_tree import BinomialTree
from src.pricing_models.monte_carlo_unified import MLSurrogate, MonteCarloPricerUni


# ------------------------------
# Black-Scholes Formula (Local Helper)
# ------------------------------
def black_scholes_price(S, K, T, r, sigma, option_type="call", q=0.0):
    if T <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# ------------------------------
# Parameters
# ------------------------------
S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0
option_type = "call"
num_steps = 500
num_simulations = 100_000

# ------------------------------
# Initialize pricers
# ------------------------------
bt_pricer = BinomialTree(num_steps=num_steps)
mc_pricer_cpu = MonteCarloPricerUni(
    num_simulations=num_simulations, use_numba=True, use_gpu=False
)
mc_pricer_gpu = MonteCarloPricerUni(
    num_simulations=num_simulations, use_numba=True, use_gpu=True
)

print(f"\n{'='*60}")
print(f"BENCHMARKING (N={num_simulations} sims, Steps={num_steps})")
print(f"{'='*60}")

# ------------------------------
# Benchmark Black-Scholes
# ------------------------------
start = time.perf_counter()
bs_price = black_scholes_price(S, K, T, r, sigma, option_type, q)
bs_time = (time.perf_counter() - start) * 1000
print(f"Black-Scholes   | Price: {bs_price:.4f} | Time: {bs_time:.2f} ms")

# ------------------------------
# Benchmark Binomial Tree
# ------------------------------
# WARMUP: Run once to trigger Numba compilation
_ = bt_pricer.price(S, K, T, r, sigma, option_type, "european", q)

start = time.perf_counter()
bt_price = bt_pricer.price(S, K, T, r, sigma, option_type, "european", q)
bt_time = (time.perf_counter() - start) * 1000
print(f"Binomial Tree   | Price: {bt_price:.4f} | Time: {bt_time:.2f} ms")

# ------------------------------
# Benchmark Monte Carlo CPU
# ------------------------------
# WARMUP: Run once to trigger Numba compilation
_ = mc_pricer_cpu.price(S, K, T, r, sigma, option_type, q)

start = time.perf_counter()
mc_price_cpu = mc_pricer_cpu.price(S, K, T, r, sigma, option_type, q)
delta_cpu, gamma_cpu = mc_pricer_cpu.delta_gamma(S, K, T, r, sigma, option_type, q)
mc_cpu_time = (time.perf_counter() - start) * 1000
print(
    f"Monte Carlo CPU | Price: {mc_price_cpu:.4f} | Delta: {delta_cpu:.4f} | Gamma: {gamma_cpu:.6f} | Time: {mc_cpu_time:.2f} ms"
)

# ------------------------------
# Benchmark Monte Carlo GPU
# ------------------------------
if mc_pricer_gpu.use_gpu:
    try:
        # WARMUP
        _ = mc_pricer_gpu.price(S, K, T, r, sigma, option_type, q)

        start = time.perf_counter()
        mc_price_gpu = mc_pricer_gpu.price(S, K, T, r, sigma, option_type, q)
        delta_gpu, gamma_gpu = mc_pricer_gpu.delta_gamma(
            S, K, T, r, sigma, option_type, q
        )
        mc_gpu_time = (time.perf_counter() - start) * 1000
        print(
            f"Monte Carlo GPU | Price: {mc_price_gpu:.4f} | Delta: {delta_gpu:.4f} | Gamma: {gamma_gpu:.6f} | Time: {mc_gpu_time:.2f} ms"
        )
    except Exception as e:
        mc_gpu_time = None
        print(f"GPU benchmark skipped: {e}")
else:
    mc_gpu_time = None
    print("GPU not available, skipping GPU benchmark")

# ------------------------------
# Benchmark ML Surrogate
# ------------------------------
# FIX: Increased training data from 10 to 50 points to capture Gamma (curvature)
import pandas as pd

S_grid = np.linspace(80, 120, 50)
df_train = pd.DataFrame(
    {
        "S": S_grid,
        "K": np.full(len(S_grid), K),
        "T": np.full(len(S_grid), T),
        "r": np.full(len(S_grid), r),
        "sigma": np.full(len(S_grid), sigma),
        "q": np.full(len(S_grid), q),
    }
)

ml_model = MLSurrogate()
start = time.perf_counter()
ml_model.fit(df_train, mc_pricer_cpu)
ml_fit_time = (time.perf_counter() - start) * 1000

df_test = pd.DataFrame(
    [[S, K, T, r, sigma, q]], columns=["S", "K", "T", "r", "sigma", "q"]
)
start = time.perf_counter()
ml_pred = ml_model.predict(df_test)
ml_predict_time = (time.perf_counter() - start) * 1000

# Extract values from DataFrame
pred_price = ml_pred["price"].iloc[0]
pred_delta = ml_pred["delta"].iloc[0]
pred_gamma = ml_pred["gamma"].iloc[0]

print(
    f"ML Surrogate    | Price: {pred_price:.4f} | Delta: {pred_delta:.4f} | Gamma: {pred_gamma:.6f} | Fit: {ml_fit_time:.2f} ms | Predict: {ml_predict_time:.2f} ms"
)
print(f"{'='*60}\n")
