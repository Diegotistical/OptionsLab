"""
Benchmark Monte Carlo pricer vs. Black-Scholes and Binomial Tree
with Delta/Gamma, CPU & GPU comparison.
"""

import time

import numpy as np
from scipy.stats import norm

from pricing_models.binomial_tree import BinomialTree
from pricing_models.monte_carlo_unified import MonteCarloPricer


# ------------------------------
# Black-Scholes Formula
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
mc_pricer_cpu = MonteCarloPricer(
    num_simulations=num_simulations, use_numba=True, use_gpu=False
)
mc_pricer_gpu = MonteCarloPricer(
    num_simulations=num_simulations, use_numba=True, use_gpu=True
)

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
start = time.perf_counter()
bt_price = bt_pricer.price(S, K, T, r, sigma, option_type, "european", q)
bt_time = (time.perf_counter() - start) * 1000
print(f"Binomial Tree   | Price: {bt_price:.4f} | Time: {bt_time:.2f} ms")

# ------------------------------
# Benchmark Monte Carlo CPU
# ------------------------------
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
    start = time.perf_counter()
    mc_price_gpu = mc_pricer_gpu.price(S, K, T, r, sigma, option_type, q)
    delta_gpu, gamma_gpu = mc_pricer_gpu.delta_gamma(S, K, T, r, sigma, option_type, q)
    mc_gpu_time = (time.perf_counter() - start) * 1000
    print(
        f"Monte Carlo GPU | Price: {mc_price_gpu:.4f} | Delta: {delta_gpu:.4f} | Gamma: {gamma_gpu:.6f} | Time: {mc_gpu_time:.2f} ms"
    )
else:
    mc_gpu_time = None
    print("GPU not available, skipping GPU benchmark")

# ------------------------------
# Optional ML Surrogate
# ------------------------------
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pricing_models.monte_carlo_unified import MLSurrogate

# Generate small training dataset
S_grid = np.linspace(80, 120, 10)
X_train = np.column_stack(
    [
        S_grid,
        np.full(10, K),
        np.full(10, T),
        np.full(10, r),
        np.full(10, sigma),
        np.full(10, q),
    ]
)
y_train = np.array(
    [
        [
            mc_pricer_cpu.price(s, K, T, r, sigma, option_type, q),
            mc_pricer_cpu.delta_gamma(s, K, T, r, sigma, option_type, q)[0],
            mc_pricer_cpu.delta_gamma(s, K, T, r, sigma, option_type, q)[1],
        ]
        for s in S_grid
    ]
)

ml_model = MLSurrogate(
    Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                MultiOutputRegressor(
                    GradientBoostingRegressor(
                        n_estimators=200, max_depth=5, random_state=42
                    )
                ),
            ),
        ]
    )
)
start = time.perf_counter()
ml_model.fit(X_train, y_train)
ml_fit_time = (time.perf_counter() - start) * 1000

X_test = np.array([[S, K, T, r, sigma, q]])
start = time.perf_counter()
ml_pred = ml_model.predict(X_test)
ml_predict_time = (time.perf_counter() - start) * 1000
print(
    f"ML Surrogate    | Price: {ml_pred[0,0]:.4f} | Delta: {ml_pred[0,1]:.4f} | Gamma: {ml_pred[0,2]:.6f} | Fit: {ml_fit_time:.2f} ms | Predict: {ml_predict_time:.2f} ms"
)
