# tests/test_benchmarks.py

import os
import sys
import time

import numpy as np

# Add project root to path so we can import 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pricing_models.binomial_tree import BinomialTree
from src.pricing_models.black_scholes import black_scholes
from src.pricing_models.monte_carlo_unified import MonteCarloPricerUni


def benchmark_function(func, args, num_runs=10):
    """Run pricing function multiple times and report avg execution time"""
    prices = []
    start_time = time.perf_counter()
    for _ in range(num_runs):
        price = func(**args)
        prices.append(price)
    elapsed = time.perf_counter() - start_time
    avg_time = elapsed / num_runs
    return np.mean(prices), avg_time


def benchmark_model(model, S, K, T, r, sigma, option_type, num_runs=10):
    """Wrapper for models that are class-based"""
    prices = []
    start_time = time.perf_counter()
    for _ in range(num_runs):
        # Note: Updated to match signature of Unified/Binomial pricers
        # BinomialTree.price takes 'exercise_style', MonteCarlo takes 'q'
        # We'll use a try-except or keyword unpacking to handle variations if needed,
        # but here we'll pass standard arguments.
        price = model.price(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)
        prices.append(price)
    elapsed = time.perf_counter() - start_time
    avg_time = elapsed / num_runs
    return np.mean(prices), avg_time


def main():
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    option_type = "call"
    num_runs = 100

    # Arguments for the Black-Scholes function
    bs_args = dict(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)

    # Initialize class-based models
    bt = BinomialTree(num_steps=500)
    # Using MonteCarloPricerUni from the unified module
    mc = MonteCarloPricerUni(num_simulations=50000, num_steps=100, seed=42)

    print("Benchmarking option pricing models...\n")

    # Black-Scholes
    price, avg_time = benchmark_function(black_scholes, bs_args, num_runs)
    print(
        f"{'Black-Scholes':15} | Price: {price:.4f} | Avg Time per Run: {avg_time*1000:.2f} ms"
    )

    # Binomial Tree
    price, avg_time = benchmark_model(bt, S, K, T, r, sigma, option_type, num_runs)
    print(
        f"{'Binomial Tree':15} | Price: {price:.4f} | Avg Time per Run: {avg_time*1000:.2f} ms"
    )

    # Monte Carlo
    price, avg_time = benchmark_model(mc, S, K, T, r, sigma, option_type, num_runs)
    print(
        f"{'Monte Carlo':15} | Price: {price:.4f} | Avg Time per Run: {avg_time*1000:.2f} ms"
    )


if __name__ == "__main__":
    main()
