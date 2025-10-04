# tests/test_benchmarks.py

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


import time

import numpy as np

from pricing_models.binomial_tree import BinomialTree
from pricing_models.black_scholes import black_scholes
from pricing_models.monte_carlo_unified import MonteCarloPricer


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
    mc = MonteCarloPricer(num_simulations=50000, num_steps=100, seed=42)

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

# Run this script to benchmark the option pricing models
# It will print the average execution time for each model
# python -m tests.test_benchmarks
# Ensure you have the necessary models implemented in the src/pricing_models directory
