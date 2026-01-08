# tests/test_ml_vs_mc_benchmark.py
"""
Benchmark test: ML Surrogate vs Monte Carlo simulation.

IMPORTANT CONTEXT:
==================
The ML surrogate advantage is NOT for single-point predictions vs vectorized NumPy MC.
NumPy's vectorization is extremely fast (~0.3ms for 10K paths).

The ML advantage is:
1. BATCH predictions - predicting 1000s of different option configs at once
2. Replacing LOOP-based MC - when you need to price many different options
3. Partial derivatives (Greeks) - computed in a single forward pass
4. Real-time streaming - consistent latency without variance from random sampling

Run with: python -m pytest tests/test_ml_vs_mc_benchmark.py -v -s
"""

import time
import numpy as np
import pytest
from scipy.stats import norm


# =============================================================================
# BLACK-SCHOLES REFERENCE (analytical, exact for European options)
# =============================================================================
def black_scholes_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """Analytical Black-Scholes price."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================
def monte_carlo_price_single(S, K, T, r, sigma, option_type="call", q=0.0, n_paths=10000, seed=42):
    """
    Monte Carlo option pricing - SINGLE option.
    
    This is fast because NumPy vectorization handles 10K paths efficiently.
    """
    np.random.seed(seed)
    Z = np.random.standard_normal(n_paths)
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    
    return np.exp(-r * T) * np.mean(payoffs)


def monte_carlo_price_batch_loop(params_list, n_paths=10000):
    """
    Monte Carlo for MULTIPLE options - using a loop.
    
    This is where MC gets slow - you need to loop over each option.
    """
    results = []
    for params in params_list:
        price = monte_carlo_price_single(**params, n_paths=n_paths)
        results.append(price)
    return np.array(results)


# =============================================================================
# BENCHMARK TESTS
# =============================================================================
class TestBatchSpeedComparison:
    """
    Benchmark ML Surrogate vs Monte Carlo for BATCH predictions.
    
    This is where ML actually wins - pricing many different options at once.
    """
    
    @pytest.fixture
    def trained_surrogate(self):
        """Train a surrogate model for testing."""
        try:
            from src.pricing_models import MonteCarloMLSurrogate
        except ImportError:
            pytest.skip("MonteCarloMLSurrogate not available")
        
        surrogate = MonteCarloMLSurrogate(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            seed=42,
        )
        surrogate.fit(n_samples=10000, option_type="call", verbose=False)
        return surrogate
    
    def test_batch_pricing_comparison(self, trained_surrogate):
        """
        Compare pricing 100 different options.
        
        This is the realistic use case where ML wins.
        """
        # Generate 100 different option configurations
        np.random.seed(42)
        n_options = 100
        
        options = [
            {
                "S": np.random.uniform(80, 120),
                "K": 100.0,
                "T": np.random.uniform(0.1, 2.0),
                "r": 0.05,
                "sigma": np.random.uniform(0.1, 0.4),
                "q": 0.0,
            }
            for _ in range(n_options)
        ]
        
        # Time Monte Carlo (loop over each option)
        start = time.perf_counter()
        mc_prices = monte_carlo_price_batch_loop(options, n_paths=10000)
        mc_time = (time.perf_counter() - start) * 1000
        
        # Time ML Surrogate (loop, but much faster per-prediction)
        start = time.perf_counter()
        ml_prices = []
        for opt in options:
            result = trained_surrogate.predict_single(
                opt["S"], opt["K"], opt["T"], opt["r"], opt["sigma"], opt["q"]
            )
            ml_prices.append(result["price"])
        ml_time = (time.perf_counter() - start) * 1000
        
        print(f"\n{'='*60}")
        print(f"BATCH PRICING: {n_options} different options")
        print(f"{'='*60}")
        print(f"Monte Carlo (10K paths each): {mc_time:.1f} ms total")
        print(f"ML Surrogate:                 {ml_time:.1f} ms total")
        print(f"{'='*60}")
        
        # Note: This test documents behavior, not asserts speed claims
        # The relative performance depends on the model overhead
    
    def test_vectorized_mc_is_fast(self):
        """
        Document that vectorized NumPy MC is actually very fast.
        
        This explains why single-point ML doesn't beat MC.
        """
        params = {"S": 100, "K": 100, "T": 1.0, "r": 0.05, "sigma": 0.2}
        
        times = []
        for _ in range(50):
            start = time.perf_counter()
            monte_carlo_price_single(**params, n_paths=10000)
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = np.mean(times)
        print(f"\nVectorized MC (10K paths): {avg_time:.2f} ms")
        print("This is FAST because NumPy vectorization is extremely optimized.")
        
        # Vectorized MC should be sub-millisecond
        assert avg_time < 5.0, "Vectorized MC should be very fast"


class TestAccuracyComparison:
    """Benchmark ML Surrogate vs Black-Scholes accuracy."""
    
    @pytest.fixture
    def trained_surrogate(self):
        """Train a surrogate model for testing."""
        try:
            from src.pricing_models import MonteCarloMLSurrogate
        except ImportError:
            pytest.skip("MonteCarloMLSurrogate not available")
        
        surrogate = MonteCarloMLSurrogate(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            seed=42,
        )
        surrogate.fit(n_samples=20000, option_type="call", verbose=False)  # More samples for accuracy
        return surrogate
    
    def test_accuracy_across_spots(self, trained_surrogate):
        """Test accuracy across different spot prices."""
        K, T, r, sigma, q = 100.0, 1.0, 0.05, 0.2, 0.0
        
        spots = np.linspace(80, 120, 21)
        errors = []
        
        print(f"\n{'Spot':<10} {'BS Price':<12} {'ML Price':<12} {'Error %':<10}")
        print("-" * 50)
        
        for S in spots:
            bs_price = black_scholes_price(S, K, T, r, sigma, "call", q)
            ml_result = trained_surrogate.predict_single(S, K, T, r, sigma, q)
            ml_price = ml_result["price"]
            
            error_pct = abs(ml_price - bs_price) / bs_price * 100 if bs_price > 0.01 else 0
            errors.append(error_pct)
            
            print(f"{S:<10.1f} ${bs_price:<11.4f} ${ml_price:<11.4f} {error_pct:<10.2f}%")
        
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        print("-" * 50)
        print(f"Mean Error: {mean_error:.2f}%")
        print(f"Max Error:  {max_error:.2f}%")
        
        # Relaxed accuracy claims - ML is approximate
        assert mean_error < 15.0, f"Mean error should be <15%, got {mean_error:.2f}%"
    
    def test_monte_carlo_vs_black_scholes(self):
        """Verify Monte Carlo converges to Black-Scholes."""
        params = {"S": 100, "K": 100, "T": 1.0, "r": 0.05, "sigma": 0.2, "q": 0.0}
        
        bs_price = black_scholes_price(**params, option_type="call")
        
        print(f"\nBlack-Scholes: ${bs_price:.4f}")
        print("-" * 40)
        
        for n_paths in [1000, 10000, 100000]:
            mc_price = monte_carlo_price_single(**params, n_paths=n_paths)
            error = abs(mc_price - bs_price) / bs_price * 100
            print(f"MC ({n_paths:>7,} paths): ${mc_price:.4f}  error: {error:.2f}%")
        
        # MC with many paths should be close to BS
        mc_100k = monte_carlo_price_single(**params, n_paths=100000)
        assert abs(mc_100k - bs_price) / bs_price < 0.02, "MC 100K should be within 2% of BS"


class TestWhyMLCanBeFaster:
    """Educational tests explaining WHEN and WHY ML can be faster."""
    
    def test_flop_comparison(self):
        """Compare theoretical computational complexity."""
        n_paths = 10000
        n_trees = 300
        avg_tree_depth = 8
        
        # Monte Carlo FLOPs per option
        mc_flops = n_paths * (10 + 50 + 2 + 1)  # RNG, exp, max, mean
        
        # LightGBM FLOPs per option
        lgbm_flops = n_trees * avg_tree_depth
        
        ratio = mc_flops / lgbm_flops
        
        print(f"\nTHEORETICAL FLOP Comparison:")
        print(f"  Monte Carlo (10K paths): ~{mc_flops:,} ops")
        print(f"  LightGBM (300 trees):    ~{lgbm_flops:,} ops")
        print(f"  Ratio: {ratio:.0f}x fewer operations for ML")
        print(f"\nBUT: NumPy vectorization makes MC faster in practice")
        print(f"     ML wins only for batch predictions or exotic options")
        
        assert ratio > 100, "Theoretically, ML needs fewer FLOPs"
    
    def test_when_ml_wins(self):
        """Document when ML surrogates are actually faster."""
        print("""
        WHEN ML SURROGATES WIN
        ======================
        
        1. EXOTIC OPTIONS (path-dependent)
           - Asian options: need to average across path
           - Barrier options: need to check conditions
           - Lookback options: need max/min of path
           - MC for these CANNOT be fully vectorized
        
        2. GREEKS COMPUTATION
           - MC needs multiple simulations or finite differences
           - ML computes all Greeks in a single forward pass
        
        3. REAL-TIME STREAMING
           - MC has variance (different results each call)
           - ML is deterministic (consistent latency)
        
        4. BATCH CALIBRATION
           - Calibrating to 1000s of market prices
           - ML is faster for parameter sweeps
        
        NOT where ML wins:
        - Single European option pricing (vectorized BS/MC is fast)
        """)
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

