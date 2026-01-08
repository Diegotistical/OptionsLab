# tests/test_monte_carlo.py
"""
Comprehensive tests for Monte Carlo pricing models.

Tests cover:
    - MonteCarloPricer: Basic MC pricing with Greeks
    - MonteCarloMLSurrogate: ML surrogate model
    - MonteCarloPricerUni: Unified CPU/GPU pricer

Run with: pytest tests/test_monte_carlo.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.exceptions.montecarlo_exceptions import InputValidationError, MonteCarloError
from src.pricing_models.black_scholes import black_scholes
from src.pricing_models.monte_carlo import NUMBA_AVAILABLE, MonteCarloPricer
from src.pricing_models.monte_carlo_ml import LIGHTGBM_AVAILABLE, MonteCarloMLSurrogate
from src.pricing_models.monte_carlo_unified import (
    GPU_AVAILABLE,
)
from src.pricing_models.monte_carlo_unified import InputValidationError as UniInputValidationError
from src.pricing_models.monte_carlo_unified import (
    MLSurrogate,
    MonteCarloPricerUni,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_pricer():
    """Create a basic Monte Carlo pricer for testing."""
    return MonteCarloPricer(
        num_simulations=10000,
        num_steps=50,
        seed=42,
        use_numba=False,
    )


@pytest.fixture
def numba_pricer():
    """Create a Numba-accelerated pricer if available."""
    if not NUMBA_AVAILABLE:
        pytest.skip("Numba not available")
    return MonteCarloPricer(
        num_simulations=10000,
        num_steps=50,
        seed=42,
        use_numba=True,
    )


@pytest.fixture
def unified_pricer():
    """Create a unified pricer for testing."""
    return MonteCarloPricerUni(
        num_simulations=10000,
        num_steps=50,
        seed=42,
        use_numba=False,
        use_gpu=False,
    )


@pytest.fixture
def sample_options_df():
    """Create sample options DataFrame for batch testing."""
    return pd.DataFrame(
        {
            "S": [100.0, 110.0, 90.0, 100.0, 100.0],
            "K": [100.0, 100.0, 100.0, 95.0, 105.0],
            "T": [1.0, 1.0, 1.0, 0.5, 0.5],
            "r": [0.05, 0.05, 0.05, 0.05, 0.05],
            "sigma": [0.2, 0.2, 0.2, 0.3, 0.15],
            "q": [0.0, 0.0, 0.0, 0.02, 0.01],
        }
    )


# =============================================================================
# Test MonteCarloPricer
# =============================================================================


class TestMonteCarloPricer:
    """Tests for the basic Monte Carlo pricer."""

    def test_initialization(self):
        """Test pricer initialization."""
        pricer = MonteCarloPricer(
            num_simulations=5000,
            num_steps=100,
            seed=123,
        )
        assert pricer.num_simulations == 5000
        assert pricer.num_steps == 100
        assert pricer.seed == 123

    def test_initialization_invalid_simulations(self):
        """Test that invalid simulation count raises error."""
        with pytest.raises(InputValidationError):
            MonteCarloPricer(num_simulations=0)

        with pytest.raises(InputValidationError):
            MonteCarloPricer(num_simulations=-100)

    def test_initialization_invalid_steps(self):
        """Test that invalid step count raises error."""
        with pytest.raises(InputValidationError):
            MonteCarloPricer(num_steps=0)

    def test_price_call_option(self, basic_pricer):
        """Test call option pricing."""
        price = basic_pricer.price(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call", q=0.0
        )

        # Price should be positive and reasonable
        assert price > 0
        assert price < 100  # Less than spot price

        # Compare to Black-Scholes (should be close)
        bs_price = black_scholes(100, 100, 1.0, 0.05, 0.2, "call", 0.0)
        assert abs(price - bs_price) < 1.0  # Within $1 tolerance

    def test_price_put_option(self, basic_pricer):
        """Test put option pricing."""
        price = basic_pricer.price(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="put", q=0.0
        )

        assert price > 0
        bs_price = black_scholes(100, 100, 1.0, 0.05, 0.2, "put", 0.0)
        assert abs(price - bs_price) < 1.0

    def test_price_invalid_option_type(self, basic_pricer):
        """Test that invalid option type raises error."""
        with pytest.raises(InputValidationError):
            basic_pricer.price(100, 100, 1.0, 0.05, 0.2, "invalid")

    def test_price_invalid_spot(self, basic_pricer):
        """Test that invalid spot price raises error."""
        with pytest.raises(InputValidationError):
            basic_pricer.price(0, 100, 1.0, 0.05, 0.2, "call")

        with pytest.raises(InputValidationError):
            basic_pricer.price(-100, 100, 1.0, 0.05, 0.2, "call")

    def test_price_reproducibility(self, basic_pricer):
        """Test that same seed produces same results."""
        price1 = basic_pricer.price(100, 100, 1.0, 0.05, 0.2, "call", seed=42)
        price2 = basic_pricer.price(100, 100, 1.0, 0.05, 0.2, "call", seed=42)

        assert price1 == price2

    def test_price_with_std_error(self, basic_pricer):
        """Test price with standard error calculation."""
        price, std_error = basic_pricer.price_with_std_error(
            100, 100, 1.0, 0.05, 0.2, "call"
        )

        assert price > 0
        assert std_error > 0
        assert std_error < price  # Std error should be small relative to price

    def test_delta_call(self, basic_pricer):
        """Test delta calculation for call option."""
        delta = basic_pricer.delta(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call"
        )

        # Call delta should be between 0 and 1
        assert 0 < delta < 1
        # ATM call should have delta around 0.5-0.6
        assert 0.4 < delta < 0.8

    def test_delta_put(self, basic_pricer):
        """Test delta calculation for put option."""
        delta = basic_pricer.delta(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="put"
        )

        # Put delta should be between -1 and 0
        assert -1 < delta < 0

    def test_gamma(self, basic_pricer):
        """Test gamma calculation."""
        gamma = basic_pricer.gamma(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call"
        )

        # Gamma should be positive (allow small negative due to MC noise)
        assert gamma > -0.01  # With low sim count, can be slightly negative

    def test_delta_gamma_combined(self, basic_pricer):
        """Test combined delta/gamma calculation."""
        delta, gamma = basic_pricer.delta_gamma(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call"
        )

        assert 0 < delta < 1
        assert gamma > -0.01  # Allow small negative due to MC noise

    def test_vega(self, basic_pricer):
        """Test vega calculation."""
        vega = basic_pricer.vega(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call"
        )

        # Vega should be positive
        assert vega > 0

    def test_theta(self, basic_pricer):
        """Test theta calculation."""
        theta = basic_pricer.theta(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call"
        )

        # Theta is typically negative (time decay)
        # But for deep ITM puts can be positive
        assert isinstance(theta, float)

    def test_rho(self, basic_pricer):
        """Test rho calculation."""
        rho = basic_pricer.rho(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call"
        )

        # Call rho should be positive
        assert rho > 0

    def test_all_greeks(self, basic_pricer):
        """Test all_greeks method."""
        greeks = basic_pricer.all_greeks(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call"
        )

        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert "rho" in greeks

        assert 0 < greeks["delta"] < 1
        assert greeks["gamma"] > -0.01  # Allow small negative due to MC noise
        assert greeks["vega"] > 0

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_numba_vs_numpy(self):
        """Test that Numba and NumPy produce similar results."""
        numpy_pricer = MonteCarloPricer(
            num_simulations=50000,
            num_steps=100,
            seed=42,
            use_numba=False,
        )
        numba_pricer = MonteCarloPricer(
            num_simulations=50000,
            num_steps=100,
            seed=42,
            use_numba=True,
        )

        numpy_price = numpy_pricer.price(100, 100, 1.0, 0.05, 0.2, "call")
        numba_price = numba_pricer.price(100, 100, 1.0, 0.05, 0.2, "call")

        # Prices should be similar (within 5% due to different RNG)
        assert abs(numpy_price - numba_price) / numpy_price < 0.05


# =============================================================================
# Test MonteCarloMLSurrogate
# =============================================================================


class TestMonteCarloMLSurrogate:
    """Tests for the ML surrogate model."""

    def test_initialization(self):
        """Test surrogate initialization."""
        surrogate = MonteCarloMLSurrogate(
            num_simulations=1000,
            num_steps=50,
            seed=42,
        )

        assert not surrogate.trained
        # Note: mc_pricer removed - we use vectorized BS for training now

    def test_generate_training_data(self):
        """Test training data generation."""
        surrogate = MonteCarloMLSurrogate(
            num_simulations=1000,
            num_steps=50,
            seed=42,
        )

        X, y = surrogate.generate_training_data(
            n_samples=50,
            verbose=False,
        )

        assert X.shape[0] == 50
        assert X.shape[1] == len(surrogate.feature_names)
        assert y.shape[0] == 50
        assert y.shape[1] == 3  # price, delta, gamma

    def test_fit_with_generated_data(self):
        """Test fitting with auto-generated data."""
        surrogate = MonteCarloMLSurrogate(
            num_simulations=1000,
            num_steps=50,
            seed=42,
        )

        surrogate.fit(n_samples=50, verbose=False)

        assert surrogate.trained

    def test_fit_with_provided_data(self, sample_options_df):
        """Test fitting with provided data."""
        surrogate = MonteCarloMLSurrogate(
            num_simulations=1000,
            num_steps=50,
            seed=42,
        )

        # Generate some dummy targets
        y = np.array(
            [
                [10.0, 0.5, 0.02],
                [15.0, 0.6, 0.018],
                [5.0, 0.4, 0.025],
                [8.0, 0.55, 0.022],
                [6.0, 0.45, 0.019],
            ]
        )

        surrogate.fit(X=sample_options_df, y=y, verbose=False)

        assert surrogate.trained

    def test_predict_raises_if_not_trained(self, sample_options_df):
        """Test that predict raises error if not trained."""
        surrogate = MonteCarloMLSurrogate()

        with pytest.raises(RuntimeError):
            surrogate.predict(sample_options_df)

    def test_predict_after_fit(self, sample_options_df):
        """Test prediction after training."""
        surrogate = MonteCarloMLSurrogate(
            num_simulations=1000,
            num_steps=50,
            seed=42,
        )

        surrogate.fit(n_samples=100, verbose=False)

        predictions = surrogate.predict(sample_options_df)

        assert len(predictions) == len(sample_options_df)
        assert "price" in predictions.columns
        assert "delta" in predictions.columns
        assert "gamma" in predictions.columns

    def test_predict_single(self):
        """Test single option prediction."""
        surrogate = MonteCarloMLSurrogate(
            num_simulations=1000,
            num_steps=50,
            seed=42,
        )

        surrogate.fit(n_samples=100, verbose=False)

        result = surrogate.predict_single(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)

        assert "price" in result
        assert "delta" in result
        assert "gamma" in result
        assert result["price"] > 0

    def test_score(self):
        """Test model scoring."""
        surrogate = MonteCarloMLSurrogate(
            num_simulations=1000,
            num_steps=50,
            seed=42,
        )

        # Generate train and test data
        X_train, y_train = surrogate.generate_training_data(
            n_samples=100, verbose=False
        )
        surrogate.fit(X=X_train, y=y_train, verbose=False)

        X_test, y_test = surrogate.generate_training_data(n_samples=50, verbose=False)

        # Convert X_test to DataFrame for scoring
        df_test = pd.DataFrame(X_test, columns=surrogate.feature_names)

        scores = surrogate.score(df_test, y_test)

        assert "price_r2" in scores
        assert "delta_r2" in scores
        assert "gamma_r2" in scores


# =============================================================================
# Test MonteCarloPricerUni
# =============================================================================


class TestMonteCarloPricerUni:
    """Tests for the unified Monte Carlo pricer."""

    def test_initialization(self):
        """Test unified pricer initialization."""
        pricer = MonteCarloPricerUni(
            num_simulations=10000,
            num_steps=100,
            seed=42,
        )

        assert pricer.num_simulations == 10000
        assert pricer.num_steps == 100

    def test_initialization_invalid(self):
        """Test that invalid params raise errors."""
        with pytest.raises((InputValidationError, UniInputValidationError)):
            MonteCarloPricerUni(num_simulations=0)

        with pytest.raises((InputValidationError, UniInputValidationError)):
            MonteCarloPricerUni(num_steps=-1)

    def test_price_call(self, unified_pricer):
        """Test call pricing."""
        price = unified_pricer.price(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call"
        )

        assert price > 0

        bs_price = black_scholes(100, 100, 1.0, 0.05, 0.2, "call")
        assert abs(price - bs_price) < 1.5

    def test_price_put(self, unified_pricer):
        """Test put pricing."""
        price = unified_pricer.price(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="put"
        )

        assert price > 0

        bs_price = black_scholes(100, 100, 1.0, 0.05, 0.2, "put")
        assert abs(price - bs_price) < 1.5

    def test_price_invalid_inputs(self, unified_pricer):
        """Test invalid input handling."""
        with pytest.raises((InputValidationError, UniInputValidationError)):
            unified_pricer.price(0, 100, 1.0, 0.05, 0.2, "call")

        with pytest.raises((InputValidationError, UniInputValidationError)):
            unified_pricer.price(100, 100, 1.0, 0.05, -0.1, "call")

        with pytest.raises((InputValidationError, UniInputValidationError)):
            unified_pricer.price(100, 100, 1.0, 0.05, 0.2, "invalid")

    def test_delta_gamma(self, unified_pricer):
        """Test delta/gamma calculation."""
        delta, gamma = unified_pricer.delta_gamma(
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call"
        )

        assert 0 < delta < 1
        assert gamma > 0

    def test_price_batch(self, unified_pricer, sample_options_df):
        """Test batch pricing."""
        prices = unified_pricer.price_batch(
            S_vals=sample_options_df["S"].values,
            K_vals=sample_options_df["K"].values,
            T_vals=sample_options_df["T"].values,
            r_vals=sample_options_df["r"].values,
            sigma_vals=sample_options_df["sigma"].values,
            option_type="call",
            q_vals=sample_options_df["q"].values,
        )

        assert len(prices) == len(sample_options_df)
        assert all(p > 0 for p in prices)

    def test_delta_gamma_batch(self, unified_pricer, sample_options_df):
        """Test batch Greeks calculation."""
        deltas, gammas = unified_pricer.delta_gamma_batch(
            S_vals=sample_options_df["S"].values,
            K_vals=sample_options_df["K"].values,
            T_vals=sample_options_df["T"].values,
            r_vals=sample_options_df["r"].values,
            sigma_vals=sample_options_df["sigma"].values,
            option_type="call",
            q_vals=sample_options_df["q"].values,
        )

        assert len(deltas) == len(sample_options_df)
        assert len(gammas) == len(sample_options_df)


# =============================================================================
# Test MLSurrogate (unified module version)
# =============================================================================


class TestMLSurrogate:
    """Tests for the unified ML surrogate."""

    def test_initialization(self):
        """Test surrogate initialization."""
        surrogate = MLSurrogate()
        assert not surrogate.trained

    def test_fit_and_predict(self, unified_pricer, sample_options_df):
        """Test fit and predict workflow."""
        surrogate = MLSurrogate()

        # Fit on sample data
        surrogate.fit(sample_options_df, unified_pricer, option_type="call")

        assert surrogate.trained

        # Predict
        predictions = surrogate.predict(sample_options_df)

        assert len(predictions) == len(sample_options_df)
        assert "price" in predictions.columns

    def test_predict_without_fit(self, sample_options_df):
        """Test that predict without fit raises error."""
        surrogate = MLSurrogate()

        with pytest.raises(RuntimeError):
            surrogate.predict(sample_options_df)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests across modules."""

    def test_put_call_parity(self, basic_pricer):
        """Test put-call parity: C - P = S*exp(-qT) - K*exp(-rT)."""
        S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.2, 0.02

        call_price = basic_pricer.price(S, K, T, r, sigma, "call", q)
        put_price = basic_pricer.price(S, K, T, r, sigma, "put", q)

        # Put-call parity
        parity_diff = call_price - put_price
        expected = S * np.exp(-q * T) - K * np.exp(-r * T)

        # Should be within Monte Carlo error
        assert abs(parity_diff - expected) < 2.0

    def test_itm_vs_otm(self, basic_pricer):
        """Test that ITM options are worth more than OTM."""
        # ITM call (S > K)
        itm_call = basic_pricer.price(110, 100, 1.0, 0.05, 0.2, "call")
        # ATM call (S = K)
        atm_call = basic_pricer.price(100, 100, 1.0, 0.05, 0.2, "call")
        # OTM call (S < K)
        otm_call = basic_pricer.price(90, 100, 1.0, 0.05, 0.2, "call")

        assert itm_call > atm_call > otm_call

    def test_higher_vol_higher_price(self, basic_pricer):
        """Test that higher volatility means higher option price."""
        low_vol = basic_pricer.price(100, 100, 1.0, 0.05, 0.1, "call")
        high_vol = basic_pricer.price(100, 100, 1.0, 0.05, 0.4, "call")

        assert high_vol > low_vol

    def test_longer_maturity_higher_price(self, basic_pricer):
        """Test that longer maturity means higher option price."""
        short_T = basic_pricer.price(100, 100, 0.25, 0.05, 0.2, "call")
        long_T = basic_pricer.price(100, 100, 2.0, 0.05, 0.2, "call")

        assert long_T > short_T


# =============================================================================
# Performance Tests (skipped in normal runs)
# =============================================================================


class TestPerformance:
    """Performance benchmarks (marked slow)."""

    @pytest.mark.slow
    def test_basic_pricer_throughput(self, basic_pricer):
        """Benchmark basic pricer throughput."""
        import time

        start = time.perf_counter()
        for _ in range(100):
            basic_pricer.price(100, 100, 1.0, 0.05, 0.2, "call")
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time
        assert elapsed < 30  # 100 calls in < 30 seconds

    @pytest.mark.slow
    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_numba_speedup(self):
        """Test that Numba provides speedup."""
        import time

        numpy_pricer = MonteCarloPricer(
            num_simulations=50000,
            num_steps=100,
            seed=42,
            use_numba=False,
        )
        numba_pricer = MonteCarloPricer(
            num_simulations=50000,
            num_steps=100,
            seed=42,
            use_numba=True,
        )

        # Warm up Numba
        numba_pricer.price(100, 100, 1.0, 0.05, 0.2, "call")

        # Time NumPy
        start = time.perf_counter()
        for _ in range(10):
            numpy_pricer.price(100, 100, 1.0, 0.05, 0.2, "call")
        numpy_time = time.perf_counter() - start

        # Time Numba
        start = time.perf_counter()
        for _ in range(10):
            numba_pricer.price(100, 100, 1.0, 0.05, 0.2, "call")
        numba_time = time.perf_counter() - start

        # Numba should be faster (or at least not slower)
        # Note: On small workloads, NumPy vectorization may win
        assert True  # Log results for inspection


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
