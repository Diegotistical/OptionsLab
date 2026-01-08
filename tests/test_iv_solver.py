# tests/test_iv_solver.py
"""
Tests for the Implied Volatility Solver module.
"""

import numpy as np
import pytest
from scipy.stats import norm


# =============================================================================
# FIXTURES
# =============================================================================
@pytest.fixture
def iv_solver():
    """Import IV solver module."""
    from src.pricing_models.iv_solver import (
        black_scholes_price,
        black_scholes_vega,
        implied_volatility,
        implied_volatility_vectorized,
        iv_surface_from_prices,
    )

    return {
        "implied_volatility": implied_volatility,
        "implied_volatility_vectorized": implied_volatility_vectorized,
        "black_scholes_price": black_scholes_price,
        "black_scholes_vega": black_scholes_vega,
        "iv_surface_from_prices": iv_surface_from_prices,
    }


# =============================================================================
# NEWTON-RAPHSON CONVERGENCE TESTS
# =============================================================================
class TestNewtonRaphsonConvergence:
    """Test Newton-Raphson solver convergence."""

    def test_atm_call_iv(self, iv_solver):
        """Test IV for at-the-money call."""
        true_sigma = 0.20
        S, K, T, r, q = 100, 100, 1.0, 0.05, 0.0

        price = iv_solver["black_scholes_price"](S, K, T, r, true_sigma, "call", q)
        computed_iv = iv_solver["implied_volatility"](price, S, K, T, r, "call", q)

        assert abs(computed_iv - true_sigma) < 1e-6

    def test_atm_put_iv(self, iv_solver):
        """Test IV for at-the-money put."""
        true_sigma = 0.25
        S, K, T, r, q = 100, 100, 1.0, 0.05, 0.0

        price = iv_solver["black_scholes_price"](S, K, T, r, true_sigma, "put", q)
        computed_iv = iv_solver["implied_volatility"](price, S, K, T, r, "put", q)

        assert abs(computed_iv - true_sigma) < 1e-6

    def test_itm_call_iv(self, iv_solver):
        """Test IV for in-the-money call."""
        true_sigma = 0.30
        S, K, T, r, q = 120, 100, 1.0, 0.05, 0.0

        price = iv_solver["black_scholes_price"](S, K, T, r, true_sigma, "call", q)
        computed_iv = iv_solver["implied_volatility"](price, S, K, T, r, "call", q)

        assert abs(computed_iv - true_sigma) < 1e-5

    def test_otm_put_iv(self, iv_solver):
        """Test IV for out-of-the-money put."""
        true_sigma = 0.35
        S, K, T, r, q = 120, 100, 1.0, 0.05, 0.0

        price = iv_solver["black_scholes_price"](S, K, T, r, true_sigma, "put", q)
        computed_iv = iv_solver["implied_volatility"](price, S, K, T, r, "put", q)

        assert abs(computed_iv - true_sigma) < 1e-5


# =============================================================================
# EDGE CASE TESTS
# =============================================================================
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_short_maturity(self, iv_solver):
        """Test IV for short time to maturity."""
        true_sigma = 0.20
        S, K, T, r, q = 100, 100, 0.01, 0.05, 0.0  # ~4 days

        price = iv_solver["black_scholes_price"](S, K, T, r, true_sigma, "call", q)
        computed_iv = iv_solver["implied_volatility"](price, S, K, T, r, "call", q)

        assert abs(computed_iv - true_sigma) < 1e-4

    def test_deep_itm_call(self, iv_solver):
        """Test IV for deep in-the-money call."""
        true_sigma = 0.20
        S, K, T, r, q = 150, 100, 1.0, 0.05, 0.0

        price = iv_solver["black_scholes_price"](S, K, T, r, true_sigma, "call", q)
        computed_iv = iv_solver["implied_volatility"](price, S, K, T, r, "call", q)

        # Deep ITM is harder, allow slightly more tolerance
        assert abs(computed_iv - true_sigma) < 1e-3

    def test_deep_otm_put(self, iv_solver):
        """Test IV for deep out-of-the-money put."""
        true_sigma = 0.40
        S, K, T, r, q = 150, 100, 1.0, 0.05, 0.0

        price = iv_solver["black_scholes_price"](S, K, T, r, true_sigma, "put", q)
        if price > 0.01:  # Only test if price is meaningful
            computed_iv = iv_solver["implied_volatility"](price, S, K, T, r, "put", q)
            assert abs(computed_iv - true_sigma) < 1e-3

    def test_high_volatility(self, iv_solver):
        """Test IV for high volatility option."""
        true_sigma = 1.0  # 100% volatility
        S, K, T, r, q = 100, 100, 1.0, 0.05, 0.0

        price = iv_solver["black_scholes_price"](S, K, T, r, true_sigma, "call", q)
        computed_iv = iv_solver["implied_volatility"](price, S, K, T, r, "call", q)

        assert abs(computed_iv - true_sigma) < 1e-4

    def test_with_dividends(self, iv_solver):
        """Test IV with dividend yield."""
        true_sigma = 0.25
        S, K, T, r, q = 100, 100, 1.0, 0.05, 0.02

        price = iv_solver["black_scholes_price"](S, K, T, r, true_sigma, "call", q)
        computed_iv = iv_solver["implied_volatility"](price, S, K, T, r, "call", q)

        assert abs(computed_iv - true_sigma) < 1e-5


# =============================================================================
# VECTORIZED TESTS
# =============================================================================
class TestVectorized:
    """Test vectorized IV computation."""

    def test_batch_iv_computation(self, iv_solver):
        """Test batch IV computation."""
        true_sigmas = np.array([0.15, 0.20, 0.25, 0.30, 0.35])
        S, K, T, r, q = 100, 100, 1.0, 0.05, 0.0

        prices = np.array(
            [
                iv_solver["black_scholes_price"](S, K, T, r, sig, "call", q)
                for sig in true_sigmas
            ]
        )

        computed_ivs = iv_solver["implied_volatility_vectorized"](
            prices, S, K, T, r, "call", q
        )

        np.testing.assert_allclose(computed_ivs, true_sigmas, rtol=1e-4)

    def test_mixed_options(self, iv_solver):
        """Test batch with mixed calls and puts."""
        S, K, T, r, q = 100, 100, 1.0, 0.05, 0.0
        true_sigma = 0.20

        call_price = iv_solver["black_scholes_price"](S, K, T, r, true_sigma, "call", q)
        put_price = iv_solver["black_scholes_price"](S, K, T, r, true_sigma, "put", q)

        prices = np.array([call_price, put_price])
        types = ["call", "put"]

        ivs = iv_solver["implied_volatility_vectorized"](prices, S, K, T, r, types, q)

        np.testing.assert_allclose(ivs, [true_sigma, true_sigma], rtol=1e-4)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================
class TestErrorHandling:
    """Test error handling."""

    def test_negative_price_raises(self, iv_solver):
        """Test that negative price raises error."""
        with pytest.raises(ValueError, match="positive"):
            iv_solver["implied_volatility"](-1.0, 100, 100, 1.0, 0.05, "call")

    def test_zero_spot_raises(self, iv_solver):
        """Test that zero spot raises error."""
        with pytest.raises(ValueError, match="positive"):
            iv_solver["implied_volatility"](10.0, 0, 100, 1.0, 0.05, "call")

    def test_arbitrage_violation_raises(self, iv_solver):
        """Test that price below intrinsic raises error."""
        # Call with S=150, K=100 has intrinsic ~50
        with pytest.raises(ValueError, match="intrinsic"):
            iv_solver["implied_volatility"](1.0, 150, 100, 1.0, 0.05, "call")


# =============================================================================
# IV SURFACE TESTS
# =============================================================================
class TestIVSurface:
    """Test IV surface construction."""

    def test_surface_from_prices(self, iv_solver):
        """Test building IV surface from option prices."""
        S = 100
        strikes = np.array([90, 95, 100, 105, 110])
        maturities = np.array([0.25, 0.5, 1.0])
        true_sigma = 0.20
        r, q = 0.05, 0.0

        # Generate prices
        call_prices = np.zeros((len(strikes), len(maturities)))
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                call_prices[i, j] = iv_solver["black_scholes_price"](
                    S, K, T, r, true_sigma, "call", q
                )

        option_data = {
            "spot": S,
            "strikes": strikes,
            "maturities": maturities,
            "call_prices": call_prices,
        }

        surface = iv_solver["iv_surface_from_prices"](option_data, r, q)

        # All IVs should be close to true_sigma
        np.testing.assert_allclose(surface["call_iv"], true_sigma, rtol=1e-3)
