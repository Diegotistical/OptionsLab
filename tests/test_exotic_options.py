# tests/test_exotic_options.py
"""
Tests for the Exotic Options module.
"""

import numpy as np
import pytest
from scipy.stats import norm


# =============================================================================
# FIXTURES
# =============================================================================
@pytest.fixture
def exotic_options():
    """Import exotic options module."""
    from src.pricing_models.exotic_options import (
        AmericanOption,
        AsianOption,
        BarrierOption,
        price_american,
        price_asian,
        price_barrier,
    )

    return {
        "AsianOption": AsianOption,
        "BarrierOption": BarrierOption,
        "AmericanOption": AmericanOption,
        "price_asian": price_asian,
        "price_barrier": price_barrier,
        "price_american": price_american,
    }


def black_scholes_call(S, K, T, r, sigma, q=0.0):
    """Reference Black-Scholes call price."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma, q=0.0):
    """Reference Black-Scholes put price."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# =============================================================================
# ASIAN OPTION TESTS
# =============================================================================
class TestAsianOptions:
    """Test Asian option pricing."""

    def test_asian_call_positive_price(self, exotic_options):
        """Test Asian call returns positive price."""
        asian = exotic_options["AsianOption"](
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, seed=42
        )
        price = asian.price(n_paths=10000, avg_type="arithmetic", option_type="call")

        assert price > 0

    def test_asian_put_positive_price(self, exotic_options):
        """Test Asian put returns positive price."""
        asian = exotic_options["AsianOption"](
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, seed=42
        )
        price = asian.price(n_paths=10000, avg_type="arithmetic", option_type="put")

        assert price > 0

    def test_asian_cheaper_than_european(self, exotic_options):
        """Asian call should be cheaper than European call (lower variance)."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

        asian = exotic_options["AsianOption"](S=S, K=K, T=T, r=r, sigma=sigma, seed=42)
        asian_price = asian.price(
            n_paths=50000, avg_type="arithmetic", option_type="call"
        )

        european_price = black_scholes_call(S, K, T, r, sigma)

        # Asian call should be cheaper (averaging reduces value)
        assert asian_price < european_price

    def test_geometric_closed_form_matches_mc(self, exotic_options):
        """Geometric Asian closed-form should match Monte Carlo."""
        asian = exotic_options["AsianOption"](
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, seed=42
        )

        mc_price = asian.price(
            n_paths=100000, n_steps=252, avg_type="geometric", option_type="call"
        )
        closed_form = asian.price_geometric_closed_form(option_type="call")

        # Should match within Monte Carlo error
        assert abs(mc_price - closed_form) / closed_form < 0.05

    def test_asian_reproducibility(self, exotic_options):
        """Same seed should give same price."""
        asian1 = exotic_options["AsianOption"](
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, seed=42
        )
        asian2 = exotic_options["AsianOption"](
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, seed=42
        )

        price1 = asian1.price(n_paths=1000)
        price2 = asian2.price(n_paths=1000)

        assert price1 == price2


# =============================================================================
# BARRIER OPTION TESTS
# =============================================================================
class TestBarrierOptions:
    """Test Barrier option pricing."""

    def test_up_and_out_call_positive(self, exotic_options):
        """Up-and-out call should have positive price."""
        barrier = exotic_options["BarrierOption"](
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, barrier=120, seed=42
        )
        price = barrier.price(
            n_paths=10000, barrier_type="up-and-out", option_type="call"
        )

        assert price >= 0

    def test_down_and_out_put_positive(self, exotic_options):
        """Down-and-out put should have positive price."""
        barrier = exotic_options["BarrierOption"](
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, barrier=80, seed=42
        )
        price = barrier.price(
            n_paths=10000, barrier_type="down-and-out", option_type="put"
        )

        assert price >= 0

    def test_knock_out_cheaper_than_european(self, exotic_options):
        """Knock-out option should be cheaper than European."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

        # Up-and-out call
        barrier = exotic_options["BarrierOption"](
            S=S, K=K, T=T, r=r, sigma=sigma, barrier=130, seed=42
        )
        barrier_price = barrier.price(
            n_paths=50000, barrier_type="up-and-out", option_type="call"
        )

        european_price = black_scholes_call(S, K, T, r, sigma)

        # Knock-out must be cheaper (can become worthless)
        assert barrier_price < european_price

    def test_knock_in_plus_knock_out_equals_european(self, exotic_options):
        """Knock-in + Knock-out should approximately equal European."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

        # Up-and-out + Up-and-in = European call
        barrier_out = exotic_options["BarrierOption"](
            S=S, K=K, T=T, r=r, sigma=sigma, barrier=120, seed=42
        )
        barrier_in = exotic_options["BarrierOption"](
            S=S, K=K, T=T, r=r, sigma=sigma, barrier=120, seed=42
        )

        out_price = barrier_out.price(
            n_paths=100000, barrier_type="up-and-out", option_type="call"
        )
        in_price = barrier_in.price(
            n_paths=100000, barrier_type="up-and-in", option_type="call"
        )

        european_price = black_scholes_call(S, K, T, r, sigma)

        # Sum should be close to European
        combined = out_price + in_price
        assert abs(combined - european_price) / european_price < 0.1

    def test_invalid_barrier_raises(self, exotic_options):
        """Zero or negative barrier should raise error."""
        barrier = exotic_options["BarrierOption"](
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, barrier=0
        )
        with pytest.raises(ValueError, match="positive"):
            barrier.price(barrier_type="up-and-out")


# =============================================================================
# AMERICAN OPTION TESTS
# =============================================================================
class TestAmericanOptions:
    """Test American option pricing."""

    def test_american_put_positive(self, exotic_options):
        """American put should have positive price."""
        american = exotic_options["AmericanOption"](
            S=100, K=100, T=1.0, r=0.05, sigma=0.2, seed=42
        )
        price = american.price(n_paths=10000, option_type="put")

        assert price > 0

    def test_american_put_greater_than_european(self, exotic_options):
        """American put should be more valuable than European put."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

        american = exotic_options["AmericanOption"](
            S=S, K=K, T=T, r=r, sigma=sigma, seed=42
        )
        american_price = american.price(n_paths=50000, option_type="put")

        european_price = black_scholes_put(S, K, T, r, sigma)

        # American put >= European put (early exercise value)
        assert american_price >= european_price * 0.95  # Allow some MC error

    def test_american_call_equals_european_no_dividend(self, exotic_options):
        """American call should equal European call with no dividends."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

        american = exotic_options["AmericanOption"](
            S=S, K=K, T=T, r=r, sigma=sigma, q=0.0, seed=42
        )
        american_price = american.price(n_paths=50000, option_type="call")

        european_price = black_scholes_call(S, K, T, r, sigma)

        # Should be very close (no early exercise for non-dividend call)
        assert abs(american_price - european_price) / european_price < 0.1

    def test_itm_put_early_exercise_premium(self, exotic_options):
        """Deep ITM put should have significant early exercise premium."""
        S, K, T, r, sigma = 70, 100, 1.0, 0.05, 0.2  # Deep ITM put

        american = exotic_options["AmericanOption"](
            S=S, K=K, T=T, r=r, sigma=sigma, seed=42
        )
        american_price = american.price(n_paths=50000, option_type="put")

        european_price = black_scholes_put(S, K, T, r, sigma)

        # American should be noticeably more valuable
        assert american_price > european_price


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================
class TestConvenienceFunctions:
    """Test convenience pricing functions."""

    def test_price_asian(self, exotic_options):
        """Test price_asian convenience function."""
        price = exotic_options["price_asian"](
            S=100,
            K=100,
            T=1.0,
            r=0.05,
            sigma=0.2,
            avg_type="arithmetic",
            n_paths=10000,
            seed=42,
        )
        assert price > 0

    def test_price_barrier(self, exotic_options):
        """Test price_barrier convenience function."""
        price = exotic_options["price_barrier"](
            S=100,
            K=100,
            T=1.0,
            r=0.05,
            sigma=0.2,
            barrier=120,
            barrier_type="up-and-out",
            n_paths=10000,
            seed=42,
        )
        assert price >= 0

    def test_price_american(self, exotic_options):
        """Test price_american convenience function."""
        price = exotic_options["price_american"](
            S=100,
            K=100,
            T=1.0,
            r=0.05,
            sigma=0.2,
            option_type="put",
            n_paths=10000,
            seed=42,
        )
        assert price > 0
