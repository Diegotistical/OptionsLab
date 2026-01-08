import numpy as np
import pytest

from src.pricing_models.black_scholes import black_scholes


@pytest.mark.parametrize(
    "S, K, T, r, sigma",
    [
        (100, 100, 1.0, 0.05, 0.2),  # ATM
        (110, 100, 0.5, 0.02, 0.3),  # ITM Call
        (90, 100, 2.0, 0.05, 0.15),  # OTM Call
    ],
)
def test_put_call_parity(S, K, T, r, sigma):
    """
    Verifies C - P = S - K * exp(-rT)
    """
    call_price = black_scholes(S, K, T, r, sigma, option_type="call")
    put_price = black_scholes(S, K, T, r, sigma, option_type="put")

    lhs = call_price - put_price
    rhs = S - K * np.exp(-r * T)

    # Floating point arithmetic requires a small tolerance
    assert np.isclose(lhs, rhs, atol=1e-5), f"Parity violated: {lhs} != {rhs}"
