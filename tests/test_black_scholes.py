import pytest
from src.pricing_models.black_scholes import black_scholes_price

def test_call_price_matches_known_value():
    # Known analytical case
    price = black_scholes_price(
        S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call"
    )
    assert pytest.approx(price, 0.01) == 10.45

def test_put_price_matches_known_value():
    price = black_scholes_price(
        S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put"
    )
    assert pytest.approx(price, 0.01) == 5.57
