import numpy as np
import pytest

from src.risk_analysis.var import VaRAnalyzer


def test_monte_carlo_var_basic():
    np.random.seed(42)  # ensure reproducibility

    var_analyzer = VaRAnalyzer(confidence_level=0.95, time_horizon_days=1)

    initial_price = 100
    mu = 0.05
    sigma = 0.2
    portfolio_value = 1_000_000
    num_simulations = 10_000

    results = var_analyzer.monte_carlo_var(
        initial_price=initial_price,
        mu=mu,
        sigma=sigma,
        scale=portfolio_value,  # Renamed from portfolio_value to scale
        num_simulations=num_simulations,
        use_log_returns=True,
        # binomial_pricer=None,  # Removed as it is not in the function signature
    )

    assert "var" in results
    assert "cvar" in results
    assert results["var"] > 0  # VaR is returned as a positive loss value
    assert results["cvar"] > results["var"]  # CVaR is greater loss than VaR


def test_monte_carlo_var_raises_on_bad_inputs():
    var_analyzer = VaRAnalyzer()

    with pytest.raises(Exception):
        var_analyzer.monte_carlo_var(
            initial_price=100,
            mu=0.05,
            sigma=-0.1,  # invalid negative volatility
            scale=1000,
        )

    with pytest.raises(Exception):
        var_analyzer.monte_carlo_var(
            initial_price=100,
            mu=0.05,
            sigma=0.1,
            num_simulations=-100, # invalid
        )