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
        portfolio_value=portfolio_value,
        num_simulations=num_simulations,
        use_log_returns=True,
        binomial_pricer=None
    )

    assert 'var' in results
    assert 'cvar' in results
    assert results['var'] < 0  # VaR is a loss (negative)
    assert results['cvar'] < results['var']  # CVaR more extreme than VaR

def test_monte_carlo_var_raises_on_bad_inputs():
    var_analyzer = VaRAnalyzer()

    with pytest.raises(Exception):
        var_analyzer.monte_carlo_var(
            initial_price=100,
            mu=0.05,
            sigma=-0.1,  # invalid negative volatility
            portfolio_value=1000
        )
    
    with pytest.raises(Exception):
        var_analyzer.monte_carlo_var(
            initial_price=100,
            mu=0.05,
            sigma=0.1,
            portfolio_value=-1000  # invalid negative portfolio value
        )
