# tests/unit/test_risk_analysis.py

import numpy as np
import pandas as pd
import pytest

from src.risk_analysis.expected_shortfall import ExpectedShortfall
from src.risk_analysis.sensitivity_analysis import SensitivityAnalysis
from src.risk_analysis.stress_testing import StressTester, StressScenario

def test_historical_es_simple():
    returns = np.array([0.1, -0.05, -0.2, 0.02, -0.01])
    es_95 = ExpectedShortfall.historical_es(returns, alpha=0.95)
    assert es_95 >= 0.0

def test_parametric_es():
    es = ExpectedShortfall.parametric_es_gaussian(mu=0.0, sigma=0.02, alpha=0.95)
    assert es >= 0.0

def test_sensitivity_delta_gamma_vega():
    # trivial price_fn: price = underlying_price * notional (delta=notional)
    def price_fn(df):
        return (df["underlying_price"].values * df["notional"].values)

    df = pd.DataFrame({
        "underlying_price": np.array([100.0, 120.0]),
        "notional": np.array([1.0, 2.0]),
        "implied_volatility": np.array([0.2, 0.25])
    })

    sa = SensitivityAnalysis()
    delta = sa.compute_delta(df, price_fn, bump=1e-2)
    # delta should be approx equal to notional
    assert np.allclose(delta, df["notional"].values, rtol=1e-3)

def test_stress_tester_basic():
    def price_fn(df):
        # price = underlying_price * notional - implied_vol * 100 (toy)
        return df["underlying_price"].values * df["notional"].values - df["implied_volatility"].values * 100.0

    df = pd.DataFrame({
        "underlying_price": np.array([100.0, 120.0]),
        "notional": np.array([1.0, 2.0]),
        "implied_volatility": np.array([0.2, 0.25])
    })

    tester = StressTester(price_fn)
    scenarios = [StressScenario("down10", "underlying_price", -0.10, "relative")]
    report = tester.run_scenarios(df, scenarios)
    assert "down10" in report.index
    assert "total_pnl" in report.columns
