"""
Shared utilities for the Streamlit UI: safe wrappers, caching, plotting,
and thin adapters to `src` modules.
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --- Attempt imports from your repo (tolerant to either function/class styles)
# Black–Scholes (function expected: price_call/price_put or price(..., option_type))
try:
    from pricing_models.black_scholes import black_scholes as bs_price  # type: ignore
except Exception:
    bs_price = None

# BinomialTree (class)
try:
    from pricing_models.binomial_tree import BinomialTree
except Exception:
    BinomialTree = None

# Monte Carlo (class)
try:
    from pricing_models.monte_carlo import MonteCarloPricer
except Exception:
    MonteCarloPricer = None

# Risk analytics
try:
    from risk_analysis.var import historical_var as historical_var
except Exception:
    historical_var = None

try:
    from risk_analysis.expected_shortfall import ExpectedShortfall as expected_shortfall
except Exception:
    expected_shortfall = None

# Volatility surface core + utils
try:
    from volatility_surface.surface_generator import VolatilitySurfaceGenerator
except Exception:
    VolatilitySurfaceGenerator = None

# Arbitrage utils optionally present
try:
    from volatility_surface.utils.arbitrage_utils import check_butterfly_arbitrage
except Exception:
    check_butterfly_arbitrage = None


# ---------- Cache Helpers ----------

@st.cache_resource(show_spinner=False)
def get_binomial_tree(n_steps: int = 500):
    if BinomialTree is None:
        raise RuntimeError("BinomialTree not found. Check src/pricing_models/binomial_tree.py")
    return BinomialTree(num_steps=n_steps)

@st.cache_resource(show_spinner=False)
def get_mc_pricer(num_sim: int = 50_000, num_steps: int = 100, seed: Optional[int] = 42):
    if MonteCarloPricer is None:
        raise RuntimeError("MonteCarloPricer not found. Check src/pricing_models/monte_carlo.py")
    return MonteCarloPricer(num_simulations=num_sim, num_steps=num_steps, seed=seed)

@st.cache_data(show_spinner=False)
def load_readme(max_lines: int = 80) -> str:
    path = os.path.join(os.getcwd(), "README.md")
    if not os.path.exists(path):
        return "_README.md not found_"
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()[:max_lines]
    return "".join(lines)

# ---------- Sidebar Status ----------

def show_repo_status() -> None:
    items = [
        ("Black–Scholes", "✅" if bs_price else "⚠️"),
        ("Binomial Tree", "✅" if BinomialTree else "⚠️"),
        ("Monte Carlo", "✅" if MonteCarloPricer else "⚠️"),
        ("VaR/ES", "✅" if (historical_var and expected_shortfall) else "⚠️"),
        ("Vol Surface", "✅" if VolatilitySurfaceGenerator else "⚠️"),
    ]
    for name, status in items:
        st.write(f"{status} {name}")

# ---------- Pricing Wrappers ----------

def price_black_scholes(S: float, K: float, T: float, r: float, sigma: float,
                        option_type: Literal["call", "put"], q: float = 0.0) -> float:
    """
    Thin wrapper that tries common signatures for your BS implementation.
    """
    if bs_price is None:
        raise RuntimeError("Black–Scholes price function not found.")
    try:
        # Preference: price(S,K,T,r,sigma, option_type, q)
        return float(bs_price(S, K, T, r, sigma, option_type, q))
    except TypeError:
        # Fallbacks: price_call/price_put or price(..., is_call=True)
        try:
            if option_type == "call":
                from src.pricing_models.black_scholes import price_call  # type: ignore
                return float(price_call(S, K, T, r, sigma, q))
            else:
                from src.pricing_models.black_scholes import price_put  # type: ignore
                return float(price_put(S, K, T, r, sigma, q))
        except Exception:
            # Attempt signature with is_call
            return float(bs_price(S, K, T, r, sigma, is_call=(option_type == "call"), q=q))

def price_binomial(S: float, K: float, T: float, r: float, sigma: float,
                   option_type: Literal["call", "put"], q: float = 0.0,
                   n_steps: int = 500, style: Literal["european", "american"] = "european") -> float:
    tree = get_binomial_tree(n_steps)
    return float(tree.price(S, K, T, r, sigma, option_type, style, q))

def price_monte_carlo(S: float, K: float, T: float, r: float, sigma: float,
                      option_type: Literal["call", "put"], q: float = 0.0,
                      num_sim: int = 50_000, num_steps: int = 100, seed: Optional[int] = 42) -> float:
    mc = get_mc_pricer(num_sim, num_steps, seed)
    return float(mc.price(S, K, T, r, sigma, option_type, q))

def greeks_mc_delta_gamma(S: float, K: float, T: float, r: float, sigma: float,
                          option_type: Literal["call", "put"], q: float = 0.0,
                          num_sim: int = 50_000, num_steps: int = 100, seed: Optional[int] = 42,
                          h: float = 1e-3) -> Tuple[float, float]:
    mc = get_mc_pricer(num_sim, num_steps, seed)
    # Central differences using existing MC pricer for clarity
    p_down = mc.price(S - h, K, T, r, sigma, option_type, q)
    p_mid  = mc.price(S,     K, T, r, sigma, option_type, q)
    p_up   = mc.price(S + h, K, T, r, sigma, option_type, q)
    delta = (p_up - p_down) / (2 * h)
    gamma = (p_up - 2 * p_mid + p_down) / (h ** 2)
    return float(delta), float(gamma)

# ---------- Risk Wrappers ----------

def compute_var_es(returns: pd.Series, level: float = 0.95) -> Tuple[Optional[float], Optional[float]]:
    var_v = None
    es_v = None
    if historical_var:
        var_v = float(historical_var(returns, level=level))
    if expected_shortfall:
        es_v = float(expected_shortfall(returns, level=level))
    return var_v, es_v

# ---------- Vol Surface Helpers ----------

@dataclass
class SurfaceResult:
    strikes: np.ndarray
    maturities: np.ndarray
    iv_grid: np.ndarray

def build_surface(strikes: np.ndarray, maturities: np.ndarray, ivs: np.ndarray,
                  strike_points: int = 50, maturity_points: int = 50,
                  method: str = "cubic", extrapolate: bool = False, benchmark: bool = True) -> SurfaceResult:
    if VolatilitySurfaceGenerator is None:
        raise RuntimeError("VolatilitySurfaceGenerator not found.")
    gen = VolatilitySurfaceGenerator(
        strikes=strikes,
        maturities=maturities,
        ivs=ivs,
        strike_points=strike_points,
        maturity_points=maturity_points,
        interp_method=method,
        allow_extrapolation=extrapolate,
        benchmark=benchmark
    )
    gS, gT, gIV = gen.generate_surface()
    return SurfaceResult(strikes=gS, maturities=gT, iv_grid=gIV)

# ---------- Micro-benchmark ----------

def timeit_ms(fn, *args, **kwargs) -> Tuple[Any, float]:
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    dt_ms = (time.perf_counter() - start) * 1000.0
    return out, dt_ms
