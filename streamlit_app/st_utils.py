# streamlit_app/st_utils.py
from __future__ import annotations

import sys
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --- Make repo root and src importable (works locally and on Streamlit Cloud)
ROOT = Path(__file__).resolve().parents[1]     # project root
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --- Import from your src packages (tolerant of absence)
try:
    from pricing_models.black_scholes import black_scholes as bs_price  # type: ignore
except Exception:
    bs_price = None

try:
    from pricing_models.binomial_tree import BinomialTree
except Exception:
    BinomialTree = None

try:
    from pricing_models.monte_carlo import MonteCarloPricer
except Exception:
    MonteCarloPricer = None

try:
    from pricing_models.monte_carlo_ml import MonteCarloML
except Exception:
    MonteCarloML = None

try:
    from pricing_models.monte_carlo_unified import MonteCarloPricerUni
except Exception:
    MonteCarloPricerUni = None

try:
    from risk_analysis.var import VaRAnalyzer
except Exception:
    VaRAnalyzer = None

try:
    from risk_analysis.expected_shortfall import ExpectedShortfall as expected_shortfall
except Exception:
    expected_shortfall = None

try:
    from volatility_surface.surface_generator import VolatilitySurfaceGenerator
except Exception:
    VolatilitySurfaceGenerator = None

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
def get_mc_pricer(
    num_sim: int = 50_000, num_steps: int = 100, seed: Optional[int] = 42, use_numba: bool = False
):
    if MonteCarloPricer is None:
        raise RuntimeError("MonteCarloPricer not found. Check src/pricing_models/monte_carlo.py")
    return MonteCarloPricer(num_simulations=num_sim, num_steps=num_steps, seed=seed, use_numba=use_numba)

@st.cache_resource(show_spinner=False)
def get_mc_ml_surrogate(num_sim: int = 50_000, num_steps: int = 100, seed: Optional[int] = 42):
    if MonteCarloML is None:
        raise RuntimeError("MonteCarloML not found. Check src/pricing_models/monte_carlo_ml.py")
    return MonteCarloML(num_simulations=num_sim, num_steps=num_steps, seed=seed)

@st.cache_resource(show_spinner=False)
def get_mc_unified_pricer(
    num_sim: int = 50_000, num_steps: int = 100, seed: Optional[int] = 42,
    use_numba: bool = True, use_gpu: bool = False
):
    if MonteCarloPricerUni is None:
        raise RuntimeError("MonteCarloPricerUni not found. Check src/pricing_models/monte_carlo_unified.py")
    return MonteCarloPricerUni(
        num_simulations=num_sim, num_steps=num_steps, seed=seed, use_numba=use_numba, use_gpu=use_gpu
    )

@st.cache_data(show_spinner=False)
def load_readme(max_lines: int = 80) -> str:
    path = ROOT / "README.md"
    if not path.exists():
        return "_README.md not found_"
    return "".join(path.read_text(encoding="utf-8").splitlines(True)[:max_lines])


# ---------- Sidebar Status ----------
def show_repo_status() -> None:
    items = [
        ("Black–Scholes", "✅" if bs_price else "⚠️"),
        ("Binomial Tree", "✅" if BinomialTree else "⚠️"),
        ("Monte Carlo", "✅" if MonteCarloPricer else "⚠️"),
        ("Monte Carlo ML", "✅" if MonteCarloML else "⚠️"),
        ("Unified MC (CPU/GPU)", "✅" if MonteCarloPricerUni else "⚠️"),
        ("VaR/ES", "✅" if (VaRAnalyzer and expected_shortfall) else "⚠️"),
        ("Vol Surface", "✅" if VolatilitySurfaceGenerator else "⚠️"),
    ]
    for name, status in items:
        st.write(f"{status} {name}")


# ---------- Pricing Wrappers ----------
def price_black_scholes(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: Literal["call", "put"], q: float = 0.0
) -> float:
    if bs_price is None:
        raise RuntimeError("Black–Scholes price function not found.")
    try:
        return float(bs_price(S, K, T, r, sigma, option_type, q))
    except TypeError:
        try:
            if option_type == "call":
                from pricing_models.black_scholes import price_call  # type: ignore
                return float(price_call(S, K, T, r, sigma, q))
            else:
                from pricing_models.black_scholes import price_put  # type: ignore
                return float(price_put(S, K, T, r, sigma, q))
        except Exception:
            return float(bs_price(S, K, T, r, sigma, is_call=(option_type == "call"), q=q))

def price_binomial(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: Literal["call", "put"], q: float = 0.0,
    n_steps: int = 500, style: Literal["european", "american"] = "european"
) -> float:
    tree = get_binomial_tree(n_steps)
    return float(tree.price(S, K, T, r, sigma, option_type, style, q))

def price_monte_carlo(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: Literal["call", "put"], q: float = 0.0,
    num_sim: int = 50_000, num_steps: int = 100, seed: Optional[int] = 42, use_numba: bool = False
) -> float:
    mc = get_mc_pricer(num_sim, num_steps, seed, use_numba=use_numba)
    return float(mc.price(S, K, T, r, sigma, option_type, q))

def greeks_mc_delta_gamma(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: Literal["call", "put"], q: float = 0.0,
    num_sim: int = 50_000, num_steps: int = 100, seed: Optional[int] = 42,
    h: float = 1e-3, use_numba: bool = False
) -> Tuple[float, float]:
    mc = get_mc_pricer(num_sim, num_steps, seed, use_numba=use_numba)
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
    if VaRAnalyzer:
        var_v = float(VaRAnalyzer(returns, level=level))
    if expected_shortfall:
        es_v = float(expected_shortfall(returns, level=level))
    return var_v, es_v


# ---------- Vol Surface Helpers ----------
@dataclass
class SurfaceResult:
    strikes: np.ndarray
    maturities: np.ndarray
    iv_grid: np.ndarray

def build_surface(
    strikes: np.ndarray, maturities: np.ndarray, ivs: np.ndarray,
    strike_points: int = 50, maturity_points: int = 50,
    method: str = "cubic", extrapolate: bool = False, benchmark: bool = True
) -> SurfaceResult:
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
