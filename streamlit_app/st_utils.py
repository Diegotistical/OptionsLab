from __future__ import annotations
import sys
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple
import logging  # Added for debugging

import numpy as np
import pandas as pd
import streamlit as st

# At the top of st_utils.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # streamlit_app/
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))  # <- add src to PYTHONPATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("monte_carlo")

# --- Safe import helper
def safe_import(module_path: str, class_name: str):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception as e:
        logger.error(f"Failed to import {module_path}.{class_name}: {e}")
        return None

bs_price = safe_import("pricing_models.black_scholes", "black_scholes")
BinomialTree = safe_import("pricing_models.binomial_tree", "BinomialTree")
MonteCarloPricer = safe_import("pricing_models.monte_carlo", "MonteCarloPricer")
MonteCarloML = safe_import("pricing_models.monte_carlo_ml", "MonteCarloML")
MonteCarloPricerUni = safe_import("pricing_models.monte_carlo_unified", "MonteCarloPricerUni")
VaRAnalyzer = safe_import("risk_analysis.var", "VaRAnalyzer")
expected_shortfall = safe_import("risk_analysis.expected_shortfall", "ExpectedShortfall")
VolatilitySurfaceGenerator = safe_import("volatility_surface.surface_generator", "VolatilitySurfaceGenerator")
check_butterfly_arbitrage = safe_import("volatility_surface.utils.arbitrage_utils", "check_butterfly_arbitrage")

# ---------- Internal fallback Monte Carlo (loop-based) ----------
def _simulate_payoffs_fallback(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"],
    num_sim: int = 50_000,
    num_steps: int = 100,
    seed: Optional[int] = 42,
    q: float = 0.0,
) -> np.ndarray:
    """Robust loop-based fallback that works on Streamlit Cloud"""
    try:
        # Reset seed properly for reproducibility
        np.random.seed(seed)
        dt = T / num_steps
        Z = np.random.standard_normal((num_sim, num_steps))
        
        # Initialize price paths
        S_paths = np.zeros((num_sim, num_steps))
        S_paths[:, 0] = S
        
        # Generate paths with dividend yield
        for t in range(1, num_steps):
            S_paths[:, t] = S_paths[:, t-1] * np.exp(
                (r - q - 0.5 * sigma**2) * dt + 
                sigma * np.sqrt(dt) * Z[:, t]
            )
        
        # Calculate terminal payoffs
        if option_type == "call":
            payoff = np.maximum(S_paths[:, -1] - K, 0.0)
        else:
            payoff = np.maximum(K - S_paths[:, -1], 0.0)
            
        return np.exp(-r * T) * payoff
    except Exception as e:
        logger.error(f"Monte Carlo fallback failed: {str(e)}")
        raise

# ---------- Cache Helpers ----------
@st.cache_resource(show_spinner=False)
def get_binomial_tree(n_steps: int = 500):
    if BinomialTree is None:
        return None
    try:
        return BinomialTree(num_steps=n_steps)
    except Exception as e:
        logger.error(f"BinomialTree init failed: {str(e)}")
        return None

@st.cache_resource(show_spinner=False)
def get_mc_pricer(num_sim: int = 50_000, num_steps: int = 100, seed: Optional[int] = 42, use_numba: bool = False):
    if MonteCarloPricer is None:
        return None
    try:
        return MonteCarloPricer(num_simulations=int(num_sim), num_steps=int(num_steps), seed=seed, use_numba=use_numba)
    except Exception as e:
        logger.error(f"MonteCarloPricer init failed: {str(e)}")
        return None

@st.cache_resource(show_spinner=False)
def get_mc_ml_surrogate(num_sim: int = 50_000, num_steps: int = 100, seed: Optional[int] = 42):
    if MonteCarloML is None:
        return None
    try:
        return MonteCarloML(num_simulations=int(num_sim), num_steps=int(num_steps), seed=seed)
    except Exception as e:
        logger.error(f"MonteCarloML init failed: {str(e)}")
        return None

@st.cache_resource(show_spinner=False)
def get_mc_unified_pricer(num_sim: int = 50_000, num_steps: int = 100, seed: Optional[int] = 42, use_numba: bool = True, use_gpu: bool = False):
    if MonteCarloPricerUni is None:
        return None
    try:
        return MonteCarloPricerUni(num_simulations=int(num_sim), num_steps=int(num_steps), seed=seed, use_numba=use_numba, use_gpu=use_gpu)
    except Exception as e:
        logger.error(f"MonteCarloPricerUni init failed: {str(e)}")
        return None

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
def price_black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: Literal["call","put"], q: float = 0.0) -> Optional[float]:
    if bs_price is None:
        return None
    try:
        return float(bs_price(S, K, T, r, sigma, option_type, q))
    except Exception as e:
        logger.error(f"BS pricing failed: {str(e)}")
        return None

def price_binomial(S: float, K: float, T: float, r: float, sigma: float, option_type: Literal["call","put"], q: float = 0.0, n_steps: int = 500, style: Literal["european","american"]="european") -> Optional[float]:
    tree = get_binomial_tree(n_steps)
    if tree is None:
        return None
    try:
        return float(tree.price(S, K, T, r, sigma, option_type, style, q))
    except Exception as e:
        logger.error(f"Binomial pricing failed: {str(e)}")
        return None

def price_monte_carlo(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call","put"],
    q: float = 0.0,
    num_sim: int = 50_000,
    num_steps: int = 100,
    seed: Optional[int] = 42,
    use_numba: bool = False
) -> Optional[float]:
    """
    Try to use external MonteCarloPricer if available; otherwise run a built-in fallback.
    Always returns a float (unless inputs are invalid).
    """
    # Attempt to use optimized pricer
    mc = get_mc_pricer(num_sim=num_sim, num_steps=num_steps, seed=seed, use_numba=use_numba)
    if mc is not None:
        try:
            return float(mc.price(S, K, T, r, sigma, option_type, q))
        except Exception as e:
            logger.warning(f"External MC pricer failed: {str(e)}. Using fallback.")
    
    # Fallback: loop-based simulation (works on Streamlit Cloud)
    try:
        discounted = _simulate_payoffs_fallback(S, K, T, r, sigma, option_type, num_sim=int(num_sim), num_steps=int(num_steps), seed=seed, q=q)
        return float(np.mean(discounted))
    except Exception as e:
        logger.error(f"MC fallback failed: {str(e)}")
        return None

def greeks_mc_delta_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call","put"],
    q: float = 0.0,
    num_sim: int = 50_000,
    num_steps: int = 100,
    seed: Optional[int] = 42,
    h: float = 1e-3,
    use_numba: bool = False
) -> Tuple[Optional[float],Optional[float]]:
    """
    Compute delta and gamma via central finite differences.
    Prefer external pricer for accuracy/performance; otherwise use the fallback.
    """
    mc = get_mc_pricer(num_sim=num_sim, num_steps=num_steps, seed=seed, use_numba=use_numba)
    # If external pricer works, use it
    if mc is not None:
        try:
            p_down = float(mc.price(S - h, K, T, r, sigma, option_type, q))
            p_mid  = float(mc.price(S, K, T, r, sigma, option_type, q))
            p_up   = float(mc.price(S + h, K, T, r, sigma, option_type, q))
            delta = (p_up - p_down) / (2*h)
            gamma = (p_up - 2*p_mid + p_down) / (h**2)
            return float(delta), float(gamma)
        except Exception as e:
            logger.warning(f"External MC greeks failed: {str(e)}. Using fallback.")

    # Fallback: loop-based simulation
    try:
        disc_down = _simulate_payoffs_fallback(S - h, K, T, r, sigma, option_type, num_sim=int(num_sim), num_steps=int(num_steps), seed=seed, q=q)
        disc_mid  = _simulate_payoffs_fallback(S,     K, T, r, sigma, option_type, num_sim=int(num_sim), num_steps=int(num_steps), seed=seed, q=q)
        disc_up   = _simulate_payoffs_fallback(S + h, K, T, r, sigma, option_type, num_sim=int(num_sim), num_steps=int(num_steps), seed=seed, q=q)
        
        p_down = float(np.mean(disc_down))
        p_mid  = float(np.mean(disc_mid))
        p_up   = float(np.mean(disc_up))
        
        delta = (p_up - p_down) / (2*h)
        gamma = (p_up - 2*p_mid + p_down) / (h**2)
        return float(delta), float(gamma)
    except Exception as e:
        logger.error(f"Greeks fallback failed: {str(e)}")
        return None, None

# ---------- Risk Wrappers ----------
def compute_var_es(returns: pd.Series, level: float = 0.95) -> Tuple[Optional[float], Optional[float]]:
    var_v, es_v = None, None
    if VaRAnalyzer:
        try:
            var_v = float(VaRAnalyzer(returns, level=level))
        except Exception as e:
            logger.error(f"VaR calculation failed: {str(e)}")
            var_v = None
    if expected_shortfall:
        try:
            es_v = float(expected_shortfall(returns, level=level))
        except Exception as e:
            logger.error(f"ES calculation failed: {str(e)}")
            es_v = None
    return var_v, es_v

# ---------- Vol Surface Helpers ----------
@dataclass
class SurfaceResult:
    strikes: np.ndarray
    maturities: np.ndarray
    iv_grid: np.ndarray

def build_surface(strikes: np.ndarray, maturities: np.ndarray, ivs: np.ndarray, strike_points: int = 50, maturity_points: int = 50, method: str = "cubic", extrapolate: bool = False, benchmark: bool = True) -> Optional[SurfaceResult]:
    if VolatilitySurfaceGenerator is None:
        return None
    try:
        gen = VolatilitySurfaceGenerator(strikes=strikes, maturities=maturities, ivs=ivs, strike_points=strike_points, maturity_points=maturity_points, interp_method=method, allow_extrapolation=extrapolate, benchmark=benchmark)
        gS, gT, gIV = gen.generate_surface()
        return SurfaceResult(strikes=gS, maturities=gT, iv_grid=gIV)
    except Exception as e:
        logger.error(f"Surface generation failed: {str(e)}")
        return None

# ---------- Micro-benchmark ----------
def timeit_ms(fn, *args, **kwargs) -> Tuple[Any,float]:
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    dt_ms = (time.perf_counter() - start) * 1000.0
    return out, dt_ms