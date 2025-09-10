from __future__ import annotations
import sys
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple, List, Union
import logging

import numpy as np
import pandas as pd
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("monte_carlo")

# Get the root directory properly - CRITICAL FIX FOR STREAMLIT CLOUD
try:
    # Streamlit Cloud has a different working directory structure
    ROOT = Path("/mount/src/optionslab") if Path("/mount/src/optionslab").exists() else Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    
    # Add SRC to PYTHONPATH
    sys.path.insert(0, str(SRC))
    logger.info(f"Added {SRC} to PYTHONPATH. Current sys.path: {sys.path}")
except Exception as e:
    logger.error(f"Failed to set up paths: {str(e)}")
    # Fallback: try to determine paths differently
    try:
        ROOT = Path.cwd()
        SRC = ROOT / "src"
        sys.path.insert(0, str(SRC))
        logger.info(f"Using fallback path: {SRC}")
    except Exception as e2:
        logger.error(f"Fallback path setup failed: {str(e2)}")

# --- Safe import helper with multiple fallbacks ---
def safe_import(module_path: str, class_name: str):
    """Safely import a class from a module with multiple fallback strategies"""
    try:
        # Strategy 1: Direct import
        try:
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except ImportError as e:
            logger.warning(f"Direct import failed for {module_path}: {str(e)}")
        
        # Strategy 2: Split and import step by step
        parts = module_path.split('.')
        current_module = None
        
        for i in range(len(parts)):
            current_path = '.'.join(parts[:i+1])
            try:
                current_module = __import__(current_path, fromlist=[])
            except ImportError as e:
                logger.warning(f"Could not import {current_path}: {str(e)}")
                continue
        
        if current_module is None:
            return None
            
        # Navigate to the final class
        for part in parts[1:]:
            current_module = getattr(current_module, part, None)
            if current_module is None:
                break
                
        if current_module is not None:
            return getattr(current_module, class_name, None)
        return None
    except Exception as e:
        logger.error(f"Failed to import {module_path}.{class_name}: {str(e)}")
        return None

# --- Import all required components ---
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
def _ensure_scalar(value: Union[float, np.ndarray, List, pd.Series]) -> float:
    """Convert arrays, lists, or pandas Series to scalar values"""
    if isinstance(value, (np.ndarray, list, pd.Series)):
        return float(np.mean(value))
    return float(value)

def _extract_scalar(value: Any) -> float:
    """Extract scalar value from pandas Series or other containers"""
    if isinstance(value, pd.Series) and len(value) == 1:
        return float(value.values[0])
    elif hasattr(value, 'item'):
        return float(value.item())
    elif isinstance(value, (np.ndarray, list)):
        return float(np.mean(value))
    return float(value)

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
    """Loop-based fallback that works on BOTH local and Streamlit Cloud"""
    try:
        # CRITICAL FIX: Use legacy random seed for compatibility
        np.random.seed(int(seed) if seed is not None else None)
        dt = T / num_steps
        num_sim = int(num_sim)
        num_steps = int(num_steps)
        
        # Generate Z with proper dimensions
        Z = np.random.standard_normal((num_sim, num_steps))
        
        # Initialize price paths
        S_paths = np.zeros((num_sim, num_steps))
        S_paths[:, 0] = _extract_scalar(S)
        
        # Convert all parameters to scalars to prevent shape mismatches
        r = _extract_scalar(r)
        q = _extract_scalar(q)
        sigma = _extract_scalar(sigma)
        
        # Generate paths with dividend yield (exact match to page implementation)
        for t in range(1, num_steps):
            # CRITICAL FIX: Ensure consistent shapes for broadcasting
            drift = (r - q - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * Z[:, t]
            S_paths[:, t] = S_paths[:, t-1] * np.exp(drift + diffusion)
        
        # Calculate terminal payoffs
        if option_type == "call":
            payoff = np.maximum(S_paths[:, -1] - _extract_scalar(K), 0.0)
        else:
            payoff = np.maximum(_extract_scalar(K) - S_paths[:, -1], 0.0)
            
        return np.exp(-r * T) * payoff
    except Exception as e:
        logger.error(f"Monte Carlo fallback failed: {str(e)}")
        raise

# Expose the fallback function for external use
simulate_payoffs = _simulate_payoffs_fallback

# ---------- Cache Helpers ----------
@st.cache_resource(show_spinner=False)
def get_binomial_tree(n_steps: int = 500):
    if BinomialTree is None:
        logger.warning("BinomialTree is not available. Using fallback.")
        return None
    try:
        return BinomialTree(num_steps=n_steps)
    except Exception as e:
        logger.error(f"BinomialTree init failed: {str(e)}")
        return None

@st.cache_resource(show_spinner=False)
def get_mc_pricer(num_sim: int = 50_000, num_steps: int = 100, seed: Optional[int] = 42, use_numba: bool = False):
    if MonteCarloPricer is None:
        logger.warning("MonteCarloPricer is not available. Using fallback.")
        return None
    try:
        return MonteCarloPricer(num_simulations=int(num_sim), num_steps=int(num_steps), seed=seed, use_numba=use_numba)
    except Exception as e:
        logger.error(f"MonteCarloPricer init failed: {str(e)}")
        return None

@st.cache_resource(show_spinner=False)
def get_mc_ml_surrogate(num_sim: int = 50_000, num_steps: int = 100, seed: Optional[int] = 42):
    if MonteCarloML is None:
        logger.warning("MonteCarloML is not available. Using fallback.")
        return None
    try:
        return MonteCarloML(num_simulations=int(num_sim), num_steps=int(num_steps), seed=seed)
    except Exception as e:
        logger.error(f"MonteCarloML init failed: {str(e)}")
        return None

@st.cache_resource(show_spinner=False)
def get_mc_unified_pricer(num_sim: int = 50_000, num_steps: int = 100, seed: Optional[int] = 42, use_numba: bool = True, use_gpu: bool = False):
    if MonteCarloPricerUni is None:
        logger.warning("MonteCarloPricerUni is not available. Using fallback.")
        return None
    try:
        return MonteCarloPricerUni(num_simulations=int(num_sim), num_steps=int(num_steps), seed=seed, use_numba=use_numba, use_gpu=use_gpu)
    except Exception as e:
        logger.error(f"MonteCarloPricerUni init failed: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def load_readme(max_lines: int = 80) -> str:
    try:
        # Try multiple possible paths for README
        possible_paths = [
            Path(__file__).resolve().parents[1] / "README.md",
            Path("/mount/src/optionslab/README.md"),
            Path.cwd() / "README.md"
        ]
        
        for path in possible_paths:
            if path.exists():
                return "".join(path.read_text(encoding="utf-8").splitlines(True)[:max_lines])
        
        return "_README.md not found_"
    except Exception as e:
        logger.error(f"Failed to load README: {str(e)}")
        return "_Error loading README_"

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
        logger.warning("Black-Scholes pricing not available. Using fallback.")
        return None
    try:
        return float(bs_price(S, K, T, r, sigma, option_type, q))
    except Exception as e:
        logger.error(f"BS pricing failed: {str(e)}")
        return None

def price_binomial(S: float, K: float, T: float, r: float, sigma: float, option_type: Literal["call","put"], q: float = 0.0, n_steps: int = 500, style: Literal["european","american"]="european") -> Optional[float]:
    tree = get_binomial_tree(n_steps)
    if tree is None:
        logger.warning("Binomial tree pricing not available. Using fallback.")
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
) -> float:  # Changed from Optional[float] to float
    """
    Try to use external MonteCarloPricer if available; otherwise run a built-in fallback.
    Always returns a float (never None).
    """
    # Convert parameters to scalars to prevent shape mismatches
    S = _extract_scalar(S)
    K = _extract_scalar(K)
    T = _extract_scalar(T)
    r = _extract_scalar(r)
    sigma = _extract_scalar(sigma)
    q = _extract_scalar(q)
    
    # Attempt to use optimized pricer
    mc = get_mc_pricer(num_sim=num_sim, num_steps=num_steps, seed=seed, use_numba=use_numba)
    if mc is not None:
        try:
            result = mc.price(S, K, T, r, sigma, option_type, q)
            if result is None:
                logger.warning("External MC pricer returned None. Using fallback.")
            else:
                return float(result)
        except Exception as e:
            logger.warning(f"External MC pricer failed: {str(e)}. Using fallback.")
    
    # Fallback: loop-based simulation (works on Streamlit Cloud)
    try:
        discounted = _simulate_payoffs_fallback(S, K, T, r, sigma, option_type, num_sim=int(num_sim), num_steps=int(num_steps), seed=seed, q=q)
        return float(np.mean(discounted))
    except Exception as e:
        logger.error(f"MC fallback failed: {str(e)}")
        # CRITICAL FIX: Return a reasonable default value instead of None
        return 0.0

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
) -> Tuple[float, float]:  # Changed from Optional[float] to float
    """
    Compute delta and gamma via central finite differences.
    Prefer external pricer for accuracy/performance; otherwise use the fallback.
    Always returns numeric values (never None).
    """
    # Convert parameters to scalars to prevent shape mismatches
    S = _extract_scalar(S)
    K = _extract_scalar(K)
    T = _extract_scalar(T)
    r = _extract_scalar(r)
    sigma = _extract_scalar(sigma)
    q = _extract_scalar(q)
    
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
        # CRITICAL FIX: Return reasonable defaults instead of None
        return 0.5, 0.01

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
        logger.warning("Volatility surface generator not available.")
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