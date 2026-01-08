# streamlit_app/st_utils.py
"""
Streamlit utility functions and cached wrappers for OptionsLab.

This module provides:
    - Cached pricer instances for performance
    - Fallback implementations for Streamlit Cloud compatibility
    - Helper functions for visualizations
    - Wrapper functions with consistent error handling

All pricing functions use absolute imports from src.* modules.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("st_utils")


# =============================================================================
# Path Setup for Cross-Platform Compatibility
# =============================================================================


def _setup_paths() -> Path:
    """
    Set up Python path for both local and Streamlit Cloud environments.

    Returns:
        Root path of the project.
    """
    # Streamlit Cloud uses /mount/src/
    cloud_root = Path("/mount/src/optionslab")
    if cloud_root.exists():
        root = cloud_root
    else:
        # Local development: find project root
        current = Path(__file__).resolve()
        for parent in [current.parent, *current.parents]:
            if (parent / "src").exists() and (parent / "setup.py").exists():
                root = parent
                break
        else:
            root = Path.cwd()

    # Add root to path if not present
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        logger.info(f"Added {root_str} to sys.path")

    return root


# Execute path setup at import time
PROJECT_ROOT = _setup_paths()


# =============================================================================
# Import Pricing Modules with Fallback
# =============================================================================

# Try imports from src package
try:
    from src.pricing_models.black_scholes import black_scholes as bs_price

    logger.info("Imported black_scholes from src.pricing_models")
except ImportError as e:
    logger.warning(f"Failed to import black_scholes: {e}")
    bs_price = None

try:
    from src.pricing_models.binomial_tree import BinomialTree

    logger.info("Imported BinomialTree from src.pricing_models")
except ImportError as e:
    logger.warning(f"Failed to import BinomialTree: {e}")
    BinomialTree = None

try:
    from src.pricing_models.monte_carlo import MonteCarloPricer

    logger.info("Imported MonteCarloPricer from src.pricing_models")
except ImportError as e:
    logger.warning(f"Failed to import MonteCarloPricer: {e}")
    MonteCarloPricer = None

try:
    from src.pricing_models.monte_carlo_ml import MonteCarloMLSurrogate as MonteCarloML

    logger.info("Imported MonteCarloMLSurrogate from src.pricing_models")
except ImportError as e:
    logger.warning(f"Failed to import MonteCarloMLSurrogate: {e}")
    MonteCarloML = None

try:
    from src.pricing_models.monte_carlo_unified import MonteCarloPricerUni

    logger.info("Imported MonteCarloPricerUni from src.pricing_models")
except ImportError as e:
    logger.warning(f"Failed to import MonteCarloPricerUni: {e}")
    MonteCarloPricerUni = None

try:
    from src.risk_analysis.var import VaRAnalyzer

    logger.info("Imported VaRAnalyzer from src.risk_analysis")
except ImportError as e:
    logger.warning(f"Failed to import VaRAnalyzer: {e}")
    VaRAnalyzer = None

try:
    from src.risk_analysis.expected_shortfall import ExpectedShortfall

    expected_shortfall = ExpectedShortfall
    logger.info("Imported ExpectedShortfall from src.risk_analysis")
except ImportError as e:
    logger.warning(f"Failed to import ExpectedShortfall: {e}")
    expected_shortfall = None

try:
    from src.volatility_surface.surface_generator import VolatilitySurfaceGenerator

    logger.info("Imported VolatilitySurfaceGenerator")
except ImportError as e:
    logger.warning(f"Failed to import VolatilitySurfaceGenerator: {e}")
    VolatilitySurfaceGenerator = None


# =============================================================================
# Helper Functions
# =============================================================================


def _ensure_scalar(value: Union[float, np.ndarray, List, pd.Series]) -> float:
    """
    Convert arrays, lists, or pandas Series to scalar values.

    Args:
        value: Input value (may be array-like).

    Returns:
        Scalar float value.
    """
    if isinstance(value, (np.ndarray, list, pd.Series)):
        return float(np.mean(value))
    return float(value)


def _extract_scalar(value: Any) -> float:
    """
    Extract scalar value from various container types.

    Args:
        value: Input value.

    Returns:
        Scalar float value.
    """
    if isinstance(value, pd.Series) and len(value) == 1:
        return float(value.values[0])
    elif hasattr(value, "item"):
        return float(value.item())
    elif isinstance(value, (np.ndarray, list)):
        return float(np.mean(value))
    return float(value)


# =============================================================================
# Fallback Monte Carlo Implementation
# =============================================================================


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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback Monte Carlo simulation for when optimized pricers unavailable.

    This implementation works on all platforms including Streamlit Cloud.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to maturity.
        r: Risk-free rate.
        sigma: Volatility.
        option_type: 'call' or 'put'.
        num_sim: Number of simulations.
        num_steps: Time steps per path.
        seed: Random seed.
        q: Dividend yield.

    Returns:
        Tuple of (discounted_payoffs, price_paths).
    """
    # Convert to scalars
    S = _extract_scalar(S)
    K = _extract_scalar(K)
    T = _extract_scalar(T)
    r = _extract_scalar(r)
    sigma = _extract_scalar(sigma)
    q = _extract_scalar(q)

    np.random.seed(int(seed) if seed is not None else None)
    dt = T / num_steps

    # Generate random normals
    Z = np.random.standard_normal((num_sim, num_steps))

    # Initialize and simulate paths
    S_paths = np.zeros((num_sim, num_steps))
    S_paths[:, 0] = S

    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion_factor = sigma * np.sqrt(dt)

    for t in range(1, num_steps):
        S_paths[:, t] = S_paths[:, t - 1] * np.exp(drift + diffusion_factor * Z[:, t])

    # Calculate payoffs
    if option_type == "call":
        payoff = np.maximum(S_paths[:, -1] - K, 0.0)
    else:
        payoff = np.maximum(K - S_paths[:, -1], 0.0)

    discounted = np.exp(-r * T) * payoff

    return discounted, S_paths


# Public alias
simulate_payoffs = _simulate_payoffs_fallback


# =============================================================================
# Cached Pricer Instances
# =============================================================================


@st.cache_resource(show_spinner=False)
def get_binomial_tree(n_steps: int = 500) -> Optional[Any]:
    """
    Get cached BinomialTree instance.

    Args:
        n_steps: Number of tree steps.

    Returns:
        BinomialTree instance or None if unavailable.
    """
    if BinomialTree is None:
        logger.warning("BinomialTree not available")
        return None
    try:
        return BinomialTree(num_steps=n_steps)
    except Exception as e:
        logger.error(f"BinomialTree init failed: {e}")
        return None


@st.cache_resource(show_spinner=False)
def get_mc_pricer(
    num_sim: int = 50_000,
    num_steps: int = 100,
    seed: Optional[int] = 42,
    use_numba: bool = False,
) -> Optional[Any]:
    """
    Get cached MonteCarloPricer instance.

    Args:
        num_sim: Number of simulations.
        num_steps: Time steps per path.
        seed: Random seed.
        use_numba: Enable Numba acceleration.

    Returns:
        MonteCarloPricer instance or None if unavailable.
    """
    if MonteCarloPricer is None:
        logger.warning("MonteCarloPricer not available")
        return None
    try:
        return MonteCarloPricer(
            num_simulations=int(num_sim),
            num_steps=int(num_steps),
            seed=seed,
            use_numba=use_numba,
        )
    except Exception as e:
        logger.error(f"MonteCarloPricer init failed: {e}")
        return None


@st.cache_resource(show_spinner=False)
def get_mc_ml_surrogate(
    num_sim: int = 50_000,
    num_steps: int = 100,
    seed: Optional[int] = 42,
) -> Optional[Any]:
    """
    Get cached MonteCarloMLSurrogate instance.

    Args:
        num_sim: MC simulations for training data.
        num_steps: Time steps for MC.
        seed: Random seed.

    Returns:
        MonteCarloMLSurrogate instance or None if unavailable.
    """
    if MonteCarloML is None:
        logger.warning("MonteCarloMLSurrogate not available")
        return None
    try:
        return MonteCarloML(
            num_simulations=int(num_sim),
            num_steps=int(num_steps),
            seed=seed,
        )
    except Exception as e:
        logger.error(f"MonteCarloMLSurrogate init failed: {e}")
        return None


@st.cache_resource(show_spinner=False)
def get_mc_unified_pricer(
    num_sim: int = 50_000,
    num_steps: int = 100,
    seed: Optional[int] = 42,
    use_numba: bool = True,
    use_gpu: bool = False,
) -> Optional[Any]:
    """
    Get cached MonteCarloPricerUni instance.

    Args:
        num_sim: Number of simulations.
        num_steps: Time steps.
        seed: Random seed.
        use_numba: Enable Numba JIT.
        use_gpu: Enable GPU acceleration.

    Returns:
        MonteCarloPricerUni instance or None if unavailable.
    """
    if MonteCarloPricerUni is None:
        logger.warning("MonteCarloPricerUni not available")
        return None
    try:
        return MonteCarloPricerUni(
            num_simulations=int(num_sim),
            num_steps=int(num_steps),
            seed=seed,
            use_numba=use_numba,
            use_gpu=use_gpu,
        )
    except Exception as e:
        logger.error(f"MonteCarloPricerUni init failed: {e}")
        return None


# =============================================================================
# Pricing Wrapper Functions
# =============================================================================


def price_black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"],
    q: float = 0.0,
) -> Optional[float]:
    """
    Price option using Black-Scholes formula.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to maturity.
        r: Risk-free rate.
        sigma: Volatility.
        option_type: 'call' or 'put'.
        q: Dividend yield.

    Returns:
        Option price or None if unavailable.
    """
    if bs_price is None:
        logger.warning("Black-Scholes not available")
        return None
    try:
        return float(bs_price(S, K, T, r, sigma, option_type, q))
    except Exception as e:
        logger.error(f"BS pricing failed: {e}")
        return None


def price_binomial(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"],
    q: float = 0.0,
    n_steps: int = 500,
    style: Literal["european", "american"] = "european",
) -> Optional[float]:
    """
    Price option using binomial tree.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to maturity.
        r: Risk-free rate.
        sigma: Volatility.
        option_type: 'call' or 'put'.
        q: Dividend yield.
        n_steps: Tree steps.
        style: 'european' or 'american'.

    Returns:
        Option price or None if unavailable.
    """
    tree = get_binomial_tree(n_steps)
    if tree is None:
        return None
    try:
        return float(tree.price(S, K, T, r, sigma, option_type, style, q))
    except Exception as e:
        logger.error(f"Binomial pricing failed: {e}")
        return None


def price_monte_carlo(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"],
    q: float = 0.0,
    num_sim: int = 50_000,
    num_steps: int = 100,
    seed: Optional[int] = 42,
    use_numba: bool = False,
) -> float:
    """
    Price option using Monte Carlo simulation.

    Uses optimized pricer if available, otherwise falls back to
    pure NumPy implementation.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to maturity.
        r: Risk-free rate.
        sigma: Volatility.
        option_type: 'call' or 'put'.
        q: Dividend yield.
        num_sim: Number of simulations.
        num_steps: Time steps.
        seed: Random seed.
        use_numba: Enable Numba acceleration.

    Returns:
        Option price (never None).
    """
    # Convert to scalars
    S = _extract_scalar(S)
    K = _extract_scalar(K)
    T = _extract_scalar(T)
    r = _extract_scalar(r)
    sigma = _extract_scalar(sigma)
    q = _extract_scalar(q)

    # Try optimized pricer
    mc = get_mc_pricer(num_sim, num_steps, seed, use_numba)
    if mc is not None:
        try:
            result = mc.price(S, K, T, r, sigma, option_type, q)
            if result is not None:
                return float(result)
        except Exception as e:
            logger.warning(f"MC pricer failed: {e}, using fallback")

    # Fallback
    try:
        discounted, _ = _simulate_payoffs_fallback(
            S, K, T, r, sigma, option_type, num_sim, num_steps, seed, q
        )
        return float(np.mean(discounted))
    except Exception as e:
        logger.error(f"MC fallback failed: {e}")
        return 0.0


def greeks_mc_delta_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"],
    q: float = 0.0,
    num_sim: int = 50_000,
    num_steps: int = 100,
    seed: Optional[int] = 42,
    h: float = 1e-3,
    use_numba: bool = False,
) -> Tuple[float, float]:
    """
    Compute Delta and Gamma using Monte Carlo.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to maturity.
        r: Risk-free rate.
        sigma: Volatility.
        option_type: 'call' or 'put'.
        q: Dividend yield.
        num_sim: Number of simulations.
        num_steps: Time steps.
        seed: Random seed.
        h: Finite difference step.
        use_numba: Enable Numba.

    Returns:
        Tuple of (delta, gamma).
    """
    # Convert to scalars
    S = _extract_scalar(S)
    K = _extract_scalar(K)
    T = _extract_scalar(T)
    r = _extract_scalar(r)
    sigma = _extract_scalar(sigma)
    q = _extract_scalar(q)

    # Try optimized pricer
    mc = get_mc_pricer(num_sim, num_steps, seed, use_numba)
    if mc is not None:
        try:
            delta, gamma = mc.delta_gamma(S, K, T, r, sigma, option_type, q, h, seed)
            return float(delta), float(gamma)
        except Exception as e:
            logger.warning(f"MC Greeks failed: {e}, using fallback")

    # Fallback using finite differences
    try:
        p_down = price_monte_carlo(
            S - h, K, T, r, sigma, option_type, q, num_sim, num_steps, seed
        )
        p_mid = price_monte_carlo(
            S, K, T, r, sigma, option_type, q, num_sim, num_steps, seed
        )
        p_up = price_monte_carlo(
            S + h, K, T, r, sigma, option_type, q, num_sim, num_steps, seed
        )

        delta = (p_up - p_down) / (2 * h)
        gamma = (p_up - 2 * p_mid + p_down) / (h**2)
        return float(delta), float(gamma)
    except Exception as e:
        logger.error(f"Greeks fallback failed: {e}")
        return 0.5, 0.01


# =============================================================================
# Risk Analysis Wrappers
# =============================================================================


def compute_var_es(
    returns: pd.Series,
    level: float = 0.95,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute Value at Risk and Expected Shortfall.

    Args:
        returns: Return series.
        level: Confidence level.

    Returns:
        Tuple of (VaR, ES) or None values if unavailable.
    """
    var_v, es_v = None, None

    if VaRAnalyzer:
        try:
            var_v = float(VaRAnalyzer(returns, level=level))
        except Exception as e:
            logger.error(f"VaR failed: {e}")

    if expected_shortfall:
        try:
            es_v = float(expected_shortfall(returns, level=level))
        except Exception as e:
            logger.error(f"ES failed: {e}")

    return var_v, es_v


# =============================================================================
# Vol Surface Helpers
# =============================================================================


@dataclass
class SurfaceResult:
    """Result container for volatility surface generation."""

    strikes: np.ndarray
    maturities: np.ndarray
    iv_grid: np.ndarray


def build_surface(
    strikes: np.ndarray,
    maturities: np.ndarray,
    ivs: np.ndarray,
    strike_points: int = 50,
    maturity_points: int = 50,
    method: str = "cubic",
    extrapolate: bool = False,
    benchmark: bool = True,
) -> Optional[SurfaceResult]:
    """
    Build interpolated volatility surface.

    Args:
        strikes: Input strike prices.
        maturities: Input maturities.
        ivs: Implied volatilities.
        strike_points: Output grid strike points.
        maturity_points: Output grid maturity points.
        method: Interpolation method.
        extrapolate: Allow extrapolation.
        benchmark: Enable benchmarking.

    Returns:
        SurfaceResult or None if unavailable.
    """
    if VolatilitySurfaceGenerator is None:
        logger.warning("VolatilitySurfaceGenerator not available")
        return None

    try:
        gen = VolatilitySurfaceGenerator(
            strikes=strikes,
            maturities=maturities,
            ivs=ivs,
            strike_points=strike_points,
            maturity_points=maturity_points,
            interp_method=method,
            allow_extrapolation=extrapolate,
            benchmark=benchmark,
        )
        gS, gT, gIV = gen.generate_surface()
        return SurfaceResult(strikes=gS, maturities=gT, iv_grid=gIV)
    except Exception as e:
        logger.error(f"Surface generation failed: {e}")
        return None


# =============================================================================
# README Loading
# =============================================================================


@st.cache_data(show_spinner=False)
def load_readme(max_lines: int = 80) -> str:
    """
    Load README.md content.

    Args:
        max_lines: Maximum lines to return.

    Returns:
        README content as string.
    """
    possible_paths = [
        PROJECT_ROOT / "README.md",
        Path("/mount/src/optionslab/README.md"),
        Path.cwd() / "README.md",
    ]

    for path in possible_paths:
        try:
            if path.exists():
                lines = path.read_text(encoding="utf-8").splitlines(True)[:max_lines]
                return "".join(lines)
        except Exception:
            continue

    return "_README.md not found_"


# =============================================================================
# Sidebar Status Display
# =============================================================================


def show_repo_status() -> None:
    """Display module availability status in sidebar."""
    items = [
        ("Black–Scholes", "✅" if bs_price else "⚠️"),
        ("Binomial Tree", "✅" if BinomialTree else "⚠️"),
        ("Monte Carlo", "✅" if MonteCarloPricer else "⚠️"),
        ("Monte Carlo ML", "✅" if MonteCarloML else "⚠️"),
        ("Unified MC", "✅" if MonteCarloPricerUni else "⚠️"),
        ("VaR/ES", "✅" if VaRAnalyzer and expected_shortfall else "⚠️"),
        ("Vol Surface", "✅" if VolatilitySurfaceGenerator else "⚠️"),
    ]

    for name, status in items:
        st.write(f"{status} {name}")


# =============================================================================
# Timing Utility
# =============================================================================


def timeit_ms(fn, *args, **kwargs) -> Tuple[Any, float]:
    """
    Time a function call in milliseconds.

    Args:
        fn: Function to time.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Tuple of (result, elapsed_ms).
    """
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms
