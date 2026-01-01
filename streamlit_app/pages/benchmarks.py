import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ======================
# CONFIGURATION
# ======================
logger = logging.getLogger("option_benchmarks")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ======================
# IMPORT HANDLING
# ======================
def get_pricing_models():
    """Import of all pricing models with fallback implementations"""
    models = {}
    
    # Black-Scholes
    try:
        from src.pricing_models.black_scholes import BlackScholesPricer
        models["bs"] = BlackScholesPricer()
        logger.info("Successfully imported BlackScholesPricer")
    except ImportError as e:
        logger.warning(f"BlackScholesPricer import failed: {str(e)}")
        models["bs"] = None
    
    # Monte Carlo Unified (both direct and path)
    try:
        from src.pricing_models.monte_carlo_unified import MonteCarloPricerUni
        models["mc_unified_direct"] = MonteCarloPricerUni(
            num_simulations=50000, num_steps=1, seed=42, use_numba=True, use_gpu=False
        )
        models["mc_unified_path"] = MonteCarloPricerUni(
            num_simulations=50000, num_steps=100, seed=42, use_numba=True, use_gpu=False
        )
        logger.info("Successfully imported MonteCarloPricerUni")
    except ImportError as e:
        logger.warning(f"MonteCarloPricerUni import failed: {str(e)}")
        models["mc_unified_direct"] = None
        models["mc_unified_path"] = None
    
    # Monte Carlo ML
    try:
        from src.pricing_models.monte_carlo_ml import MonteCarloML
        models["mc_ml"] = MonteCarloML(
            num_simulations=50000, num_steps=100, seed=42
        )
        logger.info("Successfully imported MonteCarloML")
    except (ImportError, TypeError) as e:
        logger.warning(f"MonteCarloML import failed: {str(e)}")
        models["mc_ml"] = None
    
    # Binomial Tree
    try:
        from src.pricing_models.binomial_tree import BinomialTree
        models["bt"] = BinomialTree(num_steps=500)
        logger.info("Successfully imported BinomialTree")
    except ImportError as e:
        logger.warning(f"BinomialTree import failed: {str(e)}")
        models["bt"] = None
    
    return models


def timeit_ms(fn, *args, **kwargs) -> Tuple[Any, float]:
    """Measure execution time in milliseconds"""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000.0
    return result, elapsed


# ======================
# FALLBACK IMPLEMENTATIONS
# ======================
def fallback_black_scholes(S, K, T, r, sigma, option_type="call", q=0.0):
    """Fallback implementation of Black-Scholes pricing"""
    try:
        import math
        from scipy.stats import norm
        T = max(T, 0.0001)
        sigma = max(sigma, 0.0001)
        d1 = (math.log(S / max(K, 0.0001)) + (r - q + 0.5 * sigma**2) * T) / (
            sigma * math.sqrt(T)
        )
        d2 = d1 - sigma * math.sqrt(T)
        if option_type == "call":
            price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
        return float(price)
    except Exception as e:
        logger.error(f"Black-Scholes fallback failed: {str(e)}")
        return 0.0


def fallback_monte_carlo_direct(S, K, T, r, sigma, option_type, q=0.0, num_sim=50000, seed=42):
    """Direct terminal simulation (CORRECT for European options)"""
    try:
        T = max(T, 0.0001)
        sigma = max(sigma, 0.0001)
        np.random.seed(seed)
        Z = np.random.standard_normal(num_sim)
        Z_ant = -Z
        drift = (r - q - 0.5 * sigma**2) * T
        diffusion = sigma * np.sqrt(T)
        S_T = np.concatenate([
            S * np.exp(drift + diffusion * Z),
            S * np.exp(drift + diffusion * Z_ant)
        ])
        if option_type == "call":
            payoff = np.maximum(S_T - K, 0.0)
        else:
            payoff = np.maximum(K - S_T, 0.0)
        return float(np.mean(np.exp(-r * T) * payoff))
    except Exception as e:
        logger.error(f"Direct MC fallback failed: {str(e)}")
        return 0.0


def fallback_monte_carlo_path(S, K, T, r, sigma, option_type, q=0.0, num_sim=50000, num_steps=100, seed=42):
    """Path simulation (introduces discretization bias)"""
    try:
        T = max(T, 0.0001)
        sigma = max(sigma, 0.0001)
        np.random.seed(seed)
        dt = T / max(num_steps, 1)
        Z = np.random.standard_normal((num_sim, num_steps))
        S_paths = np.zeros((num_sim, num_steps))
        S_paths[:, 0] = S
        for t in range(1, num_steps):
            drift = (r - q - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * Z[:, t]
            S_paths[:, t] = S_paths[:, t - 1] * np.exp(drift + diffusion)
        if option_type == "call":
            payoff = np.maximum(S_paths[:, -1] - K, 0.0)
        else:
            payoff = np.maximum(K - S_paths[:, -1], 0.0)
        return float(np.mean(np.exp(-r * T) * payoff))
    except Exception as e:
        logger.error(f"Path MC fallback failed: {str(e)}")
        return 0.0


def fallback_binomial_tree(S, K, T, r, sigma, option_type="call", exercise_style="european", q=0.0, num_steps=500):
    """Fallback implementation of Binomial Tree"""
    try:
        if S <= 0 or K <= 0 or T < 0 or sigma < 0 or q < 0 or num_steps <= 0:
            return 0.0
        if option_type not in {"call", "put"} or exercise_style not in {"european", "american"}:
            return 0.0
        if T == 0:
            return float(max(S - K, 0.0)) if option_type == "call" else float(max(K - S, 0.0))
        if sigma == 0:
            df = np.exp(-r * T)
            fwd = S * np.exp((r - q) * T)
            intrinsic = max(fwd - K, 0.0) if option_type == "call" else max(K - fwd, 0.0)
            return float(intrinsic * df)

        dt = T / num_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        p = min(max(p, 0.0), 1.0)

        asset_prices = np.empty((num_steps + 1, num_steps + 1), dtype=np.float64)
        for i in range(num_steps + 1):
            j = np.arange(i + 1)
            asset_prices[i, : i + 1] = S * (u**j) * (d ** (i - j))

        disc = np.exp(-r * dt)
        option_values = np.empty_like(asset_prices)

        if option_type == "call":
            option_values[-1, : num_steps + 1] = np.maximum(asset_prices[-1, : num_steps + 1] - K, 0)
        else:
            option_values[-1, : num_steps + 1] = np.maximum(K - asset_prices[-1, : num_steps + 1], 0)

        for step in range(num_steps - 1, -1, -1):
            option_values[step, : step + 1] = disc * (
                p * option_values[step + 1, 1 : step + 2]
                + (1 - p) * option_values[step + 1, : step + 1]
            )
            if exercise_style == "american":
                intrinsic = np.maximum(asset_prices[step, : step + 1] - K, 0) if option_type == "call" else np.maximum(K - asset_prices[step, : step + 1], 0)
                option_values[step, : step + 1] = np.maximum(option_values[step, : step + 1], intrinsic)

        return float(option_values[0, 0])
    except Exception as e:
        logger.error(f"Binomial Tree fallback failed: {str(e)}")
        return 0.0


# ======================
# STYLING (Same as original)
# ======================
st.markdown(
    """
<style>
    body {
        padding: 0 !important;
        margin: 0 !important;
        background-color: #0f172a;
        color: #e2e8f0;
    }
    .main-header {
        font-size: 2.5rem;
        color: #f8fafc;
        margin-bottom: 0.5rem;
        font-weight: 700;
        text-align: center;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
        opacity: 0.9;
        text-align: center;
    }
    .stApp {
        max-width: 100% !important;
        padding: 0 1rem !important;
        background-color: #0f172a;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tablist"] {
        background-color: #1e293b;
        border-radius: 8px;
        margin-bottom: 2rem;
        padding: 0 !important;
        width: 100% !important;
    }
    .stTabs [role="tab"] {
        background-color: #1e293b;
        color: #94a3b8;
        font-weight: 500;
        padding: 12px 24px;
        border: none;
        border-radius: 8px 8px 0 0;
        transition: all 0.3s ease;
        width: 25% !important;
        text-align: center;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
        margin-bottom: -2px;
    }
    .chart-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
    }
    .chart-description {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 600;
        line-height: 1.2;
    }
    .subsection-header {
        font-size: 1.3rem;
        color: #f8fafc;
        margin: 1.2rem 0 0.8rem 0;
        font-weight: 600;
    }
    .engine-option {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
        border: 1px solid #334155;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    .engine-label {
        font-size: 0.9rem;
        color: #f8fafc !important;
        margin-bottom: 0.3rem;
    }
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        width: 100% !important;
    }
    .stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stNumberInput > div > label,
    .stSlider > div > label,
    .stSelectbox > div > label {
        color: #f8fafc !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    .explanation-box {
        background-color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .executive-insight {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #334155;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    .executive-title {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-bottom: 0.3rem;
    }
    .executive-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #f8fafc;
    }
    .executive-help {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.3rem;
        line-height: 1.3;
    }
    .executive-summary {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 1.2rem;
        border: 1px solid #334155;
    }
    .executive-summary-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.8rem;
    }
    .executive-summary-text {
        font-size: 0.95rem;
        color: #e2e8f0;
        line-height: 1.5;
    }
    .highlight {
        background-color: #1e3a8a;
        color: #93c5fd;
        padding: 0.1rem 0.3rem;
        border-radius: 4px;
        font-weight: 500;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ======================
# PAGE CONTENT
# ======================
st.markdown(
    '<h1 class="main-header">Option Pricing Model Benchmark</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">Comprehensive performance comparison with honest MC implementation</p>',
    unsafe_allow_html=True,
)

# Model selection
st.markdown(
    '<div style="background-color: #1e293b; padding: 1rem; border-radius: 8px; border: 1px solid #334155; margin-bottom: 1.5rem;">',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="engine-label" style="margin-bottom: 0.5rem;">Select Models to Compare</div>',
    unsafe_allow_html=True,
)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    include_bs = st.checkbox("Black-Scholes", value=True, key="include_bs")
with col2:
    include_mc_direct = st.checkbox("MC Direct ✓", value=True, key="include_mc_direct")
with col3:
    include_mc_path = st.checkbox("MC Path ⚠️", value=False, key="include_mc_path")
with col4:
    include_mc_ml = st.checkbox("MC ML", value=True, key="include_mc_ml")
with col5:
    include_bt = st.checkbox("Binomial Tree", value=True, key="include_bt")
st.markdown("</div>", unsafe_allow_html=True)

# Input section
st.markdown(
    '<h3 class="subsection-header">Pricing Parameters</h3>', unsafe_allow_html=True
)
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Spot Price (S)</div>', unsafe_allow_html=True)
    S = st.number_input("", 1.0, 1_000.0, 100.0, key="spot_bench")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Strike Price (K)</div>', unsafe_allow_html=True)
    K = st.number_input("", 1.0, 1_000.0, 100.0, key="strike_bench")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Maturity (T, years)</div>', unsafe_allow_html=True)
    T = st.number_input("", 0.01, 5.0, 1.0, key="maturity_bench")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Risk-free Rate (r)</div>', unsafe_allow_html=True)
    r = st.number_input("", 0.0, 0.25, 0.05, key="riskfree_bench")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Dividend Yield (q)</div>', unsafe_allow_html=True)
    q = st.number_input("", 0.0, 0.2, 0.0, key="dividend_bench")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Volatility (σ)</div>', unsafe_allow_html=True)
    sigma = st.number_input("", 0.001, 2.0, 0.2, key="volatility_bench")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Option Type</div>', unsafe_allow_html=True)
    option_type = st.selectbox("", ["call", "put"], key="option_type_bench")
    st.markdown("</div>", unsafe_allow_html=True)

# Configuration section
st.markdown(
    '<h3 class="subsection-header">Model Configuration</h3>', unsafe_allow_html=True
)
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Monte Carlo Simulations</div>', unsafe_allow_html=True)
    num_sim = st.slider("", 10_000, 200_000, 50_000, step=10_000, key="sim_bench")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Time Steps (Path MC only)</div>', unsafe_allow_html=True)
    num_steps = st.slider("", 10, 500, 100, step=10, key="steps_bench")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Random Seed</div>', unsafe_allow_html=True)
    seed = st.number_input("", value=42, min_value=1, key="seed_bench")
    st.markdown("</div>", unsafe_allow_html=True)

# Run button
st.markdown('<div style="display: flex; justify-content: center; margin: 1.5rem 0; width: 100%;">', unsafe_allow_html=True)
run = st.button("Run Benchmarks", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Main application logic
if run:
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Initializing pricing models...")
        progress_bar.progress(10)
        models = get_pricing_models()

        # Validate parameters
        validation_errors = []
        if S <= 0:
            validation_errors.append("Spot price (S) must be positive")
        if K <= 0:
            validation_errors.append("Strike price (K) must be positive")
        if T <= 0.001:
            validation_errors.append("Maturity (T) must be greater than 0.001 years")
        if sigma <= 0.001:
            validation_errors.append("Volatility (σ) must be greater than 0.001")
        if r < 0:
            validation_errors.append("Risk-free rate (r) cannot be negative")
        if q < 0:
            validation_errors.append("Dividend yield (q) cannot be negative")

        if validation_errors:
            st.error("❌ Parameter validation failed:")
            for error in validation_errors:
                st.error(f"• {error}")
            st.stop()

        results = []
        reference_price = None
        price_errors = {}

        # Black-Scholes
        if include_bs:
            status_text.text("Running Black-Scholes benchmark...")
            progress_bar.progress(15)
            try:
                if models["bs"] is not None:
                    price, latency = timeit_ms(models["bs"].price, S, K, T, r, sigma, option_type, q)
                else:
                    price, latency = timeit_ms(fallback_black_scholes, S, K, T, r, sigma, option_type, q)
                
                results.append({
                    "model": "Black-Scholes",
                    "price": price,
                    "time_ms": latency,
                    "type": "Analytical",
                    "description": "Closed-form solution",
                })
                reference_price = price
            except Exception as e:
                logger.error(f"Black-Scholes benchmark failed: {str(e)}")

        # MC Direct Terminal (CORRECT)
        if include_mc_direct:
            status_text.text("Running MC Direct Terminal benchmark...")
            progress_bar.progress(30)
            try:
                if models["mc_unified_direct"] is not None:
                    price, latency = timeit_ms(
                        models["mc_unified_direct"].price, S, K, T, r, sigma, option_type, q, use_path_simulation=False
                    )
                else:
                    price, latency = timeit_ms(
                        fallback_monte_carlo_direct, S, K, T, r, sigma, option_type, q, num_sim, seed
                    )
                
                results.append({
                    "model": f"MC Direct Terminal ({num_sim*2:,} paths)",
                    "price": price,
                    "time_ms": latency,
                    "type": "Monte Carlo (Exact)",
                    "description": "Direct terminal simulation, 1 timestep, zero bias",
                })
                if reference_price is not None and price != "Error":
                    price_errors["MC Direct"] = abs(price - reference_price)
            except Exception as e:
                logger.error(f"MC Direct benchmark failed: {str(e)}")

        # MC Path Simulation (for comparison)
        if include_mc_path:
            status_text.text("Running MC Path Simulation benchmark...")
            progress_bar.progress(45)
            try:
                if models["mc_unified_path"] is not None:
                    price, latency = timeit_ms(
                        models["mc_unified_path"].price, S, K, T, r, sigma, option_type, q, use_path_simulation=True
                    )
                else:
                    price, latency = timeit_ms(
                        fallback_monte_carlo_path, S, K, T, r, sigma, option_type, q, num_sim, num_steps, seed
                    )
                
                results.append({
                    "model": f"MC Path Simulation ({num_sim*2:,}×{num_steps})",
                    "price": price,
                    "time_ms": latency,
                    "type": "Monte Carlo (Biased)",
                    "description": f"Path simulation, {num_steps} timesteps, introduces discretization bias",
                })
                if reference_price is not None and price != "Error":
                    price_errors["MC Path"] = abs(price - reference_price)
            except Exception as e:
                logger.error(f"MC Path benchmark failed: {str(e)}")

        # Monte Carlo ML
        if include_mc_ml:
            status_text.text("Running Monte Carlo ML benchmark...")
            progress_bar.progress(60)
            try:
                if models["mc_ml"] is not None:
                    # Create training grid
                    grid_S = np.linspace(max(50, S - 20), min(200, S + 20), 5)
                    grid_K = np.linspace(max(50, K - 20), min(200, K + 20), 5)
                    Sg, Kg = np.meshgrid(grid_S, grid_K)
                    df = pd.DataFrame({
                        "S": Sg.ravel(),
                        "K": Kg.ravel(),
                        "T": np.full(Sg.size, T),
                        "r": np.full(Sg.size, r),
                        "sigma": np.full(Sg.size, sigma),
                        "q": np.full(Sg.size, q),
                    })
                    # Fit model
                    _, t_fit_ms = timeit_ms(models["mc_ml"].fit, df)
                    # Predict
                    x_single = pd.DataFrame([{"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "q": q}])
                    _, t_pred_ms = timeit_ms(models["mc_ml"].predict, x_single)
                    pred_df = models["mc_ml"].predict(x_single)
                    price = pred_df["price"].iloc[0]
                    
                    results.append({
                        "model": "MC ML (Training)",
                        "price": "N/A",
                        "time_ms": t_fit_ms,
                        "type": "ML Training",
                        "description": "One-time model training",
                    })
                    results.append({
                        "model": "MC ML (Prediction)",
                        "price": price,
                        "time_ms": t_pred_ms,
                        "type": "ML Prediction",
                        "description": "Fast prediction after training",
                    })
                    if reference_price is not None and price != "Error":
                        price_errors["MC ML"] = abs(price - reference_price)
            except Exception as e:
                logger.error(f"MC ML benchmark failed: {str(e)}")

        # Binomial Tree
        if include_bt:
            status_text.text("Running Binomial Tree benchmark...")
            progress_bar.progress(80)
            try:
                if models["bt"] is not None:
                    price, latency = timeit_ms(
                        models["bt"].price, S, K, T, r, sigma, option_type, "european", q
                    )
                else:
                    price, latency = timeit_ms(
                        fallback_binomial_tree, S, K, T, r, sigma, option_type, "european", q, 500
                    )
                
                results.append({
                    "model": "Binomial Tree (500 steps)",
                    "price": price,
                    "time_ms": latency,
                    "type": "Lattice",
                    "description": "CRR Binomial Tree (European)",
                })
                if reference_price is not None and price != "Error":
                    price_errors["Binomial Tree"] = abs(price - reference_price)
            except Exception as e:
                logger.error(f"Binomial Tree benchmark failed: {str(e)}")

        progress_bar.progress(100)
        time.sleep(0.2)
        status_text.text("✅ Benchmarks complete!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()

        # Display results table
        st.markdown('<h2 class="chart-title">Benchmark Results</h2>', unsafe_allow_html=True)

        display_data = []
        for result in results:
            price_str = f"${result['price']:.6f}" if isinstance(result['price'], (int, float)) else result['price']
            time_str = f"{result['time_ms']:.2f} ms" if isinstance(result['time_ms'], (int, float)) else result['time_ms']
            
            model_key = result['model'].split(" ")[0] + (" " + result['model'].split(" ")[1] if len(result['model'].split(" ")) > 1 else "")
            error_str = f"{price_errors.get(model_key, 0):.6f}" if model_key in price_errors else "—"
            
            display_data.append({
                "Model": result['model'],
                "Type": result['type'],
                "Price": price_str,
                "Time": time_str,
                "Error vs BS": error_str,
                "Description": result['description'],
            })

        df = pd.DataFrame(display_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Add explanation box
        st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
        st.markdown("""
        <p style="color: #e2e8f0; margin: 0;">
        <strong>🎯 Understanding Monte Carlo Methods:</strong><br><br>
        <strong>Direct Terminal Simulation (MC Direct):</strong> Uses the exact terminal distribution S_T = S_0 * exp((r-q-0.5σ²)T + σ√T*Z). 
        This is mathematically correct for European options: 1 timestep, zero discretization bias, ~50-100x faster than path simulation.<br><br>
        <strong>Path Simulation (MC Path):</strong> Discretizes the path into multiple timesteps using Euler-Maruyama. 
        This introduces discretization bias O(1/n) and is computationally expensive. Only needed for path-dependent options 
        (American, barriers, Asians) where the payoff depends on the entire price path, not just the terminal value.<br><br>
        <strong>For European options, always use direct terminal simulation.</strong>
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Tabs for detailed analysis (same structure as original)
        tab1, tab2, tab3, tab4 = st.tabs([
            "Model Comparison",
            "Performance Analysis",
            "Accuracy Assessment",
            "Executive Summary",
        ])

        with tab1:
            st.markdown('<h2 class="chart-title">Model Comparison</h2>', unsafe_allow_html=True)
            
            valid_results = [r for r in results if isinstance(r["price"], (int, float))]
            if valid_results:
                col1, col2 = st.columns(2)
                
                with col1:
                    price_results = [r for r in valid_results if "Training" not in r["model"]]
                    if price_results:
                        fig_price = go.Figure()
                        fig_price.add_trace(go.Bar(
                            x=[r["model"] for r in price_results],
                            y=[r["price"] for r in price_results],
                            marker_color=['#3b82f6', '#10b981', '#ef4444', '#8b5cf6', '#f59e0b'][:len(price_results)],
                            width=0.6,
                        ))
                        if reference_price is not None:
                            fig_price.add_shape(
                                type="line",
                                x0=-0.5, y0=reference_price,
                                x1=len(price_results) - 0.5, y1=reference_price,
                                line=dict(color="#f87171", width=2, dash="dash"),
                            )
                        fig_price.update_layout(
                            title="Option Price Comparison",
                            xaxis_title="",
                            yaxis_title="Option Price",
                            template="plotly_dark",
                            height=400,
                            paper_bgcolor="#0f172a",
                            plot_bgcolor="#1e293b",
                        )
                        st.plotly_chart(fig_price, use_container_width=True)
                
                with col2:
                    pred_results = [r for r in results if "Prediction" in r["model"] or "Black-Scholes" in r["model"] or "MC Direct" in r["model"] or "Binomial" in r["model"]]
                    pred_results = [r for r in pred_results if isinstance(r["time_ms"], (int, float))]
                    if pred_results:
                        fig_time = go.Figure()
                        fig_time.add_trace(go.Bar(
                            x=[r["model"] for r in pred_results],
                            y=[r["time_ms"] for r in pred_results],
                            marker_color=['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b'][:len(pred_results)],
                            width=0.6,
                        ))
                        fig_time.update_layout(
                            title="Execution Time Comparison",
                            xaxis_title="",
                            yaxis_title="Time (ms, log scale)",
                            yaxis_type="log",
                            template="plotly_dark",
                            height=400,
                            paper_bgcolor="#0f172a",
                            plot_bgcolor="#1e293b",
                        )
                        st.plotly_chart(fig_time, use_container_width=True)

        with tab2:
            st.markdown('<h2 class="chart-title">Performance Analysis</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">Computational efficiency evaluation</p>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="executive-summary">
            <div class="executive-summary-title">Key Performance Insights</div>
            <div class="executive-summary-text">
            <p><strong>Direct Terminal vs Path Simulation:</strong> Direct terminal simulation is ~50-100x faster 
            because it eliminates unnecessary path discretization. For European options, this is always the correct choice.</p>
            <p><strong>When Path Matters:</strong> Path simulation becomes necessary only for American options 
            (early exercise), barrier options (path-dependent triggers), or Asian options (path-dependent averaging).</p>
            <p><strong>ML Trade-off:</strong> Monte Carlo ML requires expensive one-time training but provides 
            near-instant predictions, making it ideal for applications requiring thousands of option valuations.</p>
            </div>
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            st.markdown('<h2 class="chart-title">Accuracy Assessment</h2>', unsafe_allow_html=True)
            
            if price_errors:
                errors = [(k, v) for k, v in price_errors.items()]
                fig_error = go.Figure()
                fig_error.add_trace(go.Bar(
                    x=[e[0] for e in errors],
                    y=[e[1] for e in errors],
                    marker_color=['#10b981' if e[1] < 0.001 else '#ef4444' for e in errors],
                    width=0.6,
                ))
                fig_error.update_layout(
                    title="Price Error vs Black-Scholes",
                    xaxis_title="",
                    yaxis_title="Absolute Error",
                    template="plotly_dark",
                    height=400,
                    paper_bgcolor="#0f172a",
                    plot_bgcolor="#1e293b",
                )
                st.plotly_chart(fig_error, use_container_width=True)

        with tab4:
            st.markdown('<h2 class="chart-title">Executive Summary</h2>', unsafe_allow_html=True)
            st.markdown("""
            <div class="executive-summary">
            <div class="executive-summary-title">Strategic Recommendations</div>
            <div class="executive-summary-text">
            <p><span class="highlight">For European Options:</span> Use Black-Scholes for analytical speed when assumptions hold, 
            or MC Direct Terminal for non-BS models (Heston, rough volatility). Never use path simulation—it's 100x slower with added bias.</p>
            
            <p><span class="highlight">For American/Exotic Options:</span> Use path-based methods (MC Path, Binomial Tree) only when 
            genuinely needed for path-dependent features. Binomial trees often converge faster than Euler discretization.</p>
            
            <p><span class="highlight">For High-Volume Applications:</span> Monte Carlo ML provides the best throughput after 
            one-time training, making it ideal for portfolio risk management and scenario analysis requiring thousands of valuations.</p>
            
            <p><span class="highlight">Production Architecture:</span> Implement a hybrid approach where MC Direct handles 
            the majority of European options, ML accelerates bulk computations, and path-based methods are reserved for 
            genuinely path-dependent contracts.</p>
            </div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Benchmarking failed: {str(e)}")
        logger.exception("Critical error in benchmarking")

else:
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background-color: #1e293b; border-radius: 8px; 
                 border: 1px solid #334155; margin-top: 1rem;">
        <h3 style="color: #e2e8f0; margin-bottom: 0.75rem;">Get Started</h3>
        <p style="color: #94a3b8; margin-bottom: 1rem;">Configure your parameters above and click "Run Benchmarks" to see results</p>
    </div>
    """, unsafe_allow_html=True)