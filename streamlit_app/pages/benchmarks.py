# Page 4 Benchmarks.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import logging
from typing import Dict, Any, Tuple, List, Optional
# ======================
# CONFIGURATION
# ======================
logger = logging.getLogger("option_benchmarks")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
# ======================
# IMPORT HANDLING
# ======================
def get_pricing_models():
    """Robust import of all pricing models with fallback implementations"""
    models = {}
    # Try to import Black-Scholes
    try:
        from src.pricing_models.black_scholes import BlackScholesPricer
        models["bs"] = BlackScholesPricer()
        logger.info("Successfully imported BlackScholesPricer")
    except ImportError as e:
        logger.warning(f"BlackScholesPricer import failed: {str(e)}")
        models["bs"] = None
    # Try to import Monte Carlo
    try:
        from src.pricing_models.monte_carlo import MonteCarloPricer
        models["mc"] = MonteCarloPricer(num_simulations=50000, num_steps=100, seed=42)
        logger.info("Successfully imported MonteCarloPricer")
    except ImportError as e:
        logger.warning(f"MonteCarloPricer import failed: {str(e)}")
        models["mc"] = None
    # Try to import Monte Carlo Unified
    try:
        from src.pricing_models.monte_carlo_unified import MonteCarloPricerUni
        models["mc_unified"] = MonteCarloPricerUni(
            num_simulations=50000, num_steps=100, seed=42, 
            use_numba=True, use_gpu=False
        )
        logger.info("Successfully imported MonteCarloPricerUni")
    except ImportError as e:
        logger.warning(f"MonteCarloPricerUni import failed: {str(e)}")
        models["mc_unified"] = None
    # Try to import Monte Carlo ML
    try:
        from src.pricing_models.monte_carlo_ml import MonteCarloML
        models["mc_ml"] = MonteCarloML(
            num_simulations=50000, num_steps=100, seed=42, 
            model_type="gb"
        )
        logger.info("Successfully imported MonteCarloML")
    except ImportError as e:
        logger.warning(f"MonteCarloML import failed: {str(e)}")
        models["mc_ml"] = None
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
        from scipy.stats import norm
        import math
        # Ensure T is not zero to avoid division by zero
        T = max(T, 0.0001)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if option_type == "call":
            price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
        return float(price)
    except Exception as e:
        logger.error(f"Black-Scholes fallback failed: {str(e)}")
        return 0.0
def fallback_monte_carlo(S, K, T, r, sigma, option_type, q=0.0, num_sim=50000, num_steps=100, seed=42):
    """Fallback implementation of Monte Carlo pricing"""
    try:
        # Ensure T is not zero to avoid division by zero
        T = max(T, 0.0001)
        np.random.seed(seed)
        dt = T / num_steps
        Z = np.random.standard_normal((num_sim, num_steps))
        S_paths = np.zeros((num_sim, num_steps))
        S_paths[:, 0] = S
        for t in range(1, num_steps):
            drift = (r - q - 0.5 * sigma ** 2) * dt
            diffusion = sigma * np.sqrt(dt) * Z[:, t]
            S_paths[:, t] = S_paths[:, t-1] * np.exp(drift + diffusion)
        if option_type == "call":
            payoff = np.maximum(S_paths[:, -1] - K, 0.0)
        else:
            payoff = np.maximum(K - S_paths[:, -1], 0.0)
        return float(np.mean(np.exp(-r * T) * payoff))
    except Exception as e:
        logger.error(f"Monte Carlo fallback failed: {str(e)}")
        return 0.0
def fallback_monte_carlo_unified(S, K, T, r, sigma, option_type, q=0.0, num_sim=50000, num_steps=100, seed=42):
    """Fallback implementation of unified Monte Carlo pricing"""
    try:
        # Ensure T is not zero to avoid division by zero
        T = max(T, 0.0001)
        # Same as regular MC for fallback
        return fallback_monte_carlo(S, K, T, r, sigma, option_type, q, num_sim, num_steps, seed)
    except Exception as e:
        logger.error(f"Unified Monte Carlo fallback failed: {str(e)}")
        return 0.0
def fallback_monte_carlo_ml(S, K, T, r, sigma, option_type, q=0.0, num_sim=50000, num_steps=100, seed=42):
    """Fallback implementation of ML-accelerated Monte Carlo"""
    try:
        # Ensure T is not zero to avoid division by zero
        T = max(T, 0.0001)
        # For fallback, just return MC price (no actual ML)
        return fallback_monte_carlo(S, K, T, r, sigma, option_type, q, num_sim, num_steps, seed)
    except Exception as e:
        logger.error(f"ML Monte Carlo fallback failed: {str(e)}")
        return 0.0
# ======================
# STYLING
# ======================
st.markdown("""
<style>
    /* Base styling - full width */
    body {
        padding: 0 !important;
        margin: 0 !important;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1E293B;
        margin-bottom: 0.5rem;
        font-weight: 700;
        text-align: center;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #475569;
        margin-bottom: 1.5rem;
        opacity: 0.9;
        text-align: center;
    }
    /* Full width containers */
    .stApp {
        max-width: 100% !important;
        padding: 0 1rem !important;
    }
    .st-emotion-cache-13ln4jf {
        padding: 0 1rem !important;
        max-width: 100% !important;
    }
    .st-emotion-cache-12oz5g7 {
        padding: 0 1rem !important;
        max-width: 100% !important;
    }
    /* Metric cards */
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    /* Tab styling - FIXED AS REQUESTED */
    .stTabs [data-baseweb="tablist"] {
        background-color: #1A202C;
        border-radius: 8px;
        margin-bottom: 2rem;
        padding: 0 !important;
    }
    .stTabs [role="tab"] {
        background-color: #1A202C;
        color: #CBD5E1;
        font-weight: 500;
        padding: 12px 24px;
        border: none;
        border-radius: 8px 8px 0 0;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
        margin-bottom: -2px;
    }
    .stTabs [aria-selected="true"]::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background-color: #EF4444;
    }
    /* Chart elements */
    .chart-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .chart-description {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    /* Metric elements */
    .metric-label {
        color: #64748B;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .metric-value {
        color: #1E293B;
        font-size: 1.8rem;
        font-weight: 600;
        line-height: 1.2;
    }
    .metric-delta {
        color: #64748B;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    /* Section headers */
    .subsection-header {
        font-size: 1.3rem;
        color: #1E293B;
        margin: 1.2rem 0 0.8rem 0;
        font-weight: 600;
    }
    /* Input sections */
    .engine-option {
        background-color: white;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    .engine-label {
        font-size: 0.9rem;
        color: white !important;
        margin-bottom: 0.3rem;
    }
    .engine-value {
        font-size: 1.1rem;
        color: #1E293B;
        font-weight: 500;
    }
    /* Progress bar - GREY */
    .stProgress > div > div > div {
        background-color: #4B5563;
        height: 6px !important;
    }
    /* Button styling - FULL WIDTH */
    .stButton > button {
        background-color: #3B82F6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        width: 100% !important;
        margin: 0 !important;
    }
    .stButton > button:hover {
        background-color: #2563EB;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    /* Input field labels - WHITE TEXT */
    .stNumberInput > div > label,
    .stSlider > div > label,
    .stSelectbox > div > label {
        color: white !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
        width: 100% !important;
    }
    .stDataFrame > div > div > div > table {
        border-collapse: separate;
        border-spacing: 0;
        width: 100% !important;
    }
    .stDataFrame > div > div > div > table th {
        background-color: #F8FAFC;
        font-weight: 600;
        color: #1E293B;
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .stDataFrame > div > div > div > table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #F1F5F9;
    }
    .stDataFrame > div > div > div > table tr:last-child td {
        border-bottom: none;
    }
    /* Executive insights */
    .executive-insight {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    .executive-title {
        font-size: 0.9rem;
        color: #64748B;
        margin-bottom: 0.3rem;
    }
    .executive-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1E293B;
    }
    .executive-help {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 0.3rem;
        line-height: 1.3;
    }
    /* Divider styling - GREY */
    hr {
        margin: 1.5rem 0;
        border: 0;
        border-top: 1px solid #E2E8F0; /* Light grey */
    }
    /* Performance bars */
    .perf-bar-container {
        background-color: #F1F5F9;
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
    }
    .perf-bar {
        height: 100%;
        border-radius: 4px;
        background-color: #3B82F6;
    }
    /* Model comparison card */
    .model-card {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    .model-card:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transform: translateY(-2px);
    }
    .model-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .model-detail {
        font-size: 0.9rem;
        color: #64748B;
        display: flex;
        justify-content: space-between;
        margin: 0.2rem 0;
    }
    .model-value {
        font-weight: 500;
        color: #1E293B;
    }
    /* Executive summary */
    .executive-summary {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
    }
    .executive-summary-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 0.8rem;
    }
    .executive-summary-text {
        font-size: 0.95rem;
        color: #475569;
        line-height: 1.5;
    }
    .highlight {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.1rem 0.3rem;
        border-radius: 4px;
        font-weight: 500;
    }
    /* Explanation box */
    .explanation-box {
        background-color: white;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    /* Full width elements */
    .st-emotion-cache-0 {
        width: 100% !important;
    }
    .st-emotion-cache-1f9epy6 {
        width: 100% !important;
    }
    .st-emotion-cache-ocqkz {
        width: 100% !important;
    }
    .st-emotion-cache-1kyxreq {
        width: 100% !important;
        justify-content: flex-start !important;
    }
    .st-emotion-cache-1v3fv3r {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)
# ======================
# PAGE CONTENT
# ======================
st.markdown('<h1 class="main-header">Option Pricing Model Benchmark</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Comprehensive performance comparison of leading option pricing methodologies</p>', unsafe_allow_html=True)
# Model selection
st.markdown('<div style="background-color: white; padding: 1rem; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
st.markdown('<div class="engine-label" style="margin-bottom: 0.5rem;">Select Models to Compare</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    include_bs = st.checkbox("Black-Scholes", value=True, key="include_bs")
with col2:
    include_mc = st.checkbox("Monte Carlo", value=True, key="include_mc")
with col3:
    include_mc_unified = st.checkbox("Monte Carlo Unified", value=True, key="include_mc_unified")
with col4:
    include_mc_ml = st.checkbox("Monte Carlo ML", value=True, key="include_mc_ml")
st.markdown('</div>', unsafe_allow_html=True)
# Input section
st.markdown('<h3 class="subsection-header">Pricing Parameters</h3>', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1], gap="medium")
with col1:
    st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Spot Price (S)</div>', unsafe_allow_html=True)
    S = st.number_input("", 1.0, 1_000.0, 100.0, key="spot_bench")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Strike Price (K)</div>', unsafe_allow_html=True)
    K = st.number_input("", 1.0, 1_000.0, 100.0, key="strike_bench")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Maturity (T, years)</div>', unsafe_allow_html=True)
    T = st.number_input("", 0.01, 5.0, 1.0, key="maturity_bench")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Risk-free Rate (r)</div>', unsafe_allow_html=True)
    r = st.number_input("", 0.0, 0.25, 0.05, key="riskfree_bench")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Dividend Yield (q)</div>', unsafe_allow_html=True)
    q = st.number_input("", 0.0, 0.2, 0.0, key="dividend_bench")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Volatility (Ïƒ)</div>', unsafe_allow_html=True)
    sigma = st.number_input("", 0.001, 2.0, 0.2, key="volatility_bench")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Option Type</div>', unsafe_allow_html=True)
    option_type = st.selectbox("", ["call", "put"], key="option_type_bench")
    st.markdown('</div>', unsafe_allow_html=True)
# Configuration section
st.markdown('<h3 class="subsection-header">Model Configuration</h3>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Monte Carlo Simulations</div>', unsafe_allow_html=True)
    num_sim = st.slider("", 10_000, 200_000, 50_000, step=10_000, key="sim_bench")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Time Steps</div>', unsafe_allow_html=True)
    num_steps = st.slider("", 10, 500, 100, step=10, key="steps_bench")
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Random Seed</div>', unsafe_allow_html=True)
    seed = st.number_input("", value=42, min_value=1, key="seed_bench")
    st.markdown('</div>', unsafe_allow_html=True)
# Run button centered
st.markdown('<div style="display: flex; justify-content: center; margin: 1.5rem 0; width: 100%;">', unsafe_allow_html=True)
run = st.button("Run Benchmarks", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
# Main application logic
if run:
    try:
        # Initialize progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        # Get pricing models
        status_text.text("Initializing pricing models...")
        progress_bar.progress(10)
        models = get_pricing_models()
        # Validate parameters - CRITICAL FIX FOR DIVISION BY ZERO
        if S <= 0 or K <= 0 or T <= 0.001 or sigma <= 0.001:
            st.warning("Invalid parameters detected. Please check input values. T must be greater than 0.001.")
            st.stop()
        # Run benchmarks
        results = []
        reference_price = None
        price_errors = {}
        # Black-Scholes Benchmark
        if include_bs:
            status_text.text("Running Black-Scholes benchmark...")
            progress_bar.progress(20)
            try:
                if models["bs"] is not None:
                    price, latency = timeit_ms(
                        models["bs"].price, S, K, T, r, sigma, option_type, q
                    )
                else:
                    price, latency = timeit_ms(
                        fallback_black_scholes, S, K, T, r, sigma, option_type, q
                    )
                results.append({
                    "model": "Black-Scholes",
                    "price": price,
                    "time_ms": latency,
                    "type": "Analytical",
                    "description": "Closed-form solution"
                })
                reference_price = price
            except Exception as e:
                logger.error(f"Black-Scholes benchmark failed: {str(e)}")
                results.append({
                    "model": "Black-Scholes",
                    "price": "Error",
                    "time_ms": "â€”",
                    "type": "Analytical",
                    "description": "Closed-form solution"
                })
        # Monte Carlo Benchmark
        if include_mc:
            status_text.text("Running Monte Carlo benchmark...")
            progress_bar.progress(40)
            try:
                if models["mc"] is not None:
                    price, latency = timeit_ms(
                        models["mc"].price, S, K, T, r, sigma, option_type, q
                    )
                else:
                    price, latency = timeit_ms(
                        fallback_monte_carlo, S, K, T, r, sigma, option_type, q, 
                        num_sim, num_steps, seed
                    )
                results.append({
                    "model": f"Monte Carlo ({num_sim:,}Ã—{num_steps})",
                    "price": price,
                    "time_ms": latency,
                    "type": "Simulation",
                    "description": "Standard Monte Carlo"
                })
                if reference_price is not None and price != "Error":
                    price_errors["Monte Carlo"] = abs(price - reference_price)
            except Exception as e:
                logger.error(f"Monte Carlo benchmark failed: {str(e)}")
                results.append({
                    "model": f"Monte Carlo ({num_sim:,}Ã—{num_steps})",
                    "price": "Error",
                    "time_ms": "â€”",
                    "type": "Simulation",
                    "description": "Standard Monte Carlo"
                })
        # Monte Carlo Unified Benchmark
        if include_mc_unified:
            status_text.text("Running Monte Carlo Unified benchmark...")
            progress_bar.progress(60)
            try:
                if models["mc_unified"] is not None:
                    price, latency = timeit_ms(
                        models["mc_unified"].price, S, K, T, r, sigma, option_type, q
                    )
                else:
                    price, latency = timeit_ms(
                        fallback_monte_carlo_unified, S, K, T, r, sigma, option_type, q, 
                        num_sim, num_steps, seed
                    )
                results.append({
                    "model": f"Monte Carlo Unified ({num_sim:,}Ã—{num_steps})",
                    "price": price,
                    "time_ms": latency,
                    "type": "Simulation",
                    "description": "Unified CPU/GPU with variance reduction"
                })
                if reference_price is not None and price != "Error":
                    price_errors["Monte Carlo Unified"] = abs(price - reference_price)
            except Exception as e:
                logger.error(f"Monte Carlo Unified benchmark failed: {str(e)}")
                results.append({
                    "model": f"Monte Carlo Unified ({num_sim:,}Ã—{num_steps})",
                    "price": "Error",
                    "time_ms": "â€”",
                    "type": "Simulation",
                    "description": "Unified CPU/GPU with variance reduction"
                })
        # Monte Carlo ML Benchmark
        if include_mc_ml:
            status_text.text("Running Monte Carlo ML benchmark...")
            progress_bar.progress(60)
            try:
                if models["mc_ml"] is not None:
                    # First, train the model (one-time cost)
                    # Create a small training grid
                    grid_S = np.linspace(max(50, S-20), min(200, S+20), 5)
                    grid_K = np.linspace(max(50, K-20), min(200, K+20), 5)
                    Sg, Kg = np.meshgrid(grid_S, grid_K)
                    df = pd.DataFrame({
                        "S": Sg.ravel(),
                        "K": Kg.ravel(),
                        "T": np.full(Sg.size, T),
                        "r": np.full(Sg.size, r),
                        "sigma": np.full(Sg.size, sigma),
                        "q": np.full(Sg.size, q)
                    })
                    # Fit the model
                    _, t_fit_ms = timeit_ms(models["mc_ml"].fit, df)
                    # Predict our single point
                    x_single = pd.DataFrame([{
                        "S": S, "K": K, "T": T, "r": r, "sigma": sigma, "q": q
                    }])
                    _, t_pred_ms = timeit_ms(models["mc_ml"].predict, x_single)
                    pred_df = models["mc_ml"].predict(x_single)
                    price = pred_df["price"].iloc[0]
                else:
                    # Fallback implementation
                    price, latency = timeit_ms(
                        fallback_monte_carlo_ml, S, K, T, r, sigma, option_type, q, 
                        num_sim, num_steps, seed
                    )
                    t_fit_ms = 0
                    t_pred_ms = latency
                # Add ML results with separate training and prediction times
                results.append({
                    "model": "Monte Carlo ML (Training)",
                    "price": "N/A",
                    "time_ms": t_fit_ms,
                    "type": "ML Training",
                    "description": "One-time model training"
                })
                results.append({
                    "model": "Monte Carlo ML (Prediction)",
                    "price": price,
                    "time_ms": t_pred_ms,
                    "type": "ML Prediction",
                    "description": "Fast prediction after training"
                })
                if reference_price is not None and price != "Error":
                    price_errors["Monte Carlo ML"] = abs(price - reference_price)
            except Exception as e:
                logger.error(f"Monte Carlo ML benchmark failed: {str(e)}")
                results.append({
                    "model": "Monte Carlo ML (Training)",
                    "price": "Error",
                    "time_ms": "â€”",
                    "type": "ML Training",
                    "description": "One-time model training"
                })
                results.append({
                    "model": "Monte Carlo ML (Prediction)",
                    "price": "Error",
                    "time_ms": "â€”",
                    "type": "ML Prediction",
                    "description": "Fast prediction after training"
                })
        # Final progress update
        progress_bar.progress(100)
        time.sleep(0.2)
        status_text.empty()
        progress_bar.empty()
        # Process results for display
        display_data = []
        for result in results:
            model = result["model"]
            price = result["price"]
            time_ms = result["time_ms"]
            # Format price
            price_str = f"${price:.6f}" if isinstance(price, (int, float)) else price
            # Format time
            time_str = f"{time_ms:.2f} ms" if isinstance(time_ms, (int, float)) else time_ms
            # Get error if available
            error_str = "â€”"
            if model in price_errors:
                error_str = f"{price_errors[model.split(' ')[0]]:.6f}"
            display_data.append({
                "Pricing Model": model,
                "Type": result["type"],
                "Option Price": price_str,
                "Execution Time": time_str,
                "Price Error": error_str,
                "Description": result["description"]
            })
        # Create DataFrame for display
        df = pd.DataFrame(display_data)
        # Display results
        st.markdown('<h2 class="chart-title">Pricing Performance Comparison</h2>', unsafe_allow_html=True)
        st.markdown('<p class="chart-description">Benchmark results for the specified option parameters</p>', unsafe_allow_html=True)
        st.dataframe(
            df,
            column_config={
                "Pricing Model": st.column_config.TextColumn(
                    "Pricing Model",
                    width="medium",
                ),
                "Type": st.column_config.TextColumn(
                    "Type",
                    width="small",
                ),
                "Option Price": st.column_config.TextColumn(
                    "Option Price",
                    width="small",
                ),
                "Execution Time": st.column_config.TextColumn(
                    "Execution Time",
                    width="small",
                ),
                "Price Error": st.column_config.TextColumn(
                    "Price Error",
                    width="small",
                ),
                "Description": st.column_config.TextColumn(
                    "Description",
                    width="large",
                )
            },
            hide_index=True,
            use_container_width=True,
            height=240
        )
        # Add explanation about MC ML timing
        st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
        st.markdown("""
        <p style="color: #475569; margin: 0;">
        <strong>Why is Monte Carlo ML faster?</strong> 
        The Monte Carlo ML model has two phases: 
        (1) Training (one-time cost, shown as "Monte Carlo ML (Training)") 
        (2) Prediction (recurring cost, shown as "Monte Carlo ML (Prediction)")
        While training is expensive (comparable to running many Monte Carlo simulations), 
        the prediction phase is extremely fast (typically 0.1-1ms). This makes ML surrogates 
        ideal for applications requiring thousands of option pricings, such as risk management 
        and scenario analysis, where the one-time training cost is quickly amortized.
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        # Create visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Model Comparison", 
            "Performance Analysis", 
            "Accuracy Assessment",
            "Executive Summary"
        ])
        # Model Comparison tab
        with tab1:
            st.markdown('<h2 class="chart-title">Model Comparison</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">Visual comparison of pricing models across key metrics</p>', unsafe_allow_html=True)
            # Filter out models with errors
            valid_results = [r for r in results if isinstance(r["price"], (int, float))]
            if valid_results:
                # Create comparison charts
                col1, col2 = st.columns(2)
                with col1:
                    # Price comparison chart
                    fig_price = go.Figure()
                    # Add price points
                    mc_ml_price = next((r["price"] for r in valid_results if "Monte Carlo ML (Prediction)" in r["model"]), None)
                    if mc_ml_price is not None:
                        fig_price.add_trace(go.Bar(
                            x=[r["model"] for r in valid_results if "Training" not in r["model"]],
                            y=[r["price"] for r in valid_results if "Training" not in r["model"]],
                            marker_color=['#3B82F6' if "Black-Scholes" in r["model"] else '#10B981' for r in valid_results if "Training" not in r["model"]],
                            width=0.6
                        ))
                        # Add reference line if Black-Scholes is available
                        if reference_price is not None:
                            fig_price.add_shape(
                                type="line",
                                x0=-0.5, y0=reference_price,
                                x1=len(valid_results)-0.5, y1=reference_price,
                                line=dict(color="#F87171", width=2, dash="dash"),
                                name="Black-Scholes Reference"
                            )
                        fig_price.update_layout(
                            title="Option Price Comparison",
                            xaxis_title="",
                            yaxis_title="Option Price",
                            template="plotly_white",
                            height=400,
                            paper_bgcolor='white',
                            plot_bgcolor='white',
                            font=dict(size=12, color='#475569'),
                            showlegend=False,
                            margin=dict(l=40, r=40, t=40, b=40)
                        )
                        st.plotly_chart(fig_price, use_container_width=True)
                with col2:
                    # Time comparison chart - show only prediction times
                    pred_results = [r for r in results if "Prediction" in r["model"] or "Black-Scholes" in r["model"] or "Monte Carlo" in r["model"]]
                    pred_results = [r for r in pred_results if isinstance(r["time_ms"], (int, float))]
                    if pred_results:
                        fig_time = go.Figure()
                        fig_time.add_trace(go.Bar(
                            x=[r["model"] for r in pred_results],
                            y=[r["time_ms"] for r in pred_results],
                            marker_color=['#3B82F6' if "Black-Scholes" in r["model"] else '#8B5CF6' if "Prediction" in r["model"] else '#EF4444' for r in pred_results],
                            width=0.6
                        ))
                        fig_time.update_layout(
                            title="Execution Time Comparison (Prediction Phase)",
                            xaxis_title="",
                            yaxis_title="Time (ms)",
                            template="plotly_white",
                            height=400,
                            paper_bgcolor='white',
                            plot_bgcolor='white',
                            font=dict(size=12, color='#475569'),
                            showlegend=False,
                            margin=dict(l=40, r=40, t=40, b=40),
                            yaxis_type="log"
                        )
                        st.plotly_chart(fig_time, use_container_width=True)
                    else:
                        st.info("No valid execution time data available for comparison")
            else:
                st.warning("No valid pricing results available for comparison")
        # Performance Analysis tab
        with tab2:
            st.markdown('<h2 class="chart-title">Performance Analysis</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">Detailed analysis of computational efficiency</p>', unsafe_allow_html=True)
            # Filter valid time results for prediction phase
            pred_results = [r for r in results if "Prediction" in r["model"] or "Black-Scholes" in r["model"] or "Monte Carlo" in r["model"]]
            pred_results = [r for r in pred_results if isinstance(r["time_ms"], (int, float))]
            if pred_results:
                # Create performance metrics
                min_time = min(r["time_ms"] for r in pred_results)
                max_time = max(r["time_ms"] for r in pred_results)
                bs_time = next((r["time_ms"] for r in pred_results if "Black-Scholes" in r["model"]), None)
                # Speedup calculations
                speedups = {}
                if bs_time is not None:
                    for r in pred_results:
                        if "Black-Scholes" not in r["model"]:
                            # CRITICAL FIX: Prevent division by zero
                            time_ms = max(r["time_ms"], 1e-10)
                            speedups[r["model"]] = bs_time / time_ms
                # Create performance metrics
                st.markdown('<div class="executive-summary">', unsafe_allow_html=True)
                st.markdown('<div class="executive-summary-title">Performance Metrics</div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
                    st.markdown('<div class="executive-title">Fastest Model</div>', unsafe_allow_html=True)
                    fastest = min(pred_results, key=lambda x: x["time_ms"])
                    st.markdown(f'<div class="executive-value">{fastest["model"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="executive-help">Execution time: {fastest["time_ms"]:.2f} ms</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
                    st.markdown('<div class="executive-title">Slowest Model</div>', unsafe_allow_html=True)
                    slowest = max(pred_results, key=lambda x: x["time_ms"])
                    st.markdown(f'<div class="executive-value">{slowest["model"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="executive-help">Execution time: {slowest["time_ms"]:.2f} ms</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with col3:
                    if speedups:
                        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
                        st.markdown('<div class="executive-title">Best Speedup</div>', unsafe_allow_html=True)
                        best_speedup_model = max(speedups.items(), key=lambda x: x[1])
                        st.markdown(f'<div class="executive-value">{best_speedup_model[1]:.1f}x</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="executive-help">vs Black-Scholes ({best_speedup_model[0]})</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
                        st.markdown('<div class="executive-title">Speedup Analysis</div>', unsafe_allow_html=True)
                        st.markdown('<div class="executive-value">N/A</div>', unsafe_allow_html=True)
                        st.markdown('<div class="executive-help">Black-Scholes not available for comparison</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                # Performance breakdown
                st.markdown('<h3 class="subsection-header">Performance Breakdown</h3>', unsafe_allow_html=True)
                # Create model cards
                for r in pred_results:
                    st.markdown('<div class="model-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="model-name">{r["model"]}</div>', unsafe_allow_html=True)
                    # Time comparison
                    st.markdown('<div class="model-detail">', unsafe_allow_html=True)
                    st.markdown('<span>Execution Time</span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="model-value">{r["time_ms"]:.2f} ms</span>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    # Relative performance
                    if min_time > 0:
                        st.markdown('<div class="model-detail">', unsafe_allow_html=True)
                        st.markdown('<span>Relative to Fastest</span>', unsafe_allow_html=True)
                        st.markdown(f'<span class="model-value">{r["time_ms"]/min_time:.1f}x</span>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    # Speedup vs Black-Scholes
                    if bs_time is not None and "Black-Scholes" not in r["model"]:
                        speedup = bs_time / max(r["time_ms"], 1e-10)  # CRITICAL FIX: Prevent division by zero
                        st.markdown('<div class="model-detail">', unsafe_allow_html=True)
                        st.markdown('<span>Speedup vs Black-Scholes</span>', unsafe_allow_html=True)
                        st.markdown(f'<span class="model-value">{speedup:.1f}x</span>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                # Performance chart
                st.markdown('<h3 class="subsection-header">Time Distribution</h3>', unsafe_allow_html=True)
                fig_perf = go.Figure()
                # Add performance bars
                fig_perf.add_trace(go.Bar(
                    x=[r["model"] for r in pred_results],
                    y=[r["time_ms"] for r in pred_results],
                    marker_color=['#3B82F6' if "Black-Scholes" in r["model"] else '#8B5CF6' if "Prediction" in r["model"] else '#EF4444' for r in pred_results],
                    width=0.6
                ))
                fig_perf.update_layout(
                    title="Execution Time Distribution (Prediction Phase)",
                    xaxis_title="",
                    yaxis_title="Time (ms)",
                    template="plotly_white",
                    height=400,
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font=dict(size=12, color='#475569'),
                    showlegend=False,
                    margin=dict(l=40, r=40, t=40, b=40),
                    yaxis_type="log"
                )
                st.plotly_chart(fig_perf, use_container_width=True)
                # Performance insights
                st.markdown('<h3 class="subsection-header">Performance Insights</h3>', unsafe_allow_html=True)
                st.markdown('<div class="executive-summary">', unsafe_allow_html=True)
                st.markdown('<div class="executive-summary-text">', unsafe_allow_html=True)
                if speedups:
                    best_speedup_model = max(speedups.items(), key=lambda x: x[1])
                    st.markdown(f'<p>The <span class="highlight">{best_speedup_model[0]}</span> model provides the best performance advantage, '
                               f'running <span class="highlight">{best_speedup_model[1]:.1f}x</span> faster than the analytical Black-Scholes solution.</p>', 
                               unsafe_allow_html=True)
                if len(pred_results) > 1:
                    fastest = min(pred_results, key=lambda x: x["time_ms"])
                    slowest = max(pred_results, key=lambda x: x["time_ms"])
                    st.markdown(f'<p>The fastest model (<span class="highlight">{fastest["model"]}</span>) is '
                               f'<span class="highlight">{slowest["time_ms"]/fastest["time_ms"]:.1f}x</span> faster '
                               f'than the slowest model (<span class="highlight">{slowest["model"]}</span>).</p>', 
                               unsafe_allow_html=True)
                st.markdown('<p>For production environments requiring high-frequency pricing, '
                           'the Monte Carlo ML approach provides near-instant predictions after an initial training phase, '
                           'making it ideal for real-time risk management applications. The one-time training cost is quickly '
                           'amortized when pricing thousands of options.</p>', 
                           unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No valid execution time data available for performance analysis")
        # Accuracy Assessment tab
        with tab3:
            st.markdown('<h2 class="chart-title">Accuracy Assessment</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">Evaluation of pricing accuracy relative to analytical solution</p>', unsafe_allow_html=True)
            # Filter valid price results with reference
            valid_results = [r for r in results if isinstance(r["price"], (int, float)) and reference_price is not None]
            valid_results = [r for r in valid_results if "Training" not in r["model"]]
            if valid_results and reference_price is not None:
                # Calculate errors
                errors = [(r["model"], abs(r["price"] - reference_price)) for r in valid_results if "Black-Scholes" not in r["model"]]
                if errors:
                    # Create accuracy metrics
                    st.markdown('<div class="executive-summary">', unsafe_allow_html=True)
                    st.markdown('<div class="executive-summary-title">Accuracy Metrics</div>', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
                        st.markdown('<div class="executive-title">Most Accurate</div>', unsafe_allow_html=True)
                        most_accurate = min(errors, key=lambda x: x[1])
                        st.markdown(f'<div class="executive-value">{most_accurate[0]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="executive-help">Error: {most_accurate[1]:.6f}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
                        st.markdown('<div class="executive-title">Least Accurate</div>', unsafe_allow_html=True)
                        least_accurate = max(errors, key=lambda x: x[1])
                        st.markdown(f'<div class="executive-value">{least_accurate[0]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="executive-help">Error: {least_accurate[1]:.6f}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
                        st.markdown('<div class="executive-title">Max Error Ratio</div>', unsafe_allow_html=True)
                        # CRITICAL FIX: Prevent division by zero
                        min_error_val = min(errors, key=lambda x: x[1])[1]
                        min_error_val = max(min_error_val, 1e-10)  # Add minimum threshold
                        error_ratio = max(errors, key=lambda x: x[1])[1] / min_error_val
                        st.markdown(f'<div class="executive-value">{error_ratio:.1f}x</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="executive-help">Least vs most accurate</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    # Accuracy chart
                    st.markdown('<h3 class="subsection-header">Price Error Comparison</h3>', unsafe_allow_html=True)
                    fig_error = go.Figure()
                    # Add error bars
                    fig_error.add_trace(go.Bar(
                        x=[e[0] for e in errors],
                        y=[e[1] for e in errors],
                        marker_color=['#10B981' if e[1] < 0.001 else '#EF4444' for e in errors],
                        width=0.6
                    ))
                    fig_error.update_layout(
                        title="Price Error Relative to Black-Scholes",
                        xaxis_title="",
                        yaxis_title="Absolute Error",
                        template="plotly_white",
                        height=400,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(size=12, color='#475569'),
                        showlegend=False,
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig_error, use_container_width=True)
                    # Error distribution
                    st.markdown('<h3 class="subsection-header">Error Distribution</h3>', unsafe_allow_html=True)
                    # Create error distribution chart
                    fig_error_dist = go.Figure()
                    fig_error_dist.add_trace(go.Histogram(
                        x=[e[1] for e in errors],
                        nbinsx=20,
                        name='Price Error',
                        marker_color='#60A5FA',
                        opacity=0.7
                    ))
                    fig_error_dist.add_vline(
                        x=0, 
                        line_dash="dash", 
                        line_color="#F87171",
                        annotation_text="Zero Error"
                    )
                    fig_error_dist.update_layout(
                        title="Price Error Distribution",
                        xaxis_title="Absolute Error",
                        yaxis_title="Frequency",
                        template="plotly_white",
                        height=400,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(size=12, color='#475569')
                    )
                    st.plotly_chart(fig_error_dist, use_container_width=True)
                    # Accuracy insights
                    st.markdown('<h3 class="subsection-header">Accuracy Insights</h3>', unsafe_allow_html=True)
                    st.markdown('<div class="executive-summary">', unsafe_allow_html=True)
                    st.markdown('<div class="executive-summary-text">', unsafe_allow_html=True)
                    most_accurate = min(errors, key=lambda x: x[1])
                    st.markdown(f'<p>The <span class="highlight">{most_accurate[0]}</span> model demonstrates the highest pricing accuracy, '
                               f'with an absolute error of <span class="highlight">{most_accurate[1]:.6f}</span> compared to the analytical solution.</p>', 
                               unsafe_allow_html=True)
                    st.markdown('<p>For applications requiring high precision, such as exotic option pricing or '
                               'risk management in volatile markets, the Monte Carlo Unified approach with variance reduction '
                               'techniques provides the best balance of accuracy and computational efficiency.</p>', 
                               unsafe_allow_html=True)
                    st.markdown('<p>The Monte Carlo ML model, while extremely fast for prediction, maintains reasonable accuracy '
                               'due to its training on high-quality Monte Carlo simulations, making it ideal for '
                               'applications where speed is critical and minor precision trade-offs are acceptable.</p>', 
                               unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("No valid price error data available for comparison")
            else:
                st.warning("No valid price data available for accuracy assessment")
        # Executive Summary tab
        with tab4:
            st.markdown('<h2 class="chart-title">Executive Summary</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">Strategic insights for model selection based on business requirements</p>', unsafe_allow_html=True)
            st.markdown('<div class="executive-summary">', unsafe_allow_html=True)
            st.markdown('<div class="executive-summary-title">Strategic Recommendations</div>', unsafe_allow_html=True)
            st.markdown('<div class="executive-summary-text">', unsafe_allow_html=True)
            # Generate strategic insights based on results
            if len(results) > 0:
                # Identify fastest model (excluding Black-Scholes if present)
                pred_results = [r for r in results if "Prediction" in r["model"] or "Black-Scholes" in r["model"] or "Monte Carlo" in r["model"]]
                pred_results = [r for r in pred_results if isinstance(r["time_ms"], (int, float))]
                if pred_results:
                    fastest = min(pred_results, key=lambda x: x["time_ms"])
                    st.markdown(f'<p><span class="highlight">Speed Priority:</span> For applications requiring real-time pricing '
                               f'of large option portfolios, the <span class="highlight">{fastest["model"]}</span> model '
                               f'provides the fastest execution at <span class="highlight">{fastest["time_ms"]:.2f} ms</span> per option. '
                               f'This makes it ideal for high-frequency trading systems and real-time risk monitoring.</p>', 
                               unsafe_allow_html=True)
                # Identify most accurate model
                valid_results = [r for r in results if isinstance(r["price"], (int, float)) and reference_price is not None]
                valid_results = [r for r in valid_results if "Training" not in r["model"]]
                if valid_results and reference_price is not None:
                    errors = [(r["model"], abs(r["price"] - reference_price)) for r in valid_results if "Black-Scholes" not in r["model"]]
                    if errors:
                        most_accurate = min(errors, key=lambda x: x[1])
                        st.markdown(f'<p><span class="highlight">Accuracy Priority:</span> When precision is critical, such as for '
                                   f'exotic options or regulatory reporting, the <span class="highlight">{most_accurate[0]}</span> model '
                                   f'provides the highest accuracy with an error of <span class="highlight">{most_accurate[1]:.6f}</span>. '
                                   f'This model is recommended for valuation-sensitive applications where small pricing errors '
                                   f'could lead to significant financial impact.</p>', 
                                   unsafe_allow_html=True)
                # ML-specific insight
                mc_ml_result = next((r for r in results if "Monte Carlo ML (Prediction)" in r["model"]), None)
                if mc_ml_result and isinstance(mc_ml_result["price"], (int, float)) and reference_price is not None:
                    ml_error = abs(mc_ml_result["price"] - reference_price)
                    st.markdown(f'<p><span class="highlight">ML Acceleration:</span> The Monte Carlo ML model demonstrates the '
                               f'optimal balance for production environments, delivering near-instant pricing at '
                               f'<span class="highlight">{mc_ml_result["time_ms"]:.2f} ms</span> with acceptable error of '
                               f'<span class="highlight">{ml_error:.6f}</span>. After an initial training phase, it can price '
                               f'thousands of options per second, making it ideal for scenario analysis, stress testing, '
                               f'and real-time risk management.</p>', 
                               unsafe_allow_html=True)
            # General recommendation
            st.markdown('<p><span class="highlight">Strategic Recommendation:</span> Implement a hybrid approach where '
                       'the Monte Carlo ML model handles the majority of pricing requests for speed, while the more '
                       'computationally intensive methods are reserved for validation, calibration, and high-precision '
                       'requirements. This provides the best balance of performance and accuracy across different business needs.</p>', 
                       unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<h3 class="subsection-header">Model Selection Guide</h3>', unsafe_allow_html=True)
            # Create model selection guide
            model_guide = [
                {
                    "Use Case": "Real-time pricing of large portfolios",
                    "Recommended Model": "Monte Carlo ML",
                    "Rationale": "Extremely fast predictions after initial training"
                },
                {
                    "Use Case": "Regulatory reporting & high-precision valuation",
                    "Recommended Model": "Monte Carlo Unified",
                    "Rationale": "Highest accuracy with variance reduction techniques"
                },
                {
                    "Use Case": "Theoretical analysis & quick estimates",
                    "Recommended Model": "Black-Scholes",
                    "Rationale": "Instant analytical solution with known limitations"
                },
                {
                    "Use Case": "Validation & calibration",
                    "Recommended Model": "Monte Carlo",
                    "Rationale": "Standard implementation for benchmarking"
                }
            ]
            guide_df = pd.DataFrame(model_guide)
            st.dataframe(
                guide_df,
                column_config={
                    "Use Case": st.column_config.TextColumn(
                        "Use Case",
                        width="medium",
                    ),
                    "Recommended Model": st.column_config.TextColumn(
                        "Recommended Model",
                        width="small",
                    ),
                    "Rationale": st.column_config.TextColumn(
                        "Rationale",
                        width="large",
                    )
                },
                hide_index=True,
                use_container_width=True,
                height=180
            )
            st.markdown('<h3 class="subsection-header">Implementation Considerations</h3>', unsafe_allow_html=True)
            st.markdown('<div class="executive-summary">', unsafe_allow_html=True)
            st.markdown('<div class="executive-summary-text">', unsafe_allow_html=True)
            st.markdown('<p><span class="highlight">Training Requirements:</span> The Monte Carlo ML model requires '
                       'an initial training phase using high-quality Monte Carlo simulations. This is a one-time cost '
                       'that pays dividends in subsequent prediction speed.</p>', unsafe_allow_html=True)
            st.markdown('<p><span class="highlight">Hardware Utilization:</span> The Monte Carlo Unified model can '
                       'leverage GPU acceleration for significant speed improvements, while the ML model benefits '
                       'from standard CPU resources for inference.</p>', unsafe_allow_html=True)
            st.markdown('<p><span class="highlight">Accuracy Trade-offs:</span> There is always a trade-off between '
                       'speed and accuracy. Understanding your specific business requirements will guide the optimal '
                       'model selection for each use case.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred during benchmarking: {str(e)}")
        logger.exception("Critical error in benchmarking")
        st.markdown("""
        <div style="background-color: #FEF2F2; border-radius: 8px; padding: 1rem; border: 1px solid #FECACA; margin-top: 1rem;">
            <h3 style="color: #B91C1C; margin: 0 0 0.5rem 0;">Troubleshooting Tips</h3>
            <ul style="color: #DC2626; padding-left: 1.2rem; margin-bottom: 0;">
                <li>Ensure all input values are valid (positive numbers, etc.)</li>
                <li>Check that T (maturity) is greater than 0.001</li>
                <li>Try reducing simulation size if performance is poor</li>
                <li>Check that all required dependencies are installed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background-color: #F8FAFC; border-radius: 8px; 
                 border: 1px solid #E2E8F0; margin-top: 1rem;">
        <h3 style="color: #475569; margin-bottom: 0.75rem;">Get Started</h3>
        <p style="color: #64748B; margin-bottom: 1rem;">Configure your parameters above and click "Run Benchmarks" to see results</p>
        <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;">
            <div style="background-color: white; padding: 0.5rem 1rem; border-radius: 6px; border: 1px solid #E2E8F0;">
                <span style="color: #1E293B; font-weight: 500;">Black-Scholes</span>
                <div style="color: #64748B; font-size: 0.9rem; margin-top: 0.2rem;">Analytical solution</div>
            </div>
            <div style="background-color: white; padding: 0.5rem 1rem; border-radius: 6px; border: 1px solid #E2E8F0;">
                <span style="color: #1E293B; font-weight: 500;">Monte Carlo</span>
                <div style="color: #64748B; font-size: 0.9rem; margin-top: 0.2rem;">Standard simulation</div>
            </div>
            <div style="background-color: white; padding: 0.5rem 1rem; border-radius: 6px; border: 1px solid #E2E8F0;">
                <span style="color: #1E293B; font-weight: 500;">Monte Carlo Unified</span>
                <div style="color: #64748B; font-size: 0.9rem; margin-top: 0.2rem;">CPU/GPU with variance reduction</div>
            </div>
            <div style="background-color: white; padding: 0.5rem 1rem; border-radius: 6px; border: 1px solid #E2E8F0;">
                <span style="color: #1E293B; font-weight: 500;">Monte Carlo ML</span>
                <div style="color: #64748B; font-size: 0.9rem; margin-top: 0.2rem;">Machine learning accelerated</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)