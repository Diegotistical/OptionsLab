import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st
from plotly.subplots import make_subplots

# ======================
# CONFIGURATION
# ======================
logger = logging.getLogger("financial_analytics")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ======================
# DARK MODE STYLING
# ======================
st.markdown(
    """
<style>
    /* Base styling - full width dark theme */
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
    /* Full width containers */
    .stApp {
        max-width: 100% !important;
        padding: 0 1rem !important;
        background-color: #0f172a;
    }
    /* Metric cards */
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    /* Button styling - Full width */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.8rem 2rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        width: 100% !important;
        margin: 0.5rem 0 !important;
        max-width: 100% !important;
    }
    .stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    /* Input sections */
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
    /* Section headers */
    .subsection-header {
        font-size: 1.3rem;
        color: #f8fafc;
        margin: 1.2rem 0 0.8rem 0;
        font-weight: 600;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.5rem;
    }
    /* Executive insights */
    .executive-insight {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #334155;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
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
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: #1e293b;
        border-radius: 8px 8px 0px 0px;
        gap: 1rem;
        padding: 0.5rem 1rem;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
    /* Navigation styling */
    .nav-container {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 2rem;
        border: 1px solid #334155;
    }
    .nav-button {
        background-color: #334155;
        border: none;
        color: #e2e8f0;
        padding: 1rem 2rem;
        border-radius: 8px;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    .nav-button:hover {
        background-color: #3b82f6;
        transform: translateY(-2px);
    }
    .nav-button.active {
        background-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ======================
# CORE FUNCTIONS - SHARED
# ======================


def timeit_ms(fn, *args, **kwargs) -> Tuple[Any, float]:
    """Measure execution time in milliseconds"""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000.0
    return result, elapsed


def compute_var_es(
    returns: pd.Series, level: float = 0.95
) -> Tuple[Optional[float], Optional[float]]:
    """Compute Value at Risk and Expected Shortfall with robust error handling"""
    try:
        if returns.empty or len(returns) < 10:
            return None, None

        returns = pd.to_numeric(returns, errors="coerce").dropna()
        if len(returns) < 10:
            return None, None

        var = -np.quantile(returns, 1 - level)
        losses_beyond_var = returns[returns < -var]
        es = -losses_beyond_var.mean() if len(losses_beyond_var) > 0 else -returns.min()

        return float(var), float(es)

    except Exception as e:
        st.error(f"Error computing VaR/ES: {str(e)}")
        return None, None


def generate_synthetic_returns(
    n: int, mu: float, sigma: float, distribution: str = "normal", seed: int = 42
) -> pd.Series:
    """Generate synthetic returns with different distributions"""
    rng = np.random.default_rng(seed)

    if distribution == "normal":
        returns = rng.normal(mu, sigma, n)
    elif distribution == "student_t":
        returns = rng.standard_t(4, n) * sigma / np.sqrt(4) + mu
    elif distribution == "skewed":
        from scipy.stats import skewnorm

        returns = skewnorm.rvs(5, loc=mu, scale=sigma, size=n, random_state=seed)
    else:
        returns = rng.normal(mu, sigma, n)

    return pd.Series(returns, name="synthetic_returns")


# ======================
# OPTION PRICING FUNCTIONS
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
            price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(
                -r * T
            ) * norm.cdf(d2)
        else:
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(
                -q * T
            ) * norm.cdf(-d1)
        return float(price)
    except Exception as e:
        logger.error(f"Black-Scholes fallback failed: {str(e)}")
        return 0.0


def fallback_monte_carlo(
    S, K, T, r, sigma, option_type, q=0.0, num_sim=50000, num_steps=100, seed=42
):
    """Fallback implementation of Monte Carlo pricing"""
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
        logger.error(f"Monte Carlo fallback failed: {str(e)}")
        return 0.0


# --- NEW: Binomial Tree Fallback ---
def fallback_binomial_tree(
    S,
    K,
    T,
    r,
    sigma,
    option_type="call",
    exercise_style="european",
    q=0.0,
    num_steps=500,
):
    """Fallback implementation replicating the BinomialTree.price logic"""
    try:
        # --- Replicate validation logic ---
        if not (
            isinstance(S, (int, float))
            and isinstance(K, (int, float))
            and isinstance(T, (int, float))
            and isinstance(r, (int, float))
            and isinstance(sigma, (int, float))
            and isinstance(q, (int, float))
        ):
            logger.error("Binomial Tree fallback: Inputs must be numeric.")
            return 0.0
        if S <= 0 or K <= 0:
            logger.error("Binomial Tree fallback: Spot/strike must be positive.")
            return 0.0
        if T < 0 or sigma < 0 or q < 0:
            logger.error("Binomial Tree fallback: T/sigma/q must be non-negative.")
            return 0.0
        if option_type not in {"call", "put"}:
            logger.error("Binomial Tree fallback: option_type must be 'call' or 'put'.")
            return 0.0
        if exercise_style not in {"european", "american"}:
            logger.error(
                "Binomial Tree fallback: exercise_style must be 'european' or 'american'."
            )
            return 0.0
        if num_steps <= 0:
            logger.error("Binomial Tree fallback: num_steps must be positive.")
            return 0.0

        # Handle edge cases
        if T == 0:
            if option_type == "call":
                return float(max(S - K, 0.0))
            else:  # put
                return float(max(K - S, 0.0))
        if sigma == 0:
            df = np.exp(-r * T)
            fwd = S * np.exp((r - q) * T)
            if option_type == "call":
                intrinsic = max(fwd - K, 0.0)
            else:  # put
                intrinsic = max(K - fwd, 0.0)
            return float(intrinsic * df)

        # --- Compute tree parameters ---
        dt = T / num_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        p = min(max(p, 0.0), 1.0)  # Clamp probability

        # --- Build asset price tree ---
        asset_prices = np.empty((num_steps + 1, num_steps + 1), dtype=np.float64)
        for i in range(num_steps + 1):
            j = np.arange(i + 1)
            asset_prices[i, : i + 1] = S * (u**j) * (d ** (i - j))

        # --- Backward induction ---
        disc = np.exp(-r * dt)
        option_values = np.empty_like(asset_prices)

        # Terminal payoffs
        if option_type == "call":
            option_values[-1, : num_steps + 1] = np.maximum(
                asset_prices[-1, : num_steps + 1] - K, 0
            )
        else:  # put
            option_values[-1, : num_steps + 1] = np.maximum(
                K - asset_prices[-1, : num_steps + 1], 0
            )

        # Backward induction loop
        for step in range(num_steps - 1, -1, -1):
            option_values[step, : step + 1] = disc * (
                p * option_values[step + 1, 1 : step + 2]
                + (1 - p) * option_values[step + 1, : step + 1]
            )
            # American early exercise
            if exercise_style == "american":
                if option_type == "call":
                    intrinsic = np.maximum(asset_prices[step, : step + 1] - K, 0)
                else:  # put
                    intrinsic = np.maximum(K - asset_prices[step, : step + 1], 0)
                option_values[step, : step + 1] = np.maximum(
                    option_values[step, : step + 1], intrinsic
                )

        return float(option_values[0, 0])
    except Exception as e:
        logger.error(f"Binomial Tree fallback failed: {str(e)}")
        return 0.0


# ======================
# PLOTTING FUNCTIONS
# ======================


def create_option_pricing_chart(results: List[Dict]) -> go.Figure:
    """Create option pricing comparison chart"""
    fig = go.Figure()

    valid_results = [
        r
        for r in results
        if isinstance(r.get("price"), (int, float)) and "Training" not in r["model"]
    ]

    if valid_results:
        fig.add_trace(
            go.Bar(
                x=[r["model"] for r in valid_results],
                y=[r["price"] for r in valid_results],
                marker_color=[
                    (
                        "#3b82f6"
                        if "Black-Scholes" in r["model"]
                        else "#10b981" if "Binomial Tree" in r["model"] else "#ef4444"
                    )
                    for r in valid_results
                ],  # Added color for Binomial Tree
                text=[f"${r['price']:.4f}" for r in valid_results],
                textposition="auto",
            )
        )

    fig.update_layout(
        title="Option Pricing Comparison",
        xaxis_title="Pricing Model",
        yaxis_title="Option Price",
        template="plotly_dark",
        height=400,
        showlegend=False,
    )

    return fig


def create_performance_chart(results: List[Dict]) -> go.Figure:
    """Create performance comparison chart"""
    fig = go.Figure()

    # Include all valid models for performance comparison, not just 'Prediction'
    valid_results = [r for r in results if isinstance(r.get("time_ms"), (int, float))]

    if valid_results:
        fig.add_trace(
            go.Bar(
                x=[r["model"] for r in valid_results],
                y=[r["time_ms"] for r in valid_results],
                marker_color=[
                    (
                        "#3b82f6"
                        if "Black-Scholes" in r["model"]
                        else "#10b981" if "Binomial Tree" in r["model"] else "#8b5cf6"
                    )
                    for r in valid_results
                ],  # Added color for Binomial Tree
                text=[f"{r['time_ms']:.1f}ms" for r in valid_results],
                textposition="auto",
            )
        )

    fig.update_layout(
        title="Model Performance (Execution Time)",
        xaxis_title="Pricing Model",
        yaxis_title="Execution Time (ms)",
        template="plotly_dark",
        height=400,
        yaxis_type="log",
    )

    return fig


def create_risk_distribution_plot(
    returns: pd.Series, var: float, es: float, level: float
) -> go.Figure:
    """Create risk distribution plot"""
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=returns, nbinsx=50, name="Returns", opacity=0.7, marker_color="#3b82f6"
        )
    )

    if var is not None:
        fig.add_vline(
            x=-var,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"VaR {level*100:.1f}%",
        )

    if es is not None:
        fig.add_vline(
            x=-es,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=f"ES {level*100:.1f}%",
        )

    fig.update_layout(
        title="Return Distribution with Risk Metrics",
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=400,
    )

    return fig


# ======================
# MAIN APPLICATION
# ======================

st.set_page_config(
    page_title="Financial Analytics Suite",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    '<h1 class="main-header">Financial Analytics Suite</h1>', unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">Professional-grade option pricing and risk analysis platform</p>',
    unsafe_allow_html=True,
)

# Navigation
st.markdown('<div class="nav-container">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    option_benchmark_btn = st.button(
        "Option Pricing Benchmarks", use_container_width=True
    )
with col2:
    risk_analysis_btn = st.button("Risk Analysis Dashboard", use_container_width=True)
with col3:
    market_insights_btn = st.button(
        "Market Insights", use_container_width=True, disabled=True
    )
st.markdown("</div>", unsafe_allow_html=True)

# Initialize session state for navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "option_benchmark"

if option_benchmark_btn:
    st.session_state.current_page = "option_benchmark"
if risk_analysis_btn:
    st.session_state.current_page = "risk_analysis"

# ======================
# OPTION PRICING BENCHMARK PAGE
# ======================

if st.session_state.current_page == "option_benchmark":
    st.markdown(
        '<div class="subsection-header"> Option Pricing Model Benchmark</div>',
        unsafe_allow_html=True,
    )

    # Model selection
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown(
        '<div class="engine-label">Select Pricing Models</div>', unsafe_allow_html=True
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        include_bs = st.checkbox("Black-Scholes", value=True)
    with col2:
        include_mc = st.checkbox("Monte Carlo", value=True)
    with col3:
        include_mc_advanced = st.checkbox("Advanced MC", value=True)
    with col4:
        include_ml = st.checkbox("ML Pricing", value=True)
    # --- ADD Binomial Tree Checkbox ---
    with col1:  # Can add to any column, using col1 here
        include_bt = st.checkbox("Binomial Tree", value=True)
    # --- END ADD ---
    st.markdown("</div>", unsafe_allow_html=True)

    # Pricing parameters
    st.markdown(
        '<div class="subsection-header">Parameters</div>', unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="engine-option">', unsafe_allow_html=True)
        st.markdown(
            '<div class="engine-label">Spot Price (S)</div>', unsafe_allow_html=True
        )
        S = st.number_input("", 50.0, 200.0, 100.0, key="spot_price")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="engine-option">', unsafe_allow_html=True)
        st.markdown(
            '<div class="engine-label">Strike Price (K)</div>', unsafe_allow_html=True
        )
        K = st.number_input("", 50.0, 200.0, 100.0, key="strike_price")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="engine-option">', unsafe_allow_html=True)
        st.markdown(
            '<div class="engine-label">Maturity (T)</div>', unsafe_allow_html=True
        )
        T = st.number_input("", 0.1, 5.0, 1.0, key="maturity")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="engine-option">', unsafe_allow_html=True)
        st.markdown(
            '<div class="engine-label">Risk-free Rate (r)</div>', unsafe_allow_html=True
        )
        r = st.number_input("", 0.0, 0.1, 0.05, key="risk_free")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="engine-option">', unsafe_allow_html=True)
        st.markdown(
            '<div class="engine-label">Volatility (σ)</div>', unsafe_allow_html=True
        )
        sigma = st.number_input("", 0.1, 0.8, 0.2, key="volatility")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="engine-option">', unsafe_allow_html=True)
        st.markdown(
            '<div class="engine-label">Option Type</div>', unsafe_allow_html=True
        )
        option_type = st.selectbox("", ["call", "put"], key="option_type")
        st.markdown("</div>", unsafe_allow_html=True)

    # Run benchmark
    if st.button("🚀 Run Pricing Benchmark", use_container_width=True):
        with st.spinner("Running pricing benchmarks..."):
            results = []

            # Get pricing models
            try:
                from src.pricing_models.binomial_tree import BinomialTree

                binomial_model = BinomialTree(
                    num_steps=500
                )  # Using default steps as per original
                logger.info("Successfully imported BinomialTree")
            except ImportError as e:
                logger.warning(f"BinomialTree import failed: {str(e)}")
                binomial_model = None

            # Black-Scholes
            if include_bs:
                price, latency = timeit_ms(
                    fallback_black_scholes, S, K, T, r, sigma, option_type
                )
                results.append(
                    {
                        "model": "Black-Scholes",
                        "price": price,
                        "time_ms": latency,
                        "type": "Analytical",
                    }
                )

            # Monte Carlo
            if include_mc:
                price, latency = timeit_ms(
                    fallback_monte_carlo, S, K, T, r, sigma, option_type, num_sim=10000
                )
                results.append(
                    {
                        "model": "Monte Carlo (Basic)",
                        "price": price,
                        "time_ms": latency,
                        "type": "Simulation",
                    }
                )

            # Advanced Monte Carlo
            if include_mc_advanced:
                price, latency = timeit_ms(
                    fallback_monte_carlo,
                    S,
                    K,
                    T,
                    r,
                    sigma,
                    option_type,
                    num_sim=50000,
                    num_steps=100,
                )
                results.append(
                    {
                        "model": "Monte Carlo (Advanced)",
                        "price": price,
                        "time_ms": latency,
                        "type": "Simulation",
                    }
                )

            # --- NEW: Binomial Tree Benchmark ---
            if include_bt:
                try:
                    if binomial_model is not None:
                        price, latency = timeit_ms(
                            binomial_model.price,
                            S,
                            K,
                            T,
                            r,
                            sigma,
                            option_type,
                            "european",
                            q=0.0,
                        )
                    else:
                        # Fallback implementation uses european by default
                        price, latency = timeit_ms(
                            fallback_binomial_tree,
                            S,
                            K,
                            T,
                            r,
                            sigma,
                            option_type,
                            "european",
                            q=0.0,
                            num_steps=500,  # Using default steps
                        )
                    results.append(
                        {
                            "model": "Binomial Tree (Eur)",
                            "price": price,
                            "time_ms": latency,
                            "type": "Lattice",
                        }
                    )
                except Exception as e:
                    logger.error(f"Binomial Tree benchmark failed: {str(e)}")
                    # Optionally add an error result, or just skip
                    st.error(f"Binomial Tree benchmark failed: {str(e)}")
            # --- END NEW ---

            # ML Pricing (simulated)
            if include_ml:
                # Simulate ML pricing being faster but slightly different
                bs_price = fallback_black_scholes(S, K, T, r, sigma, option_type)
                ml_price = bs_price * (
                    1 + np.random.normal(0, 0.01)
                )  # Small random variation
                results.append(
                    {
                        "model": "ML Pricing",
                        "price": ml_price,
                        "time_ms": 1.5,  # Very fast
                        "type": "Machine Learning",
                    }
                )

            # Store results in session state
            st.session_state.option_results = results

            st.success("✅ Benchmark completed!")

    # Display results if available
    if "option_results" in st.session_state:
        results = st.session_state.option_results

        # Results overview
        st.markdown(
            '<div class="subsection-header">Results</div>', unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            fig_price = create_option_pricing_chart(results)
            st.plotly_chart(fig_price, use_container_width=True)
        with col2:
            fig_perf = create_performance_chart(results)
            st.plotly_chart(fig_perf, use_container_width=True)

        # Detailed results table
        st.markdown(
            '<div class="subsection-header">Detailed Results</div>',
            unsafe_allow_html=True,
        )

        results_df = pd.DataFrame(results)
        results_df["Price"] = results_df["price"].apply(
            lambda x: f"${x:.4f}" if isinstance(x, (int, float)) else "N/A"
        )
        results_df["Time"] = results_df["time_ms"].apply(
            lambda x: f"{x:.2f} ms" if isinstance(x, (int, float)) else "N/A"
        )

        st.dataframe(
            results_df[["model", "type", "Price", "Time"]],
            use_container_width=True,
            hide_index=True,
        )

# ======================
# RISK ANALYSIS PAGE
# ======================

elif st.session_state.current_page == "risk_analysis":
    st.markdown(
        '<div class="subsection-header">Risk Analysis Dashboard</div>',
        unsafe_allow_html=True,
    )

    # Data source selection
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Data Source</div>', unsafe_allow_html=True)
    data_source = st.radio(
        "",
        ["Generate Synthetic", "Upload CSV", "Use Option Results"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Confidence level
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown(
        '<div class="engine-label">Confidence Level</div>', unsafe_allow_html=True
    )
    confidence_level = st.slider(
        "", 0.80, 0.995, 0.95, 0.005, label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Data generation/upload
    if data_source == "Generate Synthetic":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_days = st.number_input("Sample Size (days)", 100, 5000, 1000)
        with col2:
            mu_return = st.number_input("Mean Return", -0.01, 0.01, 0.0005, 0.0001)
        with col3:
            sigma_return = st.number_input("Volatility", 0.001, 0.1, 0.02, 0.001)

        returns_series = generate_synthetic_returns(n_days, mu_return, sigma_return)

    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload returns CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                returns_series = df[numeric_cols[0]].dropna()
            else:
                st.error("No numeric columns found")
                returns_series = generate_synthetic_returns(1000, 0.0005, 0.02)
        else:
            returns_series = generate_synthetic_returns(1000, 0.0005, 0.02)

    else:  # Use Option Results
        if "option_results" in st.session_state:
            # Generate returns based on option pricing results volatility
            results = st.session_state.option_results
            if results:
                # Use the volatility from option pricing as base for returns
                base_vol = sigma if "sigma" in locals() else 0.2
                returns_series = generate_synthetic_returns(1000, 0.0005, base_vol)
                st.info("Using volatility from option pricing analysis")
            else:
                returns_series = generate_synthetic_returns(1000, 0.0005, 0.02)
                st.warning("No option results found, using default parameters")
        else:
            returns_series = generate_synthetic_returns(1000, 0.0005, 0.02)
            st.warning("No option results found, using default parameters")

    # Compute risk metrics
    var, es = compute_var_es(returns_series, confidence_level)
    metrics = {
        "mean": returns_series.mean(),
        "volatility": returns_series.std(),
        "sharpe": (
            returns_series.mean() / returns_series.std()
            if returns_series.std() > 1e-10
            else 0
        ),
        "skewness": returns_series.skew(),
        "kurtosis": returns_series.kurtosis(),
        "var_95": -np.quantile(returns_series, 0.05),
        "var_99": -np.quantile(returns_series, 0.01),
    }

    # Risk dashboard
    st.markdown(
        '<div class="subsection-header">Risk Metrics</div>', unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
        st.markdown(
            '<div class="executive-title">Value at Risk</div>', unsafe_allow_html=True
        )
        var_display = f"{-var:.4%}" if var is not None else "N/A"
        st.markdown(
            f'<div class="executive-value" style="color: #ef4444;">{var_display}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="executive-title">({confidence_level*100:.1f}% confidence)</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
        st.markdown(
            '<div class="executive-title">Expected Shortfall</div>',
            unsafe_allow_html=True,
        )
        es_display = f"{-es:.4%}" if es is not None else "N/A"
        st.markdown(
            f'<div class="executive-value" style="color: #dc2626;">{es_display}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="executive-title">Average tail loss</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
        st.markdown(
            '<div class="executive-title">Daily Volatility</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="executive-value">{returns_series.std():.4%}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="executive-title">Standard deviation</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
        st.markdown(
            '<div class="executive-title">Sharpe Ratio</div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="executive-value">{metrics["sharpe"]:.3f}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="executive-title">Risk-adjusted return</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Risk visualization
    st.markdown(
        '<div class="subsection-header">Risk Analysis</div>', unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs(
        ["📊 Distribution", "⏰ Time Series", "📈 Advanced Metrics"]
    )

    with tab1:
        fig_risk = create_risk_distribution_plot(
            returns_series, var, es, confidence_level
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with tab2:
        # Cumulative returns plot
        fig_ts = go.Figure()
        cumulative_returns = (1 + returns_series).cumprod() - 1
        fig_ts.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                mode="lines",
                name="Cumulative Returns",
                line=dict(color="#10b981"),
            )
        )
        fig_ts.update_layout(
            title="Cumulative Returns Over Time", template="plotly_dark", height=400
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            # Statistical metrics
            st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
            st.markdown(
                '<div class="executive-title">Statistical Metrics</div>',
                unsafe_allow_html=True,
            )
            stats_data = {
                "Mean Return": f"{metrics['mean']:.4%}",
                "Skewness": f"{metrics['skewness']:.3f}",
                "Kurtosis": f"{metrics['kurtosis']:.3f}",
                "VaR 95%": f"{metrics['var_95']:.4%}",
                "VaR 99%": f"{metrics['var_99']:.4%}",
            }
            for key, value in stats_data.items():
                st.markdown(
                    f'<div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<span style="color: #94a3b8;">{key}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<span style="color: #f8fafc;">{value}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            # Worst losses
            st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
            st.markdown(
                '<div class="executive-title">Worst Daily Losses</div>',
                unsafe_allow_html=True,
            )
            worst_losses = returns_series.nsmallest(5)
            for i, loss in enumerate(worst_losses, 1):
                st.markdown(
                    f'<div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<span style="color: #94a3b8;">#{i}</span>', unsafe_allow_html=True
                )
                st.markdown(
                    f'<span style="color: #ef4444;">{loss:.4%}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
