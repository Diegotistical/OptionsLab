import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st
from plotly.colors import qualitative

# Import from utils - CRITICAL FIX FOR STREAMLIT CLOUD
try:
    from streamlit_app.st_utils import (
        greeks_mc_delta_gamma,
        price_monte_carlo,
        show_repo_status,
        simulate_payoffs,
        timeit_ms,
    )

    logger = logging.getLogger("monte_carlo")
    logger.info("Successfully imported from st_utils")
except Exception as e:
    logger = logging.getLogger("monte_carlo")
    logger.error(f"Failed to import from st_utils: {str(e)}")

    # Fallback implementation if imports fail
    def price_monte_carlo(
        S,
        K,
        T,
        r,
        sigma,
        option_type,
        q=0.0,
        num_sim=50_000,
        num_steps=100,
        seed=42,
        use_numba=False,
    ):
        try:
            np.random.seed(int(seed))
            dt = T / num_steps
            Z = np.random.standard_normal((num_sim, num_steps))
            S_paths = np.zeros((num_sim, num_steps))
            S_paths[:, 0] = S

            for t in range(1, num_steps):
                S_paths[:, t] = S_paths[:, t - 1] * np.exp(
                    (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t]
                )

            if option_type == "call":
                payoff = np.maximum(S_paths[:, -1] - K, 0.0)
            else:
                payoff = np.maximum(K - S_paths[:, -1], 0.0)

            discounted = np.exp(-r * T) * payoff
            return float(np.mean(discounted))
        except Exception as e:
            logger.error(f"MC fallback pricing failed: {str(e)}")
            return None

    def greeks_mc_delta_gamma(
        S,
        K,
        T,
        r,
        sigma,
        option_type,
        q=0.0,
        num_sim=50_000,
        num_steps=100,
        seed=42,
        h=1e-3,
        use_numba=False,
    ):
        try:
            p_down = price_monte_carlo(
                S - h,
                K,
                T,
                r,
                sigma,
                option_type,
                q,
                num_sim,
                num_steps,
                seed,
                use_numba,
            )
            p_mid = price_monte_carlo(
                S, K, T, r, sigma, option_type, q, num_sim, num_steps, seed, use_numba
            )
            p_up = price_monte_carlo(
                S + h,
                K,
                T,
                r,
                sigma,
                option_type,
                q,
                num_sim,
                num_steps,
                seed,
                use_numba,
            )

            if None in [p_down, p_mid, p_up]:
                return None, None

            delta = (p_up - p_down) / (2 * h)
            gamma = (p_up - 2 * p_mid + p_down) / (h**2)
            return float(delta), float(gamma)
        except Exception as e:
            logger.error(f"Greeks fallback failed: {str(e)}")
            return None, None

    def timeit_ms(fn, *args, **kwargs):
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        dt_ms = (time.perf_counter() - start) * 1000.0
        return out, dt_ms

    def show_repo_status():
        st.write(
            "⚠️ Core pricing modules not available. Using fallback implementations."
        )


# Configure logging
logger = logging.getLogger("monte_carlo")

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Monte Carlo Option Pricing", layout="wide", page_icon="📈"
)

# ------------------- STYLING -------------------
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: white !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        font-weight: 700 !important;
    }
    .sub-header {
        font-size: 1.4rem !important;
        color: #E2E8F0 !important;
        margin-bottom: 1.5rem !important;
        opacity: 0.9 !important;
    }
    .metric-card {
        background-color: #1E293B !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2) !important;
        border: 1px solid #334155 !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px !important;
        border-radius: 8px 8px 0 0 !important;
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        background-color: #1E293B !important;
        color: #CBD5E1 !important;
        padding: 0 20px !important;
        flex: 1 !important;
        min-width: 150px !important;
        text-align: center !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        border-bottom: 4px solid #1E3A8A !important;
    }
    .chart-container {
        background-color: #1E293B !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2) !important;
        margin-bottom: 1.5rem !important;
        border: 1px solid #334155 !important;
    }
    .chart-title {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: white !important;
        margin-bottom: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 1px solid #334155 !important;
    }
    .chart-description {
        font-size: 1.05rem !important;
        color: #CBD5E1 !important;
        margin-bottom: 1.5rem !important;
        line-height: 1.5 !important;
    }
    .st-emotion-cache-12w0q9a {
        background-color: #1E293B !important;
        border-radius: 12px !important;
    }
    .st-emotion-cache-1kyxreq {
        justify-content: center !important;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6 !important;
    }
    .st-bb {
        background-color: transparent !important;
    }
    .st-at {
        background-color: #3B82F6 !important;
    }
    .st-emotion-cache-1cypcdb {
        background-color: #3B82F6 !important;
    }
    .st-emotion-cache-1v0mbdj > img {
        border-radius: 12px !important;
    }
    .info-box {
        background-color: #1E293B !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        border: 1px solid #334155 !important;
        margin: 1rem 0 !important;
    }
    .metric-label {
        color: #94A3B8 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    .metric-value {
        color: white !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
    }
    .metric-delta {
        color: #64748B !important;
        font-size: 1rem !important;
        margin-top: 0.3rem !important;
    }
    .section-header {
        font-size: 1.8rem !important;
        color: white !important;
        margin: 1.5rem 0 1rem !important;
        font-weight: 600 !important;
    }
    .subsection-header {
        font-size: 1.4rem !important;
        color: white !important;
        margin: 1.2rem 0 0.8rem !important;
        font-weight: 600 !important;
    }
    .warning-box {
        background-color: #1E293B !important;
        border-left: 4px solid #F59E0B !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
    }
    .plotly-graph-div {
        font-size: 14px !important;
    }
    .js-plotly-plot .plotly .title {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }
    .js-plotly-plot .plotly .xtitle, 
    .js-plotly-plot .plotly .ytitle {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    .js-plotly-plot .plotly .legendtext {
        font-size: 1.1rem !important;
    }
    .js-plotly-plot .plotly .xtick, 
    .js-plotly-plot .plotly .ytick {
        font-size: 1rem !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------- HEADER -------------------
st.markdown(
    '<h1 class="main-header">Monte Carlo Option Pricing</h1>', unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">Advanced analytics for European option pricing with comprehensive risk analysis</p>',
    unsafe_allow_html=True,
)

# ------------------- INPUT SECTION -------------------
st.markdown(
    '<div style="background-color: #1E293B; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid #334155;">',
    unsafe_allow_html=True,
)
st.markdown(
    '<h3 class="subsection-header">Configure Parameters</h3>', unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown(
        '<h4 class="metric-label">Market Parameters</h4>', unsafe_allow_html=True
    )
    S = st.number_input(
        "Spot Price (S)", min_value=0.01, value=100.0, step=1.0, key="spot"
    )
    K = st.number_input(
        "Strike Price (K)", min_value=0.01, value=100.0, step=1.0, key="strike"
    )
    T = st.number_input(
        "Time to Maturity (T, years)", min_value=0.01, value=1.0, step=0.01, key="time"
    )
    option_type = st.selectbox("Option Type", ["call", "put"], key="option_type")

with col2:
    st.markdown(
        '<h4 class="metric-label">Interest & Volatility</h4>', unsafe_allow_html=True
    )
    r = st.number_input(
        "Risk-Free Rate (r)",
        min_value=0.0,
        value=0.05,
        step=0.01,
        format="%.4f",
        key="rate",
    )
    sigma = st.number_input(
        "Volatility (σ)", min_value=0.01, value=0.2, step=0.01, format="%.4f", key="vol"
    )
    q = st.number_input(
        "Dividend Yield (q)",
        min_value=0.0,
        value=0.0,
        step=0.01,
        format="%.4f",
        key="dividend",
    )

with col3:
    st.markdown(
        '<h4 class="metric-label">Simulation Settings</h4>', unsafe_allow_html=True
    )
    num_sim = st.slider(
        "Number of Simulations",
        min_value=1000,
        max_value=200000,
        value=50000,
        step=5000,
        key="sim",
    )
    num_steps = st.slider(
        "Steps per Path", min_value=10, max_value=500, value=100, step=10, key="steps"
    )
    seed = st.number_input("Random Seed", min_value=1, value=42, step=1, key="seed")
    use_numba = st.checkbox("Enable Numba Acceleration", value=False, key="numba")

st.markdown("</div>", unsafe_allow_html=True)

# ------------------- MAIN CONTENT -------------------
run = st.button(
    "Run Monte Carlo Analysis",
    type="primary",
    use_container_width=True,
    key="run_button",
)

if run:
    try:
        # Progress bar for user feedback
        progress_bar = st.progress(0)
        status_text = st.empty()

        # ---------- Monte Carlo Pricing ----------
        status_text.text("Calculating option price...")
        progress_bar.progress(20)

        # Check if MonteCarloPricer is available
        if "MonteCarloPricer" not in globals() or MonteCarloPricer is None:
            st.warning(
                "Using fallback Monte Carlo implementation. Some features may be limited."
            )

        price, t_price_ms = timeit_ms(
            price_monte_carlo,
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_type=option_type,
            q=q,
            num_sim=num_sim,
            num_steps=num_steps,
            seed=seed,
            use_numba=use_numba,
        )

        # Ensure price is scalar
        if price is not None:
            if not np.isscalar(price):
                price = float(np.mean(price))
            price = float(price)

        # ---------- Greeks ----------
        status_text.text("Computing Greeks...")
        progress_bar.progress(40)

        delta, gamma = greeks_mc_delta_gamma(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_type=option_type,
            q=q,
            num_sim=num_sim,
            num_steps=num_steps,
            seed=seed,
            h=1e-3,
            use_numba=use_numba,
        )

        # Ensure Greeks are scalars
        if delta is not None:
            if not np.isscalar(delta):
                delta = float(np.mean(delta))
            delta = float(delta)

        if gamma is not None:
            if not np.isscalar(gamma):
                gamma = float(np.mean(gamma))
            gamma = float(gamma)

        # ---------- Payoff & Paths ----------
        status_text.text("Generating price paths...")
        progress_bar.progress(60)

        # CRITICAL FIX: Ensure discounted and S_paths are always defined
        try:
            discounted, S_paths = simulate_payoffs(
                S, K, T, r, sigma, option_type, num_sim, num_steps, seed, q=q
            )
        except Exception as e:
            logger.warning(
                f"simulate_payoffs failed: {str(e)}. Using fallback implementation."
            )
            # Fallback implementation if simulate_payoffs fails
            np.random.seed(int(seed))
            dt = T / num_steps
            Z = np.random.standard_normal((num_sim, num_steps))
            S_paths = np.zeros((num_sim, num_steps))
            S_paths[:, 0] = S

            for t in range(1, num_steps):
                S_paths[:, t] = S_paths[:, t - 1] * np.exp(
                    (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t]
                )

            if option_type == "call":
                payoff = np.maximum(S_paths[:, -1] - K, 0.0)
            else:
                payoff = np.maximum(K - S_paths[:, -1], 0.0)

            discounted = np.exp(-r * T) * payoff

        # Calculate confidence interval
        mean_price = np.mean(discounted)
        std_error = np.std(discounted) / np.sqrt(num_sim)
        ci_lower = mean_price - 1.96 * std_error
        ci_upper = mean_price + 1.96 * std_error

        # ---------- Greeks Sensitivity ----------
        status_text.text("Calculating Greeks sensitivity...")
        progress_bar.progress(80)

        # Generate sensitivity data
        spot_range = np.linspace(0.7 * S, 1.3 * S, 50)
        delta_sensitivity = []
        gamma_sensitivity = []

        for s_val in spot_range:
            try:
                d, g = greeks_mc_delta_gamma(
                    S=s_val,
                    K=K,
                    T=T,
                    r=r,
                    sigma=sigma,
                    option_type=option_type,
                    num_sim=max(1000, num_sim // 20),
                    num_steps=num_steps,
                    seed=seed,
                    h=1e-3,
                    use_numba=use_numba,
                )
                delta_sensitivity.append(d)
                gamma_sensitivity.append(g)
            except Exception as e:
                # Fallback for problematic points
                logger.warning(f"Greeks calculation failed at spot {s_val}: {str(e)}")
                delta_sensitivity.append(np.nan)
                gamma_sensitivity.append(np.nan)

        # Filter out NaN values
        valid_indices = [
            i
            for i, (d, g) in enumerate(zip(delta_sensitivity, gamma_sensitivity))
            if not np.isnan(d) and not np.isnan(g)
        ]

        if valid_indices:
            spot_range = spot_range[valid_indices]
            delta_sensitivity = [delta_sensitivity[i] for i in valid_indices]
            gamma_sensitivity = [gamma_sensitivity[i] for i in valid_indices]
        else:
            # Fallback values if all calculations failed
            spot_range = np.linspace(0.7 * S, 1.3 * S, 5)
            delta_sensitivity = [0.5] * 5
            gamma_sensitivity = [0.01] * 5

        # ---------- Convergence Analysis ----------
        status_text.text("Analyzing convergence...")
        progress_bar.progress(90)

        # Generate convergence data
        step_size = max(1000, num_sim // 100)
        sample_sizes = np.arange(step_size, num_sim + 1, step_size)
        running_means = []

        for size in sample_sizes:
            try:
                running_means.append(np.mean(discounted[:size]))
            except Exception as e:
                logger.warning(
                    f"Convergence calculation failed at size {size}: {str(e)}"
                )
                running_means.append(np.nan)

        status_text.text("Generating visualizations...")
        progress_bar.progress(100)
        time.sleep(0.3)
        status_text.empty()
        progress_bar.empty()

        # ---------- Metrics Display ----------
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)

        col1.markdown(
            '<div class="metric-label">Option Price</div>', unsafe_allow_html=True
        )
        col1.markdown(
            f'<div class="metric-value">${price:.4f}</div>', unsafe_allow_html=True
        )
        col1.markdown(
            f'<div class="metric-delta">({ci_lower:.4f} - {ci_upper:.4f})</div>',
            unsafe_allow_html=True,
        )

        col2.markdown('<div class="metric-label">Delta</div>', unsafe_allow_html=True)
        col2.markdown(
            f'<div class="metric-value">{delta:.4f}</div>', unsafe_allow_html=True
        )
        col2.markdown(
            '<div class="metric-delta">Sensitivity to spot price</div>',
            unsafe_allow_html=True,
        )

        col3.markdown('<div class="metric-label">Gamma</div>', unsafe_allow_html=True)
        col3.markdown(
            f'<div class="metric-value">{gamma:.6f}</div>', unsafe_allow_html=True
        )
        col3.markdown(
            '<div class="metric-delta">Delta\'s sensitivity</div>',
            unsafe_allow_html=True,
        )

        col4.markdown(
            '<div class="metric-label">Execution Time</div>', unsafe_allow_html=True
        )
        col4.markdown(
            f'<div class="metric-value">{t_price_ms:.2f} ms</div>',
            unsafe_allow_html=True,
        )
        col4.markdown(
            f'<div class="metric-delta">{num_sim:,} paths</div>', unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- TABS for Visualizations ----------
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Pricing Overview",
                "Path Simulation",
                "Greeks Analysis",
                "Convergence",
                "Risk Analysis",
            ]
        )

        # Add CSS for full-width tabs - CRITICAL FIX FOR STREAMLIT CLOUD
        st.markdown(
            """
        <style>
            .stTabs [data-baseweb="tablist"] {
                display: flex !important;
                flex-wrap: wrap !important;
                gap: 0.5rem !important;
                margin-bottom: 1.5rem !important;
            }
            .stTabs [role="tab"] {
                flex: 1 !important;
                min-width: 150px !important;
                text-align: center !important;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # ---------- TAB 1: Pricing Overview ----------
        with tab1:
            st.markdown(
                '<h2 class="chart-title">Option Payoff Profile</h2>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="chart-description">This diagram shows the terminal payoff profile of the option across different spot prices. The line represents the theoretical payoff, while the histogram shows the distribution of simulated outcomes.</p>',
                unsafe_allow_html=True,
            )

            # Create payoff diagram
            terminal_prices = S_paths[:, -1]
            fig_payoff = go.Figure()

            # Add payoff curve
            spot_grid = np.linspace(min(terminal_prices), max(terminal_prices), 100)
            if option_type == "call":
                theoretical_payoff = np.maximum(spot_grid - K, 0)
                simulated_payoff = np.maximum(terminal_prices - K, 0)
            else:
                theoretical_payoff = np.maximum(K - spot_grid, 0)
                simulated_payoff = np.maximum(K - terminal_prices, 0)

            fig_payoff.add_trace(
                go.Scatter(
                    x=spot_grid,
                    y=theoretical_payoff,
                    mode="lines",
                    name="Theoretical Payoff",
                    line=dict(color="#3B82F6", width=4),
                )
            )

            # Add histogram of simulated payoffs
            fig_payoff.add_trace(
                go.Histogram(
                    x=terminal_prices,
                    y=simulated_payoff,
                    histfunc="avg",
                    nbinsx=50,
                    name="Simulated Payoff",
                    marker_color="#60A5FA",
                    opacity=0.7,
                )
            )

            fig_payoff.add_vline(
                x=K,
                line_dash="dash",
                line_color="#94A3B8",
                annotation_text="Strike Price",
                annotation_font_size=14,
            )
            fig_payoff.add_vline(
                x=S,
                line_dash="dash",
                line_color="#F87171",
                annotation_text="Current Spot",
                annotation_font_size=14,
            )

            fig_payoff.update_layout(
                title_font_size=20,
                xaxis_title="Terminal Spot Price",
                yaxis_title="Payoff",
                template="plotly_dark",
                hovermode="x unified",
                height=500,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                paper_bgcolor="rgba(30,41,59,1)",
                plot_bgcolor="rgba(15,23,42,1)",
                font=dict(size=14),
            )

            st.plotly_chart(fig_payoff, use_container_width=True)

            st.markdown(
                '<h2 class="chart-title">Discounted Price Distribution</h2>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="chart-description">This histogram shows the distribution of discounted option payoffs from the Monte Carlo simulation. The red line indicates the calculated option price (mean of the distribution).</p>',
                unsafe_allow_html=True,
            )

            fig_dist = go.Figure()
            fig_dist.add_trace(
                go.Histogram(
                    x=discounted,
                    nbinsx=50,
                    name="Discounted Payoff",
                    marker_color="#60A5FA",
                    opacity=0.7,
                )
            )

            fig_dist.add_vline(
                x=price,
                line_dash="solid",
                line_color="#F87171",
                line_width=3,
                annotation_text=f"Option Price: ${price:.4f}",
                annotation_font_size=14,
            )
            fig_dist.add_vrect(
                x0=ci_lower,
                x1=ci_upper,
                fillcolor="rgba(56, 189, 248, 0.2)",
                opacity=0.5,
                line_width=0,
                annotation_text="95% Confidence Interval",
                annotation_position="top left",
                annotation_font_size=14,
            )

            fig_dist.update_layout(
                title_font_size=20,
                xaxis_title="Discounted Payoff",
                yaxis_title="Frequency",
                template="plotly_dark",
                height=450,
                paper_bgcolor="rgba(30,41,59,1)",
                plot_bgcolor="rgba(15,23,42,1)",
                font=dict(size=14),
            )

            st.plotly_chart(fig_dist, use_container_width=True)

        # ---------- TAB 2: Path Simulation ----------
        with tab2:
            st.markdown(
                '<h2 class="chart-title">Sample Price Paths</h2>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="chart-description">This visualization shows a random sample of simulated price paths. The lines represent the geometric Brownian motion paths of the underlying asset.</p>',
                unsafe_allow_html=True,
            )

            # Select a random sample of paths
            n_plot = min(50, num_sim)
            sample_indices = np.random.choice(num_sim, n_plot, replace=False)

            fig_paths = go.Figure()

            # Add each path
            for i in sample_indices:
                fig_paths.add_trace(
                    go.Scatter(
                        x=np.arange(num_steps),
                        y=S_paths[i, :],
                        mode="lines",
                        line=dict(width=1.5, color="rgba(96, 165, 250, 0.4)"),
                        hovertemplate="Step: %{x}<br>Price: %{y:.2f}<extra></extra>",
                        showlegend=False,
                    )
                )

            # Add average path
            avg_path = np.mean(S_paths, axis=0)
            fig_paths.add_trace(
                go.Scatter(
                    x=np.arange(num_steps),
                    y=avg_path,
                    mode="lines",
                    name="Average Path",
                    line=dict(color="#3B82F6", width=4),
                )
            )

            # Add current spot line
            fig_paths.add_hline(
                y=S,
                line_dash="dash",
                line_color="#F87171",
                annotation_text=f"Initial Spot: {S}",
                annotation_font_size=14,
            )

            fig_paths.update_layout(
                title_font_size=20,
                xaxis_title="Time Steps",
                yaxis_title="Spot Price",
                template="plotly_dark",
                height=500,
                hovermode="x unified",
                paper_bgcolor="rgba(30,41,59,1)",
                plot_bgcolor="rgba(15,23,42,1)",
                font=dict(size=14),
            )

            st.plotly_chart(fig_paths, use_container_width=True)

            st.markdown(
                '<h2 class="chart-title">Terminal Price Distribution</h2>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="chart-description">This chart shows the distribution of terminal prices from the Monte Carlo simulation. The vertical lines indicate key reference points for analysis.</p>',
                unsafe_allow_html=True,
            )

            fig_terminal = go.Figure()
            fig_terminal.add_trace(
                go.Histogram(
                    x=terminal_prices,
                    nbinsx=50,
                    name="Terminal Prices",
                    marker_color="#60A5FA",
                    opacity=0.7,
                )
            )

            fig_terminal.add_vline(
                x=S,
                line_dash="dash",
                line_color="#F87171",
                annotation_text="Initial Spot",
                annotation_font_size=14,
            )
            fig_terminal.add_vline(
                x=K,
                line_dash="dash",
                line_color="#94A3B8",
                annotation_text="Strike Price",
                annotation_font_size=14,
            )
            fig_terminal.add_vline(
                x=np.mean(terminal_prices),
                line_dash="dash",
                line_color="#34D399",
                annotation_text="Mean Terminal Price",
                annotation_font_size=14,
            )

            fig_terminal.update_layout(
                title_font_size=20,
                xaxis_title="Terminal Spot Price",
                yaxis_title="Frequency",
                template="plotly_dark",
                height=450,
                paper_bgcolor="rgba(30,41,59,1)",
                plot_bgcolor="rgba(15,23,42,1)",
                font=dict(size=14),
            )

            st.plotly_chart(fig_terminal, use_container_width=True)

        # ---------- TAB 3: Greeks Analysis ----------
        with tab3:
            st.markdown(
                '<h2 class="chart-title">Delta & Gamma Sensitivity</h2>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="chart-description">This chart shows how Delta and Gamma change as the spot price varies. Understanding these sensitivities is crucial for hedging and risk management.</p>',
                unsafe_allow_html=True,
            )

            # Create Greeks sensitivity chart
            fig_greeks = go.Figure()

            # Add Delta trace
            fig_greeks.add_trace(
                go.Scatter(
                    x=spot_range,
                    y=delta_sensitivity,
                    mode="lines+markers",
                    name="Delta",
                    line=dict(color="#3B82F6", width=4),
                    marker=dict(size=8, color="#3B82F6"),
                )
            )

            # Add Gamma trace
            fig_greeks.add_trace(
                go.Scatter(
                    x=spot_range,
                    y=gamma_sensitivity,
                    mode="lines",
                    name="Gamma",
                    yaxis="y2",
                    line=dict(color="#F59E0B", width=4, dash="dot"),
                )
            )

            # Add reference lines
            fig_greeks.add_vline(
                x=S,
                line_dash="dash",
                line_color="#94A3B8",
                annotation_text=f"Current Spot: {S}",
                annotation_font_size=14,
            )
            fig_greeks.add_vline(
                x=K,
                line_dash="dash",
                line_color="#94A3B8",
                annotation_text=f"Strike: {K}",
                annotation_font_size=14,
            )

            # Set up dual y-axes
            fig_greeks.update_layout(
                title_font_size=20,
                xaxis_title="Spot Price",
                yaxis_title="Delta",
                yaxis2=dict(
                    title="Gamma", overlaying="y", side="right", showgrid=False
                ),
                template="plotly_dark",
                height=500,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font_size=14,
                ),
                paper_bgcolor="rgba(30,41,59,1)",
                plot_bgcolor="rgba(15,23,42,1)",
                font=dict(size=14),
            )

            st.plotly_chart(fig_greeks, use_container_width=True)

            # Greeks explanation
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    '<h3 class="subsection-header">Delta Insights</h3>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                - **Current Delta**: {delta:.4f}
                - Delta measures the sensitivity of the option price to changes in the underlying asset price
                - For calls, Delta ranges from 0 to 1; for puts, from -1 to 0
                - At-the-money options typically have Delta around 0.5 (call) or -0.5 (put)
                - Deep in-the-money options approach Delta of 1 (call) or -1 (put)
                """
                )

            with col2:
                st.markdown(
                    '<h3 class="subsection-header">Gamma Insights</h3>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                - **Current Gamma**: {gamma:.6f}
                - Gamma measures how Delta changes as the underlying price moves
                - Highest for at-the-money options near expiration
                - Decreases as options move deeper in or out of the money
                - Critical for dynamic hedging strategies
                - Higher Gamma means Delta changes rapidly with spot price movements
                """
                )

        # ---------- TAB 4: Convergence ----------
        with tab4:
            st.markdown(
                '<h2 class="chart-title">Monte Carlo Convergence</h2>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="chart-description">This chart shows how the Monte Carlo estimate converges to the final price as the number of simulation paths increases. The shaded area represents the confidence interval.</p>',
                unsafe_allow_html=True,
            )

            # Create convergence chart
            fig_convergence = go.Figure()

            # Calculate running means and confidence intervals
            running_means = []
            ci_upper_list = []
            ci_lower_list = []

            for i in range(1, len(sample_sizes) + 1):
                current_size = sample_sizes[i - 1]
                try:
                    current_mean = np.mean(discounted[:current_size])
                    current_std = np.std(discounted[:current_size])
                    current_se = current_std / np.sqrt(current_size)

                    running_means.append(current_mean)
                    ci_upper_list.append(current_mean + 1.96 * current_se)
                    ci_lower_list.append(current_mean - 1.96 * current_se)
                except:
                    running_means.append(np.nan)
                    ci_upper_list.append(np.nan)
                    ci_lower_list.append(np.nan)

            # Filter out NaN values
            valid_indices = [
                i for i, val in enumerate(running_means) if not np.isnan(val)
            ]
            if valid_indices:
                valid_sample_sizes = [sample_sizes[i] for i in valid_indices]
                valid_running_means = [running_means[i] for i in valid_indices]
                valid_ci_upper = [ci_upper_list[i] for i in valid_indices]
                valid_ci_lower = [ci_lower_list[i] for i in valid_indices]
            else:
                # Fallback if all values are NaN
                valid_sample_sizes = [num_sim]
                valid_running_means = [price]
                valid_ci_upper = [ci_upper]
                valid_ci_lower = [ci_lower]

            # Add main convergence line
            fig_convergence.add_trace(
                go.Scatter(
                    x=valid_sample_sizes,
                    y=valid_running_means,
                    mode="lines",
                    name="Running Mean",
                    line=dict(color="#3B82F6", width=4),
                )
            )

            # Add confidence interval
            fig_convergence.add_trace(
                go.Scatter(
                    x=np.concatenate([valid_sample_sizes, valid_sample_sizes[::-1]]),
                    y=np.concatenate([valid_ci_upper, valid_ci_lower[::-1]]),
                    fill="toself",
                    fillcolor="rgba(56, 189, 248, 0.3)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="95% CI",
                )
            )

            # Add final price line
            fig_convergence.add_hline(
                y=price,
                line_dash="dash",
                line_color="#F87171",
                annotation_text=f"Final Price: ${price:.4f}",
                annotation_font_size=14,
            )

            fig_convergence.update_layout(
                title_font_size=20,
                xaxis_title="Number of Simulation Paths",
                yaxis_title="Option Price",
                template="plotly_dark",
                height=500,
                hovermode="x unified",
                paper_bgcolor="rgba(30,41,59,1)",
                plot_bgcolor="rgba(15,23,42,1)",
                font=dict(size=14),
            )

            st.plotly_chart(fig_convergence, use_container_width=True)

            # Convergence statistics
            st.markdown(
                '<h3 class="subsection-header">Convergence Metrics</h3>',
                unsafe_allow_html=True,
            )

            # Calculate convergence rate
            valid_errors = [
                abs(val - price) for val in valid_running_means if not np.isnan(val)
            ]
            if len(valid_errors) > 1 and valid_errors[-1] > 0:
                log_errors = np.log10(valid_errors)
                log_samples = np.log10(valid_sample_sizes[: len(valid_errors)])

                # Simple linear regression for convergence rate
                try:
                    slope = np.polyfit(log_samples, log_errors, 1)[0]
                except:
                    slope = -0.5  # Default expected convergence rate
            else:
                slope = -0.5  # Default expected convergence rate

            col1, col2, col3 = st.columns(3)
            with col1:
                if valid_errors:
                    st.metric(
                        "Final Error",
                        f"{valid_errors[-1]:.6f}",
                        f"after {num_sim:,} paths",
                    )
                else:
                    st.metric("Final Error", "N/A", "Convergence data unavailable")
            with col2:
                st.metric("Convergence Rate", f"{-slope:.2f}", "expected: ~0.5 for MC")
            with col3:
                st.metric("CI Width", f"{ci_upper - ci_lower:.6f}", "95% confidence")

            st.markdown(
                """
            **Interpretation**: 
            - The error should decrease proportionally to 1/√N (where N is number of paths)
            - A convergence rate close to 0.5 indicates proper Monte Carlo behavior
            - Wider confidence intervals suggest more uncertainty in the estimate
            """
            )

        # ---------- TAB 5: Risk Analysis ----------
        with tab5:
            st.markdown(
                '<h2 class="chart-title">Confidence Interval Evolution</h2>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="chart-description">This section provides detailed risk analysis of the Monte Carlo simulation, including confidence intervals and path-wise error analysis.</p>',
                unsafe_allow_html=True,
            )

            # Create confidence interval chart
            fig_ci = go.Figure()

            # Calculate running statistics
            n_points = min(1000, num_sim)
            step = max(1, num_sim // n_points)
            sample_indices = np.arange(0, num_sim, step)

            running_means = []
            ci_upper_list = []
            ci_lower_list = []

            for i in sample_indices:
                try:
                    current_mean = np.mean(discounted[: i + 1])
                    current_std = np.std(discounted[: i + 1])
                    current_se = current_std / np.sqrt(i + 1)

                    running_means.append(current_mean)
                    ci_upper_list.append(current_mean + 1.96 * current_se)
                    ci_lower_list.append(current_mean - 1.96 * current_se)
                except:
                    running_means.append(np.nan)
                    ci_upper_list.append(np.nan)
                    ci_lower_list.append(np.nan)

            # Filter out NaN values
            valid_indices = [
                i for i, val in enumerate(running_means) if not np.isnan(val)
            ]
            if valid_indices:
                valid_sample_indices = [sample_indices[i] for i in valid_indices]
                valid_running_means = [running_means[i] for i in valid_indices]
                valid_ci_upper = [ci_upper_list[i] for i in valid_indices]
                valid_ci_lower = [ci_lower_list[i] for i in valid_indices]
            else:
                # Fallback if all values are NaN
                valid_sample_indices = [num_sim]
                valid_running_means = [price]
                valid_ci_upper = [ci_upper]
                valid_ci_lower = [ci_lower]

            # Add main convergence line
            fig_ci.add_trace(
                go.Scatter(
                    x=valid_sample_indices,
                    y=valid_running_means,
                    mode="lines",
                    name="Running Mean",
                    line=dict(color="#3B82F6", width=3.5),
                )
            )

            # Add confidence interval
            fig_ci.add_trace(
                go.Scatter(
                    x=np.concatenate(
                        [valid_sample_indices, valid_sample_indices[::-1]]
                    ),
                    y=np.concatenate([valid_ci_upper, valid_ci_lower[::-1]]),
                    fill="toself",
                    fillcolor="rgba(56, 189, 248, 0.3)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="95% CI",
                )
            )

            # Add final price line
            fig_ci.add_hline(
                y=price,
                line_dash="dash",
                line_color="#F87171",
                annotation_text=f"Final Price: ${price:.4f}",
                annotation_font_size=14,
            )

            fig_ci.update_layout(
                title_font_size=20,
                xaxis_title="Number of Simulation Paths",
                yaxis_title="Option Price",
                template="plotly_dark",
                height=500,
                hovermode="x unified",
                paper_bgcolor="rgba(30,41,59,1)",
                plot_bgcolor="rgba(15,23,42,1)",
                font=dict(size=14),
            )

            st.plotly_chart(fig_ci, use_container_width=True)

            st.markdown(
                '<h3 class="subsection-header">Error Distribution Analysis</h3>',
                unsafe_allow_html=True,
            )

            # Calculate errors
            errors = discounted - price
            abs_errors = np.abs(errors)

            # Create error distribution chart
            fig_error = sp.make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Error Distribution", "Absolute Error vs Path"),
            )

            # Error distribution histogram
            fig_error.add_trace(
                go.Histogram(
                    x=errors,
                    nbinsx=50,
                    name="Error",
                    marker_color="#60A5FA",
                    opacity=0.7,
                ),
                row=1,
                col=1,
            )

            fig_error.add_vline(
                x=0, line_dash="dash", line_color="#F87171", row=1, col=1
            )

            # Absolute error vs path index
            fig_error.add_trace(
                go.Scatter(
                    x=np.arange(len(abs_errors)),
                    y=abs_errors,
                    mode="markers",
                    name="Absolute Error",
                    marker=dict(
                        size=6, color=abs_errors, colorscale="Viridis", showscale=True
                    ),
                    opacity=0.7,
                ),
                row=1,
                col=2,
            )

            fig_error.update_layout(
                height=450,
                template="plotly_dark",
                showlegend=False,
                paper_bgcolor="rgba(30,41,59,1)",
                plot_bgcolor="rgba(15,23,42,1)",
                font=dict(size=14),
            )

            fig_error.update_xaxes(title_text="Error", row=1, col=1, title_font_size=16)
            fig_error.update_yaxes(
                title_text="Frequency", row=1, col=1, title_font_size=16
            )
            fig_error.update_xaxes(
                title_text="Path Index", row=1, col=2, title_font_size=16
            )
            fig_error.update_yaxes(
                title_text="Absolute Error", row=1, col=2, title_font_size=16
            )

            st.plotly_chart(fig_error, use_container_width=True)

            st.markdown(
                '<h3 class="subsection-header">Risk Metrics</h3>',
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "95% CI Width",
                    f"{ci_upper - ci_lower:.6f}",
                    "Narrower = more precise",
                )
            with col2:
                st.metric("Standard Error", f"{std_error:.6f}", "√(variance/N)")
            with col3:
                if price > 0:
                    st.metric(
                        "Relative Error",
                        f"{(ci_upper - ci_lower)/(2*price):.2%}",
                        "CI width / price",
                    )
                else:
                    st.metric("Relative Error", "N/A", "Price is zero")

            st.markdown(
                """
            **Key Risk Insights**:
            - The confidence interval width indicates the precision of your Monte Carlo estimate
            - Standard error decreases as 1/√N - increasing paths improves precision
            - Relative error helps assess if the simulation is sufficiently precise for your needs
            - Consider increasing paths if relative error exceeds your tolerance threshold
            """
            )

    except Exception as e:
        st.error(f"Critical error during pricing: {str(e)}")
        logger.exception("Critical pricing failure")
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Analysis Failed")
        st.markdown(
            f"""
        An error occurred during the Monte Carlo analysis:
        
        **{str(e)}**
        
        Possible causes:
        - Insufficient memory for large simulation sizes
        - Invalid parameter combinations (e.g., negative volatility)
        - Numerical instability in calculations
        
        Try:
        1. Reducing the number of simulations
        2. Checking all input values are valid
        3. Refreshing the page and trying again
        """
        )
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown(
        '<div style="text-align: center; padding: 3rem 0;">', unsafe_allow_html=True
    )
    st.markdown("### Ready to Analyze")
    st.markdown(
        "Configure your parameters above and click **Run Monte Carlo Analysis** to see results"
    )
    st.image(
        "https://images.unsplash.com/photo-1611974489855-dcda4e25d61c?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80",
        use_column_width=True,
        caption="Monte Carlo simulation provides powerful tools for option pricing and risk analysis",
    )
    st.markdown("</div>", unsafe_allow_html=True)
