# streamlit_app/pages/1_MonteCarlo_Basic.py
"""
Monte Carlo Basic Pricing - Streamlit Page.

Sleek, modern design with full-width layout and inline inputs.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Monte Carlo Pricing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    from components import (
        apply_custom_css,
        format_greek,
        format_price,
        format_time_ms,
        get_chart_layout,
        page_header,
        section_divider,
    )
except ImportError:
    from streamlit_app.components import (
        apply_custom_css,
        format_greek,
        format_price,
        format_time_ms,
        get_chart_layout,
        page_header,
        section_divider,
    )

try:
    from src.greeks import compute_greeks_unified
    from src.pricing_models import (
        GPU_AVAILABLE,
        NUMBA_AVAILABLE,
        MonteCarloPricer,
        black_scholes,
    )
except ImportError:
    MonteCarloPricer = None
    black_scholes = None
    compute_greeks_unified = None
    NUMBA_AVAILABLE = False
    GPU_AVAILABLE = False

apply_custom_css()

# =============================================================================
# HEADER
# =============================================================================
page_header(
    "Monte Carlo Option Pricing",
    "Simulate option prices with vectorized Monte Carlo simulation",
)

# =============================================================================
# INPUT SECTION (NOT SIDEBAR)
# =============================================================================
st.markdown('<div class="input-section">', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1.2, 0.8])

with col1:
    st.markdown("**Asset Parameters**")
    S = st.number_input(
        "Spot Price ($)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0
    )
    K = st.number_input(
        "Strike Price ($)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0
    )

with col2:
    st.markdown("**Time & Type**")
    T = st.number_input(
        "Time to Maturity (years)", min_value=0.01, max_value=5.0, value=1.0, step=0.05
    )
    option_type = st.selectbox("Option Type", ["call", "put"])

with col3:
    st.markdown("**Market Conditions**")
    r_pct = st.number_input(
        "Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5
    )
    sigma_pct = st.number_input(
        "Volatility (%)", min_value=1.0, max_value=200.0, value=20.0, step=1.0
    )
    r = r_pct / 100.0
    sigma = sigma_pct / 100.0

with col4:
    st.markdown("**Simulation Settings**")
    num_sims = st.select_slider(
        "Simulations",
        options=[10000, 25000, 50000, 100000, 200000],
        value=50000,
    )
    num_steps = st.select_slider(
        "Time Steps",
        options=[1, 10, 25, 50, 100],
        value=1,
        help="Use 1 for fastest European option pricing",
    )

with col5:
    st.markdown("**Run**")
    seed = st.number_input("Seed", min_value=1, max_value=99999, value=42)
    run = st.button("🚀 Price Option", type="primary", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# RESULTS
# =============================================================================
if run:
    if MonteCarloPricer is None:
        st.error("Monte Carlo pricer not available. Check installation.")
        st.stop()

    with st.spinner("Running MC simulation..."):
        pricer = MonteCarloPricer(
            num_simulations=num_sims,
            num_steps=num_steps,
            seed=seed,
        )

        # Time pricing
        t_start = time.perf_counter()
        mc_price = pricer.price(S, K, T, r, sigma, option_type)
        t_price = (time.perf_counter() - t_start) * 1000

        # Time Greeks using unified interface
        t_start = time.perf_counter()
        greeks = compute_greeks_unified(
            pricer, S, K, T, r, sigma, option_type, include_second_order=False
        )
        delta = greeks["delta"]
        gamma = greeks["gamma"]
        vega = greeks["vega"]
        theta = greeks["theta"]
        rho = greeks["rho"]
        t_greeks = (time.perf_counter() - t_start) * 1000

        t_total = t_price + t_greeks
        backend_name = "NumPy"  # Default backend
        bs_price = (
            black_scholes(S, K, T, r, sigma, option_type) if black_scholes else None
        )

    section_divider()

    # Metrics row
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">MC Price</div>
            <div class="metric-value">{format_price(mc_price)}</div>
            <div class="metric-delta">{format_time_ms(t_price)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        if bs_price:
            error = abs(mc_price - bs_price)
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">BS Price</div>
                <div class="metric-value">{format_price(bs_price)}</div>
                <div class="metric-delta">Err: ${error:.4f}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Delta (Δ)</div>
            <div class="metric-value">{format_greek(delta)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Gamma (Γ)</div>
            <div class="metric-value">{format_greek(gamma, 6)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Vega (ν)</div>
            <div class="metric-value">{format_greek(vega, 2)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col6:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Theta (Θ)</div>
            <div class="metric-value">{format_greek(theta, 2)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col7:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Total Time</div>
            <div class="metric-value">{format_time_ms(t_total)}</div>
            <div class="metric-delta">{backend_name}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    section_divider()

    # Charts
    tab1, tab2, tab3 = st.tabs(
        ["📈 Price Paths", "🎯 Payoff Profile", "📊 Greeks Sensitivity"]
    )

    with tab1:
        np.random.seed(seed)
        n_paths = min(100, num_sims)
        dt = T / num_steps  # Now num_steps is defined

        paths = np.zeros((n_paths, num_steps + 1))
        paths[:, 0] = S

        for t_idx in range(num_steps):
            z = np.random.standard_normal(n_paths)
            paths[:, t_idx + 1] = paths[:, t_idx] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
            )

        fig = go.Figure()

        time_axis = np.linspace(0, T, num_steps + 1)
        for i in range(min(50, n_paths)):
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=paths[i],
                    mode="lines",
                    line=dict(width=0.8, color=f"rgba(96, 165, 250, 0.3)"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        fig.add_hline(
            y=K, line_dash="dash", line_color="#ef4444", annotation_text=f"Strike: ${K}"
        )

        mean_path = np.mean(paths, axis=0)
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=mean_path,
                mode="lines",
                line=dict(width=3, color="#60a5fa"),
                name="Mean Path",
            )
        )

        fig.update_layout(**get_chart_layout("Monte Carlo Price Paths", 450))
        fig.update_xaxes(title_text="Time (years)")
        fig.update_yaxes(title_text="Stock Price ($)")

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        spot_range = np.linspace(S * 0.6, S * 1.4, 100)

        if option_type == "call":
            intrinsic = np.maximum(spot_range - K, 0)
        else:
            intrinsic = np.maximum(K - spot_range, 0)

        # Use BS for speed
        if black_scholes:
            option_values = np.array(
                [black_scholes(s, K, T, r, sigma, option_type) for s in spot_range]
            )
        else:
            option_values = intrinsic * 1.1

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=spot_range,
                y=intrinsic,
                mode="lines",
                line=dict(width=2, color="#94a3b8", dash="dash"),
                name="Intrinsic Value",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=spot_range,
                y=option_values,
                mode="lines",
                line=dict(width=3, color="#60a5fa"),
                name="Option Value",
            )
        )

        fig.add_vline(
            x=K, line_dash="dash", line_color="#ef4444", annotation_text="Strike"
        )
        fig.add_vline(
            x=S, line_dash="dot", line_color="#10b981", annotation_text="Current"
        )

        fig.update_layout(**get_chart_layout("Payoff Profile", 400))
        fig.update_xaxes(title_text="Stock Price ($)")
        fig.update_yaxes(title_text="Option Value ($)")

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            # Delta vs Spot (use BS for speed)
            spot_range = np.linspace(S * 0.6, S * 1.4, 50)

            from scipy.stats import norm

            def bs_delta(s, k, t, r_rate, vol, opt_type):
                d1 = (np.log(s / k) + (r_rate + 0.5 * vol**2) * t) / (vol * np.sqrt(t))
                if opt_type == "call":
                    return norm.cdf(d1)
                else:
                    return norm.cdf(d1) - 1

            deltas = [bs_delta(s, K, T, r, sigma, option_type) for s in spot_range]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=spot_range,
                    y=deltas,
                    mode="lines",
                    line=dict(width=3, color="#a78bfa"),
                    fill="tozeroy",
                    fillcolor="rgba(167, 139, 250, 0.2)",
                )
            )

            fig.add_vline(x=S, line_dash="dot", line_color="#10b981")
            fig.update_layout(**get_chart_layout("Delta vs Spot Price", 350))
            fig.update_xaxes(title_text="Spot Price ($)")
            fig.update_yaxes(title_text="Delta")

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Gamma vs Spot
            def bs_gamma(s, k, t, r_rate, vol):
                d1 = (np.log(s / k) + (r_rate + 0.5 * vol**2) * t) / (vol * np.sqrt(t))
                return norm.pdf(d1) / (s * vol * np.sqrt(t))

            gammas = [bs_gamma(s, K, T, r, sigma) for s in spot_range]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=spot_range,
                    y=gammas,
                    mode="lines",
                    line=dict(width=3, color="#34d399"),
                    fill="tozeroy",
                    fillcolor="rgba(52, 211, 153, 0.2)",
                )
            )

            fig.add_vline(x=S, line_dash="dot", line_color="#10b981")
            fig.update_layout(**get_chart_layout("Gamma vs Spot Price", 350))
            fig.update_xaxes(title_text="Spot Price ($)")
            fig.update_yaxes(title_text="Gamma")

            st.plotly_chart(fig, use_container_width=True)
