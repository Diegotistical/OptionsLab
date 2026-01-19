# streamlit_app/pages/7_Exotic_Options.py
"""
Exotic Options Pricing - Streamlit Page.

Price Asian, Barrier, Lookback, Autocallable, and Cliquet options.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Exotic Options",
    page_icon="🎯",
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

# Import exotic options
try:
    from src.greeks import ExoticAdapter, compute_greeks_unified
    from src.pricing_models.exotic_options import (
        AsianOption,
        AutocallableOption,
        BarrierOption,
        CliquetOption,
        LookbackOption,
    )

    EXOTICS_AVAILABLE = True
except ImportError as e:
    EXOTICS_AVAILABLE = False
    st.error(f"Exotic options not available: {e}")

apply_custom_css()

# =============================================================================
# HEADER
# =============================================================================
page_header(
    "Exotic Options Pricing",
    "Monte Carlo pricing for path-dependent options",
)

section_divider()

# =============================================================================
# INPUT SECTION
# =============================================================================
st.markdown('<div class="input-section">', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns([1.2, 1, 1, 1.5, 0.8])

with col1:
    st.markdown("**Exotic Type**")
    exotic_type = st.selectbox(
        "Option Type",
        ["Asian", "Barrier", "Lookback", "Autocallable", "Cliquet"],
        key="exotic_type",
    )
    option_type = st.selectbox("Call/Put", ["call", "put"], key="exotic_cp")

with col2:
    st.markdown("**Asset Parameters**")
    S = st.number_input(
        "Spot Price ($)", min_value=1.0, max_value=500.0, value=100.0, step=1.0
    )
    K = st.number_input(
        "Strike Price ($)", min_value=1.0, max_value=500.0, value=100.0, step=1.0
    )

with col3:
    st.markdown("**Market**")
    T = st.number_input(
        "Maturity (years)", min_value=0.1, max_value=5.0, value=1.0, step=0.1
    )
    r_pct = st.number_input(
        "Rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5
    )
    sigma_pct = st.number_input(
        "Vol (%)", min_value=5.0, max_value=100.0, value=20.0, step=1.0
    )
    r = r_pct / 100.0
    sigma = sigma_pct / 100.0

with col4:
    st.markdown("**Exotic Parameters**")

    if exotic_type == "Asian":
        avg_type = st.selectbox("Averaging", ["arithmetic", "geometric"])
        n_paths = st.select_slider(
            "Paths", options=[10000, 25000, 50000, 100000], value=50000
        )
        n_steps = st.select_slider("Steps", options=[50, 100, 252, 500], value=252)

    elif exotic_type == "Barrier":
        barrier = st.number_input(
            "Barrier Level ($)", min_value=50.0, max_value=200.0, value=120.0, step=5.0
        )
        barrier_type = st.selectbox(
            "Barrier Type", ["up-and-out", "up-and-in", "down-and-out", "down-and-in"]
        )
        n_paths = st.select_slider(
            "Paths", options=[10000, 25000, 50000, 100000], value=50000
        )
        n_steps = st.select_slider("Steps", options=[50, 100, 252, 500], value=252)

    elif exotic_type == "Lookback":
        lookback_type = st.selectbox("Lookback Type", ["floating", "fixed"])
        n_paths = st.select_slider(
            "Paths", options=[10000, 25000, 50000, 100000], value=50000
        )
        n_steps = st.select_slider("Steps", options=[50, 100, 252, 500], value=252)

    elif exotic_type == "Autocallable":
        col_a, col_b = st.columns(2)
        with col_a:
            autocall_barrier = st.number_input(
                "Autocall Barrier (%)", min_value=100.0, max_value=150.0, value=105.0
            )
            coupon_rate = st.number_input(
                "Coupon (%)", min_value=1.0, max_value=20.0, value=8.0
            )
        with col_b:
            ki_barrier = st.number_input(
                "KI Barrier (%)", min_value=50.0, max_value=100.0, value=70.0
            )
            obs_freq = st.selectbox(
                "Observation", ["monthly", "quarterly", "semi-annual"]
            )
        n_paths = 25000
        n_steps = 252

    else:  # Cliquet
        col_a, col_b = st.columns(2)
        with col_a:
            local_floor = st.number_input(
                "Local Floor (%)", min_value=-10.0, max_value=0.0, value=-5.0
            )
            local_cap = st.number_input(
                "Local Cap (%)", min_value=1.0, max_value=20.0, value=10.0
            )
        with col_b:
            global_floor = st.number_input(
                "Global Floor (%)", min_value=0.0, max_value=50.0, value=0.0
            )
            n_periods = st.number_input("Periods", min_value=4, max_value=24, value=12)
        n_paths = 25000
        n_steps = 252

with col5:
    st.markdown("**Run**")
    seed = st.number_input(
        "Seed", min_value=1, max_value=99999, value=42, key="exotic_seed"
    )
    run = st.button("🚀 Price", type="primary", use_container_width=True)
    run_greeks = st.button("📊 + Greeks", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# PRICING
# =============================================================================
if (run or run_greeks) and EXOTICS_AVAILABLE:
    section_divider()

    with st.spinner(f"Pricing {exotic_type} option..."):
        t_start = time.perf_counter()

        if exotic_type == "Asian":
            option = AsianOption(S=S, K=K, T=T, r=r, sigma=sigma, seed=seed)
            price = option.price(
                n_paths=n_paths,
                n_steps=n_steps,
                avg_type=avg_type,
                option_type=option_type,
            )
            exotic_params = f"Averaging: {avg_type}"

        elif exotic_type == "Barrier":
            option = BarrierOption(
                S=S, K=K, T=T, r=r, sigma=sigma, barrier=barrier, seed=seed
            )
            price = option.price(
                n_paths=n_paths,
                n_steps=n_steps,
                barrier_type=barrier_type,
                option_type=option_type,
            )
            exotic_params = f"Barrier: ${barrier} ({barrier_type})"

        elif exotic_type == "Lookback":
            option = LookbackOption(S=S, K=K, T=T, r=r, sigma=sigma, seed=seed)
            price = option.price(
                n_paths=n_paths,
                n_steps=n_steps,
                lookback_type=lookback_type,
                option_type=option_type,
            )
            exotic_params = f"Type: {lookback_type}"

        elif exotic_type == "Autocallable":
            option = AutocallableOption(
                S=S,
                K=K,
                T=T,
                r=r,
                sigma=sigma,
                seed=seed,
                autocall_barrier=autocall_barrier / 100,
                coupon_rate=coupon_rate / 100,
                ki_barrier=ki_barrier / 100,
            )
            price = option.price(
                n_paths=n_paths, n_steps=n_steps, observation_freq=obs_freq
            )
            exotic_params = f"Autocall: {autocall_barrier}%, Coupon: {coupon_rate}%"

        else:  # Cliquet
            option = CliquetOption(
                S=S,
                K=K,
                T=T,
                r=r,
                sigma=sigma,
                seed=seed,
                local_floor=local_floor / 100,
                local_cap=local_cap / 100,
                global_floor=global_floor / 100,
            )
            price = option.price(n_paths=n_paths, n_steps=n_steps, n_periods=n_periods)
            exotic_params = f"Floor: {local_floor}%, Cap: {local_cap}%"

        t_price = (time.perf_counter() - t_start) * 1000

        # Greeks if requested
        greeks = None
        if run_greeks and exotic_type in ["Asian", "Barrier", "Lookback"]:
            t_start = time.perf_counter()
            if exotic_type == "Asian":
                adapter = ExoticAdapter(
                    option, n_paths=10000, n_steps=50, avg_type=avg_type
                )
            elif exotic_type == "Barrier":
                adapter = ExoticAdapter(
                    option, n_paths=10000, n_steps=50, barrier_type=barrier_type
                )
            else:
                adapter = ExoticAdapter(
                    option, n_paths=10000, n_steps=50, lookback_type=lookback_type
                )
            greeks = compute_greeks_unified(
                adapter, S, K, T, r, sigma, option_type, include_second_order=False
            )
            t_greeks = (time.perf_counter() - t_start) * 1000

    # Display results
    if greeks:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
    else:
        col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">{exotic_type} Price</div>
            <div class="metric-value">{format_price(price)}</div>
            <div class="metric-delta">{format_time_ms(t_price)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Parameters</div>
            <div class="metric-value" style="font-size: 1rem;">{exotic_params}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Paths × Steps</div>
            <div class="metric-value" style="font-size: 1.2rem;">{n_paths:,} × {n_steps}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        vanilla_price = (
            np.maximum(S - K, 0) if option_type == "call" else np.maximum(K - S, 0)
        )
        exotic_premium = price - vanilla_price if vanilla_price > 0 else price
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Exotic Premium</div>
            <div class="metric-value">{format_price(exotic_premium)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    if greeks:
        with col5:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Delta (Δ)</div>
                <div class="metric-value">{format_greek(greeks['delta'])}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col6:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Vega (ν)</div>
                <div class="metric-value">{format_greek(greeks['vega'], 2)}</div>
                <div class="metric-delta">{format_time_ms(t_greeks)}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Payoff diagram
    section_divider()

    st.markdown("### Payoff Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Terminal price distribution
        spot_range = np.linspace(S * 0.6, S * 1.4, 100)

        if exotic_type in ["Asian", "Barrier", "Lookback"]:
            # European vs Exotic payoff
            if option_type == "call":
                european_payoff = np.maximum(spot_range - K, 0)
            else:
                european_payoff = np.maximum(K - spot_range, 0)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=spot_range,
                    y=european_payoff,
                    mode="lines",
                    name="European Payoff",
                    line=dict(color="#94a3b8", dash="dash", width=2),
                )
            )
            fig.add_vline(
                x=K, line_dash="dash", line_color="#ef4444", annotation_text="Strike"
            )
            fig.add_vline(
                x=S, line_dash="dot", line_color="#10b981", annotation_text="Spot"
            )

            if exotic_type == "Barrier" and barrier:
                fig.add_vline(
                    x=barrier,
                    line_dash="dash",
                    line_color="#f59e0b",
                    annotation_text="Barrier",
                )

            fig.update_layout(**get_chart_layout("Payoff Profile", 350))
            fig.update_xaxes(title_text="Terminal Price ($)")
            fig.update_yaxes(title_text="Payoff ($)")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Sample paths
        np.random.seed(seed)
        sample_paths = 20
        dt = T / 100
        paths = np.zeros((sample_paths, 101))
        paths[:, 0] = S

        for t in range(100):
            z = np.random.randn(sample_paths)
            paths[:, t + 1] = paths[:, t] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
            )

        fig = go.Figure()
        time_axis = np.linspace(0, T, 101)

        for i in range(sample_paths):
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=paths[i],
                    mode="lines",
                    line=dict(width=0.8, color=f"rgba(96, 165, 250, 0.4)"),
                    showlegend=False,
                )
            )

        fig.add_hline(
            y=K, line_dash="dash", line_color="#ef4444", annotation_text="Strike"
        )
        if exotic_type == "Barrier":
            fig.add_hline(
                y=barrier,
                line_dash="dash",
                line_color="#f59e0b",
                annotation_text="Barrier",
            )

        fig.update_layout(**get_chart_layout("Sample Price Paths", 350))
        fig.update_xaxes(title_text="Time (years)")
        fig.update_yaxes(title_text="Price ($)")
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# HELP SECTION
# =============================================================================
if not run and not run_greeks:
    st.info("👆 Select an exotic option type and click **Price** to compute value.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <h4 style="color: #60a5fa;">Asian Options</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem;">
                Payoff based on average price over life.
                Lower volatility → cheaper than vanilla.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <h4 style="color: #a78bfa;">Barrier Options</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem;">
                Activated/deactivated if barrier touched.
                Knock-out = cheaper, Knock-in = contingent.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <h4 style="color: #34d399;">Lookback Options</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem;">
                Strike set at min/max achieved price.
                Most expensive exotics due to hindsight.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
