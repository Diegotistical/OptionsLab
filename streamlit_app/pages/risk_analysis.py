# streamlit_app/pages/risk_analysis.py
"""
Risk Analysis - Streamlit Page.

VaR, Expected Shortfall, and option risk analytics.
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
from scipy import stats

st.set_page_config(
    page_title="Risk Analysis",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    from components import (
        apply_custom_css,
        format_time_ms,
        get_chart_layout,
        page_header,
        section_divider,
    )
except ImportError:
    from streamlit_app.components import (
        apply_custom_css,
        format_time_ms,
        get_chart_layout,
        page_header,
        section_divider,
    )

apply_custom_css()


# =============================================================================
# VECTORIZED RISK CALCULATIONS
# =============================================================================


def compute_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Compute Value at Risk (vectorized)."""
    return -np.percentile(returns, (1 - confidence) * 100)


def compute_es(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Compute Expected Shortfall (vectorized)."""
    var = compute_var(returns, confidence)
    tail = returns[returns <= -var]
    return -np.mean(tail) if len(tail) > 0 else var


def generate_returns(
    n: int, mean: float, std: float, dist: str, seed: int
) -> np.ndarray:
    """Generate synthetic returns (vectorized)."""
    rng = np.random.default_rng(seed)
    if dist == "normal":
        return rng.normal(mean, std, n)
    elif dist == "t":
        return stats.t.rvs(df=5, loc=mean, scale=std, size=n, random_state=seed)
    elif dist == "skewed":
        return stats.skewnorm.rvs(a=-2, loc=mean, scale=std, size=n, random_state=seed)
    return rng.normal(mean, std, n)


# =============================================================================
# HEADER
# =============================================================================
page_header(
    "Risk Analysis", "Value at Risk, Expected Shortfall, and Option Risk Metrics"
)

# =============================================================================
# TABS FOR DIFFERENT ANALYSES
# =============================================================================
tab_basic, tab_option = st.tabs(["📊 Basic Risk Metrics", "📈 Option Position Risk"])

# =============================================================================
# TAB 1: BASIC RISK
# =============================================================================
with tab_basic:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 0.8])

    with col1:
        st.markdown("**Distribution**")
        n_samples = st.select_slider(
            "Samples",
            options=[5000, 10000, 25000, 50000, 100000],
            value=10000,
            key="r1_n",
        )
        distribution = st.selectbox("Type", ["normal", "t", "skewed"], key="r1_dist")

    with col2:
        st.markdown("**Parameters**")
        mean_ret = st.number_input("Mean (%)", value=0.0, step=0.1, key="r1_mean") / 100
        std_ret = (
            st.number_input("Volatility (%)", value=2.0, step=0.1, key="r1_std") / 100
        )

    with col3:
        st.markdown("**Risk**")
        confidence = st.select_slider(
            "Confidence", options=[0.90, 0.95, 0.99], value=0.95, key="r1_conf"
        )
        seed = st.number_input("Seed", value=42, key="r1_seed")

    with col4:
        st.markdown("**Run**")
        st.write("")
        run_basic = st.button(
            "📊 Analyze", type="primary", use_container_width=True, key="r1_run"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if run_basic:
        t_start = time.perf_counter()

        returns = generate_returns(n_samples, mean_ret, std_ret, distribution, seed)
        var = compute_var(returns, confidence)
        es = compute_es(returns, confidence)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        t_total = (time.perf_counter() - t_start) * 1000

        section_divider()

        c1, c2, c3, c4, c5, c6 = st.columns(6)

        with c1:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">VaR ({confidence:.0%})</div><div class="metric-value">{var*100:.2f}%</div></div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">ES</div><div class="metric-value">{es*100:.2f}%</div></div>""",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">Std Dev</div><div class="metric-value">{np.std(returns)*100:.2f}%</div></div>""",
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">Skew</div><div class="metric-value">{skewness:.3f}</div></div>""",
                unsafe_allow_html=True,
            )
        with c5:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">Kurtosis</div><div class="metric-value">{kurtosis:.3f}</div></div>""",
                unsafe_allow_html=True,
            )
        with c6:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">Time</div><div class="metric-value">{format_time_ms(t_total)}</div></div>""",
                unsafe_allow_html=True,
            )

        section_divider()

        # Charts
        c1, c2 = st.columns(2)

        with c1:
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=returns * 100, nbinsx=60, marker_color="#60a5fa", opacity=0.7
                )
            )
            fig.add_vline(
                x=-var * 100,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"VaR: {var*100:.1f}%",
            )
            fig.add_vline(
                x=-es * 100,
                line_dash="dot",
                line_color="#f97316",
                annotation_text=f"ES: {es*100:.1f}%",
            )
            fig.update_layout(**get_chart_layout("Return Distribution", 350))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # VaR by confidence
            confs = [0.90, 0.95, 0.975, 0.99]
            vars_l = [compute_var(returns, c) * 100 for c in confs]
            es_l = [compute_es(returns, c) * 100 for c in confs]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=[f"{c:.0%}" for c in confs],
                    y=vars_l,
                    name="VaR",
                    marker_color="#3b82f6",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=[f"{c:.0%}" for c in confs],
                    y=es_l,
                    name="ES",
                    marker_color="#ef4444",
                )
            )
            fig.update_layout(
                **get_chart_layout("Risk by Confidence", 350), barmode="group"
            )
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2: OPTION RISK
# =============================================================================
with tab_option:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 0.8])

    with col1:
        st.markdown("**Option**")
        S0 = st.number_input("Spot ($)", value=100.0, step=1.0, key="r2_spot")
        K = st.number_input("Strike ($)", value=100.0, step=1.0, key="r2_strike")
        premium = st.number_input("Premium ($)", value=10.0, step=0.5, key="r2_prem")

    with col2:
        st.markdown("**Position**")
        opt_type = st.selectbox("Type", ["call", "put"], key="r2_type")
        position = st.number_input("Contracts", value=100, step=10, key="r2_pos")
        days = st.number_input("Days", value=5, step=1, key="r2_days")

    with col3:
        st.markdown("**Simulation**")
        n_sims = st.select_slider(
            "Samples", options=[5000, 10000, 25000, 50000], value=10000, key="r2_n"
        )
        vol_pct = st.number_input("Daily Vol (%)", value=2.0, step=0.1, key="r2_vol")
        confidence = st.select_slider(
            "Confidence", options=[0.90, 0.95, 0.99], value=0.95, key="r2_conf"
        )

    with col4:
        st.markdown("**Run**")
        st.write("")
        run_opt = st.button(
            "📈 Calculate P&L", type="primary", use_container_width=True, key="r2_run"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if run_opt:
        t_start = time.perf_counter()

        # Generate spot changes
        daily_vol = vol_pct / 100
        period_vol = daily_vol * np.sqrt(days)
        rng = np.random.default_rng(42)
        spot_changes = rng.normal(0, period_vol, n_sims)

        # Calculate P&L
        S_final = S0 * (1 + spot_changes)
        if opt_type == "call":
            payoff = np.maximum(S_final - K, 0)
        else:
            payoff = np.maximum(K - S_final, 0)

        pnl = (payoff - premium) * position

        # Risk metrics
        pnl_var = -np.percentile(pnl, (1 - confidence) * 100)
        pnl_es = -np.mean(pnl[pnl <= -pnl_var]) if np.any(pnl <= -pnl_var) else pnl_var

        t_total = (time.perf_counter() - t_start) * 1000

        section_divider()

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">VaR ({confidence:.0%})</div><div class="metric-value">${pnl_var:,.0f}</div></div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">ES</div><div class="metric-value">${pnl_es:,.0f}</div></div>""",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">Max Loss</div><div class="metric-value">${pnl.min():,.0f}</div></div>""",
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">Max Gain</div><div class="metric-value">${pnl.max():,.0f}</div></div>""",
                unsafe_allow_html=True,
            )
        with c5:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">Time</div><div class="metric-value">{format_time_ms(t_total)}</div></div>""",
                unsafe_allow_html=True,
            )

        section_divider()

        # P&L histogram
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(x=pnl, nbinsx=60, marker_color="#a78bfa", opacity=0.8)
        )
        fig.add_vline(x=0, line_color="#94a3b8")
        fig.add_vline(
            x=-pnl_var,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"VaR: ${pnl_var:,.0f}",
        )
        fig.update_layout(**get_chart_layout("Option P&L Distribution", 400))
        fig.update_xaxes(title_text="P&L ($)")
        st.plotly_chart(fig, use_container_width=True)
