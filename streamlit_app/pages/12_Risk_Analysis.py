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
tab_basic, tab_option, tab_stress, tab_arb = st.tabs(
    ["Basic Risk Metrics", "Option Position Risk", "Stress Test", "Arbitrage Check"]
)

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

# =============================================================================
# TAB 3: STRESS TEST
# =============================================================================
with tab_stress:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 0.8])

    with col1:
        st.markdown("**Portfolio**")
        port_value = st.number_input(
            "Portfolio Value ($)", value=1000000, step=10000, key="s_val"
        )
        port_beta = st.number_input("Beta", value=1.0, step=0.1, key="s_beta")

    with col2:
        st.markdown("**Stress Scenario**")
        scenario = st.selectbox(
            "Scenario",
            [
                "2008 Financial Crisis (-50%)",
                "COVID Crash Mar 2020 (-34%)",
                "Black Monday 1987 (-22%)",
                "Tech Crash 2022 (-28%)",
                "Custom",
            ],
            key="s_scenario",
        )

    with col3:
        st.markdown("**Custom Shock**")
        custom_shock = st.number_input(
            "Equity Shock (%)", value=-20.0, step=5.0, key="s_custom"
        )
        vol_shock = st.number_input("Vol Shock (%)", value=50.0, step=10.0, key="s_vol")

    with col4:
        st.markdown("**Run**")
        st.write("")
        run_stress = st.button(
            "Stress Test", type="primary", use_container_width=True, key="s_run"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if run_stress:
        section_divider()

        # Define shocks
        shocks = {
            "2008 Financial Crisis (-50%)": -0.50,
            "COVID Crash Mar 2020 (-34%)": -0.34,
            "Black Monday 1987 (-22%)": -0.22,
            "Tech Crash 2022 (-28%)": -0.28,
            "Custom": custom_shock / 100,
        }

        shock = shocks[scenario]
        portfolio_loss = port_value * shock * port_beta

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">Scenario</div><div class="metric-value" style="font-size: 1rem;">{scenario.split("(")[0]}</div></div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">Market Shock</div><div class="metric-value" style="color: #ef4444;">{shock:.0%}</div></div>""",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">Portfolio Impact</div><div class="metric-value" style="color: #ef4444;">${portfolio_loss:,.0f}</div></div>""",
                unsafe_allow_html=True,
            )
        with c4:
            new_val = port_value + portfolio_loss
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">New Value</div><div class="metric-value">${new_val:,.0f}</div></div>""",
                unsafe_allow_html=True,
            )

        section_divider()

        # Sensitivity table
        st.markdown("**Sensitivity Analysis**")
        shocks_range = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20]
        betas = [0.5, 0.75, 1.0, 1.25, 1.5]

        results = []
        for b in betas:
            row = {"Beta": b}
            for s in shocks_range:
                row[f"{s:+.0%}"] = port_value * s * b
            results.append(row)

        df = pd.DataFrame(results)
        st.dataframe(
            df.style.format(
                {"Beta": "{:.2f}"} | {c: "${:,.0f}" for c in df.columns if c != "Beta"}
            ),
            hide_index=True,
        )

# =============================================================================
# TAB 4: ARBITRAGE CHECK
# =============================================================================
with tab_arb:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 0.8])

    with col1:
        st.markdown("**Butterfly Spread**")
        K1 = st.number_input("Low Strike ($)", value=95.0, step=1.0, key="a_k1")
        K2 = st.number_input("Mid Strike ($)", value=100.0, step=1.0, key="a_k2")
        K3 = st.number_input("High Strike ($)", value=105.0, step=1.0, key="a_k3")

    with col2:
        st.markdown("**Option Prices**")
        C1 = st.number_input("C(K1) Price ($)", value=8.0, step=0.5, key="a_c1")
        C2 = st.number_input("C(K2) Price ($)", value=5.0, step=0.5, key="a_c2")
        C3 = st.number_input("C(K3) Price ($)", value=2.5, step=0.5, key="a_c3")

    with col3:
        st.markdown("**Put Prices (Put-Call Parity)**")
        spot = st.number_input("Spot ($)", value=100.0, step=1.0, key="a_spot")
        r_pct = st.number_input("Rate (%)", value=5.0, step=0.5, key="a_r")
        T = st.number_input("Time (yrs)", value=0.25, step=0.05, key="a_T")

    with col4:
        st.markdown("**Check**")
        st.write("")
        run_arb = st.button(
            "Check Arbitrage", type="primary", use_container_width=True, key="a_run"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if run_arb:
        section_divider()

        # Butterfly arbitrage: C(K1) - 2C(K2) + C(K3) >= 0 for convexity
        butterfly = C1 - 2 * C2 + C3

        # Put-Call parity: C - P = S - K*exp(-rT)
        r = r_pct / 100
        parity_k1 = C1 - (spot - K1 * np.exp(-r * T))
        parity_k2 = C2 - (spot - K2 * np.exp(-r * T))
        parity_k3 = C3 - (spot - K3 * np.exp(-r * T))

        # Calendar spread: For same strike, longer dated should be worth more

        c1, c2, c3 = st.columns(3)

        with c1:
            color = "#10b981" if butterfly >= 0 else "#ef4444"
            status = "PASSED" if butterfly >= 0 else "VIOLATION"
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Butterfly Convexity</div>
                <div class="metric-value" style="color: {color};">{status}</div>
                <div class="metric-delta">C(K1) - 2C(K2) + C(K3) = ${butterfly:.2f}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Implied Put (K2)</div>
                <div class="metric-value">${parity_k2:.2f}</div>
                <div class="metric-delta">via Put-Call Parity</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with c3:
            spread = K2 - K1
            max_payoff = spread
            cost = C1 - 2 * C2 + C3
            profit = max_payoff - cost if cost > 0 else 0
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Butterfly P&L</div>
                <div class="metric-value">${profit:.2f}</div>
                <div class="metric-delta">Max at K2=${K2:.0f}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        section_divider()

        # Payoff diagram
        spot_range = np.linspace(K1 - 10, K3 + 10, 100)

        # Long butterfly: +1 C(K1), -2 C(K2), +1 C(K3)
        payoff_k1 = np.maximum(spot_range - K1, 0)
        payoff_k2 = np.maximum(spot_range - K2, 0)
        payoff_k3 = np.maximum(spot_range - K3, 0)

        butterfly_payoff = payoff_k1 - 2 * payoff_k2 + payoff_k3 - (C1 - 2 * C2 + C3)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=spot_range,
                y=butterfly_payoff,
                mode="lines",
                name="Butterfly P&L",
                line=dict(color="#a78bfa", width=3),
                fill="tozeroy",
                fillcolor="rgba(167, 139, 250, 0.2)",
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        fig.add_vline(
            x=K1, line_dash="dot", line_color="#60a5fa", annotation_text=f"K1=${K1}"
        )
        fig.add_vline(
            x=K2, line_dash="dot", line_color="#10b981", annotation_text=f"K2=${K2}"
        )
        fig.add_vline(
            x=K3, line_dash="dot", line_color="#f59e0b", annotation_text=f"K3=${K3}"
        )

        fig.update_layout(**get_chart_layout("Butterfly Spread Payoff", 400))
        fig.update_xaxes(title_text="Spot at Expiry ($)")
        fig.update_yaxes(title_text="P&L ($)")
        st.plotly_chart(fig, use_container_width=True)
