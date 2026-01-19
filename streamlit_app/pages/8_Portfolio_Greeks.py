# streamlit_app/pages/8_Portfolio_Greeks.py
"""
Portfolio Greeks - Streamlit Page.

Aggregate Greeks across multiple option positions with scenario analysis.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Portfolio Greeks",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    from components import (
        apply_custom_css,
        format_greek,
        format_price,
        get_chart_layout,
        page_header,
        section_divider,
    )
except ImportError:
    from streamlit_app.components import (
        apply_custom_css,
        format_greek,
        format_price,
        get_chart_layout,
        page_header,
        section_divider,
    )

try:
    from src.greeks import compute_greeks
    from src.pricing_models import black_scholes

    AVAILABLE = True
except ImportError:
    AVAILABLE = False

apply_custom_css()

# =============================================================================
# HEADER
# =============================================================================
page_header(
    "Portfolio Greeks",
    "Aggregate risk metrics across multiple option positions",
)

section_divider()

# =============================================================================
# SESSION STATE FOR POSITIONS
# =============================================================================
if "positions" not in st.session_state:
    st.session_state.positions = []

# =============================================================================
# ADD POSITION
# =============================================================================
st.markdown("### ➕ Add Position")
st.markdown('<div class="input-section">', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 0.8])

with col1:
    st.markdown("**Option**")
    pos_type = st.selectbox("Type", ["call", "put"], key="add_type")
    pos_side = st.selectbox("Side", ["long", "short"], key="add_side")

with col2:
    st.markdown("**Strike & Quantity**")
    pos_K = st.number_input(
        "Strike ($)",
        min_value=50.0,
        max_value=200.0,
        value=100.0,
        step=5.0,
        key="add_K",
    )
    pos_qty = st.number_input(
        "Quantity", min_value=1, max_value=1000, value=10, key="add_qty"
    )

with col3:
    st.markdown("**Market**")
    pos_T = st.number_input(
        "Maturity (yrs)",
        min_value=0.01,
        max_value=2.0,
        value=0.25,
        step=0.05,
        key="add_T",
    )
    pos_sigma = st.number_input(
        "Vol (%)", min_value=5.0, max_value=100.0, value=25.0, step=1.0, key="add_sigma"
    )

with col4:
    st.markdown("**Common**")
    spot = st.number_input(
        "Spot ($)", min_value=50.0, max_value=200.0, value=100.0, step=1.0, key="spot"
    )
    rate = st.number_input(
        "Rate (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.25, key="rate"
    )

with col5:
    st.markdown("**Actions**")
    st.write("")
    add_pos = st.button("➕ Add", type="primary", use_container_width=True)
    clear_all = st.button("🗑️ Clear All", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# Handle actions
if add_pos:
    position = {
        "type": pos_type,
        "side": pos_side,
        "K": pos_K,
        "qty": pos_qty,
        "T": pos_T,
        "sigma": pos_sigma / 100.0,
    }
    st.session_state.positions.append(position)
    st.rerun()

if clear_all:
    st.session_state.positions = []
    st.rerun()

# =============================================================================
# CURRENT POSITIONS
# =============================================================================
if st.session_state.positions:
    section_divider()
    st.markdown("### 📋 Current Positions")

    r = rate / 100.0
    S = spot

    # Calculate Greeks for each position
    position_data = []
    total_delta = 0
    total_gamma = 0
    total_vega = 0
    total_theta = 0
    total_value = 0

    for i, pos in enumerate(st.session_state.positions):
        K = pos["K"]
        T = pos["T"]
        sigma = pos["sigma"]
        qty = pos["qty"]
        sign = 1 if pos["side"] == "long" else -1

        # Calculate Greeks using BS analytical
        from scipy.stats import norm

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if pos["type"] == "call":
            delta = norm.cdf(d1)
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2)
            ) / 365
        else:
            delta = norm.cdf(d1) - 1
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
            ) / 365

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol move

        # Apply sign and quantity
        pos_delta = sign * qty * delta * 100  # Delta per share, 100 shares per contract
        pos_gamma = sign * qty * gamma * 100
        pos_vega = sign * qty * vega * 100
        pos_theta = sign * qty * theta * 100
        pos_value = sign * qty * price * 100

        total_delta += pos_delta
        total_gamma += pos_gamma
        total_vega += pos_vega
        total_theta += pos_theta
        total_value += pos_value

        position_data.append(
            {
                "Type": f"{pos['side'].upper()} {pos['type'].upper()}",
                "Strike": f"${K:.0f}",
                "Qty": qty,
                "Expiry": f"{T:.2f}y",
                "IV": f"{sigma*100:.0f}%",
                "Price": f"${price:.2f}",
                "Delta": f"{pos_delta:.0f}",
                "Gamma": f"{pos_gamma:.1f}",
                "Vega": f"${pos_vega:.0f}",
                "Theta": f"${pos_theta:.0f}",
                "Value": f"${pos_value:,.0f}",
            }
        )

    # Display positions table
    df = pd.DataFrame(position_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Remove position buttons
    cols = st.columns(len(st.session_state.positions) + 1)
    for i, _ in enumerate(st.session_state.positions):
        with cols[i]:
            if st.button(f"❌ Remove #{i+1}", key=f"remove_{i}"):
                st.session_state.positions.pop(i)
                st.rerun()

    section_divider()

    # ==========================================================================
    # AGGREGATE GREEKS
    # ==========================================================================
    st.markdown("### 📈 Portfolio Greeks")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        color = "#10b981" if total_delta >= 0 else "#ef4444"
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Total Delta (Δ)</div>
            <div class="metric-value" style="color: {color};">{total_delta:+,.0f}</div>
            <div class="metric-delta">shares equivalent</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        color = "#10b981" if total_gamma >= 0 else "#ef4444"
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Total Gamma (Γ)</div>
            <div class="metric-value" style="color: {color};">{total_gamma:+,.1f}</div>
            <div class="metric-delta">Δ change per $1</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        color = "#10b981" if total_vega >= 0 else "#ef4444"
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Total Vega (ν)</div>
            <div class="metric-value" style="color: {color};">${total_vega:+,.0f}</div>
            <div class="metric-delta">P&L per 1% vol</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        color = "#10b981" if total_theta >= 0 else "#ef4444"
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Total Theta (Θ)</div>
            <div class="metric-value" style="color: {color};">${total_theta:+,.0f}</div>
            <div class="metric-delta">daily decay</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Portfolio Value</div>
            <div class="metric-value">${total_value:+,.0f}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    section_divider()

    # ==========================================================================
    # SCENARIO ANALYSIS
    # ==========================================================================
    st.markdown("### 🎯 Scenario P&L Heatmap")

    # Generate P&L for spot/vol grid
    spot_moves = np.linspace(-0.15, 0.15, 13)  # -15% to +15%
    vol_moves = np.linspace(-0.10, 0.10, 9)  # -10% to +10% vol

    pnl_grid = np.zeros((len(vol_moves), len(spot_moves)))

    for i, dv in enumerate(vol_moves):
        for j, ds in enumerate(spot_moves):
            new_S = S * (1 + ds)
            scenario_pnl = 0

            for pos in st.session_state.positions:
                K = pos["K"]
                T = pos["T"]
                new_sigma = pos["sigma"] + dv
                qty = pos["qty"]
                sign = 1 if pos["side"] == "long" else -1

                if new_sigma < 0.01:
                    new_sigma = 0.01

                d1 = (np.log(new_S / K) + (r + 0.5 * new_sigma**2) * T) / (
                    new_sigma * np.sqrt(T)
                )
                d2 = d1 - new_sigma * np.sqrt(T)

                if pos["type"] == "call":
                    new_price = new_S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                    orig_d1 = (np.log(S / K) + (r + 0.5 * pos["sigma"] ** 2) * T) / (
                        pos["sigma"] * np.sqrt(T)
                    )
                    orig_d2 = orig_d1 - pos["sigma"] * np.sqrt(T)
                    orig_price = S * norm.cdf(orig_d1) - K * np.exp(-r * T) * norm.cdf(
                        orig_d2
                    )
                else:
                    new_price = K * np.exp(-r * T) * norm.cdf(-d2) - new_S * norm.cdf(
                        -d1
                    )
                    orig_d1 = (np.log(S / K) + (r + 0.5 * pos["sigma"] ** 2) * T) / (
                        pos["sigma"] * np.sqrt(T)
                    )
                    orig_d2 = orig_d1 - pos["sigma"] * np.sqrt(T)
                    orig_price = K * np.exp(-r * T) * norm.cdf(-orig_d2) - S * norm.cdf(
                        -orig_d1
                    )

                scenario_pnl += sign * qty * (new_price - orig_price) * 100

            pnl_grid[i, j] = scenario_pnl

    fig = go.Figure(
        data=go.Heatmap(
            x=[f"{ds*100:+.0f}%" for ds in spot_moves],
            y=[f"{dv*100:+.0f}%" for dv in vol_moves],
            z=pnl_grid,
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="P&L ($)"),
            text=[[f"${v:,.0f}" for v in row] for row in pnl_grid],
            texttemplate="%{text}",
            textfont=dict(size=10),
        )
    )

    fig.update_layout(**get_chart_layout("Scenario P&L (Spot × Vol)", 450))
    fig.update_xaxes(title_text="Spot Move (%)")
    fig.update_yaxes(title_text="Vol Move (%)")

    st.plotly_chart(fig, use_container_width=True)

    # Delta hedge suggestion
    section_divider()
    st.markdown("### 🔒 Delta Hedge")

    col1, col2, col3 = st.columns(3)

    with col1:
        hedge_shares = -total_delta
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Shares to Hedge</div>
            <div class="metric-value">{hedge_shares:+,.0f}</div>
            <div class="metric-delta">{"BUY" if hedge_shares > 0 else "SELL"} shares</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        hedge_cost = abs(hedge_shares) * S
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Hedge Cost</div>
            <div class="metric-value">${hedge_cost:,.0f}</div>
            <div class="metric-delta">at current spot</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">After Hedge</div>
            <div class="metric-value">Δ ≈ 0</div>
            <div class="metric-delta">Γ = {total_gamma:+,.1f} (unchanged)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

else:
    st.info("👆 Add option positions above to analyze portfolio Greeks.")

    st.markdown(
        """
    <div class="metric-card">
        <h3 style="color: #60a5fa; margin-bottom: 1rem;">📊 Portfolio Analysis Features</h3>
        <ul style="color: #cbd5e1; line-height: 2;">
            <li><strong>Aggregate Greeks:</strong> Sum Delta, Gamma, Vega, Theta across positions</li>
            <li><strong>Scenario Analysis:</strong> P&L heatmap for spot × vol moves</li>
            <li><strong>Delta Hedging:</strong> Calculate shares needed to neutralize delta</li>
            <li><strong>Position Tracking:</strong> Add/remove positions dynamically</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )
