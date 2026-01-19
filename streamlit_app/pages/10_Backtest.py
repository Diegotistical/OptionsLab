# streamlit_app/pages/10_Backtest.py
"""
Strategy Backtester - Streamlit Page.

Historical P&L simulation for delta hedging and gamma strategies.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    from components import (
        apply_custom_css,
        format_price,
        get_chart_layout,
        page_header,
        section_divider,
    )
except ImportError:
    from streamlit_app.components import (
        apply_custom_css,
        format_price,
        get_chart_layout,
        page_header,
        section_divider,
    )

try:
    from src.backtesting import BacktestEngine, BacktestResult

    BACKTEST_AVAILABLE = True
except ImportError as e:
    BACKTEST_AVAILABLE = False
    st.error(f"Backtest module not available: {e}")

apply_custom_css()

# =============================================================================
# HEADER
# =============================================================================
page_header(
    "Strategy Backtester",
    "Historical P&L simulation with real market data",
)

if not BACKTEST_AVAILABLE:
    st.error("📦 Install yfinance: `pip install yfinance`")
    st.stop()

section_divider()

# =============================================================================
# INPUT
# =============================================================================
st.markdown('<div class="input-section">', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 0.8])

with col1:
    st.markdown("**Asset**")
    ticker = st.text_input("Ticker", value="SPY", max_chars=10, key="bt_ticker").upper()
    strike = st.number_input(
        "Strike ($)", min_value=50.0, max_value=1000.0, value=450.0, step=10.0
    )

with col2:
    st.markdown("**Period**")
    start_date = st.date_input("Start", value=datetime.now() - timedelta(days=180))
    end_date = st.date_input("End", value=datetime.now() - timedelta(days=1))

with col3:
    st.markdown("**Option**")
    option_type = st.selectbox("Type", ["call", "put"], key="bt_type")
    maturity = st.number_input(
        "Initial TTM (yrs)", min_value=0.05, max_value=1.0, value=0.25, step=0.05
    )

with col4:
    st.markdown("**Strategy**")
    sigma_pct = st.number_input(
        "Assumed IV (%)", min_value=5.0, max_value=100.0, value=20.0, step=1.0
    )
    hedge_freq = st.selectbox("Hedge Frequency", ["daily", "weekly"])

with col5:
    st.markdown("**Run**")
    st.write("")
    run_bt = st.button("🚀 Backtest", type="primary", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# BACKTEST
# =============================================================================
if run_bt:
    section_divider()

    with st.spinner(f"Running backtest on {ticker}..."):
        try:
            engine = BacktestEngine()
            result = engine.run_delta_hedge(
                ticker=ticker,
                start_date=str(start_date),
                end_date=str(end_date),
                strike=strike,
                maturity_years=maturity,
                sigma=sigma_pct / 100,
                option_type=option_type,
                hedge_frequency=hedge_freq,
            )

            # Also get realized vol comparison
            vol_comparison = engine.compare_model_to_realized(
                ticker, str(start_date), str(end_date)
            )

        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.stop()

    # Results
    st.markdown("### 📊 Results")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    pnl_color = "#10b981" if result.total_pnl >= 0 else "#ef4444"

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Total P&L</div>
            <div class="metric-value" style="color: {pnl_color};">${result.total_pnl:+,.0f}</div>
            <div class="metric-delta">{result.total_return_pct:+.1f}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Premium Received</div>
            <div class="metric-value">${result.initial_value:,.0f}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        sharpe_color = "#10b981" if result.sharpe_ratio > 0.5 else "#f59e0b"
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value" style="color: {sharpe_color};">{result.sharpe_ratio:.2f}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value">{result.max_drawdown:.1%}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">{result.win_rate:.1%}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col6:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Rebalances</div>
            <div class="metric-value">{result.n_trades}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    section_divider()

    # P&L Chart
    st.markdown("### 📈 Cumulative P&L")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=result.dates,
            y=result.cumulative_pnl,
            mode="lines",
            name="Cumulative P&L",
            line=dict(color="#60a5fa", width=2),
            fill="tozeroy",
            fillcolor="rgba(96, 165, 250, 0.2)",
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")

    fig.update_layout(
        **get_chart_layout(
            f"Delta Hedge P&L - {ticker} {option_type.upper()} ${strike}", 400
        )
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="P&L ($)")

    st.plotly_chart(fig, use_container_width=True)

    # Daily P&L histogram
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Daily P&L Distribution")

        fig2 = go.Figure()
        fig2.add_trace(
            go.Histogram(
                x=result.daily_pnl,
                nbinsx=30,
                marker_color="#a78bfa",
                opacity=0.7,
            )
        )

        fig2.add_vline(x=0, line_dash="dash", line_color="#ef4444")
        fig2.add_vline(
            x=np.mean(result.daily_pnl),
            line_dash="dot",
            line_color="#10b981",
            annotation_text=f"Mean: ${np.mean(result.daily_pnl):.0f}",
        )

        fig2.update_layout(**get_chart_layout("Daily P&L Distribution", 350))
        fig2.update_xaxes(title_text="Daily P&L ($)")
        fig2.update_yaxes(title_text="Frequency")

        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("### 📉 Realized vs Implied Vol")

        if vol_comparison.get("realized_vol_series"):
            fig3 = go.Figure()

            fig3.add_trace(
                go.Scatter(
                    x=vol_comparison["dates"],
                    y=[v * 100 for v in vol_comparison["realized_vol_series"]],
                    mode="lines",
                    name="Realized Vol (20d)",
                    line=dict(color="#34d399", width=2),
                )
            )

            fig3.add_hline(
                y=sigma_pct,
                line_dash="dash",
                line_color="#f59e0b",
                annotation_text=f"Assumed IV: {sigma_pct}%",
            )

            fig3.update_layout(
                **get_chart_layout("Realized vs Implied Volatility", 350)
            )
            fig3.update_xaxes(title_text="Date")
            fig3.update_yaxes(title_text="Volatility (%)")

            st.plotly_chart(fig3, use_container_width=True)

    # Analysis
    section_divider()
    st.markdown("### 💡 Analysis")

    realized_mean = vol_comparison.get("realized_vol_mean", 0) * 100
    vol_diff = sigma_pct - realized_mean

    if vol_diff > 2:
        st.success(
            f"✅ **Volatility Edge:** Sold IV ({sigma_pct:.1f}%) was {vol_diff:.1f}% above realized ({realized_mean:.1f}%). This is a profitable volatility trade."
        )
    elif vol_diff < -2:
        st.warning(
            f"⚠️ **Volatility Loss:** Sold IV ({sigma_pct:.1f}%) was {abs(vol_diff):.1f}% below realized ({realized_mean:.1f}%). Short vol lost money."
        )
    else:
        st.info(
            f"📊 IV ({sigma_pct:.1f}%) ≈ Realized ({realized_mean:.1f}%). P&L driven by path and gamma."
        )

# Help
if not run_bt:
    st.info(
        "👆 Configure parameters and click **Backtest** to simulate historical P&L."
    )

    st.markdown(
        """
    <div class="metric-card">
        <h4 style="color: #60a5fa;">📈 What This Tests</h4>
        <ul style="color: #cbd5e1; line-height: 1.8;">
            <li><strong>Delta Hedging:</strong> Sell option → hedge with stock → rebalance</li>
            <li><strong>Gamma P&L:</strong> Profit/loss from stock path convexity</li>
            <li><strong>IV vs Realized:</strong> Compare assumed vol to what happened</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )
