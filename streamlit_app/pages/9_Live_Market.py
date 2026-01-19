# streamlit_app/pages/9_Live_Market.py
"""
Live Market Data - Streamlit Page.

Real-time options chain viewer with model calibration.
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
    page_title="Live Market",
    page_icon="📡",
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

# Import market data
try:
    from src.data.market_data import (
        YFINANCE_AVAILABLE,
        get_expiries,
        get_iv_surface,
        get_options_chain,
        get_stock_price,
    )

    DATA_AVAILABLE = YFINANCE_AVAILABLE
except ImportError as e:
    DATA_AVAILABLE = False
    st.warning(f"Market data module not available: {e}")

apply_custom_css()

# =============================================================================
# HEADER
# =============================================================================
page_header(
    "Live Market Data",
    "Real-time options chains from Yahoo Finance",
)

if not DATA_AVAILABLE:
    st.error("📦 Install yfinance: `pip install yfinance`")
    st.stop()

section_divider()

# =============================================================================
# TICKER INPUT
# =============================================================================
st.markdown('<div class="input-section">', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])

with col1:
    st.markdown("**Symbol**")
    ticker = st.text_input("Ticker", value="SPY", max_chars=10, key="ticker").upper()

with col2:
    st.markdown("**Data**")
    fetch_price = st.button("📊 Get Quote", use_container_width=True)

with col3:
    st.markdown("**Options**")
    fetch_chain = st.button("⛓️ Options Chain", use_container_width=True)

with col4:
    st.markdown("**Analysis**")
    fetch_surface = st.button("🌊 Vol Surface", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# STOCK QUOTE
# =============================================================================
if fetch_price or "last_ticker" not in st.session_state:
    st.session_state.last_ticker = ticker

if fetch_price:
    with st.spinner(f"Fetching {ticker}..."):
        quote = get_stock_price(ticker)

    section_divider()

    if "error" in quote:
        st.error(f"Error: {quote['error']}")
    else:
        col1, col2, col3, col4, col5 = st.columns(5)

        color = "#10b981" if quote["change"] >= 0 else "#ef4444"
        arrow = "▲" if quote["change"] >= 0 else "▼"

        with col1:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">{ticker}</div>
                <div class="metric-value">${quote['price']:.2f}</div>
                <div class="metric-delta" style="color: {color};">{arrow} ${abs(quote['change']):.2f} ({quote['change_pct']:+.2f}%)</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Previous Close</div>
                <div class="metric-value">${quote['previous_close']:.2f}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            vol_str = f"{quote['volume']:,}" if quote["volume"] else "N/A"
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Volume</div>
                <div class="metric-value" style="font-size: 1.2rem;">{vol_str}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            if quote.get("market_cap", 0) > 0:
                cap = quote["market_cap"]
                if cap > 1e12:
                    cap_str = f"${cap/1e12:.1f}T"
                elif cap > 1e9:
                    cap_str = f"${cap/1e9:.1f}B"
                else:
                    cap_str = f"${cap/1e6:.0f}M"
            else:
                cap_str = "N/A"
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Market Cap</div>
                <div class="metric-value" style="font-size: 1.2rem;">{cap_str}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col5:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Updated</div>
                <div class="metric-value" style="font-size: 0.9rem;">Live</div>
                <div class="metric-delta">5min cache</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

# =============================================================================
# OPTIONS CHAIN
# =============================================================================
if fetch_chain:
    section_divider()
    st.markdown("### ⛓️ Options Chain")

    # Get expiries
    with st.spinner("Loading expiration dates..."):
        expiries = get_expiries(ticker)

    if not expiries:
        st.error("No options available for this ticker")
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_expiry = st.selectbox(
                "Expiration", expiries[:8], key="expiry_select"
            )

        with st.spinner(f"Loading {selected_expiry} chain..."):
            chain = get_options_chain(ticker, selected_expiry)

        if chain.empty:
            st.warning("No data for this expiration")
        else:
            spot = chain["spot"].iloc[0]
            dte = chain["dte"].iloc[0]

            st.markdown(f"**Spot: ${spot:.2f} | DTE: {dte} days**")

            # Split calls/puts
            calls = chain[chain["type"] == "call"].sort_values("strike")
            puts = chain[chain["type"] == "put"].sort_values("strike")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📈 Calls**")
                cols_to_show = ["strike", "bid", "ask", "last", "volume", "implied_vol"]
                available = [c for c in cols_to_show if c in calls.columns]
                display_calls = calls[available].copy()
                if "implied_vol" in display_calls.columns:
                    display_calls["implied_vol"] = (
                        display_calls["implied_vol"] * 100
                    ).round(1).astype(str) + "%"
                st.dataframe(
                    display_calls, use_container_width=True, hide_index=True, height=400
                )

            with col2:
                st.markdown("**📉 Puts**")
                display_puts = puts[available].copy()
                if "implied_vol" in display_puts.columns:
                    display_puts["implied_vol"] = (
                        display_puts["implied_vol"] * 100
                    ).round(1).astype(str) + "%"
                st.dataframe(
                    display_puts, use_container_width=True, hide_index=True, height=400
                )

            # Vol smile
            section_divider()
            st.markdown("### 😊 Volatility Smile")

            fig = go.Figure()

            calls_valid = calls[calls["implied_vol"] > 0]
            puts_valid = puts[puts["implied_vol"] > 0]

            fig.add_trace(
                go.Scatter(
                    x=calls_valid["strike"],
                    y=calls_valid["implied_vol"] * 100,
                    mode="lines+markers",
                    name="Calls",
                    line=dict(color="#10b981", width=2),
                    marker=dict(size=6),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=puts_valid["strike"],
                    y=puts_valid["implied_vol"] * 100,
                    mode="lines+markers",
                    name="Puts",
                    line=dict(color="#ef4444", width=2),
                    marker=dict(size=6),
                )
            )

            fig.add_vline(
                x=spot,
                line_dash="dash",
                line_color="#60a5fa",
                annotation_text=f"Spot: ${spot:.0f}",
            )

            fig.update_layout(
                **get_chart_layout(
                    f"{ticker} Implied Volatility Smile - {selected_expiry}", 400
                )
            )
            fig.update_xaxes(title_text="Strike Price ($)")
            fig.update_yaxes(title_text="Implied Volatility (%)")

            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# VOLATILITY SURFACE
# =============================================================================
if fetch_surface:
    section_divider()
    st.markdown("### 🌊 Implied Volatility Surface")

    with st.spinner("Building volatility surface (fetching multiple expiries)..."):
        surface = get_iv_surface(ticker, n_expiries=5)

    if surface.empty:
        st.error("Could not build volatility surface")
    else:
        # Pivot for 3D surface
        pivot = (
            surface.pivot_table(
                values="implied_vol", index="dte", columns="strike", aggfunc="mean"
            )
            .dropna(axis=1, how="all")
            .dropna(axis=0, how="all")
        )

        if not pivot.empty:
            fig = go.Figure(
                data=[
                    go.Surface(
                        x=pivot.columns.values,
                        y=pivot.index.values,
                        z=pivot.values * 100,
                        colorscale="Viridis",
                        colorbar=dict(title="IV (%)"),
                    )
                ]
            )

            fig.update_layout(
                title=dict(
                    text=f"{ticker} IV Surface", font=dict(size=18, color="#f8fafc")
                ),
                scene=dict(
                    xaxis_title="Strike ($)",
                    yaxis_title="Days to Expiry",
                    zaxis_title="IV (%)",
                    bgcolor="rgba(15, 23, 42, 0.9)",
                ),
                paper_bgcolor="rgba(30, 41, 59, 0.8)",
                font=dict(color="#cbd5e1"),
                height=500,
                margin=dict(l=0, r=0, t=50, b=0),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Term structure
            section_divider()
            st.markdown("### 📊 ATM Term Structure")

            atm_iv = surface.groupby("dte")["implied_vol"].mean() * 100

            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=atm_iv.index,
                    y=atm_iv.values,
                    mode="lines+markers",
                    line=dict(color="#a78bfa", width=3),
                    marker=dict(size=10),
                    fill="tozeroy",
                    fillcolor="rgba(167, 139, 250, 0.2)",
                )
            )

            fig2.update_layout(
                **get_chart_layout("ATM Implied Volatility Term Structure", 350)
            )
            fig2.update_xaxes(title_text="Days to Expiry")
            fig2.update_yaxes(title_text="IV (%)")

            st.plotly_chart(fig2, use_container_width=True)

# =============================================================================
# HELP
# =============================================================================
if not fetch_price and not fetch_chain and not fetch_surface:
    st.info("👆 Enter a ticker symbol and click a button to fetch live data.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <h4 style="color: #60a5fa;">📡 Data Sources</h4>
            <ul style="color: #cbd5e1; line-height: 1.8;">
                <li><strong>Yahoo Finance:</strong> Free, no API key required</li>
                <li><strong>Caching:</strong> 5-minute TTL to respect rate limits</li>
                <li><strong>Rate Limiting:</strong> 500ms between requests</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <h4 style="color: #a78bfa;">🎯 Popular Tickers</h4>
            <p style="color: #cbd5e1;">
                Try: <code>SPY</code>, <code>QQQ</code>, <code>AAPL</code>, 
                <code>NVDA</code>, <code>TSLA</code>, <code>MSFT</code>
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
