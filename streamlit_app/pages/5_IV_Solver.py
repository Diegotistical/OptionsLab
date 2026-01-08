# streamlit_app/pages/5_IV_Solver.py
"""
Implied Volatility Solver & Market Data Page.

Features:
- Compute IV from option price
- Fetch real option chains from Yahoo Finance
- Build and visualize IV surfaces
"""

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="IV Solver & Market Data",
    page_icon="📈",
    layout="wide",
)

# ============================================================================
# IMPORTS
# ============================================================================
try:
    from src.pricing_models.iv_solver import (
        black_scholes_price,
        implied_volatility,
        implied_volatility_vectorized,
        iv_surface_from_prices,
    )

    IV_SOLVER_AVAILABLE = True
except ImportError:
    IV_SOLVER_AVAILABLE = False

try:
    from src.utils.market_data import (
        YFINANCE_AVAILABLE,
        OptionChainParser,
        YahooFinanceFetcher,
    )

    MARKET_DATA_AVAILABLE = YFINANCE_AVAILABLE
except ImportError:
    MARKET_DATA_AVAILABLE = False

# ============================================================================
# HEADER
# ============================================================================
st.title("📈 Implied Volatility Solver & Market Data")
st.markdown("Compute implied volatility from option prices and fetch real market data.")

# Feature badges
col1, col2, col3 = st.columns(3)
with col1:
    if IV_SOLVER_AVAILABLE:
        st.success("✅ IV Solver Available")
    else:
        st.error("❌ IV Solver Not Available")
with col2:
    if MARKET_DATA_AVAILABLE:
        st.success("✅ Yahoo Finance Available")
    else:
        st.warning("⚠️ yfinance not installed")
with col3:
    st.info("📊 Real-time Data")

st.divider()

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3 = st.tabs(["🧮 IV Calculator", "📊 Market Data", "🌊 IV Surface"])

# ============================================================================
# TAB 1: IV CALCULATOR
# ============================================================================
with tab1:
    st.header("Single Option IV Calculator")

    if not IV_SOLVER_AVAILABLE:
        st.error("IV Solver module not available. Check imports.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Option Parameters")
            market_price = st.number_input(
                "Market Price ($)", value=10.45, min_value=0.01, step=0.1
            )
            spot = st.number_input(
                "Spot Price ($)", value=100.0, min_value=0.01, step=1.0
            )
            strike = st.number_input(
                "Strike Price ($)", value=100.0, min_value=0.01, step=1.0
            )

        with col2:
            st.subheader("Market Conditions")
            time_to_maturity = st.number_input(
                "Time to Maturity (years)",
                value=1.0,
                min_value=0.01,
                max_value=5.0,
                step=0.1,
            )
            risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 15.0, 5.0, 0.5) / 100
            dividend_yield = st.slider("Dividend Yield (%)", 0.0, 10.0, 0.0, 0.25) / 100
            option_type = st.selectbox("Option Type", ["call", "put"])

        if st.button(
            "🔍 Compute Implied Volatility", type="primary", use_container_width=True
        ):
            try:
                iv = implied_volatility(
                    market_price=market_price,
                    S=spot,
                    K=strike,
                    T=time_to_maturity,
                    r=risk_free_rate,
                    option_type=option_type,
                    q=dividend_yield,
                )

                # Display result
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Implied Volatility", f"{iv*100:.2f}%")
                with col2:
                    # Verify by repricing
                    repriced = black_scholes_price(
                        spot,
                        strike,
                        time_to_maturity,
                        risk_free_rate,
                        iv,
                        option_type,
                        dividend_yield,
                    )
                    st.metric("Repriced Value", f"${repriced:.4f}")
                with col3:
                    error = abs(repriced - market_price)
                    st.metric("Pricing Error", f"${error:.6f}")

                # Additional info
                moneyness = spot / strike
                st.info(
                    f"Moneyness (S/K): {moneyness:.4f} | {'ITM' if (moneyness > 1 and option_type == 'call') or (moneyness < 1 and option_type == 'put') else 'OTM'}"
                )

            except ValueError as e:
                st.error(f"Could not compute IV: {e}")

# ============================================================================
# TAB 2: MARKET DATA
# ============================================================================
with tab2:
    st.header("Real-Time Option Chain Data")

    if not MARKET_DATA_AVAILABLE:
        st.warning("yfinance not installed. Run: `pip install yfinance`")
        st.code("pip install yfinance", language="bash")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            ticker = st.text_input("Ticker Symbol", value="AAPL", max_chars=10).upper()

        with col2:
            fetch_button = st.button(
                "📥 Fetch Data", type="primary", use_container_width=True
            )

        if fetch_button or "option_chain" in st.session_state:
            if fetch_button:
                with st.spinner(f"Fetching data for {ticker}..."):
                    try:
                        fetcher = YahooFinanceFetcher()

                        # Get stock data
                        stock_data = fetcher.get_stock_data(ticker, period="1mo")
                        st.session_state["stock_data"] = stock_data

                        # Get option chain
                        option_chain = fetcher.get_option_chain(ticker)
                        st.session_state["option_chain"] = option_chain
                        st.session_state["ticker"] = ticker

                    except Exception as e:
                        st.error(f"Error fetching data: {e}")
                        st.stop()

            if "stock_data" in st.session_state and "option_chain" in st.session_state:
                stock_data = st.session_state["stock_data"]
                option_chain = st.session_state["option_chain"]
                ticker = st.session_state.get("ticker", "UNKNOWN")

                # Stock info
                st.subheader(f"📈 {ticker} Stock Info")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${stock_data.current_price:.2f}")
                with col2:
                    if not stock_data.history.empty:
                        change = (
                            stock_data.history["Close"].iloc[-1]
                            - stock_data.history["Close"].iloc[-2]
                        )
                        pct = change / stock_data.history["Close"].iloc[-2] * 100
                        st.metric("Daily Change", f"${change:.2f}", f"{pct:.2f}%")
                with col3:
                    st.metric("Expirations", len(option_chain.expiration_dates))
                with col4:
                    st.metric(
                        "Total Options",
                        len(option_chain.calls) + len(option_chain.puts),
                    )

                # Parse option chain
                parsed = OptionChainParser.parse_yahoo_chain(option_chain)
                liquid = OptionChainParser.filter_liquid_options(
                    parsed, min_volume=10, min_open_interest=50
                )

                st.subheader("📋 Option Chain")

                # Filter controls
                col1, col2, col3 = st.columns(3)
                with col1:
                    exp_filter = st.selectbox(
                        "Expiration", ["All"] + option_chain.expiration_dates
                    )
                with col2:
                    type_filter = st.selectbox("Type", ["All", "call", "put"])
                with col3:
                    show_liquid_only = st.checkbox("Liquid Options Only", value=True)

                # Apply filters
                display_df = liquid if show_liquid_only else parsed
                if exp_filter != "All":
                    display_df = display_df[display_df["expiration"] == exp_filter]
                if type_filter != "All":
                    display_df = display_df[display_df["option_type"] == type_filter]

                # Display
                columns_to_show = [
                    "strike",
                    "expiration",
                    "option_type",
                    "bid",
                    "ask",
                    "mid_price",
                    "volume",
                    "open_interest",
                    "implied_volatility",
                    "moneyness",
                ]
                available_cols = [c for c in columns_to_show if c in display_df.columns]

                st.dataframe(
                    display_df[available_cols].head(50),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    f"Showing {min(50, len(display_df))} of {len(display_df)} options"
                )

# ============================================================================
# TAB 3: IV SURFACE
# ============================================================================
with tab3:
    st.header("Implied Volatility Surface")

    if not IV_SOLVER_AVAILABLE:
        st.error("IV Solver not available")
    else:
        st.markdown("Build an IV surface from generated or market data.")

        data_source = st.radio(
            "Data Source",
            ["Generated (Demo)", "Market Data (if fetched)"],
            horizontal=True,
        )

        if data_source == "Generated (Demo)":
            col1, col2 = st.columns(2)
            with col1:
                demo_spot = st.number_input("Spot Price", value=100.0, key="demo_spot")
                demo_sigma = st.slider("Base Volatility (%)", 10, 50, 20) / 100
            with col2:
                demo_r = st.slider("Risk-Free Rate (%)", 0, 10, 5, key="demo_r") / 100
                smile_strength = st.slider("Smile Strength", 0.0, 0.5, 0.1)

            if st.button("🌊 Generate IV Surface", type="primary"):
                # Generate strikes and maturities
                strikes = np.linspace(80, 120, 15)
                maturities = np.array(
                    [0.083, 0.25, 0.5, 1.0, 2.0]
                )  # 1M, 3M, 6M, 1Y, 2Y

                # Generate prices with volatility smile
                call_prices = np.zeros((len(strikes), len(maturities)))
                for i, K in enumerate(strikes):
                    for j, T in enumerate(maturities):
                        # Add smile effect
                        moneyness = np.log(demo_spot / K)
                        smile_sigma = demo_sigma * (1 + smile_strength * moneyness**2)
                        call_prices[i, j] = black_scholes_price(
                            demo_spot, K, T, demo_r, smile_sigma, "call", 0
                        )

                # Compute IV surface
                option_data = {
                    "spot": demo_spot,
                    "strikes": strikes,
                    "maturities": maturities,
                    "call_prices": call_prices,
                }
                surface = iv_surface_from_prices(option_data, demo_r, 0)

                # Plot
                fig = go.Figure(
                    data=[
                        go.Surface(
                            x=surface["maturities"],
                            y=surface["moneyness"],
                            z=surface["call_iv"] * 100,
                            colorscale="Viridis",
                            colorbar=dict(title="IV (%)"),
                        )
                    ]
                )
                fig.update_layout(
                    title="Implied Volatility Surface",
                    scene=dict(
                        xaxis_title="Maturity (years)",
                        yaxis_title="Moneyness (K/S)",
                        zaxis_title="IV (%)",
                    ),
                    width=800,
                    height=600,
                )
                st.plotly_chart(fig, use_container_width=True)

                # IV smile plot
                st.subheader("IV Smile by Maturity")
                smile_df = pd.DataFrame(
                    surface["call_iv"] * 100,
                    columns=[f"{m:.2f}Y" for m in surface["maturities"]],
                    index=surface["moneyness"],
                )
                fig2 = px.line(smile_df, title="IV Smile Across Maturities")
                fig2.update_layout(
                    xaxis_title="Moneyness (K/S)",
                    yaxis_title="Implied Volatility (%)",
                )
                st.plotly_chart(fig2, use_container_width=True)

        else:  # Market data
            if "option_chain" not in st.session_state:
                st.warning("Please fetch market data first in the 'Market Data' tab.")
            else:
                st.info("Using fetched market data to build IV surface.")

                if st.button("🌊 Build IV Surface from Market Data"):
                    option_chain = st.session_state["option_chain"]
                    spot = option_chain.spot_price

                    # Parse and filter
                    parsed = OptionChainParser.parse_yahoo_chain(option_chain)
                    calls = parsed[
                        (parsed["option_type"] == "call") & (parsed["mid_price"] > 0.1)
                    ]

                    # Compute IVs
                    ivs = []
                    for _, row in calls.iterrows():
                        try:
                            iv = implied_volatility(
                                row["mid_price"],
                                spot,
                                row["strike"],
                                max(row.get("T", 0.1), 0.01),
                                0.05,
                                "call",
                                0,
                            )
                            ivs.append(iv)
                        except Exception:
                            ivs.append(np.nan)

                    calls = calls.copy()
                    calls["computed_iv"] = ivs
                    calls = calls.dropna(subset=["computed_iv"])

                    if len(calls) > 10:
                        fig = px.scatter(
                            calls,
                            x="moneyness",
                            y=calls["computed_iv"] * 100,
                            color="expiration",
                            title="IV Smile from Market Data",
                            labels={
                                "y": "Implied Volatility (%)",
                                "moneyness": "Moneyness (K/S)",
                            },
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough data points to build surface.")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown(
    """
**References:**
- Manaster & Koehler (1982) - Newton-Raphson IV Solver
- Dupire (1994) - Local Volatility Surface

See `docs/references.md` for full citations.
"""
)
