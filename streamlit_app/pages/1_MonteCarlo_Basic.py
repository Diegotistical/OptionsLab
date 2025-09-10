import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import logging  # Added for debugging

from streamlit_app.st_utils import (
    price_monte_carlo,
    greeks_mc_delta_gamma,
    timeit_ms,
    show_repo_status,
    simulate_payoffs  # <-- IMPORT THE EXPOSED FUNCTION
)

# Configure logging
logger = logging.getLogger("monte_carlo")

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Monte Carlo Option Pricing (Basic)",
    layout="wide"
)

# ------------------- HEADER -------------------
st.title("📈 Monte Carlo Option Pricing (Basic)")
st.markdown("""
Use Monte Carlo simulation to price **European call/put options** and compute Greeks.  
Adjust parameters in the sidebar, then click **Run Monte Carlo Pricing**.
""")

# ------------------- SIDEBAR INPUTS -------------------
with st.sidebar:
    st.header("Model Parameters")

    # Market Inputs
    S = st.number_input("Spot Price (S)", min_value=0.0, value=100.0, step=1.0)
    K = st.number_input("Strike Price (K)", min_value=0.0, value=100.0, step=1.0)
    T = st.number_input("Time to Maturity (T, years)", min_value=0.01, value=1.0, step=0.01)
    r = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01, format="%.4f")
    sigma = st.number_input("Volatility (σ)", value=0.2, step=0.01, format="%.4f")
    q = st.number_input("Dividend Yield (q)", value=0.0, step=0.01, format="%.4f")
    option_type = st.selectbox("Option Type", ["call", "put"])

    # Simulation Inputs
    num_sim = st.number_input("Number of Simulations", min_value=1000, value=50_000, step=1000)
    num_steps = st.number_input("Steps per Path", min_value=1, value=100, step=1)
    seed = st.number_input("Random Seed", value=42, step=1)
    use_numba = st.checkbox("Use Numba Acceleration", value=False)

    st.markdown("---")
    show_repo_status()
    run = st.button("🚀 Run Monte Carlo Pricing")

# ------------------- MAIN CONTENT -------------------
st.markdown("### Results")

if run:
    try:
        # ---------- Monte Carlo Pricing ----------
        price, t_price_ms = timeit_ms(
            price_monte_carlo,
            S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type, q=q,
            num_sim=num_sim, num_steps=num_steps, seed=seed, use_numba=use_numba
        )

        # ---------- Greeks ----------
        delta, gamma = greeks_mc_delta_gamma(
            S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type, q=q,
            num_sim=num_sim, num_steps=num_steps, seed=seed, h=1e-3, use_numba=use_numba
        )

        # ---------- Display Metrics ----------
        col1, col2, col3 = st.columns(3)
        col1.metric("Option Price", f"{price:.4f}" if price is not None else "N/A")
        col2.metric("Delta", f"{delta:.4f}" if delta is not None else "N/A")
        col3.metric("Gamma", f"{gamma:.4f}" if gamma is not None else "N/A")

        # ---------- Confidence Interval ----------
        try:
            # Use the SAME fallback function as pricing for consistency
            discounted = simulate_payoffs(S, K, T, r, sigma, option_type, num_sim, num_steps, seed, q)
            mean_price = np.mean(discounted)
            std_error = np.std(discounted) / np.sqrt(num_sim)
            ci_lower = mean_price - 1.96 * std_error
            ci_upper = mean_price + 1.96 * std_error
            st.markdown(f"**95% Confidence Interval:** {ci_lower:.4f} - {ci_upper:.4f}")
        except Exception as e:
            st.error(f"Confidence interval calculation failed: {str(e)}")

        # ---------- Histogram ----------
        st.markdown("#### Payoff Distribution")
        st.bar_chart(pd.DataFrame({"Discounted Payoff": discounted}), y="Discounted Payoff", use_container_width=True)
    
    except Exception as e:
        st.error(f"Critical error during pricing: {str(e)}")
        logger.exception("Critical pricing failure")

else:
    st.info("Set parameters in the sidebar and click **Run Monte Carlo Pricing** to see results.")