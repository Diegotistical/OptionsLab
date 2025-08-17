"""
Option Pricing Dashboard
Compare Black–Scholes, Binomial Tree, and Monte Carlo side by side,
with latency micro-benchmarks and (MC) Delta/Gamma.
"""

import streamlit as st
import numpy as np
import pandas as pd

from utils import (
    price_black_scholes, price_binomial, price_monte_carlo,
    greeks_mc_delta_gamma, timeit_ms
)

st.set_page_config(page_title="Option Pricing", page_icon="💸", layout="wide")
st.title("💸 Option Pricing")

with st.sidebar:
    st.subheader("Inputs")
    S = st.number_input("Spot S", 1.0, 1e6, 100.0, step=1.0)
    K = st.number_input("Strike K", 1.0, 1e6, 100.0, step=1.0)
    T = st.number_input("Time to Maturity (years)", 1e-4, 50.0, 1.0, step=0.05, format="%.4f")
    r = st.number_input("Risk-free rate r", -0.50, 1.00, 0.05, step=0.005, format="%.4f")
    sigma = st.number_input("Volatility σ", 0.0, 5.0, 0.20, step=0.01, format="%.4f")
    q = st.number_input("Dividend yield q", 0.0, 1.0, 0.0, step=0.005, format="%.4f")
    option_type = st.selectbox("Option Type", ["call", "put"])
    st.markdown("---")
    st.markdown("**Binomial Tree**")
    n_steps = st.slider("Steps", 10, 3000, 500, step=10)
    style = st.selectbox("Exercise Style", ["european", "american"])
    st.markdown("---")
    st.markdown("**Monte Carlo**")
    num_sim = st.slider("Simulations", 1_000, 200_000, 50_000, step=1_000, help="Higher → more accurate, slower")
    num_steps = st.slider("Time Steps", 10, 2_000, 100, step=10)
    seed = st.number_input("Seed", 0, 10_000, 42)

# Compute (timed)
rows = []

# Black–Scholes
try:
    bs_price_val, t_bs = timeit_ms(price_black_scholes, S, K, T, r, sigma, option_type, q)
    rows.append(("Black–Scholes", bs_price_val, t_bs))
except Exception as e:
    rows.append(("Black–Scholes (error)", np.nan, np.nan))
    st.warning(f"Black–Scholes unavailable: {e}")

# Binomial
try:
    bt_price_val, t_bt = timeit_ms(price_binomial, S, K, T, r, sigma, option_type, q, n_steps, style)
    rows.append((f"Binomial ({style}, {n_steps} steps)", bt_price_val, t_bt))
except Exception as e:
    rows.append(("Binomial (error)", np.nan, np.nan))
    st.warning(f"Binomial Tree unavailable: {e}")

# Monte Carlo
try:
    mc_price_val, t_mc = timeit_ms(price_monte_carlo, S, K, T, r, sigma, option_type, q, num_sim, num_steps, seed)
    rows.append((f"Monte Carlo ({num_sim}×{num_steps})", mc_price_val, t_mc))
except Exception as e:
    rows.append(("Monte Carlo (error)", np.nan, np.nan))
    st.warning(f"Monte Carlo unavailable: {e}")

df = pd.DataFrame(rows, columns=["Model", "Price", "Latency (ms)"])
st.subheader("Results")
st.dataframe(df, use_container_width=True)

# MC Greeks
st.subheader("Greeks (Monte Carlo • central differences)")
try:
    delta, gamma = greeks_mc_delta_gamma(S, K, T, r, sigma, option_type, q, num_sim, num_steps, seed, h=1e-3)
    c1, c2 = st.columns(2)
    c1.metric("Delta", f"{delta:.6f}")
    c2.metric("Gamma", f"{gamma:.6e}")
except Exception as e:
    st.info(f"MC Greeks unavailable: {e}")
