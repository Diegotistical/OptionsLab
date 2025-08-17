"""
Benchmarks Dashboard
Quick micro-benchmarks across models for a single parameter set.
"""

import streamlit as st
import pandas as pd
from utils import (
    timeit_ms,
    price_black_scholes, price_binomial, price_monte_carlo
)

st.set_page_config(page_title="Benchmarks", page_icon="⏱️", layout="wide")
st.title("⏱️ Benchmarks")

with st.sidebar:
    S = st.number_input("Spot S", 1.0, 1e6, 100.0, step=1.0)
    K = st.number_input("Strike K", 1.0, 1e6, 100.0, step=1.0)
    T = st.number_input("Time to Maturity (years)", 1e-4, 50.0, 1.0, step=0.05, format="%.4f")
    r = st.number_input("Risk-free rate r", -0.50, 1.00, 0.05, step=0.005, format="%.4f")
    sigma = st.number_input("Volatility σ", 0.0, 5.0, 0.20, step=0.01, format="%.4f")
    q = st.number_input("Dividend yield q", 0.0, 1.0, 0.0, step=0.005, format="%.4f")
    option_type = st.selectbox("Option Type", ["call", "put"])
    st.markdown("---")
    n_steps = st.slider("Binomial Steps", 10, 3000, 500, step=10)
    style = st.selectbox("Exercise Style", ["european", "american"])
    num_sim = st.slider("MC Simulations", 1_000, 200_000, 50_000, step=1000)
    num_steps = st.slider("MC Time Steps", 10, 2000, 100, step=10)
    seed = st.number_input("Seed", 0, 10_000, 42)

rows = []

# Black–Scholes
try:
    p, t = timeit_ms(price_black_scholes, S, K, T, r, sigma, option_type, q)
    rows.append(("Black–Scholes", p, t))
except Exception as e:
    rows.append(("Black–Scholes (error)", None, None))
    st.warning(f"Black–Scholes import failed: {e}")

# Binomial
try:
    p, t = timeit_ms(price_binomial, S, K, T, r, sigma, option_type, q, n_steps, style)
    rows.append((f"Binomial ({style}, {n_steps})", p, t))
except Exception as e:
    rows.append(("Binomial (error)", None, None))
    st.warning(f"Binomial Tree import failed: {e}")

# Monte Carlo
try:
    p, t = timeit_ms(price_monte_carlo, S, K, T, r, sigma, option_type, q, num_sim, num_steps, seed)
    rows.append((f"Monte Carlo ({num_sim}×{num_steps})", p, t))
except Exception as e:
    rows.append(("Monte Carlo (error)", None, None))
    st.warning(f"Monte Carlo import failed: {e}")

df = pd.DataFrame(rows, columns=["Model", "Price", "Latency (ms)"])
st.subheader("Latency Comparison")
st.dataframe(df, use_container_width=True)
