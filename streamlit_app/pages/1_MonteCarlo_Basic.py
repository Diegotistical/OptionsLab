# 1_MonteCarlo_Basic.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from streamlit_app.st_utils import price_monte_carlo, timeit_ms


st.set_page_config(page_title="MC – Basic", page_icon="⛳", layout="wide")
st.title("Monte Carlo Pricer — Basic")

with st.sidebar:
    st.markdown("### Simulation Settings")
    num_sim = st.slider("Simulations", 5_000, 200_000, 50_000, step=5_000)
    num_steps = st.slider("Time Steps", 10, 500, 100, step=10)
    seed = st.number_input("Seed", value=42)
    use_numba = st.toggle("Use Numba (if available)", value=False)

col1, col2 = st.columns([1,1], gap="large")

with col1:
    st.markdown("### Option Inputs")
    S = st.number_input("Spot (S)", 1.0, 1_000.0, 100.0)
    K = st.number_input("Strike (K)", 1.0, 1_000.0, 100.0)
    T = st.number_input("Maturity (T, years)", 0.01, 5.0, 1.0)
    r = st.number_input("Risk-free (r)", 0.0, 0.25, 0.05)
    sigma = st.number_input("Volatility (σ)", 0.001, 2.0, 0.2)
    q = st.number_input("Dividend (q)", 0.0, 0.2, 0.0)
    option_type = st.selectbox("Option Type", ["call", "put"])

run = st.button("Run Pricing")

with col2:
    st.markdown("### Results")

if run:
    pricer = price_monte_carlo(num_sim, num_steps, seed, use_numba=use_numba)

    # Price + Greeks (Δ, Γ via finite differences)
    (price, t_price_ms) = timeit_ms(pricer.price, S, K, T, r, sigma, option_type, q)
    # Greeks using same MonteCarloPricer instance
    def greeks_delta_gamma():
        h = max(S * 1e-4, 1e-4)
        p_dn = pricer.price(S - h, K, T, r, sigma, option_type, q)
        p_mid = price
        p_up = pricer.price(S + h, K, T, r, sigma, option_type, q)
        delta = (p_up - p_dn) / (2*h)
        gamma = (p_up - 2*p_mid + p_dn) / (h*h)
        return delta, gamma

    (dg, t_greeks_ms) = timeit_ms(greeks_delta_gamma)
    delta, gamma = dg

    st.success(f"**Price** = {price:.6f}")
    st.write(f"Δ = {delta:.6f} | Γ = {gamma:.6f}")
    st.caption(f"Timing — price: {t_price_ms:.2f} ms | Δ,Γ: {t_greeks_ms:.2f} ms")

    # Terminal distribution plot
    with st.expander("Show terminal price distribution"):
        # Access private method intentionally for demo
        terminal = pricer._simulate_terminal_prices(S, T, r, sigma, q)  # noqa: SLF001
        fig, ax = plt.subplots()
        ax.hist(terminal, bins=60, alpha=0.65, edgecolor="black")
        ax.axvline(K, color="red", linestyle="--", label="Strike")
        ax.set_title("Terminal Price Distribution")
        ax.set_xlabel("Terminal Price")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig, clear_figure=True)
