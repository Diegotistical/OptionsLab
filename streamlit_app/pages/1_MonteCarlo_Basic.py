# streamlit_app/pages/1_MonteCarlo_Basic.py
import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd

from streamlit_app.st_utils import (
    price_monte_carlo,
    greeks_mc_delta_gamma,
    timeit_ms,
    show_repo_status
)

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

def simulate_payoffs(S, K, T, r, sigma, option_type, num_sim, num_steps, seed):
    """
    Vectorized black-scholes log-Euler terminal price simulator with antithetic-like paths.
    Returns discounted payoffs array length = num_sim
    """
    # Use reproducible numpy RNG for fallback simulation
    rng = np.random.default_rng(int(seed) if seed is not None else None)
    dt = float(T) / int(num_steps)
    # generate normal variates (num_sim x num_steps)
    Z = rng.standard_normal((int(num_sim), int(num_steps)))
    S_paths = np.empty_like(Z)
    S_paths[:, 0] = S
    # step forward (we use index 0..num_steps-1; terminal uses last column)
    for t in range(1, int(num_steps)):
        S_paths[:, t] = S_paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t])
    if option_type == "call":
        payoff = np.maximum(S_paths[:, -1] - K, 0.0)
    else:
        payoff = np.maximum(K - S_paths[:, -1], 0.0)
    return np.exp(-r * T) * payoff

# Safe wrapper that tries st_utils.price_monte_carlo first, falls back if any error / None
def safe_price(S, K, T, r, sigma, option_type, q=0.0, num_sim=50_000, num_steps=100, seed=42, use_numba=False):
    # try the imported pricer first (if present) but catch any exception
    try:
        if callable(price_monte_carlo):
            p = price_monte_carlo(
                float(S), float(K), float(T), float(r), float(sigma),
                option_type, float(q),
                num_sim=int(num_sim), num_steps=int(num_steps), seed=int(seed) if seed is not None else None,
                use_numba=bool(use_numba)
            )
            # if the imported pricer returns an array of payoffs, reduce to scalar
            if isinstance(p, (list, tuple, np.ndarray)):
                return float(np.mean(p))
            if p is None:
                raise RuntimeError("Imported price function returned None")
            return float(p)
    except Exception as exc:  # fallback silently to internal simulator, but show a warning
        st.warning(f"Imported Monte Carlo pricer unavailable or failed: {exc}. Using internal fallback simulator.")

    # fallback
    payoffs = simulate_payoffs(S, K, T, r, sigma, option_type, num_sim=int(num_sim), num_steps=int(num_steps), seed=seed)
    return float(np.mean(payoffs))

# Safe wrapper for Greeks: try imported greeks, else fallback finite differences using safe_price
def safe_greeks(S, K, T, r, sigma, option_type, q=0.0, num_sim=50_000, num_steps=100, seed=42, h=1e-3, use_numba=False):
    try:
        if callable(greeks_mc_delta_gamma):
            d, g = greeks_mc_delta_gamma(
                float(S), float(K), float(T), float(r), float(sigma),
                option_type, float(q),
                num_sim=int(num_sim), num_steps=int(num_steps), seed=int(seed) if seed is not None else None,
                h=float(h), use_numba=bool(use_numba)
            )
            # If a vector/array was returned accidentally, reduce to scalars
            if isinstance(d, (list, tuple, np.ndarray)):
                d = float(np.mean(d))
            if isinstance(g, (list, tuple, np.ndarray)):
                g = float(np.mean(g))
            if d is None or g is None:
                raise RuntimeError("Imported greeks returned None")
            return float(d), float(g)
    except Exception as exc:
        st.warning(f"Imported greeks unavailable or failed: {exc}. Using finite-difference fallback.")

    # fallback finite-difference using safe_price to guarantee consistent fallback behavior
    h = float(h)
    p_up = safe_price(S + h, K, T, r, sigma, option_type, q=q, num_sim=num_sim, num_steps=num_steps, seed=seed, use_numba=use_numba)
    p_mid = safe_price(S, K, T, r, sigma, option_type, q=q, num_sim=num_sim, num_steps=num_steps, seed=seed, use_numba=use_numba)
    p_down = safe_price(S - h, K, T, r, sigma, option_type, q=q, num_sim=num_sim, num_steps=num_steps, seed=seed, use_numba=use_numba)
    delta = (p_up - p_down) / (2.0 * h)
    gamma = (p_up - 2.0 * p_mid + p_down) / (h ** 2)
    return float(delta), float(gamma)

if run:
    # ---------- Monte Carlo Pricing ----------
    with st.spinner("Running Monte Carlo pricing..."):
        price, t_price_ms = timeit_ms(
            safe_price,
            S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type, q=q,
            num_sim=num_sim, num_steps=num_steps, seed=seed, use_numba=use_numba
        )
        # Guarantee scalar
        if isinstance(price, (list, tuple, np.ndarray)):
            price = float(np.mean(price))
        if price is None:
            price = float("nan")

        # ---------- Greeks ----------
        delta, gamma = safe_greeks(
            S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type, q=q,
            num_sim=num_sim, num_steps=num_steps, seed=seed, h=1e-3, use_numba=use_numba
        )

    # ---------- Display Metrics ----------
    col1, col2, col3 = st.columns(3)
    # use try/except formatting to avoid crash if something weird happens
    try:
        col1.metric("Option Price", f"{float(price):.4f}")
    except Exception:
        col1.metric("Option Price", "N/A")
    try:
        col2.metric("Delta", f"{float(delta):.4f}")
    except Exception:
        col2.metric("Delta", "N/A")
    try:
        col3.metric("Gamma", f"{float(gamma):.4f}")
    except Exception:
        col3.metric("Gamma", "N/A")

    st.caption(f"Pricing computed in {t_price_ms:.2f} ms using Monte Carlo with {int(num_sim):,} paths and {int(num_steps)} steps.")

    # ---------- Confidence Interval ----------
    # use the fallback simulator (vectorized) to compute the distribution for CI/histogram
    discounted = simulate_payoffs(S, K, T, r, sigma, option_type, num_sim=int(num_sim), num_steps=int(num_steps), seed=seed)
    mean_price = float(np.mean(discounted))
    std_error = float(np.std(discounted, ddof=0) / np.sqrt(int(num_sim)))
    ci_lower = mean_price - 1.96 * std_error
    ci_upper = mean_price + 1.96 * std_error
    st.markdown(f"**95% Confidence Interval:** {ci_lower:.4f} - {ci_upper:.4f}")

    # ---------- Histogram ----------
    st.markdown("#### Payoff Distribution")
    st.bar_chart(pd.DataFrame({"Discounted Payoff": discounted}), y="Discounted Payoff", use_container_width=True)

else:
    st.info("Set parameters in the sidebar and click **Run Monte Carlo Pricing** to see results.")
