import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import logging

from streamlit_app.st_utils import (
    price_monte_carlo,
    greeks_mc_delta_gamma,
    timeit_ms,
    show_repo_status
)

# ------------------- LOGGING -------------------
logger = logging.getLogger("monte_carlo")

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Monte Carlo Option Pricing",
    layout="wide"
)

# ------------------- HEADER -------------------
st.title("📈 Monte Carlo Option Pricing Dashboard")
st.markdown("""
Price **European call/put options**, compute Greeks, and visualize Monte Carlo simulations.  
Use the sidebar to set parameters, then click **Run Monte Carlo Pricing**.
""")

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.header("📊 Model Parameters")

    with st.expander("Market Inputs", expanded=True):
        S = st.number_input("Spot Price (S)", min_value=0.0, value=100.0, step=1.0)
        K = st.number_input("Strike Price (K)", min_value=0.0, value=100.0, step=1.0)
        T = st.number_input("Time to Maturity (T, years)", min_value=0.01, value=1.0, step=0.01)
        r = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01, format="%.4f")
        sigma = st.number_input("Volatility (σ)", value=0.2, step=0.01, format="%.4f")
        q = st.number_input("Dividend Yield (q)", value=0.0, step=0.01, format="%.4f")
        option_type = st.selectbox("Option Type", ["call", "put"])

    with st.expander("Simulation Inputs", expanded=True):
        num_sim = st.number_input("Number of Simulations", min_value=1000, value=50_000, step=1000)
        num_steps = st.number_input("Steps per Path", min_value=1, value=100, step=1)
        seed = st.number_input("Random Seed", value=42, step=1)
        use_numba = st.checkbox("Use Numba Acceleration", value=False)

    st.markdown("---")
    show_repo_status()
    run = st.button("🚀 Run Monte Carlo Pricing")

# ------------------- HELPER FUNCTION -------------------
def simulate_payoffs(S, K, T, r, sigma, option_type, num_sim, num_steps, seed, q=0.0):
    np.random.seed(seed)
    dt = T / num_steps
    Z = np.random.standard_normal((num_sim, num_steps))
    S_paths = np.zeros_like(Z)
    S_paths[:, 0] = S
    for t in range(1, num_steps):
        S_paths[:, t] = S_paths[:, t-1] * np.exp(
            (r - q - 0.5 * sigma**2) * dt +
            sigma * np.sqrt(dt) * Z[:, t]
        )
    payoff = np.maximum(S_paths[:, -1] - K, 0) if option_type == "call" else np.maximum(K - S_paths[:, -1], 0)
    discounted = np.exp(-r*T) * payoff
    return discounted, S_paths

# ------------------- MAIN CONTENT -------------------
st.markdown("### Results & Visualizations")

if run:
    try:
        # ---------- Monte Carlo Pricing ----------
        price, t_price_ms = timeit_ms(
            price_monte_carlo,
            S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type, q=q,
            num_sim=num_sim, num_steps=num_steps, seed=seed, use_numba=use_numba
        )
        price = float(np.mean(price)) if price is not None and not np.isscalar(price) else price

        # ---------- Greeks ----------
        delta, gamma = greeks_mc_delta_gamma(
            S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type, q=q,
            num_sim=num_sim, num_steps=num_steps, seed=seed, h=1e-3, use_numba=use_numba
        )
        delta = float(np.mean(delta)) if delta is not None else None
        gamma = float(np.mean(gamma)) if gamma is not None else None

        # ---------- Metrics ----------
        col1, col2, col3 = st.columns(3)
        col1.metric("Option Price", f"{price:.4f}" if price else "N/A")
        col2.metric("Delta", f"{delta:.4f}" if delta else "N/A")
        col3.metric("Gamma", f"{gamma:.4f}" if gamma else "N/A")
        st.caption(f"Computed in {t_price_ms:.2f} ms using {num_sim:,} paths and {num_steps} steps.")

        # ---------- Payoff & Paths ----------
        discounted, S_paths = simulate_payoffs(S, K, T, r, sigma, option_type, num_sim, num_steps, seed, q=q)
        mean_price = np.mean(discounted)
        std_error = np.std(discounted) / np.sqrt(num_sim)
        ci_lower = mean_price - 1.96 * std_error
        ci_upper = mean_price + 1.96 * std_error
        st.markdown(f"**95% Confidence Interval:** {ci_lower:.4f} - {ci_upper:.4f}")

        # ---------- Payoff Distribution ----------
        st.markdown("#### 1️⃣ Payoff Distribution")
        st.bar_chart(pd.DataFrame({"Discounted Payoff": discounted}), y="Discounted Payoff", use_container_width=True)

        # ---------- Sample Spot Paths ----------
        st.markdown("#### 2️⃣ Sample Spot Price Paths")
        n_plot = min(100, num_sim)
        st.line_chart(pd.DataFrame(S_paths[:n_plot, :].T), use_container_width=True)

        # ---------- Confidence Interval Band ----------
        st.markdown("#### 3️⃣ Confidence Interval Band over Paths")
        cum_mean = np.cumsum(discounted) / np.arange(1, num_sim+1)
        ci_upper_band = cum_mean + 1.96 * np.std(discounted[:len(cum_mean)]) / np.sqrt(np.arange(1, num_sim+1))
        ci_lower_band = cum_mean - 1.96 * np.std(discounted[:len(cum_mean)]) / np.sqrt(np.arange(1, num_sim+1))
        df_ci = pd.DataFrame({
            "Running Mean": cum_mean,
            "CI Upper": ci_upper_band,
            "CI Lower": ci_lower_band
        })
        st.line_chart(df_ci, use_container_width=True)

        # ---------- Convergence Plot ----------
        st.markdown("#### 4️⃣ Convergence of Price vs Number of Simulations")
        step = max(1, num_sim // 100)
        running_mean = np.cumsum(discounted[:step*100:step]) / np.arange(1, len(discounted[:step*100:step])+1)
        st.line_chart(pd.DataFrame({"Running Mean Price": running_mean}), use_container_width=True)

        # ---------- Spot vs Terminal Price ----------
        st.markdown("#### 5️⃣ Spot vs Terminal Price Scatter")
        st.scatter_chart(pd.DataFrame({"Spot": np.repeat(S, num_sim), "Terminal": S_paths[:, -1]}))

        # ---------- Delta & Gamma Sensitivity ----------
        st.markdown("#### 6️⃣ Delta & Gamma Sensitivity")
        spot_grid = np.linspace(0.8*S, 1.2*S, 50)
        delta_grid, gamma_grid = [], []
        for s_val in spot_grid:
            d, g = greeks_mc_delta_gamma(
                S=s_val, K=K, T=T, r=r, sigma=sigma, option_type=option_type,
                num_sim=num_sim//10, num_steps=num_steps, seed=seed, h=1e-3, use_numba=use_numba
            )
            delta_grid.append(float(d))
            gamma_grid.append(float(g))
        df_greeks = pd.DataFrame({"Spot": spot_grid, "Delta": delta_grid, "Gamma": gamma_grid})
        st.line_chart(df_greeks.set_index("Spot"), use_container_width=True)

    except Exception as e:
        st.error(f"Critical error during pricing: {str(e)}")
        logger.exception("Critical pricing failure")

else:
    st.info("Set parameters in the sidebar and click **Run Monte Carlo Pricing** to see results.")
