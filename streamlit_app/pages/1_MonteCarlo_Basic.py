# ------------------- Monte Carlo Dashboard -------------------
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from streamlit_app.st_utils import (
    price_monte_carlo,
    greeks_mc_delta_gamma,
    timeit_ms,
    simulate_payoffs
)

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Monte Carlo Option Pricing (Classic)",
    layout="wide"
)

# ------------------- HEADER -------------------
st.title("Monte Carlo Option Pricing Dashboard (Classic)")
st.markdown(
    "Centralized inputs, fast pricing, Greeks, and interactive visualizations."
)

# ------------------- CENTRAL INPUTS -------------------
st.markdown("<h2 style='text-align:center'>Monte Carlo Parameters</h2>", unsafe_allow_html=True)
with st.container():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        S = st.number_input("Spot Price (S)", value=100.0)
        K = st.number_input("Strike Price (K)", value=100.0)
        T = st.number_input("Time to Maturity (T, years)", value=1.0)
        r = st.number_input("Risk-Free Rate (r)", value=0.05, format="%.4f")
        sigma = st.number_input("Volatility (σ)", value=0.2, format="%.4f")
        q = st.number_input("Dividend Yield (q)", value=0.0, format="%.4f")
        option_type = st.selectbox("Option Type", ["Call","Put"])
        num_sim = st.number_input("Number of Simulations", value=50_000, step=5000)
        num_steps = st.number_input("Steps per Path", value=100)
        seed = st.number_input("Random Seed", value=42)
        use_numba = st.checkbox("Use Numba Acceleration", value=False)
        run = st.button("Run Monte Carlo Pricing")

st.markdown("---")

# ------------------- MAIN CONTENT -------------------
if run:
    try:
        # ---------- Monte Carlo Pricing ----------
        price, t_price_ms = timeit_ms(
            price_monte_carlo,
            S=S, K=K, T=T, r=r, sigma=sigma,
            option_type=option_type.lower(), q=q,
            num_sim=num_sim, num_steps=num_steps,
            seed=seed, use_numba=use_numba
        )
        price = float(np.mean(price)) if price is not None else np.nan

        # ---------- Greeks ----------
        delta, gamma = greeks_mc_delta_gamma(
            S=S, K=K, T=T, r=r, sigma=sigma,
            option_type=option_type.lower(), q=q,
            num_sim=num_sim, num_steps=num_steps,
            seed=seed, h=1e-3, use_numba=use_numba
        )
        delta = float(np.mean(delta))
        gamma = float(np.mean(gamma))

        # ---------- Metrics ----------
        st.markdown("### Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Option Price", f"{price:.4f}")
        col2.metric("Delta", f"{delta:.4f}")
        col3.metric("Gamma", f"{gamma:.4f}")
        st.caption(f"Computed in {t_price_ms:.2f} ms with {num_sim:,} paths × {num_steps} steps.")

        # ---------- Simulate Full Payoffs ----------
        discounted_payoffs = simulate_payoffs(S, K, T, r, sigma, option_type.lower(),
                                              num_sim, num_steps, seed, q=q)
        mean_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs)/np.sqrt(num_sim)
        ci_lower, ci_upper = mean_price - 1.96*std_error, mean_price + 1.96*std_error

        # ---------- Confidence Interval Band ----------
        st.markdown("### 95% Confidence Interval")
        ci_fig = go.Figure()
        ci_fig.add_trace(go.Scatter(y=[ci_lower]*num_sim, mode='lines', name='Lower Bound'))
        ci_fig.add_trace(go.Scatter(y=[mean_price]*num_sim, mode='lines', name='Mean'))
        ci_fig.add_trace(go.Scatter(y=[ci_upper]*num_sim, mode='lines', name='Upper Bound'))
        st.plotly_chart(ci_fig, use_container_width=True)

        # ---------- Histogram of Payoffs ----------
        st.markdown("### Payoff Distribution")
        hist_fig = px.histogram(pd.DataFrame({"Payoff": discounted_payoffs}), x="Payoff", nbins=50)
        st.plotly_chart(hist_fig, use_container_width=True)

        # ---------- Convergence Plot ----------
        st.markdown("### Convergence of Running Average")
        running_avg = np.cumsum(discounted_payoffs) / np.arange(1, num_sim+1)
        conv_fig = px.line(pd.DataFrame({"Running Average": running_avg}), y="Running Average")
        st.plotly_chart(conv_fig, use_container_width=True)

        # ---------- Spot vs Terminal Price Scatter ----------
        st.markdown("### Spot vs Terminal Price")
        terminal_prices = discounted_payoffs * np.exp(r*T)  # undo discount
        scatter_df = pd.DataFrame({"Spot": np.full(num_sim, S), "Terminal": terminal_prices})
        scatter_fig = px.scatter(scatter_df, x="Spot", y="Terminal")
        st.plotly_chart(scatter_fig, use_container_width=True)

        # ---------- Delta/Gamma Sensitivity ----------
        st.markdown("### Delta/Gamma Sensitivity vs Spot")
        spot_grid = np.linspace(0.8*S, 1.2*S, 50)
        delta_grid, gamma_grid = [], []
        for s in spot_grid:
            d, g = greeks_mc_delta_gamma(
                S=s, K=K, T=T, r=r, sigma=sigma,
                option_type=option_type.lower(), q=q,
                num_sim=num_sim//5, num_steps=num_steps, seed=seed, h=1e-3, use_numba=use_numba
            )
            delta_grid.append(np.mean(d))
            gamma_grid.append(np.mean(g))
        sens_df = pd.DataFrame({"Spot": spot_grid, "Delta": delta_grid, "Gamma": gamma_grid})
        sens_fig = go.Figure()
        sens_fig.add_trace(go.Scatter(x=sens_df.Spot, y=sens_df.Delta, mode='lines', name='Delta'))
        sens_fig.add_trace(go.Scatter(x=sens_df.Spot, y=sens_df.Gamma, mode='lines', name='Gamma'))
        st.plotly_chart(sens_fig, use_container_width=True)

    except Exception as e:
        st.error(f"Monte Carlo computation failed: {str(e)}")

else:
    st.info("Set parameters above and click 'Run Monte Carlo Pricing' to see results.")
