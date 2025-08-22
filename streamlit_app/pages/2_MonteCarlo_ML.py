import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import get_mc_pricer, get_mc_ml_surrogate, timeit_ms

st.set_page_config(page_title="MC – ML Surrogate", page_icon="🤖", layout="wide")
st.title("Monte Carlo — ML Surrogate (Gradient Boosting)")

with st.sidebar:
    st.markdown("### Simulation Settings")
    num_sim = st.slider("Simulations (MC target gen)", 10_000, 100_000, 30_000, step=5_000)
    num_steps = st.slider("Time Steps", 10, 250, 100, step=10)
    seed = st.number_input("Seed", value=42)

    st.markdown("---")
    st.markdown("### Training Grid")
    n_grid = st.slider("Training points per axis", 5, 25, 10)
    s_low, s_high = st.slider("S range", 50, 200, (80, 120))
    k_low, k_high = st.slider("K range", 50, 200, (80, 120))
    t_fixed = st.slider("T (fixed)", 0.05, 2.0, 1.0)
    r_fixed = st.slider("r (fixed)", 0.0, 0.15, 0.05)
    sigma_fixed = st.slider("σ (fixed)", 0.05, 0.8, 0.20)
    q_fixed = st.slider("q (fixed)", 0.0, 0.10, 0.0)

st.markdown("### Predict Inputs")
col1, col2 = st.columns([1,1])
with col1:
    S = st.number_input("Spot (S)", 1.0, 1_000.0, 100.0)
    K = st.number_input("Strike (K)", 1.0, 1_000.0, 100.0)
    T = st.number_input("Maturity (T, years)", 0.01, 5.0, 1.0)
with col2:
    r = st.number_input("Risk-free (r)", 0.0, 0.25, 0.05)
    sigma = st.number_input("Volatility (σ)", 0.001, 2.0, 0.2)
    q = st.number_input("Dividend (q)", 0.0, 0.2, 0.0)

option_type = st.selectbox("Option Type (targets are for calls in this demo)", ["call", "put"])
train = st.button("Fit Surrogate & Compare")

if train:
    # Build training dataframe on grid
    grid_S = np.linspace(s_low, s_high, n_grid)
    grid_K = np.linspace(k_low, k_high, n_grid)
    Sg, Kg = np.meshgrid(grid_S, grid_K)
    df = pd.DataFrame({
        "S": Sg.ravel(),
        "K": Kg.ravel(),
        "T": np.full(Sg.size, t_fixed),
        "r": np.full(Sg.size, r_fixed),
        "sigma": np.full(Sg.size, sigma_fixed),
        "q": np.full(Sg.size, q_fixed)
    })

    mc = get_mc_pricer(num_sim, num_steps, seed)
    ml = get_mc_ml_surrogate(num_sim, num_steps, seed)

    # Fit (generates MC targets internally if y=None)
    (_, t_fit_ms) = timeit_ms(ml.fit, df, None)

    # Predict single point with MC vs ML
    x_single = pd.DataFrame([{"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "q": q}])

    (price_mc, t_mc_ms) = timeit_ms(mc.price, S, K, T, r, sigma, option_type, q)
    (pred_df, t_ml_ms) = timeit_ms(ml.predict, x_single)

    st.success(f"Surrogate trained on {len(df):,} points | Fit time: {t_fit_ms:.0f} ms")
    st.write(f"**MC** price: {price_mc:.6f} ({t_mc_ms:.1f} ms)  |  **ML** price≈Δ≈Γ: {pred_df.iloc[0].to_dict()} ({t_ml_ms:.3f} ms)")

    # Error heatmap across grid (price)
    with st.expander("Show error heatmap across training grid (price)"):
        # Compare MC vs ML on grid for price only (calls)
        prices_mc = []
        for _, row in df.iterrows():
            prices_mc.append(mc.price(row.S, row.K, row.T, row.r, row.sigma, "call", row.q))
        prices_mc = np.array(prices_mc)

        preds = ml.predict(df)[["price"]].values.ravel()
        err = (preds - prices_mc).reshape(Sg.shape)

        fig, ax = plt.subplots()
        im = ax.imshow(err, origin="lower", extent=[s_low, s_high, k_low, k_high], aspect="auto")
        ax.set_title("ML − MC Price Error (call)")
        ax.set_xlabel("S")
        ax.set_ylabel("K")
        fig.colorbar(im, ax=ax, label="Error")
        st.pyplot(fig, clear_figure=True)
