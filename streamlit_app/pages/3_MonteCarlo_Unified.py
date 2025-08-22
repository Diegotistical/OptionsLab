import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from utils import get_mc_unified_pricer, timeit_ms

st.set_page_config(page_title="MC – Unified (CPU/GPU)", page_icon="🚀", layout="wide")
st.title("Monte Carlo — Unified (CPU/GPU + Antithetic)")

with st.sidebar:
    st.markdown("### Engine")
    num_sim = st.slider("Simulations", 10_000, 200_000, 50_000, step=10_000)
    num_steps = st.slider("Time Steps", 10, 500, 100, step=10)
    seed = st.number_input("Seed", value=42)
    use_numba = st.toggle("Use Numba", value=True)
    use_gpu = st.toggle("Use GPU (CuPy)", value=False)

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

with col2:
    st.markdown("### Surface Grid (S,T)")
    s_low, s_high = st.slider("S range", 50, 200, (80, 120))
    t_low, t_high = st.slider("T range", 0.05, 2.0, (0.1, 1.5))
    nS = st.slider("#S points", 5, 40, 25)
    nT = st.slider("#T points", 5, 40, 25)

run = st.button("Run Unified MC + Surface")

if run:
    mc = get_mc_unified_pricer(num_sim, num_steps, seed, use_numba=use_numba, use_gpu=use_gpu)

    # Single price timing
    (price, t_ms) = timeit_ms(mc.price, S, K, T, r, sigma, option_type, q)
    st.success(f"Price = {price:.6f}  |  time: {t_ms:.2f} ms  |  N={num_sim:,}, steps={num_steps}, Numba={use_numba}, GPU={use_gpu}")

    # Surface (S,T) at fixed K
    Sg = np.linspace(s_low, s_high, nS)
    Tg = np.linspace(t_low, t_high, nT)
    Sm, Tm = np.meshgrid(Sg, Tg)
    Z = np.zeros_like(Sm)

    # Compute prices grid
    for i in range(nT):
        for j in range(nS):
            Z[i, j] = mc.price(Sm[i, j], K, Tm[i, j], r, sigma, option_type, q)

    # 3D surface
    with st.expander("Show Price Surface (S,T)", expanded=True):
        fig = plt.figure(figsize=(9,5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(Sm, Tm, Z, cmap="viridis", edgecolor="k", alpha=0.85)
        ax.set_xlabel("S")
        ax.set_ylabel("T")
        ax.set_zlabel("Price")
        ax.set_title(f"Unified MC Price Surface (K={K})")
        st.pyplot(fig, clear_figure=True)
