"""
OptionsLab – Streamlit Frontend
Entry point that sets global page config, loads shared sidebar, and
delegates to Streamlit's multi-page app (files in streamlit_app/pages).
"""

import streamlit as st
from utils import show_repo_status, load_readme

st.set_page_config(
    page_title="Options Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Options Lab")
st.caption("Pricing • Greeks • Risk • Volatility Surface • Benchmarks")

with st.sidebar:
    st.markdown("### 📦 Repo Snapshot")
    show_repo_status()
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("Use the left “Pages” to navigate: Pricing, Risk Analysis, Vol Surface, Benchmarks.")
    st.markdown("—")

# Lightweight landing content
col1, col2 = st.columns([1,1])
with col1:
    st.subheader("What’s inside?")
    st.markdown(
        """
- **Option Pricing**: Black–Scholes (closed form), CRR **Binomial Tree** (Numba), and **Monte Carlo** (CPU / optional GPU).
- **Greeks**: Finite-difference Greeks + model-specific methods.
- **Risk**: VaR, Expected Shortfall, Sensitivity, Stress Testing.
- **Volatility Surface**: Feature engineering, ML models (SVR/MLP/XGBoost), interpolation, arbitrage checks.
- **Benchmarks**: Latency comparisons and micro-profiling.
        """
    )
with col2:
    st.subheader("Quick Links")
    st.write("- Navigate pages from the sidebar.")
    st.write("- Configure defaults in `src/common/config.py` if present.")
    st.write("- Add more models under `src/volatility_surface/models/` to auto-display.")

st.markdown("---")
st.subheader("README (excerpt)")
st.markdown(load_readme(max_lines=40))

