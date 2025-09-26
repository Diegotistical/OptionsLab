"""
OptionsLab – Streamlit Frontend
Entry point that sets global page config, loads shared sidebar, and
delegates to Streamlit's multi-page app (files in streamlit_app/pages).
"""
import sys
from pathlib import Path

# Make sure 'src' is importable as a package (useful on Streamlit Cloud)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

import streamlit as st
from st_utils import show_repo_status, load_readme

# Set page config with dark theme
st.set_page_config(
    page_title="Options Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode and full width
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 100%;
    }
    .sidebar .sidebar-content {
        width: 280px;
    }
    .css-1d391kg {
        padding-top: 0rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #1a1a1a;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# Main content
st.title("OptionsLab Dashboard")
st.caption("Pricing • Greeks • Risk • Volatility Surface • Benchmarks")

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Models Available", "3", "+1")
with col2:
    st.metric("Greeks Supported", "5", "Δ, Γ, Θ, ν, ρ")
with col3:
    st.metric("Risk Metrics", "4", "VaR, ES, Stress, Sens")
with col4:
    st.metric("ML Models", "3", "SVR, MLP, XGB")

st.markdown("---")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("🚀 What's Inside?")
    st.markdown("""
    - **Option Pricing**: Black–Scholes (closed form), CRR **Binomial Tree** (Numba), and **Monte Carlo** (CPU / optional GPU)
    - **Greeks**: Finite-difference Greeks + model-specific methods
    - **Risk**: VaR, Expected Shortfall, Sensitivity, Stress Testing
    - **Volatility Surface**: Feature engineering, ML models (SVR/MLP/XGBoost), interpolation, arbitrage checks
    - **Benchmarks**: Latency comparisons and micro-profiling
    """)

with col2:
    st.subheader("🔗 Quick Links")
    st.markdown("""
    - Navigate pages using the sidebar menu
    - Configure defaults in `src/common/config.py`
    - Add models under `src/volatility_surface/models/`
    - Check `src/pricing_models/` for pricing engines
    - Review `src/risk/` for risk management tools
    """)

st.markdown("---")
st.subheader("📖 README")
st.markdown(load_readme(max_lines=40))