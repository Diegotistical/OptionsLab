# streamlit_app/app.py
"""
OptionsLab – Premium Streamlit Dashboard.

Main entry point with modern, sleek design.
"""

import sys
from pathlib import Path

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Import components
try:
    from components import apply_custom_css, page_header, section_divider
except ImportError:
    from streamlit_app.components import apply_custom_css, page_header, section_divider

# Apply theme
apply_custom_css()

# =============================================================================
# MAIN CONTENT
# =============================================================================

page_header("Dashboard", "Advanced Options Pricing, Greeks & Risk Analytics")

# Overview metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        """
    <div class="metric-card">
        <div class="metric-label">Pricing Models</div>
        <div class="metric-value">4</div>
        <div class="metric-delta">BS · MC · ML · Binomial</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
    <div class="metric-card">
        <div class="metric-label">Greeks</div>
        <div class="metric-value">5</div>
        <div class="metric-delta">Δ · Γ · Θ · ν · ρ</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
    <div class="metric-card">
        <div class="metric-label">Risk Metrics</div>
        <div class="metric-value">4</div>
        <div class="metric-delta">VaR · ES · Stress · Sens</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
    <div class="metric-card">
        <div class="metric-label">Acceleration</div>
        <div class="metric-value">✓</div>
        <div class="metric-delta">Numba · GPU</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        """
    <div class="metric-card">
        <div class="metric-label">ML Models</div>
        <div class="metric-value">3</div>
        <div class="metric-delta">LightGBM · XGB · MLP</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

section_divider()

# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
    <div class="metric-card" style="height: 220px;">
        <h3 style="color: #60a5fa; margin-bottom: 1rem;">⚡ Monte Carlo Pricing</h3>
        <p style="color: #cbd5e1; line-height: 1.6;">
            High-performance Monte Carlo simulation with Numba JIT compilation
            and optional GPU acceleration via CuPy. Vectorized batch processing
            for 100x speedup.
        </p>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 1rem;">
            → 100K sims in milliseconds
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
    <div class="metric-card" style="height: 220px;">
        <h3 style="color: #a78bfa; margin-bottom: 1rem;">🧠 ML Surrogate Models</h3>
        <p style="color: #cbd5e1; line-height: 1.6;">
            Train LightGBM surrogate models for instant option pricing.
            Vectorized Black-Scholes training generates 10K samples in &lt;1 second.
        </p>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 1rem;">
            → Inference in microseconds
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
    <div class="metric-card" style="height: 220px;">
        <h3 style="color: #34d399; margin-bottom: 1rem;">📈 Risk Analytics</h3>
        <p style="color: #cbd5e1; line-height: 1.6;">
            Comprehensive risk metrics including Value at Risk, Expected Shortfall,
            stress testing, and sensitivity analysis with interactive visualizations.
        </p>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 1rem;">
            → VaR, ES, Greeks surfaces
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

section_divider()

# Quick navigation
st.markdown(
    """
<h3 style="color: #f8fafc; margin-bottom: 1rem;">🚀 Quick Start</h3>
<p style="color: #94a3b8; margin-bottom: 1.5rem;">
    Select a page from the sidebar menu to begin. Each page provides interactive
    pricing and analysis tools with full-width visualizations.
</p>
""",
    unsafe_allow_html=True,
)

# Page links - Row 1
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Monte Carlo Basic", use_container_width=True):
        st.switch_page("pages/1_MonteCarlo_Basic.py")

with col2:
    if st.button("Monte Carlo ML", use_container_width=True):
        st.switch_page("pages/2_MonteCarlo_ML.py")

with col3:
    if st.button("Monte Carlo Unified", use_container_width=True):
        st.switch_page("pages/3_MonteCarlo_Unified.py")

with col4:
    if st.button("Binomial Tree", use_container_width=True):
        st.switch_page("pages/4_Binomial_Tree.py")

# Page links - Row 2 (New pages)
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Advanced Models", use_container_width=True):
        st.switch_page("pages/6_Advanced_Models.py")

with col2:
    if st.button("Exotic Options", use_container_width=True):
        st.switch_page("pages/7_Exotic_Options.py")

with col3:
    if st.button("Portfolio Greeks", use_container_width=True):
        st.switch_page("pages/8_Portfolio_Greeks.py")

with col4:
    if st.button("Risk Analysis", use_container_width=True):
        st.switch_page("pages/12_Risk_Analysis.py")

# Page links - Row 3 (Advanced features)
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Live Market", use_container_width=True):
        st.switch_page("pages/9_Live_Market.py")

with col2:
    if st.button("Backtest", use_container_width=True):
        st.switch_page("pages/10_Backtest.py")

with col3:
    if st.button("Volatility Surface", use_container_width=True):
        st.switch_page("pages/13_Volatility_Surface.py")

with col4:
    if st.button("Benchmarks", use_container_width=True):
        st.switch_page("pages/11_Benchmarks.py")

section_divider()

# System info
with st.expander("📊 System Information", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Available Modules:**")

        try:
            from src.pricing_models import (
                GPU_AVAILABLE,
                LIGHTGBM_AVAILABLE,
                NUMBA_AVAILABLE,
            )

            st.write(
                f"- Numba JIT: {'✅ Available' if NUMBA_AVAILABLE else '❌ Not installed'}"
            )
            st.write(
                f"- GPU (CuPy): {'✅ Available' if GPU_AVAILABLE else '❌ Not installed'}"
            )
            st.write(
                f"- LightGBM: {'✅ Available' if LIGHTGBM_AVAILABLE else '❌ Using sklearn'}"
            )
        except ImportError:
            st.write("Could not check module availability")

    with col2:
        st.markdown("**Keyboard Shortcuts:**")
        st.write("- `r` - Rerun page")
        st.write("- `c` - Clear cache")
        st.write("- `m` - Toggle sidebar")
