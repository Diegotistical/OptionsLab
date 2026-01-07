# streamlit_app/pages/3_MonteCarlo_Unified.py
"""
Monte Carlo Unified Pricer - Streamlit Page.

High-performance MC with Numba and GPU acceleration.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Monte Carlo Unified",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

try:
    from components import (
        apply_custom_css, page_header, section_divider,
        get_chart_layout, format_price, format_greek, format_time_ms
    )
except ImportError:
    from streamlit_app.components import (
        apply_custom_css, page_header, section_divider,
        get_chart_layout, format_price, format_greek, format_time_ms
    )

try:
    from src.pricing_models import MonteCarloPricerUni, NUMBA_AVAILABLE, GPU_AVAILABLE
except ImportError:
    MonteCarloPricerUni = None
    NUMBA_AVAILABLE = False
    GPU_AVAILABLE = False

apply_custom_css()

# =============================================================================
# HEADER
# =============================================================================
page_header("Unified Monte Carlo", "High-performance pricing with Numba JIT and GPU acceleration")

# Hardware status
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    status = "✅ Available" if NUMBA_AVAILABLE else "❌ Not installed"
    st.markdown(f"**Numba JIT:** {status}")
with col2:
    status = "✅ Available" if GPU_AVAILABLE else "❌ Not installed"
    st.markdown(f"**GPU (CuPy):** {status}")

section_divider()

# =============================================================================
# INPUT SECTION
# =============================================================================
st.markdown('<div class="input-section">', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

with col1:
    st.markdown("**Asset**")
    S = st.number_input("Spot Price ($)", min_value=1.0, max_value=500.0, value=100.0, step=1.0, key="uni_spot")
    K = st.number_input("Strike Price ($)", min_value=1.0, max_value=500.0, value=100.0, step=1.0, key="uni_strike")

with col2:
    st.markdown("**Time & Type**")
    T = st.number_input("Maturity (years)", min_value=0.01, max_value=5.0, value=1.0, step=0.05, key="uni_time")
    option_type = st.selectbox("Option Type", ["call", "put"], key="uni_type")

with col3:
    st.markdown("**Market**")
    r_pct = st.number_input("Rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5, key="uni_rate")
    sigma_pct = st.number_input("Vol (%)", min_value=1.0, max_value=100.0, value=20.0, step=1.0, key="uni_vol")
    r, sigma = r_pct / 100.0, sigma_pct / 100.0

with col4:
    st.markdown("**Simulation**")
    num_sims = st.select_slider("Sims", options=[10000, 25000, 50000, 100000, 250000], value=50000, key="uni_sims")
    use_numba = st.checkbox("Use Numba", value=NUMBA_AVAILABLE, disabled=not NUMBA_AVAILABLE, key="uni_numba")

with col5:
    st.markdown("**Run**")
    st.write("")
    run_single = st.button("🎯 Single Price", type="primary", use_container_width=True)
    run_surface = st.button("🌊 Price Surface", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# SINGLE PRICING
# =============================================================================
if run_single:
    if MonteCarloPricerUni is None:
        st.error("Unified pricer not available.")
        st.stop()
    
    with st.spinner("Running simulation..."):
        pricer = MonteCarloPricerUni(
            num_simulations=num_sims,
            num_steps=100,
            use_numba=use_numba,
            use_gpu=False
        )
        
        t_start = time.perf_counter()
        price = pricer.price(S, K, T, r, sigma, option_type)
        t_price = (time.perf_counter() - t_start) * 1000
        
        t_start = time.perf_counter()
        delta, gamma = pricer.delta_gamma(S, K, T, r, sigma, option_type)
        t_greeks = (time.perf_counter() - t_start) * 1000
    
    section_divider()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Price</div>
            <div class="metric-value">{format_price(price)}</div>
            <div class="metric-delta">{format_time_ms(t_price)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Delta</div>
            <div class="metric-value">{format_greek(delta)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Gamma</div>
            <div class="metric-value">{format_greek(gamma, 6)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Simulations</div>
            <div class="metric-value">{num_sims:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        backend = "Numba" if use_numba else "NumPy"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Backend</div>
            <div class="metric-value">{backend}</div>
            <div class="metric-delta">Greeks: {format_time_ms(t_greeks)}</div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# SURFACE GENERATION
# =============================================================================
if run_surface:
    if MonteCarloPricerUni is None:
        st.error("Unified pricer not available.")
        st.stop()
    
    section_divider()
    
    progress = st.progress(0, text="Initializing pricer...")
    
    pricer = MonteCarloPricerUni(
        num_simulations=min(num_sims, 25000),  # Limit for surface
        num_steps=50,
        use_numba=use_numba,
        use_gpu=False
    )
    
    # Generate grid
    n_spot = 20
    n_time = 15
    
    spot_range = np.linspace(S * 0.7, S * 1.3, n_spot)
    time_range = np.linspace(0.1, T, n_time)
    
    price_surface = np.zeros((n_time, n_spot))
    delta_surface = np.zeros((n_time, n_spot))
    
    total_points = n_spot * n_time
    t_start = time.perf_counter()
    
    # Use vectorized batch pricing for speed
    for i, t_val in enumerate(time_range):
        progress.progress(int((i + 1) / n_time * 80), text=f"Computing T={t_val:.2f}...")
        
        S_vals = spot_range
        K_vals = np.full(n_spot, K)
        T_vals = np.full(n_spot, t_val)
        r_vals = np.full(n_spot, r)
        sigma_vals = np.full(n_spot, sigma)
        
        prices = pricer.price_batch(S_vals, K_vals, T_vals, r_vals, sigma_vals, option_type)
        deltas, _ = pricer.delta_gamma_batch(S_vals, K_vals, T_vals, r_vals, sigma_vals, option_type)
        
        price_surface[i] = prices
        delta_surface[i] = deltas
    
    t_total = (time.perf_counter() - t_start) * 1000
    
    progress.progress(100, text="Complete!")
    time.sleep(0.2)
    progress.empty()
    
    # Timing info
    st.markdown(f"""
    <div class="metric-card" style="display: inline-block; padding: 0.5rem 1rem;">
        <span style="color: #94a3b8;">Surface computed:</span>
        <span style="color: #10b981; font-weight: 600;">{total_points} points in {format_time_ms(t_total)}</span>
        <span style="color: #94a3b8;">({t_total / total_points:.2f}ms per point)</span>
    </div>
    """, unsafe_allow_html=True)
    
    section_divider()
    
    # Display surfaces
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[go.Surface(
            x=spot_range,
            y=time_range,
            z=price_surface,
            colorscale='Viridis',
            opacity=0.9,
            name='Price'
        )])
        
        fig.update_layout(
            title=dict(text="Price Surface", font=dict(size=18, color='#f8fafc')),
            scene=dict(
                xaxis_title='Spot ($)',
                yaxis_title='Time (years)',
                zaxis_title='Price ($)',
                bgcolor='rgba(15, 23, 42, 0.9)'
            ),
            paper_bgcolor='rgba(30, 41, 59, 0.8)',
            font=dict(color='#cbd5e1'),
            height=450,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(data=[go.Surface(
            x=spot_range,
            y=time_range,
            z=delta_surface,
            colorscale='RdYlGn',
            opacity=0.9,
            name='Delta'
        )])
        
        fig.update_layout(
            title=dict(text="Delta Surface", font=dict(size=18, color='#f8fafc')),
            scene=dict(
                xaxis_title='Spot ($)',
                yaxis_title='Time (years)',
                zaxis_title='Delta',
                bgcolor='rgba(15, 23, 42, 0.9)'
            ),
            paper_bgcolor='rgba(30, 41, 59, 0.8)',
            font=dict(color='#cbd5e1'),
            height=450,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmaps
    section_divider()
    st.markdown("**Heatmaps**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=go.Heatmap(
            x=spot_range,
            y=time_range,
            z=price_surface,
            colorscale='Viridis',
            colorbar=dict(title='Price ($)')
        ))
        fig.update_layout(**get_chart_layout("Price Heatmap", 350))
        fig.update_xaxes(title_text="Spot Price ($)")
        fig.update_yaxes(title_text="Time (years)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(data=go.Heatmap(
            x=spot_range,
            y=time_range,
            z=delta_surface,
            colorscale='RdYlGn',
            colorbar=dict(title='Delta')
        ))
        fig.update_layout(**get_chart_layout("Delta Heatmap", 350))
        fig.update_xaxes(title_text="Spot Price ($)")
        fig.update_yaxes(title_text="Time (years)")
        st.plotly_chart(fig, use_container_width=True)

# Show help if nothing running
if not run_single and not run_surface:
    st.info("👆 Configure parameters and click **Single Price** for one calculation or **Price Surface** for full grid.")
    
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #60a5fa; margin-bottom: 1rem;">Unified Pricer Features</h3>
        <ul style="color: #cbd5e1; line-height: 2;">
            <li><strong>Numba JIT:</strong> Module-level compiled kernels for 5-10x CPU speedup</li>
            <li><strong>GPU Acceleration:</strong> CuPy-based CUDA kernels for massive parallelism</li>
            <li><strong>Vectorized Batch:</strong> Price hundreds of options in a single call</li>
            <li><strong>Common Random Numbers:</strong> Variance reduction for Greeks</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)