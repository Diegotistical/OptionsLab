# streamlit_app/pages/4_Binomial_Tree.py
"""
Binomial Tree Pricing - Streamlit Page.

European and American option pricing with tree visualization.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Binomial Tree",
    page_icon="🌳",
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
    from src.pricing_models import BinomialTree, black_scholes
except ImportError:
    BinomialTree = None
    black_scholes = None

apply_custom_css()

# =============================================================================
# HEADER
# =============================================================================
page_header("Binomial Tree Pricing", "Cox-Ross-Rubinstein model for European and American options")

# =============================================================================
# INPUT SECTION
# =============================================================================
st.markdown('<div class="input-section">', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 0.8])

with col1:
    st.markdown("**Asset**")
    S = st.number_input("Spot Price ($)", min_value=1.0, max_value=500.0, value=100.0, step=1.0, key="bt_spot")
    K = st.number_input("Strike Price ($)", min_value=1.0, max_value=500.0, value=100.0, step=1.0, key="bt_strike")

with col2:
    st.markdown("**Time & Type**")
    T = st.number_input("Maturity (years)", min_value=0.01, max_value=5.0, value=1.0, step=0.05, key="bt_time")
    option_type = st.selectbox("Option Type", ["call", "put"], key="bt_type")

with col3:
    st.markdown("**Market**")
    r_pct = st.number_input("Rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5, key="bt_rate")
    sigma_pct = st.number_input("Vol (%)", min_value=1.0, max_value=100.0, value=20.0, step=1.0, key="bt_vol")
    r, sigma = r_pct / 100.0, sigma_pct / 100.0

with col4:
    st.markdown("**Tree Settings**")
    n_steps = st.select_slider("Steps", options=[10, 25, 50, 100, 200, 500, 1000], value=100, key="bt_steps")
    style = st.selectbox("Exercise Style", ["european", "american"], key="bt_style")

with col5:
    st.markdown("**Run**")
    st.write("")
    run = st.button("🌳 Price Option", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# RESULTS
# =============================================================================
if run:
    if BinomialTree is None:
        st.error("Binomial Tree not available. Check installation.")
        st.stop()
    
    with st.spinner("Building tree..."):
        tree = BinomialTree(num_steps=n_steps)
        
        t_start = time.perf_counter()
        bt_price = tree.price(S, K, T, r, sigma, option_type, style)
        t_price = (time.perf_counter() - t_start) * 1000
        
        # Get BS for comparison (European only)
        bs_price = None
        if black_scholes and style == "european":
            bs_price = black_scholes(S, K, T, r, sigma, option_type)
    
    section_divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tree Price</div>
            <div class="metric-value">{format_price(bt_price)}</div>
            <div class="metric-delta">{format_time_ms(t_price)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if bs_price:
            error = abs(bt_price - bs_price)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">BS Price</div>
                <div class="metric-value">{format_price(bs_price)}</div>
                <div class="metric-delta">Error: ${error:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Exercise</div>
                <div class="metric-value">American</div>
                <div class="metric-delta">Early exercise possible</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tree Steps</div>
            <div class="metric-value">{n_steps:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Style</div>
            <div class="metric-value">{style.title()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    section_divider()
    
    # Convergence and tree visualization
    tab1, tab2, tab3 = st.tabs(["📈 Convergence", "🌳 Tree Structure", "📊 Sensitivity"])
    
    with tab1:
        # Convergence plot
        step_range = [5, 10, 20, 50, 100, 200, 500]
        if n_steps > 500:
            step_range.append(n_steps)
        
        prices = []
        for steps in step_range:
            t = BinomialTree(num_steps=steps)
            p = t.price(S, K, T, r, sigma, option_type, style)
            prices.append(p)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=step_range, y=prices,
            mode='lines+markers',
            line=dict(width=3, color='#60a5fa'),
            marker=dict(size=10),
            name='Tree Price'
        ))
        
        if bs_price:
            fig.add_hline(y=bs_price, line_dash="dash", line_color="#10b981",
                          annotation_text=f"BS: ${bs_price:.4f}")
        
        fig.update_layout(**get_chart_layout("Convergence to Black-Scholes", 400))
        fig.update_xaxes(title_text="Number of Steps", type="log")
        fig.update_yaxes(title_text="Option Price ($)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Simple tree visualization (limited steps)
        viz_steps = min(6, n_steps)  # Only visualize first 6 steps
        
        # Build price lattice
        dt = T / viz_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        
        # Create nodes
        nodes_x = []
        nodes_y = []
        node_text = []
        
        for i in range(viz_steps + 1):
            for j in range(i + 1):
                price = S * (u ** (i - j)) * (d ** j)
                nodes_x.append(i)
                nodes_y.append(price)
                node_text.append(f"${price:.2f}")
        
        # Create edges
        edge_x = []
        edge_y = []
        
        for i in range(viz_steps):
            for j in range(i + 1):
                current_price = S * (u ** (i - j)) * (d ** j)
                up_price = S * (u ** (i + 1 - j)) * (d ** j)
                down_price = S * (u ** (i - j)) * (d ** (j + 1))
                
                # Up edge
                edge_x.extend([i, i + 1, None])
                edge_y.extend([current_price, up_price, None])
                
                # Down edge
                edge_x.extend([i, i + 1, None])
                edge_y.extend([current_price, down_price, None])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='#475569'),
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=nodes_x, y=nodes_y,
            mode='markers+text',
            marker=dict(size=12, color='#60a5fa'),
            text=node_text,
            textposition='top center',
            textfont=dict(size=10, color='#cbd5e1'),
            name='Stock Price'
        ))
        
        fig.add_hline(y=K, line_dash="dash", line_color="#ef4444", annotation_text="Strike")
        
        fig.update_layout(**get_chart_layout(f"Binomial Tree ({viz_steps} steps shown)", 450))
        fig.update_xaxes(title_text="Time Step")
        fig.update_yaxes(title_text="Stock Price ($)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        if n_steps > viz_steps:
            st.caption(f"Note: Only first {viz_steps} steps shown. Full tree has {n_steps} steps.")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Price vs Spot
            spot_range = np.linspace(S * 0.6, S * 1.4, 30)
            tree_prices = []
            
            for s in spot_range:
                t = BinomialTree(num_steps=min(n_steps, 100))
                p = t.price(s, K, T, r, sigma, option_type, style)
                tree_prices.append(p)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=spot_range, y=tree_prices,
                mode='lines',
                line=dict(width=3, color='#a78bfa'),
                fill='tozeroy',
                fillcolor='rgba(167, 139, 250, 0.2)'
            ))
            
            fig.add_vline(x=K, line_dash="dash", line_color="#ef4444")
            fig.add_vline(x=S, line_dash="dot", line_color="#10b981")
            
            fig.update_layout(**get_chart_layout("Price vs Spot", 350))
            fig.update_xaxes(title_text="Spot Price ($)")
            fig.update_yaxes(title_text="Option Price ($)")
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Price vs Volatility
            vol_range = np.linspace(0.05, 0.6, 30)
            vol_prices = []
            
            for v in vol_range:
                t = BinomialTree(num_steps=min(n_steps, 100))
                p = t.price(S, K, T, r, v, option_type, style)
                vol_prices.append(p)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=vol_range * 100, y=vol_prices,
                mode='lines',
                line=dict(width=3, color='#34d399'),
                fill='tozeroy',
                fillcolor='rgba(52, 211, 153, 0.2)'
            ))
            
            fig.add_vline(x=sigma * 100, line_dash="dot", line_color="#10b981")
            
            fig.update_layout(**get_chart_layout("Price vs Volatility", 350))
            fig.update_xaxes(title_text="Volatility (%)")
            fig.update_yaxes(title_text="Option Price ($)")
            
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Configure parameters and click **Price Option** to run binomial tree pricing.")
    
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #60a5fa; margin-bottom: 1rem;">Binomial Tree Model</h3>
        <p style="color: #cbd5e1; line-height: 1.8;">
            The Cox-Ross-Rubinstein (CRR) binomial model prices options by building a tree
            of possible stock prices over time, then working backwards to find the option value.
        </p>
        <ul style="color: #94a3b8; margin-top: 1rem; line-height: 2;">
            <li><strong>European:</strong> Exercise only at expiration</li>
            <li><strong>American:</strong> Early exercise possible (important for puts)</li>
            <li><strong>Convergence:</strong> Price approaches Black-Scholes as steps increase</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)