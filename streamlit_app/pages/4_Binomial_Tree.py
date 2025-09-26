import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

try:
    from pricing_models.binomial_tree import BinomialTree, OptionType, ExerciseStyle
except ImportError:
    st.error("❌ Could not import BinomialTree from pricing_models.binomial_tree")
    st.info("Make sure the module is available in src/pricing_models/")

# Page config
st.set_page_config(
    page_title="Binomial Tree Pricing",
    page_icon="🌳",
    layout="wide"
)

# Custom CSS for dark mode
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 100%;
    }
    .stSlider > div > div > div > div {
        background: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌳 Binomial Tree Pricing")
st.caption("CRR Binomial Tree Model for European & American Options")

# Sidebar parameters
st.sidebar.header("⚙️ Pricing Parameters")

# Input parameters in two columns
col1, col2 = st.columns(2)

with col1:
    S = st.number_input("Spot Price (S)", min_value=0.1, value=100.0, step=1.0)
    K = st.number_input("Strike Price (K)", min_value=0.1, value=100.0, step=1.0)
    T = st.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0, step=0.1)
    r = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=5.0, step=0.1) / 100

with col2:
    sigma = st.number_input("Volatility (%)", min_value=0.1, value=20.0, step=1.0) / 100
    q = st.number_input("Dividend Yield (%)", min_value=0.0, value=0.0, step=0.1) / 100
    option_type = st.selectbox("Option Type", ["call", "put"])
    exercise_style = st.selectbox("Exercise Style", ["european", "american"])

# Advanced parameters
with st.expander("🔧 Advanced Parameters"):
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        num_steps = st.slider("Tree Steps", min_value=10, max_value=1000, value=100, step=10)
        h = st.number_input("Greek Bump Size", min_value=0.0001, value=0.0001, step=0.0001, format="%.4f")
    
    with adv_col2:
        show_tree = st.checkbox("Visualize Tree (First 5 Steps)", value=True)
        calculate_greeks = st.checkbox("Calculate Greeks", value=True)

# Pricing button
if st.button("🎯 Calculate Price", use_container_width=True):
    try:
        # Initialize model
        bt = BinomialTree(num_steps=num_steps)
        
        # Calculate price
        price = bt.price(S, K, T, r, sigma, option_type, exercise_style, q)
        
        # Display results
        st.success(f"**Option Price: ${price:.4f}**")
        
        # Calculate Greeks if requested
        if calculate_greeks:
            delta = bt.delta(S, K, T, r, sigma, option_type, exercise_style, q, h)
            gamma = bt.gamma(S, K, T, r, sigma, option_type, exercise_style, q, h)
            
            greek_col1, greek_col2, greek_col3 = st.columns(3)
            with greek_col1:
                st.metric("Delta", f"{delta:.4f}")
            with greek_col2:
                st.metric("Gamma", f"{gamma:.4f}")
            with greek_col3:
                intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
                st.metric("Intrinsic Value", f"${intrinsic:.4f}")
        
        # Visualize tree (first 5 steps for clarity)
        if show_tree and num_steps >= 5:
            st.subheader("🌿 Binomial Tree Visualization")
            
            # Create a simplified tree for visualization
            viz_steps = min(5, num_steps)
            dt = T / viz_steps
            u = np.exp(sigma * np.sqrt(dt))
            d = 1.0 / u
            
            # Generate tree data
            tree_data = []
            for i in range(viz_steps + 1):
                for j in range(i + 1):
                    price_node = S * (u ** j) * (d ** (i - j))
                    tree_data.append({
                        'step': i,
                        'node': j,
                        'price': price_node
                    })
            
            df_tree = pd.DataFrame(tree_data)
            
            # Create interactive tree visualization
            fig = go.Figure()
            
            # Add nodes and edges
            for i in range(viz_steps):
                current_nodes = df_tree[df_tree['step'] == i]
                next_nodes = df_tree[df_tree['step'] == i + 1]
                
                for _, node in current_nodes.iterrows():
                    # Connect to up and down nodes
                    up_node = next_nodes[next_nodes['node'] == node['node'] + 1]
                    down_node = next_nodes[next_nodes['node'] == node['node']]
                    
                    if not up_node.empty:
                        fig.add_trace(go.Scatter(
                            x=[i, i+1], y=[node['price'], up_node.iloc[0]['price']],
                            mode='lines', line=dict(color='white', width=1),
                            showlegend=False
                        ))
                    
                    if not down_node.empty:
                        fig.add_trace(go.Scatter(
                            x=[i, i+1], y=[node['price'], down_node.iloc[0]['price']],
                            mode='lines', line=dict(color='white', width=1),
                            showlegend=False
                        ))
            
            # Add nodes
            for step in range(viz_steps + 1):
                step_nodes = df_tree[df_tree['step'] == step]
                fig.add_trace(go.Scatter(
                    x=step_nodes['step'],
                    y=step_nodes['price'],
                    mode='markers+text',
                    marker=dict(size=15, color='#ff4b4b'),
                    text=step_nodes['price'].round(2),
                    textposition="middle center",
                    name=f'Step {step}',
                    textfont=dict(color='white', size=10)
                ))
            
            fig.update_layout(
                title=f"Binomial Tree (First {viz_steps} Steps)",
                xaxis_title="Time Steps",
                yaxis_title="Asset Price",
                showlegend=False,
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional information
        with st.expander("📊 Model Details"):
            dt = T / num_steps
            u = np.exp(sigma * np.sqrt(dt))
            d = 1.0 / u
            p = (np.exp((r - q) * dt) - d) / (u - d)
            
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.write("**Tree Parameters:**")
                st.write(f"Up Factor (u): {u:.6f}")
                st.write(f"Down Factor (d): {d:.6f}")
                st.write(f"Risk-Neutral Prob (p): {p:.6f}")
                st.write(f"Time Step (dt): {dt:.6f}")
            
            with info_col2:
                st.write("**Model Info:**")
                st.write(f"Steps: {num_steps}")
                st.write(f"Option Type: {option_type.title()}")
                st.write(f"Exercise: {exercise_style.title()}")
                st.write(f"Dividend Yield: {q:.2%}")
                
    except Exception as e:
        st.error(f"❌ Error in calculation: {str(e)}")

# Theory section
with st.expander("📚 Binomial Tree Theory"):
    st.markdown("""
    ### Cox-Ross-Rubinstein (CRR) Model
    
    The binomial tree model prices options by creating a lattice of possible asset price paths.
    
    **Key Formulas:**
    - Up factor: $u = e^{\\sigma\\sqrt{\\Delta t}}$
    - Down factor: $d = 1/u = e^{-\\sigma\\sqrt{\\Delta t}}$
    - Risk-neutral probability: $p = \\frac{e^{(r-q)\\Delta t} - d}{u - d}$
    
    **Process:**
    1. Construct asset price tree forward in time
    2. Calculate option payoffs at maturity
    3. Work backward through the tree, discounting expected values
    4. For American options, compare with early exercise value at each node
    
    **Advantages:**
    - Handles American exercise features
    - Intuitive and transparent methodology
    - Converges to Black-Scholes as steps increase
    """)

# Example presets
st.sidebar.markdown("---")
st.sidebar.header("💡 Example Presets")

preset = st.sidebar.selectbox("Load Preset", [
    "Custom",
    "ATM Call (European)",
    "ITM Put (American)", 
    "OTM Call (High Vol)"
])

if preset != "Custom":
    if preset == "ATM Call (European)":
        st.sidebar.info("ATM European Call: S=K=100, T=1, σ=20%")
    elif preset == "ITM Put (American)":
        st.sidebar.info("ITM American Put: S=95, K=100, T=0.5, σ=25%")
    elif preset == "OTM Call (High Vol)":
        st.sidebar.info("OTM Call with High Vol: S=100, K=110, T=2, σ=40%")