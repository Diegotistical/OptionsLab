import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- 1. Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add the project root to sys.path so we can find 'src'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- 2. Imports ---
try:
    # Now we can import cleanly starting from 'src'
    from src.pricing_models.binomial_tree import BinomialTree, ExerciseStyle, OptionType
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.write(f"📂 Debug: Project Root identified as: {PROJECT_ROOT}")
    st.info("Ensure the file 'src/pricing_models/binomial_tree.py' exists relative to that root.")
    st.stop()
except AttributeError:
    st.error("❌ Stale Code Detected")
    st.warning("Please restart the app (Ctrl+C and run again) to load the updated backend code.")
    st.stop()

# --- 3. Page Configuration ---
st.set_page_config(page_title="Binomial Tree Pricing", page_icon="🌳", layout="wide")

st.markdown(
    """
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
    .performance-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff4b4b;
        margin: 0.5rem 0;
    }
    .numba-badge {
        background: #00a4db;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🌳 Binomial Tree Pricing")
st.caption("Production-Grade CRR Model • O(N) Memory • Analytical Greeks")

# --- 4. Sidebar Parameters ---
st.sidebar.header("⚙️ Pricing Parameters")

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

with st.expander("🔧 Advanced Parameters"):
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        num_steps = st.slider("Tree Steps", min_value=10, max_value=2000, value=200, step=10)
    with adv_col2:
        show_tree = st.checkbox("Visualize Tree (First 5 Steps)", value=True)
        calculate_greeks = st.checkbox("Calculate Greeks", value=True)
        convergence_analysis = st.checkbox("Convergence Analysis", value=True)

st.sidebar.markdown("---")
st.sidebar.header("🚀 Performance Info")
st.sidebar.markdown('<div class="numba-badge">Numba JIT Compiled</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
**Optimization Features:**
- O(N) Memory (1D Array)
- Analytical Greeks (Single Pass)
- FastMath Enabled
""")

# --- 5. Main Logic ---
if st.button("🎯 Calculate Price & Analyze", use_container_width=True):
    try:
        bt = BinomialTree(num_steps=num_steps)
        start_time = time.time()
        
        if calculate_greeks:
            if hasattr(bt, "calculate_all"):
                results = bt.calculate_all(S, K, T, r, sigma, option_type, exercise_style, q)
                price = results["price"]
                delta = results["delta"]
                gamma = results["gamma"]
            else:
                st.warning("⚠️ Using legacy calculation methods. Reboot app if issues persist.")
                price = bt.price(S, K, T, r, sigma, option_type, exercise_style, q)
                delta = bt.delta(S, K, T, r, sigma, option_type, exercise_style, q)
                gamma = bt.gamma(S, K, T, r, sigma, option_type, exercise_style, q)
        else:
            price = bt.price(S, K, T, r, sigma, option_type, exercise_style, q)
            delta, gamma = 0.0, 0.0

        pricing_time = (time.time() - start_time) * 1000

        # Intrinsic & Time Value
        if option_type.lower() == "call":
            intrinsic_value = max(S - K, 0.0)
        else:
            intrinsic_value = max(K - S, 0.0)

        time_value = max(price - intrinsic_value, 0.0)

        # Display Results
        st.success(f"**Option Price: ${price:.4f}**")

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Intrinsic Value", f"${intrinsic_value:.4f}")
        with col2: st.metric("Time Value", f"${time_value:.4f}")
        with col3:
            moneyness = "ITM" if intrinsic_value > 0 else "ATM" if intrinsic_value == 0 else "OTM"
            st.metric("Moneyness", moneyness)

        # Performance Metrics
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        with perf_col1: st.metric("Calculation Time", f"{pricing_time:.2f} ms")
        with perf_col2:
            steps_per_ms = num_steps / pricing_time if pricing_time > 0 else float("inf")
            st.metric("Steps/ms", f"{steps_per_ms:.1f}")
        with perf_col3:
            total_nodes = ((num_steps + 1) * (num_steps + 2)) // 2
            st.metric("Virtual Nodes", f"{total_nodes:,}")

        # Performance Card
        with st.container():
            st.markdown('<div class="performance-card">', unsafe_allow_html=True)
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            with perf_col1:
                st.write("**Optimization**")
                st.write("✅ O(N) Memory")
                st.write("✅ Analytical Greeks")
                st.write("✅ Numba JIT")
            with perf_col2:
                st.write("**Tree Statistics**")
                dt = T / num_steps
                u = np.exp(sigma * np.sqrt(dt))
                st.write(f"Steps: {num_steps:,}")
                st.write(f"Δt: {dt:.6f}")
                st.write(f"u: {u:.6f}")
            with perf_col3:
                st.write("**Model Info**")
                st.write(f"Type: {option_type.title()}")
                st.write(f"Exercise: {exercise_style.title()}")
                mem_kb = (num_steps * 8) / 1024
                st.write(f"Memory: ~{mem_kb:.2f} KB")
            st.markdown("</div>", unsafe_allow_html=True)

        # Greeks Display
        if calculate_greeks:
            st.subheader("📊 Greeks Calculation")
            greek_col1, greek_col2, greek_col3, greek_col4, greek_col5 = st.columns(5)
            with greek_col1: st.metric("Delta", f"{delta:.4f}", help="Price sensitivity to underlying")
            with greek_col2: st.metric("Gamma", f"{gamma:.4f}", help="Delta sensitivity to underlying")
            with greek_col3: st.metric("Intrinsic Value", f"${intrinsic_value:.4f}")
            with greek_col4: st.metric("Time Value", f"${time_value:.4f}")
            with greek_col5: st.metric("Greeks Time", "Included")

        # Tree Visualization
        if show_tree and num_steps >= 5:
            st.subheader("🌿 Binomial Tree Visualization")
            viz_steps = min(5, num_steps)
            dt = T / viz_steps
            u = np.exp(sigma * np.sqrt(dt))
            d = 1.0 / u

            tree_data = []
            for i in range(viz_steps + 1):
                for j in range(i + 1):
                    price_node = S * (u**j) * (d ** (i - j))
                    tree_data.append({"step": i, "node": j, "price": price_node})

            df_tree = pd.DataFrame(tree_data)
            fig = go.Figure()

            # Edges
            for i in range(viz_steps):
                current_nodes = df_tree[df_tree["step"] == i]
                next_nodes = df_tree[df_tree["step"] == i + 1]
                for _, node in current_nodes.iterrows():
                    up_node = next_nodes[next_nodes["node"] == node["node"] + 1]
                    down_node = next_nodes[next_nodes["node"] == node["node"]]
                    if not up_node.empty:
                        fig.add_trace(go.Scatter(
                            x=[i, i + 1], y=[node["price"], up_node.iloc[0]["price"]],
                            mode="lines", line=dict(color="white", width=1), showlegend=False
                        ))
                    if not down_node.empty:
                        fig.add_trace(go.Scatter(
                            x=[i, i + 1], y=[node["price"], down_node.iloc[0]["price"]],
                            mode="lines", line=dict(color="white", width=1), showlegend=False
                        ))

            # Nodes
            for step in range(viz_steps + 1):
                step_nodes = df_tree[df_tree["step"] == step]
                fig.add_trace(go.Scatter(
                    x=step_nodes["step"], y=step_nodes["price"],
                    mode="markers+text", marker=dict(size=15, color="#ff4b4b"),
                    text=step_nodes["price"].round(2), textposition="middle center",
                    name=f"Step {step}", textfont=dict(color="white", size=10)
                ))
            fig.update_layout(title=f"First {viz_steps} Steps", template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)

        # Convergence Analysis
        if convergence_analysis:
            st.subheader("📈 Convergence Analysis")
            st.info("Analyzing how price converges as tree steps increase...")
            
            convergence_steps = [10, 25, 50, 100, 250, 500, 1000]
            if num_steps not in convergence_steps:
                convergence_steps.append(num_steps)
                convergence_steps.sort()

            convergence_data = []
            progress_bar = st.progress(0)
            for i, steps in enumerate(convergence_steps):
                bt_temp = BinomialTree(num_steps=steps)
                conv_price = bt_temp.price(S, K, T, r, sigma, option_type, exercise_style, q)
                convergence_data.append({"steps": steps, "price": conv_price})
                progress_bar.progress((i + 1) / len(convergence_steps))

            df_conv = pd.DataFrame(convergence_data)
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=df_conv["steps"], y=df_conv["price"],
                mode="lines+markers", name="Price Convergence",
                line=dict(color="#ff4b4b", width=3), marker=dict(size=8)
            ))
            fig_conv.add_hline(y=price, line_dash="dash", line_color="green", annotation_text=f"Final: ${price:.4f}")
            fig_conv.update_layout(title="Convergence vs Steps", template="plotly_dark", height=400)
            st.plotly_chart(fig_conv, use_container_width=True)

        # Probability Distribution
        st.subheader("📊 Risk-Neutral Probability Distribution")
        dt = T / num_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        p = max(0.0, min(1.0, p))

        if num_steps > 100:
            st.info("Using Log-Normal approximation for large tree visualization.")
            mu = np.log(S) + (r - q - 0.5 * sigma**2) * T
            x = np.linspace(S*0.5, S*1.5, 100)
            pdf = (1 / (x * sigma * np.sqrt(T) * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2 * T))
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='Log-Normal PDF', line=dict(color="#00a4db")))
        else:
            terminal_prices, probabilities = [], []
            for j in range(num_steps + 1):
                terminal_price = S * (u**j) * (d ** (num_steps - j))
                try:
                    log_n_fact = math.lgamma(num_steps + 1)
                    log_k_fact = math.lgamma(j + 1)
                    log_nk_fact = math.lgamma(num_steps - j + 1)
                    log_comb = log_n_fact - log_k_fact - log_nk_fact
                    prob = math.exp(log_comb + j * math.log(p) + (num_steps - j) * math.log(1 - p))
                except: prob = 0
                terminal_prices.append(terminal_price)
                probabilities.append(prob)
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Bar(x=terminal_prices, y=probabilities, name="Probability", marker_color="#00a4db"))

        fig_dist.add_vline(x=K, line_dash="dash", line_color="red", annotation_text=f"Strike: ${K}")
        fig_dist.update_layout(title="Terminal Price Distribution", template="plotly_dark", height=400)
        st.plotly_chart(fig_dist, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error in calculation: {str(e)}")

# --- 6. Theory ---
with st.expander("📚 Binomial Tree Theory & Optimization"):
    tab1, tab2, tab3 = st.tabs(["Theory", "Optimization", "Algorithm"])
    with tab1:
        st.markdown(r"""
        **Key Formulas:**
        - Up factor: $u = e^{\sigma\sqrt{\Delta t}}$
        - Down factor: $d = 1/u = e^{-\sigma\sqrt{\Delta t}}$
        - Risk-neutral probability: $p = \frac{e^{(r-q)\Delta t} - d}{u - d}$
        """)
    with tab2:
        st.markdown("""
        **Why this is fast:**
        1. **O(N) Memory:** Stores only current column of values.
        2. **Analytical Greeks:** Calculated during backward pass (Step 1 & 2).
        3. **Numba JIT:** Compiled to machine code.
        """)
    with tab3:
        st.markdown("""
        ```python
        # Single array of size N+1
        values = initialize_payoffs()
        for step from n-1 down to 0:
            values[:step+1] = discount * (p * values[1:] + (1-p) * values[:-1])
        ```
        """)