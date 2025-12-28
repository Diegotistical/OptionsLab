import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- 1. Robust Path Setup ---
# Automatically finds the 'src' folder regardless of this file's depth
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]  # Adjusts to OptionsLab root
src_path = project_root / "src"

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# --- 2. Backend Import Strategy ---
try:
    from pricing_models.binomial_tree import BinomialTree, ExerciseStyle, OptionType
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    IMPORT_ERROR = str(e)

# --- 3. Pure Python Fallback (For "Disable Numba" Benchmarking) ---
def pure_python_binomial_pricer(S, K, T, r, sigma, q, n, option_type, exercise_style):
    """
    Slow, pure-Python implementation to demonstrate the speed difference
    when Numba acceleration is disabled.
    """
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    df = math.exp(-r * dt)
    drift = math.exp((r - q) * dt)
    p = (drift - d) / (u - d)
    p = max(0.0, min(1.0, p))

    # Initialize leaves
    values = [0.0] * (n + 1)
    for j in range(n + 1):
        spot = S * (d ** (n - j)) * (u ** j)
        if "call" in option_type:
            values[j] = max(spot - K, 0.0)
        else:
            values[j] = max(K - spot, 0.0)

    # Backward induction
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            continuation = df * (p * values[j + 1] + (1 - p) * values[j])
            
            if "american" in exercise_style:
                spot = S * (d ** (i - j)) * (u ** j)
                if "call" in option_type:
                    intrinsic = max(spot - K, 0.0)
                else:
                    intrinsic = max(K - spot, 0.0)
                values[j] = max(continuation, intrinsic)
            else:
                values[j] = continuation
                
    return values[0]

# --- 4. Page Configuration & CSS ---
st.set_page_config(page_title="Binomial Tree Pricing", page_icon="🌳", layout="wide")

st.markdown(
    """
<style>
    /* Global Spacing */
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Custom Card Styling */
    .metric-card {
        background-color: #262730;
        border: 1px solid #464b59;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
    }
    
    /* Performance Badge */
    .badge-fast {
        background-color: #00cc96;
        color: #000;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-slow {
        background-color: #ef553b;
        color: white;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Slider Color */
    .stSlider > div > div > div > div { background: #ff4b4b; }
</style>
""",
    unsafe_allow_html=True,
)

if not BACKEND_AVAILABLE:
    st.error("❌ Critical Error: Backend Not Found")
    st.info(f"Could not import `BinomialTree`. Python path includes: {src_path}")
    st.code(IMPORT_ERROR)
    st.stop()

# --- 5. UI Layout ---

# Title Section
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.title("🌳 Binomial Tree Pricing")
    st.caption("Production-Grade CRR Model • O(N) Memory • Analytical Greeks")

# Sidebar Configuration
st.sidebar.header("⚙️ Pricing Parameters")

# Sidebar Inputs
with st.sidebar:
    st.subheader("Asset & Market")
    S = st.number_input("Spot Price (S)", 0.1, 10000.0, 100.0, 1.0)
    K = st.number_input("Strike Price (K)", 0.1, 10000.0, 100.0, 1.0)
    r = st.number_input("Risk-Free Rate (%)", 0.0, 100.0, 5.0, 0.1) / 100
    q = st.number_input("Dividend Yield (%)", 0.0, 100.0, 0.0, 0.1) / 100
    
    st.subheader("Option Properties")
    sigma = st.number_input("Volatility (%)", 0.1, 500.0, 20.0, 1.0) / 100
    T = st.number_input("Time to Maturity (Yrs)", 0.01, 10.0, 1.0, 0.1)
    
    col_type, col_style = st.columns(2)
    with col_type:
        option_type = st.selectbox("Type", ["Call", "Put"])
    with col_style:
        exercise_style = st.selectbox("Style", ["European", "American"])

    st.markdown("---")
    st.subheader("🔧 Engine Settings")
    num_steps = st.slider("Lattice Steps (N)", 10, 2500, 200, 10)
    
    use_numba = st.toggle("🚀 Enable Numba Acceleration", value=True)
    
    with st.expander("Visualization & Analytics", expanded=False):
        show_tree = st.checkbox("Show Tree Diagram", value=True)
        calculate_greeks = st.checkbox("Calculate Greeks", value=True)
        convergence_analysis = st.checkbox("Run Convergence Test", value=False)

# --- 6. Main Calculation Engine ---

# Display Current Engine Status in Top Right
with col_badge:
    st.write("") # Spacer
    if use_numba:
        st.markdown('<div style="text-align: right;"><span class="badge-fast">⚡ Numba Active</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align: right;"><span class="badge-slow">🐌 Pure Python</span></div>', unsafe_allow_html=True)

if st.button("Calculate Option Price", type="primary", use_container_width=True):
    try:
        # 1. Normalize Inputs (Robust String Handling)
        opt_clean = str(option_type).lower().strip()
        ex_clean = str(exercise_style).lower().strip()
        
        start_time = time.time()
        
        # 2. Execution Logic
        if use_numba:
            # -- FAST PATH --
            bt = BinomialTree(num_steps=num_steps)
            if calculate_greeks and hasattr(bt, "calculate_all"):
                results = bt.calculate_all(S, K, T, r, sigma, opt_clean, ex_clean, q)
                price = results["price"]
                delta = results["delta"]
                gamma = results["gamma"]
            else:
                price = bt.price(S, K, T, r, sigma, opt_clean, ex_clean, q)
                delta, gamma = 0.0, 0.0
        else:
            # -- SLOW PATH --
            price = pure_python_binomial_pricer(S, K, T, r, sigma, q, num_steps, opt_clean, ex_clean)
            delta, gamma = 0.0, 0.0
            
        end_time = time.time()
        calc_time_ms = (end_time - start_time) * 1000

        # 3. Robust Intrinsic Value Calculation
        # Explicitly casting to float to prevent any TypeErrors
        if "call" in opt_clean:
            intrinsic_val = max(float(S) - float(K), 0.0)
        else:
            intrinsic_val = max(float(K) - float(S), 0.0)
            
        time_val = max(float(price) - intrinsic_val, 0.0)
        moneyness = "ITM" if intrinsic_val > 0 else "ATM" if intrinsic_val == 0 else "OTM"

        # --- 7. Results Display ---
        
        # Row 1: Key Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Option Price", f"${price:.4f}", help="Theoretical Fair Value")
        m2.metric("Intrinsic Value", f"${intrinsic_val:.4f}", help="Value if exercised immediately")
        m3.metric("Time Value", f"${time_val:.4f}", help="Premium paid for uncertainty")
        m4.metric("Moneyness", moneyness, delta_color="off")

        # Row 2: Performance & Engine Stats (Custom Card)
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0; color: #aaa;">Engine Performance</h4>
                    <p style="font-size: 1.5rem; font-weight: bold; margin: 0;">{calc_time_ms:.2f} ms</p>
                    <p style="font-size: 0.8rem; color: #666;">Time to compute {num_steps} steps</p>
                </div>
                <div style="text-align: right; border-left: 1px solid #444; padding-left: 20px;">
                    <p style="margin: 2px;">Steps/ms: <b>{num_steps / calc_time_ms if calc_time_ms > 0 else 0:.1f}</b></p>
                    <p style="margin: 2px;">Virtual Nodes: <b>{((num_steps + 1) * (num_steps + 2)) // 2:,}</b></p>
                    <p style="margin: 2px;">Memory: <b>O(N)</b></p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Row 3: Greeks (Conditional)
        if calculate_greeks and use_numba:
            st.subheader("📊 Option Greeks")
            g1, g2, g3 = st.columns(3)
            g1.metric("Delta (Δ)", f"{delta:.4f}", help="Sensitivity to Spot Price")
            g2.metric("Gamma (Γ)", f"{gamma:.4f}", help="Sensitivity of Delta")
            g3.info("Analytical Greeks calculated during backward induction pass.")
        elif calculate_greeks and not use_numba:
            st.warning("⚠️ Analytical Greeks are only available in Numba Accelerated mode.")

        # --- 8. Visualizations ---
        
        # Tabbed Interface for Charts
        tab_tree, tab_conv, tab_dist = st.tabs(["🌿 Lattice Tree", "📈 Convergence", "🔔 Probability"])
        
        with tab_tree:
            if show_tree and num_steps >= 5:
                viz_steps = min(5, num_steps)
                dt = T / viz_steps
                u = np.exp(sigma * np.sqrt(dt))
                d = 1.0 / u
                
                # Build light visualization tree
                tree_data = []
                for i in range(viz_steps + 1):
                    for j in range(i + 1):
                        p_node = S * (u**j) * (d**(i-j))
                        tree_data.append({"step": i, "node": j, "price": p_node})
                df_tree = pd.DataFrame(tree_data)

                fig = go.Figure()
                
                # Draw Edges
                for i in range(viz_steps):
                    curr_layer = df_tree[df_tree["step"] == i]
                    next_layer = df_tree[df_tree["step"] == i + 1]
                    for _, node in curr_layer.iterrows():
                        up = next_layer[next_layer["node"] == node["node"] + 1]
                        dn = next_layer[next_layer["node"] == node["node"]]
                        
                        if not up.empty:
                            fig.add_trace(go.Scatter(
                                x=[i, i+1], y=[node["price"], up.iloc[0]["price"]],
                                mode="lines", line=dict(color="rgba(255,255,255,0.3)", width=1), showlegend=False
                            ))
                        if not dn.empty:
                            fig.add_trace(go.Scatter(
                                x=[i, i+1], y=[node["price"], dn.iloc[0]["price"]],
                                mode="lines", line=dict(color="rgba(255,255,255,0.3)", width=1), showlegend=False
                            ))
                
                # Draw Nodes
                for s in range(viz_steps + 1):
                    nodes = df_tree[df_tree["step"] == s]
                    fig.add_trace(go.Scatter(
                        x=nodes["step"], y=nodes["price"],
                        mode="markers+text", marker=dict(size=24, color="#ff4b4b"),
                        text=nodes["price"].round(2), textfont=dict(color="white", size=10),
                        name=f"Step {s}"
                    ))
                    
                fig.update_layout(
                    title=f"Binomial Lattice (First {viz_steps} Steps)",
                    template="plotly_dark", height=450,
                    xaxis=dict(title="Time Steps", showgrid=False),
                    yaxis=dict(title="Asset Price", showgrid=True)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Tree visualization disabled or steps too low.")

        with tab_conv:
            if convergence_analysis:
                st.write("Analyzing price stability as N increases...")
                conv_steps = sorted(list(set([10, 50, 100, 200, 500, num_steps])))
                conv_res = []
                
                # Progress bar for analysis
                prog_bar = st.progress(0)
                for i, n in enumerate(conv_steps):
                    bt_temp = BinomialTree(num_steps=n)
                    p_temp = bt_temp.price(S, K, T, r, sigma, opt_clean, ex_clean, q)
                    conv_res.append(p_temp)
                    prog_bar.progress((i + 1) / len(conv_steps))
                
                fig_conv = go.Figure()
                fig_conv.add_trace(go.Scatter(
                    x=conv_steps, y=conv_res, mode='lines+markers',
                    name='Price', line=dict(color='#00cc96', width=3)
                ))
                fig_conv.add_hline(y=price, line_dash="dash", annotation_text="Final Price")
                fig_conv.update_layout(title="Convergence Analysis", template="plotly_dark", height=400)
                st.plotly_chart(fig_conv, use_container_width=True)
            else:
                st.info("Enable 'Run Convergence Test' in the sidebar to view this chart.")

        with tab_dist:
            # Quick Log-Normal Distribution visualization
            st.write("Risk-Neutral Terminal Price Distribution")
            
            x = np.linspace(S*0.4, S*1.6, 200)
            mu = np.log(S) + (r - q - 0.5 * sigma**2) * T
            pdf = (1 / (x * sigma * np.sqrt(T) * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2 * T))
            
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Scatter(x=x, y=pdf, fill='tozeroy', name='PDF', line=dict(color='#ab63fa')))
            fig_dist.add_vline(x=K, line_dash="dash", line_color="white", annotation_text=f"Strike ${K}")
            fig_dist.update_layout(template="plotly_dark", height=400, xaxis_title="Price at Maturity")
            st.plotly_chart(fig_dist, use_container_width=True)

    except Exception as e:
        st.error("An unexpected error occurred during calculation.")
        st.exception(e)