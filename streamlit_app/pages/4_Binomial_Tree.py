import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- 1. Robust Path Setup ---
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "src"

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# --- 2. Backend Import Strategy ---
try:
    from pricing_models.binomial_tree import BinomialTree
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

# --- 3. Pure Python Fallback ---
def pure_python_binomial_pricer(S, K, T, r, sigma, q, n, option_type, exercise_style):
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    df = math.exp(-r * dt)
    drift = math.exp((r - q) * dt)
    p = (drift - d) / (u - d)
    p = max(0.0, min(1.0, p))

    values = [0.0] * (n + 1)
    is_call = (option_type == "call")
    
    # Initialize Terminal Leaves
    for j in range(n + 1):
        spot = S * (d ** (n - j)) * (u ** j)
        if is_call:
            values[j] = max(spot - K, 0.0)
        else:
            values[j] = max(K - spot, 0.0)

    # Backward Induction
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            continuation = df * (p * values[j + 1] + (1 - p) * values[j])
            
            if exercise_style == "american":
                spot = S * (d ** (i - j)) * (u ** j)
                if is_call:
                    intrinsic = max(spot - K, 0.0)
                else:
                    intrinsic = max(K - spot, 0.0)
                values[j] = max(continuation, intrinsic)
            else:
                values[j] = continuation
                
    return values[0]

# --- 4. Page Config & CSS ---
st.set_page_config(page_title="Binomial Tree Pricing", page_icon="🌳", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .metric-container {
        background: #1E1E1E; padding: 20px; border-radius: 12px;
        border: 1px solid #333; margin-top: 10px;
    }
    .input-card {
        background: #262730; padding: 20px; border-radius: 10px; margin-bottom: 20px;
    }
    .badge {
        padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 0.8em;
    }
    .badge-on { background: #00cc96; color: black; }
    .badge-off { background: #ff4b4b; color: white; }
</style>
""", unsafe_allow_html=True)

# --- 5. Main Layout ---
col_head, col_stat = st.columns([3, 1])
with col_head:
    st.title("🌳 Binomial Tree Pricing")
    st.caption("Production-Grade CRR Model • O(N) Memory • Analytical Greeks")

# --- CONTROL PANEL (Replaces Sidebar) ---
with st.container():
    st.markdown("### ⚙️ Control Panel")
    with st.expander("Show Pricing Inputs", expanded=True):
        # Row 1: Market Data
        c1, c2, c3, c4 = st.columns(4)
        with c1: S = st.number_input("Spot Price ($)", 1.0, 10000.0, 100.0)
        with c2: K = st.number_input("Strike Price ($)", 1.0, 10000.0, 100.0)
        with c3: r = st.number_input("Risk-Free Rate (%)", 0.0, 100.0, 5.0) / 100
        with c4: sigma = st.number_input("Volatility (%)", 1.0, 500.0, 20.0) / 100

        # Row 2: Option Properties
        c5, c6, c7, c8 = st.columns(4)
        with c5: T = st.number_input("Maturity (Years)", 0.01, 10.0, 1.0)
        with c6: q = st.number_input("Dividend Yield (%)", 0.0, 100.0, 0.0) / 100
        with c7: 
            # STRICT INPUTS: We define exactly what the strings are here
            option_type = st.selectbox("Option Type", ["Call", "Put"]) 
        with c8: 
            exercise_style = st.selectbox("Exercise Style", ["European", "American"])

        # Row 3: Engine Settings
        st.markdown("---")
        c9, c10 = st.columns([1, 1])
        with c9:
            num_steps = st.slider("Lattice Steps (N)", 10, 2500, 200, 10)
        with c10:
            st.write("") # Spacer
            use_numba = st.checkbox("🚀 Enable Numba Acceleration", value=True)
            show_tree = st.checkbox("Visualize Tree", value=False)

# Update Status Badge
with col_stat:
    st.write("")
    if use_numba:
        st.markdown('<span class="badge badge-on">⚡ Numba Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-off">🐌 Python Mode</span>', unsafe_allow_html=True)

# --- 6. Execution & Results ---
if st.button("Calculate Option Price", type="primary"):
    
    # 1. Prepare Inputs for Backend (LowerCase)
    opt_lower = "call" if option_type == "Call" else "put"
    style_lower = "european" if exercise_style == "European" else "american"
    
    start_t = time.time()
    
    # 2. Run Model
    try:
        if use_numba and BACKEND_AVAILABLE:
            bt = BinomialTree(num_steps=num_steps)
            # Try/Catch for calculate_all existence
            if hasattr(bt, "calculate_all"):
                res = bt.calculate_all(S, K, T, r, sigma, opt_lower, style_lower, q)
                price, delta, gamma = res["price"], res["delta"], res["gamma"]
            else:
                price = bt.price(S, K, T, r, sigma, opt_lower, style_lower, q)
                delta, gamma = 0.0, 0.0
        else:
            price = pure_python_binomial_pricer(S, K, T, r, sigma, q, num_steps, opt_lower, style_lower)
            delta, gamma = 0.0, 0.0
            
    except Exception as e:
        st.error(f"Calculation Error: {e}")
        st.stop()
        
    calc_time = (time.time() - start_t) * 1000

    # 3. FIXED INTRINSIC VALUE LOGIC (Determinisitic)
    # We rely on the exact variable from the dropdown, not string matching
    if option_type == "Call":
        intrinsic_val = max(S - K, 0.0)
    else:
        intrinsic_val = max(K - S, 0.0)
        
    time_val = max(price - intrinsic_val, 0.0)
    moneyness = "ITM" if intrinsic_val > 0 else "ATM" if intrinsic_val == 0 else "OTM"

    # 4. Display Results
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Option Price", f"${price:.4f}")
    m2.metric("Intrinsic Value", f"${intrinsic_val:.4f}", help="Value if exercised now")
    m3.metric("Time Value", f"${time_val:.4f}")
    m4.metric("Moneyness", moneyness)
    
    if use_numba and BACKEND_AVAILABLE:
        st.markdown("---")
        g1, g2, g3 = st.columns(3)
        g1.metric("Delta (Δ)", f"{delta:.4f}")
        g2.metric("Gamma (Γ)", f"{gamma:.4f}")
        g3.caption(f"⏱️ Calc Time: {calc_time:.2f} ms")
    st.markdown('</div>', unsafe_allow_html=True)

    # 5. Visualization (Tree)
    if show_tree and num_steps <= 20: # Limit to 20 for visual sanity
        st.subheader("Tree Visualization")
        viz_steps = min(5, num_steps)
        dt = T / viz_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        
        tree_data = []
        for i in range(viz_steps + 1):
            for j in range(i + 1):
                p_node = S * (u**j) * (d**(i-j))
                tree_data.append({"step": i, "node": j, "price": p_node})
        df_tree = pd.DataFrame(tree_data)

        fig = go.Figure()
        for s in range(viz_steps + 1):
            nodes = df_tree[df_tree["step"] == s]
            fig.add_trace(go.Scatter(
                x=nodes["step"], y=nodes["price"],
                mode="markers+text", marker=dict(size=18, color="#00cc96"),
                text=nodes["price"].round(1), textposition="top center",
                name=f"Step {s}"
            ))
        
        # Draw connections
        for i in range(viz_steps):
            curr = df_tree[df_tree["step"] == i]
            nex = df_tree[df_tree["step"] == i+1]
            for _, n in curr.iterrows():
                up = nex[nex["node"] == n["node"] + 1]
                dn = nex[nex["node"] == n["node"]]
                if not up.empty:
                    fig.add_trace(go.Scatter(x=[i, i+1], y=[n["price"], up.iloc[0]["price"]], mode="lines", line=dict(color="gray", width=1), showlegend=False))
                if not dn.empty:
                    fig.add_trace(go.Scatter(x=[i, i+1], y=[n["price"], dn.iloc[0]["price"]], mode="lines", line=dict(color="gray", width=1), showlegend=False))

        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    elif show_tree:
        st.warning("⚠️ Tree visualization disabled for steps > 20 to prevent browser lag.")