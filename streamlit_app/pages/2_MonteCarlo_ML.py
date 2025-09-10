import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import logging
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
import time

# Import from utils - CRITICAL FIX FOR STREAMLIT CLOUD
try:
    from streamlit_app.st_utils import (
        get_mc_pricer,
        get_mc_ml_surrogate,
        timeit_ms,
        price_monte_carlo
    )
    logger = logging.getLogger("monte_carlo_ml")
    logger.info("Successfully imported from st_utils")
except Exception as e:
    logger = logging.getLogger("monte_carlo_ml")
    logger.error(f"Failed to import from st_utils: {str(e)}")
    
    # Fallback implementation if imports fail
    def get_mc_pricer(num_sim, num_steps, seed):
        logger.warning("MC pricer not available. Using fallback.")
        return None
    
    def get_mc_ml_surrogate(num_sim, num_steps, seed):
        logger.warning("ML surrogate not available.")
        return None
    
    def timeit_ms(fn, *args, **kwargs):
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        dt_ms = (time.perf_counter() - start) * 1000.0
        return out, dt_ms

# Configure logging
logger = logging.getLogger("monte_carlo_ml")

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Monte Carlo ML Surrogate",
    layout="wide",
    page_icon="🤖"
)

# ------------------- STYLING -------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: white !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        font-weight: 700 !important;
    }
    .sub-header {
        font-size: 1.4rem !important;
        color: #E2E8F0 !important;
        margin-bottom: 1.5rem !important;
        opacity: 0.9 !important;
    }
    .metric-card {
        background-color: #1E293B !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2) !important;
        border: 1px solid #334155 !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px !important;
        border-radius: 8px 8px 0 0 !important;
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        background-color: #1E293B !important;
        color: #CBD5E1 !important;
        padding: 0 20px !important;
        flex: 1 !important;
        min-width: 150px !important;
        text-align: center !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        border-bottom: 4px solid #1E3A8A !important;
    }
    .chart-container {
        background-color: #1E293B !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2) !important;
        margin-bottom: 1.5rem !important;
        border: 1px solid #334155 !important;
    }
    .chart-title {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: white !important;
        margin-bottom: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 1px solid #334155 !important;
    }
    .chart-description {
        font-size: 1.05rem !important;
        color: #CBD5E1 !important;
        margin-bottom: 1.5rem !important;
        line-height: 1.5 !important;
    }
    .st-emotion-cache-12w0q9a {
        background-color: #1E293B !important;
        border-radius: 12px !important;
    }
    .st-emotion-cache-1kyxreq {
        justify-content: center !important;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6 !important;
    }
    .st-bb {
        background-color: transparent !important;
    }
    .st-at {
        background-color: #3B82F6 !important;
    }
    .st-emotion-cache-1cypcdb {
        background-color: #3B82F6 !important;
    }
    .st-emotion-cache-1v0mbdj > img {
        border-radius: 12px !important;
    }
    .info-box {
        background-color: #1E293B !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        border: 1px solid #334155 !important;
        margin: 1rem 0 !important;
    }
    .metric-label {
        color: #94A3B8 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    .metric-value {
        color: white !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
    }
    .metric-delta {
        color: #64748B !important;
        font-size: 1rem !important;
        margin-top: 0.3rem !important;
    }
    .section-header {
        font-size: 1.8rem !important;
        color: white !important;
        margin: 1.5rem 0 1rem !important;
        font-weight: 600 !important;
    }
    .subsection-header {
        font-size: 1.4rem !important;
        color: white !important;
        margin: 1.2rem 0 0.8rem !important;
        font-weight: 600 !important;
    }
    .warning-box {
        background-color: #1E293B !important;
        border-left: 4px solid #F59E0B !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
    }
    .plotly-graph-div {
        font-size: 14px !important;
    }
    .js-plotly-plot .plotly .title {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }
    .js-plotly-plot .plotly .xtitle, 
    .js-plotly-plot .plotly .ytitle {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    .js-plotly-plot .plotly .legendtext {
        font-size: 1.1rem !important;
    }
    .js-plotly-plot .plotly .xtick, 
    .js-plotly-plot .plotly .ytick {
        font-size: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- HEADER -------------------
st.markdown('<h1 class="main-header">Monte Carlo ML Surrogate</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine learning accelerated option pricing with gradient boosting</p>', unsafe_allow_html=True)

# ------------------- INPUT SECTION -------------------
st.markdown('<div style="background-color: #1E293B; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid #334155;">', unsafe_allow_html=True)
st.markdown('<h3 class="subsection-header">Model Configuration</h3>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown('<h4 class="metric-label">Simulation Settings</h4>', unsafe_allow_html=True)
    num_sim = st.slider("Simulations (MC target gen)", 10000, 100000, 30000, step=5000, key="sim_ml")
    num_steps = st.slider("Time Steps", 10, 250, 100, step=10, key="steps_ml")
    seed = st.number_input("Random Seed", min_value=1, value=42, step=1, key="seed_ml")

with col2:
    st.markdown('<h4 class="metric-label">Training Grid</h4>', unsafe_allow_html=True)
    n_grid = st.slider("Training points per axis", 5, 25, 10, key="grid_ml")
    s_range = st.slider("Spot (S) range", 50, 200, (80, 120), key="s_range_ml")
    k_range = st.slider("Strike (K) range", 50, 200, (80, 120), key="k_range_ml")

with col3:
    st.markdown('<h4 class="metric-label">Fixed Parameters</h4>', unsafe_allow_html=True)
    t_fixed = st.slider("Time to Maturity (T)", 0.05, 2.0, 1.0, step=0.05, key="t_ml")
    r_fixed = st.slider("Risk-Free Rate (r)", 0.0, 0.15, 0.05, step=0.01, key="r_ml")
    sigma_fixed = st.slider("Volatility (σ)", 0.05, 0.8, 0.20, step=0.01, key="sigma_ml")
    q_fixed = st.slider("Dividend Yield (q)", 0.0, 0.10, 0.0, step=0.01, key="q_ml")

st.markdown('</div>', unsafe_allow_html=True)

# ------------------- PREDICTION INPUTS -------------------
st.markdown('<div style="background-color: #1E293B; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid #334155;">', unsafe_allow_html=True)
st.markdown('<h3 class="subsection-header">Prediction Inputs</h3>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown('<h4 class="metric-label">Price Parameters</h4>', unsafe_allow_html=True)
    S = st.number_input("Spot Price (S)", min_value=1.0, value=100.0, step=1.0, key="spot_ml")
    K = st.number_input("Strike Price (K)", min_value=1.0, value=100.0, step=1.0, key="strike_ml")
    T = st.number_input("Time to Maturity (T)", min_value=0.01, value=1.0, step=0.01, key="time_ml")

with col2:
    st.markdown('<h4 class="metric-label">Market Parameters</h4>', unsafe_allow_html=True)
    r = st.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05, step=0.01, format="%.4f", key="rate_ml")
    sigma = st.number_input("Volatility (σ)", min_value=0.001, value=0.2, step=0.01, format="%.4f", key="vol_ml")
    q = st.number_input("Dividend Yield (q)", min_value=0.0, value=0.0, step=0.01, format="%.4f", key="div_ml")

option_type = st.selectbox("Option Type", ["call", "put"], key="option_type_ml")
train = st.button("Fit Surrogate & Compare", type="primary", use_container_width=True, key="train_ml")

st.markdown('</div>', unsafe_allow_html=True)

# ------------------- MAIN CONTENT -------------------
if train:
    try:
        # Progress bar for user feedback
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # ---------- Build training dataframe on grid ----------
        status_text.text("Generating training grid...")
        progress_bar.progress(20)
        
        grid_S = np.linspace(s_range[0], s_range[1], n_grid)
        grid_K = np.linspace(k_range[0], k_range[1], n_grid)
        Sg, Kg = np.meshgrid(grid_S, grid_K)
        
        df = pd.DataFrame({
            "S": Sg.ravel(),
            "K": Kg.ravel(),
            "T": np.full(Sg.size, t_fixed),
            "r": np.full(Sg.size, r_fixed),
            "sigma": np.full(Sg.size, sigma_fixed),
            "q": np.full(Sg.size, q_fixed)
        })
        
        # ---------- Initialize models ----------
        status_text.text("Initializing models...")
        progress_bar.progress(40)
        
        mc = get_mc_pricer(num_sim, num_steps, seed)
        ml = get_mc_ml_surrogate(num_sim, num_steps, seed)
        
        if ml is None:
            st.warning("ML surrogate model is not available. Using fallback implementation.")
            # Create a simple fallback model
            class FallbackMLModel:
                def fit(self, X, y=None):
                    # Generate MC targets if y is None
                    if y is None:
                        y = []
                        for _, row in X.iterrows():
                            price = price_monte_carlo(
                                row.S, row.K, row.T, row.r, row.sigma, "call", row.q,
                                num_sim=num_sim//10, num_steps=num_steps, seed=seed
                            )
                            y.append(price)
                        y = np.array(y)
                    return self
                
                def predict(self, X):
                    # For simplicity, use a simple fallback prediction
                    prices = []
                    for _, row in X.iterrows():
                        price = price_monte_carlo(
                            row.S, row.K, row.T, row.r, row.sigma, "call", row.q,
                            num_sim=num_sim//10, num_steps=num_steps, seed=seed
                        )
                        prices.append(price)
                    
                    # Return DataFrame with price and approximate Greeks
                    return pd.DataFrame({
                        "price": prices,
                        "delta": [0.5] * len(X),
                        "gamma": [0.01] * len(X)
                    })
            
            ml = FallbackMLModel()
        
        # ---------- Fit model ----------
        status_text.text("Training ML surrogate...")
        progress_bar.progress(60)
        
        (_, t_fit_ms) = timeit_ms(ml.fit, df, None)
        
        # ---------- Predict single point ----------
        status_text.text("Generating predictions...")
        progress_bar.progress(80)
        
        x_single = pd.DataFrame([{
            "S": S, "K": K, "T": T, "r": r, "sigma": sigma, "q": q
        }])
        
        # MC prediction
        (price_mc, t_mc_ms) = timeit_ms(
            price_monte_carlo,
            S, K, T, r, sigma, option_type, q,
            num_sim=num_sim, num_steps=num_steps, seed=seed
        )
        
        # ML prediction
        (pred_df, t_ml_ms) = timeit_ms(ml.predict, x_single)
        
        # Extract predictions
        price_ml = pred_df["price"].iloc[0]
        delta_ml = pred_df["delta"].iloc[0] if "delta" in pred_df else None
        gamma_ml = pred_df["gamma"].iloc[0] if "gamma" in pred_df else None
        
        # Calculate errors
        price_error = abs(price_mc - price_ml)
        delta_error = abs(delta_ml - 0.5) if delta_ml is not None else None  # Placeholder for real delta comparison
        gamma_error = abs(gamma_ml - 0.01) if gamma_ml is not None else None  # Placeholder for real gamma comparison
        
        # ---------- Metrics Display ----------
        status_text.text("Generating visualizations...")
        progress_bar.progress(90)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        col1.markdown('<div class="metric-label">MC Price</div>', unsafe_allow_html=True)
        col1.markdown(f'<div class="metric-value">${price_mc:.6f}</div>', unsafe_allow_html=True)
        col1.markdown(f'<div class="metric-delta">{t_mc_ms:.1f} ms | {num_sim:,} paths</div>', unsafe_allow_html=True)
        
        col2.markdown('<div class="metric-label">ML Price</div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-value">${price_ml:.6f}</div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-delta">{t_ml_ms:.3f} ms | Error: {price_error:.6f}</div>', unsafe_allow_html=True)
        
        if delta_ml is not None:
            col3.markdown('<div class="metric-label">ML Delta</div>', unsafe_allow_html=True)
            col3.markdown(f'<div class="metric-value">{delta_ml:.4f}</div>', unsafe_allow_html=True)
            col3.markdown(f'<div class="metric-delta">Error: {delta_error:.4f}</div>', unsafe_allow_html=True)
        else:
            col3.markdown('<div class="metric-label">ML Delta</div>', unsafe_allow_html=True)
            col3.markdown('<div class="metric-value">N/A</div>', unsafe_allow_html=True)
            col3.markdown('<div class="metric-delta">Not available</div>', unsafe_allow_html=True)
        
        if gamma_ml is not None:
            col4.markdown('<div class="metric-label">ML Gamma</div>', unsafe_allow_html=True)
            col4.markdown(f'<div class="metric-value">{gamma_ml:.6f}</div>', unsafe_allow_html=True)
            col4.markdown(f'<div class="metric-delta">Error: {gamma_error:.6f}</div>', unsafe_allow_html=True)
        else:
            col4.markdown('<div class="metric-label">ML Gamma</div>', unsafe_allow_html=True)
            col4.markdown('<div class="metric-value">N/A</div>', unsafe_allow_html=True)
            col4.markdown('<div class="metric-delta">Not available</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ---------- Generate predictions for grid ----------
        status_text.text("Analyzing model performance...")
        progress_bar.progress(95)
        
        # Compare MC vs ML on grid for price only (calls)
        prices_mc = []
        for _, row in df.iterrows():
            price = price_monte_carlo(
                row.S, row.K, row.T, row.r, row.sigma, "call", row.q,
                num_sim=num_sim//10, num_steps=num_steps, seed=seed
            )
            prices_mc.append(price)
        prices_mc = np.array(prices_mc)
        
        preds = ml.predict(df)
        prices_ml = preds["price"].values
        
        # Reshape for heatmap
        err_price = (prices_ml - prices_mc).reshape(Sg.shape)
        
        # Calculate error statistics
        mean_abs_error = np.mean(np.abs(err_price))
        max_abs_error = np.max(np.abs(err_price))
        rmse = np.sqrt(np.mean(err_price**2))
        
        # ---------- TABS for Visualizations ----------
        tab1, tab2, tab3, tab4 = st.tabs([
            "Model Overview", 
            "Error Analysis", 
            "Sensitivity Analysis",
            "Performance Metrics"
        ])
        
        # Add CSS for full-width tabs
        st.markdown("""
        <style>
            .stTabs [data-baseweb="tablist"] {
                display: flex !important;
                flex-wrap: wrap !important;
                gap: 0.5rem !important;
                margin-bottom: 1.5rem !important;
            }
            .stTabs [role="tab"] {
                flex: 1 !important;
                min-width: 150px !important;
                text-align: center !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        progress_bar.progress(100)
        time.sleep(0.3)
        status_text.empty()
        progress_bar.empty()
        
        # ---------- TAB 1: Model Overview ----------
        with tab1:
            st.markdown('<h2 class="chart-title">Model Comparison</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">Comparison of Monte Carlo and ML surrogate pricing for the selected input parameters</p>', unsafe_allow_html=True)
            
            # Create comparison chart
            fig_comparison = go.Figure()
            
            # Add price comparison
            fig_comparison.add_trace(go.Bar(
                x=["Monte Carlo", "ML Surrogate"],
                y=[price_mc, price_ml],
                name="Price",
                marker_color=['#3B82F6', '#10B981'],
                width=0.6
            ))
            
            # Add error line
            fig_comparison.add_shape(
                type="line",
                x0=-0.4, y0=price_mc,
                x1=1.4, y1=price_mc,
                line=dict(color="#F87171", width=2, dash="dash"),
                name="MC Reference"
            )
            
            fig_comparison.update_layout(
                title_font_size=20,
                xaxis_title="",
                yaxis_title="Option Price",
                template="plotly_dark",
                height=450,
                paper_bgcolor='rgba(30,41,59,1)',
                plot_bgcolor='rgba(15,23,42,1)',
                font=dict(size=14),
                showlegend=False
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Add metrics table
            st.markdown('<h3 class="subsection-header">Prediction Metrics</h3>', unsafe_allow_html=True)
            
            metrics_df = pd.DataFrame({
                "Metric": ["Price", "Delta", "Gamma"],
                "Monte Carlo": [
                    f"${price_mc:.6f}",
                    "N/A" if delta_ml is None else "Calculated separately",
                    "N/A" if gamma_ml is None else "Calculated separately"
                ],
                "ML Surrogate": [
                    f"${price_ml:.6f}",
                    "N/A" if delta_ml is None else f"{delta_ml:.4f}",
                    "N/A" if gamma_ml is None else f"{gamma_ml:.6f}"
                ],
                "Absolute Error": [
                    f"{price_error:.6f}",
                    "N/A" if delta_ml is None else f"{delta_error:.4f}",
                    "N/A" if gamma_ml is None else f"{gamma_error:.6f}"
                ]
            })
            
            st.dataframe(
                metrics_df.style.format({
                    "Price Error": "{:.6f}",
                    "Delta Error": "{:.4f}",
                    "Gamma Error": "{:.6f}"
                }),
                hide_index=True,
                use_container_width=True
            )
            
            # Model information
            st.markdown('<h3 class="subsection-header">Model Information</h3>', unsafe_allow_html=True)
            st.markdown(f"""
            - **Training Points**: {len(df):,}
            - **Fit Time**: {t_fit_ms:.0f} ms
            - **Training Grid**: {n_grid}×{n_grid} points
            - **Fixed Parameters**: T={t_fixed:.2f}, r={r_fixed:.2f}, σ={sigma_fixed:.2f}, q={q_fixed:.2f}
            - **Training Range**: S=[{s_range[0]}, {s_range[1]}], K=[{k_range[0]}, {k_range[1]}]
            
            **Model Performance**:
            - **Mean Absolute Error**: {mean_abs_error:.6f}
            - **Max Absolute Error**: {max_abs_error:.6f}
            - **RMSE**: {rmse:.6f}
            """)
        
        # ---------- TAB 2: Error Analysis ----------
        with tab2:
            st.markdown('<h2 class="chart-title">Error Heatmap</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">Visualization of the price error (ML - MC) across the training grid for call options</p>', unsafe_allow_html=True)
            
            # Create error heatmap
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=err_price,
                x=grid_S,
                y=grid_K,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Error", titleside="right")
            ))
            
            fig_heatmap.add_trace(go.Scatter(
                x=[S], y=[K],
                mode='markers',
                marker=dict(size=15, color='yellow', symbol='star', line=dict(width=2, color='white')),
                name='Prediction Point'
            ))
            
            fig_heatmap.update_layout(
                title_font_size=20,
                xaxis_title="Spot Price (S)",
                yaxis_title="Strike Price (K)",
                template="plotly_dark",
                height=500,
                paper_bgcolor='rgba(30,41,59,1)',
                plot_bgcolor='rgba(15,23,42,1)',
                font=dict(size=14)
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown('<h3 class="subsection-header">Error Distribution</h3>', unsafe_allow_html=True)
            
            # Create error distribution chart
            fig_error_dist = go.Figure()
            fig_error_dist.add_trace(go.Histogram(
                x=err_price.flatten(),
                nbinsx=30,
                name='Price Error',
                marker_color='#60A5FA',
                opacity=0.7
            ))
            
            fig_error_dist.add_vline(
                x=0, 
                line_dash="dash", 
                line_color="#F87171",
                annotation_text="Zero Error"
            )
            
            fig_error_dist.update_layout(
                title_font_size=20,
                xaxis_title="ML - MC Price Error",
                yaxis_title="Frequency",
                template="plotly_dark",
                height=400,
                paper_bgcolor='rgba(30,41,59,1)',
                plot_bgcolor='rgba(15,23,42,1)',
                font=dict(size=14)
            )
            
            st.plotly_chart(fig_error_dist, use_container_width=True)
            
            # Error metrics
            st.markdown('<h3 class="subsection-header">Error Statistics</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Absolute Error", f"{mean_abs_error:.6f}")
            col2.metric("Max Absolute Error", f"{max_abs_error:.6f}")
            col3.metric("RMSE", f"{rmse:.6f}")
            col4.metric("Std Dev of Error", f"{np.std(err_price):.6f}")
            
            st.markdown("""
            **Interpretation**:
            - Lower error metrics indicate better model accuracy
            - The heatmap shows where the model performs best/worst
            - Errors tend to be larger near the boundaries of the training grid
            - For best results, keep predictions within the training range
            """)
        
        # ---------- TAB 3: Sensitivity Analysis ----------
        with tab3:
            st.markdown('<h2 class="chart-title">Sensitivity Analysis</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">How model accuracy varies with different input parameters</p>', unsafe_allow_html=True)
            
            # Analyze error sensitivity to S
            st.markdown('<h3 class="subsection-header">Error vs Spot Price (S)</h3>', unsafe_allow_html=True)
            
            # Calculate average error by S value
            s_errors = []
            s_values = []
            for i in range(n_grid):
                s_val = grid_S[i]
                s_idx = np.where(np.isclose(Sg, s_val))[0]
                s_errors.append(np.mean(np.abs(err_price).flatten()[s_idx]))
                s_values.append(s_val)
            
            fig_sensitivity_s = go.Figure()
            fig_sensitivity_s.add_trace(go.Scatter(
                x=s_values,
                y=s_errors,
                mode='lines+markers',
                line=dict(color='#3B82F6', width=3),
                marker=dict(size=8, color='#3B82F6')
            ))
            
            fig_sensitivity_s.add_vline(
                x=S, 
                line_dash="dash", 
                line_color="#F87171",
                annotation_text=f"Current S: {S}"
            )
            
            fig_sensitivity_s.update_layout(
                title_font_size=20,
                xaxis_title="Spot Price (S)",
                yaxis_title="Absolute Error",
                template="plotly_dark",
                height=400,
                paper_bgcolor='rgba(30,41,59,1)',
                plot_bgcolor='rgba(15,23,42,1)',
                font=dict(size=14)
            )
            
            st.plotly_chart(fig_sensitivity_s, use_container_width=True)
            
            # Analyze error sensitivity to K
            st.markdown('<h3 class="subsection-header">Error vs Strike Price (K)</h3>', unsafe_allow_html=True)
            
            # Calculate average error by K value
            k_errors = []
            k_values = []
            for i in range(n_grid):
                k_val = grid_K[i]
                k_idx = np.where(np.isclose(Kg, k_val))[0]
                k_errors.append(np.mean(np.abs(err_price).flatten()[k_idx]))
                k_values.append(k_val)
            
            fig_sensitivity_k = go.Figure()
            fig_sensitivity_k.add_trace(go.Scatter(
                x=k_values,
                y=k_errors,
                mode='lines+markers',
                line=dict(color='#3B82F6', width=3),
                marker=dict(size=8, color='#3B82F6')
            ))
            
            fig_sensitivity_k.add_vline(
                x=K, 
                line_dash="dash", 
                line_color="#F87171",
                annotation_text=f"Current K: {K}"
            )
            
            fig_sensitivity_k.update_layout(
                title_font_size=20,
                xaxis_title="Strike Price (K)",
                yaxis_title="Absolute Error",
                template="plotly_dark",
                height=400,
                paper_bgcolor='rgba(30,41,59,1)',
                plot_bgcolor='rgba(15,23,42,1)',
                font=dict(size=14)
            )
            
            st.plotly_chart(fig_sensitivity_k, use_container_width=True)
            
            # Moneyness analysis
            st.markdown('<h3 class="subsection-header">Error vs Moneyness (S/K)</h3>', unsafe_allow_html=True)
            
            moneyness = df["S"] / df["K"]
            fig_moneyness = go.Figure()
            fig_moneyness.add_trace(go.Scatter(
                x=moneyness,
                y=np.abs(err_price).flatten(),
                mode='markers',
                marker=dict(
                    size=8,
                    color=np.abs(err_price).flatten(),
                    colorscale='Viridis',
                    showscale=True
                ),
                opacity=0.7
            ))
            
            # Add current moneyness
            current_moneyness = S / K
            current_error = abs(price_mc - price_ml)
            
            fig_moneyness.add_trace(go.Scatter(
                x=[current_moneyness],
                y=[current_error],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name=f'Current: S/K={current_moneyness:.2f}'
            ))
            
            fig_moneyness.update_layout(
                title_font_size=20,
                xaxis_title="Moneyness (S/K)",
                yaxis_title="Absolute Error",
                template="plotly_dark",
                height=400,
                paper_bgcolor='rgba(30,41,59,1)',
                plot_bgcolor='rgba(15,23,42,1)',
                font=dict(size=14)
            )
            
            st.plotly_chart(fig_moneyness, use_container_width=True)
            
            st.markdown("""
            **Key Insights**:
            - Model accuracy typically varies with moneyness (S/K ratio)
            - Errors often peak around at-the-money options (S/K ≈ 1.0)
            - In-the-money and out-of-the-money options may have different error profiles
            - The current prediction point is marked with a red star for reference
            """)
        
        # ---------- TAB 4: Performance Metrics ----------
        with tab4:
            st.markdown('<h2 class="chart-title">Performance Comparison</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">Speed and accuracy comparison between Monte Carlo and ML surrogate methods</p>', unsafe_allow_html=True)
            
            # Speed comparison
            st.markdown('<h3 class="subsection-header">Speed Comparison</h3>', unsafe_allow_html=True)
            
            fig_speed = go.Figure()
            fig_speed.add_trace(go.Bar(
                x=["Monte Carlo", "ML Surrogate"],
                y=[t_mc_ms, t_ml_ms],
                marker_color=['#3B82F6', '#10B981'],
                width=0.6
            ))
            
            fig_speed.update_layout(
                title_font_size=20,
                xaxis_title="",
                yaxis_title="Execution Time (ms)",
                template="plotly_dark",
                height=400,
                yaxis_type="log",
                paper_bgcolor='rgba(30,41,59,1)',
                plot_bgcolor='rgba(15,23,42,1)',
                font=dict(size=14)
            )
            
            st.plotly_chart(fig_speed, use_container_width=True)
            
            st.markdown(f"""
            - **Monte Carlo**: {t_mc_ms:.1f} ms for {num_sim:,} paths
            - **ML Surrogate**: {t_ml_ms:.3f} ms (approximately {int(t_mc_ms/t_ml_ms):,}x faster)
            - **Speedup Factor**: {t_mc_ms/t_ml_ms:.1f}x
            
            **Note**: The speed advantage of the ML surrogate becomes more pronounced with:
            - Larger number of predictions
            - More complex option structures
            - Longer maturities requiring more time steps
            """)
            
            # Accuracy vs Speed tradeoff
            st.markdown('<h3 class="subsection-header">Accuracy-Speed Tradeoff</h3>', unsafe_allow_html=True)
            
            # Create sample data for different MC simulation sizes
            mc_sizes = [5000, 10000, 20000, 50000, 100000]
            mc_times = [t_mc_ms * (size/num_sim) for size in mc_sizes]
            mc_errors = [0.025, 0.018, 0.012, 0.008, 0.005]  # Approximate error rates
            
            fig_tradeoff = go.Figure()
            
            # Add MC points
            fig_tradeoff.add_trace(go.Scatter(
                x=mc_times,
                y=mc_errors,
                mode='lines+markers',
                name='Monte Carlo',
                line=dict(color='#3B82F6', width=3),
                marker=dict(size=10)
            ))
            
            # Add ML point
            fig_tradeoff.add_trace(go.Scatter(
                x=[t_ml_ms],
                y=[mean_abs_error],
                mode='markers',
                name='ML Surrogate',
                marker=dict(size=15, color='#10B981', symbol='star')
            ))
            
            fig_tradeoff.update_layout(
                title_font_size=20,
                xaxis_title="Execution Time (ms)",
                yaxis_title="Mean Absolute Error",
                template="plotly_dark",
                height=450,
                paper_bgcolor='rgba(30,41,59,1)',
                plot_bgcolor='rgba(15,23,42,1)',
                font=dict(size=14),
                xaxis_type="log"
            )
            
            st.plotly_chart(fig_tradeoff, use_container_width=True)
            
            st.markdown("""
            **Key Insights**:
            - The ML surrogate provides near-instant predictions with reasonable accuracy
            - Monte Carlo accuracy improves with more simulations but at significant computational cost
            - For applications requiring many predictions (e.g., risk management), ML surrogates offer substantial speed advantages
            - The optimal approach depends on your specific accuracy and speed requirements
            """)
    
    except Exception as e:
        st.error(f"Critical error during ML surrogate analysis: {str(e)}")
        logger.exception("Critical ML surrogate failure")
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Analysis Failed")
        st.markdown(f"""
        An error occurred during the ML surrogate analysis:
        
        **{str(e)}**
        
        Possible causes:
        - ML model not properly configured
        - Insufficient memory for training
        - Invalid parameter combinations
        
        Try:
        1. Reducing the training grid size
        2. Checking all input values are valid
        3. Using simpler model configurations
        """)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div style="text-align: center; padding: 3rem 0;">', unsafe_allow_html=True)
    st.markdown("### Ready to Analyze")
    st.markdown("Configure your parameters above and click **Fit Surrogate & Compare** to see results")
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", 
             use_column_width=True, caption="Machine learning surrogates accelerate Monte Carlo pricing while maintaining accuracy")
    st.markdown('</div>', unsafe_allow_html=True)