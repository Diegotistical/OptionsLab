# streamlit_app/pages/volatility_surface.py
"""
Volatility Surface - Streamlit Page.

ML-based implied volatility surface modeling and visualization.
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
from scipy.interpolate import griddata
from scipy.stats import norm

st.set_page_config(
    page_title="Volatility Surface",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    from components import (
        apply_custom_css,
        format_time_ms,
        get_chart_layout,
        page_header,
        section_divider,
    )
except ImportError:
    from streamlit_app.components import (
        apply_custom_css,
        format_time_ms,
        get_chart_layout,
        page_header,
        section_divider,
    )

apply_custom_css()

# =============================================================================
# ML MODEL IMPORTS
# =============================================================================
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Benchmark and Data Loading imports
try:
    from src.benchmarks.vol_surface_benchmark import (
        VolSurfaceBenchmark,
        generate_synthetic_surface,
    )
    from src.data.data_loader import OptionChainLoader, OptionChainDataset

    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

# PINN Model import
try:
    from src.volatility_surface.models.pinn_model import (
        PINNVolatilityModel,
        create_pinn_model,
        TORCH_AVAILABLE,
        DEVICE_NAME,
    )
    PINN_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    PINN_AVAILABLE = False
    DEVICE_NAME = "N/A"


# =============================================================================
# VECTORIZED SURFACE CALCULATIONS
# =============================================================================


def generate_iv_surface_vectorized(
    n_strikes: int = 20,
    n_maturities: int = 15,
    atm_vol: float = 0.20,
    skew: float = -0.10,
    term_slope: float = -0.02,
    smile_curvature: float = 0.05,
    seed: int = 42,
) -> tuple:
    """Generate synthetic IV surface with realistic features."""
    rng = np.random.default_rng(seed)

    moneyness = np.linspace(0.8, 1.2, n_strikes)
    maturities = np.linspace(0.05, 2.0, n_maturities)

    M, T = np.meshgrid(moneyness, maturities)

    # Build IV surface (all vectorized)
    iv = np.full_like(M, atm_vol)
    iv += skew * (1 - M)  # Skew
    iv += smile_curvature * (M - 1) ** 2  # Smile
    iv += term_slope * np.log(T + 0.1)  # Term structure
    iv += rng.normal(0, 0.005, iv.shape)  # Small noise
    iv = np.maximum(iv, 0.05)  # Ensure positive

    return moneyness, maturities, iv, M, T


def create_training_data(M, T, iv):
    """Create training data for ML models."""
    X = np.column_stack([M.ravel(), T.ravel()])
    y = iv.ravel()
    return X, y


def create_ml_model(model_name: str):
    """Create ML model pipeline."""
    if model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
    elif model_name == "XGBoost":
        if XGB_AVAILABLE:
            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=6, random_state=42
            )
    elif model_name == "MLP":
        model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    elif model_name == "SVR":
        model = SVR(kernel="rbf", C=10, gamma="scale")
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    return Pipeline([("scaler", StandardScaler()), ("model", model)])


# =============================================================================
# HEADER
# =============================================================================
page_header(
    "Volatility Surface", "ML-based implied volatility modeling and visualization"
)

# =============================================================================
# INPUT SECTION
# =============================================================================
st.markdown('<div class="input-section">', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 0.8])

with col1:
    st.markdown("**Surface Parameters**")
    atm_vol = (
        st.slider(
            "ATM Volatility (%)", min_value=5, max_value=60, value=20, key="vs_atm"
        )
        / 100
    )
    skew = (
        st.slider("Skew", min_value=-30, max_value=10, value=-10, key="vs_skew") / 100
    )

with col2:
    st.markdown("**Surface Shape**")
    smile = (
        st.slider("Smile Curvature", min_value=0, max_value=20, value=5, key="vs_smile")
        / 100
    )
    term_slope = (
        st.slider("Term Slope", min_value=-10, max_value=5, value=-2, key="vs_term")
        / 100
    )

with col3:
    st.markdown("**Grid Resolution**")
    n_strikes = st.select_slider(
        "Strike Points", options=[10, 15, 20, 30, 40], value=20, key="vs_strikes"
    )
    n_maturities = st.select_slider(
        "Maturity Points", options=[8, 10, 15, 20, 25], value=15, key="vs_mat"
    )

with col4:
    st.markdown("**ML Model**")
    model_name = st.selectbox(
        "Select Model",
        ["Random Forest", "XGBoost", "MLP", "SVR"],
        key="vs_model",
        help="ML model for interpolation/smoothing",
    )
    seed = st.number_input(
        "Random Seed", min_value=1, max_value=99999, value=42, key="vs_seed"
    )

with col5:
    st.markdown("**Actions**")
    st.write("")
    generate_btn = st.button(
        "🌊 Generate Data", type="primary", use_container_width=True
    )
    train_btn = st.button("🧠 Train & Fit", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
if "iv_data" not in st.session_state:
    st.session_state.iv_data = None
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None

# =============================================================================
# GENERATE DATA
# =============================================================================
if generate_btn:
    t_start = time.perf_counter()

    moneyness, maturities, iv_grid, M, T = generate_iv_surface_vectorized(
        n_strikes=n_strikes,
        n_maturities=n_maturities,
        atm_vol=atm_vol,
        skew=skew,
        term_slope=term_slope,
        smile_curvature=smile,
        seed=seed,
    )

    t_gen = (time.perf_counter() - t_start) * 1000

    st.session_state.iv_data = {
        "moneyness": moneyness,
        "maturities": maturities,
        "iv_grid": iv_grid,
        "M": M,
        "T": T,
    }

    st.success(
        f"✅ Generated {n_strikes * n_maturities} data points in {format_time_ms(t_gen)}"
    )

# =============================================================================
# TRAIN MODEL
# =============================================================================
if train_btn:
    if st.session_state.iv_data is None:
        st.warning("Please generate data first!")
    elif not SKLEARN_AVAILABLE:
        st.error("scikit-learn not available")
    else:
        data = st.session_state.iv_data

        X, y = create_training_data(data["M"], data["T"], data["iv_grid"])

        t_start = time.perf_counter()

        model = create_ml_model(model_name)
        model.fit(X, y)

        t_train = (time.perf_counter() - t_start) * 1000

        # Evaluate
        y_pred = model.predict(X)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

        st.session_state.trained_model = {
            "model": model,
            "name": model_name,
            "rmse": rmse,
            "r2": r2,
            "train_time": t_train,
        }

        st.success(
            f"✅ {model_name} trained in {format_time_ms(t_train)} | R²: {r2:.4f} | RMSE: {rmse:.4f}"
        )

# =============================================================================
# DISPLAY RESULTS
# =============================================================================
if st.session_state.iv_data is not None:
    data = st.session_state.iv_data
    iv_grid = data["iv_grid"]
    moneyness = data["moneyness"]
    maturities = data["maturities"]
    M, T = data["M"], data["T"]

    section_divider()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Min IV</div>
            <div class="metric-value">{iv_grid.min()*100:.1f}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Max IV</div>
            <div class="metric-value">{iv_grid.max()*100:.1f}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Grid Points</div>
            <div class="metric-value">{len(moneyness) * len(maturities)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        if st.session_state.trained_model:
            tm = st.session_state.trained_model
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">{tm['name']} R²</div>
                <div class="metric-value">{tm['r2']:.4f}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div class="metric-card">
                <div class="metric-label">Model</div>
                <div class="metric-value">Not trained</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    section_divider()

    # Visualizations
    tab1, tab2, tab3 = st.tabs(["🌊 3D Surface", "🗺️ Heatmap", "📈 Slices"])

    with tab1:
        # Use ML predictions if model trained
        if st.session_state.trained_model:
            model = st.session_state.trained_model["model"]
            M_fine, T_fine = np.meshgrid(
                np.linspace(moneyness.min(), moneyness.max(), 40),
                np.linspace(maturities.min(), maturities.max(), 40),
            )
            X_fine = np.column_stack([M_fine.ravel(), T_fine.ravel()])
            iv_fine = model.predict(X_fine).reshape(M_fine.shape)
            title = f"IV Surface ({st.session_state.trained_model['name']} Fitted)"
        else:
            M_fine, T_fine = M, T
            iv_fine = iv_grid
            title = "IV Surface (Raw Data)"

        fig = go.Figure(
            data=[
                go.Surface(
                    x=M_fine,
                    y=T_fine,
                    z=iv_fine * 100,
                    colorscale="Viridis",
                    opacity=0.95,
                    contours=dict(
                        z=dict(show=True, usecolormap=True, highlightcolor="white")
                    ),
                )
            ]
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color="#f8fafc")),
            scene=dict(
                xaxis_title="Moneyness (K/S)",
                yaxis_title="Time to Expiry (years)",
                zaxis_title="Implied Volatility (%)",
                bgcolor="rgba(15, 23, 42, 0.9)",
            ),
            paper_bgcolor="rgba(30, 41, 59, 0.8)",
            font=dict(color="#cbd5e1"),
            height=500,
            margin=dict(l=0, r=0, t=50, b=0),
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(
                data=go.Heatmap(
                    x=moneyness,
                    y=maturities,
                    z=iv_grid * 100,
                    colorscale="Viridis",
                    colorbar=dict(title="IV (%)"),
                )
            )
            fig.update_layout(**get_chart_layout("Raw Data", 350))
            fig.update_xaxes(title_text="Moneyness")
            fig.update_yaxes(title_text="Time (years)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if st.session_state.trained_model:
                fig = go.Figure(
                    data=go.Heatmap(
                        x=M_fine[0],
                        y=T_fine[:, 0],
                        z=iv_fine * 100,
                        colorscale="Viridis",
                        colorbar=dict(title="IV (%)"),
                    )
                )
                fig.update_layout(**get_chart_layout("ML Fitted", 350))
                fig.update_xaxes(title_text="Moneyness")
                fig.update_yaxes(title_text="Time (years)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Train a model to see fitted surface")

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            colors = ["#60a5fa", "#a78bfa", "#34d399", "#f97316", "#ef4444"]

            for idx in [
                0,
                len(maturities) // 4,
                len(maturities) // 2,
                3 * len(maturities) // 4,
                -1,
            ]:
                fig.add_trace(
                    go.Scatter(
                        x=moneyness,
                        y=iv_grid[idx] * 100,
                        mode="lines",
                        line=dict(
                            width=2.5,
                            color=colors[min(idx, len(colors) - 1) if idx >= 0 else -1],
                        ),
                        name=f"T = {maturities[idx]:.2f}y",
                    )
                )

            fig.update_layout(**get_chart_layout("Volatility Smile", 350))
            fig.update_xaxes(title_text="Moneyness")
            fig.update_yaxes(title_text="IV (%)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()

            for idx in [
                0,
                len(moneyness) // 4,
                len(moneyness) // 2,
                3 * len(moneyness) // 4,
                -1,
            ]:
                fig.add_trace(
                    go.Scatter(
                        x=maturities,
                        y=iv_grid[:, idx] * 100,
                        mode="lines",
                        line=dict(
                            width=2.5,
                            color=colors[min(idx, len(colors) - 1) if idx >= 0 else -1],
                        ),
                        name=f"K/S = {moneyness[idx]:.2f}",
                    )
                )

            fig.update_layout(**get_chart_layout("Term Structure", 350))
            fig.update_xaxes(title_text="Time (years)")
            fig.update_yaxes(title_text="IV (%)")
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info(
        "👆 Click **Generate Data** to create a synthetic IV surface, then **Train & Fit** to model it with ML."
    )

    st.markdown(
        """
    <div class="metric-card">
        <h3 style="color: #60a5fa; margin-bottom: 1rem;">Available ML Models</h3>
        <ul style="color: #cbd5e1; line-height: 2;">
            <li><strong>Random Forest:</strong> Ensemble of decision trees, robust</li>
            <li><strong>XGBoost:</strong> Gradient boosting, high accuracy</li>
            <li><strong>MLP:</strong> Neural network, captures complex patterns</li>
            <li><strong>SVR:</strong> Support Vector Regression, good for smooth surfaces</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

# =============================================================================
# BENCHMARK COMPARISON SECTION
# =============================================================================
section_divider()
st.markdown("## 📊 Model Benchmark Comparison")

if BENCHMARK_AVAILABLE:
    with st.expander("⚙️ Benchmark Settings", expanded=False):
        bench_col1, bench_col2, bench_col3 = st.columns(3)
        
        with bench_col1:
            available_models = ["svi", "sabr", "mlp", "rf"]
            if PINN_AVAILABLE:
                available_models.append("pinn")
            selected_models = st.multiselect(
                "Select Models to Compare",
                available_models,
                default=["svi", "sabr", "mlp"],
                key="bench_models",
            )
        
        with bench_col2:
            n_trials = st.slider("Number of Trials", 1, 10, 3, key="bench_trials")
            test_size = st.slider("Test Size (%)", 10, 40, 20, key="bench_test") / 100
        
        with bench_col3:
            bench_strikes = st.slider("Strikes per maturity", 20, 50, 30, key="bench_strikes")
            bench_seed = st.number_input("Seed", 1, 99999, 42, key="bench_seed")
    
    run_benchmark_btn = st.button("🏃 Run Benchmark", type="primary", key="run_bench")
    
    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = None
    
    if run_benchmark_btn:
        if not selected_models:
            st.warning("Please select at least one model")
        else:
            with st.spinner("Running benchmark... This may take a minute"):
                # Generate benchmark data
                bench_data = generate_synthetic_surface(
                    n_strikes=bench_strikes,
                    maturities=[0.1, 0.25, 0.5, 1.0, 2.0],
                    seed=bench_seed,
                )
                
                # Run benchmark
                benchmark = VolSurfaceBenchmark(
                    models=selected_models,
                    verbose=False,
                )
                results = benchmark.run(bench_data, n_trials=n_trials, test_size=test_size)
                st.session_state.benchmark_results = results
                
            st.success(f"✅ Benchmark complete! Tested {len(selected_models)} models over {n_trials} trials")
    
    if st.session_state.benchmark_results is not None:
        results = st.session_state.benchmark_results
        results_df = results.to_dataframe()
        
        # Display results table
        st.markdown("### Results Summary")
        
        # Format results for display
        display_df = results_df[["RMSE", "MAE", "MAPE (%)", "Calibration (ms)", "Arb-Free (%)"]].copy()
        display_df = display_df.round(4)
        
        st.dataframe(
            display_df.style.highlight_min(
                subset=["RMSE", "MAE", "MAPE (%)"], 
                color="#22c55e20"
            ).highlight_max(
                subset=["Arb-Free (%)"],
                color="#22c55e20"
            ),
            use_container_width=True,
        )
        
        # Best model callout
        best = results.best_model("RMSE")
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid #22c55e;">
                <div class="metric-label">Best Model (by RMSE)</div>
                <div class="metric-value" style="color: #22c55e;">{best}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Visualization
        bench_col1, bench_col2 = st.columns(2)
        
        with bench_col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=display_df.index.tolist(),
                y=display_df["RMSE"].values,
                marker_color="#60a5fa",
                text=display_df["RMSE"].round(4).values,
                textposition="auto",
            ))
            fig.update_layout(**get_chart_layout("RMSE by Model", 300))
            fig.update_yaxes(title_text="RMSE")
            st.plotly_chart(fig, use_container_width=True)
        
        with bench_col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=display_df.index.tolist(),
                y=display_df["Calibration (ms)"].values,
                marker_color="#a78bfa",
                text=display_df["Calibration (ms)"].round(1).values,
                textposition="auto",
            ))
            fig.update_layout(**get_chart_layout("Calibration Time", 300))
            fig.update_yaxes(title_text="Time (ms)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Export results
        csv_data = results_df.to_csv()
        st.download_button(
            "📥 Download Results (CSV)",
            csv_data,
            "benchmark_results.csv",
            "text/csv",
            key="download_bench",
        )
else:
    st.info("Benchmark module not available. Check if `src/benchmarks` is properly installed.")

# =============================================================================
# DATA UPLOAD SECTION
# =============================================================================
section_divider()
st.markdown("## 📁 Data Upload")

if BENCHMARK_AVAILABLE:
    upload_col1, upload_col2 = st.columns([2, 1])
    
    with upload_col1:
        data_source = st.radio(
            "Select Data Source",
            ["Synthetic (Built-in)", "📡 yfinance (Live)", "Upload CSV", "Upload Parquet"],
            horizontal=True,
            key="data_source",
        )
    
    with upload_col2:
        if data_source == "Synthetic (Built-in)":
            synth_spot = st.number_input("Spot Price", 50.0, 500.0, 100.0, key="synth_spot")
        elif data_source == "📡 yfinance (Live)":
            yf_ticker = st.text_input("Ticker", "AMD", key="yf_ticker")
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload Option Chain CSV",
            type=["csv"],
            help="CSV with columns: strike, T (or expiry), implied_vol (or IV)",
            key="csv_upload",
        )
        
        if uploaded_file is not None:
            try:
                import io
                df = pd.read_csv(io.StringIO(uploaded_file.read().decode("utf-8")))
                dataset = OptionChainDataset(data=df, underlying_price=100.0)
                st.session_state.uploaded_data = dataset
                st.success(f"✅ Loaded {dataset.n_options} options from CSV")
                
                # Preview
                st.dataframe(dataset.data.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
    
    elif data_source == "Upload Parquet":
        uploaded_file = st.file_uploader(
            "Upload Option Chain Parquet",
            type=["parquet"],
            key="parquet_upload",
        )
        
        if uploaded_file is not None:
            try:
                import io
                df = pd.read_parquet(io.BytesIO(uploaded_file.read()))
                dataset = OptionChainDataset(data=df, underlying_price=100.0)
                st.session_state.uploaded_data = dataset
                st.success(f"✅ Loaded {dataset.n_options} options from Parquet")
                
                st.dataframe(dataset.data.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Error loading Parquet: {e}")
    
    elif data_source == "📡 yfinance (Live)":
        st.info("⚠️ yfinance has rate limits. Uses caching + exponential backoff to avoid bans.")
        
        yf_expiries = st.slider("Number of expiries", 1, 5, 3, key="yf_expiries")
        
        if st.button(f"📡 Fetch {yf_ticker} Options", type="primary", key="fetch_yf"):
            with st.spinner(f"Fetching {yf_ticker} options (rate limited)..."):
                try:
                    dataset = OptionChainLoader.from_yfinance(
                        ticker=yf_ticker,
                        n_expiries=yf_expiries,
                        use_cache=True,
                    )
                    st.session_state.uploaded_data = dataset
                    st.success(f"✅ Loaded {dataset.n_options} options for {yf_ticker} (spot=${dataset.underlying_price:.2f})")
                    st.dataframe(dataset.data.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Error fetching {yf_ticker}: {e}")
                    st.info("Try using Synthetic data to avoid rate limits.")
    
    else:  # Synthetic
        if st.button("🔄 Generate Synthetic Data", key="gen_synth"):
            dataset = OptionChainLoader.from_synthetic(
                n_strikes=40,
                maturities=[0.1, 0.25, 0.5, 1.0, 2.0],
                spot=synth_spot,
                seed=42,
            )
            st.session_state.uploaded_data = dataset
            st.success(f"✅ Generated {dataset.n_options} synthetic options")
            
            st.dataframe(dataset.data.head(10), use_container_width=True)
    
    # Convert uploaded data to model input
    if "uploaded_data" in st.session_state and st.session_state.uploaded_data is not None:
        if st.button("🔧 Convert to Model Input", key="convert_data"):
            try:
                model_input = st.session_state.uploaded_data.to_model_input()
                st.session_state.model_input_data = model_input
                st.success(f"✅ Converted {len(model_input)} data points to model input format")
                st.dataframe(model_input.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Error converting data: {e}")

else:
    st.info("Data loading module not available. Check if `src/data` is properly installed.")

# =============================================================================
# PINN MODEL SECTION
# =============================================================================
if PINN_AVAILABLE:
    section_divider()
    st.markdown("## 🧠 Physics-Informed Neural Network (PINN)")
    
    # Show GPU status
    gpu_color = "#22c55e" if "AMD" in DEVICE_NAME or "NVIDIA" in DEVICE_NAME else "#f97316"
    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 4px solid {gpu_color};">
            <div class="metric-label">Compute Device</div>
            <div class="metric-value" style="color: {gpu_color};">{DEVICE_NAME}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div class="metric-card">
            <p style="color: #cbd5e1;">
                PINN models combine neural network flexibility with <strong>arbitrage-free constraints</strong> 
                built into the loss function. This guarantees no calendar or butterfly arbitrage.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    pinn_col1, pinn_col2 = st.columns(2)
    
    with pinn_col1:
        constraint_strength = st.select_slider(
            "Constraint Strength",
            options=["weak", "medium", "strong"],
            value="medium",
            key="pinn_strength",
            help="Stronger = more arbitrage-free, weaker = better fit",
        )
    
    with pinn_col2:
        pinn_epochs = st.slider("Training Epochs", 100, 1000, 300, 50, key="pinn_epochs")
    
    if "model_input_data" in st.session_state and st.session_state.model_input_data is not None:
        if st.button("🚀 Train PINN Model", type="primary", key="train_pinn"):
            with st.spinner(f"Training PINN ({pinn_epochs} epochs)..."):
                try:
                    pinn = create_pinn_model(
                        constraint_strength=constraint_strength,
                        epochs=pinn_epochs,
                    )
                    training_metrics = pinn.train(
                        st.session_state.model_input_data,
                        val_split=0.2,
                    )
                    st.session_state.pinn_model = pinn
                    st.session_state.pinn_metrics = training_metrics
                    
                    st.success("✅ PINN training complete!")
                    
                    # Display metrics
                    pinn_m1, pinn_m2, pinn_m3 = st.columns(3)
                    with pinn_m1:
                        st.metric("Final MSE", f"{training_metrics['final_mse']:.6f}")
                    with pinn_m2:
                        st.metric("Calendar Penalty", f"{training_metrics['final_calendar_penalty']:.6f}")
                    with pinn_m3:
                        st.metric("Butterfly Penalty", f"{training_metrics['final_butterfly_penalty']:.6f}")
                    
                    # Check arbitrage
                    arb_metrics = pinn.check_arbitrage(st.session_state.model_input_data)
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-left: 4px solid #22c55e;">
                            <div class="metric-label">Arbitrage-Free</div>
                            <div class="metric-value" style="color: #22c55e;">{arb_metrics.arbitrage_free_pct:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"PINN training failed: {e}")
    else:
        st.info("Upload or generate data first, then convert to model input to train PINN.")

