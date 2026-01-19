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
