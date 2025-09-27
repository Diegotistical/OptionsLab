# streamlit_app / pages / volatility_surface.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
import traceback
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
import json
import hashlib
import math

# =============================
# Enhanced Import System
# =============================

# Optional external: joblib for grid search parallelism (not required)
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

# Optional for BS cdf
try:
    from scipy.stats import norm
except Exception:
    # minimal fallback for norm.cdf
    class _NormFallback:
        @staticmethod
        def cdf(x):
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    norm = _NormFallback()

# =============================
# Logging
# =============================
logger = logging.getLogger("vol_surface_prod")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)

# =============================
# Add src to sys.path if present - Enhanced path detection
# =============================
def setup_import_paths():
    """Enhanced path setup to find src directory"""
    current_file = Path(__file__).resolve()
    
    # Try multiple possible locations for src
    possible_paths = [
        current_file.parents[1] / "src",  # ../src
        current_file.parent / "src",      # ./src
        Path.cwd() / "src",               # cwd/src
        Path.cwd().parent / "src",        # ../src from cwd
    ]
    
    for path in possible_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
            logger.info(f"Added {path} to sys.path")
            return str(path)
    
    logger.warning("Could not find src directory in standard locations")
    return None

setup_import_paths()

# =============================
# Enhanced Safe import helper
# =============================
def try_import(module_path: str, attr: Optional[str] = None, fallback_module: Optional[str] = None) -> Tuple[Optional[Any], Optional[str]]:
    """
    Enhanced import helper that tries multiple import strategies
    """
    import_strategies = []
    
    # Strategy 1: Direct import
    if attr:
        import_strategies.append(f"from {module_path} import {attr}")
    else:
        import_strategies.append(f"import {module_path}")
    
    # Strategy 2: Try with fallback path
    if fallback_module:
        fallback_path = f"{fallback_module}.{module_path.split('.')[-1]}"
        if attr:
            import_strategies.append(f"from {fallback_path} import {attr}")
        else:
            import_strategies.append(f"import {fallback_path}")
    
    for strategy in import_strategies:
        try:
            if strategy.startswith("from"):
                # from module import attr
                parts = strategy.split()
                module = __import__(parts[1], fromlist=[parts[3]])
                result = getattr(module, parts[3])
                return result, None
            else:
                # import module
                module_name = strategy.split()[1]
                module = __import__(module_name, fromlist=['*'])
                return module, None
        except ImportError:
            continue
        except Exception as e:
            tb = traceback.format_exc()
            logger.debug(f"Import failed with strategy {strategy}: {e}")
            continue
    
    return None, f"All import strategies failed for {module_path}.{attr if attr else ''}"

# =============================
# Try to import user modules with enhanced fallbacks
# =============================

# Try multiple import strategies for each module
VolatilitySurfaceGenerator, _ = try_import("volatility_surface.surface_generator", "VolatilitySurfaceGenerator", "src.volatility_surface")
MLPModel, _ = try_import("volatility_surface.models.mlp_model", "MLPModel", "src.volatility_surface.models")
RandomForestVolatilityModel, _ = try_import("volatility_surface.models.random_forest", "RandomForestVolatilityModel", "src.volatility_surface.models")
SVRModel, _ = try_import("volatility_surface.models.svr_model", "SVRModel", "src.volatility_surface.models")
XGBoostModel, _ = try_import("volatility_surface.models.xgboost_model", "XGBoostModel", "src.volatility_surface.models")
feature_engineering_module, _ = try_import("volatility_surface.utils.feature_engineering", None, "src.volatility_surface.utils")
arbitrage_checks_module, _ = try_import("volatility_surface.utils.arbitrage_checks", None, "src.volatility_surface.utils")
arbitrage_enforcement_module, _ = try_import("volatility_surface.utils.arbitrage_enforcement", None, "src.volatility_surface.utils")
grid_search_module, _ = try_import("volatility_surface.utils.grid_search", None, "src.volatility_surface.utils")

# Also try direct module imports as fallback
if VolatilitySurfaceGenerator is None:
    VolatilitySurfaceGenerator, _ = try_import("surface_generator", "VolatilitySurfaceGenerator", "src")

MODEL_CLASS_MAP = {
    "MLP Neural Network": MLPModel,
    "Random Forest": RandomForestVolatilityModel,
    "SVR": SVRModel,
    "XGBoost": XGBoostModel
}
AVAILABLE_MODELS = [name for name, cls in MODEL_CLASS_MAP.items() if cls is not None]
if not AVAILABLE_MODELS:
    AVAILABLE_MODELS = ["DummyModel"]
    logger.info("No custom models found, using DummyModel fallback")

# =============================
# DummyModel fallback
# =============================
class DummyModel:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.feature_names_in_ = ["moneyness","log_moneyness","time_to_maturity","ttm_squared","risk_free_rate","historical_volatility","volatility_skew"]
        self.name = "DummyModel"
        
    def train(self, df: pd.DataFrame, val_split: float = 0.2) -> Dict[str, float]:
        logger.info("DummyModel.train called (no-op)")
        return {"train_rmse": float("nan"), "val_rmse": float("nan"), "val_r2": float("nan")}
    
    def predict_volatility(self, df: pd.DataFrame) -> np.ndarray:
        if "implied_volatility" in df.columns:
            return df["implied_volatility"].to_numpy()
        # simple smile function based on moneyness & ttm (consistent)
        m = df["moneyness"].to_numpy()
        t = df["time_to_maturity"].to_numpy()
        base = 0.2 + 0.05 * np.sin(2 * np.pi * m) * np.exp(-t)
        smile = 0.03 * (m - 1.0) ** 2
        return np.clip(base + smile, 0.03, 0.6)

# Model factory
def create_model_instance(name: str, **kwargs):
    cls = MODEL_CLASS_MAP.get(name)
    if cls is None:
        logger.info(f"Using DummyModel for {name}")
        return DummyModel(**kwargs)
    try:
        return cls(**kwargs)
    except Exception:
        logger.exception(f"Failed to instantiate model {name}; using DummyModel")
        return DummyModel(**kwargs)

# =============================
# UI Configuration and Styling
# =============================

def setup_dark_theme():
    """Configure dark theme CSS"""
    st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background: linear-gradient(135deg, #0c0d13 0%, #1a1d29 100%);
        max-width: 100% !important;
        padding: 0 !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100% !important;
    }
    .section {
        background: rgba(30, 33, 48, 0.9);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #ff4b4b;
    }
    .metric-card {
        background: rgba(40, 44, 62, 0.8);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #2a2f45;
    }
    .model-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        font-size: 0.9em;
    }
    .status-available {
        background: rgba(0, 200, 83, 0.2);
        border-left: 3px solid #00c853;
    }
    .status-unavailable {
        background: rgba(255, 75, 75, 0.2);
        border-left: 3px solid #ff4b4b;
    }
    .stButton>button {
        background: linear-gradient(45deg, #ff6b6b, #ff4b4b);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #ff4b4b, #ff6b6b);
        color: white;
    }
    .stSelectbox, .stNumberInput, .stSlider {
        background: rgba(40, 44, 62, 0.8);
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================
# Rest of the utility functions (unchanged)
# =============================

def build_prediction_grid(m_start=0.7, m_end=1.3, m_steps=40, t_start=0.05, t_end=2.0, t_steps=40) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    m = np.linspace(m_start, m_end, m_steps)
    t = np.linspace(t_start, t_end, t_steps)
    M, T = np.meshgrid(m, t, indexing='xy')
    flat_m = M.ravel()
    flat_t = T.ravel()
    grid_df = pd.DataFrame({
        "moneyness": flat_m,
        "log_moneyness": np.log(np.clip(flat_m, 1e-12, None)),
        "time_to_maturity": flat_t,
        "ttm_squared": flat_t ** 2,
        "risk_free_rate": np.full(flat_m.shape, 0.03),
        "historical_volatility": np.full(flat_m.shape, 0.2),
        "volatility_skew": np.zeros(flat_m.shape)
    })
    return M, T, grid_df

def safe_model_predict_volatility(model: Any, df: pd.DataFrame) -> np.ndarray:
    try:
        if hasattr(model, "predict_volatility"):
            out = model.predict_volatility(df)
        elif hasattr(model, "predict"):
            out = model.predict(df)
        else:
            out = model(df) if callable(model) else np.full(len(df), 0.2)
        out = np.asarray(out).astype(float).ravel()
        if out.shape[0] != len(df):
            if out.size == 1:
                return np.full(len(df), float(out))
            out = np.resize(out, len(df))
        return out
    except Exception:
        logger.exception("Model prediction failed; returning fallback constant surface")
        return np.full(len(df), 0.2)

def cache_key(model_name: str, params: Dict[str, Any], m_steps: int, t_steps: int, extra: Optional[Dict] = None) -> str:
    payload = {"model": model_name, "params": params, "m": m_steps, "t": t_steps, "extra": extra or {}}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

if 'pred_cache' not in st.session_state:
    st.session_state['pred_cache'] = {}

@st.cache_data(show_spinner=False)
def generate_fallback_data(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    spots = rng.uniform(90, 110, n_samples)
    strikes = rng.uniform(80, 120, n_samples)
    ttms = rng.uniform(0.1, 2.0, n_samples)
    moneyness = strikes / spots
    ivs = 0.2 + 0.05 * np.sin(2 * np.pi * moneyness) * np.exp(-ttms) + 0.03 * (moneyness - 1)**2
    ivs += rng.normal(0, 0.07, n_samples)
    ivs = np.clip(ivs, 0.03, 0.6)
    df = pd.DataFrame({
        "underlying_price": spots,
        "strike_price": strikes,
        "time_to_maturity": ttms,
        "risk_free_rate": rng.uniform(0.01, 0.05, n_samples),
        "historical_volatility": rng.uniform(0.12, 0.28, n_samples),
        "implied_volatility": ivs
    })
    df["moneyness"] = df["underlying_price"] / df["strike_price"]
    df["log_moneyness"] = np.log(np.clip(df["moneyness"], 1e-12, None))
    df["ttm_squared"] = df["time_to_maturity"] ** 2
    df["volatility_skew"] = df["implied_volatility"] - df["historical_volatility"]
    return df

@st.cache_data(show_spinner=False)
def generate_surface_data_via_generator(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    if VolatilitySurfaceGenerator is None:
        return generate_fallback_data(n_samples, seed)
    try:
        rng = np.random.default_rng(seed)
        base_strikes = np.linspace(80, 120, 50)
        base_maturities = np.linspace(0.1, 2.0, 20)
        S, T = np.meshgrid(base_strikes, base_maturities, indexing='xy')
        base_ivs = 0.2 + 0.05 * np.sin(2 * np.pi * (S / np.mean(base_strikes))) * np.exp(-T)
        generator = VolatilitySurfaceGenerator(base_strikes, base_maturities, base_ivs,
                                               strike_points=50, maturity_points=20, interp_method='cubic')
        spots = rng.uniform(90, 110, n_samples)
        strikes = rng.uniform(80, 120, n_samples)
        ttms = rng.uniform(0.1, 2.0, n_samples)
        ivs = generator.get_surface_batch(strikes, ttms)
        df = pd.DataFrame({
            "underlying_price": spots,
            "strike_price": strikes,
            "time_to_maturity": ttms,
            "risk_free_rate": rng.uniform(0.01, 0.05, n_samples),
            "historical_volatility": rng.uniform(0.12, 0.28, n_samples),
            "implied_volatility": ivs
        })
        if feature_engineering_module and hasattr(feature_engineering_module, "engineer_features"):
            try:
                df = feature_engineering_module.engineer_features(df)
            except Exception:
                logger.exception("feature_engineering failed")
        return df
    except Exception:
        logger.exception("Surface generator error; falling back")
        return generate_fallback_data(n_samples, seed)

def black_scholes_price(S, K, T, r, sigma, option_type="call", q=0.0):
    try:
        T = max(T, 1e-12)
        sigma = max(sigma, 1e-12)
        d1 = (math.log(max(S, 1e-12) / max(K, 1e-12)) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if option_type == "call":
            return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    except Exception:
        logger.exception("Black-Scholes price failure")
        return 0.0

def bs_price_vectorized(S_arr, K_arr, T_arr, r, sigma_arr, option_type="call", q=0.0):
    out = np.zeros_like(S_arr, dtype=float)
    for i in range(len(out)):
        out[i] = black_scholes_price(float(S_arr[i]), float(K_arr[i]), float(T_arr[i]), float(r), float(sigma_arr[i]), option_type, float(q))
    return out

def compute_greeks_from_iv_grid(M, T, Z_pred, option_type="call", spot_assumption=100.0, r=0.03, q=0.0, h_frac=1e-3):
    try:
        shape = Z_pred.shape
        flat_m = M.ravel()
        flat_t = T.ravel()
        flat_sigma = Z_pred.ravel()
        S0 = spot_assumption
        K = flat_m * S0
        Tvec = flat_t
        h = max(1e-4, h_frac * S0)
        p0 = bs_price_vectorized(np.full_like(K, S0), K, Tvec, r, flat_sigma, option_type, q)
        p_up = bs_price_vectorized(np.full_like(K, S0 + h), K, Tvec, r, flat_sigma, option_type, q)
        p_down = bs_price_vectorized(np.full_like(K, S0 - h), K, Tvec, r, flat_sigma, option_type, q)
        delta = (p_up - p_down) / (2 * h)
        gamma = (p_up - 2 * p0 + p_down) / (h * h)
        return delta.reshape(shape), gamma.reshape(shape)
    except Exception:
        logger.exception("Greeks computation failed")
        return np.full_like(Z_pred, np.nan), np.full_like(Z_pred, np.nan)

def run_arbitrage_checks_vectorized(vol2d: np.ndarray, strike_grid: np.ndarray, ttm_grid: np.ndarray) -> Dict[str, Any]:
    results = {}
    try:
        if arbitrage_checks_module and hasattr(arbitrage_checks_module, "check_arbitrage_violations"):
            try:
                res = arbitrage_checks_module.check_arbitrage_violations(vol2d, strike_grid, strike_grid, ttm_grid)
                results["basic_checks"] = res
            except Exception:
                logger.exception("arbitrage_checks call failed")
                results["basic_checks_error"] = traceback.format_exc()
        else:
            results["basic_checks"] = {"note": "arbitrage_checks missing, skipped"}
    except Exception:
        results["basic_checks_error"] = traceback.format_exc()
    try:
        if arbitrage_enforcement_module and hasattr(arbitrage_enforcement_module, "detect_arbitrage_violations"):
            try:
                res = arbitrage_enforcement_module.detect_arbitrage_violations(vol2d)
                results["enforcement_checks"] = res
            except Exception:
                logger.exception("arbitrage_enforcement call failed")
                results["enforcement_checks_error"] = traceback.format_exc()
        else:
            results["enforcement_checks"] = {"note": "arbitrage_enforcement missing, skipped"}
    except Exception:
        results["enforcement_checks_error"] = traceback.format_exc()
    return results

def export_plotly_fig_png(fig: go.Figure, filename: str = "figure.png") -> Optional[bytes]:
    try:
        img_bytes = fig.to_image(format="png", engine="kaleido")
        return img_bytes
    except Exception:
        logger.exception("PNG export failed (kaleido missing?)")
        return None

def export_plotly_fig_html(fig: go.Figure, filename: str = "figure.html") -> str:
    try:
        return fig.to_html(full_html=True, include_plotlyjs='cdn')
    except Exception:
        logger.exception("HTML export failed")
        return fig.to_html(full_html=True, include_plotlyjs='cdn')

def fig_surface(M, T, Z, title="Volatility Surface"):
    fig = go.Figure(go.Surface(x=M, y=T, z=Z, colorscale="Viridis", cmin=np.nanmin(Z), cmax=np.nanmax(Z)))
    fig.update_layout(title=title, template="plotly_dark", scene=dict(xaxis_title="Moneyness", yaxis_title="TTM", zaxis_title="Implied Vol"), height=720, margin=dict(t=50))
    return fig

def fig_heatmap(M, T, Z, title="Heatmap"):
    fig = go.Figure(go.Heatmap(z=Z, x=M[0,:], y=T[:,0], colorscale="Viridis"))
    fig.update_layout(title=title, template="plotly_dark", xaxis_title="Moneyness", yaxis_title="TTM", height=600)
    return fig

def fig_contour_slices(M, T, Z, slice_m=1.0, slice_t=1.0):
    t_idx = int(np.argmin(np.abs(T[:,0] - slice_t)))
    m_idx = int(np.argmin(np.abs(M[0,:] - slice_m)))
    fig = make_subplots(rows=2, cols=2, specs=[[{"type":"contour","rowspan":2}, {"type":"scatter"}],[None, {"type":"scatter"}]],
                        column_widths=[0.62, 0.38], subplot_titles=("Contour (IV)", f"Slice @ TTM={T[t_idx,0]:.3f}", f"Slice @ moneyness={M[0,m_idx]:.3f}"))
    fig.add_trace(go.Contour(z=Z, x=M[0,:], y=T[:,0], colorscale="Viridis", contours=dict(showlabels=True)), row=1, col=1)
    fig.add_trace(go.Scatter(x=M[t_idx,:], y=Z[t_idx,:], mode="lines+markers"), row=1, col=2)
    fig.add_trace(go.Scatter(x=T[:,0], y=Z[:,m_idx], mode="lines+markers"), row=2, col=2)
    fig.update_layout(template="plotly_dark", height=700)
    return fig

def fig_residuals_scatter(pred, true):
    df_p = pd.DataFrame({"pred": pred.ravel(), "true": true.ravel()})
    fig = px.scatter(df_p, x="true", y="pred", marginal_x="histogram", marginal_y="histogram", trendline="ols")
    fig.update_layout(template="plotly_dark", title="Predicted vs True IV")
    return fig

def build_animation_frames(M, T, Z, t_axis_steps=30, title="Animated Surface"):
    frames = []
    t_len = min(Z.shape[0], t_axis_steps)
    idxs = np.linspace(0, Z.shape[0] - 1, t_len).astype(int)
    base = go.Surface(x=M, y=T, z=Z, colorscale="Viridis", showscale=False)
    fig = go.Figure(data=[base])
    for i in idxs:
        frame_z = np.copy(Z)
        frames.append(go.Frame(data=[go.Surface(x=M, y=T, z=Z)], name=str(i)))
    steps = []
    for i, f in enumerate(frames):
        step = {"args": [[f.name], {"frame": {"duration": 200, "redraw": True}, "mode": "immediate"}],
                "label": str(i), "method": "animate"}
        steps.append(step)
    sliders = [{"active": 0, "pad": {"t": 50}, "steps": steps}]
    fig.frames = frames
    fig.update_layout(template="plotly_dark", title=title, height=720, sliders=sliders,
                      updatemenus=[{"type": "buttons", "buttons": [{"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True}]}], "pad": {"r": 10, "t": 10}}])
    return fig

def compute_and_plot_greeks(M, T, Z_pred, option_type="call", spot_assumption=100.0, r=0.03, q=0.0, h_frac=1e-3):
    delta_grid, gamma_grid = compute_greeks_from_iv_grid(M, T, Z_pred, option_type=option_type, spot_assumption=spot_assumption, r=r, q=q, h_frac=h_frac)
    delta_fig = go.Figure(go.Surface(x=M, y=T, z=delta_grid, colorscale="RdBu"))
    delta_fig.update_layout(title="Delta Surface", template="plotly_dark", scene=dict(xaxis_title="Moneyness", yaxis_title="TTM", zaxis_title="Delta"), height=640)
    gamma_fig = go.Figure(go.Surface(x=M, y=T, z=gamma_grid, colorscale="RdBu"))
    gamma_fig.update_layout(title="Gamma Surface", template="plotly_dark", scene=dict(xaxis_title="Moneyness", yaxis_title="TTM", zaxis_title="Gamma"), height=640)
    return delta_grid, gamma_grid, delta_fig, gamma_fig

def synthetic_true_surface(M, T):
    base = 0.2 + 0.05 * np.sin(2 * np.pi * M) * np.exp(-T)
    smile = 0.03 * (M - 1.0) ** 2
    return np.clip(base + smile + 0.02 * np.exp(-T), 0.03, 0.6)

# =============================
# Enhanced Streamlit UI
# =============================

def main():
    st.set_page_config(
        page_title="Volatility Surface Explorer", 
        layout="wide", 
        page_icon="📊",
        initial_sidebar_state="collapsed"
    )
    
    setup_dark_theme()
    
    # Header Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">📊 Volatility Surface Explorer</h1>
        <p style="color: white; opacity: 0.9; font-size: 1.1rem;">Production-ready visualization with Greeks, animations, and arbitrage checks</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        use_generator = st.checkbox("Use Surface Generator", value=VolatilitySurfaceGenerator is not None, 
                                   help="Use advanced surface generator when available")
        n_samples = st.slider("Dataset Size", 200, 5000, 1500, step=100)
        
    with col2:
        viz_model = st.selectbox("Model Type", AVAILABLE_MODELS, index=0)
        m_steps = st.slider("Moneyness Grid", 12, 100, 40)
        
    with col3:
        option_type = st.selectbox("Option Type", ["call", "put"], index=0)
        t_steps = st.slider("TTM Grid", 6, 60, 30)
        
    with col4:
        spot_assumption = st.number_input("Spot Price", min_value=1.0, value=100.0, step=1.0)
        r = st.number_input("Risk-free Rate", min_value=0.0, value=0.03, step=0.005)
        q = st.number_input("Dividend Yield", min_value=0.0, value=0.0, step=0.001)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Status and Controls
    col5, col6 = st.columns([2, 1])
    
    with col5:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### 🔧 Model Status & Training")
        
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            st.markdown("**Module Availability:**")
            modules = [
                ("VolatilitySurfaceGenerator", VolatilitySurfaceGenerator),
                ("MLP Model", MLPModel),
                ("Random Forest", RandomForestVolatilityModel),
                ("SVR Model", SVRModel),
                ("XGBoost", XGBoostModel)
            ]
            
            for name, obj in modules:
                status_class = "status-available" if obj is not None else "status-unavailable"
                icon = "✅" if obj is not None else "❌"
                st.markdown(f'<div class="model-status {status_class}">{icon} {name}</div>', unsafe_allow_html=True)
        
        with status_col2:
            if st.button("🚀 Train Model", use_container_width=True):
                with st.spinner("Training model..."):
                    mdl = create_model_instance(viz_model)
                    try:
                        metrics = mdl.train(df, val_split=0.2) if hasattr(mdl, "train") else {}
                        st.session_state['last_trained'] = (viz_model, mdl, metrics)
                        st.success("Training completed successfully!")
                        if metrics:
                            st.json(metrics)
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
            
            if st.button("🔄 Refresh Predictions", use_container_width=True):
                st.session_state['pred_cache'] = {}
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col6:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### 📈 Visualization")
        
        vis_options = ["3D Surface", "Heatmap", "Contour + Slices", "Residuals & Scatter", "Animation (TTM)", "Greeks (Delta/Gamma)"]
        vis_choice = st.selectbox("Visualization Type", vis_options, index=0)
        
        if vis_choice == "Contour + Slices":
            slice_col1, slice_col2 = st.columns(2)
            with slice_col1:
                slice_m = st.slider("Slice Moneyness", 0.7, 1.3, 1.0, 0.01)
            with slice_col2:
                slice_t = st.slider("Slice TTM", 0.05, 2.0, 1.0, 0.05)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Generation
    if use_generator and VolatilitySurfaceGenerator:
        df = generate_surface_data_via_generator(n_samples)
    else:
        df = generate_fallback_data(n_samples)
    
    # Model Instance
    if 'last_trained' in st.session_state and st.session_state['last_trained'][0] == viz_model:
        model_instance = st.session_state['last_trained'][1]
    else:
        model_instance = create_model_instance(viz_model)
    
    # Build grid & predict
    M_grid, T_grid, grid_df = build_prediction_grid(0.7, 1.3, m_steps, 0.05, 2.0, t_steps)
    params = getattr(model_instance, "params", {}) if hasattr(model_instance, "params") else {}
    ck = cache_key(viz_model, params, m_steps, t_steps)
    pred_cache = st.session_state['pred_cache']
    
    if ck in pred_cache:
        preds = pred_cache[ck]
    else:
        with st.spinner("Computing predictions..."):
            preds = safe_model_predict_volatility(model_instance, grid_df)
            pred_cache[ck] = preds
    
    try:
        Z_pred = np.array(preds).reshape(M_grid.shape)
    except Exception:
        try:
            Z_pred = np.array(preds).reshape((M_grid.shape[1], M_grid.shape[0])).T
        except Exception:
            logger.exception("Prediction reshape failed")
            Z_pred = np.full(M_grid.shape, np.nan)
            st.error("Prediction reshape failed — model output ordering unexpected.")
    
    Z_true = synthetic_true_surface(M_grid, T_grid)
    resid = Z_pred - Z_true
    
    # Visualization Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📊 Visualization Results")
    
    if vis_choice == "3D Surface":
        fig = fig_surface(M_grid, T_grid, Z_pred, title=f"{viz_model} Predicted Surface")
        st.plotly_chart(fig, use_container_width=True)
        
    elif vis_choice == "Heatmap":
        fig = fig_heatmap(M_grid, T_grid, Z_pred, title=f"{viz_model} Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        
    elif vis_choice == "Contour + Slices":
        fig = fig_contour_slices(M_grid, T_grid, Z_pred, slice_m, slice_t)
        st.plotly_chart(fig, use_container_width=True)
        
    elif vis_choice == "Residuals & Scatter":
        fig_r = fig_residuals_scatter(Z_pred, Z_true)
        st.plotly_chart(fig_r, use_container_width=True)
        fig_res = fig_heatmap(M_grid, T_grid, resid, title="Residuals (Pred - True)")
        st.plotly_chart(fig_res, use_container_width=True)
        
    elif vis_choice == "Animation (TTM)":
        frames_fig = build_animation_frames(M_grid, T_grid, Z_pred, t_axis_steps=min(20, T_grid.shape[0]), 
                                          title=f"{viz_model} Animated Surface")
        st.plotly_chart(frames_fig, use_container_width=True)
        
    elif vis_choice == "Greeks (Delta/Gamma)":
        with st.spinner("Computing Greeks..."):
            delta_grid, gamma_grid, delta_fig, gamma_fig = compute_and_plot_greeks(
                M_grid, T_grid, Z_pred, option_type, spot_assumption, r, q)
        
        col7, col8 = st.columns(2)
        with col7:
            st.plotly_chart(delta_fig, use_container_width=True)
        with col8:
            st.plotly_chart(gamma_fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Metrics and Analysis Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📈 Performance Metrics")
    
    col9, col10, col11, col12 = st.columns(4)
    
    with col9:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("IV Min", f"{np.nanmin(Z_pred):.4f}")
        st.metric("IV Mean", f"{np.nanmean(Z_pred):.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col10:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("IV Max", f"{np.nanmax(Z_pred):.4f}")
        rmse = np.sqrt(np.nanmean((Z_pred - Z_true) ** 2))
        st.metric("RMSE", f"{rmse:.6f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col11:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Grid Size", f"{m_steps}×{t_steps}")
        st.metric("Data Points", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col12:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Model Info**")
        st.text(f"Type: {viz_model}")
        st.text(f"Params: {len(params)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Arbitrage Checks
    with st.expander("🔍 Arbitrage Analysis", expanded=False):
        with st.spinner("Running arbitrage checks..."):
            strike_grid = M_grid * spot_assumption
            ttm_grid = T_grid
            arb_res = run_arbitrage_checks_vectorized(Z_pred, strike_grid, ttm_grid)
            st.json(arb_res)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Volatility Surface Explorer • Built with Streamlit • Production-ready visualization system</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()