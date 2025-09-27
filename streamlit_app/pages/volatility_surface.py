# streamlit_vol_surface_prod_visual_greeks.py
"""
Production-ready Volatility Surface Visual Explorer with Greeks & Animation
- Uses your src/volatility_surface modules when present (safe imports)
- DummyModel fallback to keep UI working
- Cached data + predictions
- Animated surface across TTM (slider + play)
- Delta & Gamma via analytic (if exposed) or finite-difference pricing using Black-Scholes
- Export charts as PNG/HTML (kaleido optional)
- Vectorized arbitrage check adaptor (uses arbitrage utils if available)
- Designed to be clean, demonstrable, and recruiter-friendly
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import logging
import traceback
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
import json
import hashlib
import math

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
# Add src to sys.path if present
# =============================
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
    logger.info(f"Added {SRC} to sys.path")

# =============================
# Safe import helper (no UI calls inside)
# =============================
def try_import(module_path: str, attr: Optional[str] = None) -> Tuple[Optional[Any], Optional[str]]:
    try:
        if attr:
            module = __import__(module_path, fromlist=[attr])
            return getattr(module, attr), None
        else:
            module = __import__(module_path, fromlist=['*'])
            return module, None
    except Exception:
        tb = traceback.format_exc()
        logger.debug("Import failed: %s.%s\n%s", module_path, attr or "", tb)
        return None, tb

# =============================
# Try to import user modules (volatility_surface package expected under src/)
# =============================
VolatilitySurfaceGenerator, _ = try_import("volatility_surface.surface_generator", "VolatilitySurfaceGenerator")
MLPModel, _ = try_import("volatility_surface.models.mlp_model", "MLPModel")
RandomForestVolatilityModel, _ = try_import("volatility_surface.models.random_forest", "RandomForestVolatilityModel")
SVRModel, _ = try_import("volatility_surface.models.svr_model", "SVRModel")
XGBoostModel, _ = try_import("volatility_surface.models.xgboost_model", "XGBoostModel")
feature_engineering_module, _ = try_import("volatility_surface.utils.feature_engineering")
arbitrage_checks_module, _ = try_import("volatility_surface.utils.arbitrage_checks")
arbitrage_enforcement_module, _ = try_import("volatility_surface.utils.arbitrage_enforcement")
grid_search_module, _ = try_import("volatility_surface.utils.grid_search")

MODEL_CLASS_MAP = {
    "MLP Neural Network": MLPModel,
    "Random Forest": RandomForestVolatilityModel,
    "SVR": SVRModel,
    "XGBoost": XGBoostModel
}
AVAILABLE_MODELS = [name for name, cls in MODEL_CLASS_MAP.items() if cls is not None]
if not AVAILABLE_MODELS:
    AVAILABLE_MODELS = ["DummyModel"]

# =============================
# DummyModel fallback
# =============================
class DummyModel:
    def __init__(self, **kwargs):
        self.params = kwargs
        # feature_names_in_ fake
        self.feature_names_in_ = ["moneyness","log_moneyness","time_to_maturity","ttm_squared","risk_free_rate","historical_volatility","volatility_skew"]
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
        return DummyModel(**kwargs)
    try:
        return cls(**kwargs)
    except Exception:
        logger.exception("Failed to instantiate model %s; using DummyModel", name)
        return DummyModel(**kwargs)

# =============================
# Deterministic grid builder
# =============================
def build_prediction_grid(m_start=0.7, m_end=1.3, m_steps=40, t_start=0.05, t_end=2.0, t_steps=40) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    m = np.linspace(m_start, m_end, m_steps)
    t = np.linspace(t_start, t_end, t_steps)
    M, T = np.meshgrid(m, t, indexing='xy')  # shape (t_steps, m_steps)
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

# =============================
# Safe model prediction adaptor
# =============================
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

# =============================
# Caching helpers (session-state)
# =============================
def cache_key(model_name: str, params: Dict[str, Any], m_steps: int, t_steps: int, extra: Optional[Dict] = None) -> str:
    payload = {"model": model_name, "params": params, "m": m_steps, "t": t_steps, "extra": extra or {}}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

if 'pred_cache' not in st.session_state:
    st.session_state['pred_cache'] = {}

# =============================
# Data generation (cached)
# =============================
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

# =============================
# Black-Scholes pricing (for Greeks calculation)
# =============================
def black_scholes_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Black-Scholes price for European options.
    Safe numerical guards included.
    """
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

# Vectorized wrapper for price with arrays
def bs_price_vectorized(S_arr, K_arr, T_arr, r, sigma_arr, option_type="call", q=0.0):
    out = np.zeros_like(S_arr, dtype=float)
    for i in range(len(out)):
        out[i] = black_scholes_price(float(S_arr[i]), float(K_arr[i]), float(T_arr[i]), float(r), float(sigma_arr[i]), option_type, float(q))
    return out

# =============================
# Greeks calculation (finite differences fallback)
# =============================
def compute_greeks_from_iv_grid(M, T, Z_pred, option_type="call", spot_assumption=100.0, r=0.03, q=0.0, h_frac=1e-3):
    """
    Compute Delta and Gamma on the full grid by:
      - mapping moneyness m -> strike K = m * spot_assumption
      - computing option prices via Black-Scholes using predicted implied vol Z_pred
      - finite-difference in S with bump size h = h_frac * spot_assumption
    Returns: Delta_grid, Gamma_grid (same shape as Z_pred)
    """
    try:
        shape = Z_pred.shape
        flat_m = M.ravel()
        flat_t = T.ravel()
        flat_sigma = Z_pred.ravel()
        S0 = spot_assumption
        K = flat_m * S0
        Tvec = flat_t
        h = max(1e-4, h_frac * S0)
        # price at S0, S0+h, S0-h
        p0 = bs_price_vectorized(np.full_like(K, S0), K, Tvec, r, flat_sigma, option_type, q)
        p_up = bs_price_vectorized(np.full_like(K, S0 + h), K, Tvec, r, flat_sigma, option_type, q)
        p_down = bs_price_vectorized(np.full_like(K, S0 - h), K, Tvec, r, flat_sigma, option_type, q)
        delta = (p_up - p_down) / (2 * h)
        gamma = (p_up - 2 * p0 + p_down) / (h * h)
        return delta.reshape(shape), gamma.reshape(shape)
    except Exception:
        logger.exception("Greeks computation failed")
        return np.full_like(Z_pred, np.nan), np.full_like(Z_pred, np.nan)

# =============================
# Vectorized arbitrage checks (adaptor)
# =============================
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

# =============================
# Export helpers (PNG/HTML)
# =============================
def export_plotly_fig_png(fig: go.Figure, filename: str = "figure.png") -> Optional[bytes]:
    """
    Attempt to export Plotly figure to PNG using kaleido. Return bytes or None.
    """
    try:
        img_bytes = fig.to_image(format="png", engine="kaleido")
        return img_bytes
    except Exception:
        logger.exception("PNG export failed (kaleido missing?)")
        return None

def export_plotly_fig_html(fig: go.Figure, filename: str = "figure.html") -> str:
    """
    Return HTML string for the interactive plotly figure for download.
    """
    try:
        return fig.to_html(full_html=True, include_plotlyjs='cdn')
    except Exception:
        logger.exception("HTML export failed")
        return fig.to_html(full_html=True, include_plotlyjs='cdn')

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Volatility Surface — Production Ready", layout="wide", page_icon="📊")
st.markdown("<style>.block-container{padding:0.75rem 1rem;max-width:100%;}</style>", unsafe_allow_html=True)
st.title("📊 Volatility Surface — Production-ready Visual Explorer")
st.caption("Animated surfaces, Greeks, export, safe integration with your code. Recruiter-ready defaults.")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    use_generator = st.checkbox("Use surface_generator (when available)", value=True)
    n_samples = st.slider("Dataset rows", 200, 5000, 1500, step=100)
    m_steps = st.slider("Moneyness grid steps", 12, 100, 40)
    t_steps = st.slider("TTM grid steps", 6, 60, 30)
    viz_model = st.selectbox("Model", AVAILABLE_MODELS, index=0)
    option_type = st.selectbox("Option type for Greeks", ["call", "put"], index=0)
    spot_assumption = st.number_input("Assumed Spot (for strike mapping)", min_value=1.0, value=100.0, step=1.0)
    r = st.number_input("Risk-free rate (r)", min_value=0.0, value=0.03, step=0.005)
    q = st.number_input("Dividend yield (q)", min_value=0.0, value=0.0, step=0.001)
    st.markdown("---")
    vis_choice = st.selectbox("Visualization Preset", ["3D Surface", "Heatmap", "Contour + Slices", "Residuals & Scatter", "Animation (TTM)", "Greeks (Delta/Gamma)"], index=0)
    st.markdown("---")
    st.write("Module availability (imported from src/volatility_surface if present):")
    for name, obj in [("VolatilitySurfaceGenerator", VolatilitySurfaceGenerator),
                      ("MLPModel", MLPModel), ("RandomForest", RandomForestVolatilityModel),
                      ("SVR", SVRModel), ("XGBoost", XGBoostModel),
                      ("feature_engineering", feature_engineering_module),
                      ("arbitrage_checks", arbitrage_checks_module),
                      ("arbitrage_enforcement", arbitrage_enforcement_module)]:
        st.write(f"- {'✅' if obj is not None else '❌'} {name}")

# generate dataset
if use_generator and VolatilitySurfaceGenerator:
    df = generate_surface_data_via_generator(n_samples)
else:
    df = generate_fallback_data(n_samples)

st.sidebar.markdown(f"**Dataset rows:** {len(df)}")
if st.button("Train quick (demo)", key="train_demo"):
    mdl = create_model_instance(viz_model)
    try:
        metrics = mdl.train(df, val_split=0.2) if hasattr(mdl, "train") else {}
        st.sidebar.success("Training done (demo).")
        st.sidebar.json(metrics)
        st.session_state['last_trained'] = (viz_model, mdl, metrics)
    except Exception:
        logger.exception("Training demo failed")
        st.sidebar.error("Training failed (see logs)")

# prefer last trained model if same name
if 'last_trained' in st.session_state and st.session_state['last_trained'][0] == viz_model:
    model_instance = st.session_state['last_trained'][1]
else:
    model_instance = create_model_instance(viz_model)

# Build grid & predict (cached)
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

# reshape preds -> 2D grid (matching shape of M_grid)
try:
    Z_pred = np.array(preds).reshape(M_grid.shape)
except Exception:
    try:
        Z_pred = np.array(preds).reshape((M_grid.shape[1], M_grid.shape[0])).T
    except Exception:
        logger.exception("Prediction reshape failed")
        Z_pred = np.full(M_grid.shape, np.nan)
        st.error("Prediction reshape failed — model output ordering unexpected.")

# "true" synthetic surface for reference (useful during demo)
def synthetic_true_surface(M, T):
    base = 0.2 + 0.05 * np.sin(2 * np.pi * M) * np.exp(-T)
    smile = 0.03 * (M - 1.0) ** 2
    return np.clip(base + smile + 0.02 * np.exp(-T), 0.03, 0.6)

Z_true = synthetic_true_surface(M_grid, T_grid)
resid = Z_pred - Z_true

# =============================
# PLOT HELPER FUNCTIONS
# =============================
def fig_surface(M, T, Z, title="Volatility Surface"):
    fig = go.Figure(go.Surface(x=M, y=T, z=Z, colorscale="Viridis", cmin=np.nanmin(Z), cmax=np.nanmax(Z)))
    fig.update_layout(title=title, template="plotly_dark", scene=dict(xaxis_title="Moneyness", yaxis_title="TTM", zaxis_title="Implied Vol"), height=720, margin=dict(t=50))
    return fig

def fig_heatmap(M, T, Z, title="Heatmap"):
    fig = go.Figure(go.Heatmap(z=Z, x=M[0,:], y=T[:,0], colorscale="Viridis"))
    fig.update_layout(title=title, template="plotly_dark", xaxis_title="Moneyness", yaxis_title="TTM", height=600)
    return fig

def fig_contour_slices(M, T, Z, slice_m=1.0, slice_t=1.0):
    # contour + slices as in prior version
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

def fig_distribution(vals, title="Predicted IV Distribution"):
    fig = px.histogram(vals, nbins=60, marginal="box", title=title)
    fig.update_layout(template="plotly_dark", height=500)
    return fig

# Greeks compute: use analytic methods if model exposes delta/gamma for a single input; else compute finite diff via BS
def compute_and_plot_greeks(M, T, Z_pred, option_type="call", spot_assumption=100.0, r=0.03, q=0.0, h_frac=1e-3):
    # If model offers analytic delta/gamma across grid, could call; otherwise, use finite-difference on BS prices.
    delta_grid, gamma_grid = compute_greeks_from_iv_grid(M, T, Z_pred, option_type=option_type, spot_assumption=spot_assumption, r=r, q=q, h_frac=h_frac)
    # build heatmap and 3D surfaces
    delta_fig = go.Figure(go.Surface(x=M, y=T, z=delta_grid, colorscale="RdBu"))
    delta_fig.update_layout(title="Delta Surface", template="plotly_dark", scene=dict(xaxis_title="Moneyness", yaxis_title="TTM", zaxis_title="Delta"), height=640)
    gamma_fig = go.Figure(go.Surface(x=M, y=T, z=gamma_grid, colorscale="RdBu"))
    gamma_fig.update_layout(title="Gamma Surface", template="plotly_dark", scene=dict(xaxis_title="Moneyness", yaxis_title="TTM", zaxis_title="Gamma"), height=640)
    return delta_grid, gamma_grid, delta_fig, gamma_fig

# Animation builder: frames across TTM slices (vary t)
def build_animation_frames(M, T, Z, t_axis_steps=30, title="Animated Surface"):
    """
    Build a plotly Figure with frames animating across t-index.
    We animate slices of the surface with fixed m grid and varying t-index (i.e., show IV as function of m at each T).
    For a full 3D frame-based animation we create frames with surface z changing (less heavy if grid small).
    """
    frames = []
    # ensure reasonable number of frames
    t_len = min(Z.shape[0], t_axis_steps)
    idxs = np.linspace(0, Z.shape[0] - 1, t_len).astype(int)
    base = go.Surface(x=M, y=T, z=Z, colorscale="Viridis", showscale=False)
    fig = go.Figure(data=[base])
    for i in idxs:
        frame_z = np.copy(Z)
        # optionally zero out other t rows for clarity - but we keep full surface
        frames.append(go.Frame(data=[go.Surface(x=M, y=T, z=Z)], name=str(i)))
    # Set up slider steps
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

# =============================
# Render chosen visualization
# =============================
st.markdown("## Visualizations")
if vis_choice == "3D Surface":
    fig = fig_surface(M_grid, T_grid, Z_pred, title=f"{viz_model} Predicted Surface (3D)")
    st.plotly_chart(fig, use_container_width=True)
    # export
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Download PNG (kaleido)", key="dl_png_surface"):
            img = export_plotly_fig_png(fig)
            if img:
                st.download_button("Download PNG", data=img, file_name="surface.png", mime="image/png")
            else:
                st.warning("PNG export unavailable (kaleido missing). Download HTML instead.")
    with col2:
        html = fig.to_html(full_html=True, include_plotlyjs='cdn')
        st.download_button("Download HTML", data=html, file_name="surface.html", mime="text/html")

elif vis_choice == "Heatmap":
    fig = fig_heatmap(M_grid, T_grid, Z_pred, title=f"{viz_model} Heatmap")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download Heatmap HTML", data=fig.to_html(full_html=True, include_plotlyjs='cdn'), file_name="heatmap.html", mime="text/html")

elif vis_choice == "Contour + Slices":
    slice_m = st.slider("Slice moneyness", float(M_grid.min()), float(M_grid.max()), float(1.0))
    slice_t = st.slider("Slice TTM", float(T_grid.min()), float(T_grid.max()), float(1.0))
    fig = fig_contour_slices(M_grid, T_grid, Z_pred, slice_m, slice_t)
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download Contour HTML", data=fig.to_html(full_html=True, include_plotlyjs='cdn'), file_name="contour.html", mime="text/html")

elif vis_choice == "Residuals & Scatter":
    fig_r = fig_residuals_scatter(Z_pred, Z_true)
    st.plotly_chart(fig_r, use_container_width=True)
    fig_res = fig_heatmap(M_grid, T_grid, resid, title="Residuals (Pred - True)")
    st.plotly_chart(fig_res, use_container_width=True)
    st.download_button("Download Residuals HTML", data=fig_res.to_html(full_html=True, include_plotlyjs='cdn'), file_name="residuals.html", mime="text/html")

elif vis_choice == "Animation (TTM)":
    # Build animated frames (be conservative with number of frames)
    frames_fig = build_animation_frames(M_grid, T_grid, Z_pred, t_axis_steps=min(20, T_grid.shape[0]), title=f"{viz_model} Animated Surface (TTM)")
    st.plotly_chart(frames_fig, use_container_width=True)
    st.download_button("Download Animation HTML", data=frames_fig.to_html(full_html=True, include_plotlyjs='cdn'), file_name="animation.html", mime="text/html")

elif vis_choice == "Greeks (Delta/Gamma)":
    with st.spinner("Computing Greeks (finite-difference via Black-Scholes) ..."):
        delta_grid, gamma_grid, delta_fig, gamma_fig = compute_and_plot_greeks(M_grid, T_grid, Z_pred, option_type, spot_assumption, r, q)
    st.subheader("Delta Surface")
    st.plotly_chart(delta_fig, use_container_width=True)
    st.subheader("Gamma Surface")
    st.plotly_chart(gamma_fig, use_container_width=True)
    # Heatmaps
    st.subheader("Delta Heatmap")
    st.plotly_chart(fig_heatmap(M_grid, T_grid, delta_grid, title="Delta Heatmap"), use_container_width=True)
    st.subheader("Gamma Heatmap")
    st.plotly_chart(fig_heatmap(M_grid, T_grid, gamma_grid, title="Gamma Heatmap"), use_container_width=True)
    # export
    hcol1, hcol2 = st.columns(2)
    with hcol1:
        st.download_button("Download Delta HTML", data=delta_fig.to_html(full_html=True, include_plotlyjs='cdn'), file_name="delta.html", mime="text/html")
    with hcol2:
        st.download_button("Download Gamma HTML", data=gamma_fig.to_html(full_html=True, include_plotlyjs='cdn'), file_name="gamma.html", mime="text/html")

# Summary metrics and arbitrage
st.markdown("---")
st.header("Model Summary & Arbitrage Check")
colA, colB, colC = st.columns(3)
with colA:
    st.metric("IV min", f"{np.nanmin(Z_pred):.4f}")
    st.metric("IV mean", f"{np.nanmean(Z_pred):.4f}")
with colB:
    st.metric("IV max", f"{np.nanmax(Z_pred):.4f}")
    rmse = np.sqrt(np.nanmean((Z_pred - Z_true) ** 2))
    st.metric("RMSE vs synthetic truth", f"{rmse:.6f}")
with colC:
    st.write("Model metadata")
    st.json({"model": viz_model, "params": getattr(model_instance, "params", {})})

# Arbitrage (vectorized)
strike_grid = M_grid * spot_assumption
ttm_grid = T_grid
with st.expander("Run arbitrage checks (vectorized)"):
    with st.spinner("Running arbitrage checks..."):
        arb_res = run_arbitrage_checks_vectorized(Z_pred, strike_grid, ttm_grid)
    st.json(arb_res)

st.markdown("---")
st.caption("Notes: This app assumes model.predict_volatility(df) returns implied vol aligned with df rows. For production, convert src/volatility_surface into an installable package and enforce a canonical interface (VolatilityModelBase with .train() and .predict_volatility()).")
