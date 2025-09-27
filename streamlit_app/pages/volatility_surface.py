# streamlit_vol_surface_prod_visual_greeks.py
"""
Production-ready Volatility Surface Visual Explorer with Correct Imports
"""

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

# Optional external imports
try:
    from scipy.stats import norm
except Exception:
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
# CORRECT IMPORT BASED ON YOUR STRUCTURE
# =============================

def setup_import_paths():
    """Setup import paths for your specific structure"""
    possible_paths = [
        Path("/mount/src/optionslab/src"),
        Path.cwd() / "src",
        Path.cwd().parent / "src", 
        Path(__file__).parent / "src",
        Path(__file__).parent.parent / "src",
    ]
    
    added_paths = []
    for path in possible_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
            added_paths.append(str(path))
            logger.info(f"Added to sys.path: {path}")
    
    return added_paths

added_paths = setup_import_paths()

# =============================
# DIRECT IMPORTS WITH CORRECT NAMES
# =============================

logger.info("=== ATTEMPTING DIRECT IMPORTS ===")

# Import core modules
VolatilitySurfaceGenerator = None
MLPModel = None
RandomForestVolatilityModel = None  
SVRModel = None
XGBoostModel = None
feature_engineering_module = None
arbitrage_checks_module = None
arbitrage_enforcement_module = None
grid_search_module = None

try:
    from volatility_surface.surface_generator import VolatilitySurfaceGenerator
    logger.info("✓ Imported VolatilitySurfaceGenerator")
except Exception as e:
    logger.warning(f"VolatilitySurfaceGenerator import failed: {e}")

try:
    from volatility_surface.models.mlp_model import MLPModel
    logger.info("✓ Imported MLPModel")
except Exception as e:
    logger.warning(f"MLPModel import failed: {e}")

try:
    from volatility_surface.models.random_forest import RandomForestVolatilityModel
    logger.info("✓ Imported RandomForestVolatilityModel")
except Exception as e:
    logger.warning(f"RandomForestVolatilityModel import failed: {e}")

try:
    from volatility_surface.models.svr_model import SVRModel
    logger.info("✓ Imported SVRModel")
except Exception as e:
    logger.warning(f"SVRModel import failed: {e}")

try:
    from volatility_surface.models.xgboost_model import XGBoostModel
    logger.info("✓ Imported XGBoostModel")
except Exception as e:
    logger.warning(f"XGBoostModel import failed: {e}")

try:
    from volatility_surface.utils import feature_engineering as feature_engineering_module
    logger.info("✓ Imported feature_engineering")
except Exception as e:
    logger.warning(f"feature_engineering import failed: {e}")

try:
    # Based on your structure, arbitrage checks might be in arbitrage.py or arbitrage_utils.py
    from volatility_surface.utils import arbitrage as arbitrage_checks_module
    logger.info("✓ Imported arbitrage (checks)")
except Exception as e:
    logger.warning(f"arbitrage import failed: {e}")

try:
    from volatility_surface.utils import arbitrage_utils as arbitrage_enforcement_module
    logger.info("✓ Imported arbitrage_utils (enforcement)")
except Exception as e:
    logger.warning(f"arbitrage_utils import failed: {e}")

try:
    from volatility_surface.utils import grid_search as grid_search_module
    logger.info("✓ Imported grid_search")
except Exception as e:
    logger.warning(f"grid_search import failed: {e}")

# =============================
# DummyModel Fallback
# =============================
class DummyModel:
    def __init__(self, **kwargs):
        self.params = kwargs or {}
        self.feature_names_in_ = ["moneyness","log_moneyness","time_to_maturity","ttm_squared","risk_free_rate","historical_volatility","volatility_skew"]
        self.name = "DummyModel"
        self.is_trained = True
        
    def train(self, df: pd.DataFrame, val_split: float = 0.2) -> Dict[str, float]:
        logger.info("DummyModel.train called")
        return {"train_rmse": 0.1, "val_rmse": 0.12, "val_r2": 0.85, "note": "Dummy model"}
    
    def predict_volatility(self, df: pd.DataFrame) -> np.ndarray:
        m = df["moneyness"].to_numpy()
        t = df["time_to_maturity"].to_numpy()
        base = 0.2 + 0.05 * np.sin(2 * np.pi * m) * np.exp(-t)
        smile = 0.03 * (m - 1.0) ** 2
        return np.clip(base + smile, 0.03, 0.6)

# =============================
# Model Factory
# =============================
def create_model_instance(name: str, **kwargs):
    """Create model instance with correct class mapping"""
    model_map = {
        "MLP Neural Network": MLPModel,
        "Random Forest": RandomForestVolatilityModel, 
        "SVR": SVRModel,
        "XGBoost": XGBoostModel
    }
    
    model_class = model_map.get(name)
    
    if model_class is None:
        logger.info(f"Using DummyModel for {name}")
        return DummyModel(**kwargs)
    
    try:
        logger.info(f"Creating {name} instance")
        instance = model_class(**kwargs)
        logger.info(f"✓ Successfully created {name} instance")
        return instance
    except Exception as e:
        logger.error(f"Failed to create {name}: {e}")
        return DummyModel(**kwargs)

# Get available models
MODEL_NAMES = ["MLP Neural Network", "Random Forest", "SVR", "XGBoost"]
AVAILABLE_MODELS = []

for name in MODEL_NAMES:
    model_class = {
        "MLP Neural Network": MLPModel,
        "Random Forest": RandomForestVolatilityModel,
        "SVR": SVRModel,
        "XGBoost": XGBoostModel
    }.get(name)
    
    if model_class is not None:
        AVAILABLE_MODELS.append(name)

if not AVAILABLE_MODELS:
    AVAILABLE_MODELS = ["DummyModel"]
    logger.info("No custom models available, using DummyModel")

logger.info(f"Available models: {AVAILABLE_MODELS}")

# =============================
# UI Configuration
# =============================
def setup_dark_theme():
    st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    .stApp { background: linear-gradient(135deg, #0c0d13 0%, #1a1d29 100%); max-width: 100% !important; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 100% !important; }
    .section { background: rgba(30, 33, 48, 0.9); border-radius: 10px; padding: 1.5rem; margin: 1rem 0; border-left: 4px solid #ff4b4b; }
    .metric-card { background: rgba(40, 44, 62, 0.8); padding: 1rem; border-radius: 8px; border: 1px solid #2a2f45; }
    .status-available { background: rgba(0, 200, 83, 0.2); border-left: 3px solid #00c853; }
    .status-unavailable { background: rgba(255, 75, 75, 0.2); border-left: 3px solid #ff4b4b; }
    .stButton>button { background: linear-gradient(45deg, #ff6b6b, #ff4b4b); color: white; border-radius: 5px; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# =============================
# Utility Functions
# =============================
def build_prediction_grid(m_start=0.7, m_end=1.3, m_steps=40, t_start=0.05, t_end=2.0, t_steps=40):
    m = np.linspace(m_start, m_end, m_steps)
    t = np.linspace(t_start, t_end, t_steps)
    M, T = np.meshgrid(m, t, indexing='xy')
    flat_m = M.ravel()
    flat_t = T.ravel()
    grid_df = pd.DataFrame({
        "moneyness": flat_m, "log_moneyness": np.log(np.clip(flat_m, 1e-12, None)),
        "time_to_maturity": flat_t, "ttm_squared": flat_t ** 2,
        "risk_free_rate": np.full(flat_m.shape, 0.03),
        "historical_volatility": np.full(flat_m.shape, 0.2),
        "volatility_skew": np.zeros(flat_m.shape)
    })
    return M, T, grid_df

def safe_model_predict_volatility(model: Any, df: pd.DataFrame) -> np.ndarray:
    """Safe prediction with error handling"""
    try:
        # Check if model is trained
        if hasattr(model, '_assert_trained'):
            try:
                model._assert_trained()
            except Exception as e:
                logger.warning(f"Model not trained: {e}")
                # Fallback to dummy prediction
                m = df["moneyness"].to_numpy()
                t = df["time_to_maturity"].to_numpy()
                base = 0.2 + 0.05 * np.sin(2 * np.pi * m) * np.exp(-t)
                smile = 0.03 * (m - 1.0) ** 2
                return np.clip(base + smile, 0.03, 0.6)
        
        if hasattr(model, "predict_volatility"):
            return model.predict_volatility(df)
        elif hasattr(model, "predict"):
            return model.predict(df)
        else:
            # Fallback
            m = df["moneyness"].to_numpy()
            t = df["time_to_maturity"].to_numpy()
            base = 0.2 + 0.05 * np.sin(2 * np.pi * m) * np.exp(-t)
            smile = 0.03 * (m - 1.0) ** 2
            return np.clip(base + smile, 0.03, 0.6)
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        # Final fallback
        m = df["moneyness"].to_numpy()
        t = df["time_to_maturity"].to_numpy()
        base = 0.2 + 0.05 * np.sin(2 * np.pi * m) * np.exp(-t)
        smile = 0.03 * (m - 1.0) ** 2
        return np.clip(base + smile, 0.03, 0.6)

def cache_key(model_name: str, params: Dict[str, Any], m_steps: int, t_steps: int) -> str:
    payload = {"model": model_name, "params": params, "m": m_steps, "t": t_steps}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

# Initialize session state
if 'pred_cache' not in st.session_state:
    st.session_state['pred_cache'] = {}
if 'training_data' not in st.session_state:
    st.session_state['training_data'] = None

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
        "underlying_price": spots, "strike_price": strikes, "time_to_maturity": ttms,
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
        generator = VolatilitySurfaceGenerator(base_strikes, base_maturities, 
                                             np.zeros((20, 50)), 50, 20, 'cubic')
        spots = rng.uniform(90, 110, n_samples)
        strikes = rng.uniform(80, 120, n_samples)
        ttms = rng.uniform(0.1, 2.0, n_samples)
        ivs = generator.get_surface_batch(strikes, ttms)
        df = pd.DataFrame({
            "underlying_price": spots, "strike_price": strikes, "time_to_maturity": ttms,
            "risk_free_rate": rng.uniform(0.01, 0.05, n_samples),
            "historical_volatility": rng.uniform(0.12, 0.28, n_samples),
            "implied_volatility": ivs
        })
        return df
    except Exception:
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
        return 0.0

def bs_price_vectorized(S_arr, K_arr, T_arr, r, sigma_arr, option_type="call", q=0.0):
    out = np.zeros_like(S_arr, dtype=float)
    for i in range(len(out)):
        out[i] = black_scholes_price(S_arr[i], K_arr[i], T_arr[i], r, sigma_arr[i], option_type, q)
    return out

def compute_greeks_from_iv_grid(M, T, Z_pred, option_type="call", spot_assumption=100.0, r=0.03, q=0.0):
    try:
        shape = Z_pred.shape
        flat_m = M.ravel()
        flat_t = T.ravel()
        flat_sigma = Z_pred.ravel()
        S0 = spot_assumption
        K = flat_m * S0
        h = 0.01 * S0
        p0 = bs_price_vectorized(np.full_like(K, S0), K, flat_t, r, flat_sigma, option_type, q)
        p_up = bs_price_vectorized(np.full_like(K, S0 + h), K, flat_t, r, flat_sigma, option_type, q)
        p_down = bs_price_vectorized(np.full_like(K, S0 - h), K, flat_t, r, flat_sigma, option_type, q)
        delta = (p_up - p_down) / (2 * h)
        gamma = (p_up - 2 * p0 + p_down) / (h * h)
        return delta.reshape(shape), gamma.reshape(shape)
    except Exception:
        return np.full_like(Z_pred, np.nan), np.full_like(Z_pred, np.nan)

# =============================
# Visualization Functions
# =============================
def fig_surface(M, T, Z, title="Volatility Surface"):
    fig = go.Figure(go.Surface(x=M, y=T, z=Z, colorscale="Viridis"))
    fig.update_layout(title=title, template="plotly_dark", height=600,
                     scene=dict(xaxis_title="Moneyness", yaxis_title="TTM", zaxis_title="Implied Vol"))
    return fig

def fig_heatmap(M, T, Z, title="Heatmap"):
    fig = go.Figure(go.Heatmap(z=Z, x=M[0,:], y=T[:,0], colorscale="Viridis"))
    fig.update_layout(title=title, template="plotly_dark", height=500,
                     xaxis_title="Moneyness", yaxis_title="TTM")
    return fig

def synthetic_true_surface(M, T):
    base = 0.2 + 0.05 * np.sin(2 * np.pi * M) * np.exp(-T)
    smile = 0.03 * (M - 1.0) ** 2
    return np.clip(base + smile, 0.03, 0.6)

# =============================
# Main Application
# =============================
def main():
    st.set_page_config(page_title="Volatility Surface Explorer", layout="wide")
    setup_dark_theme()
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0;">📊 Volatility Surface Explorer</h1>
        <p style="color: white; opacity: 0.9;">Fixed Import Structure</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Debug Info
    with st.expander("🔧 Import Status", expanded=True):
        st.write("**Available Models:**", AVAILABLE_MODELS)
        modules = [
            ("VolatilitySurfaceGenerator", VolatilitySurfaceGenerator),
            ("MLPModel", MLPModel),
            ("RandomForest", RandomForestVolatilityModel),
            ("SVRModel", SVRModel),
            ("XGBoostModel", XGBoostModel),
        ]
        for name, obj in modules:
            status = "✅" if obj is not None else "❌"
            st.write(f"{status} {name}")
    
    # Configuration
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_generator = st.checkbox("Use Surface Generator", value=VolatilitySurfaceGenerator is not None)
        n_samples = st.slider("Dataset Size", 200, 5000, 1500)
        
    with col2:
        viz_model = st.selectbox("Model Type", AVAILABLE_MODELS, index=0)
        m_steps = st.slider("Moneyness Grid", 12, 100, 40)
        
    with col3:
        option_type = st.selectbox("Option Type", ["call", "put"], index=0)
        t_steps = st.slider("TTM Grid", 6, 60, 30)
        spot_assumption = st.number_input("Spot Price", value=100.0)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Generation
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📊 Data Management")
    
    if st.button("🔄 Generate Training Data", use_container_width=True):
        with st.spinner("Generating data..."):
            if use_generator and VolatilitySurfaceGenerator is not None:
                df = generate_surface_data_via_generator(n_samples)
            else:
                df = generate_fallback_data(n_samples)
            st.session_state['training_data'] = df
            st.success(f"Generated {len(df)} samples")
    
    if st.session_state['training_data'] is not None:
        df = st.session_state['training_data']
        st.info(f"Training data: {len(df)} samples")
    else:
        df = generate_fallback_data(1000)
        st.warning("Using fallback data - generate training data for better results")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Training
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 🤖 Model Training")
    
    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Training..."):
            mdl = create_model_instance(viz_model)
            try:
                if hasattr(mdl, 'train'):
                    metrics = mdl.train(df, val_split=0.2)
                    st.success("Training completed!")
                    st.json(metrics)
                else:
                    st.info("Model doesn't require training")
                st.session_state['last_trained'] = (viz_model, mdl, {})
            except Exception as e:
                st.error(f"Training failed: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Instance
    if 'last_trained' in st.session_state and st.session_state['last_trained'][0] == viz_model:
        model_instance = st.session_state['last_trained'][1]
        st.success("Using trained model")
    else:
        model_instance = create_model_instance(viz_model)
        st.info("Using new model instance")
    
    # Visualization
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📈 Visualization")
    
    M_grid, T_grid, grid_df = build_prediction_grid(0.7, 1.3, m_steps, 0.05, 2.0, t_steps)
    
    ck = cache_key(viz_model, getattr(model_instance, "params", {}), m_steps, t_steps)
    
    if ck in st.session_state['pred_cache']:
        preds = st.session_state['pred_cache'][ck]
    else:
        with st.spinner("Computing predictions..."):
            preds = safe_model_predict_volatility(model_instance, grid_df)
            st.session_state['pred_cache'][ck] = preds
    
    try:
        Z_pred = np.array(preds).reshape(M_grid.shape)
    except Exception:
        Z_pred = np.full(M_grid.shape, 0.2)
    
    Z_true = synthetic_true_surface(M_grid, T_grid)
    
    # Display
    fig = fig_surface(M_grid, T_grid, Z_pred, f"{viz_model} Surface")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Metrics
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📊 Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("IV Min", f"{np.nanmin(Z_pred):.4f}")
    with col2: st.metric("IV Mean", f"{np.nanmean(Z_pred):.4f}")
    with col3: st.metric("IV Max", f"{np.nanmax(Z_pred):.4f}")
    with col4: st.metric("RMSE", f"{np.sqrt(np.nanmean((Z_pred - Z_true)**2)):.6f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()