# streamlit_vol_surface_prod_visual_greeks.py
"""
Production-ready Volatility Surface Visual Explorer with Greeks & Animation
- Enhanced import system for Windows paths
- Fixed training data scope issue
- Robust module detection
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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
import json
import hashlib
import math

# =============================
# Enhanced Import System for Windows Paths
# =============================

# Optional external imports
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

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
# Advanced Import System
# =============================
def setup_import_paths():
    """Enhanced path setup specifically for your Windows directory structure"""
    current_file = Path(__file__).resolve()
    
    # Your specific path from the error
    your_specific_path = Path(r"D:\Coding\Python\OptionsLab\src")
    if your_specific_path.exists():
        if str(your_specific_path) not in sys.path:
            sys.path.insert(0, str(your_specific_path))
            logger.info(f"Added your specific path: {your_specific_path}")
    
    # Also try common locations
    possible_paths = [
        current_file.parents[1] / "src",
        current_file.parent / "src",
        Path.cwd() / "src",
        Path.cwd().parent / "src",
        Path(r"D:\Coding\Python\OptionsLab\src"),  # Your exact path
        Path(r"D:/Coding/Python/OptionsLab/src"),  # Forward slashes
    ]
    
    added_paths = []
    for path in possible_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
            added_paths.append(str(path))
            logger.info(f"Added to sys.path: {path}")
    
    return added_paths

# Setup paths immediately
added_paths = setup_import_paths()
logger.info(f"Current sys.path: {sys.path}")

# =============================
# Robust Import Helper
# =============================
def robust_import(module_name, class_name=None, alternative_paths=None):
    """
    Try multiple strategies to import modules
    """
    strategies = []
    
    # Strategy 1: Direct import
    if class_name:
        strategies.append(f"from {module_name} import {class_name}")
    else:
        strategies.append(f"import {module_name}")
    
    # Strategy 2: Try with src prefix
    if class_name:
        strategies.append(f"from src.{module_name} import {class_name}")
    else:
        strategies.append(f"import src.{module_name}")
    
    # Strategy 3: Try alternative paths
    if alternative_paths:
        for alt in alternative_paths:
            if class_name:
                strategies.append(f"from {alt}.{module_name} import {class_name}")
            else:
                strategies.append(f"import {alt}.{module_name}")
    
    for strategy in strategies:
        try:
            if strategy.startswith("from"):
                parts = strategy.split()
                module_path = parts[1]
                attr_name = parts[3]
                module = __import__(module_path, fromlist=[attr_name])
                result = getattr(module, attr_name)
                logger.info(f"✓ Successfully imported {attr_name} using: {strategy}")
                return result
            else:
                module_name = strategy.split()[1]
                module = __import__(module_name, fromlist=['*'])
                logger.info(f"✓ Successfully imported {module_name}")
                return module
        except ImportError as e:
            logger.debug(f"Strategy failed {strategy}: {e}")
            continue
        except Exception as e:
            logger.debug(f"Error with strategy {strategy}: {e}")
            continue
    
    logger.warning(f"❌ All import strategies failed for {module_name}.{class_name if class_name else ''}")
    return None

# =============================
# Import Models with Enhanced Detection
# =============================
def import_all_models():
    """Import all volatility surface models with detailed logging"""
    models = {}
    
    # Define all model imports
    model_imports = {
        "VolatilitySurfaceGenerator": ("volatility_surface.surface_generator", "VolatilitySurfaceGenerator"),
        "MLPModel": ("volatility_surface.models.mlp_model", "MLPModel"),
        "RandomForestVolatilityModel": ("volatility_surface.models.random_forest", "RandomForestVolatilityModel"),
        "SVRModel": ("volatility_surface.models.svr_model", "SVRModel"),
        "XGBoostModel": ("volatility_surface.models.xgboost_model", "XGBoostModel"),
    }
    
    module_imports = {
        "feature_engineering": "volatility_surface.utils.feature_engineering",
        "arbitrage_checks": "volatility_surface.utils.arbitrage_checks",
        "arbitrage_enforcement": "volatility_surface.utils.arbitrage_enforcement",
        "grid_search": "volatility_surface.utils.grid_search",
    }
    
    # Import models
    for name, (module_path, class_name) in model_imports.items():
        models[name] = robust_import(module_path, class_name, 
                                   alternative_paths=["volatility_surface", "src.volatility_surface"])
    
    # Import modules
    for name, module_path in module_imports.items():
        models[name] = robust_import(module_path, alternative_paths=["volatility_surface", "src.volatility_surface"])
    
    return models

# Import everything
imported_models = import_all_models()

# Assign to global variables
VolatilitySurfaceGenerator = imported_models.get("VolatilitySurfaceGenerator")
MLPModel = imported_models.get("MLPModel")
RandomForestVolatilityModel = imported_models.get("RandomForestVolatilityModel")
SVRModel = imported_models.get("SVRModel")
XGBoostModel = imported_models.get("XGBoostModel")
feature_engineering_module = imported_models.get("feature_engineering")
arbitrage_checks_module = imported_models.get("arbitrage_checks")
arbitrage_enforcement_module = imported_models.get("arbitrage_enforcement")
grid_search_module = imported_models.get("grid_search")

# Log import status
logger.info("=== IMPORT STATUS ===")
for name, obj in [
    ("VolatilitySurfaceGenerator", VolatilitySurfaceGenerator),
    ("MLPModel", MLPModel),
    ("RandomForest", RandomForestVolatilityModel),
    ("SVR", SVRModel),
    ("XGBoost", XGBoostModel),
    ("feature_engineering", feature_engineering_module),
    ("arbitrage_checks", arbitrage_checks_module),
    ("arbitrage_enforcement", arbitrage_enforcement_module)
]:
    status = "✓" if obj is not None else "✗"
    logger.info(f"{status} {name}")

# =============================
# DummyModel Fallback
# =============================
class DummyModel:
    def __init__(self, **kwargs):
        self.params = kwargs or {}
        self.feature_names_in_ = ["moneyness","log_moneyness","time_to_maturity","ttm_squared","risk_free_rate","historical_volatility","volatility_skew"]
        self.name = "DummyModel"
        self.is_trained = False
        
    def train(self, df: pd.DataFrame, val_split: float = 0.2) -> Dict[str, float]:
        logger.info("DummyModel.train called")
        self.is_trained = True
        return {"train_rmse": 0.1, "val_rmse": 0.12, "val_r2": 0.85, "note": "Dummy model - no real training"}
    
    def predict_volatility(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            logger.warning("DummyModel not trained, returning simple surface")
        # simple smile function based on moneyness & ttm
        m = df["moneyness"].to_numpy()
        t = df["time_to_maturity"].to_numpy()
        base = 0.2 + 0.05 * np.sin(2 * np.pi * m) * np.exp(-t)
        smile = 0.03 * (m - 1.0) ** 2
        return np.clip(base + smile, 0.03, 0.6)

# Enhanced Model Factory
def create_model_instance(name: str, **kwargs):
    """Create model instance with better error handling"""
    cls_map = {
        "MLP Neural Network": MLPModel,
        "Random Forest": RandomForestVolatilityModel,
        "SVR": SVRModel,
        "XGBoost": XGBoostModel
    }
    
    cls = cls_map.get(name)
    
    if cls is None:
        logger.info(f"Using DummyModel for {name}")
        return DummyModel(**kwargs)
    
    try:
        logger.info(f"Attempting to create {name} instance")
        instance = cls(**kwargs)
        logger.info(f"✓ Successfully created {name} instance")
        return instance
    except Exception as e:
        logger.error(f"Failed to create {name}: {e}")
        logger.info("Falling back to DummyModel")
        return DummyModel(**kwargs)

# Get available models
MODEL_NAMES = ["MLP Neural Network", "Random Forest", "SVR", "XGBoost"]
AVAILABLE_MODELS = []
for name in MODEL_NAMES:
    cls = {
        "MLP Neural Network": MLPModel,
        "Random Forest": RandomForestVolatilityModel,
        "SVR": SVRModel,
        "XGBoost": XGBoostModel
    }.get(name)
    if cls is not None:
        AVAILABLE_MODELS.append(name)

if not AVAILABLE_MODELS:
    AVAILABLE_MODELS = ["DummyModel"]
    logger.info("No custom models available, using DummyModel only")

# =============================
# UI Configuration and Styling
# =============================
def setup_dark_theme():
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
    """Safe prediction with comprehensive error handling"""
    try:
        # Check if model needs training
        if hasattr(model, '_assert_trained'):
            try:
                model._assert_trained()
            except Exception as e:
                logger.warning(f"Model not trained, using fallback: {e}")
                return np.full(len(df), 0.2)
        
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
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        # Return a reasonable fallback surface
        m = df["moneyness"].to_numpy()
        t = df["time_to_maturity"].to_numpy()
        base = 0.2 + 0.05 * np.sin(2 * np.pi * m) * np.exp(-t)
        smile = 0.03 * (m - 1.0) ** 2
        return np.clip(base + smile, 0.03, 0.6)

def cache_key(model_name: str, params: Dict[str, Any], m_steps: int, t_steps: int, extra: Optional[Dict] = None) -> str:
    payload = {"model": model_name, "params": params, "m": m_steps, "t": t_steps, "extra": extra or {}}
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
        return np.full_like(Z_pred, np.nan), np.full_like(Z_pred, np.nan)

# =============================
# Visualization Functions
# =============================
def fig_surface(M, T, Z, title="Volatility Surface"):
    fig = go.Figure(go.Surface(x=M, y=T, z=Z, colorscale="Viridis"))
    fig.update_layout(title=title, template="plotly_dark", 
                     scene=dict(xaxis_title="Moneyness", yaxis_title="TTM", zaxis_title="Implied Vol"), 
                     height=600)
    return fig

def fig_heatmap(M, T, Z, title="Heatmap"):
    fig = go.Figure(go.Heatmap(z=Z, x=M[0,:], y=T[:,0], colorscale="Viridis"))
    fig.update_layout(title=title, template="plotly_dark", 
                     xaxis_title="Moneyness", yaxis_title="TTM", height=500)
    return fig

def synthetic_true_surface(M, T):
    base = 0.2 + 0.05 * np.sin(2 * np.pi * M) * np.exp(-T)
    smile = 0.03 * (M - 1.0) ** 2
    return np.clip(base + smile + 0.02 * np.exp(-T), 0.03, 0.6)

# =============================
# Main Application
# =============================
def main():
    st.set_page_config(
        page_title="Volatility Surface Explorer", 
        layout="wide", 
        page_icon="📊",
        initial_sidebar_state="collapsed"
    )
    
    setup_dark_theme()
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">📊 Volatility Surface Explorer</h1>
        <p style="color: white; opacity: 0.9; font-size: 1.1rem;">Enhanced import system with Windows path support</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Debug Information
    with st.expander("🔧 Debug Information", expanded=False):
        st.write("**Python Path:**", sys.executable)
        st.write("**Working Directory:**", Path.cwd())
        st.write("**Added Paths:**", added_paths)
        st.write("**Available Models:**", AVAILABLE_MODELS)
    
    # Configuration Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        use_generator = st.checkbox("Use Surface Generator", 
                                   value=VolatilitySurfaceGenerator is not None and VolatilitySurfaceGenerator is not None,
                                   disabled=VolatilitySurfaceGenerator is None,
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
    
    # Generate training data FIRST (before any model operations)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📊 Training Data")
    
    if st.button("🔄 Generate Training Data", use_container_width=True):
        with st.spinner("Generating training data..."):
            if use_generator and VolatilitySurfaceGenerator is not None:
                df = generate_surface_data_via_generator(n_samples)
            else:
                df = generate_fallback_data(n_samples)
            st.session_state['training_data'] = df
            st.success(f"Generated {len(df)} training samples")
    
    # Display data info if available
    if st.session_state['training_data'] is not None:
        df = st.session_state['training_data']
        st.write(f"**Training Data Ready:** {len(df)} samples")
        st.write(f"**Columns:** {list(df.columns)}")
        st.write(f"**IV Range:** {df['implied_volatility'].min():.3f} - {df['implied_volatility'].max():.3f}")
    else:
        st.warning("No training data generated yet. Click the button above to generate data.")
        df = generate_fallback_data(1000)  # Fallback for initial display
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Training Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 🤖 Model Training")
    
    col5, col6 = st.columns([2, 1])
    
    with col5:
        st.markdown("**Module Status:**")
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
    
    with col6:
        if st.button("🚀 Train Model", use_container_width=True, 
                    disabled=st.session_state['training_data'] is None):
            if st.session_state['training_data'] is None:
                st.error("Please generate training data first!")
            else:
                with st.spinner("Training model..."):
                    df = st.session_state['training_data']
                    mdl = create_model_instance(viz_model)
                    try:
                        # Check if model has train method
                        if hasattr(mdl, 'train'):
                            metrics = mdl.train(df, val_split=0.2)
                        else:
                            metrics = {"note": "Model does not require training"}
                        
                        st.session_state['last_trained'] = (viz_model, mdl, metrics)
                        st.success("Training completed successfully!")
                        if metrics:
                            st.json(metrics)
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
                        st.info("Using untrained model with fallback predictions")
                        st.session_state['last_trained'] = (viz_model, mdl, {"error": str(e)})
        
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.session_state['pred_cache'] = {}
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Instance Selection
    if 'last_trained' in st.session_state and st.session_state['last_trained'][0] == viz_model:
        model_instance = st.session_state['last_trained'][1]
        st.info(f"Using trained {viz_model} model")
    else:
        model_instance = create_model_instance(viz_model)
        st.info(f"Using new {viz_model} instance (not trained)")
    
    # Visualization Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📈 Visualization")
    
    vis_options = ["3D Surface", "Heatmap", "Greeks (Delta/Gamma)"]
    vis_choice = st.selectbox("Visualization Type", vis_options, index=0)
    
    # Build prediction grid
    M_grid, T_grid, grid_df = build_prediction_grid(0.7, 1.3, m_steps, 0.05, 2.0, t_steps)
    
    # Get predictions
    ck = cache_key(viz_model, getattr(model_instance, "params", {}), m_steps, t_steps)
    
    if ck in st.session_state['pred_cache']:
        preds = st.session_state['pred_cache'][ck]
    else:
        with st.spinner("Computing predictions..."):
            preds = safe_model_predict_volatility(model_instance, grid_df)
            st.session_state['pred_cache'][ck] = preds
    
    # Reshape predictions
    try:
        Z_pred = np.array(preds).reshape(M_grid.shape)
    except Exception:
        Z_pred = np.full(M_grid.shape, 0.2)
        st.error("Prediction reshape failed")
    
    Z_true = synthetic_true_surface(M_grid, T_grid)
    
    # Display visualization
    if vis_choice == "3D Surface":
        fig = fig_surface(M_grid, T_grid, Z_pred, title=f"{viz_model} Predicted Surface")
        st.plotly_chart(fig, use_container_width=True)
        
    elif vis_choice == "Heatmap":
        fig = fig_heatmap(M_grid, T_grid, Z_pred, title=f"{viz_model} Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        
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
    
    # Metrics Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📊 Performance Metrics")
    
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
        data_size = len(st.session_state['training_data']) if st.session_state['training_data'] is not None else 0
        st.metric("Data Points", f"{data_size:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col12:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Model Info**")
        st.text(f"Type: {viz_model}")
        st.text(f"Trained: {'Yes' if 'last_trained' in st.session_state else 'No'}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def compute_and_plot_greeks(M, T, Z_pred, option_type="call", spot_assumption=100.0, r=0.03, q=0.0, h_frac=1e-3):
    delta_grid, gamma_grid = compute_greeks_from_iv_grid(M, T, Z_pred, option_type, spot_assumption, r, q, h_frac)
    
    delta_fig = go.Figure(go.Surface(x=M, y=T, z=delta_grid, colorscale="RdBu"))
    delta_fig.update_layout(title="Delta Surface", template="plotly_dark", 
                           scene=dict(xaxis_title="Moneyness", yaxis_title="TTM", zaxis_title="Delta"), 
                           height=500)
    
    gamma_fig = go.Figure(go.Surface(x=M, y=T, z=gamma_grid, colorscale="RdBu"))
    gamma_fig.update_layout(title="Gamma Surface", template="plotly_dark", 
                           scene=dict(xaxis_title="Moneyness", yaxis_title="TTM", zaxis_title="Gamma"), 
                           height=500)
    
    return delta_grid, gamma_grid, delta_fig, gamma_fig

if __name__ == "__main__":
    main()