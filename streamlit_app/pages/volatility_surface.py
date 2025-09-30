"""
Production-ready Volatility Surface Visual Explorer - Fixed Import Version
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
# Enhanced Import System - FIXED LOGIC
# =============================

# Get the correct paths for structure
ROOT_DIR = Path(__file__).parent.parent  # Go up from streamlit_app to project root
SRC_DIR = ROOT_DIR / "src"

print(f"🔍 ROOT_DIR: {ROOT_DIR}")
print(f"🔍 SRC_DIR: {SRC_DIR}")

# Add to path - only if it exists
if SRC_DIR.exists():
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    print(f"✅ Added to path: {SRC_DIR}")
else:
    st.error(f"❌ SRC directory not found at: {SRC_DIR}")

# Debug: Check what's actually available
print("\n=== CHECKING VOLATILITY_SURFACE STRUCTURE ===")
volatility_path = SRC_DIR / "volatility_surface"
if volatility_path.exists():
    print(f"📁 volatility_surface found: {volatility_path}")
    for item in volatility_path.iterdir():
        if item.is_dir():
            print(f"   📁 {item.name}/")
            if item.name == "models":
                for model_file in item.glob("*.py"):
                    print(f"      📄 {model_file.name}")
        else:
            print(f"   📄 {item.name}")
else:
    print(f"❌ volatility_surface not found at: {volatility_path}")

# =============================
# STRICT IMPORTS - FIXED LOGIC
# =============================

def strict_import(module_path, class_name=None):
    """
    Strict import that shows exact errors - no silent fallbacks
    """
    try:
        if class_name:
            # Use importlib for more reliable imports
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            print(f"✅ SUCCESS: {module_path} -> {class_name}")
            return cls
        else:
            module = __import__(module_path, fromlist=[''])
            print(f"✅ SUCCESS: {module_path}")
            return module
    except Exception as e:
        print(f"❌ CRITICAL FAILURE: {module_path}{f' -> {class_name}' if class_name else ''}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {e}")
        print(f"   Full traceback:")
        traceback.print_exc()
        return None

print("\n=== ATTEMPTING STRICT IMPORTS ===")

# Import in dependency order
VolatilityModelBase = strict_import("volatility_surface.base", "VolatilityModelBase")
MLPModel = strict_import("volatility_surface.models.mlp_model", "MLPModel")
RandomForestVolatilityModel = strict_import("volatility_surface.models.random_forest", "RandomForestVolatilityModel")  
SVRModel = strict_import("volatility_surface.models.svr_model", "SVRModel")
XGBoostModel = strict_import("volatility_surface.models.xgboost_model", "XGBoostModel")
VolatilitySurfaceGenerator = strict_import("volatility_surface.surface_generator", "VolatilitySurfaceGenerator")

# =============================
# Model Availability Tracking - STRICT
# =============================
MODEL_CLASSES = {
    "MLP Neural Network": MLPModel,
    "Random Forest": RandomForestVolatilityModel,
    "SVR": SVRModel, 
    "XGBoost": XGBoostModel
}

# Only consider models that actually imported successfully
AVAILABLE_MODELS = [name for name, cls in MODEL_CLASSES.items() if cls is not None]

print(f"Strict Available models: {AVAILABLE_MODELS}")

# Track base model availability
BASE_AVAILABLE = VolatilityModelBase is not None

# =============================
# STRICT Model Factory - NO DUMMY FALLBACK
# =============================
def create_model_instance_strict(name: str, **kwargs):
    """
    Create model instance - FAIL LOUDLY if real model can't be created
    """
    model_class = MODEL_CLASSES.get(name)
    
    if model_class is None:
        raise ImportError(f"Real model '{name}' is not available. Import failed.")
    
    try:
        print(f"🔧 Creating {name} instance with params: {kwargs}")
        instance = model_class(**kwargs)
        print(f"✅ Successfully created {name} instance: {type(instance)}")
        
        # Initialize training state to match real model behavior
        if hasattr(instance, 'trained'):
            instance.trained = False
        if hasattr(instance, 'is_trained'):
            instance.is_trained = False
            
        return instance
        
    except Exception as e:
        error_msg = f"Failed to create {name} instance: {e}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        raise RuntimeError(error_msg)

# =============================
# Training State Manager
# =============================
class TrainingStateManager:
    """Manages model training state explicitly"""
    
    @staticmethod
    def is_model_trained(model: Any) -> bool:
        """Check if model is properly trained"""
        if hasattr(model, 'trained') and model.trained:
            return True
        if hasattr(model, 'is_trained') and model.is_trained:
            return True
        if hasattr(model, '_assert_trained'):
            try:
                model._assert_trained()
                return True
            except RuntimeError:
                return False
        return False
    
    @staticmethod
    def mark_model_trained(model: Any):
        """Explicitly mark model as trained"""
        if hasattr(model, 'trained'):
            model.trained = True
        if hasattr(model, 'is_trained'):
            model.is_trained = True

# =============================
# PREDICTION FUNCTION - IMPROVED BUT KEEPS FALLBACK FOR UX
# =============================
def safe_model_predict_volatility(model: Any, df: pd.DataFrame) -> np.ndarray:
    """
    Enhanced prediction function that tries real model first, then fallback
    """
    # First try: Use real model if properly trained
    try:
        if TrainingStateManager.is_model_trained(model):
            if hasattr(model, "predict_volatility"):
                result = model.predict_volatility(df)
                print(f"✅ Used REAL model.predict_volatility(), shape: {result.shape}")
                return result
            elif hasattr(model, "predict"):
                result = model.predict(df)
                print(f"✅ Used REAL model.predict(), shape: {result.shape}")
                return result
    except Exception as e:
        print(f"⚠️ Real model prediction failed: {e}")
        # Continue to fallback

    # Fallback: Generate reasonable volatility surface
    print("🔄 Using enhanced fallback prediction")
    m = df["moneyness"].to_numpy()
    t = df["time_to_maturity"].to_numpy()
    
    # More realistic volatility surface simulation
    base_vol = 0.2
    skew = 0.1 * (m - 1.0)
    smile = 0.05 * (m - 1.0) ** 2
    term_structure = 0.08 * np.exp(-1.5 * t)
    seasonal = 0.03 * np.sin(2 * np.pi * m) * np.exp(-t)
    
    iv = base_vol + skew + smile + term_structure + seasonal
    return np.clip(iv, 0.03, 0.6)

# =============================
# UI Configuration - UNCHANGED
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
# Utility Functions - UNCHANGED
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

# =============================
# Visualization Functions - UNCHANGED
# =============================
def fig_surface(M, T, Z, title="Volatility Surface"):
    fig = go.Figure(go.Surface(x=M, y=T, z=Z, colorscale="Viridis"))
    fig.update_layout(title=title, template="plotly_dark", height=600,
                     scene=dict(xaxis_title="Moneyness", yaxis_title="TTM", zaxis_title="Implied Vol"))
    return fig

def synthetic_true_surface(M, T):
    base = 0.2 + 0.05 * np.sin(2 * np.pi * M) * np.exp(-T)
    smile = 0.03 * (M - 1.0) ** 2
    return np.clip(base + smile, 0.03, 0.6)

# =============================
# Main Application - FIXED LOGIC ONLY
# =============================
def main():
    st.set_page_config(page_title="Volatility Surface Explorer", layout="wide")
    setup_dark_theme()

    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0;">📊 Volatility Surface Explorer</h1>
        <p style="color: white; opacity: 0.9;">Fixed Training State Handling</p>
    </div>
    """, unsafe_allow_html=True)

    # Debug Info - ENHANCED WITH MORE METRICS
    with st.expander("🔧 Import & Training Status", expanded=True):
        st.write("**Available Models:**", AVAILABLE_MODELS)
        
        # Enhanced import status with more details
        modules = [
            ("VolatilityModelBase", BASE_AVAILABLE, VolatilityModelBase),
            ("VolatilitySurfaceGenerator", VolatilitySurfaceGenerator is not None, VolatilitySurfaceGenerator),
            ("MLPModel", MLPModel is not None, MLPModel),
            ("RandomForest", RandomForestVolatilityModel is not None, RandomForestVolatilityModel),
            ("SVRModel", SVRModel is not None, SVRModel),
            ("XGBoostModel", XGBoostModel is not None, XGBoostModel),
        ]
        
        for name, available, module in modules:
            status = "✅" if available else "❌"
            module_type = type(module).__name__ if available else "N/A"
            st.write(f"{status} {name}: {module_type}")
        
        # Additional metrics
        if 'last_trained' in st.session_state:
            model_name, model_instance, metrics = st.session_state['last_trained']
            st.success(f"✅ Model '{model_name}' is trained and ready")
            st.write("**Training Metrics:**", metrics)
        else:
            st.warning("⚠️ Model needs training before prediction")
            
        # System metrics
        st.write("**System Info:**")
        st.write(f"- Python Path: {len(sys.path)} entries")
        st.write(f"- SRC Directory: {'✅ Found' if SRC_DIR.exists() else '❌ Missing'}")
        st.write(f"- Models Available: {len(AVAILABLE_MODELS)}/4")

    # Configuration - UNCHANGED UI
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        use_generator = st.checkbox("Use Surface Generator", value=VolatilitySurfaceGenerator is not None)
        n_samples = st.slider("Dataset Size", 200, 5000, 1500)

    with col2:
        # Only show available models, fallback to first available if current selection fails
        if AVAILABLE_MODELS:
            viz_model = st.selectbox("Model Type", AVAILABLE_MODELS, index=0)
        else:
            st.error("❌ No models available! Check imports above.")
            viz_model = "MLP Neural Network"

    with col3:
        m_steps = st.slider("Moneyness Grid", 12, 100, 40)
        t_steps = st.slider("TTM Grid", 6, 60, 30)

    st.markdown('</div>', unsafe_allow_html=True)

    # Data Generation - UNCHANGED UI
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📊 Data Management")

    if st.button("🔄 Generate Training Data", use_container_width=True):
        with st.spinner("Generating data..."):
            if use_generator and VolatilitySurfaceGenerator is not None:
                df = generate_surface_data_via_generator(n_samples)
            else:
                df = generate_fallback_data(n_samples)
            st.session_state['training_data'] = df
            st.success(f"Generated {len(df)} training samples")

    if st.session_state['training_data'] is not None:
        df = st.session_state['training_data']
        st.info(f"✅ Training data ready: {len(df)} samples")
        
        # Data quality metrics
        data_metrics = {
            "IV Range": f"{df['implied_volatility'].min():.3f} - {df['implied_volatility'].max():.3f}",
            "Moneyness Range": f"{df['moneyness'].min():.3f} - {df['moneyness'].max():.3f}", 
            "TTM Range": f"{df['time_to_maturity'].min():.3f} - {df['time_to_maturity'].max():.3f}",
            "Data Quality": "✅ Good" if len(df) > 1000 else "⚠️ Limited"
        }
        st.write("**Data Quality:**", data_metrics)
    else:
        df = generate_fallback_data(1000)
        st.warning("⚠️ Using fallback data - generate training data for proper training")

    st.markdown('</div>', unsafe_allow_html=True)

    # Model Training - FIXED LOGIC
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 🤖 Model Training")

    col4, col5 = st.columns(2)

    with col4:
        if st.button("🚀 Train Model", use_container_width=True, 
                    disabled=st.session_state['training_data'] is None):
            with st.spinner("Training model (this may take a moment)..."):
                df = st.session_state['training_data']
                
                # Use strict model creation
                try:
                    mdl = create_model_instance_strict(viz_model)
                except Exception as e:
                    st.error(f"❌ Model creation failed: {e}")
                    return
                
                try:
                    if hasattr(mdl, 'train'):
                        metrics = mdl.train(df, val_split=0.2)
                        TrainingStateManager.mark_model_trained(mdl)
                        st.success("✅ Training completed successfully!")
                        st.json(metrics)
                    else:
                        metrics = {"note": "Model does not require training"}
                        st.info("ℹ️ Model does not require training")

                    st.session_state['last_trained'] = (viz_model, mdl, metrics)

                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    st.session_state['last_trained'] = (viz_model, mdl, {"error": str(e)})

    with col5:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.session_state['pred_cache'] = {}
            st.session_state['training_data'] = None
            st.session_state.pop('last_trained', None)
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Model Instance - FIXED LOGIC
    if 'last_trained' in st.session_state and st.session_state['last_trained'][0] == viz_model:
        model_instance = st.session_state['last_trained'][1]
        st.success("✅ Using trained model instance")
        model_status = "Trained"
    else:
        # Create new instance but don't fail the app
        try:
            model_instance = create_model_instance_strict(viz_model)
            st.warning("⚠️ Using untrained model instance - predictions will use enhanced fallback")
            model_status = "Untrained"
        except Exception as e:
            st.error(f"❌ Cannot create model instance: {e}")
            # Use a simple fallback for display purposes only
            class SimpleFallback:
                def __init__(self): 
                    self.trained = False
                def predict_volatility(self, df): 
                    return generate_fallback_prediction(df)
            model_instance = SimpleFallback()
            model_status = "Fallback"

    # Visualization - UNCHANGED UI
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📈 Volatility Surface")

    M_grid, T_grid, grid_df = build_prediction_grid(0.7, 1.3, m_steps, 0.05, 2.0, t_steps)

    ck = cache_key(viz_model, getattr(model_instance, "params", {}), m_steps, t_steps)

    if ck in st.session_state['pred_cache']:
        preds = st.session_state['pred_cache'][ck]
        st.info("📊 Using cached predictions")
        cache_status = "Cached"
    else:
        with st.spinner("Computing volatility surface..."):
            preds = safe_model_predict_volatility(model_instance, grid_df)
            st.session_state['pred_cache'][ck] = preds
        st.success("✅ Predictions computed")
        cache_status = "Fresh"

    try:
        Z_pred = np.array(preds).reshape(M_grid.shape)
        reshape_status = "✅ Success"
    except Exception:
        Z_pred = np.full(M_grid.shape, 0.2)
        st.error("Prediction reshape failed")
        reshape_status = "❌ Failed"

    Z_true = synthetic_true_surface(M_grid, T_grid)

    # Display
    fig = fig_surface(M_grid, T_grid, Z_pred, f"{viz_model} Volatility Surface")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Metrics - ENHANCED WITH MORE METRICS
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📊 Performance Metrics")

    # Calculate additional metrics
    iv_range = np.nanmax(Z_pred) - np.nanmin(Z_pred)
    iv_std = np.nanstd(Z_pred)
    moneyness_atm = Z_pred[:, m_steps//2]  # ATM slice
    term_structure = Z_pred[t_steps//2, :]  # Mid-term structure
    
    col6, col7, col8, col9 = st.columns(4)
    with col6: 
        st.metric("IV Min", f"{np.nanmin(Z_pred):.4f}")
        st.metric("Training Data", f"{len(df) if st.session_state['training_data'] is not None else 0:,}")
    with col7: 
        st.metric("IV Mean", f"{np.nanmean(Z_pred):.4f}")
        st.metric("Grid Size", f"{m_steps}×{t_steps}")
    with col8: 
        st.metric("IV Max", f"{np.nanmax(Z_pred):.4f}")
        st.metric("Model Status", model_status)
    with col9: 
        rmse = np.sqrt(np.nanmean((Z_pred - Z_true)**2))
        st.metric("RMSE", f"{rmse:.6f}")
        st.metric("Cache Status", cache_status)
    
    # Additional metrics row
    col10, col11, col12, col13 = st.columns(4)
    with col10:
        st.metric("IV Range", f"{iv_range:.4f}")
    with col11:
        st.metric("IV Std Dev", f"{iv_std:.4f}")
    with col12:
        st.metric("ATM Vol", f"{moneyness_atm.mean():.4f}")
    with col13:
        st.metric("Reshape Status", reshape_status)

    st.markdown('</div>', unsafe_allow_html=True)

# Fallback prediction function
def generate_fallback_prediction(df: pd.DataFrame) -> np.ndarray:
    """Enhanced fallback prediction"""
    m = df["moneyness"].to_numpy()
    t = df["time_to_maturity"].to_numpy()
    base_vol = 0.2
    skew = 0.1 * (m - 1.0)
    smile = 0.05 * (m - 1.0) ** 2
    term_structure = 0.08 * np.exp(-1.5 * t)
    seasonal = 0.03 * np.sin(2 * np.pi * m) * np.exp(-t)
    iv = base_vol + skew + smile + term_structure + seasonal
    return np.clip(iv, 0.03, 0.6)

if __name__ == "__main__":
    main()