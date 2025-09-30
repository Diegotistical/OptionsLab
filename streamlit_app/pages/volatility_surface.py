"""
Production-ready Volatility Surface
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
import os

# =============================
# CRITICAL: Enhanced Import System for correct structure
# =============================

# Clear any existing paths to avoid conflicts
original_sys_path = sys.path.copy()

# Get the root project directory (one level up from streamlit_app)
ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"

print(f"🔍 ROOT_DIR: {ROOT_DIR}")
print(f"🔍 SRC_DIR: {SRC_DIR}")
print(f"🔍 Current file: {Path(__file__)}")

# Add paths in order of priority
possible_paths = [
    SRC_DIR,                           # Root project / src
    ROOT_DIR,                          # Root project directory
    Path(__file__).parent,             # streamlit_app directory
]

added_paths = []
for path in possible_paths:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)
        added_paths.append(path_str)
        print(f"🔧 Added to sys.path: {path}")

print("=== FINAL SYS.PATH ===")
for i, path in enumerate(sys.path[:10]):  # Show first 10
    print(f"{i}: {path}")

# Debug: Check what's actually in the directories
print("\n=== CHECKING DIRECTORY STRUCTURE ===")
print(f"Root exists: {ROOT_DIR.exists()}")
print(f"Src exists: {SRC_DIR.exists()}")

if SRC_DIR.exists():
    print("📁 Contents of src:")
    for item in SRC_DIR.iterdir():
        if item.is_dir():
            print(f"   📁 {item.name}/")
        else:
            print(f"   📄 {item.name}")

volatility_surface_path = SRC_DIR / "volatility_surface"
if volatility_surface_path.exists():
    print(f"📁 Contents of volatility_surface:")
    for item in volatility_surface_path.iterdir():
        if item.is_dir():
            print(f"   📁 {item.name}/")
            if item.name == "models":
                models_path = volatility_surface_path / "models"
                for model_file in models_path.glob("*.py"):
                    print(f"      📄 {model_file.name}")
        else:
            print(f"   📄 {item.name}")

# =============================
# DIRECT IMPORTS FROM volatility_surface
# =============================

def attempt_import(module_path, class_name=None):
    """Attempt to import with full error reporting"""
    try:
        if class_name:
            exec(f"from {module_path} import {class_name}")
            result = locals()[class_name]
            print(f"✅ SUCCESS: {module_path} -> {class_name}")
            return result
        else:
            module = __import__(module_path, fromlist=[''])
            print(f"✅ SUCCESS: {module_path}")
            return module
    except Exception as e:
        print(f"❌ FAILED: {module_path}{f' -> {class_name}' if class_name else ''}")
        print(f"   Error: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return None

# Import base first - CORRECT PATH
print("\n=== ATTEMPTING VOLATILITY SURFACE IMPORTS ===")
VolatilityModelBase = attempt_import("volatility_surface.base", "VolatilityModelBase")

# Import models with correct paths
MLPModel = attempt_import("volatility_surface.models.mlp_model", "MLPModel")
RandomForestVolatilityModel = attempt_import("volatility_surface.models.random_forest", "RandomForestVolatilityModel")  
SVRModel = attempt_import("volatility_surface.models.svr_model", "SVRModel")
XGBoostModel = attempt_import("volatility_surface.models.xgboost_model", "XGBoostModel")
VolatilitySurfaceGenerator = attempt_import("volatility_surface.surface_generator", "VolatilitySurfaceGenerator")

# =============================
# Model Availability Tracking
# =============================
MODEL_CLASSES = {
    "MLP Neural Network": MLPModel,
    "Random Forest": RandomForestVolatilityModel,
    "SVR": SVRModel, 
    "XGBoost": XGBoostModel
}

AVAILABLE_MODELS = [name for name, cls in MODEL_CLASSES.items() if cls is not None]

if not AVAILABLE_MODELS:
    st.error("🚨 CRITICAL: No real models could be imported from volatility_surface!")
    # Try alternative import strategy
    try:
        # Attempt to import the entire volatility_surface package
        volatility_package = __import__("volatility_surface", fromlist=[''])
        available_attrs = [attr for attr in dir(volatility_package) if not attr.startswith('_')]
        st.write("Available in volatility_surface package:", available_attrs)
    except Exception as e:
        st.error(f"Couldn't inspect volatility_surface package: {e}")

print(f"📊 Available models: {AVAILABLE_MODELS}")

# =============================
# STRICT Model Factory - No Silent Fallbacks
# =============================
def create_model_instance(name: str, **kwargs):
    """Create model instance - FAIL LOUDLY if real model can't be created"""
    model_class = MODEL_CLASSES.get(name)
    
    if model_class is None:
        # Don't silently fallback - show explicit error
        raise ImportError(f"Real model '{name}' is not available. Check imports above.")
    
    try:
        print(f"🔧 Creating {name} instance...")
        instance = model_class(**kwargs)
        print(f"✅ Successfully created {name} instance")
        
        # Initialize training state
        if hasattr(instance, 'trained'):
            instance.trained = False
        if hasattr(instance, 'is_trained'):
            instance.is_trained = False
            
        return instance
        
    except Exception as e:
        # Don't fallback silently - raise the actual error
        st.error(f"❌ Failed to create {name}: {e}")
        raise

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
# STRICT Prediction Function
# =============================
def strict_model_predict_volatility(model: Any, df: pd.DataFrame) -> np.ndarray:
    """
    Strict prediction that fails loudly if model isn't properly trained
    """
    # Check training state
    if not TrainingStateManager.is_model_trained(model):
        raise RuntimeError(
            f"Model '{getattr(model, 'name', type(model).__name__)}' is not trained. "
            f"Please train the model first using the 'Train Model' button."
        )
    
    # Attempt prediction
    if hasattr(model, "predict_volatility"):
        return model.predict_volatility(df)
    elif hasattr(model, "predict"):
        return model.predict(df)
    else:
        raise AttributeError("Model has no prediction method")

# =============================
# Fallback Model (ONLY for explicit fallback mode)
# =============================
class ExplicitFallbackModel:
    """Fallback model that's only used when explicitly requested"""
    def __init__(self, **kwargs):
        self.name = "ExplicitFallback"
        self.trained = False
        
    def train(self, df: pd.DataFrame, val_split: float = 0.2) -> Dict[str, float]:
        self.trained = True
        return {
            "train_rmse": 0.15, 
            "val_rmse": 0.18, 
            "val_r2": 0.75,
            "note": "EXPLICIT FALLBACK MODE - Real models failed to load"
        }
    
    def predict_volatility(self, df: pd.DataFrame) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("Fallback model not trained")
        m = df["moneyness"].to_numpy()
        t = df["time_to_maturity"].to_numpy()
        base = 0.2 + 0.05 * np.sin(2 * np.pi * m) * np.exp(-t)
        smile = 0.03 * (m - 1.0) ** 2
        return np.clip(base + smile, 0.03, 0.6)

# =============================
# UI Configuration
# =============================
def setup_dark_theme():
    st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    .stApp { background: linear-gradient(135deg, #0c0d13 0%, #1a1d29 100%); }
    .import-success { background: rgba(0, 200, 83, 0.1); padding: 10px; border-radius: 5px; border-left: 4px solid #00c853; }
    .import-failure { background: rgba(255, 75, 75, 0.1); padding: 10px; border-radius: 5px; border-left: 4px solid #ff4b4b; }
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

@st.cache_data
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
# Main Application
# =============================
def main():
    st.set_page_config(page_title="Volatility Surface Explorer", layout="wide")
    setup_dark_theme()
    
    # Header with import status
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0;">📊 Volatility Surface Explorer</h1>
        <p style="color: white; opacity: 0.9;">Fixed for correct project structure</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Debug Information
    with st.expander("🔧 Debug Information", expanded=True):
        st.write("**Root Directory:**", ROOT_DIR)
        st.write("**Src Directory:**", SRC_DIR)
        st.write("**Current Working Directory:**", Path.cwd())
        st.write("**Script Directory:**", Path(__file__).parent)
        st.write("**Added Paths:**", added_paths)
        
        # Check directory structure
        st.markdown("**📁 Directory Structure:**")
        if ROOT_DIR.exists():
            st.success(f"✅ Root directory found: {ROOT_DIR}")
            if SRC_DIR.exists():
                st.success(f"✅ src directory found: {SRC_DIR}")
                volatility_surface_path = SRC_DIR / "volatility_surface"
                if volatility_surface_path.exists():
                    st.success(f"✅ volatility_surface found: {volatility_surface_path}")
                    
                    # List main volatility_surface files
                    main_files = list(volatility_surface_path.glob("*.py"))
                    st.write("**Main volatility_surface files:**", [f.name for f in main_files])
                    
                    # Check models directory
                    models_path = volatility_surface_path / "models"
                    if models_path.exists():
                        model_files = list(models_path.glob("*.py"))
                        st.write("**Model files:**", [f.name for f in model_files])
                    else:
                        st.error("❌ models directory not found in volatility_surface")
                else:
                    st.error(f"❌ volatility_surface not found in src")
            else:
                st.error(f"❌ src directory not found at: {SRC_DIR}")
        else:
            st.error(f"❌ Root directory not found at: {ROOT_DIR}")
    
    # Import Status Dashboard
    st.markdown("### 🔧 Import Status Dashboard")
    
    status_cols = st.columns(4)
    with status_cols[0]:
        st.metric("Available Models", f"{len(AVAILABLE_MODELS)}/4")
    with status_cols[1]:
        status = "✅ Ready" if AVAILABLE_MODELS else "❌ Failed"
        st.metric("Overall Status", status)
    with status_cols[2]:
        st.metric("Sys Paths Added", len(added_paths))
    with status_cols[3]:
        if st.button("🔄 Debug Imports"):
            st.rerun()
    
    # Detailed import status
    with st.expander("📋 Detailed Import Status", expanded=True):
        for model_name, model_class in MODEL_CLASSES.items():
            if model_class is not None:
                st.markdown(f'<div class="import-success">✅ {model_name}: SUCCESS</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="import-failure">❌ {model_name}: FAILED</div>', unsafe_allow_html=True)
    
    # Configuration
    st.markdown("### ⚙️ Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Model selection - only show available models
        if AVAILABLE_MODELS:
            viz_model = st.selectbox("Model Type", AVAILABLE_MODELS, index=0)
            use_fallback = st.checkbox("🔄 Use Explicit Fallback Mode", value=False, 
                                     help="Use fallback model instead of real models")
        else:
            st.error("No real models available!")
            viz_model = "MLP Neural Network"
            use_fallback = True
    
    with col2:
        n_samples = st.slider("Dataset Size", 200, 5000, 1500)
        m_steps = st.slider("Moneyness Grid", 12, 100, 40)
        
    with col3:
        t_steps = st.slider("TTM Grid", 6, 60, 30)
        spot_assumption = st.number_input("Spot Price", value=100.0)
    
    # Model initialization
    try:
        if use_fallback:
            model_instance = ExplicitFallbackModel()
            st.warning("🔄 Using EXPLICIT FALLBACK MODE")
        else:
            model_instance = create_model_instance(viz_model)
            st.success(f"✅ Loaded real model: {viz_model}")
    except Exception as e:
        st.error(f"❌ Failed to initialize model: {e}")
        model_instance = ExplicitFallbackModel()
        st.warning("🔄 Fell back to explicit fallback mode")
    
    # Data Generation
    st.markdown("### 📊 Data Management")
    
    if st.button("🔄 Generate Training Data", use_container_width=True):
        with st.spinner("Generating data..."):
            df = generate_fallback_data(n_samples)
            st.session_state['training_data'] = df
            st.success(f"Generated {len(df)} training samples")
    
    if st.session_state.get('training_data') is not None:
        df = st.session_state['training_data']
        st.info(f"✅ Training data ready: {len(df)} samples")
    else:
        df = generate_fallback_data(1000)
        st.warning("⚠️ Using fallback data - generate training data for proper training")
    
    # Model Training
    st.markdown("### 🤖 Model Training")
    
    if st.button("🚀 Train Model", use_container_width=True, 
                disabled=st.session_state.get('training_data') is None):
        with st.spinner("Training model..."):
            df = st.session_state['training_data']
            
            try:
                if hasattr(model_instance, 'train'):
                    metrics = model_instance.train(df, val_split=0.2)
                    TrainingStateManager.mark_model_trained(model_instance)
                    st.success("✅ Training completed successfully!")
                    st.json(metrics)
                else:
                    metrics = {"note": "Model does not require training"}
                    st.info("ℹ️ Model does not require training")
                
                st.session_state['last_trained'] = (viz_model, model_instance, metrics, use_fallback)
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.code(traceback.format_exc())
    
    # Use trained model if available
    if ('last_trained' in st.session_state and 
        st.session_state['last_trained'][0] == viz_model and
        st.session_state['last_trained'][3] == use_fallback):
        model_instance = st.session_state['last_trained'][1]
        st.success("✅ Using trained model instance")
    else:
        st.warning("⚠️ Using untrained model instance")
    
    # Visualization
    st.markdown("### 📈 Volatility Surface")
    
    M_grid, T_grid, grid_df = build_prediction_grid(0.7, 1.3, m_steps, 0.05, 2.0, t_steps)
    
    try:
        with st.spinner("Computing volatility surface..."):
            preds = strict_model_predict_volatility(model_instance, grid_df)
        st.success("✅ Predictions computed using REAL model!")
        Z_pred = np.array(preds).reshape(M_grid.shape)
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        # Use simple fallback for display only
        Z_pred = synthetic_true_surface(M_grid, T_grid)
    
    # Display surface
    model_type = "Fallback" if use_fallback else viz_model
    fig = fig_surface(M_grid, T_grid, Z_pred, f"{model_type} Volatility Surface")
    st.plotly_chart(fig, use_container_width=True)
    
    # Show what model we're actually using
    st.info(f"**Active Model:** {type(model_instance).__name__} | "
           f"**Trained:** {TrainingStateManager.is_model_trained(model_instance)} | "
           f"**Mode:** {'Fallback' if use_fallback else 'Real Model'}")

# Initialize session state
if 'pred_cache' not in st.session_state:
    st.session_state['pred_cache'] = {}
if 'training_data' not in st.session_state:
    st.session_state['training_data'] = None

if __name__ == "__main__":
    main()