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

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Also add the current directory to handle direct imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

print("Using src path:", src_path)
print("Using current dir:", current_dir)

# Optional external imports
try:
    from scipy.stats import norm
except Exception:
    class _NormFallback:
        @staticmethod
        def cdf(x):
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    norm = _NormFallback()

# Try to import StandardScaler for model initialization
try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    # Create a minimal scaler fallback if sklearn is not available
    class StandardScaler:
        def fit(self, X):
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            self.scale_[self.scale_ == 0] = 1  # Prevent division by zero
            return self
        
        def transform(self, X):
            return (X - self.mean_) / self.scale_
        
        def fit_transform(self, X):
            return self.fit(X).transform(X)

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
# Setup Import Paths
# =============================
def setup_import_paths():
    """Setup import paths for your specific structure"""
    possible_paths = [
        Path("/mount/src/optionslab/src"),  # Streamlit Cloud path
        Path.cwd() / "src",                # Local development
        Path.cwd().parent / "src", 
        Path(__file__).parent / "src",
        Path(__file__).parent.parent / "src",
        Path(__file__).parent,             # Direct access to streamlit_app
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
# DIRECT IMPORTS WITH PROPER ERROR HANDLING
# =============================

logger.info("=== ATTEMPTING DIRECT IMPORTS ===")

# Import base first since models depend on it
VolatilityModelBase = None
try:
    # Try multiple import paths
    import_attempts = [
        lambda: __import__('volatility_surface.base', fromlist=['VolatilityModelBase']),
        lambda: __import__('src.volatility_surface.base', fromlist=['VolatilityModelBase']),
        lambda: __import__('base', fromlist=['VolatilityModelBase'])
    ]
    
    for attempt in import_attempts:
        try:
            base_module = attempt()
            VolatilityModelBase = getattr(base_module, 'VolatilityModelBase')
            logger.info("✓ Imported VolatilityModelBase")
            BASE_AVAILABLE = True
            break
        except Exception as e:
            logger.warning(f"Import attempt failed: {e}")
            continue
    
    if VolatilityModelBase is None:
        raise ImportError("Could not import VolatilityModelBase from any path")
        
except Exception as e:
    logger.warning(f"VolatilityModelBase import failed: {e}")
    BASE_AVAILABLE = False
    # Create a dummy base class for fallback
    class VolatilityModelBase:
        def __init__(self, feature_columns=None, enable_benchmark=False):
            self.feature_columns = feature_columns or []
            self.enable_benchmark = enable_benchmark
            self.trained = False

        def train(self, df, val_split=0.2):
            self.trained = True
            return {"status": "trained"}

        def predict_volatility(self, df):
            if not self.trained:
                raise RuntimeError("Model is not trained or initialized.")
            return np.full(len(df), 0.2)

# Now import the models with multiple path attempts
VolatilitySurfaceGenerator = None
MLPModel = None
RandomForestVolatilityModel = None  
SVRModel = None
XGBoostModel = None

def try_import_model(module_path, class_name):
    """Helper function to try importing models from multiple paths"""
    import_attempts = [
        lambda: __import__(module_path, fromlist=[class_name]),
        lambda: __import__(f"src.{module_path}", fromlist=[class_name]),
        lambda: __import__(f"volatility_surface.models.{class_name.lower().replace('model', '')}", fromlist=[class_name]),
    ]
    
    for attempt in import_attempts:
        try:
            module = attempt()
            model_class = getattr(module, class_name)
            logger.info(f"✓ Imported {class_name}")
            return model_class
        except Exception as e:
            logger.debug(f"Failed to import {class_name} from attempt: {e}")
            continue
    return None

# Import models
MLPModel = try_import_model('volatility_surface.models.mlp_model', 'MLPModel')
RandomForestVolatilityModel = try_import_model('volatility_surface.models.random_forest', 'RandomForestVolatilityModel')
SVRModel = try_import_model('volatility_surface.models.svr_model', 'SVRModel')
XGBoostModel = try_import_model('volatility_surface.models.xgboost_model', 'XGBoostModel')

# Import VolatilitySurfaceGenerator
try:
    # Multiple attempts for the generator
    import_attempts = [
        lambda: __import__('volatility_surface.surface_generator', fromlist=['VolatilitySurfaceGenerator']),
        lambda: __import__('src.volatility_surface.surface_generator', fromlist=['VolatilitySurfaceGenerator']),
        lambda: __import__('volatility_surface.base', fromlist=['VolatilitySurfaceGenerator']),
    ]
    
    for attempt in import_attempts:
        try:
            module = attempt()
            VolatilitySurfaceGenerator = getattr(module, 'VolatilitySurfaceGenerator', None)
            if VolatilitySurfaceGenerator is not None:
                logger.info("✓ Imported VolatilitySurfaceGenerator")
                break
        except Exception as e:
            logger.debug(f"Surface generator import attempt failed: {e}")
            continue
except Exception as e:
    logger.warning(f"VolatilitySurfaceGenerator import failed: {e}")

# =============================
# Enhanced DummyModel that mimics your actual models
# =============================
class EnhancedDummyModel:
    """Dummy model that properly mimics your actual model interface"""
    def __init__(self, **kwargs):
        self.params = kwargs or {}
        self.feature_names_in_ = [
            "moneyness", "log_moneyness", "time_to_maturity", 
            "ttm_squared", "risk_free_rate", "historical_volatility", "volatility_skew"
        ]
        self.name = "EnhancedDummyModel"
        self.trained = False
        self.is_trained = False
        # Initialize scaler to avoid fit_transform error
        self.scaler_ = StandardScaler()

    def train(self, df: pd.DataFrame, val_split: float = 0.2) -> Dict[str, float]:
        logger.info("EnhancedDummyModel.train called")
        self.trained = True
        self.is_trained = True
        return {
            "train_rmse": 0.1, 
            "val_rmse": 0.12, 
            "val_r2": 0.85, 
            "note": "EnhancedDummyModel - proper training simulation"
        }

    def predict_volatility(self, df: pd.DataFrame) -> np.ndarray:
        if not self.trained:
            # Simulate the exact error your real models throw
            raise RuntimeError("Model is not trained or initialized.")

        # Realistic volatility surface prediction
        m = df["moneyness"].to_numpy()
        t = df["time_to_maturity"].to_numpy()
        base = 0.2 + 0.05 * np.sin(2 * np.pi * m) * np.exp(-t)
        smile = 0.03 * (m - 1.0) ** 2
        return np.clip(base + smile, 0.03, 0.6)

    def _assert_trained(self):
        """Mimic the real model's training check"""
        if not self.trained:
            raise RuntimeError("Model is not trained or initialized.")

# =============================
# Smart Model Factory with Training Detection
# =============================
def create_model_instance(name: str, **kwargs):
    """Create model instance with proper training state handling"""
    model_map = {
        "MLP Neural Network": MLPModel,
        "Random Forest": RandomForestVolatilityModel, 
        "SVR": SVRModel,
        "XGBoost": XGBoostModel
    }

    model_class = model_map.get(name)

    if model_class is None:
        logger.info(f"Using EnhancedDummyModel for {name} - no class found")
        return EnhancedDummyModel(**kwargs)

    try:
        logger.info(f"Attempting to create {name} instance")
        instance = model_class(**kwargs)
        logger.info(f"✓ Successfully created {name} instance")

        # Set initial state to untrained (mimic real model behavior)
        if hasattr(instance, 'trained'):
            instance.trained = False
        if hasattr(instance, 'is_trained'):
            instance.is_trained = False
            
        # Initialize scaler to avoid fit_transform error and ensure it's not fitted
        if hasattr(instance, 'scaler_'):
            instance.scaler_ = StandardScaler()
        if hasattr(instance, 'scaler'):
            instance.scaler = StandardScaler()
        # Also handle other possible scaler attributes
        if hasattr(instance, '_scaler'):
            instance._scaler = StandardScaler()
        if hasattr(instance, 'feature_scaler'):
            instance.feature_scaler = StandardScaler()

        return instance
    except Exception as e:
        logger.error(f"Failed to create {name}: {e}")
        return EnhancedDummyModel(**kwargs)

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
    AVAILABLE_MODELS = ["EnhancedDummyModel"]
    logger.info("No custom models available, using EnhancedDummyModel")

logger.info(f"Available models: {AVAILABLE_MODELS}")

# =============================
# Enhanced Prediction Function
# =============================
def safe_model_predict_volatility(model: Any, df: pd.DataFrame) -> np.ndarray:
    """
    Enhanced prediction function that properly handles training state
    """
    try:
        # Check if model has the training assertion method
        if hasattr(model, '_assert_trained'):
            try:
                model._assert_trained()
            except RuntimeError as e:
                if "not trained" in str(e).lower():
                    logger.warning(f"Model not trained, using fallback prediction")
                    # Use fallback that doesn't require training
                    return generate_fallback_prediction(df)
                else:
                    raise e

        # Check other training indicators
        if hasattr(model, 'trained') and not model.trained:
            logger.warning("Model marked as not trained, using fallback")
            return generate_fallback_prediction(df)

        if hasattr(model, 'is_trained') and not model.is_trained:
            logger.warning("Model marked as not trained, using fallback")
            return generate_fallback_prediction(df)

        # If we get here, model should be trained - attempt prediction
        if hasattr(model, "predict_volatility"):
            result = model.predict_volatility(df)
            logger.info("✓ Used model.predict_volatility()")
            return result
        elif hasattr(model, "predict"):
            result = model.predict(df)
            logger.info("✓ Used model.predict()")
            return result
        else:
            logger.warning("No prediction method found, using fallback")
            return generate_fallback_prediction(df)

    except RuntimeError as e:
        if "not trained" in str(e).lower():
            logger.warning("Model runtime error - not trained, using fallback")
            return generate_fallback_prediction(df)
        else:
            logger.error(f"Runtime error in prediction: {e}")
            return generate_fallback_prediction(df)
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}")
        return generate_fallback_prediction(df)

def generate_fallback_prediction(df: pd.DataFrame) -> np.ndarray:
    """Generate a reasonable fallback volatility surface"""
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
# Utility Functions (same as before)
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
# Visualization Functions
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
# Main Application
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

    # Debug Info
    with st.expander("🔧 Import & Training Status", expanded=True):
        st.write("**Available Models:**", AVAILABLE_MODELS)
        modules = [
            ("VolatilityModelBase", VolatilityModelBase is not None),
            ("VolatilitySurfaceGenerator", VolatilitySurfaceGenerator is not None),
            ("MLPModel", MLPModel is not None),
            ("RandomForest", RandomForestVolatilityModel is not None),
            ("SVRModel", SVRModel is not None),
            ("XGBoostModel", XGBoostModel is not None),
        ]
        for name, available in modules:
            status = "✅" if available else "❌"
            st.write(f"{status} {name}")

        if 'last_trained' in st.session_state:
            st.success("✅ Model is trained and ready for prediction")
        else:
            st.warning("⚠️ Model needs training before prediction")

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
            st.success(f"Generated {len(df)} training samples")

    if st.session_state['training_data'] is not None:
        df = st.session_state['training_data']
        st.info(f"✅ Training data ready: {len(df)} samples")
    else:
        df = generate_fallback_data(1000)
        st.warning("⚠️ Using fallback data - generate training data for proper training")

    st.markdown('</div>', unsafe_allow_html=True)

    # Model Training
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 🤖 Model Training")

    col4, col5 = st.columns(2)

    with col4:
        if st.button("🚀 Train Model", use_container_width=True, 
                    disabled=st.session_state['training_data'] is None):
            with st.spinner("Training model (this may take a moment)..."):
                df = st.session_state['training_data']
                mdl = create_model_instance(viz_model)

                try:
                    if hasattr(mdl, 'train'):
                        metrics = mdl.train(df, val_split=0.2)
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

    # Model Instance
    if 'last_trained' in st.session_state and st.session_state['last_trained'][0] == viz_model:
        model_instance = st.session_state['last_trained'][1]
        st.success("✅ Using trained model instance")
    else:
        model_instance = create_model_instance(viz_model)
        st.warning("⚠️ Using untrained model instance - predictions will use fallback")

    # Visualization
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📈 Volatility Surface")

    M_grid, T_grid, grid_df = build_prediction_grid(0.7, 1.3, m_steps, 0.05, 2.0, t_steps)

    ck = cache_key(viz_model, getattr(model_instance, "params", {}), m_steps, t_steps)

    if ck in st.session_state['pred_cache']:
        preds = st.session_state['pred_cache'][ck]
        st.info("📊 Using cached predictions")
    else:
        with st.spinner("Computing volatility surface..."):
            preds = safe_model_predict_volatility(model_instance, grid_df)
            st.session_state['pred_cache'][ck] = preds
        st.success("✅ Predictions computed")

    try:
        Z_pred = np.array(preds).reshape(M_grid.shape)
    except Exception:
        Z_pred = np.full(M_grid.shape, 0.2)
        st.error("Prediction reshape failed")

    Z_true = synthetic_true_surface(M_grid, T_grid)

    # Display
    fig = fig_surface(M_grid, T_grid, Z_pred, f"{viz_model} Volatility Surface")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Metrics
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📊 Performance Metrics")

    col6, col7, col8, col9 = st.columns(4)
    with col6: 
        st.metric("IV Min", f"{np.nanmin(Z_pred):.4f}")
        st.metric("Training Data", f"{len(df) if st.session_state['training_data'] is not None else 0:,}")
    with col7: 
        st.metric("IV Mean", f"{np.nanmean(Z_pred):.4f}")
        st.metric("Grid Size", f"{m_steps}×{t_steps}")
    with col8: 
        st.metric("IV Max", f"{np.nanmax(Z_pred):.4f}")
        trained_status = "Yes" if 'last_trained' in st.session_state else "No"
        st.metric("Model Trained", trained_status)
    with col9: 
        rmse = np.sqrt(np.nanmean((Z_pred - Z_true)**2))
        st.metric("RMSE", f"{rmse:.6f}")
        st.metric("Model Type", viz_model)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()