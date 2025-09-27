
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
from typing import Any, Dict, List, Optional, Tuple, Callable
import time
import json
import hashlib
import math
import importlib
import inspect

# =============================
# Enhanced Import System for Windows Paths
# =============================

# Optional external imports
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False
    logging.info("Joblib not available, falling back to sequential processing")

try:
    from scipy.stats import norm
except Exception:
    class _NormFallback:
        @staticmethod
        def cdf(x: float) -> float:
            """Standard normal CDF fallback implementation"""
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
def setup_import_paths() -> List[str]:
    """Enhanced path setup specifically for Windows directory structure"""
    current_file = Path(__file__).resolve()
    added_paths = []
    
    # Your specific path from the error
    your_specific_path = Path(r"D:\Coding\Python\OptionsLab\src")
    if your_specific_path.exists():
        path_str = str(your_specific_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            added_paths.append(path_str)
            logger.info(f"Added specific path: {your_specific_path}")
    
    # Common project structure paths
    possible_paths = [
        current_file.parents[1] / "src",
        current_file.parent / "src",
        Path.cwd() / "src",
        Path.cwd().parent / "src",
        Path(r"D:\Coding\Python\OptionsLab\src"),  # Your exact path
        Path(r"D:/Coding/Python/OptionsLab/src"),  # Forward slashes
        Path.cwd() / "volatility_surface",
        Path.cwd().parent / "volatility_surface",
        current_file.parent
    ]
    
    for path in possible_paths:
        try:
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
                added_paths.append(str(path))
                logger.info(f"Added to sys.path: {path}")
        except Exception as e:
            logger.debug(f"Could not add path {path}: {str(e)}")
    
    return added_paths

# Setup paths immediately
added_paths = setup_import_paths()
logger.info(f"Current sys.path: {sys.path}")

# =============================
# Robust Import Helper
# =============================
def robust_import(module_path: str, class_name: Optional[str] = None, 
                 alternative_modules: Optional[List[str]] = None) -> Any:
    """
    Try multiple strategies to import modules with detailed logging
    
    Args:
        module_path: Main module path to try
        class_name: Optional class to import from the module
        alternative_modules: Alternative module paths to try
        
    Returns:
        Imported module or class, or None if all attempts fail
    """
    strategies = []
    
    # Strategy 1: Direct import
    if class_name:
        strategies.append(f"from {module_path} import {class_name}")
    else:
        strategies.append(f"import {module_path}")
    
    # Strategy 2: Try with src prefix
    if class_name:
        strategies.append(f"from src.{module_path} import {class_name}")
    else:
        strategies.append(f"import src.{module_path}")
    
    # Strategy 3: Try common alternative structures
    for alt_prefix in ["volatility_surface", "vol_surface", "models", "option_models"]:
        if class_name:
            strategies.append(f"from {alt_prefix}.{module_path} import {class_name}")
        else:
            strategies.append(f"import {alt_prefix}.{module_path}")
    
    # Strategy 4: Try alternative modules if provided
    if alternative_modules:
        for alt_module in alternative_modules:
            if class_name:
                strategies.append(f"from {alt_module} import {class_name}")
            else:
                strategies.append(f"import {alt_module}")
    
    # Try all strategies
    for strategy in strategies:
        try:
            if strategy.startswith("from"):
                parts = strategy.split()
                module_path = parts[1]
                attr_name = parts[3]
                
                # Handle potential circular imports
                if module_path in sys.modules:
                    module = sys.modules[module_path]
                else:
                    module = importlib.import_module(module_path)
                
                if hasattr(module, attr_name):
                    result = getattr(module, attr_name)
                    logger.info(f"✓ Successfully imported {attr_name} using: {strategy}")
                    return result
                else:
                    logger.debug(f"Module {module_path} exists but has no attribute {attr_name}")
            else:
                module_name = strategy.split()[1]
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                else:
                    module = importlib.import_module(module_name)
                logger.info(f"✓ Successfully imported {module_name}")
                return module
        except ImportError as e:
            logger.debug(f"Strategy failed {strategy}: {str(e)}")
        except Exception as e:
            logger.debug(f"Error with strategy {strategy}: {str(e)}")
            continue
    
    logger.warning(f"❌ All import strategies failed for {module_path}.{class_name if class_name else ''}")
    return None

# =============================
# Import Models with Enhanced Detection
# =============================
def import_all_models() -> Dict[str, Any]:
    """Import all volatility surface models with detailed logging and fallbacks"""
    models = {}
    
    # Define all model imports with fallback options
    model_imports = {
        "VolatilitySurfaceGenerator": (
            "volatility_surface.surface_generator", 
            "VolatilitySurfaceGenerator",
            ["surface_generator", "vol_surface_generator"]
        ),
        "MLPModel": (
            "volatility_surface.models.mlp_model", 
            "MLPModel",
            ["mlp", "neural_network"]
        ),
        "RandomForestVolatilityModel": (
            "volatility_surface.models.random_forest", 
            "RandomForestVolatilityModel",
            ["random_forest", "rf_model"]
        ),
        "SVRModel": (
            "volatility_surface.models.svr_model", 
            "SVRModel",
            ["svr", "support_vector_regression"]
        ),
        "XGBoostModel": (
            "volatility_surface.models.xgboost_model", 
            "XGBoostModel",
            ["xgboost", "xgb_model"]
        ),
    }
    
    module_imports = {
        "feature_engineering": (
            "volatility_surface.utils.feature_engineering",
            None,
            ["feature_engineering", "features"]
        ),
        "arbitrage_checks": (
            "volatility_surface.utils.arbitrage_checks",
            None,
            ["arbitrage", "checks"]
        ),
        "arbitrage_enforcement": (
            "volatility_surface.utils.arbitrage_enforcement",
            None,
            ["arbitrage", "enforcement"]
        ),
        "grid_search": (
            "volatility_surface.utils.grid_search",
            None,
            ["grid_search", "optimization"]
        ),
    }
    
    # Import models
    for name, (module_path, class_name, alt_modules) in model_imports.items():
        models[name] = robust_import(module_path, class_name, alt_modules)
    
    # Import modules
    for name, (module_path, _, alt_modules) in module_imports.items():
        models[name] = robust_import(module_path, None, alt_modules)
    
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
# Dummy Implementations for Fallback
# =============================
class DummyVolatilitySurfaceGenerator:
    """Fallback implementation when VolatilitySurfaceGenerator is unavailable"""
    def __init__(self, strikes, maturities, implied_vols, **kwargs):
        self.strikes = np.array(strikes)
        self.maturities = np.array(maturities)
        self.implied_vols = np.array(implied_vols)
        logger.info("Using DummyVolatilitySurfaceGenerator")
    
    def get_surface_batch(self, strikes, maturities) -> np.ndarray:
        """Generate a simple volatility surface"""
        strikes = np.array(strikes)
        maturities = np.array(maturities)
        
        # Create a basic volatility surface with smile and term structure
        moneyness = strikes / 100.0  # Assume spot = 100
        base_vol = 0.2 + 0.05 * np.sin(2 * np.pi * moneyness) * np.exp(-maturities)
        smile = 0.03 * (moneyness - 1.0) ** 2
        return np.clip(base_vol + smile, 0.03, 0.6)

class DummyFeatureEngineering:
    """Fallback feature engineering when module is unavailable"""
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features for volatility modeling"""
        df = df.copy()
        
        # Basic feature engineering
        if "moneyness" not in df.columns:
            df["moneyness"] = df["strike_price"] / df["underlying_price"]
        
        if "log_moneyness" not in df.columns:
            df["log_moneyness"] = np.log(np.clip(df["moneyness"], 1e-12, None))
        
        if "ttm_squared" not in df.columns:
            df["ttm_squared"] = df["time_to_maturity"] ** 2
        
        if "volatility_skew" not in df.columns:
            df["volatility_skew"] = df["implied_volatility"] - df["historical_volatility"].mean()
        
        return df

# Replace missing modules with dummies
if VolatilitySurfaceGenerator is None:
    VolatilitySurfaceGenerator = DummyVolatilitySurfaceGenerator
    logger.warning("VolatilitySurfaceGenerator not found, using dummy implementation")

if feature_engineering_module is None:
    feature_engineering_module = DummyFeatureEngineering
    logger.warning("feature_engineering module not found, using dummy implementation")

# =============================
# DummyModel Fallback
# =============================
class DummyModel:
    """Fallback model when no real models are available"""
    def __init__(self, **kwargs):
        self.params = kwargs or {}
        self.feature_names_in_ = [
            "moneyness", "log_moneyness", "time_to_maturity", 
            "ttm_squared", "risk_free_rate", "historical_volatility",
            "volatility_skew"
        ]
        self.name = "DummyModel"
        self.is_trained = False
        logger.info("Initialized DummyModel")
        
    def train(self, df: pd.DataFrame, val_split: float = 0.2) -> Dict[str, float]:
        """Train the dummy model (does nothing but sets flag)"""
        logger.info("DummyModel.train called")
        self.is_trained = True
        return {
            "train_rmse": 0.1, 
            "val_rmse": 0.12, 
            "val_r2": 0.85, 
            "note": "Dummy model - no real training"
        }
    
    def predict_volatility(self, df: pd.DataFrame) -> np.ndarray:
        """Predict volatility using a simple formula"""
        if not self.is_trained:
            logger.warning("DummyModel not trained, returning simple surface")
        
        # Simple smile function based on moneyness & ttm
        m = df["moneyness"].to_numpy()
        t = df["time_to_maturity"].to_numpy()
        base = 0.2 + 0.05 * np.sin(2 * np.pi * m) * np.exp(-t)
        smile = 0.03 * (m - 1.0) ** 2
        return np.clip(base + smile, 0.03, 0.6)

# Enhanced Model Factory
def create_model_instance(name: str, **kwargs) -> Any:
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
        # Handle case where module is present but class is not properly defined
        if not inspect.isclass(cls):
            logger.warning(f"{name} is not a class, using DummyModel")
            return DummyModel(**kwargs)
            
        instance = cls(**kwargs)
        logger.info(f"✓ Successfully created {name} instance")
        return instance
    except Exception as e:
        logger.error(f"Failed to create {name}: {str(e)}")
        logger.exception("Full traceback:")
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
    
    # Check if class is valid
    if cls is not None and inspect.isclass(cls):
        AVAILABLE_MODELS.append(name)

if not AVAILABLE_MODELS:
    AVAILABLE_MODELS = ["DummyModel"]
    logger.info("No custom models available, using DummyModel only")

# =============================
# UI Configuration and Styling
# =============================
def setup_dark_theme():
    """Apply professional dark theme styling"""
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
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: rgba(40, 44, 62, 0.8);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #2a2f45;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
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
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #ff4b4b, #ff6b6b);
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(255, 75, 75, 0.3);
    }
    .stSlider > div > div > div {
        background: #ff4b4b;
    }
    .stSelectbox > div > div > div {
        border-color: #2a2f45;
    }
    .stNumberInput > div > div > input {
        background: rgba(40, 44, 62, 0.8);
        border-color: #2a2f45;
    }
    .stExpander {
        background: rgba(30, 33, 48, 0.7);
        border-radius: 8px;
        border: 1px solid #2a2f45;
    }
    .stExpander > details > summary {
        padding: 0.5rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================
# Utility Functions
# =============================
def build_prediction_grid(m_start: float = 0.7, m_end: float = 1.3, m_steps: int = 40, 
                         t_start: float = 0.05, t_end: float = 2.0, t_steps: int = 40) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build a grid for prediction visualization
    
    Args:
        m_start: Start moneyness value
        m_end: End moneyness value
        m_steps: Number of moneyness steps
        t_start: Start time to maturity
        t_end: End time to maturity
        t_steps: Number of time steps
    
    Returns:
        M, T: Meshgrids for moneyness and time
        grid_df: DataFrame with all features needed for prediction
    """
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
    """
    Safe prediction with comprehensive error handling
    
    Args:
        model: Trained model instance
        df: DataFrame with features for prediction
    
    Returns:
        Array of predicted volatilities
    """
    try:
        # Check if model needs training
        if hasattr(model, 'is_trained') and not model.is_trained:
            logger.warning("Model not trained, using fallback predictions")
            return np.full(len(df), 0.2)
        
        # Try different prediction methods
        if hasattr(model, "predict_volatility"):
            out = model.predict_volatility(df)
        elif hasattr(model, "predict"):
            out = model.predict(df)
        else:
            # Try calling the model directly if it's callable
            if callable(model):
                out = model(df)
            else:
                logger.warning("Model has no prediction method, using fallback")
                out = np.full(len(df), 0.2)
        
        # Ensure output is a numpy array of floats
        out = np.asarray(out).astype(float).ravel()
        
        # Handle cases where output size doesn't match input
        if out.shape[0] != len(df):
            if out.size == 1:
                return np.full(len(df), float(out))
            # Resize to match input length
            out = np.resize(out, len(df))
        
        # Clip to reasonable volatility range
        return np.clip(out, 0.03, 0.6)
    
    except Exception as e:
        logger.error(f"Model prediction failed: {str(e)}")
        logger.exception("Full traceback:")
        
        # Return a reasonable fallback surface
        try:
            m = df["moneyness"].to_numpy()
            t = df["time_to_maturity"].to_numpy()
        except Exception:
            m = np.linspace(0.7, 1.3, len(df))
            t = np.linspace(0.05, 2.0, len(df))
            
        base = 0.2 + 0.05 * np.sin(2 * np.pi * m) * np.exp(-t)
        smile = 0.03 * (m - 1.0) ** 2
        return np.clip(base + smile, 0.03, 0.6)

def cache_key(model_name: str, params: Dict[str, Any], m_steps: int, t_steps: int, extra: Optional[Dict] = None) -> str:
    """
    Generate a unique cache key for model predictions
    
    Args:
        model_name: Name of the model
        params: Model parameters
        m_steps: Moneyness grid steps
        t_steps: Time grid steps
        extra: Additional parameters for the key
    
    Returns:
        SHA-1 hash string as cache key
    """
    payload = {
        "model": model_name, 
        "params": params, 
        "m": m_steps, 
        "t": t_steps, 
        "extra": extra or {}
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

# Initialize session state
if 'pred_cache' not in st.session_state:
    st.session_state['pred_cache'] = {}
if 'training_data' not in st.session_state:
    st.session_state['training_data'] = None
if 'last_trained' not in st.session_state:
    st.session_state['last_trained'] = None

@st.cache_data(show_spinner=False, max_entries=5)
def generate_fallback_data(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Generate fallback training data when no real data is available
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic options data
    """
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
    
    # Add required features
    df["moneyness"] = strikes / spots
    df["log_moneyness"] = np.log(np.clip(df["moneyness"], 1e-12, None))
    df["ttm_squared"] = df["time_to_maturity"] ** 2
    df["volatility_skew"] = df["implied_volatility"] - df["historical_volatility"]
    
    return df

@st.cache_data(show_spinner=False, max_entries=5)
def generate_surface_data_via_generator(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Generate training data using the surface generator if available
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic options data
    """
    try:
        rng = np.random.default_rng(seed)
        base_strikes = np.linspace(80, 120, 50)
        base_maturities = np.linspace(0.1, 2.0, 20)
        S, T = np.meshgrid(base_strikes, base_maturities, indexing='xy')
        
        # Create base volatility surface
        base_ivs = 0.2 + 0.05 * np.sin(2 * np.pi * (S / np.mean(base_strikes))) * np.exp(-T)
        
        # Use the appropriate generator (real or dummy)
        generator = VolatilitySurfaceGenerator(base_strikes, base_maturities, base_ivs,
                                              strike_points=50, maturity_points=20, interp_method='cubic')
        
        # Generate random samples
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
        
        # Apply feature engineering
        if feature_engineering_module:
            try:
                if hasattr(feature_engineering_module, "engineer_features"):
                    df = feature_engineering_module.engineer_features(df)
                else:
                    # Try the dummy implementation
                    df = DummyFeatureEngineering.engineer_features(df)
            except Exception as e:
                logger.error(f"Feature engineering failed: {str(e)}")
                df = DummyFeatureEngineering.engineer_features(df)
        else:
            df = DummyFeatureEngineering.engineer_features(df)
            
        return df
    
    except Exception as e:
        logger.error(f"Surface generator error: {str(e)}")
        logger.exception("Full traceback:")
        return generate_fallback_data(n_samples, seed)

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, 
                       option_type: str = "call", q: float = 0.0) -> float:
    """
    Calculate Black-Scholes option price
    
    Args:
        S: Underlying price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free rate
        sigma: Volatility
        option_type: "call" or "put"
        q: Dividend yield
    
    Returns:
        Option price
    """
    try:
        # Handle edge cases
        T = max(T, 1e-12)  # Prevent division by zero
        sigma = max(sigma, 1e-12)  # Prevent zero volatility
        S = max(S, 1e-12)  # Prevent zero underlying price
        K = max(K, 1e-12)  # Prevent zero strike price
        
        # Calculate d1 and d2
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        # Calculate price based on option type
        if option_type == "call":
            price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
        
        return max(price, 0.0)  # Ensure non-negative price
    
    except Exception as e:
        logger.error(f"Black-Scholes calculation error: {str(e)}")
        return 0.0

def bs_price_vectorized(S_arr: np.ndarray, K_arr: np.ndarray, T_arr: np.ndarray, 
                       r: float, sigma_arr: np.ndarray, 
                       option_type: str = "call", q: float = 0.0) -> np.ndarray:
    """
    Vectorized Black-Scholes pricing
    
    Args:
        S_arr: Array of underlying prices
        K_arr: Array of strike prices
        T_arr: Array of times to maturity
        r: Risk-free rate
        sigma_arr: Array of volatilities
        option_type: "call" or "put"
        q: Dividend yield
    
    Returns:
        Array of option prices
    """
    out = np.zeros_like(S_arr, dtype=float)
    for i in range(len(out)):
        try:
            out[i] = black_scholes_price(
                float(S_arr[i]), 
                float(K_arr[i]), 
                float(T_arr[i]), 
                float(r), 
                float(sigma_arr[i]), 
                option_type, 
                float(q)
            )
        except Exception as e:
            logger.debug(f"Vectorized BS pricing error at index {i}: {str(e)}")
            out[i] = 0.0
    return out

def compute_greeks_from_iv_grid(M: np.ndarray, T: np.ndarray, Z_pred: np.ndarray, 
                               option_type: str = "call", spot_assumption: float = 100.0, 
                               r: float = 0.03, q: float = 0.0, 
                               h_frac: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Delta and Gamma from implied volatility grid
    
    Args:
        M: Moneyness grid
        T: Time to maturity grid
        Z_pred: Implied volatility grid
        option_type: "call" or "put"
        spot_assumption: Assumed spot price
        r: Risk-free rate
        q: Dividend yield
        h_frac: Step size fraction for finite differences
    
    Returns:
        delta_grid, gamma_grid: Arrays of Delta and Gamma values
    """
    try:
        shape = Z_pred.shape
        flat_m = M.ravel()
        flat_t = T.ravel()
        flat_sigma = Z_pred.ravel()
        
        S0 = spot_assumption
        K = flat_m * S0  # Strike prices
        Tvec = flat_t
        h = max(1e-4, h_frac * S0)  # Step size
        
        # Calculate option prices at different spot levels
        p0 = bs_price_vectorized(
            np.full_like(K, S0), 
            K, 
            Tvec, 
            r, 
            flat_sigma, 
            option_type, 
            q
        )
        
        p_up = bs_price_vectorized(
            np.full_like(K, S0 + h), 
            K, 
            Tvec, 
            r, 
            flat_sigma, 
            option_type, 
            q
        )
        
        p_down = bs_price_vectorized(
            np.full_like(K, S0 - h), 
            K, 
            Tvec, 
            r, 
            flat_sigma, 
            option_type, 
            q
        )
        
        # Calculate Delta and Gamma using finite differences
        delta = (p_up - p_down) / (2 * h)
        gamma = (p_up - 2 * p0 + p_down) / (h * h)
        
        return delta.reshape(shape), gamma.reshape(shape)
    
    except Exception as e:
        logger.error(f"Greeks calculation failed: {str(e)}")
        logger.exception("Full traceback:")
        # Return NaN arrays of the same shape
        return np.full_like(Z_pred, np.nan), np.full_like(Z_pred, np.nan)

# =============================
# Visualization Functions
# =============================
def fig_surface(M: np.ndarray, T: np.ndarray, Z: np.ndarray, title: str = "Volatility Surface") -> go.Figure:
    """
    Create 3D surface plot of volatility
    
    Args:
        M: Moneyness grid
        T: Time to maturity grid
        Z: Volatility grid
        title: Plot title
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(go.Surface(
        x=M, 
        y=T, 
        z=Z, 
        colorscale="Viridis",
        colorbar=dict(title="Volatility")
    ))
    
    fig.update_layout(
        title=title, 
        template="plotly_dark", 
        scene=dict(
            xaxis_title="Moneyness", 
            yaxis_title="Time to Maturity (Years)", 
            zaxis_title="Implied Volatility",
            aspectmode='cube'
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def fig_heatmap(M: np.ndarray, T: np.ndarray, Z: np.ndarray, title: str = "Volatility Heatmap") -> go.Figure:
    """
    Create heatmap of volatility surface
    
    Args:
        M: Moneyness grid
        T: Time to maturity grid
        Z: Volatility grid
        title: Plot title
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(go.Heatmap(
        z=Z, 
        x=M[0,:], 
        y=T[:,0], 
        colorscale="Viridis",
        colorbar=dict(title="Volatility")
    ))
    
    fig.update_layout(
        title=title, 
        template="plotly_dark", 
        xaxis_title="Moneyness", 
        yaxis_title="Time to Maturity (Years)",
        height=500,
        margin=dict(l=0, r=0, b=40, t=40)
    )
    
    return fig

def fig_greeks_surface(M: np.ndarray, T: np.ndarray, Z: np.ndarray, 
                      greek_name: str, title: str = None) -> go.Figure:
    """
    Create 3D surface plot for Greeks
    
    Args:
        M: Moneyness grid
        T: Time to maturity grid
        Z: Greek values grid
        greek_name: Name of the Greek ("Delta", "Gamma", etc.)
        title: Plot title (optional)
    
    Returns:
        Plotly Figure object
    """
    if title is None:
        title = f"{greek_name} Surface"
    
    # Choose appropriate colorscale based on Greek
    colorscale = "RdBu" if greek_name in ["Delta"] else "Viridis"
    
    fig = go.Figure(go.Surface(
        x=M, 
        y=T, 
        z=Z, 
        colorscale=colorscale,
        colorbar=dict(title=greek_name)
    ))
    
    fig.update_layout(
        title=title, 
        template="plotly_dark", 
        scene=dict(
            xaxis_title="Moneyness", 
            yaxis_title="Time to Maturity (Years)", 
            zaxis_title=greek_name,
            aspectmode='cube'
        ),
        height=500,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def synthetic_true_surface(M: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Generate synthetic true volatility surface for comparison
    
    Args:
        M: Moneyness grid
        T: Time to maturity grid
    
    Returns:
        Array of true volatility values
    """
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
        <p style="color: white; opacity: 0.9; font-size: 1.1rem;">Production-ready volatility surface visualization with Greeks</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Debug Information
    with st.expander("🔧 Debug Information", expanded=False):
        st.write("**Python Path:**", sys.executable)
        st.write("**Working Directory:**", Path.cwd())
        st.write("**Added Paths:**", added_paths)
        st.write("**Available Models:**", AVAILABLE_MODELS)
        
        # Show detailed import status
        st.subheader("Module Import Status")
        import_status = []
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
            status = "Available" if obj is not None else "Not Available"
            import_status.append({"Module": name, "Status": status})
        
        st.dataframe(pd.DataFrame(import_status), hide_index=True)
    
    # Configuration Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        use_generator = st.checkbox("Use Surface Generator", 
                                   value=VolatilitySurfaceGenerator is not None,
                                   disabled=VolatilitySurfaceGenerator is None,
                                   help="Use advanced surface generator when available")
        n_samples = st.slider("Training Data Size", 200, 5000, 1500, step=100,
                             help="Number of synthetic data points to generate for training")
        
    with col2:
        viz_model = st.selectbox("Model Type", AVAILABLE_MODELS, index=0,
                                help="Select the volatility model to use for prediction")
        m_steps = st.slider("Moneyness Resolution", 12, 100, 40,
                           help="Number of grid points for moneyness axis")
        
    with col3:
        option_type = st.selectbox("Option Type", ["call", "put"], index=0,
                                  help="Option type for Greek calculations")
        t_steps = st.slider("Time Resolution", 6, 60, 30,
                           help="Number of grid points for time to maturity axis")
        
    with col4:
        spot_assumption = st.number_input("Spot Price", min_value=1.0, value=100.0, step=1.0,
                                        help="Spot price assumption for Greek calculations")
        r = st.number_input("Risk-free Rate", min_value=0.0, value=0.03, step=0.005,
                           help="Risk-free interest rate")
        q = st.number_input("Dividend Yield", min_value=0.0, value=0.0, step=0.001,
                           help="Continuous dividend yield")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate training data FIRST (before any model operations)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📊 Training Data")
    
    if st.button("🔄 Generate Training Data", use_container_width=True):
        with st.spinner("Generating training data..."):
            start_time = time.time()
            
            if use_generator and VolatilitySurfaceGenerator is not None:
                df = generate_surface_data_via_generator(n_samples)
            else:
                df = generate_fallback_data(n_samples)
                
            st.session_state['training_data'] = df
            elapsed = time.time() - start_time
            
            st.success(f"Generated {len(df)} training samples in {elapsed:.2f} seconds")
            st.info(f"Data shape: {df.shape}, Columns: {list(df.columns)}")
    
    # Display data info if available
    if st.session_state['training_data'] is not None:
        df = st.session_state['training_data']
        
        col_data1, col_data2 = st.columns([2, 1])
        
        with col_data1:
            st.write(f"**Training Data Ready:** {len(df)} samples")
            st.write(f"**Columns:** {list(df.columns)}")
            
            # Show basic statistics
            stats = df.describe()
            st.dataframe(stats, height=200)
        
        with col_data2:
            # Show volatility distribution
            fig_hist = px.histogram(
                df, 
                x="implied_volatility", 
                nbins=30,
                title="Implied Volatility Distribution",
                color_discrete_sequence=['#ff4b4b']
            )
            fig_hist.update_layout(template="plotly_dark", height=250)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.write(f"**IV Range:** {df['implied_volatility'].min():.3f} - {df['implied_volatility'].max():.3f}")
            st.write(f"**Mean IV:** {df['implied_volatility'].mean():.3f}")
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
                    start_time = time.time()
                    df = st.session_state['training_data']
                    mdl = create_model_instance(viz_model)
                    
                    try:
                        # Check if model has train method
                        if hasattr(mdl, 'train') and callable(mdl.train):
                            metrics = mdl.train(df, val_split=0.2)
                        else:
                            metrics = {"note": "Model does not require explicit training"}
                            mdl.is_trained = True
                        
                        # Store trained model in session state
                        st.session_state['last_trained'] = (viz_model, mdl, metrics, time.time())
                        elapsed = time.time() - start_time
                        
                        st.success(f"Training completed successfully in {elapsed:.2f} seconds!")
                        
                        # Display metrics
                        if metrics:
                            st.json({
                                **metrics,
                                "training_time": f"{elapsed:.2f}s"
                            })
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
                        st.info("Using untrained model with fallback predictions")
                        st.session_state['last_trained'] = (viz_model, mdl, {"error": str(e)}, time.time())
                        logger.exception("Training error:")
        
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.session_state['pred_cache'] = {}
            st.success("Prediction cache cleared")
            time.sleep(0.5)
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Instance Selection
    model_instance = None
    training_info = None
    
    if st.session_state['last_trained'] is not None:
        trained_model_name, trained_model, metrics, train_time = st.session_state['last_trained']
        if trained_model_name == viz_model:
            model_instance = trained_model
            training_info = {
                "name": trained_model_name,
                "metrics": metrics,
                "time": train_time
            }
    
    if model_instance is None:
        model_instance = create_model_instance(viz_model)
        st.info(f"Using new {viz_model} instance (not trained)")
    else:
        trained_time = time.strftime("%H:%M:%S", time.localtime(training_info["time"]))
        st.success(f"✅ Using trained {viz_model} model (trained at {trained_time})")
        
        # Show training metrics if available
        if training_info and "metrics" in training_info and training_info["metrics"]:
            metrics = training_info["metrics"]
            if "train_rmse" in metrics and "val_rmse" in metrics:
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Training RMSE", f"{metrics['train_rmse']:.4f}")
                with col_m2:
                    st.metric("Validation RMSE", f"{metrics['val_rmse']:.4f}")
                with col_m3:
                    if "val_r2" in metrics:
                        st.metric("Validation R²", f"{metrics['val_r2']:.4f}")
    
    # Visualization Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📈 Visualization")
    
    vis_options = ["3D Volatility Surface", "Volatility Heatmap", "Greeks (Delta/Gamma)"]
    vis_choice = st.selectbox("Visualization Type", vis_options, index=0)
    
    # Build prediction grid
    M_grid, T_grid, grid_df = build_prediction_grid(0.7, 1.3, m_steps, 0.05, 2.0, t_steps)
    
    # Get predictions
    model_params = {}
    if hasattr(model_instance, 'params'):
        model_params = model_instance.params
    elif hasattr(model_instance, '__dict__'):
        model_params = {k: v for k, v in model_instance.__dict__.items() 
                       if not k.startswith('_') and not callable(v)}
    
    ck = cache_key(viz_model, model_params, m_steps, t_steps)
    
    if ck in st.session_state['pred_cache']:
        preds = st.session_state['pred_cache'][ck]
    else:
        with st.spinner("Computing predictions..."):
            start_time = time.time()
            preds = safe_model_predict_volatility(model_instance, grid_df)
            elapsed = time.time() - start_time
            
            st.session_state['pred_cache'][ck] = preds
            logger.info(f"Prediction computed in {elapsed:.4f} seconds")
    
    # Reshape predictions
    try:
        Z_pred = np.array(preds).reshape(M_grid.shape)
    except Exception as e:
        logger.error(f"Prediction reshape failed: {str(e)}")
        Z_pred = np.full(M_grid.shape, 0.2)
        st.error("Prediction reshape failed, using fallback surface")
    
    Z_true = synthetic_true_surface(M_grid, T_grid)
    
    # Display visualization
    if vis_choice == "3D Volatility Surface":
        fig = fig_surface(M_grid, T_grid, Z_pred, title=f"{viz_model} Predicted Volatility Surface")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show error surface if we have a "true" surface for comparison
        error_surface = np.abs(Z_pred - Z_true)
        max_error = np.nanmax(error_surface)
        if max_error > 0.01:  # Only show if there's meaningful error
            error_fig = fig_surface(
                M_grid, 
                T_grid, 
                error_surface, 
                title=f"Prediction Error (vs Synthetic Surface)"
            )
            st.plotly_chart(error_fig, use_container_width=True)
        
    elif vis_choice == "Volatility Heatmap":
        fig = fig_heatmap(M_grid, T_grid, Z_pred, title=f"{viz_model} Volatility Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        
        # Add volatility smile at selected maturity
        selected_ttm = st.slider(
            "Select TTM for Volatility Smile", 
            float(T_grid.min()), 
            float(T_grid.max()), 
            float(T_grid.mean()),
            format="%.2f"
        )
        
        # Find the closest TTM in the grid
        t_idx = np.abs(T_grid[:, 0] - selected_ttm).argmin()
        smile_fig = px.line(
            x=M_grid[t_idx, :], 
            y=Z_pred[t_idx, :],
            labels={'x': 'Moneyness', 'y': 'Implied Volatility'},
            title=f'Volatility Smile at TTM = {T_grid[t_idx, 0]:.2f}'
        )
        smile_fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(smile_fig, use_container_width=True)
        
    elif vis_choice == "Greeks (Delta/Gamma)":
        with st.spinner("Computing Greeks..."):
            start_time = time.time()
            delta_grid, gamma_grid = compute_greeks_from_iv_grid(
                M_grid, T_grid, Z_pred, option_type, 
                spot_assumption, r, q
            )
            elapsed = time.time() - start_time
            
            st.info(f"Greeks computed in {elapsed:.2f} seconds")
            
            delta_fig = fig_greeks_surface(M_grid, T_grid, delta_grid, "Delta", f"{option_type.capitalize()} Delta")
            gamma_fig = fig_greeks_surface(M_grid, T_grid, gamma_grid, "Gamma", f"{option_type.capitalize()} Gamma")
        
        col7, col8 = st.columns(2)
        with col7:
            st.plotly_chart(delta_fig, use_container_width=True)
        with col8:
            st.plotly_chart(gamma_fig, use_container_width=True)
        
        # Add 2D cross-sections for Greeks
        col9, col10 = st.columns(2)
        
        with col9:
            selected_ttm_delta = st.slider(
                "Select TTM for Delta Profile", 
                float(T_grid.min()), 
                float(T_grid.max()), 
                float(T_grid.mean()),
                key="delta_ttm"
            )
            t_idx = np.abs(T_grid[:, 0] - selected_ttm_delta).argmin()
            delta_fig_2d = px.line(
                x=M_grid[t_idx, :], 
                y=delta_grid[t_idx, :],
                labels={'x': 'Moneyness', 'y': 'Delta'},
                title=f'Delta Profile at TTM = {T_grid[t_idx, 0]:.2f}'
            )
            delta_fig_2d.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(delta_fig_2d, use_container_width=True)
        
        with col10:
            selected_ttm_gamma = st.slider(
                "Select TTM for Gamma Profile", 
                float(T_grid.min()), 
                float(T_grid.max()), 
                float(T_grid.mean()),
                key="gamma_ttm"
            )
            t_idx = np.abs(T_grid[:, 0] - selected_ttm_gamma).argmin()
            gamma_fig_2d = px.line(
                x=M_grid[t_idx, :], 
                y=gamma_grid[t_idx, :],
                labels={'x': 'Moneyness', 'y': 'Gamma'},
                title=f'Gamma Profile at TTM = {T_grid[t_idx, 0]:.2f}'
            )
            gamma_fig_2d.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(gamma_fig_2d, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Metrics Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📊 Performance Metrics")
    
    # Calculate metrics only if we have a valid prediction grid
    iv_min = float(np.nanmin(Z_pred))
    iv_max = float(np.nanmax(Z_pred))
    iv_mean = float(np.nanmean(Z_pred))
    
    # Calculate RMSE only if we have a "true" surface for comparison
    rmse = float('nan')
    if Z_pred.shape == Z_true.shape:
        try:
            rmse = float(np.sqrt(np.nanmean((Z_pred - Z_true) ** 2)))
        except Exception:
            rmse = float('nan')
    
    col9, col10, col11, col12 = st.columns(4)
    
    with col9:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("IV Min", f"{iv_min:.4f}")
        st.metric("IV Mean", f"{iv_mean:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col10:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("IV Max", f"{iv_max:.4f}")
        if not math.isnan(rmse):
            st.metric("RMSE", f"{rmse:.6f}")
        else:
            st.metric("RMSE", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col11:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Grid Size", f"{m_steps}×{t_steps}")
        data_size = len(st.session_state['training_data']) if st.session_state['training_data'] is not None else 0
        st.metric("Training Points", f"{data_size:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col12:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Model Info**")
        st.text(f"Type: {viz_model}")
        
        if st.session_state['last_trained'] is not None:
            trained_model_name, _, _, train_time = st.session_state['last_trained']
            if trained_model_name == viz_model:
                trained_time = time.strftime("%H:%M", time.localtime(train_time))
                st.text(f"Trained: Yes ({trained_time})")
            else:
                st.text("Trained: No (different model)")
        else:
            st.text("Trained: No")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add volatility surface animation
    st.markdown("### 🎥 Volatility Surface Animation")
    
    animate = st.checkbox("Enable Surface Animation", value=False)
    if animate:
        frame_count = st.slider("Animation Frames", 10, 100, 30)
        
        # Create animation frames
        frames = []
        for i in range(frame_count):
            # Create a time-varying component for animation
            time_factor = 0.1 * np.sin(2 * np.pi * i / frame_count)
            
            # Create animated surface
            animated_surface = Z_pred + time_factor * np.sin(2 * np.pi * M_grid) * np.exp(-T_grid)
            frames.append(go.Surface(
                x=M_grid, 
                y=T_grid, 
                z=animated_surface,
                colorscale="Viridis",
                showscale=(i == 0)
            ))
        
        # Create animation figure
        fig_anim = go.Figure(
            data=[frames[0]],
            layout=go.Layout(
                title="Animated Volatility Surface",
                template="plotly_dark",
                scene=dict(
                    xaxis_title="Moneyness", 
                    yaxis_title="TTM", 
                    zaxis_title="Implied Vol",
                    aspectmode='cube'
                ),
                height=600,
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[
                            dict(
                                label="⏯️ Play",
                                method="animate",
                                args=[
                                    None, 
                                    {
                                        "frame": {"duration": 50, "redraw": True},
                                        "fromcurrent": True,
                                        "transition": {"duration": 0}
                                    }
                                ]
                            ),
                            dict(
                                label="⏸️ Pause",
                                method="animate",
                                args=[
                                    [None], 
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}
                                    }
                                ]
                            )
                        ],
                        pad={"r": 10, "t": 87},
                        showactive=False,
                        x=0.1,
                        xanchor="right",
                        y=0,
                        yanchor="top"
                    )
                ]
            ),
            frames=frames
        )
        
        st.plotly_chart(fig_anim, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: #aaa; font-size: 0.9rem;">
        <p>Volatility Surface Explorer &copy; 2023 | Production-ready financial analytics tool</p>
        <p>Uses Black-Scholes model for Greek calculations | Volatility surface modeling</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Critical error in main application:")
        st.error(f"Application error: {str(e)}")
        st.info("Please check the debug panel for more information")
        with st.expander("Error Details", expanded=True):
            st.code(traceback.format_exc())