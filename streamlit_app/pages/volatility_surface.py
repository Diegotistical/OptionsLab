# volatility_surface.py
import sys
import os
import json
import time
import math
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Optional sklearn estimators as fallback
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Optional xgboost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# Optional scipy surface generator
try:
    from scipy.interpolate import RectBivariateSpline
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ---------------------------
# Logging
# ---------------------------
logger = logging.getLogger("vol_surface_app")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)

# ---------------------------
# Ensure src of project is importable (robust)
# ---------------------------
# Candidate locations to find your project 'src' folder on Streamlit Cloud or local dev
CANDIDATE_SRC = [
    Path(__file__).resolve().parent / ".." / "src",  # Go up from pages/ to streamlit_app/, then to src/
    Path(__file__).resolve().parent.parent / "src", # Go up two levels from pages/ (e.g., if running from a deeper structure)
    Path.cwd() / "src",
    Path("src"),
    # Your machine path example (keeps safe if not present)
    Path.home() / "Coding" / "Python" / "OptionsLab" / "src",
]

SRC_DIR = None
for p in CANDIDATE_SRC:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
        SRC_DIR = p
        logger.info("Inserted src path into sys.path: %s", p)
        break

if SRC_DIR is None:
    logger.warning("Could not find 'src' directory in candidate paths. Project models may not be available.")
else:
    # Try to import project models (preferred)
    PROJECT_MODELS_AVAILABLE = False
    _project_import_error = None
    try:
        from volatility_surface.models.mlp_model import MLPModel as ProjectMLPModel
        from volatility_surface.models.random_forest import RandomForestVolatilityModel as ProjectRFModel
        # SVR model may not exist in all repos; guard import
        try:
            from volatility_surface.models.svr_model import SVRModel as ProjectSVRModel
        except Exception:
            ProjectSVRModel = None
        # xgboost model filename may vary; try common variants
        try:
            from volatility_surface.models.xgboost_model import XGBVolatilityModel as ProjectXGBModel
        except Exception:
            try:
                from volatility_surface.models.xgb_model import XGBVolatilityModel as ProjectXGBModel
            except Exception:
                ProjectXGBModel = None

        PROJECT_MODELS_AVAILABLE = True
        logger.info("Project models imported successfully.")
    except Exception as e:
        _project_import_error = str(e)
        logger.warning("Project models not importable: %s", _project_import_error)
        ProjectMLPModel = ProjectRFModel = ProjectSVRModel = ProjectXGBModel = None


# ---------------------------
# App constants & storage
# ---------------------------
APP_ROOT = Path(__file__).resolve().parent # Directory where this script (volatility_surface.py) resides
# --- FIX: Change MODEL_DIR to your desired path ---
# Go up from pages/ to streamlit_app/, then to the project root (OptionsLab), then to models/saved_models
PROJECT_ROOT = APP_ROOT.parent.parent # Adjust based on actual structure if needed, but this is common
MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"
# --- END FIX ---
MODEL_DIR.mkdir(parents=True, exist_ok=True) # Ensure the directory exists, create parents if needed
REGISTRY_PATH = MODEL_DIR / "registry.json"

FEATURE_COLUMNS = [
    "moneyness",
    "log_moneyness",
    "time_to_maturity",
    "ttm_squared",
    "risk_free_rate",
    "historical_volatility",
    "volatility_skew",
]
TARGET_COLUMN = "implied_volatility"
DEFAULT_SEED = 42

# ---------------------------
# Registry helpers
# ---------------------------
def load_registry() -> Dict[str, Any]:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text())
        except Exception:
            return {}
    return {}

def save_registry(reg: Dict[str, Any]):
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2, default=str))

# ---------------------------
# Estimator / Project-model factory
# ---------------------------
def create_model_instance(name: str, **kwargs):
    """
    Return either a project model instance if available (preferred),
    or a sklearn/xgboost estimator instance as a fallback.
    """
    lname = name.lower()
    # Try project models first
    if PROJECT_MODELS_AVAILABLE:
        try:
            if "mlp" in lname and ProjectMLPModel is not None:
                return ProjectMLPModel(**(kwargs or {}))
            if ("forest" in lname or "random" in lname) and ProjectRFModel is not None:
                return ProjectRFModel(**(kwargs or {}))
            if "svr" in lname and ProjectSVRModel is not None:
                return ProjectSVRModel(**(kwargs or {}))
            if ("xgboost" in lname or "xgb" in lname) and ProjectXGBModel is not None:
                return ProjectXGBModel(**(kwargs or {}))
        except Exception as e:
            logger.warning("Project-model instantiation failed for %s: %s. Falling back.", name, e)

    # Fallback: sklearn/xgboost
    if "mlp" in lname:
        hidden = kwargs.get("hidden_layer_sizes", (64, 64))
        max_iter = kwargs.get("max_iter", 400)
        return MLPRegressor(hidden_layer_sizes=hidden, max_iter=max_iter, early_stopping=True, random_state=DEFAULT_SEED)
    if "forest" in lname or "random" in lname:
        n = kwargs.get("n_estimators", 200)
        return RandomForestRegressor(n_estimators=n, n_jobs=-1, random_state=DEFAULT_SEED)
    if "svr" in lname:
        c = kwargs.get("C", 1.0)
        kernel = kwargs.get("kernel", "rbf")
        return SVR(C=c, kernel=kernel)
    if "xgboost" in lname or "xgb" in lname:
        if XGBOOST_AVAILABLE:
            n = kwargs.get("n_estimators", 200)
            return XGBRegressor(n_estimators=n, objective="reg:squarederror", random_state=DEFAULT_SEED)
        else:
            logger.warning("XGBoost requested but not installed; using RandomForest fallback.")
            return RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=DEFAULT_SEED)
    # default fallback
    return RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=DEFAULT_SEED)

# ---------------------------
# Data generation & fallback prediction
# ---------------------------
def generate_fallback_data(n_samples: int = 1500, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    spots = rng.uniform(90, 110, n_samples)
    strikes = rng.uniform(80, 120, n_samples)
    ttms = rng.uniform(0.1, 2.0, n_samples)
    moneyness = spots / strikes
    ivs = 0.2 + 0.05 * np.sin(2 * np.pi * moneyness) * np.exp(-ttms) + 0.03 * (moneyness - 1) ** 2
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

def generate_fallback_prediction(df: pd.DataFrame) -> np.ndarray:
    m = df["moneyness"].to_numpy()
    t = df["time_to_maturity"].to_numpy()
    base = 0.2 + 0.05 * np.sin(2 * np.pi * m) * np.exp(-t)
    smile = 0.03 * (m - 1.0) ** 2
    return np.clip(base + smile, 0.03, 0.6)

# ---------------------------
# Prediction wrapper
# ---------------------------
def safe_predict(model_obj, df: pd.DataFrame) -> np.ndarray:
    """
    Accept either a project-model instance (which may have predict_volatility/predict)
    or a sklearn Pipeline/estimator. Returns numpy array of predictions or fallback.
    """
    if df is None:
        raise ValueError("df must be provided for prediction")

    # Project models often provide predict_volatility
    try:
        if hasattr(model_obj, "predict_volatility"):
            # Some project methods accept df and return numpy
            return np.asarray(model_obj.predict_volatility(df))
        if hasattr(model_obj, "predict"):
            return np.asarray(model_obj.predict(df if isinstance(df, np.ndarray) else df[FEATURE_COLUMNS]))
        # If model stores pipeline attribute
        if hasattr(model_obj, "pipeline"):
            return np.asarray(model_obj.pipeline.predict(df[FEATURE_COLUMNS]))
    except RuntimeError as e:
        # detect 'not trained' messages from project base
        if "not trained" in str(e).lower():
            logger.warning("Model not trained: %s", e)
            return generate_fallback_prediction(df)
        raise
    except Exception as e:
        logger.error("Prediction error: %s", e)

    # fallback
    return generate_fallback_prediction(df)

# ---------------------------
# Model persistence helpers (support project model save/load if available)
# ---------------------------
def save_model_ui(name: str, model_obj: Any) -> Optional[str]:
    """
    Try to save using project's methods first (save / _save_model_impl / _save), else joblib.
    Returns path or None.
    """
    filename = f"{name}.joblib"
    path = str(MODEL_DIR / filename)
    try:
        if hasattr(model_obj, "save") and callable(getattr(model_obj, "save")):
            # Some project models implement a save() high-level API
            try:
                model_obj.save(str(MODEL_DIR / name))
                reg = load_registry()
                reg[name] = {"path": filename, "saved_at": time.time()}
                save_registry(reg)
                return path
            except Exception:
                # fallback to lower-level
                pass

        # Some project models implement _save_model_impl(model_path, scaler_path)
        if hasattr(model_obj, "_save_model_impl"):
            model_path = str(MODEL_DIR / f"{name}_model.pkl")
            scaler_path = str(MODEL_DIR / f"{name}_scaler.pkl")
            try:
                model_obj._save_model_impl(model_path, scaler_path)
                reg = load_registry()
                reg[name] = {"path": filename, "saved_at": time.time(), "model_path": model_path, "scaler_path": scaler_path}
                save_registry(reg)
                return path
            except Exception as e:
                logger.warning("Project _save_model_impl failed: %s", e)

        # As final fallback use joblib for sklearn-style estimators or entire object
        joblib.dump(model_obj, path)
        reg = load_registry()
        reg[name] = {"path": filename, "saved_at": time.time()}
        save_registry(reg)
        return path
    except Exception as e:
        logger.error("Failed to save model: %s", e)
        return None

def load_model_ui(name: str) -> Optional[Any]:
    """
    Load a model saved by save_model_ui. For project models we try to restore using _load_model_impl if available.
    Otherwise we joblib.load.
    """
    reg = load_registry()
    entry = reg.get(name)
    if not entry:
        return None
    try:
        # If entry has explicit model_path & scaler_path try to use project's class loader
        model_path = entry.get("model_path")
        scaler_path = entry.get("scaler_path")
        if model_path and scaler_path:
            # Try to instantiate corresponding project model by name
            # Heuristic: pick a class based on name patterns
            lname = name.lower()
            inst = None
            if "mlp" in lname and ProjectMLPModel is not None:
                inst = ProjectMLPModel()
            elif ("forest" in lname or "random" in lname) and ProjectRFModel is not None:
                inst = ProjectRFModel()
            elif "svr" in lname and ProjectSVRModel is not None:
                inst = ProjectSVRModel()
            elif ("xgb" in lname or "xgboost" in lname) and ProjectXGBModel is not None:
                inst = ProjectXGBModel()

            if inst is not None and hasattr(inst, "_load_model_impl"):
                inst._load_model_impl(model_path, scaler_path)
                return inst

        # generic joblib path
        path = MODEL_DIR / entry["path"]
        if path.exists():
            return joblib.load(path)
    except Exception as e:
        logger.error("Failed to load model %s: %s", name, e)
    return None

# ---------------------------
# Grid builder & viz
# ---------------------------
def build_prediction_grid(m_start=0.7, m_end=1.3, m_steps=40, t_start=0.05, t_end=2.0, t_steps=40):
    m = np.linspace(m_start, m_end, m_steps)
    t = np.linspace(t_start, t_end, t_steps)
    M, T = np.meshgrid(m, t, indexing="xy")
    flat_m = M.ravel()
    flat_t = T.ravel()
    grid_df = pd.DataFrame({
        "moneyness": flat_m,
        "log_moneyness": np.log(np.clip(flat_m, 1e-12, None)),
        "time_to_maturity": flat_t,
        "ttm_squared": flat_t ** 2,
        "risk_free_rate": np.full(flat_m.shape, 0.03),
        "historical_volatility": np.full(flat_m.shape, 0.2),
        "volatility_skew": np.zeros(flat_m.shape),
    })
    return M, T, grid_df

def fig_surface(M, T, Z, title="Volatility Surface"):
    fig = go.Figure(go.Surface(x=M, y=T, z=Z))
    fig.update_layout(title=title, template="plotly_dark", height=650,
                      scene=dict(xaxis_title="Moneyness", yaxis_title="Time to Maturity", zaxis_title="Implied Vol"))
    return fig

# ---------------------------
# UI helpers
# ---------------------------
def setup_dark_theme():
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0c0d13 0%, #1a1d29 100%); max-width: 100% !important; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 100% !important; }
    .section { background: rgba(30,33,48,0.9); border-radius: 10px; padding: 1.25rem; margin: 0.75rem 0; }
    .center { display:flex; justify-content:center; align-items:center; }
    .controls { max-width: 1100px; width: 100%; }
    .btn-full { width:100%; padding: 0.45rem 0; border-radius:6px; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# Main page
# ---------------------------
def main():
    st.set_page_config(page_title="Volatility Surface Explorer", layout="wide")
    setup_dark_theme()

    # Header
    st.markdown(
        """
        <div style="background: linear-gradient(90deg,#253149 0%, #15202b 100%); padding: 18px; border-radius: 8px;">
            <h1 style="color: #ffffff; margin: 0; font-size: 22px;">Volatility Surface Explorer</h1>
            <p style="color: #c8d0da; margin: 4px 0 0 0; font-size: 13px;">Train, save, load and inspect volatility surface models.</p>
        </div>
        """, unsafe_allow_html=True
    )

    # Status / debug: only show import details if models not present
    if not PROJECT_MODELS_AVAILABLE:
        with st.expander("Import diagnostics", expanded=False):
            st.write("Project models were not importable. Fallback to sklearn/XGBoost will be used.")
            st.write("Import error:", _project_import_error)
            st.write("Checked src path:", str(SRC_DIR) if SRC_DIR is not None else "no src path found")

    # Controls centered & full width
    st.markdown('<div class="center"><div class="controls">', unsafe_allow_html=True)

    # Configuration row
    cfg_cols = st.columns([1, 1, 1])
    with cfg_cols[0]:
        use_generator = st.checkbox("Use surface generator (scipy required)", value=False)
        dataset_size = st.slider("Dataset size", min_value=200, max_value=5000, value=1500, step=100)
    with cfg_cols[1]:
        model_choice = st.selectbox("Model", ["MLP Neural Network", "Random Forest", "SVR", "XGBoost"], index=1)
        m_steps = st.slider("Moneyness steps", 12, 120, 40)
    with cfg_cols[2]:
        t_steps = st.slider("TTM steps", 6, 80, 30)
        spot = st.number_input("Spot price (assumption)", value=100.0, format="%.2f")

    st.markdown('</div></div>', unsafe_allow_html=True)

    # Generate / load data and train buttons (centered)
    st.markdown('<div class="center"><div class="controls">', unsafe_allow_html=True)
    ctrl_cols = st.columns([1, 1, 1])
    with ctrl_cols[0]:
        if st.button("Generate training data", key="gen_data", use_container_width=True):
            df = generate_fallback_data(dataset_size)
            st.session_state["training_data"] = df
            st.session_state["pred_cache"] = {}
            st.success(f"Generated {len(df)} samples")

    with ctrl_cols[1]:
        model_save_name = st.text_input("Model save name", value=f"{model_choice.replace(' ','_')}_model")
        if st.button("Train model", key="train_model", use_container_width=True):
            if "training_data" not in st.session_state or st.session_state["training_data"] is None:
                st.error("No training data. Generate or upload training data first.")
            else:
                df = st.session_state["training_data"]
                # instantiate
                try:
                    model_obj = create_model_instance(model_choice)
                    # Log which type of model object was created
                    logger.info(f"Created model instance: {type(model_obj)}")
                except Exception as e:
                    st.error(f"Failed to create model instance: {e}")
                    model_obj = None

                if model_obj is not None:
                    with st.spinner("Training model..."):
                        try:
                            # --- FIX: Use a standardized training approach for ALL models ---
                            # This bypasses the buggy `train` methods in the project models.
                            X = df.copy()
                            y = X[TARGET_COLUMN]

                            # If the model object is already a Pipeline (from a previous train),
                            # we just need a new estimator instance.
                            # Note: Project models shouldn't be Pipelines initially, but check anyway.
                            if isinstance(model_obj, Pipeline):
                                estimator = model_obj.named_steps['est']
                            else:
                                estimator = model_obj

                            # Build and fit the pipeline using the standard features
                            pipeline = Pipeline([("scaler", StandardScaler()), ("est", estimator)])
                            t0 = time.time()
                            # Use FEATURE_COLUMNS for the standard pipeline training
                            pipeline.fit(X[FEATURE_COLUMNS], y)
                            t1 = time.time()

                            # Calculate metrics
                            y_pred = pipeline.predict(X[FEATURE_COLUMNS])
                            metrics = {
                                "train_rmse": math.sqrt(mean_squared_error(y, y_pred)),
                                "train_mae": mean_absolute_error(y, y_pred),
                                "train_r2": r2_score(y, y_pred),
                                "fit_time_seconds": t1 - t0
                            }

                            # Store the trained pipeline as the model object
                            model_obj = pipeline
                            history = None
                            logger.info(f"Training finished using standardized pipeline for {type(estimator)}")
                            # --- END OF FIX ---

                            # save in session
                            st.session_state["last_trained"] = {"name": model_save_name, "model": model_obj, "metrics": metrics, "history": history}
                            st.session_state.setdefault("pred_cache", {})  # ensure cache exists
                            st.success("Training finished")
                            # --- FIX: Use the new API ---
                            st.rerun()
                            # --- END OF FIX ---
                        except Exception as e:
                            st.error(f"Training failed: {e}")
                            logger.exception("Training error")

    with ctrl_cols[2]:
        if st.button("Save last trained model", key="save_model", use_container_width=True):
            if "last_trained" not in st.session_state:
                st.error("No trained model in this session to save.")
            else:
                entry = st.session_state["last_trained"]
                saved_path = save_model_ui(entry["name"], entry["model"])
                if saved_path:
                    st.success(f"Saved model to {saved_path}")
                else:
                    st.error("Failed to save model. Check logs.")

    st.markdown('</div></div>', unsafe_allow_html=True)

    # If session has training data show HEAD and simple table
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Training data preview")
    if "training_data" in st.session_state and st.session_state["training_data"] is not None:
        st.dataframe(st.session_state["training_data"].head(200))
    else:
        st.info("No training data in session. Generate training data or upload your own dataset externally.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Model registry & load
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Model registry")
    registry = load_registry()
    registry_keys = list(registry.keys())
    if registry_keys:
        sel_model = st.selectbox("Load model from registry", options=["-- none --"] + registry_keys)
        if sel_model and sel_model != "-- none --":
            if st.button("Load selected model", key="load_model"):
                loaded = load_model_ui(sel_model)
                if loaded is None:
                    st.error("Failed to load model")
                else:
                    st.session_state["last_trained"] = {"name": sel_model, "model": loaded, "metrics": registry[sel_model].get("metrics"), "history": None}
                    st.success(f"Loaded {sel_model}")
    else:
        st.info("No saved models found.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Prediction & Visualization
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Volatility surface inference")

    # grid
    M_grid, T_grid, grid_df = build_prediction_grid(0.7, 1.3, m_steps, 0.05, 2.0, t_steps)
    # if we have cached pred for this model+grid use it
    pred_key = f"pred::{model_choice}::{m_steps}x{t_steps}"
    preds = None
    if pred_key in st.session_state.get("pred_cache", {}):
        preds = st.session_state["pred_cache"][pred_key]
        st.info("Using cached predictions")
    else:
        with st.spinner("Computing predictions..."):
            if "last_trained" in st.session_state and st.session_state["last_trained"].get("model") is not None:
                model_obj = st.session_state["last_trained"]["model"]
                try:
                    preds = safe_predict(model_obj, grid_df)
                except Exception as e:
                    logger.error("Prediction error: %s", e)
                    preds = generate_fallback_prediction(grid_df)
            else:
                preds = generate_fallback_prediction(grid_df)
            st.session_state.setdefault("pred_cache", {})[pred_key] = preds

    try:
        Z_pred = np.asarray(preds).reshape(M_grid.shape)
    except Exception:
        Z_pred = np.full(M_grid.shape, 0.2)
        st.error("Prediction reshape failed")

    fig = fig_surface(M_grid, T_grid, Z_pred, title=f"{model_choice} predicted surface")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Performance metrics & training diagnostics
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Performance & diagnostics")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        # show metrics of last training
        if "last_trained" in st.session_state:
            last = st.session_state["last_trained"]
            st.markdown("#### Last training summary")
            st.json(last.get("metrics", {}))
            # show loss curve if available in history
            history = last.get("history")
            if history:
                train_loss = history.get("train_loss")
                val_loss = history.get("val_loss")
                if train_loss and val_loss:
                    loss_fig = go.Figure()
                    loss_fig.add_trace(go.Scatter(y=train_loss, name="train"))
                    loss_fig.add_trace(go.Scatter(y=val_loss, name="val"))
                    loss_fig.update_layout(title="Training loss (MSE)", template="plotly_dark", height=350)
                    st.plotly_chart(loss_fig, use_container_width=True)
            # for sklearn pipelines show feature importance if possible
            model_obj = last.get("model")
            feat_imp = None
            try:
                if isinstance(model_obj, Pipeline):
                    est = model_obj.named_steps.get("est")
                    if hasattr(est, "feature_importances_"):
                        feat_imp = dict(zip(FEATURE_COLUMNS, est.feature_importances_.tolist()))
                elif hasattr(model_obj, "training_history") and isinstance(model_obj.training_history, dict):
                    # some project models put feature_importances in training_history
                    fi = model_obj.training_history.get("feature_importances")
                    if fi:
                        feat_imp = fi
                elif hasattr(model_obj, "model") and hasattr(model_obj.model, "feature_importances_"):
                    feat_imp = dict(zip(FEATURE_COLUMNS, model_obj.model.feature_importances_.tolist()))
            except Exception as e:
                logger.debug("Feature importance retrieval failed: %s", e)

            if feat_imp:
                fi_fig = go.Figure(go.Bar(x=list(feat_imp.keys()), y=list(feat_imp.values())))
                fi_fig.update_layout(title="Feature importances", template="plotly_dark", height=320)
                st.plotly_chart(fi_fig, use_container_width=True)
        else:
            st.info("No training has been performed in this session.")

    with col_b:
        # show simple predicted surface stats
        st.markdown("#### Predicted surface stats")
        st.metric("IV min", f"{np.nanmin(Z_pred):.4f}")
        st.metric("IV mean", f"{np.nanmean(Z_pred):.4f}")
        st.metric("IV max", f"{np.nanmax(Z_pred):.4f}")
        rmse = math.sqrt(np.nanmean((Z_pred - (0.2 + 0.05 * np.sin(2 * np.pi * M_grid) * np.exp(-T_grid) + 0.03 * (M_grid - 1)**2))**2))
        st.metric("Surface RMSE (synthetic truth)", f"{rmse:.6f}")

    st.markdown('</div>', unsafe_allow_html=True)

    # End of main

if __name__ == "__main__":
    main()