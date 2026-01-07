# streamlit_app/pages/2_MonteCarlo_ML.py
"""
Monte Carlo ML Surrogate - Streamlit Page.

Train and use ML surrogate models for instant option pricing.
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
from scipy.stats import norm
import streamlit as st

st.set_page_config(
    page_title="Monte Carlo ML",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

try:
    from components import (
        apply_custom_css, page_header, section_divider,
        get_chart_layout, format_price, format_greek, format_time_ms
    )
except ImportError:
    from streamlit_app.components import (
        apply_custom_css, page_header, section_divider,
        get_chart_layout, format_price, format_greek, format_time_ms
    )

try:
    from src.pricing_models import MonteCarloMLSurrogate, black_scholes, LIGHTGBM_AVAILABLE
except ImportError:
    MonteCarloMLSurrogate = None
    black_scholes = None
    LIGHTGBM_AVAILABLE = False

# Optimization module imports
OPTUNA_AVAILABLE = False
ONNX_AVAILABLE = False

try:
    import optuna
    from src.optimization import (
        OptunaStudyManager,
        LightGBMSearchSpace,
        ONNXExporter,
        ONNXValidator,
        ONNXInferenceEngine,
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    pass

try:
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    pass

apply_custom_css()


# =============================================================================
# VECTORIZED BLACK-SCHOLES (for accurate comparison)
# =============================================================================
def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """Black-Scholes price calculation."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type="call", q=0.0):
    """Black-Scholes delta."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return np.exp(-q * T) * (norm.cdf(d1) - 1)


def bs_gamma(S, K, T, r, sigma, q=0.0):
    """Black-Scholes gamma."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


# =============================================================================
# HEADER
# =============================================================================
page_header("ML Surrogate Model", "Train LightGBM models for instant option pricing predictions")

# Feature availability badges
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"**LightGBM:** {'✅ Available' if LIGHTGBM_AVAILABLE else '⚠️ sklearn fallback'}")
with col2:
    st.markdown(f"**Optuna:** {'✅ Available' if OPTUNA_AVAILABLE else '❌ Not installed'}")
with col3:
    st.markdown(f"**ONNX Runtime:** {'✅ Available' if ONNX_AVAILABLE else '❌ Not installed'}")

section_divider()

# =============================================================================
# INPUT SECTION
# =============================================================================
st.markdown('<div class="input-section">', unsafe_allow_html=True)

# Check if we have optimized params to use
has_optimized = "optuna_result" in st.session_state
if has_optimized:
    opt_params = st.session_state["optuna_result"].best_params
    st.success(f"✅ Using optimized hyperparameters from Optuna study")

col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1])

with col1:
    st.markdown("**Option Parameters**")
    S = st.number_input("Spot Price ($)", min_value=1.0, max_value=500.0, value=100.0, step=1.0, key="ml_spot")
    K = st.number_input("Strike Price ($)", min_value=1.0, max_value=500.0, value=100.0, step=1.0, key="ml_strike")
    T = st.number_input("Time to Maturity (years)", min_value=0.01, max_value=5.0, value=1.0, step=0.05, key="ml_time")

with col2:
    st.markdown("**Market Conditions**")
    r_pct = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5, key="ml_rate")
    sigma_pct = st.number_input("Volatility (%)", min_value=1.0, max_value=100.0, value=20.0, step=1.0, key="ml_vol")
    option_type = st.selectbox("Option Type", ["call", "put"], key="ml_type")
    r = r_pct / 100.0
    sigma = sigma_pct / 100.0

with col3:
    st.markdown("**Training Settings**")
    n_samples = st.select_slider(
        "Training Samples",
        options=[2000, 5000, 10000, 20000, 50000],
        value=10000,
        key="ml_samples"
    )
    
    # Default values (or from Optuna if available)
    default_trees = opt_params.get("n_estimators", 300) if has_optimized else 300
    default_depth = opt_params.get("max_depth", 8) if has_optimized else 8
    default_lr = opt_params.get("learning_rate", 0.1) if has_optimized else 0.1
    
    n_estimators = st.select_slider(
        "Model Trees",
        options=[100, 200, 300, 400, 500, 750, 1000],
        value=min([100, 200, 300, 400, 500, 750, 1000], key=lambda x: abs(x - default_trees)),
        key="ml_trees"
    )
    max_depth = st.slider("Tree Depth", min_value=3, max_value=12, value=default_depth, key="ml_depth")
    learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=default_lr, step=0.01, key="ml_lr")

with col4:
    st.markdown("**Actions**")
    st.write("")
    train_btn = st.button("🎯 Train & Predict", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TRAINING & RESULTS
# =============================================================================
if train_btn:
    if MonteCarloMLSurrogate is None:
        st.error("ML Surrogate not available. Check installation.")
        st.stop()
    
    progress = st.progress(0, text="Initializing...")
    
    # Create surrogate with UI params
    progress.progress(10, text="Creating ML surrogate...")
    
    t_start = time.perf_counter()
    surrogate = MonteCarloMLSurrogate(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        seed=42
    )
    
    progress.progress(30, text=f"Generating {n_samples:,} training samples...")
    
    # Train with focused parameter ranges around the target
    t_train_start = time.perf_counter()
    
    # Generate training data with ranges centered around target values
    param_ranges = {
        "S_range": (max(10, S * 0.5), S * 1.5),
        "K_range": (max(10, K * 0.5), K * 1.5),
        "T_range": (0.05, max(T * 1.5, 2.0)),
        "r_range": (max(0.001, r * 0.5), min(r * 2, 0.15)),
        "sigma_range": (max(0.05, sigma * 0.5), min(sigma * 2, 0.8)),
        "q_range": (0.0, 0.03),
    }
    
    surrogate.fit(
        n_samples=n_samples,
        option_type=option_type,
        verbose=False
    )
    t_train = (time.perf_counter() - t_train_start) * 1000
    
    progress.progress(70, text="Running predictions...")
    
    # Predict single point
    t_pred_start = time.perf_counter()
    result = surrogate.predict_single(S, K, T, r, sigma, 0.0)
    t_pred = (time.perf_counter() - t_pred_start) * 1000
    
    ml_price = result["price"]
    ml_delta = result["delta"]
    ml_gamma = result["gamma"]
    
    # Black-Scholes for comparison
    bs_price_val = bs_price(S, K, T, r, sigma, option_type)
    bs_delta_val = bs_delta(S, K, T, r, sigma, option_type)
    bs_gamma_val = bs_gamma(S, K, T, r, sigma)
    
    progress.progress(100, text="Complete!")
    time.sleep(0.2)
    progress.empty()
    
    section_divider()
    
    # Metrics display
    error = abs(ml_price - bs_price_val)
    pct_err = (error / bs_price_val * 100) if bs_price_val > 0 else 0
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ML Price</div>
            <div class="metric-value">{format_price(ml_price)}</div>
            <div class="metric-delta">{format_time_ms(t_pred)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        delta_class = "negative" if pct_err > 5 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">BS Price</div>
            <div class="metric-value">{format_price(bs_price_val)}</div>
            <div class="metric-delta {delta_class}">Err: {pct_err:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ML Delta</div>
            <div class="metric-value">{format_greek(ml_delta)}</div>
            <div class="metric-delta">BS: {bs_delta_val:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ML Gamma</div>
            <div class="metric-value">{format_greek(ml_gamma, 6)}</div>
            <div class="metric-delta">BS: {bs_gamma_val:.6f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Training Time</div>
            <div class="metric-value">{format_time_ms(t_train)}</div>
            <div class="metric-delta">{n_samples:,} samples</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value">{n_estimators}</div>
            <div class="metric-delta">trees</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Accuracy warning
    if pct_err > 5:
        st.warning(f"⚠️ ML prediction error is {pct_err:.1f}%. Try increasing training samples or tree count.")
    
    section_divider()
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Price Comparison", "🎨 Price Surface", "📈 Error Analysis"])
    
    with tab1:
        spot_range = np.linspace(S * 0.7, S * 1.3, 40)
        
        ml_prices = []
        bs_prices = []
        
        for s in spot_range:
            res = surrogate.predict_single(s, K, T, r, sigma, 0.0)
            ml_prices.append(res["price"])
            bs_prices.append(bs_price(s, K, T, r, sigma, option_type))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=spot_range, y=bs_prices,
            mode='lines',
            line=dict(width=3, color='#10b981', dash='dash'),
            name='Black-Scholes (exact)'
        ))
        
        fig.add_trace(go.Scatter(
            x=spot_range, y=ml_prices,
            mode='lines',
            line=dict(width=3, color='#60a5fa'),
            name='ML Surrogate'
        ))
        
        fig.add_vline(x=K, line_dash="dash", line_color="#ef4444", annotation_text="Strike")
        fig.add_vline(x=S, line_dash="dot", line_color="#f97316", annotation_text="Current")
        
        fig.update_layout(**get_chart_layout("ML vs Analytical Pricing", 400))
        fig.update_xaxes(title_text="Spot Price ($)")
        fig.update_yaxes(title_text="Option Price ($)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("**Price Surface (Spot vs Volatility)**")
        
        spot_grid = np.linspace(S * 0.8, S * 1.2, 15)
        vol_grid = np.linspace(0.1, 0.5, 15)
        
        prices_grid = np.zeros((len(vol_grid), len(spot_grid)))
        
        for i, vol in enumerate(vol_grid):
            for j, spot in enumerate(spot_grid):
                res = surrogate.predict_single(spot, K, T, r, vol, 0.0)
                prices_grid[i, j] = res["price"]
        
        fig = go.Figure(data=[go.Surface(
            x=spot_grid,
            y=vol_grid * 100,
            z=prices_grid,
            colorscale='Viridis',
            opacity=0.9
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Spot Price ($)',
                yaxis_title='Volatility (%)',
                zaxis_title='Option Price ($)',
                bgcolor='rgba(15, 23, 42, 0.9)'
            ),
            paper_bgcolor='rgba(30, 41, 59, 0.8)',
            font=dict(color='#cbd5e1'),
            height=500,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Error distribution
            errors = np.array(ml_prices) - np.array(bs_prices)
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=errors,
                nbinsx=25,
                marker_color='#a78bfa',
                opacity=0.8
            ))
            
            fig.update_layout(**get_chart_layout("Prediction Error Distribution", 350))
            fig.update_xaxes(title_text="Error (ML - BS)")
            fig.update_yaxes(title_text="Frequency")
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Error Statistics</div>
                <div style="color: #cbd5e1; margin-top: 0.5rem;">
                    Mean Error: ${np.mean(errors):.4f}<br>
                    Std Error: ${np.std(errors):.4f}<br>
                    Max Abs Error: ${np.max(np.abs(errors)):.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Delta comparison
            ml_deltas = []
            bs_deltas = []
            for s in spot_range:
                res = surrogate.predict_single(s, K, T, r, sigma, 0.0)
                ml_deltas.append(res["delta"])
                bs_deltas.append(bs_delta(s, K, T, r, sigma, option_type))
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=spot_range, y=bs_deltas,
                mode='lines',
                line=dict(width=2, color='#10b981', dash='dash'),
                name='BS Delta'
            ))
            
            fig.add_trace(go.Scatter(
                x=spot_range, y=ml_deltas,
                mode='lines',
                line=dict(width=3, color='#34d399'),
                name='ML Delta'
            ))
            
            fig.add_vline(x=S, line_dash="dot", line_color="#f97316")
            fig.update_layout(**get_chart_layout("Delta Comparison", 350))
            fig.update_xaxes(title_text="Spot Price ($)")
            fig.update_yaxes(title_text="Delta")
            
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Configure parameters and click **Train & Predict** to train the ML surrogate model.")
    
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #60a5fa; margin-bottom: 1rem;">How it works</h3>
        <ol style="color: #cbd5e1; line-height: 2;">
            <li><strong>Training Data:</strong> Generates samples using vectorized Black-Scholes (fast)</li>
            <li><strong>Feature Engineering:</strong> Adds moneyness, log-moneyness, normalized time</li>
            <li><strong>Model Training:</strong> LightGBM gradient boosting with multi-output regression</li>
            <li><strong>Inference:</strong> Instant predictions for any option parameters</li>
        </ol>
        <p style="color: #94a3b8; margin-top: 1rem;">
            💡 <strong>Tip:</strong> Increase training samples (10K+) and trees (300+) for better accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)

# OPTUNA OPTIMIZATION & ONNX EXPORT

section_divider()

st.markdown("### 🔧 Advanced Features")

if OPTUNA_AVAILABLE or ONNX_AVAILABLE:
    adv_tab1, adv_tab2 = st.tabs(["⚡ Hyperparameter Optimization", "📦 Model Export (ONNX)"])
    
    with adv_tab1:
        if OPTUNA_AVAILABLE:
            st.markdown("""
            **Optuna Hyperparameter Tuning**
            
            Automatically find optimal hyperparameters for the LightGBM surrogate model using 
            Bayesian optimization with pruning.
            """)
            
            opt_col1, opt_col2 = st.columns(2)
            
            with opt_col1:
                n_trials = st.slider("Number of Trials", min_value=5, max_value=100, value=20, step=5, key="opt_trials")
                opt_study_name = st.text_input("Study Name", value="mc_ml_study", key="opt_study_name")
            
            with opt_col2:
                opt_n_samples = st.select_slider(
                    "Training Samples per Trial",
                    options=[2000, 5000, 10000],
                    value=5000,
                    key="opt_samples"
                )
                opt_seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42, key="opt_seed")
            
            optimize_btn = st.button("🚀 Run Optimization Study", type="primary", key="run_optuna")
            
            if optimize_btn:
                with st.spinner(f"Running {n_trials} Optuna trials..."):
                    try:
                        # Create study manager
                        study_path = ROOT / "models" / "optimization_results"
                        study_path.mkdir(parents=True, exist_ok=True)
                        
                        manager = OptunaStudyManager(
                            study_name=opt_study_name,
                            storage=f"sqlite:///{study_path / 'optuna_studies.db'}",
                            seed=opt_seed,
                        )
                        
                        search_space = LightGBMSearchSpace(
                            n_estimators_range=(100, 500),
                            max_depth_range=(4, 10),
                            learning_rate_range=(0.01, 0.2),
                        )
                        
                        # Simple objective for demonstration
                        def objective(trial, trial_seed):
                            params = {k: v for k, v in trial.params.items()}
                            
                            model = MonteCarloMLSurrogate(
                                n_estimators=params.get("n_estimators", 200),
                                max_depth=params.get("max_depth", 6),
                                learning_rate=params.get("learning_rate", 0.1),
                                seed=trial_seed,
                            )
                            model.fit(n_samples=opt_n_samples, option_type="call", verbose=False)
                            
                            # Score on test points
                            test_spots = np.linspace(80, 120, 10)
                            errors = []
                            for s in test_spots:
                                pred = model.predict_single(s, 100, 1.0, 0.05, 0.2, 0.0)
                                true_price = bs_price(s, 100, 1.0, 0.05, 0.2, "call")
                                errors.append(abs(pred["price"] - true_price))
                            
                            return np.mean(errors)
                        
                        result = manager.optimize(
                            objective=objective,
                            search_space=search_space,
                            n_trials=n_trials,
                            show_progress_bar=False,
                        )
                        
                        # Store in session state
                        st.session_state["optuna_result"] = result
                        
                        # Display results
                        st.success(f"✅ Optimization complete! Best score: {result.best_value:.6f}")
                        
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.markdown("**Best Hyperparameters:**")
                            st.json(result.best_params)
                        
                        with res_col2:
                            st.markdown("**Study Statistics:**")
                            st.markdown(f"""
                            - **Trials:** {result.n_trials}
                            - **Complete:** {result.n_complete}
                            - **Pruned:** {result.n_pruned}
                            - **Duration:** {result.duration_seconds:.1f}s
                            """)
                        
                        # Save results
                        result.save(study_path / f"{opt_study_name}_result.json")
                        st.info(f"📁 Results saved to `models/optimization_results/{opt_study_name}_result.json`")
                        
                    except Exception as e:
                        st.error(f"Optimization failed: {str(e)}")
        else:
            st.warning("⚠️ Optuna not installed. Run `pip install optuna` to enable hyperparameter optimization.")
    
    with adv_tab2:
        if ONNX_AVAILABLE and OPTUNA_AVAILABLE:
            st.markdown("""
            **ONNX Model Export**
            
            Export trained models to ONNX format for:
            - Faster inference in production
            - Cross-platform deployment
            - Integration with other systems
            """)
            
            # Check if we have a trained model in session state
            if "optuna_result" in st.session_state:
                best_params = st.session_state["optuna_result"].best_params
                st.success(f"✅ Using optimized parameters from study")
                st.json(best_params)
                
                export_btn = st.button("📦 Export to ONNX", type="primary", key="export_onnx")
                
                if export_btn:
                    with st.spinner("Exporting model to ONNX..."):
                        try:
                            # Train model with best params
                            model = MonteCarloMLSurrogate(
                                n_estimators=best_params.get("n_estimators", 200),
                                max_depth=best_params.get("max_depth", 6),
                                learning_rate=best_params.get("learning_rate", 0.1),
                                seed=42,
                            )
                            model.fit(n_samples=10000, option_type="call", verbose=False)
                            
                            # Export
                            onnx_path = ROOT / "models" / "saved_models" / "mc_ml_surrogate.onnx"
                            onnx_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            feature_names = model.feature_names
                            
                            # Use sklearn export since model is a Pipeline
                            export_result = ONNXExporter.export_sklearn(
                                model=model.model,
                                output_path=onnx_path,
                                feature_names=feature_names,
                            )
                            
                            if export_result.success:
                                st.success(f"✅ Model exported to `{onnx_path}`")
                                st.markdown(f"**Model size:** {export_result.model_size_bytes / 1024:.1f} KB")
                                
                                # Validate
                                st.markdown("**Validation:**")
                                test_X = np.random.rand(100, len(feature_names)).astype(np.float32)
                                validator = ONNXValidator(rtol=1e-3, atol=1e-4)
                                validation = validator.validate(
                                    native_model=model.model,
                                    onnx_path=onnx_path,
                                    X_test=test_X,
                                )
                                
                                if validation.passed:
                                    st.success(f"✅ Validation passed (correlation: {validation.pearson_correlation:.6f})")
                                else:
                                    st.warning(f"⚠️ Validation warning: {validation.diagnostics}")
                            else:
                                st.error(f"Export failed: {export_result.error}")
                                
                        except Exception as e:
                            st.error(f"Export failed: {str(e)}")
            else:
                st.info("👆 Run an optimization study first to get optimized hyperparameters for export.")
                
            # Load existing ONNX model
            section_divider()
            st.markdown("**Load Existing ONNX Model**")
            
            onnx_files = list((ROOT / "models" / "saved_models").glob("*.onnx")) if (ROOT / "models" / "saved_models").exists() else []
            
            if onnx_files:
                selected_onnx = st.selectbox(
                    "Select ONNX model",
                    options=[f.name for f in onnx_files],
                    key="select_onnx"
                )
                
                if st.button("🔄 Load ONNX Model", key="load_onnx"):
                    onnx_path = ROOT / "models" / "saved_models" / selected_onnx
                    engine = ONNXInferenceEngine(onnx_path)
                    st.session_state["onnx_engine"] = engine
                    st.success(f"✅ Loaded {selected_onnx}")
                    st.json(engine.get_model_info())
            else:
                st.info("No ONNX models found in `models/saved_models/`")
        else:
            missing = []
            if not ONNX_AVAILABLE:
                missing.append("onnxruntime")
            if not OPTUNA_AVAILABLE:
                missing.append("optuna")
            st.warning(f"⚠️ Missing dependencies: {', '.join(missing)}. Run `pip install {' '.join(missing)}`")

else:
    st.info("""
    **Advanced features require additional packages:**
    
    ```bash
    pip install optuna onnx onnxruntime onnxmltools
    ```
    
    These enable:
    - **Optuna**: Automatic hyperparameter optimization
    - **ONNX**: Model export for production deployment
    """)

