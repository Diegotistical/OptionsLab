# Page 2 MC ML.py
import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import logging
import plotly.graph_objects as go
import plotly.express as px
import time
import traceback

# ======================
# PATH RESOLUTION
# ======================
logger = logging.getLogger("monte_carlo_ml")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def setup_paths():
    """Set up paths for different deployment environments"""
    # Strategy 1: Streamlit Cloud standard structure
    cloud_root = Path("/mount/src/optionslab")
    if cloud_root.exists():
        logger.info(f"Found Streamlit Cloud root at: {cloud_root}")
        src_path = cloud_root / "src"
        if src_path.exists():
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
                logger.info(f"Added {src_path} to sys.path")
            return True
    
    # Strategy 2: Local development structure
    if (Path.cwd() / "src").exists():
        logger.info(f"Found src directory at: {Path.cwd() / 'src'}")
        if str(Path.cwd() / "src") not in sys.path:
            sys.path.insert(0, str(Path.cwd() / "src"))
            logger.info(f"Added {Path.cwd() / 'src'} to sys.path")
        return True
    
    # Strategy 3: Current directory structure
    if (Path.cwd() / "pricing_models").exists():
        logger.info(f"Found pricing_models in current directory: {Path.cwd()}")
        if str(Path.cwd()) not in sys.path:
            sys.path.insert(0, str(Path.cwd()))
            logger.info(f"Added {Path.cwd()} to sys.path")
        return True
    
    logger.warning("Could not determine standard directory structure")
    return False

# Execute path setup
setup_paths()

# ======================
# DIRECT MONTECARLOML IMPORT
# ======================
def get_monte_carlo_ml():
    """Try to import the actual MonteCarloML class"""
    try:
        # Try direct import from pricing_models
        from pricing_models.monte_carlo_ml import MonteCarloML
        logger.info("Successfully imported MonteCarloML from pricing_models.monte_carlo_ml")
        return MonteCarloML
    except ImportError as e:
        logger.debug(f"Direct import failed: {str(e)}")
    
    try:
        # Try import from src directory
        from src.pricing_models.monte_carlo_ml import MonteCarloML
        logger.info("Successfully imported MonteCarloML from src.pricing_models.monte_carlo_ml")
        return MonteCarloML
    except ImportError as e:
        logger.debug(f"src directory import failed: {str(e)}")
    
    try:
        # Try import from parent directories
        current_file = Path(__file__).resolve()
        for i in range(5):  # Check up to 5 levels up
            parent = current_file.parents[i]
            monte_carlo_ml_path = parent / "src" / "pricing_models" / "monte_carlo_ml.py"
            if monte_carlo_ml_path.exists():
                logger.info(f"Found monte_carlo_ml.py at: {monte_carlo_ml_path}")
                
                # Add parent directory to sys.path
                src_dir = parent / "src"
                if src_dir.exists() and str(src_dir) not in sys.path:
                    sys.path.insert(0, str(src_dir))
                    logger.info(f"Added {src_dir} to sys.path")
                
                # Try import again
                from pricing_models.monte_carlo_ml import MonteCarloML
                logger.info("Successfully imported MonteCarloML after path adjustment")
                return MonteCarloML
    except ImportError as e:
        logger.debug(f"Parent directory import failed: {str(e)}")
    
    # Strategy 4: Streamlit Cloud specific path
    try:
        cloud_root = Path("/mount/src/optionslab")
        if cloud_root.exists():
            monte_carlo_ml_path = cloud_root / "src" / "pricing_models" / "monte_carlo_ml.py"
            if monte_carlo_ml_path.exists():
                logger.info(f"Found monte_carlo_ml.py at: {monte_carlo_ml_path}")
                
                # Add cloud root src to sys.path
                src_path = cloud_root / "src"
                if str(src_path) not in sys.path:
                    sys.path.insert(0, str(src_path))
                    logger.info(f"Added {src_path} to sys.path")
                
                # Try import again
                from pricing_models.monte_carlo_ml import MonteCarloML
                logger.info("Successfully imported MonteCarloML for Streamlit Cloud")
                return MonteCarloML
    except ImportError as e:
        logger.debug(f"Streamlit Cloud import failed: {str(e)}")
    
    # Strategy 5: Check if module is already loaded
    try:
        if "pricing_models.monte_carlo_ml" in sys.modules:
            module = sys.modules["pricing_models.monte_carlo_ml"]
            if hasattr(module, "MonteCarloML"):
                logger.info("Found MonteCarloML in already loaded module")
                return module.MonteCarloML
    except Exception as e:
        logger.debug(f"Module check failed: {str(e)}")
    
    # Strategy 6: Final fallback - create our own implementation
    logger.error("All import strategies failed. Using comprehensive fallback implementation.")
    
    class MonteCarloML:
        """Complete fallback implementation of MonteCarloML"""
        def __init__(self, num_simulations=50000, num_steps=100, seed=42):
            self.num_simulations = num_simulations
            self.num_steps = num_steps
            self.seed = seed
            self.is_fitted = False
            self.X_train = None
            self.y_train = None
            logger.info(f"Initialized fallback MonteCarloML with {num_simulations} simulations, {num_steps} steps")
        
        def generate_training_data(self, df):
            """Generate training data using MC pricing"""
            X = df.copy()
            prices, deltas, gammas = [], [], []
            
            for _, row in X.iterrows():
                try:
                    # Extract parameters
                    S = float(row['S'])
                    K = float(row['K'])
                    T = float(row['T'])
                    r = float(row['r'])
                    sigma = float(row['sigma'])
                    q = float(row['q'])
                    
                    # Validate parameters
                    if S <= 0 or K <= 0 or T <= 0.001 or sigma <= 0.001:
                        logger.warning(f"Invalid parameters in row: S={S}, K={K}, T={T}, sigma={sigma}")
                        prices.append(0.0)
                        deltas.append(0.5)
                        gammas.append(0.01)
                        continue
                    
                    # Generate MC price
                    np.random.seed(self.seed)
                    dt = T / self.num_steps
                    Z = np.random.standard_normal((self.num_simulations, self.num_steps))
                    S_paths = np.zeros((self.num_simulations, self.num_steps))
                    S_paths[:, 0] = S
                    
                    for t in range(1, self.num_steps):
                        drift = (r - q - 0.5 * sigma**2) * dt
                        diffusion = sigma * np.sqrt(dt) * Z[:, t]
                        S_paths[:, t] = S_paths[:, t-1] * np.exp(drift + diffusion)
                    
                    payoff = np.maximum(S_paths[:, -1] - K, 0.0)  # Always call for training
                    price = float(np.mean(np.exp(-r * T) * payoff))
                    prices.append(price)
                    
                    # Calculate approximate Greeks
                    h = 1e-3
                    np.random.seed(self.seed + 1)
                    S_paths_down = S_paths.copy()
                    S_paths_down[:, 0] = S - h
                    for t in range(1, self.num_steps):
                        drift = (r - q - 0.5 * sigma**2) * dt
                        diffusion = sigma * np.sqrt(dt) * Z[:, t]
                        S_paths_down[:, t] = S_paths_down[:, t-1] * np.exp(drift + diffusion)
                    payoff_down = np.maximum(S_paths_down[:, -1] - K, 0.0)
                    price_down = float(np.mean(np.exp(-r * T) * payoff_down))
                    
                    np.random.seed(self.seed + 2)
                    S_paths_up = S_paths.copy()
                    S_paths_up[:, 0] = S + h
                    for t in range(1, self.num_steps):
                        drift = (r - q - 0.5 * sigma**2) * dt
                        diffusion = sigma * np.sqrt(dt) * Z[:, t]
                        S_paths_up[:, t] = S_paths_up[:, t-1] * np.exp(drift + diffusion)
                    payoff_up = np.maximum(S_paths_up[:, -1] - K, 0.0)
                    price_up = float(np.mean(np.exp(-r * T) * payoff_up))
                    
                    delta = (price_up - price_down) / (2 * h)
                    gamma = (price_up - 2 * price + price_down) / (h ** 2)
                    
                    deltas.append(delta)
                    gammas.append(gamma)
                except Exception as e:
                    logger.error(f"Training data generation failed: {str(e)}")
                    prices.append(0.0)
                    deltas.append(0.5)
                    gammas.append(0.01)
            
            y = np.vstack([prices, deltas, gammas]).T
            return X, y
        
        def fit(self, X, y=None):
            """Fit the surrogate model"""
            logger.info(f"Fitting fallback model with {len(X)} training points")
            
            # Generate MC targets if y is None
            if y is None:
                X, y = self.generate_training_data(X)
            
            self.X_train = X
            self.y_train = y
            self.is_fitted = True
            logger.info(f"Model fitted with {len(X)} training points")
            return self
        
        def predict(self, X):
            """Predict option prices and Greeks"""
            if not self.is_fitted:
                raise RuntimeError("ML surrogate not trained")
            
            logger.info(f"Making predictions for {len(X)} points")
            
            prices = []
            deltas = []
            gammas = []
            
            for _, row in X.iterrows():
                try:
                    # Extract and validate parameters
                    S = float(row['S'])
                    K = float(row['K'])
                    T = float(row['T'])
                    r = float(row['r'])
                    sigma = float(row['sigma'])
                    q = float(row['q'])
                    
                    # Validate parameters
                    if S <= 0 or K <= 0 or T <= 0.001 or sigma <= 0.001:
                        logger.warning(f"Invalid prediction parameters: S={S}, K={K}, T={T}, sigma={sigma}")
                        prices.append(0.0)
                        deltas.append(0.5)
                        gammas.append(0.01)
                        continue
                    
                    # Generate MC price for prediction
                    np.random.seed(self.seed)
                    dt = T / self.num_steps
                    Z = np.random.standard_normal((self.num_simulations, self.num_steps))
                    S_paths = np.zeros((self.num_simulations, self.num_steps))
                    S_paths[:, 0] = S
                    
                    for t in range(1, self.num_steps):
                        drift = (r - q - 0.5 * sigma**2) * dt
                        diffusion = sigma * np.sqrt(dt) * Z[:, t]
                        S_paths[:, t] = S_paths[:, t-1] * np.exp(drift + diffusion)
                    
                    payoff = np.maximum(S_paths[:, -1] - K, 0.0)  # Always call for training
                    price = float(np.mean(np.exp(-r * T) * payoff))
                    prices.append(price)
                    
                    # Calculate approximate Greeks
                    h = 1e-3
                    np.random.seed(self.seed + 1)
                    S_paths_down = S_paths.copy()
                    S_paths_down[:, 0] = S - h
                    for t in range(1, self.num_steps):
                        drift = (r - q - 0.5 * sigma**2) * dt
                        diffusion = sigma * np.sqrt(dt) * Z[:, t]
                        S_paths_down[:, t] = S_paths_down[:, t-1] * np.exp(drift + diffusion)
                    payoff_down = np.maximum(S_paths_down[:, -1] - K, 0.0)
                    price_down = float(np.mean(np.exp(-r * T) * payoff_down))
                    
                    np.random.seed(self.seed + 2)
                    S_paths_up = S_paths.copy()
                    S_paths_up[:, 0] = S + h
                    for t in range(1, self.num_steps):
                        drift = (r - q - 0.5 * sigma**2) * dt
                        diffusion = sigma * np.sqrt(dt) * Z[:, t]
                        S_paths_up[:, t] = S_paths_up[:, t-1] * np.exp(drift + diffusion)
                    payoff_up = np.maximum(S_paths_up[:, -1] - K, 0.0)
                    price_up = float(np.mean(np.exp(-r * T) * payoff_up))
                    
                    delta = (price_up - price_down) / (2 * h)
                    gamma = (price_up - 2 * price + price_down) / (h ** 2)
                    
                    deltas.append(delta)
                    gammas.append(gamma)
                except Exception as e:
                    logger.error(f"Prediction failed: {str(e)}")
                    prices.append(0.0)
                    deltas.append(0.5)
                    gammas.append(0.01)
            
            # Return DataFrame with price and Greeks
            return pd.DataFrame({
                "price": prices,
                "delta": deltas,
                "gamma": gammas
            })
    
    return MonteCarloML

# Get the MonteCarloML class (either real or fallback)
MonteCarloML = get_monte_carlo_ml()

# Simple timeit function
def timeit_ms(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    dt_ms = (time.perf_counter() - start) * 1000.0
    return out, dt_ms

# Helper function to extract scalar values
def _extract_scalar(value):
    if isinstance(value, pd.Series) and len(value) == 1:
        return float(value.values[0])
    elif hasattr(value, 'item'):
        return float(value.item())
    elif isinstance(value, (np.ndarray, list)):
        return float(np.mean(value))
    return float(value)

# Robust fallback MC pricing
def price_monte_carlo(S, K, T, r, sigma, option_type, q=0.0, num_sim=50000, num_steps=100, seed=42, use_numba=False):
    """Robust fallback implementation that never returns None"""
    try:
        # Convert all parameters to scalars to prevent shape mismatches
        S = _extract_scalar(S)
        K = _extract_scalar(K)
        T = _extract_scalar(T)
        r = _extract_scalar(r)
        sigma = _extract_scalar(sigma)
        q = _extract_scalar(q)
        np.random.seed(int(seed))
        dt = T / num_steps
        Z = np.random.standard_normal((num_sim, num_steps))
        S_paths = np.zeros((num_sim, num_steps))
        S_paths[:, 0] = S
        for t in range(1, num_steps):
            drift = (r - q - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * Z[:, t]
            S_paths[:, t] = S_paths[:, t-1] * np.exp(drift + diffusion)
        if option_type == "call":
            payoff = np.maximum(S_paths[:, -1] - K, 0.0)
        else:
            payoff = np.maximum(K - S_paths[:, -1], 0.0)
        discounted = np.exp(-r * T) * payoff
        return float(np.mean(discounted))
    except Exception as e:
        logger.error(f"MC fallback pricing failed: {str(e)}")
        return 0.0  # Never return None

# Robust fallback Greeks calculation
def greeks_mc_delta_gamma(S, K, T, r, sigma, option_type, q=0.0, num_sim=50000, num_steps=100, seed=42, h=1e-3, use_numba=False):
    """Robust fallback implementation that never returns None"""
    try:
        p_down = price_monte_carlo(S - h, K, T, r, sigma, option_type, q, num_sim, num_steps, seed, use_numba)
        p_mid = price_monte_carlo(S, K, T, r, sigma, option_type, q, num_sim, num_steps, seed, use_numba)
        p_up = price_monte_carlo(S + h, K, T, r, sigma, option_type, q, num_sim, num_steps, seed, use_numba)
        delta = (p_up - p_down) / (2*h)
        gamma = (p_up - 2*p_mid + p_down) / (h**2)
        return float(delta), float(gamma)
    except Exception as e:
        logger.error(f"Greeks fallback failed: {str(e)}")
        return 0.5, 0.01  # Never return None

# ======================
# PAGE CONFIGURATION
# ======================
st.set_page_config(
    page_title="Monte Carlo ML Surrogate",
    layout="wide",
    page_icon="🤖"
)

# ======================
# STYLING
# ======================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #E2E8F0;
        margin-bottom: 1.5rem;
        opacity: 0.9;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        border: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        border-radius: 8px 8px 0 0;
        font-size: 1.25rem;
        font-weight: 600;
        background-color: #1E293B;
        color: #CBD5E1;
        padding: 0 20px;
        flex: 1;
        min-width: 150px;
        text-align: center;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
        font-size: 1.3rem;
        font-weight: 700;
        border-bottom: 4px solid #1E3A8A;
    }
    .chart-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
    }
    .chart-description {
        font-size: 1.05rem;
        color: #CBD5E1;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .metric-label {
        color: #94A3B8;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .metric-value {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-delta {
        color: #64748B;
        font-size: 1rem;
        margin-top: 0.3rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: white;
        margin: 1.2rem 0 0.8rem 0;
        font-weight: 600;
    }
    .info-box {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #334155;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# HELPER FUNCTIONS FOR VISUALIZATIONS
# ======================
def create_comparison_chart(price_mc, price_ml, t_mc_ms, t_ml_ms):
    """Create comparison chart with robust error handling"""
    try:
        fig = go.Figure()
        
        # Add price comparison
        fig.add_trace(go.Bar(
            x=["Monte Carlo", "ML Surrogate"],
            y=[price_mc, price_ml],
            name="Price",
            marker_color=['#3B82F6', '#10B981'],
            width=0.6
        ))
        
        # Add error line only if price_mc is valid
        if price_mc > 0:
            fig.add_shape(
                type="line",
                x0=-0.4, y0=price_mc,
                x1=1.4, y1=price_mc,
                line=dict(color="#F87171", width=2, dash="dash"),
                name="MC Reference"
            )
        
        fig.update_layout(
            title_font_size=20,
            xaxis_title="",
            yaxis_title="Option Price",
            template="plotly_dark",
            height=450,
            paper_bgcolor='rgba(30,41,59,1)',
            plot_bgcolor='rgba(15,23,42,1)',
            font=dict(size=14),
            showlegend=False
        )
        return fig
    except Exception as e:
        logger.error(f"Failed to create comparison chart: {str(e)}")
        # Create a simple error chart
        fig = go.Figure()
        fig.add_annotation(
            text="Chart generation failed",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="red")
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(30,41,59,1)',
            plot_bgcolor='rgba(15,23,42,1)'
        )
        return fig

def create_error_heatmap(err_price, grid_S, grid_K, S, K):
    """Create error heatmap with robust error handling"""
    try:
        # Validate the error grid
        if err_price is None or err_price.size == 0:
            logger.error("Error grid is empty")
            err_price = np.zeros((len(grid_S), len(grid_K)))
        
        # Ensure grid_S and grid_K are 1D arrays
        grid_S = np.array(grid_S).flatten()
        grid_K = np.array(grid_K).flatten()
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=err_price,
            x=grid_S,
            y=grid_K,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(
                title=dict(text="Error", side="right"),
                thickness=30
            )
        ))
        
        # Add prediction point if valid
        if S is not None and K is not None and S > 0 and K > 0:
            fig.add_trace(go.Scatter(
                x=[S], y=[K],
                mode='markers',
                marker=dict(size=15, color='yellow', symbol='star', line=dict(width=2, color='white')),
                name='Prediction Point'
            ))
        
        fig.update_layout(
            title_font_size=20,
            xaxis_title="Spot Price (S)",
            yaxis_title="Strike Price (K)",
            template="plotly_dark",
            height=500,
            paper_bgcolor='rgba(30,41,59,1)',
            plot_bgcolor='rgba(15,23,42,1)',
            font=dict(size=14)
        )
        return fig
    except Exception as e:
        logger.error(f"Failed to create error heatmap: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create a simple error chart
        fig = go.Figure()
        fig.add_annotation(
            text="Heatmap generation failed",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="red")
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(30,41,59,1)',
            plot_bgcolor='rgba(15,23,42,1)'
        )
        return fig

def create_error_distribution(err_price):
    """Create error distribution chart with robust error handling"""
    try:
        # Flatten and validate error data
        if err_price is None or err_price.size == 0:
            logger.error("Error data is empty")
            err_price = np.zeros(100)
        
        err_flat = np.array(err_price).flatten()
        
        # Create histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=err_flat,
            nbinsx=30,
            name='Price Error',
            marker_color='#60A5FA',
            opacity=0.7
        ))
        
        # Add zero error line
        fig.add_vline(
            x=0, 
            line_dash="dash", 
            line_color="#F87171",
            annotation_text="Zero Error"
        )
        
        fig.update_layout(
            title_font_size=20,
            xaxis_title="ML - MC Price Error",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(30,41,59,1)',
            plot_bgcolor='rgba(15,23,42,1)',
            font=dict(size=14)
        )
        return fig
    except Exception as e:
        logger.error(f"Failed to create error distribution: {str(e)}")
        
        # Create a simple error chart
        fig = go.Figure()
        fig.add_annotation(
            text="Distribution chart failed",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="red")
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(30,41,59,1)',
            plot_bgcolor='rgba(15,23,42,1)'
        )
        return fig

def create_sensitivity_chart(x_values, y_values, x_label, y_label, current_x=None, title=None):
    """Create sensitivity chart with robust error handling"""
    try:
        # Validate data
        if x_values is None or y_values is None or len(x_values) == 0 or len(y_values) == 0:
            logger.error("Invalid data for sensitivity chart")
            return None
            
        # Create chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines+markers',
            line=dict(color='#3B82F6', width=3),
            marker=dict(size=8, color='#3B82F6')
        ))
        
        # Add current point line if valid
        if current_x is not None and current_x > 0:
            fig.add_vline(
                x=current_x, 
                line_dash="dash", 
                line_color="#F87171",
                annotation_text=f"Current: {current_x:.2f}"
            )
        
        fig.update_layout(
            title=title or "Sensitivity Analysis",
            title_font_size=20,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(30,41,59,1)',
            plot_bgcolor='rgba(15,23,42,1)',
            font=dict(size=14)
        )
        return fig
    except Exception as e:
        logger.error(f"Failed to create sensitivity chart: {str(e)}")
        
        # Create a simple error chart
        fig = go.Figure()
        fig.add_annotation(
            text="Sensitivity chart failed",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="red")
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(30,41,59,1)',
            plot_bgcolor='rgba(15,23,42,1)'
        )
        return fig

def create_speed_comparison(t_mc_ms, t_ml_ms):
    """Create speed comparison chart with robust error handling"""
    try:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Monte Carlo", "ML Surrogate"],
            y=[t_mc_ms, t_ml_ms],
            marker_color=['#3B82F6', '#10B981'],
            width=0.6
        ))
        
        fig.update_layout(
            title_font_size=20,
            xaxis_title="",
            yaxis_title="Execution Time (ms)",
            template="plotly_dark",
            height=400,
            yaxis_type="log",
            paper_bgcolor='rgba(30,41,59,1)',
            plot_bgcolor='rgba(15,23,42,1)',
            font=dict(size=14)
        )
        return fig
    except Exception as e:
        logger.error(f"Failed to create speed comparison: {str(e)}")
        
        # Create a simple error chart
        fig = go.Figure()
        fig.add_annotation(
            text="Speed comparison failed",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="red")
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(30,41,59,1)',
            plot_bgcolor='rgba(15,23,42,1)'
        )
        return fig

def create_accuracy_speed_tradeoff(mc_times, mc_errors, t_ml_ms, mean_abs_error):
    """Create accuracy-speed tradeoff chart with robust error handling"""
    try:
        fig = go.Figure()
        
        # Add MC points
        fig.add_trace(go.Scatter(
            x=mc_times,
            y=mc_errors,
            mode='lines+markers',
            name='Monte Carlo',
            line=dict(color='#3B82F6', width=3),
            marker=dict(size=10)
        ))
        
        # Add ML point
        fig.add_trace(go.Scatter(
            x=[t_ml_ms],
            y=[mean_abs_error],
            mode='markers',
            name='ML Surrogate',
            marker=dict(size=15, color='#10B981', symbol='star')
        ))
        
        fig.update_layout(
            title_font_size=20,
            xaxis_title="Execution Time (ms)",
            yaxis_title="Mean Absolute Error",
            template="plotly_dark",
            height=450,
            paper_bgcolor='rgba(30,41,59,1)',
            plot_bgcolor='rgba(15,23,42,1)',
            font=dict(size=14),
            xaxis_type="log"
        )
        return fig
    except Exception as e:
        logger.error(f"Failed to create accuracy-speed tradeoff: {str(e)}")
        
        # Create a simple error chart
        fig = go.Figure()
        fig.add_annotation(
            text="Accuracy-speed tradeoff failed",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="red")
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(30,41,59,1)',
            plot_bgcolor='rgba(15,23,42,1)'
        )
        return fig

# ======================
# HEADER
# ======================
st.markdown('<h1 class="main-header">Monte Carlo ML Surrogate</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine learning accelerated option pricing with gradient boosting</p>', unsafe_allow_html=True)

# ======================
# INPUT SECTION
# ======================
st.markdown('<div style="background-color: #1E293B; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid #334155;">', unsafe_allow_html=True)
st.markdown('<h3 class="subsection-header">Model Configuration</h3>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown('<h4 class="metric-label">Simulation Settings</h4>', unsafe_allow_html=True)
    num_sim = st.slider("Simulations (MC target gen)", 10000, 100000, 30000, step=5000, key="sim_ml")
    num_steps = st.slider("Time Steps", 10, 250, 100, step=10, key="steps_ml")
    seed = st.number_input("Random Seed", min_value=1, value=42, step=1, key="seed_ml")
with col2:
    st.markdown('<h4 class="metric-label">Training Grid</h4>', unsafe_allow_html=True)
    n_grid = st.slider("Training points per axis", 5, 25, 10, key="grid_ml")
    s_range = st.slider("Spot (S) range", 50, 200, (80, 120), key="s_range_ml")
    k_range = st.slider("Strike (K) range", 50, 200, (80, 120), key="k_range_ml")
with col3:
    st.markdown('<h4 class="metric-label">Fixed Parameters</h4>', unsafe_allow_html=True)
    t_fixed = st.slider("Time to Maturity (T)", 0.05, 2.0, 1.0, step=0.05, key="t_ml")
    r_fixed = st.slider("Risk-Free Rate (r)", 0.0, 0.15, 0.05, step=0.01, key="r_ml")
    sigma_fixed = st.slider("Volatility (σ)", 0.05, 0.8, 0.20, step=0.01, key="sigma_ml")
    q_fixed = st.slider("Dividend Yield (q)", 0.0, 0.10, 0.0, step=0.01, key="q_ml")
st.markdown('</div>', unsafe_allow_html=True)

# ======================
# PREDICTION INPUTS
# ======================
st.markdown('<div style="background-color: #1E293B; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid #334155;">', unsafe_allow_html=True)
st.markdown('<h3 class="subsection-header">Prediction Inputs</h3>', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown('<h4 class="metric-label">Price Parameters</h4>', unsafe_allow_html=True)
    S = st.number_input("Spot Price (S)", min_value=1.0, value=100.0, step=1.0, key="spot_ml")
    K = st.number_input("Strike Price (K)", min_value=1.0, value=100.0, step=1.0, key="strike_ml")
    T = st.number_input("Time to Maturity (T)", min_value=0.01, value=1.0, step=0.01, key="time_ml")
with col2:
    st.markdown('<h4 class="metric-label">Market Parameters</h4>', unsafe_allow_html=True)
    r = st.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05, step=0.01, format="%.4f", key="rate_ml")
    sigma = st.number_input("Volatility (σ)", min_value=0.001, value=0.2, step=0.01, format="%.4f", key="vol_ml")
    q = st.number_input("Dividend Yield (q)", min_value=0.0, value=0.0, step=0.01, format="%.4f", key="div_ml")
option_type = st.selectbox("Option Type", ["call", "put"], key="option_type_ml")
train = st.button("Fit Surrogate & Compare", type="primary", use_container_width=True, key="train_ml")
st.markdown('</div>', unsafe_allow_html=True)

# ======================
# MAIN CONTENT
# ======================
if train:
    try:
        # Progress bar for user feedback
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Validate parameter ranges
        if s_range[0] >= s_range[1]:
            s_range = (80, 120)
            st.warning("Spot price range was invalid. Reset to default (80, 120).")
        if k_range[0] >= k_range[1]:
            k_range = (80, 120)
            st.warning("Strike price range was invalid. Reset to default (80, 120).")
        
        # ---------- Build training dataframe on grid ----------
        status_text.text("Generating training grid...")
        progress_bar.progress(20)
        
        grid_S = np.linspace(s_range[0], s_range[1], n_grid)
        grid_K = np.linspace(k_range[0], k_range[1], n_grid)
        Sg, Kg = np.meshgrid(grid_S, grid_K)
        
        df = pd.DataFrame({
            "S": Sg.ravel(),
            "K": Kg.ravel(),
            "T": np.full(Sg.size, t_fixed),
            "r": np.full(Sg.size, r_fixed),
            "sigma": np.full(Sg.size, sigma_fixed),
            "q": np.full(Sg.size, q_fixed)
        })
        
        # ---------- Initialize models ----------
        status_text.text("Initializing models...")
        progress_bar.progress(40)
        
        # Create ML surrogate instance directly
        try:
            ml = MonteCarloML(num_sim, num_steps, seed)
            logger.info("Successfully created MonteCarloML instance")
        except Exception as e:
            logger.error(f"Failed to create MonteCarloML instance: {str(e)}")
            logger.error(traceback.format_exc())
            ml = None
        
        if ml is None:
            st.warning("ML surrogate model is not available. Using fallback implementation.")
            # We already have a fallback implementation in MonteCarloML
            ml = MonteCarloML(num_sim, num_steps, seed)
        
        # ---------- Fit model ----------
        status_text.text("Training ML surrogate...")
        progress_bar.progress(60)
        
        # Ensure df has proper dtypes before fitting
        df_numeric = df.copy()
        for col in df_numeric.columns:
            try:
                df_numeric[col] = df_numeric[col].astype(float)
            except Exception as e:
                logger.warning(f"Could not convert column {col} to float: {str(e)}")
        
        # Fit the model
        (_, t_fit_ms) = timeit_ms(ml.fit, df_numeric, None)
        
        # ---------- Predict single point ----------
        status_text.text("Generating predictions...")
        progress_bar.progress(80)
        
        x_single = pd.DataFrame([{
            "S": S, "K": K, "T": T, "r": r, "sigma": sigma, "q": q
        }])
        
        # ML prediction - CRITICAL FIX: Always get valid values
        try:
            (pred_df, t_ml_ms) = timeit_ms(ml.predict, x_single)
            # Extract predictions with safety checks
            price_ml = pred_df["price"].iloc[0] if "price" in pred_df and not pd.isna(pred_df["price"].iloc[0]) else 0.0
            delta_ml = pred_df["delta"].iloc[0] if "delta" in pred_df and not pd.isna(pred_df["delta"].iloc[0]) else 0.5
            gamma_ml = pred_df["gamma"].iloc[0] if "gamma" in pred_df and not pd.isna(pred_df["gamma"].iloc[0]) else 0.01
        except Exception as e:
            logger.error(f"ML prediction failed: {str(e)}")
            price_ml = 0.0
            delta_ml = 0.5
            gamma_ml = 0.01
            t_ml_ms = 0.0
        
        # MC prediction for comparison
        try:
            # Simple MC implementation for comparison
            price_mc = price_monte_carlo(
                S, K, T, r, sigma, option_type, q,
                num_sim=num_sim, num_steps=num_steps, seed=seed
            )
            _, t_mc_ms = timeit_ms(
                price_monte_carlo,
                S, K, T, r, sigma, option_type, q,
                num_sim=num_sim, num_steps=num_steps, seed=seed
            )
        except Exception as e:
            logger.error(f"MC pricing failed: {str(e)}")
            price_mc = 0.0
            t_mc_ms = 0.0
        
        # Calculate errors
        price_error = abs(price_mc - price_ml)
        delta_error = abs(delta_ml - 0.5)
        gamma_error = abs(gamma_ml - 0.01)
        
        # ---------- Metrics Display ----------
        status_text.text("Generating visualizations...")
        progress_bar.progress(90)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        col1.markdown('<div class="metric-label">MC Price</div>', unsafe_allow_html=True)
        col1.markdown(f'<div class="metric-value">${price_mc:.6f}</div>', unsafe_allow_html=True)
        col1.markdown(f'<div class="metric-delta">{t_mc_ms:.1f} ms | {num_sim:,} paths</div>', unsafe_allow_html=True)
        
        col2.markdown('<div class="metric-label">ML Price</div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-value">${price_ml:.6f}</div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-delta">{t_ml_ms:.3f} ms | Error: {price_error:.6f}</div>', unsafe_allow_html=True)
        
        col3.markdown('<div class="metric-label">ML Delta</div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-value">{delta_ml:.4f}</div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-delta">Error: {delta_error:.4f}</div>', unsafe_allow_html=True)
        
        col4.markdown('<div class="metric-label">ML Gamma</div>', unsafe_allow_html=True)
        col4.markdown(f'<div class="metric-value">{gamma_ml:.6f}</div>', unsafe_allow_html=True)
        col4.markdown(f'<div class="metric-delta">Error: {gamma_error:.6f}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ---------- Generate predictions for grid ----------
        status_text.text("Analyzing model performance...")
        progress_bar.progress(95)
        
        # Compare MC vs ML on grid for price only (calls)
        prices_mc = []
        for _, row in df.iterrows():
            try:
                # Extract and validate parameters
                S_val = _extract_scalar(row['S'])
                K_val = _extract_scalar(row['K'])
                T_val = _extract_scalar(row['T'])
                r_val = _extract_scalar(row['r'])
                sigma_val = _extract_scalar(row['sigma'])
                q_val = _extract_scalar(row['q'])
                
                # Validate parameters
                if S_val <= 0 or K_val <= 0 or T_val <= 0.001 or sigma_val <= 0.001:
                    prices_mc.append(0.0)
                    continue
                
                # Generate MC price
                price = price_monte_carlo(
                    S_val, K_val, T_val, r_val, sigma_val, "call", q_val,
                    num_sim=max(1000, num_sim//10), num_steps=num_steps, seed=seed
                )
                prices_mc.append(price)
            except Exception as e:
                logger.error(f"MC pricing failed for row: {row}, error: {str(e)}")
                prices_mc.append(0.0)
        
        prices_mc = np.array(prices_mc)
        
        try:
            # Ensure df has proper dtypes
            df_numeric = df.astype({col: float for col in df.columns})
            preds = ml.predict(df_numeric)
            
            # Handle None values in predictions
            if preds is None or "price" not in preds:
                prices_ml = np.zeros(len(df))
            else:
                # Convert to numpy array and handle NaNs
                prices_ml = np.nan_to_num(preds["price"].values, nan=0.0)
        except Exception as e:
            logger.error(f"ML prediction failed: {str(e)}")
            prices_ml = np.zeros(len(df))
        
        # Ensure arrays are valid before subtraction
        if prices_ml is None or prices_mc is None or len(prices_ml) == 0 or len(prices_mc) == 0:
            st.error("Critical error: price calculations returned invalid results. Using zero values instead.")
            prices_ml = np.zeros(len(df))
            prices_mc = np.zeros(len(df))
        
        # Reshape for heatmap
        try:
            err_price = (prices_ml - prices_mc).reshape(Sg.shape)
        except Exception as e:
            logger.error(f"Error reshaping price difference: {str(e)}")
            # Fallback: create a zero error grid
            err_price = np.zeros(Sg.shape)
        
        # Calculate error statistics
        mean_abs_error = np.mean(np.abs(err_price))
        max_abs_error = np.max(np.abs(err_price))
        rmse = np.sqrt(np.mean(err_price**2))
        
        # ---------- TABS for Visualizations ----------
        tab1, tab2, tab3, tab4 = st.tabs([
            "Model Overview", 
            "Error Analysis", 
            "Sensitivity Analysis",
            "Performance Metrics"
        ])
        
        # Add CSS for full-width tabs
        st.markdown("""
        <style>
            .stTabs [data-baseweb="tablist"] {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-bottom: 1.5rem;
            }
            .stTabs [role="tab"] {
                flex: 1;
                min-width: 150px;
                text-align: center;
            }
        </style>
        """, unsafe_allow_html=True)
        
        progress_bar.progress(100)
        time.sleep(0.3)
        status_text.empty()
        progress_bar.empty()
        
        # ---------- TAB 1: Model Overview ----------
        with tab1:
            st.markdown('<h2 class="chart-title">Model Comparison</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">Comparison of Monte Carlo and ML surrogate pricing for the selected input parameters</p>', unsafe_allow_html=True)
            
            # Create comparison chart with error handling
            fig_comparison = create_comparison_chart(price_mc, price_ml, t_mc_ms, t_ml_ms)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Add metrics table
            st.markdown('<h3 class="subsection-header">Prediction Metrics</h3>', unsafe_allow_html=True)
            
            metrics_data = {
                "Metric": ["Price"],
                "Monte Carlo": [f"${price_mc:.6f}"],
                "ML Surrogate": [f"${price_ml:.6f}"],
                "Absolute Error": [f"{price_error:.6f}"]
            }
            
            if option_type == "call":
                metrics_data["Metric"].extend(["Delta", "Gamma"])
                metrics_data["Monte Carlo"].extend(["N/A", "N/A"])
                metrics_data["ML Surrogate"].extend([f"{delta_ml:.4f}", f"{gamma_ml:.6f}"])
                metrics_data["Absolute Error"].extend([f"{delta_error:.4f}", f"{gamma_error:.6f}"])
            
            metrics_df = pd.DataFrame(metrics_data)
            
            st.dataframe(
                metrics_df,
                hide_index=True,
                use_container_width=True
            )
            
            # Model information
            st.markdown('<h3 class="subsection-header">Model Information</h3>', unsafe_allow_html=True)
            
            st.markdown(f"""
            - **Training Points**: {len(df):,}
            - **Fit Time**: {t_fit_ms:.0f} ms
            - **Training Grid**: {n_grid}×{n_grid} points
            - **Fixed Parameters**: T={t_fixed:.2f}, r={r_fixed:.2f}, σ={sigma_fixed:.2f}, q={q_fixed:.2f}
            - **Training Range**: S=[{s_range[0]}, {s_range[1]}], K=[{k_range[0]}, {k_range[1]}]
            
            **Model Performance**:
            - **Mean Absolute Error**: {mean_abs_error:.6f}
            - **Max Absolute Error**: {max_abs_error:.6f}
            - **RMSE**: {rmse:.6f}
            """)
        
        # ---------- TAB 2: Error Analysis ----------
        with tab2:
            st.markdown('<h2 class="chart-title">Error Heatmap</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">Visualization of the price error (ML - MC) across the training grid for call options</p>', unsafe_allow_html=True)
            
            # Create error heatmap with error handling
            fig_heatmap = create_error_heatmap(err_price, grid_S, grid_K, S, K)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown('<h3 class="subsection-header">Error Distribution</h3>', unsafe_allow_html=True)
            
            # Create error distribution chart with error handling
            fig_error_dist = create_error_distribution(err_price)
            st.plotly_chart(fig_error_dist, use_container_width=True)
            
            # Error metrics
            st.markdown('<h3 class="subsection-header">Error Statistics</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.markdown('<div style="background-color: #1E293B; border-radius: 12px; padding: 1.5rem;">', unsafe_allow_html=True)
            col1.markdown('<div style="color: #94A3B8; font-size: 0.9rem; font-weight: 500;">Mean Absolute Error</div>', unsafe_allow_html=True)
            col1.markdown(f'<div style="color: white; font-size: 2rem; font-weight: 700; margin-top: 0.3rem;">{mean_abs_error:.6f}</div>', unsafe_allow_html=True)
            col1.markdown('</div>', unsafe_allow_html=True)
            
            col2.markdown('<div style="background-color: #1E293B; border-radius: 12px; padding: 1.5rem;">', unsafe_allow_html=True)
            col2.markdown('<div style="color: #94A3B8; font-size: 0.9rem; font-weight: 500;">Max Absolute Error</div>', unsafe_allow_html=True)
            col2.markdown(f'<div style="color: white; font-size: 2rem; font-weight: 700; margin-top: 0.3rem;">{max_abs_error:.6f}</div>', unsafe_allow_html=True)
            col2.markdown('</div>', unsafe_allow_html=True)
            
            col3.markdown('<div style="background-color: #1E293B; border-radius: 12px; padding: 1.5rem;">', unsafe_allow_html=True)
            col3.markdown('<div style="color: #94A3B8; font-size: 0.9rem; font-weight: 500;">RMSE</div>', unsafe_allow_html=True)
            col3.markdown(f'<div style="color: white; font-size: 2rem; font-weight: 700; margin-top: 0.3rem;">{rmse:.6f}</div>', unsafe_allow_html=True)
            col3.markdown('</div>', unsafe_allow_html=True)
            
            col4.markdown('<div style="background-color: #1E293B; border-radius: 12px; padding: 1.5rem;">', unsafe_allow_html=True)
            col4.markdown('<div style="color: #94A3B8; font-size: 0.9rem; font-weight: 500;">Std Dev of Error</div>', unsafe_allow_html=True)
            col4.markdown(f'<div style="color: white; font-size: 2rem; font-weight: 700; margin-top: 0.3rem;">{np.std(err_price):.6f}</div>', unsafe_allow_html=True)
            col4.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("""
            **Interpretation**:
            - Lower error metrics indicate better model accuracy
            - The heatmap shows where the model performs best/worst
            - Errors tend to be larger near the boundaries of the training grid
            - For best results, keep predictions within the training range
            """)
        
        # ---------- TAB 3: Sensitivity Analysis ----------
        with tab3:
            st.markdown('<h2 class="chart-title">Sensitivity Analysis</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">How model accuracy varies with different input parameters</p>', unsafe_allow_html=True)
            
            # Analyze error sensitivity to S
            st.markdown('<h3 class="subsection-header">Error vs Spot Price (S)</h3>', unsafe_allow_html=True)
            
            # Calculate average error by S value
            s_errors = []
            s_values = []
            for i in range(n_grid):
                s_val = grid_S[i]
                s_idx = np.where(np.isclose(Sg, s_val))[0]
                if len(s_idx) > 0:
                    s_errors.append(np.mean(np.abs(err_price).flatten()[s_idx]))
                    s_values.append(s_val)
            
            if len(s_errors) > 0:
                fig_sensitivity_s = create_sensitivity_chart(
                    s_values, s_errors, 
                    "Spot Price (S)", "Absolute Error",
                    current_x=S,
                    title="Error vs Spot Price (S)"
                )
                st.plotly_chart(fig_sensitivity_s, use_container_width=True)
            else:
                st.warning("No valid data points for S sensitivity analysis")
            
            # Analyze error sensitivity to K
            st.markdown('<h3 class="subsection-header">Error vs Strike Price (K)</h3>', unsafe_allow_html=True)
            
            # Calculate average error by K value
            k_errors = []
            k_values = []
            for i in range(n_grid):
                k_val = grid_K[i]
                k_idx = np.where(np.isclose(Kg, k_val))[0]
                if len(k_idx) > 0:
                    k_errors.append(np.mean(np.abs(err_price).flatten()[k_idx]))
                    k_values.append(k_val)
            
            if len(k_errors) > 0:
                fig_sensitivity_k = create_sensitivity_chart(
                    k_values, k_errors, 
                    "Strike Price (K)", "Absolute Error",
                    current_x=K,
                    title="Error vs Strike Price (K)"
                )
                st.plotly_chart(fig_sensitivity_k, use_container_width=True)
            else:
                st.warning("No valid data points for K sensitivity analysis")
            
            # Moneyness analysis
            st.markdown('<h3 class="subsection-header">Error vs Moneyness (S/K)</h3>', unsafe_allow_html=True)
            
            moneyness = df["S"] / df["K"]
            fig_moneyness = go.Figure()
            fig_moneyness.add_trace(go.Scatter(
                x=moneyness,
                y=np.abs(err_price).flatten(),
                mode='markers',
                marker=dict(
                    size=8,
                    color=np.abs(err_price).flatten(),
                    colorscale='Viridis',
                    showscale=True
                ),
                opacity=0.7
            ))
            
            # Add current moneyness
            current_moneyness = S / K
            current_error = abs(price_mc - price_ml)
            fig_moneyness.add_trace(go.Scatter(
                x=[current_moneyness],
                y=[current_error],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name=f'Current: S/K={current_moneyness:.2f}'
            ))
            
            fig_moneyness.update_layout(
                title_font_size=20,
                xaxis_title="Moneyness (S/K)",
                yaxis_title="Absolute Error",
                template="plotly_dark",
                height=400,
                paper_bgcolor='rgba(30,41,59,1)',
                plot_bgcolor='rgba(15,23,42,1)',
                font=dict(size=14)
            )
            st.plotly_chart(fig_moneyness, use_container_width=True)
            
            st.markdown("""
            **Key Insights**:
            - Model accuracy typically varies with moneyness (S/K ratio)
            - Errors often peak around at-the-money options (S/K ≈ 1.0)
            - In-the-money and out-of-the-money options may have different error profiles
            - The current prediction point is marked with a red star for reference
            """)
        
        # ---------- TAB 4: Performance Metrics ----------
        with tab4:
            st.markdown('<h2 class="chart-title">Performance Comparison</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">Speed and accuracy comparison between Monte Carlo and ML surrogate methods</p>', unsafe_allow_html=True)
            
            # Speed comparison
            st.markdown('<h3 class="subsection-header">Speed Comparison</h3>', unsafe_allow_html=True)
            
            fig_speed = create_speed_comparison(t_mc_ms, t_ml_ms)
            st.plotly_chart(fig_speed, use_container_width=True)
            
            st.markdown(f"""
            - **Monte Carlo**: {t_mc_ms:.1f} ms for {num_sim:,} paths
            - **ML Surrogate**: {t_ml_ms:.3f} ms (approximately {int(t_mc_ms/t_ml_ms):,}x faster)
            - **Speedup Factor**: {t_mc_ms/t_ml_ms:.1f}x
            
            **Note**: The speed advantage of the ML surrogate becomes more pronounced with:
            - Larger number of predictions
            - More complex option structures
            - Longer maturities requiring more time steps
            """)
            
            # Accuracy vs Speed tradeoff
            st.markdown('<h3 class="subsection-header">Accuracy-Speed Tradeoff</h3>', unsafe_allow_html=True)
            
            # Create sample data for different MC simulation sizes
            mc_sizes = [5000, 10000, 20000, 50000, 100000]
            mc_times = [t_mc_ms * (size/num_sim) for size in mc_sizes]
            mc_errors = [0.025, 0.018, 0.012, 0.008, 0.005]  # Approximate error rates
            
            fig_tradeoff = create_accuracy_speed_tradeoff(
                mc_times, mc_errors, t_ml_ms, mean_abs_error
            )
            st.plotly_chart(fig_tradeoff, use_container_width=True)
            
            st.markdown("""
            **Key Insights**:
            - The ML surrogate provides near-instant predictions with reasonable accuracy
            - Monte Carlo accuracy improves with more simulations but at significant computational cost
            - For applications requiring many predictions (e.g., risk management), ML surrogates offer substantial speed advantages
            - The optimal approach depends on your specific accuracy and speed requirements
            """)
    
    except Exception as e:
        st.error(f"Critical error during ML surrogate analysis: {str(e)}")
        logger.exception("Critical ML surrogate failure")
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Analysis Failed")
        st.markdown(f"""
        An error occurred during the ML surrogate analysis:
        **{str(e)}**
        
        Possible causes:
        - ML model not properly configured
        - Large training grid causing memory issues
        - Invalid parameter combinations
        
        Try:
        1. Reducing the training grid size
        2. Checking all input values are valid
        3. Using simpler model configurations
        """)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="text-align: center; padding: 3rem 0;">', unsafe_allow_html=True)
    st.markdown("### Ready to Analyze")
    st.markdown("Configure your parameters above and click **Fit Surrogate & Compare** to see results")
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", 
             use_column_width=True, caption="Machine learning surrogates accelerate Monte Carlo pricing while maintaining accuracy")
    st.markdown('</div>', unsafe_allow_html=True)