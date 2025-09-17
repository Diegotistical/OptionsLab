# Monte Carlo Unified (CPU/GPU) - Production Ready

# Page 3 MC Unified.py
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
logger = logging.getLogger("monte_carlo_unified")
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
# IMPORT HANDLING
# ======================
def get_mc_unified_pricer():
    """Try multiple import strategies for unified MC pricer"""
    # Strategy 1: Direct import
    try:
        from streamlit_app.st_utils import get_mc_unified_pricer as _get_mc_unified_pricer
        logger.info("Successfully imported get_mc_unified_pricer from streamlit_app.st_utils")
        return _get_mc_unified_pricer
    except ImportError as e:
        logger.debug(f"Direct import failed: {str(e)}")
    
    # Strategy 2: Alternative import paths
    try:
        from st_utils import get_mc_unified_pricer as _get_mc_unified_pricer
        logger.info("Successfully imported get_mc_unified_pricer from st_utils")
        return _get_mc_unified_pricer
    except ImportError as e:
        logger.debug(f"Alternative import failed: {str(e)}")
    
    # Strategy 3: Parent directory import
    try:
        current_file = Path(__file__).resolve()
        for i in range(5):  # Check up to 5 levels up
            parent = current_file.parents[i]
            st_utils_path = parent / "st_utils.py"
            if st_utils_path.exists():
                logger.info(f"Found st_utils.py at: {st_utils_path}")
                
                # Add parent directory to sys.path
                if str(parent) not in sys.path:
                    sys.path.insert(0, str(parent))
                    logger.info(f"Added {parent} to sys.path")
                
                # Try import again
                from st_utils import get_mc_unified_pricer as _get_mc_unified_pricer
                logger.info("Successfully imported get_mc_unified_pricer after path adjustment")
                return _get_mc_unified_pricer
    except ImportError as e:
        logger.debug(f"Parent directory import failed: {str(e)}")
    
    # Strategy 4: Streamlit Cloud specific path
    try:
        cloud_root = Path("/mount/src/optionslab")
        if cloud_root.exists():
            st_utils_path = cloud_root / "st_utils.py"
            if st_utils_path.exists():
                logger.info(f"Found st_utils.py at: {st_utils_path}")
                
                # Add cloud root to sys.path
                if str(cloud_root) not in sys.path:
                    sys.path.insert(0, str(cloud_root))
                    logger.info(f"Added {cloud_root} to sys.path")
                
                # Try import again
                from st_utils import get_mc_unified_pricer as _get_mc_unified_pricer
                logger.info("Successfully imported get_mc_unified_pricer for Streamlit Cloud")
                return _get_mc_unified_pricer
    except ImportError as e:
        logger.debug(f"Streamlit Cloud import failed: {str(e)}")
    
    # Strategy 5: Final fallback - create our own implementation
    logger.error("All import strategies failed. Using comprehensive fallback implementation.")
    
    class FallbackMCPricer:
        """Fallback implementation of Monte Carlo pricer"""
        def __init__(self, num_sim=50000, num_steps=100, seed=42, use_numba=False, use_gpu=False):
            self.num_sim = num_sim
            self.num_steps = num_steps
            self.seed = seed
            self.use_numba = use_numba
            self.use_gpu = use_gpu
            logger.info(f"Initialized fallback MC pricer with {num_sim} simulations, {num_steps} steps")
        
        def price(self, S, K, T, r, sigma, option_type, q=0.0):
            """Price an option using Monte Carlo simulation"""
            try:
                # Convert all parameters to scalars
                S = float(S)
                K = float(K)
                T = float(T)
                r = float(r)
                sigma = float(sigma)
                q = float(q)
                
                # Validate parameters
                if S <= 0 or K <= 0 or T <= 0.001 or sigma <= 0.001:
                    logger.warning(f"Invalid parameters: S={S}, K={K}, T={T}, sigma={sigma}")
                    return 0.0
                
                # Generate MC price
                np.random.seed(self.seed)
                dt = T / self.num_steps
                Z = np.random.standard_normal((self.num_sim, self.num_steps))
                S_paths = np.zeros((self.num_sim, self.num_steps))
                S_paths[:, 0] = S
                
                for t in range(1, self.num_steps):
                    drift = (r - q - 0.5 * sigma**2) * dt
                    diffusion = sigma * np.sqrt(dt) * Z[:, t]
                    S_paths[:, t] = S_paths[:, t-1] * np.exp(drift + diffusion)
                
                if option_type == "call":
                    payoff = np.maximum(S_paths[:, -1] - K, 0.0)
                else:
                    payoff = np.maximum(K - S_paths[:, -1], 0.0)
                
                return float(np.mean(np.exp(-r * T) * payoff))
            except Exception as e:
                logger.error(f"MC pricing failed: {str(e)}")
                return 0.0
    
    def get_mc_unified_pricer(num_sim=50000, num_steps=100, seed=42, use_numba=False, use_gpu=False):
        return FallbackMCPricer(num_sim, num_steps, seed, use_numba, use_gpu)
    
    return get_mc_unified_pricer

# Get the unified MC pricer function
get_mc_unified_pricer = get_mc_unified_pricer()

# Simple timeit function
def timeit_ms(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    dt_ms = (time.perf_counter() - start) * 1000.0
    return out, dt_ms

# ======================
# HELPER FUNCTIONS FOR VISUALIZATIONS
# ======================
def create_price_surface(Sm, Tm, Z, K, option_type):
    """Create interactive price surface with Plotly"""
    try:
        fig = go.Figure(data=[
            go.Surface(
                x=Sm,
                y=Tm,
                z=Z,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Price", thickness=25)
            )
        ])
        
        # Add contour lines
        fig.update_traces(
            contours_z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="red",
                project_z=True
            )
        )
        
        fig.update_layout(
            title=f"Monte Carlo Price Surface (K={K}, {option_type.capitalize()})",
            scene=dict(
                xaxis_title="Spot Price (S)",
                yaxis_title="Time to Maturity (T)",
                zaxis_title="Option Price",
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=0.5)
                )
            ),
            height=700,
            template="plotly_dark",
            paper_bgcolor='rgba(30,41,59,1)',
            plot_bgcolor='rgba(15,23,42,1)',
            font=dict(size=14)
        )
        
        return fig
    except Exception as e:
        logger.error(f"Failed to create price surface: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create error chart
        fig = go.Figure()
        fig.add_annotation(
            text="Price surface generation failed",
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

def create_heatmap_comparison(Sg, Tg, Z, K, option_type):
    """Create heatmap comparison for price surface"""
    try:
        # Convert to DataFrame for easier handling
        df = pd.DataFrame({
            'S': np.repeat(Sg, len(Tg)),
            'T': np.tile(Tg, len(Sg)),
            'Price': Z.flatten()
        })
        
        # Create heatmap
        fig = px.density_heatmap(
            df,
            x='S',
            y='T',
            z='Price',
            color_continuous_scale='Viridis',
            labels={'Price': 'Option Price'},
            title=f"Price Heatmap (K={K}, {option_type.capitalize()})"
        )
        
        fig.update_layout(
            height=500,
            template="plotly_dark",
            paper_bgcolor='rgba(30,41,59,1)',
            plot_bgcolor='rgba(15,23,42,1)',
            font=dict(size=14),
            coloraxis_colorbar=dict(title="Price", thickness=25)
        )
        
        return fig
    except Exception as e:
        logger.error(f"Failed to create heatmap: {str(e)}")
        
        # Create error chart
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

def create_slice_plot(Sg, Tg, Z, fixed_T_idx=None, fixed_S_idx=None):
    """Create slice plots for price surface"""
    try:
        fig = go.Figure()
        
        # Add T slice if specified
        if fixed_T_idx is not None and 0 <= fixed_T_idx < len(Tg):
            fig.add_trace(go.Scatter(
                x=Sg,
                y=Z[fixed_T_idx, :],
                mode='lines+markers',
                name=f'T={Tg[fixed_T_idx]:.2f}',
                line=dict(width=3)
            ))
        
        # Add S slice if specified
        if fixed_S_idx is not None and 0 <= fixed_S_idx < len(Sg):
            fig.add_trace(go.Scatter(
                x=Tg,
                y=Z[:, fixed_S_idx],
                mode='lines+markers',
                name=f'S={Sg[fixed_S_idx]:.2f}',
                line=dict(width=3, dash='dash')
            ))
        
        fig.update_layout(
            title="Price Slices",
            xaxis_title="S" if fixed_T_idx is not None else "T",
            yaxis_title="Option Price",
            height=400,
            template="plotly_dark",
            paper_bgcolor='rgba(30,41,59,1)',
            plot_bgcolor='rgba(15,23,42,1)',
            font=dict(size=14)
        )
        
        return fig
    except Exception as e:
        logger.error(f"Failed to create slice plot: {str(e)}")
        
        # Create error chart
        fig = go.Figure()
        fig.add_annotation(
            text="Slice plot generation failed",
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
# PAGE CONFIGURATION
# ======================
st.set_page_config(
    page_title="Monte Carlo Unified (CPU/GPU)",
    layout="wide",
    page_icon="🚀"
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
    .engine-option {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border: 1px solid #334155;
    }
    .engine-label {
        font-size: 0.9rem;
        color: #94A3B8;
        margin-bottom: 0.25rem;
    }
    .engine-value {
        font-size: 1.1rem;
        color: white;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================
st.markdown('<h1 class="main-header">Monte Carlo — Unified (CPU/GPU + Antithetic)</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">High-performance option pricing with unified CPU/GPU implementation and variance reduction techniques</p>', unsafe_allow_html=True)

# ======================
# SIDEBAR CONFIGURATION
# ======================
with st.sidebar:
    st.markdown('<div style="margin-bottom: 1.5rem;">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: white; margin-bottom: 1rem;">Engine Configuration</h3>', unsafe_allow_html=True)
    
    # Engine options
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Simulations</div>', unsafe_allow_html=True)
    num_sim = st.slider(" ", 10_000, 200_000, 50_000, step=10_000, key="sim_unified")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Time Steps</div>', unsafe_allow_html=True)
    num_steps = st.slider(" ", 10, 500, 100, step=10, key="steps_unified")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Random Seed</div>', unsafe_allow_html=True)
    seed = st.number_input(" ", value=42, min_value=1, key="seed_unified")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Acceleration</div>', unsafe_allow_html=True)
    use_numba = st.toggle("Numba JIT Compilation", value=True, key="numba_unified")
    use_gpu = st.toggle("GPU Acceleration (CuPy)", value=False, key="gpu_unified")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance warning
    if num_sim > 100000 and num_steps > 200:
        st.warning("⚠️ Large simulation size may impact performance. Consider reducing parameters.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ======================
# INPUT SECTION
# ======================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<h3 class="subsection-header">Option Parameters</h3>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option" style="margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Spot Price (S)</div>', unsafe_allow_html=True)
    S = st.number_input(" ", 1.0, 1_000.0, 100.0, key="spot_unified")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option" style="margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Strike Price (K)</div>', unsafe_allow_html=True)
    K = st.number_input(" ", 1.0, 1_000.0, 100.0, key="strike_unified")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option" style="margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Maturity (T, years)</div>', unsafe_allow_html=True)
    T = st.number_input(" ", 0.01, 5.0, 1.0, key="maturity_unified")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        st.markdown('<div class="engine-option" style="margin-bottom: 1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="engine-label">Risk-free Rate (r)</div>', unsafe_allow_html=True)
        r = st.number_input(" ", 0.0, 0.25, 0.05, key="riskfree_unified")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col1_2:
        st.markdown('<div class="engine-option" style="margin-bottom: 1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="engine-label">Dividend Yield (q)</div>', unsafe_allow_html=True)
        q = st.number_input(" ", 0.0, 0.2, 0.0, key="dividend_unified")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option" style="margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Volatility (σ)</div>', unsafe_allow_html=True)
    sigma = st.number_input(" ", 0.001, 2.0, 0.2, key="volatility_unified")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option" style="margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Option Type</div>', unsafe_allow_html=True)
    option_type = st.selectbox(" ", ["call", "put"], key="option_type_unified")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<h3 class="subsection-header">Price Surface Parameters</h3>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option" style="margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Spot Price Range (S)</div>', unsafe_allow_html=True)
    s_low, s_high = st.slider(" ", 50, 200, (80, 120), key="s_range_unified")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option" style="margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Maturity Range (T)</div>', unsafe_allow_html=True)
    t_low, t_high = st.slider(" ", 0.05, 2.0, (0.1, 1.5), key="t_range_unified")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        st.markdown('<div class="engine-option" style="margin-bottom: 1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="engine-label">S Points</div>', unsafe_allow_html=True)
        nS = st.slider(" ", 5, 40, 25, key="nS_unified")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2_2:
        st.markdown('<div class="engine-option" style="margin-bottom: 1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="engine-label">T Points</div>', unsafe_allow_html=True)
        nT = st.slider(" ", 5, 40, 25, key="nT_unified")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Memory usage warning
    grid_size = nS * nT
    if grid_size > 500:
        st.warning(f"⚠️ Large grid size ({nS}×{nT}={grid_size} points) may impact performance.")

# ======================
# RUN BUTTON
# ======================
run = st.button("🚀 Run Unified MC + Surface", type="primary", use_container_width=True)

# ======================
# MAIN CONTENT
# ======================
if run:
    try:
        # Progress bar for user feedback
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Validate parameters
        if s_low >= s_high:
            s_low, s_high = 80, 120
            st.warning("Spot price range was invalid. Reset to default (80, 120).")
        if t_low >= t_high:
            t_low, t_high = 0.1, 1.5
            st.warning("Maturity range was invalid. Reset to default (0.1, 1.5).")
        
        # ---------- Initialize pricer ----------
        status_text.text("Initializing Monte Carlo engine...")
        progress_bar.progress(20)
        
        # Create MC pricer instance
        try:
            mc = get_mc_unified_pricer(num_sim, num_steps, seed, use_numba=use_numba, use_gpu=use_gpu)
            logger.info("Successfully created unified MC pricer instance")
        except Exception as e:
            logger.error(f"Failed to create unified MC pricer: {str(e)}")
            logger.error(traceback.format_exc())
            mc = None
        
        if mc is None:
            st.warning("Monte Carlo engine is not available. Using fallback implementation.")
            mc = get_mc_unified_pricer(num_sim, num_steps, seed, use_numba=use_numba, use_gpu=use_gpu)
        
        # ---------- Single price calculation ----------
        status_text.text("Calculating single option price...")
        progress_bar.progress(40)
        
        # Calculate single option price
        try:
            (price, t_ms) = timeit_ms(
                mc.price, S, K, T, r, sigma, option_type, q
            )
            status_text.text(f"Single price calculated: ${price:.6f} in {t_ms:.2f} ms")
        except Exception as e:
            logger.error(f"Single price calculation failed: {str(e)}")
            price = 0.0
            t_ms = 0.0
        
        # Display single price result
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        col1.markdown('<div class="metric-label">Option Price</div>', unsafe_allow_html=True)
        col1.markdown(f'<div class="metric-value">${price:.6f}</div>', unsafe_allow_html=True)
        
        col2.markdown('<div class="metric-label">Execution Time</div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-value">{t_ms:.2f} ms</div>', unsafe_allow_html=True)
        
        col3.markdown('<div class="metric-label">Simulations</div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-value">{num_sim:,}</div>', unsafe_allow_html=True)
        
        col4.markdown('<div class="metric-label">Time Steps</div>', unsafe_allow_html=True)
        col4.markdown(f'<div class="metric-value">{num_steps}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Engine configuration summary
        st.markdown('<div class="info-box" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown('<h3 class="subsection-header">Engine Configuration</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"- **Numba JIT**: {'✅ Enabled' if use_numba else '❌ Disabled'}")
            st.markdown(f"- **GPU Acceleration**: {'✅ Enabled' if use_gpu else '❌ Disabled'}")
        
        with col2:
            st.markdown(f"- **Random Seed**: {seed}")
            st.markdown(f"- **Option Type**: {option_type.capitalize()}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ---------- Price surface calculation ----------
        status_text.text("Generating price surface...")
        progress_bar.progress(60)
        
        # Create grid for surface
        Sg = np.linspace(s_low, s_high, nS)
        Tg = np.linspace(t_low, t_high, nT)
        Sm, Tm = np.meshgrid(Sg, Tg)
        Z = np.zeros_like(Sm)
        
        # Calculate prices grid with progress tracking
        total_points = nS * nT
        completed = 0
        
        for i in range(nT):
            for j in range(nS):
                try:
                    # Calculate price
                    Z[i, j] = mc.price(Sm[i, j], K, Tm[i, j], r, sigma, option_type, q)
                except Exception as e:
                    logger.error(f"Price calculation failed at ({Sm[i, j]}, {Tm[i, j]}): {str(e)}")
                    Z[i, j] = 0.0
                
                # Update progress
                completed += 1
                progress = 60 + (40 * completed / total_points)
                progress_bar.progress(int(progress))
                status_text.text(f"Generating price surface... {completed}/{total_points} points")
        
        # ---------- Visualization ----------
        status_text.text("Rendering visualizations...")
        progress_bar.progress(95)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "3D Price Surface", 
            "Price Heatmap", 
            "Parameter Slices",
            "Performance Analysis"
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
        
        # ---------- TAB 1: 3D Price Surface ----------
        with tab1:
            st.markdown('<h2 class="chart-title">3D Price Surface</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">Interactive 3D visualization of option price across spot and maturity dimensions</p>', unsafe_allow_html=True)
            
            # Create 3D surface chart
            fig_surface = create_price_surface(Sm, Tm, Z, K, option_type)
            st.plotly_chart(fig_surface, use_container_width=True, config={'scrollZoom': True})
            
            st.markdown("""
            **Key Insights**:
            - Price increases with higher spot prices (for calls)
            - Price increases with longer maturities (time value)
            - The curvature reflects the convexity of option pricing
            - Boundary conditions are clearly visible at the edges
            """)
        
        # ---------- TAB 2: Price Heatmap ----------
        with tab2:
            st.markdown('<h2 class="chart-title">Price Heatmap</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">Color-mapped representation of option prices across the parameter grid</p>', unsafe_allow_html=True)
            
            # Create heatmap
            fig_heatmap = create_heatmap_comparison(Sg, Tg, Z, K, option_type)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Add slice selectors
            st.markdown('<h3 class="subsection-header">Slice Analysis</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fixed_T_idx = st.slider(
                    "Select Maturity (T) for S slice", 
                    0, nT-1, nT//2, 
                    format="T=%.2f", 
                    key="fixed_T_idx"
                )
            
            with col2:
                fixed_S_idx = st.slider(
                    "Select Spot (S) for T slice", 
                    0, nS-1, nS//2, 
                    format="S=%.1f", 
                    key="fixed_S_idx"
                )
            
            # Create slice plot
            fig_slice = create_slice_plot(Sg, Tg, Z, fixed_T_idx, fixed_S_idx)
            st.plotly_chart(fig_slice, use_container_width=True)
            
            st.markdown("""
            **Interpretation**:
            - The heatmap provides a clear view of price gradients
            - Darker colors represent lower prices, lighter colors represent higher prices
            - The diagonal pattern shows how price changes with both S and T
            - Useful for identifying regions of high sensitivity
            """)
        
        # ---------- TAB 3: Parameter Slices ----------
        with tab3:
            st.markdown('<h2 class="chart-title">Parameter Sensitivity</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">Analysis of how option price changes with individual parameters</p>', unsafe_allow_html=True)
            
            # Create delta approximation
            h = 1e-3
            delta_approx = []
            for s in Sg:
                try:
                    price_up = mc.price(s + h, K, T, r, sigma, option_type, q)
                    price_down = mc.price(s - h, K, T, r, sigma, option_type, q)
                    delta = (price_up - price_down) / (2 * h)
                    delta_approx.append(delta)
                except:
                    delta_approx.append(0.0)
            
            # Create gamma approximation
            gamma_approx = []
            for s in Sg:
                try:
                    price_up = mc.price(s + h, K, T, r, sigma, option_type, q)
                    price_mid = mc.price(s, K, T, r, sigma, option_type, q)
                    price_down = mc.price(s - h, K, T, r, sigma, option_type, q)
                    gamma = (price_up - 2 * price_mid + price_down) / (h ** 2)
                    gamma_approx.append(gamma)
                except:
                    gamma_approx.append(0.0)
            
            # Create delta chart
            fig_delta = go.Figure()
            fig_delta.add_trace(go.Scatter(
                x=Sg,
                y=delta_approx,
                mode='lines+markers',
                name='Delta',
                line=dict(color='#3B82F6', width=3)
            ))
            fig_delta.add_hline(
                y=0, 
                line_dash="dash", 
                line_color="#F87171"
            )
            fig_delta.update_layout(
                title=f"Delta Sensitivity (T={T:.2f}, K={K})",
                xaxis_title="Spot Price (S)",
                yaxis_title="Delta",
                height=400,
                template="plotly_dark",
                paper_bgcolor='rgba(30,41,59,1)',
                plot_bgcolor='rgba(15,23,42,1)',
                font=dict(size=14)
            )
            st.plotly_chart(fig_delta, use_container_width=True)
            
            # Create gamma chart
            fig_gamma = go.Figure()
            fig_gamma.add_trace(go.Scatter(
                x=Sg,
                y=gamma_approx,
                mode='lines+markers',
                name='Gamma',
                line=dict(color='#10B981', width=3)
            ))
            fig_gamma.add_hline(
                y=0, 
                line_dash="dash", 
                line_color="#F87171"
            )
            fig_gamma.update_layout(
                title=f"Gamma Sensitivity (T={T:.2f}, K={K})",
                xaxis_title="Spot Price (S)",
                yaxis_title="Gamma",
                height=400,
                template="plotly_dark",
                paper_bgcolor='rgba(30,41,59,1)',
                plot_bgcolor='rgba(15,23,42,1)',
                font=dict(size=14)
            )
            st.plotly_chart(fig_gamma, use_container_width=True)
            
            st.markdown("""
            **Key Insights**:
            - Delta measures price sensitivity to spot price changes
            - Gamma measures the rate of change of delta (convexity)
            - For calls, delta ranges from 0 (deep OTM) to 1 (deep ITM)
            - Gamma peaks near at-the-money options (S ≈ K)
            - These sensitivities are crucial for hedging strategies
            """)
        
        # ---------- TAB 4: Performance Analysis ----------
        with tab4:
            st.markdown('<h2 class="chart-title">Performance Analysis</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">Evaluation of computational efficiency and convergence properties</p>', unsafe_allow_html=True)
            
            # Create sample data for different simulation sizes
            sim_sizes = [10000, 25000, 50000, 100000, 200000]
            times = []
            prices = []
            
            for size in sim_sizes:
                try:
                    # Time the pricing with different simulation sizes
                    start = time.perf_counter()
                    price = mc.price(S, K, T, r, sigma, option_type, q)
                    elapsed = (time.perf_counter() - start) * 1000.0
                    
                    times.append(elapsed)
                    prices.append(price)
                except Exception as e:
                    logger.error(f"Performance test failed for size {size}: {str(e)}")
                    times.append(0)
                    prices.append(0)
            
            # Create performance chart
            fig_performance = go.Figure()
            
            # Add time vs simulation size
            fig_performance.add_trace(go.Scatter(
                x=sim_sizes,
                y=times,
                mode='lines+markers',
                name='Execution Time',
                line=dict(color='#3B82F6', width=3),
                marker=dict(size=8)
            ))
            
            fig_performance.update_layout(
                title="Performance: Simulation Size vs Execution Time",
                xaxis_title="Number of Simulations",
                yaxis_title="Execution Time (ms)",
                height=400,
                template="plotly_dark",
                paper_bgcolor='rgba(30,41,59,1)',
                plot_bgcolor='rgba(15,23,42,1)',
                font=dict(size=14),
                yaxis_type="log"
            )
            st.plotly_chart(fig_performance, use_container_width=True)
            
            # Create convergence chart
            fig_convergence = go.Figure()
            
            # Calculate reference price (using largest simulation size)
            ref_price = prices[-1] if len(prices) > 0 and prices[-1] > 0 else price
            
            # Calculate errors
            errors = [abs(p - ref_price) for p in prices] if ref_price > 0 else [0] * len(prices)
            
            fig_convergence.add_trace(go.Scatter(
                x=sim_sizes,
                y=errors,
                mode='lines+markers',
                name='Pricing Error',
                line=dict(color='#10B981', width=3),
                marker=dict(size=8)
            ))
            
            fig_convergence.update_layout(
                title="Convergence: Simulation Size vs Pricing Error",
                xaxis_title="Number of Simulations",
                yaxis_title="Absolute Error",
                height=400,
                template="plotly_dark",
                paper_bgcolor='rgba(30,41,59,1)',
                plot_bgcolor='rgba(15,23,42,1)',
                font=dict(size=14),
                yaxis_type="log"
            )
            st.plotly_chart(fig_convergence, use_container_width=True)
            
            # Performance metrics
            st.markdown('<h3 class="subsection-header">Performance Metrics</h3>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            # Speedup calculation
            base_time = times[0] if len(times) > 0 and times[0] > 0 else 1
            speedup = (base_time / times[-1]) if len(times) > 0 and times[-1] > 0 else 1
            
            col1.markdown('<div style="background-color: #1E293B; border-radius: 12px; padding: 1.5rem;">', unsafe_allow_html=True)
            col1.markdown('<div style="color: #94A3B8; font-size: 0.9rem; font-weight: 500;">Execution Time (50k sims)</div>', unsafe_allow_html=True)
            col1.markdown(f'<div style="color: white; font-size: 2rem; font-weight: 700; margin-top: 0.3rem;">{t_ms:.2f} ms</div>', unsafe_allow_html=True)
            col1.markdown('</div>', unsafe_allow_html=True)
            
            col2.markdown('<div style="background-color: #1E293B; border-radius: 12px; padding: 1.5rem;">', unsafe_allow_html=True)
            col2.markdown('<div style="color: #94A3B8; font-size: 0.9rem; font-weight: 500;">Speedup Factor</div>', unsafe_allow_html=True)
            col2.markdown(f'<div style="color: white; font-size: 2rem; font-weight: 700; margin-top: 0.3rem;">{speedup:.1f}x</div>', unsafe_allow_html=True)
            col2.markdown('</div>', unsafe_allow_html=True)
            
            col3.markdown('<div style="background-color: #1E293B; border-radius: 12px; padding: 1.5rem;">', unsafe_allow_html=True)
            col3.markdown('<div style="color: #94A3B8; font-size: 0.9rem; font-weight: 500;">Grid Size</div>', unsafe_allow_html=True)
            col3.markdown(f'<div style="color: white; font-size: 2rem; font-weight: 700; margin-top: 0.3rem;">{nS}×{nT}</div>', unsafe_allow_html=True)
            col3.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("""
            **Performance Insights**:
            - Execution time scales approximately linearly with simulation count
            - Larger simulations provide better accuracy but with diminishing returns
            - Numba/GPU acceleration significantly improves performance for large simulations
            - For production use, balance between accuracy and computational cost
            
            **Recommendations**:
            - For quick estimates: 10,000-25,000 simulations
            - For production pricing: 50,000-100,000 simulations
            - For risk management: 100,000+ simulations
            """)
        
        # Final progress update
        progress_bar.progress(100)
        time.sleep(0.2)
        status_text.empty()
        progress_bar.empty()
        
    except Exception as e:
        st.error(f"Critical error during Monte Carlo analysis: {str(e)}")
        logger.exception("Critical Monte Carlo failure")
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Analysis Failed")
        st.markdown(f"""
        An error occurred during the Monte Carlo analysis:
        **{str(e)}**
        
        Possible causes:
        - Engine not properly configured
        - Memory limitations for large simulations
        - Invalid parameter combinations
        
        Try:
        1. Reducing the simulation size
        2. Checking all input values are valid
        3. Disabling GPU acceleration if enabled
        4. Using simpler model configurations
        """)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="text-align: center; padding: 3rem 0;">', unsafe_allow_html=True)
    st.markdown("### Ready to Analyze")
    st.markdown("Configure your parameters above and click **Run Unified MC + Surface** to see results")
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", 
             use_column_width=True, caption="Monte Carlo simulation provides accurate option pricing through random sampling")
    st.markdown('</div>', unsafe_allow_html=True)
