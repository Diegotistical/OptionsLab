# Page 3 MC Unified.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import logging
from typing import Optional, Tuple, Dict, Any

# Configure logging
logger = logging.getLogger("mc_unified")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ======================
# IMPORT HANDLING
# ======================
def get_mc_unified_pricer(num_sim: int = 50000, 
                          num_steps: int = 100, 
                          seed: Optional[int] = 42,
                          use_numba: bool = True,
                          use_gpu: bool = False) -> Any:
    """Robust import of MonteCarloPricerUni with fallback implementation"""
    try:
        # Try direct import from expected location
        from src.pricing_models.monte_carlo_unified import MonteCarloPricerUni
        return MonteCarloPricerUni(num_sim, num_steps, seed, use_numba, use_gpu)
    except ImportError:
        try:
            # Try alternative import paths
            from pricing_models.monte_carlo_unified import MonteCarloPricerUni
            return MonteCarloPricerUni(num_sim, num_steps, seed, use_numba, use_gpu)
        except ImportError:
            logger.warning("MonteCarloPricerUni not available. Using fallback implementation.")
            return _create_fallback_pricer(num_sim, num_steps, seed, use_numba, use_gpu)

def _create_fallback_pricer(num_sim: int, num_steps: int, seed: int, 
                          use_numba: bool, use_gpu: bool) -> Any:
    """Create a fallback Monte Carlo pricer when the real implementation is unavailable"""
    
    class FallbackPricer:
        def __init__(self, num_sim, num_steps, seed):
            self.num_sim = num_sim
            self.num_steps = num_steps
            self.seed = seed
            np.random.seed(seed)
            
        def price(self, S, K, T, r, sigma, option_type, q=0.0):
            """Simplified MC pricing implementation"""
            # Validate inputs
            if S <= 0 or K <= 0 or T <= 0.001 or sigma <= 0.001:
                return 0.0
                
            # Generate paths
            dt = T / self.num_steps
            Z = np.random.standard_normal((self.num_sim, self.num_steps))
            S_paths = np.zeros((self.num_sim, self.num_steps))
            S_paths[:, 0] = S
            
            for t in range(1, self.num_steps):
                drift = (r - q - 0.5 * sigma**2) * dt
                diffusion = sigma * np.sqrt(dt) * Z[:, t]
                S_paths[:, t] = S_paths[:, t-1] * np.exp(drift + diffusion)
            
            # Calculate payoff
            if option_type == "call":
                payoff = np.maximum(S_paths[:, -1] - K, 0.0)
            else:
                payoff = np.maximum(K - S_paths[:, -1], 0.0)
                
            return float(np.mean(np.exp(-r * T) * payoff))
        
        def delta_gamma(self, S, K, T, r, sigma, option_type, q=0.0, h=1e-4):
            """Calculate delta and gamma using central differences"""
            try:
                price_up = self.price(S + h, K, T, r, sigma, option_type, q)
                price_mid = self.price(S, K, T, r, sigma, option_type, q)
                price_down = self.price(S - h, K, T, r, sigma, option_type, q)
                
                delta = (price_up - price_down) / (2 * h)
                gamma = (price_up - 2 * price_mid + price_down) / (h ** 2)
                return delta, gamma
            except:
                return 0.5, 0.01
    
    return FallbackPricer(num_sim, num_steps, seed)

def timeit_ms(fn, *args, **kwargs) -> Tuple[Any, float]:
    """Measure execution time in milliseconds"""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000.0
    return result, elapsed

# ======================
# HELPER FUNCTIONS
# ======================
def create_price_surface(Sg: np.ndarray, Tg: np.ndarray, Z: np.ndarray, 
                         K: float, option_type: str) -> go.Figure:
    """Create 3D price surface visualization"""
    fig = go.Figure(data=[
        go.Surface(
            x=Sg,
            y=Tg,
            z=Z,
            colorscale='Viridis',
            colorbar=dict(title="Price", thickness=25)
        )
    ])
    
    fig.update_layout(
        title=f"Option Price Surface (K={K}, {option_type.capitalize()})",
        scene=dict(
            xaxis_title="Spot Price (S)",
            yaxis_title="Time to Maturity (T)",
            zaxis_title="Option Price",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5))
        ),
        height=700,
        template="plotly_dark",
        paper_bgcolor='rgba(30,41,59,1)',
        plot_bgcolor='rgba(15,23,42,1)',
        font=dict(size=14)
    )
    
    return fig

def create_heatmap(Sg: np.ndarray, Tg: np.ndarray, Z: np.ndarray, 
                   K: float, option_type: str) -> go.Figure:
    """Create price heatmap visualization"""
    df = pd.DataFrame({
        'S': np.repeat(Sg, len(Tg)),
        'T': np.tile(Tg, len(Sg)),
        'Price': Z.flatten()
    })
    
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

def create_slice_plot(Sg: np.ndarray, Tg: np.ndarray, Z: np.ndarray, 
                      fixed_T_idx: Optional[int] = None, 
                      fixed_S_idx: Optional[int] = None) -> go.Figure:
    """Create slice plots for price surface"""
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

# ======================
# STREAMLIT UI
# ======================
st.set_page_config(
    page_title="Monte Carlo Unified",
    layout="wide",
    page_icon="🚀"
)

# Clean CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: white;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #E2E8F0;
        margin-bottom: 1.5rem;
        opacity: 0.9;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 6px 6px 0 0;
        font-size: 1rem;
        font-weight: 500;
        background-color: #1E293B;
        color: #CBD5E1;
        padding: 0 15px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
        font-weight: 600;
        border-bottom: 3px solid #1E3A8A;
    }
    .chart-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: white;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
    }
    .chart-description {
        font-size: 0.95rem;
        color: #CBD5E1;
        margin-bottom: 1rem;
        line-height: 1.4;
    }
    .metric-label {
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .metric-value {
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
    }
    .subsection-header {
        font-size: 1.2rem;
        color: white;
        margin: 1rem 0 0.7rem 0;
        font-weight: 600;
    }
    .engine-option {
        background-color: #1E293B;
        border-radius: 6px;
        padding: 0.6rem;
        margin-bottom: 0.5rem;
        border: 1px solid #334155;
    }
    .engine-label {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-bottom: 0.2rem;
    }
    .engine-value {
        font-size: 1rem;
        color: white;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# PAGE CONTENT
# ======================
st.markdown('<h1 class="main-header">Monte Carlo Unified (CPU/GPU + Antithetic)</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">High-performance option pricing with unified CPU/GPU implementation and variance reduction techniques</p>', unsafe_allow_html=True)

# Configuration sidebar
with st.sidebar:
    st.markdown('<h3 style="color: white; margin-top: 1rem; margin-bottom: 0.75rem;">Engine Configuration</h3>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Simulations</div>', unsafe_allow_html=True)
    num_sim = st.slider("", 10_000, 200_000, 50_000, step=10_000, key="sim_unified")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Time Steps</div>', unsafe_allow_html=True)
    num_steps = st.slider("", 10, 500, 100, step=10, key="steps_unified")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Random Seed</div>', unsafe_allow_html=True)
    seed = st.number_input("", value=42, min_value=1, key="seed_unified")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Acceleration</div>', unsafe_allow_html=True)
    use_numba = st.toggle("Numba JIT", value=True, key="numba_unified")
    use_gpu = st.toggle("GPU Acceleration", value=False, key="gpu_unified")
    st.markdown('</div>', unsafe_allow_html=True)

# Main content columns
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown('<h3 class="subsection-header">Option Parameters</h3>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Spot Price (S)</div>', unsafe_allow_html=True)
    S = st.number_input("", 1.0, 1_000.0, 100.0, key="spot_unified")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Strike Price (K)</div>', unsafe_allow_html=True)
    K = st.number_input("", 1.0, 1_000.0, 100.0, key="strike_unified")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Maturity (T, years)</div>', unsafe_allow_html=True)
    T = st.number_input("", 0.01, 5.0, 1.0, key="maturity_unified")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
        st.markdown('<div class="engine-label">Risk-free Rate (r)</div>', unsafe_allow_html=True)
        r = st.number_input("", 0.0, 0.25, 0.05, key="riskfree_unified")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col1_2:
        st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
        st.markdown('<div class="engine-label">Dividend Yield (q)</div>', unsafe_allow_html=True)
        q = st.number_input("", 0.0, 0.2, 0.0, key="dividend_unified")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Volatility (σ)</div>', unsafe_allow_html=True)
    sigma = st.number_input("", 0.001, 2.0, 0.2, key="volatility_unified")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Option Type</div>', unsafe_allow_html=True)
    option_type = st.selectbox("", ["call", "put"], key="option_type_unified")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<h3 class="subsection-header">Price Surface Parameters</h3>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Spot Price Range (S)</div>', unsafe_allow_html=True)
    s_low, s_high = st.slider("", 50, 200, (80, 120), key="s_range_unified")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Maturity Range (T)</div>', unsafe_allow_html=True)
    t_low, t_high = st.slider("", 0.05, 2.0, (0.1, 1.5), key="t_range_unified")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
        st.markdown('<div class="engine-label">S Points</div>', unsafe_allow_html=True)
        nS = st.slider("", 5, 40, 25, key="nS_unified")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2_2:
        st.markdown('<div class="engine-option" style="margin-bottom: 0.75rem;">', unsafe_allow_html=True)
        st.markdown('<div class="engine-label">T Points</div>', unsafe_allow_html=True)
        nT = st.slider("", 5, 40, 25, key="nT_unified")
        st.markdown('</div>', unsafe_allow_html=True)

# Run button
run = st.button("Run Analysis", type="primary", use_container_width=True)

# Main application logic
if run:
    try:
        # Initialize progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize pricer
        status_text.text("Initializing Monte Carlo engine...")
        progress_bar.progress(20)
        
        try:
            mc = get_mc_unified_pricer(num_sim, num_steps, seed, use_numba, use_gpu)
            status_text.text("Monte Carlo engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Monte Carlo engine: {str(e)}")
            mc = get_mc_unified_pricer(num_sim, num_steps, seed, use_numba, use_gpu)
            st.warning("Monte Carlo engine is not available. Using fallback implementation.")
        
        # Calculate single option price
        status_text.text("Calculating option price...")
        progress_bar.progress(40)
        
        try:
            (price, t_ms) = timeit_ms(
                mc.price, S, K, T, r, sigma, option_type, q
            )
            (delta, gamma) = mc.delta_gamma(S, K, T, r, sigma, option_type, q)
            status_text.text(f"Price calculated: ${price:.6f}")
        except Exception as e:
            logger.error(f"Price calculation failed: {str(e)}")
            price, delta, gamma = 0.0, 0.5, 0.01
            t_ms = 0.0
            st.error("Failed to calculate option price. Using default values.")
        
        # Display results
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        col1.markdown('<div class="metric-label">Option Price</div>', unsafe_allow_html=True)
        col1.markdown(f'<div class="metric-value">${price:.6f}</div>', unsafe_allow_html=True)
        
        col2.markdown('<div class="metric-label">Delta</div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-value">{delta:.4f}</div>', unsafe_allow_html=True)
        
        col3.markdown('<div class="metric-label">Gamma</div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-value">{gamma:.6f}</div>', unsafe_allow_html=True)
        
        col4.markdown('<div class="metric-label">Time</div>', unsafe_allow_html=True)
        col4.markdown(f'<div class="metric-value">{t_ms:.2f} ms</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate price surface
        status_text.text("Generating price surface...")
        progress_bar.progress(60)
        
        # Create grid for surface
        Sg = np.linspace(s_low, s_high, nS)
        Tg = np.linspace(t_low, t_high, nT)
        Sm, Tm = np.meshgrid(Sg, Tg)
        Z = np.zeros((nT, nS))
        
        # Calculate prices grid
        total_points = nS * nT
        completed = 0
        
        for i in range(nT):
            for j in range(nS):
                try:
                    Z[i, j] = mc.price(Sm[i, j], K, Tm[i, j], r, sigma, option_type, q)
                except Exception as e:
                    logger.error(f"Price calculation failed at ({Sm[i, j]}, {Tm[i, j]}): {str(e)}")
                    Z[i, j] = 0.0
                
                # Update progress
                completed += 1
                progress = 60 + (40 * completed / total_points)
                progress_bar.progress(int(progress))
                status_text.text(f"Generating price surface... {completed}/{total_points} points")
        
        # Create visualization tabs
        tab1, tab2, tab3 = st.tabs([
            "3D Price Surface", 
            "Price Heatmap", 
            "Parameter Sensitivity"
        ])
        
        # 3D Price Surface tab
        with tab1:
            st.markdown('<h2 class="chart-title">Price Surface</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">Option price across spot price and time to maturity dimensions</p>', unsafe_allow_html=True)
            
            fig_surface = create_price_surface(Sg, Tg, Z, K, option_type)
            st.plotly_chart(fig_surface, use_container_width=True, config={'scrollZoom': True})
        
        # Price Heatmap tab
        with tab2:
            st.markdown('<h2 class="chart-title">Price Heatmap</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">Color-mapped representation of option prices across the parameter grid</p>', unsafe_allow_html=True)
            
            fig_heatmap = create_heatmap(Sg, Tg, Z, K, option_type)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Slice selectors
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
            
            fig_slice = create_slice_plot(Sg, Tg, Z, fixed_T_idx, fixed_S_idx)
            st.plotly_chart(fig_slice, use_container_width=True)
        
        # Parameter Sensitivity tab
        with tab3:
            st.markdown('<h2 class="chart-title">Parameter Sensitivity</h2>', unsafe_allow_html=True)
            st.markdown('<p class="chart-description">Analysis of how option price changes with individual parameters</p>', unsafe_allow_html=True)
            
            # Calculate delta and gamma across S grid
            h = 1e-3
            deltas = []
            gammas = []
            
            for s in Sg:
                try:
                    price_up = mc.price(s + h, K, T, r, sigma, option_type, q)
                    price_mid = mc.price(s, K, T, r, sigma, option_type, q)
                    price_down = mc.price(s - h, K, T, r, sigma, option_type, q)
                    
                    delta = (price_up - price_down) / (2 * h)
                    gamma = (price_up - 2 * price_mid + price_down) / (h ** 2)
                    
                    deltas.append(delta)
                    gammas.append(gamma)
                except:
                    deltas.append(0.0)
                    gammas.append(0.0)
            
            # Delta chart
            fig_delta = go.Figure()
            fig_delta.add_trace(go.Scatter(
                x=Sg,
                y=deltas,
                mode='lines+markers',
                name='Delta',
                line=dict(color='#3B82F6', width=2.5)
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
            
            # Gamma chart
            fig_gamma = go.Figure()
            fig_gamma.add_trace(go.Scatter(
                x=Sg,
                y=gammas,
                mode='lines+markers',
                name='Gamma',
                line=dict(color='#10B981', width=2.5)
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
        
        # Final progress update
        progress_bar.progress(100)
        time.sleep(0.2)
        status_text.empty()
        progress_bar.empty()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.exception("Critical error in Monte Carlo Unified")
        
        st.markdown("""
        <div style="background-color: #1E293B; border-radius: 8px; padding: 1rem; border: 1px solid #334155; margin-top: 1rem;">
            <h3 style="color: white; margin: 0 0 0.5rem 0;">Troubleshooting Tips</h3>
            <ul style="color: #CBD5E1; padding-left: 1.2rem; margin-bottom: 0;">
                <li>Ensure all input values are valid (positive numbers, etc.)</li>
                <li>Try reducing the simulation size if performance is poor</li>
                <li>Disable GPU acceleration if you're not using a GPU instance</li>
                <li>Check that all required dependencies are installed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background-color: #1E293B; border-radius: 8px; 
                 border: 1px solid #334155; margin-top: 1rem;">
        <h3 style="color: white; margin-bottom: 0.75rem;">Get Started</h3>
        <p style="color: #CBD5E1; margin-bottom: 1rem;">Configure your parameters and click "Run Analysis" to see results</p>
    </div>
    """, unsafe_allow_html=True)