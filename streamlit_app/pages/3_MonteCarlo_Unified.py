# Page 3 MC Unified.py
import logging
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ======================
# CONFIGURATION
# ======================
logger = logging.getLogger("mc_unified")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ======================
# IMPORT HANDLING
# ======================
def verify_pricer_methods(pricer) -> bool:
    """Verify that the pricer has all required methods"""
    required_methods = ['price', 'delta_gamma', 'price_batch', 'delta_gamma_batch']
    missing = [m for m in required_methods if not hasattr(pricer, m)]
    if missing:
        logger.error(f"Pricer missing methods: {missing}")
        return False
    logger.info(f"Pricer has all required methods: {required_methods}")
    return True


def get_mc_unified_pricer(
    num_sim: int = 50000,
    num_steps: int = 100,
    seed: Optional[int] = 42,
    use_numba: bool = True,
    use_gpu: bool = False,
) -> Any:
    """Robust import of MonteCarloPricerUni with fallback implementation"""
    try:
        from src.pricing_models.monte_carlo_unified import MonteCarloPricerUni

        pricer = MonteCarloPricerUni(num_sim, num_steps, seed, use_numba, use_gpu)
        
        if verify_pricer_methods(pricer):
            logger.info("Successfully loaded MonteCarloPricerUni with all methods")
            return pricer
        else:
            logger.warning("MonteCarloPricerUni missing required methods. Using fallback.")
            raise ImportError("Incomplete pricer implementation")
            
    except ImportError as e:
        logger.warning(f"Primary import failed: {e}")
        try:
            from pricing_models.monte_carlo_unified import MonteCarloPricerUni

            pricer = MonteCarloPricerUni(num_sim, num_steps, seed, use_numba, use_gpu)
            
            if verify_pricer_methods(pricer):
                logger.info("Successfully loaded MonteCarloPricerUni (alt path) with all methods")
                return pricer
            else:
                raise ImportError("Incomplete pricer implementation")
                
        except ImportError:
            logger.warning(
                "MonteCarloPricerUni not available. Using fallback implementation."
            )
            return _create_fallback_pricer(num_sim, num_steps, seed, use_numba, use_gpu)


def _create_fallback_pricer(
    num_sim: int, num_steps: int, seed: int, use_numba: bool, use_gpu: bool
) -> Any:
    """Create a fallback Monte Carlo pricer when the real implementation is unavailable"""

    class FallbackPricer:
        def __init__(self, num_sim, num_steps, seed):
            self.num_sim = num_sim
            self.num_steps = num_steps
            self.seed = seed
            np.random.seed(seed)
            logger.info(f"Initialized FallbackPricer with {num_sim} simulations")

        def price(self, S, K, T, r, sigma, option_type, q=0.0, seed=None, use_path_simulation=False):
            """Direct terminal simulation (correct for European options)"""
            if S <= 0 or K <= 0 or T <= 0.001 or sigma <= 0.001:
                return 0.0

            if seed is not None:
                np.random.seed(seed)

            # ALWAYS use direct terminal for European options
            Z = np.random.standard_normal(self.num_sim)
            Z_ant = -Z
            
            drift = (r - q - 0.5 * sigma**2) * T
            diffusion = sigma * np.sqrt(T)
            
            S_T = np.concatenate([
                S * np.exp(drift + diffusion * Z),
                S * np.exp(drift + diffusion * Z_ant)
            ])

            if option_type == "call":
                payoff = np.maximum(S_T - K, 0.0)
            else:
                payoff = np.maximum(K - S_T, 0.0)

            return float(np.mean(np.exp(-r * T) * payoff))

        def delta_gamma(self, S, K, T, r, sigma, option_type, q=0.0, h=None, seed=None):
            """Calculate delta and gamma using central differences with CRN"""
            if h is None:
                h = max(1e-4 * S, 1e-5)

            if seed is not None:
                base_seed = seed
            else:
                base_seed = self.seed + 1

            # Use same random numbers for all three points (CRN)
            np.random.seed(base_seed)
            Z = np.random.standard_normal(self.num_sim)
            Z_ant = -Z

            drift = (r - q - 0.5 * sigma**2) * T
            diffusion = sigma * np.sqrt(T)

            # Calculate for S-h, S, S+h using same Z
            S_arr = np.array([S - h, S, S + h])
            prices = []
            
            for S_i in S_arr:
                S_T = np.concatenate([
                    S_i * np.exp(drift + diffusion * Z),
                    S_i * np.exp(drift + diffusion * Z_ant)
                ])
                
                if option_type == "call":
                    payoff = np.maximum(S_T - K, 0.0)
                else:
                    payoff = np.maximum(K - S_T, 0.0)
                
                price = float(np.mean(np.exp(-r * T) * payoff))
                prices.append(price)

            delta = (prices[2] - prices[0]) / (2 * h)
            gamma = (prices[2] - 2 * prices[1] + prices[0]) / (h**2)

            # Validation
            if option_type == "call":
                delta = max(0.0, min(1.0, delta))
            else:
                delta = max(-1.0, min(0.0, delta))
            gamma = max(0.0, gamma)

            return float(delta), float(gamma)

        def price_batch(
            self, S_vals, K_vals, T_vals, r_vals, sigma_vals, option_type, q_vals=0.0, use_path_simulation=False
        ):
            """Vectorized pricing for multiple points at once"""
            if isinstance(q_vals, (int, float)):
                q_vals = np.full_like(S_vals, q_vals)

            prices = np.zeros(len(S_vals))

            for i in range(len(S_vals)):
                try:
                    prices[i] = self.price(
                        S_vals[i], K_vals[i], T_vals[i], r_vals[i], sigma_vals[i],
                        option_type, q_vals[i], use_path_simulation=use_path_simulation
                    )
                except Exception as e:
                    logger.warning(f"Failed to price point {i}: {e}")
                    prices[i] = 0.0

            return prices

        def delta_gamma_batch(
            self, S_vals, K_vals, T_vals, r_vals, sigma_vals, option_type, q_vals=0.0, h=None
        ):
            """Vectorized Greek calculation for multiple points at once"""
            if isinstance(q_vals, (int, float)):
                q_vals = np.full_like(S_vals, q_vals)

            deltas = np.zeros(len(S_vals))
            gammas = np.zeros(len(S_vals))

            for i in range(len(S_vals)):
                try:
                    deltas[i], gammas[i] = self.delta_gamma(
                        S_vals[i], K_vals[i], T_vals[i], r_vals[i], sigma_vals[i],
                        option_type, q_vals[i], h
                    )
                except Exception as e:
                    logger.warning(f"Failed to calculate Greeks for point {i}: {e}")
                    deltas[i] = 0.0
                    gammas[i] = 0.0

            return deltas, gammas

    return FallbackPricer(num_sim, num_steps, seed)


def timeit_ms(fn, *args, **kwargs) -> tuple:
    """Measure execution time in milliseconds"""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000.0
    return result, elapsed


# ======================
# OPTIMIZED SURFACE GENERATION
# ======================
def generate_surface_data(mc, Sg, Tg, K, r, sigma, option_type, q, batch_size=25, use_path_simulation=False):
    """Generate surface data in batches for better performance"""
    nS = len(Sg)
    nT = len(Tg)
    Sm, Tm = np.meshgrid(Sg, Tg)

    S_flat = Sm.flatten()
    T_flat = Tm.flatten()
    K_flat = np.full_like(S_flat, K)
    r_flat = np.full_like(S_flat, r)
    sigma_flat = np.full_like(S_flat, sigma)
    q_flat = np.full_like(S_flat, q)

    Z = np.zeros_like(S_flat)
    deltas_grid = np.zeros_like(S_flat)
    gammas_grid = np.zeros_like(S_flat)

    if not hasattr(mc, 'price_batch') or not hasattr(mc, 'delta_gamma_batch'):
        logger.error("Pricer missing batch methods!")
        raise AttributeError("Pricer missing required batch methods")

    total_points = len(S_flat)
    num_batches = (total_points + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_points)

        try:
            Z[start_idx:end_idx] = mc.price_batch(
                S_flat[start_idx:end_idx],
                K_flat[start_idx:end_idx],
                T_flat[start_idx:end_idx],
                r_flat[start_idx:end_idx],
                sigma_flat[start_idx:end_idx],
                option_type,
                q_flat[start_idx:end_idx],
                use_path_simulation=use_path_simulation
            )

            deltas, gammas = mc.delta_gamma_batch(
                S_flat[start_idx:end_idx],
                K_flat[start_idx:end_idx],
                T_flat[start_idx:end_idx],
                r_flat[start_idx:end_idx],
                sigma_flat[start_idx:end_idx],
                option_type,
                q_flat[start_idx:end_idx],
            )
            deltas_grid[start_idx:end_idx] = deltas
            gammas_grid[start_idx:end_idx] = gammas
        except Exception as e:
            logger.error(f"Failed to process batch {batch_idx}: {e}")
            Z[start_idx:end_idx] = 0.0
            deltas_grid[start_idx:end_idx] = 0.0
            gammas_grid[start_idx:end_idx] = 0.0

    Z = Z.reshape((nT, nS))
    deltas_grid = deltas_grid.reshape((nT, nS))
    gammas_grid = gammas_grid.reshape((nT, nS))

    return Sm, Tm, Z, deltas_grid, gammas_grid


# ======================
# HELPER FUNCTIONS (Same as original)
# ======================
def create_price_surface(
    Sg: np.ndarray, Tg: np.ndarray, Z: np.ndarray, K: float, option_type: str
) -> go.Figure:
    """Create 3D price surface visualization"""
    fig = go.Figure(
        data=[
            go.Surface(
                x=Sg,
                y=Tg,
                z=Z,
                colorscale="Viridis",
                colorbar=dict(title="Price", thickness=25),
            )
        ]
    )

    fig.update_layout(
        title=f"Option Price Surface (K={K}, {option_type.capitalize()})",
        scene=dict(
            xaxis_title="Spot Price (S)",
            yaxis_title="Time to Maturity (T)",
            zaxis_title="Option Price",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5)),
        ),
        height=700,
        template="plotly_dark",
        paper_bgcolor="rgba(30,41,59,1)",
        plot_bgcolor="rgba(15,23,42,1)",
        font=dict(size=14),
    )

    return fig


def create_greeks_heatmap(
    Sg: np.ndarray,
    Tg: np.ndarray,
    deltas: np.ndarray,
    gammas: np.ndarray,
    K: float,
    option_type: str,
) -> tuple:
    """Create heatmaps for Greeks"""
    df_delta = pd.DataFrame(
        {
            "S": np.repeat(Sg, len(Tg)),
            "T": np.tile(Tg, len(Sg)),
            "Delta": deltas.flatten(),
        }
    )

    fig_delta = px.density_heatmap(
        df_delta,
        x="S",
        y="T",
        z="Delta",
        color_continuous_scale="RdBu",
        labels={"Delta": "Delta"},
        title=f"Delta Heatmap (K={K}, {option_type.capitalize()})",
    )

    fig_delta.update_layout(
        height=400,
        template="plotly_dark",
        paper_bgcolor="rgba(30,41,59,1)",
        plot_bgcolor="rgba(15,23,42,1)",
        font=dict(size=14),
        coloraxis_colorbar=dict(title="Delta", thickness=25),
    )

    df_gamma = pd.DataFrame(
        {
            "S": np.repeat(Sg, len(Tg)),
            "T": np.tile(Tg, len(Sg)),
            "Gamma": gammas.flatten(),
        }
    )

    fig_gamma = px.density_heatmap(
        df_gamma,
        x="S",
        y="T",
        z="Gamma",
        color_continuous_scale="Viridis",
        labels={"Gamma": "Gamma"},
        title=f"Gamma Heatmap (K={K}, {option_type.capitalize()})",
    )

    fig_gamma.update_layout(
        height=400,
        template="plotly_dark",
        paper_bgcolor="rgba(30,41,59,1)",
        plot_bgcolor="rgba(15,23,42,1)",
        font=dict(size=14),
        coloraxis_colorbar=dict(title="Gamma", thickness=25),
    )

    return fig_delta, fig_gamma


# ======================
# STREAMLIT UI (Same styling as original)
# ======================
st.set_page_config(page_title="Monte Carlo Unified", layout="wide", page_icon="🚀")

st.markdown(
    """
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
    .stTabs [data-baseweb="tablist"] {
        display: flex !important;
        flex-wrap: nowrap !important;
        gap: 0.5rem !important;
        margin-bottom: 1.5rem !important;
        width: 100% !important;
        justify-content: stretch !important;
    }
    .stTabs [role="tab"] {
        flex: 1 !important;
        min-width: 0 !important;
        height: 60px !important;
        border-radius: 10px 10px 0 0 !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        background-color: #1E293B !important;
        color: white !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
        font-weight: 700 !important;
        border-bottom: 5px solid #1E3A8A !important;
        transform: translateY(-2px) !important;
    }
    .stTabs [role="tab"]:hover:not([aria-selected="true"]) {
        background-color: #334155 !important;
        color: white !important;
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
    .subsection-header {
        font-size: 1.4rem;
        color: white;
        margin: 1.2rem 0 0.8rem 0;
        font-weight: 600;
    }
    .engine-option {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
        border: 1px solid #334155;
    }
    .engine-label {
        font-size: 0.9rem;
        color: white;
        margin-bottom: 0.3rem;
    }
    .stButton > button {
        background-color: #3B82F6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        background-color: #2563EB;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .stNumberInput > div > label,
    .stSlider > div > label,
    .stSelectbox > div > label {
        color: white !important;
        font-weight: 500 !important;
    }
    .info-box {
        background-color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #e2e8f0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ======================
# PAGE CONTENT
# ======================
st.markdown(
    '<h1 class="main-header">Monte Carlo Unified (Direct Terminal Simulation)</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">High-performance option pricing using exact terminal distribution (1 timestep, zero bias)</p>',
    unsafe_allow_html=True,
)

# Info box explaining the method
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown("""
<strong>📊 Technical Note: Direct Terminal Simulation</strong><br>
For European options, this implementation uses the exact terminal distribution:<br>
<code>S_T = S_0 * exp((r - q - 0.5*σ²)*T + σ*√T*Z)</code><br><br>
This means:<br>
• ✅ <strong>1 timestep</strong> (not 100) - matches the exact mathematical solution<br>
• ✅ <strong>Zero discretization bias</strong> - no Euler-Maruyama approximation error<br>
• ✅ <strong>~100x faster</strong> - eliminates unnecessary path discretization<br>
• ✅ <strong>Pure sampling variance</strong> - interpretable convergence properties
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Engine Configuration
st.markdown(
    '<h3 class="subsection-header">Engine Configuration</h3>', unsafe_allow_html=True
)
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Simulations (paths)</div>', unsafe_allow_html=True)
    num_sim = st.slider("", 10_000, 200_000, 30_000, step=10_000, key="sim_unified")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Random Seed</div>', unsafe_allow_html=True)
    seed = st.number_input("", value=42, min_value=1, key="seed_unified")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Acceleration</div>', unsafe_allow_html=True)
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        use_numba = st.toggle("Numba JIT", value=True, key="numba_unified")
    with col2_2:
        use_gpu = st.toggle("GPU Acceleration", value=False, key="gpu_unified")
    st.markdown("</div>", unsafe_allow_html=True)

# Option Parameters
st.markdown(
    '<h3 class="subsection-header">Option Parameters</h3>', unsafe_allow_html=True
)
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Spot Price (S)</div>', unsafe_allow_html=True)
    S = st.number_input("", 1.0, 1_000.0, 100.0, key="spot_unified")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Strike Price (K)</div>', unsafe_allow_html=True)
    K = st.number_input("", 1.0, 1_000.0, 100.0, key="strike_unified")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Maturity (T, years)</div>', unsafe_allow_html=True)
    T = st.number_input("", 0.01, 5.0, 1.0, key="maturity_unified")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Risk-free Rate (r)</div>', unsafe_allow_html=True)
    r = st.number_input("", 0.0, 0.25, 0.05, key="riskfree_unified")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Dividend Yield (q)</div>', unsafe_allow_html=True)
    q = st.number_input("", 0.0, 0.2, 0.0, key="dividend_unified")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Volatility (σ)</div>', unsafe_allow_html=True)
    sigma = st.number_input("", 0.001, 2.0, 0.2, key="volatility_unified")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Option Type</div>', unsafe_allow_html=True)
    option_type = st.selectbox("", ["call", "put"], key="option_type_unified")
    st.markdown("</div>", unsafe_allow_html=True)

# Price Surface Parameters
st.markdown(
    '<h3 class="subsection-header">Price Surface Parameters</h3>',
    unsafe_allow_html=True,
)
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Spot Price Range (S)</div>', unsafe_allow_html=True)
    s_low, s_high = st.slider("", 50, 200, (80, 120), key="s_range_unified")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Maturity Range (T)</div>', unsafe_allow_html=True)
    t_low, t_high = st.slider("", 0.05, 2.0, (0.1, 1.5), key="t_range_unified")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Grid Resolution</div>', unsafe_allow_html=True)
    grid_size = st.select_slider(
        "",
        options=["Low (5×5)", "Medium (15×15)", "High (25×25)"],
        value="Medium (15×15)",
        key="grid_size_unified",
    )
    nS = nT = {"Low (5×5)": 5, "Medium (15×15)": 15, "High (25×25)": 25}[grid_size]
    st.markdown("</div>", unsafe_allow_html=True)

# Run button
st.markdown(
    '<div style="display: flex; justify-content: center; margin: 1.5rem 0;">',
    unsafe_allow_html=True,
)
run = st.button("Run Analysis", type="primary", use_container_width=False)
st.markdown("</div>", unsafe_allow_html=True)

# Main application logic
if run:
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Initializing Monte Carlo engine...")
        progress_bar.progress(10)

        mc = get_mc_unified_pricer(num_sim, 1, seed, use_numba, use_gpu)  # num_steps=1 for direct terminal
        
        if not verify_pricer_methods(mc):
            st.error("❌ Pricer initialization failed - missing required methods")
            st.stop()
        
        status_text.text("✅ Monte Carlo engine initialized successfully")
        progress_bar.progress(20)

        # Calculate single option price (ALWAYS use direct terminal)
        status_text.text("Calculating option price (direct terminal)...")
        
        (price, t_ms) = timeit_ms(mc.price, S, K, T, r, sigma, option_type, q, use_path_simulation=False)
        (delta, gamma) = timeit_ms(
            mc.delta_gamma, S, K, T, r, sigma, option_type, q
        )[0]
        
        status_text.text(f"✅ Price calculated: ${price:.6f}")
        progress_bar.progress(30)

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

        st.markdown("</div>", unsafe_allow_html=True)

        # Generate price surface
        status_text.text(f"Generating price surface ({nS}×{nT} points)...")
        progress_bar.progress(40)

        Sg = np.linspace(s_low, s_high, nS)
        Tg = np.linspace(t_low, t_high, nT)

        # Use direct terminal simulation for surface
        Sm, Tm, Z, deltas_grid, gammas_grid = generate_surface_data(
            mc, Sg, Tg, K, r, sigma, option_type, q, batch_size=25, use_path_simulation=False
        )

        status_text.text(f"✅ Surface generated successfully")
        progress_bar.progress(60)

        # Create visualization tabs
        tab1, tab2, tab3 = st.tabs(
            ["Price Surface", "Greek Analysis", "Parameter Sensitivity"]
        )

        with tab1:
            st.markdown('<h2 class="chart-title">Price Surface</h2>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="chart-description">Option price across spot price and time to maturity ({nS}×{nT} points, direct terminal simulation)</p>',
                unsafe_allow_html=True,
            )

            fig_surface = create_price_surface(Sg, Tg, Z, K, option_type)
            st.plotly_chart(fig_surface, use_container_width=True, config={"scrollZoom": True})

        with tab2:
            st.markdown('<h2 class="chart-title">Greek Analysis</h2>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="chart-description">Delta and gamma visualization ({nS}×{nT} points)</p>',
                unsafe_allow_html=True,
            )

            fig_delta, fig_gamma = create_greeks_heatmap(Sg, Tg, deltas_grid, gammas_grid, K, option_type)

            st.markdown('<h3 class="subsection-header">Delta Heatmap</h3>', unsafe_allow_html=True)
            st.plotly_chart(fig_delta, use_container_width=True)

            st.markdown('<h3 class="subsection-header">Gamma Heatmap</h3>', unsafe_allow_html=True)
            st.plotly_chart(fig_gamma, use_container_width=True)

        with tab3:
            st.markdown('<h2 class="chart-title">Parameter Sensitivity</h2>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="chart-description">Analysis of how option price and Greeks change with parameters ({nS}×{nT} points)</p>',
                unsafe_allow_html=True,
            )

            st.markdown('<h3 class="subsection-header">Price Sensitivity</h3>', unsafe_allow_html=True)
            fig_price = go.Figure()
            fig_price.add_trace(
                go.Scatter(
                    x=Sg, y=Z[nT // 2, :], mode="lines+markers",
                    name=f"T={Tg[nT//2]:.2f}", line=dict(color="#3B82F6", width=2.5),
                )
            )
            fig_price.update_layout(
                title=f"Price Sensitivity (T={Tg[nT//2]:.2f}, K={K})",
                xaxis_title="Spot Price (S)", yaxis_title="Option Price", height=400,
                template="plotly_dark", paper_bgcolor="rgba(30,41,59,1)", plot_bgcolor="rgba(15,23,42,1)",
            )
            st.plotly_chart(fig_price, use_container_width=True)

            st.markdown('<h3 class="subsection-header">Delta Sensitivity</h3>', unsafe_allow_html=True)
            fig_delta_sens = go.Figure()
            fig_delta_sens.add_trace(
                go.Scatter(
                    x=Sg, y=deltas_grid[nT // 2, :], mode="lines+markers",
                    name=f"T={Tg[nT//2]:.2f}", line=dict(color="#3B82F6", width=2.5),
                )
            )
            fig_delta_sens.add_hline(y=0, line_dash="dash", line_color="#F87171")
            fig_delta_sens.update_layout(
                title=f"Delta Sensitivity (T={Tg[nT//2]:.2f}, K={K})",
                xaxis_title="Spot Price (S)", yaxis_title="Delta", height=400,
                template="plotly_dark", paper_bgcolor="rgba(30,41,59,1)", plot_bgcolor="rgba(15,23,42,1)",
            )
            st.plotly_chart(fig_delta_sens, use_container_width=True)

        # Final progress
        progress_bar.progress(100)
        time.sleep(0.2)
        status_text.text("✅ Analysis complete!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        logger.exception("Critical error in Monte Carlo Unified")
        
        with st.expander("Error Details"):
            st.code(traceback.format_exc())

else:
    st.markdown(
        """
    <div style="text-align: center; padding: 2rem 0; background-color: #1E293B; border-radius: 8px; 
                 border: 1px solid #334155; margin-top: 1rem;">
        <h3 style="color: white; margin-bottom: 0.75rem;">Get Started</h3>
        <p style="color: #CBD5E1; margin-bottom: 1rem;">Configure your parameters above and click "Run Analysis" to see results</p>
    </div>
    """,
        unsafe_allow_html=True,
    )