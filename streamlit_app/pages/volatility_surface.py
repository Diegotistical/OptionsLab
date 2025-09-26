"""
Advanced Volatility Surface Dashboard
Professional-grade volatility surface construction with 3D visualization and arbitrage checks
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import griddata, Rbf
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set page config first
st.set_page_config(
    page_title="Advanced Volatility Surface", 
    page_icon="🗺️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# DARK MODE STYLING
# ======================
st.markdown("""
<style>
    /* Base styling - full width dark theme */
    body {
        padding: 0 !important;
        margin: 0 !important;
        background-color: #0f172a;
        color: #e2e8f0;
    }
    .main-header {
        font-size: 2.5rem;
        color: #f8fafc;
        margin-bottom: 0.5rem;
        font-weight: 700;
        text-align: center;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
        opacity: 0.9;
        text-align: center;
    }
    /* Full width containers */
    .stApp {
        max-width: 100% !important;
        padding: 0 1rem !important;
        background-color: #0f172a;
    }
    /* Metric cards */
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    /* Button styling - Full width */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.8rem 2rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        width: 100% !important;
        margin: 0.5rem 0 !important;
        max-width: 100% !important;
    }
    .stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    /* Input sections */
    .engine-option {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
        border: 1px solid #334155;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    .engine-label {
        font-size: 0.9rem;
        color: #f8fafc !important;
        margin-bottom: 0.3rem;
    }
    /* Section headers */
    .subsection-header {
        font-size: 1.3rem;
        color: #f8fafc;
        margin: 1.2rem 0 0.8rem 0;
        font-weight: 600;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.5rem;
    }
    /* Executive insights */
    .executive-insight {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #334155;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .executive-title {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-bottom: 0.3rem;
    }
    .executive-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #f8fafc;
    }
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: #1e293b;
        border-radius: 8px 8px 0px 0px;
        gap: 1rem;
        padding: 0.5rem 1rem;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# CORE VOLATILITY SURFACE FUNCTIONS
# ======================

class VolatilitySurfaceResult:
    """Container for volatility surface results"""
    def __init__(self, strikes, maturities, iv_grid, spot_price=100):
        self.strikes = strikes
        self.maturities = maturities
        self.iv_grid = iv_grid
        self.spot_price = spot_price

def build_volatility_surface(strikes: np.ndarray, maturities: np.ndarray, ivs: np.ndarray,
                           strike_points: int = 100, maturity_points: int = 100,
                           method: str = 'cubic', extrapolate: bool = False,
                           spot_price: float = 100) -> VolatilitySurfaceResult:
    """Build volatility surface with advanced interpolation"""
    try:
        # Validate inputs
        if len(strikes) < 4 or len(maturities) < 4:
            raise ValueError("Need at least 4 data points for surface construction")
        
        # Remove outliers and invalid values
        valid_mask = (ivs > 0.01) & (ivs < 2.0) & (strikes > 0) & (maturities > 0)
        strikes = strikes[valid_mask]
        maturities = maturities[valid_mask]
        ivs = ivs[valid_mask]
        
        if len(strikes) < 4:
            raise ValueError("Not enough valid data points after filtering")
        
        # Create grid for interpolation
        strike_min, strike_max = strikes.min(), strikes.max()
        maturity_min, maturity_max = maturities.min(), maturities.max()
        
        # Expand grid slightly if extrapolation is allowed
        if extrapolate:
            strike_range = strike_max - strike_min
            maturity_range = maturity_max - maturity_min
            strike_min = max(strike_min - strike_range * 0.1, strike_min * 0.5)
            strike_max = strike_max + strike_range * 0.1
            maturity_min = max(maturity_min - maturity_range * 0.1, 0.01)
            maturity_max = maturity_max + maturity_range * 0.1
        
        strike_grid = np.linspace(strike_min, strike_max, strike_points)
        maturity_grid = np.linspace(maturity_min, maturity_max, maturity_points)
        strike_mesh, maturity_mesh = np.meshgrid(strike_grid, maturity_grid)
        
        # Perform interpolation
        if method == 'rbf':
            # Radial basis function interpolation
            rbf = Rbf(strikes, maturities, ivs, function='thin_plate', smooth=0.1)
            iv_grid = rbf(strike_mesh, maturity_mesh)
        else:
            # Grid-based interpolation
            points = np.column_stack((strikes, maturities))
            iv_grid = griddata(points, ivs, (strike_mesh, maturity_mesh), 
                             method=method, fill_value=np.nan if not extrapolate else ivs.mean())
        
        # Clean up any extreme values
        iv_grid = np.clip(iv_grid, 0.01, 1.0)
        
        return VolatilitySurfaceResult(strike_grid, maturity_grid, iv_grid, spot_price)
    
    except Exception as e:
        raise ValueError(f"Surface construction failed: {str(e)}")

def check_butterfly_arbitrage(strikes: np.ndarray, iv_slice: np.ndarray, spot_price: float = 100) -> Dict[str, Any]:
    """Check for butterfly arbitrage in volatility slice"""
    try:
        if len(strikes) < 3:
            return {"error": "Need at least 3 strikes for butterfly check"}
        
        # Calculate second derivative approximation
        d2iv_dk2 = np.gradient(np.gradient(iv_slice, strikes), strikes)
        
        # Butterfly arbitrage condition: second derivative should be positive for convexity
        arbitrage_points = np.where(d2iv_dk2 < -0.001)[0]
        
        return {
            "has_arbitrage": len(arbitrage_points) > 0,
            "arbitrage_points": int(len(arbitrage_points)),
            "convexity_score": float(np.mean(d2iv_dk2)),
            "min_convexity": float(np.min(d2iv_dk2)),
            "max_convexity": float(np.max(d2iv_dk2))
        }
    except Exception as e:
        return {"error": f"Butterfly check failed: {str(e)}"}

def check_calendar_arbitrage(surface: VolatilitySurfaceResult) -> Dict[str, Any]:
    """Check for calendar arbitrage in volatility surface"""
    try:
        # Check that variance is increasing with time for each strike
        variances = surface.iv_grid ** 2 * surface.maturities
        calendar_violations = 0
        
        for i in range(len(surface.strikes)):
            var_slice = variances[:, i]
            # Variance should be non-decreasing with time
            decreasing_var = np.where(np.diff(var_slice) < -0.0001)[0]
            calendar_violations += len(decreasing_var)
        
        return {
            "has_calendar_arbitrage": calendar_violations > 0,
            "calendar_violations": calendar_violations,
            "total_checks": len(surface.strikes) * (len(surface.maturities) - 1)
        }
    except Exception as e:
        return {"error": f"Calendar arbitrage check failed: {str(e)}"}

def generate_synthetic_surface(spot_price: float = 100, num_points: int = 100) -> pd.DataFrame:
    """Generate realistic synthetic volatility surface data"""
    np.random.seed(42)
    
    # Realistic parameters for volatility surface
    strikes = np.linspace(spot_price * 0.6, spot_price * 1.4, 20)
    maturities = np.linspace(0.1, 2.0, 15)
    
    data = []
    for strike in strikes:
        for maturity in maturities:
            # Realistic volatility surface: smile effect + term structure
            moneyness = strike / spot_price
            atm_vol = 0.18 + 0.02 * np.sqrt(maturity)  # Term structure
            smile_effect = 0.25 * (moneyness - 1) ** 2  # Volatility smile
            iv = atm_vol + smile_effect + np.random.normal(0, 0.01)
            
            data.append({
                'strike': strike,
                'maturity': maturity,
                'iv': max(0.05, min(0.8, iv))  # Realistic bounds
            })
    
    return pd.DataFrame(data)

# ======================
# PLOTTING FUNCTIONS
# ======================

def create_3d_volatility_surface(surface: VolatilitySurfaceResult) -> go.Figure:
    """Create interactive 3D volatility surface plot"""
    fig = go.Figure(data=[go.Surface(
        x=surface.strikes,
        y=surface.maturities,
        z=surface.iv_grid,
        colorscale='Viridis',
        colorbar=dict(title="Implied Volatility", titleside="right"),
        hovertemplate='<b>Strike:</b> %{x:.1f}<br><b>Maturity:</b> %{y:.2f} yrs<br><b>IV:</b> %{z:.3f}<extra></extra>'
    )])
    
    fig.update_layout(
        title="3D Volatility Surface",
        scene=dict(
            xaxis_title="Strike Price",
            yaxis_title="Time to Maturity (Years)",
            zaxis_title="Implied Volatility",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        template="plotly_dark",
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def create_volatility_slices(surface: VolatilitySurfaceResult) -> go.Figure:
    """Create 2D slices of volatility surface"""
    fig = go.Figure()
    
    # Add slices for different maturities
    maturity_indices = [0, len(surface.maturities)//4, len(surface.maturities)//2, 
                      3*len(surface.maturities)//4, -1]
    
    for i, idx in enumerate(maturity_indices):
        if idx < len(surface.maturities):
            fig.add_trace(go.Scatter(
                x=surface.strikes,
                y=surface.iv_grid[idx, :],
                mode='lines',
                name=f'{surface.maturities[idx]:.2f} yrs',
                line=dict(width=3),
                hovertemplate='<b>Strike:</b> %{x:.1f}<br><b>IV:</b> %{y:.3f}<extra></extra>'
            ))
    
    fig.update_layout(
        title="Volatility Smile Slices by Maturity",
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility",
        template="plotly_dark",
        height=400,
        showlegend=True
    )
    
    return fig

def create_term_structure(surface: VolatilitySurfaceResult, atm_strike: float = 100) -> go.Figure:
    """Create term structure plot for ATM volatility"""
    fig = go.Figure()
    
    # Find closest strike to ATM
    atm_idx = np.argmin(np.abs(surface.strikes - atm_strike))
    
    fig.add_trace(go.Scatter(
        x=surface.maturities,
        y=surface.iv_grid[:, atm_idx],
        mode='lines+markers',
        name='ATM Term Structure',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=6),
        hovertemplate='<b>Maturity:</b> %{x:.2f} yrs<br><b>IV:</b> %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="ATM Volatility Term Structure",
        xaxis_title="Time to Maturity (Years)",
        yaxis_title="Implied Volatility",
        template="plotly_dark",
        height=400
    )
    
    return fig

def create_arbitrage_heatmap(surface: VolatilitySurfaceResult) -> go.Figure:
    """Create heatmap showing potential arbitrage regions"""
    # Calculate convexity for each point
    convexity_map = np.zeros_like(surface.iv_grid)
    
    for i in range(len(surface.maturities)):
        iv_slice = surface.iv_grid[i, :]
        d2iv_dk2 = np.gradient(np.gradient(iv_slice, surface.strikes), surface.strikes)
        convexity_map[i, :] = d2iv_dk2
    
    fig = go.Figure(data=go.Heatmap(
        x=surface.strikes,
        y=surface.maturities,
        z=convexity_map,
        colorscale='RdBu',
        colorbar=dict(title="Convexity"),
        hovertemplate='<b>Strike:</b> %{x:.1f}<br><b>Maturity:</b> %{y:.2f} yrs<br><b>Convexity:</b> %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Butterfly Arbitrage Check (Convexity Heatmap)",
        xaxis_title="Strike Price",
        yaxis_title="Time to Maturity (Years)",
        template="plotly_dark",
        height=500
    )
    
    return fig

# ======================
# MAIN APPLICATION
# ======================

st.markdown('<h1 class="main-header">🗺️ Advanced Volatility Surface Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional-grade volatility surface construction and analysis</p>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Data Source</div>', unsafe_allow_html=True)
    data_source = st.radio("", ["Upload CSV", "Generate Synthetic"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Interpolation Method</div>', unsafe_allow_html=True)
    interpolation_method = st.selectbox("", ["cubic", "linear", "nearest", "rbf"], 
                                      label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="engine-option">', unsafe_allow_html=True)
        st.markdown('<div class="engine-label">Strike Points</div>', unsafe_allow_html=True)
        strike_points = st.slider("", 20, 200, 100, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="engine-option">', unsafe_allow_html=True)
        st.markdown('<div class="engine-label">Maturity Points</div>', unsafe_allow_html=True)
        maturity_points = st.slider("", 20, 200, 100, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Advanced Settings</div>', unsafe_allow_html=True)
    extrapolate = st.checkbox("Allow Extrapolation", value=False)
    spot_price = st.number_input("Spot Price", value=100.0, min_value=1.0, max_value=1000.0, step=10.0)
    st.markdown('</div>', unsafe_allow_html=True)

# Main content area
try:
    # Data loading/generation
    if data_source == "Upload CSV":
        st.markdown('<div class="subsection-header">📤 Data Upload</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload option data CSV", type=["csv"],
                                       help="CSV should contain columns: strike, maturity, iv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                df.columns = [col.strip().lower() for col in df.columns]
                
                # Validate required columns
                required_cols = ['strike', 'maturity', 'iv']
                if not all(col in df.columns for col in required_cols):
                    st.error("❌ CSV must contain columns: strike, maturity, iv")
                    st.stop()
                
                # Clean data
                df = df.dropna()
                if len(df) < 4:
                    st.error("❌ Need at least 4 valid data points")
                    st.stop()
                
                st.success(f"✅ Loaded {len(df)} option data points")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.stop()
        else:
            # Show example and generate synthetic data
            st.info("💡 Upload a CSV with strike, maturity, iv columns or use synthetic data")
            df = generate_synthetic_surface(spot_price)
    
    else:  # Generate Synthetic
        st.markdown('<div class="subsection-header">🔧 Synthetic Data Generation</div>', unsafe_allow_html=True)
        df = generate_synthetic_surface(spot_price)
        st.success(f"✅ Generated {len(df)} synthetic option data points")

    # Display data summary
    st.markdown('<div class="subsection-header">📊 Data Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
        st.markdown('<div class="executive-title">Data Points</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="executive-value">{len(df):,}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
        st.markdown('<div class="executive-title">Strike Range</div>', unsafe_allow_html=True)
        strike_range = f"${df['strike'].min():.1f} - ${df['strike'].max():.1f}"
        st.markdown(f'<div class="executive-value">{strike_range}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
        st.markdown('<div class="executive-title">Maturity Range</div>', unsafe_allow_html=True)
        maturity_range = f"{df['maturity'].min():.2f} - {df['maturity'].max():.2f} yrs"
        st.markdown(f'<div class="executive-value">{maturity_range}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
        st.markdown('<div class="executive-title">IV Range</div>', unsafe_allow_html=True)
        iv_range = f"{df['iv'].min():.3f} - {df['iv'].max():.3f}"
        st.markdown(f'<div class="executive-value">{iv_range}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Build volatility surface
    if st.button("🚀 Build Volatility Surface", use_container_width=True):
        with st.spinner("Constructing volatility surface..."):
            try:
                surface = build_volatility_surface(
                    strikes=df['strike'].values,
                    maturities=df['maturity'].values,
                    ivs=df['iv'].values,
                    strike_points=strike_points,
                    maturity_points=maturity_points,
                    method=interpolation_method,
                    extrapolate=extrapolate,
                    spot_price=spot_price
                )
                
                st.session_state.surface = surface
                st.success("✅ Volatility surface constructed successfully!")
                
            except Exception as e:
                st.error(f"❌ Surface construction failed: {str(e)}")
                st.stop()

    # Display results if surface is available
    if 'surface' in st.session_state:
        surface = st.session_state.surface
        
        # Arbitrage checks
        st.markdown('<div class="subsection-header">🛡️ Arbitrage Checks</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            # Butterfly arbitrage check
            mid_maturity_idx = len(surface.maturities) // 2
            butterfly_check = check_butterfly_arbitrage(
                surface.strikes, 
                surface.iv_grid[mid_maturity_idx, :],
                spot_price
            )
            
            st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
            st.markdown('<div class="executive-title">Butterfly Arbitrage</div>', unsafe_allow_html=True)
            if 'error' not in butterfly_check:
                status = "✅ Clean" if not butterfly_check['has_arbitrage'] else "❌ Detected"
                color = "#10b981" if not butterfly_check['has_arbitrage'] else "#ef4444"
                st.markdown(f'<div class="executive-value" style="color: {color};">{status}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="executive-title">Convexity: {butterfly_check["convexity_score"]:.4f}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="executive-value" style="color: #f59e0b;">⚠️ Error</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Calendar arbitrage check
            calendar_check = check_calendar_arbitrage(surface)
            
            st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
            st.markdown('<div class="executive-title">Calendar Arbitrage</div>', unsafe_allow_html=True)
            if 'error' not in calendar_check:
                status = "✅ Clean" if not calendar_check['has_calendar_arbitrage'] else "❌ Detected"
                color = "#10b981" if not calendar_check['has_calendar_arbitrage'] else "#ef4444"
                st.markdown(f'<div class="executive-value" style="color: {color};">{status}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="executive-title">Violations: {calendar_check["calendar_violations"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="executive-value" style="color: #f59e0b;">⚠️ Error</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Interactive visualization tabs
        st.markdown('<div class="subsection-header">📈 Interactive Visualizations</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["3D Surface", "Volatility Slices", "Term Structure", "Arbitrage Map"])
        
        with tab1:
            fig_3d = create_3d_volatility_surface(surface)
            st.plotly_chart(fig_3d, use_container_width=True)
        
        with tab2:
            fig_slices = create_volatility_slices(surface)
            st.plotly_chart(fig_slices, use_container_width=True)
        
        with tab3:
            fig_term = create_term_structure(surface, spot_price)
            st.plotly_chart(fig_term, use_container_width=True)
        
        with tab4:
            fig_arbitrage = create_arbitrage_heatmap(surface)
            st.plotly_chart(fig_arbitrage, use_container_width=True)
        
        # Surface statistics
        st.markdown('<div class="subsection-header">📋 Surface Statistics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
            st.markdown('<div class="executive-title">Surface Metrics</div>', unsafe_allow_html=True)
            
            metrics = {
                'Grid Size': f"{len(surface.strikes)} × {len(surface.maturities)}",
                'Mean IV': f"{surface.iv_grid.mean():.3f}",
                'IV Std Dev': f"{surface.iv_grid.std():.3f}",
                'Min IV': f"{surface.iv_grid.min():.3f}",
                'Max IV': f"{surface.iv_grid.max():.3f}",
            }
            
            for key, value in metrics.items():
                st.markdown(f'<div style="display: flex; justify-content: space-between; margin: 0.5rem 0; padding: 0.2rem 0; border-bottom: 1px solid #334155;">', unsafe_allow_html=True)
                st.markdown(f'<span style="color: #94a3b8;">{key}</span>', unsafe_allow_html=True)
                st.markdown(f'<span style="color: #f8fafc; font-weight: 500;">{value}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
            st.markdown('<div class="executive-title">ATM Volatility Term Structure</div>', unsafe_allow_html=True)
            
            # ATM volatilities at key maturities
            atm_idx = np.argmin(np.abs(surface.strikes - spot_price))
            key_maturities = [0.25, 0.5, 1.0, 2.0]
            
            for maturity in key_maturities:
                maturity_idx = np.argmin(np.abs(surface.maturities - maturity))
                if maturity_idx < len(surface.maturities):
                    iv = surface.iv_grid[maturity_idx, atm_idx]
                    st.markdown(f'<div style="display: flex; justify-content: space-between; margin: 0.5rem 0; padding: 0.2rem 0; border-bottom: 1px solid #334155;">', unsafe_allow_html=True)
                    st.markdown(f'<span style="color: #94a3b8;">{maturity} yr</span>', unsafe_allow_html=True)
                    st.markdown(f'<span style="color: #f8fafc; font-weight: 500;">{iv:.3f}</span>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Export functionality
        st.markdown('<div class="subsection-header">💾 Export Results</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            # Export surface data
            surface_df = pd.DataFrame({
                'strike': np.repeat(surface.strikes, len(surface.maturities)),
                'maturity': np.tile(surface.maturities, len(surface.strikes)),
                'iv': surface.iv_grid.flatten()
            })
            
            csv_data = surface_df.to_csv(index=False)
            st.download_button(
                label="📊 Export Surface Data (CSV)",
                data=csv_data,
                file_name="volatility_surface.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Generate report
            report_text = f"""
VOLATILITY SURFACE REPORT
=========================
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

SURFACE PARAMETERS
------------------
Spot Price: ${spot_price:.2f}
Interpolation Method: {interpolation_method}
Grid Size: {len(surface.strikes)} × {len(surface.maturities)}
Extrapolation: {extrapolate}

ARBITRAGE CHECKS
----------------
Butterfly Arbitrage: {'Clean' if not butterfly_check.get('has_arbitrage', True) else 'Detected'}
Calendar Arbitrage: {'Clean' if not calendar_check.get('has_calendar_arbitrage', True) else 'Detected'}

SURFACE STATISTICS
------------------
Mean IV: {surface.iv_grid.mean():.3f}
IV Range: {surface.iv_grid.min():.3f} - {surface.iv_grid.max():.3f}
            """
            
            st.download_button(
                label="📄 Download Surface Report",
                data=report_text,
                file_name="volatility_surface_report.txt",
                mime="text/plain",
                use_container_width=True
            )

except Exception as e:
    st.error(f"❌ Application error: {str(e)}")
    st.info("💡 Please check your data and parameters, then try again.")

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; color: #64748b; font-size: 0.9rem;">', unsafe_allow_html=True)
st.markdown('🗺️ Professional Volatility Surface Analytics • For institutional use only', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)