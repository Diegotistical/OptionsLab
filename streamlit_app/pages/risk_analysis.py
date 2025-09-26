"""
Advanced Risk Analysis Dashboard
Professional-grade risk analysis with interactive visualizations
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from typing import Tuple, Optional, Dict, Any
import io

# Set page config first
st.set_page_config(
    page_title="Advanced Risk Analysis", 
    page_icon="🛡️", 
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
# CORE FUNCTIONS
# ======================

def compute_var_es(returns: pd.Series, level: float = 0.95) -> Tuple[Optional[float], Optional[float]]:
    """Compute Value at Risk and Expected Shortfall with robust error handling"""
    try:
        if returns.empty or len(returns) < 10:
            return None, None
        
        returns = pd.to_numeric(returns, errors='coerce').dropna()
        if len(returns) < 10:
            return None, None
            
        var = -np.quantile(returns, 1 - level)
        losses_beyond_var = returns[returns < -var]
        es = -losses_beyond_var.mean() if len(losses_beyond_var) > 0 else -returns.min()
            
        return float(var), float(es)
    
    except Exception as e:
        st.error(f"Error computing VaR/ES: {str(e)}")
        return None, None

def generate_synthetic_returns(n: int, mu: float, sigma: float, 
                             distribution: str = "normal", 
                             seed: int = 42) -> pd.Series:
    """Generate synthetic returns with different distributions"""
    rng = np.random.default_rng(seed)
    
    if distribution == "normal":
        returns = rng.normal(mu, sigma, n)
    elif distribution == "student_t":
        returns = rng.standard_t(4, n) * sigma / np.sqrt(4) + mu
    elif distribution == "skewed":
        from scipy.stats import skewnorm
        returns = skewnorm.rvs(5, loc=mu, scale=sigma, size=n, random_state=seed)
    else:
        returns = rng.normal(mu, sigma, n)
    
    return pd.Series(returns, name="synthetic_returns")

def create_distribution_plot(returns: pd.Series, var: float, es: float, level: float) -> go.Figure:
    """Create interactive distribution plot with risk metrics"""
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Returns',
        opacity=0.7,
        marker_color='#3b82f6',
        hovertemplate='Return: %{x:.4f}<br>Count: %{y}<extra></extra>'
    ))
    
    # VaR line
    fig.add_vline(x=-var, line_dash="dash", line_color="#ef4444", 
                  annotation_text=f"VaR {level*100:.1f}%: {-var:.4f}")
    
    # ES line
    fig.add_vline(x=-es, line_dash="dash", line_color="#dc2626",
                  annotation_text=f"ES {level*100:.1f}%: {-es:.4f}")
    
    # Tail region shading
    tail_returns = returns[returns < -var]
    if len(tail_returns) > 0:
        fig.add_trace(go.Scatter(
            x=tail_returns,
            y=[0] * len(tail_returns),
            mode='markers',
            marker=dict(color='red', size=4, opacity=0.6),
            name='Tail Events',
            hovertemplate='Tail Return: %{x:.4f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Return Distribution with Risk Metrics",
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=500,
        showlegend=True,
        font=dict(color='#e2e8f0')
    )
    
    return fig

def create_qq_plot(returns: pd.Series) -> go.Figure:
    """Create Q-Q plot for normality assessment"""
    try:
        # Calculate theoretical quantiles
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)))
        sample_quantiles = np.percentile(returns, np.linspace(1, 99, len(returns)))
        
        fig = go.Figure()
        
        # Q-Q line
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sample_quantiles,
            mode='markers',
            name='Q-Q Points',
            marker=dict(color='#3b82f6', size=4)
        ))
        
        # Perfect normality line
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Normal Line',
            line=dict(color='#ef4444', dash='dash')
        ))
        
        fig.update_layout(
            title="Q-Q Plot (Normality Assessment)",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            template="plotly_dark",
            height=400,
            showlegend=True
        )
        
        return fig
    except Exception as e:
        # Fallback simple plot
        fig = go.Figure()
        fig.add_annotation(text="Q-Q Plot Not Available", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(template="plotly_dark", height=400)
        return fig

def create_time_series_plots(returns: pd.Series) -> go.Figure:
    """Create interactive time series analysis"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Cumulative Returns', 'Rolling Volatility (30-day)'),
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4]
    )
    
    # Cumulative returns
    cumulative_returns = (1 + returns).cumprod() - 1
    fig.add_trace(
        go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values,
                  name='Cumulative Returns', line=dict(color='#10b981')),
        row=1, col=1
    )
    
    # Rolling volatility (annualized)
    rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
    fig.add_trace(
        go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                  name='Rolling Vol', line=dict(color='#ef4444')),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        template="plotly_dark",
        showlegend=True,
        font=dict(color='#e2e8f0')
    )
    
    fig.update_yaxes(tickformat=".1%", row=1, col=1)
    fig.update_yaxes(tickformat=".1%", row=2, col=1)
    
    return fig

def compute_comprehensive_metrics(returns: pd.Series) -> Dict[str, Any]:
    """Compute comprehensive risk metrics"""
    metrics = {}
    
    try:
        metrics['volatility'] = returns.std()
        metrics['sharpe'] = returns.mean() / returns.std() if returns.std() > 1e-10 else 0
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        metrics['drawdown_duration'] = (drawdown < 0).astype(int).groupby((drawdown < 0).diff().ne(0).cumsum()).sum().max()
        
        # VaR at different levels
        for level in [0.95, 0.99, 0.995]:
            metrics[f'var_{int(level*100)}'] = -np.quantile(returns, 1 - level)
        
        # Worst losses
        metrics['worst_losses'] = returns.nsmallest(5).values
        
        # Tail risk metrics
        metrics['expected_shortfall_95'] = -returns[returns < -metrics['var_95']].mean() if len(returns[returns < -metrics['var_95']]) > 0 else 0
        
    except Exception as e:
        st.error(f"Error computing metrics: {str(e)}")
    
    return metrics

# ======================
# MAIN APPLICATION
# ======================

st.markdown('<h1 class="main-header">🛡️ Advanced Risk Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional-grade risk analytics with interactive visualizations</p>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Data Source</div>', unsafe_allow_html=True)
    mode = st.radio("", ["Upload CSV", "Generate Synthetic"], horizontal=True, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="engine-option">', unsafe_allow_html=True)
    st.markdown('<div class="engine-label">Confidence Level</div>', unsafe_allow_html=True)
    level = st.slider("", 0.80, 0.995, 0.95, 0.005, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if mode == "Upload CSV":
        st.info("📁 Upload CSV with single column of daily returns")
    else:
        st.markdown('<div class="engine-option">', unsafe_allow_html=True)
        st.markdown('<div class="engine-label">Sample Size (days)</div>', unsafe_allow_html=True)
        n = st.slider("", 100, 10000, 1000, 100, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="engine-option">', unsafe_allow_html=True)
            st.markdown('<div class="engine-label">Mean Return</div>', unsafe_allow_html=True)
            mu = st.number_input("", -0.01, 0.01, 0.0005, 0.0001, format="%.5f", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="engine-option">', unsafe_allow_html=True)
            st.markdown('<div class="engine-label">Volatility</div>', unsafe_allow_html=True)
            sigma = st.number_input("", 0.0001, 0.10, 0.02, 0.0001, format="%.4f", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="engine-option">', unsafe_allow_html=True)
        st.markdown('<div class="engine-label">Distribution</div>', unsafe_allow_html=True)
        distribution = st.selectbox("", ["normal", "student_t", "skewed"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

# Main application logic
try:
    # Data loading/generation
    if mode == "Upload CSV":
        st.markdown('<div class="subsection-header">📤 Data Upload</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload returns CSV", type=["csv"], 
                                       label_visibility="collapsed")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) == 0:
                    st.error("❌ No numeric columns found in uploaded file")
                    st.stop()
                
                series = df[numeric_cols[0]].astype(float).dropna()
                st.success(f"✅ Loaded {len(series):,} data points")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.stop()
        else:
            # Show example format
            example_data = pd.DataFrame({'returns': np.random.normal(0.0005, 0.02, 100)})
            st.info("💡 Expected CSV format:")
            st.dataframe(example_data.head(), use_container_width=True)
            st.stop()
    else:
        st.markdown('<div class="subsection-header">🔧 Synthetic Data Generation</div>', unsafe_allow_html=True)
        series = generate_synthetic_returns(n, mu, sigma, distribution)
        st.success(f"✅ Generated {len(series):,} synthetic returns")

    # Data validation
    if len(series) < 30:
        st.warning(f"⚠️ Limited data points ({len(series)}). Consider larger sample.")
    
    if series.std() < 1e-10:
        st.error("❌ Zero volatility detected. Check data quality.")
        st.stop()

    # Compute risk metrics
    var, es = compute_var_es(series, level)
    metrics = compute_comprehensive_metrics(series)

    # Key metrics dashboard
    st.markdown('<div class="subsection-header">📊 Risk Dashboard</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
        st.markdown('<div class="executive-title">Sample Size</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="executive-value">{len(series):,}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
        st.markdown('<div class="executive-title">Daily Volatility</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="executive-value">{series.std():.4%}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
        st.markdown('<div class="executive-title">Sharpe Ratio</div>', unsafe_allow_html=True)
        sharpe = series.mean() / series.std() if series.std() > 1e-10 else 0
        st.markdown(f'<div class="executive-value">{sharpe:.3f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
        st.markdown('<div class="executive-title">Max Drawdown</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="executive-value">{metrics.get("max_drawdown", 0):.4%}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # VaR/ES metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
        st.markdown(f'<div class="executive-title">Value at Risk ({level*100:.1f}%)</div>', unsafe_allow_html=True)
        var_display = f"{-var:.4%}" if var is not None else "N/A"
        st.markdown(f'<div class="executive-value" style="color: #ef4444;">{var_display}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
        st.markdown(f'<div class="executive-title">Expected Shortfall ({level*100:.1f}%)</div>', unsafe_allow_html=True)
        es_display = f"{-es:.4%}" if es is not None else "N/A"
        st.markdown(f'<div class="executive-value" style="color: #dc2626;">{es_display}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Interactive visualization tabs
    st.markdown('<div class="subsection-header">📈 Interactive Analytics</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "⏰ Time Series", "📋 Risk Metrics", "📤 Export"])
    
    with tab1:
        if var is not None and es is not None:
            fig_dist = create_distribution_plot(series, var, es, level)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_qq = create_qq_plot(series)
            st.plotly_chart(fig_qq, use_container_width=True)
        
        with col2:
            # Additional distribution stats
            st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
            st.markdown('<div class="executive-title">Distribution Characteristics</div>', unsafe_allow_html=True)
            
            stats_data = {
                'Skewness': f"{metrics.get('skewness', 0):.3f}",
                'Kurtosis': f"{metrics.get('kurtosis', 0):.3f}",
                'Jarque-Bera p-value': f"{stats.jarque_bera(series)[1]:.4f}",
                'Normality': 'Normal' if abs(metrics.get('skewness', 0)) < 0.5 and metrics.get('kurtosis', 0) < 3.5 else 'Non-Normal'
            }
            
            for key, value in stats_data.items():
                st.markdown(f'<div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">', unsafe_allow_html=True)
                st.markdown(f'<span style="color: #94a3b8;">{key}:</span>', unsafe_allow_html=True)
                st.markdown(f'<span style="color: #f8fafc; font-weight: 500;">{value}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        fig_ts = create_time_series_plots(series)
        st.plotly_chart(fig_ts, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
            st.markdown('<div class="executive-title">📈 Statistical Measures</div>', unsafe_allow_html=True)
            
            stat_metrics = {
                'Mean Return': f"{series.mean():.4%}",
                'Volatility': f"{series.std():.4%}",
                'Sharpe Ratio': f"{sharpe:.3f}",
                'Skewness': f"{metrics.get('skewness', 0):.3f}",
                'Kurtosis': f"{metrics.get('kurtosis', 0):.3f}",
            }
            
            for key, value in stat_metrics.items():
                st.markdown(f'<div style="display: flex; justify-content: space-between; margin: 0.3rem 0; padding: 0.2rem 0; border-bottom: 1px solid #334155;">', unsafe_allow_html=True)
                st.markdown(f'<span style="color: #94a3b8;">{key}</span>', unsafe_allow_html=True)
                st.markdown(f'<span style="color: #f8fafc; font-weight: 500;">{value}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
            st.markdown('<div class="executive-title">🛡️ Risk Measures</div>', unsafe_allow_html=True)
            
            risk_metrics = {
                'Max Drawdown': f"{metrics.get('max_drawdown', 0):.4%}",
                'VaR 95%': f"{metrics.get('var_95', 0):.4%}",
                'VaR 99%': f"{metrics.get('var_99', 0):.4%}",
                'VaR 99.5%': f"{metrics.get('var_995', 0):.4%}",
                'Expected Shortfall 95%': f"{metrics.get('expected_shortfall_95', 0):.4%}",
            }
            
            for key, value in risk_metrics.items():
                st.markdown(f'<div style="display: flex; justify-content: space-between; margin: 0.3rem 0; padding: 0.2rem 0; border-bottom: 1px solid #334155;">', unsafe_allow_html=True)
                st.markdown(f'<span style="color: #94a3b8;">{key}</span>', unsafe_allow_html=True)
                st.markdown(f'<span style="color: #f8fafc; font-weight: 500;">{value}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Worst losses table
        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
        st.markdown('<div class="executive-title">📉 Worst Daily Losses</div>', unsafe_allow_html=True)
        
        worst_losses = series.nsmallest(5)
        for i, (idx, loss) in enumerate(worst_losses.items(), 1):
            st.markdown(f'<div style="display: flex; justify-content: space-between; margin: 0.3rem 0; padding: 0.2rem 0; border-bottom: 1px solid #334155;">', unsafe_allow_html=True)
            st.markdown(f'<span style="color: #94a3b8;">#{i} Worst</span>', unsafe_allow_html=True)
            st.markdown(f'<span style="color: #ef4444; font-weight: 500;">{loss:.4%}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="executive-insight">', unsafe_allow_html=True)
        st.markdown('<div class="executive-title">💾 Export Risk Report</div>', unsafe_allow_html=True)
        
        # Generate comprehensive report
        report_text = f"""
RISK ANALYSIS REPORT
===================
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS
------------------
Sample Size: {len(series):,}
Mean Daily Return: {series.mean():.4%}
Daily Volatility: {series.std():.4%}
Sharpe Ratio: {sharpe:.3f}
Skewness: {metrics.get('skewness', 0):.3f}
Kurtosis: {metrics.get('kurtosis', 0):.3f}

RISK METRICS
------------
Value at Risk ({level*100:.1f}%): {-var:.4%}
Expected Shortfall ({level*100:.1f}%): {-es:.4%}
Max Drawdown: {metrics.get('max_drawdown', 0):.4%}
VaR 95%: {metrics.get('var_95', 0):.4%}
VaR 99%: {metrics.get('var_99', 0):.4%}

WORST LOSSES
------------
{chr(10).join([f'{i+1}. {loss:.4%}' for i, loss in enumerate(series.nsmallest(5))])}
        """
        
        st.download_button(
            label="📥 Download Full Report",
            data=report_text,
            file_name=f"risk_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        # Export data
        csv_data = series.to_csv(index=False)
        st.download_button(
            label="📊 Export Returns Data (CSV)",
            data=csv_data,
            file_name="returns_data.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"❌ Application error: {str(e)}")
    st.info("💡 Please refresh the page and try again with valid parameters.")

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; color: #64748b; font-size: 0.9rem;">', unsafe_allow_html=True)
st.markdown('🔒 Professional Risk Analytics Tool • For institutional use only', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)