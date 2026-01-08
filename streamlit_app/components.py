# streamlit_app/components.py
"""
Reusable UI components for OptionsLab Streamlit pages.

This module provides consistent styling and layout components
to ensure a sleek, modern look across all pages.

Components:
    - page_header: Standard page header
    - metric_card: Styled metric display
    - input_grid: Option parameter input form
    - chart_container: Consistent chart wrapper
    - apply_custom_css: Dark theme styling
"""

from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st


def apply_custom_css() -> None:
    """Apply global dark-theme custom CSS for sleek, modern look."""
    st.markdown(
        """
    <style>
        /* Full width layout */
        .main .block-container {
            max-width: 100%;
            padding: 1.5rem 2rem;
        }
        
        /* Remove default padding */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        }
        
        /* Headers */
        .page-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .page-subtitle {
            color: #94a3b8;
            font-size: 1.1rem;
            margin-bottom: 1.5rem;
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border-radius: 16px;
            padding: 1.25rem;
            border: 1px solid #475569;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
        
        .metric-label {
            color: #94a3b8;
            font-size: 0.85rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        
        .metric-value {
            color: #f8fafc;
            font-size: 2rem;
            font-weight: 700;
        }
        
        .metric-delta {
            color: #10b981;
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }
        
        .metric-delta.negative {
            color: #ef4444;
        }
        
        /* Input section */
        .input-section {
            background: #1e293b;
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid #334155;
            margin-bottom: 1.5rem;
        }
        
        .input-label {
            color: #cbd5e1;
            font-size: 0.9rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.2s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: #1e293b;
            border-radius: 12px;
            padding: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: #94a3b8;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            color: white;
        }
        
        /* Selectbox and Slider styling */
        .stSelectbox > div > div, .stSlider > div {
            background: #1e293b;
            border-radius: 8px;
        }
        
        /* Number inputs */
        .stNumberInput > div > div > input {
            background: #1e293b;
            border: 1px solid #475569;
            border-radius: 8px;
            color: #f8fafc;
        }
        
        /* Section divider */
        .section-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, #475569, transparent);
            margin: 2rem 0;
        }
        
        /* Chart container */
        .chart-container {
            background: #1e293b;
            border-radius: 16px;
            padding: 1rem;
            border: 1px solid #334155;
        }
        
        /* Hide streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "") -> None:
    """
    Render a styled page header.

    Args:
        title: Main page title
        subtitle: Optional subtitle description
    """
    st.markdown(f'<h1 class="page-header">{title}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def metric_card(
    label: str, value: str, delta: Optional[str] = None, delta_color: str = "normal"
) -> None:
    """
    Render a styled metric card.

    Args:
        label: Metric label text
        value: Metric value to display
        delta: Optional delta/change text
        delta_color: 'normal' (green), 'negative' (red), or 'neutral' (gray)
    """
    delta_class = "negative" if delta_color == "negative" else ""
    delta_html = (
        f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ""
    )

    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """,
        unsafe_allow_html=True,
    )


def section_divider() -> None:
    """Render a styled section divider."""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def input_section_start() -> None:
    """Start an input section container."""
    st.markdown('<div class="input-section">', unsafe_allow_html=True)


def input_section_end() -> None:
    """End an input section container."""
    st.markdown("</div>", unsafe_allow_html=True)


def option_input_grid(
    key_prefix: str = "",
    show_dividends: bool = True,
) -> Dict[str, Any]:
    """
    Render standard option parameter inputs in a grid layout.

    Returns dict with: S, K, T, r, sigma, q, option_type
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Asset**")
        S = st.number_input(
            "Spot Price ($)",
            min_value=1.0,
            max_value=1000.0,
            value=100.0,
            step=1.0,
            key=f"{key_prefix}spot",
        )
        K = st.number_input(
            "Strike Price ($)",
            min_value=1.0,
            max_value=1000.0,
            value=100.0,
            step=1.0,
            key=f"{key_prefix}strike",
        )

    with col2:
        st.markdown("**Time**")
        T = st.number_input(
            "Time to Maturity (years)",
            min_value=0.01,
            max_value=5.0,
            value=1.0,
            step=0.05,
            key=f"{key_prefix}time",
        )
        option_type = st.selectbox(
            "Option Type", ["call", "put"], key=f"{key_prefix}type"
        )

    with col3:
        st.markdown("**Market**")
        r = (
            st.number_input(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.5,
                key=f"{key_prefix}rate",
            )
            / 100.0
        )
        sigma = (
            st.number_input(
                "Volatility (%)",
                min_value=1.0,
                max_value=200.0,
                value=20.0,
                step=1.0,
                key=f"{key_prefix}vol",
            )
            / 100.0
        )

    with col4:
        st.markdown("**Simulation**")
        if show_dividends:
            q = (
                st.number_input(
                    "Dividend Yield (%)",
                    min_value=0.0,
                    max_value=20.0,
                    value=0.0,
                    step=0.5,
                    key=f"{key_prefix}div",
                )
                / 100.0
            )
        else:
            q = 0.0

        run_btn = st.button(
            "🚀 Calculate",
            type="primary",
            use_container_width=True,
            key=f"{key_prefix}run",
        )

    return {
        "S": S,
        "K": K,
        "T": T,
        "r": r,
        "sigma": sigma,
        "q": q,
        "option_type": option_type,
        "run": run_btn,
    }


def simulation_input_grid(key_prefix: str = "") -> Dict[str, Any]:
    """
    Render Monte Carlo simulation settings.

    Returns dict with: num_sims, num_steps, seed, use_numba
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        num_sims = st.select_slider(
            "Simulations",
            options=[1000, 5000, 10000, 25000, 50000, 100000],
            value=10000,
            key=f"{key_prefix}sims",
        )

    with col2:
        num_steps = st.select_slider(
            "Time Steps",
            options=[10, 25, 50, 100, 200, 500],
            value=50,
            key=f"{key_prefix}steps",
        )

    with col3:
        seed = st.number_input(
            "Random Seed",
            min_value=1,
            max_value=99999,
            value=42,
            key=f"{key_prefix}seed",
        )

    return {"num_sims": num_sims, "num_steps": num_steps, "seed": seed}


def get_chart_layout(title: str = "", height: int = 400) -> Dict[str, Any]:
    """Get consistent Plotly chart layout settings."""
    return {
        "title": {"text": title, "font": {"size": 18, "color": "#f8fafc"}},
        "paper_bgcolor": "rgba(30, 41, 59, 0.8)",
        "plot_bgcolor": "rgba(15, 23, 42, 0.9)",
        "font": {"color": "#cbd5e1"},
        "height": height,
        "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
        "xaxis": {
            "gridcolor": "#334155",
            "zerolinecolor": "#475569",
        },
        "yaxis": {
            "gridcolor": "#334155",
            "zerolinecolor": "#475569",
        },
        "legend": {
            "bgcolor": "rgba(30, 41, 59, 0.9)",
            "bordercolor": "#475569",
        },
    }


def format_price(value: float) -> str:
    """Format price value for display."""
    return f"${value:,.4f}"


def format_greek(value: float, decimals: int = 4) -> str:
    """Format Greek value for display."""
    return f"{value:,.{decimals}f}"


def format_time_ms(ms: float) -> str:
    """Format time in milliseconds for display."""
    if ms < 1:
        return f"{ms * 1000:.1f}μs"
    elif ms < 1000:
        return f"{ms:.1f}ms"
    else:
        return f"{ms / 1000:.2f}s"
