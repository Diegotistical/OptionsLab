# streamlit_app/pages/benchmarks.py
"""
Benchmarks - Streamlit Page.

Compare performance of different pricing engines.
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
import streamlit as st

st.set_page_config(
    page_title="Benchmarks",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    from components import (
        apply_custom_css,
        format_price,
        format_time_ms,
        get_chart_layout,
        page_header,
        section_divider,
    )
except ImportError:
    from streamlit_app.components import (
        apply_custom_css,
        format_price,
        format_time_ms,
        get_chart_layout,
        page_header,
        section_divider,
    )

# Import pricing models - availability check only
try:
    from src.pricing_models import GPU_AVAILABLE, LIGHTGBM_AVAILABLE, NUMBA_AVAILABLE

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    NUMBA_AVAILABLE = False
    GPU_AVAILABLE = False
    LIGHTGBM_AVAILABLE = False

apply_custom_css()

# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================


def benchmark_function(func, *args, n_trials: int = 5, **kwargs) -> tuple:
    """Run function multiple times and return (result, avg_time_ms, std_time_ms)."""
    times = []
    result = None

    for _ in range(n_trials):
        t_start = time.perf_counter()
        result = func(*args, **kwargs)
        times.append((time.perf_counter() - t_start) * 1000)

    return result, np.mean(times), np.std(times)


def run_pricing_benchmarks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    num_sims: int,
    use_numba: bool,
) -> list:
    """Run all pricing benchmarks and return results."""
    results = []

    # Black-Scholes (vectorized, instant)
    try:
        from src.pricing_models import black_scholes

        price, avg_time, std_time = benchmark_function(
            black_scholes, S, K, T, r, sigma, option_type, n_trials=10
        )
        results.append(
            {
                "model": "Black-Scholes",
                "price": price,
                "time_ms": avg_time,
                "std_ms": std_time,
                "status": "✅",
            }
        )
    except Exception as e:
        results.append(
            {
                "model": "Black-Scholes",
                "price": 0,
                "time_ms": 0,
                "std_ms": 0,
                "status": f"❌ {e}",
            }
        )

    # Monte Carlo Basic
    try:
        from src.pricing_models import MonteCarloPricer

        pricer = MonteCarloPricer(num_simulations=num_sims, use_numba=use_numba)
        price, avg_time, std_time = benchmark_function(
            pricer.price, S, K, T, r, sigma, option_type, n_trials=3
        )
        label = f"MC Basic {'(Numba)' if use_numba else '(NumPy)'}"
        results.append(
            {
                "model": label,
                "price": price,
                "time_ms": avg_time,
                "std_ms": std_time,
                "status": "✅",
            }
        )
    except Exception as e:
        results.append(
            {
                "model": "MC Basic",
                "price": 0,
                "time_ms": 0,
                "std_ms": 0,
                "status": f"❌ {e}",
            }
        )

    # Monte Carlo Unified
    try:
        from src.pricing_models import MonteCarloPricerUni

        pricer = MonteCarloPricerUni(num_simulations=num_sims, use_numba=use_numba)
        price, avg_time, std_time = benchmark_function(
            pricer.price, S, K, T, r, sigma, option_type, n_trials=3
        )
        label = f"MC Unified {'(Numba)' if use_numba else '(NumPy)'}"
        results.append(
            {
                "model": label,
                "price": price,
                "time_ms": avg_time,
                "std_ms": std_time,
                "status": "✅",
            }
        )
    except Exception as e:
        results.append(
            {
                "model": "MC Unified",
                "price": 0,
                "time_ms": 0,
                "std_ms": 0,
                "status": f"❌ {e}",
            }
        )

    # Binomial Tree
    try:
        from src.pricing_models import BinomialTree

        tree = BinomialTree(num_steps=500)
        price, avg_time, std_time = benchmark_function(
            tree.price, S, K, T, r, sigma, option_type, "european", n_trials=5
        )
        results.append(
            {
                "model": "Binomial Tree (500 steps)",
                "price": price,
                "time_ms": avg_time,
                "std_ms": std_time,
                "status": "✅",
            }
        )
    except Exception as e:
        results.append(
            {
                "model": "Binomial Tree",
                "price": 0,
                "time_ms": 0,
                "std_ms": 0,
                "status": f"❌ {e}",
            }
        )

    # ML Surrogate (if trained)
    try:
        from src.pricing_models import MonteCarloMLSurrogate

        surrogate = MonteCarloMLSurrogate()

        # Quick training
        t_train_start = time.perf_counter()
        surrogate.fit(n_samples=1000, option_type=option_type, verbose=False)
        t_train = (time.perf_counter() - t_train_start) * 1000

        # Prediction speed
        def predict_wrapper():
            return surrogate.predict_single(S, K, T, r, sigma, 0.0)["price"]

        price, avg_time, std_time = benchmark_function(predict_wrapper, n_trials=20)

        results.append(
            {
                "model": f"ML Surrogate (train: {t_train:.0f}ms)",
                "price": price,
                "time_ms": avg_time,
                "std_ms": std_time,
                "status": "✅",
            }
        )
    except Exception as e:
        results.append(
            {
                "model": "ML Surrogate",
                "price": 0,
                "time_ms": 0,
                "std_ms": 0,
                "status": f"❌ {e}",
            }
        )

    return results


# =============================================================================
# HEADER
# =============================================================================
page_header("Performance Benchmarks", "Compare pricing engine speed and accuracy")

# Status indicators
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f"**Numba JIT:** {'✅ Available' if NUMBA_AVAILABLE else '❌ Not installed'}"
    )
with col2:
    st.markdown(
        f"**GPU (CuPy):** {'✅ Available' if GPU_AVAILABLE else '❌ Not installed'}"
    )
with col3:
    st.markdown(
        f"**LightGBM:** {'✅ Available' if LIGHTGBM_AVAILABLE else '❌ Using sklearn'}"
    )

section_divider()

# =============================================================================
# INPUT SECTION
# =============================================================================
st.markdown('<div class="input-section">', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1])

with col1:
    st.markdown("**Test Option**")
    S = st.number_input("Spot Price ($)", value=100.0, step=1.0, key="bm_spot")
    K = st.number_input("Strike Price ($)", value=100.0, step=1.0, key="bm_strike")
    T = st.number_input("Maturity (years)", value=1.0, step=0.1, key="bm_time")

with col2:
    st.markdown("**Market Parameters**")
    r_pct = st.number_input("Rate (%)", value=5.0, step=0.5, key="bm_rate")
    sigma_pct = st.number_input("Volatility (%)", value=20.0, step=1.0, key="bm_vol")
    option_type = st.selectbox("Option Type", ["call", "put"], key="bm_type")
    r, sigma = r_pct / 100, sigma_pct / 100

with col3:
    st.markdown("**Simulation Settings**")
    num_sims = st.select_slider(
        "MC Simulations",
        options=[10000, 25000, 50000, 100000],
        value=50000,
        key="bm_sims",
    )
    use_numba = st.checkbox(
        "Enable Numba",
        value=NUMBA_AVAILABLE,
        disabled=not NUMBA_AVAILABLE,
        key="bm_numba",
    )

with col4:
    st.markdown("**Run**")
    st.write("")
    run_basic = st.button("⏱️ Run Benchmarks", type="primary", use_container_width=True)
    run_scaling = st.button("📈 Scaling Test", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# BASIC BENCHMARKS
# =============================================================================
if run_basic:
    if not MODELS_AVAILABLE:
        st.error("Pricing models not available. Check installation.")
        st.stop()

    with st.spinner("Running benchmarks..."):
        t_total_start = time.perf_counter()
        results = run_pricing_benchmarks(
            S, K, T, r, sigma, option_type, num_sims, use_numba
        )
        t_total = (time.perf_counter() - t_total_start) * 1000

    section_divider()

    # Results table
    df = pd.DataFrame(results)

    # Display metrics
    n_models = len([r for r in results if r["status"] == "✅"])
    fastest = min(
        [r["time_ms"] for r in results if r["status"] == "✅" and r["time_ms"] > 0]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Models Tested</div>
            <div class="metric-value">{n_models}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Fastest Time</div>
            <div class="metric-value">{format_time_ms(fastest)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Total Benchmark Time</div>
            <div class="metric-value">{format_time_ms(t_total)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    section_divider()

    # Results chart
    tab1, tab2 = st.tabs(["📊 Comparison", "📋 Details"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Speed comparison
            valid_results = [
                r for r in results if r["status"] == "✅" and r["time_ms"] > 0
            ]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=[r["model"] for r in valid_results],
                    y=[r["time_ms"] for r in valid_results],
                    error_y=dict(
                        type="data", array=[r["std_ms"] for r in valid_results]
                    ),
                    marker_color=[
                        "#3b82f6",
                        "#60a5fa",
                        "#8b5cf6",
                        "#a78bfa",
                        "#10b981",
                    ],
                )
            )

            fig.update_layout(**get_chart_layout("Execution Time (ms)", 400))
            fig.update_layout(yaxis_type="log")
            fig.update_xaxes(title_text="Model", tickangle=-30)
            fig.update_yaxes(title_text="Time (ms, log scale)")

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Price comparison
            bs_price = next(
                (r["price"] for r in results if "Black-Scholes" in r["model"]), None
            )

            if bs_price:
                fig = go.Figure()

                for r in valid_results:
                    error = abs(r["price"] - bs_price) if bs_price else 0
                    fig.add_trace(
                        go.Bar(
                            x=[r["model"]],
                            y=[r["price"]],
                            name=r["model"],
                            text=[f"${r['price']:.4f}<br>Err: ${error:.4f}"],
                            textposition="outside",
                        )
                    )

                fig.add_hline(
                    y=bs_price,
                    line_dash="dash",
                    line_color="#10b981",
                    annotation_text=f"BS: ${bs_price:.4f}",
                )

                fig.update_layout(**get_chart_layout("Price Comparison", 400))
                fig.update_layout(showlegend=False)
                fig.update_xaxes(title_text="Model", tickangle=-30)
                fig.update_yaxes(title_text="Price ($)")

                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Detailed results table
        st.dataframe(
            df[["model", "price", "time_ms", "std_ms", "status"]].rename(
                columns={
                    "model": "Model",
                    "price": "Price ($)",
                    "time_ms": "Avg Time (ms)",
                    "std_ms": "Std Dev (ms)",
                    "status": "Status",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

# =============================================================================
# SCALING BENCHMARK
# =============================================================================
if run_scaling:
    if not MODELS_AVAILABLE:
        st.error("Pricing models not available.")
        st.stop()

    section_divider()
    st.markdown("### Scaling Analysis")

    with st.spinner("Running scaling tests..."):
        sim_counts = [1000, 5000, 10000, 25000, 50000, 100000]
        mc_times_numpy = []
        mc_times_numba = []

        for n in sim_counts:
            # NumPy
            try:
                from src.pricing_models import MonteCarloPricer

                pricer = MonteCarloPricer(num_simulations=n, use_numba=False)
                _, t, _ = benchmark_function(
                    pricer.price, S, K, T, r, sigma, option_type, n_trials=2
                )
                mc_times_numpy.append(t)
            except Exception:
                mc_times_numpy.append(None)

            # Numba
            if NUMBA_AVAILABLE:
                try:
                    pricer = MonteCarloPricer(num_simulations=n, use_numba=True)
                    _, t, _ = benchmark_function(
                        pricer.price, S, K, T, r, sigma, option_type, n_trials=2
                    )
                    mc_times_numba.append(t)
                except Exception:
                    mc_times_numba.append(None)

    # Plot scaling
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=sim_counts,
            y=mc_times_numpy,
            mode="lines+markers",
            name="NumPy",
            line=dict(width=3, color="#60a5fa"),
            marker=dict(size=10),
        )
    )

    if mc_times_numba:
        fig.add_trace(
            go.Scatter(
                x=sim_counts,
                y=mc_times_numba,
                mode="lines+markers",
                name="Numba JIT",
                line=dict(width=3, color="#10b981"),
                marker=dict(size=10),
            )
        )

    fig.update_layout(**get_chart_layout("MC Pricing Time vs Simulation Count", 400))
    fig.update_xaxes(title_text="Number of Simulations", type="log")
    fig.update_yaxes(title_text="Time (ms)")

    st.plotly_chart(fig, use_container_width=True)

    # Speedup calculation
    if mc_times_numba and all(t is not None for t in mc_times_numba):
        speedups = [
            np_t / nb_t for np_t, nb_t in zip(mc_times_numpy, mc_times_numba) if nb_t
        ]
        avg_speedup = np.mean(speedups)

        st.markdown(
            f"""
        <div class="metric-card" style="display: inline-block; padding: 1rem 2rem;">
            <span style="color: #94a3b8;">Average Numba Speedup:</span>
            <span style="color: #10b981; font-weight: 700; font-size: 1.5rem; margin-left: 0.5rem;">{avg_speedup:.1f}x</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

# Help text
if not run_basic and not run_scaling:
    st.info(
        "👆 Configure parameters and run benchmarks to compare pricing engine performance."
    )

    st.markdown(
        """
    <div class="metric-card">
        <h3 style="color: #60a5fa; margin-bottom: 1rem;">Benchmark Guide</h3>
        <ul style="color: #cbd5e1; line-height: 2;">
            <li><strong>Black-Scholes:</strong> Closed-form, microseconds</li>
            <li><strong>Monte Carlo:</strong> Simulation-based, scales linearly with sim count</li>
            <li><strong>Binomial Tree:</strong> Lattice method, O(n²) in steps</li>
            <li><strong>ML Surrogate:</strong> Trained model, instant inference</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )
