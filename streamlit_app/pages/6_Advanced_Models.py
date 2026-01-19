# streamlit_app/pages/6_Advanced_Models.py
"""
Advanced Pricing Models - Streamlit Page.

Compare Heston, SABR, Jump-Diffusion, and Finite Difference models.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Advanced Models",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    from components import (
        apply_custom_css,
        format_greek,
        format_price,
        format_time_ms,
        get_chart_layout,
        page_header,
        section_divider,
    )
except ImportError:
    from streamlit_app.components import (
        apply_custom_css,
        format_greek,
        format_price,
        format_time_ms,
        get_chart_layout,
        page_header,
        section_divider,
    )

# Import models
try:
    from src.greeks import (
        FDMAdapter,
        HestonAdapter,
        JumpDiffusionAdapter,
        compute_greeks_unified,
    )
    from src.pricing_models import (
        CrankNicolsonSolver,
        HestonPricer,
        MertonJumpDiffusion,
        SABRModel,
        black_scholes,
    )

    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    st.error(f"Models not available: {e}")

apply_custom_css()

# =============================================================================
# HEADER
# =============================================================================
page_header(
    "Advanced Pricing Models",
    "Compare Heston, SABR, Jump-Diffusion, and Finite Difference methods",
)

section_divider()

# =============================================================================
# MODEL SELECTOR
# =============================================================================
st.markdown('<div class="input-section">', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns([1.2, 1, 1, 1.5, 0.8])

with col1:
    st.markdown("**Model Selection**")
    model_choice = st.selectbox(
        "Pricing Model",
        ["Heston", "SABR", "Merton Jump-Diffusion", "Crank-Nicolson FDM"],
        key="model_choice",
    )
    option_type = st.selectbox("Option Type", ["call", "put"], key="adv_type")

with col2:
    st.markdown("**Asset Parameters**")
    S = st.number_input(
        "Spot Price ($)", min_value=1.0, max_value=500.0, value=100.0, step=1.0
    )
    K = st.number_input(
        "Strike Price ($)", min_value=1.0, max_value=500.0, value=100.0, step=1.0
    )

with col3:
    st.markdown("**Market**")
    T = st.number_input(
        "Maturity (years)", min_value=0.01, max_value=5.0, value=1.0, step=0.1
    )
    r_pct = st.number_input(
        "Rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5
    )
    r = r_pct / 100.0

with col4:
    st.markdown("**Model Parameters**")

    if model_choice == "Heston":
        col_a, col_b = st.columns(2)
        with col_a:
            v0 = st.number_input(
                "v₀ (Initial Var)",
                min_value=0.001,
                max_value=0.5,
                value=0.04,
                step=0.01,
                format="%.3f",
            )
            kappa = st.number_input(
                "κ (Mean Rev)", min_value=0.1, max_value=10.0, value=2.0, step=0.1
            )
            theta = st.number_input(
                "θ (Long Var)",
                min_value=0.001,
                max_value=0.5,
                value=0.04,
                step=0.01,
                format="%.3f",
            )
        with col_b:
            sigma_v = st.number_input(
                "σᵥ (Vol of Vol)", min_value=0.01, max_value=2.0, value=0.3, step=0.05
            )
            rho = st.number_input(
                "ρ (Correlation)",
                min_value=-0.99,
                max_value=0.99,
                value=-0.7,
                step=0.05,
            )
        sigma = np.sqrt(v0)  # Use initial vol for Greeks

    elif model_choice == "SABR":
        col_a, col_b = st.columns(2)
        with col_a:
            alpha = st.number_input(
                "α (Initial Vol)", min_value=0.01, max_value=1.0, value=0.25, step=0.01
            )
            beta = st.number_input(
                "β (CEV Exp)", min_value=0.0, max_value=1.0, value=0.5, step=0.1
            )
        with col_b:
            rho_sabr = st.number_input(
                "ρ (Correlation)",
                min_value=-0.99,
                max_value=0.99,
                value=-0.3,
                step=0.05,
            )
            nu = st.number_input(
                "ν (Vol of Vol)", min_value=0.01, max_value=2.0, value=0.4, step=0.05
            )
        sigma = alpha

    elif model_choice == "Merton Jump-Diffusion":
        col_a, col_b = st.columns(2)
        with col_a:
            sigma_pct = st.number_input(
                "σ (%)", min_value=1.0, max_value=100.0, value=20.0, step=1.0
            )
            lambda_j = st.number_input(
                "λ (Jump Rate)", min_value=0.01, max_value=2.0, value=0.1, step=0.05
            )
        with col_b:
            mu_j = st.number_input(
                "μⱼ (Jump Mean)", min_value=-0.5, max_value=0.5, value=-0.1, step=0.05
            )
            sigma_j = st.number_input(
                "σⱼ (Jump Vol)", min_value=0.01, max_value=0.5, value=0.15, step=0.05
            )
        sigma = sigma_pct / 100.0

    else:  # Crank-Nicolson FDM
        col_a, col_b = st.columns(2)
        with col_a:
            sigma_pct = st.number_input(
                "Volatility (%)", min_value=1.0, max_value=100.0, value=20.0, step=1.0
            )
            n_space = st.number_input(
                "Space Steps", min_value=50, max_value=500, value=200, step=50
            )
        with col_b:
            n_time = st.number_input(
                "Time Steps", min_value=20, max_value=200, value=100, step=20
            )
            exercise = st.selectbox("Exercise", ["european", "american"])
        sigma = sigma_pct / 100.0

with col5:
    st.markdown("**Run**")
    st.write("")
    run = st.button("🚀 Price", type="primary", use_container_width=True)
    run_compare = st.button("📊 Compare All", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# SINGLE MODEL PRICING
# =============================================================================
if run and MODELS_AVAILABLE:
    section_divider()

    with st.spinner(f"Running {model_choice}..."):
        t_start = time.perf_counter()

        if model_choice == "Heston":
            pricer = HestonPricer(
                kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho, v0=v0
            )
            price = pricer.price_european(S, K, T, r, 0.0, option_type)
            adapter = HestonAdapter(pricer)
            greeks = compute_greeks_unified(
                adapter, S, K, T, r, sigma, option_type, include_second_order=False
            )

        elif model_choice == "SABR":
            pricer = SABRModel(alpha=alpha, beta=beta, rho=rho_sabr, nu=nu)
            F = S * np.exp(r * T)
            price = pricer.price(F, K, T, r, option_type)
            iv = pricer.implied_vol(F, K, T)
            greeks = {
                "price": price,
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0,
            }
            greeks["implied_vol"] = iv

        elif model_choice == "Merton Jump-Diffusion":
            pricer = MertonJumpDiffusion(lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j)
            price = pricer.price(S, K, T, r, sigma, option_type)
            adapter = JumpDiffusionAdapter(pricer)
            greeks = compute_greeks_unified(
                adapter, S, K, T, r, sigma, option_type, include_second_order=False
            )

        else:  # Crank-Nicolson
            pricer = CrankNicolsonSolver(s_max=3 * S, n_space=n_space, n_time=n_time)
            price = pricer.price(S, K, T, r, sigma, option_type, exercise)
            adapter = FDMAdapter(pricer, exercise)
            greeks = compute_greeks_unified(
                adapter, S, K, T, r, sigma, option_type, include_second_order=False
            )

        t_total = (time.perf_counter() - t_start) * 1000

    # BS reference
    bs_price = black_scholes(S, K, T, r, sigma, option_type) if black_scholes else None

    # Display results
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">{model_choice} Price</div>
            <div class="metric-value">{format_price(price)}</div>
            <div class="metric-delta">{format_time_ms(t_total)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        if bs_price:
            diff = price - bs_price
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">BS Reference</div>
                <div class="metric-value">{format_price(bs_price)}</div>
                <div class="metric-delta">Diff: {diff:+.4f}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col3:
        delta = greeks.get("delta", 0)
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Delta (Δ)</div>
            <div class="metric-value">{format_greek(delta)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        gamma = greeks.get("gamma", 0)
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Gamma (Γ)</div>
            <div class="metric-value">{format_greek(gamma, 6)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col5:
        vega = greeks.get("vega", 0)
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Vega (ν)</div>
            <div class="metric-value">{format_greek(vega, 2)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col6:
        if model_choice == "SABR":
            iv = greeks.get("implied_vol", 0)
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">SABR IV</div>
                <div class="metric-value">{iv*100:.2f}%</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            theta_val = greeks.get("theta", 0)
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Theta (Θ)</div>
                <div class="metric-value">{format_greek(theta_val, 2)}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

# =============================================================================
# COMPARE ALL MODELS
# =============================================================================
if run_compare and MODELS_AVAILABLE:
    section_divider()
    st.markdown("### Model Comparison")

    with st.spinner("Computing all models..."):
        results = {}

        # Black-Scholes baseline
        t0 = time.perf_counter()
        bs_price = black_scholes(S, K, T, r, sigma, option_type)
        results["Black-Scholes"] = {
            "price": bs_price,
            "time_ms": (time.perf_counter() - t0) * 1000,
        }

        # Heston
        try:
            t0 = time.perf_counter()
            heston = HestonPricer(
                kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=sigma**2
            )
            h_price = heston.price_european(S, K, T, r, 0.0, option_type)
            results["Heston"] = {
                "price": h_price,
                "time_ms": (time.perf_counter() - t0) * 1000,
            }
        except Exception as e:
            results["Heston"] = {"price": None, "error": str(e)}

        # SABR
        try:
            t0 = time.perf_counter()
            sabr = SABRModel(alpha=sigma, beta=0.5, rho=-0.3, nu=0.4)
            F = S * np.exp(r * T)
            s_price = sabr.price(F, K, T, r, option_type)
            results["SABR"] = {
                "price": s_price,
                "time_ms": (time.perf_counter() - t0) * 1000,
            }
        except Exception as e:
            results["SABR"] = {"price": None, "error": str(e)}

        # Merton JD
        try:
            t0 = time.perf_counter()
            jd = MertonJumpDiffusion(lambda_j=0.1, mu_j=-0.1, sigma_j=0.15)
            jd_price = jd.price(S, K, T, r, sigma, option_type)
            results["Merton JD"] = {
                "price": jd_price,
                "time_ms": (time.perf_counter() - t0) * 1000,
            }
        except Exception as e:
            results["Merton JD"] = {"price": None, "error": str(e)}

        # FDM
        try:
            t0 = time.perf_counter()
            fdm = CrankNicolsonSolver(s_max=3 * S, n_space=200, n_time=100)
            fdm_price = fdm.price(S, K, T, r, sigma, option_type)
            results["Crank-Nicolson"] = {
                "price": fdm_price,
                "time_ms": (time.perf_counter() - t0) * 1000,
            }
        except Exception as e:
            results["Crank-Nicolson"] = {"price": None, "error": str(e)}

    # Display comparison
    cols = st.columns(len(results))
    for i, (name, data) in enumerate(results.items()):
        with cols[i]:
            if data.get("price") is not None:
                diff = data["price"] - bs_price if name != "Black-Scholes" else 0
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="metric-label">{name}</div>
                    <div class="metric-value">{format_price(data['price'])}</div>
                    <div class="metric-delta">{format_time_ms(data['time_ms'])} | Δ{diff:+.3f}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="metric-label">{name}</div>
                    <div class="metric-value" style="color: #ef4444;">Error</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    # Bar chart comparison
    section_divider()

    valid_results = {k: v for k, v in results.items() if v.get("price") is not None}

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(valid_results.keys()),
            y=[v["price"] for v in valid_results.values()],
            marker_color=["#60a5fa", "#a78bfa", "#34d399", "#f59e0b", "#ef4444"][
                : len(valid_results)
            ],
            text=[f"${v['price']:.2f}" for v in valid_results.values()],
            textposition="outside",
        )
    )

    fig.add_hline(
        y=bs_price,
        line_dash="dash",
        line_color="#94a3b8",
        annotation_text=f"BS: ${bs_price:.2f}",
    )

    fig.update_layout(**get_chart_layout("Model Price Comparison", 400))
    fig.update_yaxes(title_text="Option Price ($)")

    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# HELP SECTION
# =============================================================================
if not run and not run_compare:
    st.info(
        "👆 Select a model, configure parameters, and click **Price** or **Compare All**."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <h3 style="color: #60a5fa; margin-bottom: 1rem;">🔬 Available Models</h3>
            <ul style="color: #cbd5e1; line-height: 2;">
                <li><strong>Heston:</strong> Stochastic volatility with mean reversion</li>
                <li><strong>SABR:</strong> Volatility smile interpolation</li>
                <li><strong>Merton JD:</strong> Log-normal jumps + diffusion</li>
                <li><strong>Crank-Nicolson:</strong> PDE solver for American options</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <h3 style="color: #a78bfa; margin-bottom: 1rem;">📈 When to Use Each</h3>
            <ul style="color: #cbd5e1; line-height: 2;">
                <li><strong>Heston:</strong> Equity options with vol clustering</li>
                <li><strong>SABR:</strong> Interest rate / FX smiles</li>
                <li><strong>Merton JD:</strong> Crash risk / tail events</li>
                <li><strong>FDM:</strong> American exercise features</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )
