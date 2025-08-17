"""
Volatility Surface Dashboard
Upload raw option points (strike, maturity, iv), build an interpolated surface,
and visualize it in 3D with basic arbitrage checks.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from utils import build_surface, check_butterfly_arbitrage

st.set_page_config(page_title="Volatility Surface", page_icon="🗺️", layout="wide")
st.title("🗺️ Volatility Surface")

with st.sidebar:
    st.subheader("Surface Settings")
    method = st.selectbox("Interpolation", ["linear", "cubic", "nearest"], index=1)
    extrapolate = st.checkbox("Allow Extrapolation", value=False)
    strike_points = st.slider("Strike Grid Points", 10, 200, 60, step=5)
    maturity_points = st.slider("Maturity Grid Points", 10, 200, 60, step=5)

st.subheader("Data")
st.caption("Upload CSV with columns: `strike, maturity, iv` (maturity in years, iv in decimal).")
upl = st.file_uploader("Upload option points", type=["csv"])

if upl is None:
    st.info("No data uploaded. Showing a synthetic demo surface.")
    strikes = np.linspace(80, 120, 60)
    maturities = np.linspace(0.1, 2.0, 60)
    # Synthetic smile: higher IV away from ATM, upward term structure
    S0 = 100
    mny = (strikes - S0)/S0
    iv = 0.18 + 0.25*(mny**2) + 0.05*maturities
    df = pd.DataFrame({"strike": strikes.repeat(len(maturities)),
                       "maturity": np.tile(maturities, len(strikes)),
                       "iv": np.tile(iv, len(maturities))})
else:
    df = pd.read_csv(upl).dropna()
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"strike", "maturity", "iv"}.issubset(df.columns):
        st.error("CSV must have columns: strike, maturity, iv")
        st.stop()

try:
    res = build_surface(
        strikes=df["strike"].values.astype(float),
        maturities=df["maturity"].values.astype(float),
        ivs=df["iv"].values.astype(float),
        strike_points=strike_points,
        maturity_points=maturity_points,
        method=method,
        extrapolate=extrapolate,
        benchmark=True
    )
except Exception as e:
    st.error(f"Surface generation failed: {e}")
    st.stop()

# 3D Plot
surf = go.Surface(
    x=res.strikes,
    y=res.maturities,
    z=res.iv_grid,
    colorbar=dict(title="IV"),
    showscale=True
)
layout = go.Layout(
    title="Interpolated Implied Volatility Surface",
    scene=dict(
        xaxis_title="Strike",
        yaxis_title="Maturity (years)",
        zaxis_title="Implied Volatility"
    ),
    height=650
)
fig = go.Figure(data=[surf], layout=layout)
st.plotly_chart(fig, use_container_width=True)

# Arbitrage checks (if available)
if check_butterfly_arbitrage is not None:
    try:
        arb_summary = check_butterfly_arbitrage(
            strikes=res.strikes[0, :],  # strikes grid (row 0)
            iv_row=res.iv_grid[res.iv_grid.shape[0]//2, :]  # mid maturity row
        )
        st.subheader("Arbitrage Check (Butterfly, mid-maturity slice)")
        st.json(arb_summary)
    except Exception as e:
        st.info(f"Arbitrage check skipped: {e}")
