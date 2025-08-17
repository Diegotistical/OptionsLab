"""
Risk Analysis Dashboard
Load PnL/returns or generate synthetic series, then compute VaR/ES and show distribution.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import compute_var_es

st.set_page_config(page_title="Risk Analysis", page_icon="🛡️", layout="wide")
st.title("🛡️ Risk Analysis")

with st.sidebar:
    st.subheader("Data")
    mode = st.radio("Source", ["Upload CSV", "Generate Synthetic"])
    level = st.slider("Confidence Level", 0.80, 0.995, 0.95, step=0.005)
    st.caption("CSV should contain a single column of returns (daily).")

if mode == "Upload CSV":
    upl = st.file_uploader("Upload returns CSV", type=["csv"])
    if upl:
        df = pd.read_csv(upl)
        series = df.iloc[:, 0].astype(float).dropna()
    else:
        st.stop()
else:
    st.subheader("Synthetic Returns")
    n = st.slider("Length (days)", 100, 5000, 1000, step=100)
    mu = st.number_input("Mean (daily)", -0.01, 0.01, 0.0005, step=0.0001, format="%.5f")
    s = st.number_input("Std (daily)", 0.0001, 0.10, 0.02, step=0.0001, format="%.4f")
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(mu, s, size=n), name="returns")

st.subheader("Summary")
st.write(series.describe())

var_v, es_v = compute_var_es(series, level=level)
c1, c2 = st.columns(2)
c1.metric(f"Historical VaR ({int(level*100)}%)", f"{var_v:.4%}" if var_v is not None else "N/A")
c2.metric(f"Expected Shortfall ({int(level*100)}%)", f"{es_v:.4%}" if es_v is not None else "N/A")

fig, ax = plt.subplots(figsize=(8,4))
ax.hist(series, bins=50)
ax.set_title("Return Distribution")
ax.set_xlabel("Return")
ax.set_ylabel("Frequency")
st.pyplot(fig)
