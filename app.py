from __future__ import annotations

import pandas as pd
import streamlit as st

from model import simulate_ramsey

st.set_page_config(
    page_title="Two-Household Ramsey Dashboard",
    layout="wide",
)

st.title("Two-Household Ramsey Equilibrium Dashboard")
st.caption(
    "Interactive dashboard for the log-utility, Cobb-Douglas two-household Ramsey model. "
    "The app uses the backward-shooting construction to select the active block and then computes the path."
)

with st.sidebar:
    st.header("Inputs")
    alpha = st.number_input("α in (0, 1)", min_value=0.0001, max_value=0.9999, value=0.35, step=0.01, format="%.4f")
    delta1 = st.number_input("δ₁ in (0, 1)", min_value=0.0001, max_value=0.9999, value=0.95, step=0.01, format="%.4f")
    delta2 = st.number_input("δ₂ in (0, δ₁)", min_value=0.0001, max_value=0.9998, value=0.50, step=0.01, format="%.4f")
    x10 = st.number_input("x¹₀", min_value=0.0, value=0.80, step=0.05, format="%.6f")
    x20 = st.number_input("x²₀", min_value=0.0, value=0.20, step=0.05, format="%.6f")
    horizon = st.slider("Number of periods shown", min_value=5, max_value=200, value=50, step=1)

    with st.expander("Advanced"):
        max_blocks = st.number_input(
            "Maximum active-block length searched",
            min_value=10,
            max_value=10000,
            value=1000,
            step=100,
        )

errors = []
if not (0 < alpha < 1):
    errors.append("α must be in (0, 1).")
if not (0 < delta1 < 1):
    errors.append("δ₁ must be in (0, 1).")
if not (0 < delta2 < delta1):
    errors.append("δ₂ must be in (0, δ₁).")
if x10 + x20 <= 0:
    errors.append("At least one of x¹₀ and x²₀ must be positive.")

if x10 + x20 > 1:
    st.warning(
        "The TeX file assumes K₀ = x¹₀ + x²₀ ≤ 1, where the maximum sustainable stock is 1. "
        "The app will still compute the formula-based path, but this input is outside the paper's maintained assumption."
    )

if errors:
    for error in errors:
        st.error(error)
    st.stop()

try:
    rows, info = simulate_ramsey(
        alpha=alpha,
        delta1=delta1,
        delta2=delta2,
        x10=x10,
        x20=x20,
        horizon=horizon,
        max_blocks=int(max_blocks),
    )
except Exception as exc:
    st.error(str(exc))
    st.stop()

df = pd.DataFrame(rows)

col1, col2, col3, col4 = st.columns(4)
col1.metric("K₀", f"{info.K0:.6g}")
col2.metric("Initial share x²₀/K₀", f"{info.initial_share_2:.6g}")
if info.block_length == 0:
    col3.metric("Active-block length", "0")
    col4.metric("Final-exit ratio z", "not used")
else:
    col3.metric("Active-block length", str(info.block_length))
    col4.metric("Final-exit ratio z", f"{info.final_exit_ratio_z:.8g}")

st.subheader("Capital paths")
chart_df = df.set_index("t")[["K_t", "x1_t", "x2_t"]]
st.line_chart(chart_df)

st.subheader("Computed sequence")
st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download results as CSV",
    data=csv,
    file_name="ramsey_equilibrium_path.csv",
    mime="text/csv",
)

with st.expander("Model details used by this app"):
    st.markdown(
        r"""
The app implements the normalized backward-shooting formulas from the TeX file.

Let

$$L=\frac{1-\alpha}{2}, \qquad B=\frac{1+\alpha}{2}, \qquad
\bar z=\frac{\delta_1(1-\alpha)}{\delta_1(1-\alpha)+\delta_2(1+\alpha)}.$$

For a positive initial impatient share, the app finds the unique block length $n$ and final-exit ratio
$z \in (L,\bar z]$ such that $Q_n(z)=x_0^2/(x_0^1+x_0^2)$.
Then it converts the normalized ratios into levels using

$$K_t = \sigma_t K_{t-1}^{\alpha}.$$

After household 2 exits, the patient-household continuation is

$$K_t=\alpha\delta_1 K_{t-1}^{\alpha},\qquad x_t^1=K_t,\qquad x_t^2=0.$$
        """
    )
