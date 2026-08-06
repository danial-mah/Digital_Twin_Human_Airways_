"""
pages/3_DOE_Analysis.py
========================
Design of Experiments (DOE) parameter space exploration.

Tabs
----
  Parallel Coordinates — visualise all 26 parameters at once; colour by snapshot
  Pairwise Scatter     — select any two parameters for a scatter + regression line
  Correlation Heatmap  — 26×26 Pearson correlation matrix
  LHS Coverage         — compare original DOE vs Latin Hypercube upscaling (1000 pts)

Why is this useful?
-------------------
The DOE was generated via Latin Hypercube Sampling to fill the 26-dimensional
parameter space uniformly. Parallel coordinates reveal correlations and clusters.
The LHS coverage plot shows how 1000 virtual shapes improve coverage compared
to the original 100 CFD runs.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_io import load_doe, get_param_cols, PARAM_LABELS
from core.lhs import latin_hypercube, doe_bounds, discrepancy_score
from core.i18n import t, lang_selector

st.set_page_config(page_title="DOE Analysis", page_icon="📐", layout="wide")
lang_selector()
st.title(t("doe_title"))
st.caption(t("doe_caption"))

# ── Data ───────────────────────────────────────────────────────────────────────
doe_df     = load_doe()
param_cols = get_param_cols(doe_df)
labels     = [PARAM_LABELS.get(c, c) for c in param_cols]

# Sidebar: highlight a specific run
with st.sidebar:
    st.header(t("sidebar_controls"))
    highlight = st.slider(t("doe_highlight"), 1, 100, 1)

tab1, tab2, tab3, tab4 = st.tabs([
    t("doe_tab_parallel"),
    t("doe_tab_scatter"),
    t("doe_tab_corr"),
    t("doe_tab_lhs"),
])

# ── Tab 1: Parallel coordinates ────────────────────────────────────────────────
with tab1:
    st.subheader(t("doe_sub_parallel"))
    st.caption(
        "Each line is one simulation run. Colour encodes the snapshot number. "
        "Drag the axes to reorder; click an axis range to filter."
    )

    pc_df = doe_df[param_cols + ["snapshot_num"]].copy()
    pc_df.columns = labels + ["Run"]

    fig_pc = px.parallel_coordinates(
        pc_df,
        dimensions=labels,
        color="Run",
        color_continuous_scale="Viridis",
    )
    fig_pc.update_layout(height=520, margin=dict(l=60, r=40, t=20, b=40))
    st.plotly_chart(fig_pc, width='stretch')

    # Per-parameter range table
    st.subheader(t("doe_sub_ranges"))
    range_df = (
        doe_df[param_cols]
        .agg(["min", "max", "mean", "std"])
        .T.round(3)
        .rename_axis("Parameter")
        .reset_index()
    )
    range_df["Label"] = range_df["Parameter"].map(PARAM_LABELS)
    st.dataframe(
        range_df[["Label", "min", "max", "mean", "std"]],
        hide_index=True,
        width='stretch',
    )

# ── Tab 2: Pairwise scatter ────────────────────────────────────────────────────
with tab2:
    st.subheader(t("doe_sub_pairwise"))
    c1, c2 = st.columns(2)
    xp = c1.selectbox(t("lbl_x_axis"), param_cols, index=0,
                       format_func=lambda c: PARAM_LABELS.get(c, c))
    yp = c2.selectbox(t("lbl_y_axis"), param_cols, index=3,
                       format_func=lambda c: PARAM_LABELS.get(c, c))

    fig_sc = px.scatter(
        doe_df, x=xp, y=yp,
        color="snapshot_num",
        color_continuous_scale="Viridis",
        hover_data={"snapshot_num": True},
        labels={
            xp: PARAM_LABELS.get(xp, xp),
            yp: PARAM_LABELS.get(yp, yp),
            "snapshot_num": "Run",
        },
        title=f"{PARAM_LABELS.get(xp, xp)}  vs  {PARAM_LABELS.get(yp, yp)}",
    )

    # Manual OLS trendline (no statsmodels needed)
    xd = doe_df[xp].values.astype(float)
    yd = doe_df[yp].values.astype(float)
    m, b = np.polyfit(xd, yd, 1)
    x_line = np.linspace(xd.min(), xd.max(), 50)
    fig_sc.add_scatter(x=x_line, y=m * x_line + b,
                       mode="lines", line=dict(color="white", dash="dash"),
                       name="OLS trend", showlegend=True)

    # Star marker for highlighted run
    h_row = doe_df[doe_df["snapshot_num"] == highlight].iloc[0]
    fig_sc.add_trace(
        go.Scatter(
            x=[h_row[xp]], y=[h_row[yp]],
            mode="markers",
            marker=dict(size=18, color="red", symbol="star",
                        line=dict(width=2, color="white")),
            name=f"Run {highlight}",
        )
    )
    fig_sc.update_layout(height=520)
    st.plotly_chart(fig_sc, width='stretch')

    # Pearson correlation between selected pair
    corr_val = doe_df[[xp, yp]].corr().iloc[0, 1]
    st.metric(
        f"Pearson r  ({PARAM_LABELS.get(xp, xp)} vs {PARAM_LABELS.get(yp, yp)})",
        f"{corr_val:.4f}",
    )

# ── Tab 3: Correlation heatmap ─────────────────────────────────────────────────
with tab3:
    st.subheader(t("doe_sub_corr"))
    st.caption(
        "Values near ±1 indicate strong linear dependence between parameters. "
        "The DOE was designed to minimise correlation, so most values should be ~0."
    )

    corr_matrix = doe_df[param_cols].corr()
    corr_matrix.index   = labels
    corr_matrix.columns = labels

    fig_hm = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        text_auto=".2f",
    )
    fig_hm.update_layout(
        height=700,
        xaxis=dict(tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )
    st.plotly_chart(fig_hm, width='stretch')

# ── Tab 4: LHS coverage ────────────────────────────────────────────────────────
with tab4:
    st.subheader(t("doe_sub_lhs"))
    st.markdown("""
    Latin Hypercube Sampling (LHS) generates 1000 new parameter combinations
    within the original bounds. These virtual shapes are evaluated by the RBF
    surrogate (see **RBF Inference** page) instead of running expensive CFD.
    """)

    @st.cache_data(show_spinner=False)
    def get_lhs_samples():
        low, high = doe_bounds(doe_df, param_cols)
        return latin_hypercube(1000, low, high, seed=42)

    with st.spinner("Generating 1000 LHS samples…"):
        lhs_samples = get_lhs_samples()

    low, high = doe_bounds(doe_df, param_cols)

    c1, c2 = st.columns(2)
    xp2 = c1.selectbox("X parameter (LHS view)", param_cols, index=0,
                        format_func=lambda c: PARAM_LABELS.get(c, c), key="lhs_x")
    yp2 = c2.selectbox("Y parameter (LHS view)", param_cols, index=3,
                        format_func=lambda c: PARAM_LABELS.get(c, c), key="lhs_y")

    xi = param_cols.index(xp2)
    yi = param_cols.index(yp2)

    fig_lhs = go.Figure()
    fig_lhs.add_trace(go.Scatter(
        x=lhs_samples[:, xi], y=lhs_samples[:, yi],
        mode="markers",
        marker=dict(size=4, color="#aec7e8", opacity=0.5),
        name="LHS (1000 virtual)",
    ))
    fig_lhs.add_trace(go.Scatter(
        x=doe_df[xp2].values, y=doe_df[yp2].values,
        mode="markers",
        marker=dict(size=9, color="red", symbol="circle-open", line=dict(width=2)),
        name="Original DOE (100)",
    ))
    fig_lhs.update_layout(
        xaxis_title=PARAM_LABELS.get(xp2, xp2),
        yaxis_title=PARAM_LABELS.get(yp2, yp2),
        title="Parameter coverage: LHS (blue) vs original DOE (red)",
        height=500,
    )
    st.plotly_chart(fig_lhs, width='stretch')

    # Discrepancy metrics
    d_doe = discrepancy_score(doe_df[param_cols].values)
    d_lhs = discrepancy_score(lhs_samples)
    
    mc1, mc2 = st.columns(2)
    mc1.metric(t("doe_discrepancy"), f"{d_doe:.5f}",
               help="Lower = more uniform coverage")
    mc2.metric(t("doe_lhs_discr"), f"{d_lhs:.5f}")
