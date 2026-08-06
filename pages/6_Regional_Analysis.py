"""
pages/6_Regional_Analysis.py
=============================
Mean static pressure per anatomical region of the human airway.

The mesh is divided into 35+ named selections (epiglottis, glottis,
larynx, trachea sections, left/right bronchial branches at three levels).
This page computes statistics per region for a selected snapshot, and
compares mean pressure across all 100 snapshots per region.

Tabs
----
  Per-Snapshot View   — bar chart of mean ± std per region for a chosen run
  Cross-Snapshot View — heatmap: 100 runs × regions, showing pressure variation
  Export              — download CSV of regional statistics
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_io import (
    STRIDE,
    load_pressure_snapshot,
    load_settings,
    load_doe,
    get_anatomical_regions,
    region_slice,
)
from core.i18n import t, lang_selector

st.set_page_config(page_title="Regional Analysis", page_icon="🫀", layout="wide")
lang_selector()
st.title(t("reg_title"))
st.caption(t("reg_caption"))

# ── Load static data ───────────────────────────────────────────────────────────
settings = load_settings()
doe_df   = load_doe()
regions  = get_anatomical_regions(settings)

with st.sidebar:
    st.header(t("sidebar_controls"))
    snap_sel = st.slider(t("lbl_snapshot"), 1, 100, 1)
    sort_by  = st.radio(t("reg_sort_by"), [t("reg_sort_mean"), t("reg_sort_anat")], index=0)

# ── Compute per-region stats for selected snapshot ─────────────────────────────
@st.cache_data(show_spinner=False)
def regional_stats(snap: int) -> pd.DataFrame:
    pressure = load_pressure_snapshot(snap, STRIDE)
    rows = []
    for reg in regions:
        sl   = region_slice(settings, reg, STRIDE)
        p_r  = pressure[sl]
        if len(p_r) == 0:
            continue
        rows.append({
            "region":   reg,
            "mean_Pa":  float(p_r.mean()),
            "std_Pa":   float(p_r.std()),
            "min_Pa":   float(p_r.min()),
            "max_Pa":   float(p_r.max()),
            "n_nodes":  int(len(p_r)),
        })
    return pd.DataFrame(rows)

# ── Compute mean pressure per region for ALL snapshots (for heatmap) ──────────
@st.cache_data(show_spinner=False)
def all_regional_means() -> pd.DataFrame:
    """Returns DataFrame shape (100, n_regions) of mean pressures."""
    data = {}
    for reg in regions:
        data[reg] = []
    for i in range(1, 101):
        pressure = load_pressure_snapshot(i, STRIDE)
        for reg in regions:
            sl  = region_slice(settings, reg, STRIDE)
            p_r = pressure[sl]
            data[reg].append(float(p_r.mean()) if len(p_r) > 0 else np.nan)
    return pd.DataFrame(data, index=range(1, 101))

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_snap, tab_cross, tab_export = st.tabs([
    t("reg_tab_snap"),
    t("reg_tab_heat"),
    t("reg_tab_export"),
])

# ─ Tab 1: Per-snapshot ─────────────────────────────────────────────────────────
with tab_snap:
    with st.spinner(f"Computing regional stats for snapshot {snap_sel}…"):
        reg_df = regional_stats(snap_sel)

    if sort_by == t("reg_sort_mean"):
        reg_df = reg_df.sort_values("mean_Pa", ascending=False).reset_index(drop=True)

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        fig_bar = px.bar(
            reg_df,
            x="region", y="mean_Pa",
            error_y="std_Pa",
            color="mean_Pa",
            color_continuous_scale="RdYlBu_r",
            labels={"mean_Pa": "Mean Pressure (Pa)", "region": "Anatomical Region"},
            title=f"Mean static pressure by region — Snapshot {snap_sel}",
        )
        fig_bar.update_xaxes(tickangle=50, tickfont=dict(size=9))
        fig_bar.update_layout(
            height=520,
            coloraxis_showscale=True,
            coloraxis_colorbar_title="Mean P (Pa)",
        )
        st.plotly_chart(fig_bar, width='stretch')

    with col_table:
        display_df = reg_df[["region", "mean_Pa", "std_Pa", "min_Pa", "max_Pa", "n_nodes"]].copy()
        display_df.columns = ["Region", "Mean (Pa)", "Std (Pa)", "Min (Pa)", "Max (Pa)", "Nodes"]
        st.dataframe(
            display_df.round(2),
            hide_index=True,
            height=520,
            width='stretch',
        )

    # Radial plot — useful for comparing regions on a circular axis
    st.subheader(t("reg_sub_radar"))
    top_regions = reg_df.head(12)
    fig_radar = go.Figure(go.Scatterpolar(
        r=top_regions["mean_Pa"].tolist() + [top_regions["mean_Pa"].iloc[0]],
        theta=top_regions["region"].tolist() + [top_regions["region"].iloc[0]],
        fill="toself",
        fillcolor="rgba(78,179,211,0.3)",
        line=dict(color="#4EB3D3"),
        name=f"Snapshot {snap_sel}",
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        height=450,
        title="Mean pressure per region (top 12)",
    )
    st.plotly_chart(fig_radar, width='stretch')

# ─ Tab 2: Cross-snapshot heatmap ───────────────────────────────────────────────
with tab_cross:
    st.subheader(t("reg_sub_heatmap"))
    st.caption(
        "Each cell shows the mean pressure (Pa) for one DOE run in one region. "
        "Rows are snapshots (runs); columns are anatomical regions."
    )

    with st.spinner("Loading all 100 snapshots for regional heatmap…"):
        heatmap_df = all_regional_means()

    fig_hm = px.imshow(
        heatmap_df.T,                          # regions on y-axis, runs on x-axis
        color_continuous_scale="Jet",
        aspect="auto",
        labels={"x": "Snapshot (DOE Run)", "y": "Anatomical Region", "color": "Mean P (Pa)"},
        title="Mean pressure heatmap — all snapshots × all regions",
    )
    fig_hm.update_layout(
        height=700,
        xaxis=dict(tickmode="linear", dtick=5),
        yaxis=dict(tickfont=dict(size=9)),
    )
    st.plotly_chart(fig_hm, width='stretch')

    # Most-variable regions
    st.subheader(t("reg_sub_variable"))
    variability = heatmap_df.std().sort_values(ascending=False)
    fig_var = px.bar(
        x=variability.index.tolist(),
        y=variability.values.tolist(),
        labels={"x": "Region", "y": "Std Dev of Mean Pressure (Pa)"},
        title="Standard deviation of mean pressure across all runs",
        color=variability.values.tolist(),
        color_continuous_scale="Reds",
    )
    fig_var.update_xaxes(tickangle=50, tickfont=dict(size=9))
    fig_var.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig_var, width='stretch')

# ─ Tab 3: Export ───────────────────────────────────────────────────────────────
with tab_export:
    st.subheader(t("reg_sub_export"))

    export_type = st.radio(t("reg_tab_export"), [
        t("reg_export_sel"),
        t("reg_export_all"),
    ])

    if export_type == t("reg_export_sel"):
        csv_bytes = regional_stats(snap_sel).to_csv(index=False).encode()
        st.download_button(
            t("reg_dl_sel"),
            csv_bytes,
            file_name=f"regional_stats_snapshot{snap_sel}.csv",
            mime="text/csv",
        )
    else:
        with st.spinner("Computing all regional means for export…"):
            export_df = all_regional_means()
        export_df.index.name = "snapshot"
        csv_bytes = export_df.reset_index().to_csv(index=False).encode()
        st.download_button(
            t("reg_dl_all"),
            csv_bytes,
            file_name="all_snapshots_regional_means.csv",
            mime="text/csv",
        )
