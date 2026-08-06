"""
pages/2_Pressure_Field.py
=========================
Interactive 3D pressure field viewer.

Features
--------
- Select any of the 100 DOE snapshots via slider.
- Toggle between reference mesh and snapshot-specific deformed geometry.
- Filter view to a named anatomical region.
- Colour map selection and opacity control.
- Metrics: min / max / mean / std dev of pressure.
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_io import (
    STRIDE,
    load_doe,
    load_pressure_snapshot,
    load_ref_coords,
    load_snapshot_coords,
    load_settings,
    get_anatomical_regions,
    region_slice,
    get_param_cols,
    PARAM_LABELS,
)
from core.i18n import t, lang_selector

st.set_page_config(
    page_title="Pressure Field",
    page_icon="🌡️",
    layout="wide",
)
lang_selector()
st.title(t("pres_title"))
st.caption(t("pres_caption"))

# ── Static data ────────────────────────────────────────────────────────────────
doe_df   = load_doe()
settings = load_settings()
regions  = get_anatomical_regions(settings)
ref_coords = load_ref_coords(STRIDE)
param_cols = get_param_cols(doe_df)

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header(t("sidebar_vis"))

    snap = st.slider(t("lbl_snapshot"), 1, 100, 1)

    st.divider()
    use_deformed = st.toggle(
        t("pres_deformed"),
        value=False,
        help="Load run-specific mesh coordinates (~50 MB). Slower on first load.",
    )
    region_sel = st.selectbox(t("lbl_region"), ["All"] + regions)
    cmap = st.selectbox(
        t("lbl_colour_map"),
        ["Jet", "Viridis", "Plasma", "RdBu_r", "Turbo", "Hot"],
        index=0,
    )
    opacity = st.slider(t("lbl_opacity"), 0.1, 1.0, 0.65, step=0.05)
    pt_size = st.slider(t("lbl_point_size"), 1, 5, 2)

    st.divider()
    # Show DOE parameters for selected run
    row = doe_df[doe_df["snapshot_num"] == snap].iloc[0]
    st.subheader(f"Run {snap} — {t('pres_params_hdr')}")
    for col in ["A_glotis", "A_epiglotis", "d_trachea", "l_trachea", "r_curvature",
                "teta_branch_trachea", "l_l", "l_r"]:
        st.metric(PARAM_LABELS.get(col, col), f"{row[col]:.2f}")

# ── Load snapshot data ─────────────────────────────────────────────────────────
with st.spinner(f"Loading snapshot {snap}…"):
    pressure = load_pressure_snapshot(snap, STRIDE)
    coords   = load_snapshot_coords(snap, STRIDE) if use_deformed else ref_coords

# Apply region filter
if region_sel != "All":
    sl   = region_slice(settings, region_sel, STRIDE)
    coords_v   = coords[sl]
    pressure_v = pressure[sl]
else:
    coords_v   = coords
    pressure_v = pressure

# ── Summary metrics ────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(t("met_min_pres"),  f"{pressure_v.min():.1f} Pa")
m2.metric(t("met_max_pres"),  f"{pressure_v.max():.1f} Pa")
m3.metric(t("met_mean_pres"), f"{pressure_v.mean():.1f} Pa")
m4.metric(t("met_std_pres"),  f"{pressure_v.std():.1f} Pa")
m5.metric(t("met_nodes"),     f"{len(coords_v):,}")

# ── 3D scatter coloured by pressure ───────────────────────────────────────────
fig = go.Figure(
    go.Scatter3d(
        x=coords_v[:, 0],
        y=coords_v[:, 1],
        z=coords_v[:, 2],
        mode="markers",
        marker=dict(
            size=pt_size,
            color=pressure_v,
            colorscale=cmap,
            colorbar=dict(
                title="Pressure (Pa)",
                thickness=14,
                len=0.7,
            ),
            opacity=opacity,
            cmin=float(pressure_v.min()),
            cmax=float(pressure_v.max()),
        ),
        hovertemplate="P: %{marker.color:.1f} Pa<extra></extra>",
    )
)

geo_label = "deformed" if use_deformed else "reference"
fig.update_layout(
    scene=dict(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        zaxis_title="Z (m)",
        aspectmode="data",
        bgcolor="rgb(12, 14, 20)",
        xaxis=dict(gridcolor="#2a2a2a", showbackground=True, backgroundcolor="rgb(12,14,20)"),
        yaxis=dict(gridcolor="#2a2a2a", showbackground=True, backgroundcolor="rgb(12,14,20)"),
        zaxis=dict(gridcolor="#2a2a2a", showbackground=True, backgroundcolor="rgb(12,14,20)"),
    ),
    paper_bgcolor="rgb(12, 14, 20)",
    font=dict(color="white"),
    height=680,
    margin=dict(l=0, r=0, b=0, t=40),
    title=f"Static Pressure — Run {snap} · {region_sel} · {geo_label} geometry",
)

st.plotly_chart(fig, width='stretch')
