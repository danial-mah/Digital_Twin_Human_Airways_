"""
pages/1_Geometry_Explorer.py
============================
POD-based interactive shape viewer with real-time slider response.
Uses @st.fragment with controls in the main area (sidebar widgets are not
allowed inside fragments in Streamlit >= 1.37).
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
    STRIDE, load_all_coords, load_doe, load_ref_coords,
    get_param_cols, load_precomputed_pod,
)
from core.pod import compute_pod, cumulative_energy, modes_for_energy, reconstruct
from core.i18n import t, lang_selector

st.set_page_config(page_title="Geometry Explorer", page_icon="🔬", layout="wide")
lang_selector()

st.title(t("geo_title"))
st.caption(t("geo_caption"))

# ── Data loading (runs once, cached) ──────────────────────────────────────────

doe_df = load_doe()

_pre = load_precomputed_pod("geometry")
if _pre is not None:
    mean_geo   = _pre["mean"]
    modes_geo  = _pre["modes"]
    scores_geo = _pre["scores"]
    svals_geo  = _pre["svalues"]
else:
    with st.spinner("Loading geometry snapshots and computing POD…"):
        X_geo = load_all_coords(STRIDE)

    @st.cache_data(show_spinner=False)
    def geo_pod(X):
        return compute_pod(X)

    mean_geo, modes_geo, scores_geo, svals_geo = geo_pod(X_geo)

energy   = cumulative_energy(svals_geo)
n95      = modes_for_energy(svals_geo, 0.95)
n99      = modes_for_energy(svals_geo, 0.99)
std_devs = svals_geo / np.sqrt(max(len(scores_geo) - 1, 1))

# Static sidebar — no interactive widgets, just info
with st.sidebar:
    st.header(t("geo_title")[:20])
    st.metric(t("geo_modes_95"), n95)
    st.metric(t("geo_modes_99"), n99)
    st.metric(t("geo_total_modes"), len(svals_geo))
    st.divider()
    st.caption(t("geo_caption"))

# ── Fragment: everything interactive lives here ───────────────────────────────

@st.fragment
def geometry_viewer(n_sliders, mean_geo, modes_geo, scores_geo, svals_geo, std_devs, energy):

    for i in range(n_sliders):
        if f"geo_mode_{i}" not in st.session_state:
            st.session_state[f"geo_mode_{i}"] = 0.0

    ctrl_col, view_col = st.columns([1, 3], gap="medium")

    with ctrl_col:
        st.markdown(t("geo_controls"))

        b1, b2 = st.columns(2)
        if b1.button(t("btn_reset"), key="btn_reset", width='stretch'):
            for i in range(n_sliders):
                st.session_state[f"geo_mode_{i}"] = 0.0
            st.rerun(scope="fragment")

        if b2.button(t("btn_random"), key="btn_random", width='stretch'):
            rng = np.random.default_rng()
            for i in range(n_sliders):
                st.session_state[f"geo_mode_{i}"] = float(
                    rng.uniform(-2.0 * std_devs[i], 2.0 * std_devs[i])
                )
            st.rerun(scope="fragment")

        def _load_snapshot():
            val = st.session_state["snap_pick_geo"]
            if val != t("geo_choose"):
                idx = int(val.split()[1]) - 1
                for i in range(n_sliders):
                    st.session_state[f"geo_mode_{i}"] = float(scores_geo[idx, i])

        st.selectbox(
            t("geo_load_snap"),
            [t("geo_choose")] + [f"Run {i}" for i in range(1, 101)],
            key="snap_pick_geo",
            on_change=_load_snapshot,
        )

        st.divider()
        st.markdown(t("geo_pod_sliders"))

        coeffs = np.zeros(len(svals_geo))
        for i in range(n_sliders):
            ep = float(
                energy[i] * 100 if i == 0
                else (energy[i] - energy[i - 1]) * 100
            )
            coeffs[i] = st.slider(
                f"Mode {i+1}  ({ep:.1f}%)",
                min_value=float(-3.0 * std_devs[i]),
                max_value=float( 3.0 * std_devs[i]),
                step=float(std_devs[i] / 30),
                key=f"geo_mode_{i}",
                format="%.3f",
            )

        dev = float(np.linalg.norm(coeffs[:n_sliders]))
        st.metric(t("geo_deviation"), f"{dev:.3f}")

        st.divider()
        st.markdown(t("geo_visualisation"))
        _color_opts = [t("geo_disp_mean"), t("geo_mode_contrib"), t("geo_z_coord"), t("geo_uniform")]
        color_by = st.radio(t("lbl_colour_by"), _color_opts, key="geo_color_mode")
        show_mean_ghost = st.checkbox(t("geo_show_ghost"), value=False, key="geo_ghost")

        st.markdown(t("geo_active_coeff"))
        coeff_df = pd.DataFrame({
            "Mode": [f"M{i+1}" for i in range(n_sliders)],
            "σᵢ":   [f"{std_devs[i]:.3g}" for i in range(n_sliders)],
            "cᵢ":   [f"{coeffs[i]:.3g}"   for i in range(n_sliders)],
        })
        st.dataframe(coeff_df, hide_index=True, height=180)

    with view_col:
        rec_coords  = reconstruct(mean_geo, modes_geo, coeffs).reshape(-1, 3)
        mean_coords = mean_geo.reshape(-1, 3)

        # ── Choose colour array ───────────────────────────────────────────
        if color_by == "Displacement from mean":
            disp  = np.linalg.norm(rec_coords - mean_coords, axis=1)
            c_arr = disp
            cscale = "Hot"
            cbar   = dict(title="Displacement (m)", thickness=12)
            cbar_show = True
        elif color_by == t("geo_mode_contrib"):
            # Show which mode has the largest absolute contribution at each point
            active = [i for i in range(n_sliders) if abs(coeffs[i]) > 1e-9]
            if active:
                # Weighted sum of mode spatial patterns
                mode_contrib = sum(
                    abs(coeffs[i]) * np.linalg.norm(
                        modes_geo[:, i].reshape(-1, 3), axis=1
                    )
                    for i in active
                )
                c_arr = mode_contrib
            else:
                c_arr = np.zeros(len(rec_coords))
            cscale = "Plasma"
            cbar   = dict(title="Mode influence", thickness=12)
            cbar_show = True
        elif color_by == t("geo_z_coord"):
            c_arr  = rec_coords[:, 2]
            cscale = "Viridis"
            cbar   = dict(title="Z (m)", thickness=12)
            cbar_show = True
        else:
            c_arr  = "#4EB3D3"
            cscale = None
            cbar   = None
            cbar_show = False

        # ── Build figure ──────────────────────────────────────────────────
        scene_cfg = dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data", bgcolor="rgb(15,17,25)",
            xaxis=dict(gridcolor="#333", showbackground=True, backgroundcolor="rgb(15,17,25)"),
            yaxis=dict(gridcolor="#333", showbackground=True, backgroundcolor="rgb(15,17,25)"),
            zaxis=dict(gridcolor="#333", showbackground=True, backgroundcolor="rgb(15,17,25)"),
        )

        traces = []

        # Optional mean-shape ghost
        if show_mean_ghost:
            traces.append(go.Scatter3d(
                x=mean_coords[:, 0], y=mean_coords[:, 1], z=mean_coords[:, 2],
                mode="markers",
                marker=dict(size=1.2, color="rgba(150,150,150,0.18)"),
                hoverinfo="skip",
                name="Mean shape",
                showlegend=True,
            ))

        # Deformed shape
        marker_cfg = dict(size=1.8, opacity=0.75)
        if cbar_show:
            marker_cfg.update(
                color=c_arr, colorscale=cscale,
                colorbar=cbar,
                cmin=float(np.min(c_arr)) if not isinstance(c_arr, str) else None,
                cmax=float(np.max(c_arr)) if not isinstance(c_arr, str) else None,
            )
        else:
            marker_cfg["color"] = c_arr

        traces.append(go.Scatter3d(
            x=rec_coords[:, 0], y=rec_coords[:, 1], z=rec_coords[:, 2],
            mode="markers",
            marker=marker_cfg,
            hoverinfo="skip",
            name="Deformed shape",
            showlegend=show_mean_ghost,
        ))

        fig3d = go.Figure(data=traces)
        fig3d.update_layout(
            scene=scene_cfg,
            paper_bgcolor="rgb(15,17,25)", font=dict(color="white"),
            height=580, margin=dict(l=0, r=0, b=0, t=30),
            title=f"Geometry — coloured by {color_by}",
            legend=dict(x=0.01, y=0.99, font=dict(size=10)),
        )
        st.plotly_chart(fig3d, width='stretch', key="geo_3d_plot")

        k_plot = min(30, len(svals_geo))
        fig_e = px.line(
            x=list(range(1, k_plot + 1)),
            y=(energy[:k_plot] * 100).tolist(),
            labels={"x": "Modes", "y": "Cumul. Energy (%)"},
            title=t("geo_pod_energy"),
        )
        fig_e.add_hline(y=95, line_dash="dash", line_color="orange", annotation_text="95%")
        fig_e.add_hline(y=99, line_dash="dash", line_color="red",    annotation_text="99%")
        fig_e.update_layout(height=220, margin=dict(l=10, r=10, b=10, t=30))
        st.plotly_chart(fig_e, width='stretch', key="geo_energy_plot")


n_sliders = st.sidebar.slider(t("geo_modes_display"), 1, min(20, len(svals_geo)), 10)
geometry_viewer(n_sliders, mean_geo, modes_geo, scores_geo, svals_geo, std_devs, energy)
