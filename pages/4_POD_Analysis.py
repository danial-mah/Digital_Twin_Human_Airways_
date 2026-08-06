"""
pages/4_POD_Analysis.py
========================
POD (Proper Orthogonal Decomposition) analysis for both geometry and pressure.

This page answers the question:
  "How many modes are needed to represent the data faithfully?"

Sections
--------
  Energy Curves     — cumulative variance for geometry and pressure POD
  Mode Shapes       — 3D visualisation of individual POD modes
  Reconstruction    — compare original vs POD-reconstructed snapshot
  Validation        — K-Fold and LOO reconstruction error statistics
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
    load_all_coords,
    load_all_pressures,
    load_ref_coords,
    load_doe,
)
from core.pod import (
    compute_pod,
    cumulative_energy,
    modes_for_energy,
    reconstruct,
    reconstruction_error,
)
from core.data_io import load_precomputed_pod
from core.i18n import t, lang_selector

st.set_page_config(page_title="POD Analysis", page_icon="📊", layout="wide")
lang_selector()
st.title(t("pod_title"))
st.caption(t("pod_caption"))

# ── Load data ──────────────────────────────────────────────────────────────────
doe_df = load_doe()

_pre_g = load_precomputed_pod("geometry")
_pre_p = load_precomputed_pod("pressure")

if _pre_g is not None:
    mean_geo, modes_geo, scores_geo, sv_geo = (
        _pre_g["mean"], _pre_g["modes"], _pre_g["scores"], _pre_g["svalues"]
    )
else:
    with st.spinner("Loading geometry snapshots for POD…"):
        X_geo = load_all_coords(STRIDE)
    @st.cache_data(show_spinner=False)
    def geo_pod(X): return compute_pod(X)
    with st.spinner("Computing geometry POD…"):
        mean_geo, modes_geo, scores_geo, sv_geo = geo_pod(X_geo)

with st.spinner("Loading pressure snapshots…"):
    P_pres = load_all_pressures(STRIDE)

if _pre_p is not None:
    mean_pres, modes_pres, scores_pres, sv_pres = (
        _pre_p["mean"], _pre_p["modes"], _pre_p["scores"], _pre_p["svalues"]
    )
else:
    @st.cache_data(show_spinner=False)
    def pres_pod(P): return compute_pod(P)
    with st.spinner("Computing pressure POD…"):
        mean_pres, modes_pres, scores_pres, sv_pres = pres_pod(P_pres)

ref_coords = load_ref_coords(STRIDE)

# Pre-compute energy
e_geo  = cumulative_energy(sv_geo)
e_pres = cumulative_energy(sv_pres)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_e, tab_m, tab_r, tab_v = st.tabs([
    t("pod_tab_energy"),
    t("pod_tab_modes"),
    t("pod_tab_recon"),
    t("pod_tab_val"),
])

# ─ Tab 1: Energy curves ────────────────────────────────────────────────────────
with tab_e:
    st.subheader("Cumulative Energy vs Number of Modes")
    st.markdown("""
    The energy of mode *i* = σᵢ² / Σ σⱼ² where σ are the singular values.
    Cumulative energy tells us how many modes are needed to represent
    a given percentage of the total variance in the dataset.
    """)

    k_plot = min(40, len(sv_geo), len(sv_pres))
    modes_axis = list(range(1, k_plot + 1))

    fig_e = go.Figure()
    fig_e.add_trace(go.Scatter(
        x=modes_axis, y=(e_geo[:k_plot] * 100).tolist(),
        mode="lines+markers", name="Geometry (Coordinates)",
        line=dict(color="#4EB3D3", width=2), marker=dict(size=5),
    ))
    fig_e.add_trace(go.Scatter(
        x=modes_axis, y=(e_pres[:k_plot] * 100).tolist(),
        mode="lines+markers", name="Pressure Field",
        line=dict(color="#F4A261", width=2), marker=dict(size=5),
    ))
    fig_e.add_hline(y=95, line_dash="dash", line_color="gray",
                    annotation_text="95 %")
    fig_e.add_hline(y=99, line_dash="dot",  line_color="gray",
                    annotation_text="99 %")
    fig_e.update_layout(
        xaxis_title="Number of Modes",
        yaxis_title="Cumulative Energy (%)",
        yaxis_range=[0, 101],
        legend=dict(x=0.6, y=0.1),
        height=450,
    )
    st.plotly_chart(fig_e, width='stretch')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("pod_geo_modes95"),  modes_for_energy(sv_geo,  0.95))
    c2.metric(t("pod_geo_modes99"),  modes_for_energy(sv_geo,  0.99))
    c3.metric(t("pod_pres_modes95"), modes_for_energy(sv_pres, 0.95))
    c4.metric(t("pod_pres_modes99"), modes_for_energy(sv_pres, 0.99))

    st.subheader(t("pod_sub_sv"))
    sv_df = pd.DataFrame({
        "Mode":     list(range(1, k_plot + 1)) * 2,
        "Value":    list(sv_geo[:k_plot]) + list(sv_pres[:k_plot]),
        "Type":     ["Geometry"] * k_plot + ["Pressure"] * k_plot,
    })
    fig_sv = px.bar(sv_df, x="Mode", y="Value", color="Type",
                    barmode="group", log_y=True,
                    title="Singular values (log scale)",
                    color_discrete_map={"Geometry": "#4EB3D3", "Pressure": "#F4A261"})
    fig_sv.update_layout(height=350)
    st.plotly_chart(fig_sv, width='stretch')

# ─ Tab 2: Mode shapes ──────────────────────────────────────────────────────────
with tab_m:
    st.subheader("3D Mode Shape Visualisation")
    st.caption(
        "A mode shape shows *where* the airway geometry or pressure varies "
        "most strongly for that mode. Positive values (warm) and negative (cool) "
        "indicate the direction of variation."
    )

    field_sel  = st.radio(t("pod_mode_field"), [t("pod_geo_label"), t("pod_pres_label")], horizontal=True)
    mode_idx   = st.slider(t("pod_mode_number"), 1, min(20, len(sv_geo)), 1)

    if field_sel == t("pod_geo_label"):
        mode_vec = modes_geo[:, mode_idx - 1].reshape(-1, 3)
        colour = np.linalg.norm(mode_vec, axis=1)
        colour_label = t("pod_disp_mag")
        coords_plot = ref_coords
    else:
        mode_vec     = modes_pres[:, mode_idx - 1]
        colour       = mode_vec
        colour_label = t("pod_pres_mode_val")
        coords_plot  = ref_coords

    fig_m = go.Figure(go.Scatter3d(
        x=coords_plot[:, 0],
        y=coords_plot[:, 1],
        z=coords_plot[:, 2],
        mode="markers",
        marker=dict(
            size=2,
            color=colour,
            colorscale="RdBu_r",
            colorbar=dict(title=colour_label, thickness=14),
            opacity=0.7,
        ),
        hoverinfo="skip",
    ))
    energy_pct = (
        e_geo[mode_idx - 1] if field_sel.startswith("Geometry") else e_pres[mode_idx - 1]
    ) * 100
    prev_pct = (
        e_geo[mode_idx - 2] if mode_idx > 1 else 0.0
    ) * 100 if field_sel.startswith("Geometry") else (
        e_pres[mode_idx - 2] if mode_idx > 1 else 0.0
    ) * 100
    mode_pct = energy_pct - prev_pct

    fig_m.update_layout(
        scene=dict(aspectmode="data",
                   bgcolor="rgb(15,17,25)",
                   xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
                   zaxis=dict(gridcolor="#333")),
        paper_bgcolor="rgb(15,17,25)",
        font=dict(color="white"),
        height=620,
        margin=dict(l=0, r=0, b=0, t=40),
        title=f"Mode {mode_idx} — {mode_pct:.1f} % individual energy",
    )
    st.plotly_chart(fig_m, width='stretch')

# ─ Tab 3: Reconstruction ───────────────────────────────────────────────────────
with tab_r:
    st.subheader("Original vs POD Reconstruction")

    col_l, col_r = st.columns(2)
    snap_r   = col_l.slider(t("pod_snap_recon"), 1, 100, 1, key="rec_snap")
    k_modes  = col_r.slider(t("pod_modes_used"), 1, 40, 10, key="rec_k")

    # Pressure reconstruction
    actual_pres  = P_pres[snap_r - 1]
    rec_pres     = reconstruct(mean_pres, modes_pres[:, :k_modes], scores_pres[snap_r - 1, :k_modes])
    rel_err      = float(np.linalg.norm(actual_pres - rec_pres) / (np.linalg.norm(actual_pres) + 1e-12))

    st.metric(f'{t("pod_rel_err")} ({k_modes} {t("lbl_modes")})', f"{rel_err * 100:.3f} %")

    # Side-by-side 3D plots
    c1, c2 = st.columns(2)

    def pressure_scatter(coords, pressure, title):
        fig = go.Figure(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(size=1.5, color=pressure, colorscale="Jet",
                        colorbar=dict(title="Pa", thickness=10), opacity=0.65),
            hoverinfo="skip",
        ))
        fig.update_layout(
            scene=dict(aspectmode="data", bgcolor="rgb(15,17,25)",
                       xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
                       zaxis=dict(gridcolor="#333")),
            paper_bgcolor="rgb(15,17,25)", font=dict(color="white"),
            height=480, margin=dict(l=0, r=0, b=0, t=40), title=title,
        )
        return fig

    c1.plotly_chart(pressure_scatter(ref_coords, actual_pres, "Original"), width='stretch')
    c2.plotly_chart(pressure_scatter(ref_coords, rec_pres,    f"Reconstructed ({k_modes} modes)"),
                    width='stretch')

    # Error map
    err_field = np.abs(actual_pres - rec_pres)
    st.subheader("Absolute Error Map")
    fig_err = go.Figure(go.Scatter3d(
        x=ref_coords[:, 0], y=ref_coords[:, 1], z=ref_coords[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=err_field, colorscale="Hot",
                    colorbar=dict(title="|Error| (Pa)", thickness=14), opacity=0.7),
        hoverinfo="skip",
    ))
    fig_err.update_layout(
        scene=dict(aspectmode="data", bgcolor="rgb(15,17,25)",
                   xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
                   zaxis=dict(gridcolor="#333")),
        paper_bgcolor="rgb(15,17,25)", font=dict(color="white"),
        height=500, margin=dict(l=0, r=0, b=0, t=40),
        title="Absolute point-wise reconstruction error",
    )
    st.plotly_chart(fig_err, width='stretch')

# ─ Tab 4: Validation ───────────────────────────────────────────────────────────
with tab_v:
    st.subheader("Reconstruction Error vs Number of Modes")
    st.caption(
        "Relative L2 error averaged over all 100 snapshots as a function "
        "of the number of POD modes used."
    )

    @st.cache_data(show_spinner=False)
    def error_vs_modes(P, mean_p, modes_p, scores_p):
        k_range = list(range(1, min(31, P.shape[0])))
        errs = []
        for k in k_range:
            e = reconstruction_error(P, mean_p, modes_p[:, :k], scores_p[:, :k])
            errs.append(float(e.mean()))
        return k_range, errs

    with st.spinner("Computing error vs modes…"):
        k_range, errs = error_vs_modes(P_pres, mean_pres, modes_pres, scores_pres)

    fig_val = px.line(
        x=k_range, y=[e * 100 for e in errs],
        labels={"x": "Modes", "y": "Mean Relative L2 Error (%)"},
        title="Pressure reconstruction error vs number of POD modes",
        markers=True,
    )
    fig_val.update_layout(height=400)
    st.plotly_chart(fig_val, width='stretch')

    errs_arr = np.array(errs)
    k_1pct = next((k for k, e in zip(k_range, errs_arr) if e < 0.01), k_range[-1])
    k_5pct = next((k for k, e in zip(k_range, errs_arr) if e < 0.05), k_range[-1])
    v1, v2 = st.columns(2)
    v1.metric("Modes for < 1 % error", k_1pct)
    v2.metric("Modes for < 5 % error", k_5pct)

    st.divider()
    st.subheader("K-Fold Reconstruction Error")
    st.caption(
        "Splits the 100 snapshots into k folds, reconstructs each held-out fold "
        "using POD trained on the remaining folds. Measures how well POD generalises."
    )

    kf_col1, kf_col2 = st.columns(2)
    n_folds_pod = kf_col1.slider(t("lbl_n_folds"), 2, 10, 5, key="pod_kfold")
    k_modes_kf  = kf_col2.slider(t("pod_kfold_modes"), 1, 30, 10, key="pod_kfold_k")

    if st.button(t("pod_run_kfold"), key="pod_kfold_btn"):
        with st.spinner(f"Running {n_folds_pod}-fold reconstruction validation…"):
            rng = np.random.default_rng(42)
            indices = rng.permutation(100)
            folds   = np.array_split(indices, n_folds_pod)
            fold_errs = []

            for fold_idx, test_idx in enumerate(folds):
                train_idx = np.concatenate([folds[j] for j in range(n_folds_pod)
                                            if j != fold_idx])
                P_train = P_pres[train_idx]
                P_test  = P_pres[test_idx]

                m_tr, modes_tr, scores_tr, _ = compute_pod(P_train)
                # Project test snapshots onto train modes
                test_scores = (P_test - m_tr) @ modes_tr[:, :k_modes_kf]
                rec_test    = m_tr + test_scores @ modes_tr[:, :k_modes_kf].T
                fold_rel_err = float(
                    np.linalg.norm(P_test - rec_test) /
                    (np.linalg.norm(P_test) + 1e-12)
                )
                fold_errs.append(fold_rel_err)

        fold_df = pd.DataFrame({
            "Fold":            [f"Fold {i+1}" for i in range(n_folds_pod)],
            "Relative L2 error": fold_errs,
        })
        fig_kf_pod = px.bar(
            fold_df, x="Fold", y="Relative L2 error",
            color="Relative L2 error", color_continuous_scale="Blues",
            text_auto=".4f",
            title=f"{n_folds_pod}-Fold POD reconstruction error (k={k_modes_kf} modes)",
        )
        fig_kf_pod.update_layout(height=320)
        st.plotly_chart(fig_kf_pod, width='stretch')

        kf1, kf2, kf3 = st.columns(3)
        kf1.metric("Mean fold error", f"{np.mean(fold_errs)*100:.3f} %")
        kf2.metric("Worst fold",      f"{np.max(fold_errs)*100:.3f} %")
        kf3.metric("Best fold",       f"{np.min(fold_errs)*100:.3f} %")
        st.caption(
            "A consistent error across folds means POD generalises well. "
            "A high worst-fold error suggests some snapshots are outliers."
        )
