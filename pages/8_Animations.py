"""
pages/8_Animations.py
=====================
Animated POD visualisations — geometry mode sweeps and pressure snapshot reel.
Animations are generated on demand (button click) to stay within cloud memory limits.
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_io import (
    STRIDE, load_all_pressures, load_ref_coords, load_precomputed_pod,
)
from core.pod import compute_pod, cumulative_energy, modes_for_energy, reconstruct
from core.i18n import t, lang_selector

st.set_page_config(page_title="Animations", page_icon="🎬", layout="wide")
lang_selector()
st.title(t("anim_title"))
st.caption(t("anim_caption"))

# ── Load data (lightweight — only precomputed NPZ files) ──────────────────────

_pre_g = load_precomputed_pod("geometry")
_pre_p = load_precomputed_pod("pressure")

if _pre_g is None or _pre_p is None:
    st.error(t("anim_error_precomp"))
    st.stop()

mean_geo   = _pre_g["mean"]
modes_geo  = _pre_g["modes"]
scores_geo = _pre_g["scores"]
svals_geo  = _pre_g["svalues"]

mean_pres   = _pre_p["mean"]
modes_pres  = _pre_p["modes"]
scores_pres = _pre_p["scores"]
svals_pres  = _pre_p["svalues"]

ref_coords = load_ref_coords(STRIDE)
std_geo    = svals_geo  / np.sqrt(max(len(scores_geo)  - 1, 1))
std_pres   = svals_pres / np.sqrt(max(len(scores_pres) - 1, 1))
energy_geo = cumulative_energy(svals_geo)

# ── Shared layout builder ─────────────────────────────────────────────────────

def anim_layout(title: str, n_frames: int, frame_labels: list, height: int = 600) -> dict:
    return dict(
        scene=dict(
            aspectmode="data", bgcolor="rgb(12,14,20)",
            xaxis=dict(gridcolor="#2a2a2a", showbackground=True, backgroundcolor="rgb(12,14,20)"),
            yaxis=dict(gridcolor="#2a2a2a", showbackground=True, backgroundcolor="rgb(12,14,20)"),
            zaxis=dict(gridcolor="#2a2a2a", showbackground=True, backgroundcolor="rgb(12,14,20)"),
        ),
        paper_bgcolor="rgb(12,14,20)", font=dict(color="white"),
        height=height, margin=dict(l=0, r=0, b=0, t=50),
        title=title,
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=0, x=0.5, xanchor="center", yanchor="top", pad=dict(t=10),
            buttons=[
                dict(label="▶ Play",  method="animate",
                     args=[None, dict(frame=dict(duration=100, redraw=True),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method="animate",
                        args=[[f"f{k}"], dict(mode="immediate",
                               frame=dict(duration=100, redraw=True))],
                        label=frame_labels[k])
                   for k in range(n_frames)],
            transition=dict(duration=0), x=0, y=0,
            currentvalue=dict(visible=True, prefix="Frame: ", font=dict(color="#aaa")),
            len=1.0,
        )],
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_geo, tab_reel, tab_pres = st.tabs([
    t("anim_tab_geo"),
    t("anim_tab_reel"),
    t("anim_tab_pres"),
])


# ── Tab 1: Geometry mode sweep ────────────────────────────────────────────────
with tab_geo:
    st.subheader(t("anim_sub_geo"))
    st.caption("Morphs the airway from −3σ to +3σ along one POD mode.")

    c1, c2, c3 = st.columns(3)
    mode_idx     = c1.slider(t("anim_pod_mode"), 1, min(10, len(svals_geo)), 1, key="g_mode") - 1
    n_frames_g   = c2.slider(t("lbl_frames"), 10, 25, 16, key="g_frames")
    sigma_range  = c3.slider(t("anim_sigma"), 1.0, 4.0, 3.0, step=0.5, key="g_sigma")

    ep = float(svals_geo[mode_idx]**2 / (svals_geo**2).sum() * 100)
    st.info(f"Mode {mode_idx+1} captures **{ep:.1f}%** of geometry variance  |  "
            f"σ = {std_geo[mode_idx]:.4g} m")

    if st.button(t("anim_btn_geo"), type="primary", key="gen_geo"):
        sigma  = float(std_geo[mode_idx])
        alphas = np.linspace(-sigma_range * sigma, sigma_range * sigma, n_frames_g)

        with st.spinner(f"Building {n_frames_g} frames…"):
            frames = []
            for k, alpha in enumerate(alphas):
                c = np.zeros(modes_geo.shape[1])
                c[mode_idx] = alpha
                rec = reconstruct(mean_geo, modes_geo, c).reshape(-1, 3)
                frames.append(go.Frame(
                    data=[go.Scatter3d(
                        x=rec[:, 0], y=rec[:, 1], z=rec[:, 2],
                        mode="markers",
                        marker=dict(size=1.5, color="#4EB3D3", opacity=0.55),
                        hoverinfo="skip",
                    )],
                    name=f"f{k}",
                ))

        rec0 = reconstruct(mean_geo, modes_geo, np.zeros(modes_geo.shape[1])).reshape(-1, 3)
        fig = go.Figure(
            data=[go.Scatter3d(
                x=rec0[:, 0], y=rec0[:, 1], z=rec0[:, 2],
                mode="markers",
                marker=dict(size=1.5, color="#4EB3D3", opacity=0.55),
                hoverinfo="skip",
            )],
            frames=frames,
        )
        labels = [f"{a/float(std_geo[mode_idx]):.1f}σ" for a in alphas]
        fig.update_layout(**anim_layout(
            f"Geometry Mode {mode_idx+1} Sweep (±{sigma_range}σ)",
            n_frames_g, labels,
        ))
        st.plotly_chart(fig, width="stretch", key="fig_geo")


# ── Tab 2: Pressure snapshot reel ────────────────────────────────────────────
with tab_reel:
    st.subheader(t("anim_sub_reel"))
    st.caption("Steps through all 100 CFD snapshots.")

    c1, c2 = st.columns(2)
    step       = c1.slider(t("anim_every_n"), 1, 5, 2, key="reel_step")
    cmap_reel  = c2.selectbox(t("lbl_colour_map"), ["Jet","Viridis","Plasma","Turbo"], key="reel_cmap")

    snap_indices = list(range(0, 100, step))
    st.info(f"{len(snap_indices)} frames  ·  ~{len(snap_indices)*42}K points total")

    if st.button(t("anim_btn_reel"), type="primary", key="gen_reel"):
        with st.spinner("Loading pressure fields…"):
            P_all = load_all_pressures(STRIDE)

        p_min = float(P_all.min())
        p_max = float(P_all.max())

        with st.spinner(f"Building {len(snap_indices)} frames…"):
            frames = []
            for k, idx in enumerate(snap_indices):
                frames.append(go.Frame(
                    data=[go.Scatter3d(
                        x=ref_coords[:, 0], y=ref_coords[:, 1], z=ref_coords[:, 2],
                        mode="markers",
                        marker=dict(
                            size=2, color=P_all[idx].tolist(),
                            colorscale=cmap_reel,
                            cmin=p_min, cmax=p_max, opacity=0.65,
                        ),
                        hoverinfo="skip",
                    )],
                    name=f"f{k}",
                ))

        fig = go.Figure(
            data=[go.Scatter3d(
                x=ref_coords[:, 0], y=ref_coords[:, 1], z=ref_coords[:, 2],
                mode="markers",
                marker=dict(
                    size=2, color=P_all[snap_indices[0]].tolist(),
                    colorscale=cmap_reel, cmin=p_min, cmax=p_max,
                    colorbar=dict(title="P (Pa)", thickness=12),
                    opacity=0.65,
                ),
                hoverinfo="skip",
            )],
            frames=frames,
        )
        labels = [f"Run {snap_indices[k]+1}" for k in range(len(snap_indices))]
        fig.update_layout(**anim_layout(
            "Static Pressure — DOE Snapshot Reel",
            len(snap_indices), labels,
        ))
        st.plotly_chart(fig, width="stretch", key="fig_reel")
        st.caption(f"Global pressure range: {p_min:.0f} – {p_max:.0f} Pa")


# ── Tab 3: Pressure mode sweep ────────────────────────────────────────────────
with tab_pres:
    st.subheader(t("anim_sub_pres"))
    st.caption("Animates the pressure field by sweeping one POD mode from −3σ to +3σ.")

    c1, c2, c3 = st.columns(3)
    pmode_idx    = c1.slider(t("anim_pod_mode"), 1, min(10, len(svals_pres)), 1, key="p_mode") - 1
    n_frames_p   = c2.slider(t("lbl_frames"), 10, 25, 16, key="p_frames")
    sigma_range_p= c3.slider(t("anim_sigma"), 1.0, 4.0, 3.0, step=0.5, key="p_sigma")
    cmap_pmode   = st.selectbox(t("lbl_colour_map"), ["Jet","Viridis","Plasma","Turbo"], key="p_cmap")

    ep2 = float(svals_pres[pmode_idx]**2 / (svals_pres**2).sum() * 100)
    st.info(f"Pressure Mode {pmode_idx+1} captures **{ep2:.1f}%** of pressure variance")

    if st.button(t("anim_btn_pres"), type="primary", key="gen_pres"):
        sigma_p = float(std_pres[pmode_idx])
        alphas_p = np.linspace(-sigma_range_p * sigma_p, sigma_range_p * sigma_p, n_frames_p)

        with st.spinner(f"Building {n_frames_p} frames…"):
            fields = []
            for alpha in alphas_p:
                c = np.zeros(modes_pres.shape[1])
                c[pmode_idx] = alpha
                fields.append(mean_pres + modes_pres @ c)

        p_min2 = min(f.min() for f in fields)
        p_max2 = max(f.max() for f in fields)

        frames = [
            go.Frame(
                data=[go.Scatter3d(
                    x=ref_coords[:, 0], y=ref_coords[:, 1], z=ref_coords[:, 2],
                    mode="markers",
                    marker=dict(
                        size=2, color=f.tolist(),
                        colorscale=cmap_pmode,
                        cmin=p_min2, cmax=p_max2, opacity=0.65,
                    ),
                    hoverinfo="skip",
                )],
                name=f"f{k}",
            )
            for k, f in enumerate(fields)
        ]

        fig = go.Figure(
            data=[go.Scatter3d(
                x=ref_coords[:, 0], y=ref_coords[:, 1], z=ref_coords[:, 2],
                mode="markers",
                marker=dict(
                    size=2, color=fields[0].tolist(),
                    colorscale=cmap_pmode, cmin=p_min2, cmax=p_max2,
                    colorbar=dict(title="P (Pa)", thickness=12),
                    opacity=0.65,
                ),
                hoverinfo="skip",
            )],
            frames=frames,
        )
        labels = [f"{a/sigma_p:.1f}σ" for a in alphas_p]
        fig.update_layout(**anim_layout(
            f"Pressure Mode {pmode_idx+1} Sweep (±{sigma_range_p}σ)",
            n_frames_p, labels,
        ))
        st.plotly_chart(fig, width="stretch", key="fig_pmode")
