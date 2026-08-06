"""
pages/15_Patient_Comparison.py
================================
Two-Patient Airway Comparison — Human Airways Digital Twin.

Renders two patients side-by-side: pressure fields, resistance metrics,
regional analysis, and a delta-pressure map showing the difference.

Each patient can be defined by:
  • A DOE snapshot (1-100) — uses the real CFD pressure field
  • Custom RBF prediction — adjust 26 geometry sliders directly
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_io import (
    STRIDE, load_pressure_snapshot, load_ref_coords,
    load_doe, get_param_cols, PARAM_LABELS, PARAM_GROUPS,
    load_precomputed_pod, load_all_pressures,
)
from core.pod import compute_pod, modes_for_energy, reconstruct
from core.rbf import build_rbf, predict
from core.i18n import t, lang_selector

st.set_page_config(page_title="Patient Comparison", page_icon="⚖️", layout="wide")
lang_selector()
st.title(t("cmp2_title"))
st.caption(t("cmp2_caption"))

# ── Load shared data ──────────────────────────────────────────────────────────
doe_df     = load_doe()
param_cols = get_param_cols(doe_df)
ref_coords = load_ref_coords(STRIDE)

_pre_p = load_precomputed_pod("pressure")
if _pre_p is not None:
    mean_pres, modes_pres, scores_pres, sv_pres = (
        _pre_p["mean"], _pre_p["modes"], _pre_p["scores"], _pre_p["svalues"]
    )
else:
    with st.spinner("Loading pressure POD…"):
        P_mat = load_all_pressures(STRIDE)
        mean_pres, modes_pres, scores_pres, sv_pres = compute_pod(P_mat)

params_raw  = doe_df[param_cols].values.astype(float)
low, high   = params_raw.min(0), params_raw.max(0)
params_norm = (params_raw - low) / (high - low + 1e-12)
k_pres      = modes_for_energy(sv_pres, 0.99)

@st.cache_data(show_spinner=False)
def get_rbf(k: int, kern: str = "thin_plate_spline"):
    return build_rbf(params_norm, scores_pres[:, :k], kernel=kern)

rbf_model = get_rbf(k_pres)


# ── Helper: predict pressure from RBF for a parameter vector ─────────────────
def predict_pressure(param_vec: np.ndarray) -> np.ndarray:
    pnorm = (param_vec - low) / (high - low + 1e-12)
    scores = predict(rbf_model, pnorm.reshape(1, -1))[0]
    return reconstruct(mean_pres, modes_pres[:, :k_pres], scores)


# ── Helper: resistance metrics ────────────────────────────────────────────────
def resistance_metrics(pressure: np.ndarray) -> dict:
    dp   = float(pressure.max() - pressure.min())
    mean = float(pressure.mean())
    std  = float(pressure.std())
    dp_mean_all = float(scores_pres[:, 0].std() * sv_pres[0] / np.sqrt(99) + 1e-6)
    # resistance index relative to mean of all 100 snapshots
    all_dp = np.array([
        scores_pres[i, 0] * sv_pres[0] for i in range(100)
    ])
    # Use actual all-snapshot DeltaP distribution
    def _dp(i):
        p = load_pressure_snapshot(i + 1, STRIDE)
        return float(p.max() - p.min())
    all_dp_field = np.array([_dp(i) for i in range(100)])
    resist_idx = dp / (all_dp_field.mean() + 1e-9) * 100
    return {
        "ΔP (Pa)": round(dp, 2),
        "Mean P (Pa)": round(mean, 2),
        "Std P (Pa)": round(std, 2),
        "Resistance index (%)": round(resist_idx, 1),
    }


@st.cache_data(show_spinner=False)
def all_snapshot_dp() -> np.ndarray:
    arr = np.empty(100)
    for i in range(100):
        p = load_pressure_snapshot(i + 1, STRIDE)
        arr[i] = p.max() - p.min()
    return arr

dp_all = all_snapshot_dp()
dp_mean_ref = dp_all.mean()


def resistance_index(dp: float) -> float:
    return dp / (dp_mean_ref + 1e-9) * 100


# ── Sidebar: patient mode selection ──────────────────────────────────────────
_doe_opt = t("cmp2_doe_snap")
_rbf_opt = t("cmp2_custom_rbf")

with st.sidebar:
    st.header(t("cmp2_patient_a"))
    mode_a = st.radio(t("cmp2_input_mode"), [_doe_opt, _rbf_opt], key="mode_a")
    if mode_a == _doe_opt:
        snap_a = st.slider(t("cmp2_snap_a"), 1, 100, 10, key="snap_a")
    st.divider()
    st.header(t("cmp2_patient_b"))
    mode_b = st.radio(t("cmp2_input_mode"), [_doe_opt, _rbf_opt], key="mode_b")
    if mode_b == _doe_opt:
        snap_b = st.slider(t("cmp2_snap_b"), 1, 100, 55, key="snap_b")
    st.divider()
    colorscale = st.selectbox(t("lbl_colour_map"), ["Jet", "RdBu_r", "Viridis", "Plasma"])
    show_delta = st.checkbox(t("cmp2_delta_map"), value=True,
                             help="Third view: A−B pressure difference")


# ── Custom parameter inputs (expanders, below the main layout) ───────────────
def param_sliders(suffix: str) -> np.ndarray:
    """Render 26 geometry sliders grouped by anatomy. Returns param vector."""
    vec = []
    for group, cols in PARAM_GROUPS.items():
        with st.expander(group, expanded=False):
            for col in cols:
                idx = list(param_cols).index(col)
                val = st.slider(
                    PARAM_LABELS.get(col, col),
                    float(low[idx]), float(high[idx]),
                    float((low[idx] + high[idx]) / 2),
                    key=f"{col}_{suffix}",
                )
                vec.append(val)
    return np.array(vec)


# ── Pressure loading / prediction ─────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader(f"🔵 {t('cmp2_patient_a')}")
    if mode_a == _doe_opt:
        pressure_a = load_pressure_snapshot(snap_a, STRIDE)
        label_a    = f"Snapshot {snap_a}"
        params_a   = doe_df[param_cols].iloc[snap_a - 1].values.astype(float)
    else:
        st.markdown("**Set geometry parameters:**")
        params_a  = param_sliders("A")
        with st.spinner(t("cmp2_spinner")):
            pressure_a = predict_pressure(params_a)
        label_a = "Custom A"

with col_b:
    st.subheader(f"🔴 {t('cmp2_patient_b')}")
    if mode_b == _doe_opt:
        pressure_b = load_pressure_snapshot(snap_b, STRIDE)
        label_b    = f"Snapshot {snap_b}"
        params_b   = doe_df[param_cols].iloc[snap_b - 1].values.astype(float)
    else:
        st.markdown("**Set geometry parameters:**")
        params_b  = param_sliders("B")
        with st.spinner(t("cmp2_spinner")):
            pressure_b = predict_pressure(params_b)
        label_b = "Custom B"

# ── Derived metrics ───────────────────────────────────────────────────────────
dp_a = float(pressure_a.max() - pressure_a.min())
dp_b = float(pressure_b.max() - pressure_b.min())
ri_a = resistance_index(dp_a)
ri_b = resistance_index(dp_b)

delta_p = pressure_a - pressure_b

st.divider()

# ── Metric comparison row ─────────────────────────────────────────────────────
m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
m1.metric(t("cmp2_dp_a"),     f"{dp_a:.1f}")
m2.metric(t("cmp2_dp_b"),     f"{dp_b:.1f}",
          delta=f"{dp_b - dp_a:+.1f}", delta_color="inverse")
m3.metric(t("cmp2_resist_a"), f"{ri_a:.0f}")
m4.metric(t("cmp2_resist_b"), f"{ri_b:.0f}",
          delta=f"{ri_b - ri_a:+.0f}", delta_color="inverse")
m5.metric(f"{t('met_mean_pres')} A", f"{pressure_a.mean():.1f} Pa")
m6.metric(f"{t('met_mean_pres')} B", f"{pressure_b.mean():.1f} Pa")
m7.metric(t("cmp2_max_diff"), f"{abs(delta_p).max():.1f} Pa")
m8.metric(t("cmp2_higher"),
          "A" if ri_a > ri_b else ("B" if ri_b > ri_a else "Equal"),
          delta="↑ A harder to breathe" if ri_a > ri_b else "↑ B harder to breathe",
          delta_color="inverse")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_3d, tab_delta, tab_region, tab_params = st.tabs([
    t("cmp2_tab_3d"),
    t("cmp2_tab_delta"),
    t("cmp2_tab_regional"),
    t("cmp2_tab_params"),
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Side-by-side 3D pressure fields
# ════════════════════════════════════════════════════════════════════════════
with tab_3d:
    # Shared colour range so both maps are comparable
    vmin = min(pressure_a.min(), pressure_b.min())
    vmax = max(pressure_a.max(), pressure_b.max())

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[f"Patient A — {label_a}", f"Patient B — {label_b}"],
        horizontal_spacing=0.02,
    )

    mk_common = dict(size=1.8, colorscale=colorscale, cmin=vmin, cmax=vmax)

    fig.add_trace(go.Scatter3d(
        x=ref_coords[:, 0], y=ref_coords[:, 1], z=ref_coords[:, 2],
        mode="markers",
        marker=dict(**mk_common, color=pressure_a,
                    colorbar=dict(title="Pa", x=0.44, thickness=10)),
        hovertemplate="P: %{marker.color:.1f} Pa<extra></extra>",
        name="Patient A",
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=ref_coords[:, 0], y=ref_coords[:, 1], z=ref_coords[:, 2],
        mode="markers",
        marker=dict(**mk_common, color=pressure_b,
                    colorbar=dict(title="Pa", x=1.01, thickness=10)),
        hovertemplate="P: %{marker.color:.1f} Pa<extra></extra>",
        name="Patient B",
    ), row=1, col=2)

    scene = dict(
        bgcolor="rgb(10, 12, 18)", aspectmode="data",
        xaxis=dict(showbackground=False, gridcolor="#222"),
        yaxis=dict(showbackground=False, gridcolor="#222"),
        zaxis=dict(showbackground=False, gridcolor="#222"),
    )
    fig.update_layout(
        scene=scene, scene2=scene,
        paper_bgcolor="rgb(10, 12, 18)", font=dict(color="white"),
        height=640, margin=dict(l=0, r=0, b=0, t=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Both maps share the same pressure colour scale for direct comparison.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Delta pressure map (A − B)
# ════════════════════════════════════════════════════════════════════════════
with tab_delta:
    st.markdown(
        f"**Colour = pressure of Patient A minus pressure of Patient B (Pa).** "
        f"Red = A has higher pressure here. Blue = B has higher pressure here."
    )

    abs_max = float(np.abs(delta_p).max())

    fig_delta = go.Figure(go.Scatter3d(
        x=ref_coords[:, 0], y=ref_coords[:, 1], z=ref_coords[:, 2],
        mode="markers",
        marker=dict(
            size=2,
            color=delta_p,
            colorscale="RdBu_r",
            cmin=-abs_max, cmax=abs_max,
            colorbar=dict(title="ΔP A−B (Pa)", thickness=14),
            opacity=0.85,
        ),
        hovertemplate="ΔP A−B: %{marker.color:.2f} Pa<extra></extra>",
    ))
    fig_delta.update_layout(
        scene=dict(
            bgcolor="rgb(10, 12, 18)", aspectmode="data",
            xaxis=dict(showbackground=False, gridcolor="#222"),
            yaxis=dict(showbackground=False, gridcolor="#222"),
            zaxis=dict(showbackground=False, gridcolor="#222"),
        ),
        paper_bgcolor="rgb(10, 12, 18)", font=dict(color="white"),
        height=640, margin=dict(l=0, r=0, b=0, t=30),
        title=f"Δ Pressure: {label_a} − {label_b}  |  max |ΔP| = {abs_max:.1f} Pa",
    )
    st.plotly_chart(fig_delta, use_container_width=True)

    # Histogram of differences
    fig_hist = px.histogram(
        x=delta_p, nbins=80,
        title=f"Distribution of nodal pressure differences (A − B)",
        labels={"x": "Pressure difference (Pa)"},
        color_discrete_sequence=["#4EB3D3"],
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="white")
    fig_hist.update_layout(height=300, template="plotly_dark", margin=dict(t=40, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Regional comparison
# ════════════════════════════════════════════════════════════════════════════
with tab_region:
    st.markdown(
        "Mean pressure per airway depth layer. Uses pressure as a depth proxy "
        "(high pressure = near inlet, low pressure = near bronchial outlets)."
    )

    n_bins = 10
    edges  = np.linspace(
        min(pressure_a.min(), pressure_b.min()),
        max(pressure_a.max(), pressure_b.max()),
        n_bins + 1,
    )
    labels_r = [
        "Terminal bronchi", "3rd-gen R", "3rd-gen L",
        "2nd-gen R", "2nd-gen L", "Main bronchi",
        "Lower trachea", "Mid trachea", "Upper trachea",
        "Glottis / Larynx",
    ]

    rows = []
    for i in range(n_bins):
        lo_b, hi_b = edges[i], edges[i + 1]
        mask_a = (pressure_a >= lo_b) & (pressure_a < hi_b)
        mask_b = (pressure_b >= lo_b) & (pressure_b < hi_b)

        rows.append({
            "Region":      labels_r[i],
            "Mean P — A (Pa)": float(pressure_a[mask_a].mean()) if mask_a.any() else np.nan,
            "Mean P — B (Pa)": float(pressure_b[mask_b].mean()) if mask_b.any() else np.nan,
            "Nodes — A": int(mask_a.sum()),
            "Nodes — B": int(mask_b.sum()),
        })

    reg_df = pd.DataFrame(rows).dropna()
    reg_df["Difference (Pa)"] = reg_df["Mean P — A (Pa)"] - reg_df["Mean P — B (Pa)"]

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Bar(
        name=f"Patient A ({label_a})",
        x=reg_df["Region"], y=reg_df["Mean P — A (Pa)"],
        marker_color="#4EB3D3", opacity=0.85,
    ))
    fig_reg.add_trace(go.Bar(
        name=f"Patient B ({label_b})",
        x=reg_df["Region"], y=reg_df["Mean P — B (Pa)"],
        marker_color="#E63946", opacity=0.85,
    ))
    fig_reg.update_layout(
        barmode="group",
        title="Mean pressure per anatomical region",
        yaxis_title="Mean pressure (Pa)",
        xaxis_title="Airway region (bronchi → glottis)",
        height=420, template="plotly_dark",
        legend=dict(x=0.75, y=0.95),
    )
    st.plotly_chart(fig_reg, use_container_width=True)

    # Difference chart
    fig_diff = px.bar(
        reg_df, x="Region", y="Difference (Pa)",
        color="Difference (Pa)",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title="Pressure difference per region (A − B)",
    )
    fig_diff.update_layout(height=340, template="plotly_dark")
    st.plotly_chart(fig_diff, use_container_width=True)

    st.dataframe(reg_df.round(2), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Parameter comparison
# ════════════════════════════════════════════════════════════════════════════
with tab_params:
    st.markdown("Geometry parameters for both patients — normalised to [0,1] DOE range.")

    cmp_data = []
    for i, col in enumerate(param_cols):
        norm_a = float((params_a[i] - low[i]) / (high[i] - low[i] + 1e-12))
        norm_b = float((params_b[i] - low[i]) / (high[i] - low[i] + 1e-12))
        cmp_data.append({
            "Parameter":     PARAM_LABELS.get(col, col),
            f"A ({label_a})": round(float(params_a[i]), 2),
            f"B ({label_b})": round(float(params_b[i]), 2),
            "A (norm)":      round(norm_a, 3),
            "B (norm)":      round(norm_b, 3),
            "Difference":    round(float(params_a[i] - params_b[i]), 3),
        })
    cmp_df = pd.DataFrame(cmp_data)

    # Radar / parallel-coordinates style — use horizontal bar chart of normalised values
    fig_par = go.Figure()
    fig_par.add_trace(go.Bar(
        name=f"Patient A ({label_a})",
        x=cmp_df["Parameter"], y=cmp_df["A (norm)"],
        marker_color="#4EB3D3", opacity=0.8,
    ))
    fig_par.add_trace(go.Bar(
        name=f"Patient B ({label_b})",
        x=cmp_df["Parameter"], y=cmp_df["B (norm)"],
        marker_color="#E63946", opacity=0.8,
    ))
    fig_par.update_layout(
        barmode="overlay",
        title="Normalised geometry parameters (0 = min DOE, 1 = max DOE)",
        yaxis_title="Normalised value",
        xaxis_tickangle=-45,
        height=420, template="plotly_dark",
        legend=dict(x=0.8, y=0.98),
    )
    st.plotly_chart(fig_par, use_container_width=True)

    # Key differences
    cmp_df["Abs diff (norm)"] = (cmp_df["A (norm)"] - cmp_df["B (norm)"]).abs()
    top_diff = cmp_df.nlargest(5, "Abs diff (norm)")[["Parameter", f"A ({label_a})", f"B ({label_b})", "Difference"]]

    st.markdown("#### Top 5 parameter differences")
    st.dataframe(top_diff, use_container_width=True, hide_index=True)
    st.dataframe(cmp_df.drop(columns=["A (norm)", "B (norm)", "Abs diff (norm)"]),
                 use_container_width=True, hide_index=True)
