"""
pages/14_Drug_Deposition.py
===========================
Drug Particle Deposition Simulator — Human Airways Digital Twin.

Physics model (standard 1-D trumpet model adapted to 3-D point cloud)
----------------------------------------------------------------------
Local airflow velocity is derived from the RBF-predicted pressure field:
  v_local ∝ inlet_vel × (glottis_area / local_cross_section) / n_branches_at_depth

Three deposition mechanisms:
  1. Inertial impaction  — Stokes number at bends / bifurcations (large particles)
  2. Gravitational sedimentation — terminal settling in slow-flow segments (1-10 µm)
  3. Brownian diffusion  — thermal motion in narrow segments (<0.5 µm)

Total deposition at each node = 1 − ∏(1 − P_mech)

Inputs
------
  Patient: DOE snapshot (1-100) — uses real CFD pressure field
  Particle: size (µm), injection velocity (m/s)
  Airflow:  inlet velocity (m/s) from DOE or manual override
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
    load_ref_coords,
    load_doe,
    get_param_cols,
    load_precomputed_pod,
    load_all_pressures,
)
from core.pod import compute_pod, modes_for_energy, reconstruct
from core.rbf import build_rbf, predict
from core.i18n import t, lang_selector

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Drug Deposition", page_icon="💊", layout="wide")
lang_selector()
st.title(t("drug_title"))
st.caption(t("drug_caption"))

# ── Physical constants ────────────────────────────────────────────────────────
MU    = 1.81e-5   # Pa·s — air dynamic viscosity at 37 °C
G     = 9.81      # m/s²
KB    = 1.38e-23  # J/K — Boltzmann constant
T     = 310.0     # K — body temperature
LAM   = 66e-9     # m — mean free path of air at body conditions
RHO_P = 1200.0    # kg/m³ — typical drug particle density (default)

# Anatomical label colours for regional chart
REGION_COLORS = {
    "glottis / epiglottis": "#E63946",
    "larynx":               "#F4A261",
    "trachea":              "#FFC300",
    "left bronchi":         "#4EB3D3",
    "right bronchi":        "#2A9D8F",
    "sub-branches":         "#06D6A0",
}

# ── Load pressure POD and DOE ─────────────────────────────────────────────────
doe_pressure_path = Path(__file__).parent.parent / "Pressure" / "doe_pressure.csv"
doe_pres_df = pd.read_csv(doe_pressure_path, sep=";") if doe_pressure_path.exists() else None

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
def get_rbf(k, kern="thin_plate_spline"):
    return build_rbf(params_norm, scores_pres[:, :k], kernel=kern)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header(t("drug_hdr_patient"))
    snap = st.slider(t("lbl_snapshot"), 1, 100, 1,
                     help="Pick one of 100 real CFD patient geometries.")

    # Airflow from DOE if available
    default_inlet = 2.0
    if doe_pres_df is not None and "inlet_vel" in doe_pres_df.columns:
        default_inlet = float(doe_pres_df["inlet_vel"].iloc[snap - 1])

    st.divider()
    st.header(t("drug_hdr_airflow"))
    inlet_vel = st.slider(
        t("drug_velocity"), 0.3, 5.0, float(round(default_inlet, 2)), 0.05,
        help="Resting breathing ~0.5–1.5 m/s; forced inhalation up to 4–5 m/s."
    )

    st.divider()
    st.header(t("drug_hdr_particle"))
    part_size_um = st.slider(
        t("drug_diameter"), 0.5, 20.0, 5.0, 0.5,
        help="1–5 µm optimal for bronchial deposition. >10 µm deposits in throat."
    )
    part_inj_vel = st.slider(
        t("drug_inject_v"), 0.1, 10.0, 3.0, 0.1,
        help="Speed at which particles are injected from inhaler device."
    )
    rho_p = st.number_input(
        t("drug_density"), 800, 2000, 1200, 100,
        help="Typical pharmaceutical aerosol: 1000–1300 kg/m³."
    )
    st.divider()
    colorscale = st.selectbox(
        t("lbl_colour_map"), ["Hot", "Jet", "Plasma", "RdYlGn_r"], index=0,
        help="Hot / Jet show high deposition as red/yellow."
    )


# ── Load pressure field ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_pressure(snap_n: int) -> np.ndarray:
    return load_pressure_snapshot(snap_n, STRIDE)

pressure = get_pressure(snap)

# ── DOE geometry params for trachea diameter ──────────────────────────────────
row       = doe_df.iloc[snap - 1]
d_trachea = float(row.get("d_trachea", 16.5)) * 1e-3   # mm → m
A_glotis  = float(row.get("A_glotis", 150.0))           # mm²
# Effective glottis diameter
D_glotis  = np.sqrt(4 * A_glotis * 1e-6 / np.pi)        # m

# ── Physics model ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_deposition(
    pressure: np.ndarray,
    inlet_vel: float,
    part_size_um: float,
    part_inj_vel: float,
    d_trachea_m: float,
    D_glotis_m: float,
    rho_p: float,
) -> dict:
    d_p = part_size_um * 1e-6   # µm → m

    # Cunningham slip correction (important for < 1 µm)
    C_c = 1.0 + (LAM / d_p) * (2.34 + 1.05 * np.exp(-0.39 * d_p / LAM))

    # Depth proxy: 0 = glottis (high pressure) → 1 = terminal bronchi (low pressure)
    p_min, p_max = pressure.min(), pressure.max()
    depth = (p_max - pressure) / (p_max - p_min + 1e-12)
    depth = np.clip(depth, 0, 1)

    # Local diameter: trachea (d_trachea) → terminal branches (~2 mm)
    D_terminal = 0.002   # 2 mm at 3rd-generation bronchi
    log_ratio   = np.log(D_terminal / (d_trachea_m + 1e-9))
    D_local     = d_trachea_m * np.exp(depth * log_ratio)
    D_local     = np.clip(D_local, D_terminal, d_trachea_m * 1.5)

    # Number of parallel bronchial pathways at each depth level
    # ~1 at trachea → ~8 at 3rd generation
    n_branches = np.maximum(1.0, 2.0 ** (depth * 3.0))

    # Local velocity (continuity + branching)
    v_local = inlet_vel * (D_glotis_m / d_trachea_m) ** 2 * (d_trachea_m / D_local) ** 2 / n_branches
    v_local = np.clip(v_local, 0.01, inlet_vel * 8)

    # Stokes number  Stk = ρ_p d_p² v / (18 µ D)
    Stk = (rho_p * d_p ** 2 * v_local) / (18.0 * MU * D_local)

    # ── Mechanism 1: inertial impaction at bends / bifurcations ──────────────
    # Chan & Lippmann (1980) correlation for 90° bends: P_imp = f(Stk)
    # At typical 30°–45° bifurcations: Stk_50 ~ 0.24–0.35
    Stk_50 = 0.24
    P_imp_bend = Stk / (Stk + Stk_50)

    # Bifurcations cluster near depth ≈ 0.25, 0.50, 0.75
    bif = (
        np.exp(-((depth - 0.25) ** 2) / 0.008)
        + np.exp(-((depth - 0.50) ** 2) / 0.008)
        + np.exp(-((depth - 0.75) ** 2) / 0.008)
    )
    bif /= bif.max() + 1e-12
    P_imp = P_imp_bend * (0.25 + 0.75 * bif)

    # ── Mechanism 2: gravitational sedimentation ─────────────────────────────
    V_ts    = (rho_p * d_p ** 2 * G * C_c) / (18.0 * MU)
    L_seg   = D_local * 6.0   # tube length ~ 6 × diameter (aspect ratio)
    P_sed   = 1.0 - np.exp(-L_seg * V_ts / (v_local * D_local + 1e-15))
    P_sed   = np.clip(P_sed, 0, 0.95)

    # ── Mechanism 3: Brownian diffusion ─────────────────────────────────────
    D_diff = (KB * T * C_c) / (3.0 * np.pi * MU * d_p)
    r_tube = D_local / 2.0
    P_diff = 1.0 - np.exp(-np.pi * D_diff * L_seg / (v_local * r_tube ** 2 + 1e-20))
    P_diff = np.clip(P_diff, 0, 0.95)

    # ── Combined deposition probability ─────────────────────────────────────
    P_dep = 1.0 - (1.0 - P_imp) * (1.0 - P_sed) * (1.0 - P_diff)
    P_dep = np.clip(P_dep, 0, 1)

    # ── Terminal settling velocity and diffusivity (for display) ──────────
    return {
        "P_dep":   P_dep,
        "P_imp":   P_imp,
        "P_sed":   P_sed,
        "P_diff":  P_diff,
        "depth":   depth,
        "Stk":     Stk,
        "v_local": v_local,
        "D_local": D_local,
        "V_ts":    V_ts,
        "D_diff":  D_diff,
    }


result = compute_deposition(
    pressure, inlet_vel, part_size_um, part_inj_vel,
    d_trachea, D_glotis, float(rho_p)
)
P_dep   = result["P_dep"]
depth   = result["depth"]
v_local = result["v_local"]
D_local = result["D_local"]

# ── Summary metrics ───────────────────────────────────────────────────────────
# Upper airway: depth < 0.35 | Trachea: 0.35–0.55 | Bronchi: > 0.55
mask_upper   = depth < 0.35
mask_trachea = (depth >= 0.35) & (depth < 0.55)
mask_bronchi = depth >= 0.55

dep_upper   = P_dep[mask_upper].mean()   if mask_upper.any()   else 0.0
dep_trachea = P_dep[mask_trachea].mean() if mask_trachea.any() else 0.0
dep_bronchi = P_dep[mask_bronchi].mean() if mask_bronchi.any() else 0.0
dep_overall = P_dep.mean()

# Drug delivery efficiency: fraction that reaches bronchi
# (lower deposition in upper airway = better drug delivery)
efficiency = dep_bronchi / (dep_upper + dep_trachea + dep_bronchi + 1e-9)

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric(t("drug_overall"), f"{dep_overall*100:.1f}%")
mc2.metric(t("drug_upper"), f"{dep_upper*100:.1f}%",
           delta=f"{'high — wasted' if dep_upper > 0.4 else 'ok'}",
           delta_color="inverse")
mc3.metric(t("drug_trachea"), f"{dep_trachea*100:.1f}%")
mc4.metric(t("drug_bronchi"), f"{dep_bronchi*100:.1f}%",
           delta=f"{'good target' if dep_bronchi > 0.3 else 'low'}",
           delta_color="normal")
mc5.metric(t("drug_efficiency"), f"{efficiency*100:.0f}%",
           help="Fraction of deposited particles that reach the bronchi (vs wasted in upper airway).")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_map, tab_region, tab_sweep, tab_physics = st.tabs([
    t("drug_tab_map"),
    t("drug_tab_region"),
    t("drug_tab_optimal"),
    t("drug_tab_physics"),
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — 3D deposition heatmap
# ════════════════════════════════════════════════════════════════════════════
with tab_map:
    c_left, c_right = st.columns([3, 1])

    with c_right:
        st.markdown("#### What you're seeing")
        st.markdown(f"""
        **Particle size:** {part_size_um:.1f} µm
        **Injection velocity:** {part_inj_vel:.1f} m/s
        **Inlet airflow:** {inlet_vel:.2f} m/s

        **Colour:** deposition probability per node.
        Bright = particles are likely to stick here.
        Dark = particles pass through.

        **Dominant mechanism at {part_size_um:.1f} µm:**
        {"🔴 Inertial impaction" if part_size_um > 8 else
         "🟡 Sedimentation (+ impaction)" if part_size_um > 2 else
         "🔵 Brownian diffusion"}
        """)

        st.markdown("---")
        st.markdown(t("drug_mech_breakdown"))
        st.metric(t("drug_impaction"),    f"{result['P_imp'].mean()*100:.1f}%")
        st.metric(t("drug_sedimentation"),f"{result['P_sed'].mean()*100:.1f}%")
        st.metric(t("drug_diffusion"),    f"{result['P_diff'].mean()*100:.1f}%")

    with c_left:
        fig_dep = go.Figure(go.Scatter3d(
            x=ref_coords[:, 0], y=ref_coords[:, 1], z=ref_coords[:, 2],
            mode="markers",
            marker=dict(
                size=2.5,
                color=P_dep,
                colorscale=colorscale,
                cmin=0, cmax=1,
                colorbar=dict(
                    title="Deposition<br>probability",
                    thickness=14,
                    tickformat=".0%",
                ),
                opacity=0.85,
            ),
            hovertemplate=(
                "Deposition: %{marker.color:.1%}<extra></extra>"
            ),
        ))
        fig_dep.update_layout(
            scene=dict(
                aspectmode="data",
                bgcolor="rgb(10, 12, 18)",
                xaxis=dict(title="X (m)", showbackground=False, gridcolor="#222"),
                yaxis=dict(title="Y (m)", showbackground=False, gridcolor="#222"),
                zaxis=dict(title="Z (m)", showbackground=False, gridcolor="#222"),
            ),
            paper_bgcolor="rgb(10, 12, 18)",
            font=dict(color="white"),
            height=680,
            margin=dict(l=0, r=0, b=0, t=30),
            title=f"Deposition map — Snapshot {snap} | {part_size_um:.1f} µm particles",
        )
        st.plotly_chart(fig_dep, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Regional breakdown
# ════════════════════════════════════════════════════════════════════════════
with tab_region:
    n_bins = 12
    edges  = np.linspace(0, 1, n_bins + 1)
    labels = [
        "Glottis/Epiglottis",
        "Larynx", "Upper Trachea",
        "Mid Trachea", "Lower Trachea",
        "Main Bronchi", "L-Main", "R-Main",
        "2nd-L Branches", "2nd-R Branches",
        "3rd-L Branches", "3rd-R Branches",
    ]
    dep_by_region = []
    for i in range(n_bins):
        mask = (depth >= edges[i]) & (depth < edges[i + 1])
        if mask.any():
            dep_by_region.append({
                "Region":            labels[i],
                "Depth":             (edges[i] + edges[i + 1]) / 2,
                "Deposition (%)":    float(P_dep[mask].mean() * 100),
                "Impaction (%)":     float(result["P_imp"][mask].mean() * 100),
                "Sedimentation (%)": float(result["P_sed"][mask].mean() * 100),
                "Diffusion (%)":     float(result["P_diff"][mask].mean() * 100),
                "Mean velocity (m/s)": float(v_local[mask].mean()),
                "Mean diameter (mm)":  float(D_local[mask].mean() * 1e3),
            })

    reg_df = pd.DataFrame(dep_by_region)

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Bar(
        name="Impaction",
        x=reg_df["Region"], y=reg_df["Impaction (%)"],
        marker_color="#E63946", opacity=0.9,
    ))
    fig_reg.add_trace(go.Bar(
        name="Sedimentation",
        x=reg_df["Region"], y=reg_df["Sedimentation (%)"],
        marker_color="#F4A261", opacity=0.9,
    ))
    fig_reg.add_trace(go.Bar(
        name="Diffusion",
        x=reg_df["Region"], y=reg_df["Diffusion (%)"],
        marker_color="#4EB3D3", opacity=0.9,
    ))
    fig_reg.update_layout(
        barmode="group",
        title=f"Deposition mechanism by airway region — {part_size_um:.1f} µm particles",
        yaxis_title="Mean deposition probability (%)",
        xaxis_title="Anatomical region (inlet → terminal bronchi)",
        height=420, template="plotly_dark",
        legend=dict(x=0.75, y=0.95),
    )
    st.plotly_chart(fig_reg, use_container_width=True)

    # Total deposition per region as horizontal bar
    fig_total = px.bar(
        reg_df, y="Region", x="Deposition (%)",
        orientation="h",
        color="Deposition (%)",
        color_continuous_scale="Hot_r",
        title="Total deposition probability per region",
        labels={"Deposition (%)": "Mean deposition (%)"},
    )
    fig_total.update_layout(height=380, template="plotly_dark")
    st.plotly_chart(fig_total, use_container_width=True)

    st.dataframe(
        reg_df.round(2),
        use_container_width=True, hide_index=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Optimal particle size sweep
# ════════════════════════════════════════════════════════════════════════════
with tab_sweep:
    st.markdown(t("drug_optimal_intro"))

    @st.cache_data(show_spinner=False)
    def size_sweep(pressure_hash, inlet_vel, part_inj_vel, d_trachea_m, D_glotis_m, rho_p):
        sizes = np.linspace(0.5, 20.0, 60)
        rows  = []
        for s in sizes:
            r = compute_deposition(
                pressure, inlet_vel, s, part_inj_vel,
                d_trachea_m, D_glotis_m, float(rho_p)
            )
            dep   = r["depth"]
            P     = r["P_dep"]
            m_upper   = dep < 0.35
            m_trachea = (dep >= 0.35) & (dep < 0.55)
            m_bronchi = dep >= 0.55

            dep_u = P[m_upper].mean()   if m_upper.any()   else 0.0
            dep_t = P[m_trachea].mean() if m_trachea.any() else 0.0
            dep_b = P[m_bronchi].mean() if m_bronchi.any() else 0.0
            eff   = dep_b / (dep_u + dep_t + dep_b + 1e-9)

            rows.append({
                "size_um":         s,
                "Overall (%)":     P.mean() * 100,
                "Upper airway (%)":dep_u * 100,
                "Trachea (%)":     dep_t * 100,
                "Bronchi (%)":     dep_b * 100,
                "Efficiency (%)":  eff * 100,
                "Impaction (%)":   r["P_imp"].mean() * 100,
                "Sedimentation (%)": r["P_sed"].mean() * 100,
                "Diffusion (%)":   r["P_diff"].mean() * 100,
            })
        return pd.DataFrame(rows)

    sweep_df = size_sweep(
        int(pressure.mean() * 1000),
        inlet_vel, part_inj_vel, d_trachea, D_glotis, int(rho_p)
    )

    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df["size_um"], y=sweep_df["Bronchi (%)"],
        name="Bronchi deposition", line=dict(color="#06D6A0", width=2.5),
        fill="tozeroy", fillcolor="rgba(6,214,160,0.15)",
    ))
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df["size_um"], y=sweep_df["Upper airway (%)"],
        name="Upper airway (wasted)", line=dict(color="#E63946", width=2),
    ))
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df["size_um"], y=sweep_df["Trachea (%)"],
        name="Trachea", line=dict(color="#F4A261", width=2),
    ))
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df["size_um"], y=sweep_df["Efficiency (%)"],
        name="Delivery efficiency", line=dict(color="white", width=2, dash="dot"),
    ))
    # Mark current selection
    fig_sweep.add_vline(
        x=part_size_um, line_dash="dash", line_color="yellow",
        annotation_text=f"Current: {part_size_um:.1f} µm",
        annotation_position="top right",
    )
    fig_sweep.update_layout(
        title="Deposition vs particle size (airflow: {:.2f} m/s, patient: snapshot {})".format(
            inlet_vel, snap
        ),
        xaxis_title="Particle diameter (µm)",
        yaxis_title="Mean deposition probability (%)",
        height=440, template="plotly_dark",
        legend=dict(x=0.65, y=0.95),
    )
    st.plotly_chart(fig_sweep, use_container_width=True)

    # Mechanism breakdown over size
    fig_mech = go.Figure()
    fig_mech.add_trace(go.Scatter(
        x=sweep_df["size_um"], y=sweep_df["Impaction (%)"],
        name="Inertial impaction", line=dict(color="#E63946", width=2),
    ))
    fig_mech.add_trace(go.Scatter(
        x=sweep_df["size_um"], y=sweep_df["Sedimentation (%)"],
        name="Sedimentation", line=dict(color="#F4A261", width=2),
    ))
    fig_mech.add_trace(go.Scatter(
        x=sweep_df["size_um"], y=sweep_df["Diffusion (%)"],
        name="Brownian diffusion", line=dict(color="#4EB3D3", width=2),
    ))
    fig_mech.add_vline(x=part_size_um, line_dash="dash", line_color="yellow")
    fig_mech.update_layout(
        title="Dominant mechanism vs particle size",
        xaxis_title="Particle diameter (µm)",
        yaxis_title="Mean probability (%)",
        height=360, template="plotly_dark",
        legend=dict(x=0.65, y=0.95),
    )
    st.plotly_chart(fig_mech, use_container_width=True)

    # Optimal recommendation
    best_idx = sweep_df["Efficiency (%)"].idxmax()
    best_size = sweep_df.loc[best_idx, "size_um"]
    st.success(
        f"**Optimal particle size for this patient (snapshot {snap}): "
        f"{best_size:.1f} µm** — maximises bronchial delivery efficiency "
        f"({sweep_df.loc[best_idx, 'Efficiency (%)']:.0f}%)."
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Physics model explanation
# ════════════════════════════════════════════════════════════════════════════
with tab_physics:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Deposition mechanisms")
        st.markdown(r"""
        #### 1. Inertial Impaction
        When a particle is too heavy to follow streamline bends at bifurcations,
        it continues in a straight line and hits the airway wall.

        **Stokes number:**
        $$\text{Stk} = \frac{\rho_p \, d_p^2 \, v}{18 \, \mu \, D}$$

        Deposition at a bifurcation (Chan & Lippmann 1980):
        $$P_\text{imp} = \frac{\text{Stk}}{\text{Stk} + \text{Stk}_{50}}$$

        High Stk (large/fast particles) → **deposits in upper airway**.

        #### 2. Gravitational Sedimentation
        Particles settle under gravity in slow-flow segments:

        $$V_{ts} = \frac{\rho_p \, d_p^2 \, g \, C_c}{18 \, \mu}$$
        $$P_\text{sed} = 1 - e^{-L \, V_{ts} / (v \, D)}$$

        Dominant for **1–10 µm** particles in lower bronchi.

        #### 3. Brownian Diffusion
        Sub-micron particles undergo thermal motion and diffuse to the wall:

        $$D_\text{Brown} = \frac{k_B T C_c}{3 \pi \mu d_p}$$

        Dominant for **< 0.5 µm** particles everywhere.
        """)

    with c2:
        st.subheader("Model simplifications")
        st.markdown(r"""
        #### Local velocity estimate
        No actual velocity field is stored — only the pressure field. We derive
        local velocity using conservation of mass + branching geometry:

        $$v_\text{local} \approx v_\text{inlet} \times \frac{A_\text{glottis}}{A_\text{local}} \times \frac{1}{N_\text{branches}(\text{depth})}$$

        where depth ∈ [0,1] is mapped from the pressure field (low pressure = deep).

        #### Local diameter estimate
        $$D_\text{local} = D_\text{trachea} \times \left(\frac{D_\text{terminal}}{D_\text{trachea}}\right)^{\text{depth}}$$

        where $D_\text{terminal}$ = 2 mm (3rd-generation bronchi).

        #### Cunningham correction
        For particles approaching the mean free path of air:
        $$C_c = 1 + \frac{\lambda}{d_p}\left(2.34 + 1.05 \, e^{-0.39 \, d_p/\lambda}\right)$$

        with $\lambda$ = 66 nm at body conditions.

        #### Combined deposition
        $$P_\text{total} = 1 - (1-P_\text{imp})(1-P_\text{sed})(1-P_\text{diff})$$

        This assumes mechanisms act independently at each airway segment.
        """)

    st.info(
        "This model is a spatially-resolved adaptation of the **ICRP trumpet model** "
        "for respiratory deposition. It is physically grounded but simplified — actual "
        "deposition requires full Lagrangian particle tracking in the CFD velocity field. "
        "Use this as a qualitative guide and comparative tool across patients/particle sizes."
    )
