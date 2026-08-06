"""
pages/7_Design_Space.py
========================
Design Landscape — understand the parameter space and what drives pressure.

Tabs
----
1. Parameter Sensitivity  — Pearson correlation of each param with mean pressure
2. Design Landscape       — 2D map of all 100 runs in POD score space, colored by pressure
3. Param–Pressure Matrix  — scatter grid of key params vs mean pressure
4. Sobol Indices          — variance-based global sensitivity (first-order + total)
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
    STRIDE, load_doe, load_all_pressures, get_param_cols,
    PARAM_LABELS, load_precomputed_pod,
)
from core.pod import compute_pod, modes_for_energy
from core.rbf import build_rbf, predict
from core.lhs import latin_hypercube
from core.i18n import t, lang_selector

st.set_page_config(page_title="Design Space", page_icon="🗺️", layout="wide")
lang_selector()
st.title(t("ds_title"))
st.caption(t("ds_caption"))

# ── Load data ─────────────────────────────────────────────────────────────────
doe_df     = load_doe()
param_cols = get_param_cols(doe_df)
params_raw = doe_df[param_cols].values.astype(float)

with st.spinner("Loading pressure snapshots…"):
    P_all = load_all_pressures(STRIDE)

mean_pressure_per_run = P_all.mean(axis=1)   # (100,) mean pressure per snapshot

# POD for pressure (precomputed or computed)
_pre = load_precomputed_pod("pressure")
if _pre is not None:
    mean_pres   = _pre["mean"]
    modes_pres  = _pre["modes"]
    scores_pres = _pre["scores"]
    sv_pres     = _pre["svalues"]
else:
    @st.cache_data(show_spinner=False)
    def _pres_pod(P):
        return compute_pod(P)
    with st.spinner("Computing pressure POD…"):
        mean_pres, modes_pres, scores_pres, sv_pres = _pres_pod(P_all)

# POD for geometry (precomputed or computed)
_pre_g = load_precomputed_pod("geometry")
if _pre_g is not None:
    scores_geo = _pre_g["scores"]
else:
    from core.data_io import load_all_coords
    @st.cache_data(show_spinner=False)
    def _geo_pod(X):
        return compute_pod(X)
    with st.spinner("Computing geometry POD…"):
        _, _, scores_geo, _ = _geo_pod(load_all_coords(STRIDE))

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_sens, tab_land, tab_matrix, tab_sobol = st.tabs([
    t("ds_tab_sens"),
    t("ds_tab_land"),
    t("ds_tab_grid"),
    t("ds_tab_sobol"),
])

# ── Tab 1: Parameter Sensitivity ──────────────────────────────────────────────
with tab_sens:
    st.subheader(t("ds_sub_sensitivity"))
    st.markdown(
        "Pearson correlation coefficient between each of the 26 geometry parameters "
        "and the **mean static pressure** across the 100 DOE snapshots.  "
        "Red = positive correlation (larger → higher pressure), "
        "blue = negative."
    )

    corr = np.array([
        float(np.corrcoef(params_raw[:, i], mean_pressure_per_run)[0, 1])
        for i in range(len(param_cols))
    ])
    labels = [PARAM_LABELS.get(c, c) for c in param_cols]

    sens_df = pd.DataFrame({
        "Parameter":   labels,
        "Raw name":    param_cols,
        "Correlation": corr,
        "AbsCorr":     np.abs(corr),
    }).sort_values("AbsCorr", ascending=True)

    fig_sens = px.bar(
        sens_df, x="Correlation", y="Parameter",
        orientation="h",
        color="Correlation",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title="Pearson Correlation: Parameter → Mean Pressure",
        labels={"Correlation": "r"},
        height=700,
    )
    fig_sens.update_layout(yaxis=dict(tickfont=dict(size=11)))
    st.plotly_chart(fig_sens, width='stretch')

    col_top, col_bot = st.columns(2)
    top3 = sens_df.nlargest(3, "AbsCorr")[["Parameter", "Correlation"]]
    with col_top:
        st.markdown("**Top 3 most influential parameters**")
        st.dataframe(top3.round(4), hide_index=True)

    # Also show correlation with first POD mode score
    corr_mode1 = np.array([
        float(np.corrcoef(params_raw[:, i], scores_pres[:, 0])[0, 1])
        for i in range(len(param_cols))
    ])
    sens_m1 = pd.DataFrame({
        "Parameter":   labels,
        "r with Mode 1": corr_mode1,
        "AbsCorr":     np.abs(corr_mode1),
    }).sort_values("AbsCorr", ascending=False)

    with col_bot:
        st.markdown("**Top 3 params correlated with Pressure Mode 1**")
        st.dataframe(sens_m1[["Parameter","r with Mode 1"]].head(3).round(4), hide_index=True)

    st.divider()
    st.markdown("**Full sensitivity table**")
    full_df = pd.DataFrame({
        "Parameter":          labels,
        "r (mean pressure)":  corr.round(4),
        "r (pressure mode 1)":corr_mode1.round(4),
    }).sort_values("r (mean pressure)", key=np.abs, ascending=False)
    st.dataframe(full_df, hide_index=True, width='stretch')


# ── Tab 2: Design Landscape ───────────────────────────────────────────────────
with tab_land:
    st.subheader("All 100 DOE runs mapped into the pressure POD space")
    st.markdown(
        "Each point is one CFD snapshot, plotted by its first two **pressure POD scores**. "
        "Color = mean static pressure.  "
        "Hover to see the snapshot number and key geometry parameters."
    )

    # Build hover text
    hover_texts = []
    for i, row in doe_df.iterrows():
        snap = int(row["snapshot_num"])
        key_params = ", ".join(
            f"{PARAM_LABELS.get(c,c).split('(')[0].strip()}: {row[c]:.1f}"
            for c in ["A_glotis", "d_trachea", "l_trachea", "teta_branch_trachea"]
        )
        hover_texts.append(f"Run {snap}<br>{key_params}<br>Mean P: {mean_pressure_per_run[i]:.1f} Pa")

    fig_land = go.Figure(go.Scatter(
        x=scores_pres[:, 0], y=scores_pres[:, 1],
        mode="markers+text",
        text=[str(int(s)) for s in doe_df["snapshot_num"]],
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(
            size=10,
            color=mean_pressure_per_run,
            colorscale="Jet",
            colorbar=dict(title="Mean P (Pa)", thickness=14),
            line=dict(width=0.5, color="white"),
        ),
        hovertext=hover_texts,
        hoverinfo="text",
    ))
    fig_land.update_layout(
        title="Design Landscape — Pressure POD Score Space",
        xaxis_title="Pressure POD Mode 1 score",
        yaxis_title="Pressure POD Mode 2 score",
        height=600,
        paper_bgcolor="rgb(15,17,25)", plot_bgcolor="rgb(20,22,30)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
    )
    st.plotly_chart(fig_land, width='stretch')

    # Same for geometry space
    st.subheader(t("ds_sub_landscape"))
    fig_geo_land = go.Figure(go.Scatter(
        x=scores_geo[:, 0], y=scores_geo[:, 1],
        mode="markers",
        marker=dict(
            size=10, color=mean_pressure_per_run,
            colorscale="Jet",
            colorbar=dict(title="Mean P (Pa)", thickness=14),
            line=dict(width=0.5, color="white"),
        ),
        hovertext=hover_texts, hoverinfo="text",
    ))
    fig_geo_land.update_layout(
        title="Design Landscape — Geometry POD Score Space",
        xaxis_title="Geometry POD Mode 1 score",
        yaxis_title="Geometry POD Mode 2 score",
        height=500,
        paper_bgcolor="rgb(15,17,25)", plot_bgcolor="rgb(20,22,30)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
    )
    st.plotly_chart(fig_geo_land, width='stretch')

    # Geometry vs Pressure POD scores scatter
    st.subheader("Geometry POD Mode 1 vs Pressure POD Mode 1")
    # Add a linear trendline manually
    x_cross = scores_geo[:, 0]
    y_cross = scores_pres[:, 0]
    m_cross, b_cross = np.polyfit(x_cross, y_cross, 1)
    x_line = np.linspace(x_cross.min(), x_cross.max(), 50)

    fig_cross = px.scatter(
        x=x_cross, y=y_cross,
        color=mean_pressure_per_run, color_continuous_scale="Jet",
        labels={"x": "Geo Mode 1", "y": "Pres Mode 1", "color": "Mean P (Pa)"},
        title="Cross-space correlation: does geometry Mode 1 predict pressure Mode 1?",
    )
    fig_cross.add_scatter(x=x_line, y=m_cross * x_line + b_cross,
                          mode="lines", line=dict(color="white", dash="dash"),
                          name=f"OLS (slope={m_cross:.3f})", showlegend=True)
    fig_cross.update_layout(height=420)
    st.plotly_chart(fig_cross, width='stretch')


# ── Tab 3: Param–Pressure Grid ────────────────────────────────────────────────
with tab_matrix:
    st.subheader("Key parameter vs mean pressure scatter plots")

    # Pick top 6 most correlated parameters
    top_params = pd.DataFrame({
        "col":    param_cols,
        "label":  labels,
        "abscorr": np.abs(corr),
    }).nlargest(6, "abscorr")

    cols_grid = st.columns(3)
    for idx, (_, row_p) in enumerate(top_params.iterrows()):
        col = row_p["col"]
        lbl = row_p["label"]
        i = param_cols.index(col)
        with cols_grid[idx % 3]:
            xd = params_raw[:, i]
            yd = mean_pressure_per_run
            m, b = np.polyfit(xd, yd, 1)
            x_fit = np.linspace(xd.min(), xd.max(), 40)
            fig_s = px.scatter(
                x=xd, y=yd,
                labels={"x": lbl, "y": "Mean P (Pa)"},
                color=yd, color_continuous_scale="Jet",
                title=lbl.split("(")[0].strip(),
            )
            fig_s.add_scatter(x=x_fit, y=m * x_fit + b,
                              mode="lines", line=dict(color="white", dash="dash"),
                              showlegend=False)
            fig_s.update_layout(
                height=300, margin=dict(l=10, r=10, b=30, t=40),
                showlegend=False,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_s, width='stretch')

    # Full parameter correlation heatmap
    st.subheader("Parameter–parameter correlation matrix")
    corr_matrix = np.corrcoef(params_raw.T)
    fig_heat = px.imshow(
        corr_matrix,
        x=labels, y=labels,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        zmin=-1, zmax=1,
        title="Pearson correlation between all 26 DOE parameters",
        height=700,
    )
    fig_heat.update_layout(
        xaxis=dict(tickfont=dict(size=9), tickangle=45),
        yaxis=dict(tickfont=dict(size=9)),
    )
    st.plotly_chart(fig_heat, width='stretch')


# ── Tab 4: Sobol Sensitivity Indices ─────────────────────────────────────────
with tab_sobol:
    st.subheader(t("ds_sub_sobol"))
    st.markdown("""
    Sobol indices measure **how much of the output variance** is explained by each input parameter.
    Unlike Pearson correlation (linear only), Sobol captures **non-linear and interaction effects**.

    - **First-order S₁**: variance explained by parameter alone
    - **Total Sₜ**: variance explained including all interactions with other parameters
    - **Sₜ − S₁**: the interaction contribution

    Uses the **Saltelli (2010)** estimator with LHS + RBF surrogate — no extra CFD needed.
    """)

    col1, col2 = st.columns(2)
    N_sobol = col1.select_slider(
        t("ds_sample_n"),
        options=[128, 256, 512, 1024], value=512, key="sobol_N"
    )
    sobol_output = col2.radio(
        t("ds_output_var"),
        [t("ds_mean_pres"), t("ds_delta_p")],
        key="sobol_out", horizontal=True,
    )

    st.caption(
        f"Requires {N_sobol * (len(param_cols) + 2):,} RBF evaluations "
        f"({N_sobol} × ({len(param_cols)} params + 2)). "
        f"Each takes <1 ms — total ~{N_sobol*(len(param_cols)+2)/1000:.1f}s."
    )

    if st.button(t("ds_btn_sobol"), type="primary", key="run_sobol"):

        # ── Build RBF surrogate ──────────────────────────────────────────────
        params_norm_s = (params_raw - params_raw.min(axis=0)) / (
            params_raw.max(axis=0) - params_raw.min(axis=0) + 1e-12
        )
        lo_s = params_raw.min(axis=0)
        hi_s = params_raw.max(axis=0)

        k_s = modes_for_energy(scores_pres.std(axis=0) if hasattr(scores_pres, 'std')
                               else np.ones(scores_pres.shape[1]), 0.99)
        # Use precomputed scores directly
        k_s = min(modes_for_energy(
            np.array([np.linalg.norm(scores_pres[:, i]) for i in range(scores_pres.shape[1])]),
            0.99
        ), scores_pres.shape[1])

        rbf_s = build_rbf(params_norm_s, scores_pres[:, :k_s], kernel="thin_plate_spline")

        def evaluate_surrogate(X_norm):
            """Evaluate RBF → reconstruct → compute scalar output."""
            sc = predict(rbf_s, X_norm)
            results = np.empty(len(X_norm))
            for i in range(len(X_norm)):
                field = mean_pressure_per_run.mean() + (
                    (mean_pres + modes_pres[:, :k_s] @ sc[i]) - mean_pres
                ).mean()
                # Simpler: just use predicted mean pressure
                pred_field = mean_pres + modes_pres[:, :k_s] @ sc[i]
                if sobol_output == t("ds_mean_pres"):
                    results[i] = float(pred_field.mean())
                else:
                    results[i] = float(pred_field.max() - pred_field.min())
            return results

        # ── Saltelli estimator ───────────────────────────────────────────────
        with st.spinner(f"Running Saltelli estimator (N={N_sobol}, d={len(param_cols)})…"):
            rng = np.random.default_rng(42)
            d   = len(param_cols)

            # Two independent samples in [0,1]^d
            A_raw = latin_hypercube(N_sobol, lo_s, hi_s, seed=42)
            B_raw = latin_hypercube(N_sobol, lo_s, hi_s, seed=123)
            A_norm = (A_raw - lo_s) / (hi_s - lo_s + 1e-12)
            B_norm = (B_raw - lo_s) / (hi_s - lo_s + 1e-12)

            fA = evaluate_surrogate(A_norm)
            fB = evaluate_surrogate(B_norm)
            var_Y = float(np.var(np.concatenate([fA, fB])))

            S1 = np.zeros(d)
            ST = np.zeros(d)

            prog = st.progress(0, text="Computing per-parameter Sobol indices…")
            for i in range(d):
                # A_Bi: copy of A with column i from B
                A_Bi      = A_norm.copy()
                A_Bi[:, i]= B_norm[:, i]
                fABi      = evaluate_surrogate(A_Bi)

                # Saltelli 2010 estimators
                S1[i] = float(np.mean(fB * (fABi - fA))) / (var_Y + 1e-12)
                ST[i] = float(np.mean((fA - fABi) ** 2)) / (2 * var_Y + 1e-12)
                prog.progress((i + 1) / d, text=f"Parameter {i+1}/{d}")
            prog.empty()

        # ── Clamp to [0, 1] (estimator can give small negatives) ─────────────
        S1 = np.clip(S1, 0, 1)
        ST = np.clip(ST, 0, 1)

        sobol_df = pd.DataFrame({
            "Parameter":   labels,
            "Raw name":    param_cols,
            "S1 (first-order)": S1.round(4),
            "ST (total)":       ST.round(4),
            "Interactions":     (ST - S1).clip(0).round(4),
        }).sort_values("ST (total)", ascending=False)

        # ── Grouped bar chart ────────────────────────────────────────────────
        top_n = st.slider(t("ds_top_n"), 5, len(param_cols), 15, key="sobol_topn")
        plot_df = sobol_df.head(top_n)

        fig_sobol = go.Figure()
        fig_sobol.add_bar(
            x=plot_df["Parameter"], y=plot_df["S1 (first-order)"],
            name="S₁ (first-order)", marker_color="#4EB3D3",
        )
        fig_sobol.add_bar(
            x=plot_df["Parameter"], y=plot_df["Interactions"],
            name="Interaction contribution (Sₜ−S₁)", marker_color="#F4A261",
        )
        fig_sobol.update_layout(
            barmode="stack",
            xaxis=dict(tickangle=45, tickfont=dict(size=10)),
            yaxis_title="Sobol Index",
            title=f"Sobol Sensitivity Indices — output: {sobol_output} (N={N_sobol})",
            legend=dict(x=0.6, y=0.98),
            height=500,
        )
        st.plotly_chart(fig_sobol, width='stretch')

        # ── Comparison: Pearson vs Sobol ─────────────────────────────────────
        st.subheader("Pearson correlation vs Sobol S₁ — do they agree?")
        comp_df = pd.DataFrame({
            "Parameter":  sobol_df["Parameter"],
            "|Pearson r|": np.abs(corr)[
                [param_cols.index(c) for c in sobol_df["Raw name"]]
            ].round(4),
            "Sobol S₁":   sobol_df["S1 (first-order)"].values,
        }).head(15)

        fig_comp = px.scatter(
            comp_df, x="|Pearson r|", y="Sobol S₁",
            text="Parameter", size_max=10,
            title="Pearson |r| vs Sobol S₁ — divergence reveals non-linear effects",
        )
        fig_comp.update_traces(textposition="top center", textfont=dict(size=9))
        fig_comp.update_layout(height=420)
        st.plotly_chart(fig_comp, width='stretch')

        st.caption(
            "Points above the diagonal = parameter has more influence than Pearson suggests "
            "(non-linear effects). Points on the diagonal = purely linear relationship."
        )

        st.divider()
        st.markdown("**Full Sobol indices table**")
        st.dataframe(sobol_df.reset_index(drop=True), hide_index=True, width='stretch')
