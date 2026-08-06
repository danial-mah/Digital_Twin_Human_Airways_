"""
pages/8_Delivery_Packages.py
============================
Overview of the three data delivery packages produced by this digital twin:

  VR Archive  — standalone WebXR viewer for any browser / Meta Quest
  HELYX       — STL meshes + parameter tables for CFD pre-processing
  Domino      — ML-ready dataset (params, POD bases, pressure fields)

Shows package status, file contents, and live data previews where available.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent

from core.i18n import t, lang_selector

EXPORT_DIR   = ROOT / "export"
VR_DIR       = EXPORT_DIR / "vr_archive"
VR_ZIP       = EXPORT_DIR / "vr_archive.zip"
HELYX_DIR    = EXPORT_DIR / "helyx"
HELYX_ZIP    = EXPORT_DIR / "helyx_package.zip"
DOMINO_DIR   = EXPORT_DIR / "domino"
DOMINO_ZIP   = EXPORT_DIR / "domino_dataset.zip"
MESH_DIR     = EXPORT_DIR / "mesh"

st.set_page_config(page_title="Delivery Packages", page_icon="📦", layout="wide")
lang_selector()
st.title(t("pkg_title"))
st.caption(t("pkg_caption"))

# ── Status overview ────────────────────────────────────────────────────────────

vr_ready     = VR_DIR.exists() and any(VR_DIR.glob("data/*.bin"))
helyx_ready  = HELYX_DIR.exists() and any(HELYX_DIR.rglob("*.stl"))
domino_ready = (DOMINO_DIR / "shapes_1000_summary.csv").exists()

c1, c2, c3 = st.columns(3)
with c1:
    if vr_ready:
        st.success(t("pkg_vr_ready"))
    else:
        st.warning(t("pkg_vr_run"))
    st.caption(t("pkg_vr_cap"))

with c2:
    if helyx_ready:
        st.success(t("pkg_helyx_ready"))
    else:
        st.warning(t("pkg_helyx_run"))
    st.caption(t("pkg_helyx_cap"))

with c3:
    if domino_ready:
        st.success(t("pkg_domino_ready"))
    else:
        st.warning(t("pkg_domino_run"))
    st.caption(t("pkg_domino_cap"))

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_vr, tab_helyx, tab_domino = st.tabs([
    t("pkg_tab_vr"),
    t("pkg_tab_helyx"),
    t("pkg_tab_domino"),
])


# ══════════════════════════════════════════════════════════════════════════════
# VR Archive tab
# ══════════════════════════════════════════════════════════════════════════════
with tab_vr:
    left, right = st.columns([1, 1])

    with left:
        st.subheader(t("pkg_what"))
        st.markdown("""
        A self-contained archive that lets anyone explore the airway pressure
        field in 3D — **no Streamlit server, no Python, no installation required.**

        Works on:
        - Any desktop browser (Chrome, Firefox, Edge)
        - **Meta Quest** browser (immersive VR — walk inside the trachea)
        - Any WebXR-compatible device on the same WiFi

        #### How to use it
        ```
        1. Unzip  vr_archive.zip
        2. Open a terminal in the folder
        3. python serve.py
        4. Open  http://localhost:8765  in any browser
        ```
        For **Meta Quest**: connect to the same WiFi network, open the Quest
        browser, and navigate to `http://YOUR-PC-IP:8765`.
        The "Enter VR" button appears automatically on WebXR-compatible devices.

        #### Controls
        | Action | Orbit mode | Fly mode (press F) |
        |--------|-----------|---------------------|
        | Look around | Left-drag | Left-drag |
        | Move | — | WASD / arrow keys |
        | Up / Down | — | Q / E |
        | Zoom / Speed | Scroll | Scroll |
        """)

    with right:
        st.subheader(t("pkg_contents"))

        if vr_ready:
            # Read manifest
            manifest_path = VR_DIR / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    mf = json.load(f)
                snaps = mf.get("snapshots", [])
                stride = mf.get("stride", 50)

                m1, m2, m3 = st.columns(3)
                m1.metric(t("pkg_snapshots"), len(snaps))
                m2.metric(t("pkg_nodes"), f"{snaps[0]['n_pts']:,}" if snaps else "—")
                m3.metric(t("pkg_stride"), stride)

                # Pressure range across all snapshots
                dp_vals = [s["dp_Pa"] for s in snaps if "dp_Pa" in s]
                if dp_vals:
                    fig_vr = px.histogram(
                        x=dp_vals, nbins=20,
                        labels={"x": "DeltaP (Pa)"},
                        title="DeltaP distribution across archived snapshots",
                        color_discrete_sequence=["#4EB3D3"],
                    )
                    fig_vr.update_layout(height=260, margin=dict(t=40, b=20))
                    st.plotly_chart(fig_vr, use_container_width=True)

            # File listing
            st.markdown(t("pkg_files_archive"))
            files = sorted(VR_DIR.glob("**/*"))
            rows = []
            for f in files:
                if f.is_file():
                    rows.append({"file": str(f.relative_to(VR_DIR)), "size": f"{f.stat().st_size/1e3:.0f} KB"})
            if rows:
                st.dataframe(pd.DataFrame(rows).head(10), use_container_width=True, hide_index=True)
                if len(rows) > 10:
                    st.caption(f"... and {len(rows)-10} more binary blob files (snapshot_NNN.bin)")
        else:
            st.info("Archive not yet generated. Run:\n```\npython scripts/export_vr_archive.py\n```")

        # Download
        if VR_ZIP.exists():
            st.download_button(
                label=f"{t('pkg_dl_vr')}  ({VR_ZIP.stat().st_size/1e6:.0f} MB)",
                data=VR_ZIP.read_bytes(),
                file_name="vr_archive.zip",
                mime="application/zip",
                type="primary",
            )


# ══════════════════════════════════════════════════════════════════════════════
# HELYX tab
# ══════════════════════════════════════════════════════════════════════════════
with tab_helyx:
    left, right = st.columns([1, 1])

    with left:
        st.subheader(t("pkg_what"))
        st.markdown("""
        A delivery package for a **CFD engineering team** using
        [HELYX](https://engys.com/products/helyx) (an OpenFOAM-based CFD pre-processor).

        The STL meshes are triangulated surfaces of the airway reconstructed from
        the Ansys Twin Builder point cloud via Delaunay 3D triangulation.
        The engineering team can use them directly in `snappyHexMesh` to create
        a CFD-quality hex mesh and run new full-resolution simulations.

        #### Workflow the HELYX team follows
        ```
        1. Import STL → Geometry panel in HELYX
        2. snappyHexMesh (refinement levels 2-3 near walls)
        3. Assign boundary conditions:
             inlet  (epiglottis)  → velocity inlet
             outlet (bronchial)   → pressure outlet
             wall   (airway wall) → no-slip
        4. Solver: simpleFoam (steady RANS) or pimpleFoam (transient)
        5. Compare DeltaP vs the surrogate predictions
        ```

        #### What they can validate
        The RBF surrogate predicts DeltaP in < 1 ms. HELYX runs the full
        Navier-Stokes equations, which take hours. Comparing the two tells the
        team where the surrogate is accurate and where it needs more training data.
        """)

    with right:
        st.subheader(t("pkg_contents"))

        if helyx_ready:
            stl_actual  = list((HELYX_DIR / "stl" / "actual").glob("*.stl"))
            stl_virtual = list((HELYX_DIR / "stl" / "virtual").glob("*.stl"))

            m1, m2 = st.columns(2)
            m1.metric(t("pkg_actual_stl"), len(stl_actual))
            m2.metric(t("pkg_virtual_stl"), len(stl_virtual))

            # File tree summary
            sections = [
                ("stl/actual/",   f"{len(stl_actual)} STL files  — real CFD geometry"),
                ("stl/virtual/",  f"{len(stl_virtual)} STL files  — representative virtual shapes"),
                ("params/",       "doe_100_actual.csv, lhs_1000_virtual.csv, lhs_1000_summary.csv"),
                ("pod/",          "pod_basis_geometry.npz, pod_scores_1000.npy"),
                ("HELYX_README.txt", "Format, coordinate system, full workflow"),
            ]
            df_tree = pd.DataFrame(sections, columns=["path", "description"])
            st.dataframe(df_tree, use_container_width=True, hide_index=True)

            # README viewer
            readme = HELYX_DIR / "HELYX_README.txt"
            if readme.exists():
                with st.expander("View HELYX_README.txt", expanded=False):
                    st.text(readme.read_text(encoding="utf-8"))

            # Params preview
            params_path = HELYX_DIR / "params" / "lhs_1000_summary.csv"
            if params_path.exists():
                df_sum = pd.read_csv(params_path)
                st.markdown("**Virtual shapes — resistance summary (sample)**")
                col_show = ["shape_id", "dp_Pa", "mean_p_Pa", "resist_index"]
                st.dataframe(
                    df_sum[col_show].head(10),
                    use_container_width=True, hide_index=True,
                )
        else:
            st.info(
                "Package not yet generated. Run:\n"
                "```\npython scripts/export_mesh.py\n"
                "python scripts/export_helyx.py\n```"
            )

        if HELYX_ZIP.exists():
            st.download_button(
                label=f"{t('pkg_dl_helyx')}  ({HELYX_ZIP.stat().st_size/1e6:.0f} MB)",
                data=HELYX_ZIP.read_bytes(),
                file_name="helyx_package.zip",
                mime="application/zip",
                type="primary",
            )


# ══════════════════════════════════════════════════════════════════════════════
# Domino Dataset tab
# ══════════════════════════════════════════════════════════════════════════════
with tab_domino:
    st.subheader("What is it?")
    st.markdown("""
    [Domino Data Lab](https://domino.ai) is an MLOps platform where data science teams
    build, deploy, and monitor machine learning models. The dataset here packages
    the 1000-shape virtual patient cohort as a ready-to-use Domino dataset.

    A data scientist on Domino could immediately:
    - Build a **regression model** predicting DeltaP from the 26 geometric parameters
    - **Cluster** the 1000 shapes to find patient archetypes
    - Train a **neural network** surrogate (as an alternative to the RBF)
    - Run **sensitivity analysis** to find which geometry parameters matter most
    """)

    st.divider()

    if not domino_ready:
        st.info("Dataset not yet generated. Run: `python scripts/export_domino.py`")
    else:
        # ── Load data ─────────────────────────────────────────────────────────
        summary  = pd.read_csv(DOMINO_DIR / "shapes_1000_summary.csv")
        params   = pd.read_csv(DOMINO_DIR / "shapes_1000_params.csv", index_col=0)
        doe_real = pd.read_csv(DOMINO_DIR / "doe_100_actual.csv")

        with open(DOMINO_DIR / "dataset_metadata.json") as f:
            meta = json.load(f)

        # ── Key metrics ───────────────────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(t("pkg_virtual"),     f"{len(summary):,}")
        m2.metric(t("pkg_geo_params"),  meta["geometry_parameters"])
        m3.metric(t("pkg_pres_modes"),  meta.get("k_pressure_modes", "—"))
        m4.metric(t("pkg_mean_dp"),     f"{summary['dp_Pa'].mean():.1f} Pa")
        m5.metric("DeltaP range",       f"{summary['dp_Pa'].min():.0f}–{summary['dp_Pa'].max():.0f} Pa")

        st.divider()

        # ── Plots ─────────────────────────────────────────────────────────────
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            fig_hist = px.histogram(
                summary, x="dp_Pa", nbins=50,
                title="DeltaP distribution — 1000 virtual airways",
                labels={"dp_Pa": "DeltaP — Airway Resistance (Pa)"},
                color_discrete_sequence=["#F4A261"],
            )
            dp_ref = float(doe_real["A_glotis"].count())  # placeholder — use mean
            # Real baseline from actual snapshots
            fig_hist.update_layout(height=320, margin=dict(t=40, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)

        with r1c2:
            fig_ri = px.histogram(
                summary, x="resist_index", nbins=50,
                title="Resistance index distribution (100% = mean shape)",
                labels={"resist_index": "Resistance index (%)"},
                color_discrete_sequence=["#4EB3D3"],
            )
            fig_ri.add_vline(x=100, line_dash="dash", line_color="white",
                             annotation_text="Mean shape")
            fig_ri.update_layout(height=320, margin=dict(t=40, b=20))
            st.plotly_chart(fig_ri, use_container_width=True)

        # Key scatter: glottis area vs trachea diameter, colored by DeltaP
        r2c1, r2c2 = st.columns(2)

        with r2c1:
            if "A_glotis" in summary.columns and "d_trachea" in summary.columns:
                fig_sc = px.scatter(
                    summary, x="A_glotis", y="d_trachea", color="dp_Pa",
                    color_continuous_scale="Jet",
                    title="Glottis area vs Trachea diameter (coloured by DeltaP)",
                    labels={
                        "A_glotis":  "Glottis Area (mm2)",
                        "d_trachea": "Trachea Diameter (mm)",
                        "dp_Pa":     "DeltaP (Pa)",
                    },
                    opacity=0.6,
                )
                fig_sc.update_layout(height=340, margin=dict(t=40, b=20))
                st.plotly_chart(fig_sc, use_container_width=True)

        with r2c2:
            # Most vs least resistive shapes
            top5    = summary.nlargest(5,  "dp_Pa")[["shape_id", "dp_Pa", "resist_index"]]
            bottom5 = summary.nsmallest(5, "dp_Pa")[["shape_id", "dp_Pa", "resist_index"]]
            combined = pd.concat([top5, bottom5])
            combined["category"] = (
                ["High resistance"] * 5 + ["Low resistance"] * 5
            )
            fig_bar = px.bar(
                combined, x="shape_id", y="dp_Pa", color="category",
                color_discrete_map={
                    "High resistance": "#E63946",
                    "Low resistance":  "#06D6A0",
                },
                title="Top 5 most & least resistive virtual airways",
                labels={"dp_Pa": "DeltaP (Pa)", "shape_id": "Shape ID"},
            )
            fig_bar.update_layout(height=340, margin=dict(t=40, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Correlation with DeltaP ───────────────────────────────────────────
        st.subheader(t("pkg_geo_params_drive"))
        st.caption(t("pkg_corr_caption"))

        param_cols_in_summary = [
            c for c in summary.columns
            if c not in ("shape_id", "dp_Pa", "mean_p_Pa", "resist_index")
        ]
        if param_cols_in_summary:
            corr = summary[param_cols_in_summary + ["dp_Pa"]].corr()["dp_Pa"].drop("dp_Pa")
            corr_df = corr.abs().sort_values(ascending=False).reset_index()
            corr_df.columns = ["parameter", "abs_correlation"]
            corr_df["correlation"] = corr[corr_df["parameter"]].values

            fig_corr = px.bar(
                corr_df.head(15), x="parameter", y="correlation",
                color="correlation",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                title="Top 15 parameters by |correlation| with DeltaP",
                labels={"correlation": "Pearson r", "parameter": "Geometry parameter"},
            )
            fig_corr.update_layout(height=360, margin=dict(t=40, b=60))
            st.plotly_chart(fig_corr, use_container_width=True)

        # ── File listing ──────────────────────────────────────────────────────
        st.subheader(t("pkg_dataset_files"))
        file_rows = []
        for f in sorted(DOMINO_DIR.glob("*")):
            if f.is_file():
                desc = meta.get("files", {}).get(f.name, "")
                file_rows.append({
                    "file":        f.name,
                    "size":        f"{f.stat().st_size/1e6:.2f} MB",
                    "description": desc[:90] + "…" if len(desc) > 90 else desc,
                })
        st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)

        # ── Download ──────────────────────────────────────────────────────────
        if DOMINO_ZIP.exists():
            st.download_button(
                label=f"{t('pkg_dl_domino')}  ({DOMINO_ZIP.stat().st_size/1e6:.0f} MB)",
                data=DOMINO_ZIP.read_bytes(),
                file_name="domino_dataset.zip",
                mime="application/zip",
                type="primary",
            )
        else:
            st.caption("Run `python scripts/export_domino.py` to generate the ZIP.")
