"""
pages/5_RBF_Inference.py
=========================
RBF surrogate model — pressure prediction, validation, and clinical insights.

Tabs
----
1. Predict New Shape   — 26 sliders → RBF → pressure field + airway resistance ΔP
2. LOO Validation      — leave-one-out cross-validation (per-snapshot errors)
3. K-Fold Validation   — k-fold CV (faster, fold-level summary + comparison with LOO)
4. Convergence Study   — LOO error as function of number of POD modes (bias-variance)
5. 1000 Virtual Shapes — LHS upscaling + resistance distribution
6. Patient Demo        — how this digital twin is used clinically
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
    STRIDE, load_all_pressures, load_ref_coords,
    load_doe, get_param_cols, PARAM_LABELS, load_precomputed_pod,
)
from core.pod import compute_pod, modes_for_energy, reconstruct
from core.rbf import build_rbf, predict, loo_errors, kfold_errors
import core.gp as gp_mod
from core.lhs import latin_hypercube, doe_bounds
from core.i18n import t, lang_selector

st.set_page_config(page_title="RBF / GP Inference", page_icon="🔮", layout="wide")
lang_selector()
st.title(t("rbf_title"))
st.caption(t("rbf_caption"))

# ── Load data ─────────────────────────────────────────────────────────────────
doe_df     = load_doe()
param_cols = get_param_cols(doe_df)
ref_coords = load_ref_coords(STRIDE)

_pre_p = load_precomputed_pod("pressure")
if _pre_p is not None:
    mean_pres   = _pre_p["mean"]
    modes_pres  = _pre_p["modes"]
    scores_pres = _pre_p["scores"]
    sv_pres     = _pre_p["svalues"]
else:
    with st.spinner("Loading pressure snapshots…"):
        P_pres = load_all_pressures(STRIDE)
    @st.cache_data(show_spinner=False)
    def pres_pod(P): return compute_pod(P)
    with st.spinner("Computing pressure POD…"):
        mean_pres, modes_pres, scores_pres, sv_pres = pres_pod(P_pres)

k_rbf_default = modes_for_energy(sv_pres, 0.99)
params_raw    = doe_df[param_cols].values.astype(float)
low, high     = params_raw.min(axis=0), params_raw.max(axis=0)
params_norm   = (params_raw - low) / (high - low + 1e-12)

# Reference ΔP (mean shape) — used to normalise resistance index
_mean_scores_ref = np.zeros(scores_pres.shape[1])
_pres_ref        = mean_pres  # the mean field itself is the reference

with st.sidebar:
    st.header(t("rbf_sidebar_hdr"))
    k_rbf  = st.slider(t("rbf_pod_modes"), 1, min(40, len(sv_pres)), k_rbf_default)
    kernel = st.selectbox(t("rbf_kernel"),
                          ["thin_plate_spline", "multiquadric", "linear", "cubic"])
    st.divider()
    st.caption("thin_plate_spline: φ(r) = r² log r — smooth, good for high-dim data.")

    st.header(t("gp_sidebar_hdr"))
    gp_kernel = st.selectbox(
        t("gp_kernel"),
        ["matern_52", "matern_32", "rbf", "rational_quadratic"],
        help=(
            "matern_52: twice differentiable — good default for physical data.\n"
            "rbf: infinitely smooth (squared exponential / Gaussian).\n"
            "matern_32: once differentiable — sharper features.\n"
            "rational_quadratic: mixture of length-scales."
        ),
    )
    gp_restarts = st.slider(
        t("gp_restarts"), 1, 5, 2,
        help="More restarts → better optimisation, slower training.",
    )
    st.divider()
    st.caption(
        "GP hyperparameters (length-scale, amplitude) are fitted by "
        "maximising the marginal log-likelihood of the training data."
    )

@st.cache_data(show_spinner=False)
def get_rbf(params, scores, k, kern):
    return build_rbf(params, scores[:, :k], kernel=kern)

with st.spinner("Building RBF surrogate…"):
    rbf_model = get_rbf(params_norm, scores_pres, k_rbf, kernel)

# ΔP for the mean pressure field (reference baseline)
dp_ref = float(mean_pres.max() - mean_pres.min())

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_pred, tab_loo, tab_kfold, tab_conv, tab_lhs, tab_demo, tab_compare = st.tabs([
    t("rbf_tab_predict"),
    t("rbf_tab_loo"),
    t("rbf_tab_kfold"),
    t("rbf_tab_conv"),
    t("rbf_tab_lhs"),
    t("rbf_tab_demo"),
    t("rbf_tab_compare"),
])

# ── Tab 1: Prediction ─────────────────────────────────────────────────────────
with tab_pred:
    st.subheader(t("rbf_sub_predict"))

    new_params = np.empty(len(param_cols))
    groups = {
        "Main airway (glottis, trachea)": [
            "A_glotis", "A_epiglotis", "r_curvature",
            "d_trachea", "l_trachea", "teta_branch_trachea",
        ],
        "First-level branches (L / R)": [
            "l_l", "l_r", "teta_branch_l", "teta_branch_r",
            "l_ll", "l_lr", "l_rl", "l_rr",
            "teta_branch_ll", "teta_branch_lr", "teta_branch_rl", "teta_branch_rr",
        ],
        "Second-level sub-branches": [
            "l_lll", "l_llr", "l_lrl", "l_lrr",
            "l_rll", "l_rlr", "l_rrl", "l_rrr",
        ],
    }

    for grp_name, grp_cols in groups.items():
        with st.expander(grp_name, expanded=(grp_name == "Main airway (glottis, trachea)")):
            cols_ui = st.columns(3)
            for j, col in enumerate(grp_cols):
                i = param_cols.index(col)
                c_min  = float(params_raw[:, i].min())
                c_max  = float(params_raw[:, i].max())
                c_mean = float(params_raw[:, i].mean())
                new_params[i] = cols_ui[j % 3].slider(
                    PARAM_LABELS.get(col, col),
                    min_value=c_min, max_value=c_max, value=c_mean,
                    step=float((c_max - c_min) / 50),
                    key=f"rbf_{col}", format="%.2f",
                )

    if st.button(t("rbf_btn_predict"), type="primary"):
        new_norm      = ((new_params - low) / (high - low + 1e-12)).reshape(1, -1)
        pred_scores   = predict(rbf_model, new_norm)[0]
        pred_pressure = reconstruct(mean_pres, modes_pres[:, :k_rbf], pred_scores)

        dp_pred = float(pred_pressure.max() - pred_pressure.min())
        resist_idx = dp_pred / dp_ref * 100.0

        st.success(t("rbf_predicted_ok"))

        # ── Metrics row ─────────────────────────────────────────────────────
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric(t("met_min_pres"),    f"{pred_pressure.min():.1f} Pa")
        m2.metric(t("met_max_pres"),    f"{pred_pressure.max():.1f} Pa")
        m3.metric(t("met_mean_pres"),   f"{pred_pressure.mean():.1f} Pa")
        m4.metric(t("met_std_pres"),    f"{pred_pressure.std():.1f} Pa")
        m5.metric(t("met_delta_p"),     f"{dp_pred:.1f} Pa",
                  delta=f"{dp_pred - dp_ref:+.1f} vs mean shape",
                  delta_color="inverse")
        m6.metric(t("met_resist_idx"),   f"{resist_idx:.1f} %",
                  help="ΔP relative to mean-shape baseline (100% = average resistance)")

        st.caption(
            "**ΔP** = max − min pressure across the airway. "
            "Higher ΔP means more resistance to airflow — harder to breathe. "
            f"Mean-shape baseline ΔP = {dp_ref:.1f} Pa."
        )

        fig_pred = go.Figure(go.Scatter3d(
            x=ref_coords[:, 0], y=ref_coords[:, 1], z=ref_coords[:, 2],
            mode="markers",
            marker=dict(
                size=2, color=pred_pressure, colorscale="Jet",
                colorbar=dict(title="Pressure (Pa)", thickness=14),
                opacity=0.65,
            ),
            hovertemplate="P: %{marker.color:.1f} Pa<extra></extra>",
        ))
        fig_pred.update_layout(
            scene=dict(aspectmode="data", bgcolor="rgb(12,14,20)",
                       xaxis=dict(gridcolor="#2a2a2a"),
                       yaxis=dict(gridcolor="#2a2a2a"),
                       zaxis=dict(gridcolor="#2a2a2a")),
            paper_bgcolor="rgb(12,14,20)", font=dict(color="white"),
            height=620, margin=dict(l=0, r=0, b=0, t=40),
            title="RBF-Predicted Static Pressure",
        )
        st.plotly_chart(fig_pred, width='stretch')


# ── Tab 2: LOO Validation ────────────────────────────────────────────────────
with tab_loo:
    st.subheader(t("rbf_sub_loo"))
    st.markdown("""
    Each of the 100 snapshots is removed in turn; the RBF is rebuilt on the remaining 99
    and predicts the removed one. The most rigorous estimate of generalisation error,
    but requires rebuilding the model 100 times (~30 s).
    """)

    if st.button(t("rbf_btn_loo"), key="run_loo"):
        with st.spinner("Running LOO (100 iterations)…"):
            errors = loo_errors(params_norm, scores_pres[:, :k_rbf], kernel=kernel)

        err_df = pd.DataFrame({
            "Snapshot": list(range(1, 101)),
            "LOO Error": errors.tolist(),
        })
        fig_loo = px.bar(
            err_df, x="Snapshot", y="LOO Error",
            color="LOO Error", color_continuous_scale="Reds",
            title="LOO error per snapshot  (‖predicted − actual‖ in POD score space)",
            labels={"LOO Error": "‖error‖"},
        )
        fig_loo.update_layout(height=400)
        st.plotly_chart(fig_loo, width='stretch')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("rbf_mean_loo"), f"{errors.mean():.4f}")
        c2.metric(t("rbf_max_loo"),  f"{errors.max():.4f}")
        c3.metric(t("rbf_min_loo"),  f"{errors.min():.4f}")
        c4.metric(t("rbf_std_loo"),  f"{errors.std():.4f}")


# ── Tab 3: K-Fold Validation ─────────────────────────────────────────────────
with tab_kfold:
    st.subheader(t("rbf_sub_kfold"))
    st.markdown("""
    The 100 snapshots are shuffled and split into **k equal folds**.
    The RBF is rebuilt k times, each time trained on k−1 folds and tested on the remaining fold.
    Much faster than LOO while still giving a robust generalisation estimate.
    """)

    col1, col2 = st.columns(2)
    n_folds = col1.slider(t("lbl_n_folds"), 2, 10, 5, key="kfold_k")
    kfold_seed = col2.number_input(t("lbl_random_seed"), value=42, step=1, key="kfold_seed")

    st.caption(
        f"**{n_folds}-fold**: trains on {100 - 100//n_folds} samples, tests on "
        f"~{100//n_folds} samples per fold. Rebuilds RBF {n_folds} times."
    )

    if st.button(f"Run {n_folds}-Fold CV", type="primary", key="run_kfold"):
        with st.spinner(f"Running {n_folds}-fold cross-validation…"):
            fold_errs = kfold_errors(
                params_norm, scores_pres[:, :k_rbf],
                k=n_folds, kernel=kernel, seed=int(kfold_seed),
            )

        fold_df = pd.DataFrame({
            "Fold":       [f"Fold {i+1}" for i in range(n_folds)],
            "Mean Error": fold_errs.tolist(),
        })

        fig_kf = px.bar(
            fold_df, x="Fold", y="Mean Error",
            color="Mean Error", color_continuous_scale="Blues",
            title=f"{n_folds}-Fold CV — mean ‖error‖ per fold",
            text_auto=".4f",
        )
        fig_kf.update_layout(height=350)
        st.plotly_chart(fig_kf, width='stretch')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("rbf_mean_cv_err"), f"{fold_errs.mean():.4f}")
        c2.metric(t("rbf_std_cv_err"),  f"{fold_errs.std():.4f}")
        c3.metric(t("rbf_best_fold"),   f"{fold_errs.min():.4f}")
        c4.metric(t("rbf_worst_fold"),  f"{fold_errs.max():.4f}")

        st.divider()
        st.markdown("**LOO vs K-Fold: when to use which**")
        st.markdown("""
        | Method | Models built | Time | Bias | Use when |
        |--------|-------------|------|------|----------|
        | LOO    | 100         | ~30s | Low  | Small dataset, need most accurate estimate |
        | 5-Fold | 5           | ~2s  | Slightly higher | Quick check, larger datasets |
        | 10-Fold| 10          | ~4s  | Low  | Good balance of speed and accuracy |

        For 100 samples, **LOO and 10-fold give nearly identical results** in practice.
        """)


# ── Tab 4: Convergence Study ─────────────────────────────────────────────────
with tab_conv:
    st.subheader(t("rbf_sub_conv"))
    st.markdown("""
    How does RBF accuracy change as we use more POD modes?

    - **Too few modes**: underfitting — the pressure field isn't captured well
    - **Too many modes**: overfitting — high-frequency modes contain noise the RBF struggles to interpolate
    - The optimal k is at the **elbow** of this curve
    """)

    max_k_conv = st.slider(t("rbf_max_k_conv"), 5, min(25, len(sv_pres)), 15, key="conv_maxk")
    cv_method  = st.radio(t("rbf_cv_method"), ["5-Fold (fast)", "LOO (slow, ~30s)"],
                          horizontal=True, key="conv_method")

    if st.button(t("rbf_btn_conv"), type="primary", key="run_conv"):
        k_range = list(range(1, max_k_conv + 1))
        conv_errors = []

        progress_bar = st.progress(0, text="Computing…")
        for idx, ki in enumerate(k_range):
            if cv_method.startswith("5"):
                errs = kfold_errors(params_norm, scores_pres[:, :ki], k=5, kernel=kernel)
                conv_errors.append(float(errs.mean()))
            else:
                errs = loo_errors(params_norm, scores_pres[:, :ki], kernel=kernel)
                conv_errors.append(float(errs.mean()))
            progress_bar.progress((idx + 1) / len(k_range),
                                   text=f"k = {ki}/{max_k_conv}")
        progress_bar.empty()

        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=k_range, y=conv_errors,
            mode="lines+markers",
            line=dict(color="#4EB3D3", width=2),
            marker=dict(size=7),
            name="CV error",
        ))
        # Mark the minimum
        best_k   = k_range[int(np.argmin(conv_errors))]
        best_err = min(conv_errors)
        fig_conv.add_vline(x=best_k, line_dash="dash", line_color="#F4A261",
                           annotation_text=f"Optimal k={best_k}")
        fig_conv.update_layout(
            xaxis_title="Number of POD modes",
            yaxis_title="Mean CV error",
            title="RBF generalisation error vs POD truncation",
            height=420,
        )
        st.plotly_chart(fig_conv, width='stretch')

        co1, co2, co3 = st.columns(3)
        co1.metric(t("rbf_optimal_k"),  best_k)
        co2.metric(t("rbf_min_cv_err"), f"{best_err:.4f}")
        co3.metric("Error at k=99%",
                   f"{conv_errors[modes_for_energy(sv_pres, 0.99)-1]:.4f}")

        st.info(
            f"The elbow is at **k={best_k} modes**. "
            f"Using more modes beyond this point doesn't reduce prediction error "
            f"— it increases it due to noise amplification in the RBF interpolation."
        )


# ── Tab 5: 1000 Virtual Shapes ────────────────────────────────────────────────
with tab_lhs:
    st.subheader(t("rbf_sub_lhs"))
    st.markdown("""
    LHS generates 1000 parameter vectors within the original DOE bounds.
    The RBF surrogate predicts the pressure field for each — no CFD needed.
    """)

    @st.cache_data(show_spinner=False)
    def lhs_predictions(params_norm_arr, lo, hi, mean_p, modes_p, k, kern):
        lhs_raw  = latin_hypercube(1000, lo, hi, seed=42)
        lhs_norm = (lhs_raw - lo) / (hi - lo + 1e-12)
        rbf_tmp  = build_rbf(params_norm_arr, scores_pres[:, :k], kernel=kern)
        pred_sc  = predict(rbf_tmp, lhs_norm)
        mean_pressures = np.array([
            reconstruct(mean_p, modes_p[:, :k], pred_sc[i]).mean()
            for i in range(len(lhs_raw))
        ])
        dp_values = np.array([
            float(reconstruct(mean_p, modes_p[:, :k], pred_sc[i]).max()
                  - reconstruct(mean_p, modes_p[:, :k], pred_sc[i]).min())
            for i in range(len(lhs_raw))
        ])
        return lhs_raw, mean_pressures, dp_values

    with st.spinner("Predicting pressure for 1000 LHS shapes…"):
        lhs_raw, mean_preds, dp_preds = lhs_predictions(
            params_norm, low, high, mean_pres, modes_pres, k_rbf, kernel
        )

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(
            x=mean_preds, nbins=50,
            labels={"x": "Mean Pressure (Pa)"},
            title="Mean pressure distribution — 1000 virtual airways",
            color_discrete_sequence=["#4EB3D3"],
        )
        fig_hist.update_layout(height=340)
        st.plotly_chart(fig_hist, width='stretch')

    with c2:
        fig_dp = px.histogram(
            x=dp_preds, nbins=50,
            labels={"x": "ΔP — Airway Resistance (Pa)"},
            title="Airway resistance (ΔP) distribution — 1000 virtual airways",
            color_discrete_sequence=["#F4A261"],
        )
        fig_dp.add_vline(x=dp_ref, line_dash="dash", line_color="white",
                         annotation_text="Mean shape ΔP")
        fig_dp.update_layout(height=340)
        st.plotly_chart(fig_dp, width='stretch')

    # Scatter design space coloured by ΔP
    pa_idx, pb_idx = 0, 3
    fig_sc = px.scatter(
        x=lhs_raw[:, pa_idx], y=lhs_raw[:, pb_idx],
        color=dp_preds,
        color_continuous_scale="Jet",
        labels={
            "x": PARAM_LABELS.get(param_cols[pa_idx], param_cols[pa_idx]),
            "y": PARAM_LABELS.get(param_cols[pb_idx], param_cols[pb_idx]),
            "color": "ΔP (Pa)",
        },
        title="Design space coloured by airway resistance ΔP",
    )
    fig_sc.update_layout(height=440)
    st.plotly_chart(fig_sc, width='stretch')

    r1, r2, r3 = st.columns(3)
    r1.metric(t("rbf_mean_dp"), f"{dp_preds.mean():.1f} Pa")
    r2.metric(t("rbf_max_dp"), f"{dp_preds.max():.1f} Pa")
    r3.metric(t("rbf_min_dp"), f"{dp_preds.min():.1f} Pa")


# ── Tab 6: Patient Demo ───────────────────────────────────────────────────────
with tab_demo:
    st.subheader("🏥 Clinical Use Case — How This Digital Twin Works in Practice")
    st.markdown("""
    This page demonstrates how the surrogate model would be used for a **real patient**.
    """)

    col_flow, col_img = st.columns([2, 1])
    with col_flow:
        st.markdown("""
        ### Workflow

        **Traditional approach (hours):**
        ```
        CT scan → Segmentation → CFD mesh → Simulation (hours) → Pressure field
        ```

        **With this Digital Twin (<1 ms after training):**
        ```
        CT scan → Measure 26 anatomical parameters → RBF prediction → Pressure field
        ```

        ### Step-by-step

        1. **Patient gets a CT scan** of the upper airways
        2. A radiologist (or automated tool) extracts the **26 geometric measurements**:
           glottis area, trachea diameter, bronchial lengths, branching angles, etc.
        3. Those 26 numbers are entered into the sliders on the **Predict New Shape** tab
        4. The RBF surrogate predicts the full 3D pressure field in **< 1 millisecond**
        5. The clinician reads:
           - The **ΔP (airway resistance)** — how hard is it for this patient to breathe?
           - The **pressure field** — where is the bottleneck in the airway?
           - Whether the patient's ΔP is **above or below average** (resistance index)

        ### Clinical relevance

        | ΔP range | Interpretation |
        |----------|---------------|
        | Low ΔP   | Low resistance — easy breathing |
        | High ΔP  | High resistance — obstructed airway (e.g. tracheal stenosis) |
        | Very high | Potential obstructive sleep apnoea, requires intervention |

        ### What makes this a Digital Twin (not just a model)

        A Digital Twin is updated as the patient's anatomy changes — e.g. after surgery,
        a new CT scan produces new measurements, and the surrogate instantly predicts
        the post-operative pressure field, allowing surgeons to evaluate outcomes
        **before the operation**.
        """)

    with col_img:
        st.markdown(f"### {t('demo_stenosis')}")
        st.caption("Narrow the trachea diameter and watch resistance increase.")

        d_trachea_demo = st.slider(
            t("demo_trachea"),
            min_value=float(params_raw[:, param_cols.index("d_trachea")].min()),
            max_value=float(params_raw[:, param_cols.index("d_trachea")].max()),
            value=float(params_raw[:, param_cols.index("d_trachea")].mean()),
            key="demo_trachea",
        )
        A_glotis_demo = st.slider(
            t("demo_glottis"),
            min_value=float(params_raw[:, param_cols.index("A_glotis")].min()),
            max_value=float(params_raw[:, param_cols.index("A_glotis")].max()),
            value=float(params_raw[:, param_cols.index("A_glotis")].mean()),
            key="demo_glotis",
        )

        # Build a demo parameter vector (mean everywhere, vary trachea + glottis)
        demo_params = params_raw.mean(axis=0).copy()
        demo_params[param_cols.index("d_trachea")] = d_trachea_demo
        demo_params[param_cols.index("A_glotis")]  = A_glotis_demo
        demo_norm = ((demo_params - low) / (high - low + 1e-12)).reshape(1, -1)

        demo_scores   = predict(rbf_model, demo_norm)[0]
        demo_pressure = reconstruct(mean_pres, modes_pres[:, :k_rbf], demo_scores)
        demo_dp       = float(demo_pressure.max() - demo_pressure.min())
        demo_ri       = demo_dp / dp_ref * 100.0

        st.metric("ΔP", f"{demo_dp:.1f} Pa",
                  delta=f"{demo_dp - dp_ref:+.1f} vs average",
                  delta_color="inverse")
        st.metric("Resistance index", f"{demo_ri:.1f} %")

        if demo_ri > 120:
            st.error(t("demo_high_res"))
        elif demo_ri > 105:
            st.warning(t("demo_slight_res"))
        else:
            st.success(t("demo_normal_res"))


# ── Tab 7: GP vs RBF Comparison ───────────────────────────────────────────────
with tab_compare:
    import time as _time

    st.subheader(t("rbf_sub_compare"))
    st.markdown(f"""
    Both surrogates map the same **{len(param_cols)} geometry parameters → {k_rbf_default} pressure POD modes**.
    This tab trains both models, runs K-fold cross-validation, and measures inference speed
    so you can make an informed choice for your use case.

    | | RBF | Gaussian Process |
    |---|---|---|
    | **Type** | Deterministic interpolant | Probabilistic regression |
    | **Kernel** | `{kernel}` | `{gp_kernel}` |
    | **Uncertainty** | None | Posterior std per prediction |
    | **Hyperparameters** | Manual (kernel + smoothing) | Optimised automatically (MLL) |
    | **Scaling** | O(n³) solve, O(n) query | O(n³) train, O(n) query |
    """)

    cv_folds_cmp = st.slider(t("rbf_cv_folds"), 2, 10, 5, key="cmp_folds")
    st.caption(
        f"Will train **{cv_folds_cmp * 2}** models total ({cv_folds_cmp} per method). "
        "GP training includes kernel hyperparameter optimisation — may take 20–60 s."
    )

    if st.button(t("rbf_btn_compare"), type="primary", key="run_compare"):
        scores_k = scores_pres[:, :k_rbf_default]

        # ── Train both on full dataset + measure training time ────────────────
        with st.spinner("Training RBF…"):
            t0 = _time.perf_counter()
            rbf_cmp = build_rbf(params_norm, scores_k, kernel=kernel)
            rbf_train_time = _time.perf_counter() - t0

        with st.spinner(f"Training GP ({gp_kernel}, {gp_restarts} restarts × {k_rbf_default} modes)…"):
            t0 = _time.perf_counter()
            gp_models = gp_mod.build_gp(
                params_norm, scores_k,
                kernel_name=gp_kernel, n_restarts=gp_restarts,
            )
            gp_train_time = _time.perf_counter() - t0

        # ── Inference speed: time 200 queries on random points ─────────────
        rng_inf = np.random.default_rng(0)
        test_pts = rng_inf.uniform(0, 1, size=(200, params_norm.shape[1]))

        t0 = _time.perf_counter()
        for _ in range(200):
            predict(rbf_cmp, test_pts[[0]])
        rbf_inf_us = (_time.perf_counter() - t0) / 200 * 1e6   # µs per query

        t0 = _time.perf_counter()
        for _ in range(200):
            gp_mod.predict(gp_models, test_pts[[0]])
        gp_inf_us = (_time.perf_counter() - t0) / 200 * 1e6

        # ── K-fold CV for both ─────────────────────────────────────────────
        with st.spinner(f"Running {cv_folds_cmp}-fold CV for RBF…"):
            t0 = _time.perf_counter()
            rbf_fold_errs = kfold_errors(
                params_norm, scores_k, k=cv_folds_cmp, kernel=kernel,
            )
            rbf_cv_time = _time.perf_counter() - t0

        with st.spinner(f"Running {cv_folds_cmp}-fold CV for GP (slow — rebuilds GP per fold)…"):
            t0 = _time.perf_counter()
            gp_fold_errs = gp_mod.kfold_errors(
                params_norm, scores_k,
                k=cv_folds_cmp, kernel_name=gp_kernel, n_restarts=1,
            )
            gp_cv_time = _time.perf_counter() - t0

        # ── Results ────────────────────────────────────────────────────────
        st.success("Comparison complete!")
        st.divider()

        # --- Headline metrics ---
        st.markdown("### Summary")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric(t("rbf_train_time"),   f"{rbf_train_time*1e3:.0f} ms")
        c2.metric(t("gp_train_time"),    f"{gp_train_time:.1f} s",
                  delta=f"{gp_train_time/max(rbf_train_time,1e-9):.0f}× slower",
                  delta_color="off")
        c3.metric(t("rbf_inference"),    f"{rbf_inf_us:.1f} µs")
        c4.metric(t("gp_inference"),     f"{gp_inf_us:.1f} µs",
                  delta=f"{gp_inf_us/max(rbf_inf_us,1e-9):.1f}× vs RBF",
                  delta_color="off")
        c5.metric(t("rbf_mean_cv_err"),  f"{rbf_fold_errs.mean():.4f}")
        c6.metric(t("gp_mean_cv_err"),   f"{gp_fold_errs.mean():.4f}",
                  delta=f"{gp_fold_errs.mean() - rbf_fold_errs.mean():+.4f}",
                  delta_color="inverse")

        # --- Training time bar ---
        st.divider()
        st.markdown(f"### {t('cmp_training_time')}")
        fig_train = go.Figure(go.Bar(
            x=["RBF", "GP"],
            y=[rbf_train_time, gp_train_time],
            marker_color=["#4EB3D3", "#F4A261"],
            text=[f"{rbf_train_time*1e3:.0f} ms", f"{gp_train_time:.2f} s"],
            textposition="outside",
        ))
        fig_train.update_layout(
            yaxis_title="Time (s)", title="Full-dataset training time",
            height=320, yaxis_type="log",
        )
        st.plotly_chart(fig_train, width='stretch')
        st.caption("Log scale. GP training includes marginal-likelihood optimisation of kernel hyperparameters.")

        # --- K-fold CV accuracy side by side ---
        st.divider()
        st.markdown(f"### {t('cmp_kfold_acc')}")
        fold_labels = [f"Fold {i+1}" for i in range(cv_folds_cmp)]
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(
            name="RBF", x=fold_labels, y=rbf_fold_errs.tolist(),
            marker_color="#4EB3D3",
            text=[f"{e:.4f}" for e in rbf_fold_errs], textposition="outside",
        ))
        fig_cv.add_trace(go.Bar(
            name="GP", x=fold_labels, y=gp_fold_errs.tolist(),
            marker_color="#F4A261",
            text=[f"{e:.4f}" for e in gp_fold_errs], textposition="outside",
        ))
        fig_cv.update_layout(
            barmode="group",
            yaxis_title="Mean ‖error‖ (POD score space)",
            title=f"{cv_folds_cmp}-Fold CV error — RBF vs GP",
            height=380,
        )
        st.plotly_chart(fig_cv, width='stretch')

        # --- Inference time bar ---
        st.divider()
        st.markdown(f"### {t('cmp_inf_speed')}")
        fig_inf = go.Figure(go.Bar(
            x=["RBF", "GP"],
            y=[rbf_inf_us, gp_inf_us],
            marker_color=["#4EB3D3", "#F4A261"],
            text=[f"{rbf_inf_us:.1f} µs", f"{gp_inf_us:.1f} µs"],
            textposition="outside",
        ))
        fig_inf.update_layout(
            yaxis_title="Microseconds per query (µs)",
            title="Inference latency (200-query average)",
            height=300,
        )
        st.plotly_chart(fig_inf, width='stretch')

        # --- Predicted field comparison (mean shape) ---
        st.divider()
        st.markdown(f"### {t('cmp_pred_field')}")
        mean_params_norm = params_norm.mean(axis=0).reshape(1, -1)

        rbf_scores_mean  = predict(rbf_cmp, mean_params_norm)[0]
        gp_scores_mean   = gp_mod.predict(gp_models, mean_params_norm)[0]

        rbf_field = reconstruct(mean_pres, modes_pres[:, :k_rbf_default], rbf_scores_mean)
        gp_field  = reconstruct(mean_pres, modes_pres[:, :k_rbf_default], gp_scores_mean)

        diff_field = np.abs(rbf_field - gp_field)
        vmin = min(rbf_field.min(), gp_field.min())
        vmax = max(rbf_field.max(), gp_field.max())

        def _field_scatter(coords, values, title, cmin, cmax):
            fig = go.Figure(go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=2, color=values, colorscale="Jet",
                    cmin=cmin, cmax=cmax,
                    colorbar=dict(title="Pa", thickness=12),
                    opacity=0.6,
                ),
            ))
            fig.update_layout(
                scene=dict(aspectmode="data", bgcolor="rgb(12,14,20)",
                           xaxis=dict(gridcolor="#2a2a2a"),
                           yaxis=dict(gridcolor="#2a2a2a"),
                           zaxis=dict(gridcolor="#2a2a2a")),
                paper_bgcolor="rgb(12,14,20)", font=dict(color="white"),
                height=450, margin=dict(l=0, r=0, b=0, t=40), title=title,
            )
            return fig

        col_rbf, col_gp, col_diff = st.columns(3)
        with col_rbf:
            st.plotly_chart(_field_scatter(ref_coords, rbf_field, "RBF prediction", vmin, vmax),
                            width='stretch')
        with col_gp:
            st.plotly_chart(_field_scatter(ref_coords, gp_field, "GP prediction", vmin, vmax),
                            width='stretch')
        with col_diff:
            st.plotly_chart(_field_scatter(ref_coords, diff_field,
                                           "|RBF − GP| absolute difference", 0, diff_field.max()),
                            width='stretch')

        rbf_dp = float(rbf_field.max() - rbf_field.min())
        gp_dp  = float(gp_field.max()  - gp_field.min())
        st.caption(
            f"RBF ΔP = {rbf_dp:.1f} Pa   |   GP ΔP = {gp_dp:.1f} Pa   |   "
            f"Max pointwise difference = {diff_field.max():.2f} Pa   "
            f"(mean = {diff_field.mean():.2f} Pa)"
        )

        # --- Summary table ---
        st.divider()
        st.markdown(f"### {t('cmp_summary')}")
        winner_acc   = "RBF" if rbf_fold_errs.mean() <= gp_fold_errs.mean() else "GP"
        winner_speed = "RBF" if rbf_inf_us <= gp_inf_us else "GP"
        summary_df = pd.DataFrame({
            "Metric":          [
                "Training time", "Inference latency",
                f"CV mean error ({cv_folds_cmp}-fold)", f"CV std ({cv_folds_cmp}-fold)",
                f"CV best fold",  f"CV worst fold",
            ],
            "RBF":             [
                f"{rbf_train_time*1e3:.0f} ms",
                f"{rbf_inf_us:.1f} µs",
                f"{rbf_fold_errs.mean():.4f}",
                f"{rbf_fold_errs.std():.4f}",
                f"{rbf_fold_errs.min():.4f}",
                f"{rbf_fold_errs.max():.4f}",
            ],
            "GP":              [
                f"{gp_train_time:.2f} s",
                f"{gp_inf_us:.1f} µs",
                f"{gp_fold_errs.mean():.4f}",
                f"{gp_fold_errs.std():.4f}",
                f"{gp_fold_errs.min():.4f}",
                f"{gp_fold_errs.max():.4f}",
            ],
            "Winner":          [
                "RBF (faster)",
                winner_speed,
                winner_acc,
                "—", "—", "—",
            ],
        })
        st.dataframe(summary_df, hide_index=True, width=700)

        st.info(
            f"**Accuracy winner:** {winner_acc} (lower CV error)   |   "
            f"**Speed winner:** {winner_speed} (lower inference latency)\n\n"
            "GP provides additional value not measured here: **uncertainty estimates** "
            "(posterior standard deviation per prediction) — useful for flagging "
            "extrapolation outside the training distribution."
        )
