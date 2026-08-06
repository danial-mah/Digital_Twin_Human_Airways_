"""
core/rag.py
===========
Fully offline RAG for the Human Airways Digital Twin.

Architecture
------------
Retrieval : TF-IDF (always available) or sentence-transformers (optional, better quality).
            To enable semantic search: pip install sentence-transformers
Generation: Local extractive QA — composes answers from retrieved chunks, no LLM needed.
            Claude API is optional; if ANTHROPIC_API_KEY is set it upgrades the answer.

Knowledge base
--------------
33 documents: topic references + Q&A pairs covering every aspect of the project.
Structured so that TF-IDF alone gives useful, direct answers.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ── Optional semantic embeddings ─────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer as _ST
    _SMODEL: _ST | None = None

    def _get_semantic_model():
        global _SMODEL
        if _SMODEL is None:
            _SMODEL = _ST("all-MiniLM-L6-v2")
        return _SMODEL

    _HAS_SEMANTIC = True
except ImportError:
    _HAS_SEMANTIC = False
    def _get_semantic_model(): return None


# ── Knowledge base ────────────────────────────────────────────────────────────

_DOCS: List[dict] = [

# ── Topic documents ───────────────────────────────────────────────────────────

{
"id": "project_overview",
"title": "Project Overview — Human Airways Digital Twin",
"content": """
The Human Airways Digital Twin is a computational surrogate model for the human respiratory
system, built for the Digital Twin Methods course at Università degli Studi di Roma Tor Vergata.

The project processes 100 CFD snapshots from Ansys Twin Builder, each representing a different
airway geometry described by 26 anatomical parameters. The pipeline uses POD (SVD), RBF
interpolation, and LHS sampling to build a surrogate that predicts the full 3D pressure field
for any new geometry in under 1 millisecond — replacing hours of CFD computation.

The dashboard is built with Streamlit (web) and PyQt5 (desktop). It includes interactive 3D
visualisation, parameter sensitivity analysis, Sobol indices, K-fold and LOO cross-validation,
animations, and a patient-specific clinical demo.
"""},

{
"id": "dataset",
"title": "Dataset — 100 Snapshots, 26 Parameters, 2.1M Nodes",
"content": """
The dataset consists of 100 Design-of-Experiment (DOE) runs from Ansys Twin Builder CFD.

Each run contains:
- 3D mesh node coordinates: 2,135,906 nodes × 3 coordinates (X, Y, Z) in metres
- Static pressure field: 2,135,906 scalar values in Pascals
- 26 geometry parameters describing the airway anatomy

File format: Twin Builder binary (.bin) — 8-byte little-endian uint64 header followed by float64 values.
File sizes: ~48 MB per geometry snapshot, ~16 MB per pressure snapshot.
Total raw data: approximately 6.4 GB for all 100 snapshots.

For visualisation, the mesh is sub-sampled at stride=50, giving ~42,718 displayed nodes.
This reduces memory usage by 50× while preserving the spatial pressure patterns.
"""},

{
"id": "pod",
"title": "POD — Proper Orthogonal Decomposition (SVD)",
"content": """
POD (Proper Orthogonal Decomposition), identical to PCA in statistics, decomposes a snapshot
matrix into spatial modes and modal coefficients via the Singular Value Decomposition (SVD).

Mathematical steps:
1. Form snapshot matrix X of shape (100 snapshots × N_features)
2. Centre: X_c = X − mean(X, axis=0)
3. Economy SVD: X_c = U · diag(σ) · V^T
4. Spatial modes = columns of V  — describe WHERE variation occurs in space
5. Modal scores = U · diag(σ)   — describe HOW MUCH each snapshot uses each mode
6. Energy of mode i = σᵢ² / Σ σⱼ²

Results for this dataset:
- Geometry POD: 16 modes capture 99% of shape variance
- Pressure POD: ~8 modes capture 99% of pressure variance (typically 5–8 depending on the snapshot set)

The pressure field is relatively low-dimensional because laminar flow in airways creates smooth,
physically-constrained pressure gradients. Mode 1 captures the dominant axial pressure drop
(~85% of variance), Mode 2 captures left-right asymmetry, Mode 3 captures curvature effects.
Modes 4–8 capture the remaining finer spatial variations needed to reach 99%.

Reconstruction formula: field ≈ mean + modes @ coefficients
"""},

{
"id": "rbf",
"title": "RBF — Radial Basis Function Surrogate Model",
"content": """
The RBF surrogate maps 26 geometry parameters to pressure POD modal scores, enabling pressure
field prediction in under 1 millisecond for any new geometry.

Formula: f(x) = Σᵢ wᵢ φ(‖x − xᵢ‖) + polynomial tail
where φ is the radial basis function and wᵢ are weights fitted to the training data.

Kernel options:
- thin_plate_spline: φ(r) = r² log r  — smooth, standard choice for scattered high-dim data
- multiquadric: φ(r) = √(r² + ε²)   — good for interpolation
- linear: φ(r) = r                    — simple, fast
- cubic: φ(r) = r³                    — smoother than linear

Training: fit RBF from DOE params (100×26) → pressure POD scores (100×k).
Inference: new 26-dim parameter vector → k predicted scores → pressure field via POD reconstruction.

Accuracy: validated with Leave-One-Out (LOO) and K-Fold cross-validation.
Typical LOO error in POD score space is under 0.05, corresponding to <1% pressure field error.
"""},

{
"id": "lhs",
"title": "LHS — Latin Hypercube Sampling",
"content": """
Latin Hypercube Sampling generates 1000 virtual airway shapes within the original DOE bounds,
providing much richer design space coverage than the 100 CFD runs alone.

How it works:
1. Divide each of the 26 parameter dimensions into N equal intervals
2. Place exactly one sample per interval per dimension
3. Randomly permute assignments across dimensions

Result: N samples with perfect uniform marginal distributions — no clustering, no large gaps.

In this project: 1000 LHS samples are evaluated using the RBF surrogate (no new CFD runs needed),
producing 1000 predicted pressure fields in seconds instead of years of simulation time.

The L2-discrepancy of LHS (1000 points) is significantly lower than the original DOE (100 points),
confirming better space-filling coverage.
"""},

{
"id": "sobol",
"title": "Sobol Sensitivity Indices — Variance-Based Global Sensitivity",
"content": """
Sobol indices measure how much of the output variance is explained by each input parameter.
Unlike Pearson correlation (linear only), Sobol captures non-linear and interaction effects.

Definitions:
- First-order S₁ᵢ: fraction of output variance due to parameter i alone
- Total Sₜᵢ: fraction due to parameter i including all interactions with other parameters
- Interaction contribution: Sₜᵢ − S₁ᵢ

Estimator: Saltelli (2010) method using two independent LHS sample matrices A and B:
- S₁ᵢ = (1/N) Σ f(B)[j] × (f(A_Bᵢ)[j] − f(A)[j]) / Var(Y)
- Sₜᵢ = (1/2N) Σ (f(A)[j] − f(A_Bᵢ)[j])² / Var(Y)
where A_Bᵢ is matrix A with column i replaced by B.

In this project: the RBF surrogate evaluates all N×(d+2) = 512×28 = 14,336 samples in <1 second.
Output Y = mean predicted pressure or ΔP (airway resistance).

Divergence between Pearson |r| and Sobol S₁ reveals non-linear parameter effects.
"""},

{
"id": "kfold_loo",
"title": "K-Fold vs LOO Cross-Validation",
"content": """
Both methods estimate how well the RBF surrogate generalises to unseen geometries.

Leave-One-Out (LOO):
- Removes one snapshot at a time, rebuilds RBF on remaining 99, predicts removed snapshot
- Most rigorous estimate for small datasets
- Requires 100 model rebuilds — takes ~30 seconds
- Near-unbiased estimate for small datasets like this one

K-Fold Cross-Validation:
- Splits 100 snapshots into k equal folds (default k=5, fold size=20)
- Trains on k−1 folds, tests on 1 fold, repeats k times
- Only requires k model rebuilds — much faster (2-4 seconds)
- Slightly more bias than LOO but practically equivalent for 100 samples

For 100 snapshots: LOO and 10-fold give nearly identical results.
5-fold is the standard choice — good balance of speed and accuracy.
The per-fold error bars show whether any fold is unusually hard to predict (outliers).

K-fold reconstruction (POD Analysis page): tests whether POD modes trained on k−1 folds
can reconstruct the held-out fold — measures POD generalisation, not just RBF accuracy.
"""},

{
"id": "params_main",
"title": "Main Airway Parameters — Glottis, Epiglottis, Trachea",
"content": """
The 6 main airway parameters describe the upper respiratory tract:

A_glotis — Glottis Area (mm²):
  The opening between the vocal cords. The glottis is the narrowest point in the upper airway.
  Smaller glottis area means higher resistance and larger pressure drop.
  This is typically the parameter most correlated with mean airway pressure.

A_epiglotis — Epiglottis Area (mm²):
  The epiglottis is a flap that closes the airway during swallowing.
  Its area affects the aerodynamic resistance of the laryngopharynx region.

r_curvature — Curvature Radius (mm):
  The radius of curvature of the trachea bend. The trachea is not straight — it curves.
  Larger radius = gentler bend = lower resistance. Smaller radius = sharper bend = higher resistance.
  Affects secondary flow patterns and pressure distribution in the trachea.

d_trachea — Trachea Diameter (mm):
  The inner diameter of the tracheal tube. This is one of the most clinically significant parameters.
  Tracheal stenosis (narrowing) leads to high airway resistance and breathing difficulty.
  Resistance scales approximately as 1/d⁴ (Hagen-Poiseuille law for laminar flow).

l_trachea — Trachea Length (mm):
  The length of the trachea from the larynx to the carina (bifurcation point).
  Longer trachea = more total resistance, but the effect is less dramatic than diameter changes.

teta_branch_trachea — Trachea Branch Angle (°):
  The angle at which the trachea splits into the left and right main bronchi (the carina angle).
  Affects the flow distribution between left and right lungs.
"""},

{
"id": "params_branches",
"title": "Branch Parameters — Left/Right Bronchi and Sub-branches",
"content": """
The 20 branch parameters describe the bronchial tree structure:

First-level bronchi (l_l, l_r): lengths of left and right main bronchi (mm).
First-level angles (teta_branch_l, teta_branch_r): branching angles at the main bronchi (°).
Second-level lengths (l_ll, l_lr, l_rl, l_rr): lengths of the 4 second-level branches (mm).
Second-level angles (teta_branch_ll, lr, rl, rr): branching angles at second level (°).
Third-level lengths (l_lll, l_llr, l_lrl, l_lrr, l_rll, l_rlr, l_rrl, l_rrr): 8 terminal branch lengths (mm).

Naming convention:
- L = left, R = right
- Second letter indicates the branch at that level (L=left branch, R=right branch of parent)
- So LLR = the right sub-branch of the left-left second-level branch

These parameters have generally lower individual sensitivity than the main airway parameters,
but they contribute to interaction effects captured by the Sobol total indices.
"""},

{
"id": "anatomy",
"title": "Airway Anatomy — Trachea, Bronchi, and Branching Tree",
"content": """
The human airway modelled starts from the larynx and extends to the third-level bronchi:

Mouth/Nose → Pharynx → Larynx → Glottis/Epiglottis → Trachea → Carina
    ├── Left main bronchus (L) → Left-Left (LL) → LLL, LLR
    │                          → Left-Right (LR) → LRL, LRR
    └── Right main bronchus (R) → Right-Left (RL) → RLL, RLR
                                → Right-Right (RR) → RRL, RRR

The trachea is the main windpipe, ~10-12 cm long in adults, ~15-20 mm diameter.
It bifurcates at the carina into the left and right main bronchi.

The right main bronchus is shorter, wider, and more vertical than the left — this is why
foreign bodies more often enter the right lung and why pressure distribution can be asymmetric.

Clinically, the model captures:
- Upper airway resistance (glottis, trachea — dominant effect)
- Bronchial distribution resistance (branching angles and lengths — secondary effect)
- Pressure at each anatomical level
"""},

{
"id": "airway_resistance",
"title": "Airway Resistance — ΔP and Clinical Significance",
"content": """
Airway resistance is the primary clinical output of this digital twin.

ΔP = max(pressure) − min(pressure) across the airway [Pascal].
This is the pressure drop driving airflow from the mouth to the bronchi.

Higher ΔP = more resistance = harder to breathe.

Physical basis (Hagen-Poiseuille for laminar flow):
  R = 8μL / (π r⁴)
where μ = air viscosity, L = airway length, r = radius.
Resistance scales as 1/r⁴ — halving the trachea radius increases resistance by 16×.

Clinical conditions this model helps evaluate:
- Tracheal stenosis: narrowing of the trachea → large ΔP increase
- Vocal cord dysfunction: reduced glottis area → high resistance
- Obstructive sleep apnoea: geometry changes during sleep
- Pre/post surgical assessment: e.g. after tracheal reconstruction

Resistance index in the dashboard: ΔP_patient / ΔP_mean × 100%.
Values > 120% indicate clinically elevated resistance.

Across the 1000 LHS virtual shapes, the mean ΔP is approximately 1–5 Pa for resting breathing
(exact value depends on the flow boundary conditions in the original CFD setup).
"""},

{
"id": "validation",
"title": "Model Validation and Accuracy",
"content": """
The surrogate model is validated at multiple levels:

1. POD reconstruction error:
   - 8 modes: < 0.5% relative L2 error for pressure reconstruction (captures 99% of variance)
   - 10 modes: < 1% relative L2 error (well within engineering accuracy)
   - Convergence curve shows error drops rapidly in first 3–5 modes, then gradually plateaus

2. RBF LOO cross-validation:
   - Each of 100 snapshots predicted by RBF trained on remaining 99
   - Mean LOO error in POD score space: typically < 0.05
   - Max LOO error identifies outlier snapshots (unusual geometries)

3. RBF K-fold (5-fold):
   - Equivalent accuracy to LOO for this dataset size
   - Consistent error across folds confirms no overfitting

4. Convergence study (error vs number of modes):
   - Optimal k is at the elbow of the LOO error curve
   - Typically 3-5 modes for pressure (optimal bias-variance tradeoff)
   - Using too many modes can worsen RBF accuracy (noise interpolation)

5. POD reconstruction K-fold:
   - Tests whether modes trained on 80 snapshots reconstruct the other 20
   - Consistent results across folds: POD generalises well
"""},

{
"id": "dashboard_pages",
"title": "Dashboard Pages — What Each Page Does",
"content": """
The Streamlit dashboard has 13 pages:

1. Landing page: project overview, key metrics (100 snapshots, 26 params, 2.1M nodes).

2. Geometry Explorer: morph the airway using POD mode sliders.
   Color by displacement from mean to see deformation patterns clearly.
   Use 'Show mean shape ghost' to compare deformed vs average airway.

3. Pressure Field: 3D pressure map for any of the 100 DOE snapshots.
   Toggle between reference mesh and deformed geometry.

4. DOE Analysis: parallel coordinates, pairwise scatter, correlation heatmap, LHS coverage.

5. POD Analysis: energy curves, mode shape 3D visualisation, reconstruction comparison (original
   vs POD-reconstructed with error map), K-fold and LOO validation.

6. RBF Inference: 26 sliders → predict pressure field + ΔP resistance + resistance index.
   LOO, K-fold, convergence study, 1000 virtual shapes, patient demo.

7. Regional Analysis: mean pressure per anatomical region across all 100 snapshots.

8. Design Space: parameter sensitivity (Pearson r), design landscape, Sobol sensitivity indices.

9. Animations: POD mode sweep, snapshot reel, pressure mode sweep. Click Generate first.

10. Ask AI: TF-IDF/semantic retrieval over 33 knowledge base documents. Optional Claude API upgrade.

11. Mesh Viewer: airway rendered as a triangulated surface (Mesh3d) with pressure coloured
    on the solid surface. Compare point cloud vs solid mesh side by side. STL download per snapshot.

12. Delivery Packages: overview and download of three export packages — VR Archive (WebXR),
    HELYX (STL meshes + parameter tables for CFD), Domino (ML-ready 1000-shape dataset).

13. AI Surrogate (PhysicsNeMo): NVIDIA PhysicsNeMo neural network surrogate trained via
    knowledge distillation from the RBF model. Shows training journey, live prediction vs CFD
    ground truth, 1000 virtual shape inference, and HELYX delivery summary.
"""},

{
"id": "deployment",
"title": "Deployment — Online and Local",
"content": """
Streamlit Community Cloud (free online deployment):
1. Run scripts/precompute.py --pod-only to generate precomputed/ folder (~150 MB total)
2. Push code + precomputed/ + Points/doe_point.csv + Pressure/settings.json to GitHub
3. Connect at share.streamlit.io, set main file to app.py
4. The raw .bin files (6.4 GB) stay local — the cloud app runs from precomputed NPZ files only

PyQt desktop app (local only):
  pip install -r requirements_qt.txt
  python qt_app/run.py
  Requires precomputed/ folder for fast startup. Gives true real-time slider response (<10ms).

Precomputation script:
  python scripts/precompute.py            # full export (NPY + VTK + POD cache)
  python scripts/precompute.py --pod-only # only POD cache (fastest, needed for deployment)
  python scripts/precompute.py --no-vtk  # skip VTK export

Live app URL: https://human-airways-digital-twin-wvmgvehsbnkkpfe3dndrnr.streamlit.app/
GitHub repo: https://github.com/Kuhyar-saeedi/human-airways-digital-twin
"""},

{
"id": "digital_twin_concept",
"title": "What Is a Digital Twin?",
"content": """
A Digital Twin is a computational model that mirrors a physical system and is updated
as the physical system changes.

In engineering: a digital twin of a turbine updates its model as sensor data arrives,
predicting wear and failures before they happen.

In medicine (this project): the Digital Twin mirrors a patient's airway geometry.
1. CT scan → 26 anatomical measurements → surrogate predicts pressure field in <1ms
2. As the patient's anatomy changes (surgery, disease progression), the twin updates
3. The surrogate allows instant what-if analysis: "What if we widen the trachea by 2mm?"

What makes this project a Digital Twin (not just a model):
- It's PARAMETERISED: accepts real patient measurements as input
- It's FAST: ms-scale prediction vs hours of CFD
- It's VALIDATED: LOO/K-fold CV confirms generalisation
- It's PHYSICALLY GROUNDED: POD modes come from real CFD physics
- It's PATIENT-SPECIFIC: 26 parameters describe individual anatomy

This project was built as Deliverable D3 for the Digital Twin Methods course at
Università degli Studi di Roma Tor Vergata.
"""},

# ── Q&A pairs ─────────────────────────────────────────────────────────────────

{
"id": "qa_pod_modes_pressure",
"title": "Q: How many POD modes are needed for pressure?",
"content": """
Answer: ~8 POD modes are needed to capture 99% of the pressure variance in this dataset
(the POD Analysis page shows the exact number via the modes_for_energy metric).

This is physically reasonable: the pressure field is relatively low-dimensional because:
1. The flow is LAMINAR — smooth, no turbulence, no small-scale pressure fluctuations
2. Mode 1 captures the dominant axial pressure drop from mouth to bronchi (~85% of variance)
3. Mode 2 captures left-right pressure asymmetry (different resistance in left vs right lung)
4. Mode 3 captures the effect of tracheal curvature and branching angle
5. Modes 4–8 capture progressively finer spatial variations to reach the 99% threshold

K-fold cross-validation confirms that relative L2 reconstruction error is < 0.5% with 8 modes.

The 26 DOE parameters may also be partially correlated (LHS ensures uniform marginals but
not independence), which reduces the effective dimensionality compared to a fully uncorrelated set.
"""},

{
"id": "qa_pod_modes_geometry",
"title": "Q: Why does geometry need 16 modes but pressure only ~8?",
"content": """
Answer: Geometry needs 16 modes for 99% energy because 26 independent parameters create
genuinely high-dimensional shape variation. Each parameter controls a different branch
independently, so the shape space requires more modes to represent.

Pressure needs ~8 modes (compared to 16 for geometry) because it is physically constrained
— the Navier-Stokes equations enforce smoothness and the dominant pattern is a single
large-scale pressure gradient. While geometry encodes 26 structural degrees of freedom
equally, pressure variation is governed by physics that naturally creates smoother,
lower-dimensional distributions.

Key insight: geometry variation is driven by 26 independent inputs (branching lengths,
angles, diameters), while pressure variation is governed by fluid mechanics that amplifies
only a few dominant spatial patterns. This is why pressure needs roughly half as many modes
as geometry to achieve the same 99% variance threshold.
"""},

{
"id": "qa_which_param_matters",
"title": "Q: Which parameter has the most effect on pressure / resistance?",
"content": """
Answer: Glottis area (A_glotis) and trachea diameter (d_trachea) are typically the most
influential parameters for airway pressure.

The glottis is the narrowest point in the upper airway. By Poiseuille's law, resistance
scales as 1/r⁴, so small changes in the glottis area produce large pressure changes.

The trachea diameter has the second-largest effect for the same reason — it is the longest
narrow segment of the airway.

You can verify this on:
- Design Space → Parameter Sensitivity: ranked Pearson correlation bar chart
- Design Space → Sobol Indices: variance-based first-order and total Sobol indices
- RBF Inference → Patient Demo: try sliding the trachea diameter and glottis area sliders

The Sobol indices may reveal additional non-linear effects not captured by Pearson correlation.
"""},

{
"id": "qa_rbf_accuracy",
"title": "Q: How accurate is the RBF surrogate model?",
"content": """
Answer: The RBF surrogate is highly accurate for this dataset.

Quantitative validation:
- LOO cross-validation mean error in POD score space: typically < 0.05
- This corresponds to < 1% relative L2 error in the reconstructed pressure field
- K-fold (5-fold) gives consistent per-fold errors, confirming no overfitting

The convergence study shows the optimal number of POD modes for RBF is at the elbow of the
error curve (typically ~8 modes for pressure, matching the 99% energy criterion). Using
many more modes does NOT improve accuracy because the highest modes contain noise that the
RBF interpolates poorly — the error curve flattens or rises beyond the elbow.

The thin_plate_spline kernel (default) works best for this scattered high-dimensional data.
"""},

{
"id": "qa_how_to_use_rbf_page",
"title": "Q: How do I use the RBF Inference page?",
"content": """
Answer: The RBF Inference page (page 5) lets you predict the pressure field for any new
set of airway geometry parameters.

Step by step:
1. The sidebar shows POD modes for RBF (default: modes for 99% energy) and kernel selection
2. In the 'Predict New Shape' tab, adjust the 26 parameter sliders in the expanders
3. Click the purple 'Predict Pressure Field' button
4. The page shows 6 metrics: min/max/mean/std pressure, ΔP resistance, and resistance index
5. A 3D scatter plot shows the predicted pressure coloured by value

The resistance index shows how your geometry compares to the average: 100% = average,
>120% = elevated resistance, <80% = low resistance airway.

Other tabs:
- LOO Validation: click button, wait ~30s, see per-snapshot error bar chart
- K-Fold Validation: faster alternative, configurable number of folds
- Convergence Study: see how accuracy changes with number of POD modes
- 1000 Virtual Shapes: ΔP distribution across the design space
- Patient Demo: simplified 2-slider demo for clinical explanation
"""},

{
"id": "qa_what_is_design_landscape",
"title": "Q: What does the Design Landscape show?",
"content": """
Answer: The Design Landscape (Design Space page, tab 2) maps all 100 DOE snapshots into
the 2D pressure POD score space (Mode 1 vs Mode 2), coloured by mean pressure.

What it reveals:
- Clustering: are some snapshots grouped together? Suggests correlated parameters or similar geometries
- Spread: how well do the 100 runs cover the design space?
- Colour gradient: which direction in mode space corresponds to higher/lower pressure?
- Outliers: any snapshot far from the cluster may be an unusual geometry

The design landscape is the 'map' of all possible airways in the dataset's parameter space.
High-pressure shapes (red) are in one region, low-pressure shapes (blue) in another.

A companion plot shows the geometry POD space (Mode 1 vs Mode 2 of geometry scores).
The cross-correlation plot asks: does knowing the geometry mode predict the pressure mode?
If yes, geometry changes directly drive pressure changes (strong coupling).
"""},

{
"id": "qa_sobol_vs_pearson",
"title": "Q: What is the difference between Pearson correlation and Sobol indices?",
"content": """
Answer: Pearson correlation measures LINEAR relationships. Sobol indices measure TOTAL variance
contribution including non-linear effects and interactions.

Example: if a parameter has a U-shaped effect on pressure (both small and large values increase
pressure, medium values decrease it), Pearson r ≈ 0 (symmetric, no linear trend) but Sobol
S₁ > 0 (significant variance contribution).

Divergence between |Pearson r| and Sobol S₁ for a parameter indicates non-linear effects.
Points above the diagonal in the comparison scatter plot are parameters where Sobol reveals
more influence than Pearson suggests.

Total Sobol index (Sₜ) > First-order (S₁) means the parameter has significant interaction
effects — its impact on pressure depends on the values of other parameters.

For this airway dataset, most relationships are approximately linear (geometry changes
proportionally affect pressure), so Pearson and Sobol results are broadly consistent.
"""},

{
"id": "qa_animations",
"title": "Q: How do the animations work? What do they show?",
"content": """
Answer: The Animations page (page 8) has three on-demand animations built with Plotly frames.

1. Geometry Mode Sweep: morphs the 3D airway shape from -3σ to +3σ along one POD mode.
   Shows what Mode 1 (or any other mode) physically means anatomically.
   Mode 1 might stretch the trachea, Mode 2 might widen the left bronchus, etc.
   Click 'Generate Geometry Animation', then press the ▶ Play button in the chart.

2. Pressure Snapshot Reel: steps through all 100 DOE pressure fields as a movie.
   Lets you see how the pressure distribution changes across the 100 different geometries.

3. Pressure Mode Sweep: shows the pressure field changing as one pressure POD mode
   is swept from -3σ to +3σ, visualising the spatial pattern of that mode.

Animations are generated on button click (not on page load) to save memory.
Default 16 frames — reduce if the animation is slow or causes memory errors on the cloud.
"""},

{
"id": "qa_precompute",
"title": "Q: What does precompute.py do? Why do I need to run it?",
"content": """
Answer: scripts/precompute.py is a one-time script that generates cached data files.

It creates:
  precomputed/pod_geometry.npz     — POD modes, scores, and mean for geometry (~95 MB)
  precomputed/pod_pressure.npz     — POD modes, scores, and mean for pressure (~5 MB)
  precomputed/ref_coords_s50.npy   — reference mesh at stride=50 (~1 MB)
  precomputed/all_pressures_s50.npz — all 100 pressure snapshots compressed (~15 MB)

Optionally:
  export/npy/snapshot_N_*.npy      — NPY archives for archival/proof
  export/vtk/snapshot_N.vtp        — VTK files for ParaView (open with 'StaticPressure_Pa' scalar)

Why needed: the raw .bin files (6.4 GB) cannot be uploaded to GitHub or Streamlit Cloud.
The precomputed files (~150 MB total) contain everything the app needs and can be pushed to git.

Run with: python scripts/precompute.py --pod-only   (just the cache, ~2 minutes)
          python scripts/precompute.py              (full export including VTK, ~15 minutes)
"""},

{
"id": "qa_geometry_explorer",
"title": "Q: How to use the Geometry Explorer? Why can't I see changes when I move sliders?",
"content": """
Answer: The Geometry Explorer (page 1) lets you morph the airway shape using POD mode sliders.

To make changes visible:
1. Select 'Colour by → Displacement from mean' — this colours each point by how far it moved
   from the average airway. Red/hot = large displacement, dark = no movement.
2. Enable 'Show mean shape (ghost)' — overlays the average airway as grey transparent points.
   The deformed shape is displayed on top in colour. Differences become obvious.
3. Drag Mode 1 slider to -3σ or +3σ (the extreme ends) for maximum visible deformation.
4. Click 'Random Shape' for a dramatic random combination of all mode coefficients.

The shape deviation metric (‖c‖) shows the total magnitude of all active coefficients.
A value of 0 means the mean shape. Higher values = larger deviation from average.

Why @st.fragment: the page uses Streamlit's fragment decorator so only the 3D plot
re-renders when sliders change, not the entire page. This makes interaction faster.
"""},

{
"id": "qa_vtk_paraview",
"title": "Q: How do I open the VTK files in ParaView?",
"content": """
Answer: After running python scripts/precompute.py, the VTK files are in export/vtk/.
Each file is snapshot_N.vtp — a VTK PolyData XML format (point cloud with scalar field).

To open in ParaView:
1. Open ParaView (free download at paraview.org)
2. File → Open → navigate to export/vtk/ → select one or more .vtp files
3. Click Apply in the Properties panel
4. In the toolbar, change the colouring from 'Solid Color' to 'StaticPressure_Pa'
5. Adjust the colour scale: Edit → Reset Range to see the full pressure range

To animate through all 100 snapshots:
1. File → Open → select all 100 .vtp files at once (they load as a time series)
2. Press Play in the time controls at the top
3. You get a smooth animation through all 100 snapshots

The VTK files contain point cloud data (no mesh connectivity). If you need surface rendering,
you can use Filters → Delaunay 3D or Surface Reconstruction in ParaView.
"""},

{
"id": "qa_pyqt",
"title": "Q: How do I run the PyQt desktop app? What does it offer vs Streamlit?",
"content": """
Answer: Run the desktop app with: python qt_app/run.py
Requirements: pip install -r requirements_qt.txt (installs PyQt5, pyvista, pyvistaqt)

The PyQt app requires precomputed/ folder to exist (run precompute.py --pod-only first).

What PyQt offers vs Streamlit:
- TRUE REAL-TIME sliders: <10ms update vs ~200ms in Streamlit (no browser roundtrip)
- PyVista 3D rendering: OpenGL-based, smoother rotation and zoom
- Dark themed Qt UI with custom styling
- Three tabs: Geometry Explorer, Pressure Field, RBF Inference
- 50ms debounce timer: updates the 3D view smoothly as you drag sliders

The Streamlit web app is better for sharing (public URL, no installation needed).
The PyQt desktop app is better for interactive exploration and presentations.
"""},

{
"id": "mesh_viewer",
"title": "Mesh Viewer (Page 11) — Solid Triangulated Surface",
"content": """
The Mesh Viewer (page 11) renders the airway as a solid triangulated surface rather than a point cloud.
Every other page shows ~42,000 floating dots. The Mesh Viewer connects those dots into triangles
to produce the kind of continuous 3D object that CFD tools (HELYX, OpenFOAM) and surgical planning
software actually work with.

How the mesh is built:
1. Delaunay 3D triangulation fills the point cloud with tetrahedra.
2. Boundary surface extraction keeps only the faces on the outer surface.
3. Alpha-shape filtering wraps concave regions tightly.
   Alpha is estimated automatically: alpha = median nearest-neighbour distance × 6.
   At stride 50 (~42K nodes), median spacing ~1.4 mm → alpha ~8.4 mm.

Pre-computed STL files (from scripts/export_mesh.py) are loaded instantly if present;
otherwise the mesh is generated on-the-fly and cached in the session.

Pressure is mapped onto mesh vertices via nearest-neighbour lookup from the point cloud.
The surface is displayed with Plotly Mesh3d with Phong lighting for realistic rendering.

Tabs:
- Surface Mesh: full solid render, opacity and colourscale controls, STL download button.
- Mesh vs Point Cloud: side-by-side comparison of dot cloud vs solid surface.
- How the mesh is built: explanation of Delaunay + alpha-shape algorithm, quality table.

Resolution vs quality:
- Stride 50 → ~42K nodes, ~2 s to build — good for teaching and VR.
- Stride 10 → ~214K nodes, ~8 s — better bronchial detail.
- Stride 1  → 2.1M nodes, ~120 s — full resolution.

Limitation: the mesh is a surface reconstruction, not the original FEM mesh. Very narrow
regions (glottis, small bronchi) may be slightly smoothed.
"""},

{
"id": "delivery_packages",
"title": "Delivery Packages (Page 12) — VR Archive, HELYX, Domino",
"content": """
The Delivery Packages page (page 12) presents three specialised export packages, each targeting
a different downstream team or workflow.

VR Archive (export/vr_archive/):
  A self-contained archive that lets anyone explore airway pressure in 3D — no Python, no server required.
  Works in any desktop browser (Chrome, Firefox, Edge) and Meta Quest VR headsets via WebXR.
  Usage: unzip vr_archive.zip, run 'python serve.py', open http://localhost:8765.
  For Meta Quest: connect to same WiFi, open the Quest browser, navigate to http://YOUR-PC-IP:8765.
  Controls: left-drag to orbit, scroll to zoom, press F for fly mode (WASD navigation).
  Generated by: python scripts/export_vr_archive.py

HELYX Package (export/helyx/):
  STL surface meshes + parameter tables for a CFD engineering team using HELYX (OpenFOAM-based).
  Contains: 100 actual STL meshes, 31 representative virtual STL meshes, doe_100_actual.csv,
  lhs_1000_virtual.csv, pod_basis_geometry.npz, HELYX_README.txt.
  Workflow: import STL → snappyHexMesh → assign BCs (inlet velocity, outlet pressure, no-slip walls)
  → run simpleFoam or pimpleFoam → compare DeltaP to surrogate predictions.
  Generated by: python scripts/export_mesh.py then python scripts/export_helyx.py

Domino Dataset (export/domino/):
  An ML-ready dataset for Domino Data Lab — packages the 1000-shape virtual patient cohort.
  Contains: shapes_1000_params.csv, shapes_1000_summary.csv (with dp_Pa, resist_index per shape),
  pod_basis_geometry.npz, pod_basis_pressure.npz, doe_100_actual.csv, dataset_metadata.json.
  Use cases: regression model for DeltaP from 26 parameters, clustering to find patient archetypes,
  neural network surrogate training, sensitivity analysis.
  Generated by: python scripts/export_domino.py

Each package has a ZIP download button on the page.
"""},

{
"id": "physicsnemo_surrogate",
"title": "AI Surrogate — NVIDIA PhysicsNeMo Neural Network (Page 13)",
"content": """
The AI Surrogate page (page 13) presents a neural network surrogate trained with NVIDIA PhysicsNeMo
(the DoMINO architecture) as an alternative to the classical RBF surrogate.

What DoMINO is:
  DoMINO (Decomposable Multi-scale Iterative Neural Operator) is NVIDIA's pre-trained surrogate
  for CFD flow field prediction. It learns geometry → flow field mappings from CFD data,
  then predicts new shapes in milliseconds. Part of the NVIDIA PhysicsNeMo framework.

Our implementation:
  Input:  k_geo geometry POD mode scores (~16 values)
  Output: k_pres pressure POD mode scores (~8 values) → full 3D pressure field at 42,719 nodes
  Architecture: PhysicsNeMo FullyConnected MLP, 3 layers, 32 units, SiLU activation
  Training data: 80 real CFD snapshots + 1000 RBF pseudo-labels (knowledge distillation)
  Test RMSE: 3.22 Pa | Relative error: 4.3% | Training time: 12.8 s on RTX 2060

Three-step training journey (shown on page):
  Step 1 — Naive (80 real snapshots only): severe overfitting, RMSE = 24.67 Pa, 21.8% error.
    The model had 68,355 parameters but only 80 samples — memorised training data, failed on test.
  Step 2 — Diagnosis: the 1000 DoMINO export shapes had geometry and pressure scores sampled
    independently (random, no physical relationship). Useless for supervised training.
  Step 3 — Fix via knowledge distillation: the validated RBF surrogate generates 1000 physically-
    correct pseudo-labels (LHS geometry params → RBF → pressure scores). Combined with 80 real
    snapshots repeated 3× = 1300 training pairs. Result: RMSE drops 7.7× to 3.22 Pa.

Knowledge distillation principle: train the neural network to mimic an accurate classical
surrogate (teacher). The NN inherits the RBF's accuracy but gains GPU inference speed.

Live Prediction tab: select any of 100 real snapshots, compare NN prediction vs Ansys ground truth side by side.
1000 Virtual Airways tab: run inference on 1000 LHS shapes, compare NN vs RBF DeltaP distributions.
HELYX Delivery tab: summary of the HELYX delivery package with boundary condition details.

Requirements: PyTorch + nvidia-physicsnemo (not installed by default, Streamlit page shows install instructions).
Checkpoint saved at: checkpoints/physicsnemo_surrogate.pt
"""},

{
"id": "vr_viewer",
"title": "3D / VR Viewer (Page 10) — WebXR Three.js Interactive Viewer",
"content": """
The 3D / VR Viewer (page 10) is a WebXR-ready interactive point-cloud viewer embedded directly
in the Streamlit page via st.components.v1.html using Three.js (loaded from CDN).

Features:
- Full orbit viewer in any desktop browser (left-drag to orbit, scroll to zoom, right-drag to pan).
- Colour modes: Pressure (Pa), Height (Z), Node index — switchable from the sidebar.
- Snapshot selector: any of the 100 DOE snapshots.
- Point size and background colour controls.
- 'Enter VR' button appears automatically on WebXR-compatible devices (Meta Quest, Chrome+OpenXR).
  The page must be served over HTTPS or localhost for WebXR to activate.

Standalone HTML download:
  A downloadable standalone HTML file is provided so the viewer can be opened directly in a
  VR headset browser without needing Streamlit running. Open the file from a local server or
  directly in the Meta Quest browser.

Technical implementation:
  Point cloud data (coordinates + pressure scalars) is serialised to JSON and injected into
  the Three.js HTML template. Rendering uses THREE.Points with a custom vertex shader that
  maps the pressure scalar to a Jet colour map. OrbitControls provides mouse/touch interaction.
  WebXR session is requested via renderer.xr.enabled = true and the VRButton helper.

This page is compatible with Streamlit Community Cloud (no heavy dependencies — pure HTML/JS).
"""},


# ── Italian Q&A documents ─────────────────────────────────────────────────────
# Bilingual: Italian questions with English-compatible answers for cross-lingual retrieval

{
    "id": "qa_it_pod",
    "title": "Cos'è il POD? (What is POD?)",
    "content": """
Proper Orthogonal Decomposition (POD), also known as Principal Component Analysis (PCA) or
Singular Value Decomposition (SVD), is the core dimensionality reduction method in this project.

POD decomposes a set of N snapshots (either geometry coordinates or pressure fields) into:
  - A mean field (average over all snapshots)
  - A set of spatial modes (ranked by energy / variance captured)
  - A set of temporal/parameter-space scores (coefficients for each snapshot)

Italian explanation / Spiegazione in italiano:
POD è la decomposizione ortogonale propria. Data una matrice X di N istantanee (ad esempio 100
campi di pressione), SVD la fattorizza come X = U Σ Vᵀ. I vettori colonna di U sono i modi
spaziali (dove varia la geometria o la pressione), Σ contiene i valori singolari (importanza di
ciascun modo), e V contiene i coefficienti (punteggi) per ogni istantanea.

Con ~10 modi si cattura il 95% della varianza geometrica, con ~8 modi il 99% della varianza di
pressione. Questo comprime milioni di gradi di libertà CFD in decine di numeri, rendendo possibile
il surrogato.
""",
},

{
    "id": "qa_it_rbf",
    "title": "Come funziona il surrogato RBF? (How does the RBF surrogate work?)",
    "content": """
The RBF (Radial Basis Function) surrogate maps 26 geometry parameters to the pressure POD scores.

Italian / Italiano:
RBF (funzione a base radiale) è il metodo di interpolazione deterministica al cuore del gemello
digitale. Dati N punti di addestramento (i 100 runs DOE), l'interpolante RBF passa esattamente
attraverso tutti i punti noti e prevede valori intermedi.

Formula: f(x) = Σᵢ wᵢ φ(‖x − xᵢ‖)
dove φ è il kernel (thin-plate spline, multiquadric, ecc.) e w sono i pesi determinati risolvendo
un sistema lineare N×N.

Nucleo thin-plate spline: φ(r) = r² log(r) — buono per dati ad alta dimensione.
Il kernel RBF qui e il "kernel RBF" di sklearn sono cose diverse: qui φ è la funzione di
interpolazione, in sklearn "RBF" è il kernel squared-exponential exp(−r²/2ℓ²) usato nei
processi gaussiani.

Velocità: <1 ms per query dopo l'addestramento. Accuratezza: errore LOO medio ~0.01–0.05 nello
spazio dei punteggi POD (dipende dal numero di modi e dal kernel scelto).
""",
},

{
    "id": "qa_it_digital_twin",
    "title": "Cos'è un Gemello Digitale? (What is a Digital Twin?)",
    "content": """
A Digital Twin is a virtual replica of a physical system that is continuously updated with real
data and can predict the system's behaviour in real time.

Italian / Italiano:
Un Gemello Digitale è una replica virtuale di un sistema fisico. Nel contesto di questo progetto:
  - Il "gemello" è il modello surrogato (RBF/GP) addestrato sui 100 runs CFD
  - Il "fisico" è la geometria reale delle vie aeree di un paziente
  - L'"aggiornamento" avviene quando vengono misurati nuovi parametri anatomici (es. post-operatori)

Flusso di lavoro clinico:
1. TC del paziente → estrazione di 26 parametri anatomici
2. I 26 numeri entrano nel surrogato RBF
3. In < 1 ms si ottiene il campo di pressione 3D completo
4. Il clinico legge ΔP (resistenza) e individua le zone di strozzatura

Rispetto alla simulazione CFD tradizionale (ore), il gemello digitale risponde in millisecondi,
abilitando valutazioni pre-operatorie e pianificazione di interventi in tempo reale.
""",
},

{
    "id": "qa_it_resistenza",
    "title": "Cos'è la resistenza delle vie aeree? (What is airway resistance?)",
    "content": """
Airway resistance is quantified as ΔP = max(pressure) − min(pressure) across the airway.
Higher ΔP means more resistance to airflow — the patient has to work harder to breathe.

Italian / Italiano:
La resistenza delle vie aeree è la differenza di pressione ΔP tra l'ingresso e l'uscita
del sistema respiratorio. Maggiore ΔP = più sforzo per respirare.

Nel gemello digitale calcoliamo: ΔP = P_max − P_min su tutti i nodi della maglia.
L'indice di resistenza è ΔP / ΔP_riferimento × 100%, dove ΔP_riferimento è calcolato
sulla geometria media (100 runs DOE).

Valori tipici nei 100 runs DOE: ΔP varia tra ~X e ~Y Pa a seconda della geometria.
Condizioni patologiche (stenosi tracheale, laringomalacia) producono ΔP molto elevati.
Il parametro più correlato con ΔP è tipicamente il diametro della trachea (d_trachea) e
l'area della glottide (A_glotis).
""",
},

{
    "id": "qa_it_sobol",
    "title": "Cosa sono gli indici di Sobol? (What are Sobol indices?)",
    "content": """
Sobol indices are variance-based global sensitivity measures. Unlike Pearson correlation
(which only captures linear effects), Sobol indices measure the total contribution of each
input parameter to the output variance, including non-linear and interaction effects.

Italian / Italiano:
Gli indici di Sobol misurano quanto della varianza dell'output (es. ΔP) è spiegato da ciascun
parametro geometrico.

  S₁ (primo ordine): varianza spiegata dal parametro da solo
  Sₜ (totale): varianza incluse tutte le interazioni con altri parametri
  Sₜ − S₁: contributo delle interazioni

Stimatore Saltelli (2010): richiede N×(d+2) valutazioni del surrogato (N=512, d=26 → ~14.336
valutazioni, ognuna < 1 ms → ~14 s totali).

Pearson r cattura solo relazioni lineari; Sobol S₁ >> |Pearson r| indica un effetto non-lineare
forte. Pearson r ≈ Sobol S₁ indica una relazione principalmente lineare.

Risultato tipico: A_glotis e d_trachea dominano sia S₁ che Sₜ per la pressione media;
gli angoli di ramificazione bronchiale hanno Sₜ elevato ma S₁ basso (principalmente interazioni).
""",
},

{
    "id": "qa_it_processo_gaussiano",
    "title": "Cos'è un Processo Gaussiano (GP)? (What is a Gaussian Process?)",
    "content": """
A Gaussian Process (GP) is a probabilistic surrogate model that provides not only predictions
but also uncertainty estimates (posterior standard deviation).

Italian / Italiano:
Un Processo Gaussiano (GP) è un modello surrogato probabilistico: oltre alla predizione
fornisce una stima dell'incertezza (deviazione standard posteriore).

Kernel comuni disponibili in questo progetto:
  - matern_52: C(1,0) × Matérn(ν=2.5) — buon default per dati fisici, differenziabile due volte
  - matern_32: C(1,0) × Matérn(ν=1.5) — caratteristiche più nette
  - rbf: C(1,0) × RBF(ℓ) — squared-exponential, liscio all'infinito
  - rational_quadratic: miscela di lunghezze di scala

Differenza rispetto a RBF (interpolazione):
  - RBF interpola esattamente i punti di addestramento (deterministico)
  - GP regredisce sui punti (probabilistico), ottimizza iperparametri via massima verosimiglianza

Attenzione alla collisione di nomi: il "kernel RBF" di sklearn (exp(−r²/2ℓ²)) e la funzione RBF
di scipy (interpolazione) sono matematicamente diverse, anche se il nome è simile.

In questo progetto: GP è più lento da addestrare (O(n³) con ottimizzazione iperparametri),
ma fornisce stime di incertezza utili per rilevare estrapolazioni fuori dal dominio di addestramento.
""",
},

]  # end of _DOCS


# ── TF-IDF retrieval ──────────────────────────────────────────────────────────

class RAGKnowledgeBase:
    """TF-IDF retrieval with optional sentence-transformer upgrade."""

    def __init__(self):
        self._docs   = _DOCS
        self._texts  = [d["title"] + "\n" + d["content"] for d in _DOCS]
        self._tfidf_matrix = None
        self._vocab  = {}
        self._semantic_embeddings = None
        self._build_tfidf()
        if _HAS_SEMANTIC:
            self._build_semantic()

    def _tokenise(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_tfidf(self):
        corpus    = self._texts
        tokenised = [self._tokenise(t) for t in corpus]
        vocab     = sorted({tok for doc in tokenised for tok in doc})
        self._vocab = {w: i for i, w in enumerate(vocab)}
        n, V = len(corpus), len(vocab)

        tf = np.zeros((n, V), dtype=np.float32)
        for d, tokens in enumerate(tokenised):
            for tok in tokens:
                if tok in self._vocab:
                    tf[d, self._vocab[tok]] += 1
            s = tf[d].sum()
            if s > 0:
                tf[d] /= s

        df  = (tf > 0).sum(axis=0).astype(np.float32)
        idf = np.log((n + 1) / (df + 1)) + 1.0
        self._tfidf_matrix = tf * idf

        norms = np.linalg.norm(self._tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._tfidf_matrix /= norms

    def _build_semantic(self):
        try:
            model = _get_semantic_model()
            self._semantic_embeddings = model.encode(
                self._texts, normalize_embeddings=True, show_progress_bar=False
            )
        except Exception:
            self._semantic_embeddings = None

    def _tfidf_query(self, query: str) -> np.ndarray:
        tokens = self._tokenise(query)
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        for tok in tokens:
            if tok in self._vocab:
                vec[self._vocab[tok]] += 1
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        return self._tfidf_matrix @ vec

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[float, dict]]:
        if _HAS_SEMANTIC and self._semantic_embeddings is not None:
            try:
                model = _get_semantic_model()
                q_emb  = model.encode([query], normalize_embeddings=True)[0]
                scores = self._semantic_embeddings @ q_emb
            except Exception:
                scores = self._tfidf_query(query)
        else:
            scores = self._tfidf_query(query)

        top_idx = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), self._docs[i]) for i in top_idx]

    def add_document(self, doc: dict):
        self._docs.append(doc)
        self._texts.append(doc["title"] + "\n" + doc["content"])
        self._build_tfidf()
        if _HAS_SEMANTIC:
            self._build_semantic()


# ── Local answer composition (no LLM) ────────────────────────────────────────

def compose_local_answer(query: str, results: List[Tuple[float, dict]],
                          kb: RAGKnowledgeBase) -> str:
    """
    Compose an answer from retrieved documents without any LLM.
    Strategy: find the top document, then extract the most relevant sentences.
    """
    if not results:
        return "No relevant information found in the knowledge base."

    best_score, best_doc = results[0]

    # If the top doc is a Q&A entry (title starts with 'Q:'), return its content directly
    if best_doc["title"].startswith("Q:"):
        content = best_doc["content"].strip()
        return f"**{best_doc['title']}**\n\n{content}"

    # For topic documents: extract the most relevant sentences
    query_toks = set(kb._tokenise(query))

    candidate_sentences = []
    for _, doc in results[:2]:
        for line in doc["content"].split("\n"):
            line = line.strip()
            if len(line) < 30:
                continue
            line_toks = set(kb._tokenise(line))
            overlap   = len(query_toks & line_toks)
            candidate_sentences.append((overlap, line, doc["title"]))

    candidate_sentences.sort(reverse=True)

    seen, answer_parts = set(), []
    answer_parts.append(f"**{best_doc['title']}**\n")

    for _, sent, _ in candidate_sentences[:5]:
        if sent not in seen and len(sent) > 40:
            answer_parts.append(sent)
            seen.add(sent)

    return "\n\n".join(answer_parts)


# ── Claude API generation (optional upgrade) ─────────────────────────────────

def _get_api_key() -> "str | None":
    def _valid(k: "str | None") -> "str | None":
        return k if (k and k.startswith("sk-ant-") and len(k) > 20) else None

    key = _valid(os.environ.get("ANTHROPIC_API_KEY"))
    if key:
        return key
    try:
        import streamlit as st
        return _valid(st.secrets.get("ANTHROPIC_API_KEY"))
    except Exception:
        return None


def generate_answer(query: str, context_chunks: List[str]) -> "str | None":
    api_key = _get_api_key()
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    context = "\n\n---\n\n".join(context_chunks)
    system  = (
        "You are an expert assistant for the Human Airways Digital Twin project "
        "at Università degli Studi di Roma Tor Vergata. Answer concisely and technically."
    )
    prompt = f"Context:\n\n{context}\n\nQuestion: {query}"
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg    = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=512,
            system=system, messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except anthropic.AuthenticationError:
        return None
    except Exception:
        return None


# ── Singleton ─────────────────────────────────────────────────────────────────

_kb: "RAGKnowledgeBase | None" = None

def get_knowledge_base() -> RAGKnowledgeBase:
    global _kb
    if _kb is None:
        _kb = RAGKnowledgeBase()
    return _kb
