"""
generate_report.py
==================
Generates report.pdf — Report of Activities Performed.

Run:
    python generate_report.py

Requires: reportlab  (pip install reportlab)

Students : Kuhyar Saeedi, Danial Mahmoody, Davood Jokar, Mahyar Emami, Nima Shahrokhi
Course   : Digital Twins - Modeling and Applications
Professor: Prof. Marco E. Biancolini
Year     : 2026-2027
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

OUT_PDF = Path(__file__).parent / "report.pdf"

BLUE   = colors.HexColor("#1A73C8")
DBLUE  = colors.HexColor("#0D47A1")
LBLUE  = colors.HexColor("#D6E8F7")
LGRAY  = colors.HexColor("#F5F5F5")
DGRAY  = colors.HexColor("#424242")
MGRAY  = colors.HexColor("#757575")
TEAL   = colors.HexColor("#00695C")
GREEN  = colors.HexColor("#2E7D32")
ORANGE = colors.HexColor("#E65100")

BASE = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle(
    "Title",
    fontSize=28, leading=34, alignment=TA_CENTER,
    textColor=DBLUE, fontName="Helvetica-Bold", spaceAfter=10,
)
SUBTITLE_STYLE = ParagraphStyle(
    "Subtitle",
    fontSize=13, leading=17, alignment=TA_CENTER,
    textColor=DGRAY, fontName="Helvetica", spaceAfter=5,
)
LINK_STYLE = ParagraphStyle(
    "Link",
    fontSize=12, leading=16, alignment=TA_CENTER,
    textColor=BLUE, fontName="Helvetica-Bold", spaceAfter=5,
)
H1 = ParagraphStyle(
    "H1",
    fontSize=15, leading=20, textColor=DBLUE,
    fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=8,
)
H2 = ParagraphStyle(
    "H2",
    fontSize=12, leading=16, textColor=TEAL,
    fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=5,
)
H3 = ParagraphStyle(
    "H3",
    fontSize=11, leading=15, textColor=ORANGE,
    fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4,
)
BODY = ParagraphStyle(
    "Body",
    fontSize=10, leading=15, alignment=TA_JUSTIFY,
    fontName="Helvetica", spaceBefore=4, spaceAfter=4,
)
BULLET = ParagraphStyle(
    "Bullet",
    fontSize=10, leading=14,
    fontName="Helvetica", leftIndent=18, bulletIndent=6,
    spaceBefore=2, spaceAfter=2,
)
CODE = ParagraphStyle(
    "Code",
    fontSize=9, leading=13,
    fontName="Courier", leftIndent=12, textColor=DGRAY,
    backColor=LGRAY, spaceBefore=4, spaceAfter=4,
)
CAPTION = ParagraphStyle(
    "Caption",
    fontSize=9, leading=12, alignment=TA_CENTER,
    fontName="Helvetica-Oblique", textColor=MGRAY,
)
NOTE = ParagraphStyle(
    "Note",
    fontSize=9, leading=13, alignment=TA_LEFT,
    fontName="Helvetica-Oblique", textColor=MGRAY,
    leftIndent=12, spaceBefore=2, spaceAfter=6,
)
PAGE_LABEL = ParagraphStyle(
    "PageLabel",
    fontSize=11, leading=15, textColor=BLUE,
    fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=3,
)
CELL_STYLE = ParagraphStyle(
    "Cell",
    fontSize=9, leading=13, alignment=TA_LEFT,
    fontName="Helvetica",
)


def hr(color=BLUE):
    return HRFlowable(width="100%", thickness=1, color=color, spaceAfter=8, spaceBefore=8)


def h1(text): return Paragraph(text, H1)
def h2(text): return Paragraph(text, H2)
def h3(text): return Paragraph(text, H3)
def p(text):  return Paragraph(text, BODY)
def bullet(text): return Paragraph(f"• {text}", BULLET)
def code(text):   return Paragraph(text, CODE)
def sp(h=0.3):    return Spacer(1, h * cm)
def note(text):   return Paragraph(text, NOTE)
def page_label(n, title): return Paragraph(f"Page {n}  —  {title}", PAGE_LABEL)


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(BLUE)
    canvas.rect(2 * cm, A4[1] - 1.8 * cm, A4[0] - 4 * cm, 0.35 * cm, fill=1, stroke=0)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(DGRAY)
    canvas.drawString(2 * cm, A4[1] - 2.4 * cm,
                      "Human Airways Digital Twin  |  Universita degli Studi di Roma Tor Vergata")
    canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 2.4 * cm,
                           "Report of Activities Performed")
    canvas.setFont("Helvetica", 8)
    canvas.drawString(2 * cm, 1.2 * cm,
                      "Digital Twins - Modeling and Applications  |  Prof. M.E. Biancolini  |  2026-2027")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def meta_table(data, col_widths, header_color=BLUE):
    """Table where body cells use Paragraph so long text wraps correctly."""
    table_data = []
    for r, row in enumerate(data):
        table_data.append([
            Paragraph(str(cell), ParagraphStyle(
                "th" if r == 0 else "td",
                fontSize=9, leading=13,
                fontName="Helvetica-Bold" if r == 0 else "Helvetica",
                textColor=colors.white if r == 0 else DGRAY,
            ))
            for cell in row
        ])
    tbl = Table(table_data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), header_color),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LGRAY, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return tbl


def simple_table(data, col_widths=None, header_color=BLUE):
    if col_widths is None:
        col_widths = [5 * cm, 10.5 * cm]
    return meta_table(data, col_widths, header_color)


def build_pdf():
    doc = BaseDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=2.5 * cm, rightMargin=2.5 * cm,
        topMargin=3.0 * cm, bottomMargin=2.5 * cm,
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=on_page)])

    story = []

    # ── TITLE PAGE ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 3.0 * cm))
    story.append(Paragraph("Human Airways", TITLE_STYLE))
    story.append(Paragraph("Digital Twin Dashboard", TITLE_STYLE))
    story.append(sp(0.5))
    story.append(Paragraph("Report of Activities Performed", SUBTITLE_STYLE))
    story.append(sp(0.3))
    story.append(Paragraph(
        "Digital Twins - Modeling and Applications",
        SUBTITLE_STYLE,
    ))
    story.append(Paragraph(
        "Kuhyar Saeedi  ·  Danial Mahmoody  ·  Davood Jokar",
        SUBTITLE_STYLE,
    ))
    story.append(Paragraph(
        "Mahyar Emami  ·  Nima Shahrokhi",
        SUBTITLE_STYLE,
    ))
    story.append(sp(0.3))
    story.append(Paragraph(
        "Universita degli Studi di Roma Tor Vergata  |  Prof. Marco E. Biancolini",
        SUBTITLE_STYLE,
    ))
    story.append(Paragraph("Academic Year 2026-2027", SUBTITLE_STYLE))
    story.append(sp(0.7))
    story.append(hr())
    story.append(sp(0.4))
    story.append(Paragraph(
        "Live Demo:  https://human-airways-digital-twin-wvmgvehsbnkkpfe3dndrnr.streamlit.app",
        LINK_STYLE,
    ))
    story.append(PageBreak())

    # ── SECTION 1: INTRODUCTION ────────────────────────────────────────────────
    story.append(h1("1.  Introduction"))
    story.append(hr())
    story.append(p(
        "This document describes all activities performed during the 2026-2027 academic year "
        "for the <b>Human Respiratory Airways Digital Twin</b> project. The dataset provided "
        "by Ansys Twin Builder consists of 100 CFD simulations of the human airway, each "
        "using a distinct geometry defined by 26 anatomical parameters covering the glottis, "
        "epiglottis, trachea, and three levels of bronchial branching."
    ))
    story.append(p(
        "The project deliverable is a <b>15-page multi-page Streamlit dashboard</b> "
        "that runs locally in a browser and is also deployed online at "
        "<b>https://human-airways-digital-twin-wvmgvehsbnkkpfe3dndrnr.streamlit.app</b>. It implements the complete "
        "digital twin pipeline from raw CFD binary files to interactive surrogate prediction, "
        "and includes advanced clinical tools, a neural surrogate, an offline AI assistant "
        "(optionally upgraded to full online responses by supplying a Claude API key), "
        "VR export, and drug deposition simulation."
    ))
    story.append(p(
        "Alongside the Streamlit app, a <b>PyQt5 desktop application</b> (<i>qt_app/</i>) was developed "
        "as a fully local, browser-free alternative. It exposes the same core functionality — "
        "Geometry Explorer, Pressure Field, DOE Analysis, POD Analysis, RBF Inference, "
        "Regional Analysis, Design Space, and AI Assistant — inside a dark-themed native window "
        "with a splash-screen progress loader. It shares the identical <b>core/</b> backend as the "
        "Streamlit dashboard and is launched with <b>python qt_app/run.py</b> "
        "(dependencies: requirements_qt.txt)."
    ))
    story.append(sp(0.5))
    story.append(p("<b>Technology stack:</b>"))
    tech = [
        ["Category",         "Libraries / Tools"],
        ["Dashboard",        "Streamlit (multi-page app, st.fragment, session state, components)"],
        ["Numerics",         "NumPy, SciPy (SVD, LHS, Sobol, RBF interpolation, stats)"],
        ["Machine learning", "scikit-learn (GaussianProcessRegressor), PyTorch, NVIDIA PhysicsNeMo"],
        ["Visualisation",    "Plotly (3D scatter, contour, parallel coordinates, animations)"],
        ["AI assistant",     "TF-IDF / sentence-transformers, extractive QA, Web Speech API"],
        ["Geometry",         "pyvista (Delaunay 3D triangulation, local only)"],
        ["Language support", "Python 3.11+, bilingual EN/IT via core/i18n.py"],
    ]
    story.append(simple_table(tech, [4 * cm, 11.5 * cm]))
    story.append(PageBreak())

    # ── SECTION 2: CORE PIPELINE ───────────────────────────────────────────────
    story.append(h1("2.  Core Pipeline (core/)"))
    story.append(hr())
    story.append(p(
        "All data loading, dimensionality reduction, and surrogate logic lives in the "
        "<b>core/</b> package. Every dashboard page imports from here."
    ))

    story.append(h2("2.1  Data I/O  (core/data_io.py)"))
    for b in [
        "Reads raw Twin Builder binary files: 8-byte count header followed by N x float64 values. "
        "Functions: load_snapshot_coords(), load_pressure_snapshot(), load_ref_coords().",
        "Cloud fallback built in: if raw binary snapshots are absent (online deployment), "
        "load_pressure_snapshot() reads from precomputed/all_pressures_s50.npz and "
        "load_snapshot_coords() reconstructs from the precomputed geometry POD.",
        "load_doe() reads Points/doe_point.csv (26 parameters for 100 runs). "
        "get_param_cols() extracts the 26 named geometry parameters.",
        "get_anatomical_regions() and region_slice() map node index ranges from "
        "Pressure/settings.json to named anatomical labels (glottis, epiglottis, trachea, "
        "left and right bronchi at three levels).",
        "STRIDE = 50 constant: every 50th node is loaded (~42,718 nodes). "
        "load_all_coords() and load_all_pressures() batch-load all 100 snapshots, "
        "preferring precomputed NPZ when available.",
        "PARAM_LABELS and PARAM_GROUPS: human-readable names and grouped layouts "
        "for the 26 parameters, used across slider widgets on multiple pages.",
    ]:
        story.append(bullet(b))

    story.append(h2("2.2  POD / SVD  (core/pod.py)"))
    for b in [
        "compute_pod(X): subtracts column mean, runs numpy.linalg.svd(full_matrices=False), "
        "returns (mean, modes, scores, singular_values).",
        "cumulative_energy(svals): fraction of total variance explained by each mode.",
        "modes_for_energy(svals, threshold): minimum number of modes for a given energy fraction.",
        "reconstruct(mean, modes, scores, k): reconstructs the field from the first k modes.",
        "reconstruction_error(X, mean, modes, k): relative L2 error between original and "
        "rank-k POD reconstruction.",
        "Key results: geometry POD — 14 modes for 99.09% variance (Mode 1: 75.35%); "
        "pressure POD — 3 modes for 99.07% variance (Mode 1 alone: 93.94%, "
        "Mode 2: +4.02%, Mode 3: +1.11%).",
    ]:
        story.append(bullet(b))

    story.append(h2("2.3  RBF Surrogate  (core/rbf.py)"))
    for b in [
        "build_rbf(params_norm, scores, kernel): solves the (n+d+1) x (n+d+1) linear system. "
        "Kernel options: thin_plate_spline (default, phi(r)=r² log r), multiquadric, linear, cubic.",
        "All 26 input parameters normalised to [0, 1] before training.",
        "predict(model, x_new): evaluates the RBF at a new point. Returns k pressure POD scores.",
        "loo_errors(params, scores, k_pod, kernel): leave-one-out cross-validation — trains on "
        "n-1 samples, predicts the removed. Runs 100 iterations (~30 s). "
        "Returns per-snapshot relative L2 errors (mean ~3-5%).",
        "kfold_errors(params, scores, k, kernel, seed): K-fold cross-validation with "
        "configurable folds (default 5). Returns per-fold mean error. "
        "For 100 samples, 10-fold and LOO give nearly identical results.",
        "Inference time: < 0.1 ms per query. Validated at 3-5% relative L2 error.",
    ]:
        story.append(bullet(b))

    story.append(h2("2.4  Gaussian Process Surrogate  (core/gp.py)"))
    for b in [
        "build_gp(params_norm, scores, kernel_name, n_restarts): one scikit-learn "
        "GaussianProcessRegressor per pressure POD mode (independent outputs).",
        "Four kernel options: RBF (squared exponential / Gaussian), Matern 3/2, "
        "Matern 5/2 (default, twice differentiable), Rational Quadratic.",
        "Kernel hyperparameters (length-scale, amplitude) optimised automatically via "
        "L-BFGS-B marginal log-likelihood maximisation with n_restarts restarts.",
        "predict(gp_models, x): returns both mean POD scores AND standard deviation (sigma) "
        "per output, providing query-level uncertainty quantification.",
        "kfold_errors(params, scores, k, kernel_name, n_restarts): same K-fold protocol as RBF "
        "for head-to-head benchmarking.",
        "Training time: 1-5 s (vs ~2 ms for RBF). Inference: ~2-10 ms per query (vs < 0.1 ms). "
        "Accuracy: comparable to RBF at 3-5% relative L2 error.",
    ]:
        story.append(bullet(b))

    story.append(h2("2.5  LHS Sampling  (core/lhs.py)"))
    for b in [
        "latin_hypercube(n, lo, hi, seed): generates n samples in d dimensions using "
        "scipy.stats.qmc.LatinHypercube with reproducible seed. "
        "Exactly one sample per bin per dimension.",
        "doe_bounds(params): returns per-parameter min/max from the DOE table.",
        "discrepancy_score(samples): L2-discrepancy metric for coverage quality comparison.",
        "Used to generate 1000 virtual airway shapes within the original DOE bounds.",
    ]:
        story.append(bullet(b))

    story.append(h2("2.6  AI Assistant Backend  (core/rag.py)"))
    for b in [
        "33-document knowledge base covering project methodology, results, and exam topics.",
        "Retrieval: TF-IDF (always available) or sentence-transformers all-MiniLM-L6-v2 "
        "(optional, higher quality). Falls back gracefully if not installed.",
        "compose_local_answer(): extractive QA — assembles answers from retrieved document "
        "chunks entirely offline. No API key or internet required.",
        "generate_answer(): upgrades to Claude API if ANTHROPIC_API_KEY is set. "
        "Falls back to extractive QA otherwise.",
    ]:
        story.append(bullet(b))

    story.append(h2("2.7  Internationalisation  (core/i18n.py)"))
    for b in [
        "lang_selector(): sidebar widget to toggle between English and Italian.",
        "t(key): returns the localised string for the current session language.",
        "All 15 pages and the home page are fully bilingual — every user-visible "
        "string goes through t().",
    ]:
        story.append(bullet(b))
    story.append(PageBreak())

    # ── SECTION 3: DASHBOARD PAGES ────────────────────────────────────────────
    story.append(h1("3.  Streamlit Dashboard — 15 Pages"))
    story.append(hr())
    story.append(p(
        "The dashboard is a Streamlit multi-page application. The home page (app.py) "
        "is the entry point; the 15 sub-pages are in the pages/ directory. "
        "Navigation is via the Streamlit sidebar. Every page is bilingual (EN/IT)."
    ))

    story.append(h2("Home Page  (app.py)"))
    for b in [
        "Five metric cards at the top: DOE snapshots (100), geometry parameters (26), "
        "full mesh nodes (2,135,906), visualised nodes (~42,700 at stride 50), "
        "and static pressure field.",
        "Left column: project overview in the active language, anatomy tree diagram "
        "from mouth/larynx to left and right bronchi at three levels.",
        "Right column: nine methodology steps as expandable widgets (inspect database, "
        "geometry POD, LHS, RBF, pressure POD, coupled prediction, dashboard, CSV export, "
        "VTK-compatible export).",
        "Bottom: navigation grid with icons and one-line descriptions for all 15 pages.",
    ]:
        story.append(bullet(b))

    pages_desc = [
        (
            "1", "Geometry Explorer",
            "Real-time POD-based airway shape morphing.",
            [
                "Loads precomputed geometry POD from precomputed/pod_geometry.npz "
                "(falls back to computing SVD on all 100 snapshots if file absent).",
                "Sliders for the first N geometry POD modes, implemented with st.fragment "
                "for sub-second re-render on every slider change.",
                "Colour options: by displacement from mean shape, by POD mode amplitude, "
                "or by Z-coordinate (depth through the airway).",
                "Mean shape ghost overlay: renders the mean geometry as a faint backdrop "
                "so the user can see deformation relative to average.",
                "Snapshot selector: jump to any of the 100 real DOE snapshots and display "
                "its actual POD coefficients on the sliders.",
                "Sidebar: cumulative energy bar, mode count for 95% and 99% thresholds.",
            ],
        ),
        (
            "2", "Pressure Field",
            "Interactive 3D pressure viewer for all 100 DOE snapshots.",
            [
                "Snapshot slider (1-100) in the sidebar. On cloud: reads from "
                "precomputed/all_pressures_s50.npz (no raw binary file needed).",
                "Toggle between reference (mean) mesh and the snapshot-specific deformed geometry.",
                "Anatomical region filter: restricts the 3D view to a named region "
                "(epiglottis, glottis, upper trachea, left/right bronchi, etc.).",
                "Colour map selector and opacity slider.",
                "Four metric cards: min / max / mean / std dev of pressure for the "
                "selected snapshot.",
                "Sidebar panel shows the 26 geometry parameter values for the selected run.",
            ],
        ),
        (
            "3", "DOE Analysis",
            "Design of Experiments parameter space exploration — 4 tabs.",
            [
                "Tab 1 — Parallel Coordinates: all 26 parameters as vertical axes; "
                "each polyline is one DOE run, coloured by snapshot index. "
                "Sidebar slider highlights a specific run in a different colour.",
                "Tab 2 — Pairwise Scatter: two-parameter dropdown selectors, "
                "scatter plot with linear regression line and R2 annotation.",
                "Tab 3 — Correlation Heatmap: 26x26 Pearson correlation matrix. "
                "Identifies redundant parameter pairs (|r| > 0.8) and independent drivers.",
                "Tab 4 — LHS Coverage: side-by-side comparison of the original 100-point DOE "
                "vs a 1000-point LHS set for a user-selected parameter pair. "
                "Displays L2-discrepancy scores for both.",
            ],
        ),
        (
            "4", "POD Analysis",
            "Full decomposition analysis for geometry and pressure — 4 sections.",
            [
                "Energy Curves: cumulative variance vs mode number for both fields, "
                "with vertical lines at 95% and 99% thresholds and mode counts displayed.",
                "Mode Shapes: 3D point cloud coloured by a selected POD mode's spatial "
                "pattern (positive/negative regions). Mode selector dropdown.",
                "Reconstruction Comparison: select a snapshot and k; side-by-side original "
                "vs POD-reconstructed field with relative L2 error.",
                "Validation: K-Fold (5-fold) and LOO reconstruction error statistics "
                "as bar charts and summary tables for both geometry and pressure fields.",
            ],
        ),
        (
            "5", "RBF / GP Inference",
            "Surrogate model prediction, validation, and GP vs RBF benchmark — 7 tabs.",
            [
                "Tab 1 — Predict New Shape: 26 sliders grouped by anatomy "
                "(main airway, first-level branches, second-level sub-branches). "
                "Predict button triggers RBF inference: 3 pressure POD scores predicted, "
                "full 42K-node field reconstructed. Displays 6 metrics: "
                "min / max / mean / std pressure, airway resistance delta-P (Pa), "
                "and resistance index (delta-P relative to mean-shape baseline, 100% = average).",
                "Tab 2 — LOO Validation: runs 100 iterations, each rebuilding RBF on 99 samples "
                "and predicting the removed snapshot. Bar chart of per-snapshot error coloured "
                "by magnitude. Metric cards: mean LOO error, max error, min error, std of errors.",
                "Tab 3 — K-Fold Validation: configurable 2-10 folds (default 5), configurable seed. "
                "Per-fold error bar chart with numeric labels. Metric cards: mean CV error, "
                "std, best fold, worst fold. Comparison table: LOO takes ~30 s (100 models), "
                "5-fold takes ~2 s (5 models); for 100 samples LOO and 10-fold give nearly "
                "identical results.",
                "Tab 4 — Convergence Study: sweeps POD mode count k from 1 to configurable max. "
                "Choice of 5-fold (fast) or LOO (slow). Plots mean CV error vs k; "
                "marks optimal k at the error minimum. Metric cards: optimal k, min CV error, "
                "error at 99%-energy k.",
                "Tab 5 — 1000 Virtual Shapes: LHS generates 1000 parameter sets; "
                "RBF predicts pressure for all. Histogram of mean pressure distribution, "
                "histogram of airway resistance (delta-P) distribution with mean-shape "
                "baseline marker, and 2D design space coloured by delta-P.",
                "Tab 6 — Patient Demo: illustrative clinical workflow showing how CT-scan "
                "measurements map to the 26 parameter sliders, with live trachea/glottis "
                "sliders updating the resistance index and colour-coded alert (normal / "
                "slightly elevated / high resistance).",
                "Tab 7 — RBF vs GP Comparison: trains both surrogates on the full dataset, "
                "times inference over 200 queries, runs K-fold CV for both. "
                "Displays: training time bar (log scale), per-fold CV accuracy side by side, "
                "inference latency bar, and predicted field comparison at mean parameters. "
                "Kernel selector for both surrogate types. GP provides sigma uncertainty per "
                "prediction; RBF is deterministic. Typical results: RBF trains in ~2 ms, "
                "GP in 1-5 s; both achieve 3-5% relative L2 error in K-fold CV.",
            ],
        ),
        (
            "6", "Regional Analysis",
            "Mean static pressure per anatomical region — 3 tabs.",
            [
                "Per-Snapshot View: bar chart of mean +/- std pressure for every named "
                "region in the selected snapshot. Sidebar: snapshot slider and sort order "
                "(by mean pressure or by anatomical depth).",
                "Cross-Snapshot View: heatmap (100 snapshots x regions) showing pressure "
                "variation across the full DOE. Reveals which regions are most sensitive "
                "to geometry changes.",
                "Export tab: CSV download of regional statistics (mean, std, min, max per "
                "region per snapshot) for external analysis.",
            ],
        ),
        (
            "7", "Design Space",
            "Parameter sensitivity and landscape — 4 tabs.",
            [
                "Tab 1 — Parameter Sensitivity: Pearson correlation of each of the 26 "
                "parameters with mean pressure across 100 snapshots. "
                "Sorted horizontal bar chart; identifies dominant anatomical drivers.",
                "Tab 2 — Design Landscape: 2D scatter of all 100 runs in the first two "
                "geometry POD score dimensions, coloured by mean pressure. "
                "Reveals clustering and outliers.",
                "Tab 3 — Param-Pressure Matrix: scatter grid of top-k most correlated "
                "parameters vs mean pressure with regression lines.",
                "Tab 4 — Sobol Indices: variance-based global sensitivity computed from "
                "1000 LHS virtual shapes. Bar chart of first-order and total-order "
                "Sobol indices for all 26 parameters.",
            ],
        ),
        (
            "8", "Animations",
            "Animated POD visualisations generated on demand.",
            [
                "Requires precomputed NPZ files. Stops with a clear error if these are missing.",
                "Animation 1 — POD Mode Sweep: user selects a geometry mode; Plotly "
                "animated scatter sweeps the coefficient from -2 to +2 standard deviations.",
                "Animation 2 — Pressure Snapshot Reel: animated sequence through all 100 "
                "DOE pressure snapshots, capped at 25 frames for memory safety.",
                "All animations generated on button click to avoid blocking page load.",
            ],
        ),
        (
            "9", "Ask AI",
            "Offline AI virtual assistant with animated avatar and voice.",
            [
                "Animated avatar with four states: idle (blue), listening (green), "
                "thinking (amber), speaking (purple). State indicator updates reactively.",
                "Eight pre-suggested questions displayed as buttons.",
                "Voice input: browser Web Speech API (Chrome/Edge). No server-side audio.",
                "Voice output: browser speechSynthesis API, fully offline.",
                "Backend: core/rag.py — 33-document knowledge base, TF-IDF retrieval, "
                "extractive QA. Upgrades to Claude API if ANTHROPIC_API_KEY is set.",
                "Scrollable chat history with user/assistant message bubbles, "
                "persisted in st.session_state for the session.",
            ],
        ),
        (
            "10", "3D / VR Viewer",
            "WebXR-ready immersive airway viewer.",
            [
                "Renders geometry.stl as a solid lit mesh via Three.js/WebXR inside "
                "a Streamlit iframe. Falls back gracefully if geometry.stl is absent.",
                "Optionally overlays the pressure or deformation field as a coloured "
                "point cloud on top of the solid mesh.",
                "Orbit/zoom/pan with mouse on desktop. Fly mode with WASD keys. "
                "Enter VR button for Meta Quest / Chrome+OpenXR (HTTPS or localhost required).",
                "Standalone HTML download: self-contained file that opens in a VR browser.",
            ],
        ),
        (
            "11", "Mesh Viewer",
            "Solid surface mesh from point cloud via Delaunay triangulation.",
            [
                "Loads pre-computed STL files from export/mesh/ if available. "
                "Otherwise runs pyvista Delaunay 3D on-the-fly; if pyvista is absent "
                "(cloud deployment), falls back to point cloud only with a warning.",
                "Side-by-side panels: point cloud (left) and solid mesh (right), "
                "both coloured by pressure when a snapshot is loaded.",
                "STL download button per snapshot for use in HELYX / OpenFOAM / Blender.",
            ],
        ),
        (
            "12", "Delivery Packages",
            "Status overview of the three data delivery packages.",
            [
                "Three panels with status indicators: VR Archive, HELYX STL package, "
                "Domino dataset. Each shows ready/not-generated and file sizes.",
                "VR Archive: binary Float32 blobs of all 100 geometry snapshots, "
                "Three.js/WebXR viewer, served at http://localhost:8765.",
                "HELYX Package: STL surface files for HELYX Open-Source CFD / OpenFOAM.",
                "Domino Dataset: NPZ + CSV files for NVIDIA DoMINO/PhysicsNeMo training.",
                "Live data previews where a package is ready.",
            ],
        ),
        (
            "13", "AI Surrogate",
            "NVIDIA PhysicsNeMo neural network surrogate training report.",
            [
                "Graceful error with install instructions if PyTorch/PhysicsNeMo are absent.",
                "Three-step training journey: "
                "(1) Naive — 80 real snapshots, 68,355 parameters, overfitting, "
                "RMSE = 24.67 Pa, relative error = 21.8%. "
                "(2) Diagnosis — DoMINO-exported virtual shapes had geometry and pressure "
                "scores sampled independently, no physical link, would train on noise. "
                "(3) Knowledge Distillation — RBF generates 1000 physically-correct "
                "pseudo-labels; 80 real runs repeated 3x for ground-truth weighting; "
                "total 1,300 training samples. Result: RMSE = 3.22 Pa, error = 4.3%.",
                "Architecture: FullyConnected, 3 hidden layers, SiLU activation. "
                "Input: 14 geometry POD modes. Output: 3 pressure POD modes.",
                "Live inference from checkpoints/physicsnemo_surrogate.pt. "
                "RBF vs NN comparison panel for the same input.",
            ],
        ),
        (
            "14", "Drug Deposition",
            "Physics-based inhaled particle deposition simulator.",
            [
                "Local airflow velocity derived from RBF-predicted pressure field: "
                "proportional to inlet velocity x glottis area / local cross-section / "
                "number of branches at that depth.",
                "Three deposition mechanisms: "
                "(1) Inertial Impaction — Stokes number at bends/bifurcations, large particles. "
                "(2) Gravitational Sedimentation — terminal settling velocity, intermediate particles. "
                "(3) Brownian Diffusion — Einstein-Smoluchowski thermal motion, nano-particles.",
                "Total deposition per node = 1 minus product of (1 - P_mech) for each mechanism.",
                "Inputs: DOE snapshot (1-100), particle diameter (µm), injection velocity (m/s), "
                "inlet velocity (m/s).",
                "Optimal size sweep: evaluates 60 diameters from 0.5 to 20 µm. Plots bronchial "
                "deposition fraction vs size; identifies optimal diameter.",
                "3D heatmap of pressure field coloured by deposition probability per node.",
            ],
        ),
        (
            "15", "Patient Comparison",
            "Two-patient side-by-side airway comparison.",
            [
                "Each patient defined independently: DOE snapshot (real CFD) or "
                "custom RBF prediction via 26 geometry sliders.",
                "Two 3D scatter plots with a shared colour scale for direct comparison.",
                "Resistance metrics for both: inlet-to-outlet delta-P, mean pressure, "
                "normalised airway resistance index.",
                "Delta-pressure map: third 3D plot showing pressure A minus pressure B "
                "at every node.",
                "Regional bar charts: mean pressure per anatomical region for both "
                "patients on shared axes.",
                "Top-5 parameter difference table: five geometry parameters differing "
                "most between the two configurations, identified automatically.",
            ],
        ),
    ]

    for num, title, subtitle, bullets in pages_desc:
        story.append(KeepTogether([
            page_label(num, title),
            Paragraph(subtitle, BODY),
        ]))
        for b in bullets:
            story.append(bullet(b))
        story.append(sp(0.3))

    story.append(PageBreak())

    # ── SECTION 4: EXPORT SCRIPTS ─────────────────────────────────────────────
    story.append(h1("4.  Export Scripts  (scripts/)"))
    story.append(hr())
    story.append(p(
        "Five standalone Python scripts generate delivery packages from the digital twin. "
        "They run independently from the dashboard."
    ))

    scripts = [
        ("scripts/export_mesh.py",
         "Reconstructs surface meshes from all 100 point clouds via pyvista Delaunay 3D "
         "(~2.7 s per snapshot). Exports STL (HELYX/MeshLab) and VTP (ParaView)."),
        ("scripts/export_1000_shapes.py",
         "Generates 1000 LHS virtual airway shapes. Exports params CSV, "
         "summary CSV (predicted delta-P per shape), and POD score NPY arrays. "
         "Optional --stl flag exports STL/VTP per shape."),
        ("scripts/export_vr_archive.py",
         "Packages all 100 snapshots as Float32 binary blobs plus a standalone "
         "Three.js/WebXR viewer and a serve.py local HTTP server."),
        ("scripts/export_physicsnemo.py",
         "Packages shapes for NVIDIA PhysicsNeMo/DoMINO training: "
         "geometry STL and surface pressure VTP per shape, "
         "manifest.csv, config_domino.yaml. RBF provides physically-correct pressure "
         "pseudo-labels for virtual shapes."),
        ("scripts/export_helyx.py",
         "Assembles HELYX CFD delivery: STL files, parameter tables, "
         "HELYX_README.txt. Output: export/helyx/ + helyx_package.zip."),
    ]
    for name, desc in scripts:
        story.append(KeepTogether([
            h2(name),
            p(desc),
            sp(0.1),
        ]))

    story.append(PageBreak())

    # ── SECTION 5: KEY RESULTS ─────────────────────────────────────────────────
    story.append(h1("5.  Key Results"))
    story.append(hr())

    results = [
        ["Metric",                           "Value"],
        ["DOE snapshots",                    "100"],
        ["Geometry parameters",              "26"],
        ["Full mesh nodes per snapshot",     "2,135,906"],
        ["Visualisation nodes (stride 50)",  "~42,718"],
        ["Geometry POD Mode 1 energy",       "75.35%"],
        ["Geometry POD -- 95% variance",     "6 modes  (95.58%)"],
        ["Geometry POD -- 99% variance",     "14 modes  (99.09%)"],
        ["Pressure POD Mode 1 energy",       "93.94%"],
        ["Pressure POD Mode 2 energy",       "+4.02%  (cumulative 97.96%)"],
        ["Pressure POD Mode 3 energy",       "+1.11%  (cumulative 99.07%)"],
        ["Pressure POD -- 99% variance",     "3 modes"],
        ["RBF -- LOO mean error",            "~3-5% relative L2"],
        ["RBF -- K-Fold (5-fold) mean error","~3-5% relative L2  (comparable to LOO)"],
        ["RBF training time",                "~2 ms"],
        ["RBF inference time",               "< 0.1 ms per query"],
        ["GP training time",                 "1-5 s (kernel hyperparameter optimisation)"],
        ["GP inference time",                "~2-10 ms per query"],
        ["GP K-Fold accuracy",               "Comparable to RBF at 3-5% relative L2"],
        ["1000-shape LHS inference (RBF)",   "< 1 s total"],
        ["CFD speedup",                      "> 10,000x vs new simulation"],
        ["PhysicsNeMo -- naive error",       "21.8%  (RMSE = 24.67 Pa, 80 samples)"],
        ["PhysicsNeMo -- distilled error",   "4.3%  (RMSE = 3.22 Pa, 1,300 samples)"],
        ["NN architecture",                  "FullyConnected, 3 hidden layers, SiLU, 68,355 params"],
        ["AI knowledge base",                "33 documents"],
        ["Dashboard pages",                  "15 (+ home page), bilingual EN/IT"],
    ]
    story.append(simple_table(results, [7.5 * cm, 8.0 * cm], DBLUE))
    story.append(PageBreak())

    # ── SECTION 6: CONCLUSIONS ─────────────────────────────────────────────────
    story.append(h1("6.  Conclusions"))
    story.append(hr())
    story.append(p(
        "The Human Airways Digital Twin project delivered a comprehensive 15-page Streamlit "
        "dashboard implementing the complete POD-RBF pipeline from raw Ansys Twin Builder "
        "CFD binaries to interactive clinical tools. The dashboard is available locally "
        "and online at <b>https://human-airways-digital-twin-wvmgvehsbnkkpfe3dndrnr.streamlit.app</b>."
    ))
    for b in [
        "POD: 2.1M-node geometry reduced to 14 coefficients (99.09% variance); "
        "2.1M-node pressure to 3 coefficients (99.07% variance).",
        "RBF surrogate: 3-5% relative L2 error by both LOO and K-Fold. Inference < 0.1 ms.",
        "Gaussian Process surrogate: comparable accuracy with per-query uncertainty sigma. "
        "Four kernel options. Trained on the same 100 CFD snapshots.",
        "1000 virtual airways via LHS + RBF in < 1 second. Zero additional CFD cost.",
        "PhysicsNeMo neural surrogate: 4.3% error after knowledge distillation from RBF "
        "(down from 21.8% naive).",
        "Drug Deposition Simulator: three-mechanism physics model with optimal particle "
        "size sweep across 60 diameters.",
        "Two-Patient Comparison: delta-pressure maps, regional breakdown, parameter diff table.",
        "Offline AI assistant: 33-document knowledge base, TF-IDF retrieval, voice I/O.",
        "Bilingual (EN/IT) across all 15 pages. Three professional delivery packages.",
    ]:
        story.append(bullet(b))
    story.append(sp(0.5))
    story.append(hr())
    story.append(Paragraph(
        "Human Airways Digital Twin  |  Universita di Roma Tor Vergata  |  2026-2027",
        CAPTION,
    ))

    doc.build(story)
    print(f"Report written to: {OUT_PDF}")


if __name__ == "__main__":
    build_pdf()
