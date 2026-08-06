# Human Airways Digital Twin

Interactive Streamlit dashboard for a data-driven digital twin of the human
respiratory system, built on Ansys Twin Builder / CFD simulations.

Course: Digital Twin Methods — Università degli Studi di Roma Tor Vergata

## What it does

100 Design-of-Experiment (DOE) CFD runs, each with a different airway geometry
(26 parameters: glottis area, epiglottis area, trachea dimensions, branching
angles), provide 3D mesh coordinates and static pressure fields. The pipeline
compresses these with POD/SVD and trains RBF / Gaussian Process surrogates so
new geometries can be predicted in milliseconds instead of re-running CFD.

Pages:

- **Geometry Explorer** — morph the airway shape via POD mode sliders
- **Pressure Field** — 3D pressure map for any of the 100 DOE snapshots
- **DOE Analysis** — parallel coordinates, scatter plots, correlation heatmap, LHS coverage
- **POD Analysis** — energy curves, mode shapes, reconstruction error
- **RBF / GP Inference** — predict the full pressure field for any new 26-parameter geometry
- **Regional Analysis** — mean pressure per anatomical region
- **Design Space, Animations, 3D/VR Viewer, Mesh Viewer, Delivery Packages,
  AI Surrogate, Drug Deposition, Patient Comparison, Ask AI** — additional
  exploration and reporting tools

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Optional features

A few pages use optional heavy dependencies (`pyvista`, `trimesh`,
`sentence_transformers`, `torch` + `physicsnemo`) that are **not** required —
those pages detect their absence and fall back gracefully. Install them only
if you want the full mesh-viewer / semantic-search / neural-surrogate
experience:

```bash
pip install pyvista trimesh sentence_transformers torch physicsnemo
```

The **Ask AI** page works out of the box with a local, offline TF-IDF
extractive Q&A engine. It optionally upgrades to Claude if an
`ANTHROPIC_API_KEY` is available (as an environment variable, or via
`.streamlit/secrets.toml` — never commit that file).

## Deploying on Streamlit Community Cloud

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. "New app" → select this repo/branch → main file path: `app.py` → Deploy.
4. (Optional) In app settings → Secrets, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```

## Project structure

```
app.py                  Landing page
pages/                  Multi-page Streamlit app (1_Geometry_Explorer.py, ...)
core/                   Shared logic: data I/O, POD, RBF, GP, LHS, RAG, i18n
precomputed/            Precomputed POD bases and pressure fields (.npz)
scripts/                Offline precompute / data-prep scripts
qt_app/                 Standalone PyQt desktop viewer (not used by Streamlit)
```
