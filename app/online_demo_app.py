"""Lightweight online Streamlit demo for the human airways digital twin project.

This app is designed for sharing with a professor or reviewer online.

Important:
This app does not load the heavy raw snapshots, full PCA models, or full
surrogate models. It loads a small downsampled demo asset from demo_assets/.

Why:
GitHub and Streamlit Cloud are not good places for multi-gigabyte scientific
files. This demo gives a fast visual preview while the full pipeline remains
available locally.

Run locally:

    streamlit run app/online_demo_app.py
"""

# This line allows Python type hints to be handled in a modern, flexible way.
from __future__ import annotations

# json reads small metadata about the demo asset.
import json

# Path gives clean filesystem path handling.
from pathlib import Path

# NumPy loads the compressed demo arrays.
import numpy as np

# pandas displays metric tables.
import pandas as pd

# Plotly creates an interactive 3D view that supports rotate, zoom, and pan.
import plotly.graph_objects as go

# Streamlit builds the web app.
import streamlit as st


# PROJECT_ROOT points to the main project folder, one level above app/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DEMO_ASSET_PATH is the small compressed file created by script 09.
DEMO_ASSET_PATH = PROJECT_ROOT / "demo_assets" / "airway_demo_sample.npz"

# DEMO_METADATA_PATH explains where the demo asset came from.
DEMO_METADATA_PATH = PROJECT_ROOT / "demo_assets" / "airway_demo_metadata.json"


@st.cache_data
def load_demo_asset() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load the small demo point cloud and pressure array."""

    # If the demo file is missing, stop with a clear instruction.
    if not DEMO_ASSET_PATH.exists():
        raise FileNotFoundError(
            f"Missing demo asset: {DEMO_ASSET_PATH}. "
            "Run python scripts/09_create_online_demo_assets.py first."
        )

    # Load the compressed NumPy file.
    data = np.load(DEMO_ASSET_PATH)

    # Extract x, y, z points.
    points = data["points"]

    # Extract scalar pressure values.
    pressure = data["pressure"]

    # Load metadata if available.
    if DEMO_METADATA_PATH.exists():
        metadata = json.loads(DEMO_METADATA_PATH.read_text(encoding="utf-8"))
    else:
        metadata = {}

    # Return all demo data.
    return points, pressure, metadata


def compute_demo_metrics(points: np.ndarray, pressure: np.ndarray) -> dict[str, float]:
    """Compute simple metrics for the demo asset."""

    # Compute the average point location.
    centroid = np.mean(points, axis=0)

    # Compute simple radial distances in the xy plane.
    radial = np.sqrt((points[:, 0] - centroid[0]) ** 2 + (points[:, 1] - centroid[1]) ** 2)

    # Compute pressure and geometry summary values.
    return {
        "number_of_demo_points": int(points.shape[0]),
        "pressure_min": float(np.min(pressure)),
        "pressure_max": float(np.max(pressure)),
        "pressure_mean": float(np.mean(pressure)),
        "pressure_std": float(np.std(pressure)),
        "mean_radius_proxy": float(np.mean(radial)),
        "max_radius_proxy": float(np.max(radial)),
    }


def make_3d_figure(points: np.ndarray, pressure: np.ndarray, pressure_scale: float) -> go.Figure:
    """Create an interactive 3D airway plot."""

    # Scale pressure only for visual demonstration.
    display_pressure = pressure * pressure_scale

    # Create an interactive 3D point cloud.
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker={
            "size": 2,
            "color": display_pressure,
            "colorscale": "Viridis",
            "opacity": 0.85,
            "showscale": True,
            "colorbar": {"title": "Pressure"},
        },
        hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
    )

    # Create a figure object from the scatter plot.
    fig = go.Figure(data=[scatter])

    # Configure the 3D scene.
    fig.update_layout(
        title="Interactive Airway Demo",
        height=650,
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "aspectmode": "data",
        },
    )

    # Return the interactive figure.
    return fig


def main() -> None:
    """Render the lightweight online demo app."""

    # Configure the browser page.
    st.set_page_config(page_title="Airways Digital Twin Demo", layout="wide")

    # Show a clear title.
    st.title("Human Airways Digital Twin - Lightweight Online Demo")

    # Explain the purpose of the app.
    st.caption("A small deployable demo for online sharing. The full training pipeline remains local.")

    # Add links between demo and blog/main app when available.
    st.page_link("pages/01_Project_Blog.py", label="Open Project Blog")

    # Load the demo data.
    try:
        points, pressure, metadata = load_demo_asset()
    except FileNotFoundError as error:
        st.error(str(error))
        return

    # Sidebar controls for presentation.
    st.sidebar.header("Demo Controls")

    # This scale changes the visual color intensity, not the trained model.
    pressure_scale = st.sidebar.slider("Pressure color scale", 0.2, 2.0, 1.0, 0.1)

    # Explain that this is a visual demo, not a live surrogate prediction.
    st.sidebar.info(
        "This online version uses a small precomputed sample. "
        "It is designed for sharing progress without uploading heavy data."
    )

    # Compute metrics for display.
    metrics = compute_demo_metrics(points, pressure)

    # Show compact metric cards.
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Demo Points", f"{metrics['number_of_demo_points']:,}")
    col2.metric("Pressure Mean", f"{metrics['pressure_mean']:.3g}")
    col3.metric("Pressure Range", f"{metrics['pressure_max'] - metrics['pressure_min']:.3g}")
    col4.metric("Mean Radius Proxy", f"{metrics['mean_radius_proxy']:.3g}")

    # Show the interactive 3D point cloud.
    st.subheader("Interactive 3D Airway View")
    st.plotly_chart(make_3d_figure(points, pressure, pressure_scale), use_container_width=True)
    st.caption("Rotate, zoom, and pan the airway using mouse or touch gestures.")

    # Show the metrics table and metadata side by side.
    table_col, meta_col = st.columns([1, 1])
    with table_col:
        st.subheader("Demo Metrics")
        st.dataframe(pd.DataFrame([metrics]), use_container_width=True)

    with meta_col:
        st.subheader("Demo Asset Information")
        st.json(metadata)

    # Explain the recommended professor-sharing workflow.
    st.subheader("How This Online Demo Should Be Used")
    st.markdown(
        """
        This lightweight app is meant for online presentation:

        - It can be deployed on Streamlit Community Cloud.
        - It avoids heavy snapshot files and huge model files.
        - It shows the airway geometry, pressure coloring, and project progress.
        - The full scientific pipeline remains in the repository and can run locally.
        """
    )


# This makes main() run only when Streamlit starts this file.
if __name__ == "__main__":
    main()
