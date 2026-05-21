"""Interactive human airways digital twin dashboard.

Run from the project root with:

    streamlit run app/airways_digital_twin_app.py

The app loads trained surrogate models and PCA models, maps DOE parameters to
POD coefficients, reconstructs predicted geometry and pressure fields, computes
simple proxy metrics, and optionally exports a ParaView-readable VTP file.
"""

# This import enables modern type-hint behavior.
from __future__ import annotations

# Path gives clean filesystem path handling.
from pathlib import Path

# joblib loads trained surrogate models, scalers, and PCA models.
import joblib

# matplotlib creates 2D plots and histograms inside Streamlit.
import matplotlib.pyplot as plt

# NumPy handles prediction arrays and metric calculations.
import numpy as np

# pandas reads DOE tables and displays metric tables.
import pandas as pd

# Streamlit builds the interactive web dashboard.
import streamlit as st


# PROJECT_ROOT points to the main project folder, one level above app/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DATA_ROOT points to the raw project data folder.
DATA_ROOT = PROJECT_ROOT / "data"

# OUTPUT_ROOT points to generated outputs.
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# MODELS_DIR points to saved PCA and surrogate models.
MODELS_DIR = OUTPUT_ROOT / "models"

# PREDICTIONS_DIR points to saved predictions.
PREDICTIONS_DIR = OUTPUT_ROOT / "predictions"

# SMALL_EPSILON prevents division by zero in proxy resistance calculations.
SMALL_EPSILON = 1e-12


# REQUIRED_FILES lists every trained artifact needed by the dashboard.
REQUIRED_FILES = {
    "geometry surrogate model": MODELS_DIR / "geometry_best_surrogate.joblib",
    "pressure surrogate model": MODELS_DIR / "pressure_best_surrogate.joblib",
    "geometry input scaler": MODELS_DIR / "geometry_input_scaler.joblib",
    "pressure input scaler": MODELS_DIR / "pressure_input_scaler.joblib",
    "geometry PCA model": MODELS_DIR / "geometry_pca_model.joblib",
    "pressure PCA model": MODELS_DIR / "pressure_pca_model.joblib",
    "geometry feature names": MODELS_DIR / "geometry_feature_names.txt",
    "pressure feature names": MODELS_DIR / "pressure_feature_names.txt",
    "geometry DOE": DATA_ROOT / "geometry" / "doe.csv",
    "pressure DOE": DATA_ROOT / "pressure" / "doe.csv",
}


def relative_path(path: Path) -> str:
    # Convert a project path to a shorter path relative to the project root.
    return str(path.relative_to(PROJECT_ROOT))


def script_hint_for_file(path: Path) -> str:
    # PCA model files are created by the POD/PCA compression script.
    if path.name.endswith("_pca_model.joblib"):
        return "scripts/04_pod_pca_compression.py"

    # Surrogate, scaler, and feature-name files are created by the surrogate training script.
    if (
        path.name.endswith("_best_surrogate.joblib")
        or path.name.endswith("_input_scaler.joblib")
        or path.name.endswith("_feature_names.txt")
    ):
        return "scripts/05_train_surrogate_models.py"

    # DOE CSV files are part of the dataset and should exist before training begins.
    if path.name == "doe.csv":
        return "scripts/01_check_dataset.py"

    # Fall back to the full pipeline when the file type is unknown.
    return "the previous pipeline scripts"


def check_required_files() -> bool:
    # Collect required files that do not exist yet.
    missing = [(label, path) for label, path in REQUIRED_FILES.items() if not path.exists()]

    # If nothing is missing, the app can safely continue building the interface.
    if not missing:
        return True

    # Show one clear error per missing file with the script that likely creates/checks it.
    for _, path in missing:
        st.error(
            f"Missing file: {relative_path(path)}. "
            f"Please run {script_hint_for_file(path)} first."
        )

    # Stop the app before model loading or interface construction can crash.
    st.stop()

    # This return is only for static type checkers; st.stop() halts Streamlit execution.
    return False


def read_feature_names(path: Path) -> list[str]:
    # Read one feature name per line and ignore empty lines.
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@st.cache_resource
def load_models() -> dict[str, object]:
    # Load all trained models and scalers once, then cache them across Streamlit reruns.
    return {
        "geometry_surrogate": joblib.load(REQUIRED_FILES["geometry surrogate model"]),
        "pressure_surrogate": joblib.load(REQUIRED_FILES["pressure surrogate model"]),
        "geometry_scaler": joblib.load(REQUIRED_FILES["geometry input scaler"]),
        "pressure_scaler": joblib.load(REQUIRED_FILES["pressure input scaler"]),
        "geometry_pca": joblib.load(REQUIRED_FILES["geometry PCA model"]),
        "pressure_pca": joblib.load(REQUIRED_FILES["pressure PCA model"]),
    }


@st.cache_data
def load_doe_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Read geometry DOE; sep=None lets pandas detect comma or semicolon separators.
    geometry_doe = pd.read_csv(REQUIRED_FILES["geometry DOE"], sep=None, engine="python")

    # Read pressure DOE; this file may use a different separator.
    pressure_doe = pd.read_csv(REQUIRED_FILES["pressure DOE"], sep=None, engine="python")

    # Return both DOE tables.
    return geometry_doe, pressure_doe


@st.cache_data
def load_feature_sets() -> tuple[list[str], list[str]]:
    # Read geometry feature names saved during surrogate training.
    geometry_features = read_feature_names(REQUIRED_FILES["geometry feature names"])

    # Read pressure feature names saved during surrogate training.
    pressure_features = read_feature_names(REQUIRED_FILES["pressure feature names"])

    # Return both feature-name lists.
    return geometry_features, pressure_features


def finite_slider_bounds(series: pd.Series) -> tuple[float, float, float]:
    # Convert the DOE column to numeric values and drop missing values.
    numeric = pd.to_numeric(series, errors="coerce").dropna()

    # If no numeric values exist, provide a safe default range.
    if numeric.empty:
        return 0.0, 1.0, 0.0

    # Compute minimum value from the DOE column.
    min_value = float(numeric.min())

    # Compute maximum value from the DOE column.
    max_value = float(numeric.max())

    # Use median as a stable default slider value.
    default_value = float(numeric.median())

    # If min and max are equal, widen the slider slightly so Streamlit accepts it.
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0

    # Return slider minimum, maximum, and default values.
    return min_value, max_value, default_value


def build_sidebar_inputs(
    geometry_doe: pd.DataFrame,
    pressure_doe: pd.DataFrame,
    geometry_features: list[str],
    pressure_features: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    # Create a sidebar section for all DOE controls.
    st.sidebar.header("DOE Input Parameters")

    # The union keeps shared geometry/pressure parameters controlled by one slider.
    all_features = list(dict.fromkeys(geometry_features + pressure_features))

    # Store every selected slider value by feature name.
    selected_values: dict[str, float] = {}

    # Loop through every feature needed by either model.
    for feature in all_features:
        # Prefer geometry DOE bounds when the feature exists there.
        if feature in geometry_doe.columns:
            min_value, max_value, default_value = finite_slider_bounds(geometry_doe[feature])

        # Otherwise use pressure DOE bounds.
        elif feature in pressure_doe.columns:
            min_value, max_value, default_value = finite_slider_bounds(pressure_doe[feature])

        # If the feature is missing from both DOE tables, create a safe fallback slider.
        else:
            min_value, max_value, default_value = 0.0, 1.0, 0.0

        # Use a compact slider label and store the selected value.
        selected_values[feature] = st.sidebar.slider(feature, min_value, max_value, default_value)

    # Build the geometry-specific input dictionary in trained feature order.
    geometry_inputs = {feature: selected_values[feature] for feature in geometry_features}

    # Build the pressure-specific input dictionary in trained feature order.
    pressure_inputs = {feature: selected_values[feature] for feature in pressure_features}

    # Return both input dictionaries.
    return geometry_inputs, pressure_inputs


def predict_field(
    input_values: dict[str, float],
    feature_names: list[str],
    scaler: object,
    surrogate: object,
    pca: object,
) -> tuple[np.ndarray, np.ndarray]:
    # Build a one-row input matrix in exactly the training feature order.
    x = np.array([[input_values[feature] for feature in feature_names]], dtype=np.float32)

    # Standardize the input parameters using the saved training scaler.
    x_scaled = scaler.transform(x)

    # Predict POD/PCA coefficients from the scaled DOE input.
    coefficients = surrogate.predict(x_scaled)

    # Reconstruct the full field from the predicted POD/PCA coefficients.
    reconstructed = pca.inverse_transform(coefficients)[0]

    # Return coefficients and reconstructed field as NumPy arrays.
    return np.asarray(coefficients), np.asarray(reconstructed, dtype=np.float32)


def geometry_to_points(geometry: np.ndarray) -> tuple[np.ndarray | None, str]:
    # Flatten geometry to test whether it contains x, y, z triples.
    flat = np.asarray(geometry).ravel()

    # Direct case: all values form x, y, z triples.
    if flat.size % 3 == 0:
        return flat.reshape(-1, 3), f"Geometry reshaped directly to ({flat.size // 3}, 3)."

    # Header-like case: skip one leading value, then reshape triples.
    if flat.size > 1 and (flat.size - 1) % 3 == 0:
        return flat[1:].reshape(-1, 3), f"Geometry reshaped after skipping one leading value to ({(flat.size - 1) // 3}, 3)."

    # Failure case: return diagnostics instead of crashing.
    return None, f"Geometry has {flat.size} values and cannot be reshaped to (-1, 3)."


def load_points_bin(path: Path, label: str) -> tuple[np.ndarray | None, dict[str, object]]:
    # Start a diagnostic dictionary for this points.bin file.
    diagnostics: dict[str, object] = {
        f"{label}_points_file": relative_path(path),
        f"{label}_points_file_exists": path.exists(),
        f"{label}_points_dtype_used": None,
        f"{label}_points_array_shape": None,
        f"{label}_points_number_of_values": None,
        f"{label}_points_divisible_by_3": False,
        f"{label}_points_number_of_points": None,
        f"{label}_points_reshape_status": "file missing",
    }

    # If points.bin is missing, return diagnostics without crashing.
    if not path.exists():
        return None, diagnostics

    # Try float32 first because that is a common simple binary interpretation.
    for dtype in (np.float32, np.float64):
        # Read the binary file using the current dtype.
        values = np.fromfile(path, dtype=dtype)

        # Store general shape and value-count diagnostics.
        diagnostics[f"{label}_points_dtype_used"] = str(np.dtype(dtype))
        diagnostics[f"{label}_points_array_shape"] = str(values.shape)
        diagnostics[f"{label}_points_number_of_values"] = int(values.size)
        diagnostics[f"{label}_points_divisible_by_3"] = bool(values.size % 3 == 0)

        # If the whole array is divisible by 3, reshape it directly to x, y, z points.
        if values.size % 3 == 0:
            points = values.reshape(-1, 3).astype(np.float32, copy=False)
            diagnostics[f"{label}_points_number_of_points"] = int(points.shape[0])
            diagnostics[f"{label}_points_reshape_status"] = "reshaped directly"
            return points, diagnostics

        # If one leading value makes the rest divisible by 3, skip that header-like value.
        if values.size > 1 and (values.size - 1) % 3 == 0:
            points = values[1:].reshape(-1, 3).astype(np.float32, copy=False)
            diagnostics[f"{label}_points_number_of_points"] = int(points.shape[0])
            diagnostics[f"{label}_points_reshape_status"] = "reshaped after skipping one leading value"
            return points, diagnostics

        # Record that this dtype failed before trying the fallback dtype.
        diagnostics[f"{label}_points_reshape_status"] = f"not reshapeable as {np.dtype(dtype)}"

    # Return no points if both dtype interpretations failed.
    return None, diagnostics


def inspect_binary_shapes(
    geometry_prediction: np.ndarray,
    pressure_prediction: np.ndarray,
) -> dict[str, object]:
    # Flatten geometry and pressure so value counts are easy to compare.
    geometry_values = np.asarray(geometry_prediction).ravel()
    pressure_values = np.asarray(pressure_prediction).ravel()

    # Try to interpret predicted geometry as coordinates.
    geometry_points, geometry_status = geometry_to_points(geometry_prediction)

    # Load pressure points.bin separately; do not assume it is the same mesh as geometry.
    pressure_points, pressure_points_diagnostics = load_points_bin(DATA_ROOT / "pressure" / "points.bin", "pressure")

    # Load geometry points.bin separately for diagnostics and possible fallback display.
    raw_geometry_points, geometry_points_diagnostics = load_points_bin(DATA_ROOT / "geometry" / "points.bin", "geometry")

    # Count points from predicted geometry if it could be reshaped.
    geometry_point_count = None if geometry_points is None else int(geometry_points.shape[0])

    # Count pressure points from pressure points.bin if it could be reshaped.
    pressure_point_count = None if pressure_points is None else int(pressure_points.shape[0])

    # Check whether predicted pressure has one scalar value for each pressure mesh point.
    pressure_matches_pressure_points = pressure_point_count is not None and pressure_values.size == pressure_point_count

    # Check whether predicted pressure has one scalar value for each predicted geometry point.
    pressure_matches_geometry_points = geometry_point_count is not None and pressure_values.size == geometry_point_count

    # Decide which coordinates are safest for visualization.
    if geometry_points is not None and pressure_matches_geometry_points:
        display_points = geometry_points
        display_mode = "3D/2D projection using predicted geometry with pressure coloring"
    elif pressure_points is not None and pressure_matches_pressure_points:
        display_points = pressure_points
        display_mode = "3D/2D projection using pressure points.bin with pressure coloring"
    elif geometry_points is not None:
        display_points = geometry_points
        display_mode = "2D projection using predicted geometry without pressure coloring"
    elif raw_geometry_points is not None:
        display_points = raw_geometry_points
        display_mode = "2D projection using geometry points.bin without pressure coloring"
    else:
        display_points = None
        display_mode = "numerical results only"

    # Build a diagnostic dictionary that can be shown in the app.
    diagnostics: dict[str, object] = {
        "geometry_array_shape": str(tuple(np.asarray(geometry_prediction).shape)),
        "pressure_array_shape": str(tuple(np.asarray(pressure_prediction).shape)),
        "number_of_geometry_values": int(geometry_values.size),
        "number_of_pressure_values": int(pressure_values.size),
        "geometry_values_divisible_by_3": bool(geometry_values.size % 3 == 0),
        "pressure_values_divisible_by_3": bool(pressure_values.size % 3 == 0),
        "predicted_geometry_reshape_status": geometry_status,
        "predicted_geometry_number_of_points": geometry_point_count,
        "pressure_prediction_has_one_scalar_per_pressure_point": bool(pressure_matches_pressure_points),
        "pressure_prediction_has_one_scalar_per_geometry_point": bool(pressure_matches_geometry_points),
        "display_mode": display_mode,
        "display_point_shape": None if display_points is None else str(tuple(display_points.shape)),
        "display_points": display_points,
        "geometry_points": geometry_points,
        "pressure_points": pressure_points,
    }

    # Add pressure points.bin diagnostics.
    diagnostics.update(pressure_points_diagnostics)

    # Add geometry points.bin diagnostics.
    diagnostics.update(geometry_points_diagnostics)

    # Return diagnostics and internal arrays.
    return diagnostics


def pressure_to_scalar(pressure: np.ndarray) -> np.ndarray:
    # Flatten pressure to one scalar value per node when dimensions match the geometry.
    return np.asarray(pressure, dtype=np.float32).ravel()


def compute_proxy_metrics(points: np.ndarray | None, pressure: np.ndarray) -> dict[str, float]:
    # Start with pressure metrics because pressure exists even if geometry reshape fails.
    pressure_values = pressure_to_scalar(pressure)

    # Store pressure min, max, mean, and range.
    metrics: dict[str, float] = {
        "pressure_min": float(np.min(pressure_values)),
        "pressure_max": float(np.max(pressure_values)),
        "pressure_mean": float(np.mean(pressure_values)),
        "pressure_range": float(np.max(pressure_values) - np.min(pressure_values)),
    }

    # If points are missing, fill geometry-derived metrics with NaN.
    if points is None:
        metrics["mean_radius_proxy"] = np.nan
        metrics["constriction_index"] = np.nan
        metrics["flow_capacity_proxy"] = np.nan
        metrics["resistance_proxy"] = np.nan
        metrics["pressure_drop_proxy"] = metrics["pressure_range"]
        return metrics

    # Compute centroid of the predicted geometry.
    centroid = np.mean(points, axis=0)

    # Compute radial distances in the xy plane.
    radial = np.sqrt((points[:, 0] - centroid[0]) ** 2 + (points[:, 1] - centroid[1]) ** 2)

    # Compute simplified geometry proxy metrics.
    mean_radius = float(np.mean(radial))
    min_radius = float(np.min(radial))
    constriction = float(min_radius / mean_radius) if mean_radius > 0 else np.nan
    flow_capacity = float(mean_radius**4) if np.isfinite(mean_radius) else np.nan
    resistance = float(1.0 / max(flow_capacity, SMALL_EPSILON)) if np.isfinite(flow_capacity) else np.nan

    # Store the requested proxy metrics.
    metrics["mean_radius_proxy"] = mean_radius
    metrics["constriction_index"] = constriction
    metrics["flow_capacity_proxy"] = flow_capacity
    metrics["resistance_proxy"] = resistance
    metrics["pressure_drop_proxy"] = metrics["pressure_range"]

    # Return all proxy metrics.
    return metrics


def plot_projection(points: np.ndarray, pressure: np.ndarray) -> plt.Figure:
    # Flatten pressure values for coloring.
    pressure_values = pressure_to_scalar(pressure)

    # Downsample the display for speed when many nodes exist.
    stride = max(1, points.shape[0] // 40000)

    # Select sampled points.
    sample_points = points[::stride]

    # Select matching pressure values when dimensions match.
    sample_pressure = pressure_values[::stride] if pressure_values.shape[0] == points.shape[0] else None

    # Create a clean figure for the x-z projection.
    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw a colored scatter if pressure matches the point count.
    if sample_pressure is not None:
        scatter = ax.scatter(sample_points[:, 0], sample_points[:, 2], c=sample_pressure, s=2, cmap="viridis")
        fig.colorbar(scatter, ax=ax, label="Pressure")

    # Otherwise draw geometry points without pressure coloring.
    else:
        ax.scatter(sample_points[:, 0], sample_points[:, 2], s=2, color="#2a9d8f")

    # Label the projection axes.
    ax.set_xlabel("X")
    ax.set_ylabel("Z")

    # Add a descriptive title.
    ax.set_title("Predicted Airway Projection")

    # Keep equal aspect ratio so the geometry is not visually distorted.
    ax.set_aspect("equal", adjustable="box")

    # Add a subtle grid for presentation readability.
    ax.grid(True, alpha=0.2)

    # Tighten spacing around labels.
    fig.tight_layout()

    # Return the figure for Streamlit display.
    return fig


def plot_pressure_histogram(pressure: np.ndarray) -> plt.Figure:
    # Flatten pressure into scalar values.
    pressure_values = pressure_to_scalar(pressure)

    # Create the histogram figure.
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot pressure distribution.
    ax.hist(pressure_values, bins=60, color="#457b9d", edgecolor="white")

    # Label axes.
    ax.set_xlabel("Pressure")
    ax.set_ylabel("Point count")

    # Add a title.
    ax.set_title("Predicted Pressure Distribution")

    # Add a subtle grid.
    ax.grid(True, alpha=0.2)

    # Tighten spacing around labels.
    fig.tight_layout()

    # Return the figure.
    return fig


def save_predictions(geometry: np.ndarray, pressure: np.ndarray) -> tuple[Path, Path]:
    # Make sure the predictions folder exists.
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Build output paths for the dashboard prediction.
    geometry_path = PREDICTIONS_DIR / "dashboard_predicted_geometry.npy"
    pressure_path = PREDICTIONS_DIR / "dashboard_predicted_pressure.npy"

    # Save geometry prediction.
    np.save(geometry_path, geometry)

    # Save pressure prediction.
    np.save(pressure_path, pressure)

    # Return both output paths.
    return geometry_path, pressure_path


def export_vtp(points: np.ndarray | None, pressure: np.ndarray) -> Path | None:
    # Import PyVista only when export is requested so the app can run without it.
    try:
        import pyvista as pv
    except Exception as error:
        st.error(f"PyVista is not available, so VTP export cannot run: {error}")
        return None

    # If geometry points are unavailable, export cannot create a point cloud.
    if points is None:
        st.error("Cannot export VTP because geometry could not be reshaped to x, y, z points.")
        return None

    # Create a PyVista point cloud.
    mesh = pv.PolyData(points)

    # Flatten pressure values.
    pressure_values = pressure_to_scalar(pressure)

    # Attach pressure only when dimensions match.
    if pressure_values.shape[0] == points.shape[0]:
        mesh.point_data["pressure"] = pressure_values
    else:
        st.warning(
            f"Pressure was not attached to VTP because point count is {points.shape[0]} "
            f"but pressure length is {pressure_values.shape[0]}."
        )

    # Build the VTP output path.
    output_path = PREDICTIONS_DIR / "dashboard_predicted_airway.vtp"

    # Make sure the predictions folder exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the VTP file for ParaView.
    mesh.save(output_path)

    # Return the saved file path.
    return output_path


def main() -> None:
    # Configure the Streamlit page for a clean presentation layout.
    st.set_page_config(page_title="Airways Digital Twin", layout="wide")

    # Check all required files before trying to load models or build the interface.
    check_required_files()

    # Add a professional title and compact subtitle.
    st.title("Human Airways Digital Twin Dashboard")
    st.caption("DOE parameters -> surrogate models -> POD/PCA coefficients -> reconstructed airway fields")

    # Load trained models, scalers, and PCA objects.
    models = load_models()

    # Load DOE tables used for slider ranges.
    geometry_doe, pressure_doe = load_doe_tables()

    # Load feature names in the same order used during training.
    geometry_features, pressure_features = load_feature_sets()

    # Build sidebar sliders and get input dictionaries.
    geometry_inputs, pressure_inputs = build_sidebar_inputs(
        geometry_doe,
        pressure_doe,
        geometry_features,
        pressure_features,
    )

    # Add action buttons to the sidebar.
    st.sidebar.header("Actions")
    predict_clicked = st.sidebar.button("Predict", type="primary")
    save_clicked = st.sidebar.button("Save prediction as .npy")
    export_clicked = st.sidebar.button("Export prediction to .vtp")

    # Keep latest predictions in session state so save/export buttons can use them.
    if predict_clicked or "geometry_prediction" not in st.session_state:
        # Predict geometry coefficients and reconstruct geometry.
        geometry_coefficients, geometry_prediction = predict_field(
            geometry_inputs,
            geometry_features,
            models["geometry_scaler"],
            models["geometry_surrogate"],
            models["geometry_pca"],
        )

        # Predict pressure coefficients and reconstruct pressure.
        pressure_coefficients, pressure_prediction = predict_field(
            pressure_inputs,
            pressure_features,
            models["pressure_scaler"],
            models["pressure_surrogate"],
            models["pressure_pca"],
        )

        # Save predictions in session state for reuse.
        st.session_state.geometry_coefficients = geometry_coefficients
        st.session_state.pressure_coefficients = pressure_coefficients
        st.session_state.geometry_prediction = geometry_prediction
        st.session_state.pressure_prediction = pressure_prediction

    # Retrieve predictions from session state.
    geometry_prediction = st.session_state.geometry_prediction
    pressure_prediction = st.session_state.pressure_prediction
    geometry_coefficients = st.session_state.geometry_coefficients
    pressure_coefficients = st.session_state.pressure_coefficients

    # Inspect geometry, pressure, and points.bin shapes without assuming shared meshes.
    shape_diagnostics = inspect_binary_shapes(geometry_prediction, pressure_prediction)

    # Use predicted geometry points for geometry-based proxy metrics when available.
    geometry_points = shape_diagnostics["geometry_points"]

    # Use the safest available point set for display.
    display_points = shape_diagnostics["display_points"]

    # Compute proxy metrics for presentation.
    metrics = compute_proxy_metrics(geometry_points, pressure_prediction)

    # Handle save button after predictions exist.
    if save_clicked:
        geometry_path, pressure_path = save_predictions(geometry_prediction, pressure_prediction)
        st.sidebar.success(f"Saved: {geometry_path.name}, {pressure_path.name}")

    # Handle VTP export button after predictions exist.
    if export_clicked:
        exported_path = export_vtp(display_points, pressure_prediction)
        if exported_path:
            st.sidebar.success(f"Exported: {exported_path.name}")

    # Show top-level prediction metrics in compact cards.
    st.subheader("Prediction Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Geometry shape", str(tuple(geometry_prediction.shape)))
    col2.metric("Pressure shape", str(tuple(pressure_prediction.shape)))
    col3.metric("Pressure mean", f"{metrics['pressure_mean']:.4g}")
    col4.metric("Pressure drop proxy", f"{metrics['pressure_drop_proxy']:.4g}")

    # Show selected input parameters in two tables.
    st.subheader("Selected Input Parameters")
    left_inputs, right_inputs = st.columns(2)
    with left_inputs:
        st.write("Geometry model inputs")
        st.dataframe(pd.DataFrame([geometry_inputs]), use_container_width=True)
    with right_inputs:
        st.write("Pressure model inputs")
        st.dataframe(pd.DataFrame([pressure_inputs]), use_container_width=True)

    # Show coefficient shapes and binary diagnostics.
    st.subheader("Model And Binary Diagnostics")
    diag_col1, diag_col2, diag_col3 = st.columns(3)
    diag_col1.metric("Geometry coefficients", str(tuple(geometry_coefficients.shape)))
    diag_col2.metric("Pressure coefficients", str(tuple(pressure_coefficients.shape)))
    diag_col3.metric("Display mode", str(shape_diagnostics["display_mode"]))

    # Hide internal NumPy arrays before showing diagnostics as a table.
    visible_diagnostics = {
        key: value
        for key, value in shape_diagnostics.items()
        if key not in {"display_points", "geometry_points", "pressure_points"}
    }
    st.dataframe(pd.DataFrame([visible_diagnostics]), use_container_width=True)

    # Show proxy metrics table.
    st.subheader("Proxy Physics Metrics")
    proxy_keys = [
        "mean_radius_proxy",
        "constriction_index",
        "flow_capacity_proxy",
        "resistance_proxy",
        "pressure_drop_proxy",
    ]
    st.dataframe(pd.DataFrame([{key: metrics[key] for key in proxy_keys}]), use_container_width=True)
    st.caption("These are simplified proxy metrics for interaction and presentation, not full CFD results.")

    # Show visualizations in two columns.
    st.subheader("Visualization")
    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        if display_points is not None:
            st.pyplot(plot_projection(display_points, pressure_prediction), clear_figure=True)
        else:
            st.warning("No usable point coordinates were found. Showing numerical results only.")
    with plot_col2:
        st.pyplot(plot_pressure_histogram(pressure_prediction), clear_figure=True)

    # Show pressure statistics below plots.
    st.subheader("Pressure Statistics")
    pressure_stats = {
        "pressure_min": metrics["pressure_min"],
        "pressure_max": metrics["pressure_max"],
        "pressure_mean": metrics["pressure_mean"],
        "pressure_range": metrics["pressure_range"],
    }
    st.dataframe(pd.DataFrame([pressure_stats]), use_container_width=True)


# This makes main() run only when Streamlit executes the file.
if __name__ == "__main__":
    main()
