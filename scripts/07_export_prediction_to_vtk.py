"""Export predicted airway fields to VTP files for ParaView.

This script creates PyVista PolyData point-cloud files. The saved .vtp files can
be opened in ParaView to visualize predicted airway geometry and pressure.
"""

# This import enables modern type-hint behavior.
from __future__ import annotations

# Path gives clean filesystem path handling.
from pathlib import Path

# NumPy loads binary arrays and prediction arrays.
import numpy as np

# PyVista creates and writes VTK/VTP datasets.
import pyvista as pv


# PROJECT_ROOT points to the main project folder, one level above scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DATA_ROOT points to the raw data folder.
DATA_ROOT = PROJECT_ROOT / "data"

# PREDICTIONS_DIR points to generated prediction files.
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"

# Pressure points are used as the default point-cloud coordinates.
PRESSURE_POINTS_PATH = DATA_ROOT / "pressure" / "points.bin"

# Geometry points are a fallback coordinate source if pressure points are unavailable.
GEOMETRY_POINTS_PATH = DATA_ROOT / "geometry" / "points.bin"

# Predicted pressure values are attached to the point cloud as point data.
PREDICTED_PRESSURE_PATH = PREDICTIONS_DIR / "predicted_pressure_row0.npy"

# Predicted geometry may contain x, y, z coordinates for the point cloud.
PREDICTED_GEOMETRY_PATH = PREDICTIONS_DIR / "predicted_geometry_row0.npy"

# Main output file containing coordinates and predicted pressure.
PRESSURE_OUTPUT_PATH = PREDICTIONS_DIR / "predicted_airway_pressure_row0.vtp"

# Optional second output file emphasizing predicted geometry.
GEOMETRY_OUTPUT_PATH = PREDICTIONS_DIR / "predicted_airway_geometry_row0.vtp"


def load_binary_float32(path: Path) -> np.ndarray | None:
    # If the binary file is missing, print a helpful warning and return None.
    if not path.exists():
        print(f"WARNING: Missing binary file: {path}")
        return None

    # Load the binary values as float32, exactly as requested for this script.
    values = np.fromfile(path, dtype=np.float32)

    # Print how many values were read for diagnostics.
    print(f"Loaded {path.name} as float32 with {values.size} values.")

    # Return the raw one-dimensional array.
    return values


def load_binary_float64(path: Path) -> np.ndarray | None:
    # If the binary file is missing, print a helpful warning and return None.
    if not path.exists():
        print(f"WARNING: Missing binary file: {path}")
        return None

    # Load the binary values as float64 as a fallback when float32 cannot be reshaped.
    values = np.fromfile(path, dtype=np.float64)

    # Print how many fallback values were read for diagnostics.
    print(f"Fallback loaded {path.name} as float64 with {values.size} values.")

    # Return the raw one-dimensional array.
    return values


def reshape_points(values: np.ndarray | None, label: str) -> np.ndarray | None:
    # If no values were provided, point reshaping is impossible.
    if values is None:
        return None

    # Try the direct interpretation: all values are x, y, z coordinates.
    if values.size % 3 == 0:
        points = values.reshape(-1, 3)
        print(f"{label}: reshaped directly to points with shape {points.shape}.")
        return points

    # Some exported binary files include one leading header value; try skipping it.
    if values.size > 1 and (values.size - 1) % 3 == 0:
        points = values[1:].reshape(-1, 3)
        print(f"{label}: skipped one leading value and reshaped to points with shape {points.shape}.")
        return points

    # If neither interpretation works, print diagnostics instead of crashing.
    print(f"WARNING: {label} has {values.size} values and cannot be reshaped to (-1, 3).")
    return None


def load_prediction(path: Path, label: str) -> np.ndarray | None:
    # If the prediction file is missing, print a warning and return None.
    if not path.exists():
        print(f"WARNING: Missing prediction file for {label}: {path}")
        return None

    # Load the NumPy prediction array.
    values = np.load(path)

    # Convert to float32 for compact VTK output.
    values = np.asarray(values, dtype=np.float32)

    # Print the loaded prediction shape.
    print(f"Loaded {label} prediction with shape {values.shape}.")

    # Return the prediction array.
    return values


def prediction_as_points(predicted_geometry: np.ndarray | None) -> np.ndarray | None:
    # If predicted geometry is unavailable, it cannot be used as points.
    if predicted_geometry is None:
        return None

    # Flatten the prediction so we can inspect whether it contains triples.
    flat = predicted_geometry.ravel()

    # If all values form x, y, z triples, reshape directly.
    if flat.size % 3 == 0:
        points = flat.reshape(-1, 3)
        print(f"Predicted geometry can be used as points with shape {points.shape}.")
        return points

    # If one leading value prevents divisibility by three, try skipping it.
    if flat.size > 1 and (flat.size - 1) % 3 == 0:
        points = flat[1:].reshape(-1, 3)
        print(f"Predicted geometry used as points after skipping one leading value: {points.shape}.")
        return points

    # Otherwise, report that predicted geometry cannot define coordinates directly.
    print(
        "WARNING: Predicted geometry cannot be used directly as coordinates; "
        f"flattened size is {flat.size}."
    )
    return None


def pressure_as_scalar(predicted_pressure: np.ndarray | None) -> np.ndarray | None:
    # If pressure is unavailable, there is no scalar field to attach.
    if predicted_pressure is None:
        return None

    # Flatten pressure so point_data receives one value per point.
    pressure = predicted_pressure.ravel().astype(np.float32, copy=False)

    # If there appears to be one leading header-like value, keep diagnostics but do not remove blindly.
    print(f"Pressure flattened shape: {pressure.shape}.")

    # Return the flattened pressure values.
    return pressure


def create_polydata(points: np.ndarray, pressure: np.ndarray | None = None) -> tuple[pv.PolyData, list[str]]:
    # Create a PyVista point-cloud mesh from x, y, z coordinates.
    mesh = pv.PolyData(points)

    # Track which point fields are attached.
    attached_fields: list[str] = []

    # Attach pressure only when one scalar value exists per point.
    if pressure is not None and pressure.shape[0] == points.shape[0]:
        mesh.point_data["pressure"] = pressure
        attached_fields.append("pressure")

    # If pressure length does not match point count, print a helpful diagnostic.
    elif pressure is not None:
        print(
            "WARNING: Pressure was not attached because dimensions do not match: "
            f"points={points.shape[0]}, pressure_values={pressure.shape[0]}"
        )

    # Return the mesh and list of attached fields.
    return mesh, attached_fields


def save_mesh(mesh: pv.PolyData, path: Path) -> None:
    # Make sure the output folder exists before saving.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save the point cloud as a VTP file that ParaView can open.
    mesh.save(path)

    # Print the saved path.
    print(f"Saved VTP file: {path}")


def first_available_points(*candidates: np.ndarray | None) -> np.ndarray | None:
    # Loop over candidate point arrays in priority order.
    for candidate in candidates:
        # Return the first candidate that actually exists.
        if candidate is not None:
            return candidate

    # Return None if every candidate was unavailable.
    return None


def main() -> None:
    # Load pressure points as the preferred base coordinates.
    pressure_point_values = load_binary_float32(PRESSURE_POINTS_PATH)

    # Try to reshape pressure points into an N x 3 coordinate array.
    pressure_points = reshape_points(pressure_point_values, "pressure points.bin")

    # If float32 did not produce coordinates, try float64 because inspection suggested that format.
    if pressure_points is None:
        pressure_points = reshape_points(load_binary_float64(PRESSURE_POINTS_PATH), "pressure points.bin float64 fallback")

    # Load geometry points as a fallback coordinate source.
    geometry_point_values = load_binary_float32(GEOMETRY_POINTS_PATH)

    # Try to reshape geometry points into an N x 3 coordinate array.
    geometry_points = reshape_points(geometry_point_values, "geometry points.bin")

    # If float32 did not produce coordinates, try float64 because inspection suggested that format.
    if geometry_points is None:
        geometry_points = reshape_points(load_binary_float64(GEOMETRY_POINTS_PATH), "geometry points.bin float64 fallback")

    # Load predicted pressure values.
    predicted_pressure = load_prediction(PREDICTED_PRESSURE_PATH, "pressure")

    # Convert predicted pressure into one flat scalar array.
    pressure = pressure_as_scalar(predicted_pressure)

    # Load predicted geometry values if available.
    predicted_geometry = load_prediction(PREDICTED_GEOMETRY_PATH, "geometry")

    # Try to use predicted geometry as coordinates.
    predicted_geometry_points = prediction_as_points(predicted_geometry)

    # Choose coordinates for the pressure output: predicted geometry first, then pressure points, then geometry points.
    pressure_output_points = first_available_points(predicted_geometry_points, pressure_points, geometry_points)

    # If no coordinate source is usable, stop gracefully.
    if pressure_output_points is None:
        print("ERROR: Could not create point cloud because no usable coordinate array was found.")
        return

    # Create the pressure point cloud and attach pressure if dimensions match.
    pressure_mesh, pressure_fields = create_polydata(pressure_output_points, pressure)

    # Save the pressure visualization file.
    save_mesh(pressure_mesh, PRESSURE_OUTPUT_PATH)

    # Print diagnostics for the main output.
    print(f"Pressure output point cloud shape: {pressure_output_points.shape}")
    print(f"Pressure shape: {None if pressure is None else pressure.shape}")
    print(f"Attached fields in pressure output: {pressure_fields}")
    print(f"Pressure output file path: {PRESSURE_OUTPUT_PATH}")

    # Save a second geometry file if predicted geometry can be used as points.
    if predicted_geometry_points is not None:
        # Create geometry-only point cloud, also attaching pressure when dimensions match.
        geometry_mesh, geometry_fields = create_polydata(predicted_geometry_points, pressure)

        # Save the geometry-focused VTP file.
        save_mesh(geometry_mesh, GEOMETRY_OUTPUT_PATH)

        # Print diagnostics for the second output.
        print(f"Geometry output point cloud shape: {predicted_geometry_points.shape}")
        print(f"Attached fields in geometry output: {geometry_fields}")
        print(f"Geometry output file path: {GEOMETRY_OUTPUT_PATH}")

    # If predicted geometry could not be used, explain why the second file was not saved.
    else:
        print("Geometry VTP file was not saved because predicted geometry was not usable as x,y,z coordinates.")


# This makes main() run only when this file is executed directly.
if __name__ == "__main__":
    main()
