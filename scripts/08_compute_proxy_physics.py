"""Compute simple proxy physics metrics for a predicted airway.

This is an execution script. You run it from the project root with:

    python scripts/08_compute_proxy_physics.py

It loads the predicted geometry and pressure produced by script 06, computes
simple dashboard-friendly metrics, prints them, and saves them to a CSV file.

Important:
These metrics are proxy indicators, not real CFD simulation results.
"""

# This line allows Python type hints to be handled in a modern, flexible way.
from __future__ import annotations

# sys lets us modify Python's import search path.
# We use it so this numbered script can import helper code from src/airways/.
import sys

# Path is used to work with folder and file locations.
from pathlib import Path

# pandas is used to save the metrics dictionary as a one-row CSV table.
import pandas as pd


# This script lives in scripts/.
# Path(__file__).resolve() gives the full path to this script.
# parents[1] moves from scripts/ back to the project root.
# Adding / "src" points Python to the shared helper package folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# reshape_values_to_points tries to convert a flat geometry array into x, y, z points.
# Example:
#   [x1, y1, z1, x2, y2, z2, ...] -> [[x1, y1, z1], [x2, y2, z2], ...]
from airways.binary import reshape_values_to_points

# load_prediction loads a .npy prediction file and prints helpful diagnostics.
from airways.io import load_prediction

# PREDICTIONS_DIR is the shared path to outputs/predictions/.
from airways.paths import PREDICTIONS_DIR

# compute_proxy_metrics calculates geometry, pressure, and simplified flow indicators.
from airways.proxy_physics import compute_proxy_metrics


# This file should already exist after running script 06.
# It contains the reconstructed predicted geometry for DOE row 0.
GEOMETRY_PATH = PREDICTIONS_DIR / "predicted_geometry_row0.npy"

# This file should already exist after running script 06.
# It contains the reconstructed predicted pressure for DOE row 0.
PRESSURE_PATH = PREDICTIONS_DIR / "predicted_pressure_row0.npy"

# This is the CSV file that this script will create.
# It stores all proxy metrics in one row so they can be opened in Excel, pandas, or reports.
OUTPUT_PATH = PREDICTIONS_DIR / "proxy_physics_row0.csv"


def print_metrics(metrics: dict[str, float]) -> None:
    """Print metric names and values in a readable terminal format."""

    # Print a blank line and title to make the output easy to find in the terminal.
    print("\nProxy physics metrics:")

    # metrics.items() gives each key-value pair in the dictionary.
    # key is the metric name, for example "pressure_mean".
    # value is the metric value, for example 23.5.
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def main() -> None:
    """Run the proxy physics calculation workflow."""

    # Load predicted geometry from outputs/predictions/predicted_geometry_row0.npy.
    # If the file is missing, load_prediction returns None instead of crashing.
    geometry = load_prediction(GEOMETRY_PATH, "geometry")

    # Load predicted pressure from outputs/predictions/predicted_pressure_row0.npy.
    # If the file is missing, load_prediction returns None instead of crashing.
    pressure = load_prediction(PRESSURE_PATH, "pressure")

    # Try to reshape the geometry into point coordinates.
    # points will be a NumPy array shaped like (number_of_points, 3) if successful.
    # geometry_status is a text explanation of what happened.
    points, geometry_status = reshape_values_to_points(geometry)

    # Print the geometry reshape status so the user understands whether geometry
    # metrics such as centroid and radius proxy can be calculated.
    print(f"Geometry reshape status: {geometry_status}")

    # Compute all available proxy metrics.
    # If geometry is missing, geometry metrics are skipped.
    # If pressure is missing, pressure metrics are skipped.
    metrics = compute_proxy_metrics(points, pressure)

    # Print the metric dictionary in a readable way.
    print_metrics(metrics)

    # Make sure outputs/predictions/ exists before writing the CSV file.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Convert the metrics dictionary into a one-row pandas DataFrame.
    # Example columns:
    #   pressure_min, pressure_max, mean_radius_proxy, resistance_proxy
    metrics_table = pd.DataFrame([metrics])

    # Save the one-row table to CSV.
    # index=False avoids writing an unnecessary row-number column.
    metrics_table.to_csv(OUTPUT_PATH, index=False)

    # Print the saved path so the user knows where the report went.
    print(f"\nSaved proxy physics metrics to: {OUTPUT_PATH}")


# This condition means:
#   Run main() only when this file is executed directly.
#   Do not automatically run main() if another file imports this script.
if __name__ == "__main__":
    main()
