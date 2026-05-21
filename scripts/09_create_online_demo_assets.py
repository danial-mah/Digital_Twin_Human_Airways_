"""Create small demo assets for online Streamlit deployment.

Why this script exists:
The full project uses very large files: raw snapshots, snapshot matrices, and
large PCA models. Those files are not practical for GitHub or Streamlit Cloud.

This script creates a lightweight demo asset instead:

    demo_assets/airway_demo_sample.npz
    demo_assets/airway_demo_metadata.json

The demo asset contains only a downsampled point cloud and pressure values.
That is enough to show the airway in 3D online without uploading gigabytes.

Run from the project root:

    python scripts/09_create_online_demo_assets.py
"""

# This line allows Python type hints to be handled in a modern, flexible way.
from __future__ import annotations

# json saves small metadata in a human-readable file.
import json

# sys lets this script import shared helpers from src/airways/.
import sys

# Path gives clean filesystem path handling.
from pathlib import Path

# NumPy reads binary data and saves compact .npz demo files.
import numpy as np


# Add src/ to Python's import path so this script can use shared project paths.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import shared project paths.
from airways.paths import DATA_ROOT, PROJECT_ROOT


# DEMO_DIR is intentionally outside outputs/ because outputs/ is ignored by Git.
DEMO_DIR = PROJECT_ROOT / "demo_assets"

# DEMO_NPZ_PATH is the small compressed file the online demo app will load.
DEMO_NPZ_PATH = DEMO_DIR / "airway_demo_sample.npz"

# DEMO_METADATA_PATH stores simple information about how the demo asset was made.
DEMO_METADATA_PATH = DEMO_DIR / "airway_demo_metadata.json"

# MAX_POINTS limits the number of points in the online demo.
# A smaller value makes the browser faster and the GitHub file smaller.
MAX_POINTS = 30000


def read_float64_with_header(path: Path) -> np.ndarray:
    """Read a dataset .bin file stored as float64 with one leading header value."""

    # Read the raw binary file using the real storage format.
    raw_values = np.fromfile(path, dtype=np.float64)

    # Remove the first header value and return the real payload.
    return raw_values[1:]


def main() -> None:
    """Create the lightweight online demo asset."""

    # Create demo_assets/ if it does not exist.
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    # Use pressure points because pressure and geometry share the same point layout here.
    points_path = DATA_ROOT / "pressure" / "points.bin"

    # Use the first pressure snapshot as a lightweight pressure field for the demo.
    pressure_snapshot_path = DATA_ROOT / "pressure" / "Snapshots" / "snapshot1.bin"

    # Stop early with a clear error if required raw files are missing.
    if not points_path.exists():
        raise FileNotFoundError(f"Missing points file: {points_path}")
    if not pressure_snapshot_path.exists():
        raise FileNotFoundError(f"Missing pressure snapshot: {pressure_snapshot_path}")

    # Read points.bin correctly as float64 with one header value.
    point_values = read_float64_with_header(points_path)

    # Reshape flat point values into rows of x, y, z coordinates.
    points = point_values.reshape(-1, 3)

    # Read pressure snapshot correctly as float64 with one header value.
    pressure = read_float64_with_header(pressure_snapshot_path)

    # Check that pressure has one scalar value per point.
    if pressure.shape[0] != points.shape[0]:
        raise ValueError(
            f"Pressure length {pressure.shape[0]} does not match point count {points.shape[0]}."
        )

    # Downsample evenly so the demo remains fast in a web browser.
    stride = max(1, points.shape[0] // MAX_POINTS)
    demo_points = points[::stride].astype(np.float32)
    demo_pressure = pressure[::stride].astype(np.float32)

    # Save the small demo arrays in compressed NumPy format.
    np.savez_compressed(
        DEMO_NPZ_PATH,
        points=demo_points,
        pressure=demo_pressure,
    )

    # Build simple metadata for the demo.
    metadata = {
        "source_points": str(points_path.relative_to(PROJECT_ROOT)),
        "source_pressure_snapshot": str(pressure_snapshot_path.relative_to(PROJECT_ROOT)),
        "original_points": int(points.shape[0]),
        "demo_points": int(demo_points.shape[0]),
        "stride": int(stride),
        "asset_file": str(DEMO_NPZ_PATH.relative_to(PROJECT_ROOT)),
        "important_note": "This is a lightweight visualization demo, not the full training dataset.",
    }

    # Save metadata as readable JSON.
    DEMO_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Print a clear summary for the user.
    print(f"Saved demo asset: {DEMO_NPZ_PATH}")
    print(f"Saved metadata: {DEMO_METADATA_PATH}")
    print(f"Demo points shape: {demo_points.shape}")
    print(f"Demo pressure shape: {demo_pressure.shape}")
    print(f"Compressed asset size: {DEMO_NPZ_PATH.stat().st_size / (1024 * 1024):.2f} MB")


# This makes main() run only when this file is executed directly.
if __name__ == "__main__":
    main()
