"""Binary-data helpers for airway snapshots and point clouds.

The airway .bin files in this project are stored as float64 and contain one
leading header value. For machine learning we read them correctly, remove the
header, then optionally convert the cleaned values to float32.

Beginner example:
The wrong way for this dataset is:

    np.fromfile(path, dtype=np.float32)

The correct way is:

    raw = np.fromfile(path, dtype=np.float64)
    clean = raw[1:]
    clean = clean.astype(np.float32)

Why:
The file was written as float64. If we read it as float32, the numbers become
corrupted because Python splits each 8-byte value into two 4-byte pieces.
"""

# This import enables modern type-hint behavior.
from __future__ import annotations

# re extracts natural numbers from filenames such as snapshot100.bin.
import re

# Path gives clean filesystem path handling.
from pathlib import Path

# NumPy reads binary arrays and reshapes numeric data.
import numpy as np


# RAW_BINARY_DTYPE is the real storage type in the source .bin files.
RAW_BINARY_DTYPE = np.float64

# DEFAULT_MATRIX_DTYPE is the compact type we usually save for ML matrices.
DEFAULT_MATRIX_DTYPE = np.float32


def snapshot_number(path: Path) -> int:
    # Search for the first number in the filename stem.
    match = re.search(r"(\d+)", path.stem)

    # Return the number if found, otherwise sort unusual names at the end.
    return int(match.group(1)) if match else 10**12


def list_snapshot_files(snapshots_dir: Path) -> list[Path]:
    # Find all .bin files inside the Snapshots folder.
    snapshot_paths = list(snapshots_dir.glob("*.bin"))

    # Sort naturally so snapshot2 comes before snapshot10.
    return sorted(snapshot_paths, key=snapshot_number)


def cleaned_value_count_from_file_size(path: Path, raw_dtype: np.dtype = RAW_BINARY_DTYPE) -> int | None:
    # Get the file size in bytes.
    byte_size = path.stat().st_size

    # Get how many bytes one raw value uses.
    bytes_per_value = np.dtype(raw_dtype).itemsize

    # If the file size is not divisible by dtype size, it cannot be read cleanly.
    if byte_size % bytes_per_value != 0:
        return None

    # Convert bytes into raw value count.
    raw_value_count = byte_size // bytes_per_value

    # Remove one leading header value from the usable count.
    return raw_value_count - 1


def read_clean_binary_field(
    path: Path,
    raw_dtype: np.dtype = RAW_BINARY_DTYPE,
    output_dtype: np.dtype = DEFAULT_MATRIX_DTYPE,
) -> np.ndarray:
    # Read the raw binary file using the true storage dtype.
    raw_values = np.fromfile(path, dtype=raw_dtype)

    # Remove the leading header value.
    cleaned_values = raw_values[1:]

    # Convert to the requested output dtype, usually float32 for memory savings.
    return cleaned_values.astype(output_dtype, copy=False)


def reshape_values_to_points(values: np.ndarray | None) -> tuple[np.ndarray | None, str]:
    # If no values are available, reshaping cannot happen.
    if values is None:
        return None, "No values were provided."

    # Flatten values so we can test whether they form x, y, z triples.
    flat = np.asarray(values).ravel()

    # Direct case: all values form x, y, z triples.
    if flat.size % 3 == 0:
        return flat.reshape(-1, 3), f"Reshaped directly to ({flat.size // 3}, 3)."

    # Header-like case: skip one leading value, then reshape triples.
    if flat.size > 1 and (flat.size - 1) % 3 == 0:
        return flat[1:].reshape(-1, 3), f"Reshaped after skipping one leading value to ({(flat.size - 1) // 3}, 3)."

    # Failure case: return diagnostics instead of crashing.
    return None, f"{flat.size} values cannot be reshaped to (-1, 3)."


def load_points_bin(path: Path, label: str = "points") -> tuple[np.ndarray | None, dict[str, object]]:
    # Start a diagnostic dictionary for the points.bin file.
    diagnostics: dict[str, object] = {
        f"{label}_points_file": str(path),
        f"{label}_points_file_exists": path.exists(),
        f"{label}_points_dtype_used": None,
        f"{label}_points_array_shape": None,
        f"{label}_points_number_of_values": None,
        f"{label}_points_divisible_by_3": False,
        f"{label}_points_number_of_points": None,
        f"{label}_points_reshape_status": "file missing",
    }

    # If the file is missing, return diagnostics without crashing.
    if not path.exists():
        return None, diagnostics

    # Try float32 first for diagnostics, then float64 because this dataset uses float64.
    for dtype in (np.float32, np.float64):
        # Read the binary file with this dtype.
        values = np.fromfile(path, dtype=dtype)

        # Store shape and divisibility diagnostics.
        diagnostics[f"{label}_points_dtype_used"] = str(np.dtype(dtype))
        diagnostics[f"{label}_points_array_shape"] = str(values.shape)
        diagnostics[f"{label}_points_number_of_values"] = int(values.size)
        diagnostics[f"{label}_points_divisible_by_3"] = bool(values.size % 3 == 0)

        # Try reshaping this interpretation into points.
        points, status = reshape_values_to_points(values)

        # If it worked, return float32 points and diagnostics.
        if points is not None:
            points = points.astype(np.float32, copy=False)
            diagnostics[f"{label}_points_number_of_points"] = int(points.shape[0])
            diagnostics[f"{label}_points_reshape_status"] = status
            return points, diagnostics

        # Record failed status before trying the next dtype.
        diagnostics[f"{label}_points_reshape_status"] = f"not reshapeable as {np.dtype(dtype)}"

    # Return no points if both dtype interpretations failed.
    return None, diagnostics
