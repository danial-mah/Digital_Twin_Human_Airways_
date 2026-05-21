"""Build snapshot matrices for geometry and pressure datasets.

This script reads every snapshot file, stacks valid snapshots into a 2D matrix,
saves the matrix as a .npy file, saves the accepted snapshot names, and compares
the number of accepted snapshots with the number of DOE rows.

Important learning note:
The raw binary snapshot files are stored as float64 and include one leading
header value. We read them correctly as float64, remove the header with [1:],
then convert the cleaned data to float32 before saving the matrix.

Beginner example:
Imagine you have 100 pressure simulations. Each simulation contains about
2 million pressure values. This script turns those 100 files into one table:

    rows = simulations
    columns = pressure values

So the matrix looks like:

    pressure_matrix[0] = all pressure values from snapshot1
    pressure_matrix[1] = all pressure values from snapshot2

Why save as float32:
float32 uses less disk and memory than float64. We still read the raw files as
float64 first because that is the real dataset format, then convert safely.

Run from the project root:
    python scripts/03_build_snapshot_matrices.py
"""

# This import enables modern type-hint behavior.
from __future__ import annotations

# sys lets this script add src/ to Python's import path when run from scripts/.
import sys

# Path gives clean filesystem path handling.
from pathlib import Path

# NumPy reads binary arrays and writes .npy matrices.
import numpy as np

# pandas reads DOE CSV files and reports their shape and columns.
import pandas as pd


# Add src/ to the import path so this numbered script can use shared helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import shared project paths.
from airways.paths import DATA_ROOT, OUTPUT_ROOT

# Import shared binary helpers for snapshots.
from airways.binary import (
    DEFAULT_MATRIX_DTYPE,
    RAW_BINARY_DTYPE,
    cleaned_value_count_from_file_size,
    list_snapshot_files,
    read_clean_binary_field,
)


def choose_consistent_snapshots(snapshot_paths: list[Path]) -> tuple[list[Path], int | None]:
    # Start with no expected snapshot size.
    expected_values: int | None = None

    # Store snapshots that match the expected size.
    valid_paths: list[Path] = []

    # Inspect every snapshot file before allocating the output matrix.
    for path in snapshot_paths:
        # Compute how many usable values remain after removing the float64 header.
        value_count = cleaned_value_count_from_file_size(path)

        # Warn and skip files whose byte size is incompatible with float64.
        if value_count is None:
            print(f"WARNING: {path.name} byte size is not divisible by {np.dtype(RAW_BINARY_DTYPE).itemsize}; skipping.")
            continue

        # Warn and skip files that only contain a header or no usable values.
        if value_count <= 0:
            print(f"WARNING: {path.name} does not contain usable values after removing the header; skipping.")
            continue

        # Use the first valid snapshot as the expected size.
        if expected_values is None:
            expected_values = value_count
            valid_paths.append(path)
            continue

        # If this snapshot size differs, warn and skip it.
        if value_count != expected_values:
            print(
                f"WARNING: {path.name} has {value_count} values, "
                f"expected {expected_values}; skipping."
            )
            continue

        # Keep this snapshot because it matches the expected size.
        valid_paths.append(path)

    # Return the accepted files and the shared number of values per snapshot.
    return valid_paths, expected_values


def save_snapshot_names(paths: list[Path], output_path: Path) -> None:
    # Convert accepted snapshot paths into only their filenames.
    names = [path.name for path in paths]

    # Join filenames with newline characters for a simple text file.
    text = "\n".join(names)

    # Add a final newline when at least one name exists.
    if text:
        text += "\n"

    # Save the accepted snapshot names as UTF-8 text.
    output_path.write_text(text, encoding="utf-8")


def build_snapshot_matrix(dataset_name: str) -> tuple[tuple[int, int] | None, int]:
    # Build the path to this dataset folder.
    dataset_dir = DATA_ROOT / dataset_name

    # Build the path to this dataset's Snapshots folder.
    snapshots_dir = dataset_dir / "Snapshots"

    # Build the output folder path for this dataset.
    output_dir = OUTPUT_ROOT / dataset_name

    # Create the output folder if it does not already exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the output .npy matrix path.
    matrix_path = output_dir / f"{dataset_name}_snapshot_matrix.npy"

    # Build the output text file path for accepted snapshot names.
    names_path = output_dir / f"{dataset_name}_snapshot_names.txt"

    # Print a section header for the dataset being processed.
    print("\n" + "=" * 80)
    print(f"Building snapshot matrix for: {dataset_name}")
    print("=" * 80)

    # Stop early if the Snapshots folder is missing.
    if not snapshots_dir.exists() or not snapshots_dir.is_dir():
        print(f"WARNING: Snapshots folder not found: {snapshots_dir}")
        return None, 0

    # List snapshot files in natural order.
    snapshot_paths = list_snapshot_files(snapshots_dir)

    # Print how many snapshot files were found before filtering.
    print(f"Found snapshot files: {len(snapshot_paths)}")

    # Stop early if there are no snapshot files.
    if not snapshot_paths:
        print("WARNING: No snapshot files found.")
        return None, 0

    # Print the raw dtype used for reading the binary source files.
    print(f"Reading raw snapshots with dtype: {np.dtype(RAW_BINARY_DTYPE)}")

    # Print the saved matrix dtype.
    print(f"Saving cleaned snapshot matrix with dtype: {np.dtype(DEFAULT_MATRIX_DTYPE)}")

    # Select only snapshots with consistent value counts.
    valid_paths, values_per_snapshot = choose_consistent_snapshots(snapshot_paths)

    # Stop early if no valid snapshots remain.
    if values_per_snapshot is None or not valid_paths:
        print("WARNING: No valid snapshots could be used.")
        return None, 0

    # Print the matrix shape that will be created.
    print(f"Accepted snapshots: {len(valid_paths)}")
    print(f"Values per snapshot: {values_per_snapshot}")
    print(f"Output matrix shape: ({len(valid_paths)}, {values_per_snapshot})")

    # Create a .npy file as a memory-mapped array to avoid holding the full matrix in RAM.
    matrix = np.lib.format.open_memmap(
        matrix_path,
        mode="w+",
        dtype=DEFAULT_MATRIX_DTYPE,
        shape=(len(valid_paths), values_per_snapshot),
    )

    # Read each valid snapshot and write it into one matrix row.
    for row_index, path in enumerate(valid_paths, start=1):
        # Print progress so the user can see the script is working.
        print(f"[{dataset_name}] Reading {row_index}/{len(valid_paths)}: {path.name}")

        # Read as float64, remove the first header value, then convert to float32.
        values = read_clean_binary_field(path)

        # Double-check the size even though we already checked by file size.
        if values.size != values_per_snapshot:
            print(
                f"WARNING: {path.name} read {values.size} values, "
                f"expected {values_per_snapshot}; leaving this row unchanged."
            )
            continue

        # Store this snapshot as one row in the output matrix.
        matrix[row_index - 1, :] = values

    # Flush the memory map so all data is written to disk.
    matrix.flush()

    # Save the accepted snapshot filenames next to the matrix.
    save_snapshot_names(valid_paths, names_path)

    # Print where the outputs were saved.
    print(f"Saved matrix to: {matrix_path}")
    print(f"Saved snapshot names to: {names_path}")

    # Return the matrix shape and number of accepted snapshots.
    return (len(valid_paths), values_per_snapshot), len(valid_paths)


def load_doe(dataset_name: str) -> pd.DataFrame | None:
    # Build the path to the DOE CSV file.
    doe_path = DATA_ROOT / dataset_name / "doe.csv"

    # If the DOE file is missing, warn and return None.
    if not doe_path.exists():
        print(f"WARNING: DOE file not found for {dataset_name}: {doe_path}")
        return None

    # Try reading the DOE file safely.
    try:
        # sep=None lets pandas detect comma or semicolon separators automatically.
        doe = pd.read_csv(doe_path, sep=None, engine="python")

    # If pandas cannot read the file, print the error and return None.
    except Exception as error:
        print(f"WARNING: Could not read {doe_path}: {error}")
        return None

    # Print the shape of the DOE table.
    print(f"\n{dataset_name} DOE shape: {doe.shape}")

    # Print the DOE column names.
    print(f"{dataset_name} DOE columns: {list(doe.columns)}")

    # Return the loaded DOE table.
    return doe


def main() -> None:
    # Build the geometry snapshot matrix and remember its shape.
    geometry_matrix_shape, geometry_snapshot_count = build_snapshot_matrix("geometry")

    # Build the pressure snapshot matrix and remember its shape.
    pressure_matrix_shape, pressure_snapshot_count = build_snapshot_matrix("pressure")

    # Load the geometry DOE table.
    geometry_doe = load_doe("geometry")

    # Load the pressure DOE table.
    pressure_doe = load_doe("pressure")

    # Extract DOE shapes if the files loaded successfully.
    geometry_doe_shape = geometry_doe.shape if geometry_doe is not None else None
    pressure_doe_shape = pressure_doe.shape if pressure_doe is not None else None

    # Compare number of accepted geometry snapshots with geometry DOE rows.
    geometry_consistent = geometry_doe is not None and geometry_snapshot_count == len(geometry_doe)

    # Compare number of accepted pressure snapshots with pressure DOE rows.
    pressure_consistent = pressure_doe is not None and pressure_snapshot_count == len(pressure_doe)

    # Combine both consistency checks into one final boolean.
    all_consistent = geometry_consistent and pressure_consistent

    # Print the final summary section.
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    # Print geometry matrix shape.
    print(f"Geometry matrix shape: {geometry_matrix_shape}")

    # Print pressure matrix shape.
    print(f"Pressure matrix shape: {pressure_matrix_shape}")

    # Print geometry DOE shape.
    print(f"Geometry DOE shape: {geometry_doe_shape}")

    # Print pressure DOE shape.
    print(f"Pressure DOE shape: {pressure_doe_shape}")

    # Print whether geometry snapshot count matches DOE row count.
    print(f"Geometry snapshot count matches DOE rows: {geometry_consistent}")

    # Print whether pressure snapshot count matches DOE row count.
    print(f"Pressure snapshot count matches DOE rows: {pressure_consistent}")

    # Print whether all dimensions look consistent.
    print(f"Dimensions are consistent: {all_consistent}")


# This makes main() run only when this file is executed directly.
if __name__ == "__main__":
    main()
