"""Build snapshot matrices for geometry and pressure datasets.

This script reads every snapshot file, stacks valid snapshots into a 2D matrix,
saves the matrix as a .npy file, saves the accepted snapshot names, and compares
the number of accepted snapshots with the number of DOE rows.

Important learning note:
The previous inspection script suggested these binary files are likely float64
with one leading header value. This script uses float32 because that is the
requested experiment, but the printed value counts should be checked carefully.
"""

# This import enables modern type-hint behavior.
from __future__ import annotations

# re is used to extract numbers from filenames such as snapshot100.bin.
import re

# Path gives clean filesystem path handling.
from pathlib import Path

# NumPy reads binary arrays and writes .npy matrices.
import numpy as np

# pandas reads DOE CSV files and reports their shape and columns.
import pandas as pd


# PROJECT_ROOT points to the main project folder, one level above scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DATA_ROOT points to the folder that contains geometry/ and pressure/.
DATA_ROOT = PROJECT_ROOT / "data"

# OUTPUT_ROOT points to the folder where generated matrices will be saved.
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# SNAPSHOT_DTYPE is the binary data type requested for this script.
SNAPSHOT_DTYPE = np.float32


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


def value_count_from_file_size(path: Path, dtype: np.dtype) -> int | None:
    # Get the size of the file in bytes.
    byte_size = path.stat().st_size

    # Get how many bytes one value of the selected dtype uses.
    bytes_per_value = np.dtype(dtype).itemsize

    # If the file size is not divisible by the dtype size, it cannot be read cleanly.
    if byte_size % bytes_per_value != 0:
        return None

    # Convert bytes into number of dtype values.
    return byte_size // bytes_per_value


def choose_consistent_snapshots(snapshot_paths: list[Path], dtype: np.dtype) -> tuple[list[Path], int | None]:
    # Start with no expected snapshot size.
    expected_values: int | None = None

    # Store snapshots that match the expected size.
    valid_paths: list[Path] = []

    # Inspect every snapshot file before allocating the output matrix.
    for path in snapshot_paths:
        # Compute how many dtype values this file contains based on file size.
        value_count = value_count_from_file_size(path, dtype)

        # Warn and skip files whose byte size is incompatible with this dtype.
        if value_count is None:
            print(f"WARNING: {path.name} byte size is not divisible by {np.dtype(dtype).itemsize}; skipping.")
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

    # Print the dtype being used for this requested build.
    print(f"Reading snapshots with dtype: {np.dtype(SNAPSHOT_DTYPE)}")

    # Warn that float32 may not match the inspected binary format.
    print("Note: Previous inspection suggested float64 with one leading header; this script uses float32 by request.")

    # Select only snapshots with consistent value counts.
    valid_paths, values_per_snapshot = choose_consistent_snapshots(snapshot_paths, SNAPSHOT_DTYPE)

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
        dtype=SNAPSHOT_DTYPE,
        shape=(len(valid_paths), values_per_snapshot),
    )

    # Read each valid snapshot and write it into one matrix row.
    for row_index, path in enumerate(valid_paths, start=1):
        # Print progress so the user can see the script is working.
        print(f"[{dataset_name}] Reading {row_index}/{len(valid_paths)}: {path.name}")

        # Read the binary file as float32, according to the requested experiment.
        values = np.fromfile(path, dtype=SNAPSHOT_DTYPE)

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
