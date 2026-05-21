"""Read one geometry and pressure snapshot to explore the binary format.

Beginner explanation:
Binary files are not human-readable like CSV files. We must guess or discover
how the numbers are stored. This script reads one snapshot in different ways so
we can see which interpretation makes sense.

Simple example:
Reading this dataset as float32 gives strange values and NaN. Reading it as
float64 and skipping the first value gives realistic geometry and pressure
values. That tells us the real format is float64 with one header value.

Why this matters:
If we read the binary file with the wrong type, every later step becomes wrong.
PCA may fail, plots may look broken, and machine learning may train on garbage.

Run from the project root:
    python scripts/02_read_one_snapshot.py
"""

# This import enables modern type-hint behavior.
from __future__ import annotations

# json is used to read settings.json and outputDefinition.json.
import json

# re is used to extract snapshot numbers from filenames such as snapshot10.bin.
import re

# Path gives clean, cross-platform filesystem paths.
from pathlib import Path

# NumPy reads binary numeric files and computes statistics.
import numpy as np


# PROJECT_ROOT points to the main project folder, one level above scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DATA_ROOT points to the folder containing geometry/ and pressure/.
DATA_ROOT = PROJECT_ROOT / "data"

# REPORT_PATH is where the text report will be saved.
REPORT_PATH = PROJECT_ROOT / "outputs" / "dataset_first_snapshot_report.txt"


def snapshot_number(path: Path) -> int:
    # Search the filename stem for digits, for example 10 in snapshot10.
    match = re.search(r"(\d+)", path.stem)

    # Return the number if found; otherwise return a large value so odd names sort last.
    return int(match.group(1)) if match else 10**12


def format_json_file(path: Path) -> list[str]:
    # Start a list of text lines that will describe this JSON file.
    lines = [f"{path.name}:"]

    # If the file is missing, record a warning and return immediately.
    if not path.exists():
        lines.append(f"  WARNING: missing file: {path}")
        return lines

    # Try to parse the JSON file safely.
    try:
        # Open the file as UTF-8 text.
        with path.open("r", encoding="utf-8") as file:
            # Convert JSON text into a Python object.
            data = json.load(file)

    # If the JSON syntax is invalid, record the parsing error.
    except json.JSONDecodeError as error:
        lines.append(f"  WARNING: could not parse JSON: {error}")
        return lines

    # If reading the file fails for another reason, record that error.
    except OSError as error:
        lines.append(f"  WARNING: could not read file: {error}")
        return lines

    # Pretty-print the JSON object with indentation.
    pretty_json = json.dumps(data, indent=2)

    # Add each line of the pretty JSON to the report.
    lines.extend(f"  {line}" for line in pretty_json.splitlines())

    # Return all report lines for this JSON file.
    return lines


def summarize_array(label: str, values: np.ndarray) -> list[str]:
    # Start a list of report lines for this numeric array.
    lines = [f"{label}:"]

    # Print how many numeric values were read from the binary file.
    lines.append(f"  number of values: {values.size}")

    # If the value count is divisible by 3, it may represent x, y, z coordinates.
    if values.size % 3 == 0:
        lines.append(f"  possible number of 3D points: {values.size // 3}")

    # If the value count is not divisible by 3, say so explicitly.
    else:
        lines.append("  possible number of 3D points: not divisible by 3")

    # If the file was empty, skip statistics to avoid NumPy warnings.
    if values.size == 0:
        lines.append("  WARNING: no values found")
        return lines

    # Build a mask that is True only for finite values, excluding NaN and infinity.
    finite_mask = np.isfinite(values)

    # Count how many values are not finite; this is useful when a dtype guess is wrong.
    non_finite_count = int(values.size - np.count_nonzero(finite_mask))

    # Report the number of non-finite values before computing statistics.
    lines.append(f"  non-finite values: {non_finite_count}")

    # Keep only finite values and cast to float64 so statistics are stable.
    finite_values = values[finite_mask].astype(np.float64, copy=False)

    # If every value is invalid, report that statistics cannot be computed.
    if finite_values.size == 0:
        lines.append("  min: unavailable because no finite values were found")
        lines.append("  max: unavailable because no finite values were found")
        lines.append("  mean: unavailable because no finite values were found")
        lines.append("  std: unavailable because no finite values were found")

    # Otherwise compute statistics on the finite values only.
    else:
        # Compute minimum finite value.
        lines.append(f"  min: {float(np.min(finite_values))}")

        # Compute maximum finite value.
        lines.append(f"  max: {float(np.max(finite_values))}")

        # Compute average finite value.
        lines.append(f"  mean: {float(np.mean(finite_values))}")

        # Compute standard deviation of finite values.
        lines.append(f"  std: {float(np.std(finite_values))}")

    # Show the first 10 values so we can inspect the raw pattern.
    lines.append(f"  first 10 values: {values[:10].tolist()}")

    # Return all summary lines for this array.
    return lines


def read_binary(path: Path, dtype: np.dtype) -> np.ndarray:
    # Read the entire binary file as the requested NumPy data type.
    return np.fromfile(path, dtype=dtype)


def inspect_dtype(dataset_dir: Path, snapshot_path: Path, dtype: np.dtype, title: str) -> list[str]:
    # Start a section explaining which dtype is being tested.
    lines = [f"\n{title}"]

    # Build the path to points.bin.
    points_path = dataset_dir / "points.bin"

    # If points.bin exists, read and summarize it.
    if points_path.exists():
        points_values = read_binary(points_path, dtype)
        lines.extend(summarize_array("points.bin", points_values))

    # If points.bin is missing, record a warning.
    else:
        lines.append(f"points.bin: WARNING missing file: {points_path}")

    # If the selected snapshot exists, read and summarize it.
    if snapshot_path.exists():
        snapshot_values = read_binary(snapshot_path, dtype)
        lines.extend(summarize_array(f"snapshot ({snapshot_path.name})", snapshot_values))

    # If the snapshot is missing, record a warning.
    else:
        lines.append(f"snapshot: WARNING missing file: {snapshot_path}")

    # Return all lines for this dtype inspection.
    return lines


def infer_from_settings(dataset_dir: Path, snapshot_path: Path) -> list[str]:
    # Start a short interpretation section.
    lines = ["\nFormat clues from settings.json and file sizes:"]

    # Build the settings path.
    settings_path = dataset_dir / "settings.json"

    # If settings are missing, no metadata-based inference is possible.
    if not settings_path.exists():
        lines.append("  WARNING: settings.json is missing, so metadata inference was skipped.")
        return lines

    # Try to read settings safely.
    try:
        # Open settings.json as UTF-8 text.
        with settings_path.open("r", encoding="utf-8") as file:
            # Parse the JSON into a dictionary.
            settings = json.load(file)

    # If parsing or reading fails, record the error.
    except (json.JSONDecodeError, OSError) as error:
        lines.append(f"  WARNING: could not read settings.json for inference: {error}")
        return lines

    # Read the ids list if it exists.
    ids = settings.get("ids")

    # Read the dimensionality list if it exists.
    dimensionality = settings.get("dimensionality")

    # Continue only when ids and dimensionality have the expected structure.
    if not ids or not dimensionality:
        lines.append("  WARNING: settings.json does not contain ids and dimensionality.")
        return lines

    # Convert the id range into the expected number of mesh nodes.
    node_count = int(ids[2]) - int(ids[0]) + 1

    # Convert dimensionality into the expected number of values per node.
    components = int(dimensionality[0])

    # Compute how many field values are expected without any header.
    expected_values = node_count * components

    # Report the metadata interpretation.
    lines.append(f"  node count from settings.json: {node_count}")
    lines.append(f"  components per node from settings.json: {components}")
    lines.append(f"  expected snapshot values without header: {expected_values}")

    # Compute how many float64 values the snapshot file contains.
    snapshot_float64_values = snapshot_path.stat().st_size // np.dtype(np.float64).itemsize

    # Compute how many float32 values the snapshot file contains.
    snapshot_float32_values = snapshot_path.stat().st_size // np.dtype(np.float32).itemsize

    # Report the value counts implied by file size.
    lines.append(f"  snapshot values if float64: {snapshot_float64_values}")
    lines.append(f"  snapshot values if float32: {snapshot_float32_values}")

    # Check if float64 looks like expected field values plus one header value.
    if snapshot_float64_values == expected_values + 1:
        lines.append("  likely snapshot format: float64 values with one leading header value")

    # Check if float64 looks like exactly expected field values.
    elif snapshot_float64_values == expected_values:
        lines.append("  likely snapshot format: float64 values without extra header")

    # Check if float32 looks like expected field values plus one header value.
    elif snapshot_float32_values == expected_values + 1:
        lines.append("  likely snapshot format: float32 values with one leading header value")

    # Check if float32 looks like exactly expected field values.
    elif snapshot_float32_values == expected_values:
        lines.append("  likely snapshot format: float32 values without extra header")

    # If no simple match is found, say that the format needs more investigation.
    else:
        lines.append("  likely snapshot format: unclear from simple size checks")

    # Return all inference lines.
    return lines


def inspect_dataset(name: str) -> list[str]:
    # Build the dataset folder path.
    dataset_dir = DATA_ROOT / name

    # Start the report section for this dataset.
    lines = ["=" * 80, f"Dataset: {name}", f"Path: {dataset_dir}", "=" * 80]

    # If the dataset folder is missing, record a warning and stop this dataset.
    if not dataset_dir.exists():
        lines.append(f"WARNING: dataset folder is missing: {dataset_dir}")
        return lines

    # Build the path to the Snapshots folder.
    snapshots_dir = dataset_dir / "Snapshots"

    # If Snapshots is missing, record a warning and stop this dataset.
    if not snapshots_dir.exists() or not snapshots_dir.is_dir():
        lines.append(f"WARNING: Snapshots folder is missing: {snapshots_dir}")
        return lines

    # Find all .bin snapshot files and sort them by their numeric snapshot number.
    snapshots = sorted(snapshots_dir.glob("*.bin"), key=snapshot_number)

    # If no snapshot files exist, record a warning and stop this dataset.
    if not snapshots:
        lines.append(f"WARNING: no .bin snapshots found in {snapshots_dir}")
        return lines

    # Select the first snapshot after numeric sorting.
    first_snapshot = snapshots[0]

    # Print which snapshot file is being inspected.
    lines.append(f"First snapshot selected: {first_snapshot.name}")

    # Add readable settings.json contents to the report.
    lines.extend(format_json_file(dataset_dir / "settings.json"))

    # Add readable outputDefinition.json contents to the report.
    lines.extend(format_json_file(dataset_dir / "outputDefinition.json"))

    # Try float32 first, exactly as requested.
    lines.extend(inspect_dtype(dataset_dir, first_snapshot, np.float32, "\nReading binary files as float32 first:"))

    # Also compare float64 because these scientific binary files are often float64.
    lines.extend(inspect_dtype(dataset_dir, first_snapshot, np.float64, "\nComparison reading binary files as float64:"))

    # Add metadata and file-size clues to help decide the true format.
    lines.extend(infer_from_settings(dataset_dir, first_snapshot))

    # Return all lines for this dataset.
    return lines


def main() -> None:
    # Create a list that will collect the entire report.
    report_lines: list[str] = []

    # Inspect geometry first.
    report_lines.extend(inspect_dataset("geometry"))

    # Add a blank line between dataset sections.
    report_lines.append("")

    # Inspect pressure second.
    report_lines.extend(inspect_dataset("pressure"))

    # Join all report lines into one text block.
    report_text = "\n".join(report_lines)

    # Print the report to the terminal.
    print(report_text)

    # Make sure the outputs folder exists before saving the report.
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save the report as a text file.
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    # Print the report path so the user knows where it was saved.
    print(f"\nSaved report to: {REPORT_PATH}")


# This makes main() run only when the file is executed directly.
if __name__ == "__main__":
    main()
