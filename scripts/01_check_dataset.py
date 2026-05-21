"""Inspect the geometry and pressure datasets before processing.

Beginner explanation:
This script is like a checklist before starting a laboratory experiment. It
does not train a model and it does not change the dataset. It only checks that
the important files exist and can be opened.

What it checks:
- Is there a Snapshots folder?
- Is there a doe.csv table?
- Is there a points.bin file?
- Can settings.json and outputDefinition.json be read?

Simple example:
If the geometry folder has 100 snapshot files and a readable doe.csv file, the
script prints that geometry looks ready. If points.bin is missing, it prints a
warning instead of crashing.

Run from the project root:
    python scripts/01_check_dataset.py
"""

# This import enables modern type-hint behavior.
from __future__ import annotations

# json is used to parse settings.json and outputDefinition.json.
import json

# os is used here for os.path.getsize, as requested.
import os

# Path provides clean filesystem path handling.
from pathlib import Path

# pandas reads doe.csv files and reports their shape and columns.
import pandas as pd


# PROJECT_ROOT is the main project directory, one level above scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DATA_ROOT is the folder that contains geometry/ and pressure/.
DATA_ROOT = PROJECT_ROOT / "data"

# These are the required files and folders every dataset should contain.
REQUIRED_ITEMS = [
    "Snapshots",
    "doe.csv",
    "points.bin",
    "settings.json",
    "outputDefinition.json",
]


def print_header(title: str) -> None:
    # Print a blank line and a separator so each section is easy to see.
    print("\n" + "=" * 80)

    # Print the section title.
    print(title)

    # Print another separator under the title.
    print("=" * 80)


def file_size_mb(path: Path) -> float:
    # Get the file size in bytes and convert bytes to megabytes.
    return os.path.getsize(path) / (1024 * 1024)


def inspect_json(path: Path, label: str) -> bool:
    # If the JSON file is missing, warn the user and mark this check as failed.
    if not path.exists():
        print(f"WARNING: {label} is missing.")
        return False

    # Try to read and parse the JSON file safely.
    try:
        # Open the JSON file as UTF-8 text.
        with path.open("r", encoding="utf-8") as f:
            # Convert the JSON text into a Python object.
            data = json.load(f)

    # Catch invalid JSON syntax and print the parsing error.
    except json.JSONDecodeError as error:
        print(f"WARNING: Could not parse {label}: {error}")
        return False

    # Catch file reading problems, such as permission errors.
    except OSError as error:
        print(f"WARNING: Could not read {label}: {error}")
        return False

    # If the JSON root is a dictionary, print its top-level keys.
    if isinstance(data, dict):
        print(f"{label} keys: {list(data.keys())}")

    # If the JSON root is another type, print that type instead.
    else:
        print(f"{label} contents type: {type(data).__name__}")

    # Print a label before showing the full JSON contents.
    print(f"{label} contents:")

    # Pretty-print the JSON content with indentation.
    print(json.dumps(data, indent=2))

    # Return True because this JSON file existed and parsed correctly.
    return True


def inspect_doe(path: Path) -> bool:
    # If the DOE file is missing, warn the user and mark this check as failed.
    if not path.exists():
        print("WARNING: doe.csv is missing.")
        return False

    # Try to read the DOE table safely.
    try:
        # sep=None lets pandas detect comma or semicolon separators automatically.
        doe = pd.read_csv(path, sep=None, engine="python")

    # Catch any CSV reading problem and print the error without crashing.
    except Exception as error:
        print(f"WARNING: Could not read doe.csv: {error}")
        return False

    # Print the number of rows and columns in the DOE table.
    print(f"doe.csv shape: {doe.shape}")

    # Print every column name in the DOE table.
    print(f"doe.csv columns: {list(doe.columns)}")

    # Return True because the DOE file existed and loaded correctly.
    return True


def inspect_dataset(name: str) -> bool:
    # Build the folder path for the selected dataset.
    dataset_dir = DATA_ROOT / name

    # Print a clear title for this dataset section.
    print_header(f"Dataset: {name}")

    # Print the full folder path being inspected.
    print(f"Path: {dataset_dir}")

    # Start by assuming the dataset is ready; failed checks will change this to False.
    dataset_ready = True

    # If the dataset folder itself is missing, no later checks can work.
    if not dataset_dir.exists():
        print(f"WARNING: Dataset folder is missing: {dataset_dir}")
        return False

    # Print a section title for the required file checks.
    print("\nRequired files and folders:")

    # Check every required file and folder.
    for item in REQUIRED_ITEMS:
        # Build the path for this required item.
        path = dataset_dir / item

        # Check whether this item exists.
        exists = path.exists()

        # Print FOUND or MISSING for this item.
        print(f"- {item}: {'FOUND' if exists else 'MISSING'}")

        # Mark the dataset not ready if anything required is missing.
        if not exists:
            dataset_ready = False

    # Build the path to the snapshots folder.
    snapshots_dir = dataset_dir / "Snapshots"

    # Continue only if Snapshots exists and is actually a directory.
    if snapshots_dir.exists() and snapshots_dir.is_dir():
        # Collect all .bin files in the Snapshots folder.
        snapshots = sorted(
            [path for path in snapshots_dir.iterdir() if path.is_file() and path.suffix.lower() == ".bin"],
            key=lambda path: path.name,
        )

        # Print how many snapshot files were found.
        print(f"\nNumber of snapshot files: {len(snapshots)}")

        # Print a label before showing the first few filenames.
        print("First 5 snapshot filenames:")

        # Print up to five snapshot filenames.
        for snapshot in snapshots[:5]:
            print(f"- {snapshot.name}")

        # Warn if the folder exists but contains no snapshot files.
        if not snapshots:
            print("WARNING: No .bin snapshot files found.")
            dataset_ready = False

    # If Snapshots is missing or not a folder, warn the user.
    else:
        print("\nWARNING: Snapshots folder is missing or is not a directory.")
        dataset_ready = False

    # Build the path to points.bin.
    points_path = dataset_dir / "points.bin"

    # If points.bin exists, print its file size.
    if points_path.exists():
        print(f"\npoints.bin size: {file_size_mb(points_path):.2f} MB")

    # If points.bin is missing, warn the user.
    else:
        print("\nWARNING: points.bin is missing.")
        dataset_ready = False

    # Print a section title before reading DOE.
    print("\ndoe.csv inspection:")

    # Run DOE inspection and combine the result with previous readiness checks.
    dataset_ready = inspect_doe(dataset_dir / "doe.csv") and dataset_ready

    # Print a section title before reading settings.json.
    print("\nsettings.json inspection:")

    # Run settings.json inspection and combine the result with previous checks.
    dataset_ready = inspect_json(dataset_dir / "settings.json", "settings.json") and dataset_ready

    # Print a section title before reading outputDefinition.json.
    print("\noutputDefinition.json inspection:")

    # Run outputDefinition.json inspection and combine the result with previous checks.
    dataset_ready = inspect_json(dataset_dir / "outputDefinition.json", "outputDefinition.json") and dataset_ready

    # Return the final readiness value for this dataset.
    return dataset_ready


def main() -> None:
    # Inspect both datasets and store their readiness results.
    results = {
        "geometry": inspect_dataset("geometry"),
        "pressure": inspect_dataset("pressure"),
    }

    # Print a final summary section.
    print_header("Summary")

    # Print READY or NOT READY for each dataset.
    for name, ready in results.items():
        status = "READY" if ready else "NOT READY"
        print(f"{name}: {status}")

    # If both datasets passed all checks, print a positive final message.
    if all(results.values()):
        print("\nGeometry and pressure datasets look ready for processing.")

    # Otherwise, tell the user that something must be fixed first.
    else:
        print("\nOne or more datasets need attention before processing.")


# This makes main() run only when the file is executed directly.
if __name__ == "__main__":
    main()
