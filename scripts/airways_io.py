"""Shared input/output helpers for the human airways digital twin project.

This file contains reusable functions for reading metadata, DOE tables, mesh
points, and snapshot fields. Other scripts import these helpers so the dataset
loading logic is written once instead of repeated in every script.
"""

# This import lets us write modern type hints, such as list[Path], in a way
# that stays friendly across Python versions.
from __future__ import annotations

# json reads the settings.json files that describe the airway datasets.
import json

# re is used to extract the number from filenames such as snapshot42.bin.
import re

# dataclass creates a small structured object for dataset paths and metadata.
from dataclasses import dataclass

# Path gives clean cross-platform filesystem paths instead of raw strings.
from pathlib import Path

# Iterable is a type hint for inputs that can be looped over, such as lists.
from typing import Iterable

# NumPy handles the large binary numeric arrays efficiently.
import numpy as np

# pandas reads the DOE CSV files and manages tabular input parameters.
import pandas as pd


# PROJECT_ROOT points to the main project folder, one level above scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DATA_ROOT points to the folder that contains geometry/ and pressure/.
DATA_ROOT = PROJECT_ROOT / "data"


# frozen=True makes this metadata object immutable after it is created.
@dataclass(frozen=True)
class AirwaysDataset:
    # The dataset name, usually "geometry" or "pressure".
    name: str

    # The root folder for this dataset, for example data/geometry.
    root: Path

    # The path to the DOE CSV table for this dataset.
    doe_path: Path

    # The path to points.bin, which stores mesh point coordinates.
    points_path: Path

    # The path to settings.json, which stores metadata about the field.
    settings_path: Path

    # The folder containing snapshot*.bin files.
    snapshots_dir: Path

    # Number of values per mesh node: 3 for coordinates, 1 for pressure.
    components: int

    # Number of nodes in the airway mesh.
    node_count: int

    # Human-readable field name from settings.json.
    field_name: str

    # Physical unit from settings.json, such as Pa for pressure.
    unit: str | None

    # Dictionary of named mesh regions and their node index ranges.
    named_selections: dict[str, list[int]]


def load_dataset(name: str, data_root: Path = DATA_ROOT) -> AirwaysDataset:
    """Read metadata for a dataset such as geometry or pressure."""

    # Build the folder path for the selected dataset.
    root = data_root / name

    # Build the path to the dataset metadata file.
    settings_path = root / "settings.json"

    # Open settings.json as text using UTF-8 encoding.
    with settings_path.open("r", encoding="utf-8") as f:
        # Parse the JSON text into a Python dictionary.
        settings = json.load(f)

    # Read the id range from settings.json; the last value gives the final node id.
    ids = settings["ids"]

    # Convert the id range into a count of mesh nodes.
    node_count = int(ids[2]) - int(ids[0]) + 1

    # Read how many values each node has, such as 3 coordinates or 1 pressure.
    components = int(settings["dimensionality"][0])

    # Return one structured object containing all useful paths and metadata.
    return AirwaysDataset(
        name=name,
        root=root,
        doe_path=root / "doe.csv",
        points_path=root / "points.bin",
        settings_path=settings_path,
        snapshots_dir=root / "Snapshots",
        components=components,
        node_count=node_count,
        field_name=settings.get("name", name),
        unit=settings.get("unit"),
        named_selections=settings.get("namedSelections", {}),
    )


def load_doe(dataset: AirwaysDataset) -> pd.DataFrame:
    """Load DOE parameters and normalize the snapshot filename column."""

    # Read the CSV file; sep=None lets pandas detect comma or semicolon separators.
    doe = pd.read_csv(dataset.doe_path, sep=None, engine="python")

    # The first column contains the snapshot filename, but its name differs by dataset.
    first_column = doe.columns[0]

    # Rename the first column to a shared name so later scripts can use one convention.
    doe = doe.rename(columns={first_column: "snapshot"})

    # Force snapshot names to strings so path matching is reliable.
    doe["snapshot"] = doe["snapshot"].astype(str)

    # Return the cleaned DOE table.
    return doe


def snapshot_number(path: Path) -> int:
    # Search the filename stem for digits, for example "42" in snapshot42.
    match = re.search(r"(\d+)", path.stem)

    # Return the numeric part if it exists; otherwise use -1 for unusual names.
    return int(match.group(1)) if match else -1


def list_snapshot_paths(dataset: AirwaysDataset) -> list[Path]:
    # Find every binary snapshot file inside the dataset Snapshots folder.
    paths = list(dataset.snapshots_dir.glob("*.bin"))

    # Sort by snapshot number so snapshot2 comes before snapshot10.
    return sorted(paths, key=snapshot_number)


def selection_indices(dataset: AirwaysDataset, selection: str | None) -> np.ndarray | None:
    """Return zero-based node indices for a named airway region."""

    # If no region was requested, return None so callers use the full mesh.
    if not selection:
        return None

    # Stop early if the requested named region does not exist.
    if selection not in dataset.named_selections:
        # Build a readable list of valid region names for the error message.
        available = ", ".join(sorted(dataset.named_selections))

        # Raise a helpful error instead of failing later with a vague KeyError.
        raise ValueError(f"Unknown selection '{selection}'. Available selections: {available}")

    # Each named selection stores [start, unused_middle_value, end].
    start, _, end = dataset.named_selections[selection]

    # Create all node indices from start to end, inclusive.
    return np.arange(int(start), int(end) + 1, dtype=np.int64)


def read_binary_field(
    path: Path,
    node_count: int,
    components: int,
    indices: np.ndarray | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Read one binary field file and return an array shaped as nodes x components.

    The source files contain one 8-byte header value followed by float64 payload data.
    """

    # Calculate how many numeric values the real field should contain.
    expected = node_count * components

    # Read the entire binary file as float64 values.
    values = np.fromfile(path, dtype=np.float64)

    # If the file has one extra value, remove the first header value.
    if values.size == expected + 1:
        values = values[1:]

    # If the size is not expected, stop because reshaping would be unsafe.
    elif values.size != expected:
        raise ValueError(
            f"{path} has {values.size} float64 values; expected {expected} or {expected + 1}."
        )

    # Reshape the flat values into one row per node and one column per component.
    field = values.reshape(node_count, components)

    # If a named selection was requested, keep only those node rows.
    if indices is not None:
        field = field[indices]

    # Convert to float32 to reduce memory use while preserving enough precision.
    return field.astype(dtype, copy=False)


def load_points(dataset: AirwaysDataset, indices: np.ndarray | None = None) -> np.ndarray:
    # points.bin always stores 3D coordinates, so components is fixed to 3 here.
    return read_binary_field(dataset.points_path, dataset.node_count, 3, indices=indices)


def load_snapshot_matrix(
    dataset: AirwaysDataset,
    snapshot_names: Iterable[str],
    selection: str | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Load snapshots into a 2D matrix shaped as samples x flattened field values."""

    # Convert an optional named region into an array of node indices.
    indices = selection_indices(dataset, selection)

    # Build a lookup from filename to full path for fast snapshot access.
    path_by_name = {path.name: path for path in list_snapshot_paths(dataset)}

    # Store each flattened snapshot row before stacking them into a matrix.
    rows: list[np.ndarray] = []

    # Loop over snapshot filenames in the same order as the DOE table.
    for snapshot_name in snapshot_names:
        # Find the actual file path for this snapshot name.
        path = path_by_name.get(str(snapshot_name))

        # If the DOE references a file that is missing, report the exact name.
        if path is None:
            raise FileNotFoundError(f"Snapshot '{snapshot_name}' was not found in {dataset.snapshots_dir}.")

        # Read one snapshot as nodes x components.
        field = read_binary_field(
            path,
            dataset.node_count,
            dataset.components,
            indices=indices,
            dtype=dtype,
        )

        # Flatten the field into one long row for machine learning.
        rows.append(field.ravel())

    # Protect against accidentally training with zero snapshots.
    if not rows:
        raise ValueError(f"No snapshots were loaded for dataset '{dataset.name}'.")

    # Stack all rows into a samples x features matrix.
    return np.vstack(rows)


def feature_columns(doe: pd.DataFrame) -> list[str]:
    # Use every numeric DOE column except the snapshot filename as a model input.
    return [column for column in doe.columns if column != "snapshot" and pd.api.types.is_numeric_dtype(doe[column])]


def model_path(dataset_name: str, models_dir: Path | None = None) -> Path:
    # Use the provided model folder, or default to outputs/models.
    root = models_dir or PROJECT_ROOT / "outputs" / "models"

    # Return the standard filename for this dataset model.
    return root / f"{dataset_name}_digital_twin.joblib"
