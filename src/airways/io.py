"""Small input/output helpers shared by scripts and apps."""

# This import enables modern type-hint behavior.
from __future__ import annotations

# Path gives clean filesystem path handling.
from pathlib import Path

# NumPy loads prediction arrays.
import numpy as np

# pandas reads DOE CSV files.
import pandas as pd


def load_doe(path: Path) -> pd.DataFrame:
    # Read CSV using automatic separator detection for comma or semicolon files.
    return pd.read_csv(path, sep=None, engine="python")


def read_feature_names(path: Path) -> list[str]:
    # Read one feature name per line and ignore empty lines.
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_feature_names(feature_names: list[str], path: Path) -> None:
    # Join feature names with newline characters.
    text = "\n".join(feature_names)

    # Add a final newline for a clean text file.
    if text:
        text += "\n"

    # Save the feature names to disk.
    path.write_text(text, encoding="utf-8")


def load_prediction(path: Path, label: str) -> np.ndarray | None:
    # If the prediction file is missing, print a warning and return None.
    if not path.exists():
        print(f"WARNING: Missing {label} prediction file: {path}")
        return None

    # Load the prediction from the .npy file.
    values = np.load(path)

    # Convert to float64 for stable metric calculations.
    values = np.asarray(values, dtype=np.float64)

    # Print the loaded shape for diagnostics.
    print(f"Loaded {label} prediction with shape {values.shape}.")

    # Return the loaded prediction array.
    return values
