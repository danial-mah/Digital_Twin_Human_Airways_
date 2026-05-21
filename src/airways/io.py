"""Small input/output helpers shared by scripts and apps.

Input/output, often shortened to I/O, means reading files from disk and writing
files back to disk. These helpers keep common file-reading tasks in one place.
"""

# This line allows Python type hints to be handled in a modern, flexible way.
from __future__ import annotations

# Path is used to represent file locations such as data/geometry/doe.csv.
from pathlib import Path

# NumPy is used here for loading .npy prediction arrays.
import numpy as np

# pandas is used here for reading CSV tables such as doe.csv.
import pandas as pd


def load_doe(path: Path) -> pd.DataFrame:
    """Load a DOE CSV file into a pandas DataFrame.

    DOE means "Design of Experiments". In this project, each DOE row contains
    input parameters for one simulation/snapshot.
    """

    # pd.read_csv reads a CSV file into a table-like object called a DataFrame.
    # sep=None tells pandas to detect the separator automatically.
    # This matters because one DOE file may use commas, while another may use semicolons.
    # engine="python" is required when using automatic separator detection.
    return pd.read_csv(path, sep=None, engine="python")


def read_feature_names(path: Path) -> list[str]:
    """Read model input feature names from a text file.

    The feature-name file contains one column name per line. The order matters
    because the surrogate model expects input values in the same order used
    during training.
    """

    # path.read_text(...) reads the entire text file as one string.
    # splitlines() breaks that string into one list item per line.
    # line.strip() removes extra spaces and newline characters.
    # The final "if line.strip()" ignores blank lines.
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_feature_names(feature_names: list[str], path: Path) -> None:
    """Save model input feature names to a text file.

    Saving feature names is important because prediction scripts must use the
    same input columns, in the same order, as the training script.
    """

    # "\n".join(...) combines all feature names into one text block.
    # Each feature name is placed on its own line.
    text = "\n".join(feature_names)

    # If there is at least one feature name, add one final newline.
    # This is a common style for clean text files.
    if text:
        text += "\n"

    # write_text saves the text to disk using UTF-8 encoding.
    path.write_text(text, encoding="utf-8")


def load_prediction(path: Path, label: str) -> np.ndarray | None:
    """Load a saved prediction array from a .npy file.

    Returns None instead of crashing if the file does not exist. This makes
    scripts friendlier while the pipeline is still being built step by step.
    """

    # Check whether the file exists before trying to load it.
    # If it is missing, print a clear warning and return None.
    if not path.exists():
        print(f"WARNING: Missing {label} prediction file: {path}")
        return None

    # np.load reads a NumPy .npy file back into a NumPy array.
    values = np.load(path)

    # Convert values to float64 for stable metric calculations.
    # Even if the file was saved as float32, float64 is safer for summaries.
    values = np.asarray(values, dtype=np.float64)

    # Print the loaded shape so the user can see what was loaded.
    # Example shape: (2135906,) means about 2.1 million scalar values.
    print(f"Loaded {label} prediction with shape {values.shape}.")

    # Return the loaded prediction array to the caller.
    return values
