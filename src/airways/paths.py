"""Central project paths used by scripts and apps.

This file has one simple job: define important folder locations once.

Why this helps:
- Many scripts need to know where `data/`, `outputs/`, and `outputs/models/` are.
- If every script builds those paths by itself, mistakes become easy.
- By keeping the paths here, every script can import the same trusted locations.

Beginner example:
Instead of writing this everywhere:

    Path("outputs") / "models"

we write:

    from airways.paths import MODELS_DIR

Then every file uses the same model folder path.
"""

# This line allows Python type hints to be handled in a modern, flexible way.
# It is not essential for the math, but it helps keep annotations clean.
from __future__ import annotations

# Path is a Python object for working with folders and files.
# It is safer and clearer than manually joining strings with "/" or "\\".
from pathlib import Path


# __file__ means "the path of this current file", which is src/airways/paths.py.
# resolve() converts it into an absolute path.
# parents[2] moves two folders upward:
#   paths.py -> airways/ -> src/ -> project root
# The final result is the main project folder: Digital_Twin_Airways_Project/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# DATA_ROOT points to the folder that stores the original datasets.
# In this project that is:
#   Digital_Twin_Airways_Project/data/
DATA_ROOT = PROJECT_ROOT / "data"

# OUTPUT_ROOT points to the folder where generated files are saved.
# Examples include snapshot matrices, PCA outputs, models, figures, and predictions.
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# MODELS_DIR points specifically to the folder that stores trained models.
# Examples:
#   geometry_pca_model.joblib
#   pressure_best_surrogate.joblib
MODELS_DIR = OUTPUT_ROOT / "models"

# FIGURES_DIR points to the folder for generated plots.
# Examples:
#   geometry_explained_variance.png
#   pressure_explained_variance.png
FIGURES_DIR = OUTPUT_ROOT / "figures"

# PREDICTIONS_DIR points to the folder for prediction results.
# Examples:
#   predicted_geometry_row0.npy
#   predicted_pressure_row0.npy
#   proxy_physics_row0.csv
PREDICTIONS_DIR = OUTPUT_ROOT / "predictions"
