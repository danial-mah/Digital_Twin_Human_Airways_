"""Central project paths used by scripts and apps.

Keeping paths in one file avoids repeating the same PROJECT_ROOT, DATA_ROOT,
and OUTPUT_ROOT definitions across many scripts.
"""

# This import enables modern type-hint behavior.
from __future__ import annotations

# Path gives clean filesystem path handling.
from pathlib import Path


# PROJECT_ROOT points to the main repository folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# DATA_ROOT points to data/ where geometry and pressure datasets live.
DATA_ROOT = PROJECT_ROOT / "data"

# OUTPUT_ROOT points to outputs/ where generated artifacts are stored.
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# MODELS_DIR points to trained models and scalers.
MODELS_DIR = OUTPUT_ROOT / "models"

# FIGURES_DIR points to generated plot images.
FIGURES_DIR = OUTPUT_ROOT / "figures"

# PREDICTIONS_DIR points to generated prediction files.
PREDICTIONS_DIR = OUTPUT_ROOT / "predictions"
