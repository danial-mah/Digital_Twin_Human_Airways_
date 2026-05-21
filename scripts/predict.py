"""Generate a digital twin field prediction from a trained model.

This script loads one saved model from outputs/models/, builds one row of input
parameters, predicts the compressed PCA coordinates, reconstructs the field,
and saves the final predicted field as a NumPy .npy file.

Beginner example:
If you already trained a pressure model, this script can use one DOE row and
create a predicted pressure field:

    python scripts/predict.py --dataset pressure --doe-row 0

This file is an earlier simple prediction script. The newer full pipeline uses
scripts/06_predict_and_reconstruct.py for both geometry and pressure.
"""

# This import enables modern type-hint behavior.
from __future__ import annotations

# argparse lets the user choose options from the command line.
import argparse

# json reads custom input parameters from a JSON file.
import json

# Path provides clean filesystem path handling.
from pathlib import Path

# joblib loads the trained model saved by train_surrogates.py.
import joblib

# NumPy handles numeric arrays and saves .npy prediction files.
import numpy as np

# pandas is used to build a one-row input table.
import pandas as pd

# Import shared helpers for project paths, DOE loading, and model paths.
from airways_io import PROJECT_ROOT, load_dataset, load_doe, model_path


def parse_args() -> argparse.Namespace:
    # Create the command-line parser and describe what the script does.
    parser = argparse.ArgumentParser(description="Predict an airway field from trained surrogate inputs.")

    # Select whether the geometry or pressure model should be used.
    parser.add_argument("--dataset", choices=["geometry", "pressure"], required=True)

    # Optionally provide a JSON file containing parameter values by name.
    parser.add_argument("--params-json", type=Path, default=None, help="JSON file containing input parameters.")

    # Optionally use one existing row from the DOE table as input.
    parser.add_argument("--doe-row", type=int, default=None, help="Use a zero-based row from the dataset DOE file.")

    # Optionally choose where the prediction .npy file will be saved.
    parser.add_argument("--output", type=Path, default=None)

    # Return the parsed command-line options.
    return parser.parse_args()


def load_features(args: argparse.Namespace, feature_columns: list[str]) -> pd.DataFrame:
    # If the user gave a JSON file, use it as the input source.
    if args.params_json:
        # Open the JSON parameter file as text.
        with args.params_json.open("r", encoding="utf-8") as f:
            # Parse the JSON into a Python dictionary.
            params = json.load(f)

        # Build a one-row DataFrame in exactly the feature order the model expects.
        return pd.DataFrame([{column: params[column] for column in feature_columns}])

    # If the user gave a DOE row number, use that existing design point.
    if args.doe_row is not None:
        # Load the DOE table for the selected dataset.
        doe = load_doe(load_dataset(args.dataset))

        # Return only the selected row and only the required feature columns.
        return doe.loc[[args.doe_row], feature_columns]

    # Stop if the user did not provide any input source.
    raise ValueError("Provide either --params-json or --doe-row.")


def main() -> None:
    # Read command-line options.
    args = parse_args()

    # Load the trained model dictionary for the selected dataset.
    model = joblib.load(model_path(args.dataset))

    # Build the input matrix and convert it to float64 for scikit-learn.
    x = load_features(args, model["feature_columns"]).to_numpy(dtype=np.float64)

    # Predict PCA latent coordinates from the input parameters.
    latent = model["surrogate"].predict(x)

    # Reconstruct the flattened physical field from PCA coordinates.
    flattened = model["pca"].inverse_transform(latent)[0]

    # Convert the flat vector back to nodes x components.
    prediction = flattened.reshape(-1, model["components_per_node"])

    # Use the requested output path, or create a default path in outputs/predictions.
    out_path = args.output or PROJECT_ROOT / "outputs" / "predictions" / f"{args.dataset}_prediction.npy"

    # Make sure the output folder exists.
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the prediction array to disk.
    np.save(out_path, prediction)

    # Print the saved file path for the user.
    print(f"Saved prediction: {out_path}")

    # Print the shape so the user knows how many nodes and components were predicted.
    print(f"Prediction shape: {prediction.shape}")


# This makes main() run only when the file is executed directly.
if __name__ == "__main__":
    main()
