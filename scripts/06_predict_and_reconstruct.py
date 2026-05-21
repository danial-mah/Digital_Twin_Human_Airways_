"""Predict and reconstruct airway fields from DOE parameters.

This script proves the digital twin pipeline:

    DOE input -> surrogate model -> POD coefficients -> reconstructed field

For now it uses row index 0 from the geometry and pressure DOE tables, predicts
POD coefficients with the trained surrogate models, reconstructs fields with
PCA inverse_transform, and compares the reconstructed fields with row 0 of the
original snapshot matrices when those matrices are available.
"""

# This import enables modern type-hint behavior.
from __future__ import annotations

# Path gives clean filesystem path handling.
from pathlib import Path

# joblib loads saved surrogate models, scalers, and PCA models.
import joblib

# NumPy handles numeric arrays and error calculations.
import numpy as np

# pandas reads DOE CSV files and writes the error report.
import pandas as pd


# PROJECT_ROOT points to the main project folder, one level above scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DATA_ROOT points to the folder containing geometry/ and pressure/.
DATA_ROOT = PROJECT_ROOT / "data"

# OUTPUT_ROOT points to the generated outputs folder.
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# MODELS_DIR points to saved model files.
MODELS_DIR = OUTPUT_ROOT / "models"

# PREDICTIONS_DIR points to the folder where reconstructed predictions are saved.
PREDICTIONS_DIR = OUTPUT_ROOT / "predictions"

# ROW_INDEX selects the DOE row used as an example input.
ROW_INDEX = 0

# DATASETS stores all input and output paths for geometry and pressure.
DATASETS = {
    "geometry": {
        "doe_path": DATA_ROOT / "geometry" / "doe.csv",
        "surrogate_path": MODELS_DIR / "geometry_best_surrogate.joblib",
        "scaler_path": MODELS_DIR / "geometry_input_scaler.joblib",
        "feature_names_path": MODELS_DIR / "geometry_feature_names.txt",
        "pca_path": MODELS_DIR / "geometry_pca_model.joblib",
        "snapshot_matrix_path": OUTPUT_ROOT / "geometry" / "geometry_snapshot_matrix.npy",
        "prediction_path": PREDICTIONS_DIR / "predicted_geometry_row0.npy",
    },
    "pressure": {
        "doe_path": DATA_ROOT / "pressure" / "doe.csv",
        "surrogate_path": MODELS_DIR / "pressure_best_surrogate.joblib",
        "scaler_path": MODELS_DIR / "pressure_input_scaler.joblib",
        "feature_names_path": MODELS_DIR / "pressure_feature_names.txt",
        "pca_path": MODELS_DIR / "pressure_pca_model.joblib",
        "snapshot_matrix_path": OUTPUT_ROOT / "pressure" / "pressure_snapshot_matrix.npy",
        "prediction_path": PREDICTIONS_DIR / "predicted_pressure_row0.npy",
    },
}


def load_feature_names(path: Path) -> list[str]:
    # If the feature-name file is missing, stop because column order is important.
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature names file: {path}. "
            "Run python scripts/05_train_surrogate_models.py first."
        )

    # Read all non-empty lines as feature names.
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_doe_row(doe_path: Path, feature_names: list[str], row_index: int) -> tuple[pd.Series, np.ndarray]:
    # Read the DOE CSV file; sep=None lets pandas detect comma or semicolon separators.
    doe = pd.read_csv(doe_path, sep=None, engine="python")

    # Stop early if the requested row does not exist.
    if row_index >= len(doe):
        raise IndexError(f"DOE row index {row_index} is out of range for {doe_path}, which has {len(doe)} rows.")

    # Check that every trained feature is available in this DOE file.
    missing = [name for name in feature_names if name not in doe.columns]

    # Stop with a clear error if any required feature is missing.
    if missing:
        raise KeyError(f"Missing required feature columns in {doe_path}: {missing}")

    # Select the requested row as a pandas Series for readable printing.
    row = doe.iloc[row_index]

    # Build a one-row input matrix in exactly the order used during training.
    x = doe.loc[[row_index], feature_names].to_numpy(dtype=np.float32)

    # Return both the readable row and the numeric model input.
    return row, x


def relative_l2_error(reference: np.ndarray, prediction: np.ndarray) -> float:
    # Compute the L2 norm of the difference between true and predicted fields.
    numerator = np.linalg.norm(reference - prediction)

    # Compute the L2 norm of the true field.
    denominator = np.linalg.norm(reference)

    # Avoid division by zero if the reference field is all zeros.
    if denominator == 0:
        return float("nan")

    # Return the relative L2 error.
    return float(numerator / denominator)


def compute_errors(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    # Compute difference values once and reuse them.
    difference = reference - prediction

    # Compute relative L2 error.
    rel_l2 = relative_l2_error(reference, prediction)

    # Compute mean absolute error.
    mae = float(np.mean(np.abs(difference)))

    # Compute root mean squared error.
    rmse = float(np.sqrt(np.mean(difference**2)))

    # Return all errors in a dictionary.
    return {
        "relative_l2_error": rel_l2,
        "mae": mae,
        "rmse": rmse,
    }


def compare_with_snapshot_matrix(snapshot_matrix_path: Path, prediction: np.ndarray, row_index: int) -> dict[str, float]:
    # If the original matrix is missing, return NaN errors and explain later in output.
    if not snapshot_matrix_path.exists():
        return {
            "relative_l2_error": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
        }

    # Load the original snapshot matrix with memory mapping to avoid full RAM loading.
    matrix = np.load(snapshot_matrix_path, mmap_mode="r")

    # Stop early if the requested comparison row does not exist.
    if row_index >= matrix.shape[0]:
        raise IndexError(
            f"Snapshot matrix row {row_index} is out of range for {snapshot_matrix_path}, "
            f"which has {matrix.shape[0]} rows."
        )

    # Load only the requested original row as float32.
    reference = np.asarray(matrix[row_index], dtype=np.float32)

    # Flatten the prediction so it matches the stored snapshot matrix row format.
    prediction_flat = np.asarray(prediction, dtype=np.float32).ravel()

    # If dimensions differ, compare only when they are exactly compatible.
    if reference.shape != prediction_flat.shape:
        raise ValueError(
            f"Prediction shape {prediction_flat.shape} does not match reference row shape {reference.shape} "
            f"from {snapshot_matrix_path}."
        )

    # Compute and return the prediction errors.
    return compute_errors(reference, prediction_flat)


def process_dataset(name: str, paths: dict[str, Path], row_index: int) -> dict[str, object]:
    # Print a clear section title.
    print("\n" + "=" * 80)
    print(f"Predicting and reconstructing: {name}")
    print("=" * 80)

    # Load the feature names used during surrogate training.
    feature_names = load_feature_names(paths["feature_names_path"])

    # Load the example DOE row and numeric input matrix.
    doe_row, x = load_doe_row(paths["doe_path"], feature_names, row_index)

    # Load the trained input scaler.
    scaler = joblib.load(paths["scaler_path"])

    # Load the trained surrogate model.
    surrogate = joblib.load(paths["surrogate_path"])

    # Load the trained PCA/POD model.
    pca = joblib.load(paths["pca_path"])

    # Standardize the DOE input using the saved scaler.
    x_scaled = scaler.transform(x)

    # Predict POD/PCA coefficients from DOE parameters.
    predicted_coefficients = surrogate.predict(x_scaled)

    # Reconstruct the full flattened field using PCA inverse_transform.
    reconstructed = pca.inverse_transform(predicted_coefficients)[0]

    # Convert reconstructed values to float32 for storage.
    reconstructed = np.asarray(reconstructed, dtype=np.float32)

    # Make sure the predictions output folder exists.
    paths["prediction_path"].parent.mkdir(parents=True, exist_ok=True)

    # Save the reconstructed full field as a NumPy array.
    np.save(paths["prediction_path"], reconstructed)

    # Compare the prediction with the original snapshot matrix row if available.
    errors = compare_with_snapshot_matrix(paths["snapshot_matrix_path"], reconstructed, row_index)

    # Print the input parameters used, limited to the trained features.
    print("Input parameters used:")
    for feature_name in feature_names:
        print(f"  {feature_name}: {doe_row[feature_name]}")

    # Print the predicted coefficient shape.
    print(f"Predicted coefficient shape: {predicted_coefficients.shape}")

    # Print the reconstructed field shape.
    print(f"Reconstructed {name} shape: {reconstructed.shape}")

    # Print the output path.
    print(f"Saved reconstructed {name}: {paths['prediction_path']}")

    # Print error metrics.
    print("Errors compared with original snapshot matrix row 0:")
    print(f"  relative L2 error: {errors['relative_l2_error']}")
    print(f"  MAE: {errors['mae']}")
    print(f"  RMSE: {errors['rmse']}")

    # Return a summary row for the final CSV report.
    return {
        "dataset": name,
        "row_index": row_index,
        "predicted_coefficients_shape": str(tuple(predicted_coefficients.shape)),
        "reconstructed_shape": str(tuple(reconstructed.shape)),
        "prediction_path": str(paths["prediction_path"]),
        **errors,
    }


def main() -> None:
    # Store one error-summary row per dataset.
    report_rows: list[dict[str, object]] = []

    # Process geometry and pressure separately.
    for name, paths in DATASETS.items():
        report_rows.append(process_dataset(name, paths, ROW_INDEX))

    # Convert the report rows into a table.
    report = pd.DataFrame(report_rows)

    # Build the error report output path.
    error_report_path = PREDICTIONS_DIR / "prediction_error_row0.csv"

    # Make sure the predictions folder exists.
    error_report_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the error report table.
    report.to_csv(error_report_path, index=False)

    # Print a final table to the terminal.
    print("\n" + "=" * 80)
    print("Prediction error summary")
    print("=" * 80)
    print(report.to_string(index=False))
    print(f"\nSaved error report: {error_report_path}")


# This makes main() run only when this file is executed directly.
if __name__ == "__main__":
    main()
