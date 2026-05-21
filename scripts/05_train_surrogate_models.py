"""Train surrogate models from DOE parameters to POD/PCA coefficients.

The purpose of this script is:

    DOE input parameters -> POD/PCA coefficients

It does not reconstruct full geometry or pressure fields. Reconstruction will
come later by combining predicted POD coefficients with the saved PCA model.
"""

# This import enables modern type-hint behavior.
from __future__ import annotations

# Path gives clean filesystem path handling.
from pathlib import Path

# joblib saves trained models and scalers to disk.
import joblib

# NumPy handles numeric arrays and metric calculations.
import numpy as np

# pandas reads DOE CSV files and saves model comparison tables.
import pandas as pd

# GaussianProcessRegressor is an accurate but potentially expensive surrogate.
from sklearn.gaussian_process import GaussianProcessRegressor

# RBF, ConstantKernel, and WhiteKernel define a smooth Gaussian-process kernel.
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

# These functions compute regression quality on the test set.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# train_test_split creates reproducible train/test datasets.
from sklearn.model_selection import train_test_split

# MultiOutputRegressor lets regressors that predict one output handle many POD coefficients.
from sklearn.multioutput import MultiOutputRegressor

# RandomForestRegressor learns nonlinear mappings from DOE parameters to coefficients.
from sklearn.ensemble import RandomForestRegressor

# Ridge regression is a simple linear baseline model.
from sklearn.linear_model import Ridge

# StandardScaler standardizes input features before model training.
from sklearn.preprocessing import StandardScaler


# PROJECT_ROOT points to the main project folder, one level above scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DATA_ROOT points to data/ where DOE files live.
DATA_ROOT = PROJECT_ROOT / "data"

# OUTPUT_ROOT points to outputs/ where POD coefficients and results live.
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# MODELS_DIR is where trained surrogate models and scalers will be saved.
MODELS_DIR = OUTPUT_ROOT / "models"

# DATASETS stores all input and output paths for geometry and pressure.
DATASETS = {
    "geometry": {
        "doe_path": DATA_ROOT / "geometry" / "doe.csv",
        "coefficients_path": OUTPUT_ROOT / "geometry" / "geometry_pod_coefficients.npy",
        "results_path": OUTPUT_ROOT / "geometry" / "geometry_surrogate_results.csv",
        "best_model_path": MODELS_DIR / "geometry_best_surrogate.joblib",
        "scaler_path": MODELS_DIR / "geometry_input_scaler.joblib",
        "feature_names_path": MODELS_DIR / "geometry_feature_names.txt",
    },
    "pressure": {
        "doe_path": DATA_ROOT / "pressure" / "doe.csv",
        "coefficients_path": OUTPUT_ROOT / "pressure" / "pressure_pod_coefficients.npy",
        "results_path": OUTPUT_ROOT / "pressure" / "pressure_surrogate_results.csv",
        "best_model_path": MODELS_DIR / "pressure_best_surrogate.joblib",
        "scaler_path": MODELS_DIR / "pressure_input_scaler.joblib",
        "feature_names_path": MODELS_DIR / "pressure_feature_names.txt",
    },
}


def load_doe_features(doe_path: Path) -> tuple[pd.DataFrame, list[str]]:
    # Read the DOE CSV file; sep=None lets pandas detect comma or semicolon separators.
    doe = pd.read_csv(doe_path, sep=None, engine="python")

    # Keep only numeric columns, automatically removing text columns like snapshot names.
    numeric_doe = doe.select_dtypes(include=[np.number]).copy()

    # Remove a column named "points" if it exists, because it is not an input parameter.
    if "points" in numeric_doe.columns:
        numeric_doe = numeric_doe.drop(columns=["points"])

    # Store the remaining numeric feature names in order.
    feature_names = list(numeric_doe.columns)

    # Stop early if no usable numeric input features remain.
    if not feature_names:
        raise ValueError(f"No numeric input features found in {doe_path}.")

    # Return the numeric feature table and its column names.
    return numeric_doe, feature_names


def load_coefficients(coefficients_path: Path) -> np.ndarray:
    # Load POD coefficients with memory mapping to avoid unnecessary RAM usage.
    coefficients = np.load(coefficients_path, mmap_mode="r")

    # Convert to a normal float32 array because model training needs random row access.
    return np.asarray(coefficients, dtype=np.float32)


def align_rows(x_table: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Find the largest row count that both DOE and coefficient arrays can support.
    n_rows = min(len(x_table), y.shape[0])

    # Warn if the DOE row count and coefficient row count do not match exactly.
    if len(x_table) != y.shape[0]:
        print(
            f"WARNING: DOE rows ({len(x_table)}) and POD coefficient rows ({y.shape[0]}) differ; "
            f"using first {n_rows} rows."
        )

    # Convert the aligned DOE features into a NumPy input matrix.
    x_aligned = x_table.iloc[:n_rows].to_numpy(dtype=np.float32)

    # Keep the matching coefficient rows as the output matrix.
    y_aligned = np.asarray(y[:n_rows], dtype=np.float32)

    # Return aligned inputs and outputs.
    return x_aligned, y_aligned


def create_models(n_train: int, n_outputs: int) -> list[tuple[str, object, str]]:
    # Start with models that are always computationally reasonable.
    models: list[tuple[str, object, str]] = [
        # Ridge is a fast linear baseline.
        ("Ridge", Ridge(alpha=1.0), "trained"),

        # Random Forest captures nonlinear effects and supports multi-output regression directly.
        (
            "RandomForestRegressor",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),
            "trained",
        ),
    ]

    # Gaussian Process can be expensive because training scales poorly with sample count.
    gpr_feasible = n_train <= 250 and n_outputs <= 50

    # Add Gaussian Process only when the training problem is small enough.
    if gpr_feasible:
        # Build a smooth kernel with noise for numerical stability.
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)

        # Wrap GPR in MultiOutputRegressor so one GPR is trained for each POD coefficient.
        models.append(
            (
                "GaussianProcessRegressor",
                MultiOutputRegressor(
                    GaussianProcessRegressor(
                        kernel=kernel,
                        normalize_y=True,
                        random_state=42,
                        n_restarts_optimizer=0,
                    )
                ),
                "trained",
            )
        )

    # If GPR is too expensive, add a placeholder status instead of training it.
    else:
        models.append(
            (
                "GaussianProcessRegressor",
                None,
                f"skipped: not computationally feasible for n_train={n_train}, n_outputs={n_outputs}",
            )
        )

    # Return the model list.
    return models


def evaluate_model(model: object, x_test: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    # Predict POD coefficients for the test DOE inputs.
    y_pred = model.predict(x_test)

    # Compute R2 score across all POD coefficient outputs.
    r2 = float(r2_score(y_test, y_pred, multioutput="variance_weighted"))

    # Compute mean absolute error across all coefficients.
    mae = float(mean_absolute_error(y_test, y_pred))

    # Compute root mean squared error across all coefficients.
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # Return predictions and metrics.
    return y_pred, r2, mae, rmse


def save_feature_names(feature_names: list[str], path: Path) -> None:
    # Join feature names with newline characters.
    text = "\n".join(feature_names)

    # Add a final newline for a clean text file.
    if text:
        text += "\n"

    # Save the feature names to disk.
    path.write_text(text, encoding="utf-8")


def train_dataset(name: str, paths: dict[str, Path]) -> pd.DataFrame:
    # Print a clear section title for this dataset.
    print("\n" + "=" * 80)
    print(f"Training surrogate models for: {name}")
    print("=" * 80)

    # Stop early if the POD coefficients file has not been created yet.
    if not paths["coefficients_path"].exists():
        raise FileNotFoundError(
            f"Missing POD coefficients: {paths['coefficients_path']}. "
            "Run python scripts/04_pod_pca_compression.py first."
        )

    # Load numeric DOE features and feature names.
    x_table, feature_names = load_doe_features(paths["doe_path"])

    # Load POD coefficient matrix.
    y = load_coefficients(paths["coefficients_path"])

    # Align DOE rows with coefficient rows.
    x, y = align_rows(x_table, y)

    # Print the learning problem shapes.
    print(f"Input feature matrix X shape: {x.shape}")
    print(f"POD coefficient target y shape: {y.shape}")
    print(f"Feature names: {feature_names}")

    # Split inputs and targets into training and testing sets.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Create a scaler for DOE input parameters.
    scaler = StandardScaler()

    # Fit the scaler on training inputs and transform training inputs.
    x_train_scaled = scaler.fit_transform(x_train)

    # Transform test inputs using the same scaler.
    x_test_scaled = scaler.transform(x_test)

    # Build the list of surrogate models.
    models = create_models(n_train=x_train_scaled.shape[0], n_outputs=y_train.shape[1])

    # Store one result dictionary per model.
    results: list[dict[str, object]] = []

    # Track the best model by highest R2 score.
    best_model_name: str | None = None
    best_model: object | None = None
    best_r2 = -np.inf

    # Train and evaluate each model.
    for model_name, model, status in models:
        # If the model was marked as skipped, record that in the results table.
        if model is None:
            results.append(
                {
                    "model": model_name,
                    "status": status,
                    "r2": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                }
            )
            print(f"{model_name}: {status}")
            continue

        # Print progress before fitting.
        print(f"Training {model_name}...")

        # Train the model from standardized DOE inputs to POD coefficients.
        model.fit(x_train_scaled, y_train)

        # Evaluate the trained model on the test split.
        _, r2, mae, rmse = evaluate_model(model, x_test_scaled, y_test)

        # Store the model comparison metrics.
        results.append(
            {
                "model": model_name,
                "status": status,
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
            }
        )

        # Print this model's metrics.
        print(f"{model_name}: R2={r2:.6f}, MAE={mae:.6g}, RMSE={rmse:.6g}")

        # Update the best model if this model has the highest R2 so far.
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = model_name
            best_model = model

    # Stop if every model was skipped or failed to produce a trained model.
    if best_model is None or best_model_name is None:
        raise RuntimeError(f"No surrogate model was trained successfully for {name}.")

    # Convert result dictionaries into a DataFrame.
    results_table = pd.DataFrame(results)

    # Sort trained models above skipped models, and highest R2 first.
    results_table = results_table.sort_values(by="r2", ascending=False, na_position="last")

    # Save the model comparison table.
    results_table.to_csv(paths["results_path"], index=False)

    # Save the best trained surrogate model.
    joblib.dump(best_model, paths["best_model_path"])

    # Save the input scaler used before the surrogate model.
    joblib.dump(scaler, paths["scaler_path"])

    # Save the feature names used by the model.
    save_feature_names(feature_names, paths["feature_names_path"])

    # Print a clear comparison table.
    print("\nModel comparison:")
    print(results_table.to_string(index=False))

    # Print where the selected model and supporting files were saved.
    print(f"\nBest model by R2: {best_model_name} with R2={best_r2:.6f}")
    print(f"Saved results: {paths['results_path']}")
    print(f"Saved best surrogate: {paths['best_model_path']}")
    print(f"Saved input scaler: {paths['scaler_path']}")
    print(f"Saved feature names: {paths['feature_names_path']}")

    # Return the results table for final summary printing.
    return results_table


def main() -> None:
    # Create the models folder if it does not already exist.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Store result tables for both datasets.
    all_results: dict[str, pd.DataFrame] = {}

    # Train geometry and pressure surrogate models separately.
    for name, paths in DATASETS.items():
        # Make sure the dataset-specific output folder exists.
        paths["results_path"].parent.mkdir(parents=True, exist_ok=True)

        # Train all surrogate models for this dataset.
        all_results[name] = train_dataset(name, paths)

    # Print a final compact summary.
    print("\n" + "=" * 80)
    print("Final surrogate model summary")
    print("=" * 80)

    # Print the best row from each dataset result table.
    for name, results_table in all_results.items():
        best_row = results_table.dropna(subset=["r2"]).iloc[0]
        print(
            f"{name}: best={best_row['model']}, "
            f"R2={best_row['r2']:.6f}, "
            f"MAE={best_row['mae']:.6g}, "
            f"RMSE={best_row['rmse']:.6g}"
        )


# This makes main() run only when this file is executed directly.
if __name__ == "__main__":
    main()
