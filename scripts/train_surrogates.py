"""Train PCA/POD compression and surrogate models for airway datasets.

This script reads geometry and/or pressure snapshots, compresses the high
dimensional fields with PCA, trains a random forest to predict PCA coordinates
from DOE parameters, and saves the trained model to outputs/models/.
"""

# This import enables modern type-hint behavior.
from __future__ import annotations

# argparse lets this file accept command-line options such as --dataset pressure.
import argparse

# pathlib is used for clean filesystem paths.
from pathlib import Path

# joblib saves and loads Python machine-learning objects efficiently.
import joblib

# matplotlib creates the PCA explained-variance figure.
import matplotlib.pyplot as plt

# NumPy provides numeric arrays and mathematical operations.
import numpy as np

# PCA is the POD/PCA compression method used to reduce snapshot size.
from sklearn.decomposition import PCA

# RandomForestRegressor learns the surrogate mapping from DOE inputs to PCA scores.
from sklearn.ensemble import RandomForestRegressor

# These metrics evaluate prediction quality on the test split.
from sklearn.metrics import mean_absolute_error, r2_score

# train_test_split separates data into training and validation subsets.
from sklearn.model_selection import train_test_split

# Pipeline chains preprocessing and regression into one model object.
from sklearn.pipeline import Pipeline

# StandardScaler normalizes DOE input features before the random forest.
from sklearn.preprocessing import StandardScaler

# Import shared dataset-loading utilities from scripts/airways_io.py.
from airways_io import (
    PROJECT_ROOT,
    feature_columns,
    load_dataset,
    load_doe,
    load_points,
    load_snapshot_matrix,
    model_path,
    selection_indices,
)


def parse_args() -> argparse.Namespace:
    # Create the command-line parser and describe the script purpose.
    parser = argparse.ArgumentParser(description="Train POD/PCA and surrogate models for airway datasets.")

    # Choose whether to train geometry, pressure, or both datasets.
    parser.add_argument("--dataset", choices=["geometry", "pressure", "both"], default="both")

    # Set the maximum number of PCA modes to keep.
    parser.add_argument("--components", type=int, default=20, help="Maximum number of PCA/POD modes.")

    # Optionally train on a named airway region instead of the full mesh.
    parser.add_argument("--selection", default=None, help="Optional named airway region, for example glotis.")

    # Choose the fraction of samples reserved for validation.
    parser.add_argument("--test-size", type=float, default=0.2)

    # Set the random seed so repeated runs are reproducible.
    parser.add_argument("--random-state", type=int, default=42)

    # Choose how many trees the random forest will use.
    parser.add_argument("--n-estimators", type=int, default=300)

    # Optionally use only the first N snapshots for a faster smoke test.
    parser.add_argument("--max-snapshots", type=int, default=None)

    # Parse and return all command-line arguments.
    return parser.parse_args()


def train_one(name: str, args: argparse.Namespace) -> dict:
    # Load paths and metadata for the selected dataset.
    dataset = load_dataset(name)

    # Read the DOE table and normalize its snapshot column name.
    doe = load_doe(dataset)

    # If requested, keep only the first N rows for a quick test run.
    if args.max_snapshots:
        doe = doe.head(args.max_snapshots).copy()

    # Select numeric DOE columns to use as machine-learning input features.
    cols = feature_columns(doe)

    # Convert the DOE input table into a numeric NumPy matrix.
    x = doe[cols].to_numpy(dtype=np.float64)

    # Load all referenced snapshots into a samples x flattened-field matrix.
    y = load_snapshot_matrix(dataset, doe["snapshot"], selection=args.selection)

    # Split inputs and fields into training and testing groups.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # PCA components cannot exceed the number of training samples or features.
    n_components = min(args.components, y_train.shape[0], y_train.shape[1])

    # Stop if too little data exists to train even one component.
    if n_components < 1:
        raise ValueError("At least two snapshots are required to train a PCA surrogate.")

    # Create the PCA compressor; randomized SVD is faster for large snapshot matrices.
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=args.random_state)

    # Fit PCA on training fields and convert those fields into latent PCA scores.
    z_train = pca.fit_transform(y_train)

    # Convert validation fields into the same PCA latent space.
    z_test = pca.transform(y_test)

    # Build the surrogate model as scaling followed by random forest regression.
    surrogate = Pipeline(
        steps=[
            # Scale input parameters so features with large units do not dominate.
            ("scale", StandardScaler()),
            (
                # The regressor predicts PCA scores from DOE input parameters.
                "regressor",
                RandomForestRegressor(
                    n_estimators=args.n_estimators,
                    random_state=args.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Train the surrogate model on DOE inputs and PCA scores.
    surrogate.fit(x_train, z_train)

    # Predict PCA scores for validation DOE inputs.
    z_pred = surrogate.predict(x_test)

    # Convert predicted PCA scores back into full field values.
    y_pred = pca.inverse_transform(z_pred)

    # Store useful training and validation numbers in a dictionary.
    metrics = {
        "dataset": name,
        "selection": args.selection,
        "samples": int(y.shape[0]),
        "features": int(y.shape[1]),
        "pca_components": int(n_components),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "latent_r2": float(r2_score(z_test, z_pred, multioutput="variance_weighted")),
        "field_mae": float(mean_absolute_error(y_test, y_pred)),
    }

    # Convert the optional named region into node indices for visualization points.
    indices = selection_indices(dataset, args.selection)

    # Load mesh coordinates for the full mesh or selected region.
    points = load_points(dataset, indices=indices)

    # Pack everything needed for prediction and visualization into one object.
    model = {
        "dataset": name,
        "field_name": dataset.field_name,
        "unit": dataset.unit,
        "components_per_node": dataset.components,
        "selection": args.selection,
        "feature_columns": cols,
        "feature_defaults": doe[cols].median().to_dict(),
        "feature_min": doe[cols].min().to_dict(),
        "feature_max": doe[cols].max().to_dict(),
        "pca": pca,
        "surrogate": surrogate,
        "points": points,
        "metrics": metrics,
    }

    # Build the output path for this trained model file.
    out_path = model_path(name)

    # Create outputs/models if it does not already exist.
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the model dictionary to disk.
    joblib.dump(model, out_path)

    # Point to the folder where figures should be saved.
    figures_dir = PROJECT_ROOT / "outputs" / "figures"

    # Create outputs/figures if it does not already exist.
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Build the filename for the explained-variance plot.
    fig_path = figures_dir / f"{name}_explained_variance.png"

    # Create a new matplotlib figure.
    plt.figure(figsize=(7, 4))

    # Plot cumulative PCA energy so we can see how much information modes keep.
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")

    # Label the horizontal axis.
    plt.xlabel("PCA components")

    # Label the vertical axis.
    plt.ylabel("Cumulative explained variance")

    # Add a title naming the dataset.
    plt.title(f"{name.title()} POD/PCA energy")

    # Add a light grid to make the curve easier to read.
    plt.grid(True, alpha=0.3)

    # Adjust spacing so labels do not get cut off.
    plt.tight_layout()

    # Save the figure as a PNG image.
    plt.savefig(fig_path, dpi=160)

    # Close the figure to free memory.
    plt.close()

    # Return metrics so main() can print them.
    return metrics


def main() -> None:
    # Read all command-line arguments.
    args = parse_args()

    # Expand "both" into the two dataset names; otherwise train only one dataset.
    names = ["geometry", "pressure"] if args.dataset == "both" else [args.dataset]

    # Train each requested dataset.
    for name in names:
        # Train one dataset and collect its metrics.
        metrics = train_one(name, args)

        # Print where the model was saved.
        print(f"Saved {name} model to {model_path(name)}")

        # Print every metric on its own line.
        for key, value in metrics.items():
            print(f"  {key}: {value}")


# This makes main() run only when the file is executed as a script.
if __name__ == "__main__":
    main()
