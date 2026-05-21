"""Apply POD/PCA compression to geometry and pressure snapshot matrices.

POD/PCA reduces each full airway field into a small number of coefficients.
Instead of storing or predicting millions of values directly, we represent each
snapshot using about 20 numbers that describe the strongest patterns of change.

This script is memory-conscious: it loads .npy matrices with memory mapping and
uses scikit-learn IncrementalPCA so rows can be processed in batches.
"""

# This import enables modern type-hint behavior.
from __future__ import annotations

# argparse lets us optionally control the batch size from the command line.
import argparse

# Path gives clean filesystem path handling.
from pathlib import Path

# joblib saves the fitted PCA model to disk.
import joblib

# matplotlib creates explained-variance plots.
import matplotlib.pyplot as plt

# NumPy loads large .npy matrices and saves arrays.
import numpy as np

# pandas writes explained-variance values to CSV.
import pandas as pd

# IncrementalPCA is a scikit-learn PCA implementation designed for large data.
from sklearn.decomposition import IncrementalPCA


# PROJECT_ROOT points to the main project folder, one level above scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# OUTPUT_ROOT points to the generated output folder.
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# MODELS_DIR is where fitted PCA models will be saved.
MODELS_DIR = OUTPUT_ROOT / "models"

# FIGURES_DIR is where explained-variance plots will be saved.
FIGURES_DIR = OUTPUT_ROOT / "figures"

# DATASETS describes the input and output paths for each dataset.
DATASETS = {
    "geometry": {
        "matrix_path": OUTPUT_ROOT / "geometry" / "geometry_snapshot_matrix.npy",
        "mean_path": OUTPUT_ROOT / "geometry" / "geometry_mean.npy",
        "coefficients_path": OUTPUT_ROOT / "geometry" / "geometry_pod_coefficients.npy",
        "variance_csv_path": OUTPUT_ROOT / "geometry" / "geometry_explained_variance.csv",
        "model_path": MODELS_DIR / "geometry_pca_model.joblib",
        "figure_path": FIGURES_DIR / "geometry_explained_variance.png",
    },
    "pressure": {
        "matrix_path": OUTPUT_ROOT / "pressure" / "pressure_snapshot_matrix.npy",
        "mean_path": OUTPUT_ROOT / "pressure" / "pressure_mean.npy",
        "coefficients_path": OUTPUT_ROOT / "pressure" / "pressure_pod_coefficients.npy",
        "variance_csv_path": OUTPUT_ROOT / "pressure" / "pressure_explained_variance.csv",
        "model_path": MODELS_DIR / "pressure_pca_model.joblib",
        "figure_path": FIGURES_DIR / "pressure_explained_variance.png",
    },
}


def parse_args() -> argparse.Namespace:
    # Create a command-line parser for optional runtime settings.
    parser = argparse.ArgumentParser(description="Apply POD/PCA compression to airway snapshot matrices.")

    # Let the user choose how many snapshots are processed per IncrementalPCA batch.
    parser.add_argument("--batch-size", type=int, default=25, help="Number of snapshots per PCA batch.")

    # Return the parsed command-line arguments.
    return parser.parse_args()


def batch_slices(n_rows: int, batch_size: int, min_batch_size: int) -> list[slice]:
    # Make sure the requested batch size is at least the number of PCA components.
    batch_size = max(batch_size, min_batch_size)

    # Store slices that select row batches from the snapshot matrix.
    slices: list[slice] = []

    # Start at the first row.
    start = 0

    # Continue until all rows are assigned to a batch.
    while start < n_rows:
        # Compute the normal end row for this batch.
        end = min(start + batch_size, n_rows)

        # If the final batch would be too small, merge it into the previous batch.
        if n_rows - end > 0 and n_rows - end < min_batch_size:
            end = n_rows

        # Add this batch slice.
        slices.append(slice(start, end))

        # Move to the next batch.
        start = end

    # Return all batch slices.
    return slices


def as_float32(batch: np.ndarray) -> np.ndarray:
    # Convert a matrix batch to float32 only when it is not already float32.
    return batch.astype(np.float32, copy=False)


def fit_incremental_pca(matrix: np.ndarray, n_components: int, batch_size: int) -> IncrementalPCA:
    # Create the IncrementalPCA model with the requested number of POD/PCA modes.
    pca = IncrementalPCA(n_components=n_components)

    # Create row slices for memory-conscious training batches.
    slices = batch_slices(matrix.shape[0], batch_size, n_components)

    # Fit the PCA model one row batch at a time.
    for batch_index, row_slice in enumerate(slices, start=1):
        # Read only this batch from the memory-mapped matrix.
        batch = as_float32(matrix[row_slice])

        # Print progress so the user can see fitting is active.
        print(f"  PCA fit batch {batch_index}/{len(slices)} with rows {row_slice.start}:{row_slice.stop}")

        # Update the PCA model using this batch.
        pca.partial_fit(batch)

    # Return the fitted PCA model.
    return pca


def transform_in_batches(
    pca: IncrementalPCA,
    matrix: np.ndarray,
    coefficients_path: Path,
    batch_size: int,
) -> np.ndarray:
    # Read the number of snapshots from the matrix shape.
    n_snapshots = matrix.shape[0]

    # Read the number of PCA modes from the fitted model.
    n_components = pca.n_components_

    # Create a memory-mapped .npy file for POD/PCA coefficients.
    coefficients = np.lib.format.open_memmap(
        coefficients_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_snapshots, n_components),
    )

    # Create row slices for transformation batches.
    slices = batch_slices(n_snapshots, batch_size, n_components)

    # Transform each batch from full-field space into PCA coefficient space.
    for batch_index, row_slice in enumerate(slices, start=1):
        # Read only this batch from the snapshot matrix.
        batch = as_float32(matrix[row_slice])

        # Print progress so the user knows transformation is active.
        print(f"  PCA transform batch {batch_index}/{len(slices)} with rows {row_slice.start}:{row_slice.stop}")

        # Transform the full fields into small POD/PCA coefficient vectors.
        coefficients[row_slice, :] = pca.transform(batch).astype(np.float32)

    # Flush changes so the coefficients are fully written to disk.
    coefficients.flush()

    # Return the memory-mapped coefficient matrix object.
    return coefficients


def save_explained_variance(name: str, pca: IncrementalPCA, csv_path: Path, figure_path: Path) -> None:
    # Store the per-mode explained variance ratio from PCA.
    explained = pca.explained_variance_ratio_

    # Compute cumulative variance so we can see how much information the modes keep together.
    cumulative = np.cumsum(explained)

    # Create a table with mode numbers and variance values.
    table = pd.DataFrame(
        {
            "mode": np.arange(1, len(explained) + 1),
            "explained_variance_ratio": explained,
            "cumulative_explained_variance": cumulative,
        }
    )

    # Save the explained-variance table as CSV.
    table.to_csv(csv_path, index=False)

    # Create a figure for the cumulative explained variance curve.
    plt.figure(figsize=(7, 4))

    # Plot cumulative explained variance by PCA mode.
    plt.plot(table["mode"], table["cumulative_explained_variance"], marker="o")

    # Label the horizontal axis.
    plt.xlabel("POD/PCA mode")

    # Label the vertical axis.
    plt.ylabel("Cumulative explained variance")

    # Add a dataset-specific title.
    plt.title(f"{name.title()} POD/PCA explained variance")

    # Add a light grid for readability.
    plt.grid(True, alpha=0.3)

    # Keep the layout compact so labels fit.
    plt.tight_layout()

    # Save the plot as a PNG image.
    plt.savefig(figure_path, dpi=160)

    # Close the figure to free memory.
    plt.close()


def process_dataset(name: str, paths: dict[str, Path], batch_size: int) -> dict[str, object]:
    # Print a clear section title for this dataset.
    print("\n" + "=" * 80)
    print(f"POD/PCA compression for: {name}")
    print("=" * 80)

    # Get the input matrix path for this dataset.
    matrix_path = paths["matrix_path"]

    # Stop early if the snapshot matrix has not been built yet.
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Missing snapshot matrix: {matrix_path}. "
            "Run python scripts/03_build_snapshot_matrices.py first."
        )

    # Load the matrix with memory mapping so NumPy does not immediately load all values into RAM.
    matrix = np.load(matrix_path, mmap_mode="r")

    # Print the original full snapshot matrix shape.
    print(f"Original matrix shape: {matrix.shape}")

    # Print the stored dtype so we know whether conversion is needed.
    print(f"Original matrix dtype: {matrix.dtype}")

    # Choose up to 20 POD/PCA modes, but never more than the number of snapshots.
    n_components = min(20, matrix.shape[0])

    # Print the selected number of components.
    print(f"Using n_components: {n_components}")

    # Fit PCA/POD compression on the full-field snapshot matrix.
    pca = fit_incremental_pca(matrix, n_components=n_components, batch_size=batch_size)

    # Save the fitted PCA model with joblib.
    joblib.dump(pca, paths["model_path"])

    # Save the PCA mean field; this is the average field subtracted before PCA.
    np.save(paths["mean_path"], pca.mean_.astype(np.float32))

    # Transform full snapshots into small POD/PCA coefficient vectors.
    coefficients = transform_in_batches(pca, matrix, paths["coefficients_path"], batch_size=batch_size)

    # Save explained variance values and the variance plot.
    save_explained_variance(name, pca, paths["variance_csv_path"], paths["figure_path"])

    # Store explained variance in local variables for printing.
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # Print the coefficient matrix shape.
    print(f"POD coefficient matrix shape: {coefficients.shape}")

    # Print explained variance for the available modes, up to 20.
    print(f"Explained variance ratio of first {len(explained)} modes:")
    print(explained)

    # Print cumulative explained variance for the available modes.
    print("Cumulative explained variance:")
    print(cumulative)

    # Print where the generated outputs were saved.
    print(f"Saved PCA model: {paths['model_path']}")
    print(f"Saved mean field: {paths['mean_path']}")
    print(f"Saved POD coefficients: {paths['coefficients_path']}")
    print(f"Saved explained variance CSV: {paths['variance_csv_path']}")
    print(f"Saved explained variance plot: {paths['figure_path']}")

    # Return a compact summary for the final report.
    return {
        "matrix_shape": tuple(matrix.shape),
        "coefficient_shape": tuple(coefficients.shape),
        "explained_variance": explained,
        "cumulative_explained_variance": cumulative,
    }


def main() -> None:
    # Parse optional command-line arguments.
    args = parse_args()

    # Create model and figure output folders if needed.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Store summaries for both datasets.
    summaries: dict[str, dict[str, object]] = {}

    # Process geometry first, then pressure.
    for name, paths in DATASETS.items():
        # Make sure each dataset-specific output folder exists.
        paths["mean_path"].parent.mkdir(parents=True, exist_ok=True)

        # Run POD/PCA compression for this dataset.
        summaries[name] = process_dataset(name, paths, batch_size=args.batch_size)

    # Print a short final summary after both datasets are complete.
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    # Print compact shapes for each dataset.
    for name, summary in summaries.items():
        print(f"{name} matrix shape: {summary['matrix_shape']}")
        print(f"{name} POD coefficient matrix shape: {summary['coefficient_shape']}")


# This makes main() run only when this file is executed directly.
if __name__ == "__main__":
    main()
