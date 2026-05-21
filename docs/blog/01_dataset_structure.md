# 01 - Dataset Structure And Checks

The first step was to organize the project into a clean structure.

## Main Folders

```text
data/
  geometry/
  pressure/

outputs/
  geometry/
  pressure/
  models/
  figures/
  predictions/

scripts/
app/
src/
```

## Why This Structure Helps

The `data/` folder keeps original input data separate from generated results.

The `outputs/` folder stores anything created by the pipeline:

- snapshot matrices
- PCA models
- POD coefficients
- surrogate models
- plots
- predictions

The `scripts/` folder contains numbered execution steps. This makes the workflow easy to follow:

```bash
python scripts/01_check_dataset.py
python scripts/02_read_one_snapshot.py
python scripts/03_build_snapshot_matrices.py
```

The `src/airways/` folder contains reusable helper code. This keeps repeated logic out of the numbered scripts.

## Dataset Checker

The script:

```text
scripts/01_check_dataset.py
```

checks whether each dataset has:

- `Snapshots/`
- `doe.csv`
- `points.bin`
- `settings.json`
- `outputDefinition.json`

It also prints DOE columns, JSON contents, snapshot counts, and readiness status.

## Why This Matters

Before doing machine learning, we must confirm that the raw data is present and readable. If a file is missing, later scripts would fail in a more confusing way.
