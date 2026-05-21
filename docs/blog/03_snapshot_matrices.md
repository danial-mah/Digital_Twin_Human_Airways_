# 03 - Building Snapshot Matrices

The script:

```text
scripts/03_build_snapshot_matrices.py
```

builds large machine-learning matrices from the raw snapshot files.

## What Is A Snapshot?

One snapshot is one simulation result.

For geometry, one snapshot contains many x, y, z coordinates.

For pressure, one snapshot contains scalar pressure values.

## What Is A Snapshot Matrix?

A snapshot matrix stacks all snapshots into a table:

```text
rows = snapshots
columns = field values
```

Example:

```text
geometry matrix shape = (100, 6407718)
pressure matrix shape = (100, 2135906)
```

This means:

- 100 snapshots
- millions of values per snapshot

## Correct Reading Logic

The script now reads each file like this:

```python
raw_values = np.fromfile(path, dtype=np.float64)
cleaned_values = raw_values[1:]
snapshot = cleaned_values.astype(np.float32)
```

## Why Save Matrices?

PCA and machine learning models need data in a consistent matrix form.

The output files are:

```text
outputs/geometry/geometry_snapshot_matrix.npy
outputs/pressure/pressure_snapshot_matrix.npy
```

The script also saves the accepted snapshot names, so we know which row belongs to which snapshot file.
