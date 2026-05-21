# 04 - POD/PCA Compression

The script:

```text
scripts/04_pod_pca_compression.py
```

compresses the large geometry and pressure matrices.

## Why Compression Is Needed

Each full field contains millions of numbers.

Predicting millions of values directly would be difficult and inefficient.

POD/PCA finds the most important patterns in the data and represents each snapshot using a small number of coefficients.

## Simple Explanation

Instead of saying:

```text
snapshot = millions of field values
```

we say:

```text
snapshot = 20 important coefficients
```

Those coefficients are much easier for machine learning models to predict.

## What The Script Saves

For geometry:

```text
outputs/models/geometry_pca_model.joblib
outputs/geometry/geometry_mean.npy
outputs/geometry/geometry_pod_coefficients.npy
outputs/geometry/geometry_explained_variance.csv
```

For pressure:

```text
outputs/models/pressure_pca_model.joblib
outputs/pressure/pressure_mean.npy
outputs/pressure/pressure_pod_coefficients.npy
outputs/pressure/pressure_explained_variance.csv
```

## Explained Variance

Explained variance tells us how much information is captured by the PCA modes.

In this project, the first 20 modes capture most of the variation in both geometry and pressure.

## Key Lesson

POD/PCA is the bridge between large scientific fields and small machine-learning targets.
