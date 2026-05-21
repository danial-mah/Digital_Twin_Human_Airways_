# Human Airways Digital Twin: Learning Blog

Welcome to the learning notes for this project.

This blog explains the work step by step, in plain language. The goal is not only to run the code, but to understand what each part of the digital twin pipeline is doing.

## What This Project Builds

This project builds a data-driven digital twin for human airways.

The pipeline is:

```text
Raw dataset
-> snapshot matrices
-> POD/PCA compression
-> surrogate machine learning models
-> predicted geometry and pressure
-> visualization and proxy metrics
```

## Main Idea

The original geometry and pressure fields are very large. Instead of predicting millions of values directly, the project compresses each field into a small set of POD/PCA coefficients.

Then machine learning models learn this mapping:

```text
DOE input parameters -> POD/PCA coefficients
```

After prediction, PCA reconstructs the full field again:

```text
POD/PCA coefficients -> reconstructed geometry or pressure field
```

## Blog Pages

1. Dataset structure and checks
2. Binary snapshots and why float64 matters
3. Snapshot matrices
4. POD/PCA compression
5. Surrogate models
6. Prediction, visualization, and proxy physics

## Important Note

The proxy physics metrics are not CFD. They are simplified indicators for fast dashboard feedback.
