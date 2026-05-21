# 05 - Surrogate Models

The script:

```text
scripts/05_train_surrogate_models.py
```

trains machine learning models.

## What The Surrogate Learns

The model learns:

```text
DOE input parameters -> POD/PCA coefficients
```

It does not predict full fields directly.

## Why This Is Useful

The DOE parameters are small input variables, such as airway geometry parameters and pressure-related parameters.

The POD/PCA coefficients are compressed representations of full simulation fields.

This makes the machine learning problem much smaller.

## Models Used

The script compares:

- Ridge regression
- Random Forest regression
- Gaussian Process regression when feasible

## How The Best Model Is Chosen

Each model is evaluated using:

- R2 score
- MAE
- RMSE

The model with the best R2 score is saved.

## Saved Files

```text
outputs/models/geometry_best_surrogate.joblib
outputs/models/pressure_best_surrogate.joblib
outputs/models/geometry_input_scaler.joblib
outputs/models/pressure_input_scaler.joblib
outputs/models/geometry_feature_names.txt
outputs/models/pressure_feature_names.txt
```

The scaler and feature-name files are important because prediction must use the same input order and scaling as training.
