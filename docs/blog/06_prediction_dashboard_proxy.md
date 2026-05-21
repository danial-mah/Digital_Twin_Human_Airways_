# 06 - Prediction, Dashboard, And Proxy Physics

After PCA and surrogate training, the project can predict new airway fields.

## Prediction Pipeline

The prediction workflow is:

```text
DOE input
-> input scaler
-> surrogate model
-> predicted POD/PCA coefficients
-> PCA inverse_transform
-> reconstructed geometry and pressure
```

This is implemented in:

```text
scripts/06_predict_and_reconstruct.py
```

## VTP Export

The script:

```text
scripts/07_export_prediction_to_vtk.py
```

exports predicted fields to `.vtp`, which can be opened in ParaView.

## Proxy Physics

The script:

```text
scripts/08_compute_proxy_physics.py
```

computes simplified metrics such as:

- pressure minimum
- pressure maximum
- pressure mean
- pressure range
- mean radius proxy
- constriction index
- flow capacity proxy
- resistance proxy
- pressure drop proxy

## These Are Not CFD Results

The proxy metrics are simplified indicators.

They are useful for a dashboard, but they do not replace real fluid simulation.

## Streamlit Dashboard

The main dashboard is:

```text
app/airways_digital_twin_app.py
```

It lets users change input parameters, predict fields, view pressure statistics, inspect proxy metrics, save predictions, and export VTP files.
