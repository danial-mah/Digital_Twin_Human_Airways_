"""
core/gp.py
===========
Gaussian Process Regression (GPR) surrogate — same interface as core/rbf.py.

Each pressure POD mode is fitted with its own GPR (independent outputs).
scikit-learn's GaussianProcessRegressor handles hyperparameter optimisation
(length-scales, amplitude) automatically via marginal-likelihood maximisation.

Kernel choices
--------------
  rbf               : k(r) = σ² exp(−r²/2ℓ²)          smooth / infinitely differentiable
  matern_32         : k(r) = σ²(1+√3 r/ℓ)exp(−√3 r/ℓ) once differentiable
  matern_52         : k(r) = σ²(1+√5 r/ℓ+5r²/3ℓ²)…   twice differentiable
  rational_quadratic: k(r) = σ²(1+r²/2αℓ²)^−α          mixture of length-scales

For airway flow data, Matérn 5/2 is often a good default — smoother than
Matérn 3/2, more robust to sharp features than the pure RBF kernel.
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, ConstantKernel as C,
)


# ── Kernel factory ─────────────────────────────────────────────────────────────

def _kernel(name: str):
    """Return a sklearn kernel object for the given name."""
    ls_bounds = (1e-3, 1e3)
    amp = C(1.0, constant_value_bounds=(1e-3, 1e3))
    if name == "matern_32":
        return amp * Matern(length_scale=1.0, length_scale_bounds=ls_bounds, nu=1.5)
    if name == "matern_52":
        return amp * Matern(length_scale=1.0, length_scale_bounds=ls_bounds, nu=2.5)
    if name == "rational_quadratic":
        return amp * RationalQuadratic(
            length_scale=1.0, alpha=1.0,
            length_scale_bounds=ls_bounds, alpha_bounds=(1e-3, 1e3),
        )
    # default: RBF (squared exponential / Gaussian)
    return amp * RBF(length_scale=1.0, length_scale_bounds=ls_bounds)


# ── Public API ─────────────────────────────────────────────────────────────────

def build_gp(
    params: np.ndarray,
    targets: np.ndarray,
    kernel_name: str = "matern_52",
    n_restarts: int = 2,
) -> list[GaussianProcessRegressor]:
    """
    Fit one GPR per output column (one per POD mode).

    Parameters
    ----------
    params      : (n, d)  normalised input parameter matrix
    targets     : (n, k)  POD score matrix
    kernel_name : kernel type string (see module docstring)
    n_restarts  : number of random restarts for hyperparameter optimisation

    Returns
    -------
    List of k fitted GaussianProcessRegressor objects (one per mode).
    """
    models = []
    k = targets.shape[1]
    for j in range(k):
        gpr = GaussianProcessRegressor(
            kernel=_kernel(kernel_name),
            n_restarts_optimizer=n_restarts,
            normalize_y=True,
            alpha=1e-8,          # tiny diagonal regularisation for numerical stability
        )
        gpr.fit(params, targets[:, j])
        models.append(gpr)
    return models


def predict(models: list[GaussianProcessRegressor], new_params: np.ndarray) -> np.ndarray:
    """
    Predict POD scores at new parameter points.

    Parameters
    ----------
    models     : list of k fitted GPR models
    new_params : (m, d) new parameter vectors

    Returns
    -------
    (m, k) predicted POD scores
    """
    preds = np.column_stack([m.predict(new_params) for m in models])
    return preds


def kfold_errors(
    params: np.ndarray,
    targets: np.ndarray,
    k: int = 5,
    kernel_name: str = "matern_52",
    n_restarts: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """
    K-Fold cross-validation — same signature as core.rbf.kfold_errors.

    n_restarts is kept at 1 during CV to keep runtime manageable.

    Returns
    -------
    fold_errors : (k,) mean ‖predicted − actual‖ per fold
    """
    n = len(params)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    folds = np.array_split(indices, k)
    fold_errors = np.empty(k)

    for fold_idx, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(k) if j != fold_idx])
        models_k = build_gp(params[train_idx], targets[train_idx],
                            kernel_name=kernel_name, n_restarts=n_restarts)
        pred = predict(models_k, params[test_idx])
        fold_errors[fold_idx] = float(
            np.linalg.norm(pred - targets[test_idx]) / len(test_idx)
        )

    return fold_errors
