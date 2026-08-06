"""
core/rbf.py
===========
Radial Basis Function (RBF) surrogate model for parameter-space inference.

How RBF inference works
-----------------------
Given training pairs  (pᵢ, yᵢ)  where pᵢ ∈ ℝᵈ are DOE parameter vectors
and yᵢ ∈ ℝᵏ are POD modal scores, the RBF interpolant is:

    f(x) = Σᵢ wᵢ φ(‖x − pᵢ‖)  +  polynomial tail

where φ is the radial basis function.  Weights wᵢ are found by solving
the linear system  Φ w = y.

Common kernels
--------------
  thin_plate_spline : φ(r) = r² log(r)    — smooth, widely used
  multiquadric      : φ(r) = √(r² + ε²)  — good for scattered data
  gaussian          : φ(r) = exp(−r²/ε²) — compact support

Usage in this project
---------------------
  1. Build RBF from DOE params (100 × 26) → pressure POD scores (100 × k)
  2. At inference time: supply new 26-dim parameter vector → get k scores
  3. Reconstruct pressure field: mean + modes @ predicted_scores
"""

import numpy as np
from scipy.interpolate import RBFInterpolator


def build_rbf(
    params: np.ndarray,
    targets: np.ndarray,
    kernel: str = "thin_plate_spline",
    smoothing: float = 0.0,
) -> RBFInterpolator:
    """
    Fit an RBF interpolator.

    Parameters
    ----------
    params    : (n, d)  input parameter matrix (DOE values)
    targets   : (n, k)  target matrix (POD scores)
    kernel    : RBF kernel name (scipy convention)
    smoothing : 0.0 = exact interpolation; >0 adds regularisation

    Returns
    -------
    Fitted RBFInterpolator instance
    """
    return RBFInterpolator(params, targets, kernel=kernel, smoothing=smoothing)


def predict(rbf: RBFInterpolator, new_params: np.ndarray) -> np.ndarray:
    """
    Predict POD scores at new parameter points.

    Parameters
    ----------
    rbf        : fitted RBFInterpolator
    new_params : (m, d) new parameter vectors

    Returns
    -------
    (m, k) predicted POD scores
    """
    return rbf(new_params)


def kfold_errors(
    params: np.ndarray,
    targets: np.ndarray,
    k: int = 5,
    kernel: str = "thin_plate_spline",
    seed: int = 42,
) -> np.ndarray:
    """
    K-Fold cross-validation: mean per-sample prediction error for each fold.

    Unlike LOO which rebuilds n models, K-Fold rebuilds only k models,
    making it much faster while still giving a robust generalisation estimate.

    Parameters
    ----------
    params  : (n, d) input parameter matrix
    targets : (n, k) target matrix (POD scores)
    k       : number of folds (default 5)
    kernel  : RBF kernel
    seed    : random seed for fold shuffling

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
        rbf_k = RBFInterpolator(params[train_idx], targets[train_idx], kernel=kernel)
        pred = rbf_k(params[test_idx])
        fold_errors[fold_idx] = float(
            np.linalg.norm(pred - targets[test_idx]) / len(test_idx)
        )

    return fold_errors


def loo_errors(
    params: np.ndarray,
    targets: np.ndarray,
    kernel: str = "thin_plate_spline",
) -> np.ndarray:
    """
    Leave-One-Out cross-validation: absolute prediction error for each sample.
    Used to estimate RBF accuracy without a separate test set.

    Returns
    -------
    (n,) array of ‖predicted − actual‖ errors
    """
    n = len(params)
    errors = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rbf_loo = RBFInterpolator(params[mask], targets[mask], kernel=kernel)
        pred = rbf_loo(params[[i]])
        errors[i] = float(np.linalg.norm(pred - targets[[i]]))
    return errors
