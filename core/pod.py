"""
core/pod.py
===========
Proper Orthogonal Decomposition (POD) via truncated SVD.

Mathematical background
-----------------------
Given a snapshot matrix  X  of shape (n_snapshots, n_features):

1. Centre: X_c = X - mean(X, axis=0)

2. Economy SVD:
       X_c = U · diag(σ) · V^T
   where  U  (n × k),  σ  (k,),  V  (n_features × k),  k = min(n, p)

3. Spatial modes  = columns of V  → describe where variation occurs in space
   Modal scores   = U · diag(σ)   → describe how each snapshot uses each mode
   Energy of mode i = σᵢ² / Σ σⱼ²

The truncated SVD keeps only the first k modes, reducing dimensionality
from n_features to k while retaining most of the total variance.

Relationship to PCA
-------------------
POD on the snapshot matrix is identical to PCA with no normalisation.
The POD modes are the principal directions of maximum variance.
"""

import numpy as np


def compute_pod(X: np.ndarray, n_modes: int | None = None):
    """
    Decompose snapshot matrix X (n_snapshots × n_features) into POD modes.

    Parameters
    ----------
    X       : (n, p) snapshot matrix
    n_modes : truncate to this many modes (default: keep all min(n, p))

    Returns
    -------
    mean    : (p,)   mean snapshot
    modes   : (p, k) spatial modes, columns orthonormal
    scores  : (n, k) modal coefficients per snapshot
    svalues : (k,)   singular values in descending order
    """
    mean = X.mean(axis=0)
    Xc = X - mean  # centre around the mean shape / field

    # Economy SVD: avoids computing the full (p × p) matrix when p >> n
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    if n_modes is not None:
        k = min(n_modes, len(s))
        U, s, Vt = U[:, :k], s[:k], Vt[:k]

    modes  = Vt.T       # (p, k)  — each column is a spatial mode
    scores = U * s      # (n, k)  — each row is a snapshot's projection

    return mean, modes, scores, s


def cumulative_energy(svalues: np.ndarray) -> np.ndarray:
    """
    Fraction of total variance captured by the first k modes (k = 1, 2, …).
    Returns array of shape (K,) with values in [0, 1].
    """
    e2 = svalues ** 2
    return np.cumsum(e2) / e2.sum()


def modes_for_energy(svalues: np.ndarray, threshold: float = 0.99) -> int:
    """
    Minimum number of modes needed to capture `threshold` fraction of energy.
    """
    cum = cumulative_energy(svalues)
    idx = int(np.searchsorted(cum, threshold))
    return min(idx + 1, len(svalues))


def reconstruct(mean: np.ndarray, modes: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Reconstruct a field from its POD expansion.

    field ≈ mean + modes @ coeffs

    Parameters
    ----------
    mean   : (p,) mean field
    modes  : (p, k) spatial modes
    coeffs : (k,) modal coefficients

    Returns
    -------
    (p,) reconstructed field
    """
    return mean + modes @ coeffs


def project(Xc_row: np.ndarray, modes: np.ndarray) -> np.ndarray:
    """
    Project a single centred snapshot onto POD modes.
    Returns modal coefficients (k,).
    """
    return modes.T @ Xc_row


def reconstruction_error(X: np.ndarray, mean: np.ndarray, modes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """
    Relative L2 reconstruction error for each snapshot (array of shape (n,)).
    error_i = ||X_i - reconstructed_i|| / ||X_i||
    """
    errors = np.empty(len(X))
    for i in range(len(X)):
        rec = reconstruct(mean, modes, scores[i])
        errors[i] = np.linalg.norm(X[i] - rec) / (np.linalg.norm(X[i]) + 1e-12)
    return errors
