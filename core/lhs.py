"""
core/lhs.py
===========
Latin Hypercube Sampling (LHS) for virtual database upscaling.

Why LHS instead of pure random sampling?
-----------------------------------------
Pure random sampling can leave large regions of parameter space empty.
LHS divides each dimension into n equal intervals and places exactly
one sample per interval per dimension — guaranteeing uniform coverage
in all marginal distributions.

Steps
-----
  1. Divide each of the d dimensions into n equal bins
  2. Pick one random point inside each bin for every dimension
  3. Randomly permute the bin assignments across dimensions

Result: n samples that fill the d-dimensional space far more evenly
than n random draws, using only n × d random numbers.

Application here
----------------
  - Original DOE: 100 CFD runs with 26 geometry parameters
  - LHS upscaling: generate 1000 new parameter combinations
  - Evaluate with RBF surrogate instead of expensive CFD solver
  - Result: 1100-point design space for richer analysis
"""

import numpy as np
from scipy.stats import qmc


def latin_hypercube(
    n: int,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate n LHS samples uniformly within [bounds_low, bounds_high].

    Parameters
    ----------
    n           : number of virtual samples to generate
    bounds_low  : (d,) lower bound per parameter dimension
    bounds_high : (d,) upper bound per parameter dimension
    seed        : random seed for reproducibility

    Returns
    -------
    (n, d) array of sampled parameter values
    """
    d = len(bounds_low)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit_samples = sampler.random(n=n)                      # (n, d) in [0, 1]
    return qmc.scale(unit_samples, bounds_low, bounds_high)  # (n, d) in bounds


def doe_bounds(doe_df, param_cols: list) -> tuple:
    """
    Extract per-parameter min/max bounds from the original DOE table.

    Returns
    -------
    (bounds_low, bounds_high) — two (d,) arrays
    """
    low  = doe_df[param_cols].min().values.astype(float)
    high = doe_df[param_cols].max().values.astype(float)
    return low, high


def discrepancy_score(samples: np.ndarray) -> float:
    """
    Centered L2-discrepancy: measures how uniformly samples fill the space.
    Lower is better (0 = perfectly uniform).
    """
    unit = qmc.scale(samples,
                     samples.min(axis=0),
                     samples.max(axis=0),
                     reverse=True)
    return float(qmc.discrepancy(unit))
