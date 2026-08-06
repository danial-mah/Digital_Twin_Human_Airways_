"""
core/data_io.py
===============
All data-loading helpers for the Human Airways Digital Twin.
Works in both Streamlit and pure-Python (Qt app, scripts) contexts.

Data layout
-----------
  Points/points.bin            – reference mesh coordinates  (2,135,906 × 3 float64)
  Points/Snapshots/snapshotN.bin – deformed coords for run N (2,135,906 × 3 float64)
  Pressure/points.bin          – same reference mesh (pressure field reference)
  Pressure/Snapshots/snapshotN.bin – static pressure for run N (2,135,906 × 1 float64)
  Points/doe_point.csv         – 100 rows × 26 geometry parameters
  precomputed/                 – NPY cache (created by scripts/precompute.py)

Binary file format (Twin Builder):
  [8-byte little-endian uint64 = N] [N × float64 values]

Subsampling strategy
--------------------
  stride = 50  →  take every 50th node  →  ~42,718 nodes from 2,135,906
"""

import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd

# ── Optional Streamlit caching (falls back to identity when not available) ──────
try:
    import streamlit as st
    def _cache(fn):
        return st.cache_data(show_spinner=False)(fn)
except ImportError:
    def _cache(fn):
        return fn

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
POINTS_DIR      = BASE_DIR / "Points"
PRESSURE_DIR    = BASE_DIR / "Pressure"
PRECOMPUTED_DIR = BASE_DIR / "precomputed"

# Every 50th node → ~42,718 points displayed instead of 2,135,906
STRIDE = 50

# Human-readable labels for the 26 DOE geometry parameters
PARAM_LABELS = {
    "A_glotis":            "Glottis Area (mm²)",
    "A_epiglotis":         "Epiglottis Area (mm²)",
    "r_curvature":         "Curvature Radius (mm)",
    "d_trachea":           "Trachea Diameter (mm)",
    "l_trachea":           "Trachea Length (mm)",
    "teta_branch_trachea": "Trachea Branch Angle (°)",
    "l_l":                 "Left Main Bronchus Length (mm)",
    "l_r":                 "Right Main Bronchus Length (mm)",
    "teta_branch_l":       "Left Branch Angle (°)",
    "teta_branch_r":       "Right Branch Angle (°)",
    "l_ll":                "LL Branch Length (mm)",
    "l_lr":                "LR Branch Length (mm)",
    "l_rl":                "RL Branch Length (mm)",
    "l_rr":                "RR Branch Length (mm)",
    "teta_branch_ll":      "LL Angle (°)",
    "teta_branch_lr":      "LR Angle (°)",
    "teta_branch_rl":      "RL Angle (°)",
    "teta_branch_rr":      "RR Angle (°)",
    "l_lll":               "LLL Length (mm)",
    "l_llr":               "LLR Length (mm)",
    "l_lrl":               "LRL Length (mm)",
    "l_lrr":               "LRR Length (mm)",
    "l_rll":               "RLL Length (mm)",
    "l_rlr":               "RLR Length (mm)",
    "l_rrl":               "RRL Length (mm)",
    "l_rrr":               "RRR Length (mm)",
}

# Parameter groupings for UI organisation
PARAM_GROUPS = {
    "Main airway (glottis, trachea)": [
        "A_glotis", "A_epiglotis", "r_curvature",
        "d_trachea", "l_trachea", "teta_branch_trachea",
    ],
    "First-level branches (L / R)": [
        "l_l", "l_r", "teta_branch_l", "teta_branch_r",
        "l_ll", "l_lr", "l_rl", "l_rr",
        "teta_branch_ll", "teta_branch_lr", "teta_branch_rl", "teta_branch_rr",
    ],
    "Second-level sub-branches": [
        "l_lll", "l_llr", "l_lrl", "l_lrr",
        "l_rll", "l_rlr", "l_rrl", "l_rrr",
    ],
}


# ── Low-level binary reader ────────────────────────────────────────────────────

def read_bin(path: Path) -> np.ndarray:
    """Read a Twin Builder .bin file → 1D float64 array."""
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        data = np.fromfile(f, dtype="<f8", count=n)
    return data


# ── Cached high-level loaders ──────────────────────────────────────────────────

@_cache
def load_doe() -> pd.DataFrame:
    df = pd.read_csv(POINTS_DIR / "doe_point.csv")
    df["snapshot_num"] = df["points"].str.extract(r"snapshot(\d+)\.bin").astype(int)
    return df.sort_values("snapshot_num").reset_index(drop=True)


@_cache
def load_settings() -> dict:
    with open(PRESSURE_DIR / "settings.json") as f:
        return json.load(f)


@_cache
def load_ref_coords(stride: int = STRIDE) -> np.ndarray:
    if stride == STRIDE:
        cached = PRECOMPUTED_DIR / "ref_coords_s50.npy"
        if cached.exists():
            return np.load(cached)
    raw = read_bin(PRESSURE_DIR / "points.bin")
    return raw.reshape(-1, 3)[::stride]


@_cache
def load_snapshot_coords(n: int, stride: int = STRIDE) -> np.ndarray:
    bin_path = POINTS_DIR / "Snapshots" / f"snapshot{n}.bin"
    if bin_path.exists():
        return read_bin(bin_path).reshape(-1, 3)[::stride]
    # Cloud fallback: reconstruct from precomputed geometry POD
    pre = load_precomputed_pod("geometry")
    if pre is not None:
        rec = (pre["mean"] + pre["modes"] @ pre["scores"][n - 1]).reshape(-1, 3)
        return rec
    raise FileNotFoundError(
        f"{bin_path} not found and no precomputed geometry POD available."
    )


@_cache
def load_pressure_snapshot(n: int, stride: int = STRIDE) -> np.ndarray:
    bin_path = PRESSURE_DIR / "Snapshots" / f"snapshot{n}.bin"
    if bin_path.exists():
        return read_bin(bin_path)[::stride]
    # Cloud fallback: load from precomputed all_pressures matrix
    cached = PRECOMPUTED_DIR / "all_pressures_s50.npz"
    if cached.exists():
        return np.load(cached)["data"][n - 1]
    raise FileNotFoundError(
        f"{bin_path} not found and no precomputed all_pressures_s50.npz available. "
        "Run scripts/precompute.py first."
    )


@_cache
def load_all_coords(stride: int = STRIDE) -> np.ndarray:
    if stride == STRIDE:
        pre = load_precomputed_pod("geometry")
        if pre is not None:
            # Exact reconstruction: 100 snapshots → 100 modes → lossless
            return pre["scores"] @ pre["modes"].T + pre["mean"]
    n_pts = len(load_ref_coords(stride))
    X = np.empty((100, n_pts * 3), dtype=np.float64)
    for i in range(100):
        X[i] = load_snapshot_coords(i + 1, stride).ravel()
    return X


@_cache
def load_all_pressures(stride: int = STRIDE) -> np.ndarray:
    if stride == STRIDE:
        cached = PRECOMPUTED_DIR / "all_pressures_s50.npz"
        if cached.exists():
            return np.load(cached)["data"]
    n_pts = len(load_ref_coords(stride))
    P = np.empty((100, n_pts), dtype=np.float64)
    for i in range(100):
        P[i] = load_pressure_snapshot(i + 1, stride)
    return P


# ── Precomputed NPY cache ──────────────────────────────────────────────────────

def load_precomputed_pod(kind: str) -> "dict | None":
    """
    Load precomputed POD from precomputed/pod_<kind>.npz.
    kind: 'geometry' or 'pressure'
    Returns dict(mean, modes, scores, svalues) or None if not found.
    Generated by scripts/precompute.py.
    """
    path = PRECOMPUTED_DIR / f"pod_{kind}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return {"mean": d["mean"], "modes": d["modes"],
            "scores": d["scores"], "svalues": d["svalues"]}


def load_precomputed_ref_coords() -> "np.ndarray | None":
    """Load precomputed ref_coords at stride=50, or None."""
    path = PRECOMPUTED_DIR / "ref_coords_s50.npy"
    return np.load(path) if path.exists() else None


def precomputed_available() -> bool:
    """True if all three precomputed files exist."""
    return (
        (PRECOMPUTED_DIR / "pod_geometry.npz").exists() and
        (PRECOMPUTED_DIR / "pod_pressure.npz").exists() and
        (PRECOMPUTED_DIR / "ref_coords_s50.npy").exists()
    )


# ── Region helpers ─────────────────────────────────────────────────────────────

def region_slice(settings: dict, region: str, stride: int = STRIDE) -> slice:
    sel = settings["namedSelections"][region]
    start_full, end_full = sel[0], sel[2]
    start_i = int(np.ceil(start_full / stride))
    end_i   = int(np.floor(end_full / stride))
    return slice(start_i, end_i + 1)


def get_param_cols(doe_df: pd.DataFrame) -> list:
    return [c for c in doe_df.columns if c not in ("points", "snapshot_num")]


def get_anatomical_regions(settings: dict) -> list:
    return [k for k in settings["namedSelections"] if "workbench" not in k]
