"""
scripts/precompute.py
=====================
One-time data export and precomputation script.

Run this once from the project root:
    python scripts/precompute.py

What it produces
----------------
precomputed/
    pod_geometry.npz   – POD results for geometry  (mean, modes, scores, svalues)
    pod_pressure.npz   – POD results for pressure  (mean, modes, scores, svalues)
    ref_coords_s50.npy – reference mesh coords at stride=50

export/npy/
    snapshot_N_coords_s{stride}.npy     – deformed coordinates (N=1..100)
    snapshot_N_pressure_s{stride}.npy   – pressure field       (N=1..100)

export/vtk/
    snapshot_N.vtu  – VTK UnstructuredGrid point cloud with pressure scalar
                      (open in ParaView)

Usage
-----
    # Default: export NPY at stride=10, VTK at stride=10
    python scripts/precompute.py

    # Full resolution NPY (slow, ~6 GB)
    python scripts/precompute.py --stride 1

    # Skip VTK export
    python scripts/precompute.py --no-vtk
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Make sure project root is on the path ─────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.pod import compute_pod

# ── Paths ─────────────────────────────────────────────────────────────────────
POINTS_DIR   = ROOT / "Points"
PRESSURE_DIR = ROOT / "Pressure"
PRECOMP_DIR  = ROOT / "precomputed"
EXPORT_NPY   = ROOT / "export" / "npy"
EXPORT_VTK   = ROOT / "export" / "vtk"


# ── Standalone binary reader (no streamlit) ───────────────────────────────────

def read_bin(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return np.fromfile(f, dtype="<f8", count=n)


def load_coords(n: int, stride: int = 1) -> np.ndarray:
    return read_bin(POINTS_DIR / "Snapshots" / f"snapshot{n}.bin").reshape(-1, 3)[::stride]


def load_pressure(n: int, stride: int = 1) -> np.ndarray:
    return read_bin(PRESSURE_DIR / "Snapshots" / f"snapshot{n}.bin")[::stride]


def load_ref_coords(stride: int = 1) -> np.ndarray:
    return read_bin(PRESSURE_DIR / "points.bin").reshape(-1, 3)[::stride]


# ── Progress helper ───────────────────────────────────────────────────────────

def progress(i: int, total: int, label: str = "", width: int = 40):
    done = int(width * i / total)
    bar  = "█" * done + "░" * (width - done)
    pct  = 100 * i / total
    print(f"\r  [{bar}] {pct:5.1f}%  {label}", end="", flush=True)
    if i == total:
        print()


# ── Step 1: Precomputed POD ───────────────────────────────────────────────────

def build_precomputed(pod_stride: int = 50):
    PRECOMP_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Building precomputed POD cache (stride=50)…")

    # Reference coords
    print("  Loading reference coordinates…")
    ref = load_ref_coords(pod_stride)
    np.save(PRECOMP_DIR / "ref_coords_s50.npy", ref)
    print(f"  ref_coords saved  {ref.shape}  ({ref.nbytes/1e6:.1f} MB)")

    # Geometry POD
    print("  Loading all 100 geometry snapshots…")
    t0 = time.time()
    n_pts = len(ref)
    X = np.empty((100, n_pts * 3), dtype=np.float64)
    for i in range(100):
        progress(i + 1, 100, f"snapshot {i+1}")
        X[i] = load_coords(i + 1, pod_stride).ravel()
    print(f"  Loaded in {time.time()-t0:.1f}s  shape={X.shape}")

    print("  Computing geometry POD (SVD)…")
    t0 = time.time()
    mean_g, modes_g, scores_g, sv_g = compute_pod(X)
    print(f"  SVD done in {time.time()-t0:.1f}s  {len(sv_g)} modes")
    np.savez_compressed(
        PRECOMP_DIR / "pod_geometry.npz",
        mean=mean_g, modes=modes_g, scores=scores_g, svalues=sv_g,
    )
    print(f"  pod_geometry.npz saved  ({(PRECOMP_DIR/'pod_geometry.npz').stat().st_size/1e6:.1f} MB)")

    # Pressure POD
    print("  Loading all 100 pressure snapshots…")
    t0 = time.time()
    P = np.empty((100, n_pts), dtype=np.float64)
    for i in range(100):
        progress(i + 1, 100, f"snapshot {i+1}")
        P[i] = load_pressure(i + 1, pod_stride)
    print(f"  Loaded in {time.time()-t0:.1f}s  shape={P.shape}")

    print("  Computing pressure POD (SVD)…")
    t0 = time.time()
    mean_p, modes_p, scores_p, sv_p = compute_pod(P)
    print(f"  SVD done in {time.time()-t0:.1f}s  {len(sv_p)} modes")
    np.savez_compressed(
        PRECOMP_DIR / "pod_pressure.npz",
        mean=mean_p, modes=modes_p, scores=scores_p, svalues=sv_p,
    )
    print(f"  pod_pressure.npz saved  ({(PRECOMP_DIR/'pod_pressure.npz').stat().st_size/1e6:.1f} MB)")

    # All pressures matrix — needed by cloud deployment (no .bin files there)
    print("  Saving all_pressures_s50.npz for cloud deployment…")
    np.savez_compressed(PRECOMP_DIR / "all_pressures_s50.npz", data=P)
    print(f"  all_pressures_s50.npz saved  ({(PRECOMP_DIR/'all_pressures_s50.npz').stat().st_size/1e6:.1f} MB)")

    print("  [1/3] Done — precomputed/ is ready.\n")
    return ref, mean_p, modes_p  # returned for VTK step


# ── Step 2: Export NPY archives ───────────────────────────────────────────────

def export_npy(stride: int = 10):
    EXPORT_NPY.mkdir(parents=True, exist_ok=True)
    print(f"[2/3] Exporting NPY archives (stride={stride})…")
    t0 = time.time()
    for i in range(100):
        progress(i + 1, 100, f"snapshot {i+1}")
        coords = load_coords(i + 1, stride)
        press  = load_pressure(i + 1, stride)
        np.save(EXPORT_NPY / f"snapshot_{i+1}_coords_s{stride}.npy", coords)
        np.save(EXPORT_NPY / f"snapshot_{i+1}_pressure_s{stride}.npy", press)
    print(f"  Exported 200 NPY files in {time.time()-t0:.1f}s")
    total_mb = sum(f.stat().st_size for f in EXPORT_NPY.glob("*.npy")) / 1e6
    print(f"  Total size: {total_mb:.0f} MB  →  {EXPORT_NPY}")
    print("  [2/3] Done.\n")


# ── Step 3: Export VTK point clouds ───────────────────────────────────────────

def export_vtk(stride: int = 10):
    try:
        import pyvista as pv
    except ImportError:
        print("[3/3] pyvista not installed — skipping VTK export.")
        print("      Install with:  pip install pyvista")
        return

    EXPORT_VTK.mkdir(parents=True, exist_ok=True)
    print(f"[3/3] Exporting VTK point clouds (stride={stride})…")
    t0 = time.time()

    for i in range(100):
        progress(i + 1, 100, f"snapshot {i+1}")
        coords = load_coords(i + 1, stride)
        press  = load_pressure(i + 1, stride)

        mesh = pv.PolyData(coords)
        mesh["StaticPressure_Pa"] = press
        mesh.save(str(EXPORT_VTK / f"snapshot_{i+1}.vtp"))

    print(f"  Exported 100 VTK files in {time.time()-t0:.1f}s")
    total_mb = sum(f.stat().st_size for f in EXPORT_VTK.glob("*.vtp")) / 1e6
    print(f"  Total size: {total_mb:.0f} MB  →  {EXPORT_VTK}")
    print("  Open any .vtp in ParaView — colour by StaticPressure_Pa.")
    print("  [3/3] Done.\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Precompute and export Human Airways data.")
    parser.add_argument("--stride",    type=int, default=10,
                        help="Subsampling stride for NPY/VTK export (default 10)")
    parser.add_argument("--no-vtk",   action="store_true",
                        help="Skip VTK export")
    parser.add_argument("--no-npy",   action="store_true",
                        help="Skip per-snapshot NPY export")
    parser.add_argument("--pod-only", action="store_true",
                        help="Only build precomputed/ POD cache")
    args = parser.parse_args()

    print("=" * 60)
    print("  Human Airways Digital Twin — Precompute & Export")
    print("=" * 60)

    build_precomputed(pod_stride=50)

    if not args.pod_only:
        if not args.no_npy:
            export_npy(stride=args.stride)  
        if not args.no_vtk:
            export_vtk(stride=args.stride)

    print("=" * 60)
    print("  All done! Summary:")
    print(f"    precomputed/     →  POD cache for fast app startup")
    print(f"    export/npy/      →  {100*2} NPY archives (stride={args.stride})")
    print(f"    export/vtk/      →  100 .vtp files for ParaView")
    print("=" * 60)


if __name__ == "__main__":
    main()
