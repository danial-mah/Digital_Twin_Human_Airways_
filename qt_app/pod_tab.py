"""
qt_app/pod_tab.py
=================
POD Analysis tab — mirrors pages/4_POD_Analysis.py.

Sub-tabs:
  Energy Curves   — cumulative variance + singular values bar (matplotlib)
  Mode Shapes     — 3D spatial mode visualisation (PyVista)
  Reconstruction  — original vs reconstructed pressure + error map (PyVista)
  Validation      — reconstruction error vs modes + K-fold (matplotlib)
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox, QPushButton, QTabWidget, QSizePolicy,
    QSplitter, QSpinBox, QProgressBar,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyvista as pv
from pyvistaqt import QtInteractor

from core.pod import cumulative_energy, modes_for_energy, reconstruct, reconstruction_error, compute_pod

BG = '#0f1119'; AXES_BG = '#1a1d26'; FG = '#e0e0e0'; ACCENT = '#4EB3D3'; GRID = '#2a2d3a'
TAB_SS = """
    QTabWidget::pane   { border:1px solid #2a2a2a; }
    QTabBar::tab       { background:#1a1d26; color:#aaa; padding:6px 14px;
                         border:1px solid #2a2a2a; border-bottom:none; font-size:11px; }
    QTabBar::tab:selected { background:#252836; color:#4EB3D3; border-top:2px solid #4EB3D3; }
    QTabBar::tab:hover    { background:#252836; color:#fff; }
"""


def _fig(w=8, h=4):
    return Figure(figsize=(w, h), facecolor=BG, tight_layout=True)

def _ax(ax):
    ax.set_facecolor(AXES_BG)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
    ax.title.set_color(ACCENT)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.grid(color=GRID, alpha=0.5, lw=0.5)
    return ax

def _cv(fig):
    c = FigureCanvas(fig)
    c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    c.setMinimumHeight(300)
    return c

def _pv_interactor(parent=None):
    p = QtInteractor(parent)
    p.set_background('#0f1119')
    p.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    p.setMinimumHeight(350)
    return p


class KFoldThread(QThread):
    result = pyqtSignal(list)

    def __init__(self, P_pres, n_folds, k_modes):
        super().__init__()
        self.P = P_pres; self.k = n_folds; self.km = k_modes

    def run(self):
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(self.P))
        folds = np.array_split(idx, self.k)
        errs = []
        for i, test_idx in enumerate(folds):
            train_idx = np.concatenate([folds[j] for j in range(self.k) if j != i])
            P_tr = self.P[train_idx]; P_te = self.P[test_idx]
            m_tr, modes_tr, _, _ = compute_pod(P_tr)
            test_sc = (P_te - m_tr) @ modes_tr[:, :self.km]
            rec = m_tr + test_sc @ modes_tr[:, :self.km].T
            e = float(np.linalg.norm(P_te - rec) / (np.linalg.norm(P_te) + 1e-12))
            errs.append(e)
        self.result.emit(errs)


class ErrVsModesThread(QThread):
    result = pyqtSignal(list, list)

    def __init__(self, P_pres, mean_p, modes_p, scores_p):
        super().__init__()
        self.P = P_pres; self.mean = mean_p
        self.modes = modes_p; self.scores = scores_p

    def run(self):
        k_range = list(range(1, min(31, self.P.shape[0])))
        errs = []
        for k in k_range:
            e = reconstruction_error(self.P, self.mean,
                                     self.modes[:, :k], self.scores[:, :k])
            errs.append(float(e.mean()))
        self.result.emit(k_range, errs)


class PODTab(QWidget):

    def __init__(self, mean_geo, modes_geo, svals_geo,
                 mean_pres, modes_pres, scores_pres, svals_pres,
                 ref_coords, all_pressures, parent=None):
        super().__init__(parent)
        self.mean_geo   = mean_geo;   self.modes_geo  = modes_geo
        self.svals_geo  = svals_geo
        self.mean_pres  = mean_pres;  self.modes_pres = modes_pres
        self.scores_pres= scores_pres; self.svals_pres = svals_pres
        self.ref_coords = ref_coords;  self.P_pres     = all_pressures
        self.e_geo  = cumulative_energy(svals_geo)
        self.e_pres = cumulative_energy(svals_pres)
        self._kfold_thread = None
        self._err_thread   = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        tabs = QTabWidget(); tabs.setStyleSheet(TAB_SS)
        tabs.addTab(self._tab_energy(),     "⚡ Energy Curves")
        tabs.addTab(self._tab_modes(),      "🌀 Mode Shapes")
        tabs.addTab(self._tab_recon(),      "🔄 Reconstruction")
        tabs.addTab(self._tab_validation(), "✅ Validation")
        root.addWidget(tabs)

    # ── Energy Curves ─────────────────────────────────────────────────────────

    def _tab_energy(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)

        k_plot = min(40, len(self.svals_geo), len(self.svals_pres))
        x = list(range(1, k_plot + 1))

        fig = _fig(9, 4)
        ax  = _ax(fig.add_subplot(111))
        ax.plot(x, self.e_geo[:k_plot] * 100, 'o-', color='#4EB3D3',
                ms=4, lw=1.8, label="Geometry")
        ax.plot(x, self.e_pres[:k_plot] * 100, 's-', color='#F4A261',
                ms=4, lw=1.8, label="Pressure")
        ax.axhline(95, ls='--', color='#888', lw=1)
        ax.axhline(99, ls=':',  color='#888', lw=1)
        ax.text(k_plot * 0.6, 95.5, '95 %', color='#aaa', fontsize=8)
        ax.text(k_plot * 0.6, 99.5, '99 %', color='#aaa', fontsize=8)
        ax.set_xlabel("Number of Modes"); ax.set_ylabel("Cumulative Energy (%)")
        ax.set_title("POD Energy Curves — Geometry vs Pressure", color=ACCENT)
        ax.set_ylim(0, 101)
        ax.legend(framealpha=0.3, labelcolor=FG, fontsize=9)
        v.addWidget(_cv(fig))

        # Metrics row
        m_row = QHBoxLayout()
        for label, val in [
            ("Geo modes for 95%",  modes_for_energy(self.svals_geo,  0.95)),
            ("Geo modes for 99%",  modes_for_energy(self.svals_geo,  0.99)),
            ("Pres modes for 95%", modes_for_energy(self.svals_pres, 0.95)),
            ("Pres modes for 99%", modes_for_energy(self.svals_pres, 0.99)),
        ]:
            box = QWidget()
            bl  = QVBoxLayout(box)
            bl.setContentsMargins(8, 4, 8, 4)
            val_lbl = QLabel(str(val))
            val_lbl.setStyleSheet(f"color:{ACCENT}; font-size:20px; font-weight:bold;")
            val_lbl.setAlignment(Qt.AlignCenter)
            cap_lbl = QLabel(label)
            cap_lbl.setStyleSheet(f"color:#888; font-size:10px;")
            cap_lbl.setAlignment(Qt.AlignCenter)
            bl.addWidget(val_lbl); bl.addWidget(cap_lbl)
            box.setStyleSheet("background:#1a1d26; border-radius:6px; margin:4px;")
            m_row.addWidget(box)
        v.addLayout(m_row)

        # Singular value bar chart
        fig2 = _fig(9, 3)
        ax2  = _ax(fig2.add_subplot(111))
        xb = np.arange(k_plot)
        ax2.bar(xb - 0.2, self.svals_geo[:k_plot],  0.35, color='#4EB3D3', label="Geometry",  alpha=0.8)
        ax2.bar(xb + 0.2, self.svals_pres[:k_plot], 0.35, color='#F4A261', label="Pressure",  alpha=0.8)
        ax2.set_yscale('log')
        ax2.set_xlabel("Mode"); ax2.set_ylabel("Singular value (log)")
        ax2.set_title("Singular Value Spectrum", color=ACCENT)
        ax2.legend(framealpha=0.3, labelcolor=FG, fontsize=8)
        v.addWidget(_cv(fig2))
        return w

    # ── Mode Shapes ───────────────────────────────────────────────────────────

    def _tab_modes(self) -> QWidget:
        w = QWidget(); root = QVBoxLayout(w)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Field:"))
        self._mode_field = QComboBox()
        self._mode_field.addItems(["Geometry (displacement)", "Pressure"])
        ctrl.addWidget(self._mode_field)
        ctrl.addWidget(QLabel("  Mode:"))
        self._mode_sl = QSlider(Qt.Horizontal)
        self._mode_sl.setRange(1, 20)
        self._mode_sl.setValue(1)
        self._mode_lbl = QLabel("1")
        self._mode_sl.setFixedWidth(200)
        ctrl.addWidget(self._mode_sl); ctrl.addWidget(self._mode_lbl); ctrl.addStretch()
        root.addLayout(ctrl)

        self._mode_pv = _pv_interactor(self)
        root.addWidget(self._mode_pv)

        self._mode_info = QLabel("")
        self._mode_info.setStyleSheet(f"color:{ACCENT}; font-size:12px; padding:4px;")
        root.addWidget(self._mode_info)

        self._mode_sl.valueChanged.connect(self._update_mode)
        self._mode_field.currentIndexChanged.connect(self._update_mode)
        self._update_mode()
        return w

    def _update_mode(self):
        k    = self._mode_sl.value()
        self._mode_lbl.setText(str(k))
        geo  = self._mode_field.currentIndex() == 0

        if geo:
            vec    = self.modes_geo[:, k - 1].reshape(-1, 3)
            colour = np.linalg.norm(vec, axis=1)
            coords = self.ref_coords
            e_ind  = self.e_geo[k - 1] - (self.e_geo[k - 2] if k > 1 else 0.0)
        else:
            colour = self.modes_pres[:, k - 1]
            coords = self.ref_coords
            e_ind  = self.e_pres[k - 1] - (self.e_pres[k - 2] if k > 1 else 0.0)

        cloud = pv.PolyData(coords)
        cloud["value"] = colour
        self._mode_pv.clear()
        self._mode_pv.add_mesh(cloud, scalars="value", render_points_as_spheres=False,
                               point_size=2.5, cmap='RdBu', show_scalar_bar=True)
        self._mode_pv.reset_camera()
        self._mode_info.setText(
            f"Mode {k} — individual energy contribution: {e_ind * 100:.2f} %"
        )

    # ── Reconstruction ────────────────────────────────────────────────────────

    def _tab_recon(self) -> QWidget:
        w = QWidget(); root = QVBoxLayout(w)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Snapshot:"))
        self._rec_snap = QSlider(Qt.Horizontal)
        self._rec_snap.setRange(1, 100); self._rec_snap.setValue(1)
        self._rec_snap.setFixedWidth(150)
        self._rec_snap_lbl = QLabel("1")
        ctrl.addWidget(self._rec_snap); ctrl.addWidget(self._rec_snap_lbl)
        ctrl.addWidget(QLabel("  Modes:"))
        self._rec_k = QSlider(Qt.Horizontal)
        self._rec_k.setRange(1, min(40, self.P_pres.shape[0]))
        self._rec_k.setValue(10)
        self._rec_k.setFixedWidth(150)
        self._rec_k_lbl = QLabel("10")
        ctrl.addWidget(self._rec_k); ctrl.addWidget(self._rec_k_lbl)
        btn = QPushButton("Reconstruct"); btn.setFixedWidth(100)
        btn.clicked.connect(self._update_recon)
        ctrl.addWidget(btn); ctrl.addStretch()
        root.addLayout(ctrl)

        self._rec_err_lbl = QLabel("Relative L2 error: —")
        self._rec_err_lbl.setStyleSheet(f"color:{ACCENT}; font-size:13px; padding:4px;")
        root.addWidget(self._rec_err_lbl)

        # Two side-by-side PyVista views
        spl = QSplitter(Qt.Horizontal)
        self._rec_orig = _pv_interactor(self)
        self._rec_recon = _pv_interactor(self)
        spl.addWidget(self._rec_orig)
        spl.addWidget(self._rec_recon)
        root.addWidget(spl, stretch=2)

        # Error map
        self._err_pv = _pv_interactor(self)
        self._err_pv.setMinimumHeight(220)
        root.addWidget(self._err_pv, stretch=1)

        self._rec_snap.valueChanged.connect(lambda v: self._rec_snap_lbl.setText(str(v)))
        self._rec_k.valueChanged.connect(lambda v: self._rec_k_lbl.setText(str(v)))
        return w

    def _update_recon(self):
        snap = self._rec_snap.value()
        k    = self._rec_k.value()
        actual = self.P_pres[snap - 1]
        rec    = reconstruct(self.mean_pres, self.modes_pres[:, :k],
                             self.scores_pres[snap - 1, :k])
        err    = float(np.linalg.norm(actual - rec) / (np.linalg.norm(actual) + 1e-12))
        self._rec_err_lbl.setText(f"Relative L2 error ({k} modes): {err * 100:.3f} %")

        coords = self.ref_coords
        cmin = float(actual.min()); cmax = float(actual.max())
        rng  = [cmin, cmax] if cmax > cmin else None

        orig_cloud = pv.PolyData(coords); orig_cloud["P"] = actual
        self._rec_orig.clear()
        self._rec_orig.add_mesh(orig_cloud, scalars="P", cmap='jet',
                                 point_size=2.5, clim=rng,
                                 scalar_bar_args={"title": "Orig P (Pa)"})
        self._rec_orig.reset_camera()

        rec_cloud = pv.PolyData(coords); rec_cloud["P"] = rec
        self._rec_recon.clear()
        self._rec_recon.add_mesh(rec_cloud, scalars="P", cmap='jet',
                                  point_size=2.5, clim=rng,
                                  scalar_bar_args={"title": f"Recon {k}m (Pa)"})
        self._rec_recon.reset_camera()

        err_cloud = pv.PolyData(coords); err_cloud["E"] = np.abs(actual - rec)
        self._err_pv.clear()
        self._err_pv.add_mesh(err_cloud, scalars="E", cmap='hot',
                               point_size=2.5,
                               scalar_bar_args={"title": "|Error| (Pa)"})
        self._err_pv.reset_camera()

    # ── Validation ────────────────────────────────────────────────────────────

    def _tab_validation(self) -> QWidget:
        w = QWidget(); root = QVBoxLayout(w)

        # Error-vs-modes chart
        self._val_fig = _fig(8, 3.5)
        self._val_ax  = _ax(self._val_fig.add_subplot(111))
        self._val_cv  = _cv(self._val_fig)
        root.addWidget(self._val_cv)

        self._val_lbl = QLabel("Click 'Compute' to run error vs modes (takes ~10 s)")
        self._val_lbl.setStyleSheet(f"color:#888; font-size:11px; padding:4px;")
        root.addWidget(self._val_lbl)

        btn_err = QPushButton("Compute Error vs Modes")
        btn_err.clicked.connect(self._run_err_vs_modes)
        root.addWidget(btn_err)

        root.addWidget(self._sep())

        # K-fold section
        kf_ctrl = QHBoxLayout()
        kf_ctrl.addWidget(QLabel("Folds:"))
        self._kf_spin = QSpinBox(); self._kf_spin.setRange(2, 10); self._kf_spin.setValue(5)
        kf_ctrl.addWidget(self._kf_spin)
        kf_ctrl.addWidget(QLabel("  Modes:"))
        self._kf_k_spin = QSpinBox(); self._kf_k_spin.setRange(1, 30); self._kf_k_spin.setValue(10)
        kf_ctrl.addWidget(self._kf_k_spin)
        btn_kf = QPushButton("Run K-Fold POD")
        btn_kf.clicked.connect(self._run_kfold)
        kf_ctrl.addWidget(btn_kf); kf_ctrl.addStretch()
        root.addLayout(kf_ctrl)

        self._kf_progress = QProgressBar()
        self._kf_progress.setVisible(False)
        root.addWidget(self._kf_progress)

        self._kf_fig = _fig(8, 3)
        self._kf_ax  = _ax(self._kf_fig.add_subplot(111))
        self._kf_cv  = _cv(self._kf_fig)
        root.addWidget(self._kf_cv)

        self._kf_lbl = QLabel("")
        self._kf_lbl.setStyleSheet(f"color:{ACCENT}; font-size:11px; padding:4px;")
        root.addWidget(self._kf_lbl)
        return w

    def _sep(self):
        line = QWidget(); line.setFixedHeight(1)
        line.setStyleSheet("background:#2a2d3a;")
        return line

    def _run_err_vs_modes(self):
        self._val_lbl.setText("Computing… please wait")
        self._err_thread = ErrVsModesThread(
            self.P_pres, self.mean_pres, self.modes_pres, self.scores_pres)
        self._err_thread.result.connect(self._on_err_vs_modes)
        self._err_thread.start()

    def _on_err_vs_modes(self, k_range, errs):
        ax = self._val_ax; ax.cla(); _ax(ax)
        ax.plot(k_range, [e * 100 for e in errs], 'o-', color='#4EB3D3', ms=4, lw=1.8)
        ax.set_xlabel("Modes"); ax.set_ylabel("Mean Relative L2 Error (%)")
        ax.set_title("Pressure Reconstruction Error vs Modes", color=ACCENT)
        self._val_cv.draw()
        errs_a = np.array(errs)
        k_1 = next((k for k, e in zip(k_range, errs_a) if e < 0.01), k_range[-1])
        k_5 = next((k for k, e in zip(k_range, errs_a) if e < 0.05), k_range[-1])
        self._val_lbl.setText(f"Modes for <1% error: {k_1}  |  Modes for <5% error: {k_5}")

    def _run_kfold(self):
        self._kf_progress.setVisible(True)
        self._kf_progress.setRange(0, 0)
        k_folds = self._kf_spin.value(); k_modes = self._kf_k_spin.value()
        self._kfold_thread = KFoldThread(self.P_pres, k_folds, k_modes)
        self._kfold_thread.result.connect(self._on_kfold)
        self._kfold_thread.start()

    def _on_kfold(self, errs):
        self._kf_progress.setVisible(False)
        n = len(errs)
        ax = self._kf_ax; ax.cla(); _ax(ax)
        bars = ax.bar(range(1, n + 1), [e * 100 for e in errs],
                      color='#4EB3D3', alpha=0.8, edgecolor=GRID)
        ax.set_xticks(range(1, n + 1))
        ax.set_xticklabels([f"Fold {i}" for i in range(1, n + 1)])
        ax.set_ylabel("Relative L2 Error (%)"); ax.set_xlabel("Fold")
        ax.set_title(f"{n}-Fold POD Reconstruction Error", color=ACCENT)
        self._kf_cv.draw()
        self._kf_lbl.setText(
            f"Mean: {np.mean(errs)*100:.3f}%  |  "
            f"Worst: {np.max(errs)*100:.3f}%  |  "
            f"Best: {np.min(errs)*100:.3f}%"
        )
