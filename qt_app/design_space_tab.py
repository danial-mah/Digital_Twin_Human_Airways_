"""
qt_app/design_space_tab.py
==========================
Design Space tab — mirrors pages/7_Design_Space.py.

Sub-tabs:
  Parameter Sensitivity — Pearson correlation bar chart
  Design Landscape      — 2D scatter in POD score space, coloured by pressure
  Sobol Indices         — variance-based global sensitivity (computed on demand)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox, QPushButton, QTabWidget, QSizePolicy,
    QProgressBar, QCheckBox,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from core.data_io import PARAM_LABELS, get_param_cols
from core.rbf import build_rbf, predict
from core.pod import modes_for_energy
from core.lhs import latin_hypercube

BG = '#0f1119'; AXES_BG = '#1a1d26'; FG = '#e0e0e0'; ACCENT = '#4EB3D3'; GRID = '#2a2d3a'
TAB_SS = """
    QTabWidget::pane   { border:1px solid #2a2a2a; }
    QTabBar::tab       { background:#1a1d26; color:#aaa; padding:6px 14px;
                         border:1px solid #2a2a2a; border-bottom:none; font-size:11px; }
    QTabBar::tab:selected { background:#252836; color:#4EB3D3; border-top:2px solid #4EB3D3; }
    QTabBar::tab:hover    { background:#252836; color:#fff; }
"""


def _fig(w=9, h=5):
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
    c.setMinimumHeight(320)
    return c


class SobolThread(QThread):
    progress = pyqtSignal(int, int)   # (current, total)
    result   = pyqtSignal(object, object)  # S1, ST arrays

    def __init__(self, params_raw, param_cols, mean_pres, modes_pres,
                 scores_pres, N, use_dp):
        super().__init__()
        self.params_raw = params_raw; self.param_cols = param_cols
        self.mean_pres  = mean_pres;  self.modes_pres = modes_pres
        self.scores_pres = scores_pres; self.N = N; self.use_dp = use_dp

    def run(self):
        lo = self.params_raw.min(0); hi = self.params_raw.max(0)
        norm = lambda X: (X - lo) / (hi - lo + 1e-12)
        k = min(modes_for_energy(
            np.array([np.linalg.norm(self.scores_pres[:, i])
                      for i in range(self.scores_pres.shape[1])]), 0.99),
            self.scores_pres.shape[1])

        params_n = norm(self.params_raw)
        rbf = build_rbf(params_n, self.scores_pres[:, :k], kernel="thin_plate_spline")

        def eval_rbf(X_norm):
            sc  = predict(rbf, X_norm)
            out = np.empty(len(X_norm))
            for i in range(len(X_norm)):
                field = self.mean_pres + self.modes_pres[:, :k] @ sc[i]
                out[i] = float(field.max() - field.min()) if self.use_dp else float(field.mean())
            return out

        A = norm(latin_hypercube(self.N, lo, hi, seed=42))
        B = norm(latin_hypercube(self.N, lo, hi, seed=123))
        fA = eval_rbf(A); fB = eval_rbf(B)
        var_Y = float(np.var(np.concatenate([fA, fB]))) + 1e-12

        d = len(self.param_cols)
        S1 = np.zeros(d); ST = np.zeros(d)
        for i in range(d):
            AB = A.copy(); AB[:, i] = B[:, i]
            fAB = eval_rbf(AB)
            S1[i] = float(np.mean(fB * (fAB - fA))) / var_Y
            ST[i] = float(np.mean((fA - fAB) ** 2)) / (2 * var_Y)
            self.progress.emit(i + 1, d)

        self.result.emit(np.clip(S1, 0, 1), np.clip(ST, 0, 1))


class DesignSpaceTab(QWidget):

    def __init__(self, doe_df: pd.DataFrame, param_cols: list,
                 mean_pres, modes_pres, scores_pres, svals_pres,
                 scores_geo, all_pressures, parent=None):
        super().__init__(parent)
        self.doe_df      = doe_df
        self.param_cols  = param_cols
        self.labels      = [PARAM_LABELS.get(c, c) for c in param_cols]
        self.mean_pres   = mean_pres;  self.modes_pres  = modes_pres
        self.scores_pres = scores_pres; self.svals_pres  = svals_pres
        self.scores_geo  = scores_geo
        self.params_raw  = doe_df[param_cols].values.astype(float)
        self.P_all       = all_pressures
        self.mean_P_run  = all_pressures.mean(axis=1)
        self._corr       = self._compute_correlations()
        self._sobol_thread = None
        self._build_ui()

    def _compute_correlations(self):
        return np.array([
            float(np.corrcoef(self.params_raw[:, i], self.mean_P_run)[0, 1])
            for i in range(len(self.param_cols))
        ])

    def _build_ui(self):
        root = QVBoxLayout(self); root.setContentsMargins(6, 6, 6, 6)
        tabs = QTabWidget(); tabs.setStyleSheet(TAB_SS)
        tabs.addTab(self._tab_sensitivity(), "📊 Sensitivity")
        tabs.addTab(self._tab_landscape(),   "🌐 Design Landscape")
        tabs.addTab(self._tab_sobol(),       "🎲 Sobol Indices")
        root.addWidget(tabs)

    # ── Parameter Sensitivity ─────────────────────────────────────────────────

    def _tab_sensitivity(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        corr = self._corr
        order = np.argsort(np.abs(corr))
        labs  = [self.labels[i] for i in order]
        vals  = corr[order]

        fig = _fig(9, 8)
        ax  = _ax(fig.add_subplot(111))
        norm = mcolors.TwoSlopeNorm(0, vmin=-1, vmax=1)
        cmap = cm.get_cmap('RdBu_r')
        colors = [cmap(norm(v)) for v in vals]
        ax.barh(range(len(labs)), vals, color=colors, edgecolor=GRID, lw=0.3)
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels(labs, fontsize=7.5, color=FG)
        ax.axvline(0, color='#888', lw=0.8)
        ax.set_xlabel("Pearson r")
        ax.set_title("Parameter Sensitivity — Pearson r with Mean Pressure", color=ACCENT)
        sm = cm.ScalarMappable(cmap='RdBu_r', norm=norm); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.8)
        cb.set_label("r", color=FG, fontsize=8)
        cb.ax.tick_params(labelcolor=FG, labelsize=7)
        v.addWidget(_cv(fig))

        # Top 3 table
        top3_idx = np.argsort(np.abs(corr))[::-1][:3]
        info = "  |  ".join(
            f"{self.labels[i]}: r={corr[i]:.3f}" for i in top3_idx
        )
        lbl = QLabel(f"Top 3 most influential:  {info}")
        lbl.setStyleSheet(f"color:{ACCENT}; font-size:11px; padding:4px;")
        v.addWidget(lbl)
        return w

    # ── Design Landscape ──────────────────────────────────────────────────────

    def _tab_landscape(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)

        # Pressure POD space
        fig = _fig(9, 5)
        ax  = _ax(fig.add_subplot(111))
        sc = ax.scatter(self.scores_pres[:, 0], self.scores_pres[:, 1],
                        c=self.mean_P_run, cmap='jet', s=55,
                        edgecolors='white', linewidths=0.4)
        for i, (x, y) in enumerate(zip(self.scores_pres[:, 0], self.scores_pres[:, 1])):
            ax.text(x, y, str(int(self.doe_df["snapshot_num"].iloc[i])),
                    fontsize=5, color='white', ha='center', va='bottom')
        cb = fig.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label("Mean P (Pa)", color=FG, fontsize=8)
        cb.ax.tick_params(labelcolor=FG, labelsize=7)
        ax.set_xlabel("Pressure POD Mode 1 score")
        ax.set_ylabel("Pressure POD Mode 2 score")
        ax.set_title("Design Landscape — Pressure POD Score Space", color=ACCENT)
        v.addWidget(_cv(fig))

        # Geometry POD space
        fig2 = _fig(9, 4)
        ax2  = _ax(fig2.add_subplot(111))
        sc2 = ax2.scatter(self.scores_geo[:, 0], self.scores_geo[:, 1],
                          c=self.mean_P_run, cmap='jet', s=55,
                          edgecolors='white', linewidths=0.4)
        cb2 = fig2.colorbar(sc2, ax=ax2, shrink=0.85)
        cb2.set_label("Mean P (Pa)", color=FG, fontsize=8)
        cb2.ax.tick_params(labelcolor=FG, labelsize=7)
        ax2.set_xlabel("Geometry POD Mode 1 score")
        ax2.set_ylabel("Geometry POD Mode 2 score")
        ax2.set_title("Design Landscape — Geometry POD Score Space", color=ACCENT)
        v.addWidget(_cv(fig2))

        # Cross-correlation Geo Mode 1 vs Pres Mode 1
        fig3 = _fig(9, 3.5)
        ax3  = _ax(fig3.add_subplot(111))
        xc = self.scores_geo[:, 0]; yc = self.scores_pres[:, 0]
        ax3.scatter(xc, yc, c=self.mean_P_run, cmap='jet', s=40,
                    edgecolors='white', linewidths=0.3)
        m, b = np.polyfit(xc, yc, 1)
        xl = np.linspace(xc.min(), xc.max(), 50)
        ax3.plot(xl, m * xl + b, 'w--', lw=1.2,
                 label=f"OLS slope={m:.3f}")
        ax3.legend(framealpha=0.3, labelcolor=FG, fontsize=8)
        ax3.set_xlabel("Geo Mode 1 score")
        ax3.set_ylabel("Pres Mode 1 score")
        ax3.set_title("Cross-space: Geo Mode 1 vs Pressure Mode 1", color=ACCENT)
        v.addWidget(_cv(fig3))
        return w

    # ── Sobol Indices ─────────────────────────────────────────────────────────

    def _tab_sobol(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("N:"))
        self._sobol_n = QComboBox()
        for n in [128, 256, 512, 1024]:
            self._sobol_n.addItem(str(n))
        self._sobol_n.setCurrentIndex(2)  # 512
        ctrl.addWidget(self._sobol_n)
        self._sobol_dp = QCheckBox("Use ΔP (else mean pressure)")
        ctrl.addWidget(self._sobol_dp)
        btn = QPushButton("Compute Sobol Indices")
        btn.clicked.connect(self._run_sobol)
        ctrl.addWidget(btn); ctrl.addStretch()
        v.addLayout(ctrl)

        self._sobol_prog = QProgressBar()
        self._sobol_prog.setRange(0, len(self.param_cols))
        self._sobol_prog.setVisible(False)
        v.addWidget(self._sobol_prog)

        self._sobol_status = QLabel(
            "Sobol indices measure variance-based sensitivity including non-linear effects.\n"
            "S₁ = first-order (parameter alone)   Sₜ = total (including interactions)"
        )
        self._sobol_status.setStyleSheet(f"color:#888; font-size:10px; padding:4px;")
        v.addWidget(self._sobol_status)

        self._sobol_fig = _fig(9, 7)
        self._sobol_ax  = _ax(self._sobol_fig.add_subplot(111))
        self._sobol_cv  = _cv(self._sobol_fig)
        v.addWidget(self._sobol_cv)

        # Pearson vs Sobol comparison
        self._comp_fig = _fig(6, 4)
        self._comp_ax  = _ax(self._comp_fig.add_subplot(111))
        self._comp_cv  = _cv(self._comp_fig)
        v.addWidget(self._comp_cv)
        return w

    def _run_sobol(self):
        N      = int(self._sobol_n.currentText())
        use_dp = self._sobol_dp.isChecked()
        self._sobol_prog.setVisible(True); self._sobol_prog.setValue(0)
        self._sobol_thread = SobolThread(
            self.params_raw, self.param_cols,
            self.mean_pres, self.modes_pres, self.scores_pres,
            N, use_dp
        )
        self._sobol_thread.progress.connect(
            lambda cur, tot: self._sobol_prog.setValue(cur))
        self._sobol_thread.result.connect(self._on_sobol_done)
        self._sobol_thread.start()

    def _on_sobol_done(self, S1, ST):
        self._sobol_prog.setVisible(False)
        order = np.argsort(ST)[::-1][:15]
        labs  = [self.labels[i] for i in order]

        ax = self._sobol_ax; ax.cla(); _ax(ax)
        x = np.arange(len(order))
        ax.bar(x, S1[order], 0.6, label="S₁ (first-order)", color='#4EB3D3', alpha=0.85)
        ax.bar(x, np.maximum(ST[order] - S1[order], 0), 0.6,
               bottom=S1[order], label="Interaction (Sₜ−S₁)", color='#F4A261', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labs, rotation=45, ha='right',
                                              fontsize=8, color=FG)
        ax.set_ylabel("Sobol Index"); ax.set_ylim(0, 1)
        ax.set_title("Sobol Sensitivity Indices (top 15)", color=ACCENT)
        ax.legend(framealpha=0.3, labelcolor=FG, fontsize=9)
        self._sobol_cv.draw()

        # Pearson vs Sobol comparison
        absC = np.abs(self._corr)
        ax2 = self._comp_ax; ax2.cla(); _ax(ax2)
        ax2.scatter(absC[order], S1[order], s=40, color='#4EB3D3',
                    edgecolors='white', lw=0.4, zorder=3)
        for i, lbl in enumerate(labs):
            ax2.annotate(lbl.split("(")[0].strip(),
                         (absC[order[i]], S1[order[i]]),
                         fontsize=6, color='#aaa',
                         xytext=(3, 2), textcoords='offset points')
        lim = max(absC[order].max(), S1[order].max()) * 1.1
        ax2.plot([0, lim], [0, lim], 'w--', lw=0.8, label="1:1 line")
        ax2.set_xlabel("|Pearson r|"); ax2.set_ylabel("Sobol S₁")
        ax2.set_title("|Pearson r| vs Sobol S₁ — divergence reveals non-linear effects",
                      color=ACCENT)
        ax2.legend(framealpha=0.3, labelcolor=FG, fontsize=8)
        self._comp_cv.draw()

        self._sobol_status.setText(
            f"Done. Top parameter: {self.labels[order[0]]}  "
            f"(S₁={S1[order[0]]:.3f}, Sₜ={ST[order[0]]:.3f})"
        )
