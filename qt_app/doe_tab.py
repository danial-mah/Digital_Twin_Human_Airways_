"""
qt_app/doe_tab.py
=================
DOE Analysis tab — mirrors pages/3_DOE_Analysis.py.
All charts use Matplotlib embedded via FigureCanvasQTAgg.

Sub-tabs:
  Parallel Coordinates  — all 26 parameters, colour = run number
  Pairwise Scatter      — any two params with OLS trendline + highlighted run
  Correlation Heatmap   — 26×26 Pearson correlation matrix
  LHS Coverage          — original 100 DOE vs 1000 LHS virtual shapes
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox, QPushButton, QTabWidget, QSizePolicy, QScrollArea,
)

from core.data_io import PARAM_LABELS, get_param_cols
from core.lhs import latin_hypercube, doe_bounds, discrepancy_score

BG      = '#0f1119'
AXES_BG = '#1a1d26'
FG      = '#e0e0e0'
ACCENT  = '#4EB3D3'
GRID    = '#2a2d3a'
TAB_SS  = """
    QTabWidget::pane   { border: 1px solid #2a2a2a; }
    QTabBar::tab       { background: #1a1d26; color: #aaa;
                         padding: 6px 14px; border: 1px solid #2a2a2a;
                         border-bottom: none; font-size: 11px; }
    QTabBar::tab:selected { background: #252836; color: #4EB3D3;
                            border-top: 2px solid #4EB3D3; }
    QTabBar::tab:hover    { background: #252836; color: #fff; }
"""


def _fig(w=9, h=5) -> Figure:
    return Figure(figsize=(w, h), facecolor=BG, tight_layout=True)


def _ax(ax):
    ax.set_facecolor(AXES_BG)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(ACCENT)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, alpha=0.5, lw=0.5)
    return ax


def _canvas(fig: Figure) -> FigureCanvas:
    c = FigureCanvas(fig)
    c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    c.setMinimumHeight(320)
    return c


class DOETab(QWidget):

    def __init__(self, doe_df: pd.DataFrame, param_cols: list, parent=None):
        super().__init__(parent)
        self.doe_df     = doe_df
        self.param_cols = param_cols
        self.labels     = [PARAM_LABELS.get(c, c) for c in param_cols]
        self._lhs_cache: np.ndarray | None = None
        self._build_ui()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # Highlight-run control
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Highlight run:"))
        self._snap_sl = QSlider(Qt.Horizontal)
        self._snap_sl.setRange(1, 100)
        self._snap_sl.setValue(1)
        self._snap_sl.setFixedWidth(200)
        self._snap_lbl = QLabel("1")
        self._snap_sl.valueChanged.connect(lambda v: (
            self._snap_lbl.setText(str(v)),
            self._update_scatter(),
        ))
        ctrl.addWidget(self._snap_sl)
        ctrl.addWidget(self._snap_lbl)
        ctrl.addStretch()
        root.addLayout(ctrl)

        tabs = QTabWidget()
        tabs.setStyleSheet(TAB_SS)
        tabs.addTab(self._tab_parallel(),  "📊 Parallel Coords")
        tabs.addTab(self._tab_scatter(),   "🔵 Pairwise Scatter")
        tabs.addTab(self._tab_heatmap(),   "🔥 Correlation Heatmap")
        tabs.addTab(self._tab_lhs(),       "🎯 LHS Coverage")
        root.addWidget(tabs)

    # ── Tab 1: Parallel coordinates ───────────────────────────────────────────

    def _tab_parallel(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        data = self.doe_df[self.param_cols]
        data_n = (data - data.min()) / (data.max() - data.min() + 1e-12)
        n = len(self.param_cols)

        fig = _fig(11, 5)
        ax  = _ax(fig.add_subplot(111))

        cmap_v = cm.get_cmap('viridis')
        for i, (_, row) in enumerate(data_n.iterrows()):
            color = cmap_v(i / max(len(data_n) - 1, 1))
            ax.plot(range(n), row.values, color=color, alpha=0.35, lw=0.7)

        ax.set_xticks(range(n))
        ax.set_xticklabels(self.labels, rotation=55, ha='right', fontsize=6, color=FG)
        ax.set_ylabel("Normalised value", color=FG)
        ax.set_title("All 100 DOE Runs — 26 Parameters (colour = run #)", color=ACCENT)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='x', color=GRID, alpha=0.7, lw=0.6)

        sm = cm.ScalarMappable(cmap='viridis', norm=mcolors.Normalize(1, 100))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.01)
        cbar.set_label("Run #", color=FG, fontsize=8)
        cbar.ax.tick_params(labelcolor=FG, labelsize=7)

        v.addWidget(_canvas(fig))
        return w

    # ── Tab 2: Pairwise scatter ───────────────────────────────────────────────

    def _tab_scatter(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("X:"))
        self._sx = QComboBox(); self._sy = QComboBox()
        for lbl in self.labels:
            self._sx.addItem(lbl); self._sy.addItem(lbl)
        self._sx.setCurrentIndex(0); self._sy.setCurrentIndex(3)
        ctrl.addWidget(self._sx)
        ctrl.addWidget(QLabel("  Y:")); ctrl.addWidget(self._sy)
        btn = QPushButton("Plot"); btn.setFixedWidth(60)
        btn.clicked.connect(self._update_scatter)
        ctrl.addWidget(btn); ctrl.addStretch()
        v.addLayout(ctrl)

        self._sc_fig = _fig(8, 5)
        self._sc_ax  = _ax(self._sc_fig.add_subplot(111))
        self._sc_cv  = _canvas(self._sc_fig)
        v.addWidget(self._sc_cv)

        self._corr_lbl = QLabel("")
        self._corr_lbl.setStyleSheet(f"color:{ACCENT}; font-size:12px; padding:4px;")
        v.addWidget(self._corr_lbl)

        self._sx.currentIndexChanged.connect(self._update_scatter)
        self._sy.currentIndexChanged.connect(self._update_scatter)
        self._update_scatter()
        return w

    def _update_scatter(self):
        xi = self._sx.currentIndex(); yi = self._sy.currentIndex()
        xc = self.param_cols[xi];     yc = self.param_cols[yi]
        xd = self.doe_df[xc].values.astype(float)
        yd = self.doe_df[yc].values.astype(float)

        ax = self._sc_ax; ax.cla(); _ax(ax)
        sc = ax.scatter(xd, yd, c=self.doe_df["snapshot_num"].values,
                        cmap='viridis', s=38, zorder=3,
                        edgecolors='white', linewidths=0.3)
        cb = self._sc_fig.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label("Run #", color=FG, fontsize=8)
        cb.ax.tick_params(labelcolor=FG, labelsize=7)

        m, b = np.polyfit(xd, yd, 1)
        xl = np.linspace(xd.min(), xd.max(), 50)
        ax.plot(xl, m * xl + b, 'w--', lw=1.2, label="OLS")

        h = self._snap_sl.value()
        hr = self.doe_df[self.doe_df["snapshot_num"] == h].iloc[0]
        ax.scatter([hr[xc]], [hr[yc]], color='red', s=200, marker='*',
                   zorder=5, label=f"Run {h}")

        ax.set_xlabel(self.labels[xi]); ax.set_ylabel(self.labels[yi])
        ax.set_title(f"{self.labels[xi]}  vs  {self.labels[yi]}", color=ACCENT)
        ax.legend(framealpha=0.3, labelcolor=FG, fontsize=8)
        r = float(np.corrcoef(xd, yd)[0, 1])
        self._corr_lbl.setText(f"Pearson r = {r:.4f}")
        self._sc_cv.draw()

    # ── Tab 3: Correlation heatmap ────────────────────────────────────────────

    def _tab_heatmap(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        corr = self.doe_df[self.param_cols].corr().values
        n    = len(self.labels)

        fig = _fig(10, 9)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(AXES_BG)

        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        cb = fig.colorbar(im, ax=ax, shrink=0.85, label="Pearson r")
        cb.ax.tick_params(labelcolor=FG, labelsize=7)
        cb.ax.yaxis.label.set_color(FG)

        ax.set_xticks(range(n))
        ax.set_xticklabels(self.labels, rotation=55, ha='right', fontsize=6, color=FG)
        ax.set_yticks(range(n))
        ax.set_yticklabels(self.labels, fontsize=6, color=FG)
        ax.tick_params(colors=FG)
        ax.set_title("Pearson Correlation Matrix — 26 DOE Parameters", color=ACCENT)

        for i in range(n):
            for j in range(n):
                val = corr[i, j]
                if abs(val) > 0.3 and i != j:
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                            fontsize=4.5, color='white' if abs(val) > 0.6 else '#111')

        v.addWidget(_canvas(fig))
        return w

    # ── Tab 4: LHS Coverage ───────────────────────────────────────────────────

    def _tab_lhs(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("X:"))
        self._lx = QComboBox(); self._ly = QComboBox()
        for lbl in self.labels:
            self._lx.addItem(lbl); self._ly.addItem(lbl)
        self._lx.setCurrentIndex(0); self._ly.setCurrentIndex(3)
        ctrl.addWidget(self._lx)
        ctrl.addWidget(QLabel("  Y:")); ctrl.addWidget(self._ly)
        btn = QPushButton("Plot"); btn.setFixedWidth(60)
        btn.clicked.connect(self._update_lhs)
        ctrl.addWidget(btn); ctrl.addStretch()
        v.addLayout(ctrl)

        self._lhs_fig = _fig(8, 5)
        self._lhs_ax  = _ax(self._lhs_fig.add_subplot(111))
        self._lhs_cv  = _canvas(self._lhs_fig)
        v.addWidget(self._lhs_cv)

        self._disc_lbl = QLabel("")
        self._disc_lbl.setStyleSheet(f"color:{FG}; font-size:11px; padding:4px;")
        v.addWidget(self._disc_lbl)

        self._lx.currentIndexChanged.connect(self._update_lhs)
        self._ly.currentIndexChanged.connect(self._update_lhs)
        self._ensure_lhs()
        self._update_lhs()
        return w

    def _ensure_lhs(self):
        if self._lhs_cache is None:
            low, high = doe_bounds(self.doe_df, self.param_cols)
            self._lhs_cache = latin_hypercube(1000, low, high, seed=42)
            self._lhs_low   = low
            self._lhs_high  = high

    def _update_lhs(self):
        self._ensure_lhs()
        xi = self._lx.currentIndex(); yi = self._ly.currentIndex()
        xc = self.param_cols[xi];     yc = self.param_cols[yi]

        ax = self._lhs_ax; ax.cla(); _ax(ax)
        ax.scatter(self._lhs_cache[:, xi], self._lhs_cache[:, yi],
                   s=4, color='#aec7e8', alpha=0.5, label="LHS 1000 virtual")
        ax.scatter(self.doe_df[xc].values, self.doe_df[yc].values,
                   s=60, color='red', facecolors='none',
                   linewidths=1.5, label="Original DOE 100")
        ax.set_xlabel(self.labels[xi]); ax.set_ylabel(self.labels[yi])
        ax.set_title("Coverage: LHS virtual (blue) vs DOE (red)", color=ACCENT)
        ax.legend(framealpha=0.3, labelcolor=FG, fontsize=8)
        self._lhs_cv.draw()

        d_doe = discrepancy_score(self.doe_df[self.param_cols].values)
        d_lhs = discrepancy_score(self._lhs_cache)
        self._disc_lbl.setText(
            f"L2-discrepancy — DOE (100): {d_doe:.5f}  |  LHS (1000): {d_lhs:.5f}  "
            f"(lower = better coverage)"
        )
