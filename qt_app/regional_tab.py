"""
qt_app/regional_tab.py
======================
Regional Analysis tab — mirrors pages/6_Regional_Analysis.py.

Sub-tabs:
  Per-Snapshot  — bar chart of mean±std per anatomical region for a chosen run
  All Snapshots — heatmap 100 runs × regions (computed on demand via QThread)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QTabWidget, QSizePolicy, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QRadioButton,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from core.data_io import (
    STRIDE, load_pressure_snapshot, load_settings,
    get_anatomical_regions, region_slice,
)

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


class HeatmapThread(QThread):
    result    = pyqtSignal(object)   # pd.DataFrame
    progress  = pyqtSignal(int)

    def __init__(self, settings, regions):
        super().__init__()
        self._settings = settings; self._regions = regions

    def run(self):
        data = {r: [] for r in self._regions}
        for i in range(1, 101):
            pressure = load_pressure_snapshot(i, STRIDE)
            for r in self._regions:
                sl  = region_slice(self._settings, r, STRIDE)
                p_r = pressure[sl]
                data[r].append(float(p_r.mean()) if len(p_r) > 0 else float('nan'))
            self.progress.emit(i)
        df = pd.DataFrame(data, index=range(1, 101))
        self.result.emit(df)


class RegionalTab(QWidget):

    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._regions  = get_anatomical_regions(settings)
        self._heatmap_df: pd.DataFrame | None = None
        self._heatmap_thread = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self); root.setContentsMargins(6, 6, 6, 6)

        # Snapshot control
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Snapshot:"))
        self._snap_sl = QSlider(Qt.Horizontal)
        self._snap_sl.setRange(1, 100); self._snap_sl.setValue(1)
        self._snap_sl.setFixedWidth(200)
        self._snap_lbl = QLabel("1")
        self._snap_sl.valueChanged.connect(lambda v: (
            self._snap_lbl.setText(str(v)), self._update_bar()))
        ctrl.addWidget(self._snap_sl); ctrl.addWidget(self._snap_lbl)
        ctrl.addWidget(QLabel("   Sort by:"))
        self._sort_mean = QRadioButton("Mean pressure"); self._sort_mean.setChecked(True)
        self._sort_anat = QRadioButton("Anatomical order")
        ctrl.addWidget(self._sort_mean); ctrl.addWidget(self._sort_anat)
        self._sort_mean.toggled.connect(lambda: self._update_bar())
        ctrl.addStretch()
        root.addLayout(ctrl)

        tabs = QTabWidget(); tabs.setStyleSheet(TAB_SS)
        tabs.addTab(self._tab_snap(),  "📍 Per-Snapshot")
        tabs.addTab(self._tab_heat(),  "🗺️ All Snapshots")
        root.addWidget(tabs)

    # ── Per-Snapshot bar chart ────────────────────────────────────────────────

    def _tab_snap(self) -> QWidget:
        spl = QSplitter(Qt.Horizontal)

        # Left: table
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["Region", "Mean (Pa)", "Std (Pa)", "Min (Pa)", "Max (Pa)"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setMaximumWidth(400)
        self._table.setStyleSheet(
            "QTableWidget { background:#1a1d26; color:#e0e0e0; gridline-color:#2a2d3a; }"
            "QHeaderView::section { background:#252836; color:#4EB3D3; }"
        )
        spl.addWidget(self._table)

        # Right: chart
        right = QWidget(); rv = QVBoxLayout(right); rv.setContentsMargins(0, 0, 0, 0)
        self._bar_fig = _fig(9, 5)
        self._bar_ax  = _ax(self._bar_fig.add_subplot(111))
        self._bar_cv  = _cv(self._bar_fig)
        rv.addWidget(self._bar_cv)
        spl.addWidget(right)
        spl.setStretchFactor(0, 1); spl.setStretchFactor(1, 3)

        self._update_bar()
        return spl

    def _update_bar(self):
        snap = self._snap_sl.value()
        pressure = load_pressure_snapshot(snap, STRIDE)
        rows = []
        for reg in self._regions:
            sl  = region_slice(self._settings, reg, STRIDE)
            p_r = pressure[sl]
            if len(p_r) == 0: continue
            rows.append({
                "region": reg,
                "mean":   float(p_r.mean()),
                "std":    float(p_r.std()),
                "min":    float(p_r.min()),
                "max":    float(p_r.max()),
            })
        df = pd.DataFrame(rows)
        if self._sort_mean.isChecked():
            df = df.sort_values("mean", ascending=False).reset_index(drop=True)

        # Update table
        self._table.setRowCount(len(df))
        for i, row in df.iterrows():
            self._table.setItem(i, 0, QTableWidgetItem(row["region"]))
            for j, col in enumerate(["mean", "std", "min", "max"], 1):
                self._table.setItem(i, j, QTableWidgetItem(f"{row[col]:.2f}"))

        # Update bar chart
        ax = self._bar_ax; ax.cla(); _ax(ax)
        vals = df["mean"].values; errs = df["std"].values
        norm = mcolors.Normalize(vals.min(), vals.max())
        cmap = cm.get_cmap('RdYlBu_r')
        colors = [cmap(norm(v)) for v in vals]
        ax.bar(range(len(df)), vals, yerr=errs, color=colors,
               error_kw=dict(ecolor='#888', lw=0.8, capsize=2))
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["region"].tolist(), rotation=55, ha='right',
                           fontsize=7, color=FG)
        ax.set_ylabel("Mean Pressure (Pa)")
        ax.set_title(f"Regional Pressure — Snapshot {snap}", color=ACCENT)
        sm = cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)
        sm.set_array([]); cb = self._bar_fig.colorbar(sm, ax=ax, shrink=0.8)
        cb.set_label("Mean P (Pa)", color=FG, fontsize=8)
        cb.ax.tick_params(labelcolor=FG, labelsize=7)
        self._bar_cv.draw()

    # ── All-snapshots heatmap ─────────────────────────────────────────────────

    def _tab_heat(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)

        self._heat_btn = QPushButton("Compute Heatmap (all 100 snapshots — ~30 s)")
        self._heat_btn.clicked.connect(self._start_heatmap)
        v.addWidget(self._heat_btn)

        self._heat_prog = QProgressBar()
        self._heat_prog.setRange(0, 100); self._heat_prog.setVisible(False)
        v.addWidget(self._heat_prog)

        self._heat_fig = _fig(12, 7)
        self._heat_ax  = _ax(self._heat_fig.add_subplot(111))
        self._heat_cv  = _cv(self._heat_fig)
        v.addWidget(self._heat_cv)
        return w

    def _start_heatmap(self):
        self._heat_btn.setEnabled(False)
        self._heat_prog.setVisible(True); self._heat_prog.setValue(0)
        self._heatmap_thread = HeatmapThread(self._settings, self._regions)
        self._heatmap_thread.progress.connect(self._heat_prog.setValue)
        self._heatmap_thread.result.connect(self._on_heatmap_done)
        self._heatmap_thread.start()

    def _on_heatmap_done(self, df: pd.DataFrame):
        self._heatmap_df = df
        self._heat_prog.setVisible(False)
        self._heat_btn.setText("Recompute Heatmap")
        self._heat_btn.setEnabled(True)

        ax = self._heat_ax; ax.cla()
        ax.set_facecolor(AXES_BG)
        im = ax.imshow(df.T.values, cmap='jet', aspect='auto')
        cb = self._heat_fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("Mean P (Pa)", color=FG, fontsize=8)
        cb.ax.tick_params(labelcolor=FG, labelsize=7)
        ax.set_xticks(range(0, 100, 5))
        ax.set_xticklabels(range(1, 101, 5), fontsize=7, color=FG)
        ax.set_yticks(range(len(self._regions)))
        ax.set_yticklabels(self._regions, fontsize=6.5, color=FG)
        ax.tick_params(colors=FG)
        ax.set_xlabel("Snapshot (run)", color=FG)
        ax.set_title("Mean Pressure Heatmap — All 100 Runs × Anatomical Regions",
                     color=ACCENT)
        self._heat_cv.draw()
