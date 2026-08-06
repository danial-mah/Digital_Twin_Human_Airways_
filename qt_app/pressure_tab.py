"""
qt_app/pressure_tab.py
======================
3D pressure field viewer — select any of the 100 DOE snapshots and instantly
see the coloured pressure distribution on the airway mesh.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider,
    QGroupBox, QComboBox, QCheckBox, QSizePolicy,
)

from core.pod import reconstruct


class PressureTab(QWidget):
    """Snapshot pressure field viewer tab."""

    def __init__(self, ref_coords, mean_pres, modes_pres, scores_pres,
                 all_pressures, doe_df, parent=None):
        super().__init__(parent)
        self.ref_coords   = ref_coords
        self.mean_pres    = mean_pres
        self.modes_pres   = modes_pres
        self.scores_pres  = scores_pres
        self.all_pressures = all_pressures   # (100, N) raw pressure fields
        self.doe_df       = doe_df

        self._current_snap = 0
        self._build_ui()
        self._show_snapshot(0)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # Left panel
        left = QWidget()
        left.setFixedWidth(260)
        left.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        # Snapshot selector
        snap_grp = QGroupBox("DOE Snapshot")
        snap_layout = QVBoxLayout(snap_grp)

        self._snap_slider = QSlider(Qt.Horizontal)
        self._snap_slider.setRange(1, 100)
        self._snap_slider.setValue(1)
        self._snap_slider.setTickInterval(10)
        self._snap_slider.setTickPosition(QSlider.TicksBelow)
        self._snap_slider.valueChanged.connect(self._on_snap_change)

        self._snap_label = QLabel("Run 1")
        self._snap_label.setAlignment(Qt.AlignCenter)
        self._snap_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4EB3D3;")

        snap_layout.addWidget(self._snap_label)
        snap_layout.addWidget(self._snap_slider)
        left_layout.addWidget(snap_grp)

        # Colormap selector
        cmap_grp = QGroupBox("Colour map")
        cmap_layout = QVBoxLayout(cmap_grp)
        self._cmap_combo = QComboBox()
        for cm in ["jet", "viridis", "plasma", "hot", "coolwarm", "turbo"]:
            self._cmap_combo.addItem(cm)
        self._cmap_combo.currentTextChanged.connect(self._refresh_plot)
        cmap_layout.addWidget(self._cmap_combo)
        left_layout.addWidget(cmap_grp)

        # Options
        opts_grp = QGroupBox("Options")
        opts_layout = QVBoxLayout(opts_grp)
        self._show_scalar_bar = QCheckBox("Show scalar bar")
        self._show_scalar_bar.setChecked(True)
        self._show_scalar_bar.stateChanged.connect(self._refresh_plot)
        opts_layout.addWidget(self._show_scalar_bar)
        left_layout.addWidget(opts_grp)

        # Metrics
        self._metrics_label = QLabel()
        self._metrics_label.setStyleSheet("color: #aaa; font-size: 10px;")
        self._metrics_label.setWordWrap(True)
        left_layout.addWidget(self._metrics_label)

        # DOE params display
        self._params_label = QLabel()
        self._params_label.setStyleSheet("color: #888; font-size: 10px;")
        self._params_label.setWordWrap(True)
        left_layout.addWidget(self._params_label)

        left_layout.addStretch()
        root.addWidget(left)

        # Right panel: PyVista 3D
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#0c0e14")
        root.addWidget(self.plotter.interactor, stretch=1)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_snap_change(self, value: int):
        self._current_snap = value - 1
        self._snap_label.setText(f"Run {value}")
        self._show_snapshot(self._current_snap)

    def _refresh_plot(self):
        self._show_snapshot(self._current_snap)

    def _show_snapshot(self, idx: int):
        pressure = self.all_pressures[idx]
        coords   = self.ref_coords
        cmap     = self._cmap_combo.currentText()

        mesh = pv.PolyData(coords)
        mesh["Pressure_Pa"] = pressure

        self.plotter.clear()
        self.plotter.add_mesh(
            mesh, scalars="Pressure_Pa", cmap=cmap,
            style="points", point_size=3, opacity=0.7,
            scalar_bar_args={"title": "P (Pa)", "vertical": True,
                             "position_x": 0.85, "width": 0.08},
            show_scalar_bar=self._show_scalar_bar.isChecked(),
        )
        self.plotter.reset_camera()

        # Update metrics
        self._metrics_label.setText(
            f"Min:  {pressure.min():.1f} Pa\n"
            f"Max:  {pressure.max():.1f} Pa\n"
            f"Mean: {pressure.mean():.1f} Pa\n"
            f"Std:  {pressure.std():.1f} Pa\n"
            f"ΔP:   {pressure.max() - pressure.min():.1f} Pa"
        )

        # Update DOE params
        if self.doe_df is not None:
            row = self.doe_df.iloc[idx]
            params_str = "\n".join(
                f"{c}: {row[c]:.2f}"
                for c in ["A_glotis", "d_trachea", "l_trachea",
                          "teta_branch_trachea", "l_l", "l_r"]
                if c in row.index
            )
            self._params_label.setText(f"Key parameters:\n{params_str}")
