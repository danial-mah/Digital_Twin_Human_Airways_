"""
qt_app/geometry_tab.py
======================
Real-time POD geometry explorer using PyVista + PyQt5.
Slider changes trigger a 50ms-debounced mesh update — effectively instantaneous.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider,
    QPushButton, QGroupBox, QScrollArea, QComboBox, QSizePolicy,
)

from core.pod import reconstruct, cumulative_energy, modes_for_energy


class GeometryTab(QWidget):
    """Interactive POD shape morphing tab."""

    N_SLIDERS = 10

    def __init__(self, mean_geo, modes_geo, scores_geo, svals_geo, parent=None):
        super().__init__(parent)
        self.mean_geo   = mean_geo
        self.modes_geo  = modes_geo
        self.scores_geo = scores_geo
        self.svals_geo  = svals_geo
        self.std_devs   = svals_geo / np.sqrt(max(len(scores_geo) - 1, 1))
        self.energy     = cumulative_energy(svals_geo)
        self.n95        = modes_for_energy(svals_geo, 0.95)
        self.n99        = modes_for_energy(svals_geo, 0.99)

        self._coeffs = np.zeros(len(svals_geo))

        # Debounce timer — updates 3D view 50 ms after last slider move
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._update_mesh)

        self._build_ui()
        self._update_mesh()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # Left panel: controls
        left = QWidget()
        left.setFixedWidth(300)
        left.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        # Info
        info = QLabel(
            f"<b>POD Modes</b><br>"
            f"95% energy: {self.n95} modes<br>"
            f"99% energy: {self.n99} modes<br>"
            f"Total modes: {len(self.svals_geo)}"
        )
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        left_layout.addWidget(info)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Reset to Mean")
        btn_reset.clicked.connect(self._reset)
        btn_random = QPushButton("Random Shape")
        btn_random.clicked.connect(self._random_shape)
        btn_row.addWidget(btn_reset)
        btn_row.addWidget(btn_random)
        left_layout.addLayout(btn_row)

        # Snapshot loader
        snap_box = QGroupBox("Load real snapshot")
        snap_layout = QVBoxLayout(snap_box)
        self._snap_combo = QComboBox()
        self._snap_combo.addItem("— choose —")
        for i in range(1, 101):
            self._snap_combo.addItem(f"Run {i}")
        self._snap_combo.currentIndexChanged.connect(self._load_snapshot)
        snap_layout.addWidget(self._snap_combo)
        left_layout.addWidget(snap_box)

        # Sliders in a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        slider_container = QWidget()
        self._slider_layout = QVBoxLayout(slider_container)
        self._slider_layout.setSpacing(4)
        self._sliders: list[QSlider] = []
        self._slider_labels: list[QLabel] = []

        for i in range(self.N_SLIDERS):
            ep = float(
                self.energy[i] * 100 if i == 0
                else (self.energy[i] - self.energy[i - 1]) * 100
            )
            grp = QGroupBox(f"Mode {i+1}  ({ep:.1f}%)")
            grp.setStyleSheet("QGroupBox { font-size: 11px; color: #ccc; }")
            g_layout = QVBoxLayout(grp)

            lbl = QLabel("0.000")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #4EB3D3; font-size: 11px;")

            sldr = QSlider(Qt.Horizontal)
            sldr.setRange(-1000, 1000)
            sldr.setValue(0)
            sldr.setTickInterval(100)
            sigma = float(self.std_devs[i])

            def make_cb(idx, sigma, lbl):
                def cb(val):
                    coeff = val / 1000.0 * 3.0 * sigma
                    self._coeffs[idx] = coeff
                    lbl.setText(f"{coeff:.3f}")
                    self._timer.start()
                return cb

            sldr.valueChanged.connect(make_cb(i, sigma, lbl))
            g_layout.addWidget(lbl)
            g_layout.addWidget(sldr)
            self._slider_layout.addWidget(grp)
            self._sliders.append(sldr)
            self._slider_labels.append(lbl)

        scroll_area.setWidget(slider_container)
        left_layout.addWidget(scroll_area, stretch=1)

        # Deviation metric
        self._dev_label = QLabel("Shape deviation ‖c‖: 0.000")
        self._dev_label.setStyleSheet("color: #aaa; font-size: 11px;")
        left_layout.addWidget(self._dev_label)

        root.addWidget(left)

        # Right panel: PyVista 3D
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#0c0e14")
        root.addWidget(self.plotter.interactor, stretch=1)

    # ── Mesh update ───────────────────────────────────────────────────────────

    def _update_mesh(self):
        rec = reconstruct(self.mean_geo, self.modes_geo, self._coeffs).reshape(-1, 3)
        mesh = pv.PolyData(rec)
        self.plotter.clear()
        self.plotter.add_mesh(
            mesh, style="points", point_size=3,
            color="#4EB3D3", opacity=0.6,
            render_points_as_spheres=False,
        )
        self.plotter.reset_camera()

        dev = float(np.linalg.norm(self._coeffs))
        self._dev_label.setText(f"Shape deviation ‖c‖: {dev:.3f}")

    # ── Button / combo callbacks ──────────────────────────────────────────────

    def _reset(self):
        self._coeffs[:] = 0.0
        for s in self._sliders:
            s.blockSignals(True)
            s.setValue(0)
            s.blockSignals(False)
        for lbl in self._slider_labels:
            lbl.setText("0.000")
        self._update_mesh()

    def _random_shape(self):
        rng = np.random.default_rng()
        for i, (s, lbl) in enumerate(zip(self._sliders, self._slider_labels)):
            sigma = float(self.std_devs[i])
            coeff = float(rng.uniform(-2.0 * sigma, 2.0 * sigma))
            self._coeffs[i] = coeff
            slider_val = int(np.clip(coeff / (3.0 * sigma) * 1000, -1000, 1000))
            s.blockSignals(True)
            s.setValue(slider_val)
            s.blockSignals(False)
            lbl.setText(f"{coeff:.3f}")
        self._update_mesh()

    def _load_snapshot(self, index: int):
        if index == 0:
            return
        snap_idx = index - 1
        for i, (s, lbl) in enumerate(zip(self._sliders, self._slider_labels)):
            if i < self.scores_geo.shape[1]:
                coeff = float(self.scores_geo[snap_idx, i])
                self._coeffs[i] = coeff
                sigma = float(self.std_devs[i])
                if sigma > 0:
                    slider_val = int(np.clip(coeff / (3.0 * sigma) * 1000, -1000, 1000))
                else:
                    slider_val = 0
                s.blockSignals(True)
                s.setValue(slider_val)
                s.blockSignals(False)
                lbl.setText(f"{coeff:.3f}")
        self._update_mesh()
