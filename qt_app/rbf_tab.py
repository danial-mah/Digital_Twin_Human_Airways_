"""
qt_app/rbf_tab.py
=================
Real-time RBF surrogate inference tab.
Adjust 26 geometry parameter sliders → predicted pressure field updates live.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider,
    QPushButton, QGroupBox, QScrollArea, QSizePolicy, QComboBox,
)

from core.pod import reconstruct
from core.rbf import build_rbf, predict
from core.data_io import PARAM_LABELS, PARAM_GROUPS


class RBFTab(QWidget):
    """RBF surrogate inference tab with real-time 3D pressure preview."""

    def __init__(self, ref_coords, mean_pres, modes_pres, scores_pres, svals_pres,
                 doe_df, param_cols, parent=None):
        super().__init__(parent)
        self.ref_coords   = ref_coords
        self.mean_pres    = mean_pres
        self.modes_pres   = modes_pres
        self.scores_pres  = scores_pres
        self.svals_pres   = svals_pres
        self.doe_df       = doe_df
        self.param_cols   = param_cols

        # Prepare training data
        self.params_raw   = doe_df[param_cols].values.astype(float)
        self.low          = self.params_raw.min(axis=0)
        self.high         = self.params_raw.max(axis=0)
        self.params_norm  = (self.params_raw - self.low) / (self.high - self.low + 1e-12)

        # Default: modes covering 99% energy
        from core.pod import modes_for_energy
        k = modes_for_energy(svals_pres, 0.99)
        self._k_modes = k

        self._rbf = None
        self._new_params = self.params_raw.mean(axis=0).copy()

        # Debounce timer
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._predict_and_render)

        self._build_ui()
        self._build_rbf()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # Left panel: controls
        left = QWidget()
        left.setFixedWidth(340)
        left.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        # RBF settings
        rbf_grp = QGroupBox("RBF Settings")
        rbf_layout = QVBoxLayout(rbf_grp)
        self._kernel_combo = QComboBox()
        for k in ["thin_plate_spline", "multiquadric", "linear", "cubic"]:
            self._kernel_combo.addItem(k)
        rbf_layout.addWidget(QLabel("Kernel:"))
        rbf_layout.addWidget(self._kernel_combo)

        self._rebuild_btn = QPushButton("Rebuild RBF Surrogate")
        self._rebuild_btn.clicked.connect(self._build_rbf)
        rbf_layout.addWidget(self._rebuild_btn)

        self._status_label = QLabel("Building RBF…")
        self._status_label.setStyleSheet("color: #aaa; font-size: 10px;")
        rbf_layout.addWidget(self._status_label)
        left_layout.addWidget(rbf_grp)

        # Reset / load buttons
        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Reset to Mean")
        btn_reset.clicked.connect(self._reset_params)
        btn_row.addWidget(btn_reset)
        left_layout.addLayout(btn_row)

        # Parameter sliders in scrollable area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(4)

        self._sliders: dict[str, QSlider]  = {}
        self._val_labels: dict[str, QLabel] = {}

        for group_name, group_cols in PARAM_GROUPS.items():
            grp = QGroupBox(group_name)
            grp.setStyleSheet("QGroupBox { font-size: 10px; color: #aaa; }")
            grp_layout = QVBoxLayout(grp)

            for col in group_cols:
                if col not in self.param_cols:
                    continue
                i     = self.param_cols.index(col)
                c_min = float(self.params_raw[:, i].min())
                c_max = float(self.params_raw[:, i].max())
                c_val = float(self._new_params[i])
                label = PARAM_LABELS.get(col, col)

                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)

                lbl_name = QLabel(label.split("(")[0].strip()[:25])
                lbl_name.setFixedWidth(140)
                lbl_name.setStyleSheet("font-size: 9px; color: #ccc;")

                sldr = QSlider(Qt.Horizontal)
                sldr.setRange(0, 1000)
                span = c_max - c_min
                sldr.setValue(int((c_val - c_min) / (span + 1e-12) * 1000))

                val_lbl = QLabel(f"{c_val:.2f}")
                val_lbl.setFixedWidth(45)
                val_lbl.setStyleSheet("font-size: 9px; color: #4EB3D3;")

                def make_cb(col_name, c_min, c_max, val_lbl):
                    def cb(v):
                        value = c_min + v / 1000.0 * (c_max - c_min)
                        i = self.param_cols.index(col_name)
                        self._new_params[i] = value
                        val_lbl.setText(f"{value:.2f}")
                        self._timer.start()
                    return cb

                sldr.valueChanged.connect(make_cb(col, c_min, c_max, val_lbl))
                row_layout.addWidget(lbl_name)
                row_layout.addWidget(sldr)
                row_layout.addWidget(val_lbl)
                grp_layout.addWidget(row_widget)

                self._sliders[col]    = sldr
                self._val_labels[col] = val_lbl

            container_layout.addWidget(grp)

        scroll.setWidget(container)
        left_layout.addWidget(scroll, stretch=1)

        # Pressure metrics
        self._metrics_label = QLabel("Adjust sliders to predict.")
        self._metrics_label.setStyleSheet("color: #aaa; font-size: 10px;")
        self._metrics_label.setWordWrap(True)
        left_layout.addWidget(self._metrics_label)

        root.addWidget(left)

        # Right panel: PyVista 3D
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#0c0e14")
        root.addWidget(self.plotter.interactor, stretch=1)

    # ── RBF build and inference ───────────────────────────────────────────────

    def _build_rbf(self):
        kernel = self._kernel_combo.currentText()
        self._status_label.setText("Building RBF surrogate…")
        self._status_label.repaint()
        self._rbf = build_rbf(
            self.params_norm,
            self.scores_pres[:, :self._k_modes],
            kernel=kernel,
        )
        self._status_label.setText(
            f"RBF ready  ({self._k_modes} modes, {kernel})"
        )
        self._predict_and_render()

    def _predict_and_render(self):
        if self._rbf is None:
            return

        new_norm = ((self._new_params - self.low) / (self.high - self.low + 1e-12)).reshape(1, -1)
        pred_scores  = predict(self._rbf, new_norm)[0]
        pred_pressure = reconstruct(
            self.mean_pres, self.modes_pres[:, :self._k_modes], pred_scores
        )

        mesh = pv.PolyData(self.ref_coords)
        mesh["Pressure_Pa"] = pred_pressure

        self.plotter.clear()
        self.plotter.add_mesh(
            mesh, scalars="Pressure_Pa", cmap="jet",
            style="points", point_size=3, opacity=0.7,
            scalar_bar_args={"title": "P (Pa)", "vertical": True,
                             "position_x": 0.85, "width": 0.08},
        )
        self.plotter.reset_camera()

        self._metrics_label.setText(
            f"RBF-predicted pressure:\n"
            f"Min:  {pred_pressure.min():.1f} Pa\n"
            f"Max:  {pred_pressure.max():.1f} Pa\n"
            f"Mean: {pred_pressure.mean():.1f} Pa\n"
            f"Std:  {pred_pressure.std():.1f} Pa\n"
            f"ΔP:   {pred_pressure.max()-pred_pressure.min():.1f} Pa"
        )

    def _reset_params(self):
        mean_vals = self.params_raw.mean(axis=0)
        self._new_params = mean_vals.copy()
        for col, sldr in self._sliders.items():
            i     = self.param_cols.index(col)
            c_min = float(self.params_raw[:, i].min())
            c_max = float(self.params_raw[:, i].max())
            val   = float(mean_vals[i])
            slider_val = int((val - c_min) / (c_max - c_min + 1e-12) * 1000)
            sldr.blockSignals(True)
            sldr.setValue(slider_val)
            sldr.blockSignals(False)
            self._val_labels[col].setText(f"{val:.2f}")
        self._predict_and_render()
