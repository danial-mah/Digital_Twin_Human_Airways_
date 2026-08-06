"""
qt_app/main_window.py
=====================
Main application window: loads precomputed data, then builds the three tabs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QLabel, QProgressBar, QApplication, QSizePolicy,
)
from PyQt5.QtGui import QPalette, QColor

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _dark_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.Window,          QColor("#0f1119"))
    p.setColor(QPalette.WindowText,      QColor("#e0e0e0"))
    p.setColor(QPalette.Base,            QColor("#1a1d26"))
    p.setColor(QPalette.AlternateBase,   QColor("#141720"))
    p.setColor(QPalette.Text,            QColor("#e0e0e0"))
    p.setColor(QPalette.Button,          QColor("#252836"))
    p.setColor(QPalette.ButtonText,      QColor("#e0e0e0"))
    p.setColor(QPalette.Highlight,       QColor("#4EB3D3"))
    p.setColor(QPalette.HighlightedText, QColor("#000000"))
    return p


class SplashWidget(QWidget):
    """Loading screen shown while data is being read."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)

        title = QLabel("🫁 Human Airways Digital Twin")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #4EB3D3;")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Loading precomputed data…")
        subtitle.setStyleSheet("font-size: 13px; color: #aaa;")
        subtitle.setAlignment(Qt.AlignCenter)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedWidth(400)
        self.progress.setStyleSheet("""
            QProgressBar { border: 1px solid #333; border-radius: 4px;
                           background: #1a1d26; height: 18px; }
            QProgressBar::chunk { background: #4EB3D3; border-radius: 3px; }
        """)

        self.status = QLabel("Initialising…")
        self.status.setStyleSheet("font-size: 11px; color: #888;")
        self.status.setAlignment(Qt.AlignCenter)

        lay.addWidget(title)
        lay.addSpacing(12)
        lay.addWidget(subtitle)
        lay.addSpacing(20)
        lay.addWidget(self.progress, alignment=Qt.AlignCenter)
        lay.addWidget(self.status)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Human Airways Digital Twin")
        self.resize(1400, 850)
        self.setPalette(_dark_palette())

        # Show splash while loading
        self._splash = SplashWidget(self)
        self.setCentralWidget(self._splash)
        self.show()
        QApplication.processEvents()

        self._load_data()
        self._build_tabs()

    # ── Data loading ──────────────────────────────────────────────────────────

    def _log(self, msg: str, pct: int):
        self._splash.status.setText(msg)
        self._splash.progress.setValue(pct)
        QApplication.processEvents()

    def _load_data(self):
        from core.data_io import (
            load_doe, load_ref_coords, load_all_pressures,
            load_precomputed_pod, load_precomputed_ref_coords,
            precomputed_available, STRIDE,
        )

        self._log("Loading DOE parameter table…", 5)
        self.doe_df     = load_doe()
        self.param_cols = [c for c in self.doe_df.columns
                           if c not in ("points", "snapshot_num")]

        if precomputed_available():
            self._log("Loading precomputed geometry POD…", 20)
            pre_g = load_precomputed_pod("geometry")
            self.mean_geo   = pre_g["mean"]
            self.modes_geo  = pre_g["modes"]
            self.scores_geo = pre_g["scores"]
            self.svals_geo  = pre_g["svalues"]

            self._log("Loading precomputed pressure POD…", 45)
            pre_p = load_precomputed_pod("pressure")
            self.mean_pres   = pre_p["mean"]
            self.modes_pres  = pre_p["modes"]
            self.scores_pres = pre_p["scores"]
            self.svals_pres  = pre_p["svalues"]

            self._log("Loading precomputed reference coordinates…", 60)
            self.ref_coords = load_precomputed_ref_coords()

        else:
            self._log("Precomputed data not found — loading from .bin files…", 10)
            from core.pod import compute_pod

            self._log("Loading reference coordinates…", 15)
            self.ref_coords = load_ref_coords(STRIDE)

            self._log("Loading all geometry snapshots (this takes ~30s)…", 20)
            from core.data_io import load_all_coords
            X_geo = load_all_coords(STRIDE)
            self._log("Computing geometry POD (SVD)…", 50)
            self.mean_geo, self.modes_geo, self.scores_geo, self.svals_geo = compute_pod(X_geo)

        # Always load raw pressure snapshots for the Pressure tab viewer
        self._log("Loading all pressure snapshots…", 70)
        from core.data_io import load_all_pressures
        self.all_pressures = load_all_pressures(STRIDE)

        if not hasattr(self, "mean_pres"):
            from core.pod import compute_pod
            self._log("Computing pressure POD…", 85)
            self.mean_pres, self.modes_pres, self.scores_pres, self.svals_pres = compute_pod(
                self.all_pressures
            )

        self._log("Loading settings…", 88)
        from core.data_io import load_settings
        self.settings = load_settings()

        self._log("Data ready!", 100)
        QApplication.processEvents()

    # ── Tab building ─────────────────────────────────────────────────────────

    def _build_tabs(self):
        from qt_app.geometry_tab      import GeometryTab
        from qt_app.pressure_tab      import PressureTab
        from qt_app.rbf_tab           import RBFTab
        from qt_app.doe_tab           import DOETab
        from qt_app.pod_tab           import PODTab
        from qt_app.regional_tab      import RegionalTab
        from qt_app.design_space_tab  import DesignSpaceTab
        from qt_app.assistant_tab     import AssistantTab

        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane   { border: 1px solid #2a2a2a; }
            QTabBar::tab       { background: #1a1d26; color: #aaa;
                                 padding: 8px 16px; border: 1px solid #2a2a2a;
                                 border-bottom: none; font-size: 11px; }
            QTabBar::tab:selected { background: #252836; color: #4EB3D3;
                                    border-top: 2px solid #4EB3D3; }
            QTabBar::tab:hover    { background: #252836; color: #fff; }
        """)
        tabs.setTabPosition(QTabWidget.North)
        tabs.setUsesScrollButtons(True)

        # ── Original 3 tabs ──────────────────────────────────────────────────
        geo_tab = GeometryTab(
            self.mean_geo, self.modes_geo, self.scores_geo, self.svals_geo
        )
        tabs.addTab(geo_tab, "🔬 Geometry")

        pres_tab = PressureTab(
            self.ref_coords, self.mean_pres, self.modes_pres,
            self.scores_pres, self.all_pressures, self.doe_df
        )
        tabs.addTab(pres_tab, "🌡️ Pressure")

        rbf_tab = RBFTab(
            self.ref_coords, self.mean_pres, self.modes_pres,
            self.scores_pres, self.svals_pres,
            self.doe_df, self.param_cols
        )
        tabs.addTab(rbf_tab, "🔮 RBF")

        # ── New analytics tabs ───────────────────────────────────────────────
        self._log("Building DOE Analysis tab…", 92)
        doe_tab = DOETab(self.doe_df, self.param_cols)
        tabs.addTab(doe_tab, "📐 DOE")

        self._log("Building POD Analysis tab…", 94)
        pod_tab = PODTab(
            self.mean_geo, self.modes_geo, self.svals_geo,
            self.mean_pres, self.modes_pres, self.scores_pres, self.svals_pres,
            self.ref_coords, self.all_pressures
        )
        tabs.addTab(pod_tab, "📊 POD")

        self._log("Building Regional Analysis tab…", 96)
        regional_tab = RegionalTab(self.settings)
        tabs.addTab(regional_tab, "🫀 Regional")

        self._log("Building Design Space tab…", 97)
        design_tab = DesignSpaceTab(
            self.doe_df, self.param_cols,
            self.mean_pres, self.modes_pres, self.scores_pres, self.svals_pres,
            self.scores_geo, self.all_pressures
        )
        tabs.addTab(design_tab, "🗺️ Design Space")

        self._log("Building AI Assistant tab…", 99)
        assistant_tab = AssistantTab()
        tabs.addTab(assistant_tab, "🤖 AI Assistant")

        self.setCentralWidget(tabs)
