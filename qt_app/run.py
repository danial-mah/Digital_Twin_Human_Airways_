"""
qt_app/run.py
=============
Entry point for the Human Airways Digital Twin desktop application.

Usage (from project root):
    python qt_app/run.py

Prerequisites:
    pip install -r requirements_qt.txt
    python scripts/precompute.py --pod-only    # run once for fast startup
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from qt_app.main_window import MainWindow


def main():
    # Enable high-DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,   True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("Human Airways Digital Twin")
    app.setOrganizationName("Tor Vergata")

    # Apply global stylesheet
    app.setStyleSheet("""
        QWidget        { background-color: #0f1119; color: #e0e0e0;
                         font-family: "Segoe UI", Arial, sans-serif; font-size: 11px; }
        QPushButton    { background: #252836; border: 1px solid #3a3d4e;
                         border-radius: 4px; padding: 5px 12px; }
        QPushButton:hover   { background: #2e3245; border-color: #4EB3D3; color: #4EB3D3; }
        QPushButton:pressed { background: #4EB3D3; color: #000; }
        QSlider::groove:horizontal {
            background: #252836; height: 4px; border-radius: 2px; }
        QSlider::handle:horizontal {
            background: #4EB3D3; width: 14px; height: 14px;
            border-radius: 7px; margin: -5px 0; }
        QSlider::sub-page:horizontal { background: #4EB3D3; border-radius: 2px; }
        QComboBox      { background: #252836; border: 1px solid #3a3d4e;
                         border-radius: 4px; padding: 3px 8px; }
        QComboBox::drop-down { border: none; }
        QGroupBox      { border: 1px solid #2a2a2a; border-radius: 4px;
                         margin-top: 6px; padding-top: 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #888; }
        QScrollBar:vertical  { background: #1a1d26; width: 8px; }
        QScrollBar::handle:vertical  { background: #3a3d4e; border-radius: 4px; }
        QCheckBox::indicator { width: 14px; height: 14px;
                               border: 1px solid #3a3d4e; background: #252836; }
        QCheckBox::indicator:checked { background: #4EB3D3; }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
