"""Screenshot collector with a window (GUI version of collect_dataset.py).

    python collector_gui.py

Buttons or global hotkeys (work while Roblox is focused):  F6 = auto-capture on/off,  F7 = take one screenshot.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices, QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QVBoxLayout, QWidget)

from capture import ScreenGrabber
from collect_dataset import Collector
from config import RAW_IMAGES_DIR, Config
from hotkeys import KeyEdge

STYLE = """
QWidget { font-size: 13px; }
QPushButton { padding: 9px 14px; border-radius: 6px; background: #3a3f4b; color: white; }
QPushButton:hover { background: #4b5263; }
QPushButton#primary { background: #2e7d32; font-weight: bold; font-size: 15px; }
QPushButton#primary:hover { background: #388e3c; }
QPushButton#recording { background: #c62828; font-weight: bold; }
QLabel#preview { background: #15171c; border: 1px solid #3a3f4b; border-radius: 6px; color: #888; }
QLabel#status { color: #aab; }
"""


def bgr_to_pixmap(image) -> QPixmap:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


class CollectorWindow(QWidget):
    def __init__(self, cfg: Config, out_dir: Path = RAW_IMAGES_DIR, grabber: Optional[ScreenGrabber] = None):
        super().__init__()
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.grabber = grabber or ScreenGrabber(cfg)
        self.collector: Optional[Collector] = None
        self.auto = False
        self._saved_total = 0

        self.setWindowTitle("Dataset Collector")  # must not contain the Roblox window title
        self.setStyleSheet(STYLE)
        self.setMinimumWidth(460)

        self.preview = QLabel("Preview appears here after the first screenshot")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(420, 236)

        self.session = QLineEdit(datetime.now().strftime("session%m%d-%H%M"))
        self.interval = QDoubleSpinBox()
        self.interval.setRange(0.3, 30.0)
        self.interval.setValue(1.0)
        self.interval.setSingleStep(0.5)
        self.interval.setSuffix(" s")
        self.min_change = QDoubleSpinBox()
        self.min_change.setRange(0.0, 50.0)
        self.min_change.setValue(4.0)
        self.min_change.setToolTip("Skip a frame if it looks almost identical to the last saved one (0 = keep all)")

        form = QFormLayout()
        form.addRow("Session name", self.session)
        form.addRow("Seconds between shots", self.interval)
        form.addRow("Skip duplicates (0 = off)", self.min_change)

        self.btn_shot = QPushButton("Take screenshot   (F7)")
        self.btn_shot.setObjectName("primary")
        self.btn_shot.clicked.connect(self.take_one)
        self.btn_auto = QPushButton("Start auto capture   (F6)")
        self.btn_auto.clicked.connect(self.toggle_auto)
        self.btn_open = QPushButton("Open folder")
        self.btn_open.clicked.connect(self.open_folder)

        row = QHBoxLayout()
        row.addWidget(self.btn_auto, 1)
        row.addWidget(self.btn_open)

        self.status = QLabel("Ready.")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self.counter = QLabel()

        lay = QVBoxLayout(self)
        lay.addWidget(self.preview)
        lay.addLayout(form)
        lay.addWidget(self.btn_shot)
        lay.addLayout(row)
        lay.addWidget(self.counter)
        lay.addWidget(self.status)

        self._key_auto, self._key_shot = KeyEdge("F6"), KeyEdge("F7")
        self._poll = QTimer(self)
        self._poll.setInterval(25)
        self._poll.timeout.connect(self._poll_keys)
        self._poll.start()
        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(self._auto_tick)
        self._refresh_counter()

    # -- helpers -------------------------------------------------------------
    def _ensure_collector(self) -> Collector:
        name = self.session.text().strip().replace("_", "-").replace(" ", "-") or "session"
        if self.collector is None or self.collector.session != name:
            self.collector = Collector(self.grabber, self.out_dir, name, min_change=self.min_change.value())
        self.collector.min_change = self.min_change.value()
        return self.collector

    def _refresh_counter(self) -> None:
        total = len(list(self.out_dir.glob("*.jpg"))) if self.out_dir.exists() else 0
        self.counter.setText(f"Saved this run: {self._saved_total}   |   Total images in folder: {total}   (aim for 500+)")

    def _capture(self, force: bool) -> None:
        c = self._ensure_collector()
        path = c.capture(force=force)
        if c.last_frame is not None:
            pm = bgr_to_pixmap(c.last_frame.image).scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                         Qt.TransformationMode.SmoothTransformation)
            self.preview.setPixmap(pm)
        if path:
            self._saved_total += 1
            self.status.setText(f"Saved {path.name}")
        elif c.last_frame is None:
            self.status.setText(f"Cannot see the game: {self.grabber.status}. Open Roblox in windowed/borderless mode.")
        else:
            self.status.setText("Skipped: nothing changed since the last screenshot.")
        self._refresh_counter()

    # -- actions -------------------------------------------------------------
    def take_one(self) -> None:
        self._capture(force=True)

    def toggle_auto(self) -> None:
        self.auto = not self.auto
        if self.auto:
            self._auto_timer.start(int(self.interval.value() * 1000))
            self.btn_auto.setText("Stop auto capture   (F6)")
            self.btn_auto.setObjectName("recording")
            self.status.setText("Auto capture running... move around and look at different scenes.")
        else:
            self._auto_timer.stop()
            self.btn_auto.setText("Start auto capture   (F6)")
            self.btn_auto.setObjectName("")
            self.status.setText("Auto capture stopped.")
        self.btn_auto.style().unpolish(self.btn_auto)
        self.btn_auto.style().polish(self.btn_auto)

    def _auto_tick(self) -> None:
        self._auto_timer.setInterval(int(self.interval.value() * 1000))
        self._capture(force=False)

    def _poll_keys(self) -> None:
        if self._key_auto.pressed():
            self.toggle_auto()
        if self._key_shot.pressed():
            self.take_one()

    def open_folder(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.out_dir)))

    def closeEvent(self, event) -> None:  # noqa: N802
        self._auto_timer.stop()
        self._poll.stop()
        self.grabber.close()
        super().closeEvent(event)


def main() -> int:
    from capture import enable_dpi_awareness
    app = QApplication(sys.argv)  # Qt sets DPI awareness itself
    cfg = Config.load()
    win = CollectorWindow(cfg)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
