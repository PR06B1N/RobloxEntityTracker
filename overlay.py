"""PyQt6 transparent overlay + control panel (Stage 7).

`OverlayWindow`  - click-through, always-on-top, transparent window placed exactly over
                   the Roblox client area. Draws boxes, temporary IDs and the live counter.
`ControlPanel`   - small normal window with the live settings (opacity, confidence, FPS, ...).
`HotkeyWatcher`  - polls the global toggle key (F8 by default).
"""
from __future__ import annotations

import ctypes
import logging
import sys
from typing import Optional

from PyQt6.QtCore import QObject, QRect, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QKeySequence, QPainter, QPen, QShortcut
from PyQt6.QtWidgets import (QCheckBox, QFormLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
                             QSpinBox, QVBoxLayout, QWidget)

from capture import Region
from config import Config
from hotkeys import KeyEdge
from pipeline import FrameResult

log = logging.getLogger(__name__)
IS_WINDOWS = sys.platform == "win32"

_HWND_TOPMOST = -1
_SWP_NOACTIVATE = 0x0010
_WDA_EXCLUDEFROMCAPTURE = 0x11  # Windows 10 version 2004+


def color_for_id(track_id: int) -> QColor:
    """Stable, well-separated colour per ID (golden-ratio hue spacing)."""
    return QColor.fromHsvF((track_id * 0.61803398875) % 1.0, 0.85, 1.0)


class OverlayWindow(QWidget):
    def __init__(self, cfg: Config):
        flags = (Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                 | Qt.WindowType.Tool | Qt.WindowType.WindowTransparentForInput
                 | Qt.WindowType.NoDropShadowWindowHint)
        super().__init__(None, flags)
        self.cfg = cfg
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setWindowTitle("Entity Tracker Overlay")

        self._result: Optional[FrameResult] = None
        self._live_region: Optional[Region] = None
        self._applied_region: Optional[Region] = None
        self._affinity_done = False
        self._font = QFont("Segoe UI", 9)
        self._font.setBold(True)
        self._big_font = QFont("Segoe UI", 12)
        self._big_font.setBold(True)

    # -- public API ---------------------------------------------------------
    def set_live_region(self, region: Optional[Region]) -> None:
        """Called ~20x/second with the current Roblox client rectangle (physical pixels)."""
        self._live_region = region if region is not None and region.valid else None
        self.refresh()

    def set_result(self, result: FrameResult) -> None:
        self._result = result
        if self.isVisible():
            self.update()

    def refresh(self) -> None:
        """Show/hide and position according to settings and window availability."""
        should_show = self.cfg.overlay_enabled and self._live_region is not None
        if not should_show:
            if self.isVisible():
                self.hide()
            self._applied_region = None
            return
        region = self._live_region
        first_show = not self.isVisible()
        if first_show:
            self._set_qt_geometry(region)   # approximate, so Qt's idea of the window matches
            self.show()
            self._apply_capture_exclusion()
        if first_show or region != self._applied_region:
            self._apply_geometry(region)    # exact physical pixels on Windows
            self._applied_region = region
        self.update()

    # -- placement ----------------------------------------------------------
    def _set_qt_geometry(self, region: Region) -> None:
        dpr = self.devicePixelRatioF() or 1.0
        self.setGeometry(QRect(round(region.left / dpr), round(region.top / dpr),
                               round(region.width / dpr), round(region.height / dpr)))

    def _apply_geometry(self, region: Region) -> None:
        if IS_WINDOWS:
            # Win32 works in physical pixels, so this is exact on any display scaling.
            hwnd = int(self.winId())
            ctypes.windll.user32.SetWindowPos(
                ctypes.c_void_p(hwnd), ctypes.c_void_p(_HWND_TOPMOST),
                int(region.left), int(region.top), int(region.width), int(region.height), _SWP_NOACTIVATE)
        else:
            self._set_qt_geometry(region)

    def _apply_capture_exclusion(self) -> None:
        """Hide this window from screen capture so we never detect our own boxes."""
        if self._affinity_done or not IS_WINDOWS or not self.cfg.exclude_overlay_from_capture:
            return
        ok = ctypes.windll.user32.SetWindowDisplayAffinity(ctypes.c_void_p(int(self.winId())), _WDA_EXCLUDEFROMCAPTURE)
        self._affinity_done = True
        if not ok:
            log.warning("Could not exclude the overlay from screen capture (needs Windows 10 2004+). "
                        "Boxes may appear in captured frames; lower the overlay opacity or thickness if detection is affected.")

    # -- drawing ------------------------------------------------------------
    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        result = self._result
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setOpacity(self.cfg.overlay_opacity)
        if result is None or result.region is None or not result.region.valid:
            return
        sx = self.width() / result.region.width
        sy = self.height() / result.region.height

        painter.setFont(self._font)
        metrics = painter.fontMetrics()
        for t in result.tracks:
            rect = QRectF(t.x1 * sx, t.y1 * sy, (t.x2 - t.x1) * sx, (t.y2 - t.y1) * sy)
            color = color_for_id(t.track_id)
            painter.setPen(QPen(color, self.cfg.line_thickness))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)

            label = f"ID {t.track_id}" + (f"  {t.conf:.0%}" if self.cfg.show_confidence else "")
            w, h = metrics.horizontalAdvance(label) + 8, metrics.height() + 2
            top = rect.top() - h if rect.top() - h >= 0 else rect.top()
            tag = QRectF(rect.left(), top, w, h)
            painter.fillRect(tag, color)
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(tag, Qt.AlignmentFlag.AlignCenter, label)

        self._draw_counter(painter, result)

    def _draw_counter(self, painter: QPainter, result: FrameResult) -> None:
        line1 = f"Characters: {len(result.tracks)}"
        line2 = f"detect {result.detect_fps:.1f} fps  |  {result.device}"
        painter.setFont(self._big_font)
        w1 = painter.fontMetrics().horizontalAdvance(line1)
        h1 = painter.fontMetrics().height()
        painter.setFont(self._font)
        w2 = painter.fontMetrics().horizontalAdvance(line2)
        h2 = painter.fontMetrics().height()
        panel = QRectF(10, 10, max(w1, w2) + 20, h1 + h2 + 14)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.drawRoundedRect(panel, 6, 6)
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(self._big_font)
        painter.drawText(QRectF(panel.left() + 10, panel.top() + 4, panel.width(), h1), Qt.AlignmentFlag.AlignLeft, line1)
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(self._font)
        painter.drawText(QRectF(panel.left() + 10, panel.top() + 6 + h1, panel.width(), h2), Qt.AlignmentFlag.AlignLeft, line2)


class ControlPanel(QWidget):
    """Live settings. Every control writes straight into the shared Config."""

    overlay_changed = pyqtSignal()
    closed = pyqtSignal()

    def __init__(self, cfg: Config, device_text: str = ""):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("Entity Tracker Controls")  # must not contain the Roblox window title
        self.setMinimumWidth(360)

        self.chk_enabled = QCheckBox(f"Show overlay  (hotkey: {cfg.toggle_hotkey})")
        self.chk_enabled.setChecked(cfg.overlay_enabled)
        self.chk_enabled.toggled.connect(self._on_enabled)

        self.sld_opacity, self.lbl_opacity = self._slider(10, 100, round(cfg.overlay_opacity * 100), self._on_opacity, "%")
        self.sld_conf, self.lbl_conf = self._slider(5, 95, round(cfg.confidence * 100), self._on_conf, "%")
        self.sld_scale, self.lbl_scale = self._slider(25, 100, round(cfg.capture_scale * 100), self._on_scale, "%")

        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 60)
        self.spin_fps.setValue(min(60, cfg.detect_fps))
        self.spin_fps.setSuffix(" fps")
        self.spin_fps.valueChanged.connect(self._on_fps)

        self.spin_timeout = QSpinBox()
        self.spin_timeout.setRange(1, 30)
        self.spin_timeout.setValue(max(1, round(cfg.track_timeout_s)))
        self.spin_timeout.setSuffix(" s")
        self.spin_timeout.valueChanged.connect(self._on_timeout)

        form = QFormLayout()
        form.addRow("Overlay opacity", self._row(self.sld_opacity, self.lbl_opacity))
        form.addRow("Confidence threshold", self._row(self.sld_conf, self.lbl_conf))
        form.addRow("Capture resolution", self._row(self.sld_scale, self.lbl_scale))
        form.addRow("Detection rate", self.spin_fps)
        form.addRow("Forget lost tracks after", self.spin_timeout)

        self.lbl_status = QLabel("Starting...")
        self.lbl_status.setWordWrap(True)
        self.lbl_device = QLabel(f"Model device: {device_text}")
        note = QLabel("IDs are temporary tracking labels, not Roblox usernames.")
        note.setStyleSheet("color: gray;")

        btn_save = QPushButton("Save settings")
        btn_save.clicked.connect(self.cfg.save)

        layout = QVBoxLayout(self)
        layout.addWidget(self.chk_enabled)
        layout.addLayout(form)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.lbl_device)
        layout.addWidget(note)
        layout.addWidget(btn_save)

        if not IS_WINDOWS:  # on Windows the global HotkeyWatcher handles the key
            QShortcut(QKeySequence(cfg.toggle_hotkey), self, activated=self.toggle_overlay)

    # -- helpers ------------------------------------------------------------
    def _slider(self, lo: int, hi: int, value: int, handler, suffix: str):
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(max(lo, min(hi, value)))
        label = QLabel(f"{s.value()}{suffix}")
        label.setMinimumWidth(42)
        s.valueChanged.connect(lambda v: (label.setText(f"{v}{suffix}"), handler(v)))
        return s, label

    @staticmethod
    def _row(slider: QSlider, label: QLabel) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(slider, 1)
        lay.addWidget(label)
        return w

    # -- handlers -----------------------------------------------------------
    def toggle_overlay(self) -> None:
        self.chk_enabled.setChecked(not self.chk_enabled.isChecked())

    def _on_enabled(self, checked: bool) -> None:
        self.cfg.overlay_enabled = checked
        self.overlay_changed.emit()

    def _on_opacity(self, v: int) -> None:
        self.cfg.overlay_opacity = v / 100.0
        self.overlay_changed.emit()

    def _on_conf(self, v: int) -> None:
        self.cfg.confidence = v / 100.0

    def _on_scale(self, v: int) -> None:
        self.cfg.capture_scale = v / 100.0

    def _on_fps(self, v: int) -> None:
        self.cfg.detect_fps = v

    def _on_timeout(self, v: int) -> None:
        self.cfg.track_timeout_s = float(v)

    def set_result(self, r: FrameResult) -> None:
        window = "Roblox window: OK" if r.region is not None else f"Roblox window: {r.status}"
        self.lbl_status.setText(f"{window}\ncapture {r.capture_fps:.0f} fps | detection {r.detect_fps:.1f} fps | "
                                f"characters visible: {len(r.tracks)}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.cfg.save()
        self.closed.emit()
        super().closeEvent(event)


class HotkeyWatcher(QObject):
    """Emits `triggered` whenever the global toggle key is pressed (Windows)."""

    triggered = pyqtSignal()

    def __init__(self, key_name: str, parent: Optional[QObject] = None):
        super().__init__(parent)
        try:
            self._edge = KeyEdge(key_name)
        except ValueError:
            log.warning("Unsupported toggle hotkey %r; falling back to F8", key_name)
            self._edge = KeyEdge("F8")
        self._timer = QTimer(self)
        self._timer.setInterval(25)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    def _poll(self) -> None:
        if self._edge.pressed():
            self.triggered.emit()
