"""Roblox Entity Tracker - entry point (Stage 8).

    python main.py

Starts three cooperating parts:
    capture thread  ->  detection + tracking thread  ->  PyQt6 overlay (main thread)

Everything is external and visual: screenshots in, rectangles out. No game
memory is read, nothing is injected and no game files are touched.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from config import Config

log = logging.getLogger("main")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect and track visible Roblox characters (external, visual only).")
    p.add_argument("--model", help="path to weights (default: models/best.pt)")
    p.add_argument("--device", help="auto, cpu, cuda:0 ...")
    p.add_argument("--monitor", action="store_true", help="capture a whole monitor instead of the Roblox window")
    p.add_argument("--conf", type=float, help="initial confidence threshold (0.05-0.95)")
    p.add_argument("--detect-fps", type=int, help="initial detection rate")
    p.add_argument("--scale", type=float, help="initial capture scale (0.25-1.0)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = Config.load()
    if args.model:
        cfg.model_path = args.model
    if args.device:
        cfg.device = args.device
    if args.monitor:
        cfg.capture_mode = "monitor"
    if args.conf is not None:
        cfg.confidence = args.conf
    if args.detect_fps is not None:
        cfg.detect_fps = args.detect_fps
    if args.scale is not None:
        cfg.capture_scale = args.scale
    cfg.validate()

    from PyQt6.QtCore import QObject, QTimer, pyqtSignal
    from PyQt6.QtWidgets import QApplication, QMessageBox

    app = QApplication(sys.argv)
    app.setApplicationName("Entity Tracker")
    app.setQuitOnLastWindowClosed(False)  # the overlay is a tool window; closing the control panel quits instead

    # -- load the model (fail with a readable message, not a traceback) ------
    from detector import Detector, ModelNotFoundError
    try:
        detector = Detector(cfg.resolved_model_path, device=cfg.device, imgsz=cfg.imgsz, use_half=cfg.use_half)
    except ModelNotFoundError as exc:
        QMessageBox.critical(None, "Model not found", str(exc))
        return 1
    except Exception as exc:
        log.exception("Could not load the model")
        QMessageBox.critical(None, "Could not load the model", f"{type(exc).__name__}: {exc}")
        return 1
    log.info("Model loaded on %s", detector.describe())

    from capture import CaptureThread
    from overlay import ControlPanel, HotkeyWatcher, OverlayWindow
    from pipeline import DetectionWorker, FrameResult
    from tracker import CharacterTracker

    class Bridge(QObject):
        """Carries results from the worker thread to the GUI thread."""
        result = pyqtSignal(object)

    bridge = Bridge()
    capture = CaptureThread(cfg)
    tracker = CharacterTracker(fps=cfg.detect_fps, timeout_s=cfg.track_timeout_s, confidence=cfg.confidence,
                               low_thresh=cfg.track_low_thresh, match_thresh=cfg.track_match_thresh)
    worker = DetectionWorker(cfg, capture, detector, tracker, bridge.result.emit)

    overlay = OverlayWindow(cfg)
    panel = ControlPanel(cfg, detector.describe())
    bridge.result.connect(overlay.set_result)
    bridge.result.connect(panel.set_result)
    panel.overlay_changed.connect(overlay.refresh)
    panel.closed.connect(app.quit)

    hotkey = HotkeyWatcher(cfg.toggle_hotkey)
    hotkey.triggered.connect(panel.toggle_overlay)

    # keep the overlay glued to the Roblox window, even while it is being dragged
    follow = QTimer()
    follow.setInterval(50)
    follow.timeout.connect(lambda: overlay.set_live_region(capture.region if capture.status == "ok" else None))
    follow.start()

    def shutdown() -> None:
        follow.stop()
        worker.stop()
        capture.stop()
        worker.join(timeout=2)
        capture.join(timeout=2)
        overlay.hide()
        try:
            cfg.save()
        except OSError as exc:
            log.warning("Could not save settings: %s", exc)

    app.aboutToQuit.connect(shutdown)

    capture.start()
    worker.start()
    panel.show()
    bridge.result.emit(FrameResult(status="waiting for first frame", device=detector.describe()))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
