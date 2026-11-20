"""Detection + tracking worker thread (Stage 8 glue).

Capture, detection/tracking and the Qt UI each run on their own thread, so a
slow model never freezes the overlay and a slow overlay never delays capture.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from capture import CaptureThread, Region
from config import Config
from detector import Detector
from tracker import CharacterTracker, Track

log = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """What the overlay needs to draw one update."""
    tracks: list[Track] = field(default_factory=list)  # boxes relative to `region`, in physical px
    region: Optional[Region] = None                    # None -> game window unavailable
    status: str = "starting"
    detect_fps: float = 0.0
    capture_fps: float = 0.0
    device: str = ""


class DetectionWorker(threading.Thread):
    def __init__(self, cfg: Config, capture: CaptureThread, detector: Detector, tracker: CharacterTracker,
                 on_result: Callable[[FrameResult], None]):
        super().__init__(name="DetectionWorker", daemon=True)
        self.cfg, self.capture, self.detector, self.tracker = cfg, capture, detector, tracker
        self.on_result = on_result
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def _emit(self, result: FrameResult) -> None:
        try:
            self.on_result(result)
        except Exception:
            log.exception("Result callback failed")

    def run(self) -> None:
        last_index = 0
        next_t = time.perf_counter()
        last_loop = next_t
        fps = 0.0
        last_ok = time.monotonic()
        tracker_cleared = False
        device = self.detector.describe()

        while not self._stop_event.is_set():
            cfg = self.cfg
            delay = next_t - time.perf_counter()
            if delay > 0 and self._stop_event.wait(delay):
                break

            frame = self.capture.wait_for_frame(last_index, timeout=0.25)
            if frame is None or self.capture.status != "ok":
                self._emit(FrameResult([], None, self.capture.status, fps, self.capture.measured_fps, device))
                if not tracker_cleared and time.monotonic() - last_ok > max(1.0, cfg.track_timeout_s):
                    self.tracker.reset()
                    tracker_cleared = True
                next_t = time.perf_counter() + 0.05
                continue
            last_index = frame.index
            last_ok = time.monotonic()
            tracker_cleared = False

            try:
                effective_fps = min(cfg.detect_fps, fps) if fps > 0 else cfg.detect_fps
                self.tracker.configure(fps=max(1.0, effective_fps), timeout_s=cfg.track_timeout_s,
                                       confidence=cfg.confidence)
                # run the detector with a low threshold: ByteTrack uses weak detections to keep tracks alive
                dets = self.detector.detect(frame.image, conf=min(cfg.track_low_thresh, cfg.confidence), iou=cfg.iou)
                tracks = self.tracker.update(dets)
            except Exception as exc:
                log.exception("Detection failed")
                self._emit(FrameResult([], frame.region, f"detection error: {exc}", fps,
                                       self.capture.measured_fps, device))
                self._stop_event.wait(0.5)
                next_t = time.perf_counter()
                continue

            inv = 1.0 / frame.scale  # frame pixels -> physical window pixels
            scaled = [Track(t.track_id, t.x1 * inv, t.y1 * inv, t.x2 * inv, t.y2 * inv, t.conf) for t in tracks]

            now = time.perf_counter()
            inst = 1.0 / max(1e-6, now - last_loop)
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
            last_loop = now
            self._emit(FrameResult(scaled, frame.region, "ok", fps, self.capture.measured_fps, device))

            next_t += 1.0 / max(1, cfg.detect_fps)
            if next_t < time.perf_counter():
                next_t = time.perf_counter()  # running behind: do not queue up work
