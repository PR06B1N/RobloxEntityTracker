"""ByteTrack multi-object tracking (Stage 6).

Uses the ByteTrack implementation that ships with Ultralytics. It receives the
detector's boxes for each frame and decides which box in the new frame is the
same character as in the previous one, handing out temporary integer IDs.

IMPORTANT: these IDs are only labels for "the same blob across frames". They
are not Roblox usernames or user IDs, and they restart when tracking is reset.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from detector import Detections

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Track:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def size(self) -> tuple[float, float]:
        return self.x2 - self.x1, self.y2 - self.y1


class _ByteTrackInput:
    """The small interface BYTETracker.update() reads from a detection result.

    It needs `conf`, `cls`, `xywh` (centre x, centre y, width, height) and must support
    boolean-mask indexing (newer Ultralytics versions slice the detections).
    """

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, np.float32).reshape(-1)
        x1, y1, x2, y2 = self.xyxy[:, 0], self.xyxy[:, 1], self.xyxy[:, 2], self.xyxy[:, 3]
        self.xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1).astype(np.float32)
        self.cls = np.zeros(len(self.conf), np.float32)  # single class

    @classmethod
    def from_detections(cls, dets: Detections) -> "_ByteTrackInput":
        return cls(dets.xyxy, dets.conf)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> "_ByteTrackInput":
        return _ByteTrackInput(self.xyxy[mask], self.conf[mask])


class CharacterTracker:
    """Assigns temporary IDs to detected characters.

    fps          : how often `update()` is called (the detection rate)
    timeout_s    : how long a character that vanished may reappear with the same ID
    confidence   : detections at/above this start and drive tracks; detections between
                   `low_thresh` and `confidence` are only used to keep existing tracks alive
    """

    def __init__(self, fps: float = 15.0, timeout_s: float = 1.5, confidence: float = 0.4,
                 low_thresh: float = 0.1, match_thresh: float = 0.8):
        self.fps = float(fps)
        self.timeout_s = float(timeout_s)
        self.confidence = float(confidence)
        self.low_thresh = float(low_thresh)
        self.match_thresh = float(match_thresh)
        self._tracker = None
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Forget all tracks; the next IDs start again at 1."""
        import inspect

        from ultralytics.trackers.byte_tracker import BYTETracker, STrack
        from ultralytics.utils import IterableSimpleNamespace

        if hasattr(STrack, "reset_id"):
            STrack.reset_id()  # older Ultralytics keeps a global ID counter
        args = IterableSimpleNamespace(
            track_high_thresh=self.confidence,
            track_low_thresh=self.low_thresh,
            new_track_thresh=self.confidence,
            track_buffer=30,                 # replaced below by the real timeout in frames
            match_thresh=self.match_thresh,
            fuse_score=True,
        )
        # Ultralytics changed this constructor between versions: support both signatures.
        if "frame_rate" in inspect.signature(BYTETracker.__init__).parameters:
            self._tracker = BYTETracker(args, frame_rate=30)
        else:
            self._tracker = BYTETracker(args)
        self._apply_timeout()

    def _apply_timeout(self) -> None:
        # A lost track is dropped after this many frames (attribute name differs per version).
        frames = max(1, int(round(self.fps * self.timeout_s)))
        self._tracker.args.track_buffer = frames
        self._tracker.max_time_lost = frames
        self._tracker.max_frames_lost = frames

    def configure(self, fps: float | None = None, timeout_s: float | None = None,
                  confidence: float | None = None) -> None:
        """Change settings on the fly without losing the current tracks."""
        if fps is not None:
            self.fps = float(fps)
        if timeout_s is not None:
            self.timeout_s = float(timeout_s)
        if confidence is not None:
            self.confidence = float(confidence)
            self._tracker.args.track_high_thresh = self.confidence
            self._tracker.args.new_track_thresh = self.confidence
        self._apply_timeout()

    # ------------------------------------------------------------------
    def update(self, dets: Detections) -> list[Track]:
        """Feed one frame of detections; returns the currently visible tracks."""
        rows = np.asarray(self._tracker.update(_ByteTrackInput.from_detections(dets)))
        if rows.size == 0:
            return []
        rows = rows.reshape(-1, rows.shape[-1])  # columns: x1, y1, x2, y2, id, score, cls, det_index
        return [Track(int(r[4]), float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[5])) for r in rows]
