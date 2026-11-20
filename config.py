"""Central configuration for Roblox Entity Tracker.

Settings live in a single `Config` dataclass that is saved to / loaded from
`config.json` next to this file. Every numeric value is clamped to a safe
range by `Config.validate()` so a hand-edited file can never crash the app.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATASET_DIR = ROOT / "dataset"
RAW_IMAGES_DIR = DATASET_DIR / "raw_images"
CVAT_EXPORT_DIR = DATASET_DIR / "cvat_export"
DATA_YAML = ROOT / "data.yaml"
CONFIG_PATH = ROOT / "config.json"
DEFAULT_MODEL_PATH = MODELS_DIR / "best.pt"

CLASS_NAMES = ["character"]  # single detection class (players and NPCs alike)


def _clamp(value, low, high):
    return max(low, min(high, value))


@dataclass
class Config:
    # --- capture ---------------------------------------------------------
    window_title: str = "Roblox"       # title of the game window
    window_match: str = "exact"        # "exact" or "contains"
    capture_mode: str = "window"       # "window" (Roblox client area) or "monitor"
    monitor_index: int = 1             # used when capture_mode == "monitor" (1 = primary)
    capture_fps: int = 20              # how often the screen is grabbed
    capture_scale: float = 0.5         # 0.25 - 1.0; frames are downscaled before detection

    # --- detection -------------------------------------------------------
    model_path: str = str(DEFAULT_MODEL_PATH)
    device: str = "auto"               # "auto", "cpu", "cuda:0", ...
    use_half: bool = True              # FP16 inference (only used on CUDA)
    detect_fps: int = 15               # how often detection runs
    confidence: float = 0.40           # live-adjustable confidence threshold
    iou: float = 0.50                  # NMS IoU threshold
    imgsz: int = 640                   # YOLO inference size

    # --- tracking --------------------------------------------------------
    track_timeout_s: float = 1.5       # seconds a lost track is remembered
    track_low_thresh: float = 0.10     # ByteTrack second-association threshold
    track_match_thresh: float = 0.80   # ByteTrack IoU matching threshold

    # --- overlay ---------------------------------------------------------
    overlay_enabled: bool = True
    overlay_opacity: float = 0.90      # 0.1 - 1.0
    line_thickness: int = 2
    show_confidence: bool = True
    toggle_hotkey: str = "F8"          # global key that shows/hides the overlay
    exclude_overlay_from_capture: bool = True  # keep the overlay out of our own screenshots

    # ---------------------------------------------------------------------
    def validate(self) -> "Config":
        """Clamp / normalise every value. Returns self for chaining."""
        self.window_title = str(self.window_title)
        self.window_match = self.window_match if self.window_match in ("exact", "contains") else "exact"
        self.capture_mode = self.capture_mode if self.capture_mode in ("window", "monitor") else "window"
        self.monitor_index = int(_clamp(int(self.monitor_index), 1, 16))
        self.capture_fps = int(_clamp(int(self.capture_fps), 1, 120))
        self.capture_scale = float(_clamp(float(self.capture_scale), 0.25, 1.0))
        self.model_path = str(self.model_path)
        self.device = str(self.device) or "auto"
        self.use_half = bool(self.use_half)
        self.detect_fps = int(_clamp(int(self.detect_fps), 1, 120))
        self.confidence = float(_clamp(float(self.confidence), 0.05, 0.95))
        self.iou = float(_clamp(float(self.iou), 0.1, 0.95))
        self.imgsz = int(_clamp(int(self.imgsz), 160, 1920))
        self.imgsz -= self.imgsz % 32  # YOLO needs multiples of 32
        self.track_timeout_s = float(_clamp(float(self.track_timeout_s), 0.1, 30.0))
        self.track_low_thresh = float(_clamp(float(self.track_low_thresh), 0.01, 0.5))
        self.track_match_thresh = float(_clamp(float(self.track_match_thresh), 0.1, 0.99))
        self.overlay_enabled = bool(self.overlay_enabled)
        self.overlay_opacity = float(_clamp(float(self.overlay_opacity), 0.1, 1.0))
        self.line_thickness = int(_clamp(int(self.line_thickness), 1, 8))
        self.show_confidence = bool(self.show_confidence)
        self.toggle_hotkey = str(self.toggle_hotkey).upper() or "F8"
        self.exclude_overlay_from_capture = bool(self.exclude_overlay_from_capture)
        return self

    @property
    def resolved_model_path(self) -> Path:
        p = Path(self.model_path)
        return p if p.is_absolute() else ROOT / p

    # ---------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        """Load settings; missing file or bad values fall back to defaults."""
        path = Path(path)
        if not path.exists():
            return cls().validate()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            known = {f.name for f in fields(cls)}
            cfg = cls(**{k: v for k, v in raw.items() if k in known})
            return cfg.validate()
        except Exception as exc:  # corrupt JSON, wrong types, ...
            log.warning("Could not read %s (%s); using defaults.", path, exc)
            return cls().validate()

    def save(self, path: Path = CONFIG_PATH) -> None:
        self.validate()
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
