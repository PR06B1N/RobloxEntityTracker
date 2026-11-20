"""YOLO character detector (Stage 5).

`Detector` wraps an Ultralytics model trained on the single class `character`.
It prefers a CUDA GPU when PyTorch can use one and falls back to the CPU
automatically if anything about the GPU path fails.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)


class ModelNotFoundError(FileNotFoundError):
    """best.pt does not exist yet (train the model first)."""


@dataclass
class Detections:
    """Boxes in pixel coordinates of the image that was given to `detect()`."""
    xyxy: np.ndarray   # (N, 4) float32: x1, y1, x2, y2
    conf: np.ndarray   # (N,)  float32

    def __len__(self) -> int:
        return int(len(self.conf))

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))


def pick_device(requested: str = "auto") -> str:
    """'auto' -> 'cuda:0' when available, else 'cpu'. Explicit values are passed through."""
    if requested and requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception as exc:  # torch missing or broken
        log.warning("PyTorch check failed (%s); using CPU.", exc)
    return "cpu"


class Detector:
    def __init__(self, model_path: str | Path, device: str = "auto", imgsz: int = 640,
                 use_half: bool = True, classes: Optional[Sequence[int]] = None, max_det: int = 100):
        path = Path(model_path)
        if not path.exists():
            raise ModelNotFoundError(
                f"Model file not found: {path}\n"
                "Train it first (python train.py) after building a labelled dataset."
            )
        from ultralytics import YOLO  # slow import, so done lazily

        self.imgsz = int(imgsz)
        self.classes = list(classes) if classes is not None else None
        self.max_det = max_det
        self.model = YOLO(str(path))

        self.device = pick_device(device)
        self.half = bool(use_half) and self.device != "cpu"
        try:
            self._warmup()
        except Exception as exc:
            if self.device == "cpu":
                raise
            log.warning("GPU inference failed (%s). Falling back to CPU.", exc)
            self.device, self.half = "cpu", False
            self._warmup()

        names = list(self.model.names.values()) if isinstance(self.model.names, dict) else list(self.model.names)
        if names != ["character"]:
            log.warning("Model classes are %s, expected ['character']. Is this the right weights file?", names)

    def _warmup(self) -> None:
        dummy = np.zeros((self.imgsz, self.imgsz, 3), np.uint8)
        self.detect(dummy, conf=0.5, iou=0.5)

    def detect(self, image_bgr: np.ndarray, conf: float = 0.4, iou: float = 0.5) -> Detections:
        """Run detection on one BGR image (OpenCV layout)."""
        kwargs = {"half": True} if self.half else {}  # only pass it on GPU (newer versions warn otherwise)
        result = self.model.predict(
            image_bgr, imgsz=self.imgsz, conf=float(conf), iou=float(iou), device=self.device,
            classes=self.classes, max_det=self.max_det, verbose=False, **kwargs,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return Detections.empty()
        return Detections(
            boxes.xyxy.detach().cpu().numpy().astype(np.float32),
            boxes.conf.detach().cpu().numpy().astype(np.float32),
        )

    def describe(self) -> str:
        return f"{self.device}{' (fp16)' if self.half else ''}"
