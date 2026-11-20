"""Stage 3a - collect screenshots of Roblox for annotation.

Play Roblox in windowed/borderless mode with this running in a terminal:

    F6   start / stop automatic capture (one image every --interval seconds)
    F7   save one screenshot right now
    END  quit

Images go to dataset/raw_images/<session>_<number>.jpg. Near-duplicate frames
(you standing still) are skipped so you do not label hundreds of identical images.
Vary the maps, avatars, distances, lighting and number of characters between
sessions - dataset variety matters more than sheer quantity.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from capture import Frame, ScreenGrabber, enable_dpi_awareness
from config import RAW_IMAGES_DIR, Config
from hotkeys import KeyEdge

log = logging.getLogger("collect")


def frame_signature(image: np.ndarray) -> np.ndarray:
    """Tiny grayscale thumbnail used to compare frames cheaply."""
    small = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def is_near_duplicate(sig: np.ndarray, previous: Optional[np.ndarray], min_change: float) -> bool:
    """True when the mean absolute pixel difference to the previous saved frame is tiny."""
    if previous is None:
        return False
    return float(np.mean(np.abs(sig - previous))) < min_change


def next_index(out_dir: Path, session: str) -> int:
    """Continue numbering if this session name already has files."""
    highest = 0
    for p in out_dir.glob(f"{session}_*.jpg"):
        tail = p.stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            highest = max(highest, int(tail))
    return highest + 1


class Collector:
    def __init__(self, grabber: ScreenGrabber, out_dir: Path, session: str,
                 min_change: float = 4.0, jpeg_quality: int = 95):
        self.grabber = grabber
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.session = session
        self.min_change = min_change
        self.jpeg_quality = jpeg_quality
        self.counter = next_index(self.out_dir, session)
        self.saved = 0
        self._last_sig: Optional[np.ndarray] = None
        self.last_frame: Optional[Frame] = None  # newest grabbed frame (used for GUI previews)

    def _grab_full_res(self) -> Optional[Frame]:
        # Full resolution on purpose: annotate at the best quality, YOLO resizes later.
        return self.grabber.grab(scale=1.0)

    def capture(self, force: bool = False) -> Optional[Path]:
        """Save one frame. Returns the file path, or None if skipped / unavailable."""
        frame = self._grab_full_res()
        if frame is None:
            log.info("  skipped: %s", self.grabber.status)
            return None
        self.last_frame = frame
        sig = frame_signature(frame.image)
        if not force and is_near_duplicate(sig, self._last_sig, self.min_change):
            return None
        path = self.out_dir / f"{self.session}_{self.counter:05d}.jpg"
        ok = cv2.imwrite(str(path), frame.image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            log.error("  could not write %s", path)
            return None
        self._last_sig = sig
        self.counter += 1
        self.saved += 1
        return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Roblox screenshots for the dataset.")
    parser.add_argument("--session", default=datetime.now().strftime("s%Y%m%d-%H%M%S"),
                        help="session name used in file names (use a new one per play session)")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between automatic captures")
    parser.add_argument("--min-change", type=float, default=4.0,
                        help="skip frames whose mean pixel change is below this (0 = keep everything)")
    parser.add_argument("--max", type=int, default=0, help="stop after saving this many images (0 = no limit)")
    parser.add_argument("--out", default=str(RAW_IMAGES_DIR), help="output folder")
    parser.add_argument("--monitor", action="store_true", help="capture the whole monitor instead of the Roblox window")
    parser.add_argument("--auto", action="store_true", help="start with automatic capture already on")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    enable_dpi_awareness()

    session = args.session.replace("_", "-")  # '_' separates session and number in file names
    cfg = Config.load()
    if args.monitor:
        cfg.capture_mode = "monitor"
    cfg.validate()

    grabber = ScreenGrabber(cfg)
    collector = Collector(grabber, Path(args.out), session, min_change=args.min_change)

    toggle, single, quit_key = KeyEdge("F6"), KeyEdge("F7"), KeyEdge("END")
    auto = args.auto
    print(f"Session '{session}' -> {collector.out_dir}")
    print("F6 = start/stop auto capture | F7 = save one now | END = quit | Ctrl+C also quits")
    if sys.platform != "win32":
        print("(Hotkeys only work on Windows; non-Windows runs capture automatically.)")
        auto = True

    next_auto = time.monotonic()
    try:
        while True:
            if quit_key.pressed():
                break
            if toggle.pressed():
                auto = not auto
                print(f"Auto capture {'ON' if auto else 'OFF'}")
                next_auto = time.monotonic()
            if single.pressed():
                path = collector.capture(force=True)
                if path:
                    print(f"[{collector.saved}] saved {path.name}")
            if auto and time.monotonic() >= next_auto:
                next_auto = time.monotonic() + args.interval
                path = collector.capture()
                if path:
                    print(f"[{collector.saved}] saved {path.name}")
            if args.max and collector.saved >= args.max:
                print(f"Reached --max {args.max}.")
                break
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        grabber.close()
    print(f"Done. {collector.saved} images saved in {collector.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
