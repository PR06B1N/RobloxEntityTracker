"""Screen capture of the Roblox window (purely visual, via MSS).

Pieces
------
* `find_window_region()`  - locates the Roblox client area with read-only Win32 calls.
* `ScreenGrabber`         - synchronous "give me one frame" object.
* `CaptureThread`         - background thread that keeps the newest frame ready.

Run `python capture.py --help` for the Stage-2 self-test.
"""
from __future__ import annotations

import argparse
import ctypes
import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from config import Config

log = logging.getLogger(__name__)
IS_WINDOWS = sys.platform == "win32"


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Region:
    """A rectangle in *physical* screen pixels."""
    left: int
    top: int
    width: int
    height: int

    @property
    def valid(self) -> bool:
        return self.width > 0 and self.height > 0

    def as_mss(self) -> dict:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class Frame:
    image: np.ndarray   # BGR, already scaled by `scale`
    region: Region      # where the frame came from (unscaled, physical px)
    scale: float        # image size / region size
    timestamp: float
    index: int


# --------------------------------------------------------------------------
# Win32 window lookup (read-only; no process access)
# --------------------------------------------------------------------------
if IS_WINDOWS:  # pragma: no cover - exercised on Windows only
    from ctypes import wintypes

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    _user32.EnumWindows.argtypes = [_WNDENUMPROC, wintypes.LPARAM]
    _user32.EnumWindows.restype = wintypes.BOOL
    _user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    _user32.GetWindowTextLengthW.restype = ctypes.c_int
    _user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _user32.GetWindowTextW.restype = ctypes.c_int
    _user32.IsWindowVisible.argtypes = [wintypes.HWND]
    _user32.IsWindowVisible.restype = wintypes.BOOL
    _user32.IsIconic.argtypes = [wintypes.HWND]
    _user32.IsIconic.restype = wintypes.BOOL
    _user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    _user32.GetClientRect.restype = wintypes.BOOL
    _user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    _user32.ClientToScreen.restype = wintypes.BOOL


def list_windows() -> list[tuple[int, str, Region | None, bool]]:
    """All visible top-level windows: (hwnd, title, client region, minimized)."""
    if not IS_WINDOWS:
        return []
    found: list[tuple[int, str, Region | None, bool]] = []

    def callback(hwnd, _lparam):
        if not _user32.IsWindowVisible(hwnd):
            return True
        length = _user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buf, length + 1)
        minimized = bool(_user32.IsIconic(hwnd))
        rect = wintypes.RECT()
        region = None
        if _user32.GetClientRect(hwnd, ctypes.byref(rect)):
            origin = wintypes.POINT(0, 0)
            if _user32.ClientToScreen(hwnd, ctypes.byref(origin)):
                region = Region(origin.x, origin.y, rect.right - rect.left, rect.bottom - rect.top)
        found.append((int(hwnd) if hwnd else 0, buf.value, region, minimized))
        return True

    _user32.EnumWindows(_WNDENUMPROC(callback), 0)
    return found


def find_window_region(title: str, match: str = "exact") -> tuple[Optional[Region], str]:
    """Return (client-area region, status). Region is None unless status == 'ok'."""
    if not IS_WINDOWS:
        return None, "window capture needs Windows (use capture_mode='monitor')"
    wanted = title.strip().lower()
    candidates = []
    for _hwnd, name, region, minimized in list_windows():
        n = name.strip().lower()
        if (n == wanted) if match == "exact" else (wanted in n):
            candidates.append((region, minimized))
    if not candidates:
        return None, f"window '{title}' not found"
    usable = [(r, m) for r, m in candidates if r is not None and not m]
    if not usable:
        return None, "window minimized" if any(m for _, m in candidates) else "window has no client area"
    usable = [(r, m) for r, m in usable if r.valid]
    if not usable:
        return None, "window has zero size"
    region = max((r for r, _ in usable), key=lambda r: r.width * r.height)
    return region, "ok"


# --------------------------------------------------------------------------
# MSS backend (separate class so tests can swap it out)
# --------------------------------------------------------------------------
class MssBackend:
    """Thin wrapper around mss. Must be created and used on the same thread."""

    def __init__(self):
        self._sct = None

    def _ensure(self):
        if self._sct is None:
            import mss  # imported lazily so unit tests do not need a display
            self._sct = mss.mss()
        return self._sct

    def grab(self, region: Region) -> np.ndarray:
        """Return a BGRA uint8 array shaped (height, width, 4)."""
        shot = self._ensure().grab(region.as_mss())
        return np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)

    def monitor_region(self, index: int) -> Region:
        monitors = self._ensure().monitors  # [0] = all screens, 1.. = individual
        mon = monitors[index] if 0 < index < len(monitors) else monitors[1]
        return Region(mon["left"], mon["top"], mon["width"], mon["height"])

    def close(self):
        if self._sct is not None:
            try:
                self._sct.close()
            finally:
                self._sct = None


# --------------------------------------------------------------------------
# Synchronous grabber
# --------------------------------------------------------------------------
class ScreenGrabber:
    """Grabs one frame at a time from the Roblox window (or a monitor)."""

    REGION_REFRESH_S = 0.1  # how often the window position is re-checked

    def __init__(self, cfg: Config, backend: Optional[MssBackend] = None,
                 window_finder: Callable[[str, str], tuple[Optional[Region], str]] = find_window_region):
        self.cfg = cfg
        self.backend = backend or MssBackend()
        self._find = window_finder
        self._region: Optional[Region] = None
        self._status = "starting"
        self._last_lookup = 0.0

    @property
    def region(self) -> Optional[Region]:
        return self._region

    @property
    def status(self) -> str:
        return self._status

    def _resolve_region(self) -> None:
        now = time.monotonic()
        if now - self._last_lookup < self.REGION_REFRESH_S and self._region is not None:
            return
        self._last_lookup = now
        if self.cfg.capture_mode == "monitor":
            self._region, self._status = self.backend.monitor_region(self.cfg.monitor_index), "ok"
        else:
            self._region, self._status = self._find(self.cfg.window_title, self.cfg.window_match)

    def grab(self, scale: Optional[float] = None, index: int = 0) -> Optional[Frame]:
        """Capture one frame. Returns None when the window is unavailable."""
        self._resolve_region()
        region = self._region
        if region is None or not region.valid:
            return None
        bgra = self.backend.grab(region)
        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        s = self.cfg.capture_scale if scale is None else scale
        s = max(0.05, min(1.0, float(s)))
        if s < 0.999:
            h, w = bgr.shape[:2]
            bgr = cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        actual_scale = bgr.shape[1] / region.width
        return Frame(bgr, region, actual_scale, time.time(), index)

    def close(self) -> None:
        self.backend.close()


# --------------------------------------------------------------------------
# Background capture thread
# --------------------------------------------------------------------------
class CaptureThread(threading.Thread):
    """Continuously grabs frames and keeps only the newest one."""

    def __init__(self, cfg: Config, grabber_factory: Optional[Callable[[Config], ScreenGrabber]] = None):
        super().__init__(name="CaptureThread", daemon=True)
        self.cfg = cfg
        self._factory = grabber_factory or ScreenGrabber
        self._stop_event = threading.Event()
        self._cond = threading.Condition()
        self._frame: Optional[Frame] = None
        self._index = 0
        self.region: Optional[Region] = None
        self.status = "starting"
        self.measured_fps = 0.0

    # -- consumer API -----------------------------------------------------
    def wait_for_frame(self, last_index: int, timeout: float = 0.5) -> Optional[Frame]:
        """Block until a frame newer than `last_index` exists (or timeout)."""
        with self._cond:
            if self._frame is None or self._frame.index <= last_index:
                self._cond.wait(timeout)
            if self._frame is not None and self._frame.index > last_index:
                return self._frame
        return None

    def stop(self) -> None:
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()

    # -- thread body ------------------------------------------------------
    def run(self) -> None:
        grabber = self._factory(self.cfg)
        next_t = time.perf_counter()
        fps_t0, fps_n = time.perf_counter(), 0
        try:
            while not self._stop_event.is_set():
                period = 1.0 / max(1, self.cfg.capture_fps)
                try:
                    self._index += 1
                    frame = grabber.grab(index=self._index)
                    self.region, self.status = grabber.region, grabber.status
                    if frame is not None:
                        with self._cond:
                            self._frame = frame
                            self._cond.notify_all()
                        fps_n += 1
                    else:
                        self._index -= 1
                except Exception as exc:  # keep running through transient capture errors
                    log.warning("Capture error: %s", exc)
                    self.status = f"capture error: {exc}"
                    grabber.close()
                    self._stop_event.wait(0.5)
                    continue

                now = time.perf_counter()
                if now - fps_t0 >= 1.0:
                    self.measured_fps = fps_n / (now - fps_t0)
                    fps_t0, fps_n = now, 0
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    self._stop_event.wait(delay)
                else:
                    next_t = time.perf_counter()  # we are behind; do not try to catch up
        finally:
            grabber.close()


# --------------------------------------------------------------------------
# Stage 2 self-test
# --------------------------------------------------------------------------
def enable_dpi_awareness() -> None:
    """Make Win32 report real pixels on scaled displays (Qt does this itself in main.py)."""
    if not IS_WINDOWS:
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _main() -> int:
    enable_dpi_awareness()
    parser = argparse.ArgumentParser(description="Test Roblox window capture.")
    parser.add_argument("--seconds", type=float, default=5.0, help="how long to capture")
    parser.add_argument("--save", default="capture_test.png", help="save the last frame here")
    parser.add_argument("--list-windows", action="store_true", help="print visible windows and exit")
    parser.add_argument("--monitor", action="store_true", help="capture a whole monitor instead of the Roblox window")
    parser.add_argument("--scale", type=float, default=None, help="override capture_scale")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.list_windows:
        if not IS_WINDOWS:
            print("Window listing is Windows-only.")
            return 1
        for hwnd, title, region, minimized in list_windows():
            print(f"{hwnd:>10}  {'(min) ' if minimized else ''}{title!r}  {region}")
        return 0

    cfg = Config.load()
    if args.monitor:
        cfg.capture_mode = "monitor"
    if args.scale:
        cfg.capture_scale = args.scale
    cfg.validate()

    thread = CaptureThread(cfg)
    thread.start()
    t0, last, got, last_idx = time.time(), None, 0, 0
    try:
        while time.time() - t0 < args.seconds:
            frame = thread.wait_for_frame(last_idx, timeout=0.5)
            if frame is None:
                print(f"waiting... status: {thread.status}")
                continue
            last, last_idx, got = frame, frame.index, got + 1
    finally:
        thread.stop()
        thread.join(timeout=2)

    if last is None:
        print(f"No frames captured. Status: {thread.status}")
        print("Tip: start Roblox in windowed/borderless mode, or run `python capture.py --list-windows`.")
        return 1
    cv2.imwrite(args.save, last.image)
    print(f"Captured {got} frames in {args.seconds:.1f}s (~{got / args.seconds:.1f} fps)")
    print(f"Region (physical px): {last.region}   frame size: {last.image.shape[1]}x{last.image.shape[0]}   scale: {last.scale:.2f}")
    print(f"Saved last frame to {args.save}. Open it and check that it shows the game.")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
