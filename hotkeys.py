"""Tiny global-hotkey helper (Windows only, no extra dependencies).

It *polls* the keyboard state with `GetAsyncKeyState`, which is read-only:
no hooks are installed and no other process is touched. On non-Windows
systems every key simply reports "not pressed".
"""
from __future__ import annotations

import ctypes
import sys

IS_WINDOWS = sys.platform == "win32"

_NAMED = {
    "ESC": 0x1B, "SPACE": 0x20, "END": 0x23, "HOME": 0x24, "INSERT": 0x2D,
    "DELETE": 0x2E, "PAUSE": 0x13, "PAGEUP": 0x21, "PAGEDOWN": 0x22,
}


def vk_from_name(name: str) -> int:
    """'F8' -> 0x77, 'a' -> 0x41, '5' -> 0x35, 'END' -> 0x23. Raises ValueError if unknown."""
    key = name.strip().upper()
    if key in _NAMED:
        return _NAMED[key]
    if len(key) >= 2 and key[0] == "F" and key[1:].isdigit() and 1 <= int(key[1:]) <= 12:
        return 0x6F + int(key[1:])  # VK_F1 = 0x70
    if len(key) == 1 and (key.isalpha() or key.isdigit()):
        return ord(key)
    raise ValueError(f"Unsupported hotkey name: {name!r}")


class KeyEdge:
    """`pressed()` returns True exactly once each time the key goes down."""

    def __init__(self, name: str):
        self.name = name
        self.vk = vk_from_name(name)
        self._was_down = False

    def pressed(self) -> bool:
        if not IS_WINDOWS:
            return False
        down = bool(ctypes.windll.user32.GetAsyncKeyState(self.vk) & 0x8000)
        fired = down and not self._was_down
        self._was_down = down
        return fired
