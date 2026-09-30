"""
Play a generated trajectory with accurate timing through controller.mouse.

Windows' default sleep resolution is ~15.6 ms, but mouse samples are only
~8 ms apart, so we temporarily request 1 ms timer resolution (timeBeginPeriod)
and schedule every point against a perf_counter clock (no drift build-up).
"""

import ctypes
import time
from contextlib import contextmanager

import pyautogui

from controller import mouse, screen

_winmm = ctypes.windll.winmm


@contextmanager
def high_res_timer(ms=1):
    """Temporarily set the Windows timer resolution to `ms` milliseconds."""
    _winmm.timeBeginPeriod(ms)
    try:
        yield
    finally:
        _winmm.timeEndPeriod(ms)


def _clamp(x, y, bounds):
    """
    Keep a point on the desktop (all monitors - `bounds` from
    screen.virtual_screen_rect()) and off the exact pyautogui fail-safe corner
    pixels, so a generated path can't trigger the emergency stop by accident.
    """
    left, top, right, bottom = bounds
    px = min(max(int(round(x)), left), right - 1)
    py = min(max(int(round(y)), top), bottom - 1)
    if (px, py) in pyautogui.FAILSAFE_POINTS:
        px += 1 if px == left else -1
    return px, py


def play_trajectory(points, speed=1.0):
    """
    Move the cursor along `points` = [(x, y, dt), ...].

    dt is the delay (s) *before* moving to that point.
    speed: >1 plays faster, <1 slower (1 = recorded human speed).
    """
    last = None
    bounds = screen.virtual_screen_rect()   # re-read each time: monitors can change
    with high_res_timer():
        t_next = time.perf_counter()
        for x, y, dt in points:
            t_next += dt / speed
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            pos = _clamp(x, y, bounds)
            if pos != last:          # skip redundant OS calls for sub-pixel steps
                mouse.move_to(*pos)
                last = pos
