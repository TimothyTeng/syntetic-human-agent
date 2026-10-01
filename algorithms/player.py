"""
Play a generated trajectory with accurate timing through controller.mouse.

Windows' default sleep resolution is ~15.6 ms, but mouse samples are only
~8 ms apart, so we temporarily request 1 ms timer resolution (timeBeginPeriod)
and schedule every point against a perf_counter clock (no drift build-up).
"""

import ctypes
import time
from contextlib import contextmanager

import numpy as np
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


def resample(points, start, hz=125, jitter=0.1, rng=None):
    """
    Fill long gaps in a trajectory so the cursor updates at a mouse's polling rate.

    A real mouse reports every ~8 ms while it moves. The model's samples are spaced
    more coarsely (mean dt ~17 ms vs ~12 ms in real data), so every moving segment
    longer than 1.5 polling periods is split into evenly timed sub-steps (linear
    interpolation, +-`jitter` period variation) - but into no more steps than whole
    pixels it covers, because a slowly moving mouse sends nothing until its motion
    adds up to a pixel. Segments without movement are left alone.

    points: [(x, y, dt), ...]; start: (x, y) the cursor is at before the first point.
    """
    rng = rng or np.random.default_rng()
    period = 1.0 / hz
    out = []
    px, py = start
    for x, y, dt in points:
        n = min(int(round(dt / period)), int(max(abs(x - px), abs(y - py))))
        if dt <= 1.5 * period or n < 2:
            out.append((x, y, dt))
        else:
            w = 1 + jitter * rng.uniform(-1, 1, n)
            w = w / w.sum() * dt
            f = np.cumsum(w) / dt
            out.extend((px + (x - px) * fi, py + (y - py) * fi, float(wi)) for fi, wi in zip(f, w))
        px, py = x, y
    return out


def play_trajectory(points, speed=1.0, resample_hz=None, rng=None):
    """
    Move the cursor along `points` = [(x, y, dt), ...].

    dt is the delay (s) *before* moving to that point.
    speed: >1 plays faster, <1 slower (1 = recorded human speed).
    resample_hz: if given, fill long gaps up to this polling rate (see resample()).
    """
    if resample_hz:
        points = resample(points, mouse.get_position(), resample_hz, rng=rng)
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
