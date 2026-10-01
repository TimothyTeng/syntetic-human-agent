"""
Mouse accuracy diagnostic. Moves the cursor (no clicks) to random points on every
monitor and reports where the error comes from. Don't touch the mouse while it runs.

    python -m tools.mouse_accuracy            # 15 targets per monitor
    python -m tools.mouse_accuracy --n 30 --seed 1

For each target it measures
  * arrival - where the generated movement left the cursor (before any correction),
  * settled - the same 150 ms later: if this differs from arrival, something else is
              moving the pointer (a second mouse/touchpad, a pen, remote-control software),
  * final   - after HumanMouse's closed-loop correction.
It also prints each monitor's display scaling and this process's DPI awareness: if
clicks land off by a constant factor on one monitor, UI-element rectangles and cursor
coordinates disagree about scaling there.
"""

import argparse
import ctypes
import time
from ctypes import wintypes

import numpy as np

from algorithms import player
from algorithms.human_mouse import HumanMouse
from controller import mouse


def monitors():
    """[(left, top, right, bottom, scale %)] for every monitor."""
    out = []
    shcore = ctypes.windll.shcore

    @ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(wintypes.RECT), ctypes.c_void_p)
    def _cb(hmon, _hdc, rect, _data):
        r = rect.contents
        dx, dy = ctypes.c_uint(), ctypes.c_uint()
        try:
            shcore.GetDpiForMonitor(ctypes.c_void_p(hmon), 0, ctypes.byref(dx), ctypes.byref(dy))  # MDT_EFFECTIVE_DPI
            scale = round(dx.value / 96 * 100)
        except OSError:
            scale = None
        out.append((r.left, r.top, r.right, r.bottom, scale))
        return 1

    ctypes.windll.user32.EnumDisplayMonitors(None, None, _cb, 0)
    return out


def dpi_awareness():
    user32 = ctypes.windll.user32
    try:
        ctx = user32.GetThreadDpiAwarenessContext()
        return {0: "unaware", 1: "system aware", 2: "per-monitor aware"}.get(
            user32.GetAwarenessFromDpiAwarenessContext(ctypes.c_void_p(ctx)), "unknown")
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=15, help="targets per monitor")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    hm = HumanMouse(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    print(f"DPI awareness of this thread: {dpi_awareness()} | mouse backend: {hm.backend}")
    mons = monitors()
    for i, (l, t, r, b, scale) in enumerate(mons):
        print(f"monitor {i}: ({l}, {t}, {r}, {b}) scaling {scale}%")
    print("Starting in 3 s - don't touch the mouse.")
    time.sleep(3)

    home = mouse.get_position()
    rows = []
    for (l, t, r, b, _) in mons:
        for _ in range(args.n):
            target = (int(rng.integers(l + 40, r - 40)), int(rng.integers(t + 40, b - 40)))
            start = mouse.get_position()
            player.play_trajectory(hm.plan(start, target), speed=hm.speed, resample_hz=hm.resample_hz, rng=hm.rng)
            arrival = mouse.get_position()
            time.sleep(0.15)
            settled = mouse.get_position()
            before = hm.corrections
            hm.move_to(*target)
            final = mouse.get_position()
            err = lambda p: float(np.hypot(p[0] - target[0], p[1] - target[1]))  # noqa: E731
            rows.append((err(arrival), err(settled), err(final), hm.corrections - before))
    mouse.move_to(*home)

    a = np.array(rows)
    print(f"\n{len(rows)} moves (pixels from target):")
    for name, col in (("arrival", 0), ("settled after 150 ms", 1), ("final, after correction", 2)):
        print(f"  {name:<24} median {np.median(a[:, col]):5.1f}   max {a[:, col].max():6.1f}   "
              f"exact {np.mean(a[:, col] <= 1) * 100:3.0f}%")
    moved = np.mean(np.abs(a[:, 1] - a[:, 0]) > 1) * 100
    print(f"  corrections made: {int(a[:, 3].sum())}")
    if moved > 5:
        print(f"\nThe pointer moved on its own after {moved:.0f}% of movements - another input device or"
              " program is moving it. The closed-loop correction compensates, but find the source.")
    elif np.median(a[:, 0]) > 1:
        print("\nMovements end off target although nothing else moves the pointer - please report this output.")
    else:
        print("\nCursor positioning is exact. If clicks still look off, it is the deliberate click scatter"
              " (--click-spread, default 0.17) or the element rectangles themselves (check display scaling above).")


if __name__ == "__main__":
    main()
