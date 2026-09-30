"""
Reading behaviour: what a person does while reading a web page.

Alternates between
  * scroll bursts - a few wheel notches in quick succession, mostly downwards,
    occasionally back up,
  * reading pauses - log-normally distributed (most short, a few long),
  * mouse drifts - small idle movements made with the learned mouse model.

The movement itself comes from the trained model (via HumanMouse). The timing
parameters below are hand-set placeholders for the future entropy work;
they're collected in ReadingParams so they can be tuned or learned later.
"""

import time
from dataclasses import dataclass

import numpy as np

from controller import mouse

from . import targeting


@dataclass
class ReadingParams:
    p_scroll: float = 0.45           # chance the next action is a scroll burst
    p_pause: float = 0.35            # ... a reading pause
    p_drift: float = 0.20            # ... a small mouse drift
    notches_min: int = 1             # wheel notches per burst
    notches_max: int = 5
    notch_gap_median: float = 0.07   # seconds between notches in a burst
    p_scroll_up: float = 0.12        # chance a burst scrolls back up
    pause_median: float = 1.8        # seconds, reading pause
    pause_sigma: float = 0.6         # log-normal spread of pauses
    drift_max_px: int = 120


def _lognormal(rng, median, sigma):
    return float(median * np.exp(rng.normal(0, sigma)))


def reading_area(page_rect):
    """The central text column of the page, where a reader's cursor tends to rest."""
    left, top, right, bottom = page_rect
    w, h = right - left, bottom - top
    return (int(left + 0.2 * w), int(top + 0.15 * h), int(right - 0.2 * w), int(bottom - 0.1 * h))


def _inside(point, rect):
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def scroll_burst(rng, params=ReadingParams()):
    """Scroll a few notches in quick succession. Returns notches scrolled (neg = down)."""
    n = int(rng.integers(params.notches_min, params.notches_max + 1))
    direction = 1 if rng.random() < params.p_scroll_up else -1
    for _ in range(n):
        mouse.scroll(direction)
        time.sleep(_lognormal(rng, params.notch_gap_median, 0.4))
    return direction * n


def read_page(hm, duration, page_rect=None, params=ReadingParams()):
    """
    Simulate reading the current page for about `duration` seconds.

    hm:        HumanMouse instance (provides movement + rng).
    page_rect: web page area (controller.browser.get_page_rect()); if given,
               the cursor is first moved into the text column so wheel events
               reach the page.
    """
    rng = hm.rng
    area = reading_area(page_rect) if page_rect else None
    if area and not _inside(mouse.get_position(), area):
        hm.move_to(*targeting.random_point_in(area, rng))

    end = time.monotonic() + duration
    probs = np.array([params.p_scroll, params.p_pause, params.p_drift])
    probs = probs / probs.sum()
    while time.monotonic() < end:
        action = rng.choice(["scroll", "pause", "drift"], p=probs)
        if action == "scroll":
            scroll_burst(rng, params)
        elif action == "pause":
            remaining = end - time.monotonic()
            time.sleep(max(0.0, min(_lognormal(rng, params.pause_median, params.pause_sigma), remaining)))
        else:
            hm.drift(params.drift_max_px, bounds=area)
