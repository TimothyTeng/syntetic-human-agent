"""
PLACEHOLDER human typing: per-key delays from a log-normal distribution with
longer gaps after spaces and punctuation. To be replaced by a learned
keystroke-dynamics model later (same function signature).
"""

import time

import numpy as np

from controller import keyboard


def type_like_human(text, rng=None, wpm=45, sigma=0.35):
    """
    Type `text` into the focused window.

    wpm:   average words per minute (1 word = 5 characters).
    sigma: log-normal spread of per-key delays.
    """
    rng = rng or np.random.default_rng()
    base = 60.0 / (wpm * 5)
    for ch in text:
        keyboard.type_char(ch)
        delay = base * np.exp(rng.normal(0, sigma))
        if ch == " ":
            delay *= 1.3
        elif ch in ".,;:!?":
            delay *= 2.0
        time.sleep(delay)
