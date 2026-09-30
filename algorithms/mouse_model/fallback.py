"""
Simple non-learned trajectory generator, used when no trained model exists
(so main.py works before training) and as a baseline in evaluate.py.

Model: minimum-jerk speed profile (bell-shaped velocity, standard in motor
control research), a single random sideways arc, small positional noise and
jittered sampling intervals. Duration grows with log distance (Fitts-like).
"""

import numpy as np


def minimum_jerk(tau):
    """Position fraction 0..1 at normalised time tau in 0..1."""
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


def generate(start, end, rng=None, sample_dt=0.008):
    """
    Return a list of (x, y, dt) from start to end (end point exact).
    sample_dt: mean seconds between points (~125 Hz like a typical mouse).
    """
    rng = rng or np.random.default_rng()
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    v = end - start
    D = float(np.hypot(*v))
    if D < 1:
        return [(float(end[0]), float(end[1]), sample_dt)]

    duration = (0.18 + 0.11 * np.log2(D / 15 + 1)) * rng.uniform(0.85, 1.2)

    # Jittered time grid
    dts = rng.uniform(0.6, 1.4, size=max(3, int(duration / sample_dt))) * sample_dt
    t = np.cumsum(dts)
    tau = np.clip(t / t[-1], 0, 1)
    s = minimum_jerk(tau)

    # Sideways arc (perpendicular to the line) + small noise, fading at the end
    perp = np.array([-v[1], v[0]]) / D
    arc = rng.normal(0, 0.06) * D * np.sin(np.pi * s)
    noise = rng.normal(0, 0.6, size=(len(s), 2)) * (1 - s)[:, None]

    pts = start + s[:, None] * v + arc[:, None] * perp + noise
    pts[-1] = end
    return [(float(x), float(y), float(dt)) for (x, y), dt in zip(pts, dts)]
