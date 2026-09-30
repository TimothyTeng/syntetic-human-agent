"""
Empirical click-timing distributions learned from the dataset.

  hold  - how long the left button stays down during a click (press -> release)
  dwell - how long the cursor rests on the target before pressing

We store (a subsample of) the real measured values and sample from them with a
little multiplicative jitter, which reproduces the real distribution shape
without assuming a formula. Defaults are used if no dataset has been fitted.
"""

import json

import numpy as np

MAX_SAMPLES = 5000

# Fallback log-normal parameters (median seconds, sigma) when no data is available.
DEFAULT_HOLD = (0.095, 0.35)
DEFAULT_DWELL = (0.12, 0.6)

HOLD_RANGE = (0.03, 0.40)
DWELL_RANGE = (0.0, 1.5)


class ClickStats:
    def __init__(self, holds=None, dwells=None):
        self.holds = np.asarray(holds if holds is not None else [], dtype=float)
        self.dwells = np.asarray(dwells if dwells is not None else [], dtype=float)

    # --- Build / persist ----------------------------------------------------

    @classmethod
    def fit(cls, strokes, rng=None):
        """Collect hold and dwell times from Stroke objects."""
        rng = rng or np.random.default_rng(0)
        holds = np.array([s.hold for s in strokes if np.isfinite(s.hold)])
        dwells = np.array([s.dwell for s in strokes if np.isfinite(s.dwell)])
        holds = holds[(holds >= HOLD_RANGE[0]) & (holds <= HOLD_RANGE[1])]
        dwells = dwells[(dwells >= DWELL_RANGE[0]) & (dwells <= DWELL_RANGE[1])]
        if len(holds) > MAX_SAMPLES:
            holds = rng.choice(holds, MAX_SAMPLES, replace=False)
        if len(dwells) > MAX_SAMPLES:
            dwells = rng.choice(dwells, MAX_SAMPLES, replace=False)
        return cls(holds, dwells)

    @classmethod
    def default(cls):
        """Stats with no data: sampling falls back to log-normal defaults."""
        return cls()

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"holds": self.holds.round(4).tolist(),
                       "dwells": self.dwells.round(4).tolist()}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d.get("holds"), d.get("dwells"))

    # --- Sampling -------------------------------------------------------------

    @staticmethod
    def _sample(values, default, bounds, rng, jitter=0.07):
        if len(values):
            v = rng.choice(values) * np.exp(rng.normal(0, jitter))
        else:
            median, sigma = default
            v = median * np.exp(rng.normal(0, sigma))
        return float(np.clip(v, *bounds))

    def sample_hold(self, rng):
        """Seconds to keep the button pressed."""
        return self._sample(self.holds, DEFAULT_HOLD, HOLD_RANGE, rng)

    def sample_dwell(self, rng):
        """Seconds to rest on the target before pressing."""
        return self._sample(self.dwells, DEFAULT_DWELL, DWELL_RANGE, rng)

    def summary(self):
        """Human-readable medians, for logging."""
        med = lambda a: f"{np.median(a) * 1000:.0f} ms (n={len(a)})" if len(a) else "default"  # noqa: E731
        return f"hold {med(self.holds)}, dwell {med(self.dwells)}"
