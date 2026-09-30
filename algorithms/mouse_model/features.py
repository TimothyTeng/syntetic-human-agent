"""
Feature encoding shared by training (TensorFlow) and generation (numpy).

Each stroke with n points becomes m = n-1 steps. For step j the model sees:

    input[j]  = [dx_prev, dy_prev, logdt_prev,  # previous step (standardised), zeros at j=0
                 is_first,                      # 1 at j=0 else 0
                 rem_x, rem_y,                  # remaining vector to target (canonical units)
                 logD]                          # log pixel distance (standardised)
    target[j] = [dx, dy, logdt,                 # this step (standardised)
                 end]                           # 1 on the last step

dx, dy are in the canonical frame (see strokes.to_canonical); dt in seconds.
"""

import numpy as np

from .strokes import to_canonical

N_IN = 7
N_TARGET = 4


class NormStats:
    """Mean/std used to standardise step deltas and log distance."""

    def __init__(self, step_mean, step_std, logd_mean, logd_std):
        self.step_mean = np.asarray(step_mean, dtype=np.float32)  # (3,) dx, dy, logdt
        self.step_std = np.asarray(step_std, dtype=np.float32)
        self.logd_mean = float(logd_mean)
        self.logd_std = float(logd_std)

    @classmethod
    def fit(cls, strokes):
        """Compute statistics from a list of Stroke objects."""
        steps, logds = [], []
        for s in strokes:
            raw, D = raw_steps(s)
            steps.append(raw)
            logds.append(np.log(D))
        steps = np.concatenate(steps)
        return cls(steps.mean(0), steps.std(0) + 1e-6, np.mean(logds), np.std(logds) + 1e-6)

    def to_dict(self):
        return {"step_mean": self.step_mean, "step_std": self.step_std,
                "logd_mean": np.float32(self.logd_mean), "logd_std": np.float32(self.logd_std)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["step_mean"], d["step_std"], float(d["logd_mean"]), float(d["logd_std"]))


def raw_steps(stroke):
    """Return ((m,3) array of [dx, dy, log dt] in canonical frame, pixel distance D)."""
    canon, D, _ = to_canonical(stroke.points)
    d = np.diff(canon, axis=0)
    dt = np.clip(np.diff(stroke.times), 1e-3, 1.0)
    return np.column_stack([d, np.log(dt)]).astype(np.float32), D


def encode_stroke(stroke, stats):
    """
    Build (inputs (m, N_IN), targets (m, N_TARGET)) float32 arrays for one stroke.
    """
    raw, D = raw_steps(stroke)
    m = len(raw)
    std_steps = (raw - stats.step_mean) / stats.step_std

    # Canonical position before each step (cumulative sum of raw dx, dy)
    pos = np.vstack([[0.0, 0.0], np.cumsum(raw[:, :2], axis=0)[:-1]])
    rem = np.array([1.0, 0.0]) - pos

    X = np.zeros((m, N_IN), dtype=np.float32)
    X[1:, 0:3] = std_steps[:-1]
    X[0, 3] = 1.0
    X[:, 4:6] = rem
    X[:, 6] = (np.log(D) - stats.logd_mean) / stats.logd_std

    Y = np.zeros((m, N_TARGET), dtype=np.float32)
    Y[:, 0:3] = std_steps
    Y[-1, 3] = 1.0
    return X, Y


def make_input(prev_std_step, is_first, pos, logd_std):
    """Build one input vector during generation (mirrors encode_stroke)."""
    x = np.zeros(N_IN, dtype=np.float32)
    if not is_first:
        x[0:3] = prev_std_step
    x[3] = 1.0 if is_first else 0.0
    x[4] = 1.0 - pos[0]
    x[5] = -pos[1]
    x[6] = logd_std
    return x
