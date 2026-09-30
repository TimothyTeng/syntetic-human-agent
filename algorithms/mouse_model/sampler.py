"""
Numpy-only runtime for the trained LSTM-MDN mouse model.

Loads models/mouse_mdn.npz (exported by train.py) and generates human-like
trajectories between two screen points. No TensorFlow needed -> small RAM and
CPU footprint while the agent runs (a trajectory takes a few milliseconds).
"""

import json

import numpy as np

from .features import NormStats, make_input
from .strokes import from_canonical

LOG_SIGMA_MIN, LOG_SIGMA_MAX = -7.0, 5.0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class MouseModel:
    def __init__(self, arrays):
        self.config = json.loads(str(arrays["config"]))
        self.K = self.config["mixtures"]
        self.layers = [(arrays[f"lstm{i}_W"], arrays[f"lstm{i}_U"], arrays[f"lstm{i}_b"])
                       for i in range(self.config["layers"])]
        self.units = self.config["units"]
        self.Wd, self.bd = arrays["mdn_W"], arrays["mdn_b"]
        self.stats = NormStats.from_dict(arrays)

    @classmethod
    def load(cls, path):
        """Load a .npz produced by train.py."""
        with np.load(path, allow_pickle=False) as data:
            return cls({k: data[k] for k in data.files})

    # --- Network forward pass ------------------------------------------------------

    def _init_state(self):
        return [(np.zeros(self.units, np.float32), np.zeros(self.units, np.float32))
                for _ in self.layers]

    def _step(self, x, state):
        """One time step through all LSTM layers + MDN head. Returns (raw_output, new_state)."""
        h_in = x
        new_state = []
        u = self.units
        for (W, U, b), (h, c) in zip(self.layers, state):
            z = h_in @ W + h @ U + b
            i = _sigmoid(z[:u])
            f = _sigmoid(z[u:2 * u])
            g = np.tanh(z[2 * u:3 * u])
            o = _sigmoid(z[3 * u:])
            c = f * c + i * g
            h = o * np.tanh(c)
            new_state.append((h, c))
            h_in = h
        return h_in @ self.Wd + self.bd, new_state

    def forward_sequence(self, X):
        """Teacher-forced pass over (T, n_in) inputs -> (T, K*7+1). Used for parity tests."""
        state = self._init_state()
        outs = []
        for x in X:
            out, state = self._step(x.astype(np.float32), state)
            outs.append(out)
        return np.stack(outs)

    def _split(self, out):
        K = self.K
        logits = out[:K]
        mu = out[K:4 * K].reshape(K, 3)
        log_sigma = np.clip(out[4 * K:7 * K].reshape(K, 3), LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        return logits, mu, log_sigma, out[7 * K]

    # --- Sampling --------------------------------------------------------------------

    def generate_canonical(self, distance, rng, temperature=1.0, max_steps=300,
                           end_tolerance_px=5.0, end_tolerance_frac=0.25):
        """
        Sample one stroke in the canonical frame (start (0,0), target (1,0)).

        distance:    pixel distance of the real move (the model's condition).
        temperature: <1 = more average/smoother, >1 = more varied.
        end_tolerance_px / end_tolerance_frac:
                     the model's "stop" signal is only accepted once the cursor
                     is within max(px, frac*distance) of the target. Without it the
                     model sometimes stops early (too-short, too-straight strokes);
                     too tight and strokes wander. 0.25 was best in a sweep against
                     held-out BMDD browsing strokes (mean KS 0.135 vs 0.198 ungated).
        Returns (points (m+1, 2), dts (m,), ended_naturally: bool).
        """
        st = self.stats
        logd = (np.log(distance) - st.logd_mean) / st.logd_std
        tol = max(end_tolerance_px / distance, end_tolerance_frac)
        state = self._init_state()
        pos = np.zeros(2)
        prev = np.zeros(3, np.float32)
        points, dts = [pos.copy()], []

        for j in range(max_steps):
            x = make_input(prev, j == 0, pos, logd)
            out, state = self._step(x, state)
            logits, mu, log_sigma, end_logit = self._split(out)

            # Choose a mixture component, then sample the step from its Gaussian
            p = np.exp((logits - logits.max()) / temperature)
            k = rng.choice(self.K, p=p / p.sum())
            step_std = mu[k] + np.exp(log_sigma[k]) * np.sqrt(temperature) * rng.standard_normal(3)

            raw = step_std * st.step_std + st.step_mean     # un-standardise
            pos = pos + raw[:2]
            points.append(pos.copy())
            dts.append(float(np.exp(raw[2])))
            prev = step_std.astype(np.float32)

            near = np.hypot(1.0 - pos[0], pos[1]) < tol
            if near and rng.random() < _sigmoid(end_logit):
                return np.array(points), np.array(dts), True
        return np.array(points), np.array(dts), False

    def generate(self, start, end, rng=None, temperature=1.0, retries=3, max_error=0.35, **gen_kwargs):
        """
        Generate a screen-space trajectory from `start` to `end`.

        Returns a list of (x, y, dt) - dt = seconds since the previous point.
        The first point is *after* start; the last point is exactly `end`.
        gen_kwargs are passed to generate_canonical (e.g. end_tolerance_frac).
        Returns None if the model fails repeatedly (caller should use fallback).
        """
        rng = rng or np.random.default_rng()
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        v = end - start
        D = float(np.hypot(*v))
        angle = float(np.arctan2(v[1], v[0]))

        for _ in range(retries):
            canon, dts, ended = self.generate_canonical(D, rng, temperature, **gen_kwargs)
            err = np.array([1.0, 0.0]) - canon[-1]
            if not ended or np.hypot(*err) > max_error:
                continue
            canon = _warp_to_target(canon, err)
            pts = from_canonical(canon, start, D, angle)
            return [(float(px), float(py), float(dt)) for (px, py), dt in zip(pts[1:], dts)]
        return None


def _warp_to_target(canon, err):
    """
    Spread the final landing error along the path (proportional to arc length)
    so the stroke ends exactly on the target without a visible jump.
    """
    seg = np.hypot(*np.diff(canon, axis=0).T)
    frac = np.concatenate([[0.0], np.cumsum(seg)])
    frac = frac / frac[-1] if frac[-1] > 0 else np.linspace(0, 1, len(canon))
    return canon + frac[:, None] * err[None, :]
