"""
Numpy-only runtime for the trained LSTM-MDN keystroke-timing model
(models/typing_mdn.npz, exported by train.py). Same interface as
timing.BigramTimer: .session(wpm, rng, temperature).sample(prev, key, next).
"""

import json
import os

import numpy as np

from .features import NormStats, make_step_input
from .keys import key_class
from .timing import BigramTimer

LOG_SIGMA_MIN, LOG_SIGMA_MAX = -5.0, 3.0
RESET_AFTER = 80        # training sequences are sentences; restart context on long texts


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class TypingMDN:
    def __init__(self, arrays, aux=None):
        self.config = json.loads(str(arrays["config"]))
        self.K = self.config["mixtures"]
        self.units = self.config["units"]
        self.emb = arrays["emb"]
        self.layers = [(arrays[f"lstm{i}_W"], arrays[f"lstm{i}_U"], arrays[f"lstm{i}_b"])
                       for i in range(self.config["layers"])]
        self.Wd, self.bd = arrays["mdn_W"], arrays["mdn_b"]
        self.stats = NormStats.from_dict(arrays)
        self.wpm_scale = float(arrays["wpm_scale"]) if "wpm_scale" in arrays else 1.0
        self.aux = aux or BigramTimer.default()     # Shift timing lives in the bigram stats
        self.d = {"source": "mdn"}

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        aux_path = os.path.join(os.path.dirname(path), "typing_bigrams.json")
        aux = BigramTimer.load(aux_path) if os.path.exists(aux_path) else None
        return cls(arrays, aux)

    # --- persona / helpers ---------------------------------------------------------------

    def median_iki(self, wpm):
        # wpm_scale: measured achieved/requested speed of generated text (see train.calibrate_speed)
        return self.stats.median_from_wpm(wpm / self.wpm_scale)

    def sample_shift(self, rng):
        return self.aux.sample_shift(rng)

    def session(self, wpm, rng, temperature=1.0):
        return _MDNSession(self, wpm, rng, temperature)

    # --- forward pass --------------------------------------------------------------------

    def init_state(self):
        return [(np.zeros(self.units, np.float32), np.zeros(self.units, np.float32)) for _ in self.layers]

    def step(self, tokens, floats, state):
        x = np.concatenate([self.emb[tokens].reshape(-1), floats]).astype(np.float32)
        u = self.units
        new = []
        for (W, U, b), (h, c) in zip(self.layers, state):
            z = x @ W + h @ U + b
            i, f = _sigmoid(z[:u]), _sigmoid(z[u:2 * u])
            g, o = np.tanh(z[2 * u:3 * u]), _sigmoid(z[3 * u:])
            c = f * c + i * g
            h = o * np.tanh(c)
            new.append((h, c))
            x = h
        return x @ self.Wd + self.bd, new

    def forward_sequence(self, tokens, floats):
        """Teacher-forced pass over (T,3)/(T,F) inputs -> (T, K*5). Used for parity tests."""
        state = self.init_state()
        outs = []
        for t, f in zip(tokens, floats):
            o, state = self.step(t, f, state)
            outs.append(o)
        return np.stack(outs)

    def split(self, out):
        K = self.K
        return (out[:K], out[K:3 * K].reshape(K, 2),
                np.clip(out[3 * K:5 * K].reshape(K, 2), LOG_SIGMA_MIN, LOG_SIGMA_MAX))


class _MDNSession:
    def __init__(self, model, wpm, rng, temperature):
        self.m, self.rng, self.temp = model, rng, temperature
        self.med = model.median_iki(wpm)
        self.speed = float(model.stats.speed_from_median(self.med))
        pers = model.aux.d.get("persona") or {}
        self.hold_factor = float(np.exp(rng.normal(0, pers.get("hold_level_sd", 0.0))))
        self._reset()

    @property
    def median_iki(self):
        return self.med

    def _reset(self):
        self.state = self.m.init_state()
        self.prev_z = np.zeros(2, np.float32)
        self.n = 0

    def sample(self, prev, key, nxt=None):
        st = self.m.stats
        if prev is None or (self.n >= RESET_AFTER and prev == " "):
            self._reset()
        if prev is None:
            # first key of a sequence has no interval; hold from the model's typical value
            self.n = 1
            hold = float(np.exp(st.mean[1] + st.std[1] * self.rng.normal(0, 0.5)))
            self.prev_z = np.array([0.0, (np.log(hold) - st.mean[1]) / st.std[1]], np.float32)
            return self.med, float(np.clip(hold, 0.02, 0.6))
        tok, fl = make_step_input(key_class(prev), key_class(key), key_class(nxt) if nxt else 0,
                                  self.prev_z, self.n == 1, self.speed)
        out, self.state = self.m.step(tok, fl, self.state)
        logits, mu, log_sigma = self.m.split(out)
        p = np.exp((logits - logits.max()) / self.temp)
        k = self.rng.choice(self.m.K, p=p / p.sum())
        z = mu[k] + np.exp(log_sigma[k]) * np.sqrt(self.temp) * self.rng.standard_normal(2)
        z = np.clip(z, -6, 6).astype(np.float32)
        self.prev_z = z
        self.n += 1
        iki, hold = np.exp(z * st.std + st.mean)
        return float(iki), float(np.clip(hold * self.hold_factor, 0.02, 0.6))
