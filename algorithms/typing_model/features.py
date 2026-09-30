"""
Feature encoding for the LSTM-MDN motor-timing model, shared by training
(TensorFlow) and generation (numpy).

One sequence = the text keystrokes of one typed sentence (SHIFT etc. dropped).
For keystroke j >= 1 the model sees

    tokens[j] = [key class of previous key, this key, next key]       (ints, keys.VOCAB)
    floats[j] = [log iki_{j-1}, log hold_{j-1}   (standardised, 0 at j=1),
                 is_second,                       (1 at j=1: no previous interval)
                 speed]                           (standardised log median interval
                                                   of the typist = persona)
and predicts
    target[j] = [log iki_j, log hold_j]           (standardised)

where iki_j = press_j - press_{j-1} and hold_j = release_j - press_j.
"""

import numpy as np

from .keys import key_class

N_FLOAT = 4
N_TOK = 3
IKI_RANGE = (0.008, 5.0)
HOLD_RANGE = (0.01, 1.0)


class NormStats:
    def __init__(self, mean, std, speed_mean, speed_std, reg):
        self.mean = np.asarray(mean, np.float32)       # (2,) log iki, log hold
        self.std = np.asarray(std, np.float32)
        self.speed_mean = float(speed_mean)
        self.speed_std = float(speed_std)
        self.reg = [float(reg[0]), float(reg[1])]       # log(median iki) = a + b*log(wpm)

    def to_dict(self):
        return {"norm_mean": self.mean, "norm_std": self.std,
                "speed_mean": np.float32(self.speed_mean), "speed_std": np.float32(self.speed_std),
                "wpm_reg": np.array(self.reg, np.float32)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["norm_mean"], d["norm_std"], float(d["speed_mean"]), float(d["speed_std"]), d["wpm_reg"])

    def speed_from_median(self, med_iki):
        return (np.log(med_iki) - self.speed_mean) / self.speed_std

    def median_from_wpm(self, wpm):
        return float(np.exp(self.reg[0] + self.reg[1] * np.log(max(wpm, 5.0))))


def trial_arrays(trial):
    """(keys, iki, hold) for the text keystrokes of an Aalto Trial."""
    idx = [i for i, k in enumerate(trial.keys) if k is not None]
    if len(idx) < 3:
        return None
    keys = [trial.keys[i] for i in idx]
    press = trial.press[idx]
    hold = np.clip(trial.release[idx] - press, *HOLD_RANGE)
    iki = np.clip(np.diff(press, prepend=press[0]), *IKI_RANGE)
    return keys, iki, hold


def participant_median(trials):
    ik = []
    for tr in trials:
        a = trial_arrays(tr)
        if a is not None:
            ik.append(a[1][1:])
    if not ik:
        return None
    ik = np.concatenate(ik)
    ik = ik[ik < 2.0]
    return float(np.median(ik)) if len(ik) >= 50 else None


def encode_trial(keys, iki, hold, speed, stats):
    """-> (tokens (m,3) int32, floats (m,N_FLOAT) f32, targets (m,2) f32), m = n-1."""
    n = len(keys)
    cls = np.array([key_class(k) for k in keys], np.int32)
    nxt = np.append(cls[1:], 0)
    z = (np.column_stack([np.log(iki), np.log(hold)]) - stats.mean) / stats.std
    m = n - 1
    tok = np.column_stack([cls[:-1], cls[1:], nxt[1:]]).astype(np.int32)
    fl = np.zeros((m, N_FLOAT), np.float32)
    fl[1:, 0] = z[1:-1, 0]            # previous interval (none for the 2nd key)
    fl[:, 1] = z[:-1, 1]              # previous hold
    fl[0, 2] = 1.0
    fl[:, 3] = speed
    return tok, fl, z[1:].astype(np.float32)


def make_step_input(prev_tok, tok, next_tok, prev_z, is_second, speed):
    """One step at generation time (mirrors encode_trial)."""
    fl = np.zeros(N_FLOAT, np.float32)
    if not is_second:
        fl[0] = prev_z[0]
    fl[1] = prev_z[1]
    fl[2] = 1.0 if is_second else 0.0
    fl[3] = speed
    return np.array([prev_tok, tok, next_tok], np.int32), fl
