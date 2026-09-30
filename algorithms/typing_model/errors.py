"""
Typo model learned from transcription typing (Aalto 136M keystrokes).

Each trial is replayed into a text buffer and compared with the target sentence.
An *error onset* is a keystroke that turns a correct prefix into an incorrect one.
For every onset we record:

    type       adjacent-key substitution, other substitution, omission (skipped a
               letter), insertion (extra / doubled letter), transposition, case slip
    delay      how many more characters were typed before the first Backspace
               (0 = noticed immediately)
    extra      Backspaces beyond what was needed to reach the error (over-deletion,
               e.g. deleting the whole word)
    corrected  whether the error was fixed at all

Rates are per keystroke typed from a correct prefix, binned by the typist's speed
(fast typists make fewer errors).
"""

import json

import numpy as np

from .keys import BKSP, neighbours, timing_token

ERROR_TYPES = ["adjacent", "substitution", "omission", "insertion", "transposition", "case"]
WPM_BINS = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120]
MAX_SAMPLES = 5000


def classify_error(typed, target, p):
    """
    typed:  characters appended from the onset on (typed[0] is the wrong one)
    target: the target sentence; p = index of the error in target
    """
    t0 = target[p] if p < len(target) else ""
    t1 = target[p + 1] if p + 1 < len(target) else ""
    t2 = target[p + 2] if p + 2 < len(target) else ""
    tp = target[p - 1] if p > 0 else ""
    c0 = typed[0]
    c1 = typed[1] if len(typed) > 1 else ""
    if t0 and c0.lower() == t0.lower() and c0 != t0:
        return "case"
    if t1 and c0 == t1 and c1 == t0 and t0 != t1:
        return "transposition"
    if t1 and c0 == t1 and (not c1 or c1 == t2):
        return "omission"
    if (tp and c0 == tp) or (t0 and c1 == t0):
        return "insertion"
    if t0 and c0 in neighbours(t0):
        return "adjacent"
    return "substitution"


def _default_dict():
    return {
        "rate_by_wpm": [[20, 0.045], [40, 0.03], [60, 0.022], [80, 0.017], [100, 0.014], [120, 0.012]],
        "type_probs": {"adjacent": 0.38, "substitution": 0.1, "omission": 0.16, "insertion": 0.18,
                       "transposition": 0.1, "case": 0.08},
        "delay": [0] * 50 + [1] * 25 + [2] * 12 + [3] * 7 + [4] * 4 + [6] * 2,
        "extra": [0] * 85 + [1] * 8 + [2] * 4 + [4] * 3,
        "p_uncorrected": 0.08,
        "token_weight": {},
        "correction_run_rate_by_wpm": [[20, 0.04], [60, 0.02], [120, 0.011]],
        "source": "defaults",
    }


class ErrorStats:
    def __init__(self, d=None):
        self.d = d or _default_dict()
        self._types = list(self.d["type_probs"].keys())
        p = np.array([self.d["type_probs"][t] for t in self._types], float)
        self._type_p = p / p.sum()
        self._delay = np.asarray(self.d["delay"], int)
        self._extra = np.asarray(self.d["extra"], int)
        rb = np.array(self.d["rate_by_wpm"], float)
        self._rate_wpm, self._rate = rb[:, 0], rb[:, 1]

    # --- Fit ---------------------------------------------------------------------

    @classmethod
    def fit(cls, trials_by_participant, rng=None, verbose=True):
        """
        trials_by_participant: iterable of (wpm, [Trial, ...]) where Trial has
        .sentence, .user_input and .keys (list of chars; BKSP for Backspace, None
        for keys that do not change the text).
        """
        rng = rng or np.random.default_rng(0)
        type_counts = {t: 0 for t in ERROR_TYPES}
        delays, extras = [], []
        n_err_tok, n_opp_tok = {}, {}
        per_part = []              # (wpm, errors, opportunities, correction runs, keystrokes)
        n_uncorrected, n_onsets = 0, 0

        for n_part, (wpm, trials) in enumerate(trials_by_participant):
            errs = opps = runs = kst = 0
            for tr in trials:
                target = tr.sentence
                buf = []
                keys = [k for k in tr.keys if k is not None]
                if not keys:
                    continue
                kst += len(keys)
                runs += sum(1 for i, k in enumerate(keys) if k == BKSP and (i == 0 or keys[i - 1] != BKSP))
                i = 0
                while i < len(keys):
                    k = keys[i]
                    if k == BKSP:
                        if buf:
                            buf.pop()
                        i += 1
                        continue
                    p = len(buf)
                    was_correct = "".join(buf) == target[:p]
                    buf.append(k)
                    if not was_correct:
                        i += 1
                        continue
                    tok = timing_token(target[p]) if p < len(target) else "<other>"
                    opps += 1
                    n_opp_tok[tok] = n_opp_tok.get(tok, 0) + 1
                    if p < len(target) and k == target[p]:
                        i += 1
                        continue
                    # --- error onset at p ---
                    errs += 1
                    n_onsets += 1
                    n_err_tok[tok] = n_err_tok.get(tok, 0) + 1
                    typed = [k]
                    j = i + 1
                    while j < len(keys) and keys[j] != BKSP:
                        typed.append(keys[j])
                        j += 1
                    type_counts[classify_error(typed, target, p)] += 1
                    if j >= len(keys):                    # never corrected in this trial
                        n_uncorrected += 1
                    else:
                        delays.append(len(typed) - 1)
                        r = j
                        while r < len(keys) and keys[r] == BKSP:
                            r += 1
                        extras.append(max(0, (r - j) - len(typed)))   # beyond the error itself
                    # replay the typed continuation (it is part of the buffer too)
                    for c in typed[1:]:
                        buf.append(c)
                    i = j
            if opps:
                per_part.append((wpm, errs, opps, runs, kst))
            if verbose and (n_part + 1) % 1000 == 0:
                print(f"  errors: {n_part + 1} participants, {n_onsets} error onsets")

        pp = np.array(per_part, float)
        rate_by_wpm, run_rate = [], []
        edges = [0] + WPM_BINS + [1000]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pp[:, 0] >= lo) & (pp[:, 0] < hi)
            if m.sum() >= 20:
                centre = float(np.median(pp[m, 0]))
                rate_by_wpm.append([round(centre, 1), round(float(pp[m, 1].sum() / pp[m, 2].sum()), 5)])
                run_rate.append([round(centre, 1), round(float(pp[m, 3].sum() / pp[m, 4].sum()), 5)])
        total = sum(type_counts.values()) or 1
        overall = sum(n_err_tok.values()) / max(sum(n_opp_tok.values()), 1)
        token_weight = {t: round((n_err_tok.get(t, 0) / n) / overall, 3)
                        for t, n in n_opp_tok.items() if n >= 500}

        def sub(a):
            a = np.asarray(a, int)
            return (rng.choice(a, MAX_SAMPLES, replace=False) if len(a) > MAX_SAMPLES else a).tolist()

        d = {
            "rate_by_wpm": rate_by_wpm or _default_dict()["rate_by_wpm"],
            "type_probs": {t: round(c / total, 4) for t, c in type_counts.items()},
            "delay": sub(delays) or _default_dict()["delay"],
            "extra": sub(extras) or _default_dict()["extra"],
            "p_uncorrected": round(n_uncorrected / max(n_onsets, 1), 4),
            "token_weight": token_weight,
            "correction_run_rate_by_wpm": run_rate,
            "n_onsets": n_onsets,
            "n_participants": len(per_part),
            "source": "aalto",
        }
        return cls(d)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.d, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    @classmethod
    def default(cls):
        return cls()

    # --- Sampling --------------------------------------------------------------------

    def scale_rate(self, factor):
        self.d["rate_by_wpm"] = [[w, round(r * factor, 5)] for w, r in self.d["rate_by_wpm"]]
        self._rate = self._rate * factor

    def rate(self, wpm):
        """Error onsets per keystroke for a typist of this speed."""
        return float(np.interp(wpm, self._rate_wpm, self._rate))

    def correction_run_rate(self, wpm):
        rr = np.array(self.d.get("correction_run_rate_by_wpm") or [[60, 0.02]], float)
        return float(np.interp(wpm, rr[:, 0], rr[:, 1]))

    def p_error(self, ch, wpm, scale=1.0):
        w = self.d["token_weight"].get(timing_token(ch), 1.0)
        return min(0.5, self.rate(wpm) * w * scale)

    def sample_type(self, rng):
        return self._types[rng.choice(len(self._types), p=self._type_p)]

    def sample_delay(self, rng):
        return int(rng.choice(self._delay))

    def sample_extra(self, rng):
        return int(rng.choice(self._extra))

    def summary(self):
        d = self.d
        rates = ", ".join(f"{w:.0f}wpm {r:.1%}" for w, r in d["rate_by_wpm"])
        types = ", ".join(f"{t} {p:.0%}" for t, p in d["type_probs"].items())
        return (f"source={d['source']}; errors/keystroke: {rates}; types: {types}; "
                f"noticed immediately {np.mean(np.asarray(d['delay']) == 0):.0%}; "
                f"left uncorrected {d['p_uncorrected']:.0%}")


def make_typo(target, j, etype, rng):
    """
    Produce the erroneous keystrokes for target[j:] of the given type.
    Returns (typed_chars, n_target_chars_consumed). Falls back to a substitution
    when the requested type does not apply at this position.
    """
    t0 = target[j]
    t1 = target[j + 1] if j + 1 < len(target) else None
    if etype == "case" and t0.isalpha():
        return [t0.swapcase()], 1
    if etype == "transposition" and t1 is not None and t1 != t0 and t1 not in "\n":
        return [t1, t0], 2
    if etype == "omission" and t1 is not None and t1 != "\n":
        return [t1], 2               # skipped t0; typed the next char instead
    if etype == "insertion":
        nb = neighbours(t0)
        extra = t0 if (rng.random() < 0.5 or not nb) else nb[rng.integers(len(nb))]
        return [t0, extra], 1
    nb = neighbours(t0)
    if etype in ("adjacent", "substitution", "case", "transposition", "omission") and nb:
        return [nb[rng.integers(len(nb))]], 1
    return [], 0                     # no plausible typo here
