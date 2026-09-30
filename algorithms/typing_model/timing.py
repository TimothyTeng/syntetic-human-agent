"""
Empirical motor-timing model: per-bigram inter-key intervals and per-key hold times.

For every participant, intervals are divided by that participant's median
interval (so the shape is speed-independent). For each key-class bigram
(prev key -> key) we store quantiles of log(interval / median). At runtime an
interval is  persona_median * exp(quantile(u))  where u comes from an AR(1)
Gaussian process, giving the slow tempo drift real typing has (intervals are
not independent from key to key). Hold (press -> release) times are modelled the
same way per key.

The persona median interval is derived from words-per-minute with a log-log
regression fitted on the participants.

Fallback when nothing has been fitted: log-normal intervals with simple rules
(same-hand bigrams slower, Backspace runs faster, pause before correcting).
"""

import json
import math

import numpy as np

from .keys import BKSP, hand, needs_shift, timing_token

N_Q = 41
QGRID = np.linspace(0.0, 1.0, N_Q)
MIN_BIGRAM = 60
MIN_UNIGRAM = 60
MAX_IKI = 5.0


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _quantiles(values):
    """N_Q quantiles, with the extreme 0.5% trimmed (sensor glitches)."""
    q = np.quantile(values, np.linspace(0.005, 0.995, N_Q))
    return np.round(q, 4).tolist()


def _default_dict():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.38, 20000)
    hold = rng.normal(0, 0.25, 20000)
    return {
        "iki": {"global": _quantiles(base), "bigram": {}, "unigram": {}},
        "hold": {"global": _quantiles(hold), "unigram": {}},
        "iki_reg": [math.log(10.5), -1.0],     # log(median_iki) = a + b*log(wpm)
        "hold_reg": [math.log(0.105), -0.1],
        "rho": 0.25,
        "shift": {"lead": [0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25], "lag": [-0.02, 0.0, 0.02, 0.04, 0.06, 0.1]},
        "source": "defaults",
    }


class BigramTimer:
    def __init__(self, d=None):
        self.d = d or _default_dict()
        cv = lambda x: np.asarray(x, float)  # noqa: E731
        self.iki_global = cv(self.d["iki"]["global"])
        self.iki_bigram = {k: cv(v) for k, v in self.d["iki"]["bigram"].items()}
        self.iki_unigram = {k: cv(v) for k, v in self.d["iki"]["unigram"].items()}
        self.hold_global = cv(self.d["hold"]["global"])
        self.hold_unigram = {k: cv(v) for k, v in self.d["hold"]["unigram"].items()}
        self.fitted = self.d.get("source") != "defaults"

    # --- Fit ---------------------------------------------------------------------

    @classmethod
    def fit(cls, participants, rng=None, verbose=True):
        """
        participants: iterable of (wpm, [Trial, ...]); Trial has .keys (chars, BKSP,
        or None for non-text keys), .press and .release (seconds).
        """
        rng = rng or np.random.default_rng(0)
        big, uni, hold_uni = {}, {}, {}
        all_iki, all_hold = [], []
        reg_iki, reg_hold = [], []
        lag_pairs = []
        shift_lead, shift_lag = [], []
        iki_spread, hold_spread = [], []     # per-person spread of the (normalised) distributions

        for n, (wpm, trials) in enumerate(participants):
            for tr in trials:
                _shift_timing(tr, shift_lead, shift_lag)
            rows = []      # (prev_tok, tok, iki, hold, trial_idx)
            for ti, tr in enumerate(trials):
                prev_t, prev_tok = None, None
                for k, p, r in zip(tr.keys, tr.press, tr.release):
                    if k is None:
                        continue
                    tok = timing_token(k)
                    h = r - p
                    if prev_t is not None:
                        iki = p - prev_t
                        if 0.0 < iki < MAX_IKI:
                            rows.append((prev_tok, tok, iki, h, ti))
                    prev_t, prev_tok = p, tok
            if len(rows) < 100 or not wpm or wpm <= 0:
                continue
            iki = np.array([r[2] for r in rows])
            hold = np.array([r[3] for r in rows])
            med = float(np.median(iki[iki < 2.0]))
            ok_h = (hold > 0.005) & (hold < 1.0)
            med_h = float(np.median(hold[ok_h])) if ok_h.any() else 0.1
            reg_iki.append((math.log(wpm), math.log(med)))
            reg_hold.append((math.log(wpm), math.log(med_h)))
            li = np.log(iki / med)
            lh = np.log(np.where(ok_h, hold, med_h) / med_h)
            for (pt, t, _, _, ti), v, hv, okh in zip(rows, li, lh, ok_h):
                big.setdefault(f"{pt}|{t}", []).append(v)
                uni.setdefault(t, []).append(v)
                if okh:
                    hold_uni.setdefault(t, []).append(hv)
            all_iki.append(li)
            all_hold.append(lh[ok_h])
            iki_spread.append(np.subtract(*np.percentile(li, [75, 25])))
            hold_spread.append(np.subtract(*np.percentile(lh[ok_h], [75, 25])) if ok_h.sum() > 20 else np.nan)
            # lag-1 correlation of residuals within trials (tempo drift)
            if n % 5 == 0:
                tis = np.array([r[4] for r in rows])
                resid = li - np.median(li)
                same = tis[1:] == tis[:-1]
                lag_pairs.append(np.column_stack([resid[:-1][same], resid[1:][same]]))
            if verbose and (n + 1) % 1000 == 0:
                print(f"  timing: {n + 1} participants")

        def reg(pairs):
            a = np.array(pairs)
            b, c = np.polyfit(a[:, 0], a[:, 1], 1)
            return [round(float(c), 5), round(float(b), 5)]

        def resid_sd(pairs):
            a = np.array(pairs)
            b, c = np.polyfit(a[:, 0], a[:, 1], 1)
            return round(float(np.std(a[:, 1] - (b * a[:, 0] + c))), 4)

        def log_sd(v):
            v = np.asarray(v, float)
            v = v[np.isfinite(v) & (v > 0)]
            return round(float(np.std(np.log(v))), 4) if len(v) > 10 else 0.0

        lp = np.concatenate(lag_pairs)
        lp = lp[np.all(np.abs(lp) < 2.5, axis=1)]
        rho = float(np.corrcoef(lp[:, 0], lp[:, 1])[0, 1])
        d = {
            "iki": {"global": _quantiles(np.concatenate(all_iki)),
                    "bigram": {k: _quantiles(v) for k, v in big.items() if len(v) >= MIN_BIGRAM},
                    "unigram": {k: _quantiles(v) for k, v in uni.items() if len(v) >= MIN_UNIGRAM}},
            "hold": {"global": _quantiles(np.concatenate(all_hold)),
                     "unigram": {k: _quantiles(v) for k, v in hold_uni.items() if len(v) >= MIN_UNIGRAM}},
            "iki_reg": reg(reg_iki),
            "hold_reg": reg(reg_hold),
            "rho": round(rho, 4),
            "shift": {"lead": _sub(shift_lead, rng), "lag": _sub(shift_lag, rng)},
            # between-person variation, so personas differ like real typists do
            "persona": {"hold_level_sd": resid_sd(reg_hold), "iki_spread_sd": log_sd(iki_spread),
                        "hold_spread_sd": log_sd(hold_spread)},
            "n_participants": len(reg_iki),
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

    # --- Persona -----------------------------------------------------------------------

    def median_iki(self, wpm):
        a, b = self.d["iki_reg"]
        return math.exp(a + b * math.log(max(wpm, 5)))

    def median_hold(self, wpm):
        a, b = self.d["hold_reg"]
        return math.exp(a + b * math.log(max(wpm, 5)))

    def sample_shift(self, rng):
        """(lead, lag): Shift goes down `lead` s before the key, up `lag` s after its release."""
        sh = self.d.get("shift") or _default_dict()["shift"]
        return float(rng.choice(sh["lead"])), float(rng.choice(sh["lag"]))

    def session(self, wpm, rng, temperature=1.0):
        return _TimerSession(self, wpm, rng, temperature)

    # --- Lookup ----------------------------------------------------------------------

    def _iki_q(self, prev, key):
        pt, t = timing_token(prev) if prev is not None else None, timing_token(key)
        q = self.iki_bigram.get(f"{pt}|{t}")
        if q is not None:
            return q, 0.0
        q = self.iki_unigram.get(t)
        if q is not None:
            return q, 0.0
        # rule-based adjustments on the global shape (defaults only)
        adj = 0.0
        if key == BKSP:
            adj = -0.25 if prev == BKSP else 0.8
        elif prev == BKSP:
            adj = 0.4
        elif prev is not None and hand(prev) == hand(key) and hand(key) in (0, 1):
            adj = 0.1
        elif key == " " or prev == " ":
            adj = 0.15
        return self.iki_global, adj

    def _hold_q(self, key):
        return self.hold_unigram.get(timing_token(key), self.hold_global)


def _sub(values, rng, n=3000):
    a = np.asarray(values, float)
    if len(a) > n:
        a = rng.choice(a, n, replace=False)
    return np.round(a, 4).tolist()


def _spread(v, q, factor):
    """Widen / narrow a sampled log value around its distribution's median."""
    med = q[len(q) // 2]
    return med + (v - med) * factor


def _shift_timing(tr, leads, lags):
    """Shift press -> shifted key press (lead) and shifted key release -> Shift release (lag)."""
    shifts = [(p, r) for n, p, r in zip(tr.names, tr.press, tr.release) if n == "SHIFT"]
    if not shifts:
        return
    for k, p, r in zip(tr.keys, tr.press, tr.release):
        if k is None or k == BKSP or not needs_shift(k):
            continue
        held = [(sp, sr) for sp, sr in shifts if sp <= p <= sr]
        if held:
            sp, sr = held[-1]
            if 0 < p - sp < 1.5:
                leads.append(p - sp)
                lags.append(sr - r)


class _TimerSession:
    """Stateful sampler for one typing burst (keeps the AR(1) tempo state)."""

    def __init__(self, timer, wpm, rng, temperature):
        self.t = timer
        self.rng = rng
        self.temp = temperature
        self.med = timer.median_iki(wpm)
        self.med_hold = timer.median_hold(wpm)
        self.rho = timer.d["rho"]
        self.z = rng.standard_normal()
        pers = timer.d.get("persona") or {}
        self.med_hold *= math.exp(rng.normal(0, pers.get("hold_level_sd", 0.0)))
        self.iki_spread = float(np.clip(math.exp(rng.normal(0, pers.get("iki_spread_sd", 0.0))), 0.6, 1.7))
        self.hold_spread = float(np.clip(math.exp(rng.normal(0, pers.get("hold_spread_sd", 0.0))), 0.6, 1.7))

    @property
    def median_iki(self):
        return self.med

    def _u(self):
        self.z = self.rho * self.z + math.sqrt(1 - self.rho ** 2) * self.rng.standard_normal()
        return _phi(self.z * self.temp)

    def sample(self, prev, key, nxt=None):
        """(interval since previous press, hold) in seconds for pressing `key`."""
        q, adj = self.t._iki_q(prev, key)
        v = _spread(float(np.interp(self._u(), QGRID, q)), q, self.iki_spread)
        iki = self.med * math.exp(v + adj)
        hq = self.t._hold_q(key)
        hv = _spread(float(np.interp(_phi(self.rng.standard_normal() * self.temp), QGRID, hq)), hq, self.hold_spread)
        hold = self.med_hold * math.exp(hv)
        return iki, float(np.clip(hold, 0.02, 0.6))
