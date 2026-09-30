"""
Composition-level behaviour learned from free essay writing (KLiCKe logs):
where people pause, how long, how often they delete and rewrite, and how often
they go back into earlier text to revise it.

Pause classes (the pause *before* a keystroke, press-to-press):
    in_word          letter after letter
    word_end         space / punctuation right after a word
    between_words    first letter of a word after a plain space
    after_clause     first letter after ", " / "; " / ": "
    after_sentence   first letter after ". " / "? " / "! "
    after_paragraph  first key after Enter
    punct            before a punctuation mark
    before_delete    first Backspace of a deletion run
    in_delete        Backspace after Backspace
    after_delete     first key typed after a deletion run
    other

A pause >= LONG_PAUSE seconds is a "long" (cognitive) pause. Short pauses are
stored as ratios to the writer's median in-word interval so they scale with the
typist's speed; long pauses are stored in seconds.

Defaults (used when no fitted file exists) are rough values from the writing-
process literature and are replaced by `fit_stats.py --klicke ...`.
"""

import json

import numpy as np

from .dataset_klicke import INPUT, NAV, REMOVE

LONG_PAUSE = 2.0
MAX_FIX = 30          # in-text episodes changing more chars than this are content insertions, not fixes
MAX_SAMPLES = 4000
PAUSE_CLASSES = ["in_word", "word_end", "between_words", "after_clause", "after_sentence",
                 "after_paragraph", "punct", "before_delete", "in_delete", "after_delete", "other"]
DELETE_BUCKETS = [(1, 2), (3, 9), (10, 29), (30, 400)]

_SENT = set(".?!")
_CLAUSE = set(",;:")


def _is_word(c):
    return c == "q" or c.isalnum()


def pause_class(ch, prev_text, prev_kind, kind):
    """Label the pause before a production keystroke (see module docstring)."""
    if kind == REMOVE:
        return "in_delete" if prev_kind == REMOVE else "before_delete"
    if prev_kind == REMOVE:
        return "after_delete"
    if not prev_text:
        return "other"
    last = prev_text[-1]
    if last == "\n":
        return "after_paragraph"
    if _is_word(ch):
        if _is_word(last):
            return "in_word"
        if last == " ":
            before = prev_text[-2] if len(prev_text) >= 2 else ""
            if before in _SENT:
                return "after_sentence"
            if before in _CLAUSE:
                return "after_clause"
            return "between_words"
        if last in _SENT:
            return "after_sentence"
        if last in _CLAUSE:
            return "after_clause"
        return "other"
    if ch == " " and _is_word(last):
        return "word_end"
    if ch in _SENT or ch in _CLAUSE:
        return "punct"
    return "other"


def _default_dict():
    # Literature-ish priors (Medimorec & Risko 2017; Conijn et al. 2019)
    pc = {
        "in_word": (0.01, 1.0, 0.25), "word_end": (0.02, 1.1, 0.3), "between_words": (0.07, 1.6, 0.4),
        "after_clause": (0.20, 2.0, 0.45), "after_sentence": (0.35, 2.4, 0.5),
        "after_paragraph": (0.6, 2.5, 0.5), "punct": (0.05, 1.4, 0.4), "before_delete": (0.12, 1.8, 0.45),
        "in_delete": (0.01, 0.8, 0.3), "after_delete": (0.15, 2.0, 0.5), "other": (0.08, 1.5, 0.45),
    }
    pauses = {}
    rng = np.random.default_rng(0)
    for c, (p_long, med_ratio, sig) in pc.items():
        short = np.clip(med_ratio * np.exp(rng.normal(0, sig, 400)), 0.2, None)
        short = short[short * 0.2 < LONG_PAUSE]
        long_ = LONG_PAUSE * np.exp(np.abs(rng.normal(0, 0.8, 400)))
        pauses[c] = {"p_long": p_long, "short_ratio": short.round(3).tolist(), "long_sec": long_.round(2).tolist()}
    return {
        "pauses": pauses,
        "delete_rate_per_100": {"1-2": 2.5, "3-9": 1.2, "10-29": 0.25, "30-400": 0.05},
        "delete_sizes": {"3-9": [3, 4, 5, 6, 7, 8, 9], "10-29": [10, 12, 15, 18, 22, 27], "30-400": [30, 40, 60, 90]},
        "p_delete_after_long_pause": 0.12,
        "delete_after_long_sizes": [2, 3, 5, 8, 12, 20],
        "intext_rate_per_1000": 1.5,
        "intext_distance": [5, 10, 20, 40, 80, 150, 300],
        "intext_size": [1, 1, 2, 3, 5, 8],
        "intext_end_review_frac": 0.35,
        "writer_median_iki": [0.16, 0.19, 0.22, 0.26, 0.3],
        "source": "defaults",
    }


class CompositionStats:
    def __init__(self, d=None):
        self.d = d or _default_dict()
        self.pauses = {c: {"p_long": float(v["p_long"]),
                           "short_ratio": np.asarray(v["short_ratio"], float),
                           "long_sec": np.asarray(v["long_sec"], float)}
                       for c, v in self.d["pauses"].items()}

    # --- Fit / persist -------------------------------------------------------------

    @classmethod
    def fit(cls, essays, rng=None, verbose=True):
        """Collect composition statistics from dataset_klicke.Essay objects."""
        rng = rng or np.random.default_rng(0)
        short = {c: [] for c in PAUSE_CLASSES}
        long_ = {c: [] for c in PAUSE_CLASSES}
        n_cls = {c: 0 for c in PAUSE_CLASSES}
        del_runs = []                  # (size, pause_before)
        long_then = [0, 0]             # [long pauses, long pauses followed by deletion]
        del_after_long = []
        intext = []                    # (distance, size, frac_through_essay)
        edge_chars = 0
        writer_med = []

        for n_essay, e in enumerate(essays):
            prod = np.flatnonzero((e.kind == INPUT) | (e.kind == REMOVE))
            if len(prod) < 200:
                continue
            iki = np.diff(e.down[prod])
            classes = []
            for j in range(1, len(prod)):
                i, ip = prod[j], prod[j - 1]
                ch = e.char[i][:1] if e.kind[i] == INPUT else "\b"
                classes.append(pause_class(ch, e.prev[i], e.kind[ip], e.kind[i]))
            classes = np.array(classes)
            edge = e.at_edge[prod[1:]]
            inword = iki[(classes == "in_word") & edge & (iki > 0) & (iki < LONG_PAUSE)]
            if len(inword) < 50:
                continue
            med = float(np.median(inword))
            writer_med.append(med)

            ok = edge & (iki > 0) & (iki < 600)
            for c in PAUSE_CLASSES:
                m = ok & (classes == c)
                v = iki[m]
                n_cls[c] += len(v)
                short[c].append(v[v < LONG_PAUSE] / med)
                long_[c].append(v[v >= LONG_PAUSE])

            # Leading-edge deletion runs + in-text revision episodes
            kinds = e.kind[prod]
            t_total = e.down[prod[-1]] - e.down[prod[0]] + 1e-9
            j = 1
            last_edge_input = 0
            while j < len(prod):
                i = prod[j]
                if kinds[j] == INPUT and e.at_edge[i]:
                    edge_chars += len(e.char[i])
                    j += 1
                    continue
                if kinds[j] == REMOVE and e.at_edge[i]:
                    size, k = 0, j
                    while k < len(prod) and kinds[k] == REMOVE and e.at_edge[prod[k]]:
                        size += len(e.char[prod[k]])
                        k += 1
                    pause = iki[j - 1]
                    del_runs.append((size, pause))
                    j = k
                    continue
                # in-text edit episode: until typing resumes at the end of the text
                dist = int(e.text_len[i] - max(e.pos[i], 0))
                frac = (e.down[i] - e.down[prod[0]]) / t_total
                size, k = 0, j
                while k < len(prod) and not e.at_edge[prod[k]]:
                    size += len(e.char[prod[k]])
                    k += 1
                if 0 < dist and 0 < size <= MAX_FIX:
                    intext.append((dist, size, frac))
                j = k

            # Long pause at the leading edge -> is the next action a deletion?
            for j in range(1, len(prod) - 1):
                if iki[j - 1] >= LONG_PAUSE and edge[j - 1]:
                    long_then[0] += 1
                    if kinds[j] == REMOVE:
                        long_then[1] += 1
                        size, k = 0, j
                        while k < len(prod) and kinds[k] == REMOVE:
                            size += len(e.char[prod[k]])
                            k += 1
                        del_after_long.append(size)

            if verbose and (n_essay + 1) % 250 == 0:
                print(f"  composition: {n_essay + 1} essays")

        def sub(a, n=MAX_SAMPLES):
            a = np.concatenate(a) if isinstance(a, list) and a else np.asarray(a, float)
            if len(a) > n:
                a = rng.choice(a, n, replace=False)
            return a

        pauses = {}
        for c in PAUSE_CLASSES:
            s, l = sub(short[c]), sub(long_[c])
            total = sum(len(x) for x in short[c]) + sum(len(x) for x in long_[c])
            p_long = (sum(len(x) for x in long_[c]) / total) if total else 0.0
            pauses[c] = {"p_long": round(p_long, 4), "n": int(total),
                         "short_ratio": np.round(s, 3).tolist(), "long_sec": np.round(np.minimum(l, 600), 2).tolist()}

        sizes = np.array([s for s, _ in del_runs]) if del_runs else np.zeros(0)
        rate, size_samples = {}, {}
        for lo, hi in DELETE_BUCKETS:
            key = f"{lo}-{hi}"
            m = (sizes >= lo) & (sizes <= hi)
            rate[key] = round(100.0 * m.sum() / max(edge_chars, 1), 4)
            if lo >= 3:
                size_samples[key] = sub(sizes[m].astype(float), 2000).astype(int).tolist()
        it = np.array(intext) if intext else np.zeros((0, 3))
        d = {
            "pauses": pauses,
            "delete_rate_per_100": rate,
            "delete_sizes": size_samples,
            "p_delete_after_long_pause": round(long_then[1] / max(long_then[0], 1), 4),
            "delete_after_long_sizes": sub(np.array(del_after_long, float), 2000).astype(int).tolist(),
            "intext_rate_per_1000": round(1000.0 * len(it) / max(edge_chars, 1), 4),
            "intext_distance": sub(it[:, 0], 2000).astype(int).tolist(),
            "intext_size": sub(it[:, 1], 2000).astype(int).tolist(),
            "intext_end_review_frac": round(float((it[:, 2] > 0.9).mean()) if len(it) else 0.0, 4),
            "writer_median_iki": np.round(np.array(writer_med), 4).tolist(),
            "edge_chars": int(edge_chars),
            "n_essays": len(writer_med),
            "source": "klicke",
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

    # --- Sampling ------------------------------------------------------------------

    def sample_pause(self, cls_name, base_iki, rng, long_scale=1.0, max_pause=20.0):
        """
        Interval (s) before a keystroke of pause class `cls_name`, for a typist whose
        median in-word interval is `base_iki`. Returns (seconds, is_long).
        """
        p = self.pauses.get(cls_name) or self.pauses["other"]
        if len(p["long_sec"]) and rng.random() < p["p_long"]:
            v = rng.choice(p["long_sec"]) * np.exp(rng.normal(0, 0.1)) * long_scale
            return float(min(v, max_pause)), True
        r = rng.choice(p["short_ratio"]) if len(p["short_ratio"]) else 1.0
        return float(min(base_iki * r * np.exp(rng.normal(0, 0.05)), LONG_PAUSE)), False

    def sample_short(self, cls_name, base_iki, rng):
        """A non-thinking interval (s) for this pause class (always < LONG_PAUSE)."""
        p = self.pauses.get(cls_name) or self.pauses["other"]
        r = rng.choice(p["short_ratio"]) if len(p["short_ratio"]) else 1.0
        return float(min(base_iki * r * np.exp(rng.normal(0, 0.05)), 0.95 * LONG_PAUSE))

    def p_long(self, cls_name):
        return (self.pauses.get(cls_name) or self.pauses["other"])["p_long"]

    def delete_rate(self, bucket):
        """Leading-edge deletion runs of a size bucket per typed character."""
        return self.d["delete_rate_per_100"].get(bucket, 0.0) / 100.0

    def sample_delete_size(self, bucket, rng):
        s = self.d["delete_sizes"].get(bucket) or [5]
        return int(rng.choice(s))

    def summary(self):
        d = self.d
        med = np.median(d["writer_median_iki"]) * 1000 if d["writer_median_iki"] else float("nan")
        pl = ", ".join(f"{c} {self.pauses[c]['p_long']:.0%}" for c in
                       ["in_word", "between_words", "after_clause", "after_sentence"] if c in self.pauses)
        return (f"source={d['source']}, writer median in-word IKI {med:.0f} ms; long-pause rate: {pl}; "
                f"deletions/100ch {d['delete_rate_per_100']}; in-text revisions/1000ch {d['intext_rate_per_1000']}")
