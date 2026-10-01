"""
Empirical scroll-wheel timing learned from the Balabit Mouse Dynamics Challenge.

  burst  - wheel notches scrolled in one go (consecutive scroll events < BURST_GAP apart)
  gap    - seconds between notches inside a burst
  pause  - seconds between one burst and the next (reading, looking around)

People scroll very differently: in Balabit, 3 of 10 users send notches 15-50 ms apart
(free-spinning wheels or touchpads), the rest 125-235 ms apart (a notched wheel turned
by a finger). Pooling all events would let the heavy scrollers dominate, so values are
stored PER USER. A simulated person adopts one user's style for the whole run
(for_profile), which keeps the person consistent and the population varied.

Like click_stats, real values are stored (subsampled) and sampled with a little
jitter, so the real distribution shapes are kept without assuming a formula.
Balabit's client timestamps tick in ~16 ms steps (several notches can share one
timestamp), so sampled gaps are spread over that step.

The share of bursts scrolling UP is stored too (`p_up`), but Balabit was recorded
during remote-desktop admin work, where scrolling back up is far more common than
when reading a web page - reading.py keeps its own web-reading value by default.

    python -m algorithms.mouse_model.scroll_stats --data data/balabit           # fit -> models/scroll_stats.json
    python -m algorithms.mouse_model.scroll_stats --data data/balabit --check   # + compare with held-out sessions
"""

import argparse
import json
import os

import numpy as np

MAX_PER_USER = 1500        # stored values per user and kind
BURST_GAP = 0.5            # s; scroll events closer than this belong to one burst
TICK = 0.016               # Balabit client-timestamp resolution (s), used to spread sampled gaps
CLOCK = 0.015625           # ... exact Windows timer tick, for comparing gaps on the dataset's own clock
BURST_RANGE = (1, 25)      # notches
PAUSE_RANGE = (0.2, 60.0)  # longer gaps are breaks, not reading pauses
MIN_BURSTS = 50            # users with fewer scroll bursts are left out

# Fallbacks when no fitted file exists (the earlier hand-set reading defaults)
DEFAULT_BURST = (1, 5)       # uniform notches
DEFAULT_GAP = (0.07, 0.4)    # log-normal median, sigma
DEFAULT_PAUSE = (1.8, 0.6)


def _user_of(path):
    return os.path.basename(os.path.dirname(path))


def extract(files):
    """{user: {'bursts': [...], 'gaps': [...], 'pauses': [...], 'ups': n_up}} from Balabit session files."""
    import pandas as pd
    out = {}
    for path in files:
        df = pd.read_csv(path)
        sc = df[df["button"] == "Scroll"]
        if len(sc) < 2:
            continue
        u = out.setdefault(_user_of(path), {"bursts": [], "gaps": [], "pauses": [], "ups": 0})
        t = sc["client timestamp"].to_numpy(float)
        d = sc["state"].to_numpy()
        start = 0
        for i in range(1, len(t) + 1):
            if i == len(t) or t[i] - t[i - 1] >= BURST_GAP or d[i] != d[start]:
                u["bursts"].append(int(np.clip(i - start, *BURST_RANGE)))
                u["ups"] += d[start] == "Up"
                u["gaps"].extend(np.diff(t[start:i]).tolist())
                if i < len(t) and PAUSE_RANGE[0] <= t[i] - t[i - 1] <= PAUSE_RANGE[1]:
                    u["pauses"].append(float(t[i] - t[i - 1]))
                start = i
    return out


class ScrollStats:
    def __init__(self, users=None, p_up=None, source="defaults", profile=None):
        """users: {name: {'bursts', 'gaps', 'pauses'}}; profile: the user whose style to sample
        (None = a random user per sample, i.e. the population)."""
        self.users = {k: {kind: np.asarray(v[kind], float) for kind in ("bursts", "gaps", "pauses")}
                      for k, v in (users or {}).items()}
        self.p_up = p_up
        self.source = source
        self.profile = profile if profile in self.users else None

    # --- Build / persist ------------------------------------------------------------

    @classmethod
    def fit(cls, files, rng=None):
        rng = rng or np.random.default_rng(0)
        raw = {u: v for u, v in extract(files).items() if len(v["bursts"]) >= MIN_BURSTS}
        users, ups, total = {}, 0, 0
        for name, v in raw.items():
            sub = {}
            for kind in ("bursts", "gaps", "pauses"):
                a = np.asarray(v[kind], float)
                sub[kind] = rng.choice(a, MAX_PER_USER, replace=False) if len(a) > MAX_PER_USER else a
            users[name] = sub
            ups += v["ups"]
            total += len(v["bursts"])
        return cls(users, ups / total if total else None,
                   source=f"balabit ({len(files)} sessions, {len(users)} users, {total} bursts)")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"source": self.source, "p_up": self.p_up,
                       "users": {u: {k: np.round(a, 4).tolist() for k, a in v.items()}
                                 for u, v in self.users.items()}}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d.get("users"), d.get("p_up"), d.get("source", path))

    @classmethod
    def default(cls):
        """No data: sampling falls back to the hand-set defaults."""
        return cls()

    def profiles(self):
        return sorted(self.users)

    def for_profile(self, profile):
        """The same statistics, sampled only from one user's style."""
        s = ScrollStats.__new__(ScrollStats)
        s.users, s.p_up, s.source = self.users, self.p_up, self.source
        s.profile = profile if profile in self.users else None
        return s

    # --- Sampling ---------------------------------------------------------------------

    def _values(self, kind, rng):
        if not self.users:
            return None
        name = self.profile or self.profiles()[int(rng.integers(len(self.users)))]
        a = self.users[name][kind]
        return a if len(a) else None

    def sample_burst(self, rng):
        """Notches in one scroll burst."""
        a = self._values("bursts", rng)
        if a is None:
            return int(rng.integers(DEFAULT_BURST[0], DEFAULT_BURST[1] + 1))
        return int(rng.choice(a))

    def sample_gap(self, rng):
        """Seconds between two notches of a burst (spread over the dataset's timestamp tick)."""
        a = self._values("gaps", rng)
        if a is None:
            return float(DEFAULT_GAP[0] * np.exp(rng.normal(0, DEFAULT_GAP[1])))
        v = float(rng.choice(a)) + rng.uniform(0, TICK)
        return float(np.clip(v * np.exp(rng.normal(0, 0.07)), 0.003, BURST_GAP))

    def sample_pause(self, rng, max_seconds=PAUSE_RANGE[1]):
        """Seconds between scroll bursts while reading (at most max_seconds)."""
        a = self._values("pauses", rng)
        if a is None:
            return float(min(DEFAULT_PAUSE[0] * np.exp(rng.normal(0, DEFAULT_PAUSE[1])), max_seconds))
        a = a[a <= max_seconds]
        if not len(a):
            return float(max_seconds)
        return float(rng.choice(a) * np.exp(rng.normal(0, 0.07)))

    def summary(self):
        if not self.users:
            return "scroll timing: defaults"
        who = f"style of {self.profile}" if self.profile else "all users"
        med = lambda kind: np.median(np.concatenate([self.users[u][kind] for u in  # noqa: E731
                                                     ([self.profile] if self.profile else self.users)]))
        return (f"scroll timing ({who}): notches/burst median {med('bursts'):.0f}, "
                f"gap median {med('gaps') * 1000:.0f} ms, pause median {med('pauses'):.1f} s")


DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "models", "scroll_stats.json")
_CACHE = {}


def load_default():
    """models/scroll_stats.json if it exists, else the hand-set defaults (cached)."""
    if "stats" not in _CACHE:
        _CACHE["stats"] = ScrollStats.load(DEFAULT_PATH) if os.path.exists(DEFAULT_PATH) else ScrollStats.default()
    return _CACHE["stats"]


def main(argv=None):
    from scipy.stats import ks_2samp

    from .dataset import list_files

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/balabit", help="extracted Balabit dataset folder")
    ap.add_argument("--out", default=DEFAULT_PATH)
    ap.add_argument("--check", action="store_true", help="compare each user's style with their held-out sessions")
    args = ap.parse_args(argv)

    stats = ScrollStats.fit(list_files("balabit", args.data, split="training"))
    stats.save(args.out)
    print(f"Saved {args.out}: {stats.source}; real share of upward bursts {stats.p_up:.2f}")
    for u in stats.profiles():
        print("  " + stats.for_profile(u).summary())

    if args.check:
        held = extract(list_files("balabit", args.data, split="test_files"))
        rng = np.random.default_rng(1)
        print("\nHeld-out sessions (test_files), per user - KS distance real vs sampled (0 = identical):")
        print(f"  {'user':<8} {'bursts':>7} {'notches':>8} {'gap':>6} {'pause':>6}")
        ks_all = []
        for u in stats.profiles():
            h = held.get(u)
            if not h or len(h["bursts"]) < 30:
                continue
            s = stats.for_profile(u)
            row = []
            for kind, fn in (("bursts", s.sample_burst), ("gaps", s.sample_gap), ("pauses", s.sample_pause)):
                real = np.asarray(h[kind], float)
                gen = np.array([fn(rng) for _ in range(len(real))])
                if kind == "gaps":   # compare on the dataset's own ~15.6 ms clock
                    real, gen = np.round(real / CLOCK), np.floor(gen / CLOCK)
                row.append(ks_2samp(real, gen).statistic)
            ks_all.append(row)
            print(f"  {u:<8} {len(h['bursts']):>7} {row[0]:>8.3f} {row[1]:>6.3f} {row[2]:>6.3f}")
        if ks_all:
            m = np.mean(ks_all, axis=0)
            print(f"  {'mean':<8} {'':>7} {m[0]:>8.3f} {m[1]:>6.3f} {m[2]:>6.3f}")


if __name__ == "__main__":
    main()
