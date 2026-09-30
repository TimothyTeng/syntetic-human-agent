"""
Loader for the KLiCKe keystroke logs (Kaggle "Linking Writing Processes to Writing
Quality"): ~2.5k argumentative essays composed freely in 30 minutes, 8.4M events.

Pass either the competition .zip (read in place, nothing extracted) or train_logs.csv.

train_logs.csv columns:
    id, event_id, down_time, up_time, action_time (ms), activity, down_event,
    up_event, text_change, cursor_position, word_count
activity: Input | Remove/Cut | Nonproduction | Replace | Paste | "Move From [a, b] To [c, d]"
Letters and digits are masked as 'q'; spaces, punctuation and newlines are kept,
so word / clause / sentence boundaries can still be recovered.

Each essay is replayed into a (masked) text buffer so every production event can
be labelled with where it happened: at the leading edge (end of text) or inside
earlier text, and what character precedes the insertion point.
"""

import re
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Event kinds
INPUT, REMOVE, NAV, OTHER = 0, 1, 2, 3

_NAV_EVENTS = {"Leftclick", "Rightclick", "Middleclick", "ArrowLeft", "ArrowRight", "ArrowUp",
               "ArrowDown", "Home", "End", "PageUp", "PageDown"}
_MOVE_RE = re.compile(r"Move From \[(\d+), (\d+)\] To \[(\d+), (\d+)\]")

COLUMNS = ["id", "event_id", "down_time", "up_time", "activity", "down_event",
           "text_change", "cursor_position"]


@dataclass
class Essay:
    id: str
    kind: np.ndarray        # (n,) INPUT / REMOVE / NAV / OTHER
    down: np.ndarray        # (n,) key-down time, seconds from essay start
    up: np.ndarray          # (n,) key-up time, seconds
    char: list              # inserted text for INPUT, removed text for REMOVE, key name otherwise
    at_edge: np.ndarray     # (n,) bool - the edit touched the end of the text
    prev: list              # text just before the edit point (last 3 chars), "" at start
    text_len: np.ndarray    # (n,) text length *before* the event
    pos: np.ndarray         # (n,) edit position (insertion / removal start), -1 if none
    final_text: str


def load_logs(path, usecols=COLUMNS):
    """Read train_logs.csv from the Kaggle .zip or a plain csv path."""
    if str(path).lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("train_logs.csv"))
            with z.open(name) as f:
                return pd.read_csv(f, usecols=usecols, keep_default_na=False)
    return pd.read_csv(path, usecols=usecols, keep_default_na=False)


def iter_essays(path_or_df, max_essays=None, seed=0):
    """Yield Essay objects (optionally a random subset of `max_essays`)."""
    df = path_or_df if isinstance(path_or_df, pd.DataFrame) else load_logs(path_or_df)
    df = df.sort_values(["id", "event_id"], kind="stable")
    ids = df["id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    order = np.arange(len(starts))
    if max_essays is not None and max_essays < len(order):
        order = np.sort(np.random.default_rng(seed).choice(order, max_essays, replace=False))

    cols = {c: df[c].to_numpy() for c in COLUMNS}
    for k in order:
        s, e = starts[k], ends[k]
        yield _replay({c: v[s:e] for c, v in cols.items()})


def _replay(ev):
    n = len(ev["id"])
    kind = np.full(n, OTHER, np.int8)
    at_edge = np.zeros(n, bool)
    text_len = np.zeros(n, np.int32)
    epos = np.full(n, -1, np.int32)
    chars, prevs = [], []
    text = []                      # list of characters (masked)

    for i in range(n):
        act = ev["activity"][i]
        tc = ev["text_change"][i]
        cur = int(ev["cursor_position"][i])
        L = len(text)
        text_len[i] = L
        prev = ""

        if act == "Input" or act == "Paste":
            pos = max(0, min(cur - len(tc), L))
            prev = "".join(text[max(0, pos - 3):pos])
            text[pos:pos] = list(tc)
            epos[i] = pos
            kind[i] = INPUT if act == "Input" else OTHER   # pastes are not typing
            at_edge[i] = pos == L
        elif act == "Remove/Cut":
            pos = max(0, min(cur, L))
            prev = "".join(text[max(0, pos - 3):pos])
            del text[pos:pos + len(tc)]
            epos[i] = pos
            kind[i] = REMOVE
            at_edge[i] = pos + len(tc) >= L
        elif act == "Replace":
            old, _, new = tc.partition(" => ")
            pos = max(0, min(cur - len(new), L))
            prev = "".join(text[max(0, pos - 3):pos])
            text[pos:pos + len(old)] = list(new)
            epos[i] = pos
            kind[i] = REMOVE
            at_edge[i] = pos + len(old) >= L
        elif act.startswith("Move"):
            m = _MOVE_RE.match(act)
            if m:
                a, b, c, _ = map(int, m.groups())
                seg = text[a:b]
                del text[a:b]
                text[c:c] = seg
        elif ev["down_event"][i] in _NAV_EVENTS:
            kind[i] = NAV
            tc = ev["down_event"][i]
        chars.append(tc)
        prevs.append(prev)

    return Essay(
        id=str(ev["id"][0]),
        kind=kind,
        down=ev["down_time"].astype(np.float64) / 1000.0,
        up=ev["up_time"].astype(np.float64) / 1000.0,
        char=chars,
        at_edge=at_edge,
        prev=prevs,
        text_len=text_len,
        pos=epos,
        final_text="".join(text),
    )
