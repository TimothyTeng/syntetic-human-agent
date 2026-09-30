"""
Load raw mouse-event logs from public datasets into one common format.

Supported:
  * BMDD  - Bogazici Mouse Dynamics Dataset (native Windows logger, CSV)
            https://data.mendeley.com/datasets/w6cxr8yc7p/2
  * Balabit Mouse Dynamics Challenge (RDP capture, headerless-extension files
            named session_XXXX with a CSV header)
            https://github.com/balabit/Mouse-Dynamics-Challenge

`root` can be an extracted folder OR the BMDD split archive itself
(e.g. boun-mouse-dynamics-dataset.zip with .z01-.z09 beside it) - files are
then streamed straight from the archive via splitzip.py, no extraction needed.

Every file becomes a `Session` holding parallel numpy arrays:
    t (seconds, float), x, y (pixels), button (code), state (code), window (str or None)

The datasets are large (BMDD is ~2,500 hours), so sessions are *yielded*
one at a time by iter_sessions() instead of being loaded all at once.
"""

import glob
import io
import os
import random
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .splitzip import SplitZip

# --- Event codes -------------------------------------------------------------
MOVE, PRESS, RELEASE, DRAG, SCROLL, UNKNOWN = 0, 1, 2, 3, 4, -1
NO_BUTTON, LEFT, RIGHT, OTHER_BUTTON = 0, 1, 2, 3


@dataclass
class Session:
    """One recording file converted to numpy arrays (all the same length)."""
    user: str
    path: str
    t: np.ndarray        # timestamps in seconds (sorted ascending)
    x: np.ndarray        # cursor x in pixels
    y: np.ndarray        # cursor y in pixels
    button: np.ndarray   # NO_BUTTON / LEFT / RIGHT / OTHER_BUTTON
    state: np.ndarray    # MOVE / PRESS / RELEASE / DRAG / SCROLL / UNKNOWN
    window: np.ndarray | None = None  # foreground window name per event (BMDD only)

    def __len__(self):
        return len(self.t)


# --- Value mapping helpers -----------------------------------------------------

def _map_state(values):
    """Map text states ('Move', 'Pressed', 'Released', 'Drag', 'Scroll') to codes."""
    s = pd.Series(values).astype(str).str.strip().str.lower()
    out = np.full(len(s), UNKNOWN, dtype=np.int8)
    out[s.str.startswith("move").to_numpy()] = MOVE
    out[s.str.startswith(("press", "down")).to_numpy()] = PRESS
    out[s.str.startswith(("release", "up")).to_numpy()] = RELEASE
    out[s.str.startswith("drag").to_numpy()] = DRAG
    out[s.str.startswith(("scroll", "wheel")).to_numpy()] = SCROLL
    return out


def _map_button(values):
    """Map text buttons ('Left', 'Right', 'NoButton', 'None') to codes."""
    s = pd.Series(values).astype(str).str.strip().str.lower()
    out = np.full(len(s), OTHER_BUTTON, dtype=np.int8)
    out[s.str.startswith("left").to_numpy()] = LEFT
    out[s.str.startswith("right").to_numpy()] = RIGHT
    out[s.isin(["nobutton", "none", "", "nan", "no button"]).to_numpy()] = NO_BUTTON
    return out


def _norm(name):
    """Normalise a column name: 'Client Timestamp' -> 'clienttimestamp'."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _find_col(columns, exact=(), contains=()):
    """Return the first column whose normalised name matches, else None."""
    normed = {_norm(c): c for c in columns}
    for cand in exact:
        if cand in normed:
            return normed[cand]
    for cand in contains:
        for key, col in normed.items():
            if cand in key:
                return col
    return None


def _build_session(df, path, user, tcol, xcol, ycol, bcol, scol, wcol=None, acol=None):
    """Convert a raw DataFrame into a cleaned, time-sorted Session."""
    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=float)
    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
    button = _map_button(df[bcol]) if bcol else np.zeros(len(df), dtype=np.int8)
    if scol:
        state = _map_state(df[scol])
    elif acol:
        state = _map_state(df[acol])  # some logs only have an "action type" column
    else:
        state = np.zeros(len(df), dtype=np.int8)
    window = df[wcol].astype(str).to_numpy() if wcol else None

    # Drop broken rows (NaNs, off-screen sentinel values like 65535)
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (x < 20000) & (y < 20000)
    order = np.argsort(t[ok], kind="stable")
    sel = lambda a: a[ok][order] if a is not None else None  # noqa: E731
    return Session(user=user, path=path, t=sel(t), x=sel(x), y=sel(y),
                   button=sel(button), state=sel(state), window=sel(window))


# --- Balabit -------------------------------------------------------------------

def _balabit_files(root, split="training"):
    files = sorted(glob.glob(os.path.join(root, "**", "session_*"), recursive=True))
    if split:
        files = [f for f in files if split in f.replace("\\", "/").lower()] or files
    return files


def _load_balabit_file(path):
    """Balabit columns: record timestamp, client timestamp, button, state, x, y."""
    df = pd.read_csv(path)
    user = os.path.basename(os.path.dirname(path))
    cols = df.columns
    return _build_session(
        df, path, user,
        tcol=_find_col(cols, exact=("clienttimestamp",), contains=("timestamp",)),
        xcol=_find_col(cols, exact=("x",)),
        ycol=_find_col(cols, exact=("y",)),
        bcol=_find_col(cols, exact=("button",)),
        scol=_find_col(cols, exact=("state",)),
    )


# --- BMDD ----------------------------------------------------------------------

_archives = {}


def _is_archive(root):
    return str(root).lower().endswith(".zip") and os.path.isfile(root)


def _archive(root):
    """Open (once) and cache a SplitZip for the given final .zip part."""
    if root not in _archives:
        _archives[root] = SplitZip(root)
    return _archives[root]


def _bmdd_files(root, split="train"):
    """
    Folder: all *.csv below it. Archive: all *.csv members.
    BMDD layout: users/userN/{training, internal_tests, external_tests}/session_*.csv
    """
    if _is_archive(root):
        files = sorted(n for n in _archive(root).names() if n.lower().endswith(".csv") and "/users/" in n)
    else:
        files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if split:
        files = [f for f in files if split in f.replace("\\", "/").lower()] or files
    return files


def _user_from_path(path):
    """User id = first path component like 'user12', else the parent folder name."""
    parts = path.replace("\\", "/").split("/")
    for part in parts[:-1]:
        if re.fullmatch(r"user\d+", part.lower()):
            return part
    return parts[-2] if len(parts) > 1 else "unknown"


def _load_bmdd_file(path, root):
    """
    BMDD columns: client_timestamp, x, y, button (NaN / Left / Right),
    state (Move / Drag / Pressed / Released), window (anonymised category:
    'browsing', 'development', 'file system', ...). Columns are detected by
    name so other exports of the dataset also work.
    """
    source = io.BytesIO(_archive(root).read(path)) if _is_archive(root) else path
    df = pd.read_csv(source, low_memory=False)
    cols = df.columns
    tcol = _find_col(cols, exact=("timestamp", "clienttimestamp", "time"), contains=("time",))
    xcol = _find_col(cols, exact=("x", "xcoordinate", "xpos", "posx"), contains=("x",))
    ycol = _find_col(cols, exact=("y", "ycoordinate", "ypos", "posy"), contains=("y",))
    if not (tcol and xcol and ycol):
        raise ValueError(f"Can't detect time/x/y columns in {path}: {list(cols)}")
    return _build_session(
        df, path, _user_from_path(path), tcol, xcol, ycol,
        bcol=_find_col(cols, contains=("button",)),
        scol=_find_col(cols, contains=("state",)),
        wcol=_find_col(cols, contains=("window",)),
        acol=_find_col(cols, contains=("action", "type")),
    )


# --- Public API ------------------------------------------------------------------

def detect_format(root):
    """
    'balabit' if the folder holds extension-less session_* files (Balabit's
    format), otherwise 'bmdd' (whose session files end in .csv). Archives = BMDD.
    """
    if _is_archive(root):
        return "bmdd"
    for f in glob.iglob(os.path.join(root, "**", "session_*"), recursive=True):
        if not os.path.splitext(f)[1]:
            return "balabit"
    return "bmdd"


def list_files(dataset, root, split=None):
    """List the raw files of a dataset ('bmdd' / 'balabit' / 'auto')."""
    if dataset == "auto":
        dataset = detect_format(root)
    if dataset == "balabit":
        return _balabit_files(root, split if split is not None else "training")
    return _bmdd_files(root, split if split is not None else "train")


def iter_sessions(dataset, root, split=None, max_files=None, users=None, shuffle=True, seed=0):
    """
    Yield Session objects one file at a time.

    dataset:   'bmdd', 'balabit' or 'auto' (detect from folder contents).
    root:      folder where the dataset was extracted, or the BMDD .zip archive.
    split:     only use files whose path contains this text ('train', 'training'...).
               Pass '' to use every file.
    max_files: stop after this many files (the full BMDD is very large).
    users:     optional list of user ids to keep.
    shuffle:   visit files in random order so a partial load covers many users.
    """
    if dataset == "auto":
        dataset = detect_format(root)
    files = list_files(dataset, root, split)
    if not files:
        raise FileNotFoundError(f"No {dataset} files found under {root}")
    if users:  # filter by path first so unwanted files are never read
        files = [f for f in files if _user_from_path(f) in users]
    if shuffle:
        random.Random(seed).shuffle(files)

    count = 0
    for path in files:
        try:
            sess = _load_balabit_file(path) if dataset == "balabit" else _load_bmdd_file(path, root)
        except Exception as exc:  # skip unreadable files but tell the user
            print(f"[dataset] skipping {path}: {exc}")
            continue
        if users and sess.user not in users:
            continue
        if len(sess) == 0:
            continue
        yield sess
        count += 1
        if max_files and count >= max_files:
            break


def peek(dataset, root, n_rows=5):
    """DEBUG: print the header and first rows of the first file (check column mapping)."""
    files = list_files(dataset if dataset != "auto" else detect_format(root), root, split="")
    if not files:
        print("no files found")
        return
    print(files[0])
    source = io.BytesIO(_archive(root).read(files[0])) if _is_archive(root) else files[0]
    print(pd.read_csv(source, nrows=n_rows).to_string())
