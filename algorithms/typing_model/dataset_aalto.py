"""
Loader for the Aalto "136M Keystrokes" dataset (Dhakal et al., CHI 2018).
168k participants each transcribed 15 sentences on a physical keyboard.

Pass the downloaded Keystrokes.zip (1.4 GB) - files are read straight from the
archive, nothing is extracted.

Keystrokes/files/<id>_keystrokes.txt (tab separated):
    PARTICIPANT_ID, TEST_SECTION_ID, SENTENCE, USER_INPUT, KEYSTROKE_ID,
    PRESS_TIME, RELEASE_TIME (ms), LETTER, KEYCODE
LETTER is the typed character, or a key name: BKSP (also "\\x08"), SHIFT,
CAPS_LOCK, ARW_LEFT, CTRL, ... Some browsers reported an empty LETTER; those
trials are skipped.

Keystrokes/files/metadata_participants.txt: LAYOUT, KEYBOARD_TYPE, AVG_WPM_15,
ERROR_RATE, FINGERS, ...

License: free for non-commercial use with attribution (see the readme in the zip).
"""

import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .keys import BKSP

PREFIX = "Keystrokes/files/"
META = PREFIX + "metadata_participants.txt"

# Keys that move the cursor or delete words: a trial containing them cannot be
# replayed as a simple append/backspace buffer, so it is skipped.
_UNREPLAYABLE = {"ARW_LEFT", "ARW_RIGHT", "ARW_UP", "ARW_DOWN", "ARw_DOWN", "HOME", "END",
                 "DELETE", "CTRL", "PG_UP", "PG_DOWN", "INSERT", "ALT", "WIN"}


@dataclass
class Trial:
    participant: int
    sentence: str
    user_input: str
    keys: list                      # char, BKSP, or None (non-text key such as SHIFT)
    press: np.ndarray               # seconds, relative to the trial's first key
    release: np.ndarray
    names: list = field(default_factory=list)   # raw LETTER values (SHIFT kept for shift timing)


def load_metadata(zip_path):
    with zipfile.ZipFile(zip_path) as z, z.open(META) as f:
        return pd.read_csv(f, sep="\t", encoding_errors="replace", keep_default_na=False)


def select_participants(meta, layouts=("qwerty",), keyboards=("full", "laptop"), wpm_range=(10, 200)):
    m = meta
    wpm = pd.to_numeric(m["AVG_WPM_15"], errors="coerce")
    ok = m["LAYOUT"].isin(layouts) & m["KEYBOARD_TYPE"].isin(keyboards) & wpm.between(*wpm_range)
    return m.loc[ok, "PARTICIPANT_ID"].astype(int).to_numpy(), dict(zip(m["PARTICIPANT_ID"].astype(int), wpm))


def is_validation(pid, every=20):
    """Deterministic participant-level split: 1 in `every` participants is held out."""
    return pid % every == 0


def _key(letter):
    if letter in ("BKSP", "\x08"):
        return BKSP
    if len(letter) == 1:
        return letter
    return None


def parse_file(text, pid):
    """Parse one <id>_keystrokes.txt into a list of Trial objects."""
    rows = {}
    order = []
    lines = text.splitlines()
    for line in lines[1:]:
        f = line.split("\t")
        if len(f) != 9:
            continue
        sec = f[1]
        if sec not in rows:
            rows[sec] = []
            order.append(sec)
        try:
            rows[sec].append((int(f[5]), int(f[6]), f[7], f[2], f[3]))
        except ValueError:
            continue
    trials = []
    for sec in order:
        r = sorted(rows[sec], key=lambda x: x[0])
        names = [x[2] for x in r]
        if not r or any(n == "" for n in names) or any(n in _UNREPLAYABLE for n in names):
            continue
        t0 = r[0][0]
        press = np.array([x[0] - t0 for x in r], float) / 1000.0
        release = np.array([x[1] - t0 for x in r], float) / 1000.0
        trials.append(Trial(pid, r[0][3], r[0][4], [_key(n) for n in names], press, release, names))
    return trials


def iter_participants(zip_path, split="train", max_participants=None, seed=0, meta=None, verbose=True):
    """
    Yield (wpm, [Trial, ...]) per participant.
    split: "train" | "val" | "all"  (participant-level hold-out, see is_validation)
    """
    meta = meta if meta is not None else load_metadata(zip_path)
    pids, wpm_of = select_participants(meta)
    if split == "train":
        pids = pids[~np.array([is_validation(p) for p in pids])]
    elif split == "val":
        pids = pids[np.array([is_validation(p) for p in pids])]
    rng = np.random.default_rng(seed)
    pids = rng.permutation(pids)
    if max_participants is not None:
        pids = pids[:max_participants]
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for i, pid in enumerate(pids):
            name = f"{PREFIX}{pid}_keystrokes.txt"
            if name not in names:
                continue
            trials = parse_file(z.read(name).decode("utf-8", "replace"), int(pid))
            if trials:
                yield float(wpm_of[int(pid)]), trials
            if verbose and (i + 1) % 1000 == 0:
                print(f"  aalto: read {i + 1}/{len(pids)} participants")
