"""
Keyboard geometry and character helpers (US QWERTY).

Used for:
  - typo generation (which keys are physically next to each other)
  - timing statistics (characters are reduced to a small "key class" vocabulary)
  - the player (which physical key + Shift produces a character)

Special characters used inside keystroke plans:
    BKSP = "\b"   Backspace
    WORD_BKSP     Ctrl+Backspace: delete the previous word (see word_delete_start)
    LEFT / RIGHT  arrow keys (in-text revisions)
"""

import numpy as np

BKSP = "\b"
WORD_BKSP = "\x7f"
LEFT = "\x1b[D"
RIGHT = "\x1b[C"
NAV_KEYS = {LEFT: "left", RIGHT: "right"}

ROWS = ["`1234567890-=", "qwertyuiop[]\\", "asdfghjkl;'", "zxcvbnm,./"]
ROW_OFFSETS = [0.0, 0.5, 0.75, 1.25]   # stagger of each row in key widths

SHIFTED = {
    "~": "`", "!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6", "&": "7",
    "*": "8", "(": "9", ")": "0", "_": "-", "+": "=", "{": "[", "}": "]", "|": "\\",
    ":": ";", '"': "'", "<": ",", ">": ".", "?": "/",
}

# Physical key positions (x, y) in key widths
_POS = {}
for r, (row, off) in enumerate(zip(ROWS, ROW_OFFSETS)):
    for c, ch in enumerate(row):
        _POS[ch] = (c + off, float(r))
_POS[" "] = (5.5, 4.0)

# Left hand keys (touch-typing convention); used for hand-alternation features
LEFT_HAND = set("`12345qwertasdfgzxcvb")


def base_key(ch):
    """Physical (unshifted) key for a character, or None if not on the US layout."""
    if ch in _POS:
        return ch
    if ch.isalpha() and ch.isascii():
        return ch.lower()
    return SHIFTED.get(ch)


def needs_shift(ch):
    return (ch.isascii() and ch.isalpha() and ch.isupper()) or ch in SHIFTED


def neighbours(ch, radius=1.25):
    """Keys physically adjacent to `ch`'s key (same case / shift state as ch)."""
    k = base_key(ch)
    if k is None or k not in _POS:
        return []
    x, y = _POS[k]
    out = []
    for other, (ox, oy) in _POS.items():
        if other == k or other == " ":
            continue
        if abs(oy - y) <= 1 and np.hypot(ox - x, oy - y) <= radius:
            out.append(other)
    if needs_shift(ch):
        inv = {v: s for s, v in SHIFTED.items()}
        out = [o.upper() if o.isalpha() else inv.get(o, o) for o in out]
    return out


def hand(ch):
    k = base_key(ch)
    if k is None:
        return -1
    if k == " ":
        return 2
    return 0 if k in LEFT_HAND else 1


# --- Key-class vocabulary for timing models ---------------------------------------
# Timing depends on the physical key, not on case, so 'A' and 'a' share a class.
# Shifted symbols map to their base key; everything else is "other".

VOCAB = ["<pad>", "<other>", " ", BKSP, "\n", "<nav>"] + list("abcdefghijklmnopqrstuvwxyz0123456789") \
    + list("`-=[]\\;',./")
VOCAB_INDEX = {c: i for i, c in enumerate(VOCAB)}
PAD, OTHER = 0, 1


def word_delete_start(buf, cur):
    """
    Where Ctrl+Backspace at caret `cur` deletes back to: one space right before the
    caret (if any), then the letters/digits of the word before it - the behaviour of
    Word, Notepad and browser text fields for plain words. Plans only use it where
    that is exact (runs of whole words separated by single spaces).
    """
    i = cur
    if i > 0 and buf[i - 1] == " ":
        i -= 1
    while i > 0 and buf[i - 1].isalnum():
        i -= 1
    return i


def key_class(ch):
    """Map a character (or special key) to an index in VOCAB."""
    if ch is None:
        return PAD
    if ch == WORD_BKSP:               # timed like Backspace
        return VOCAB_INDEX[BKSP]
    if ch in NAV_KEYS:
        return VOCAB_INDEX["<nav>"]
    if ch in VOCAB_INDEX:
        return VOCAB_INDEX[ch]
    k = base_key(ch)
    if k is not None and k in VOCAB_INDEX:
        return VOCAB_INDEX[k]
    return OTHER


def timing_token(ch):
    """Short string label of a key class, used as dict keys in JSON stats."""
    return VOCAB[key_class(ch)]
