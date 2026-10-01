"""
Load the fitted typing model pieces from models/ (numpy only), with fallbacks:

    motor timing   typing_mdn.npz (LSTM-MDN)  ->  typing_bigrams.json  ->  built-in defaults
    typos          typing_errors.json         ->  defaults
    composition    typing_composition.json    ->  defaults
"""

import os

from .composition import CompositionStats
from .errors import ErrorStats
from .keys import BKSP, LEFT, RIGHT, WORD_BKSP, word_delete_start
from .timing import BigramTimer

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")


class TypingModel:
    def __init__(self, timer, errors, comp, sources=None):
        self.timer, self.errors, self.comp = timer, errors, comp
        self.sources = sources or {}

    @classmethod
    def load(cls, models_dir=DEFAULT_DIR, use_mdn=True):
        p = lambda name: os.path.join(models_dir, name)  # noqa: E731
        sources = {}
        timer = None
        if use_mdn and os.path.exists(p("typing_mdn.npz")):
            try:
                from .sampler import TypingMDN
                timer = TypingMDN.load(p("typing_mdn.npz"))
                sources["timing"] = "typing_mdn.npz"
            except Exception as exc:  # noqa: BLE001 - fall back to the bigram model
                sources["timing_error"] = f"{type(exc).__name__}: {exc}"
        if timer is None and os.path.exists(p("typing_bigrams.json")):
            timer = BigramTimer.load(p("typing_bigrams.json"))
            sources["timing"] = "typing_bigrams.json"
        if timer is None:
            timer = BigramTimer.default()
            sources["timing"] = "defaults"
        errors = ErrorStats.load(p("typing_errors.json")) if os.path.exists(p("typing_errors.json")) \
            else ErrorStats.default()
        comp = CompositionStats.load(p("typing_composition.json")) if os.path.exists(p("typing_composition.json")) \
            else CompositionStats.default()
        sources["errors"] = errors.d.get("source")
        sources["composition"] = comp.d.get("source")
        return cls(timer, errors, comp, sources)


def apply_plan(keystrokes, initial=""):
    """Replay a plan into a virtual text field (cursor starts at the end). Returns the text."""
    buf = list(initial)
    cur = len(buf)
    for ks in keystrokes:
        k = ks.key
        if k == BKSP:
            if cur > 0:
                del buf[cur - 1]
                cur -= 1
        elif k == WORD_BKSP:
            start = word_delete_start(buf, cur)
            del buf[start:cur]
            cur = start
        elif k == LEFT:
            cur = max(0, cur - 1)
        elif k == RIGHT:
            cur = min(len(buf), cur + 1)
        else:
            buf.insert(cur, k)
            cur += 1
    return "".join(buf)


def describe(keystrokes, width=100):
    """Human-readable trace of a plan: typed text with markers, for debugging."""
    out = []
    for ks in keystrokes:
        if ks.long:
            out.append(f"[{ks.iki:.1f}s]")
        if ks.key == BKSP:
            out.append("~")
        elif ks.key == WORD_BKSP:
            out.append("^")
        elif ks.key == LEFT:
            out.append("<")
        elif ks.key == RIGHT:
            out.append(">")
        elif ks.key == "\n":
            out.append("\n")
        else:
            out.append(ks.key)
    s = "".join(out)
    return "\n".join(s[i:i + width] for i in range(0, len(s), width))
