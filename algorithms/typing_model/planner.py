"""
Text -> keystroke plan.

Pass 1 (the "writer") decides WHAT keys are pressed. It types the target text at
the leading edge and injects human behaviour, keeping a virtual text buffer so
the final text is always exactly the target (unless allow_uncorrected):

    typo           a motor error (adjacent key, omission, doubled key, transposition,
                   case slip), noticed after 0..n more keys, then backspaced
    false start    in compose mode, type a different wording (from an
                   alternatives provider), pause, delete back to where it diverged
    lost thought   a long pause at a boundary, then delete the last few characters
                   and retype them
    in-text fix    in compose mode, leave a wrong / missing word or an unnoticed
                   typo, keep writing, later arrow back, fix it and arrow forward

Pass 2 assigns WHEN: each key gets (interval since the previous press, hold time)
from the motor timer (Aalto bigrams / LSTM-MDN); in compose mode the interval at
word / clause / sentence boundaries is replaced by a composition pause (KLiCKe).

Modes:
    transcribe  short inputs (search boxes, fields): motor rhythm + typos only
    compose     longer text (emails, documents): everything
    None        auto: compose if the text has >= 12 words
"""

from dataclasses import dataclass, field

import numpy as np

from .alternatives import SYNONYMS
from .composition import LONG_PAUSE, pause_class
from .dataset_klicke import INPUT, REMOVE
from .errors import make_typo
from .keys import BKSP, LEFT, NAV_KEYS, RIGHT, base_key

MAX_HOLD = 0.35          # stay below the OS key auto-repeat delay (~0.5 s)
MIN_IKI = 0.012          # fastest press-to-press interval (also leaves room for Shift changes)
NAV_REPEAT = (0.033, 0.004)


@dataclass
class TypingConfig:
    wpm: float = 45.0
    mode: str = None                 # "transcribe" | "compose" | None (auto)
    temperature: float = 1.0         # motor timing spread
    error_scale: float = 1.0         # x typo rate
    revision_scale: float = 1.0      # x false starts / lost thoughts / in-text fixes
    pause_scale: float = 0.7         # x long (thinking) pauses; 1.0 = essay-writing levels
    max_pause: float = 8.0           # cap on any single pause (s)
    allow_uncorrected: bool = False  # let a few typos stay in the final text
    intext_revisions: bool = True    # allow arrow-key navigation back into the text
    p_late_notice: float = None      # compose: share of typos noticed only later (None = from data)
    max_travel: int = 60             # compose: furthest (chars) the writer arrows back to fix


@dataclass
class Keystroke:
    key: str                  # character, BKSP, LEFT or RIGHT
    tag: str                  # type | typo | cont | fix | false_start | lost | nav | revise
    press: float = 0.0        # seconds since the first press
    hold: float = 0.08        # seconds held down
    iki: float = 0.0          # seconds since the previous press
    pclass: str = ""          # pause class (composition)
    long: bool = False        # this interval was a long (cognitive) pause
    force: str = ""           # "long" -> writer asked for a thinking pause here


def _is_word(c):
    return c.isalnum()


def _mask(c):
    return "q" if c.isalnum() else c


def auto_mode(text):
    return "compose" if len(text.split()) >= 12 else "transcribe"


# --- Pass 1: which keys ---------------------------------------------------------------

class _Writer:
    def __init__(self, target, rng, cfg, mode, errors, comp, provider):
        self.T = target
        self.eff = list(target)        # text currently being aimed for (incl. a planned flaw)
        self.buf, self.cur = [], 0
        self.keys = []
        self.rng, self.cfg, self.mode = rng, cfg, mode
        self.errors, self.comp, self.provider = errors, comp, provider
        self.flaw = None
        self.used_fs = set()
        self.attempts = {}
        compose = mode == "compose"
        rs = cfg.revision_scale if compose else 0.0

        # per-word-start probabilities from composition rates (per typed char x ~6 chars/word)
        cpw = 6.0
        fs_rate = 0.5 * comp.delete_rate("3-9") + comp.delete_rate("10-29") + comp.delete_rate("30-400")
        self.p_false_start = min(0.5, fs_rate * cpw * rs)
        self.fs_buckets = ["3-9", "10-29", "30-400"]
        w = np.array([0.5 * comp.delete_rate("3-9"), comp.delete_rate("10-29"), comp.delete_rate("30-400")])
        self.fs_w = w / w.sum() if w.sum() > 0 else np.array([0.6, 0.3, 0.1])
        self.p_flaw = min(0.2, comp.d["intext_rate_per_1000"] / 1000.0 * cpw * 0.5 * rs) \
            if cfg.intext_revisions else 0.0
        self.p_lost = min(0.2, comp.p_long("between_words") * comp.d["p_delete_after_long_pause"] * rs)

        # composition typing has more small corrections than copying a sentence
        self.err_scale = cfg.error_scale
        if compose and errors.d.get("correction_run_rate_by_wpm") and comp.d.get("source") == "klicke":
            ratio = comp.delete_rate("1-2") / max(errors.correction_run_rate(cfg.wpm), 1e-4)
            self.err_scale *= float(np.clip(ratio, 0.7, 2.5))

        # in-text fixes (KLiCKe rate) come half from typos noticed late, half from wrong /
        # missing words (p_flaw above)
        typo_per_char = max(errors.rate(cfg.wpm) * self.err_scale, 1e-3)
        self.p_late = cfg.p_late_notice if cfg.p_late_notice is not None else             min(0.5, 0.5 * comp.d["intext_rate_per_1000"] / 1000.0 / typo_per_char * rs)

        # LLM provider: false starts only where it proposed them, at the same overall rate
        self.fs_positions = provider.positions() if provider is not None else None
        if self.fs_positions:
            n_words = max(1, len(target.split()))
            self.p_fs_pos = min(1.0, self.p_false_start * n_words / len(self.fs_positions)) if compose else 0.0

    # --- buffer primitive
    def press(self, key, tag, force=""):
        self.keys.append(Keystroke(key, tag, force=force))
        if key == BKSP:
            if self.cur > 0:
                del self.buf[self.cur - 1]
                self.cur -= 1
        elif key == LEFT:
            self.cur = max(0, self.cur - 1)
        elif key == RIGHT:
            self.cur = min(len(self.buf), self.cur + 1)
        else:
            self.buf.insert(self.cur, key)
            self.cur += 1

    def _common_prefix(self):
        k = 0
        n = min(len(self.buf), len(self.eff))
        while k < n and self.buf[k] == self.eff[k]:
            k += 1
        return k

    def _delete_to_prefix(self, tag, extra=0, force=""):
        k = self._common_prefix()
        # extra over-deletion stays inside the current word
        ws = k
        while ws > 0 and _is_word(self.buf[ws - 1]):
            ws -= 1
        extra = min(extra, k - ws)
        for i in range(len(self.buf) - k + extra):
            self.press(BKSP, tag, force=force if i == 0 else "")

    # --- behaviours
    def _word_start(self, j):
        return _is_word(self.eff[j]) and (j == 0 or not _is_word(self.eff[j - 1]))

    def maybe_false_start(self, j):
        if j in self.used_fs:
            return False
        if self.fs_positions:
            if j not in self.fs_positions or self.rng.random() >= self.p_fs_pos:
                return False
        elif self.rng.random() >= self.p_false_start:
            return False
        self.used_fs.add(j)
        bucket = self.fs_buckets[self.rng.choice(3, p=self.fs_w)]
        approx = self.comp.sample_delete_size(bucket, self.rng)
        alt = self.provider.alternative("".join(self.eff), j, approx, self.rng)
        if not alt:
            return False
        for ch in alt:
            self.press(ch, "false_start")
        self._delete_to_prefix("false_start")
        return True

    def maybe_lost_thought(self, j):
        if j < 8 or self.rng.random() >= self.p_lost:
            return False
        sizes = self.comp.d.get("delete_after_long_sizes") or [3]
        n = int(np.clip(self.rng.choice(sizes), 1, 25))
        n = min(n, len(self.buf))
        for i in range(n):
            self.press(BKSP, "lost", force="long" if i == 0 else "")
        return True

    def maybe_flaw(self, j):
        """Plan an in-text revision: type a flawed version now, fix it later."""
        if self.flaw is not None or self.rng.random() >= self.p_flaw:
            return False
        end = j
        while end < len(self.eff) and _is_word(self.eff[end]):
            end += 1
        word = "".join(self.eff[j:end])
        if len(word) < 2:
            return False
        syn = SYNONYMS.get(word.lower())
        sentence_start = j == 0 or "".join(self.eff[:j]).rstrip()[-1:] in (".", "?", "!", "\n")
        if syn and (word.islower() or sentence_start):
            alt = syn[self.rng.integers(len(syn))]
            wrong, correct = (alt[:1].upper() + alt[1:] if word[:1].isupper() else alt), word   # wrong word
        elif end < len(self.eff) and self.eff[end] == " " and j > 0 and word.islower():
            wrong, correct = "", word + " "                                   # missing word
        else:
            return False
        return self._set_flaw(j, wrong, correct, len(correct))

    def _set_flaw(self, j, wrong, correct, consumed):
        """Leave `wrong` in place of eff[j:j+consumed]; fix it after `dist` more chars.
        Returns False (no flaw) if the fix could not happen within max_travel chars."""
        dists = np.asarray(self.comp.d.get("intext_distance") or [20])
        dists = dists[(dists >= 4) & (dists <= self.cfg.max_travel)]
        if not len(dists):
            return False
        dist = int(self.rng.choice(dists))
        fix_at = j + len(wrong) + dist
        if fix_at > len(self.eff) - consumed + len(wrong):   # would run past the end of the text
            return False
        self.flaw = {"start": j, "wrong": wrong, "correct": correct, "fix_at": fix_at}
        self.eff = self.eff[:j] + list(wrong) + self.eff[j + consumed:]
        return True

    def fix_flaw(self):
        f = self.flaw
        self.flaw = None
        end_wrong = f["start"] + len(f["wrong"])
        back = len(self.buf) - end_wrong
        for i in range(back):
            self.press(LEFT, "nav")
        for _ in range(len(f["wrong"])):
            self.press(BKSP, "revise")
        for ch in f["correct"]:
            self.press(ch, "revise")
        for _ in range(back):
            self.press(RIGHT, "nav")
        self.eff = self.eff[:f["start"]] + list(f["correct"]) + self.eff[end_wrong:]

    def maybe_typo(self, j):
        c = self.eff[j]
        if self.attempts.get(j, 0) >= 2 or c == "\n":
            return False
        if self.rng.random() >= self.errors.p_error(c, self.cfg.wpm, self.err_scale):
            return False
        self.attempts[j] = self.attempts.get(j, 0) + 1
        typed, consumed = make_typo(self.eff, j, self.errors.sample_type(self.rng), self.rng)
        if not typed:
            return False
        for ch in typed:
            self.press(ch, "typo")

        if self.cfg.allow_uncorrected and self.rng.random() < self.errors.d["p_uncorrected"]:
            self.eff = self.eff[:j] + typed + self.eff[j + consumed:]     # it stays
            return True
        late = self.mode == "compose" and self.cfg.intext_revisions and self.flaw is None
        if late and self.rng.random() < self.p_late:
            if self._set_flaw(j, typed, "".join(self.eff[j:j + consumed]), consumed):
                return True

        # keep typing a little before noticing - at most to the end of the next word
        d = self.errors.sample_delay(self.rng)
        p = j + consumed
        limit, words_left = p, 2
        while limit < len(self.eff) and words_left:
            if self.eff[limit] == " " and limit > p and self.eff[limit - 1] != " ":
                words_left -= 1
                if not words_left:
                    break
            limit += 1
        for ch in self.eff[p:min(p + d, limit)]:
            if ch == "\n":
                break
            self.press(ch, "cont")
        self._delete_to_prefix("fix", extra=self.errors.sample_extra(self.rng))
        return True

    def run(self):
        compose = self.mode == "compose"
        while len(self.buf) < len(self.eff) or self.buf != self.eff:
            n = len(self.buf)
            if self.flaw is not None and n >= self.flaw["fix_at"]:
                self.fix_flaw()
                continue
            if n >= len(self.eff):
                break
            if self._common_prefix() < n:          # safety net: never build on a wrong prefix
                self._delete_to_prefix("fix")
                continue
            j = n
            if compose and self._word_start(j):
                if self.maybe_false_start(j) or self.maybe_lost_thought(j) or self.maybe_flaw(j):
                    continue
            if self.maybe_typo(j):
                continue
            self.press(self.eff[j], "type")
        if self.flaw is not None:                  # read-through at the end
            self.fix_flaw()
        return self.keys


# --- Pass 2: when ---------------------------------------------------------------------

def assign_timing(keys, session, comp, cfg, mode, rng):
    base = session.median_iki
    buf, cur = [], 0
    prev_key, prev_kind = None, None
    t = 0.0
    compose = mode == "compose"
    for i, ks in enumerate(keys):
        key = ks.key
        nxt = keys[i + 1].key if i + 1 < len(keys) else None
        iki, hold = session.sample(prev_key, key, nxt)
        is_long = False
        if key in NAV_KEYS:
            if prev_key == key:                    # holding an arrow key: auto-repeat
                iki = max(0.02, rng.normal(*NAV_REPEAT))
                hold = 0.02
            else:                                   # noticing something earlier in the text
                iki, is_long = comp.sample_pause("before_delete", base, rng, cfg.pause_scale, cfg.max_pause)
                iki = max(iki, 0.35)
            pc = "nav"
        else:
            kind = REMOVE if key == BKSP else INPUT
            prev_text = "".join(_mask(c) for c in buf[max(0, cur - 3):cur])
            pc = pause_class(_mask(key) if key != BKSP else "\b", prev_text, prev_kind, kind)
            if prev_key in NAV_KEYS:
                iki = max(iki, base * rng.uniform(1.5, 3.0))       # re-orienting after arrows
            elif compose:
                if pc not in ("in_word", "in_delete"):
                    iki, is_long = comp.sample_pause(pc, base, rng, cfg.pause_scale, cfg.max_pause)
                elif rng.random() < comp.p_long(pc) * cfg.pause_scale:
                    iki, is_long = comp.sample_pause(pc, base, rng, cfg.pause_scale, cfg.max_pause)
                    is_long = True
                elif iki >= LONG_PAUSE:
                    # thinking pauses are decided above; a motor interval this long would add
                    # unplanned ones (transcription data includes reading-the-sentence gaps)
                    iki = comp.sample_short(pc, base, rng)
            if ks.force == "long":
                p = comp.pauses.get(pc) or comp.pauses["other"]
                if len(p["long_sec"]):
                    iki = max(iki, min(rng.choice(p["long_sec"]) * cfg.pause_scale, cfg.max_pause))
                else:
                    iki = max(iki, LONG_PAUSE)
                is_long = True
            prev_kind = kind
        if i == 0:
            iki = 0.0
        else:
            iki = max(iki, MIN_IKI)
        t += iki
        ks.iki, ks.press, ks.hold, ks.pclass, ks.long = float(iki), t, float(hold), pc, is_long

        # apply to the virtual buffer (for the next key's pause class)
        if key == BKSP:
            if cur > 0:
                del buf[cur - 1]
                cur -= 1
        elif key == LEFT:
            cur = max(0, cur - 1)
        elif key == RIGHT:
            cur = min(len(buf), cur + 1)
        else:
            buf.insert(cur, key)
            cur += 1
        prev_key = key

    # holds: never past the next press of the same key (it must be released first),
    # and short enough not to trigger OS auto-repeat
    last_press = {}
    for ks in reversed(keys):
        phys = base_key(ks.key) or ks.key
        h = min(ks.hold, MAX_HOLD)
        nxt_same = last_press.get(phys)
        if nxt_same is not None:
            h = min(h, nxt_same - ks.press - 0.006)
        ks.hold = max(0.012, h)
        last_press[phys] = ks.press
    return keys


def plan_keystrokes(text, model, rng=None, cfg=None, provider=None):
    """
    Build the full keystroke plan for `text`.
    model: TypingModel (timer + errors + composition stats)
    """
    rng = rng or np.random.default_rng()
    cfg = cfg or TypingConfig()
    mode = cfg.mode or auto_mode(text)
    if provider is None:
        from .alternatives import HeuristicAlternatives
        provider = HeuristicAlternatives()
    keys = _Writer(text, rng, cfg, mode, model.errors, model.comp, provider).run()
    session = model.timer.session(cfg.wpm, rng, cfg.temperature)
    return assign_timing(keys, session, model.comp, cfg, mode, rng)
