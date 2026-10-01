"""
Human-like typing, driven by the learned typing model (algorithms/typing_model).

    type_like_human(text, rng=None, wpm=45, sigma=0.35, ...)

1. plan   typing_model.planner turns the text into keystrokes with typos and their
          corrections, pauses, false starts and (for longer text) in-text revisions,
          each with a press time and hold time learned from real typists.
2. play   the plan becomes a timeline of key-down / key-up events (keys can overlap -
          "rollover" - and Shift is pressed a little before a capital and released
          after it), executed with sub-millisecond waits.

The final text in the field is exactly `text` (unless allow_uncorrected=True).
Assumes the caret is at the end of the field and nothing else types meanwhile.
Works with no fitted files (built-in defaults); run
`python -m algorithms.typing_model.fit_stats ...` to learn from the datasets.
"""

import time

import numpy as np
import pyautogui

from controller import keyboard
from algorithms.typing_model import keys as K
from algorithms.typing_model.alternatives import make_provider
from algorithms.typing_model.planner import TypingConfig, auto_mode, plan_keystrokes
from algorithms.typing_model.runtime import TypingModel

_MODEL = None


def get_model():
    """Load (once) the typing model from models/."""
    global _MODEL
    if _MODEL is None:
        _MODEL = TypingModel.load()
    return _MODEL


def type_like_human(text, rng=None, wpm=45, sigma=0.35, mode=None, revisions="heuristic",
                    allow_uncorrected=False, error_scale=1.0, revision_scale=1.0,
                    pause_scale=0.7, max_pause=8.0, dry_run=False, **config):
    """
    Type `text` into the focused window like a person would.

    wpm:        typing-speed persona (words per minute, 1 word = 5 characters)
    sigma:      timing variability; 0.35 = typical (kept from the old placeholder,
                it scales the model's sampling temperature)
    mode:       "transcribe" (short inputs: rhythm + typos), "compose" (longer text:
                also thinking pauses, false starts, going back to fix things), or
                None = automatic (compose from 12 words up)
    revisions:  "heuristic" (offline) or "llm" (ask Claude for first-draft wordings;
                needs the anthropic package + credentials, falls back to heuristic)
    allow_uncorrected: leave the occasional typo in the final text
    error_scale / revision_scale / pause_scale: multiply typo rate, revision rate
                and long-pause length (pause_scale 1.0 = timed-essay levels, which
                feel slow for emails); max_pause caps any single pause (s)
    dry_run:    plan only, do not press keys
    Returns the keystroke plan (list of typing_model.planner.Keystroke).
    """
    rng = rng or np.random.default_rng()
    cfg = TypingConfig(wpm=wpm, mode=mode, temperature=sigma / 0.35, error_scale=error_scale,
                       revision_scale=revision_scale, pause_scale=pause_scale, max_pause=max_pause,
                       allow_uncorrected=allow_uncorrected, **config)
    provider = make_provider(revisions)
    if (cfg.mode or auto_mode(text)) == "compose":
        provider.prepare(text, rng)
    plan = plan_keystrokes(text, get_model(), rng, cfg, provider)
    if not dry_run:
        play(plan, rng)
    return plan


# --- Player ---------------------------------------------------------------------------

_SPECIAL = {K.BKSP: "backspace", "\n": "enter", "\t": "tab", K.LEFT: "left", K.RIGHT: "right"}


def _physical(ch):
    """(pyautogui key name, needs_shift) for a character, or (None, False) if it can't be
    pressed as a plain key on this keyboard layout (then it is typed atomically)."""
    if ch in _SPECIAL:
        return _SPECIAL[ch], False
    base = K.base_key(ch)
    mapping = pyautogui.platformModule.keyboardMapping
    if base is None or base not in mapping or ch not in mapping or mapping[base] is None or mapping[ch] is None:
        return None, False
    mods, vk = divmod(mapping[ch], 0x100)
    bmods, bvk = divmod(mapping[base], 0x100)
    if vk != bvk or bmods != 0:            # layout differs from US: let pyautogui handle it
        return None, False
    return base, bool(mods & 1) or K.needs_shift(ch)


def build_timeline(plan, rng, timer=None):
    """
    Keystrokes -> sorted [(t, order, action, key)] with action in {"down", "up", "char"}.
    Shift is held around runs of shifted characters, pressed `lead` before the first
    and released `lag` after the last (never while an unshifted key goes down).
    """
    timer = timer or get_model().timer
    events = []
    phys = [_physical(ks.key) for ks in plan]
    for ks, (name, _) in zip(plan, phys):
        if name is None:
            events.append((ks.press, 1, "char", ks.key))
        else:
            events.append((ks.press, 1, "down", name))
            events.append((ks.press + ks.hold, 2, "up", name))

    # Shift runs
    i, n = 0, len(plan)
    last_up = -1e9
    while i < n:
        if not phys[i][1]:
            i += 1
            continue
        j = i
        while j + 1 < n and phys[j + 1][1] and plan[j + 1].press - plan[j].press < 0.6:
            j += 1
        lead, lag = timer.sample_shift(rng) if hasattr(timer, "sample_shift") else (0.08, 0.03)
        prev_press = plan[i - 1].press if i > 0 else plan[i].press - 1.0
        lo = max(prev_press, last_up) + 0.002
        down = float(np.clip(plan[i].press - lead, lo, plan[i].press - 0.002))
        last = plan[j]
        hi = plan[j + 1].press - 0.002 if j + 1 < n else np.inf
        up = float(np.clip(last.press + last.hold + lag, last.press + 0.002, hi))
        events.append((down, 0, "down", "shift"))
        events.append((up, 0, "up", "shift"))
        last_up = up
        i = j + 1
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def _wait_until(target):
    while True:
        rem = target - time.perf_counter()
        if rem <= 0:
            return
        if rem > 0.003:
            time.sleep(rem - 0.002)


def _run_events(events):
    """Execute [(t, action, key)] (t = seconds from now, action 'down'/'up') with accurate
    waits; any key still held at the end (or on an exception) is released."""
    held = set()
    t0 = time.perf_counter()
    try:
        for t, action, key in sorted(events, key=lambda e: e[0]):
            _wait_until(t0 + t)
            if action == "down":
                keyboard.key_down(key)
                held.add(key)
            else:
                keyboard.key_up(key)
                held.discard(key)
    finally:
        for key in list(held):
            keyboard.key_up(key)


def play(plan, rng=None):
    """Execute a keystroke plan on the focused window."""
    rng = rng or np.random.default_rng()
    events = build_timeline(plan, rng)
    held = set()
    t0 = time.perf_counter()
    try:
        for t, _, action, key in events:
            _wait_until(t0 + t)
            if action == "down":
                keyboard.key_down(key)
                held.add(key)
            elif action == "up":
                keyboard.key_up(key)
                held.discard(key)
            else:
                keyboard.type_char(key)
    finally:
        for key in list(held):             # never leave a key (esp. Shift) stuck down
            keyboard.key_up(key)


# --- Single keys and shortcuts ----------------------------------------------------------
# The "glue" keys between typed text (Enter, F5, Ctrl+L, Alt+Left...) get the same learned
# hold times and modifier timing as typing, instead of being pressed in zero time.

# pyautogui key name -> character the timing model knows; other keys use OTHER_KEY
_TIMING_CHARS = {"enter": "\n", "return": "\n", "tab": "\t", "space": " ",
                 "backspace": K.BKSP, "left": K.LEFT, "right": K.RIGHT}
OTHER_KEY = "\x00"     # maps to the model's "<other>" key class


def _timing_key(name):
    if name in _TIMING_CHARS:
        return _TIMING_CHARS[name]
    return name if len(name) == 1 else OTHER_KEY


def key_timer(wpm=45, rng=None, sigma=0.35):
    """
    A motor-timing session for single keys / shortcuts. Keep one per persona (as
    behaviour.Human does) so hold times stay consistent for the same "person".
    """
    return get_model().timer.session(wpm, rng or np.random.default_rng(), temperature=sigma / 0.35)


def press_like_human(key, rng=None, presses=1, timer=None, wpm=45):
    """
    Press a key (pyautogui name: 'enter', 'f5', 'down', 'a', ...) with a learned hold
    time. presses > 1 repeats it with learned key-to-key intervals (e.g. 3x 'down').
    """
    rng = rng or np.random.default_rng()
    timer = timer or key_timer(wpm, rng)
    tk = _timing_key(key)
    events, t, prev = [], 0.0, None
    for i in range(presses):
        iki, hold = timer.sample(prev, tk)
        if i:
            t = max(t + iki, events[-1][0] + 0.01)      # the previous press must be released first
        events += [(t, "down", key), (t + hold, "up", key)]
        prev = tk
    _run_events(events)


def hotkey_like_human(*keys, rng=None, timer=None, wpm=45):
    """
    Press a shortcut like a person: modifiers go down one after another, the main key
    follows after a learned Shift-style lead time and is held for a learned hold time,
    then the modifiers come up after a learned lag - which is sometimes negative, i.e.
    the modifier is let go while the main key is still down, as real typists do.

    Example: hotkey_like_human('ctrl', 'l', rng=rng)
    """
    rng = rng or np.random.default_rng()
    timer = timer or key_timer(wpm, rng)
    model_timer = get_model().timer
    *mods, main = keys
    events, t = [], 0.0
    for i, m in enumerate(mods):
        if i:   # a second modifier follows the first quickly
            t += float(np.clip(0.05 * np.exp(rng.normal(0, 0.5)), 0.015, 0.25))
        events.append((t, "down", m))
    lead, _ = model_timer.sample_shift(rng) if mods else (0.0, 0.0)
    t_main = t + float(np.clip(lead, 0.03, 0.6))
    _, hold = timer.sample(None, _timing_key(main))
    events += [(t_main, "down", main), (t_main + hold, "up", main)]
    for i, m in enumerate(reversed(mods)):
        _, lag = model_timer.sample_shift(rng)
        events.append((max(t_main + 0.015, t_main + hold + lag + 0.02 * i), "up", m))
    _run_events(events)
