"""
Reading behaviour: what a person does while reading a web page.

Alternates between
  * scroll bursts - a few wheel notches in quick succession, mostly downwards,
    occasionally back up,
  * reading pauses - log-normally distributed (most short, a few long),
  * mouse drifts - small idle movements made with the learned mouse model,
  * word tracing - the pointer follows a line of text word by word, the way some
    people read with the cursor: short hops, a pause on each word (longer for long
    words), now and then a step back, sometimes on to the next line.

Scrolling is done with the wheel or, for keyboard-minded people, PgDn / arrow keys
(only when the page itself has keyboard focus). Once the bottom of the page is
reached the reader either scrolls back up to re-read something or is done.
reading_time() sizes the whole read from the amount of text on the page.

The movement itself comes from the trained model (via HumanMouse). Scroll timing -
notches per burst, the gaps between them and the pauses between bursts - is learned
from the Balabit dataset (mouse_model/scroll_stats.py, one real user's style per
person) when ReadingParams.scroll is set; the remaining parameters are hand-set.
"""

import time
from dataclasses import dataclass

import numpy as np

from controller import browser, mouse, screen, ui_elements

from . import player, targeting
from .mouse_model.fallback import minimum_jerk


@dataclass
class ReadingParams:
    p_scroll: float = 0.45           # chance the next action is a scroll burst
    p_pause: float = 0.35            # ... a reading pause
    p_drift: float = 0.20            # ... a small mouse drift
    notches_min: int = 1             # wheel notches per burst
    notches_max: int = 5
    notch_gap_median: float = 0.07   # seconds between notches in a burst
    p_scroll_up: float = 0.12        # chance a burst scrolls back up
    pause_median: float = 1.8        # seconds, reading pause (when no learned scroll timing)
    pause_sigma: float = 0.6         # log-normal spread of pauses
    pause_max: float = 20.0          # longest single reading pause
    scroll: object = None            # learned ScrollStats (notches, gaps, pauses); None = hand-set
    drift_max_px: int = 120
    # word tracing
    p_trace: float = 0.12            # ... tracing a line of text with the cursor
    trace_words_min: int = 2         # words followed per trace
    trace_words_max: int = 10
    trace_offset_px: tuple = (3, 9)  # cursor sits this far below the words' centre line
    fixation_base: float = 0.17      # seconds resting on a word at reading_wpm = 230 ...
    fixation_per_char: float = 0.025  # ... plus this per character
    fixation_sigma: float = 0.35
    p_regress: float = 0.10          # hop back one word, then carry on
    p_next_line: float = 0.25        # at the end of a line, sweep to the next one and continue
    p_line_start: float = 0.6        # start at the beginning of the line (else early in it)
    reading_wpm: float = 230.0
    # scrolling method and end of page
    p_key_scroll: float = 0.0        # chance a scroll burst uses PgDn / arrow keys (needs `human`)
    p_page_key: float = 0.5          # ... PgDn/PgUp once, else a few arrow presses
    end_percent: float = 99.0        # scroll position (%) that counts as the bottom
    p_reread_at_end: float = 0.4     # at the bottom: scroll back up to re-read, else done
    # reading_time(): web readers read only part of a page's text
    read_fraction: float = 0.28
    read_fraction_sigma: float = 0.5


def _lognormal(rng, median, sigma):
    return float(median * np.exp(rng.normal(0, sigma)))


def reading_area(page_rect):
    """The central text column of the page, where a reader's cursor tends to rest."""
    left, top, right, bottom = page_rect
    w, h = right - left, bottom - top
    return (int(left + 0.2 * w), int(top + 0.15 * h), int(right - 0.2 * w), int(bottom - 0.1 * h))


def _inside(point, rect):
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def scroll_burst(rng, params=ReadingParams(), direction=None):
    """Scroll a few wheel notches in quick succession. Returns notches scrolled (neg = down)."""
    learned = params.scroll
    n = learned.sample_burst(rng) if learned else int(rng.integers(params.notches_min, params.notches_max + 1))
    if direction is None:
        direction = 1 if rng.random() < params.p_scroll_up else -1
    for i in range(n):
        if i:
            time.sleep(learned.sample_gap(rng) if learned else _lognormal(rng, params.notch_gap_median, 0.4))
        mouse.scroll(direction)
    return direction * n


def reading_pause(rng, params=ReadingParams()):
    """Seconds of reading without scrolling (scaled by reading speed)."""
    if params.scroll:
        secs = params.scroll.sample_pause(rng, params.pause_max)
    else:
        secs = _lognormal(rng, params.pause_median, params.pause_sigma)
    return min(secs * 230.0 / params.reading_wpm, params.pause_max)


def key_scroll(human, rng, params=ReadingParams(), direction=-1):
    """Scroll with the keyboard: PgDn/PgUp once, or a few arrow presses (direction -1 = down)."""
    if rng.random() < params.p_page_key:
        human.press("pagedown" if direction < 0 else "pageup")
    else:
        human.press("down" if direction < 0 else "up", presses=int(rng.integers(2, 7)))


def read_page(hm, duration, page_rect=None, params=ReadingParams(), on_trace=None, human=None):
    """
    Simulate reading the current page for about `duration` seconds. Returns early if
    the reader reaches the bottom of the page and is done with it.

    hm:        HumanMouse instance (provides movement + rng).
    page_rect: web page area (controller.browser.get_page_rect()); if given,
               the cursor is first moved into the text column so wheel events
               reach the page. Word tracing needs it (it only traces inside it).
    on_trace:  optional callback(list of traced word dicts), e.g. for logging.
    human:     optional behaviour.Human - enables keyboard scrolling
               (params.p_key_scroll) and hand-switch (homing) time between devices.
    """
    rng = hm.rng
    area = reading_area(page_rect) if page_rect else None
    use = human.use_device if human else (lambda device: None)
    if area and not _inside(mouse.get_position(), area):
        use("mouse")
        hm.move_to(*targeting.random_point_in(area, rng))

    end = time.monotonic() + duration
    actions = ["scroll", "pause", "drift", "trace"]
    probs = np.array([params.p_scroll, params.p_pause, params.p_drift, params.p_trace if area else 0.0])
    probs = probs / probs.sum()
    at_bottom = False
    last_view = None                                  # for pages that don't report a scroll position
    while time.monotonic() < end:
        action = rng.choice(actions, p=probs)
        if action == "scroll":
            if at_bottom and rng.random() >= params.p_reread_at_end:
                return                                  # read to the end - done with this page
            direction = 1 if at_bottom or rng.random() < params.p_scroll_up else -1
            if human and rng.random() < params.p_key_scroll and browser.page_has_keyboard_focus():
                key_scroll(human, rng, params, direction)
            else:
                use("mouse")
                scroll_burst(rng, params, direction)
            state = browser.page_scroll_state()
            if state is not None:
                at_bottom = state[0] >= params.end_percent
            elif direction < 0:                       # bottom = scrolling down moved nothing
                view = browser.page_view_signature()
                at_bottom = view is not None and view == last_view
                last_view = view
            else:
                at_bottom, last_view = False, None
        elif action == "pause":
            remaining = end - time.monotonic()
            time.sleep(max(0.0, min(reading_pause(rng, params), remaining)))
        elif action == "drift":
            use("mouse")
            hm.drift(params.drift_max_px, bounds=area)
        else:
            use("mouse")
            traced = trace_text(hm, area, params)
            if traced and on_trace:
                on_trace(traced)


def reading_time(rng, params=ReadingParams(), lo=10.0, hi=120.0):
    """
    Seconds a person would spend on the current page: the page's words (visible words
    scaled up by how much of the page is in view) times the share web readers actually
    read (~28%, log-normal), at params.reading_wpm. Clipped to [lo, hi]; None if the
    page exposes no text.
    """
    state = browser.page_scroll_state()
    if state and 0 < state[1] < 100:                  # visible words scaled up to the whole page
        words = browser.page_word_count() * 100.0 / state[1]
    else:                                             # no scroll position: count the whole page's text
        words = browser.page_total_words() or browser.page_word_count()
    if not words:
        return None
    seconds = words * params.read_fraction / params.reading_wpm * 60.0
    return float(np.clip(seconds * np.exp(rng.normal(0, params.read_fraction_sigma)), lo, hi))


# --- Word tracing -----------------------------------------------------------------------

def visible_text_lines(area, rng, max_nodes=4, min_words=3):
    """
    Lines of words currently visible inside `area`: [[{'text', 'rect'}, ...], ...].

    Text elements come from Chrome's accessibility tree; a few are picked (favouring
    the upper-middle of the view, where a reader is after scrolling) and split into
    words. OCR is the fallback for pages that expose no text. [] if neither works.
    """
    pattern, nodes = browser.page_text_nodes(area, min_words)
    lines = []
    if nodes:
        w = np.array([_position_weight(n["rect"], area) * min(len(n["text"].split()), 30) for n in nodes])
        picks = rng.choice(len(nodes), size=min(max_nodes, len(nodes)), replace=False, p=w / w.sum())
        for i in sorted(picks):
            lines += ui_elements.group_lines(ui_elements.node_words(nodes[i], pattern))
    elif screen.ocr_available():
        lines = screen.ocr_lines(area)
    return [ln for ln in lines if len(ln) >= min_words and all(_inside(_centre(w["rect"]), area) for w in ln)]


def trace_text(hm, area, params=ReadingParams()):
    """
    Follow a line of visible text with the cursor. Returns the traced words ([] if no
    text was found - then nothing moves). Word positions are fetched fresh each time,
    since any scroll invalidates them.
    """
    rng = hm.rng
    lines = visible_text_lines(area, rng)
    if not lines:
        return []
    seq = plan_trace(lines, area, rng, params)
    if not seq:
        return []
    offset = rng.uniform(*params.trace_offset_px)
    anchors = [_anchor(w["rect"], offset, rng) for w in seq]
    dwells = [_fixation(w["text"], rng, params) for w in seq]
    hm.move_to(*anchors[0])                                    # an ordinary aimed movement
    player.play_trajectory(trace_path(anchors, dwells, rng))   # then the slow reading motion
    return seq


def plan_trace(lines, area, rng, params=ReadingParams()):
    """
    Choose which words to visit, in order: a line (upper-middle of `area` favoured), a
    start word, then n words forward with occasional one-word regressions and, at the
    end of a line, sometimes a return sweep to the next line.
    """
    w = np.array([_position_weight(_line_rect(ln), area) for ln in lines])
    li = int(rng.choice(len(lines), p=w / w.sum()))
    line = lines[li]
    wi = 0 if rng.random() < params.p_line_start else int(rng.integers(0, max(1, len(line) // 2)))
    n = int(rng.integers(params.trace_words_min, params.trace_words_max + 1))
    seq = []
    while len(seq) < n:
        if wi >= len(line):
            if li + 1 < len(lines) and rng.random() < params.p_next_line:
                li, wi = li + 1, 0
                line = lines[li]
            else:
                break
        seq.append(line[wi])
        if wi > 0 and rng.random() < params.p_regress:
            seq += [line[wi - 1], line[wi]]                  # glance back, then on
        wi += 1
    return seq


def trace_path(anchors, dwells, rng, hz=100):
    """
    The slow reading motion as [(x, y, dt), ...], starting at anchors[0].

    On each word the cursor rests (dwell seconds), nearly still: it creeps forward a
    pixel or three. Between words it hops with a minimum-jerk profile lasting 60-160 ms
    (longer hops, such as a return sweep to the next line, up to 300 ms). The vertical
    offset wanders slowly, so the path is not ruler-straight.
    """
    period = 1.0 / hz
    pts = []
    x, y = map(float, anchors[0])
    wobble = 0.0
    for i, dwell in enumerate(dwells):
        # rest on the word: a few 1-px forward creeps at random moments
        times = np.sort(rng.uniform(0, dwell, int(rng.integers(0, 4))))
        last = 0.0
        for t in times:
            x += 1
            pts.append((x, y, float(t - last)))
            last = float(t)
        pending = dwell - last
        if i + 1 == len(anchors):
            break
        # hop to the next word
        wobble = float(np.clip(wobble + rng.normal(0, 0.8), -3, 3))
        tx, ty = anchors[i + 1][0], anchors[i + 1][1] + wobble
        dist = float(np.hypot(tx - x, ty - y))
        dur = float(np.clip(0.06 + 0.0011 * dist, 0.06, 0.16 if dist < 150 else 0.3) * np.exp(rng.normal(0, 0.15)))
        steps = max(2, int(round(dur / period)))
        s = minimum_jerk(np.arange(1, steps + 1) / steps)
        x0, y0 = x, y
        for k, f in enumerate(s):
            dt = period * rng.uniform(0.9, 1.1) + (pending if k == 0 else 0.0)
            pts.append((x0 + (tx - x0) * f, y0 + (ty - y0) * f + rng.normal(0, 0.3), float(dt)))
        x, y = tx, ty
    return pts


def _fixation(word, rng, params):
    """Seconds the cursor rests on a word: longer for long words, scaled by reading speed."""
    base = params.fixation_base + params.fixation_per_char * len(word)
    return _lognormal(rng, base * 230.0 / params.reading_wpm, params.fixation_sigma)


def _anchor(rect, offset, rng):
    """Cursor point for a word: ~40% into it (readers lead with the word start), just below its centre."""
    left, top, right, bottom = rect
    x = left + (right - left) * float(np.clip(rng.normal(0.4, 0.1), 0.15, 0.85))
    return int(round(x)), int(round((top + bottom) / 2 + offset))


def _centre(rect):
    return (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2


def _line_rect(line):
    return (line[0]["rect"][0], min(w["rect"][1] for w in line),
            line[-1]["rect"][2], max(w["rect"][3] for w in line))


def _position_weight(rect, area, focus=0.35, width=0.35):
    """Favour text about `focus` of the way down the view (gaussian over vertical position)."""
    h = max(area[3] - area[1], 1)
    rel = ((rect[1] + rect[3]) / 2 - area[1]) / h
    return float(np.exp(-((rel - focus) / width) ** 2)) + 1e-3
