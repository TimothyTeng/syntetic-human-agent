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

The movement itself comes from the trained model (via HumanMouse). The timing
parameters below are hand-set placeholders for the future entropy work;
they're collected in ReadingParams so they can be tuned or learned later.
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
    pause_median: float = 1.8        # seconds, reading pause
    pause_sigma: float = 0.6         # log-normal spread of pauses
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


def _lognormal(rng, median, sigma):
    return float(median * np.exp(rng.normal(0, sigma)))


def reading_area(page_rect):
    """The central text column of the page, where a reader's cursor tends to rest."""
    left, top, right, bottom = page_rect
    w, h = right - left, bottom - top
    return (int(left + 0.2 * w), int(top + 0.15 * h), int(right - 0.2 * w), int(bottom - 0.1 * h))


def _inside(point, rect):
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def scroll_burst(rng, params=ReadingParams()):
    """Scroll a few notches in quick succession. Returns notches scrolled (neg = down)."""
    n = int(rng.integers(params.notches_min, params.notches_max + 1))
    direction = 1 if rng.random() < params.p_scroll_up else -1
    for _ in range(n):
        mouse.scroll(direction)
        time.sleep(_lognormal(rng, params.notch_gap_median, 0.4))
    return direction * n


def read_page(hm, duration, page_rect=None, params=ReadingParams(), on_trace=None):
    """
    Simulate reading the current page for about `duration` seconds.

    hm:        HumanMouse instance (provides movement + rng).
    page_rect: web page area (controller.browser.get_page_rect()); if given,
               the cursor is first moved into the text column so wheel events
               reach the page. Word tracing needs it (it only traces inside it).
    on_trace:  optional callback(list of traced word dicts), e.g. for logging.
    """
    rng = hm.rng
    area = reading_area(page_rect) if page_rect else None
    if area and not _inside(mouse.get_position(), area):
        hm.move_to(*targeting.random_point_in(area, rng))

    end = time.monotonic() + duration
    actions = ["scroll", "pause", "drift", "trace"]
    probs = np.array([params.p_scroll, params.p_pause, params.p_drift, params.p_trace if area else 0.0])
    probs = probs / probs.sum()
    while time.monotonic() < end:
        action = rng.choice(actions, p=probs)
        if action == "scroll":
            scroll_burst(rng, params)
        elif action == "pause":
            remaining = end - time.monotonic()
            time.sleep(max(0.0, min(_lognormal(rng, params.pause_median, params.pause_sigma), remaining)))
        elif action == "drift":
            hm.drift(params.drift_max_px, bounds=area)
        else:
            traced = trace_text(hm, area, params)
            if traced and on_trace:
                on_trace(traced)


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
