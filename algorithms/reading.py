"""
Reading behaviour: what a person does while reading a web page.

read_article() is the reader used for browsing. It works through a page the way people
read articles - from the top down, at a pace set by the text on screen:
  * the lead is read first (a pause for the words in view at this person's reading speed),
  * then a step down - a flick of the wheel worth 40-80% of a screen (keyboard-minded people:
    PgDn or a few arrow presses) - and a pause for the words that came INTO view with it,
  * how much of the new text gets read follows an engagement level that drifts as the page
    goes on (skimming: big steps, short pauses; reading; close reading: longer pauses and
    now and then following a line with the cursor),
  * now and then a short scroll back up to re-read something, an idle mouse drift, or the
    cursor resting on a link without clicking it,
  * at the bottom: done, or a scroll back up to look at something again first.
It ends at the bottom ('finished'), when the time budget is spent ('budget' - the rest is
left unread) or, rarely, right after the lead ('bored').

read_page() is the same reader with the older signature (used by the session demo and
glance_at_source). Its building blocks:
  * scroll bursts - wheel notches in quick succession,
  * reading pauses - log-normally distributed,
  * mouse drifts - small idle movements made with the learned mouse model,
  * word tracing - the pointer follows a line of text word by word, the way some
    people read with the cursor: short hops, a pause on each word (longer for long
    words), now and then a step back, sometimes on to the next line.

Scrolling is done with the wheel or, for keyboard-minded people, PgDn / arrow keys
(only when the page itself has keyboard focus). reading_time() sizes the whole read from
the amount of text on the page.

The movement itself comes from the trained model (via HumanMouse). Scroll timing - the
gaps between notches and the pauses when no text is reported - is learned from the
Balabit dataset (mouse_model/scroll_stats.py, one real user's style per person) when
ReadingParams.scroll is set; the remaining parameters are hand-set. The decisions (step
size, words revealed, pause for them, engagement drift) are pure functions, tested offline.
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
    # read_article(): progressive reading
    step_frac: tuple = (0.4, 0.8)    # share of the screen one step down scrolls (reading)
    notch_px: float = 100.0          # pixels one wheel notch scrolls in Chrome
    p_reread_step: float = 0.10      # per step: scroll back up a little and re-read
    p_drift_step: float = 0.12       # ... an idle mouse drift
    p_hover_link: float = 0.04       # ... rest the cursor on a link in view, without clicking
    p_bored: float = 0.03            # leave right after the lead
    engagement_stay: float = 0.7     # chance the engagement level stays the same for the next step


def _lognormal(rng, median, sigma):
    return float(median * np.exp(rng.normal(0, sigma)))


def reading_area(page_rect):
    """The central text column of the page, where a reader's cursor tends to rest."""
    left, top, right, bottom = page_rect
    w, h = right - left, bottom - top
    return (int(left + 0.2 * w), int(top + 0.15 * h), int(right - 0.2 * w), int(bottom - 0.1 * h))


def _inside(point, rect):
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def scroll_burst(rng, params=ReadingParams(), direction=None, n=None):
    """Scroll a few wheel notches in quick succession (n: how many; default from the learned
    burst sizes). Returns notches scrolled (neg = down)."""
    learned = params.scroll
    if n is None:
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


# --- Progressive reading ---------------------------------------------------------------------

# engagement level -> (share of the newly revealed words actually read, chance of tracing a line)
ENGAGEMENT = {"skim": (0.12, 0.08), "read": (0.45, 0.35), "close": (0.85, 0.65)}
ENGAGEMENT_START = {"skim": 0.25, "read": 0.55, "close": 0.20}
WORDS_PER_SCREEN = 150               # assumed when a page doesn't expose its text


@dataclass
class ReadResult:
    reason: str                      # 'finished' (bottom reached), 'budget' (time spent) or 'bored'
    seconds: float                   # time spent reading
    seen: float = None               # furthest scroll position reached (%), None if the page doesn't say
    steps: int = 0                   # steps down the page
    rereads: int = 0                 # scrolls back up to re-read
    traces: int = 0                  # lines followed with the cursor


def next_engagement(rng, current, stay=0.7):
    """The engagement level for the next step: usually the same as now, else drawn afresh."""
    if current is not None and rng.random() < stay:
        return current
    names = list(ENGAGEMENT_START)
    p = np.array([ENGAGEMENT_START[k] for k in names])
    return str(names[int(rng.choice(len(names), p=p / p.sum()))])


def step_notches(rng, view_px, engagement, params=ReadingParams()):
    """Wheel notches for one step down: 40-80% of a screen when reading, nearly a whole screen
    (or more) when skimming. At least one."""
    lo, hi = params.step_frac
    if engagement == "skim":
        lo, hi = hi, hi + 0.35
    return max(1, int(round(rng.uniform(lo, hi) * view_px / params.notch_px)))


def view_top(state):
    """Top of the view as a percentage of the page, from (scroll position %, visible %)."""
    pos, view = state
    return pos * (100.0 - view) / 100.0


def revealed_words(new_state, seen_bottom, words_in_view):
    """Words that came into view for the first time: the part of the view below the furthest
    point seen so far (seen_bottom, % of the page), as a share of the words in view."""
    top, view = view_top(new_state), new_state[1]
    if view <= 0:
        return 0.0
    fresh = max(0.0, top + view - max(seen_bottom, top))
    return words_in_view * min(1.0, fresh / view)


def words_pause(rng, words, wpm, share, lo=0.4, hi=30.0):
    """Seconds to read `share` of `words` at `wpm` (log-normal spread), within [lo, hi]."""
    secs = words * share / max(wpm, 1.0) * 60.0 * float(np.exp(rng.normal(0, 0.3)))
    return float(np.clip(secs, lo, hi))


def time_pressure(budget, page_words, wpm, share=0.45):
    """How much of the usual pace fits the budget: 1 when there is time to read the page as
    usual, down to 0.2 when it is short - pauses are scaled by it, so a reader with little
    time skims through the page instead of stopping at its start."""
    natural = page_words * share / max(wpm, 1.0) * 60.0
    return float(np.clip(budget / natural, 0.2, 1.0)) if natural > 0 else 1.0


def wait_for_text(timeout=4.0, min_words=20):
    """Words in view once the page exposes its text (Chrome builds a fresh page's accessibility
    tree a moment after it loads; each try asks for it again). 0 if it never does."""
    deadline = time.monotonic() + timeout
    while True:
        words = browser.page_word_count()
        if words >= min_words or time.monotonic() >= deadline:
            return words
        time.sleep(0.6)


def _sleep_until(t):
    remaining = t - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def step_down(rng, params, notches, human=None, use=lambda device: None):
    """Scroll one step down: with the keyboard (PgDn for a big step, else arrows) for
    keyboard-minded people when the page has the focus, else the wheel - big steps as two
    flicks with a short gap, like a reader's finger."""
    if human and rng.random() < params.p_key_scroll and browser.page_has_keyboard_focus():
        if notches * params.notch_px >= 500 and rng.random() < params.p_page_key:
            human.press("pagedown")
        else:
            human.press("down", presses=int(np.clip(round(notches * params.notch_px / 40), 2, 12)))
        return
    use("mouse")
    if notches > 5 and rng.random() < 0.6:
        first = int(rng.integers(2, notches - 1))
        scroll_burst(rng, params, -1, first)
        time.sleep(float(rng.uniform(0.15, 0.6)))
        scroll_burst(rng, params, -1, notches - first)
    else:
        scroll_burst(rng, params, -1, notches)


def hover_link(hm, area, rng):
    """Rest the cursor on a link in view for a moment (considering it) without clicking."""
    try:
        links = [k for k in browser.find_links() if _inside(k["center"], area) and len(k["name"]) > 3]
    except Exception:
        links = []
    if not links:
        return False
    link = links[int(rng.integers(len(links)))]
    l, t, r, b = link["rect"]
    hm.move_to(int(rng.uniform(l + 2, max(l + 3, r - 2))), int((t + b) / 2 + rng.normal(0, 1.5)))
    time.sleep(float(rng.uniform(0.4, 1.4)))
    return True


def read_article(hm, budget, page_rect=None, params=ReadingParams(), on_trace=None, human=None, on_view=None):
    """
    Read the current page from where it is scrolled to, top-down, for at most `budget`
    seconds (see the module docstring). Returns a ReadResult.

    hm:        HumanMouse instance (provides movement + rng).
    page_rect: web page area (controller.browser.get_page_rect()); the cursor is first moved
               into the text column so wheel events reach the page. Tracing and link hovers
               need it.
    on_trace:  optional callback(list of traced word dicts), e.g. for logging.
    human:     optional behaviour.Human - keyboard scrolling (params.p_key_scroll) and the
               hand-switch (homing) time between devices.
    on_view:   optional callback((scroll position %, visible %)) - at the start and after
               every scroll, for pages that report their scroll position (what was seen).
    """
    rng = hm.rng
    t0 = time.monotonic()
    end = t0 + budget
    area = reading_area(page_rect) if page_rect else None
    view_px = (page_rect[3] - page_rect[1]) if page_rect else 700
    use = human.use_device if human else (lambda device: None)
    if area and not _inside(mouse.get_position(), area):
        use("mouse")
        hm.move_to(*targeting.random_point_in(area, rng))

    traced = [0]

    def result(reason):
        return ReadResult(reason, time.monotonic() - t0, seen, steps, rereads, traced[0])

    def read_for(seconds, p_trace):
        traced[0] += _read_for(hm, area, params, on_trace, use, rng, end, seconds, p_trace)

    words_in_view = wait_for_text() or WORDS_PER_SCREEN     # 0: the page exposes no text
    state = browser.page_scroll_state()
    if on_view and state is not None:
        on_view(state)
    seen = state[0] if state else None
    seen_bottom = view_top(state) + state[1] if state else 0.0
    page_words = words_in_view * 100.0 / state[1] if state and state[1] > 0 else words_in_view * 6
    pace = time_pressure(budget, page_words, params.reading_wpm)
    step_cap = max(2.0, budget / 5.0)                 # never stand still for a big part of the budget
    engagement = next_engagement(rng, None)
    steps = rereads = 0
    reread_at_end = False
    last_view = None

    # the lead: the text in view when the page opens (at most ~30% of the time there is)
    share, p_trace = ENGAGEMENT[engagement]
    lead = words_pause(rng, words_in_view, params.reading_wpm, max(share, 0.4), lo=1.5, hi=25.0) * pace
    read_for(min(lead, max(2.5, 0.3 * budget)), p_trace)
    if budget > 30 and rng.random() < params.p_bored:
        return result("bored")

    at_bottom = state is not None and state[0] >= params.end_percent
    while time.monotonic() < end - 0.5:
        if at_bottom:
            if reread_at_end or rng.random() >= params.p_reread_at_end or end - time.monotonic() < 8:
                return result("finished")
            reread_at_end = True                       # once: look at something further up again
            use("mouse")
            scroll_burst(rng, params, 1, int(rng.integers(3, 9)))
            rereads += 1
            _sleep_until(min(end, time.monotonic() + _lognormal(rng, 3.0, 0.4)))
            state = browser.page_scroll_state()
            if on_view and state is not None:
                on_view(state)
            at_bottom = False
            continue
        r = rng.random()
        if r < params.p_reread_step and steps > 0:     # back up a little to re-read a passage
            use("mouse")
            scroll_burst(rng, params, 1, int(rng.integers(1, 4)))
            rereads += 1
            _sleep_until(min(end, time.monotonic() + _lognormal(rng, 2.2, 0.45)))
            state = browser.page_scroll_state() or state
            continue
        r -= params.p_reread_step
        if r < params.p_drift_step:
            use("mouse")
            hm.drift(params.drift_max_px, bounds=area)
            continue
        r -= params.p_drift_step
        if r < params.p_hover_link and area:
            use("mouse")
            hover_link(hm, area, rng)
            continue

        # a step down, then read what came into view
        engagement = next_engagement(rng, engagement, params.engagement_stay)
        share, p_trace = ENGAGEMENT[engagement]
        notches = step_notches(rng, view_px, engagement, params)
        step_down(rng, params, notches, human, use)
        steps += 1
        time.sleep(0.15)                               # the page redraws
        if steps % 3 == 0:                             # the amount of text per screen changes along a page
            words_in_view = browser.page_word_count() or words_in_view
        state = browser.page_scroll_state()
        if state is not None:
            if on_view:
                on_view(state)
            fresh = revealed_words(state, seen_bottom, words_in_view)
            seen_bottom = max(seen_bottom, view_top(state) + state[1])
            seen = max(seen or 0.0, state[0])
            at_bottom = state[0] >= params.end_percent
        else:                                          # no scroll position: bottom = the view stopped changing
            view = browser.page_view_signature()
            at_bottom = view is not None and view == last_view
            last_view = view
            fresh = 0.0 if at_bottom else words_in_view * min(1.0, notches * params.notch_px / view_px)
        read_for(min(words_pause(rng, fresh, params.reading_wpm, share) * pace, step_cap), p_trace)
    return result("budget")


def _read_for(hm, area, params, on_trace, use, rng, end, seconds, p_trace):
    """Read for `seconds` (not past `end`): with chance p_trace following a line with the cursor
    - in a long pause sometimes another one after it - otherwise still. Returns the lines
    traced."""
    until = min(end, time.monotonic() + seconds)
    n, p = 0, p_trace
    while area and until - time.monotonic() > 2.0 and rng.random() < p:
        use("mouse")
        words = trace_text(hm, area, params)
        if not words:                                  # no text found in view: nothing to follow
            break
        n += 1
        if on_trace:
            on_trace(words)
        p = p_trace * 0.5
    _sleep_until(until)
    return n


def read_page(hm, duration, page_rect=None, params=ReadingParams(), on_trace=None, human=None, on_view=None):
    """read_article() under its older name and signature: read the current page for about
    `duration` seconds (less if the reader reaches the bottom and is done). Returns the
    ReadResult."""
    return read_article(hm, duration, page_rect, params, on_trace, human, on_view)


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
    try:
        pattern, nodes = browser.page_text_nodes(area, min_words)
    except Exception:                                 # the page changed under us (stale UIA elements)
        pattern, nodes = None, []
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
