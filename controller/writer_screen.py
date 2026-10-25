"""
Where things are in an OpenOffice Writer window, read from a screenshot - for the mouse, since
OpenOffice shows UI Automation nothing (or only its first window). Pure numpy on one capture
of the window, in-process: no OCR engine, no helper program.

    view = read_view(hwnd)                 # None if the window doesn't look like a document
    view.page                              # (left, top, right, bottom) of the visible page
    view.lines[-1]                         # the last line of text on screen: (left, top, right, bottom)
    view.menu_rect(MENUS.index("Insert"))  # the menu bar's 'Insert'

What it relies on (Writer 4.1's default look, measured on its screenshots):
  * the page is white and lies on a flat grey desk (about 171); the desk shows left of the page
    from below the ruler down to the bottom of the document area
  * text is dark (< 140) on the white page; the light grey frame of the text area (about 205)
    and the page's own border are not text
  * a line of text is a band of rows with dark pixels in it; a highlight tints it blue
  * the menu bar is a row of dark word clusters (File ... Help) just under the title bar
    (controller.calc_screen.menu_titles - the same in every OpenOffice program)
The document is written at its end, so the last line of text on screen is the one just typed
(Writer keeps the caret in view).
"""

import numpy as np

from .calc_screen import capture, menu_titles, selected

__all__ = ["MENUS", "View", "capture", "find_view", "formatting_boxes", "highlighted", "read_view", "selected"]   # capture / selected for callers
MENUS = ("File", "Edit", "View", "Insert", "Format", "Table", "Tools", "Window", "Help")   # the menu bar, in order
DESK = (160, 185)                    # the grey around the page
TEXT = 140                           # darker than this on the page is text
MIN_PAGE = 300                       # px: a narrower white stretch isn't the page


class View:
    """The visible part of a Writer document in screen pixels: the page, the lines of text on
    it (top to bottom), the document area (between the ruler and the scroll bar), the menu bar,
    and the Formatting toolbar's boxes (Apply Style, Font Name, Font Size; empty if not found)."""

    def __init__(self, page, lines, area, menu_titles_, menu_y, boxes=()):
        self.page, self.lines, self.area = page, lines, area
        self.menu_titles, self.menu_y = menu_titles_, menu_y
        self.boxes = list(boxes)

    @property
    def style_box(self):
        return self.boxes[0] if len(self.boxes) == 3 else None

    @property
    def size_box(self):
        return self.boxes[2] if len(self.boxes) == 3 else None

    def menu_rect(self, index):
        l, r = self.menu_titles[index]
        return l, self.menu_y - 7, r, self.menu_y + 7

    @property
    def last_line(self):
        return self.lines[-1] if self.lines else None


def _runs(mask):
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def find_view(rgb, offset=(0, 0)):
    """The View of a Writer window's capture (rgb: H x W x 3), or None."""
    g = rgb.astype(np.int16).mean(axis=2)
    h, w = g.shape
    # the page: the widest gap between two stretches of grey desk in the lower part of the window
    # (text, tables and the text area's frame break up the white, the desk is always flat)
    body = g[int(h * 0.35): int(h * 0.85)]
    desk_cols = ((body >= DESK[0]) & (body <= DESK[1])).mean(axis=0) >= 0.6
    desk = [(int(s), int(e)) for s, e in _runs(desk_cols) if e - s >= 8]
    gaps = [(a[1], b[0]) for a, b in zip(desk, desk[1:]) if b[0] - a[1] >= MIN_PAGE]
    if not gaps:
        return None
    pl, pr = max(gaps, key=lambda gap: gap[1] - gap[0])
    for _ in range(6):                                 # past the page's dark border
        if body[:, pl].mean() < 240:
            pl += 1
        if body[:, pr - 1].mean() < 240:
            pr -= 1
    # the document area: the rows where the grey desk shows just left of the page
    probe = g[:, max(0, pl - 30)]
    desk_rows = (probe >= DESK[0]) & (probe <= DESK[1])
    area_runs = [(int(s), int(e)) for s, e in _runs(desk_rows) if e - s >= 100]
    if not area_runs:
        return None
    top, bottom = max(area_runs, key=lambda r: r[1] - r[0])
    # the page's top edge if it is in view (else the page runs on above the document area)
    col = g[top:bottom, pl + 10]
    page_top = top + next((i for i, v in enumerate(col) if v >= 250), 0)
    # lines of text: bands of rows with dark pixels on the page
    sheet = g[top:bottom, pl + 4: pr - 4]
    dark = sheet < TEXT
    rows = dark.sum(axis=1) >= 2
    bands = []
    for s, e in _runs(rows):
        if bands and s - bands[-1][1] <= 3:            # a gap inside a line (between ascenders and descenders)
            bands[-1] = (bands[-1][0], e)
        else:
            bands.append((s, e))
    lines = []
    for s, e in bands:
        if not 6 <= e - s <= 90:
            continue
        cols = np.flatnonzero(dark[s:e].any(axis=0))
        lines.append((int(cols[0] + pl + 4), int(s + top), int(cols[-1] + pl + 5), int(e + top)))
    menu_y, titles = menu_titles(g[:top])
    boxes = formatting_boxes(g, menu_y + 10 if menu_y is not None else 0, top)
    ox, oy = offset
    shift = lambda r: (r[0] + ox, r[1] + oy, r[2] + ox, r[3] + oy)    # noqa: E731
    return View((pl + ox, page_top + oy, pr + ox, bottom + oy), [shift(r) for r in lines],
                (pl + ox, top + oy, pr + ox, bottom + oy), [(a + ox, b + ox) for a, b in titles],
                menu_y + oy if menu_y is not None else None, [shift(b) for b in boxes])


def formatting_boxes(g, top, bottom):
    """[Apply Style, Font Name, Font Size] text fields of the Formatting toolbar between rows top
    and bottom, or []. A field is a white run framed by thin border lines; the toolbar row with
    them is the one whose first three runs (left half of the window) are a wide, a wider and a
    narrow one - each field's rows above and below its text show the whole run."""
    w = g.shape[1]
    hits = {}
    for y in range(top, bottom):
        runs = [(int(s), int(e)) for s, e in _runs(g[y, : w // 2] >= 250) if e - s >= 30]
        runs = [r for r in runs if r[0] > 0]                   # not the toolbar's own white edge
        if len(runs) < 3:
            continue
        a, b, c = runs[:3]
        if a[1] - a[0] >= 100 and b[1] - b[0] >= 100 and 35 <= c[1] - c[0] <= 120 and a[0] < w * 0.15:
            ends = hits.setdefault((a[0], b[0], c[0]), {"ys": [], "ends": [a[1], b[1], c[1]]})
            ends["ys"].append(y)
            ends["ends"] = [min(e, n) for e, n in zip(ends["ends"], (a[1], b[1], c[1]))]
    if not hits:
        return []
    starts, found = max(hits.items(), key=lambda kv: len(kv[1]["ys"]))
    if len(found["ys"]) < 3:
        return []
    # the field's height: along its left edge, up and down to the border lines
    x = starts[0] + 2
    t, b = min(found["ys"]), max(found["ys"])
    while t > top and g[t - 1, x] >= 245:
        t -= 1
    while b < bottom - 1 and g[b + 1, x] >= 245:
        b += 1
    b += 1
    if not 12 <= b - t <= 45:
        return []
    return [(x0, t, x1, b) for x0, x1 in zip(starts, found["ends"])]


def highlighted(rgb, rect, offset=(0, 0), share=0.12):
    """True if at least `share` of rect is the strong blue of a text highlight (a box's text
    selected with Ctrl+A: the blue sits behind the words only, not the whole field)."""
    l, t, r, b = (v - o for v, o in zip(rect, (offset[0], offset[1], offset[0], offset[1])))
    part = rgb[t: b, l: r].astype(np.int16)
    blue = (part[..., 2] - part[..., 0] >= 60) & (part[..., 2] >= 150)
    return bool(blue.mean() >= share)


def read_view(hwnd):
    """The View of a Writer window as it is on screen now, or None."""
    rgb, offset = capture(hwnd)
    return find_view(rgb, offset)
