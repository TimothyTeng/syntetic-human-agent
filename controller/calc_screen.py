"""
Where things are in an OpenOffice Calc window, read from a screenshot - for the mouse, since
OpenOffice shows UI Automation nothing (or only its first window). Pure numpy on one capture
of the window, in-process: no OCR engine, no helper program.

    grid = read_grid(hwnd)                 # None if the window doesn't look like a sheet
    grid.cell_rect("C7")                   # (left, top, right, bottom) in screen pixels
    grid.column_header_rect("B")           # the 'B' box in the column header strip
    grid.menu_titles                       # [(left, right), ...] of File, Edit, View, Insert, ...

What it relies on (Calc 4.1's default look, measured on its screenshots):
  * the cells are white (255) and their grid lines are thin light grey lines (about 199 in the
    middle of a 2-3 px line); the column header strip above them and the row header strip left
    of them are flat light grey (236-248) with darker separators
  * a column edge is a grid line in the cell area that is also a separator in the column header
    strip - so a chart's axis or a border drawn over the cells is never taken for one; a row
    edge is a horizontal grid line across the first columns
  * the sheet isn't scrolled: the first column shown is A and the first row 1 (the executor
    goes to A1 before relying on this, and the sheets it builds fit on the screen)
  * the menu bar is a row of dark word clusters (File ... Help) just under the title bar

Checks after a click (both from a fresh capture):
  active_cell_at(rect)   the thick black frame of the cell cursor is around rect
  selected(rect)         rect is tinted with the selection's blue
"""

import ctypes
import ctypes.wintypes

import numpy as np

GRID_GREY = (170, 235)               # a grid line's darkest pixel
WHITE = 250
MIN_COL, MIN_ROW = 12, 10            # px: narrower columns / lower rows are noise
MIN_GRID_ROWS = 100                  # px of mostly-white rows that make a cell area


def window_rect(hwnd):
    r = ctypes.wintypes.RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right, r.bottom


def capture(hwnd):
    """The window as an RGB numpy array, and its screen offset (left, top)."""
    import pyautogui
    from .screen import virtual_screen_rect
    l, t, r, b = window_rect(hwnd)
    vl, vt, vr, vb = virtual_screen_rect()          # a maximised window reaches 8 px past the screen
    l, t, r, b = max(l, vl), max(t, vt), min(r, vr), min(b, vb)
    img = pyautogui.screenshot(region=(l, t, r - l, b - t))
    return np.asarray(img.convert("RGB")), (l, t)


def col_letters(n):
    """1 -> 'A', 27 -> 'AA'."""
    out = ""
    while n:
        n, r = divmod(n - 1, 26)
        out = chr(65 + r) + out
    return out


def col_number(letters):
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


class Grid:
    """Screen geometry of the visible sheet: column edges xs (A's left edge first), row edges
    ys (row 1's top first), the header strips, and the menu bar titles - all in screen pixels."""

    def __init__(self, xs, ys, header, menu_titles, menu_y):
        self.xs, self.ys = xs, ys
        self.header = header                 # (top, bottom) of the column header strip
        self.menu_titles, self.menu_y = menu_titles, menu_y

    @property
    def columns(self):
        return len(self.xs) - 1

    @property
    def rows(self):
        return len(self.ys) - 1

    def has(self, ref):
        """True if every cell of `ref` ('B3', 'A1:G1') is fully on screen."""
        a, _, b = ref.replace("$", "").partition(":")
        for cell in (a, b or a):
            col, row = split_ref(cell)
            if not (1 <= col <= self.columns and 1 <= row <= self.rows):
                return False
        return True

    def cell_rect(self, ref):
        col, row = split_ref(ref.replace("$", ""))
        return self.xs[col - 1], self.ys[row - 1], self.xs[col], self.ys[row]

    def range_rect(self, ref):
        a, _, b = ref.replace("$", "").partition(":")
        l, t, _, _ = self.cell_rect(a)
        _, _, r, bt = self.cell_rect(b or a)
        return l, t, r, bt

    def column_header_rect(self, letters):
        n = col_number(letters)
        return self.xs[n - 1], self.header[0], self.xs[n], self.header[1]

    def menu_rect(self, index):
        """The index-th menu bar title (0 = File), with some height around its words."""
        l, r = self.menu_titles[index]
        return l, self.menu_y - 7, r, self.menu_y + 7


def split_ref(cell):
    i = next(k for k, ch in enumerate(cell) if ch.isdigit())
    return col_number(cell[:i]), int(cell[i:])


def _runs(mask):
    """(start, end) of each run of True in a 1-D boolean array (end exclusive)."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _line_centres(score, threshold):
    """Positions where score passes threshold, one per cluster of neighbours (the best of it)."""
    out = []
    for s, e in _runs(score >= threshold):
        out.append(int(s + np.argmax(score[s:e])))
    return out


def find_grid(rgb, offset=(0, 0)):
    """The Grid of a Calc window's capture (rgb: H x W x 3), or None. offset: the capture's
    top-left on screen."""
    g = rgb.astype(np.int16).mean(axis=2)
    # grey bands: rows with next to no white and a light mean - the title bar, the column header
    # strip, the sheet tab bar, the status bar (the toolbars and the cells have white in them).
    # The column header strip is the one whose separators line up with grid lines below it.
    grey = ((g >= WHITE).mean(axis=1) < 0.15) & (g.mean(axis=1) > 200)
    bands = [(int(s), int(e)) for s, e in _runs(grey) if e - s >= 6]
    for k, (header_top, header_bottom) in enumerate(bands):
        if not 15 <= header_bottom - header_top <= 45:
            continue
        top = header_bottom + 3
        bottom = next((s for s, e in bands[k + 1:] if e - s >= 15), g.shape[0]) - 2
        if bottom - top < MIN_GRID_ROWS:
            continue
        grid = _grid_below(g, header_top, header_bottom, top, bottom)
        if grid is not None:
            xs, ys = grid
            below_title = bands[0][1] if bands and bands[0][0] == 0 else 0     # the title bar is a grey band too
            menu_y, titles = menu_titles(g[below_title:header_top])
            menu_y = None if menu_y is None else menu_y + below_title
            ox, oy = offset
            return Grid([x + ox for x in xs], [y + oy for y in ys], (header_top + oy, header_bottom + oy),
                        [(a + ox, b + ox) for a, b in titles], menu_y + oy if menu_y is not None else None)
    return None


def _grid_below(g, header_top, header_bottom, top, bottom):
    """(column edges, row edges) of the cells under a header strip candidate, or None."""
    # column edges: thin vertical grey lines in the cell area that are separators in the header
    # strip too. Lines are local minima across x with white (or nearly) a few pixels either side
    cells = g[top:bottom]
    left, right = np.roll(cells, 3, axis=1), np.roll(cells, -3, axis=1)
    vline = (cells <= GRID_GREY[1]) & (cells + 15 <= np.maximum(left, right)) \
        & (cells <= np.minimum(np.roll(cells, 1, axis=1), np.roll(cells, -1, axis=1)))
    vscore = vline.mean(axis=0)
    vscore[:3] = vscore[-3:] = 0
    head = g[header_top:header_bottom]
    mid = head[len(head) // 4: len(head) * 3 // 4]
    hsep = (mid + 10 <= np.maximum(np.roll(mid, 3, axis=1), np.roll(mid, -3, axis=1))).mean(axis=0) >= 0.6
    hsep = np.convolve(hsep.astype(int), [1, 1, 1, 1, 1], mode="same") > 0      # within 2 px
    xs = [x for x in _line_centres(vscore, 0.3) if hsep[x]]
    # column A's left edge is the one with the grey row-number strip on its left and white cells
    # on its right (the window's frame and other lines left of it are dropped)
    white = cells >= WHITE
    first = next((k for k, x in enumerate(xs) if x >= 30
                  and white[:, max(0, x - 25): x - 4].mean() < 0.1 and cells[:, max(0, x - 25): x - 4].mean() > 200
                  and white[:, x + 4: x + 25].mean() > 0.5), None)
    if first is None:
        return None
    xs = _drop_close(xs[first:], MIN_COL)
    if len(xs) < 4:
        return None
    # row edges: horizontal grid lines across the first columns (left of any chart)
    band = g[:, xs[0] + 3: xs[3] - 3]
    up, down = np.roll(band, 3, axis=0), np.roll(band, -3, axis=0)
    hline = (band <= GRID_GREY[1]) & (band + 15 <= np.maximum(up, down)) \
        & (band <= np.minimum(np.roll(band, 1, axis=0), np.roll(band, -1, axis=0)))
    hscore = hline.mean(axis=1)
    hscore[:header_bottom - 2] = 0
    hscore[bottom + 2:] = 0
    ys = _regular_rows(_drop_close(_line_centres(hscore, 0.5), MIN_ROW), header_bottom)
    return (xs, ys) if len(ys) >= 3 else None


def _drop_close(positions, gap):
    out = []
    for p in positions:
        if not out or p - out[-1] >= gap:
            out.append(p)
    return out


def _regular_rows(ys, header_bottom):
    """Row edges from the header strip down, as long as they keep the usual row height (text
    and a chart can add or hide a line; a missing one is filled in)."""
    if len(ys) < 3:
        return ys
    gaps = np.diff(ys)
    step = float(np.median(gaps))
    out = [y for y in ys if y >= header_bottom - 2][:1]
    if not out:
        return []
    k = 1
    while True:
        want = out[-1] + step
        near = [y for y in ys if abs(y - want) <= step * 0.3]
        if near:
            out.append(near[0])
        elif want <= ys[-1] + step * 0.5:
            out.append(int(round(want)))          # a line hidden under something
        else:
            break
        k += 1
    return out


def menu_titles(top_part):
    """(y, [(left, right), ...]) of the menu bar's words (File ... Help): the first band of dark
    text holding 6-12 words in the window's left half (the document's close button sits at the
    far right of the same row and is left out)."""
    dark = top_part < 110
    rows = dark.mean(axis=1)
    for s, e in _runs(rows > 0.003):
        if e - s < 6 or e - s > 24:
            continue
        cols = dark[s:e].any(axis=0)
        words = []
        for a, b in _runs(cols):
            if words and a - words[-1][1] <= 6:       # letters of one word
                words[-1] = (words[-1][0], b)
            else:
                words.append((a, b))
        words = [wd for wd in words if wd[1] < top_part.shape[1] * 0.5]
        if 6 <= len(words) <= 12 and words[0][0] < top_part.shape[1] * 0.1:
            return (s + e) // 2, words
    return None, []


def read_grid(hwnd):
    """The Grid of a Calc window as it is on screen now, or None."""
    rgb, offset = capture(hwnd)
    return find_grid(rgb, offset)


# --- checks after a click --------------------------------------------------------------------

def active_cell_at(rgb, rect, offset=(0, 0)):
    """True if the cell cursor's thick dark frame runs along all four edges of rect (a frozen
    row's line or a neighbour's frame can darken one or two of them)."""
    l, t, r, b = (v - o for v, o in zip(rect, (offset[0], offset[1], offset[0], offset[1])))
    x0, y0 = max(0, l - 2), max(0, t - 2)
    g = rgb[y0: b + 3, x0: r + 3].astype(np.int16).mean(axis=2)       # just around the cell
    l, t, r, b = l - x0, t - y0, r - x0, b - y0
    edges = [g[max(0, y - 2): y + 3, l + 4: r - 4].min(axis=0) for y in (t, b)] \
        + [g[t + 4: b - 4, max(0, x - 2): x + 3].min(axis=1) for x in (l, r)]
    return all((e < 80).mean() > 0.8 for e in edges)


def selected(rgb, rect, offset=(0, 0)):
    """True if most of rect is tinted with the selection's blue (blue clearly above red)."""
    l, t, r, b = (v - o for v, o in zip(rect, (offset[0], offset[1], offset[0], offset[1])))
    part = rgb[t + 2: b - 2, l + 2: r - 2].astype(np.int16)
    blue = (part[..., 2] - part[..., 0] >= 25)
    return bool(blue.mean() > 0.5)
