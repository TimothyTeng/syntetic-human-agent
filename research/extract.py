"""
Taking notes from a web page.

capture_page()   reads the open Chrome page through UI Automation - the accessibility
                 interface screen readers use, so nothing is injected into the page
                 and no debugging port is needed. It returns a plain, JSON-ready
                 PageCapture dict:
                     {url, title, site, accessed, main_text, headings: [{text, level}],
                      tables: [{kind, rows}], seen_intervals, total_words, method}
                 Chrome must be in the foreground (as for browser.find_links()).
parse_capture()  pure: PageCapture -> notes.Source. Splits the text into sections at
                 the headings, drops page furniture (navigation, "References",
                 "[edit]", citation marks, pronunciations) and keeps paragraphs.

Captures are saved to the research folder, so parse_capture() can be re-run and
tested offline (tests/fixtures/pages/*.json have the same format).
"""

import datetime
import re
from urllib.parse import urlsplit

import uiautomation as auto

from controller import browser, ui_elements

from .notes import Para, SectionNote, Source, TableNote
from .textutil import alpha_share, clean_line, load_list, load_map

# UIA ids uiautomation has no names for
HEADING_LEVEL_PROPERTY = 30173        # UIA_HeadingLevelPropertyId
HEADING_LEVEL_NONE = 80050            # HeadingLevel_None; HeadingLevel1..9 = 80051..80059
MAIN_LANDMARK = 80002                 # UIA_MainLandmarkTypeId
MAIN_IDS = ("mw-content-text", "bodyContent", "main-content", "content", "main", "article-body")

MAX_CHARS = 400_000
MIN_MAIN_WORDS = 150                  # a "main" element with less text than this is not the article
MAX_TABLES = 6


# --- Capture (UI Automation) ----------------------------------------------------------------

def capture_page(max_chars=MAX_CHARS):
    """Read the current Chrome page. Returns a PageCapture dict, or None if the page
    exposes no text.

    The text comes from one TextPattern call on the article element (falling back to the
    whole document, then to the visible text elements), and headings / tables from bulk
    FindAll queries. Walking the tree instead would cost one cross-process call per
    element and take tens of seconds on a long article."""
    win = browser.get_chrome_window(timeout=2)
    doc = browser.page_document(win) if win else None
    if doc is None:
        return None
    pattern = ui_elements.text_pattern(doc)
    root, method = _main_root(doc, pattern, max_chars)
    text = ui_elements.range_text(pattern, root, max_chars) if root is not None else ""
    if len(text.split()) < MIN_MAIN_WORDS:                # no article element: the whole page (menus and all)
        root, method = doc, "document"
        try:
            text = pattern.DocumentRange.GetText(max_chars) if pattern else ""
        except Exception:
            text = ""
    if not text.strip():                                  # no TextPattern: the text elements we can see
        nodes = ui_elements.text_nodes(doc, min_words=1)
        text = "\n".join(n["text"] for n in nodes)
        method = "text-nodes"
    if not text.strip():
        return None
    url = normalise_url(browser.get_current_url())
    return {
        "url": url,
        "title": browser.get_page_title() or doc.Name or "",
        "site": site_name(url),
        "accessed": datetime.date.today().isoformat(),
        "main_text": text,
        "headings": _headings(root, pattern),
        **_tables(root, doc, pattern),                    # 'tables' and 'table_lines'
        "seen_intervals": [],                             # filled in by ResearchProject.on_view while reading
        "total_words": len(text.split()),
        "method": method,
    }


def _main_root(doc, pattern, max_chars):
    """The element holding the article: the Main landmark, an <article>, or a well-known
    content id - the first of these with enough text. (None, 'document') if none.

    The order is from most to least reliable; a candidate with fewer than MIN_MAIN_WORDS
    words is a sidebar or an empty wrapper, not the article."""
    candidates = []
    # <main> / role="main" - most sites mark the article this way
    candidates += [("main-landmark", c) for c in ui_elements.find_all_fast(
        doc, properties={auto.PropertyId.LandmarkTypeProperty: MAIN_LANDMARK})]
    candidates += [("article", c) for c in ui_elements.find_all_fast(
        doc, properties={auto.PropertyId.LocalizedControlTypeProperty: "article"})]
    for aid in MAIN_IDS:                                   # the HTML id becomes the AutomationId
        candidates += [(f"id:{aid}", c) for c in ui_elements.find_all_fast(
            doc, properties={auto.PropertyId.AutomationIdProperty: aid})]
    for method, ctrl in candidates:
        if len(ui_elements.range_text(pattern, ctrl, max_chars).split()) >= MIN_MAIN_WORDS:
            return ctrl, method
    return None, "document"


def _headings(root, pattern):
    """[{text, level}] of the headings under root, in page order.

    Chrome exposes <h1>..<h6> as text elements with the UIA HeadingLevel property (30173)
    set, so one FindAll excluding HeadingLevel_None finds them all. Older UIA versions
    lack that property; then headings are found by their localized role name, with the
    level read from the ARIA properties ("level=2")."""
    found = ui_elements.find_all_fast(root, "TextControl",
                                      exclude={HEADING_LEVEL_PROPERTY: HEADING_LEVEL_NONE})
    # double-check the values: keep only elements that report HeadingLevel1..9
    found = [c for c in found if (ui_elements.property_value(c, HEADING_LEVEL_PROPERTY) or 0) > HEADING_LEVEL_NONE]
    if not found:                                          # older UIA: no heading levels, use the role name
        found = ui_elements.find_all_fast(root, properties={auto.PropertyId.LocalizedControlTypeProperty: "heading"})
    out = []
    for ctrl in found:                                     # Name is empty when the heading text is a link
        text = (ctrl.Name or "").strip() or ui_elements.range_text(pattern, ctrl, 300).strip()
        if text:
            out.append({"text": text, "level": _heading_level(ctrl)})
    return out


def _heading_level(ctrl):
    """1-9 for a heading: UIA's HeadingLevel1..9, else ARIA's "level=N", else 2."""
    value = ui_elements.property_value(ctrl, HEADING_LEVEL_PROPERTY) or 0
    if HEADING_LEVEL_NONE < value <= HEADING_LEVEL_NONE + 9:
        return value - HEADING_LEVEL_NONE
    m = re.search(r"level=(\d)", ui_elements.property_value(ctrl, auto.PropertyId.AriaPropertiesProperty) or "")
    return int(m.group(1)) if m else 2


def _tables(root, doc, pattern):
    """{'tables': small tables of the page as {kind, rows} - 'infobox' (a 2-column fact box
    near the top) or 'data'; navigation boxes and big tables are skipped - and
    'table_lines': the text lines of every table, kept or not, so parse_capture can keep
    table cells out of the paragraphs.}

    Every table's lines are recorded, even for tables that are skipped: the page text from
    TextPattern includes the cells as separate lines, and without this they would turn up
    as (odd, short) paragraphs."""
    out, lines = [], set()
    try:
        page = doc.BoundingRectangle                       # the visible part of the page (one screen)
        page_top, view_h = page.top, max(1, page.height())
    except Exception:
        page_top, view_h = 0, 1000
    for table in ui_elements.find_all_fast(root, "TableControl"):
        lines.update(ln.strip() for ln in ui_elements.range_text(pattern, table, 50_000).split("\n") if ln.strip())
        if len(out) >= MAX_TABLES:
            continue                                       # still collecting the lines of the rest
        cls = (table.ClassName or "").lower()
        if "navbox" in cls or "sidebar" in cls or "metadata" in cls:
            continue                                       # Wikipedia's link boxes and notices, not data
        try:
            grid = table.GetGridPattern()
            n_rows, n_cols = grid.RowCount, grid.ColumnCount
        except Exception:
            continue                                       # a layout table without a grid
        if not (2 <= n_rows <= 40 and 2 <= n_cols <= 5):
            continue                                       # too small to matter or too big to retype
        rows = []
        for r in range(min(n_rows, 40)):
            row = []
            for c in range(n_cols):
                try:
                    cell = grid.GetItem(r, c)
                except Exception:
                    cell = None
                row.append(_cell_text(cell, pattern))
            rows.append(row)
        top = table.BoundingRectangle.top - page_top
        # an infobox: marked as one, or a 2-column (label, value) table near the top of the page
        infobox = "infobox" in cls or (n_cols == 2 and top < 1.5 * view_h)
        out.append({"kind": "infobox" if infobox else "data", "rows": rows})
    return {"tables": out, "table_lines": sorted(lines)}


def _cell_text(cell, pattern):
    """One table cell's text on one line, without image placeholders ('' for no cell)."""
    if cell is None:
        return ""
    text = ui_elements.range_text(pattern, cell, 500) or cell.Name or ""
    return " ".join(text.replace("\ufffc", " ").split())


# --- URLs and site names -----------------------------------------------------------------------

_SITES = sorted(load_map("site_names.tsv"), key=lambda kv: -len(kv[0]))


def normalise_url(url):
    """Full URL from the address bar text (Chrome hides 'https://'), without #fragment."""
    url = (url or "").strip()
    if url and "://" not in url:
        url = "https://" + url
    return url.split("#")[0]


def site_name(url):
    """'Wikipedia', 'NASA', ... for a URL (the host name when the site is unknown)."""
    host = (urlsplit(url).hostname or "").lower()
    for suffix, name in _SITES:
        if host == suffix or host.endswith("." + suffix):
            return name
    host = re.sub(r"^www\d?\.", "", host)
    return host or "Unknown site"


def clean_title(title, site=""):
    """Page title without the site suffix: 'Apollo program - Wikipedia' -> 'Apollo program'."""
    title = re.sub(r"\s+", " ", (title or "").replace(" - Google Chrome", "")).strip()
    parts = [p.strip() for p in re.split(r"\s+[|–—-]\s+", title) if p.strip()]
    site_l = site.lower()
    while len(parts) > 1 and (parts[-1].lower() in site_l or site_l in parts[-1].lower()
                              or len(parts[-1].split()) <= 2):
        parts.pop()
    if not parts:
        return title
    return parts[0] if len(parts) > 1 and "|" in title else " - ".join(parts)


# --- Parsing (pure) ------------------------------------------------------------------------------

BOILERPLATE_HEADINGS = frozenset(load_list("boilerplate_headings.txt"))

_JUNK_LINE = re.compile(
    r"^\[?edit\]?$|^(main|further) (article|information)s?:|^see also:|^this article is about|"
    r"^not to be confused|^for other uses|^jump to|^from wikipedia|^retrieved \d|archived from|\bisbn\b|"
    r"\bdoi:|^coordinates:|\d+°\s?\d+′|^(this|the) (page|article|section) (was|is|needs|may)|"
    r"^\"?[a-z ]+\" redirects here|^cite this|^share\b|^print\b|^our editors|^subscribe|^sign up|"
    r"^advertisement$|^image:|^photo:|^credit:|^listen\b|^view all|^read next|^written by|^fact-checked by|"
    r"^last updated|^updated:|^published:|^\d+ min(ute)? read",
    re.IGNORECASE)
MIN_PARA_WORDS = 6


def is_boilerplate_heading(heading):
    """True for a section that is page furniture ("References", "See also", ...: the list in
    data/boilerplate_headings.txt), with or without an "[edit]" suffix."""
    h = re.sub(r"\[\s*edit\s*\]$", "", heading.strip(), flags=re.IGNORECASE).strip().lower()
    return h in BOILERPLATE_HEADINGS


def _locate_headings(text, headings):
    """[(start, end, heading, level)] where each heading's line is in the text, searched in
    page order. A heading matches a line that is the heading itself, possibly followed by a
    short suffix such as "[edit]" - not a paragraph that merely starts with the same word.
    Headings that can't be found are dropped."""
    lines, starts, pos = text.split("\n"), [], 0
    for line in lines:                                     # character offset of each line
        starts.append(pos)
        pos += len(line) + 1
    out, li = [], 0
    for h in headings:
        name = " ".join(h["text"].split())
        if not name:
            continue
        # search on from the previous heading's line, so a repeated name (e.g. in the table
        # of contents) doesn't pull a later heading back up the page
        for k in range(li, len(lines)):
            line = " ".join(lines[k].split())
            if line == name or (line.startswith(name) and len(line) <= len(name) + 8
                                and not line[len(name):len(name) + 1].isalnum()):
                out.append((starts[k], starts[k] + len(lines[k]), name, int(h.get("level", 2))))
                li = k + 1
                break
    return out


def _guess_headings(text):
    """Headings for pages that don't mark them: a short capitalised line without final
    punctuation, followed by a long line. Image captions (after Chrome's U+FFFC image
    placeholder), "Main article: ..." lines and lines with commas are not headings.
    Every guessed heading gets level 2."""
    out, pos, prev = [], 0, ""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        start, s = pos, line.strip()
        pos += len(line) + 1
        after_image, prev = "￼" in prev, s or prev          # prev: the last non-empty line
        nxt = next((ln.strip() for ln in lines[i + 1:] if ln.strip()), "")
        # short, capitalised, no sentence punctuation, mostly letters, and a real paragraph after it
        if (1 <= len(s.split()) <= 8 and s[:1].isupper() and not s.endswith((".", ":", ",", ";", "?", "!"))
                and "," not in s and ":" not in s and not after_image and not _JUNK_LINE.search(s)
                and alpha_share(s) > 0.7 and len(nxt.split()) >= 15):
            out.append((start, start + len(line), s, 2))
    return out


def parse_capture(cap, query=""):
    """PageCapture dict -> notes.Source (sections of clean paragraphs, small tables).

    The text is split into sections at the headings (those the page marks, else guessed
    ones). Boilerplate sections ("References", "See also", ...) are dropped together with
    their subsections. Each paragraph keeps its position on the page (0-1), so the parts
    the reader scrolled past (seen_intervals) give words_read."""
    text = cap.get("main_text", "").replace("\r\n", "\n").replace("\r", "\n")
    total = max(1, len(text))
    site = cap.get("site") or site_name(cap.get("url", ""))
    title = clean_title(cap.get("title", ""), site)

    heads = _locate_headings(text, cap.get("headings") or []) or _guess_headings(text)
    table_lines = set(cap.get("table_lines", []))
    bounds = [(0, 0, "", 1)] + heads                       # a pseudo-heading for the text before the first one
    sections, skip_below = [], None                        # skip_below: level of the boilerplate section being skipped
    for k, (start, end, heading, level) in enumerate(bounds):
        body_end = bounds[k + 1][0] if k + 1 < len(bounds) else len(text)   # the section runs to the next heading
        if skip_below is not None and level > skip_below:
            continue                                       # inside a skipped section
        skip_below = None
        if heading and is_boilerplate_heading(heading):
            skip_below = level
            continue
        if level <= 1 or heading.strip().lower() == title.lower():
            heading, level = "", 1                         # the page title: what follows is the lead
        paras = _paragraphs(text, end, body_end, total, heading, table_lines)
        if not paras:
            continue
        if sections and sections[-1].heading == heading:   # e.g. lead text before and after the title heading
            sections[-1].paras += paras
        else:
            sections.append(SectionNote(heading=_clean_heading(heading), level=max(1, level), paras=paras))

    tables = [TableNote(kind=t.get("kind", "data"), rows=rows)
              for t in cap.get("tables", []) if (rows := _clean_rows(t.get("rows", [])))]
    src = Source(url=cap.get("url", ""), title=title or site, site=site,
                 accessed=cap.get("accessed") or datetime.date.today().isoformat(), query=query,
                 sections=sections, tables=tables, seen_intervals=[list(iv) for iv in cap.get("seen_intervals", [])])
    # words of the paragraphs inside the scrolled-through parts of the page
    src.words_read = sum(len(p.text.split()) for s in sections for p in s.paras if src.seen(p.pos))
    return src


def _clean_heading(heading):
    """The heading without a trailing "[edit]" link."""
    return re.sub(r"\[\s*edit\s*\]$", "", heading, flags=re.IGNORECASE).strip()


def _paragraphs(text, start, end, total, heading, table_lines=frozenset()):
    """[Para] from the lines of text[start:end]: clean lines of real prose, each with its
    position on the page (line start / total). Image captions, table cells, junk lines
    (see _JUNK_LINE), the heading itself and short or mostly non-letter lines are dropped."""
    out, pos, after_image = [], start, False
    for line in text[start:end].split("\n"):
        line_start = pos
        pos += len(line) + 1
        raw = line.strip()
        if raw.strip("\ufffc ") == "" and "\ufffc" in raw:
            after_image = True                     # Chrome's placeholder for an image: a caption follows
            continue
        caption, after_image = after_image, False          # only the one line after the image
        if not raw or caption or raw in table_lines or _JUNK_LINE.search(raw) or raw == heading:
            continue
        para = clean_line(raw)
        if len(para.split()) < MIN_PARA_WORDS or alpha_share(para) < 0.6:
            continue
        out.append(Para(text=para, pos=round(line_start / total, 4)))
    return out


def _clean_rows(rows):
    """Table rows with clean cell text; empty rows and full-width (spanning) rows dropped."""
    out = []
    for row in rows:
        cells = [clean_line(c or "") for c in row]
        if not any(cells) or (len(set(cells)) == 1 and len(cells) > 1):
            continue
        out.append(cells)
    return out
