"""
Locate buttons, menus, text boxes and links using Windows UI Automation.

UI Automation (UIA) is the accessibility API screen readers use. Most apps
(Office ribbon, File dialogs, Notepad, Chrome web pages) publish every control
with a Name, a ControlType and an on-screen BoundingRectangle - so you can find
"the Save button" by name instead of by hard-coded coordinates.

These functions only *locate* things and return coordinates / control objects.
Actually clicking is done through mouse.py so the cursor visibly travels there.

Common control_type strings:
  'ButtonControl', 'EditControl', 'HyperlinkControl', 'MenuItemControl',
  'TabItemControl', 'ListItemControl', 'TextControl', 'DocumentControl',
  'CheckBoxControl', 'ComboBoxControl', 'WindowControl', 'PaneControl'

Tip: run list_elements(get_window('OpenOffice Writer')) to print what's available.
"""

import time

import uiautomation as auto

from . import config, mouse


def _control_type_id(control_type):
    """Convert 'ButtonControl' -> auto.ControlType.ButtonControl (int)."""
    if control_type is None or isinstance(control_type, int):
        return control_type
    return getattr(auto.ControlType, control_type)


def get_window(title_substring, timeout=config.DEFAULT_TIMEOUT):
    """
    Find a top-level window whose title contains `title_substring`.

    Returns a uiautomation WindowControl, or None if not found in `timeout` s.
    Example: get_window('Google Chrome'), get_window('OpenOffice Writer')
    """
    win = auto.WindowControl(searchDepth=1, SubName=title_substring)
    if win.Exists(maxSearchSeconds=timeout, searchIntervalSeconds=config.POLL_INTERVAL):
        return win
    return None


def get_foreground_window():
    """Return the UIA control for the window that currently has focus."""
    return auto.GetForegroundControl()


def find_element(window, name, control_type=None, partial=False,
                 timeout=5, search_depth=0xFFFFFFFF):
    """
    Find a single element inside `window` by its accessible name.

    window:       control returned by get_window() (or None = whole desktop).
    name:         the element's Name, e.g. 'Save', 'File Tab', 'Don't Save'.
    control_type: optional filter, e.g. 'ButtonControl'.
    partial:      True = Name only needs to *contain* `name`.
    timeout:      seconds to keep searching.

    Returns the control or None. The result is a generic uiautomation Control;
    to use a pattern call e.g. ctrl.GetPattern(auto.PatternId.ValuePattern).
    """
    props = {"SubName" if partial else "Name": name}
    if control_type:
        props["ControlType"] = _control_type_id(control_type)
    root = window if window is not None else auto.GetRootControl()
    ctrl = auto.Control(searchFromControl=root, searchDepth=search_depth, **props)
    if ctrl.Exists(maxSearchSeconds=timeout, searchIntervalSeconds=config.POLL_INTERVAL):
        return ctrl
    return None


def find_elements(window, control_type=None, name_contains=None,
                  visible_only=True, max_depth=40):
    """
    Return a list of every element under `window` matching the filters.

    control_type:  e.g. 'HyperlinkControl' to get all links.
    name_contains: case-insensitive substring filter on Name.
    visible_only:  skip elements that are off-screen / zero size.
    max_depth:     how deep to walk the tree (deep trees are slow to walk).
    """
    type_id = _control_type_id(control_type)
    needle = name_contains.lower() if name_contains else None
    results = []
    for ctrl, _depth in auto.WalkControl(window, includeTop=False, maxDepth=max_depth):
        if type_id is not None and ctrl.ControlType != type_id:
            continue
        if needle and needle not in (ctrl.Name or "").lower():
            continue
        if visible_only and not is_visible(ctrl):
            continue
        results.append(ctrl)
    return results


def is_visible(ctrl):
    """True if the element is on-screen and has a non-zero rectangle."""
    try:
        if ctrl.IsOffscreen:
            return False
        r = ctrl.BoundingRectangle
        return r.width() > 0 and r.height() > 0
    except Exception:
        return False


def element_rect(ctrl):
    """Return the element's screen rectangle as (left, top, right, bottom)."""
    r = ctrl.BoundingRectangle
    return r.left, r.top, r.right, r.bottom


def element_center(ctrl):
    """Return the element's centre point (x, y) in screen pixels."""
    left, top, right, bottom = element_rect(ctrl)
    return (left + right) // 2, (top + bottom) // 2


def click_element(ctrl, duration=0.0, button="left"):
    """Move the mouse to the centre of the element and click it."""
    x, y = element_center(ctrl)
    mouse.click(x, y, button=button, duration=duration)


def wait_for_element(window, name, control_type=None, partial=False,
                     timeout=config.DEFAULT_TIMEOUT):
    """Like find_element but with a longer default timeout - for slow UIs."""
    return find_element(window, name, control_type, partial, timeout=timeout)


def list_elements(window, control_type=None, max_depth=40, visible_only=True):
    """
    DEBUG helper: print Name / type / rectangle of elements in a window, so you
    can discover the exact names to pass to find_element().
    Returns the list of (name, type, rect) tuples.
    """
    rows = []
    for ctrl in find_elements(window, control_type, visible_only=visible_only, max_depth=max_depth):
        row = (ctrl.Name, ctrl.ControlTypeName, element_rect(ctrl))
        rows.append(row)
        print(f"{row[1]:<22} {row[0]!r:<50} {row[2]}")
    return rows


def list_buttons(window):
    """DEBUG helper: print every visible button in a window."""
    return list_elements(window, "ButtonControl")


# --- Text and word positions -------------------------------------------------
# Used to find where words are on screen, e.g. so the behaviour layer can follow a
# line of text with the cursor while "reading".

SINGLE_LINE_MAX_PX = 40   # a text element no taller than this is treated as one line


def text_pattern(ctrl):
    """The element's UIA TextPattern (documents, rich text), or None."""
    try:
        return ctrl.GetTextPattern()
    except Exception:
        return None


def text_nodes(root, area=None, min_words=3, max_depth=40):
    """
    Visible text elements (TextControl) under `root` with at least `min_words` words,
    top to bottom: [{'text', 'rect', 'control'}]. With `area`, only elements lying
    entirely inside that rect are returned.
    """
    nodes = []
    for ctrl in find_elements(root, "TextControl", max_depth=max_depth):
        text = (ctrl.Name or "").strip()
        if len(text.split()) < min_words:
            continue
        rect = element_rect(ctrl)
        if area and not (area[0] <= rect[0] and rect[2] <= area[2] and area[1] <= rect[1] and rect[3] <= area[3]):
            continue
        nodes.append({"text": text, "rect": rect, "control": ctrl})
    return sorted(nodes, key=lambda n: (n["rect"][1], n["rect"][0]))


def node_words(node, pattern=None, max_words=60):
    """
    Words of a text element with their screen rects: [{'text', 'rect'}] in reading order.

    pattern: TextPattern of the enclosing document (e.g. the Chrome page). Words are then
             read from the element's own text range, one Word unit at a time, which
             handles wrapped multi-line paragraphs exactly.
    Without a pattern (or if that fails) a single-line element is split proportionally
    by character count; multi-line elements then return [].
    """
    words = _pattern_words(pattern, node, max_words) if pattern is not None else []
    if words:
        return words
    if node["rect"][3] - node["rect"][1] <= SINGLE_LINE_MAX_PX:
        return split_line_proportional(node["text"], node["rect"])[:max_words]
    return []


def _pattern_words(pattern, node, max_words):
    try:
        # uiautomation's TextPattern.RangeFromChild passes the wrong object; call the COM method directly
        whole = auto.TextRange(pattern.pattern.RangeFromChild(node["control"].Element))
        word = whole.Clone()
        word.ExpandToEnclosingUnit(auto.TextUnit.Word, waitTime=0)
        l, t, r, b = node["rect"]
        out = []
        while len(out) < max_words:
            if word.CompareEndpoints(auto.TextPatternRangeEndpoint.Start, whole, auto.TextPatternRangeEndpoint.End) >= 0:
                break
            text = word.GetText(100).strip()
            rects = word.GetBoundingRectangles()
            if text and rects:
                wr = rects[0]
                if not (l - 5 <= wr.xcenter() <= r + 5 and t - 5 <= wr.ycenter() <= b + 5):
                    break                             # ran past the element
                out.append({"text": text, "rect": (wr.left, wr.top, wr.right, wr.bottom)})
            if word.Move(auto.TextUnit.Word, 1, waitTime=0) == 0:
                break
        return out
    except Exception:
        return []


def split_line_proportional(text, rect):
    """Approximate word rects on one line of text: x positions in proportion to characters."""
    left, top, right, bottom = rect
    words = text.split()
    if not words:
        return []
    per_char = (right - left) / max(len(" ".join(words)), 1)
    out, pos = [], 0
    for w in words:
        x0 = left + pos * per_char
        out.append({"text": w, "rect": (int(x0), top, int(x0 + len(w) * per_char), bottom)})
        pos += len(w) + 1
    return out


def group_lines(words):
    """Group word dicts into visual lines (top to bottom, each left to right)."""
    lines = []
    for w in sorted(words, key=lambda w: ((w["rect"][1] + w["rect"][3]) / 2, w["rect"][0])):
        cy = (w["rect"][1] + w["rect"][3]) / 2
        h = w["rect"][3] - w["rect"][1]
        if lines and abs(cy - lines[-1]["cy"]) < 0.5 * max(h, 1):
            lines[-1]["words"].append(w)
        else:
            lines.append({"cy": cy, "words": [w]})
    return [sorted(line["words"], key=lambda w: w["rect"][0]) for line in lines]


# --- Bulk queries and text ranges --------------------------------------------
# Walking a long web page element by element costs one cross-process call per element
# and property (tens of seconds on a long article). These ask UIA for every match in
# one call instead.

TREE_SCOPE_DESCENDANTS = 4


def _uia():
    """The raw IUIAutomation COM object behind uiautomation (it has no wrapper for FindAll
    with conditions)."""
    return auto.uiautomation._AutomationClient.instance().IUIAutomation


def find_all_fast(root, control_type=None, properties=None, exclude=None):
    """
    All descendants of `root` (in document order) whose control type is `control_type`
    ('TextControl' or a ControlType id) and whose properties equal `properties`
    ({PropertyId: value}); `exclude` ({PropertyId: value}) drops elements having one of
    those values. One IUIAutomationElement::FindAll call. Returns [] on any failure.
    """
    try:
        uia = _uia()
        conds = []
        if control_type is not None:
            conds.append(uia.CreatePropertyCondition(auto.PropertyId.ControlTypeProperty,
                                                     _control_type_id(control_type)))
        for pid, value in (properties or {}).items():
            conds.append(uia.CreatePropertyCondition(pid, value))
        for pid, value in (exclude or {}).items():
            conds.append(uia.CreateNotCondition(uia.CreatePropertyCondition(pid, value)))
        cond = conds[0] if conds else uia.CreateTrueCondition()
        for extra in conds[1:]:
            cond = uia.CreateAndCondition(cond, extra)
        found = root.Element.FindAll(TREE_SCOPE_DESCENDANTS, cond)
        out = []
        for i in range(found.Length if found else 0):
            ctrl = auto.Control.CreateControlFromElement(found.GetElement(i))
            if ctrl is not None:
                out.append(ctrl)
        return out
    except Exception:
        return []


def property_value(ctrl, property_id, default=None):
    """A UIA property by id (also ones uiautomation has no wrapper for, e.g. 30173 HeadingLevel)."""
    try:
        return ctrl.Element.GetCurrentPropertyValue(property_id)
    except Exception:
        return default


def range_text(pattern, ctrl, max_chars=100_000):
    """Text of `ctrl` read through the enclosing document's TextPattern ('' on failure).
    Unlike ctrl.Name this includes text inside links and nested elements."""
    try:
        # uiautomation's TextPattern.RangeFromChild passes the wrong object; call the COM method directly
        return auto.TextRange(pattern.pattern.RangeFromChild(ctrl.Element)).GetText(max_chars) or ""
    except Exception:
        return ""
