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

Tip: run list_elements(get_window('Word')) to print what's available.
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
    Example: get_window('Google Chrome'), get_window('Word')
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
