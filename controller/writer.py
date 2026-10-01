"""
OpenOffice Writer: find toolbar controls and read the document back, through UI Automation.

Like the rest of controller/, nothing here moves the mouse or presses keys; the
behaviour layer does that (research/writer_exec.py). These functions locate things and
check the result of an edit.

What Writer (Apache OpenOffice 4.1, IAccessible2 seen through UIA's MSAA proxy) exposes:
  * the document is a DocumentControl whose children are the paragraphs ON SCREEN (only
    the visible part of the document is in the tree); a paragraph's LegacyIAccessible
    Value is its text, and the paragraph with the caret has the FOCUSED state. Tables are
    TableControl > DataItemControl (cell 'A1', ...) > paragraph. The TextPattern exists
    but its calls fail.
  * the Formatting toolbar's 'Apply Style' and 'Font Size' boxes show the paragraph style
    and font size at the caret; a box with the keyboard focus has the FOCUSED state.
    GetFocusedControl() only ever reports the Writer window, so focus is read from those
    states instead. After a toolbar button is clicked, Writer goes on marking that button
    FOCUSED (and no paragraph) while keys still reach the page - until the page is
    clicked again.
  * toggle buttons (Bold, Italic, Centred, Bullets On/Off) do NOT report whether they are
    on - their states never change - so bold / italic / alignment can't be read back.

    caret_paragraph_style(win)    'Heading 1', 'Text body', ... of the paragraph with the caret
    caret_font_size(win)          12.0, ... at the caret (None if unknown)
    caret_in_table(doc)           whether the caret is inside a table
    caret_paragraph_text(doc)     text of the paragraph with the caret
    document_text(doc)            the visible text (paragraphs end in '\\r', cells in '\\x07')
    normalise(text)               text with AutoCorrect's typing changes undone, for comparisons
    toolbar_button / style_box / font_size_box / menu / menu_item / find_dialog
"""

import ctypes
import functools
import re
import time

import uiautomation as auto

from . import apps, ui_elements

WINDOW_CLASS = "SALFRAME"                     # every OpenOffice main window (Writer, Calc, Start Center)
DIALOG_CLASS = "SALSUBFRAME"                  # OpenOffice dialogs (Insert Table, Insert Break, ...)
TITLE_SUFFIX = "OpenOffice Writer"            # "Untitled 1 - OpenOffice Writer", "paper.odt - OpenOffice Writer"

STATE_PRESSED, STATE_FOCUSED, STATE_CHECKED = 0x8, 0x4, 0x10       # MSAA state bits


def _tolerant(find):
    """UIA calls into OpenOffice fail now and then while it redraws (COMError: element not
    available). Lookups retry, then report 'not found' (None) rather than stopping a run."""

    @functools.wraps(find)
    def wrapper(*args, **kw):
        """Call the lookup; on an exception wait a moment and try again (3 tries), then None."""
        for attempt in range(3):
            try:
                return find(*args, **kw)
            except Exception:
                if attempt == 2:
                    return None
                time.sleep(0.2)               # let Writer finish redrawing
    return wrapper


def _state(ctrl):
    """MSAA state bits of a control (0 if unknown)."""
    try:
        return int(ctrl.GetLegacyIAccessiblePattern().State)
    except Exception:
        return 0


def _value(ctrl):
    """MSAA value of a control ('' if none)."""
    try:
        return ctrl.GetLegacyIAccessiblePattern().Value or ""
    except Exception:
        return ""


# --- Windows --------------------------------------------------------------------------------

def is_writer_window(hwnd):
    """True for a Writer document window (not Calc, not the Start Center)."""
    return apps.window_class(hwnd) == WINDOW_CLASS and apps.window_title(hwnd).endswith(TITLE_SUFFIX)


def writer_windows():
    """Handles of every open Writer document window."""
    return [h for h in apps.top_level_windows(WINDOW_CLASS) if is_writer_window(h)]


def document_control(win, timeout=5):
    """The editing surface (DocumentControl) of a Writer window, or None. Re-find it after a
    save: the element is replaced when the document gets its file name."""
    for attempt in range(3):                  # tree walks fail now and then while Writer redraws
        try:
            doc = auto.DocumentControl(searchFromControl=win, searchDepth=6)
            return doc if doc.Exists(maxSearchSeconds=timeout, searchIntervalSeconds=0.25) else None
        except Exception:
            time.sleep(0.5)
    return None


def foreground_in_writer(win):
    """True if the window in front belongs to this Writer (the document window or one of
    its dialogs, e.g. Insert Table)."""
    user32 = ctypes.windll.user32
    fg = apps.foreground_top_level()
    if not fg:
        return False
    a, b = ctypes.c_ulong(), ctypes.c_ulong()          # process ids
    user32.GetWindowThreadProcessId(fg, ctypes.byref(a))
    user32.GetWindowThreadProcessId(win.NativeWindowHandle, ctypes.byref(b))
    return a.value == b.value                          # same process: one of OpenOffice's windows


# --- Focus and the caret --------------------------------------------------------------------

def _paragraphs(doc):
    """(paragraph control, cell or None) for every paragraph on screen, in document order."""
    out = []
    for child in doc.GetChildren():
        if child.ControlType == auto.ControlType.TableControl:
            for cell in child.GetChildren():
                out += [(p, cell) for p in cell.GetChildren()]
        else:
            out.append((child, None))
    return out


@_tolerant
def _caret_paragraph(doc):
    """(paragraph, cell or None) of the paragraph with the caret, or (None, None) if no
    paragraph has the focus (a toolbar box or a dialog has it, or it isn't on screen)."""
    for para, cell in _paragraphs(doc):
        if is_focused(para):
            return para, cell
    return None, None


def _focused_box(win):
    """The Formatting toolbar box (Apply Style, Font Name, Font Size) with the keyboard
    focus, or None."""
    bar = formatting_toolbar(win)
    if bar is None:
        return None
    for edit in ui_elements.find_all_fast(bar, "EditControl"):
        if is_focused(edit):
            return edit
    return None


@_tolerant
def _focused_toolbar_button(win):
    """A Formatting / Standard toolbar button marked FOCUSED (it was clicked), or None."""
    for name in ("Formatting", "Standard"):
        bar = auto.ToolBarControl(searchFromControl=win, Name=name, searchDepth=4)
        if not bar.Exists(0, 0):
            continue
        for ctrl in bar.GetChildren():
            if ctrl.ControlType in (auto.ControlType.ButtonControl, auto.ControlType.SplitButtonControl)                     and is_focused(ctrl):
                return ctrl
    return None


def focus_kind(win, doc=None):
    """Where keyboard input would go: 'document' (the page of this Writer window),
    'toolbar' (a toolbar button was clicked: Writer reports it focused, but typing still
    reaches the page), 'field' (a toolbar box such as Apply Style or Font Size), 'other'
    (Writer is in front but none of these has the focus - a menu, the sidebar, or the
    caret's paragraph isn't marked yet) or 'elsewhere' (another window, or one of Writer's
    dialogs, is in front)."""
    try:
        if apps.foreground_top_level() != win.NativeWindowHandle:
            return "elsewhere"
    except Exception:
        return "elsewhere"
    if _focused_box(win) is not None:
        return "field"
    doc = doc or document_control(win, timeout=1)
    if doc is not None and (_caret_paragraph(doc) or (None,))[0] is not None:
        return "document"
    if _focused_toolbar_button(win) is not None:
        return "toolbar"
    return "other"


def document_has_focus(win, doc=None):
    """True if keyboard input goes into the document of this Writer window (its page has
    the focus, or a toolbar button was just clicked)."""
    return focus_kind(win, doc) in ("document", "toolbar")


def caret_in_table(doc):
    """True if the caret is in a table cell (None if it can't be told)."""
    found = _caret_paragraph(doc)
    if found is None or found[0] is None:
        return None
    return found[1] is not None


def caret_paragraph_text(doc, assume_end=False):
    """Text of the paragraph with the caret (None if unknown). assume_end: when Writer
    doesn't say where the caret is (after a toolbar click), the last paragraph on screen -
    where the caret is when writing at the end of the document."""
    found = _caret_paragraph(doc)
    if found and found[0] is not None:
        return _value(found[0])
    if assume_end:
        paras = _tolerant(_paragraphs)(doc)
        if paras and paras[-1][1] is None:
            return _value(paras[-1][0])
    return None


@_tolerant
def table_count(doc):
    """Tables on screen (0 if none or unknown)."""
    return len(ui_elements.find_all_fast(doc, "TableControl"))


@_tolerant
def document_text(doc):
    """Text of the paragraphs on screen (Writer only exposes the visible part), or None.
    Paragraphs end in '\\r', table cells in '\\x07' (normalise() turns both into spaces)."""
    parts = []
    for para, cell in _paragraphs(doc):
        parts.append(_value(para) + ("\x07" if cell is not None else "\r"))
    return "".join(parts)


_TYPING_CHANGES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                                 "–": "-", "—": "-", "…": "...", " ": " ",
                                 "•": " ", "": " ", "": " "})     # bullet glyphs


def normalise(text):
    """Text as typed: AutoCorrect's smart quotes / dashes undone, cell markers, paragraph
    marks, line and page breaks turned into spaces, whitespace collapsed."""
    text = (text or "").translate(_TYPING_CHANGES)
    text = re.sub(r"[\r\n\x07\x0b\x0c\t]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def pending_completion(doc, typed):
    """Writer's word completion shows the rest of a long word it has seen before as
    selected text after the caret, and Enter accepts it. After typing `typed`: the number
    of suggested characters showing after it (0 if none, or if it can't be told)."""
    tail = re.search(r"[A-Za-z]{3,}$", typed or "")       # suggestions only follow a word being typed
    para = caret_paragraph_text(doc, assume_end=True) if tail else None
    if not para or para.endswith(tail.group(0)):
        return 0
    m = re.search(re.escape(tail.group(0)) + r"([a-z]{1,30})$", para)
    return len(m.group(1)) if m else 0


# --- Toolbars, menus and dialogs ------------------------------------------------------------

@_tolerant
def formatting_toolbar(win):
    """The Formatting toolbar (Apply Style, Font Size, Bold, ...), or None if it's hidden."""
    bar = auto.ToolBarControl(searchFromControl=win, Name="Formatting", searchDepth=4)
    return bar if bar.Exists(0.5, 0.1) and ui_elements.is_visible(bar) else None


@_tolerant
def toolbar_button(win, name, toolbar="Formatting", timeout=1):
    """A visible button on a toolbar ('Bold', 'Centred', 'Bullets On/Off' on Formatting;
    'Save' on Standard), or None. name may be a regex ('^Cent(red|ered)$') - the labels
    follow the UI language (en-GB 'Centred', en-US 'Centered'). Searched within the
    toolbar so the sidebar's copies of the same buttons aren't picked."""
    bar = auto.ToolBarControl(searchFromControl=win, Name=toolbar, searchDepth=4)
    if not bar.Exists(timeout, 0.1):
        return None
    for ctrl in bar.GetChildren():
        if ctrl.ControlType in (auto.ControlType.ButtonControl, auto.ControlType.SplitButtonControl) \
                and re.fullmatch(name, ctrl.Name or "") and ui_elements.is_visible(ctrl):
            return ctrl
    return None


@_tolerant
def _toolbar_box(win, name):
    """The edit field of a Formatting toolbar box, or None."""
    bar = formatting_toolbar(win)
    if bar is None:
        return None
    edit = auto.EditControl(searchFromControl=bar, Name=name, searchDepth=3)
    return edit if edit.Exists(0.5, 0.1) and ui_elements.is_visible(edit) else None


def style_box(win):
    """The 'Apply Style' box (shows the paragraph style at the caret), or None."""
    return _toolbar_box(win, "Apply Style")


def font_size_box(win):
    """The 'Font Size' box (shows the font size at the caret), or None."""
    return _toolbar_box(win, "Font Size")


def caret_paragraph_style(win):
    """Style name of the paragraph holding the caret, from the Apply Style box (None if
    the box isn't shown)."""
    box = style_box(win)
    try:
        return box.GetValuePattern().Value or None if box is not None else None
    except Exception:
        return None


def caret_font_size(win):
    """Font size (points) at the caret, from the Font Size box (None if unknown)."""
    box = font_size_box(win)
    try:
        return float(box.GetValuePattern().Value.replace(",", ".").replace("pt", "").strip())
    except Exception:
        return None


@_tolerant
def menu(win, name):
    """A menu on the menu bar ('Insert', 'Format', 'Table'), or None."""
    item = auto.MenuItemControl(searchFromControl=win, Name=name, searchDepth=3)
    return item if item.Exists(0.5, 0.1) and ui_elements.is_visible(item) else None


@_tolerant
def menu_item(parent, name, timeout=1.5):
    """An item of an open menu ('Table...', 'Manual Break...', 'Default Formatting'), or
    None. The items are children of the menu-bar item that opened them. name is a regex."""
    deadline = time.time() + timeout
    while True:
        for item in parent.GetChildren():
            if item.ControlType == auto.ControlType.MenuItemControl and re.fullmatch(name, item.Name or "") \
                    and ui_elements.is_visible(item):
                return item
        if time.time() >= deadline:
            return None
        time.sleep(0.2)


def find_dialog(title, timeout=3):
    """A Writer dialog ('Insert Table', 'Insert Break') once it is the window in front, as a
    UIA control, or None. title is a regex matched against the whole window title."""
    deadline = time.time() + timeout
    while True:
        fg = apps.foreground_top_level()
        if fg and apps.window_class(fg) == DIALOG_CLASS and re.fullmatch(title, apps.window_title(fg)):
            ctrl = apps.control_for(fg)
            if ctrl is not None:
                return ctrl
        if time.time() >= deadline:
            return None
        time.sleep(0.2)


@_tolerant
def dialog_control(dialog, name, control_type="EditControl"):
    """A control of a dialog by name ('Columns', 'Rows', 'Heading', 'Page break'), or None."""
    ctrl = getattr(auto, control_type)(searchFromControl=dialog, RegexName=name, searchDepth=4)
    return ctrl if ctrl.Exists(0.5, 0.1) else None


def is_focused(ctrl):
    """True if a control (a toolbar box, a paragraph) has the keyboard focus (MSAA FOCUSED)."""
    return bool(_state(ctrl) & STATE_FOCUSED)


def is_checked(ctrl):
    """True if a check box / radio button is ticked (MSAA CHECKED state)."""
    return bool(_state(ctrl) & STATE_CHECKED)
