"""
Google Chrome control: open, navigate, tabs, scroll, find and click links.

Navigation uses the real address bar / keyboard shortcuts. Links are found
through Chrome's accessibility tree (UI Automation), which gives the link text
and its exact on-screen rectangle - no browser debugging flags needed.

Note: Chrome builds the web-page accessibility tree lazily the first time an
accessibility client asks for it, so the very first find_links() call on a
page may return few/no links. The functions below retry for a short while.
Chrome also stops rendering (and exposing) pages whose window is fully covered
or minimized - bring Chrome to the front (focus_chrome()) before find_links().
"""

import ctypes
import re
import time

import uiautomation as auto

from . import apps, config, keyboard, mouse, ui_elements

CHROME_TITLE = "Google Chrome"
CHROME_CLASS = "Chrome_WidgetWin_1"

_user32 = ctypes.windll.user32
SW_MAXIMIZE, SW_RESTORE = 3, 9


# --- Window ------------------------------------------------------------------

def _chrome_windows():
    """All top-level Chrome windows (UIA controls), including minimised ones. Found with
    Win32 calls, so windows opening or closing meanwhile are skipped, not errors."""
    controls = (apps.control_for(h, retries=1) for h in apps.top_level_windows(CHROME_CLASS, titled=True))
    return [c for c in controls if c is not None]


def _is_browser_window(win):
    """Normal browser windows are titled '<page title> - Google Chrome'."""
    try:
        return apps.window_title(win.NativeWindowHandle).endswith(" - " + CHROME_TITLE)
    except Exception:
        return False


def _foreground_chrome():
    """The foreground window's UIA control if it is a Chrome window, else None."""
    hwnd = _user32.GetForegroundWindow()
    buf = ctypes.create_unicode_buffer(64)
    _user32.GetClassNameW(hwnd, buf, 64)
    if buf.value != CHROME_CLASS or not _user32.GetWindowTextLengthW(hwnd):
        return None
    try:
        return apps.control_for(hwnd, retries=1)
    except Exception:          # the window closed in between (e.g. the profile picker going away)
        return None


def open_chrome(timeout=config.DEFAULT_TIMEOUT, launch=None):
    """
    Open Chrome from the Start menu and wait until a *new* Chrome window is in
    the foreground: either a browser window or the "Who's using Chrome?"
    profile picker (check with get_profile_picker()).
    Already-open Chrome windows don't count, so this works while Chrome runs.
    launch: optional callable that opens Chrome instead (e.g. the behaviour
            layer's human-timed Start-menu search).
    Returns the new window's UIA control, or None on timeout.
    """
    before = {w.NativeWindowHandle for w in _chrome_windows()}
    (launch or (lambda: apps.open_via_start_menu("chrome")))()
    deadline = time.time() + timeout
    while time.time() < deadline:
        fg = _foreground_chrome()
        if fg is not None and fg.NativeWindowHandle not in before:
            return fg
        time.sleep(config.POLL_INTERVAL)
    return None


def get_chrome_window(timeout=5):
    """
    Return the Chrome *browser* window to work with (UIA control), or None:
    the foreground one if Chrome is in front, otherwise the first visible one.
    Minimised windows and the profile picker are ignored.
    """
    deadline = time.time() + timeout
    while True:
        fg = _foreground_chrome()
        if fg is not None and _is_browser_window(fg):
            return fg
        visible = [w for w in _chrome_windows()
                   if _is_browser_window(w) and ui_elements.is_visible(w)]
        if visible:
            return visible[0]
        if time.time() >= deadline:
            return None
        time.sleep(config.POLL_INTERVAL)


def wait_for_browser_window(timeout=config.DEFAULT_TIMEOUT):
    """
    Wait until a normal browser window is in the foreground, e.g. after
    choosing a profile in the picker. Returns its UIA control or None.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        fg = _foreground_chrome()
        if fg is not None and _is_browser_window(fg):
            return fg
        time.sleep(config.POLL_INTERVAL)
    return None


def focus_chrome():
    """Bring the Chrome browser window to the foreground. Returns True/False."""
    win = get_chrome_window(timeout=1)
    if win is None:  # maybe minimised - restore the first browser window
        wins = [w for w in _chrome_windows() if _is_browser_window(w)]
        if not wins:
            return False
        win = wins[0]
        _user32.ShowWindow(win.NativeWindowHandle, SW_RESTORE)
    _user32.SetForegroundWindow(win.NativeWindowHandle)
    return True


def maximize_chrome():
    """Maximise the current Chrome browser window. Returns True/False."""
    win = get_chrome_window()
    if win is None:
        return False
    _user32.ShowWindow(win.NativeWindowHandle, SW_MAXIMIZE)
    return True


# --- Profile picker ("Who's using Chrome?") ------------------------------------
# Each profile card is exposed as a button named "Open <profile name> profile",
# with the account name as a text element inside it. The names are the English
# UI strings; adjust PROFILE_BUTTON_RE for other Chrome languages.

PROFILE_BUTTON_RE = re.compile(r"^Open (.+) profile$")
CARD_HEADER_PX = 60   # top strip of a card holds the (renameable) profile name - avoid clicking it


def get_profile_picker(timeout=0):
    """
    Return the profile picker window (UIA control) if it is open, else None.
    timeout: seconds to keep checking (0 = check once).
    """
    deadline = time.time() + timeout
    while True:
        for w in _chrome_windows():
            try:
                if apps.window_title(w.NativeWindowHandle) != CHROME_TITLE:   # the picker is titled just "Google Chrome"
                    continue
            except Exception:
                continue
            _request_web_accessibility(w)
            if auto.ButtonControl(searchFromControl=w, RegexName=PROFILE_BUTTON_RE.pattern).Exists(0.5):
                return w
        if time.time() >= deadline:
            return None
        time.sleep(config.POLL_INTERVAL)


def list_profiles(timeout=5):
    """
    List the profile cards on the picker, left to right, as dicts:
        {'name':    profile name (top label, e.g. 'Work'),
         'account': account/person name (bottom label, e.g. 'Timothy Teng') or None,
         'index':   position on the picker (0-based),
         'rect':    whole card (left, top, right, bottom),
         'click_rect': safe area to click (avatar + account name, below the renameable header),
         'center':  centre of click_rect}
    Returns [] if the picker isn't open.
    """
    picker = get_profile_picker(timeout)
    if picker is None:
        return []
    profiles = []
    for ctrl, _ in auto.WalkControl(picker, maxDepth=40):
        if ctrl.ControlType != auto.ControlType.ButtonControl:
            continue
        m = PROFILE_BUTTON_RE.match(ctrl.Name or "")
        if not m:
            continue
        rect = ui_elements.element_rect(ctrl)
        texts = [c.Name for c, _ in auto.WalkControl(ctrl, maxDepth=4)
                 if c.ControlType == auto.ControlType.TextControl and c.Name]
        click_rect = (rect[0], min(rect[1] + CARD_HEADER_PX, rect[3] - 10), rect[2], rect[3])
        profiles.append({
            "name": m.group(1),
            "account": texts[0] if texts else None,
            "rect": rect,
            "click_rect": click_rect,
            "center": ((click_rect[0] + click_rect[2]) // 2, (click_rect[1] + click_rect[3]) // 2),
            "control": ctrl,
        })
    profiles.sort(key=lambda p: (p["rect"][1], p["rect"][0]))
    for i, p in enumerate(profiles):
        p["index"] = i
    return profiles


def find_profile(name, index=0, exact=False):
    """
    Find a profile card by profile name OR account name (case-insensitive).

    name:  e.g. 'Work', 'VX', 'Timothy Teng', 'sstarchive7'.
    index: which one to take when several cards match, left to right
           (e.g. three 'Timothy' profiles -> index 0, 1 or 2).
    exact: require the whole name to match; otherwise exact matches are
           preferred and substring matches are used if there are none.
    Returns the profile dict (see list_profiles) or None.
    """
    needle = name.lower()
    profiles = list_profiles()
    fields = lambda p: [(p["name"] or "").lower(), (p["account"] or "").lower()]  # noqa: E731
    matches = [p for p in profiles if needle in fields(p)]
    if not matches and not exact:
        matches = [p for p in profiles if any(needle in f for f in fields(p))]
    return matches[index] if 0 <= index < len(matches) else None


def select_profile(name, index=0, exact=False, duration=0.0):
    """
    Click a profile on the picker (see find_profile for matching rules).
    Returns True if clicked. Follow with wait_for_browser_window().
    """
    profile = find_profile(name, index, exact)
    if not profile:
        return False
    mouse.click(*profile["center"], duration=duration)
    return True


def get_guest_mode_rect(timeout=2):
    """Rect of the picker's 'Guest mode' button, or None."""
    picker = get_profile_picker(timeout)
    if picker is None:
        return None
    btn = ui_elements.find_element(picker, "Guest mode", control_type="ButtonControl", timeout=2)
    return ui_elements.element_rect(btn) if btn else None


def close_chrome():
    """Close the Chrome window (Ctrl+Shift+W closes the whole window)."""
    if focus_chrome():
        keyboard.hotkey("ctrl", "shift", "w")
        return True
    return False


# --- Navigation --------------------------------------------------------------

def navigate_to(url, type_interval=0.0):
    """
    Go to a URL by focusing the address bar (Ctrl+L), typing it and Enter.
    type_interval: seconds between keystrokes when typing the URL.
    """
    keyboard.hotkey("ctrl", "l")
    time.sleep(0.2)
    keyboard.type_text(url, interval=type_interval)
    keyboard.press_key("enter")


def get_current_url():
    """Read the text currently in Chrome's address bar ('' if not found)."""
    win = get_chrome_window()
    if not win:
        return ""
    bar = ui_elements.find_element(win, "Address and search bar",
                                   control_type="EditControl", timeout=2)
    if not bar:
        return ""
    try:
        return bar.GetPattern(auto.PatternId.ValuePattern).Value
    except Exception:
        return ""


def get_page_title():
    """Return the current tab's title (window title minus ' - Google Chrome')."""
    title = apps.get_active_window_title()
    return title.replace(" - Google Chrome", "")


def wait_for_page_load(title_hint=None, timeout=config.DEFAULT_TIMEOUT):
    """
    Wait until the page has finished loading.

    Chrome's toolbar button is named 'Stop' while loading and 'Reload' when
    done. If title_hint is given, also wait until the window title contains it.
    Returns True if loaded within timeout.
    """
    deadline = time.time() + timeout
    win = get_chrome_window()
    while time.time() < deadline:
        title_ok = title_hint is None or title_hint.lower() in apps.get_active_window_title().lower()
        reload_btn = win and auto.ButtonControl(searchFromControl=win, Name="Reload").Exists(0)
        if title_ok and reload_btn:
            return True
        time.sleep(config.POLL_INTERVAL)
    return False


def go_back():
    """Browser Back (Alt+Left)."""
    keyboard.hotkey("alt", "left")


def go_forward():
    """Browser Forward (Alt+Right)."""
    keyboard.hotkey("alt", "right")


def refresh():
    """Reload the page (F5)."""
    keyboard.press_key("f5")


# --- Locating Chrome UI parts ------------------------------------------------

def get_address_bar_rect(timeout=3):
    """Screen rect (left, top, right, bottom) of the address bar, or None."""
    win = get_chrome_window()
    if not win:
        return None
    bar = ui_elements.find_element(win, "Address and search bar",
                                   control_type="EditControl", timeout=timeout)
    return ui_elements.element_rect(bar) if bar else None


def address_bar_has_focus(timeout=1):
    """
    True only if a Chrome browser window is in the foreground AND its address
    bar has keyboard focus. Check this before typing a URL/search so keystrokes
    can never end up in another application.
    """
    fg = _foreground_chrome()
    if fg is None or not _is_browser_window(fg):
        return False
    bar = ui_elements.find_element(fg, "Address and search bar",
                                   control_type="EditControl", timeout=timeout)
    try:
        return bool(bar) and bool(bar.HasKeyboardFocus)
    except Exception:
        return False


def get_toolbar_button_rect(name, timeout=3):
    """
    Screen rect of a Chrome toolbar button by its accessible name, or None.
    Names: 'Back', 'Forward', 'Reload', 'Bookmark this tab', ...
    """
    win = get_chrome_window()
    if not win:
        return None
    btn = ui_elements.find_element(win, name, control_type="ButtonControl", timeout=timeout)
    return ui_elements.element_rect(btn) if btn else None


def get_page_rect():
    """Screen rect of the web page content area (below the toolbar), or None."""
    win = get_chrome_window()
    if not win:
        return None
    doc = _get_page_document(win)
    return ui_elements.element_rect(doc) if doc else None


# --- Tabs --------------------------------------------------------------------

def new_tab():
    """Open a new tab (Ctrl+T)."""
    keyboard.hotkey("ctrl", "t")


def close_tab():
    """Close the current tab (Ctrl+W)."""
    keyboard.hotkey("ctrl", "w")


def tab_items(win=None):
    """
    The tabs of a Chrome window (UIA TabItem controls), left to right. Only tabs in
    the tab strip count - tab widgets inside the web page are ignored.
    """
    win = win or get_chrome_window(timeout=1)
    if not win:
        return []
    page = _get_page_document(win)
    page_top = page.BoundingRectangle.top if page else None
    tabs = [t for t in ui_elements.find_elements(win, "TabItemControl", max_depth=12)
            if page_top is None or t.BoundingRectangle.bottom <= page_top]
    return sorted(tabs, key=lambda t: t.BoundingRectangle.left)


def selected_tab_index(tabs):
    """Index of the active tab in tab_items() (None if it can't be told)."""
    for i, tab in enumerate(tabs):
        try:
            if tab.GetSelectionItemPattern().IsSelected:
                return i
        except Exception:
            pass
    return None


def tab_close_button(tab):
    """The small close (x) button on a tab, or None."""
    return ui_elements.find_element(tab, "Close", "ButtonControl", partial=True, timeout=0.5, search_depth=3)


def switch_tab(n):
    """Switch to tab number n (1-8), or 9 = last tab (Ctrl+<n>)."""
    keyboard.hotkey("ctrl", str(n))


def next_tab():
    """Go to the next tab (Ctrl+Tab)."""
    keyboard.hotkey("ctrl", "tab")


# --- Scrolling ---------------------------------------------------------------

def scroll_page(amount, x=None, y=None):
    """
    Scroll the page with the mouse wheel. amount > 0 = up, < 0 = down.
    If x/y not given, the cursor is moved to the centre of the Chrome window
    first so the wheel event reaches the page.
    """
    if x is None or y is None:
        rect = apps.get_window_rect(CHROME_TITLE)
        if rect:
            x, y = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
    mouse.scroll(amount, x, y)


def page_down():
    """Keyboard Page Down."""
    keyboard.press_key("pagedown")


def page_up():
    """Keyboard Page Up."""
    keyboard.press_key("pageup")


# --- Links -------------------------------------------------------------------

def _get_page_document(win):
    """
    Return the DocumentControl holding the web page content.
    Chrome has several DocumentControls (most are hidden, 0x0 size), so pick
    the largest visible one - that is the active tab's page.
    """
    docs = [d for d, _ in auto.WalkControl(win, maxDepth=14)
            if d.ControlType == auto.ControlType.DocumentControl and ui_elements.is_visible(d)]
    if not docs:
        return None
    # Chrome wraps the page in an empty, unnamed DocumentControl of the same
    # size; the real page document carries the page title as its Name.
    # Prefer named documents, then the largest area.
    area = lambda d: d.BoundingRectangle.width() * d.BoundingRectangle.height()  # noqa: E731
    return max(docs, key=lambda d: (bool(d.Name), area(d)))


def _request_web_accessibility(win):
    """
    Best-effort: ask Chrome's page renderer windows for their accessibility
    object (WM_GETOBJECT), which is how Chrome notices an accessibility client
    and starts exposing page content (links, text) through UI Automation.
    """
    WM_GETOBJECT, OBJID_CLIENT = 0x003D, -4
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _collect(hwnd, _):
        buf = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, buf, 64)
        if buf.value == "Chrome_RenderWidgetHostHWND":
            found.append(hwnd)
        return True

    user32.EnumChildWindows(win.NativeWindowHandle, _collect, 0)
    for hwnd in found:
        user32.SendMessageW(ctypes.c_void_p(hwnd), WM_GETOBJECT, 0, ctypes.c_ssize_t(OBJID_CLIENT))


def find_links(visible_only=True, retries=3, retry_wait=0.5, max_depth=40):
    """
    Return the links on the current page as a list of dicts:
        {'name': link text, 'rect': (left, top, right, bottom),
         'center': (x, y), 'control': <uiautomation control>}

    visible_only: only links currently inside the viewport (clickable now).
    """
    win = get_chrome_window()
    if not win:
        return []
    _request_web_accessibility(win)
    for _ in range(retries):
        doc = _get_page_document(win)
        root = doc if doc else win
        view = ui_elements.element_rect(root)
        links = []
        for ctrl in ui_elements.find_elements(root, "HyperlinkControl",
                                              visible_only=visible_only, max_depth=max_depth):
            rect = ui_elements.element_rect(ctrl)
            # Extra viewport check: IsOffscreen isn't always reliable in Chrome
            if visible_only and not _inside(rect, view):
                continue
            links.append({
                "name": (ctrl.Name or "").strip(),
                "rect": rect,
                "center": ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2),
                "control": ctrl,
            })
        if links:
            return links
        time.sleep(retry_wait)  # tree may still be building - try again
    return []


def _inside(rect, view):
    """True if rect's centre lies within the view rectangle."""
    cx, cy = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
    return view[0] <= cx <= view[2] and view[1] <= cy <= view[3]


def find_link(text, exact=False, visible_only=True):
    """
    Find the first link whose text matches `text` (case-insensitive).
    exact=False means substring match. Returns the link dict or None.
    """
    needle = text.lower()
    for link in find_links(visible_only=visible_only):
        name = link["name"].lower()
        if (name == needle) if exact else (needle in name):
            return link
    return None


def click_link(text, exact=False, duration=0.0):
    """
    Find a visible link by its text, move the mouse to it and click.
    duration: seconds for the mouse movement. Returns True if clicked.
    """
    link = find_link(text, exact=exact)
    if not link:
        return False
    mouse.click(*link["center"], duration=duration)
    return True


def scroll_until_link_visible(text, exact=False, step=-3, max_scrolls=15, pause=0.4):
    """
    Scroll down (step < 0) until a link with the text is in view.
    Returns the link dict, or None if not found after max_scrolls.
    """
    for _ in range(max_scrolls):
        link = find_link(text, exact=exact)
        if link:
            return link
        scroll_page(step)
        time.sleep(pause)
    return None


# --- Page text ---------------------------------------------------------------

def page_text_nodes(area=None, min_words=3):
    """
    Visible text elements of the current page: (pattern, nodes), where nodes =
    [{'text', 'rect', 'control'}] top to bottom (only those inside `area` if given)
    and pattern is the page's TextPattern (or None), for ui_elements.node_words().
    Chrome must be in the foreground, as for find_links().
    """
    win = get_chrome_window()
    if not win:
        return None, []
    _request_web_accessibility(win)
    doc = _get_page_document(win)
    if not doc:
        return None, []
    return ui_elements.text_pattern(doc), ui_elements.text_nodes(doc, area, min_words)


def page_total_words(max_chars=400_000):
    """Words in the WHOLE current page (not just what is in view), from its text
    interface; 0 if the page doesn't expose its text."""
    win = get_chrome_window(timeout=1)
    doc = _get_page_document(win) if win else None
    if doc is None:
        return 0
    try:
        return len(doc.GetTextPattern().DocumentRange.GetText(max_chars).split())
    except Exception:
        return 0


def page_view_signature(n=5):
    """
    Positions of the first few visible text elements of the page - changes whenever
    the page scrolls. Used to notice the bottom of a page that doesn't report its
    scroll position: scrolling down no longer changes it. None if no text is found.
    """
    _, nodes = page_text_nodes(min_words=1)
    if not nodes:
        return None
    return tuple((n_["text"][:20], n_["rect"][1]) for n_ in nodes[:n])


def page_word_count():
    """Number of words in the page's currently visible text (0 if none found)."""
    _, nodes = page_text_nodes(min_words=1)
    return sum(len(n["text"].split()) for n in nodes)


def page_scroll_state():
    """
    (scroll position %, visible part of the page %) of the current page, e.g.
    (37.5, 20.0) = 37.5% scrolled down with a fifth of the page in view. 100% position
    means the bottom is reached. None if the page doesn't report it (or can't scroll).
    """
    win = get_chrome_window(timeout=1)
    doc = _get_page_document(win) if win else None
    for ctrl in (doc, doc.GetParentControl() if doc else None):
        if ctrl is None:
            continue
        try:
            sp = ctrl.GetPattern(auto.PatternId.ScrollPattern)
            if sp and sp.VerticallyScrollable:
                return float(sp.VerticalScrollPercent), float(sp.VerticalViewSize)
        except Exception:
            pass
    return None


def page_has_keyboard_focus():
    """
    True if keys would go to the page itself (Chrome in front, focus on the page and
    not in a text field or the address bar) - so PgDn / arrow keys scroll it.
    """
    fg = _foreground_chrome()
    if fg is None or not _is_browser_window(fg):
        return False
    try:
        focused = auto.GetFocusedControl()
    except Exception:
        return False
    if focused is None:
        return False
    if focused.ControlType in (auto.ControlType.EditControl, auto.ControlType.ComboBoxControl):
        return False
    doc = _get_page_document(fg)
    if doc is None:
        return False
    l, t, r, b = ui_elements.element_rect(doc)
    fx, fy = ui_elements.element_center(focused)
    return l <= fx <= r and t <= fy <= b
