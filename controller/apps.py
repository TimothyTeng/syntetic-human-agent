"""
Launching, finding and closing programs / windows.

Programs are opened the way a person would: through the Start menu search or
File Explorer. That keeps the launch going through the normal Windows shell
instead of being spawned directly by this Python process.
"""

import ctypes
import os
import time

import psutil
import pygetwindow as gw

from . import config, keyboard


def open_via_start_menu(app_name, wait_title=None, timeout=config.DEFAULT_TIMEOUT,
                        open_wait=config.START_MENU_OPEN_WAIT,
                        search_wait=config.START_MENU_SEARCH_WAIT, type_interval=0.0):
    """
    Open an application by pressing the Windows key, typing its name and Enter.

    app_name:    what you'd type into Start search, e.g. 'word', 'excel',
                 'notepad', 'chrome'.
    wait_title:  if given, wait until a window whose title contains this text
                 appears and return it (pygetwindow Window). Else returns None.
    open_wait:   seconds to wait for the Start menu to open.
    search_wait: seconds to wait for search results before pressing Enter.
    """
    keyboard.press_key("win")
    time.sleep(open_wait)
    keyboard.type_text(app_name, interval=type_interval)
    time.sleep(search_wait)
    keyboard.press_key("enter")
    if wait_title:
        return wait_for_window(wait_title, timeout=timeout)
    return None


def open_file_via_explorer(path, wait_title=None, timeout=config.DEFAULT_TIMEOUT,
                           open_wait=1.5, type_interval=0.0):
    """
    Open a file (e.g. C:\\Users\\me\\Documents\\report.docx) with its default
    program by opening File Explorer (Win+E), typing the full path into the
    address bar and pressing Enter.

    The Explorer window is left open; call close_window('File Explorer') or
    similar afterwards if you want it gone.
    """
    keyboard.hotkey("win", "e")
    time.sleep(open_wait)
    keyboard.hotkey("alt", "d")          # focus the address bar
    time.sleep(0.3)
    keyboard.type_text(os.path.abspath(path), interval=type_interval)
    keyboard.press_key("enter")
    if wait_title:
        return wait_for_window(wait_title, timeout=timeout)
    return None


def is_running(process_name):
    """True if a process with this exe name is running, e.g. 'chrome.exe'."""
    target = process_name.lower()
    for p in psutil.process_iter(["name"]):
        if (p.info["name"] or "").lower() == target:
            return True
    return False


def find_windows(title_substring):
    """Return a list of pygetwindow Windows whose title contains the text."""
    return [w for w in gw.getWindowsWithTitle(title_substring) if w.title]


def wait_for_window(title_substring, timeout=config.DEFAULT_TIMEOUT):
    """
    Wait until a window whose title contains `title_substring` exists.
    Returns the pygetwindow Window, or None on timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        wins = find_windows(title_substring)
        if wins:
            return wins[0]
        time.sleep(config.POLL_INTERVAL)
    return None


def get_active_window_title():
    """Return the title of the window that currently has focus ('' if none)."""
    win = gw.getActiveWindow()
    return win.title if win else ""


def focus_window(title_substring):
    """
    Bring a window to the front and give it keyboard focus.
    Returns True on success, False if no such window exists.
    """
    wins = find_windows(title_substring)
    if not wins:
        return False
    win = wins[0]
    if win.isMinimized:
        win.restore()
    try:
        win.activate()
    except gw.PyGetWindowException:
        # Windows sometimes refuses SetForegroundWindow; a minimize/restore
        # cycle is the standard workaround.
        win.minimize()
        win.restore()
    return True


def maximize_window(title_substring):
    """Maximize the first window matching the title. Returns True/False."""
    wins = find_windows(title_substring)
    if wins:
        wins[0].maximize()
        return True
    return False


def minimize_window(title_substring):
    """Minimize the first window matching the title. Returns True/False."""
    wins = find_windows(title_substring)
    if wins:
        wins[0].minimize()
        return True
    return False


def get_window_rect(title_substring):
    """Return (left, top, right, bottom) of the first matching window, or None."""
    wins = find_windows(title_substring)
    if not wins:
        return None
    w = wins[0]
    return w.left, w.top, w.right, w.bottom


def close_window(title_substring=None):
    """
    Close a window with Alt+F4 (like a user would).
    If title_substring is given, that window is focused first; otherwise the
    currently focused window is closed.
    Returns False if the requested window wasn't found.
    """
    if title_substring is not None and not focus_window(title_substring):
        return False
    time.sleep(0.2)
    keyboard.hotkey("alt", "f4")
    return True


# --- Win32 window queries --------------------------------------------------------
# These use plain Win32 calls rather than UI Automation: reading a UIA property of a
# window that is being created or destroyed raises a COM error, while Win32 simply
# returns an empty value.

def window_class(hwnd):
    """Window class name of a handle ('' if the window is gone)."""
    buf = ctypes.create_unicode_buffer(256)
    ctypes.windll.user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def window_title(hwnd):
    """Title of a window ('' if none or gone)."""
    n = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def top_level_windows(window_class_name=None, titled=False):
    """Handles of visible (or minimised) top-level windows, optionally of one class /
    with a title, in z-order (front first)."""
    out = []
    user32 = ctypes.windll.user32

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _collect(hwnd, _):
        if user32.IsWindowVisible(hwnd) and (window_class_name is None or window_class(hwnd) == window_class_name) \
                and (not titled or user32.GetWindowTextLengthW(hwnd)):
            out.append(hwnd)
        return True

    user32.EnumWindows(_collect, 0)
    return out


def foreground_top_level():
    """Handle of the top-level window in front (the root of whatever has focus)."""
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    return user32.GetAncestor(hwnd, 2) if hwnd else 0          # GA_ROOT


def control_for(hwnd, retries=3):
    """UIA control for a window handle, or None if the window is (still) changing."""
    import uiautomation as auto
    for _ in range(retries):
        try:
            ctrl = auto.ControlFromHandle(hwnd)
            ctrl.ClassName                                     # touch it: fails if not ready
            return ctrl
        except Exception:
            time.sleep(0.2)
    return None


# --- Alt+Tab order -------------------------------------------------------------

def _is_switcher_window(hwnd):
    """True for windows that appear in the Alt+Tab switcher (visible, titled, not a
    tool window, unowned, not cloaked like suspended UWP apps)."""
    user32 = ctypes.windll.user32
    if not user32.IsWindowVisible(hwnd) or not user32.GetWindowTextLengthW(hwnd):
        return False
    if user32.GetWindow(hwnd, 4):                         # GW_OWNER: owned popups don't show
        return False
    if user32.GetWindowLongW(hwnd, -20) & 0x00000080:     # GWL_EXSTYLE & WS_EX_TOOLWINDOW
        return False
    cloaked = ctypes.c_int(0)
    ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked))  # DWMWA_CLOAKED
    return not cloaked.value


def switcher_windows():
    """
    Window handles in Alt+Tab order (most recently used first): [0] is the window in
    front, [1] is where a single Alt+Tab goes.
    """
    out = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _collect(hwnd, _):
        if _is_switcher_window(hwnd):
            out.append(hwnd)
        return True

    ctypes.windll.user32.EnumWindows(_collect, 0)          # enumerates in z-order, top first
    return out


def foreground_window():
    """Handle of the window in front."""
    return ctypes.windll.user32.GetForegroundWindow()


def bring_to_front(hwnd):
    """Restore (if minimised) and activate a window by handle. Returns True if it is now in front."""
    user32 = ctypes.windll.user32
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)                         # SW_RESTORE
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.3)
    return user32.GetForegroundWindow() == hwnd
