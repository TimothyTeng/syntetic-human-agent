"""
Launching, finding and closing programs / windows.

Programs are opened the way a person would: through the Start menu search or
File Explorer. That keeps the launch going through the normal Windows shell
instead of being spawned directly by this Python process.
"""

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
