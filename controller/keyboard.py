"""
Keyboard control primitives.

Key names follow pyautogui: 'enter', 'tab', 'esc', 'backspace', 'delete',
'ctrl', 'shift', 'alt', 'win', 'f1'..'f12', 'up', 'down', 'left', 'right',
'home', 'end', 'pageup', 'pagedown', single characters 'a', '1', etc.
Full list: pyautogui.KEYBOARD_KEYS
"""

import time

import pyautogui
import pyperclip

from . import config  # noqa: F401


def type_text(text, interval=0.0):
    """
    Type a string into the focused window.

    interval: fixed seconds between keystrokes. For variable per-key timing,
              the entropy layer should loop over characters and call
              type_char() with its own delays instead.

    '\n' is typed as Enter. Non-ASCII characters (which pyautogui cannot type
    directly) are inserted via the clipboard.
    """
    for ch in text:
        type_char(ch)
        if interval:
            time.sleep(interval)


def type_char(ch):
    """Type a single character (use this to build custom typing rhythms)."""
    if ch == "\n":
        pyautogui.press("enter")
    elif ch == "\t":
        pyautogui.press("tab")
    elif ch.isascii():
        pyautogui.write(ch)
    else:
        # pyautogui only maps ASCII keys - paste anything else (é, ü, emoji...)
        paste_text(ch)


def press_key(key, presses=1, interval=0.0):
    """
    Press and release a key `presses` times.

    Example: press_key('down', presses=3, interval=0.1)
    """
    pyautogui.press(key, presses=presses, interval=interval)


def hotkey(*keys, interval=0.0):
    """
    Press a key combination: keys go down in order, then release in reverse.

    Example: hotkey('ctrl', 's')   hotkey('ctrl', 'shift', 's')
    """
    pyautogui.hotkey(*keys, interval=interval)


def key_down(key):
    """Hold a key down (remember to call key_up)."""
    pyautogui.keyDown(key)


def key_up(key):
    """Release a key previously held with key_down."""
    pyautogui.keyUp(key)


def backspace(n=1, interval=0.0):
    """Press Backspace n times - e.g. to correct a typo."""
    press_key("backspace", presses=n, interval=interval)


def select_all():
    """Ctrl+A - select everything in the focused field/document."""
    hotkey("ctrl", "a")


def copy():
    """Ctrl+C - copy selection to clipboard."""
    hotkey("ctrl", "c")


def paste():
    """Ctrl+V - paste clipboard contents."""
    hotkey("ctrl", "v")


def set_clipboard(text):
    """Put text on the Windows clipboard."""
    pyperclip.copy(text)


def get_clipboard():
    """Return the current clipboard text."""
    return pyperclip.paste()


def paste_text(text):
    """Insert text by placing it on the clipboard and pressing Ctrl+V."""
    set_clipboard(text)
    paste()
