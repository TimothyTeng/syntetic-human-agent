"""
Keyboard control primitives.

Key names follow pyautogui: 'enter', 'tab', 'esc', 'backspace', 'delete',
'ctrl', 'shift', 'alt', 'win', 'f1'..'f12', 'up', 'down', 'left', 'right',
'home', 'end', 'pageup', 'pagedown', single characters 'a', '1', etc.
Full list: pyautogui.KEYBOARD_KEYS

Keys are sent with win_input.SendInput carrying the real scan code (and the
extended-key flag for arrows, Home, End, ...), like a physical keyboard.
"""

import time

import pyautogui
import pyperclip

from . import config  # noqa: F401
from . import win_input


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
        press_key("enter")
    elif ch == "\t":
        press_key("tab")
    elif ch.isascii() and win_input.key_vk(ch)[0] is not None:
        vk, shift = win_input.key_vk(ch)
        if shift:
            win_input.send_key(0x10)                # VK_SHIFT
        win_input.send_key(vk)
        win_input.send_key(vk, up=True)
        if shift:
            win_input.send_key(0x10, up=True)
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
    for i in range(presses):
        if i and interval:
            time.sleep(interval)
        key_down(key)
        key_up(key)


def hotkey(*keys, interval=0.0):
    """
    Press a key combination: keys go down in order, then release in reverse.

    Example: hotkey('ctrl', 's')   hotkey('ctrl', 'shift', 's')
    """
    for i, key in enumerate(keys):
        if i and interval:
            time.sleep(interval)
        key_down(key)
    for key in reversed(keys):
        if interval:
            time.sleep(interval)
        key_up(key)


def key_down(key):
    """Hold a key down (remember to call key_up)."""
    vk, shift = win_input.key_vk(key)
    if vk is None or shift:          # not a plain key on this layout: let pyautogui handle it
        pyautogui.keyDown(key)
    else:
        win_input.send_key(vk)


def key_up(key):
    """Release a key previously held with key_down."""
    vk, shift = win_input.key_vk(key)
    if vk is None or shift:
        pyautogui.keyUp(key)
    else:
        win_input.send_key(vk, up=True)


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
