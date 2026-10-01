"""
Mouse control primitives.

All coordinates are absolute screen pixels (0,0 = top-left of primary monitor).
`duration` arguments control how long a movement takes (0 = instant teleport).
`tween` lets the caller pick an easing curve from pyautogui (e.g.
pyautogui.easeInOutQuad) - the future entropy layer can also replace movement
entirely by calling move_to() repeatedly along its own generated path.

Cursor movement is sent as real absolute mouse-move events (win_input.SendInput),
not SetCursorPos, so input listeners and the system idle timer see it like a
physical mouse. Buttons and the wheel go through pyautogui (mouse_event).
"""

import time

import pyautogui

from . import config  # noqa: F401  (applies DPI / pyautogui settings)
from . import win_input

MOVE_RATE_HZ = 125   # update rate for timed moves - a typical USB mouse polling rate

# On Windows pyautogui passes the scroll amount straight through as the raw
# wheel delta, where one physical wheel notch = 120. We convert so that
# scroll(1) == one notch.
WHEEL_DELTA = 120


def get_position():
    """Return the current mouse cursor position as (x, y)."""
    return win_input.cursor_pos()


def move_to(x, y, duration=0.0, tween=pyautogui.linear):
    """
    Move the cursor to absolute screen position (x, y).

    duration: seconds the movement takes (0 = instant).
    tween:    easing function controlling speed along the path.
    """
    if duration <= 0:
        win_input.send_mouse_move(x, y)
        return
    x0, y0 = get_position()
    steps = max(2, int(duration * MOVE_RATE_HZ))
    t0 = time.perf_counter()
    for i in range(1, steps + 1):
        f = tween(i / steps)
        win_input.send_mouse_move(round(x0 + (x - x0) * f), round(y0 + (y - y0) * f))
        delay = t0 + duration * i / steps - time.perf_counter()
        if delay > 0:
            time.sleep(delay)


def move_relative(dx, dy, duration=0.0, tween=pyautogui.linear):
    """Move the cursor by (dx, dy) pixels from its current position."""
    x0, y0 = get_position()
    move_to(x0 + dx, y0 + dy, duration=duration, tween=tween)


def click(x=None, y=None, button="left", duration=0.0):
    """
    Single click. If x/y are given, move there first (taking `duration` secs),
    otherwise click at the current cursor position.

    button: 'left', 'right' or 'middle'.
    """
    if x is not None and y is not None:
        move_to(x, y, duration=duration)
    pyautogui.click(button=button)


def double_click(x=None, y=None, button="left", duration=0.0, interval=0.0):
    """
    Double click at (x, y) or at the current position.

    interval: gap in seconds between the two clicks (must stay below the
              Windows double-click time, ~0.5 s by default).
    """
    if x is not None and y is not None:
        move_to(x, y, duration=duration)
    pyautogui.click(clicks=2, interval=interval, button=button)


def right_click(x=None, y=None, duration=0.0):
    """Right click (open context menu) at (x, y) or at the current position."""
    click(x, y, button="right", duration=duration)


def mouse_down(button="left"):
    """Press and hold a mouse button at the current position."""
    pyautogui.mouseDown(button=button)


def mouse_up(button="left"):
    """Release a held mouse button at the current position."""
    pyautogui.mouseUp(button=button)


def drag_to(x, y, duration=0.5, button="left"):
    """
    Click-and-drag from the current position to (x, y) - e.g. selecting text
    or moving a window. A non-zero duration is recommended; many apps ignore
    drags that happen instantly.
    """
    mouse_down(button)
    try:
        move_to(x, y, duration=duration)
    finally:
        mouse_up(button)


def scroll(amount, x=None, y=None):
    """
    Vertical mouse-wheel scroll.

    amount: number of wheel notches; positive = up, negative = down.
            (One notch is usually ~3 lines / ~100 px in a browser.)
            Fractions are allowed (e.g. 0.5) - like a smooth/precision wheel.
    x, y:   optionally move the cursor here first (scroll goes to the window
            under the cursor).
    """
    if x is not None and y is not None:
        move_to(x, y)
    pyautogui.scroll(int(round(amount * WHEEL_DELTA)))


def hscroll(amount, x=None, y=None):
    """Horizontal scroll in wheel notches; positive = right, negative = left."""
    if x is not None and y is not None:
        move_to(x, y)
    pyautogui.hscroll(int(round(amount * WHEEL_DELTA)))


def click_center_of(rect, button="left", duration=0.0):
    """
    Click the centre of a rectangle.

    rect: (left, top, right, bottom) - the format returned by
          ui_elements.element_rect() and browser.find_links().
    """
    left, top, right, bottom = rect
    click((left + right) // 2, (top + bottom) // 2, button=button, duration=duration)
