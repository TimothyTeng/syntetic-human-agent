"""
Low-level input through the Win32 SendInput API.

pyautogui moves the cursor with SetCursorPos and presses keys with
keybd_event(vk, scan=0). Neither looks like a real device:
  * SetCursorPos is not an input event - raw-input listeners and low-level mouse
    hooks never see the movement and the system idle timer (GetLastInputInfo)
    is not reset, so a generated trajectory is invisible and clicks appear to
    come from nowhere.
  * scan code 0 leaves e.g. the browser's KeyboardEvent.code empty, and
    without the extended-key flag arrows / Home / End / Delete arrive as their
    numeric-keypad twins.

This module sends proper mouse-move and key events instead. pyautogui's corner
fail-safe is checked before every event, exactly as pyautogui itself does.
"""

import ctypes
from ctypes import wintypes

import pyautogui

from . import config  # noqa: F401  (DPI awareness must be set before any coordinates are used)

user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_MOUSE, INPUT_KEYBOARD = 0, 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002

MAPVK_VK_TO_VSC_EX = 4

# Keys whose scan code carries the 0xE0 prefix (MapVirtualKey does not always say so)
EXTENDED_VKS = {
    0x21, 0x22, 0x23, 0x24,              # PageUp, PageDown, End, Home
    0x25, 0x26, 0x27, 0x28,              # arrows
    0x2C, 0x2D, 0x2E,                    # PrintScreen, Insert, Delete
    0x5B, 0x5C, 0x5D,                    # LWin, RWin, Apps
    0x6F, 0x90,                          # numpad Divide, NumLock
    0xA3, 0xA5,                          # RCtrl, RAlt
}

ULONG_PTR = ctypes.c_size_t


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT
user32.MapVirtualKeyW.argtypes = (wintypes.UINT, wintypes.UINT)
user32.MapVirtualKeyW.restype = wintypes.UINT


def _send(inp):
    pyautogui.failSafeCheck()
    if user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) != 1:
        raise ctypes.WinError(ctypes.get_last_error())


def _virtual_screen():
    gsm = user32.GetSystemMetrics
    return gsm(76), gsm(77), gsm(78), gsm(79)   # SM_X/YVIRTUALSCREEN, SM_CX/CYVIRTUALSCREEN


def cursor_pos():
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def send_mouse_move(x, y):
    """
    Move the cursor to screen pixel (x, y) with a real absolute mouse-move event.

    Absolute coordinates are normalised to 0..65535 over the whole virtual desktop;
    aiming at the pixel centre makes Windows' rounding land on the intended pixel.
    If it still misses (unusual DPI setups), the position is corrected with
    SetCursorPos so callers can always rely on the cursor being at (x, y).
    """
    x, y = int(x), int(y)
    left, top, width, height = _virtual_screen()
    nx = int((x - left + 0.5) * 65536 / width)
    ny = int((y - top + 0.5) * 65536 / height)
    inp = INPUT(type=INPUT_MOUSE)
    inp.u.mi = MOUSEINPUT(nx, ny, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, 0, 0)
    _send(inp)
    if cursor_pos() != (x, y):
        user32.SetCursorPos(x, y)


def scan_code(vk):
    """(scan code, is_extended) for a virtual-key code."""
    sc = user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC_EX)
    extended = (sc >> 8) in (0xE0, 0xE1) or vk in EXTENDED_VKS
    return sc & 0xFF, extended


def send_key(vk, up=False):
    """Press (up=False) or release a key by virtual-key code, with its real scan code."""
    sc, extended = scan_code(vk)
    flags = (KEYEVENTF_EXTENDEDKEY if extended else 0) | (KEYEVENTF_KEYUP if up else 0)
    inp = INPUT(type=INPUT_KEYBOARD)
    inp.u.ki = KEYBDINPUT(vk, sc, flags, 0, 0)
    _send(inp)


def key_vk(key):
    """
    (virtual-key code, needs_shift) for a pyautogui key name or character, or
    (None, False) if this keyboard layout can't produce it with one key.
    """
    mapping = pyautogui.platformModule.keyboardMapping
    if len(key) > 1:
        key = key.lower()
    code = mapping.get(key)
    if code is None or code < 0:
        return None, False
    mods, vk = divmod(code, 0x100)
    if mods & ~1:             # needs Ctrl/Alt (AltGr layouts) - not a single key + Shift
        return None, False
    return vk, bool(mods & 1)
