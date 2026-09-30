"""
Global settings for the controller package.

Timing policy: the base controller does NOT add any delays of its own beyond
the small "UI settle" waits needed for correctness (e.g. waiting for the Start
menu to open). Every function takes explicit timing parameters so the future
behaviour/entropy layer can supply its own values.
"""

import ctypes
import os

# --- DPI awareness -----------------------------------------------------------
# Make sure screen coordinates from pyautogui (mouse/screenshots) and from
# uiautomation (element rectangles) are both in *physical* pixels on EVERY
# monitor. Without this, display scaling (125%, 150%...) makes the two disagree
# and clicks land offset - especially on a second monitor with other scaling.
#
# This MUST run before `import pyautogui`: pyautogui calls SetProcessDPIAware()
# (system-level awareness only) on import, and Windows ignores any later change.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # fallback for older Windows
    except Exception:
        pass

# Some Python builds (e.g. the Microsoft Store python.exe) already declare
# system-level awareness in their manifest, which blocks the process-wide call
# above. Per-monitor awareness can still be set for the current thread
# (Windows 10 1607+); all controller calls run on the main thread.
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)
try:
    ctypes.windll.user32.SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
except Exception:
    pass

import pyautogui  # noqa: E402  (must come after the DPI setting above)

# --- pyautogui behaviour ----------------------------------------------------
# Fail-safe: slam the mouse into any screen corner to raise FailSafeException
# and stop the script. Keep this ON while developing.
pyautogui.FAILSAFE = True

# pyautogui normally sleeps 0.1 s after every call. Disable it so that the
# caller (timing layer) fully owns the pacing.
pyautogui.PAUSE = 0

# --- Default "UI settle" waits (seconds) -------------------------------------
# Minimum time for Windows UI to react. Override per call where needed.
START_MENU_OPEN_WAIT = 0.8     # after pressing the Windows key
START_MENU_SEARCH_WAIT = 1.0   # after typing the app name, before Enter
DIALOG_OPEN_WAIT = 1.0         # after triggering Save As / Open dialogs
DEFAULT_TIMEOUT = 15           # generic timeout for waiting on windows/elements
POLL_INTERVAL = 0.25           # how often wait_* functions re-check

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICONS_DIR = os.path.join(PROJECT_ROOT, "assets", "icons")
