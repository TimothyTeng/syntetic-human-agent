"""
Check that this machine can run the agent - before a run, on any new device.

    python -m tools.preflight            # static checks (nothing is opened or changed)
    python -m tools.preflight --launch   # also opens OpenOffice Writer and Chrome briefly to
                                         # check what they expose to UI Automation

Each line is OK, WARN (works, with less variety or slower) or FAIL (that part of the run
will not work), with what to do about it. Exit code 1 if anything failed.

What is checked:
  Windows / Python   Windows 10 or 11, Python 3.10+, the packages in requirements.txt
  Models             the trained mouse and typing models (without them simple fallbacks are used)
  Display            screen size, display scaling, and that mouse and UI Automation coordinates
                     agree (DPI awareness)
  Language           Windows display language and keyboard layout - names of buttons and menus
                     are matched in English; other languages fall back to keyboard shortcuts
  Chrome             installed, in the Start menu (the agent opens it like a person)
  OpenOffice         Writer installed, in the Start menu, version, accessibility bridge registered
  --launch           Writer exposes its document (needed to check typing and use the toolbar;
                     without it Writer is driven by keyboard only); Chrome exposes page text
                     (needed to read pages, follow links and trace words)
"""

import argparse
import ctypes
import glob
import importlib
import os
import platform
import subprocess
import sys
import time
import winreg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS = []


def report(level, what, detail="", fix=""):
    RESULTS.append(level)
    print(f"[{level:4}] {what}" + (f" - {detail}" if detail else ""))
    if fix and level != "OK":
        print(f"        -> {fix}")


def app_path(exe):
    """Install path of a program from the App Paths registry key (how Windows' Run dialog
    finds it), or None."""
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for view in (0, winreg.KEY_WOW64_32KEY):
            try:
                with winreg.OpenKey(hive, rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}",
                                    0, winreg.KEY_READ | view) as k:
                    path = winreg.QueryValue(k, None).strip('"')
                    if os.path.exists(path):
                        return path
            except OSError:
                continue
    return None


def start_menu_shortcut(pattern):
    """A Start-menu shortcut matching the glob pattern (all users or this user), or None."""
    roots = [os.path.join(os.environ.get("PROGRAMDATA", r"C:\ProgramData"), r"Microsoft\Windows\Start Menu\Programs"),
             os.path.join(os.environ.get("APPDATA", ""), r"Microsoft\Windows\Start Menu\Programs")]
    for root in roots:
        hits = glob.glob(os.path.join(root, "**", pattern), recursive=True)
        if hits:
            return hits[0]
    return None


# --- checks -----------------------------------------------------------------------------------

def check_system():
    ver = sys.getwindowsversion()
    name = f"Windows {platform.release()} (build {ver.build})"
    if ver.major < 10:
        report("FAIL", "Windows", name, "Windows 10 or 11 is needed")
    else:
        report("OK", "Windows", name)
    py = f"{sys.version.split()[0]} ({platform.architecture()[0]})"
    if sys.version_info < (3, 10):
        report("FAIL", "Python", py, "install Python 3.10 or newer")
    else:
        report("OK", "Python", py)
    missing = []
    for mod, pkg in (("pyautogui", "pyautogui"), ("uiautomation", "uiautomation"), ("pyperclip", "pyperclip"),
                     ("numpy", "numpy"), ("psutil", "psutil"), ("PIL", "pillow")):
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pkg)
    if missing:
        report("FAIL", "Python packages", "missing: " + ", ".join(missing), "pip install -r requirements.txt")
    else:
        report("OK", "Python packages", "all required packages import")


def check_models():
    models = os.path.join(ROOT, "models")
    for name, what in (("mouse_mdn.npz", "mouse movement model"), ("typing_mdn.npz", "typing timing model"),
                       ("typing_errors.json", "typo statistics"), ("scroll_stats.json", "scroll timing")):
        path = os.path.join(models, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            report("OK", f"Model: {what}", name)
        else:
            report("WARN", f"Model: {what}", f"{name} missing - a simpler built-in fallback is used",
                   "copy the models/ folder from the repository (git pull), or retrain (see ALGORITHMS.md)")


def check_display():
    importlib.import_module("controller.config")     # sets DPI awareness the way the agent does
    import pyautogui
    from controller import screen
    user32 = ctypes.windll.user32
    w, h = pyautogui.size()

    class DEVMODE(ctypes.Structure):
        _fields_ = [("dmDeviceName", ctypes.c_wchar * 32), ("dmSpecVersion", ctypes.c_ushort),
                    ("dmDriverVersion", ctypes.c_ushort), ("dmSize", ctypes.c_ushort),
                    ("dmDriverExtra", ctypes.c_ushort), ("dmFields", ctypes.c_ulong), ("_pos", ctypes.c_long * 2),
                    ("dmDisplayOrientation", ctypes.c_ulong), ("dmDisplayFixedOutput", ctypes.c_ulong),
                    ("dmColor", ctypes.c_short), ("dmDuplex", ctypes.c_short), ("dmYResolution", ctypes.c_short),
                    ("dmTTOption", ctypes.c_short), ("dmCollate", ctypes.c_short),
                    ("dmFormName", ctypes.c_wchar * 32), ("dmLogPixels", ctypes.c_ushort),
                    ("dmBitsPerPel", ctypes.c_ulong), ("dmPelsWidth", ctypes.c_ulong), ("dmPelsHeight", ctypes.c_ulong),
                    ("dmDisplayFlags", ctypes.c_ulong), ("dmDisplayFrequency", ctypes.c_ulong)]
    dm = DEVMODE()
    dm.dmSize = ctypes.sizeof(DEVMODE)
    physical = (dm.dmPelsWidth, dm.dmPelsHeight) if user32.EnumDisplaySettingsW(None, -1, ctypes.byref(dm)) else None
    try:
        scale = ctypes.windll.shcore.GetScaleFactorForDevice(0)
    except Exception:
        scale = None
    detail = f"{w}x{h}" + (f", scaling {scale}%" if scale else "") + f", {len(screen.monitor_rects())} monitor(s)"
    if physical and (w, h) != physical:
        report("FAIL", "Display coordinates", f"mouse sees {w}x{h} but the screen is {physical[0]}x{physical[1]}",
               "clicks would land off target - run with python.org Python, or set display scaling to 100%")
    elif w < 1280 or h < 720:
        report("WARN", "Display", detail + " - small screens crowd toolbars and pages",
               "use at least 1280x720 (windows are maximised automatically on small screens)")
    else:
        report("OK", "Display", detail)


def check_language():
    lang = ctypes.windll.kernel32.GetUserDefaultUILanguage()
    if lang & 0x3FF == 0x09:
        report("OK", "Windows language", f"English (0x{lang:04x})")
    else:
        report("WARN", "Windows language", f"not English (0x{lang:04x}) - buttons and menus are looked up by "
               "their English names; other languages fall back to keyboard shortcuts (less variety)",
               "an English Windows / Chrome / OpenOffice gives the full mix of mouse and keyboard actions")
    layout = ctypes.windll.user32.GetKeyboardLayout(0) & 0xFFFF
    if layout in (0x0409, 0x0809, 0x1009, 0x0c09, 0x1409, 0x4809):     # US, UK, CA, AU, NZ, SG English
        report("OK", "Keyboard layout", f"0x{layout:04x}")
    else:
        report("WARN", "Keyboard layout", f"0x{layout:04x} - text is typed per layout, but shortcuts such as "
               "Ctrl+] may sit on other keys", "an English (US/UK) layout is safest")


def check_apps():
    chrome = app_path("chrome.exe")
    if chrome:
        report("OK", "Google Chrome", chrome)
    else:
        report("FAIL", "Google Chrome", "not installed", "install Google Chrome")
    if not start_menu_shortcut("Google Chrome*.lnk"):
        report("WARN", "Chrome in the Start menu", "no shortcut found - the Run dialog is used instead",
               "reinstall Chrome for all users, or pin it")
    writer = app_path("swriter.exe")
    if writer:
        version = ""
        ini = os.path.join(os.path.dirname(writer), "version.ini")
        if os.path.exists(ini):
            for line in open(ini, encoding="latin-1"):
                if line.startswith("OOOBaseVersion=") or line.startswith("ProductVersion="):
                    version = line.split("=", 1)[1].strip()
                    break
        report("OK", "OpenOffice Writer", writer + (f" (version {version})" if version else ""))
        check_accessibility_bridge(os.path.dirname(writer))
        if "libreoffice" in writer.lower():
            report("WARN", "Office suite", "this is LibreOffice - the agent targets Apache OpenOffice 4.1 "
                   "(style names and toolbars differ)", "install Apache OpenOffice 4.1 (openoffice.org)")
    else:
        report("FAIL", "OpenOffice Writer", "not installed", "install Apache OpenOffice 4.1 (openoffice.org)")
    if not start_menu_shortcut("OpenOffice Writer*.lnk"):
        report("WARN", "Writer in the Start menu", "no 'OpenOffice Writer' shortcut - the Run dialog is used instead")
    timeout = os.environ.get("AGENT_TIMEOUT_SCALE")
    report("OK", "Timeouts", f"scaled x{timeout}" if timeout else "normal (set AGENT_TIMEOUT_SCALE=2 on slow machines)")


def accessibility_bridge_registered():
    """True if OpenOffice's accessibility bridge (UAccCOM.dll) is registered as a COM server -
    without it Writer shows UI Automation only its title bar."""
    for view in (winreg.KEY_WOW64_32KEY, winreg.KEY_WOW64_64KEY):
        try:
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, "CLSID", 0, winreg.KEY_READ | view) as clsids:
                i = 0
                while True:
                    try:
                        name = winreg.EnumKey(clsids, i)
                    except OSError:
                        break
                    i += 1
                    try:
                        with winreg.OpenKey(clsids, name + r"\InprocServer32") as server:
                            if "uacccom.dll" in winreg.QueryValue(server, None).lower():
                                return True
                    except OSError:
                        pass
        except OSError:
            pass
    return False


def check_accessibility_bridge(program_dir):
    dll = os.path.join(program_dir, "UAccCOM.dll")
    if not os.path.exists(dll):
        report("WARN", "Writer accessibility bridge", "UAccCOM.dll missing - Writer is driven by keyboard only "
               "(no read-back checks; Title style and font sizes left out)",
               "reinstall Apache OpenOffice 4.1 with its default components")
    elif not accessibility_bridge_registered():
        report("WARN", "Writer accessibility bridge", "UAccCOM.dll is not registered - Writer is driven by keyboard "
               "only (no read-back checks; Title style and font sizes left out)",
               rf'optional, from an administrator prompt: C:\Windows\SysWOW64\regsvr32.exe "{dll}"')
    else:
        report("OK", "Writer accessibility bridge", "UAccCOM.dll registered")


def check_writer_live():
    """Open Writer (a separate window), see whether its document is exposed, close it again."""
    from controller import apps
    from controller import writer as W
    import tasks
    path = app_path("swriter.exe")
    if not path:
        return
    running = tasks.openoffice_processes()
    if running:
        report("WARN", "OpenOffice already running", f"{len(running)} process(es)"
               + (f", {len(tasks.visible_openoffice_windows())} window(s) open" if tasks.visible_openoffice_windows()
                  else ", no window (leftover or Quickstarter)")
               + " - OpenOffice only shows UI Automation its FIRST window, so a Writer window opened now is "
                 "unreadable", "close every OpenOffice window and the Quickstarter (tray icon) before a run; "
                               "a leftover without windows is ended by the agent itself")
    before = set(W.writer_windows())
    subprocess.Popen([path])
    win = None
    deadline = time.time() + 60
    while time.time() < deadline and win is None:
        new = [h for h in W.writer_windows() if h not in before]
        win = new[0] if new else None
        time.sleep(0.5)
    if win is None:
        report("FAIL", "Writer starts", "no Writer window within 60 s", "start OpenOffice once by hand (first-run wizard)")
        return
    time.sleep(6)                                    # let it finish starting
    ctrl = apps.control_for(win)
    W.request_accessibility(win)
    doc = W.document_control(ctrl, timeout=8)
    toolbar = W.formatting_toolbar(ctrl) is not None
    if doc is not None:
        report("OK", "Writer accessibility", "document exposed" + ("" if toolbar else " (Formatting toolbar hidden)"),
               "View > Toolbars > Formatting")
        if not toolbar:
            report("WARN", "Writer Formatting toolbar", "not shown - formatting falls back to shortcuts",
                   "View > Toolbars > Formatting")
    else:
        report("WARN", "Writer accessibility", "Writer shows UI Automation only its title bar - the agent then "
               "writes by keyboard only, without checks, styles without shortcuts or font sizes",
               "close every OpenOffice window (and the Quickstarter in the tray), then run this check again - "
               "OpenOffice 4.1 only exposes the first window of a running OpenOffice"
               if accessibility_bridge_registered() else
               "the accessibility bridge isn't registered (see 'Writer accessibility bridge' above)")
    ctypes.windll.user32.PostMessageW(win, 0x10, 0, 0)          # close it (an empty document: no prompt)


def check_chrome_live():
    """Open a Chrome window on a Wikipedia page and see whether the page text is exposed."""
    from controller import browser
    path = app_path("chrome.exe")
    if not path:
        return
    subprocess.Popen([path, "--new-window", "https://en.wikipedia.org/wiki/Photosynthesis"])
    time.sleep(8)
    words = 0
    for _ in range(8):
        words = browser.page_word_count()
        if words > 50:
            break
        time.sleep(1)
    if words > 50:
        report("OK", "Chrome page text", f"{words} words in view")
    else:
        report("WARN", "Chrome page text", "the page's text wasn't exposed - reading falls back to estimates, no "
               "word tracing", "make sure the Chrome window is in front; check chrome://accessibility")
    win = browser.get_chrome_window(timeout=2)
    if win is not None:
        if browser.address_bar(win) is None:
            report("FAIL", "Chrome address bar", "not found", "update Chrome")
        else:
            report("OK", "Chrome address bar", "found")
        ctypes.windll.user32.PostMessageW(win.NativeWindowHandle, 0x10, 0, 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--launch", action="store_true", help="also open Writer and Chrome to check them live")
    args = ap.parse_args(argv)
    print("Preflight for the synthetic human agent\n")
    check_system()
    check_models()
    check_display()
    check_language()
    check_apps()
    if args.launch:
        print("\nLive checks (opening Writer and Chrome for a moment):")
        check_writer_live()
        check_chrome_live()
    fails, warns = RESULTS.count("FAIL"), RESULTS.count("WARN")
    print(f"\n{fails} failed, {warns} warnings" + ("" if args.launch else " (add --launch for the live checks)"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
