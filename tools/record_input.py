"""
Record mouse and keyboard events to CSV, to check what the agent actually sends (or
to record your own sessions and compare the two).

    python -m tools.record_input --out run.csv                # until Ctrl+C
    python -m tools.record_input --out run.csv --seconds 900  # e.g. alongside a 15-minute session
    python -m tools.record_input --summary run.csv            # statistics of a recording

Privacy: by default keys are stored only as a class (letter, digit, space, backspace,
enter, modifier, arrow, other) - enough for timing analysis, but the typed text can't
be read back from the file. --keep-keys stores the virtual-key and scan codes.
Only record your own sessions.

Columns: t (s since start), device, event, x, y, key (class or vk), scan, flags
(injected / extended), wheel (delta).
"""

import argparse
import csv
import ctypes
import sys
import time
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32")

HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
user32.SetWindowsHookExW.argtypes = (ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD)
user32.SetWindowsHookExW.restype = wintypes.HHOOK
user32.CallNextHookEx.argtypes = (wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
user32.CallNextHookEx.restype = ctypes.c_ssize_t

WH_KEYBOARD_LL, WH_MOUSE_LL = 13, 14
MOUSE_EVENTS = {0x200: "move", 0x201: "left_down", 0x202: "left_up", 0x204: "right_down", 0x205: "right_up",
                0x207: "middle_down", 0x208: "middle_up", 0x20A: "wheel", 0x20E: "hwheel"}
KEY_EVENTS = {0x100: "down", 0x101: "up", 0x104: "down", 0x105: "up"}   # WM_(SYS)KEYDOWN / UP


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vk", wintypes.DWORD), ("scan", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("extra", ctypes.c_size_t)]


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG), ("data", wintypes.DWORD),
                ("flags", wintypes.DWORD), ("time", wintypes.DWORD), ("extra", ctypes.c_size_t)]


def key_class(vk):
    """Coarse class of a virtual-key code (hides which character was typed)."""
    if 0x41 <= vk <= 0x5A:
        return "letter"
    if 0x30 <= vk <= 0x39 or 0x60 <= vk <= 0x69:
        return "digit"
    return {0x20: "space", 0x08: "backspace", 0x0D: "enter", 0x09: "tab", 0x1B: "esc",
            0x10: "shift", 0xA0: "shift", 0xA1: "shift", 0x11: "ctrl", 0xA2: "ctrl", 0xA3: "ctrl",
            0x12: "alt", 0xA4: "alt", 0xA5: "alt", 0x5B: "win", 0x5C: "win",
            0x25: "arrow", 0x26: "arrow", 0x27: "arrow", 0x28: "arrow",
            0x21: "page", 0x22: "page", 0x23: "home_end", 0x24: "home_end"}.get(vk, "other")


def record(out_path, seconds=None, keep_keys=False):
    t0 = time.perf_counter()
    f = open(out_path, "w", newline="", encoding="utf-8", buffering=1)   # line-buffered: survives being killed
    w = csv.writer(f)
    w.writerow(["t", "device", "event", "x", "y", "key", "scan", "injected", "extended", "wheel"])
    counts = {"mouse": 0, "keyboard": 0}

    @HOOKPROC
    def on_key(n, wparam, lparam):
        if n == 0 and wparam in KEY_EVENTS:
            k = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            w.writerow([f"{time.perf_counter() - t0:.4f}", "keyboard", KEY_EVENTS[wparam], "", "",
                        k.vk if keep_keys else key_class(k.vk), k.scan if keep_keys else "",
                        int(bool(k.flags & 0x10)), int(bool(k.flags & 0x01)), ""])
            counts["keyboard"] += 1
        return user32.CallNextHookEx(None, n, wparam, lparam)

    @HOOKPROC
    def on_mouse(n, wparam, lparam):
        if n == 0 and wparam in MOUSE_EVENTS:
            m = ctypes.cast(lparam, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
            wheel = ctypes.c_short(m.data >> 16).value if wparam in (0x20A, 0x20E) else ""
            w.writerow([f"{time.perf_counter() - t0:.4f}", "mouse", MOUSE_EVENTS[wparam], m.x, m.y, "", "",
                        int(bool(m.flags & 0x01)), "", wheel])
            counts["mouse"] += 1
        return user32.CallNextHookEx(None, n, wparam, lparam)

    hooks = [user32.SetWindowsHookExW(WH_KEYBOARD_LL, on_key, None, 0),
             user32.SetWindowsHookExW(WH_MOUSE_LL, on_mouse, None, 0)]
    if not all(hooks):
        sys.exit("Couldn't install the input hooks")
    if seconds:   # stop the message loop after `seconds`
        user32.SetTimer(None, 0, int(seconds * 1000), None)
    print(f"Recording to {out_path} {'for %.0f s' % seconds if seconds else 'until Ctrl+C'}...", flush=True)
    msg = wintypes.MSG()
    try:
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == 0x0113:         # WM_TIMER
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
    except KeyboardInterrupt:
        pass
    finally:
        for h in hooks:
            user32.UnhookWindowsHookEx(h)
        f.close()
    print(f"Recorded {counts['mouse']} mouse and {counts['keyboard']} keyboard events.")


def summary(path):
    """Print timing statistics of a recording."""
    import numpy as np
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        print("empty recording")
        return
    t = lambda r: float(r["t"])  # noqa: E731
    moves = [r for r in rows if r["event"] == "move"]
    print(f"{path}: {len(rows)} events over {t(rows[-1]) - t(rows[0]):.0f} s")
    if len(moves) > 1:
        dts = np.diff([t(r) for r in moves])
        bursts = dts[dts < 0.1]                   # gaps inside movements (not between them)
        print(f"  mouse moves: {len(moves)}, interval inside movements median {np.median(bursts) * 1000:.1f} ms "
              f"(~{1 / np.median(bursts):.0f} Hz)" if len(bursts) else f"  mouse moves: {len(moves)}")
    for button in ("left", "right"):
        downs = [t(r) for r in rows if r["event"] == f"{button}_down"]
        ups = [t(r) for r in rows if r["event"] == f"{button}_up"]
        holds = [min((u for u in ups if u > d), default=d) - d for d in downs]
        if downs:
            print(f"  {button} clicks: {len(downs)}, hold median {np.median(holds) * 1000:.0f} ms")
    wheels = [r for r in rows if r["event"] == "wheel"]
    if wheels:
        print(f"  wheel events: {len(wheels)}")
    keys = [r for r in rows if r["device"] == "keyboard"]
    if keys:
        downs = [r for r in keys if r["event"] == "down"]
        holds, pending = [], {}
        for r in keys:
            k = r["key"]
            if r["event"] == "down" and k not in pending:
                pending[k] = t(r)
            elif r["event"] == "up" and k in pending:
                holds.append(t(r) - pending.pop(k))
        iki = np.diff([t(r) for r in downs])
        iki = iki[iki < 2.0]                      # within typing bursts
        injected = np.mean([r["injected"] == "1" for r in keys]) * 100
        zero_scan = sum(1 for r in keys if r["scan"] == "0")
        print(f"  key presses: {len(downs)}, hold median {np.median(holds) * 1000:.0f} ms, "
              f"interval median {np.median(iki) * 1000:.0f} ms" if len(iki) else f"  key presses: {len(downs)}")
        classes = {}
        for r in downs:
            classes[r["key"]] = classes.get(r["key"], 0) + 1
        print("  key classes: " + ", ".join(f"{k} {v}" for k, v in sorted(classes.items(), key=lambda kv: -kv[1])))
        print(f"  injected flag on {injected:.0f}% of key events" +
              (f"; {zero_scan} events with scan code 0" if zero_scan else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="input_log.csv")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this long (default: Ctrl+C)")
    ap.add_argument("--keep-keys", action="store_true", help="store virtual-key / scan codes (reveals typed text)")
    ap.add_argument("--summary", metavar="CSV", help="print statistics of a recording instead of recording")
    args = ap.parse_args(argv)
    if args.summary:
        summary(args.summary)
    else:
        record(args.out, args.seconds, args.keep_keys)


if __name__ == "__main__":
    main()
