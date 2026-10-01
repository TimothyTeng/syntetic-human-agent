"""
Run research mode's self-checks on their own (optional: a research run makes the same
checks as it goes and writes them to diagnostics.json). Read-only: no keys are pressed
and the mouse isn't moved.

    python -m tools.probe_research chrome    # with an article (e.g. Wikipedia) open in Chrome
    python -m tools.probe_research writer    # with an OpenOffice Writer document open

Switch to the app within 3 s of starting.
"""

import json
import sys
import time

from controller import apps
from controller import writer as W
from research.diagnostics import page_report, writer_report
from research.extract import capture_page, parse_capture


def show(report, warnings):
    """Print a diagnostics report and its warnings."""
    print(json.dumps(report, indent=1, default=str))
    for w in warnings:
        print(f"Check: {w}")
    print("No problems found." if not warnings else "")


def probe_chrome():
    """Capture the Chrome page in front and report how it was read (diagnostics.page_report)."""
    t0 = time.monotonic()
    cap = capture_page()
    if cap is None:
        raise SystemExit("No page text found (is Chrome in front with a page loaded?)")
    print(f"Read in {time.monotonic() - t0:.1f} s")
    show(*page_report(cap, parse_capture(cap)))


def probe_writer():
    """Report what the Writer window in front exposes (diagnostics.writer_report)."""
    hwnd = apps.foreground_top_level()
    if not W.is_writer_window(hwnd):
        raise SystemExit("OpenOffice Writer isn't the window in front")
    win = apps.control_for(hwnd)
    show(*writer_report(win, W.document_control(win)))


def main():
    """Probe 'chrome' or 'writer' (the command-line argument) after a 3 s pause."""
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what not in ("chrome", "writer"):
        raise SystemExit(__doc__)
    print(f"Switch to {what.capitalize()} - checking in 3 s ...")
    time.sleep(3)
    probe_chrome() if what == "chrome" else probe_writer()


if __name__ == "__main__":
    main()
