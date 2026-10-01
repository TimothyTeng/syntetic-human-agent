"""
Save the page open in Chrome as a research page capture (the format of
tests/fixtures/pages/*.json) and show what the notes keep from it - for building test
fixtures and for checking extraction on a new site.

    python -m tools.capture_page                      # prints a summary only
    python -m tools.capture_page --save my_page       # also writes tests/fixtures/pages/my_page.json

Switch to Chrome within 3 s (UI Automation only exposes the page of a window in front).
Read-only: no keys are pressed and the mouse isn't moved.
"""

import argparse
import json
import os
import time

from research.extract import capture_page, parse_capture

PAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures", "pages")


def main():
    """Capture the page in front, print its headings, tables and the sections the notes
    keep, and save the capture with --save."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", default=None, help="file name (without .json) under tests/fixtures/pages")
    ap.add_argument("--wait", type=float, default=3.0, help="seconds to switch to Chrome")
    args = ap.parse_args()

    print(f"Switch to Chrome - reading the page in {args.wait:.0f} s ...")
    time.sleep(args.wait)
    t0 = time.monotonic()
    cap = capture_page()
    if cap is None:
        raise SystemExit("No page text found (is Chrome in front with a page loaded?)")
    print(f"Read {cap['total_words']} words via {cap['method']} in {time.monotonic() - t0:.1f} s")
    print(f"URL:   {cap['url']}\nTitle: {cap['title']}\nSite:  {cap['site']}")
    print(f"Headings ({len(cap['headings'])}):")
    for h in cap["headings"][:40]:                        # indented by level, as an outline
        print(f"  {'  ' * (h['level'] - 1)}h{h['level']} {h['text'][:70]}")
    print(f"Tables ({len(cap['tables'])}): " + ", ".join(f"{t['kind']} {len(t['rows'])}x{len(t['rows'][0])}"
                                                       for t in cap["tables"] if t["rows"]))
    src = parse_capture(cap)                              # what a research run would keep
    print(f"\nNotes: {src.title!r} - {src.words()} words kept")
    for s in src.sections:
        print(f"  [{s.heading or 'lead'}] {len(s.paras)} paragraphs, {s.words()} words"
              + (f" - {s.paras[0].text[:60]!r}" if s.paras else ""))
    if args.save:
        os.makedirs(PAGES, exist_ok=True)
        path = os.path.join(PAGES, args.save + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cap, f, indent=1, ensure_ascii=False)
        print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
