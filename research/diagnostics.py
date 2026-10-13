"""
Self-checks recorded during a research run, so no separate probing is needed first.

Every page read and the Writer window are checked as they come up, through the same
read-only UI Automation queries tools/probe_research.py makes:
  pages  how the article text was found (main landmark / article / whole page), how many
         headings and with which levels, which tables, how much usable text was kept
  writer whether Writer reports the paragraph style and font size at the caret and the
         document text (needed to verify formatting and typing), whether the Formatting
         toolbar's buttons and boxes and the menus are found by name (needed by
         mouse-minded people)
  writing which method each operation used, what had to fall back, text mismatches

Problems are logged as they are found ("Check: ...") and everything is written to
diagnostics.json in the run folder; summary() gives the end-of-run verdict.
"""

import json
import os
from collections import Counter

from controller import apps
from controller import writer as W

WEAK_METHODS = ("document", "text-nodes")         # capture methods that read more than the article


def page_report(cap, src):
    """What reading one page found. (report dict, [warnings]).

    Warns when the article element wasn't found (navigation text may end up in the notes),
    when the page has no headings (sections were guessed), when every heading is level 2
    (Chrome may not expose levels, so subsections look like sections) and when little
    usable text was kept."""
    levels = Counter(h["level"] for h in cap.get("headings", []))
    report = {
        "url": cap.get("url"), "method": cap.get("method"), "words_on_page": cap.get("total_words"),
        "headings": sum(levels.values()), "heading_levels": {f"h{k}": v for k, v in sorted(levels.items())},
        "tables": [f"{t['kind']} {len(t['rows'])}x{len(t['rows'][0]) if t['rows'] else 0}"
                   for t in cap.get("tables", [])],
        "sections_kept": [s.heading or "(lead)" for s in src.sections], "words_kept": src.words(),
    }
    warnings = []
    site = src.site
    if cap.get("method") in WEAK_METHODS:
        warnings.append(f"{site}: the article element wasn't found - read the whole page (menus may slip in)")
    if not levels:
        warnings.append(f"{site}: the page exposes no headings - sections were guessed from short lines")
    elif set(levels) == {2} and levels[2] > 3:            # _heading_level()'s default for an unknown level
        warnings.append(f"{site}: every heading came back as level 2 - heading levels may not be exposed")
    if src.words() < 150:
        warnings.append(f"{site}: only {src.words()} usable words kept from {cap.get('total_words')} on the page")
    return report, warnings


def writer_report(win, doc):
    """What Writer exposes for checking and clicking. (report dict, [warnings]).

    caret_style / caret_font_size: whether the style and font size at the caret can be read
    back from the Formatting toolbar (needed to verify those steps). document_text: whether
    the paragraphs on screen can be read (needed to check the typed text). buttons /
    style_box / menus: whether the toolbar controls and menus a mouse user clicks are found
    by name; if not, the executor falls back to keyboard shortcuts."""
    text = W.document_text(doc)
    report = {
        "window": apps.window_title(win.NativeWindowHandle), "document_has_focus": W.document_has_focus(win, doc),
        "caret_style": W.caret_paragraph_style(win), "caret_font_size": W.caret_font_size(win),
        "caret_in_table": W.caret_in_table(doc), "document_text": text is not None,
        "formatting_toolbar": W.formatting_toolbar(win) is not None,
        "buttons": {b: bool(W.toolbar_button(win, b)) for b in ("Bold", "Italic", "Align Left", "Cent(red|ered)",
                                                                  "Bullets On/Off")},
        "save_button": bool(W.toolbar_button(win, "Save", "Standard")),
        "style_box": bool(W.style_box(win)), "font_size_box": bool(W.font_size_box(win)),
        "menus": {m: bool(W.menu(win, m)) for m in ("Insert", "Format")},
    }
    warnings = []
    if report["caret_style"] is None:
        warnings.append("Writer doesn't report paragraph styles (Formatting toolbar hidden?) - style changes "
                        "can't be verified")
    if text is None:
        warnings.append("the document's text can't be read - typed text can't be checked")
    if not report["formatting_toolbar"]:
        warnings.append("the Formatting toolbar wasn't found (View > Toolbars) - clicks fall back to shortcuts")
    elif not all(report["buttons"].values()):
        warnings.append("some Formatting toolbar buttons weren't found (narrow window or another language) - "
                        "clicks fall back to shortcuts")
    if not all(report["menus"].values()):
        warnings.append("the Insert / Format menus weren't found by name - menu clicks fall back to shortcuts")
    return report, warnings


class Diagnostics:
    """The self-checks of one research run, collected in diagnostics.json as they happen."""

    def __init__(self, folder, log=print):
        """Checks for the run in `folder`; warnings are also passed to `log`."""
        self.path = os.path.join(folder, "diagnostics.json")
        self.log = log
        self.data = {"pages": [], "writer": None, "writing": {"methods": Counter(), "problems": [],
                                                           "text_mismatches": 0, "blocks_checked": 0},
                     "warnings": []}

    def warn(self, msg):
        """Record and log a warning (each distinct message once)."""
        if msg not in self.data["warnings"]:
            self.data["warnings"].append(msg)
            self.log(f"Check: {msg}")

    def page(self, cap, src):
        """Check a page just read (its capture and the notes taken from it)."""
        report, warnings = page_report(cap, src)
        self.data["pages"].append(report)
        for w in warnings:
            self.warn(w)
        self.save()

    def writer(self, win, doc):
        """Check the Writer window - once per run, the first time writing starts."""
        if self.data["writer"] is not None or win is None or doc is None:
            return
        try:
            report, warnings = writer_report(win, doc)
        except Exception as exc:                  # a check must never stop the run
            report, warnings = {"error": repr(exc)}, [f"Writer check failed: {exc!r}"]
        self.data["writer"] = report
        for w in warnings:
            self.warn(w)
        self.save()

    def writing(self, ex):
        """Add a writing stint's results: how each operation was done ("bold:key",
        "style:ribbon", "fix:...", ...) and the formatting problems the executor found."""
        self.data["writing"]["methods"].update(ex.methods)
        for p in ex.problems:
            self.data["writing"]["problems"].append(p)
            self.warn(f"writing: {p}")
        self.save()

    def block_checked(self, ok):
        """Count a block whose text was compared with the document (ok: it matched)."""
        w = self.data["writing"]
        w["blocks_checked"] += 1
        w["text_mismatches"] += 0 if ok else 1

    def save(self):
        """Write diagnostics.json (Counters are saved as plain dicts)."""
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1, default=dict)

    def summary(self):
        """One line per area: pages, Writer, writing."""
        pages = self.data["pages"]
        good = sum(1 for p in pages if p["method"] not in WEAK_METHODS)   # the article element was found
        with_heads = sum(1 for p in pages if p["headings"])
        lines = [f"Pages: {len(pages)} read, article found on {good}, headings on {with_heads}"]
        writer = self.data["writer"]
        if writer and "error" not in writer:              # "toolbar OK": the Formatting toolbar's buttons were found
            lines.append(f"Writer: style read-back {'OK' if writer['caret_style'] else 'not available'}, text "
                         f"read-back {'OK' if writer['document_text'] else 'not available'}, toolbar "
                         f"{'OK' if all(writer['buttons'].values()) else 'partly hidden'}")
        w = self.data["writing"]
        if w["blocks_checked"]:
            lines.append(f"Writing: {w['blocks_checked'] - w['text_mismatches']}/{w['blocks_checked']} blocks matched, "
                         f"{len(w['problems'])} formatting problems")
        lines.append(f"{len(self.data['warnings'])} warnings - details in {self.path}")
        return lines
