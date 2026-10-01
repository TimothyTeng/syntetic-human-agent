"""
Live smoke test of the OpenOffice Writer side of research mode, in about three minutes:
opens a new Writer document, writes a mini paper containing every kind of block (title,
subtitle, headings, bold / italic runs, a font size, centred text, bullets, a table with
a bold header row, a caption, a page break, references), saves it as .odt, then opens the
saved file (its content.xml, no Writer needed) and checks that the styles and formatting
are what was meant.

    python -m tools.writer_smoke                       # persona's own keyboard/mouse mix
    python -m tools.writer_smoke --shortcut-pref 0.95  # nearly always shortcuts
    python -m tools.writer_smoke --shortcut-pref 0.05  # nearly always toolbar and menus

Don't touch the mouse or keyboard while it runs (abort: mouse into a screen corner).
"""

import argparse
import datetime
import os
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

from algorithms.behaviour import Human
from research.ir import (Bullets, Caption, Document, Heading, PageBreak, Paragraph, Run, Subtitle, Table, Title,
                         validate)
from research.project import block_text
from research.writer_exec import WriterExecutor
from research.writer_ops import render, style_align
from tasks import click_into_document, log, open_writer_document

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "smoke")
NS = {"office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
      "style": "urn:oasis:names:tc:opendocument:xmlns:style:1.0",
      "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
      "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
      "fo": "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"}


def q(name):
    """'text:p' -> '{urn:...text...}p' for ElementTree."""
    prefix, local = name.split(":")
    return f"{{{NS[prefix]}}}{local}"


def mini_paper():
    """A short ir.Document with one of every kind of block and formatting the paper uses. The
    abstract mentions "introductions", so typing the heading "Introduction" makes word
    completion offer the "s" - which Enter would accept if it weren't dismissed."""
    return Document([
        Title("Smoke Test Paper"),
        Subtitle("A short check of every block"),
        Paragraph([Run("Compiled for testing")], align="center", size=10),
        Heading("Abstract", 1),
        Paragraph([Run("This small paragraph checks the font size before the introductions.")], size=10),
        Paragraph([Run("Keywords:", bold=True), Run(" testing, Writer")], size=10),
        Heading("Introduction", 1),
        Paragraph([Run("The "), Run("smoke test", bold=True), Run(" writes a short paper [1].")]),
        Heading("Details", 2),
        Bullets([[Run("First:", bold=True), Run(" a bullet item.")],
                 [Run("Second:", bold=True), Run(" another one.")]]),
        Caption("Table 1. A small table."),
        Table([["Year", "Event"], ["1969", "First landing"], ["1972", "Last landing"]]),
        Paragraph([Run("A paragraph after the table.")]),
        PageBreak(),
        Heading("References", 1),
        Paragraph([Run("[1] "), Run("Smoke test page", italic=True), Run(". Example. Retrieved today.")], size=10),
    ])


# --- Reading the saved .odt ---------------------------------------------------------------------

def _props(style_el):
    """Direct formatting of an automatic style: {'parent', 'bold', 'italic', 'size', 'align',
    'break'} (None where the style doesn't set it)."""
    text = style_el.find(q("style:text-properties"))
    para = style_el.find(q("style:paragraph-properties"))
    get = lambda el, attr: el.get(q(attr)) if el is not None else None  # noqa: E731
    weight, posture, size = get(text, "fo:font-weight"), get(text, "fo:font-style"), get(text, "fo:font-size")
    align = get(para, "fo:text-align")
    return {"parent": style_el.get(q("style:parent-style-name")),
            "bold": None if weight is None else weight not in ("normal", "400"),
            "italic": None if posture is None else posture != "normal",
            "size": float(size[:-2]) if size and size.endswith("pt") else None,
            "align": {"start": "left", "end": "right", "justify": "justify"}.get(align, align),
            "break": get(para, "fo:break-before")}


def _style_name(name):
    """ODF style names encode spaces: 'Text_20_body' -> 'Text body'."""
    return (name or "").replace("_20_", " ")


def odt_document(path):
    """The saved document as a list of items in order: ('p', style, align, break, in_list,
    [(text, bold, italic, size)]) - alignment None and bold / italic / size None where not
    set directly - and ('table', [[(text, bold)]])."""
    root = ET.fromstring(zipfile.ZipFile(path).read("content.xml"))
    auto = {s.get(q("style:name")): _props(s) for s in root.iter(q("style:style"))}
    out = []

    def runs_of(el, inherited, acc):
        """Collect (text, bold, italic, size) runs of a paragraph element and its spans."""
        if el.text:
            acc.append([el.text, *inherited])
        for child in el:
            if child.tag == q("text:span"):
                p = auto.get(child.get(q("text:style-name")), {})
                runs_of(child, [p.get(k) if p.get(k) is not None else v
                                for k, v in zip(("bold", "italic", "size"), inherited)], acc)
            elif child.tag == q("text:s"):
                acc.append([" " * int(child.get(q("text:c"), "1")), *inherited])
            elif child.tag in (q("text:tab"), q("text:line-break")):
                acc.append([" ", *inherited])
            elif child.tag == q("text:a"):
                runs_of(child, inherited, acc)
            if child.tail:
                acc.append([child.tail, *inherited])
        return acc

    def paragraph(el, in_list):
        """One text:p / text:h as an item."""
        name = el.get(q("text:style-name"))
        p = auto.get(name)
        style = _style_name(p["parent"] if p and p["parent"] else name)
        p = p or {}
        runs = runs_of(el, [p.get("bold"), p.get("italic"), p.get("size")], [])
        merged = []
        for text, bold, italic, size in runs:          # neighbours with the same formatting joined
            if merged and merged[-1][1:] == [bold, italic, size]:
                merged[-1][0] += text
            elif text:
                merged.append([text, bold, italic, size])
        return ("p", style, p.get("align"), p.get("break"), in_list, [tuple(r) for r in merged])

    def walk(el, in_list=False):
        """The body's paragraphs, lists and tables in document order."""
        for child in el:
            if child.tag in (q("text:p"), q("text:h")):
                out.append(paragraph(child, in_list))
            elif child.tag in (q("text:list"), q("text:list-item")):
                walk(child, True)
            elif child.tag == q("table:table"):
                rows = []
                for row in child.iter(q("table:table-row")):
                    cells = []
                    for cell in row.findall(q("table:table-cell")):
                        ps = [paragraph(p, False) for p in cell.findall(q("text:p"))]
                        runs = [r for p in ps for r in p[5]]
                        cells.append(("".join(r[0] for r in runs), any(r[1] for r in runs)))
                    rows.append(cells)
                out.append(("table", rows))

    walk(root.find(q("office:body")).find(q("office:text")))
    return out


def check(path):
    """Problems found in the saved .odt (empty list = everything is as meant).

    Each check guards against a mistake the executor could make while formatting by
    keyboard or toolbar: a wrong or missing style, formatting that leaked into the next
    run or paragraph, a lost page break, a table typed wrongly, a style's own formatting
    toggled off, a word completion accepted."""
    items = odt_document(path)
    paras = [it for it in items if it[0] == "p"]
    problems = []

    def find(text):
        """The first paragraph whose text starts with (the first 25 characters of) text, or None."""
        return next((p for p in paras if "".join(r[0] for r in p[5]).startswith(text[:25])), None)

    def text_of(p):
        return "".join(r[0] for r in p[5])

    # every styled block has its style, its text exactly (no word completion accepted), the
    # style's own alignment (the centring of the date line didn't carry over) and no
    # direct bold / italic (e.g. Ctrl+I pressed in the italic Heading 2 would switch it off)
    expect = [("Smoke Test Paper", "Title"), ("A short check of every block", "Subtitle"), ("Abstract", "Heading 1"),
              ("Introduction", "Heading 1"), ("Details", "Heading 2"), ("Table 1. A small table.", "Caption"),
              ("References", "Heading 1")]
    for text, style in expect:
        p = next((p for p in paras if text_of(p) == text), None)
        if p is None:
            near = find(text)
            problems.append(f"missing paragraph {text!r}" + (f" (found {text_of(near)!r})" if near else ""))
            continue
        if p[1] != style:
            problems.append(f"{text!r}: style {p[1]} (meant {style})")
        if (p[2] or style_align(style)) != style_align(style):
            problems.append(f"{text!r}: alignment {p[2]} (meant the style's own, {style_align(style)})")
        if any(r[1] is not None or r[2] is not None for r in p[5]):
            problems.append(f"{text!r}: direct bold / italic on a styled paragraph {p[5]}")
        if p[4]:
            problems.append(f"{text!r}: has bullets")
    # direct paragraph / font formatting: centring and a font size were applied
    p = find("Compiled for testing")
    if p and (p[2] != "center" or p[5][0][3] != 10):
        problems.append(f"date line: alignment {p[2]}, size {p[5][0][3]} (meant centre, 10 pt)")
    for text in ("This small paragraph", "Keywords:", "[1] Smoke"):
        p = find(text)
        if p is None:
            problems.append(f"missing paragraph {text!r}")
        elif any(r[3] != 10 for r in p[5]) or p[1] != "Text body":
            problems.append(f"{text!r}: style {p[1]}, sizes {[r[3] for r in p[5]]} (meant Text body, 10 pt)")
    # bold switched on for the label and off again before the rest
    p = find("Keywords:")
    if p and not (p[5][0][1] and not p[5][-1][1]):
        problems.append(f"'Keywords:' bold / rest plain wrong: {p[5]}")
    # bold in mid-paragraph covers exactly its words (not the spaces or text around them)
    p = find("The smoke test")
    if p and [r[0] for r in p[5] if r[1]] != ["smoke test"]:
        problems.append(f"bold run wrong: {p[5]}")
    # both items are bulleted body text, with a bold label
    bullets = [p for p in paras if text_of(p).startswith(("First:", "Second:"))]
    if len(bullets) != 2 or not all(b[4] and b[1] == "Text body" and b[5][0][1] for b in bullets):
        problems.append(f"bullets wrong: {[(b[1], b[4], b[5][0]) for b in bullets]}")
    # italic covers exactly the reference's title
    p = find("[1] Smoke")
    if p and [r[0] for r in p[5] if r[2]] != ["Smoke test page"]:
        problems.append(f"italic title wrong: {p[5]}")
    # one table, every cell in its place (no text typed into the wrong cell or after the table)
    tables = [it[1] for it in items if it[0] == "table"]
    if len(tables) != 1:
        problems.append(f"{len(tables)} tables (meant 1)")
    else:
        cells = [c for row in tables[0] for c in row]
        if [c[0] for c in cells] != ["Year", "Event", "1969", "First landing", "1972", "Last landing"]:
            problems.append(f"table cells {[c[0] for c in cells]}")
        if [c[1] for c in cells] != [True, True, False, False, False, False]:
            problems.append(f"table bold cells {[c[1] for c in cells]} (meant header row only)")
    # leaving the table goes back to a plain paragraph
    p = find("A paragraph after")
    if p and (p[1] != "Text body" or p[4] or p[2] not in (None, "left")):
        problems.append(f"paragraph after the table: style {p[1]}, bullets {p[4]}, alignment {p[2]}")
    # the page break survived (it sits on the References heading, or an empty paragraph before it)
    i = next((k for k, it in enumerate(items) if it[0] == "p" and text_of(it) == "References"), None)
    if i is not None and not any(it[0] == "p" and it[3] == "page" and (k == i or not text_of(it))
                                 for k, it in enumerate(items[:i + 1]) if k >= i - 1):
        problems.append("no page break before 'References'")
    return problems


def main():
    """Write the mini paper into a new Writer document as a person would, save it, check the
    .odt and exit with 1 if anything is wrong."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shortcut-pref", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--wpm", type=float, default=90, help="typing speed (fast, to keep the test short)")
    args = ap.parse_args()

    doc = mini_paper()
    assert not validate(doc), validate(doc)
    human = Human(seed=args.seed, wpm=args.wpm, shortcut_pref=args.shortcut_pref, pause_scale=0.3)
    log(f"Persona: {human.persona.describe()}")
    log("Starting in 3 s - don't touch the mouse/keyboard.")
    time.sleep(3)
    path = os.path.join(OUT, f"smoke-{datetime.datetime.now():%Y%m%d-%H%M%S}.odt")
    win, writer_doc = open_writer_document(human)
    click_into_document(human, win, writer_doc)
    ex = WriterExecutor(human, win, writer_doc, save_path=path, log=log,
                        refocus=lambda: click_into_document(human, win, ex.doc or writer_doc))
    from controller import writer as W
    mismatches = []

    def checkpoint(_, i):
        """After block i: note it if the document doesn't end with the block's text."""
        got = W.normalise(W.document_text(ex.doc or writer_doc))
        meant = W.normalise(block_text(doc.blocks[i]))
        if meant and not got.endswith(meant):
            mismatches.append(i)
            log(f"Block {i} text differs: meant ...{meant[-60:]!r}, got ...{got[-60:]!r}")

    ex.run(render(doc), on_checkpoint=checkpoint)
    log(f"Methods: {ex.summary()}")
    time.sleep(2)                        # let Writer finish writing the file
    problems = check(path) if os.path.exists(path) else [f"{path} was not saved"]
    problems += ex.problems
    for i in mismatches:                 # a typo left in / a lost key: logged, not a formatting failure
        log(f"WARNING: the text of block {i} differs from what was meant")
    for p in problems:
        log(f"PROBLEM: {p}")
    log("SMOKE TEST PASSED" if not problems else f"SMOKE TEST FAILED ({len(problems)} problems) - {path}")
    close_writer(human, win)
    sys.exit(1 if problems else 0)


def close_writer(human, win):
    """Close the test document (it is saved; a 'Save changes?' question gets Discard)."""
    from controller import apps
    if not apps.bring_to_front(win.NativeWindowHandle):
        return
    time.sleep(0.5)
    human.hotkey("ctrl", "w")            # Window > Close Window: just this document
    time.sleep(1.5)
    if apps.foreground_top_level() != win.NativeWindowHandle and apps.window_class(
            apps.foreground_top_level()) == "SALSUBFRAME":
        human.answer_prompt(r"^Discard$")


if __name__ == "__main__":
    main()
