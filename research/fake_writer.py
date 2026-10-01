"""
An offline stand-in for OpenOffice Writer that executes writer_ops operations, for tests
and dry runs.

It models the parts of Writer's behaviour the renderer relies on (each checked in the
real Writer 4.1):
  * Enter starts a paragraph in the style's next style (Title -> Subtitle, Subtitle and
    headings -> Text body, Caption and Text body continue) and carries bullets, direct
    character formatting (including an explicit "not bold"), font size and direct
    alignment along - a style's own alignment is not carried
  * bold / italic and bullets are toggles (an operation that would switch them off by
    accident shows up as wrong formatting); the alignment keys set the alignment
  * applying a style clears direct alignment (Title / Subtitle are centred by their
    style), and a heading style drops bullets; it keeps direct character formatting -
    typing into a Title / Subtitle / heading / caption with direct formatting still at
    the caret is asserted not to happen (it would e.g. un-bold the heading)
  * Default Formatting (Ctrl+M) clears direct character formatting and alignment -
    asserted to happen only in an empty paragraph; bullets stay
  * a new table cell starts unformatted; Tab in a table's last cell adds a row;
    Ctrl+End leaves the table for the paragraph after it

    fw = FakeWriter(); fw.run(render(doc)); fw.result() == expected(doc)
"""

from .ir import Bullets, Caption, Heading, PageBreak, Paragraph, Subtitle, Table, Title
from .writer_ops import BODY, NEXT_STYLE, START_STYLE, block_style, style_align


class FakeWriter:
    """A document model driven by the same operations as the real Writer. Assertions fail on
    operations that would go wrong in Writer (toggling into the wrong state, Ctrl+M in a
    paragraph with text, Enter in a table cell, a table inserted into a paragraph with text...)."""

    def __init__(self):
        """A new document: one empty Default paragraph with the caret in it."""
        self.items = [self._para(START_STYLE)]   # paragraphs, tables and page breaks in order
        self.cur = self.items[0]                  # paragraph (or cell paragraph) with the caret
        self.table = None                         # (table item, row, col) while in a table
        self.bold = self.italic = False
        self.size = None
        self.direct = False                       # direct character formatting at the caret
        self.outside = None                       # the caret's formatting outside the table

    @staticmethod
    def _para(style, align=None, bullets=False):
        """A new empty paragraph (align: direct alignment, None = the style's); runs are
        [text, bold, italic, size] lists."""
        return {"kind": "p", "style": style, "align": align, "bullets": bullets, "runs": []}

    def run(self, ops):
        """Execute operations in order (op_<kind>); returns self for chaining."""
        for op in ops:
            getattr(self, "op_" + op.kind)(op)
        return self

    # --- operations -------------------------------------------------------------------------

    def op_style(self, op):
        """Apply a paragraph style: direct alignment goes; a heading style replaces bullets."""
        self.cur["style"] = op.a
        self.cur["align"] = None
        if op.a.startswith("Heading"):
            self.cur["bullets"] = False

    def op_bullets(self, op):
        """Shift+F12 / Bullets On/Off: flip bullets; op.a is the state expected afterwards."""
        self.cur["bullets"] = not self.cur["bullets"]
        assert self.cur["bullets"] == op.a, "bullets toggled into the wrong state"

    def op_bold(self, op):
        """Ctrl+B: flip bold; op.a is the state the renderer expects afterwards."""
        self.bold = not self.bold                 # Ctrl+B toggles
        self.direct = True                        # "not bold" is direct formatting too
        assert self.bold == op.a, "bold toggled into the wrong state"

    def op_italic(self, op):
        """Ctrl+I: flip italic; op.a is the state the renderer expects afterwards."""
        self.italic = not self.italic
        self.direct = True
        assert self.italic == op.a, "italic toggled into the wrong state"

    def op_size(self, op):
        """Font size for the text typed from here on."""
        self.size = op.a
        self.direct = True

    def op_reset_chars(self, op):
        """Ctrl+M: plain text in the style's own size - and the style's own alignment, which
        is why it must never be used in a paragraph that already has text."""
        assert not any(r[0] for r in self.cur["runs"]), "Default Formatting in a paragraph with text"
        self.bold = self.italic = self.direct = False
        self.size = None
        self.cur["align"] = None

    def op_align(self, op):
        """Ctrl+L / Ctrl+E: set the alignment (in Writer they don't toggle)."""
        self.cur["align"] = op.a

    def op_type(self, op):
        """Add a run with the current character formatting (it must not contain Enter or Tab).
        In a styled paragraph (title, heading, caption) no direct formatting may be left at
        the caret - a carried-over "not bold" would un-bold a heading."""
        assert "\n" not in op.a and "\t" not in op.a, "line break inside typed text"
        assert not (self.direct and self.cur["style"] not in (BODY, "Table Contents")), \
            f"direct formatting carried into a {self.cur['style']} paragraph"
        self.cur["runs"].append([op.a, self.bold, self.italic, self.size])

    def op_enter(self, op):
        """New paragraph after the current one, in the next style, with the same alignment and
        bullets (character formatting carries over too)."""
        assert self.table is None, "Enter inside a table cell"
        new = self._para(NEXT_STYLE.get(self.cur["style"], self.cur["style"]), self.cur["align"],
                         self.cur["bullets"])
        self.items.insert(self.items.index(self.cur) + 1, new)
        self.cur = new

    def op_page_break(self, op):
        """Ctrl+Enter: a page break, then a new paragraph (same style, alignment and character
        formatting) with the caret."""
        i = self.items.index(self.cur)
        new = self._para(self.cur["style"], self.cur["align"], self.cur["bullets"])
        self.items[i + 1:i + 1] = [{"kind": "page_break"}, new]
        self.cur = new

    def op_table(self, op):
        """Insert an op.a x op.b table before the (empty) current paragraph; the caret goes
        to the first cell, unformatted."""
        assert self.table is None, "table inside a table"
        assert not self.cur["runs"], "table inserted into a paragraph with text"
        assert not self.cur["bullets"], "table inserted into a bulleted paragraph"
        rows, cols = op.a, op.b
        table = {"kind": "table",
                 "cells": [[self._para("Table Contents") for _ in range(cols)] for _ in range(rows)]}
        self.items.insert(self.items.index(self.cur), table)     # the empty paragraph stays after it
        self.table = (table, 0, 0)
        self.outside = (self.bold, self.italic, self.size, self.direct)
        self.cur = table["cells"][0][0]
        self.bold = self.italic = self.direct = False
        self.size = None

    def op_next_cell(self, op):
        """Tab: the next cell (along the row, then down); in the last cell it adds a row,
        as in Writer. The new cell starts unformatted."""
        assert self.table is not None, "Tab outside a table"
        table, r, c = self.table
        cols = len(table["cells"][0])
        r, c = (r, c + 1) if c + 1 < cols else (r + 1, 0)     # end of a row: first cell of the next
        if r == len(table["cells"]):              # Tab in the last cell adds a row
            table["cells"].append([self._para("Table Contents") for _ in range(cols)])
        self.table = (table, r, c)
        self.cur = table["cells"][r][c]
        self.bold = self.italic = self.direct = False
        self.size = None

    def op_exit_table(self, op):
        """Ctrl+End: leave the table for the paragraph after it (the document's end, since
        everything is written at the end)."""
        assert self.table is not None, "Ctrl+End expected to leave a table"
        table = self.table[0]
        self.table = None
        self.cur = self.items[self.items.index(table) + 1]       # Ctrl+End: the paragraph after it
        self.bold, self.italic, self.size, self.direct = self.outside     # that paragraph's own formatting

    def op_checkpoint(self, op):
        """No effect on the document."""
        pass

    def op_save(self, op):
        """No effect on the document."""
        pass

    # --- result -------------------------------------------------------------------------------

    def result(self):
        """The document as comparable tuples (empty paragraphs dropped): ('p' or 'li' for a
        bulleted paragraph, style, alignment, runs), ('table', cells), ('page_break',)."""
        out = []
        for it in self.items:
            if it["kind"] == "page_break":
                out.append(("page_break",))
            elif it["kind"] == "table":
                out.append(("table", tuple(tuple(_cell(p) for p in row) for row in it["cells"])))
            elif it["runs"] and any(r[0] for r in it["runs"]):
                out.append(("li" if it["bullets"] else "p", it["style"], it["align"] or style_align(it["style"]),
                            _merge(it["runs"])))
        return out


def _merge(runs):
    """Runs as (text, bold, italic, size) tuples, empty ones dropped and neighbours with the
    same formatting joined - so how the text was split into type ops doesn't matter."""
    out = []
    for text, bold, italic, size in runs:
        if not text:
            continue
        if out and out[-1][1:] == (bold, italic, size):
            out[-1] = (out[-1][0] + text, bold, italic, size)
        else:
            out.append((text, bold, italic, size))
    return tuple(out)


def _cell(para):
    """A table cell as (text, bold): bold if any of its text is."""
    runs = _merge(para["runs"])
    return ("".join(r[0] for r in runs), any(r[1] for r in runs))


def expected(doc):
    """What a correctly written document looks like, in FakeWriter.result() form."""
    out = []
    for b in doc.blocks:
        if isinstance(b, PageBreak):
            out.append(("page_break",))
        elif isinstance(b, Table):
            out.append(("table", tuple(tuple((cell, b.header and r == 0) for cell in row)
                                       for r, row in enumerate(b.rows))))
        elif isinstance(b, Bullets):
            for item in b.items:
                out.append(("li", BODY, "left", _merge([[r.text, r.bold, r.italic, None] for r in item])))
        elif isinstance(b, Paragraph):
            out.append(("p", BODY, b.align, _merge([[r.text, r.bold, r.italic, b.size] for r in b.runs])))
        elif isinstance(b, (Title, Subtitle, Heading, Caption)):
            style = block_style(b)
            out.append(("p", style, style_align(style), ((b.text, False, False, None),)))
    return out
