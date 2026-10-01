"""
Turning the paper (ir.Document) into OpenOffice Writer editing operations - pure, no UI.

    ops = render(doc)
    for op in ops: print(op)          # Op('style', 'Heading 1'), Op('type', 'Introduction', 'transcribe'), ...

The renderer keeps track of what Writer's state will be at the caret - paragraph style,
bullets, bold / italic / font size, alignment, whether it is in a table - and only emits
an operation when something has to change, the way a person formats while writing. What
it relies on was checked in Writer 4.1:

  * a new paragraph is Enter; Writer then picks the style's "next style" itself (after a
    heading or the subtitle: Text body; after the title: Subtitle; a caption and body
    text continue as they are), and bullets, direct character formatting, font size and
    DIRECT alignment carry over (a style's own centring - Title, Subtitle - doesn't)
  * direct character formatting includes "switched off": Ctrl+B after bold text stores
    an explicit not-bold, which carries over Enter like bold does and would un-bold a
    heading typed next. So once anything was toggled or sized in a paragraph, the next
    block starts with Default Formatting (Ctrl+M)
  * Ctrl+M also clears the paragraph's direct alignment, so it is only used in a fresh,
    empty paragraph (in one with text it would un-centre that text) - before the style
    and alignment are set; it keeps bullets, and a page break
  * bullets are switched on and off (Shift+F12 / Bullets On/Off) - in Writer they aren't a
    paragraph style; applying a heading style drops them
  * applying a paragraph style clears direct alignment (the style's own applies); the
    alignment keys set the alignment, they don't toggle
  * a table is inserted, filled cell by cell (Tab between cells, never after the last
    one, which would add a row) and left with Ctrl+End - the document ends with the
    paragraph after the table, since everything is written at the end
  * after a table or a page break the caret already sits in a fresh paragraph

Operations (Op.kind, args):
  style name | bullets on | bold on | italic on | size pt | reset_chars | align 'left'/'center'
  type text mode | enter | page_break | table rows cols | next_cell | exit_table
  checkpoint block_index | save ['final']
"""

from dataclasses import dataclass

from .ir import Bullets, Caption, Heading, PageBreak, Paragraph, Subtitle, Table, Title

BODY = "Text body"                   # Writer's style for running text (Ctrl+0)
START_STYLE = "Default"              # the paragraph style of a new document
# Style Writer gives the paragraph after Enter (the style's "next style")
NEXT_STYLE = {"Default": "Default", BODY: BODY, "Title": "Subtitle", "Subtitle": BODY, "Heading 1": BODY,
              "Heading 2": BODY, "Heading 3": BODY, "Caption": "Caption", "Table Contents": "Table Contents"}
# Alignment a style brings with it (the others are left-aligned)
STYLE_ALIGN = {"Title": "center", "Subtitle": "center"}
COMPOSE_MIN_WORDS = 12               # runs this long are typed as composed prose, shorter ones transcribed


@dataclass(frozen=True)
class Op:
    """One editing operation: kind (see the module docstring) and up to two arguments."""
    kind: str
    a: object = None
    b: object = None

    def __str__(self):
        """Readable form for logs, e.g. "type('Introduction', 'transcribe')"."""
        args = ", ".join(repr(x) for x in (self.a, self.b) if x is not None)
        return f"{self.kind}({args})"


def block_style(block):
    """Writer paragraph style for a block (headings deeper than 3 become Heading 3; bullet
    items are body text with bullets switched on)."""
    if isinstance(block, Title):
        return "Title"
    if isinstance(block, Subtitle):
        return "Subtitle"
    if isinstance(block, Heading):
        return f"Heading {min(max(block.level, 1), 3)}"
    if isinstance(block, Caption):
        return "Caption"
    return BODY


def style_align(style):
    """The alignment a paragraph style brings with it."""
    return STYLE_ALIGN.get(style, "left")


def type_mode(text):
    """Typing mode for a run: 'compose' (writing prose, with thinking pauses and revisions)
    for runs of COMPOSE_MIN_WORDS words or more, else 'transcribe'."""
    return "compose" if len(text.split()) >= COMPOSE_MIN_WORDS else "transcribe"


class Renderer:
    """Turns blocks into operations while tracking what Writer's state at the caret will be,
    so that only the operations that change something are emitted."""

    def __init__(self):
        """Start state: the empty Default paragraph of a new document, plain text, no direct
        formatting."""
        self.ops = []
        self.style = START_STYLE
        self.bullets = False
        self.bold = self.italic = False
        self.size = None                 # None = the style's own size
        self.direct = False              # direct character formatting at the caret (even "not bold")
        self.align = None                # direct alignment; None = the style's own
        self.fresh = True                # caret is in an empty paragraph nobody has typed in

    def emit(self, kind, a=None, b=None):
        """Append an operation."""
        self.ops.append(Op(kind, a, b))

    def alignment(self):
        """The paragraph's alignment as it shows: direct, else the style's own."""
        return self.align or style_align(self.style)

    # --- state changes ----------------------------------------------------------------------

    def set_char(self, bold=False, italic=False):
        """Bold / italic for the next run. Only differing ones are emitted (the executor
        toggles them, so an extra bold op would switch bold off again)."""
        if self.bold != bold:
            self.emit("bold", bold)
            self.bold, self.direct = bold, True
        if self.italic != italic:
            self.emit("italic", italic)
            self.italic, self.direct = italic, True

    def reset_chars(self):
        """Default Formatting (Ctrl+M) - only ever in an empty paragraph, since it also
        clears the paragraph's direct alignment."""
        if self.direct or self.bold or self.italic or self.size is not None:
            self.emit("reset_chars")
            self.bold = self.italic = self.direct = False
            self.size = self.align = None

    def set_bullets(self, on):
        """Bullets on / off (a toggle, so only emitted when it changes)."""
        if self.bullets != on:
            self.emit("bullets", on)
            self.bullets = on

    def new_paragraph(self):
        """Start a block's paragraph: Enter, unless the caret already sits in a fresh empty
        paragraph (start of the document, after a table or page break). Bullets, direct
        character formatting and direct alignment carry over; the style becomes Writer's
        next style."""
        if not self.fresh:
            self.emit("enter")
            self.style = NEXT_STYLE.get(self.style, self.style)
        self.fresh = False

    def set_paragraph(self, style, align=None, size=None, bullets=False):
        """Formatting for a new (empty) block: bullets switched off first, direct formatting
        carried over cleared (Ctrl+M), then style, bullets, font size and alignment - the
        size before the alignment, because the keyboard way to the Font Size box tabs out of
        the Apply Style box, which re-applies the style and so clears direct alignment.
        align=None: the style's own alignment."""
        align = align or style_align(style)
        if not bullets:
            self.set_bullets(False)
        if self.direct or self.size != size:
            self.reset_chars()
        if self.style != style:
            self.emit("style", style)
            self.style = style
            self.align = None                        # direct alignment goes with a new style
            if style.startswith("Heading"):
                self.bullets = False                 # heading styles have outline numbering instead
        if bullets:
            self.set_bullets(True)
        if size is not None and self.size != size:
            self.emit("size", size)
            self.size, self.direct = size, True
        if self.alignment() != align:
            self.emit("align", align)
            self.align = align

    def type(self, text, mode=None):
        """Type text (nothing for an empty string); mode defaults to type_mode(text)."""
        if text:
            self.emit("type", text, mode or type_mode(text))

    def runs(self, runs):
        """Type formatted runs, each with its own bold / italic (one type op per run, so the
        typing model's corrections never cross a formatting boundary)."""
        for r in runs:
            self.set_char(r.bold, r.italic)
            self.type(r.text)

    # --- blocks -------------------------------------------------------------------------------

    def block(self, i, b, first_of_section):
        """Operations for block number i. A new section (first_of_section) begins with a save."""
        if first_of_section and i > 0:
            self.emit("save")
        if isinstance(b, PageBreak):
            self.emit("page_break")              # the new page's paragraph keeps the formatting,
            self.fresh = True                    # cleared there by the next block (Ctrl+M keeps the break)
            return
        if isinstance(b, Table):
            self.new_paragraph()
            self.set_paragraph(BODY)             # the table goes into a plain empty paragraph
            outside = (self.bold, self.italic, self.direct)
            self.emit("table", len(b.rows), len(b.rows[0]))
            for r, row in enumerate(b.rows):
                for c, cell in enumerate(row):
                    if r or c:                   # Tab between cells, never after the last (it adds a row)
                        self.emit("next_cell")
                    self.bold = self.italic = self.direct = False      # each cell starts unformatted
                    if b.header and r == 0:
                        self.set_char(bold=True)
                    self.type(cell, "transcribe")
            self.emit("exit_table")              # Ctrl+End: the empty paragraph after the table
            self.bold, self.italic, self.direct = outside
            self.fresh = True
            return
        if isinstance(b, Bullets):
            for k, item in enumerate(b.items):
                self.new_paragraph()
                if k == 0:
                    self.set_paragraph(BODY, bullets=True)
                else:
                    self.set_char()              # the list continues; drop bold / italic carried over Enter
                self.runs(item)
            return
        self.new_paragraph()
        if isinstance(b, Paragraph):
            self.set_paragraph(BODY, b.align, b.size)
            self.runs(b.runs)
        else:
            self.set_paragraph(block_style(b))
            self.type(b.text, "transcribe")


def render(doc):
    """List of Op for the whole document, with a checkpoint after every block (where a
    writing session may stop and later resume) and a save before each new section."""
    r = Renderer()
    for i, b in enumerate(doc.blocks):
        r.block(i, b, first_of_section=isinstance(b, Heading) and b.level == 1)
        r.emit("checkpoint", i)
    r.emit("save", "final")
    return r.ops
