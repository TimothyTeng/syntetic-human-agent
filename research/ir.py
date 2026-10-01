"""
The paper as a list of blocks, independent of the word processor.

    Document([Title("Apollo Program"), Subtitle("A short research report"),
              Heading("Introduction", 1), Paragraph([Run("The "), Run("Apollo", bold=True), Run(" ...")]),
              Table([["Year", "Event"], ["1969", "First landing"]]), Caption("Table 1. ..."), ...])

compose.py builds it, writer_ops.py turns it into editing operations, to_markdown()
previews it. validate() enforces what typing into OpenOffice Writer needs: ASCII only
(anything else would be pasted), no line breaks or tabs inside text (the typing model's
corrections must never cross a paragraph or cell boundary), and nothing Writer's
AutoCorrect / AutoFormat would rewrite while typing.
"""

import re
from dataclasses import dataclass, field


@dataclass
class Run:
    """A stretch of text inside a paragraph or bullet with one formatting: plain, bold
    (e.g. "Keywords:", key terms) or italic (e.g. a source's title in the references)."""
    text: str
    bold: bool = False
    italic: bool = False


@dataclass
class Title:
    """The paper's title at the top of the first page (Writer's Title style)."""
    text: str


@dataclass
class Subtitle:
    """The line under the title (Writer's Subtitle style)."""
    text: str


@dataclass
class Heading:
    """A section heading: level 1 for sections ("Introduction"), 2 for subsections
    (Writer's Heading 1 / Heading 2 styles)."""
    text: str
    level: int = 1


@dataclass
class Paragraph:
    """A body paragraph made of runs, optionally in a smaller font (abstract, references)
    or centred (the line under the subtitle)."""
    runs: list                        # [Run]
    size: float = None                # font size in pt (None = the style's)
    align: str = "left"               # "left" | "center"

    @property
    def text(self):
        """The paragraph's plain text (all runs joined)."""
        return "".join(r.text for r in self.runs)


@dataclass
class Bullets:
    """A bulleted list (body text with bullets switched on); each item is a list of runs."""
    items: list                       # [[Run]] - one list of runs per item


@dataclass
class Table:
    """A table of plain cell text, e.g. the key facts or the sources consulted; with
    header=True the first row is set in bold."""
    rows: list                        # [[cell text]]; the first row is the header
    header: bool = True


@dataclass
class Caption:
    """A table caption ("Table 1. ..."), typed just above its table (Writer's Caption style)."""
    text: str


@dataclass
class PageBreak:
    """A page break (Ctrl+Enter), e.g. before the references."""
    pass


@dataclass
class Document:
    """The whole paper: its blocks in order, and meta (what compose.py records about it:
    topic, section names, reference number -> URL)."""
    blocks: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def texts(self):
        """Every piece of text that will be typed, with where it is (for checks)."""
        for i, b in enumerate(self.blocks):
            if isinstance(b, (Title, Subtitle, Heading, Caption)):
                yield i, "block", b.text
            elif isinstance(b, Paragraph):
                for r in b.runs:
                    yield i, "run", r.text
            elif isinstance(b, Bullets):
                for item in b.items:
                    for r in item:
                        yield i, "run", r.text
            elif isinstance(b, Table):
                for row in b.rows:
                    for cell in row:
                        yield i, "cell", cell

    def word_count(self):
        """Words in all the text to be typed (table cells included)."""
        return sum(len(t.split()) for _, _, t in self.texts())


# --- Validation -------------------------------------------------------------------------------

# Paragraph starts that AutoFormat As You Type turns into lists, borders or tables
_AUTOFORMAT_START = re.compile(r"^(\*|-|>|\d+[.)]|[a-zA-Z][.)]|[ivx]+[.)]|---|===|___|\*\*\*|###|~~~|\+-)\s")
_AUTOFORMAT_ANY = re.compile(r"---|===|___|\*\*\*|###|~~~|\+-+\+")
# Sequences AutoCorrect replaces with other characters while typing, and Writer's
# "Automatic *bold* and _underline_". (" - " becomes an en dash and straight quotes become
# curly ones; read-back checks normalise those.)
_AUTOCORRECT = re.compile(r"\((c|r|tm)\)|:\)|:\(|:-\)|-->|<--|->|<-|==>|<==|<=>|\.\.\.|--|\b(1/2|1/4|3/4)\b"
                          r"|\*[^*\s][^*]*\*|(?<![A-Za-z0-9])_[^_\s][^_]*_(?![A-Za-z0-9])", re.IGNORECASE)


def check_text(text):
    """Problems with one piece of text to be typed (empty list = fine)."""
    problems = []
    if any(ord(ch) >= 128 for ch in text):
        problems.append("non-ASCII character")
    if "\n" in text or "\t" in text or "\r" in text:
        problems.append("line break or tab")
    if _AUTOCORRECT.search(text):
        problems.append(f"AutoCorrect trigger {_AUTOCORRECT.search(text).group(0)!r}")
    if _AUTOFORMAT_ANY.search(text):
        problems.append("AutoFormat trigger")
    return problems


def validate(doc):
    """List of (block index, problem) - empty if the document is safe to type into Writer."""
    problems = []
    for i, kind, text in doc.texts():                      # characters and AutoCorrect triggers anywhere
        for p in check_text(text):
            problems.append((i, f"{p} in {text[:60]!r}"))
    for i, b in enumerate(doc.blocks):                     # AutoFormat triggers at the start of each paragraph
        starts = []                                        # / item / cell, and malformed blocks
        if isinstance(b, Paragraph):
            starts = [b.text]
            if not b.text.strip():
                problems.append((i, "empty paragraph"))
        elif isinstance(b, Bullets):
            starts = ["".join(r.text for r in item) for item in b.items]
            if not b.items:
                problems.append((i, "empty bullet list"))
        elif isinstance(b, (Title, Subtitle, Heading, Caption)):
            starts = [b.text]
        elif isinstance(b, Table):
            widths = {len(r) for r in b.rows}
            if len(widths) != 1 or not b.rows:
                problems.append((i, "table rows of different lengths"))
            starts = [c for row in b.rows for c in row]
        for s in starts:
            if _AUTOFORMAT_START.match(s):
                problems.append((i, f"AutoFormat list trigger at the start of {s[:40]!r}"))
    return problems


# --- Saving (a paper being written is resumed from outline.json) ------------------------------

_BLOCKS = {cls.__name__: cls for cls in (Title, Subtitle, Heading, Paragraph, Bullets, Table, Caption, PageBreak)}


def to_dict(doc):
    """JSON-ready dict of a Document (each block tagged with its class name), for outline.json."""
    def block(b):
        """One block as a dict; runs become dicts too."""
        d = {"type": type(b).__name__, **{k: v for k, v in vars(b).items()}}
        if isinstance(b, Paragraph):
            d["runs"] = [vars(r) for r in b.runs]
        elif isinstance(b, Bullets):
            d["items"] = [[vars(r) for r in item] for item in b.items]
        return d
    return {"blocks": [block(b) for b in doc.blocks], "meta": doc.meta}


def from_dict(d):
    """The Document saved by to_dict()."""
    blocks = []
    for raw in d["blocks"]:
        raw = dict(raw)
        cls = _BLOCKS[raw.pop("type")]
        if cls is Paragraph:
            raw["runs"] = [Run(**r) for r in raw["runs"]]
        elif cls is Bullets:
            raw["items"] = [[Run(**r) for r in item] for item in raw["items"]]
        blocks.append(cls(**raw))
    return Document(blocks, d.get("meta", {}))


# --- Preview ----------------------------------------------------------------------------------

def _md_runs(runs):
    """Runs as Markdown: **bold**, *italic* (bold wins if both)."""
    out = []
    for r in runs:
        t = r.text
        core = t.strip()
        if core and (r.bold or r.italic):
            mark = "**" if r.bold else "*"
            # keep the run's spaces outside the marks ("** x**" isn't Markdown)
            lead, trail = t[:len(t) - len(t.lstrip())], t[len(t.rstrip()):]
            t = f"{lead}{mark}{core}{mark}{trail}"
        out.append(t)
    return "".join(out)


def to_markdown(doc):
    """A Markdown preview of the paper (paper.md): headings one level below the title,
    small text in <small>, a page break as a rule."""
    lines = []
    for b in doc.blocks:
        if isinstance(b, Title):
            lines += [f"# {b.text}", ""]
        elif isinstance(b, Subtitle):
            lines += [f"*{b.text}*", ""]
        elif isinstance(b, Heading):
            lines += [f"{'#' * (b.level + 1)} {b.text}", ""]
        elif isinstance(b, Paragraph):
            size = f"<small>{_md_runs(b.runs)}</small>" if b.size and b.size < 11 else _md_runs(b.runs)
            lines += [size, ""]
        elif isinstance(b, Bullets):
            lines += [f"- {_md_runs(item)}" for item in b.items] + [""]
        elif isinstance(b, Table):
            head, *body = b.rows
            lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            lines += ["| " + " | ".join(r) + " |" for r in body] + [""]
        elif isinstance(b, Caption):
            lines += [f"*{b.text}*", ""]
        elif isinstance(b, PageBreak):
            lines += ["---", ""]
    return "\n".join(lines)
