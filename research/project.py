"""
A research project: browse for a topic, take notes from every page read, then write the
paper into OpenOffice Writer - for main.py (--process research) and for the Session's browse / write
activities.

    proj = ResearchProject(human, "Apollo program", log=log)
    ws.research = proj                      # tasks.Workspace: browse() and write() use it
    ws.browse(seconds_left)                 # searches proj.next_query(), notes from each page read
    proj.compose(target_words=800)          # the paper (ir.Document), saved to outline.json
    proj.write_next(ws, seconds_left)       # types it into Writer, section by section

Everything lives in one folder (runs/research/<topic>-<time>/): notes.json,
captures/NNN.json (raw page text), outline.json + paper.md (the composed paper and a
preview), progress.json (how far the writing got), diagnostics.json (self-checks of what
Chrome and Writer exposed, see diagnostics.py) and the .odt document. A run can be resumed from
the folder (ResearchProject.resume).
"""

import datetime
import json
import os
import re
import time

import numpy as np

from algorithms.session import TaskError
from controller import browser
from tasks import click_into_document, log as task_log

from . import ir
from .compose import ComposeConfig, compose_paper, plan_queries, words_for_time
from .diagnostics import Diagnostics
from .extract import capture_page, normalise_url, parse_capture
from .notes import ResearchNotes, _merge
from .textutil import slugify
from .writer_exec import WriterExecutor
from .writer_ops import render

RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "research")
MIN_SOURCES = 2
SEARCH_PAGE = re.compile(r"^https?://(www\.)?(google|bing|duckduckgo|search\.yahoo)\.[a-z.]+/(search|\?|$)", re.I)
MIN_WORDS = 400


class ResearchProject:
    """One topic from searching to the saved .odt: the notes taken, the composed paper and
    how far its typing got, all kept in (and resumable from) the run folder."""

    def __init__(self, human, topic, folder=None, n_sources=4, paraphrase=1, revisions="heuristic", log=print):
        """A new project in `folder` (default runs/research/<topic>-<time>). n_sources: pages
        to read before writing; paraphrase and revisions are passed to the composer and the
        Writer executor."""
        self.human, self.topic, self.log = human, topic, log
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.folder = os.path.abspath(folder or os.path.join(RUNS_DIR, f"{slugify(topic)}-{stamp}"))
        self.n_sources, self.paraphrase, self.revisions = n_sources, paraphrase, revisions
        self.notes = ResearchNotes(topic)
        self.queries_done = []
        self.current = None              # Source of the page being read
        self.paper = None                # ir.Document once composed
        self.ops = None                  # writer_ops.render(paper): the editing operations
        self.op_index = 0                # next operation to perform
        self.saved = False               # the document has been saved under its name
        self.doc_path = os.path.join(self.folder, f"{slugify(topic)}.odt")
        os.makedirs(os.path.join(self.folder, "captures"), exist_ok=True)
        self.diag = Diagnostics(self.folder, log)

    # --- browsing --------------------------------------------------------------------------

    def next_query(self):
        """(query, link hint) of the next search: the topic on Wikipedia / Britannica first,
        then the topic with the main headings of the first page read."""
        for query, hint in plan_queries(self.topic, self.notes):
            if query not in self.queries_done:
                self.queries_done.append(query)
                return query, hint
        query = f"{self.topic} facts"                  # every planned query used: a generic one
        return query, None

    def on_page(self):
        """A result page has opened: take notes from it (tasks.read_result calls this).

        Waits until the clicked result has replaced the search page, then reads it, retrying
        while Chrome is still building the accessibility tree (the first read of a fresh page
        often returns little or no text). The raw capture is saved to captures/NNN.json."""
        if not self._left_search_page():
            self.log("Still on the search results - no notes taken from this page")
            self.current = None
            return
        t0 = time.monotonic()
        cap = None
        for attempt in range(4):        # Chrome builds the page's accessibility tree a moment after loading
            cap = capture_page()
            if cap is not None and cap["total_words"] >= 150:
                break
            time.sleep(1.5)
        if cap is not None and is_search_page(cap["url"]):   # the click didn't open a result after all
            cap = None
        if cap is None:
            self.log("Couldn't read this page's text - no notes taken")
            self.current = None
            return
        n = len(os.listdir(os.path.join(self.folder, "captures"))) + 1
        with open(os.path.join(self.folder, "captures", f"{n:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(cap, f, indent=1, ensure_ascii=False)
        query = self.queries_done[-1] if self.queries_done else ""
        self.current = self.notes.add(parse_capture(cap, query))   # on_view() marks what gets read of it
        src = self.current
        self.log(f"Notes from {src.site}: {src.title!r} - {len(src.sections)} sections, {src.words()} words, "
                 f"{len(src.tables)} tables (read via {cap['method']} in {time.monotonic() - t0:.1f} s)")
        self.diag.page(cap, src)
        self.save_notes()

    def _left_search_page(self, timeout=10.0):
        """Wait until the browser shows the clicked result rather than the search results
        (a slow site keeps the results page up for a few seconds after the click)."""
        deadline = time.monotonic() + timeout
        while is_search_page(normalise_url(browser.get_current_url())):
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.5)
        browser.wait_for_page_load(timeout=10)
        return True

    def on_view(self, state):
        """The reader scrolled: (position %, visible %) -> the part of the page now in view.

        The scroll position is a percentage of the scrollable range, i.e. of (100 - visible)
        percent of the page, so the view covers [top, top + visible] percent of the page.
        That interval is merged into the source's seen_intervals and words_read is updated."""
        if self.current is None:
            return
        pos, view = state
        top = pos * (100.0 - view) / 100.0            # at 100 % scrolled the view's bottom is the page's end
        self.current.seen_intervals = _merge(self.current.seen_intervals + [[round(top, 1), round(top + view, 1)]])
        src = self.current
        src.words_read = sum(len(p.text.split()) for s in src.sections for p in s.paras if src.seen(p.pos))

    def save_notes(self):
        """Write notes.json to the run folder."""
        self.notes.save(self.folder)

    def has_material(self):
        """Enough read to start writing: at least MIN_SOURCES pages (fewer if fewer were
        asked for) and MIN_WORDS words of notes."""
        return len(self.notes.sources) >= min(MIN_SOURCES, self.n_sources) and \
            sum(s.words() for s in self.notes.sources) >= MIN_WORDS

    def needs_sources(self):
        """Fewer pages read than the n_sources asked for."""
        return len(self.notes.sources) < self.n_sources

    # --- composing ---------------------------------------------------------------------------

    def compose(self, target_words=None, seconds=None):
        """Write the paper from the notes (once; later calls keep it). The length is
        target_words, or what this person can type in `seconds` (800 words without either),
        kept to 300-2000 words. Saves outline.json, paper.md and progress.json."""
        if self.paper is not None:
            return self.paper                          # also after a resume: the paper is never re-composed
        if target_words is None:
            target_words = words_for_time(seconds or 1200, self.human.persona.wpm) if seconds else 800
        target_words = int(np.clip(target_words, 300, 2000))   # too short for a paper's frame / too long to type
        cfg = ComposeConfig(target_words=target_words, paraphrase=self.paraphrase)
        try:
            self.paper = compose_paper(self.notes, self.topic, self.human.rng, cfg)
        except ValueError as exc:
            raise TaskError(f"Couldn't write a paper from the notes: {exc}") from exc
        self.ops = render(self.paper)
        self.op_index = 0
        with open(os.path.join(self.folder, "outline.json"), "w", encoding="utf-8") as f:
            json.dump(ir.to_dict(self.paper), f, indent=1)
        with open(os.path.join(self.folder, "paper.md"), "w", encoding="utf-8") as f:
            f.write(ir.to_markdown(self.paper))
        self.save_progress()
        sections = ", ".join(self.paper.meta.get("sections", []))
        self.log(f"Composed the paper: {self.paper.word_count()} words, sections: {sections} "
                 f"({len(self.ops)} editing steps; preview in {os.path.join(self.folder, 'paper.md')})")
        return self.paper

    # --- writing -------------------------------------------------------------------------------

    @property
    def finished(self):
        """Every editing operation of the paper has been performed."""
        return self.ops is not None and self.op_index >= len(self.ops)

    def executor(self, ws, dry_run=False):
        """A WriterExecutor for the Writer document of `ws` (none with dry_run). If the document
        loses keyboard focus, it is clicked back into like a person would (refocus)."""
        refocus = (lambda: click_into_document(self.human, ws.writer_win, ws.writer_doc)) if not dry_run else None
        return WriterExecutor(self.human, None if dry_run else ws.writer_win, None if dry_run else ws.writer_doc,
                              save_path=self.doc_path, refocus=refocus, log=task_log, dry_run=dry_run,
                              revisions=self.revisions, saved=self.saved)

    def write_next(self, ws, seconds, one_section=True, dry_run=False):
        """Type the next part of the paper into the Writer document of `ws` (already in front
        with the caret at the end): until the end of the current section (one_section) or
        until about `seconds` have passed. Returns the executor (its summary / problems).

        The executor stops only between blocks: it doesn't start a block that block_seconds()
        says won't finish before the deadline, so a short time slot may write nothing (the
        caller sees op_index unchanged). Progress is saved after every block."""
        if self.paper is None:
            self.compose(seconds=seconds)
        if not dry_run:
            self.diag.writer(ws.writer_win, ws.writer_doc)     # once per run: what Writer exposes
        ex = self.executor(ws, dry_run)
        deadline = time.monotonic() + max(10.0, seconds)
        start = self.op_index
        self.op_index = ex.run(self.ops, start, deadline=deadline, on_checkpoint=self._checkpoint(ws, ex),
                               stop_before_save=one_section, estimate=self.block_seconds)
        self.saved = ex.saved
        self.save_progress()
        self.diag.writing(ex)
        done = sum(1 for op in self.ops[:self.op_index] if op.kind == "checkpoint")   # blocks finished
        total = sum(1 for op in self.ops if op.kind == "checkpoint")
        self.log(f"Wrote paper blocks {done}/{total} ({ex.summary()})")
        return ex

    def block_seconds(self, i):
        """Rough time for the block starting at ops[i]: typing at this person's effective
        rate (with thinking pauses) plus a few seconds per formatting step."""
        chars_per_s = self.human.persona.wpm * 5 / 60 / 1.8   # 5 chars a word; pauses and fixes ~1.8x slower
        total = 0.0
        for op in self.ops[i:]:
            if op.kind == "checkpoint":                       # the end of this block
                break
            total += len(op.a) / chars_per_s if op.kind == "type" else 2.0
        return total

    def _checkpoint(self, ws, ex):
        """The executor's on_checkpoint callback: after each block, record the progress and
        check the block's text in the document."""
        def check(next_index, block_index):
            """A block is finished: next_index is the operation after it."""
            self.op_index = next_index
            self.saved = ex.saved
            if not ex.dry_run:
                self.diag.block_checked(self.verify_block(ex.doc or ws.writer_doc, block_index))
        return check

    def verify_block(self, doc, i):
        """Log whether the document now ends with block i's text. True if it does.

        Both texts are normalised (whitespace, paragraph and cell marks, AutoCorrect's dashes
        and curly quotes), so only real differences count - typically a typo the typing model left in or a key
        that got lost. Writer only exposes the paragraphs on screen - the end of the document,
        where the block was just typed."""
        from controller import writer as W
        block = self.paper.blocks[i]
        expected = W.normalise(block_text(block))
        if not expected:
            return True                                       # a page break: nothing to compare
        got = W.normalise(W.document_text(doc))
        if got.endswith(expected):
            return True
        self.log(f"Block {i} ({type(block).__name__}) differs from what was meant to be typed:")
        for line in text_differences(expected, got[-len(expected) - 40:]):   # the end of the document, with slack
            self.log("  " + line)
        return False

    # --- saving / resuming ----------------------------------------------------------------------

    def save_progress(self):
        """Write progress.json: how far the writing got and whether the document has its name."""
        state = {"topic": self.topic, "op_index": self.op_index, "saved": self.saved, "document": self.doc_path,
                 "queries_done": self.queries_done, "finished": self.finished}
        with open(os.path.join(self.folder, "progress.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1)

    @classmethod
    def resume(cls, human, folder, log=print, **kw):
        """Continue a project from its folder (notes, composed paper and writing progress).

        The paper is loaded from outline.json rather than composed again (composing is
        random, so a new paper wouldn't match what is already in the .odt), and rendered to
        the same operations, so op_index points at the next block still to type. With
        saved, tasks.Workspace.write() reopens the .odt instead of starting a new one."""
        with open(os.path.join(folder, "progress.json"), encoding="utf-8") as f:
            state = json.load(f)
        proj = cls(human, state["topic"], folder=folder, log=log, **kw)
        proj.notes = ResearchNotes.load(folder)
        proj.queries_done = state.get("queries_done", [])   # so the same searches aren't repeated
        outline = os.path.join(folder, "outline.json")
        if os.path.exists(outline):                          # no outline: still reading, compose later
            with open(outline, encoding="utf-8") as f:
                proj.paper = ir.from_dict(json.load(f))
            proj.ops = render(proj.paper)
            proj.op_index = state.get("op_index", 0)
            proj.saved = state.get("saved", False)
        proj.doc_path = state.get("document", proj.doc_path)
        log(f"Resuming research on {proj.topic!r}: {len(proj.notes.sources)} sources, "
            f"{'paper composed' if proj.paper else 'no paper yet'}, step {proj.op_index}")
        return proj


def text_differences(meant, got, context=15, limit=4):
    """Where two texts differ, as short lines: "'... orbit and' -> '... obit and'"."""
    import difflib
    for n in (30, 12, 5):                                  # align `got` on the block's start (a shorter
        start = got.rfind(meant[:n])                       # prefix if the difference is right at the start)
        if start >= 0:
            got = got[start:]
            break
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, meant, got, autojunk=False).get_opcodes():
        if tag != "equal" and i1 < len(meant):
            out.append(f"meant {meant[max(0, i1 - context):i2 + context]!r} "
                       f"-> got {got[max(0, j1 - context):j2 + context]!r}")
    return out[:limit] or [f"meant ...{meant[-80:]!r}", f"got   ...{got[-80:]!r}"]


def is_search_page(url):
    """True for a search engine's results page (never a source to take notes from)."""
    return bool(SEARCH_PAGE.match(url or ""))


def block_text(block):
    """The text a block puts in the document (table cells joined by spaces)."""
    if isinstance(block, ir.Table):
        return " ".join(c for row in block.rows for c in row)
    if isinstance(block, ir.Bullets):
        return " ".join("".join(r.text for r in item) for item in block.items)
    if isinstance(block, ir.Paragraph):
        return block.text
    return getattr(block, "text", "")
