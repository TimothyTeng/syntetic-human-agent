"""
From page captures to OpenOffice Writer operations, offline: parsing (research/extract.py),
composing (research/compose.py), rendering (research/writer_ops.py) checked against
FakeWriter, the typing plans of every typed run, and the executor's dry run.
"""

import datetime
import json
import os
import re
import unittest

import numpy as np

from research import ir
from research.compose import ComposeConfig, compose_paper, plan_queries
from research.extract import clean_title, parse_capture, site_name
from research.fake_writer import FakeWriter, expected
from research.notes import ResearchNotes
from research.writer_exec import WriterExecutor
from research.writer_ops import render

PAGES = os.path.join(os.path.dirname(__file__), "fixtures", "pages")


def load_capture(name):
    """The page capture tests/fixtures/pages/<name>.json."""
    with open(os.path.join(PAGES, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def fixture_notes():
    """Notes on "Apollo program" from the three fixture pages."""
    notes = ResearchNotes("Apollo program")
    for name in ("apollo_wikipedia", "apollo_britannica", "apollo_nasa"):
        notes.add(parse_capture(load_capture(name)))
    return notes


def compose(seed=1, words=800, paraphrase=1):
    """A paper composed from the fixture notes (fixed date, so the output is repeatable)."""
    cfg = ComposeConfig(target_words=words, paraphrase=paraphrase, today=datetime.date(2026, 10, 1))
    return compose_paper(fixture_notes(), "Apollo program", np.random.default_rng(seed), cfg)


class ParseTest(unittest.TestCase):
    """Page captures -> notes (research/extract.py), and query planning from the notes."""

    def test_wikipedia_page(self):
        """Sections at the headings, boilerplate and page furniture dropped, infobox kept."""
        src = parse_capture(load_capture("apollo_wikipedia"))
        self.assertEqual(src.title, "Apollo program")
        self.assertEqual(src.site, "Wikipedia")
        headings = [s.heading for s in src.sections]
        self.assertEqual(headings, ["", "Background", "Choosing a mission mode", "Spacecraft", "Missions", "Legacy"])
        text = " ".join(p.text for s in src.sections for p in s.paras)
        self.assertNotRegex(text, r"\[\d+\]|\[edit\]|ISBN|Retrieved|Main article|From Wikipedia|/\u02c8")
        self.assertEqual(src.tables[0].kind, "infobox")
        self.assertIn(["Country", "United States"], src.tables[0].rows)

    def test_lead_paragraph_is_not_mistaken_for_a_heading(self):
        """A paragraph starting with the title word stays in the lead; "Related Topics" is dropped."""
        src = parse_capture(load_capture("apollo_britannica"))
        self.assertEqual(src.sections[0].heading, "")
        self.assertTrue(src.sections[0].paras[0].text.startswith("Apollo, the United States effort"))
        self.assertNotIn("Related Topics", [s.heading for s in src.sections])

    def test_headings_are_guessed_when_the_page_has_none(self):
        """Short lines followed by a paragraph become headings on a page without heading markup."""
        src = parse_capture(load_capture("apollo_nasa"))
        self.assertIn("Mission control", [s.heading for s in src.sections])

    def test_seen_part_of_the_page(self):
        """seen_intervals decide which positions were read, and words_read counts only those."""
        src = parse_capture(load_capture("apollo_wikipedia"))
        self.assertTrue(src.seen(0.10))
        self.assertFalse(src.seen(0.95))
        self.assertLess(src.words_read, src.words())

    def test_titles_and_sites(self):
        """Site suffixes are taken off page titles; hosts map to site names or bare domains."""
        self.assertEqual(clean_title("Apollo | History, Missions, & Facts | Britannica", "Encyclopaedia Britannica"),
                         "Apollo")
        self.assertEqual(clean_title("The Apollo Program - NASA", "NASA"), "The Apollo Program")
        self.assertEqual(site_name("https://en.wikipedia.org/wiki/X"), "Wikipedia")
        self.assertEqual(site_name("https://www.example.org/a"), "example.org")

    def test_notes_round_trip(self):
        """Notes survive saving to JSON and loading unchanged."""
        notes = fixture_notes()
        again = ResearchNotes.from_dict(json.loads(json.dumps(notes.to_dict())))
        self.assertEqual(again.to_dict(), notes.to_dict())

    def test_queries_follow_the_first_source(self):
        """Wikipedia and Britannica are searched first, then the topic with the first source's headings."""
        queries = [q for q, _ in plan_queries("Apollo program", fixture_notes())]
        self.assertEqual(queries[:2], ["Apollo program wikipedia", "Apollo program britannica"])
        self.assertTrue(any(q.endswith(("spacecraft", "missions", "legacy", "background")) for q in queries[2:]))


class ComposeTest(unittest.TestCase):
    """Notes -> paper (research/compose.py)."""

    def test_paper_structure(self):
        """Every seed gives a valid paper: frame sections, 3-5 body sections, tables, bold and italic."""
        for seed in range(5):
            doc = compose(seed)
            self.assertEqual(ir.validate(doc), [], f"seed {seed}")
            headings = [b.text for b in doc.blocks if isinstance(b, ir.Heading) and b.level == 1]
            self.assertEqual(headings[:2], ["Abstract", "Introduction"])
            self.assertIn("References", headings)
            body = headings[2:headings.index("Discussion")]
            body = [h for h in body if h != "Key Facts"]
            self.assertTrue(3 <= len(body) <= 5, body)
            self.assertIsInstance(doc.blocks[0], ir.Title)
            tables = [b for b in doc.blocks if isinstance(b, ir.Table)]
            self.assertGreaterEqual(len(tables), 2)          # key facts + sources consulted
            self.assertTrue(any(r.bold for b in doc.blocks if isinstance(b, ir.Paragraph) for r in b.runs))
            self.assertTrue(any(r.italic for b in doc.blocks if isinstance(b, ir.Paragraph) for r in b.runs))

    def test_short_papers_have_a_lean_frame(self):
        """A short paper has two body sections and no appendix, but still a table."""
        doc = compose(1, 420)
        headings = [b.text for b in doc.blocks if isinstance(b, ir.Heading) and b.level == 1]
        self.assertEqual(len(headings[2:headings.index("Key Facts")]), 2, headings)
        self.assertNotIn("Appendix: Sources Consulted", headings)
        self.assertEqual(ir.validate(doc), [])
        self.assertTrue(any(isinstance(b, ir.Table) for b in doc.blocks))

    def test_every_citation_has_a_reference(self):
        """Every [n] cited in the text has a reference entry; no citation placeholder is left."""
        doc = compose(2)
        text = " ".join(t for _, _, t in doc.texts())
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        refs = [b for b in doc.blocks if isinstance(b, ir.Paragraph) and b.runs[0].text.startswith("[")]
        numbers = {int(re.match(r"\[(\d+)\]", b.runs[0].text).group(1)) for b in refs}
        self.assertTrue(cited)
        self.assertTrue(cited <= numbers, (cited, numbers))
        self.assertNotIn("\x00", text)

    def test_length_follows_the_target(self):
        """The paper's word count is within 25% of the target."""
        for words in (450, 600, 900, 1300):
            n = compose(1, words).word_count()
            self.assertLess(abs(n - words) / words, 0.25, (words, n))

    def test_no_sentence_twice(self):
        """No long sentence appears twice in the paper (citations ignored)."""
        doc = compose(4)
        sentences = [s for b in doc.blocks if isinstance(b, ir.Paragraph)
                     for s in re.split(r"(?<=\.) ", b.text) if len(s.split()) > 8]
        sentences = [re.sub(r" \[\d+\]", "", s) for s in sentences]
        self.assertEqual(len(sentences), len(set(sentences)))

    def test_leaning_sentences_keep_their_predecessor(self):
        """No paragraph opens with a sentence that refers back ("It", "However", ...) to one left out."""
        frame = re.compile(r"^This (report|paper|short report|section) ")      # the paper's own sentences
        for seed in range(4):
            for b in compose(seed).blocks:
                if isinstance(b, ir.Paragraph) and re.match(r"(It|Its|This|They|However|These)\b", b.text) \
                        and not frame.match(b.text):
                    self.fail(f"paragraph starts with a sentence that needs the one before: {b.text[:60]!r}")

    def test_outline_round_trip(self):
        """A paper saved with ir.to_dict and loaded with ir.from_dict is the same paper."""
        doc = compose(1)
        again = ir.from_dict(json.loads(json.dumps(ir.to_dict(doc))))
        self.assertEqual(ir.to_markdown(again), ir.to_markdown(doc))


class RenderTest(unittest.TestCase):
    """Paper -> editing operations (research/writer_ops.py) and the typing of their text."""

    def test_fake_writer_ends_with_the_paper(self):
        """Replaying the operations in FakeWriter produces exactly the paper's blocks and formatting
        (and never uses Default Formatting in a paragraph that has text)."""
        for seed in range(6):
            doc = compose(seed)
            got = FakeWriter().run(render(doc)).result()
            self.assertEqual(got, expected(doc), f"seed {seed}")

    def test_every_kind_of_block(self):
        """A document with every block kind, formatting carried into places it must be cleared
        from (bold before a page break, a centred small line before a heading, a list before
        a table) comes out right."""
        from research.ir import (Bullets, Caption, Document, Heading, PageBreak, Paragraph, Run, Subtitle,
                                 Table, Title)
        doc = Document([
            Title("T"), Subtitle("S"), Paragraph([Run("centred")], align="center", size=10),
            Heading("H", 1), Paragraph([Run("Key:", bold=True), Run(" value")], size=10),
            Bullets([[Run("a", bold=True), Run(" b")], [Run("c", italic=True)]]),
            Table([["x", "y"], ["1", "2"]]), Caption("Cap"), Paragraph([Run("end", bold=True)]),
            PageBreak(), Heading("R", 1), Paragraph([Run("[1] "), Run("t", italic=True), Run(".")], size=10),
        ])
        self.assertEqual(FakeWriter().run(render(doc)).result(), expected(doc))

    def test_operations(self):
        """Operations start with a style and end with the final save; typed text has no breaks or
        tabs; a table is filled with exactly rows x columns - 1 cell moves."""
        ops = render(compose(1))
        kinds = [op.kind for op in ops]
        self.assertEqual(kinds[0], "style")
        self.assertEqual((ops[-1].kind, ops[-1].a), ("save", "final"))
        for i, op in enumerate(ops):
            if op.kind == "type":
                self.assertNotIn("\n", op.a)
                self.assertNotIn("\t", op.a)
            if op.kind == "exit_table":
                self.assertNotEqual(ops[i - 1].kind, "next_cell")
        # a table of r rows x c columns has r*c - 1 cell moves
        for i, op in enumerate(ops):
            if op.kind == "table":
                j = kinds.index("exit_table", i)
                self.assertEqual(kinds[i:j].count("next_cell"), op.a * op.b - 1)

    def test_typed_runs_replay_exactly(self):
        """The typing model's plan for each run (typos and corrections included) ends with exactly
        the run's text and never presses Enter or Tab."""
        from algorithms.human_typing import get_model, type_like_human
        from algorithms.typing_model.runtime import apply_plan
        model = get_model()
        self.assertIsNotNone(model)
        rng = np.random.default_rng(0)
        for op in render(compose(1)):
            if op.kind == "type":
                plan = type_like_human(op.a, rng=rng, wpm=55, mode=op.b, dry_run=True, max_pause=1.0)
                self.assertEqual(apply_plan(plan), op.a)
                self.assertFalse(any(k.key in ("\n", "\t") for k in plan))


class _StubHuman:
    """Just enough of a Human for a dry run: always or never prefers the keyboard."""

    class persona:
        """A persona with a middling shortcut preference."""
        shortcut_pref = 0.5

    def __init__(self, keyboard):
        """keyboard: True for a person who always uses shortcuts, False for the mouse."""
        self.keyboard = keyboard
        self.rng = np.random.default_rng(0)

    def prefers_keyboard(self, bias=0.0):
        """The fixed choice, whatever the bias."""
        return self.keyboard


class DryRunTest(unittest.TestCase):
    """The Writer executor's dry run (research/writer_exec.py): which actions each person takes."""

    def run_dry(self, keyboard):
        """Dry-run the whole paper for a keyboard (True) or mouse (False) person; the executor."""
        ex = WriterExecutor(_StubHuman(keyboard), None, None, save_path=r"C:\tmp\paper.odt", dry_run=True,
                            log=lambda *_: None)
        ops = render(compose(1))
        self.assertEqual(ex.run(ops), len(ops))
        return ex

    def test_keyboard_person(self):
        """A keyboard person uses shortcuts (Ctrl+1, Ctrl+F12 for a table, Ctrl+F11 for styles
        without one), saves as once and never clicks."""
        ex = self.run_dry(True)
        self.assertIn("HOTKEY ctrl+1", ex.actions)
        self.assertIn("HOTKEY ctrl+f12", ex.actions)            # Insert Table
        self.assertIn("HOTKEY ctrl+f11", ex.actions)            # the Apply Style box (Title has no shortcut)
        self.assertIn("HOTKEY shift+f12", ex.actions)           # bullets
        self.assertNotIn("HOTKEY ctrl+space", ex.actions)       # a non-breaking space in Writer
        self.assertEqual(ex.actions.count("SAVE AS " + r"C:\tmp\paper.odt"), 1)
        self.assertFalse(any(a.startswith("CLICK") for a in ex.actions))

    def test_mouse_person(self):
        """A mouse person clicks the Apply Style box, the toolbar and the Insert > Table... menu item."""
        ex = self.run_dry(False)
        self.assertIn("CLICK Formatting > Apply Style", ex.actions)
        self.assertIn("CLICK Insert > Table...", ex.actions)
        self.assertIn("CLICK Formatting > Bold", ex.actions)

    def test_stops_at_section_boundaries(self):
        """With stop_before_save the paper is written in several section-sized parts to the end."""
        ex = WriterExecutor(_StubHuman(True), None, None, save_path=r"C:\tmp\paper.odt", dry_run=True,
                            log=lambda *_: None)
        ops = render(compose(1))
        stops, i = [], 0
        while i < len(ops):
            i = ex.run(ops, i, stop_before_save=True)
            stops.append(i)
        self.assertGreater(len(stops), 5)
        self.assertEqual(stops[-1], len(ops))

    def test_a_new_executor_carries_on_with_the_state(self):
        """An executor started mid-paper takes bold / italic / style from the operations done
        so far (Writer can't be asked), so it doesn't toggle bold off when it means on."""
        ops = render(compose(1))
        i = next(k for k, op in enumerate(ops) if op.kind == "bold" and op.a) + 1    # just after a 'bold on'
        ex = WriterExecutor(_StubHuman(True), None, None, dry_run=True, log=lambda *_: None)
        ex.sync(ops[:i])
        self.assertTrue(ex.bold)


if __name__ == "__main__":
    unittest.main()
