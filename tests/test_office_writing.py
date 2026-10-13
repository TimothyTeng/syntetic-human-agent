"""
Normal mode's writing, offline: the generated office document (office_writing.py) and the
formatting done after typing (writer_ops select / deselect), checked against FakeWriter and
the Writer executor's dry run.
"""

import datetime
import unittest

import numpy as np

from office_writing import count_sentences, generate_office_doc
from research import ir
from research.fake_writer import FakeWriter, expected
from research.ir import Document, Paragraph, Run, Title
from research.writer_exec import WriterExecutor
from research.writer_ops import render
from tests.test_research_paper import _StubHuman, compose

TODAY = datetime.date(2026, 10, 12)


def generate(seed, n=25, kind=None):
    return generate_office_doc(np.random.default_rng(seed), n, kind, today=TODAY)


class OfficeDocumentTest(unittest.TestCase):
    """The generated document: exact length, safe to type, varied."""

    def test_exact_sentence_count(self):
        """Every target length gives exactly that many sentences (bullets and bold lines included)."""
        for seed in range(60):
            n = [1, 2, 4, 7, 12, 25, 40][seed % 7]
            doc = generate(seed, n)
            self.assertEqual(doc.meta["sentences"], n, f"seed {seed}")
            self.assertEqual(count_sentences(doc), n, f"seed {seed}")

    def test_safe_to_type(self):
        """ASCII only and nothing AutoCorrect / AutoFormat would rewrite."""
        for seed in range(60):
            self.assertEqual(ir.validate(generate(seed)), [], f"seed {seed}")

    def test_variety(self):
        """Titles, kinds and sentence structures vary from run to run; no sentence repeats within one
        document and the same structure never comes twice in a row."""
        docs = [generate(seed) for seed in range(40)]
        self.assertGreater(len({d.meta["title"] for d in docs}), 25)
        self.assertEqual({d.meta["kind"] for d in docs}, {"status report", "meeting notes", "project proposal", "memo"})
        for d in docs:
            kinds = [k for k in d.meta["structures"] if k != "reference"]
            self.assertGreaterEqual(len(set(kinds)), 6)
            self.assertFalse(any(a == b for a, b in zip(kinds, kinds[1:])), d.meta["title"])
            sentences = [s for b in d.blocks if isinstance(b, Paragraph)
                         for s in b.text.replace("? ", "?|").replace(". ", ".|").split("|")]
            self.assertEqual(len(sentences), len(set(sentences)))

    def test_frame(self):
        """A title that fits on one line, headed sections, some bold or italic text across runs."""
        docs = [generate(seed) for seed in range(30)]
        for d in docs:
            self.assertIsInstance(d.blocks[0], Title)
            self.assertLessEqual(len(d.blocks[0].text), 30)
            self.assertGreaterEqual(sum(isinstance(b, ir.Heading) for b in d.blocks), 2)
        runs = [r for d in docs for b in d.blocks if isinstance(b, Paragraph) for r in b.runs]
        self.assertTrue(any(r.bold for r in runs))
        self.assertTrue(any(r.italic for r in runs))
        self.assertTrue(any(isinstance(b, ir.Bullets) for d in docs for b in d.blocks))


class FormatAfterTypingTest(unittest.TestCase):
    """Formatting applied by highlighting text that was already typed (select, format, deselect)."""

    def test_fake_writer_ends_with_the_document(self):
        """With formatting afterwards, office documents and research papers still come out exactly as
        meant - including a title resized after typing."""
        for seed in range(25):
            doc = generate(seed)
            ops = render(doc, np.random.default_rng(seed), p_post=0.6)
            self.assertEqual(FakeWriter().run(ops).result(), expected(doc), f"office seed {seed}")
        for seed in range(4):
            doc = compose(seed)
            doc.blocks[0].size = 32.0
            ops = render(doc, np.random.default_rng(seed), p_post=0.8)
            self.assertEqual(FakeWriter().run(ops).result(), expected(doc), f"paper seed {seed}")

    def test_operations(self):
        """A bold phrase typed plain then highlighted word by word; a bold one-line paragraph and a
        resized title highlighted as a line; formatting at the caret switched off before plain text
        goes on; never typing while text is selected."""
        doc = Document([Title("Weekly Update", size=24.0), Paragraph([Run("Key decision made", bold=True)]),
                        Paragraph([Run("The "), Run("rollout plan", bold=True), Run(" is ready.")])])
        ops = [str(op) for op in render(doc, np.random.default_rng(0), p_post=1.0)]
        self.assertIn("select(2, 'line')", ops)                    # the title
        i = ops.index("type('rollout plan', 'transcribe')")
        self.assertEqual(ops[i + 1:i + 5], ["select(2, 'words')", "bold(True)", "deselect()", "bold(False)"])
        selecting = False
        for op in render(doc, np.random.default_rng(0), p_post=1.0):
            selecting = op.kind == "select" or (selecting and op.kind != "deselect")
            if selecting:
                self.assertNotEqual(op.kind, "type")

    def test_without_rng_nothing_changes(self):
        """Without an rng everything is formatted before typing, as before."""
        ops = render(compose(1))
        self.assertFalse(any(op.kind in ("select", "deselect") for op in ops))

    def test_same_seed_same_operations(self):
        """Resuming relies on the same render seed giving the same operations."""
        doc = generate(3)
        a = render(doc, np.random.default_rng(42))
        b = render(doc, np.random.default_rng(42))
        self.assertEqual(a, b)

    def run_dry(self, keyboard, seed=5):
        ex = WriterExecutor(_StubHuman(keyboard), None, None, save_path=r"C:\tmp\doc.odt", dry_run=True,
                            log=lambda *_: None)
        found = set()
        for s in range(seed, seed + 12):          # several documents, so every way shows up
            doc = generate(s)
            doc.blocks[0].size = 26.0
            ops = render(doc, np.random.default_rng(s), p_post=0.9)
            ex.__init__(_StubHuman(keyboard), None, None, save_path=r"C:\tmp\doc.odt", dry_run=True,
                        log=lambda *_: None)
            ex.human.rng = np.random.default_rng(s)
            self.assertEqual(ex.run(ops), len(ops))
            found |= set(ex.methods)
        return found

    def test_keyboard_person_highlights_with_keys(self):
        """Keyboard people highlight with Ctrl+Shift+Left / Shift+Home and resize with Ctrl+]."""
        found = self.run_dry(True)
        self.assertIn("select:ctrl_shift_left", found)
        self.assertIn("select:shift_home", found)
        self.assertIn("size:steps", found)
        self.assertIn("bold:key", found)
        self.assertFalse(any(m.startswith(("select:drag", "select:triple")) for m in found))

    def test_mouse_person_highlights_with_the_mouse(self):
        """Mouse people drag across or triple-click a line, and use the toolbar's Bold and Font Size."""
        found = self.run_dry(False)
        self.assertTrue({"select:drag", "select:triple_click"} & found)
        self.assertIn("bold:toolbar", found)
        self.assertIn("size:toolbar", found)


if __name__ == "__main__":
    unittest.main()
