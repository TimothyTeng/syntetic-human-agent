"""
Normal mode's budget sheet, offline: the generated expenses (spreadsheet_ops), the sheet the
plan builds (FakeSheet), the report's spending section and pasted table (office_writing,
writer_ops), and the Calc and Writer executors' dry runs.
"""

import datetime
import time
import unittest

import numpy as np

from office_writing import BUDGET_MIN_SENTENCES, count_sentences, generate_office_doc
from research import ir
from research.fake_writer import FakeWriter, expected
from research.ir import Table
from research.writer_exec import WriterExecutor
from research.writer_ops import render
from spreadsheet_exec import CalcExecutor
from spreadsheet_ops import (ITEMS, MAX_ROWS, MAX_TOTAL, MIN_ROWS, OPTIONAL, TOTAL, FakeSheet, generate_budget,
                             money, plan_sheet, rows_for_time)
from tests.test_research_paper import _StubHuman

TODAY = datetime.date(2026, 10, 21)


def budget(seed, n=8):
    return generate_budget(np.random.default_rng(seed), n, today=TODAY)


class BudgetTest(unittest.TestCase):
    """The expenses: last month, enough categories, nothing Calc's AutoInput would change."""

    def test_no_entry_starts_another(self):
        """AutoInput finishes a cell from the entries above it: no item, and no category (or
        'Total' typed under them), may be the start of another."""
        for column in ({i[0] for i in ITEMS}, {i[1] for i in ITEMS} | {TOTAL}):
            for a in column:
                for b in column:
                    self.assertFalse(a != b and b.lower().startswith(a.lower()), (a, b))

    def test_rows(self):
        for seed in range(80):
            b = budget(seed, n=seed % 15)
            self.assertTrue(MIN_ROWS <= len(b.rows) <= MAX_ROWS)
            self.assertGreaterEqual(len(b.categories), 3)
            self.assertEqual(len({r[3] for r in b.rows}), len(b.rows), "two amounts alike")
            self.assertEqual(b.month, "September")
            self.assertTrue(all(r[0].startswith("2026-09-") for r in b.rows))
            self.assertEqual(b.total, sum(b.spent(c) for c in b.categories))
            self.assertLessEqual(b.total, MAX_TOTAL, "a total of 1,000 or more shows differently by format")

    def test_money(self):
        self.assertEqual(money(123450), "1,234.50")
        self.assertEqual(money(705), "7.05")

    def test_rows_for_time(self):
        self.assertEqual(rows_for_time(10, 40), MIN_ROWS)
        self.assertEqual(rows_for_time(3600, 90), MAX_ROWS)


class SheetPlanTest(unittest.TestCase):
    """The operations build the sheet the report quotes."""

    def test_summary_matches_the_budget(self):
        """Every plan shows the summary the report pastes, in bold where meant, and the total
        under the expenses; the chart is always there, and the sheet is saved last."""
        for seed in range(120):
            b = budget(seed, n=MIN_ROWS + seed % (MAX_ROWS - MIN_ROWS + 1))
            sheet = FakeSheet().run(plan_sheet(b, np.random.default_rng(seed)))   # asserts no cell typed twice
            self.assertEqual(sheet.rows(b.summary_range), b.summary_rows, f"seed {seed}")
            n, k = len(b.rows), len(b.categories)
            self.assertEqual(sheet.display(f"D{n + 2}"), money(b.total))
            for ref in ("A1", "F1", "G1", f"C{n + 2}", f"G{k + 2}"):
                self.assertTrue(sheet.bold(ref), ref)
            self.assertIn("chart", sheet.done)
            self.assertEqual(sheet.done[-1], "save")

    def test_one_argument_formulas(self):
        """Calc's argument separator depends on the locale: no formula has more than one argument."""
        for seed in range(30):
            for op in plan_sheet(budget(seed), np.random.default_rng(seed)):
                if op.kind == "formula":
                    self.assertNotIn(",", op.a)
                    self.assertNotIn(";", op.a)

    def test_formula_copied_down(self):
        b = budget(5)
        sheet = FakeSheet().run(plan_sheet(b))
        n = len(b.rows) + 1
        self.assertEqual(sheet.cells[(7, 3)], f"=SUMPRODUCT(($C$2:$C${n}=F3)*$D$2:$D${n})")


class ReportTest(unittest.TestCase):
    """The report's spending section."""

    def test_exact_and_safe(self):
        for seed in range(150):
            n = [6, 7, 9, 12, 25, 35][seed % 6]
            b = budget(seed + 500, n=5 + seed % 8)
            doc = generate_office_doc(np.random.default_rng(seed), n, today=TODAY, budget=b)
            self.assertEqual(count_sentences(doc), n, f"seed {seed}")
            self.assertEqual(doc.meta["sentences"], n)
            self.assertEqual(ir.validate(doc), [])
            tables = [x for x in doc.blocks if isinstance(x, Table)]
            self.assertEqual(len(tables), 1)
            self.assertEqual((tables[0].rows, tables[0].source), (b.summary_rows, b.summary_range))
            self.assertIn("$" + money(b.total), ir.to_markdown(doc))

    def test_short_document_has_no_budget(self):
        doc = generate_office_doc(np.random.default_rng(1), BUDGET_MIN_SENTENCES - 1, today=TODAY, budget=budget(1))
        self.assertFalse(any(isinstance(x, Table) for x in doc.blocks))
        self.assertIsNone(doc.meta["budget"])

    def test_paste_comes_before_the_typed_table(self):
        """paste_table sits in the empty Text body paragraph, right before the typed fallback;
        FakeWriter (which types the fallback) still ends with the document."""
        doc = generate_office_doc(np.random.default_rng(4), 20, today=TODAY, budget=budget(4))
        ops = render(doc, np.random.default_rng(4))
        i = next(k for k, op in enumerate(ops) if op.kind == "paste_table")
        self.assertEqual(ops[i + 1].kind, "table")
        self.assertEqual(ops[i].a, budget(4).summary_range)
        self.assertEqual(FakeWriter().run(ops).result(), expected(doc))

    def test_outline_round_trip(self):
        doc = generate_office_doc(np.random.default_rng(2), 20, today=TODAY, budget=budget(2))
        self.assertEqual(ir.from_dict(ir.to_dict(doc)), doc)


class DryRunTest(unittest.TestCase):
    """The executors' dry runs."""

    def writer_run(self, paste_source):
        doc = generate_office_doc(np.random.default_rng(7), 20, today=TODAY, budget=budget(7))
        ops = render(doc)
        ex = WriterExecutor(_StubHuman(True), None, None, save_path=r"C:\tmp\doc.odt", dry_run=True,
                            log=lambda *_: None, paste_source=paste_source)
        self.assertEqual(ex.run(ops), len(ops))
        return ex

    def test_pasted_table_is_not_typed(self):
        calls = []
        ex = self.writer_run(lambda ref, rows: calls.append((ref, rows)) or "copied\n")
        self.assertEqual(calls[0][0], budget(7).summary_range)
        self.assertEqual(calls[0][1], budget(7).summary_rows)
        self.assertEqual(ex.methods["paste_table"], 1)
        self.assertFalse(any(m.startswith("table:") for m in ex.methods))

    def test_without_a_sheet_the_table_is_typed(self):
        ex = self.writer_run(None)
        self.assertNotIn("paste_table", ex.methods)
        self.assertTrue(any(m.startswith("table:") for m in ex.methods))

    def calc_run(self, keyboard, deadline=None):
        b = budget(9)
        ops = plan_sheet(b, np.random.default_rng(9))
        ex = CalcExecutor(_StubHuman(keyboard), None, save_path=r"C:	mpudget.ods", dry_run=True,
                          log=lambda *_: None)
        return ex, ex.run(ops, deadline=deadline), ops

    def test_calc_keyboard_person(self):
        """By keyboard: the Name Box with Ctrl+Shift+T, the formula copied with Ctrl+C / Ctrl+V, two
        decimals with Ctrl+Shift+1, the chart through the Insert menu's letters and the wizard's Finish."""
        ex, done, ops = self.calc_run(True)
        self.assertEqual(done, len(ops))
        self.assertTrue(ex.saved)
        for action in ("HOTKEY ctrl+shift+t", "HOTKEY ctrl+v", "HOTKEY ctrl+shift+1", "MENU Insert > Chart...",
                       "HOTKEY alt+f", "HOTKEY ctrl+shift+s"):
            self.assertIn(action, ex.actions)
        self.assertNotIn("HOTKEY ctrl+d", ex.actions)          # Selection List in Calc, not fill down
        self.assertEqual(ex.methods["chart:keys"], 1)

    def test_calc_mouse_person(self):
        ex, done, ops = self.calc_run(False)
        self.assertEqual(done, len(ops))
        self.assertEqual(ex.methods["chart:menu"], 1)

    def test_calc_short_of_time(self):
        """Past the deadline the optional steps are left out, but the chart is made and the sheet saved."""
        ex, done, ops = self.calc_run(True, deadline=time.monotonic() - 1)
        self.assertEqual(done, len([op for op in ops if op.kind not in OPTIONAL]))
        self.assertIn("chart:keys", ex.methods)
        self.assertTrue(ex.saved)


if __name__ == "__main__":
    unittest.main()
