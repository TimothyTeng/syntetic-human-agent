"""
Ctrl+Backspace word deletes, offline: OpenOffice Writer also deletes the space before the
word ('board rev|' -> 'board|'), unlike Word, Notepad and browsers. Plans made for Writer
(style 'openoffice') must still end in exactly the target text - the cases below are the ones
a live run in Writer typed wrong ('boardshould', 'questionsby', 'planwill').
"""

import unittest

import numpy as np

from algorithms.human_typing import use_word_deletes
from algorithms.typing_model import keys as K
from algorithms.typing_model.planner import Keystroke, TypingConfig, plan_keystrokes
from algorithms.typing_model.runtime import TypingModel, apply_plan


def _false_start(before, wrong, after):
    """Type `before + wrong`, Backspace it all away, type `after` - a false start."""
    plan, t = [], 0.0
    for ch in before:
        plan.append(Keystroke(ch, "type", press=t, hold=0.08))
        t += 0.15
    for ch in wrong:
        plan.append(Keystroke(ch, "false_start", press=t, hold=0.08))
        t += 0.15
    for _ in wrong:
        plan.append(Keystroke(K.BKSP, "false_start", press=t, hold=0.08))
        t += 0.1
    for ch in after:
        plan.append(Keystroke(ch, "type", press=t, hold=0.08))
        t += 0.15
    return plan


class WordDeleteStartTest(unittest.TestCase):
    def test_writer_takes_the_space_before_the_word(self):
        for text, windows, openoffice in [("board rev", "board ", "board"),
                                          ("plan can ", "plan ", "plan"),
                                          ("questions there", "questions ", "questions")]:
            buf = list(text)
            self.assertEqual(text[:K.word_delete_start(buf, len(buf))], windows)
            self.assertEqual(text[:K.word_delete_start(buf, len(buf), "openoffice")], openoffice)


class UseWordDeletesTest(unittest.TestCase):
    CASES = [("the project board ", "rev", "should revisit"),
             ("the survey questions ", "there", "by Friday."),
             ("the rollout plan ", "can ", "will move")]

    def test_windows_plans_go_wrong_in_writer(self):
        """The bug: a plan made for Windows-style Ctrl+Backspace, played in Writer."""
        before, wrong, after = self.CASES[0]
        plan = use_word_deletes(_false_start(before, wrong, after), np.random.default_rng(0), p=1.0)
        self.assertIn(K.WORD_BKSP, [k.key for k in plan])
        self.assertEqual(apply_plan(plan), before + after)
        self.assertEqual(apply_plan(plan, word_delete_style="openoffice"), "the project boardshould revisit")

    def test_openoffice_plans_type_the_space_again(self):
        for before, wrong, after in self.CASES:
            plan = use_word_deletes(_false_start(before, wrong, after), np.random.default_rng(0), p=1.0,
                                    style="openoffice")
            keys = [k.key for k in plan]
            self.assertIn(K.WORD_BKSP, keys)
            self.assertEqual(apply_plan(plan, word_delete_style="openoffice"), before + after)
            presses = [k.press for k in plan]
            self.assertEqual(presses, sorted(presses))

    def test_openoffice_skips_runs_it_cannot_predict(self):
        """After punctuation or at the start of a line Writer's deletion isn't certain: Backspaces stay."""
        for before in ("the draft timeline, ", "", "Notes\n"):
            plan = use_word_deletes(_false_start(before, "next ", "the end"), np.random.default_rng(0), p=1.0,
                                    style="openoffice")
            self.assertNotIn(K.WORD_BKSP, [k.key for k in plan])
            self.assertEqual(apply_plan(plan, word_delete_style="openoffice"), before + "the end")

    def test_planned_paragraphs_come_out_exact(self):
        model = TypingModel.load()
        rng = np.random.default_rng(3)
        text = ("The approval workflow was documented by the operations team at the start of the month. Once the "
                "vendor checked the draft timeline, the project board should revisit the survey questions by Friday.")
        used = 0
        for _ in range(40):
            cfg = TypingConfig(wpm=70, mode="compose", revision_scale=3.0, word_safe=True)
            plan = use_word_deletes(plan_keystrokes(text, model, rng, cfg), rng, p=1.0, style="openoffice")
            used += sum(k.key == K.WORD_BKSP for k in plan)
            self.assertEqual(apply_plan(plan, word_delete_style="openoffice"), text)
        self.assertGreater(used, 0)


if __name__ == "__main__":
    unittest.main()
