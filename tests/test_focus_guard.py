"""
Never typing into the wrong window, offline: the keystroke player stops when another window
comes to the front (human_typing.play's focus guard), and opening Writer notices when Start
search launched a different OpenOffice program (tasks.foreground_writer_window's stray check).
Windows and keys are faked - nothing is pressed.
"""

import unittest
from unittest import mock

import numpy as np

import tasks
from algorithms import human_typing
from algorithms.typing_model.planner import Keystroke


def _plan(text, gap=0.01):
    return [Keystroke(key=ch, tag="type", press=i * gap, hold=0.005) for i, ch in enumerate(text)]


class FocusGuardTest(unittest.TestCase):
    def _play(self, fronts, focus_guard=True):
        """Play 'hello' while the window in front follows `fronts` (one per check); returns the
        keys that went down and whether every key was let go."""
        downs, ups = [], []
        seq = iter(fronts)
        last = [fronts[0]]

        def front():
            last[0] = next(seq, last[0])
            return last[0]
        with mock.patch.object(human_typing.apps, "foreground_top_level", side_effect=front), \
                mock.patch.object(human_typing.apps, "window_title", return_value="Other"), \
                mock.patch.object(human_typing.keyboard, "key_down", side_effect=downs.append), \
                mock.patch.object(human_typing.keyboard, "key_up", side_effect=ups.append), \
                mock.patch.object(human_typing.keyboard, "type_char"):
            try:
                human_typing.play(_plan("hello"), np.random.default_rng(0), focus_guard=focus_guard)
                stopped = False
            except human_typing.FocusLost:
                stopped = True
        return downs, set(downs) <= set(ups), stopped

    def test_types_everything_while_the_window_stays_in_front(self):
        downs, released, stopped = self._play([1] * 50)
        self.assertEqual("".join(downs), "hello")
        self.assertTrue(released)
        self.assertFalse(stopped)

    def test_stops_as_soon_as_another_window_comes_to_the_front(self):
        downs, released, stopped = self._play([1, 1, 1, 2])     # the start, 'h', 'e', then another window
        self.assertTrue(stopped)
        self.assertEqual("".join(downs), "he")
        self.assertTrue(released)

    def test_guard_off_types_through_a_focus_change(self):
        downs, _, stopped = self._play([1, 2], focus_guard=False)
        self.assertFalse(stopped)
        self.assertEqual("".join(downs), "hello")


class StrayOpenOfficeWindowTest(unittest.TestCase):
    WRITER, BASE, OLD = 10, 20, 30

    def _wait(self, front, titles, stray_ignore=()):
        clock = [0.0]

        def now():
            return clock[0]

        def sleep(s):
            clock[0] += s
        stray = []
        with mock.patch.object(tasks.time, "time", side_effect=now), \
                mock.patch.object(tasks.time, "sleep", side_effect=sleep), \
                mock.patch.object(tasks.apps, "foreground_top_level", side_effect=lambda: front(clock[0])), \
                mock.patch.object(tasks.apps, "window_class", return_value=tasks.W.WINDOW_CLASS), \
                mock.patch.object(tasks.apps, "window_title", side_effect=lambda h: titles.get(h, "")), \
                mock.patch.object(tasks.apps, "control_for", side_effect=lambda h: h):
            win = tasks.foreground_writer_window(timeout=60, stray=stray, stray_ignore=stray_ignore)
        return win, stray, clock[0]

    def test_writer_is_found(self):
        titles = {self.WRITER: "Untitled 1 - OpenOffice Writer"}
        win, stray, _ = self._wait(lambda t: self.WRITER if t > 3 else 0, titles)
        self.assertEqual(win, self.WRITER)
        self.assertEqual(stray, [])

    def test_database_wizard_instead_of_writer_is_given_up_on_quickly(self):
        titles = {self.BASE: "Database Wizard"}
        win, stray, waited = self._wait(lambda t: self.BASE if t > 2 else 0, titles)
        self.assertIsNone(win)
        self.assertEqual(stray, [self.BASE])
        self.assertLess(waited, 2 + tasks.STRAY_SECONDS + 1)

    def test_a_brief_untitled_frame_before_writer_is_not_stray(self):
        titles = {self.WRITER: "Untitled 1 - OpenOffice Writer"}
        front = lambda t: self.BASE if t < 1.5 else self.WRITER       # noqa: E731 - e.g. a window still starting
        win, stray, _ = self._wait(front, titles)
        self.assertEqual(win, self.WRITER)
        self.assertEqual(stray, [])

    def test_windows_open_before_are_not_stray(self):
        titles = {self.OLD: "Untitled 1 - OpenOffice Calc", self.WRITER: "Untitled 2 - OpenOffice Writer"}
        win, stray, _ = self._wait(lambda t: self.OLD if t < 20 else self.WRITER, titles, stray_ignore={self.OLD})
        self.assertEqual(win, self.WRITER)
        self.assertEqual(stray, [])


class StartCenterIsNotTheDocumentTest(unittest.TestCase):
    """Closing the last document turns its window into the Start Center (same handle): typing
    must never go there, even for a keyboard-only (inaccessible) Writer window."""

    def _kind(self, title):
        win = mock.Mock(NativeWindowHandle=5)
        with mock.patch.object(tasks.W.apps, "foreground_top_level", return_value=5),                 mock.patch.object(tasks.W.apps, "window_class", return_value=tasks.W.WINDOW_CLASS),                 mock.patch.object(tasks.W.apps, "window_title", return_value=title),                 mock.patch.object(tasks.W, "_INACCESSIBLE", {5}):
            return tasks.W.focus_kind(win)

    def test_blind_writer_in_front_takes_typing(self):
        self.assertEqual(self._kind("Untitled 1 - OpenOffice Writer"), "document")

    def test_start_center_in_the_same_window_does_not(self):
        self.assertEqual(self._kind("OpenOffice"), "elsewhere")


if __name__ == "__main__":
    unittest.main()
