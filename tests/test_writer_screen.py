"""
Reading OpenOffice Writer's window from screenshots (controller/writer_screen.py), on captures of
the real Writer 4.1 window (tests/fixtures/writer/): a heading, a two-line paragraph and a short
last line typed; the same with the last line highlighted (Shift+Home); the Insert menu open.
"""

import os
import unittest

import numpy as np
from PIL import Image

from controller.writer_screen import MENUS, View, find_view, highlighted, selected

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "writer")
MENU_LEFTS = [11, 56, 104, 159, 221, 295, 357, 419, 503]            # File, Edit, View, Insert, Format, Table, ...
LINE_TOPS = [409, 455, 484, 523]                                    # heading, paragraph (2 lines), last line


def load(name):
    return np.asarray(Image.open(os.path.join(FIXTURES, name + ".png")).convert("RGB"))


class ViewTest(unittest.TestCase):

    def test_page_lines_and_menus(self):
        view = find_view(load("writer_typed"))
        self.assertIsInstance(view, View)
        l, t, r, b = view.page
        self.assertLessEqual(abs(l - 163), 2)
        self.assertLessEqual(abs(r - 1351), 2)
        self.assertLessEqual(abs(t - 264), 2)
        self.assertEqual([line[1] for line in view.lines], LINE_TOPS)
        last = view.last_line
        self.assertLessEqual(abs(last[0] - 277), 3)               # the text area's left edge
        self.assertLess(last[2], 470)                               # 'A short closing line' is short
        self.assertGreater(view.lines[1][2], 1100)                  # the paragraph's first line is long
        self.assertEqual(len(view.menu_titles), len(MENUS))
        self.assertEqual([int(a) for a, _ in view.menu_titles], MENU_LEFTS)

    def test_highlight(self):
        """The highlighted last line reads as selected; nothing else does, and nothing is
        selected before Shift+Home."""
        rgb = load("writer_line_selected")
        view = find_view(rgb)
        self.assertTrue(selected(rgb, view.last_line))
        self.assertFalse(any(selected(rgb, line) for line in view.lines[:-1]))
        typed = load("writer_typed")
        self.assertFalse(selected(typed, find_view(typed).last_line))

    def test_formatting_boxes(self):
        """Apply Style, Font Name and Font Size on the Formatting toolbar, in every capture; none
        of them shows highlighted text."""
        for name in ("writer_typed", "writer_line_selected"):
            rgb = load(name)
            view = find_view(rgb)
            self.assertEqual(view.boxes, [(70, 124, 244, 154), (287, 124, 487, 154), (530, 124, 592, 154)], name)
            self.assertEqual(view.style_box, view.boxes[0])
            self.assertEqual(view.size_box, view.boxes[2])
            self.assertFalse(highlighted(rgb, view.style_box))
            self.assertFalse(highlighted(rgb, view.size_box))

    def test_offset(self):
        """A capture of the window (not the whole screen) gives screen coordinates back."""
        rgb = load("writer_typed")[40:, 30:]
        view = find_view(rgb, offset=(30, 40))
        self.assertEqual([line[1] for line in view.lines], LINE_TOPS)

    def test_not_a_document(self):
        self.assertIsNone(find_view(np.full((800, 1200, 3), 171, np.uint8)))


if __name__ == "__main__":
    unittest.main()
