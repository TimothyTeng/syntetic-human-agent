"""
Reading OpenOffice Calc's grid from screenshots (controller/calc_screen.py), on captures of the
real Calc 4.1 window (tests/fixtures/calc/): the rows just typed, a chart drawn over the cells,
the columns fitted to their contents with the top row frozen, and the chart selected.
"""

import os
import unittest

import numpy as np
from PIL import Image

from controller.calc_screen import Grid, active_cell_at, find_grid, selected

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "calc")
DEFAULT_COLUMNS = [51, 173, 294, 416, 537, 659, 781, 902]           # A..G at Calc's default width
FITTED_COLUMNS = [51, 146, 342, 455, 548, 673, 785, 863]            # after Optimal Column Width
MENU_LEFTS = [11, 56, 104, 159, 221, 295, 357, 413, 495]            # File, Edit, ... Help


def load(name):
    return np.asarray(Image.open(os.path.join(FIXTURES, name + ".png")).convert("RGB"))


class GridTest(unittest.TestCase):

    def check(self, name, columns, active):
        rgb = load(name)
        grid = find_grid(rgb)
        self.assertIsInstance(grid, Grid, name)
        for got, want in zip(grid.xs, columns):
            self.assertLessEqual(abs(got - want), 2, f"{name}: column edges {grid.xs[:8]}")
        self.assertEqual(grid.ys[0], 241, name)
        self.assertGreaterEqual(grid.rows, 30, name)
        heights = np.diff(grid.ys)
        self.assertTrue(np.all(np.abs(heights - np.median(heights)) <= 3), f"{name}: row edges {grid.ys}")
        self.assertEqual(grid.header, (215, 240))
        self.assertEqual([int(a) for a, _ in grid.menu_titles], MENU_LEFTS)
        cursors = [c + str(r) for c in "ABCDEFG" for r in range(1, 15) if active_cell_at(rgb, grid.cell_rect(c + str(r)))]
        self.assertEqual(cursors, [active], name)
        return grid

    def test_rows_typed(self):
        grid = self.check("rows_typed", DEFAULT_COLUMNS, "A7")
        self.assertEqual(grid.cell_rect("B2"), (173, 266, 294, 292))
        self.assertTrue(grid.has("A1:G14"))
        self.assertFalse(grid.has("A1:A40"))

    def test_chart_over_the_cells(self):
        """A chart's axis inside a column is not taken for a column edge."""
        self.check("chart_inserted", DEFAULT_COLUMNS, "F1")

    def test_fitted_columns_and_frozen_row(self):
        """The frozen row's line and the neighbour's frame don't fake a second cell cursor."""
        self.check("widths_fitted", FITTED_COLUMNS, "A2")
        self.check("chart_selected", FITTED_COLUMNS, "A2")

    def test_window_frame_is_not_a_column(self):
        """A capture that starts left of the window's frame (a maximised window reaches past the
        screen's edge) still has column A first - a frame line taken for an edge put every click
        one column to the left in a live run."""
        rgb = load("rows_typed")
        framed = np.zeros((rgb.shape[0] + 8, rgb.shape[1] + 8, 3), np.uint8)
        framed[8:, 8:] = rgb
        framed[:, 8:10] = 128                       # the frame's dark line
        grid = find_grid(framed, offset=(-8, -8))
        self.assertEqual(grid.xs[:4], DEFAULT_COLUMNS[:4])

    def test_nothing_selected(self):
        rgb = load("rows_typed")
        grid = find_grid(rgb)
        self.assertFalse(selected(rgb, grid.range_rect("A1:D1")))

    def test_not_a_sheet(self):
        """A screen without Calc's grid (here: a blank white one) gives None, not a guess."""
        self.assertIsNone(find_grid(np.full((800, 1200, 3), 255, np.uint8)))


if __name__ == "__main__":
    unittest.main()
