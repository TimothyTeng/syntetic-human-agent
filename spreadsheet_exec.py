"""
Carrying out spreadsheet_ops operations in a real OpenOffice Calc window, as the simulated person.

    ex = CalcExecutor(human, win, save_path=r"...\\September Budget.ods", log=log)
    ex.run(plan_sheet(budget, human.rng), deadline=time.monotonic() + 240)

OpenOffice 4.1 shows UI Automation only the first window of a running OpenOffice (see
tasks.open_writer_document), and on some machines none at all. So the window's title and its
dialogs' titles (plain Win32) are the checks, and where things are on screen is read from a
screenshot (controller.calc_screen: the grid's column and row edges, the column headers, the
menu bar's titles):
  select a range   mouse people click the cell or drag across the range - checked on a fresh
                   screenshot (the cell cursor's frame, the selection's tint), the Name Box if
                   that shows anything else; otherwise Ctrl+Shift+T (the Name Box), Ctrl+A,
                   the reference typed, Enter
  rows             Tab between cells, Enter and Home at the end of a row
  copy a formula   Ctrl+C on its cell, the cells below selected, Ctrl+V, Esc
  bold             Ctrl+B (or the Bold button where it can be found)
  two decimals     Ctrl+Shift+1
  chart            Insert > Chart... (clicked, or Alt+I, C, C - Insert Cells shares the C -
                   Enter), the Chart Wizard's Finish (Alt+F), then Esc twice: out of the chart
                   and away from it
  optimal width    mouse people drag across the column headers and double-click a border
                   between two of them (checked: the columns' widths change); otherwise the
                   columns selected (Ctrl+Shift+Space), Format > Column > Optimal Column
                   Width... (Alt+O, M, O), OK
  freeze top row   the cursor on A2, Window > Freeze Window (Alt+W, F)
  rename sheet     Format > Sheet > Rename Sheet... (Alt+O, S, R), the name typed, OK
  save             Save As (Ctrl+Shift+S), the full path typed (an .ods)
Mouse people click a menu's title on the menu bar (checked: a menu window opens) and pick the
entry by its letter; else menus are opened by their letters (Alt+letter). When
a step's dialog doesn't appear the menus are closed with Esc and the step is recorded as a
problem - nothing else is pressed. Numbers, dates, formulas, references and the sheet's name
are typed without typos (a slip there would change the totals the report quotes); item names
get the typing model's usual typos and corrections.

While typing, the mouse isn't left frozen: now and then it drifts to where the person is
looking - the row being typed, the cell a formula goes into - and rests there.

Nothing is typed unless Calc's window is in front.
dry_run=True performs nothing and records the actions it would take (self.actions).
"""

import os
import re
import time
from collections import Counter

from algorithms.session import TaskError
from controller import apps
from controller import calc_screen as CS
from controller import writer as W
from spreadsheet_ops import OPTIONAL

NUMERIC = re.compile(r"[=$\d][\w$.,:()=*-]*")     # numbers, dates, references and formulas: typed exactly
OPTIONAL_RESERVE = 25.0                           # seconds kept for saving when deciding on optional steps
TITLE_SUFFIX = "OpenOffice Calc"                  # "Untitled 2 - OpenOffice Calc"
CHART_NUDGES = 8                                  # arrow presses that move a new chart off the Total row
MENUS = ("File", "Edit", "View", "Insert", "Format", "Tools", "Data", "Window", "Help")   # the menu bar, in order
P_DRIFT_ROW, P_DRIFT_FORMULA = 0.35, 0.6           # chances the mouse drifts to the row / formula being typed


def is_calc_window(hwnd):
    return apps.window_class(hwnd) == W.WINDOW_CLASS and apps.window_title(hwnd).endswith(TITLE_SUFFIX)


class CalcExecutor:
    """Performs spreadsheet_ops operations in Calc as one simulated person and collects how
    each was done (methods) and what went wrong (problems) for the log."""

    def __init__(self, human, win, save_path=None, log=print, dry_run=False):
        """win: the Calc window (a UIA control; None in a dry run); save_path: the .ods the save
        operation names the workbook."""
        self.human, self.win, self.save_path, self.log, self.dry_run = human, win, save_path, log, dry_run
        self.hwnd = None if win is None else win.NativeWindowHandle
        self.grid = None                       # controller.calc_screen.Grid, read when the mouse needs it
        self.at = "A1"                         # the cell the cursor is on, as far as we know
        self.calibrated = None                 # the screen's grid checked against a Name Box jump (_calibrate)
        self.saved = False
        self.actions = []
        self.methods = Counter()
        self.problems = []

    # --- running ------------------------------------------------------------------------------

    def run(self, ops, deadline=None):
        """Execute ops in order. With a deadline (time.monotonic()) the optional steps (OPTIONAL)
        are left out once too little time is left; everything else is needed for the sheet's
        figures and always done. Returns the number of operations performed."""
        done = 0
        for op in ops:
            if op.kind in OPTIONAL and deadline is not None and time.monotonic() > deadline - OPTIONAL_RESERVE:
                self.log(f"Short of time - leaving out {op.kind}")
                continue
            self.log(f"Sheet: {op.kind}" + (f" {op.a}" if isinstance(op.a, str) else ""))
            getattr(self, "op_" + op.kind)(op)
            done += 1
        return done

    def summary(self):
        ways = ", ".join(f"{k} x{v}" for k, v in sorted(self.methods.items()))
        return ways + (f"; problems: {'; '.join(self.problems)}" if self.problems else "")

    # --- primitives (recorded instead of performed in a dry run) --------------------------------

    def _hotkey(self, *keys):
        if self.dry_run:
            self.actions.append("HOTKEY " + "+".join(keys))
        else:
            self.human.hotkey(*keys)

    def _press(self, key, presses=1):
        if self.dry_run:
            self.actions.append(f"PRESS {key}" + (f" x{presses}" if presses > 1 else ""))
        else:
            self.human.press(key, presses)

    def _type(self, text, exact=None):
        """Type text: exactly for numbers, dates, formulas and references (or exact=True), with
        the usual typos for words."""
        if self.dry_run:
            self.actions.append(f"TYPE {text!r}")
            return
        if exact or (exact is None and NUMERIC.fullmatch(text)):
            self.human.type_short(text, error_scale=0.0)
        else:
            self.human.type_short(text)

    def _think(self, kind="glance", scale=1.0):
        if not self.dry_run:
            self.human.think(kind, scale)

    def _sleep(self, seconds):
        if not self.dry_run:
            time.sleep(seconds)

    def _keyboard(self, bias=0.0):
        return self.human.prefers_keyboard(bias)

    def _front(self):
        """Calc's window must be in front before anything is pressed (TaskError otherwise)."""
        if self.dry_run or apps.foreground_top_level() == self.hwnd:
            return
        if not apps.bring_to_front(self.hwnd) or apps.foreground_top_level() != self.hwnd:
            raise TaskError("Calc isn't the window in front - stopping before typing anything")

    def _find(self, name, control_type=None, timeout=0.5):
        """A visible control of the Calc window (where Calc exposes it to UI Automation), or None."""
        from controller import ui_elements
        if self.dry_run or W.is_inaccessible(self.win):
            return None
        ctrl = ui_elements.find_element(self.win, name, control_type, timeout=timeout)
        return ctrl if ctrl is not None and ui_elements.is_visible(ctrl) else None

    def _menu(self, path, letters, label):
        """A menu command: clicked through the menus (path: the menu-bar menu, then item regexes)
        by mouse people where Calc exposes its menus, else Alt+letters[0] and the other letters /
        keys one at a time. The way used ('menu' / 'keys')."""
        if self.dry_run:
            self.actions.append(f"MENU {label}")
            return "keys" if self._keyboard(0.1) else "menu"
        if not self._keyboard(0.1):
            menu = W.menu(self.win, path[0])
            if menu is None and self._click_menu_title(path[0]):
                for key in letters[1:]:                # the entries by their letters
                    time.sleep(0.35)
                    self._press(key)
                return "title"
            if menu is not None:
                self.human.click_element(menu)
                time.sleep(0.4)
                parent = menu
                for k, name in enumerate(path[1:]):
                    item = W.menu_item(parent, name)
                    if item is None:
                        self._press("esc", presses=k + 1)
                        break
                    self._think("glance", 0.6)
                    self.human.click_element(item)
                    time.sleep(0.4)
                    parent = item
                else:
                    return "menu"
        self._hotkey("alt", letters[0])
        for key in letters[1:]:
            time.sleep(0.35)
            self._press(key)
        return "keys"

    def _dialog(self, title, timeout=4.0):
        """The dialog with this title (a regex) once it is in front, or None (menus closed with Esc)."""
        if self.dry_run:
            return True
        dialog = W.find_dialog(title, timeout=timeout)
        if dialog is None:
            self._press("esc", presses=3)
        return dialog

    # --- the mouse, aimed from a screenshot ---------------------------------------------------------

    def _read_grid(self):
        """Read where the cells are from a screenshot of Calc (None if they can't be found)."""
        if self.dry_run or self.hwnd is None:
            return None
        try:
            self.grid = CS.read_grid(self.hwnd)
        except Exception as exc:                # a capture failing is no reason to stop
            self.log(f"Couldn't read Calc's grid from the screen ({type(exc).__name__}: {exc})")
            self.grid = None
        return self.grid

    def _inside(self, rect, margin=0.25):
        """A random point well inside rect (left, top, right, bottom)."""
        l, t, r, b = rect
        rng = self.human.rng
        return (int(rng.uniform(l + (r - l) * margin, r - (r - l) * margin)),
                int(rng.uniform(t + (b - t) * margin, b - (b - t) * margin)))

    def _select_by_mouse(self, ref):
        """Click the cell (or drag across the range) where the screenshot shows it, then check on
        a fresh one that the cell cursor / the selection is there. False (the Name Box takes over,
        whatever got selected) if the grid can't be read, the range isn't on screen or the check
        fails."""
        if self.dry_run:
            self.actions.append(f"{'DRAG' if ':' in ref else 'CLICK'} {ref}")
            self.methods["select:mouse"] += 1
            return True
        if not self.calibrated:
            return False
        grid = self._read_grid()
        if grid is None or not grid.has(ref):
            return False
        first, _, last = ref.replace("$", "").partition(":")
        if last and last != first:
            self.human.drag(self._inside(grid.cell_rect(first)), self._inside(grid.cell_rect(last)))
            how = "drag"
        else:
            self.human.click_rect(_shrink(grid.cell_rect(first), 0.2))
            how = "click"
        time.sleep(0.35)
        rgb, offset = CS.capture(self.hwnd)
        ok = CS.selected(rgb, grid.range_rect(ref), offset) if how == "drag" \
            else CS.active_cell_at(rgb, grid.cell_rect(first), offset)
        if not ok:
            self.log(f"The {how} didn't select {ref} - using the Name Box")
            self.methods[f"select:{how}_missed"] += 1
            return False
        self.methods[f"select:{how}"] += 1
        return True

    def _drift(self, ref, p):
        """With chance p, let the mouse drift to where the person is looking: just right of the
        cell `ref` (where the text being typed shows), and rest it there."""
        if self.dry_run or not self.calibrated or self.human.rng.random() >= p:
            return
        grid = self.grid if self.grid is not None and self.grid.has(ref) else self._read_grid()
        if grid is None or not grid.has(ref):
            return
        l, t, r, b = grid.cell_rect(ref)
        rng = self.human.rng
        x = r + rng.uniform(15, 160)
        y = rng.uniform(t - (b - t) * 0.4, b + (b - t) * 0.4)
        self.human.move_to(int(x), int(y))
        self.methods["mouse:drift"] += 1

    def _look_at_dialog(self):
        """Mouse people move the mouse over the dialog in front while reading it (its window rect
        is plain Win32, so this works where nothing inside it can be found)."""
        if self.dry_run or self._keyboard(0.2):
            return
        l, t, r, b = CS.window_rect(apps.foreground_top_level())
        if r - l > 60 and b - t > 60:
            self.human.move_to(*self._inside((l, t, r, b), 0.3))
            self.methods["mouse:dialog"] += 1

    def _click_menu_title(self, name):
        """Click a menu's title on the menu bar (found on a screenshot); True once a menu window
        has opened (it is closed again with Esc if something else happened)."""
        if self.dry_run or name not in MENUS:
            return False
        grid = self.grid if self.grid is not None and self.grid.menu_y else self._read_grid()
        if grid is None or len(grid.menu_titles) != len(MENUS):
            return False
        before = set(apps.top_level_windows())
        self.human.click_rect(grid.menu_rect(MENUS.index(name)))
        deadline = time.time() + 1.5
        while time.time() < deadline:
            new = [h for h in set(apps.top_level_windows()) - before if apps.window_class(h).startswith("SAL")]
            if new:
                time.sleep(0.25)
                self.methods["menu:title_click"] += 1
                return True
            time.sleep(0.1)
        self.log(f"Clicking '{name}' didn't open its menu")
        self._press("esc")
        return False

    def _wait_title(self, title, timeout):
        """True once a window whose title matches `title` (a regex) is in front."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if re.fullmatch(title, apps.window_title(apps.foreground_top_level()) or ""):
                time.sleep(0.4)                        # the dialog settles (its name box takes the focus)
                return True
            time.sleep(0.2)
        return False

    def select(self, ref):
        """Select a cell or range through the Name Box: Ctrl+Shift+T (or a click on it where Calc
        exposes it), the address in it highlighted (Ctrl+A - the shortcut leaves the cursor in
        front of it, so typing would add to it: 'C7A7'), the reference typed, Enter."""
        self._front()
        self.at = ref.split(":")[0].replace("$", "")
        if not self._keyboard(0.2) and self._select_by_mouse(ref):
            return
        box = None if self._keyboard(0.2) else self._find("Name Box", "ComboBoxControl")
        if box is not None:
            self.human.click_element(box)
            how = "click"
        else:
            self._hotkey("ctrl", "shift", "t")
            how = "key"
        self._sleep(0.3)
        self._hotkey("ctrl", "a")
        self._think("glance", 0.7)
        self._type(ref, exact=True)
        self._press("enter")
        self._sleep(0.3)
        self.methods[f"select:{how}"] += 1
        if self.calibrated is None:
            self._calibrate()

    def _calibrate(self):
        """Once, right after the Name Box put the cursor on a known cell: check the cell cursor is
        where the screenshot's grid says that cell is. The mouse only aims at cells once this
        held - a grid read off by a column would click (and type into) the wrong cells while
        still checking out against itself."""
        if self.dry_run:
            self.calibrated = True
            return
        grid = self._read_grid()
        if grid is None or not grid.has(self.at):
            return                                  # try again after the next Name Box jump
        rgb, offset = CS.capture(self.hwnd)
        self.calibrated = CS.active_cell_at(rgb, grid.cell_rect(self.at), offset)
        if not self.calibrated:
            self.log(f"The cell cursor isn't where the screen's grid puts {self.at} - not aiming the mouse at cells")
        self.methods["mouse:calibrated" if self.calibrated else "mouse:calibration_failed"] += 1

    # --- operations ------------------------------------------------------------------------------

    def op_goto(self, op):
        self.select(op.a)

    def op_rows(self, op):
        """Type rows from the active cell: Tab between cells, Enter and Home after each row."""
        self._front()
        self._think("mental", 0.6)
        col, row_no = CS.split_ref(self.at)
        for k, row in enumerate(op.a):
            self._drift(f"{CS.col_letters(col + len(row) - 1)}{row_no + k}", P_DRIFT_ROW)
            for j, value in enumerate(row):
                self._type(value)
                if j + 1 < len(row):
                    self._press("tab")
            self._press("enter")
            self._press("home")
            self._think("glance")
        self.methods["rows"] += len(op.a)

    def op_cell(self, op):
        self._front()
        self._type(op.a)
        self._press(op.b)

    def op_column(self, op):
        self._front()
        for value in op.a:
            self._type(value)
            self._press("enter")
        self.methods["rows"] += len(op.a)

    def op_formula(self, op):
        self._front()
        self._drift(self.at, P_DRIFT_FORMULA)
        self._think("mental")                          # working out the formula
        self._type(op.a, exact=True)
        self._think("confirm", 0.6)
        self._press("enter")
        self.methods["formula"] += 1

    def op_copy_to(self, op):
        """Copy the formula in op.a to the cells op.b (Ctrl+C, select, Ctrl+V), then Esc."""
        self.select(op.a)
        self._hotkey("ctrl", "c")
        self._sleep(0.3)
        self.select(op.b)
        self._hotkey("ctrl", "v")
        self._sleep(0.4)
        self._press("esc")
        self.methods["copy_formula"] += 1

    def op_bold(self, op):
        self.select(op.a)
        button = None if self._keyboard(0.2) else self._find("Bold", "ButtonControl")
        if button is not None:
            self.human.click_element(button)
            self.methods["bold:button"] += 1
        else:
            self._hotkey("ctrl", "b")
            self.methods["bold:key"] += 1

    def op_number_format(self, op):
        self.select(op.a)
        self._hotkey("ctrl", "shift", "1")
        self.methods["number_format"] += 1

    def op_chart(self, op):
        """A column chart of op.a: Insert > Chart..., Finish in the Chart Wizard, out of the chart."""
        self.select(op.a)
        self._think("mental", 0.5)
        how = self._menu(("Insert", r"Chart\.\.\."), ("i", "c", "c", "enter"), "Insert > Chart...")
        dialog = self._dialog(r"Chart Wizard", timeout=10)        # the chart module loads the first time
        if dialog is None:
            self.problems.append("the Chart Wizard didn't open - no chart")
            return
        self._look_at_dialog()
        self._think("scan")                            # a look at the preview
        self._hotkey("alt", "f")                       # ~Finish
        self._sleep(1.5)
        if not self.dry_run and W.find_dialog(r"Chart Wizard", timeout=0) is not None:
            self._press("enter")
            self._sleep(1.5)
        self._think("scan")                            # a look at the chart
        self._press("esc")                             # out of the chart's edit mode: the chart is selected
        self._sleep(0.4)
        # Calc puts the chart just under the range, over the summary's Total row: nudge it down
        # (each arrow press moves a selected object about 1 mm; with nothing selected the arrows
        # only move the cell cursor)
        self._think("glance")
        if self.dry_run:
            self.actions.append(f"PRESS down x{CHART_NUDGES}")
        else:
            for _ in range(CHART_NUDGES):
                self.human.press("down")
                time.sleep(float(self.human.rng.uniform(0.05, 0.18)))
        self._sleep(0.3)
        self._press("esc")                             # the chart no longer selected
        self._sleep(0.3)
        self.methods[f"chart:{how}"] += 1

    def op_optimal_width(self, op):
        """The columns of op.a to their optimal width (Format > Column > Optimal Column Width...)."""
        if not self._keyboard(0.1) and self._optimal_width_by_mouse(op.a):
            return
        self.select(op.a)
        self._hotkey("ctrl", "shift", "space")         # whole columns
        self._sleep(0.3)
        how = self._menu(("Format", r"Colu?mn", r"Optimal Column Width\.\.\."), ("o", "m", "o"),
                         "Format > Column > Optimal Column Width...")
        if self._dialog(r"Optimal Column Width") is None:
            self.problems.append("Optimal Column Width didn't open")
            return
        self._think("confirm")
        self._press("enter")
        self._sleep(0.4)
        self.methods[f"optimal_width:{how}"] += 1

    def _optimal_width_by_mouse(self, ref):
        """Drag across the column headers of ref's columns, double-click the border right of the
        first one (Calc fits every selected column), then check the widths changed. False (the
        columns stay selected; the menu does it) if the grid can't be read or nothing changed."""
        if self.dry_run:
            self.actions.append(f"DRAG column headers {ref}; DOUBLE-CLICK a header border")
            self.methods["optimal_width:mouse"] += 1
            return True
        if not self.calibrated:
            return False
        self._front()
        grid = self._read_grid()
        if grid is None or not grid.has(ref):
            return False
        first, _, last = ref.partition(":")
        a, b = re.match(r"[A-Z]+", first).group(), re.match(r"[A-Z]+", last or first).group()
        self.human.drag(self._inside(grid.column_header_rect(a), 0.3), self._inside(grid.column_header_rect(b), 0.3))
        time.sleep(0.3)
        self._think("glance")
        edge = grid.xs[CS.col_number(a)]
        top, bottom = grid.header
        self.human.double_click_rect((edge - 1, top + (bottom - top) // 3, edge + 1, bottom - (bottom - top) // 3))
        time.sleep(0.6)
        # click back into the sheet (whole columns selected tint every cell blue, and the grid is
        # read off white cells) - at the left end of A1, which stays put however wide A becomes
        x0, (t1, b1) = grid.xs[0], grid.ys[:2]
        self.human.click_rect((x0 + 10, t1 + 6, x0 + 30, b1 - 6))
        self.at = "A1"
        time.sleep(0.4)
        after = self._read_grid()
        if after is None or after.xs[:len(grid.xs)] == grid.xs[:len(after.xs)]:
            self.log("Double-clicking the header border didn't change the widths - using the menu")
            self.methods["optimal_width:mouse_missed"] += 1
            return False
        self.methods["optimal_width:mouse"] += 1
        return True

    def op_freeze_below(self, op):
        """Freeze the rows above op.a (Window > Freeze Window with the cursor there)."""
        self.select(op.a)
        how = self._menu(("Window", r"Freeze Window"), ("w", "f"), "Window > Freeze Window")
        self._sleep(0.4)
        self.methods[f"freeze:{how}"] += 1

    def op_rename_sheet(self, op):
        self._front()
        how = self._menu(("Format", r"Sheet", r"Rename Sheet\.\.\."), ("o", "s", "r"),
                         "Format > Sheet > Rename Sheet...")
        if self._dialog(r"Rename Sheet") is None:
            self.problems.append("Rename Sheet didn't open")
            return
        self._think("glance")
        self._hotkey("ctrl", "a")                      # the old name is selected already; to be sure
        self._type(op.a, exact=True)
        self._think("confirm")
        self._press("enter")
        self._sleep(0.4)
        self.methods[f"rename:{how}"] += 1

    def op_save(self, op):
        """Save As (Ctrl+Shift+S) to save_path the first time; Ctrl+S afterwards."""
        if not self.save_path:
            return
        self._front()
        if self.dry_run:
            self.actions.append("HOTKEY ctrl+s" if self.saved else "HOTKEY ctrl+shift+s")
            self.actions.append(f"TYPE {self.save_path!r}")
            self.saved = True
            return
        if self.saved:
            self.human.hotkey("ctrl", "s")
        else:
            # OpenOffice takes a few seconds to bring up Save As: nothing is typed until it is in
            # front (keys sent earlier would go into the sheet)
            existed = os.path.exists(self.save_path)
            self.human.hotkey("ctrl", "shift", "s")
            if not self._wait_title(r"Save As", 15):
                self.problems.append("Save As didn't open")
                return
            self._think("scan", 0.6)
            self.human.select_all_and_type(self.save_path, error_scale=0.0)
            self._think("confirm")
            self._press("enter")
            if existed:
                self.human.answer_prompt(r"^Yes$")
        deadline = time.time() + 15
        while not os.path.exists(self.save_path) and time.time() < deadline:
            time.sleep(0.3)
        if os.path.exists(self.save_path):
            self.saved = True
            self.methods["save"] += 1
            self.log(f"Saved {self.save_path}")
        else:
            self.problems.append("the spreadsheet doesn't seem to have been saved")


def _shrink(rect, frac):
    """rect with frac of its width and height taken off each side."""
    l, t, r, b = rect
    dx, dy = (r - l) * frac, (b - t) * frac
    return int(l + dx), int(t + dy), int(r - dx), int(b - dy)
