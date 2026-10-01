"""
Carrying out writer_ops operations in a real OpenOffice Writer window, as the simulated person.

    ex = WriterExecutor(human, win, doc, save_path=r"...\\paper.odt", refocus=..., log=log)
    next_index = ex.run(ops, start=0, deadline=time.monotonic() + 600)

Every operation is done the way this person would do it: keyboard-minded people use
shortcuts (Ctrl+1, Ctrl+B, Ctrl+E, Shift+F12, Ctrl+F12 for a table, Ctrl+F11 to reach the
Apply Style box), mouse-minded people click the Formatting toolbar (Apply Style box,
Bold, Centred, Bullets On/Off) and the menus (Insert > Table..., Insert > Manual Break...),
chosen per action from the persona's shortcut preference. Default Formatting is always
Ctrl+M (see op_reset_chars).
When the first method doesn't work (a hidden toolbar, a dialog that doesn't open), the
next one is tried, and the result is checked through UI Automation where Writer reports
it (paragraph style and font size at the caret, caret in a table, document focus). Writer
doesn't report bold / italic / alignment, so those follow the executor's own model.

Text is typed with the learned typing model (typos, corrections, thinking pauses), one
run at a time, so its corrections never cross a formatting boundary. When Writer's word
completion offers the rest of a long word after a run, it is dismissed (Delete) before
Enter could accept it.

dry_run=True performs nothing and records the actions it would take (self.actions).
"""

import os
import time
from collections import Counter

from algorithms.session import TaskError
from controller import apps
from controller import writer as W

from .fake_writer import FakeWriter
from .writer_ops import BODY, NEXT_STYLE, START_STYLE, Op

# Built-in shortcuts for the styles that have one (Title, Subtitle, Caption don't)
STYLE_KEYS = {"Heading 1": ("ctrl", "1"), "Heading 2": ("ctrl", "2"), "Heading 3": ("ctrl", "3"),
              BODY: ("ctrl", "0")}
TOGGLES = {"bold": ("Bold", "b"), "italic": ("Italic", "i")}       # toolbar button name, Ctrl+ key
# toolbar button name (a regex: en-GB 'Centred', en-US 'Centered'), Ctrl+ key
ALIGN = {"center": ("Cent(red|ered)", "e"), "left": ("Align Left", "l")}
BULLETS_BUTTON = "Bullets On/Off"
P_SAVE_AT_SECTION = 0.6                                  # chance of saving when a section is done (after Save As)


class WriterExecutor:
    """Performs writer_ops operations in Writer as one simulated person (human: a Human with
    a persona). Keeps its own idea of the state at the caret (style, bold, italic, table),
    checks what Writer reports through UIA (style, font size, table, focus) and collects
    how each operation was done (methods) and what went wrong (problems) for the log."""

    def __init__(self, human, win, doc, save_path=None, refocus=None, log=print, dry_run=False,
                 revisions="heuristic", saved=False):
        """
        win, doc:   the Writer window and its DocumentControl (None in a dry run)
        revisions:  how the typing model revises composed text (passed to human.type)
        save_path:  where the first save puts the file (Save As, an .odt); later saves are Ctrl+S
        refocus:    callback that puts the keyboard focus back into the document (e.g.
                    tasks.click_into_document); a TaskError is raised if focus can't be had
        saved:      the document already has its file name (resuming)
        """
        self.human, self.win, self.doc = human, win, doc
        self.save_path, self.refocus, self.log = save_path, refocus, log
        self.dry_run, self.revisions, self.saved = dry_run, revisions, saved
        self.bold = self.italic = False
        self.style = START_STYLE               # paragraph style at the caret, as far as we know
        self.in_table = False
        self.style_checked = False             # the caret's paragraph style was compared with Writer's
        self.skipping_table = False
        self.actions = []                      # dry run: what would have been done
        self.methods = Counter()               # how each kind of operation was done (for the log)
        self.problems = []

    # --- running ------------------------------------------------------------------------------

    def run(self, ops, start=0, deadline=None, on_checkpoint=None, stop_before_save=False, estimate=None):
        """
        Execute ops[start:] until the end, or until a checkpoint after `deadline`
        (time.monotonic()). on_checkpoint(next_index, block_index) is called at every
        checkpoint. stop_before_save: also stop at the first section boundary (a 'save'
        right after a checkpoint) once at least one block was written. estimate(i): seconds
        the block starting at ops[i] will take - a block that wouldn't finish before the
        deadline isn't started. Returns the index of the next operation to run.
        """
        i = start
        if start:
            self.sync(ops[:start])
        if deadline is not None and estimate and time.monotonic() + estimate(start) >= deadline:
            return start                           # not even the first block fits
        while i < len(ops):
            op = ops[i]
            if op.kind == "checkpoint":            # a block is finished: a safe place to stop
                if on_checkpoint:
                    on_checkpoint(i + 1, op.a)
                if deadline is not None and time.monotonic() + (estimate(i + 1) if estimate else 0) >= deadline:
                    return i + 1                   # time is up, or the next block wouldn't fit
                if stop_before_save and i + 1 < len(ops) and ops[i + 1].kind == "save":
                    self.do(ops[i + 1])            # end of a section: save it, then stop
                    return i + 2
            else:
                self.do(op)
            i += 1
        return i

    def sync(self, done):
        """Take the state at the caret from the operations already performed (`done`):
        a new executor carries on where an earlier one stopped, and Writer can't be asked
        whether bold or italic is on."""
        fw = FakeWriter().run(done)
        self.bold, self.italic, self.in_table = fw.bold, fw.italic, fw.table is not None
        self.style = fw.cur["style"]

    def do(self, op):
        """Perform one operation (dispatched to op_<kind>). While a table that couldn't be
        inserted is being skipped, its cell operations are dropped up to its exit_table."""
        if self.skipping_table:                 # the table couldn't be inserted: leave it out
            if op.kind == "exit_table":
                self.skipping_table = False
            return
        getattr(self, "op_" + op.kind)(op)

    # --- primitives (recorded instead of performed in a dry run) --------------------------------

    def _hotkey(self, *keys):
        """Press a key combination (e.g. 'ctrl', 'b') the way this person does."""
        if self.dry_run:
            self.actions.append("HOTKEY " + "+".join(keys))
        else:
            self.human.hotkey(*keys)

    def _press(self, key, presses=1):
        """Press a single key, `presses` times."""
        if self.dry_run:
            self.actions.append(f"PRESS {key}" + (f" x{presses}" if presses > 1 else ""))
        else:
            self.human.press(key, presses)

    def _type(self, text, mode, exact=False):
        """Type text with the typing model. mode 'compose' (thinking pauses, revisions) for
        prose being written, 'transcribe' for copying short text; exact=True types without
        typos (numbers in a dialog, a style name - a mistyped name in the Apply Style box
        would create a new style). word_safe keeps every typo and its correction inside its
        word: AutoCorrect rewrites a misspelled word as soon as a space follows it, which
        would break the correction. Returns human.type's result."""
        if self.dry_run:
            self.actions.append(f"TYPE[{mode}] {text!r}")
            return None
        kw = {"error_scale": 0.0} if exact else {}
        kw["word_safe"] = True       # a misspelled word never gets a space after it (AutoCorrect)
        if mode == "compose":
            return self.human.type(text, mode="compose", revisions=self.revisions, max_pause=4.0, **kw)
        return self.human.type(text, mode="transcribe", **kw)

    def _click(self, ctrl, label):
        """Click a UIA control with the mouse; label names it in the dry-run record."""
        if self.dry_run:
            self.actions.append(f"CLICK {label}")
        else:
            self.human.click_element(ctrl)

    def _think(self, kind="glance", scale=1.0):
        """A pause of the given kind ('glance', 'scan', 'mental', 'confirm') - skipped in a dry run."""
        if not self.dry_run:
            self.human.think(kind, scale)

    def _keyboard(self, bias=0.0):
        """Whether this person uses the keyboard (rather than the mouse) for this action;
        bias > 0 makes the keyboard more likely, for actions with a well-known shortcut."""
        return self.human.prefers_keyboard(bias)

    # --- focus ---------------------------------------------------------------------------------

    def ensure_focus(self):
        """Make sure typing goes into the document. Writer marks the caret's paragraph a
        moment after it moves, so an unclear answer ('other' while Writer is in front) is
        asked again first. A toolbar box, a menu or another window has the focus: Esc, then
        click back into the page - except inside a table, where the click and Ctrl+End
        would leave the table: there the run stops instead."""
        if self.dry_run:
            return
        kind = W.focus_kind(self.win, self.doc)
        for _ in range(3):
            if kind != "other":
                break
            time.sleep(0.3)
            self.doc = W.document_control(self.win, timeout=1) or self.doc     # replaced after a save
            kind = W.focus_kind(self.win, self.doc)
        if kind in ("document", "toolbar"):      # 'toolbar': a clicked button, keys still reach the page
            return
        self._press("esc")                     # closes a menu / leaves a box: often enough
        time.sleep(0.3)
        if W.document_has_focus(self.win, self.doc):
            return
        if self.in_table:
            raise TaskError(f"The Writer document lost keyboard focus ({kind}) inside a table - stopping")
        self.log(f"Writer document lost focus ({kind}) - clicking back into it")
        if self.refocus:
            self.refocus()
        if not W.document_has_focus(self.win, self.doc):  # never type blind into whatever has focus
            raise TaskError("The Writer document doesn't have keyboard focus - stopping before typing anything")

    def _toolbar_click(self, button_name, label=None, toolbar="Formatting"):
        """Click a toolbar button (name may be a regex). False if it can't be found."""
        if self.dry_run:
            self.actions.append(f"CLICK {toolbar} > {label or button_name}")
            return True
        btn = W.toolbar_button(self.win, button_name, toolbar)
        if btn is None:
            return False
        self.human.click_element(btn)
        time.sleep(0.2)
        return True

    def _menu_click(self, menu_name, item_name, label):
        """Open a menu ('Insert') and click one of its items (regex, e.g. 'Table\\.\\.\\.').
        False if either can't be found (an opened menu is closed again)."""
        if self.dry_run:
            self.actions.append(f"CLICK {menu_name} > {label}")
            return True
        menu = W.menu(self.win, menu_name)
        if menu is None:
            return False
        self.human.click_element(menu)
        time.sleep(0.4)                           # the menu opening
        item = W.menu_item(menu, item_name)
        if item is None:
            self._press("esc")                    # close the menu again
            return False
        self._think("glance", 0.6)                # finding the item in the list
        self.human.click_element(item)
        return True

    # --- operations ------------------------------------------------------------------------------

    def op_type(self, op):
        """Type a run of text (op.a) in mode op.b. Before the first run of a paragraph, its
        style is compared with what Writer reports (and re-applied if Writer chose another);
        after the run, a word-completion suggestion is dismissed."""
        self.ensure_focus()
        if not self.dry_run and not self.style_checked and not self.in_table:
            self._check_style()
        self._type(op.a, op.b)
        if not self.dry_run:
            self._dismiss_completion(op.a)

    def _check_style(self):
        """The paragraph about to be typed in should have self.style; Writer's "next style"
        after Enter may differ from what the renderer assumed - then apply the right one."""
        self.style_checked = True
        got = W.caret_paragraph_style(self.win)
        if got is None or got == self.style:
            return
        self.log(f"The paragraph has style {got!r}, meant {self.style!r} - applying it")
        self.methods["style:resync"] += 1
        self.op_style(Op("style", self.style))

    def _dismiss_completion(self, typed):
        """If word completion is offering the rest of a word (selected text after the caret),
        press Delete: Enter would accept it and Tab / typing on could keep it."""
        n = W.pending_completion(self.doc, typed)
        if n:
            self.log(f"Word completion offers {n} more letters after {typed[-20:]!r} - dismissing it")
            self._think("glance", 0.5)
            self._press("delete")
            self.methods["completion:dismissed"] += 1

    def op_enter(self, op):
        """New paragraph; Writer gives it the style that follows the current one (NEXT_STYLE)."""
        self._press("enter")
        self.style = NEXT_STYLE.get(self.style, self.style)
        self.style_checked = False

    def op_reset_chars(self, op):
        """Default Formatting (Ctrl+M): back to the paragraph style's own character formatting
        (and alignment). The renderer only emits it in an empty paragraph. Always the
        shortcut: Format > Default Formatting from the menu was seen to leave a carried-over
        "not bold" in place (it then un-bolded the heading typed next)."""
        self._hotkey("ctrl", "m")
        self.methods["reset:key"] += 1
        self.bold = self.italic = False

    def op_bold(self, op):
        """Switch bold on / off (op.a) - see _toggle."""
        self._toggle("bold", op.a)

    def op_italic(self, op):
        """Switch italic on / off (op.a) - see _toggle."""
        self._toggle("italic", op.a)

    def _toggle(self, what, on):
        """Bring bold / italic into state `on`. Ctrl+B / Ctrl+I and the toolbar buttons
        toggle, so pressing one in the wrong state does the opposite of what's meant. Writer
        doesn't report the buttons' state, so the executor's own idea of it decides."""
        if getattr(self, what) == on:
            return                                   # already right: pressing would switch it off
        button, key = TOGGLES[what]
        self._think("glance")
        if self._keyboard(0.3) or not self._toolbar_click(button):
            self._hotkey("ctrl", key)
            self.methods[f"{what}:key"] += 1
        else:
            self.methods[f"{what}:toolbar"] += 1
        setattr(self, what, on)

    def op_align(self, op):
        """Align the paragraph ('left' / 'center') with Ctrl+L / Ctrl+E or the toolbar."""
        button, key = ALIGN[op.a]
        if self._keyboard(0.2) or not self._toolbar_click(button, label=op.a):
            self._hotkey("ctrl", key)
            self.methods["align:key"] += 1
        else:
            self.methods["align:toolbar"] += 1

    def op_bullets(self, op):
        """Switch bullets on / off (Shift+F12 or Bullets On/Off - both toggle; the renderer
        only emits this when it changes the state)."""
        if self._keyboard(0.2) or not self._toolbar_click(BULLETS_BUTTON):
            self._hotkey("shift", "f12")
            self.methods["bullets:key"] += 1
        else:
            self.methods["bullets:toolbar"] += 1

    def op_page_break(self, op):
        """Page break (Ctrl+Enter, or Insert > Manual Break... > Page break > OK); the caret
        lands in a fresh paragraph on the new page."""
        if self._keyboard(0.2) or not self._page_break_by_menu():
            self._hotkey("ctrl", "enter")
            self.methods["page_break:key"] += 1
        else:
            self.methods["page_break:menu"] += 1
        self.style_checked = False

    def _page_break_by_menu(self):
        """Insert > Manual Break..., make sure 'Page break' is chosen, OK. False if the
        menu item or the dialog isn't found."""
        if not self._menu_click("Insert", r"Manual Break\.\.\.", "Manual Break..."):
            return False
        if self.dry_run:
            self.actions.append("CLICK Page break, OK")
            return True
        dialog = W.find_dialog(r"Insert Break|Manual Break", timeout=3)
        if dialog is None:
            self._press("esc")
            return False
        self._think("scan", 0.6)
        page = W.dialog_control(dialog, r"Page break|Page Break", "RadioButtonControl")
        if page is not None and not W.is_checked(page):
            self.human.click_element(page)
        self._think("confirm")
        self._press("enter")                      # OK is the default button
        time.sleep(0.5)
        return True

    def op_next_cell(self, op):
        """Tab to the next table cell (never emitted after the last cell: Tab there adds a row)."""
        self._press("tab")
        self.bold = self.italic = False          # a new cell starts unformatted

    def op_exit_table(self, op):
        """Leave the table: Ctrl+End goes to the end of the document, the empty paragraph
        after the table (everything is written at the end). Checked through UIA - from
        inside a cell Writer may first stop at the end of the table, so up to two more."""
        self._hotkey("ctrl", "end")
        self.bold = self.italic = False
        self.style, self.in_table, self.style_checked = BODY, False, False
        for _ in range(2):
            if self.dry_run or not W.caret_in_table(self.doc):
                return
            self.log("Caret still in the table after Ctrl+End - pressing Ctrl+End again")
            time.sleep(0.2)
            self._hotkey("ctrl", "end")
        if W.caret_in_table(self.doc):
            self.problems.append("caret still in the table after Ctrl+End")

    def op_checkpoint(self, op):
        """Nothing to do in Writer (run() handles checkpoints)."""
        pass

    # --- styles ----------------------------------------------------------------------------------

    def op_style(self, op):
        """Apply a paragraph style. Keyboard-minded: the style's shortcut (Ctrl+1, Ctrl+0),
        then the Apply Style box reached with Ctrl+F11; mouse-minded: clicking into the Apply
        Style box first. Each way is checked against the style Writer reports at the caret;
        if it didn't take, the next is tried. (Applying a style also sets its own alignment -
        the renderer knows.)"""
        style = op.a
        self.style = style
        self.style_checked = True
        keys = STYLE_KEYS.get(style)                 # None: the style has no shortcut
        if self._keyboard(0.2):
            methods = ([self._style_by_keys] if keys else []) + [self._style_by_box_keys, self._style_by_box_click]
        else:
            methods = [self._style_by_box_click] + ([self._style_by_keys] if keys else []) + [self._style_by_box_keys]
        for method in methods:
            if not method(style):
                continue                         # e.g. the toolbar is hidden: just use another way
            if self._style_is(style):
                self.methods[f"style:{method.__name__[10:]}"] += 1     # '_style_by_keys' -> 'keys'
                return
            self.log(f"Style {style!r} via {method.__name__[10:]} didn't take - trying another way")
        self.problems.append(f"couldn't apply style {style!r}")
        self.log(f"Couldn't apply the {style!r} style - carrying on without it")

    def _style_is(self, style):
        """True if the caret's paragraph now has `style` (or Writer doesn't say)."""
        if self.dry_run:
            return True
        time.sleep(0.15)
        got = W.caret_paragraph_style(self.win)
        return got is None or got.lower() == style.lower()     # not reported: assume it worked

    def _style_by_keys(self, style):
        """The style's built-in shortcut (Ctrl+1, Ctrl+0, ...). Always 'done'."""
        self._hotkey(*STYLE_KEYS[style])
        return True

    def _style_by_box_keys(self, style):
        """Ctrl+F11 puts the focus in the Apply Style box: type the style's name, Enter."""
        self._hotkey("ctrl", "f11")
        return self._type_into_style_box(style)

    def _style_by_box_click(self, style):
        """Click into the Apply Style box on the Formatting toolbar: type the name, Enter.
        False if the box isn't shown."""
        if self.dry_run:
            self.actions.append("CLICK Formatting > Apply Style")
        else:
            box = W.style_box(self.win)
            if box is None:
                return False
            self.human.click_element(box)
        return self._type_into_style_box(style)

    def _type_into_style_box(self, style):
        """With the Apply Style box (about to be) focused: replace its text with the style's
        name and Enter. False if no field got the focus."""
        if not self.dry_run:
            time.sleep(0.2)
            if not self._in_field():             # Ctrl+A in the page would select the whole document
                return False
            self._think("glance")
        self._hotkey("ctrl", "a")                # replace the style name already in the box
        self._type(style, "transcribe", exact=True)
        self._think("confirm")
        self._press("enter")
        return self._back_in_document()

    # --- font size --------------------------------------------------------------------------------

    def op_size(self, op):
        """Set the font size (op.a points) in the Font Size box: clicked, or reached from the
        keyboard (Ctrl+F11 to the Apply Style box, Tab, Tab). Checked against the size
        Writer reports at the caret."""
        pt = f"{op.a:g}"                           # 12.0 -> '12', 10.5 -> '10.5'
        if self._keyboard(0.1):
            done = self._size_by_keys(pt) or self._size_by_box(pt)
        else:
            done = self._size_by_box(pt) or self._size_by_keys(pt)
        if done and not self.dry_run:
            got = W.caret_font_size(self.win)
            if got is not None and abs(got - op.a) > 0.05:
                done = False
        if not done:
            self.problems.append(f"couldn't set font size {pt}")
            self.log(f"Couldn't set the font size to {pt} pt")

    def _size_by_keys(self, pt):
        """Ctrl+F11 (Apply Style box), Tab (Font Name), Tab (Font Size): type the size, Enter.
        Tabbing out of the Apply Style box re-applies the paragraph style (clearing direct
        alignment) - the renderer sets the size before the alignment for that reason."""
        self._hotkey("ctrl", "f11")
        if not self.dry_run:
            time.sleep(0.3)
            if not self._in_field():
                return False
        for _ in range(2):                      # to Font Name, then Font Size - a moment for each
            self._press("tab")
            if not self.dry_run:
                time.sleep(0.3)
        if not self.dry_run:
            box = W.font_size_box(self.win)
            if box is None or not W.is_focused(box):     # the Tabs went somewhere else
                self.log("The Font Size box didn't get the focus")
                self._escape_to_document()
                return False
        self._think("glance")
        self._hotkey("ctrl", "a")
        self._type(pt, "transcribe", exact=True)
        self._press("enter")
        self.methods["size:key"] += 1
        return self._back_in_document()

    def _size_by_box(self, pt):
        """Click the Font Size box, select what's in it, type the size, Enter. False if the
        box can't be found or the click didn't put the focus in it."""
        if self.dry_run:
            self.actions.append("CLICK Formatting > Font Size")
        else:
            box = W.font_size_box(self.win)
            if box is None:
                return False
            self.human.click_element(box)
            time.sleep(0.2)
            # a missed click leaves the focus in the page, where Ctrl+A + typing would
            # select and replace the whole document
            if not self._in_field():
                return False
        self._hotkey("ctrl", "a")                   # select the old size in the box
        self._type(pt, "transcribe", exact=True)
        self._press("enter")
        self.methods["size:toolbar"] += 1
        return self._back_in_document()

    def _in_field(self, timeout=1.0):
        """True once a toolbar box or a Writer dialog (not the page) has the keyboard focus.
        Ctrl+A and typing are only safe then - in the page, Ctrl+A would select the whole
        document. If no field gets the focus, Esc and False."""
        deadline = time.monotonic() + timeout
        while True:
            kind = W.focus_kind(self.win, self.doc)
            if kind == "field" or (kind == "elsewhere" and W.foreground_in_writer(self.win)):
                return True
            if time.monotonic() >= deadline:
                self.log("No text field got the focus - not typing into it")
                self._escape_to_document()
                return False
            time.sleep(0.1)

    def _back_in_document(self):
        """After Enter in a toolbar box: True once the focus is back in the page (Esc as a
        last resort)."""
        if self.dry_run:
            return True
        time.sleep(0.4)
        return W.document_has_focus(self.win, self.doc) or self._escape_to_document()

    def _escape_to_document(self):
        """Esc out of a box / menu / dialog; True if the page then has the focus."""
        self._press("esc")
        time.sleep(0.3)
        return W.document_has_focus(self.win, self.doc)

    # --- tables -----------------------------------------------------------------------------------

    def op_table(self, op):
        """Insert an op.a x op.b table through the Insert Table dialog, opened with Ctrl+F12 or
        Insert > Table..., whichever this person reaches for first. If neither works, the
        table's operations are skipped (skipping_table)."""
        rows, cols = op.a, op.b
        self.ensure_focus()
        before = 0 if self.dry_run else W.table_count(self.doc) or 0     # to tell a new table appeared
        methods = [self._table_by_keys, self._table_by_menu]
        if not self._keyboard(0.1):
            methods.reverse()
        for method in methods:
            dialog = method()
            if dialog is None:                       # the dialog didn't open: try the other way
                continue
            self._fill_table_dialog(dialog, rows, cols)
            if self.dry_run or self._table_inserted(before):
                self.methods[f"table:{method.__name__[10:]}"] += 1     # '_table_by_menu' -> 'menu'
                self.bold = self.italic = False      # the first cell starts unformatted
                self.in_table = True
                return
            self.log("The table doesn't seem to have been inserted")
        self.problems.append("couldn't insert a table")
        self.log("Couldn't insert a table - leaving it out")
        self.skipping_table = True                   # do() drops the cells up to exit_table

    def _table_by_keys(self):
        """Ctrl+F12 - the Insert Table dialog."""
        self._hotkey("ctrl", "f12")
        return self._wait_table_dialog()

    def _table_by_menu(self):
        """Insert > Table... - the Insert Table dialog."""
        if not self._menu_click("Insert", r"Table\.\.\.", "Table..."):
            return None
        return self._wait_table_dialog()

    def _wait_table_dialog(self):
        """The Insert Table dialog once it's open, or None (menus Esc'd away)."""
        if self.dry_run:
            return True
        dialog = W.find_dialog("Insert Table", timeout=3)
        if dialog is None:
            self._press("esc")                    # leave a menu that may still be open
        return dialog

    def _fill_table_dialog(self, dialog, rows, cols):
        """The dialog opens with the table's Name selected: Tab, columns, Tab, rows (Tab selects
        each field's number, so typing replaces it). The Heading option is unticked if it is
        ticked - its heading style is bold by itself, so the bold typed into the first row
        would switch it off. Then Enter (OK)."""
        self._think("scan", 0.8)
        self._press("tab")
        self._type(str(cols), "transcribe", exact=True)
        self._press("tab")
        self._type(str(rows), "transcribe", exact=True)
        if not self.dry_run:
            heading = W.dialog_control(dialog, "^Heading$", "CheckBoxControl")
            if heading is not None and W.is_checked(heading):
                self.log("Unticking the table's Heading option")
                self.human.click_element(heading)
        self._think("confirm")
        self._press("enter")
        if not self.dry_run:
            time.sleep(0.6)

    def _table_inserted(self, before):
        """True if the caret is now in a table, or (when UIA can't tell) there are more
        tables than `before`."""
        in_table = W.caret_in_table(self.doc)
        return bool(in_table) or (in_table is None and (W.table_count(self.doc) or 0) > before)

    # --- saving -----------------------------------------------------------------------------------

    def op_save(self, op):
        """Save the document. The first save is Save As (Ctrl+Shift+S) to save_path; after that
        Ctrl+S or the Standard toolbar's Save button, at some section ends only
        (P_SAVE_AT_SECTION) and always at the end ('final')."""
        if not self.save_path:
            return
        if self.saved and op.a != "final" and self.human.rng.random() > P_SAVE_AT_SECTION:
            return                               # not every section ends with a save
        if not self.saved:
            self.log(f"Saving as {self.save_path}")
            if self.dry_run:
                self.actions.append(f"SAVE AS {self.save_path}")
            else:
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                self.human.save_as(self.save_path, shortcut=("ctrl", "shift", "s"))
                self._wait_saved()
            self.saved = True
            self.methods["save:save_as"] += 1
        elif self._keyboard(0.4) or not self._toolbar_click("Save", toolbar="Standard"):
            self._hotkey("ctrl", "s")
            self.methods["save:key"] += 1
        else:
            self.methods["save:button"] += 1
        if not self.dry_run:
            time.sleep(0.8)
            self.doc = W.document_control(self.win) or self.doc     # replaced when the file is named
            self.style_checked = False

    def _wait_saved(self, timeout=15):
        """Wait until Writer's title bar shows the new file name (the Save As went through).
        Meanwhile a 'Keep current format' question (asked when a non-ODF file type was chosen
        in the dialog) is answered. True / False; a timeout is only logged. (The title is
        read through Win32: Writer's window has no UIA name.)"""
        stem = os.path.splitext(os.path.basename(self.save_path))[0].lower()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if stem in apps.window_title(self.win.NativeWindowHandle).lower():
                return True
            dialog = W.find_dialog(r".+", timeout=0)
            keep = dialog and W.dialog_control(dialog, r"Keep [Cc]urrent [Ff]ormat|Use .* Format!?", "ButtonControl")
            if keep:
                self.log("Writer asks which format to keep - keeping the current one")
                self._think("scan", 0.7)
                self.human.click_element(keep)
            time.sleep(0.3)
        self.log("Writer's title doesn't show the new file name yet")
        return False

    def summary(self):
        """One line for the log: how often each way was used ('style:keys x4, ...'), and the
        number of problems if there were any."""
        used = ", ".join(f"{k} x{v}" for k, v in sorted(self.methods.items()))
        return used + (f"; problems: {len(self.problems)}" if self.problems else "")
