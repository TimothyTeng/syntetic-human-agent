"""
Normal mode's spreadsheet: last month's spending on the project, built in OpenOffice Calc
before the report is written - pure, no UI. The report (office_writing.py) quotes its totals
and pastes its summary in as a table, so the two documents agree.

    budget = generate_budget(rng, n_rows=8)
    ops = plan_sheet(budget, rng)              # Op('goto', 'A1'), Op('rows', (...)), Op('formula', ...), ...
    sheet = FakeSheet().run(ops)               # the sheet the operations build, offline
    sheet.display("G3")                        # '186.40'

The sheet, the way a person keeps one:
  A1:D(n+1)   Date | Item | Category | Amount - one row per expense, typed row by row (Tab
              between cells, Enter and Home at the end of a row)
  C/D(n+2)    Total | =SUM(D2:D(n+1))
  F1:G(k+1)   Category | Spent - each category once, and
              =SUMPRODUCT(($C$2:$C$(n+1)=F2)*$D$2:$D$(n+1)) typed in G2, then copied (Ctrl+C)
              and pasted (Ctrl+V) over the other categories' cells, which shifts F2 along
  F/G(k+2)    Total | =SUM(G2:G(k+1))
Then the header rows and both total rows in bold and the amounts with two decimals
(Ctrl+Shift+1), a column chart of the summary (Insert > Chart...), often the columns set to
their optimal width, the top row frozen and the sheet renamed 'Budget'; saved as an .ods
(Save As). The summary range F1:G(k+2) is what the report pastes.

What it relies on in Calc (OpenOffice 4.1, checked against its menus and shortcuts):
  * Ctrl+Shift+T puts the cursor in the Name Box, which takes a cell or a range ('F1:G6');
    Enter there selects it and goes back to the sheet
  * AutoInput finishes a cell from the entries above it in the same column - so no item or
    category is the start of another one (typed in full it is never changed)
  * formulas use one argument each (SUM, SUMPRODUCT): the separator between arguments (',' or
    ';') depends on the locale
  * Ctrl+D is Selection List in Calc, not fill down - copying the formula and pasting it over
    the cells below does the same, relative references shifting
  * every amount and total stays under 1,000, so the two-decimal format (Ctrl+Shift+1) shows
    '861.30' with or without a thousands separator
  * copying a range puts its text on the clipboard: tabs between cells, a line break per row

Operations (Op.kind, args), the cell ops typing at the active cell:
  goto ref | rows rows | cell text key | column values | formula text | copy_to source target
  bold ref | number_format ref | chart ref | optimal_width ref | freeze_below cell
  rename_sheet name | save
OPTIONAL ops (optimal_width, freeze_below, rename_sheet) are left out when time runs short.
"""

import datetime
import re

import numpy as np

from research.writer_ops import Op

HEADER = ("Date", "Item", "Category", "Amount")
SUMMARY_HEADER = ("Category", "Spent")
TOTAL = "Total"
SHEET_NAME = "Budget"
OPTIONAL = frozenset({"optimal_width", "freeze_below", "rename_sheet"})
# (item, category, lowest, highest amount) - no item is the start of another, and no
# category the start of another or of 'Total' (Calc's AutoInput would finish it)
ITEMS = [
    ("Taxi to client office", "Travel", 12, 45), ("Train tickets", "Travel", 18, 60),
    ("Parking at the venue", "Travel", 6, 25), ("Hotel for the site visit", "Travel", 120, 280),
    ("Printer paper", "Supplies", 15, 40), ("Whiteboard markers", "Supplies", 8, 22),
    ("Coffee beans", "Supplies", 12, 30), ("Team lunch", "Meals", 60, 180),
    ("Client dinner", "Meals", 90, 260), ("Workshop catering", "Meals", 80, 220),
    ("Conference fee", "Training", 150, 450), ("Online course", "Training", 40, 160),
    ("Software licence", "Software", 30, 120), ("Survey tool subscription", "Software", 20, 80),
    ("Courier delivery", "Postage", 10, 35), ("Laptop stand", "Equipment", 25, 60),
    ("Headset for calls", "Equipment", 30, 90), ("Poster printing", "Printing", 20, 75),
    ("Printed handouts", "Printing", 15, 50),
]
MIN_ROWS, MAX_ROWS = 4, 12
MAX_TOTAL = 99_995                   # cents: the total stays under 1,000 (see the module docstring)
ROW_CHARS = 38                       # typed characters per expense row, about
FIXED_SECONDS = 140.0                # opening Calc, selecting, formulas, formatting, the chart and saving
                                     # (measured live: ~135 s, the Save As path typed out taking ~35 s)


def money(cents):
    """An amount as Calc shows it with two decimals: '861.30'."""
    return f"{cents // 100:,}.{cents % 100:02d}"


def plain(cents):
    """An amount as it is typed into a cell: '861.30'."""
    return f"{cents // 100}.{cents % 100:02d}"


class Budget:
    """Last month's expenses: rows of (date 'YYYY-MM-DD', item, category, cents) in the order
    they are typed."""

    def __init__(self, rows, month):
        self.rows, self.month = rows, month

    @property
    def categories(self):
        """The categories in the order they first appear (the summary's order)."""
        seen = []
        for r in self.rows:
            if r[2] not in seen:
                seen.append(r[2])
        return seen

    def spent(self, category):
        return sum(r[3] for r in self.rows if r[2] == category)

    @property
    def total(self):
        return sum(r[3] for r in self.rows)

    @property
    def top(self):
        """(category, cents, its largest item) of the category with the most spent."""
        cat = max(self.categories, key=self.spent)
        item = max((r for r in self.rows if r[2] == cat), key=lambda r: r[3])[1]
        return cat, self.spent(cat), item

    @property
    def summary_rows(self):
        """The summary as the sheet shows it (F1:G(k+2)), header and total row included - what the
        report pastes."""
        return ([list(SUMMARY_HEADER)] + [[c, money(self.spent(c))] for c in self.categories]
                + [[TOTAL, money(self.total)]])

    @property
    def summary_range(self):
        return f"F1:G{len(self.categories) + 2}"

    def to_dict(self):
        return {"rows": [list(r) for r in self.rows], "month": self.month}

    @classmethod
    def from_dict(cls, d):
        return cls([tuple(r) for r in d["rows"]], d["month"])


def rows_for_time(seconds, wpm):
    """How many expense rows a person typing at `wpm` enters in about `seconds`, leaving
    FIXED_SECONDS for the formulas, formatting, the chart and saving."""
    chars_per_s = wpm * 5 / 60 / 1.8
    return int(np.clip((seconds - FIXED_SECONDS) * 0.7 * chars_per_s / ROW_CHARS, MIN_ROWS, MAX_ROWS))


def generate_budget(rng, n_rows=8, today=None):
    """n_rows expenses from last month: at least three categories, no two amounts alike, the
    total under 1,000."""
    today = today or datetime.date.today()
    last = today.replace(day=1) - datetime.timedelta(days=1)          # last day of last month
    n_rows = int(np.clip(n_rows, MIN_ROWS, MAX_ROWS))
    while True:
        picks = [ITEMS[i] for i in rng.permutation(len(ITEMS))[:n_rows]]
        if len({p[1] for p in picks}) < 3:
            continue
        amounts, used = [], set()
        for _, _, lo, hi in picks:
            cents = int(round(rng.uniform(lo, hi) * 20)) * 5           # to 5 cents, like receipts
            while cents in used:
                cents += 5
            used.add(cents)
            amounts.append(cents)
        if sum(amounts) <= MAX_TOTAL:
            break
    days = sorted(int(d) for d in rng.permutation(last.day)[:n_rows] + 1)
    rows = [(last.replace(day=day).isoformat(), item, cat, cents)
            for day, (item, cat, _, _), cents in zip(days, picks, amounts)]
    return Budget(rows, last.strftime("%B"))


def plan_sheet(budget, rng=None):
    """The operations that build the budget's sheet (see the module docstring). rng: the order
    of the formatting steps and which optional steps this person bothers with."""
    rng = rng if rng is not None else np.random.default_rng(0)
    n, k = len(budget.rows), len(budget.categories)
    data_end, sum_end = n + 1, k + 1
    ops = [Op("goto", "A1"),
           Op("rows", tuple([HEADER] + [(d, item, cat, plain(c)) for d, item, cat, c in budget.rows])),
           Op("goto", f"C{n + 2}"), Op("cell", TOTAL, "tab"), Op("formula", f"=SUM(D2:D{data_end})"),
           Op("goto", "F1"), Op("cell", SUMMARY_HEADER[0], "tab"), Op("cell", SUMMARY_HEADER[1], "enter"),
           Op("goto", "F2"), Op("column", tuple(budget.categories)),
           Op("goto", "G2"), Op("formula", f"=SUMPRODUCT(($C$2:$C${data_end}=F2)*$D$2:$D${data_end})")]
    if k > 1:
        ops.append(Op("copy_to", "G2", f"G3:G{sum_end}"))
    ops += [Op("goto", f"F{k + 2}"), Op("cell", TOTAL, "tab"), Op("formula", f"=SUM(G2:G{sum_end})")]
    formatting = [Op("bold", "A1:D1"), Op("bold", "F1:G1"), Op("bold", f"C{n + 2}:D{n + 2}"),
                  Op("bold", f"F{k + 2}:G{k + 2}"), Op("number_format", f"D2:D{n + 2}"),
                  Op("number_format", f"G2:G{k + 2}")]
    ops += [formatting[i] for i in rng.permutation(len(formatting))]
    ops.append(Op("chart", f"F1:G{sum_end}"))
    if rng.random() < 0.7:
        ops.append(Op("optimal_width", "A1:G1"))
    if rng.random() < 0.5:
        ops.append(Op("freeze_below", "A2"))
    if rng.random() < 0.6:
        ops.append(Op("rename_sheet", SHEET_NAME))
    ops.append(Op("save"))
    return ops


# --- an offline sheet -----------------------------------------------------------------------------

_REF = re.compile(r"(\$?)([A-Z]+)(\$?)(\d+)")


def col_index(letters):
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


def col_letters(n):
    out = ""
    while n:
        n, r = divmod(n - 1, 26)
        out = chr(65 + r) + out
    return out


def parse_ref(ref):
    """'G2' -> (7, 2)."""
    m = re.fullmatch(r"([A-Z]+)(\d+)", ref.replace("$", ""))
    return col_index(m.group(1)), int(m.group(2))


def parse_range(ref):
    """'F1:G6' -> ((6, 1), (7, 6)); a single cell is a range of one."""
    a, _, b = ref.partition(":")
    return parse_ref(a), parse_ref(b or a)


def cells_of(ref):
    (c0, r0), (c1, r1) = parse_range(ref.replace("$", ""))
    return [(c, r) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def shifted(formula, rows):
    """A formula copied `rows` rows down: relative row references move, $-rows stay."""
    return _REF.sub(lambda m: m.group(1) + m.group(2) + m.group(3)
                    + str(int(m.group(4)) + (0 if m.group(3) else rows)), formula)


class FakeSheet:
    """A worksheet driven by the same operations as the real Calc: what is typed where, the
    formulas and their copies, and the formats - to check offline that the plan builds the sheet
    the report quotes. Assertions fail on what would go wrong in Calc (typing over a cell that
    has something in it, pasting over cells with data, which makes Calc ask)."""

    def __init__(self):
        self.cells = {}                  # (col, row) -> typed text or formula
        self.formats = {}                # (col, row) -> set of 'bold' / 'number'
        self.active = (1, 1)
        self.done = []                   # kinds of the operations without an effect on the cells

    def run(self, ops):
        for op in ops:
            getattr(self, "op_" + op.kind, self._other)(op)
        return self

    def _other(self, op):
        self.done.append(op.kind)

    def _put(self, text):
        assert self.active not in self.cells, f"typing over {col_letters(self.active[0])}{self.active[1]}"
        self.cells[self.active] = text

    def op_goto(self, op):
        self.active = parse_range(op.a)[0]

    def op_rows(self, op):
        """Tab between cells, Enter (down) and Home (column A) after each row."""
        for row in op.a:
            for j, value in enumerate(row):
                self._put(value)
                self.active = (self.active[0] + 1, self.active[1])
            self.active = (1, self.active[1] + 1)

    def op_cell(self, op):
        self._put(op.a)
        c, r = self.active
        self.active = (c + 1, r) if op.b == "tab" else (c, r + 1)

    def op_column(self, op):
        for value in op.a:
            self.op_cell(Op("cell", value, "enter"))

    def op_formula(self, op):
        self.op_cell(Op("cell", op.a, "enter"))

    def op_copy_to(self, op):
        (src,) = cells_of(op.a)
        for c, r in cells_of(op.b):
            assert (c, r) not in self.cells, "pasting over data"
            self.cells[(c, r)] = shifted(self.cells[src], r - src[1])

    def op_bold(self, op):
        for cell in cells_of(op.a):
            self.formats.setdefault(cell, set()).add("bold")

    def op_number_format(self, op):
        for cell in cells_of(op.a):
            self.formats.setdefault(cell, set()).add("number")

    def value(self, cell):
        """A cell's value: a float for numbers and formulas, the text otherwise (None if empty)."""
        text = self.cells.get(cell)
        if text is None or not text.startswith("="):
            try:
                return float(text)
            except (TypeError, ValueError):
                return text
        m = re.fullmatch(r"=SUM\((\w+:\w+)\)", text)
        if m:
            return sum(v for v in (self.value(x) for x in cells_of(m.group(1))) if isinstance(v, float))
        m = re.fullmatch(r"=SUMPRODUCT\(\(([$\w]+:[$\w]+)=(\w+)\)\*([$\w]+:[$\w]+)\)", text)
        if m:
            crit = self.value(parse_ref(m.group(2)))
            pairs = zip(cells_of(m.group(1)), cells_of(m.group(3)))
            return sum(self.value(s) for c, s in pairs if self.value(c) == crit)
        raise ValueError(f"formula FakeSheet can't evaluate: {text}")

    def display(self, ref):
        """A cell's text as Calc shows it (numbers with two decimals once formatted)."""
        cell = parse_ref(ref)
        v = self.value(cell)
        if isinstance(v, float):
            cents = int(round(v * 100))
            return money(cents) if "number" in self.formats.get(cell, ()) else f"{v:g}"
        return v or ""

    def rows(self, ref):
        """The displayed text of a range, row by row."""
        (c0, r0), (c1, r1) = parse_range(ref)
        return [[self.display(f"{col_letters(c)}{r}") for c in range(c0, c1 + 1)] for r in range(r0, r1 + 1)]

    def bold(self, ref):
        return "bold" in self.formats.get(parse_ref(ref), ())
