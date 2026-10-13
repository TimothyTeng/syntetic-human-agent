"""
Normal mode's writing: an everyday office document, generated fresh each run, typed into
OpenOffice Writer through the same pipeline as the research paper (research/ir.py ->
writer_ops -> writer_exec).

    doc = generate_office_doc(rng, n_sentences=25)          # an ir.Document
    proj = OfficeProject(human, n_sentences=25, log=log)    # a WritingProject for tasks.Workspace.write

A document is one of four kinds - status report, meeting notes, project proposal, memo -
about one made-up piece of work (a project, the teams on it, the numbers it is measured
by), so its sentences hang together. It has a title (often resized once typed), sometimes a
date line, two to four headed sections, paragraphs of two to five sentences, and now and
then a bulleted list or a one-line bold statement.

Sentences come from a small grammar rather than fixed text: about a dozen structures
(simple, compound with "and / but / so", fronted "Although / Because / Once / If" clauses,
passives, relative clauses, numbers that moved, questions, requests, opinions, lists of
three, expectations), each filled from pools of subjects, verbs, objects, times and states.
Sections lean on the structures that suit them (questions under "Open Questions", requests
under "Next Steps", numbers under "Key Metrics"), no structure or first word repeats back to
back and no sentence is used twice. Some key phrases are set in bold or italic.

Exactly n_sentences sentences are written (bullet items and the bold line count as one each;
the title, headings and the date line don't). Everything passes ir.validate(): ASCII only and
nothing Writer's AutoCorrect would rewrite while typing.
"""

import datetime
import os
import re

import numpy as np

from research import ir
from research.ir import Bullets, Document, Heading, Paragraph, Run, Subtitle, Title
from research.project import WritingProject
from research.textutil import slugify

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "normal")
KINDS = ("status report", "meeting notes", "project proposal", "memo")
TITLE_MAX = 30                       # characters - a title stays on one line, even made bigger
TITLE_SIZES = (22.0, 24.0, 26.0, 32.0)

# --- the piece of work a document is about ------------------------------------------------------

# (phrase used in sentences, title form)
PROJECTS = [
    ("customer onboarding portal", "Customer Onboarding"), ("warehouse scheduling system", "Warehouse Scheduling"),
    ("expense approval process", "Expense Approvals"), ("staff training programme", "Staff Training"),
    ("website redesign", "Website Redesign"), ("supplier contract renewal", "Supplier Contracts"),
    ("office move", "Office Move"), ("support ticket backlog", "Support Backlog"),
    ("quarterly budget review", "Budget Review"), ("new payroll system", "Payroll System"),
    ("customer feedback survey", "Customer Survey"), ("inventory tracking tool", "Inventory Tracking"),
    ("mobile app release", "Mobile App Release"), ("data migration", "Data Migration"),
    ("recruitment drive", "Recruitment Drive"), ("internal newsletter", "Staff Newsletter"),
]
# (subject, plural?)
TEAMS = [
    ("the operations team", False), ("the finance team", False), ("our support staff", True),
    ("the design group", False), ("the sales leads", True), ("the IT department", False),
    ("the project board", False), ("two of our analysts", True), ("the vendor", False),
    ("the regional managers", True), ("the HR team", False), ("our contractors", True),
]
# (name, unit, low, high, lower is better?)
METRICS = [
    ("average response time", "hours", 4, 30, True), ("onboarding time", "days", 3, 14, True),
    ("the number of open tickets", "", 40, 400, True), ("customer satisfaction", "percent", 60, 95, False),
    ("the error rate", "percent", 2, 15, True), ("weekly sign ups", "", 30, 300, False),
    ("processing time per request", "minutes", 8, 60, True), ("first contact resolution", "percent", 40, 85, False),
    ("delivery delays", "per week", 3, 25, True), ("training completion", "percent", 35, 98, False),
]
# (object, plural?)
OBJECTS = [
    ("the draft timeline", False), ("the remaining manual steps", True), ("the test plan", False),
    ("the budget estimate", False), ("the training material", False), ("the reporting dashboard", False),
    ("the approval workflow", False), ("the customer FAQ", False), ("the handover checklist", False),
    ("the risk register", False), ("the supplier quotes", True), ("the first set of requirements", False),
    ("the user guide", False), ("the weekly summary", False), ("the access permissions", True),
    ("the rollout plan", False), ("the cost breakdown", False), ("the survey questions", True),
]
DOC_NAMES = ["Travel Policy", "Supplier Handbook", "Onboarding Guide", "Style Guide", "Security Checklist"]
# (past, base, past participle)
VERBS = [
    ("reviewed", "review", "reviewed"), ("finalised", "finalise", "finalised"), ("tested", "test", "tested"),
    ("updated", "update", "updated"), ("simplified", "simplify", "simplified"),
    ("documented", "document", "documented"), ("rolled out", "roll out", "rolled out"),
    ("checked", "check", "checked"), ("redesigned", "redesign", "redesigned"),
    ("prioritised", "prioritise", "prioritised"), ("shared", "share", "shared"),
    ("cleaned up", "clean up", "cleaned up"), ("drafted", "draft", "drafted"),
    ("signed off", "sign off", "signed off"), ("revisited", "revisit", "revisited"),
]
TIMES_PAST = ["last week", "on Tuesday", "earlier this month", "over the past quarter", "in the last sprint",
              "before the holidays", "yesterday afternoon", "at the start of the month", "two weeks ago"]
TIMES_FUTURE = ["by Friday", "next week", "before the end of the month", "in the next sprint",
                "by the end of the quarter", "over the coming weeks", "early next month", "before the review"]
METRIC_TIMES = ["over the past quarter", "since the last report", "this month", "since the summer",
                "over the last six weeks", "compared with last year"]
STATES = ["almost ready", "behind schedule", "on track", "still under review", "more complex than expected",
          "close to finished", "waiting for approval", "in better shape than last month"]
GOALS = ["ready", "signed off", "live", "finished", "in place"]
PROBLEMS = ["a delay with the supplier", "a few data quality issues", "some pushback from users",
            "an unexpected licence cost", "a gap in the test coverage", "a shortage of reviewers",
            "conflicting requirements", "a slow sign off process"]
BENEFITS = ["save several hours a week", "cut down on rework", "make the numbers easier to trust",
            "free up time for customer calls", "reduce the number of complaints", "speed up approvals"]
OPENERS = ["In my view", "On balance", "Overall", "For now", "In short", "To be fair", "Looking ahead",
           "Realistically"]
REL_PAST = ["which led the pilot last year", "who joined the project in the spring", "which raised the issue first",
            "who handled the last migration", "which owns most of the budget"]
MODALS_SG = ["will", "should", "could", "needs to", "plans to", "is going to"]
MODALS_PL = ["will", "should", "could", "need to", "plan to", "are going to"]

# --- document frames ----------------------------------------------------------------------------

TITLES = {
    "status report": ["{P} Status Report", "{M} Update on {P}", "{P} Progress Report", "Q{q} Report on {P}",
                      "{P} Update"],
    "meeting notes": ["Notes from the {P} Meeting", "{P} Meeting Notes", "{D} Meeting on {P}", "{P} Catch Up Notes"],
    "project proposal": ["Proposal for {P}", "A Plan for {P}", "{P} Proposal", "Improving {P}"],
    "memo": ["Memo on {P}", "Changes to {P}", "{P} Update for Staff", "A Note on {P}"],
}
HEADINGS = {
    "status report": (["Summary", "Overview"], ["Progress This Month", "Key Metrics", "Risks and Issues",
                                                "What Went Well"], ["Next Steps", "Plans for Next Month"]),
    "meeting notes": (["Purpose", "Background"], ["Discussion", "Decisions", "Open Questions"],
                      ["Action Items", "Next Steps"]),
    "project proposal": (["Background", "The Problem"], ["Proposed Approach", "Costs and Benefits", "Timeline",
                                                         "Risks"], ["Recommendation", "Next Steps"]),
    "memo": (["Purpose", "Background"], ["Details", "What This Means for You", "Key Dates"],
             ["Next Steps", "Questions"]),
}
# structures each kind of section leans on (others still appear, less often)
SECTION_STYLE = [
    (r"Metric|Progress|Went Well", {"metric": 3.0, "passive": 1.5, "simple": 1.5}),
    (r"Risk|Problem|Issue", {"fronted": 2.5, "compound": 2.0, "opinion": 1.5, "concern": 2.5}),
    (r"Question", {"question": 5.0, "concern": 1.0}),
    (r"Next|Action|Plans|Timeline|Dates", {"request": 3.0, "modal": 2.5, "expect": 2.5}),
    (r"Decision|Recommendation", {"agreed": 3.5, "opinion": 2.0, "modal": 1.5}),
    (r"Approach|Benefit|Details|Means", {"benefit": 3.0, "modal": 1.5, "fronted": 1.5}),
]
BASE_WEIGHTS = {"simple": 1.5, "compound": 1.2, "fronted": 1.2, "passive": 1.0, "relative": 0.6, "metric": 0.8,
                "question": 0.3, "request": 0.5, "opinion": 0.8, "list": 0.5, "expect": 0.7, "concern": 0.5,
                "agreed": 0.3, "benefit": 0.5, "modal": 0.8}
ACTION_SECTIONS = re.compile(r"Next|Action|Plans|Dates")


class Ctx:
    """One document's world: the project, its teams, numbers and the objects it talks about."""

    def __init__(self, rng, today):
        self.rng, self.today = rng, today
        self.project, self.project_title = PROJECTS[int(rng.integers(len(PROJECTS)))]
        idx = rng.permutation(len(TEAMS))[:5]
        self.teams = [TEAMS[i] for i in idx] + [("we", True)]
        self.metrics = [METRICS[i] for i in rng.permutation(len(METRICS))[:3]]
        self.objects = [OBJECTS[i] for i in rng.permutation(len(OBJECTS))[:10]] + [(f"the {self.project}", False)]
        self.doc_name = DOC_NAMES[int(rng.integers(len(DOC_NAMES)))]
        self.used = set()

    def pick(self, items):
        """A random item."""
        return items[int(self.rng.integers(len(items)))]

    def team(self, allow_we=True):
        """(subject, plural?)."""
        pool = self.teams if allow_we else self.teams[:-1]
        return self.pick(pool)

    def obj(self):
        """(object, plural?)."""
        return self.pick(self.objects)


# --- clauses (lower case, no final punctuation) -------------------------------------------------

def past_clause(c, time=True):
    """'the finance team reviewed the test plan last week'."""
    subj, _ = c.team()
    obj, _ = c.obj()
    verb = c.pick(VERBS)[0]
    return f"{subj} {verb} {obj}" + (f" {c.pick(TIMES_PAST)}" if time and c.rng.random() < 0.6 else "")


def modal_clause(c, time=True):
    """'we should finalise the budget estimate by Friday'."""
    subj, plural = c.team()
    modal = c.pick(MODALS_PL if plural else MODALS_SG)
    obj, _ = c.obj()
    return f"{subj} {modal} {c.pick(VERBS)[1]} {obj}" + (f" {c.pick(TIMES_FUTURE)}" if time else "")


def state_clause(c):
    """'the test plan is almost ready'."""
    obj, plural = c.obj()
    return f"{obj} {'are' if plural else 'is'} {c.pick(STATES)}"


def problem_clause(c):
    """'the vendor ran into a few data quality issues'."""
    subj, _ = c.team()
    verb = c.pick(["ran into", "reported", "flagged", "had to deal with"])
    return f"{subj} {verb} {c.pick(PROBLEMS)}"


def metric_clause(c):
    """'average response time fell from 22 to 9 hours over the past quarter' (moves the good way
    more often than not)."""
    name, unit, lo, hi, lower_better = c.pick(c.metrics)
    a, b = sorted(int(x) for x in c.rng.integers(lo, hi + 1, size=2))
    if a == b:
        b = a + 1
    better = c.rng.random() < 0.75
    down = better == lower_better
    start, end = (b, a) if down else (a, b)
    verb = c.pick(["fell", "dropped", "went down"] if down else ["rose", "increased", "went up"])
    unit_text = f" {unit}" if unit else ""
    if c.rng.random() < 0.3 and unit != "percent":
        pct = int(round(abs(end - start) / max(start, 1) * 100))
        return f"{name} {verb} by about {pct} percent {c.pick(METRIC_TIMES)}"
    return f"{name} {verb} from {start} to {end}{unit_text} {c.pick(METRIC_TIMES)}"


def verb_it(base):
    """'check it', 'roll it out' (a phrasal verb's particle goes after the pronoun)."""
    verb, _, particle = base.partition(" ")
    return f"{verb} it {particle}" if particle else f"{verb} it"


def cap(text):
    """Capitalise the first letter."""
    return text[:1].upper() + text[1:]


# --- sentence structures ------------------------------------------------------------------------

def s_simple(c):
    return cap(past_clause(c)) + "."


def s_compound(c):
    first = past_clause(c, time=False)
    form = c.rng.integers(3)
    if form == 0:
        return cap(f"{first}, and {past_clause(c)}.")
    if form == 1:
        return cap(f"{first}, but {c.pick([state_clause, problem_clause])(c)}.")
    return cap(f"{problem_clause(c)}, so {modal_clause(c)}.")


def s_fronted(c):
    form = c.rng.integers(4)
    if form == 0:
        return f"Although {past_clause(c, time=False)}, {state_clause(c)}."
    if form == 1:
        return f"{c.pick(['Because', 'Since'])} {problem_clause(c)}, {modal_clause(c)}."
    if form == 2:
        return f"{c.pick(['Once', 'Now that'])} {past_clause(c, time=False)}, {modal_clause(c)}."
    obj, plural = c.obj()
    return (f"If {obj} {'are' if plural else 'is'} {c.pick(['approved', 'signed off', 'ready'])} "
            f"{c.pick(TIMES_FUTURE)}, {modal_clause(c, time=False)}.")


def s_passive(c):
    obj, plural = c.obj()
    subj, _ = c.team(allow_we=False)
    pp = c.pick(VERBS)[2]
    return cap(f"{obj} {'were' if plural else 'was'} {pp} by {subj} {c.pick(TIMES_PAST)}.")


def s_relative(c):
    subj, _ = c.team(allow_we=False)
    obj, _ = c.obj()
    return cap(f"{subj}, {c.pick(REL_PAST)}, {c.pick(VERBS)[0]} {obj}.")


def s_metric(c):
    if c.rng.random() < 0.3:
        return f"{c.pick(['As a result', 'Since then', 'Meanwhile'])}, {metric_clause(c)}."
    return cap(metric_clause(c)) + "."


def s_question(c):
    obj, _ = c.obj()
    subj, plural = c.team(allow_we=False)
    verb = c.pick(VERBS)[1]
    form = c.rng.integers(4)
    if form == 0:
        return f"Should we {verb} {obj} {c.pick(TIMES_FUTURE)}?"
    if form == 1:
        return f"Can {subj} {verb} {obj} {c.pick(TIMES_FUTURE)}?"
    if form == 2:
        return f"What would it take to {verb} {obj} {c.pick(TIMES_FUTURE)}?"
    return f"Is there a simpler way to {verb} {obj}?"


def s_request(c):
    obj, _ = c.obj()
    verb = c.pick(VERBS)[1]
    lead = c.pick(["Please", "Could you please", "Remember to", "Let us"])
    end = "?" if lead.startswith("Could") else "."
    return f"{lead} {verb} {obj} {c.pick(TIMES_FUTURE)}{end}"


def s_opinion(c):
    clause = c.pick([modal_clause, state_clause])(c)
    if c.rng.random() < 0.5:
        return f"{c.pick(OPENERS)}, {clause}."
    return f"{c.pick(['It is worth noting that', 'The main point is that', 'I think'])} {clause}."


def s_list(c):
    subj, _ = c.team()
    objs = []
    while len(objs) < 3:
        o = c.obj()[0]
        if o not in objs:
            objs.append(o)
    verb = c.pick(["focused on", "worked through", "looked at", "spent time on"])
    return cap(f"{subj} {verb} three things: {objs[0]}, {objs[1]} and {objs[2]}.")


def s_expect(c):
    obj, _ = c.obj()
    lead = c.pick(["We expect", "The plan is for", "We would like"])
    return f"{lead} {obj} to be {c.pick(GOALS)} {c.pick(TIMES_FUTURE)}."


def s_concern(c):
    lead = c.pick(["One concern is that", "The biggest risk is that", "There is a chance that"])
    return f"{lead} {c.pick([problem_clause, state_clause])(c)}."


def s_agreed(c):
    if c.rng.random() < 0.5:
        return f"It was agreed that {modal_clause(c)}."
    obj, _ = c.obj()
    who = c.pick(["The group", "The board", "Everyone", "The team"])
    return f"{who} decided to {c.pick(VERBS)[1]} {obj} {c.pick(TIMES_FUTURE)}."


def s_benefit(c):
    obj, _ = c.obj()
    subj, plural = c.team()
    want = c.pick(["want", "plan", "would like"] if plural else ["wants", "plans", "would like"])
    return cap(f"{c.pick(['changing', 'reworking', 'tidying up'])} {obj} would {c.pick(BENEFITS)}, "
               f"so {subj} {want} to {verb_it(c.pick(VERBS)[1])} {c.pick(TIMES_FUTURE)}.")


def s_modal(c):
    return cap(modal_clause(c)) + "."


STRUCTURES = {"simple": s_simple, "compound": s_compound, "fronted": s_fronted, "passive": s_passive,
              "relative": s_relative, "metric": s_metric, "question": s_question, "request": s_request,
              "opinion": s_opinion, "list": s_list, "expect": s_expect, "concern": s_concern,
              "agreed": s_agreed, "benefit": s_benefit, "modal": s_modal}


class SentenceMaker:
    """Sentences for one document: weighted structures, never the same structure or first word
    twice in a row, never the same sentence twice, always safe to type."""

    def __init__(self, ctx):
        self.c = ctx
        self.last_kind = self.last_first = None
        self.count = 0
        self.kinds = []                        # the structure of every sentence made, in order

    def weights(self, heading):
        w = dict(BASE_WEIGHTS)
        for pattern, boost in SECTION_STYLE:
            if re.search(pattern, heading or ""):
                for k, v in boost.items():
                    w[k] = w[k] * v
        return w

    def make(self, heading, kinds=None, max_chars=None):
        """One sentence for a section headed `heading` (kinds: restrict the structures)."""
        w = self.weights(heading)
        names = [k for k in (kinds or w) if k != self.last_kind] or list(kinds or w)
        p = np.array([w[k] for k in names], dtype=float)
        for _ in range(60):
            kind = names[int(self.c.rng.choice(len(names), p=p / p.sum()))]
            text = STRUCTURES[kind](self.c)
            first = text.split()[0]
            if text in self.c.used or first == self.last_first or ir.check_text(text):
                continue
            if max_chars and len(text) > max_chars:
                continue
            break
        self.c.used.add(text)
        self.last_kind, self.last_first = kind, first
        self.count += 1
        self.kinds.append(kind)
        return text


# --- formatting ---------------------------------------------------------------------------------

def emphasise(text, ctx, rng):
    """Runs for a paragraph: sometimes one key phrase in bold (the project, a metric) or the
    name of a document in italic."""
    candidates = [ctx.project] + [m[0].replace("the ", "", 1) for m in ctx.metrics]
    rng.shuffle(candidates)
    if rng.random() < 0.35:
        for phrase in candidates:
            i = text.find(phrase)
            if i > 0 and text[i - 1] == " ":
                return [Run(text[:i]), Run(phrase, bold=True), Run(text[i + len(phrase):])]
    return [Run(text)]


DOC_REFERENCES = [("The details are in the ", " for anyone who needs them."), ("See the ", " for the full list."),
                  ("This follows the ", " that was shared last month."),
                  ("Everything else stays as described in the ", ".")]


def with_doc_name(ctx):
    """Runs of a sentence that points to the document's reference guide (its name in italic)."""
    lead, tail = ctx.pick(DOC_REFERENCES)
    return [Run(lead), Run(ctx.doc_name, italic=True), Run(tail)]


# --- the document ---------------------------------------------------------------------------------

def make_title(ctx, kind):
    """A title for this kind of document that fits TITLE_MAX characters."""
    q = (ctx.today.month - 1) // 3 + 1
    options = [t.format(P=ctx.project_title, M=ctx.today.strftime("%B"), q=q, D=ctx.today.strftime("%A"))
               for t in TITLES[kind]]
    fits = [t for t in options if len(t) <= TITLE_MAX] or [ctx.project_title]
    return ctx.pick(fits)


def pick_headings(ctx, kind, n):
    """n headings in a sensible order: an opening one, some middle ones, often a closing one."""
    first, middle, last = HEADINGS[kind]
    out = [ctx.pick(first)]
    k_mid = n - 1 - (1 if n >= 3 or ctx.rng.random() < 0.5 else 0)
    mids = [middle[i] for i in sorted(ctx.rng.permutation(len(middle))[:max(0, k_mid)])]
    out += mids
    if len(out) < n:
        out.append(ctx.pick(last))
    return out


def split_counts(total, parts, rng):
    """total split into `parts` positive integers, roughly even with some jitter."""
    if parts <= 1:
        return [total]
    w = rng.uniform(0.7, 1.3, size=parts)
    raw = w / w.sum() * total
    counts = np.maximum(1, np.floor(raw)).astype(int)
    while counts.sum() < total:
        counts[int(np.argmax(raw - counts))] += 1
    while counts.sum() > total:
        counts[int(np.argmax(counts))] -= 1
    return [int(x) for x in counts]


def paragraph_sizes(n, rng):
    """Split n sentences into paragraphs of two to five (a lone sentence only if n is 1)."""
    sizes = []
    while n > 0:
        if n <= 5:
            k = n if n <= 3 or rng.random() < 0.5 else int(rng.integers(2, n - 1))
        else:
            k = int(rng.integers(2, 6))
            if n - k == 1:
                k -= 1 if k > 2 else -1
        sizes.append(k)
        n -= k
    return sizes


def generate_office_doc(rng, n_sentences=25, kind=None, today=None):
    """An office document (ir.Document) with exactly n_sentences sentences; meta records its
    kind, title and the structure of every sentence."""
    if n_sentences < 1:
        raise ValueError("n_sentences must be at least 1")
    today = today or datetime.date.today()
    ctx = Ctx(rng, today)
    kind = kind or KINDS[int(rng.integers(len(KINDS)))]
    maker = SentenceMaker(ctx)
    title = make_title(ctx, kind)
    blocks = [Title(title, size=float(rng.choice(TITLE_SIZES)) if rng.random() < 0.6 else None)]
    if rng.random() < 0.7:                     # a date line under the title
        date = f"{today.day} {today.strftime('%B %Y')}"
        line = ctx.pick([f"Prepared by the {ctx.pick(['operations', 'project', 'finance', 'admin'])} team, {date}",
                         f"Last updated {date}", f"Draft for discussion, {date}"])
        blocks.append(Subtitle(line) if rng.random() < 0.5 else Paragraph([Run(line)], align="center", size=10))

    n_sections = 1 if n_sentences < 4 else 2 if n_sentences < 9 else int(rng.integers(2, 5))
    headings = pick_headings(ctx, kind, n_sections)
    counts = split_counts(n_sentences, len(headings), rng)
    doc_name_used = False
    for heading, count in zip(headings, counts):
        blocks.append(Heading(heading, 1))
        left = count
        if re.search(r"Decision|Recommendation|Summary|Purpose", heading) and left >= 3 and rng.random() < 0.5:
            text = maker.make(heading, kinds=["agreed", "opinion", "modal", "simple"], max_chars=70)
            blocks.append(Paragraph([Run(text, bold=True)]))       # a one-line statement in bold
            left -= 1
        bullets = None
        if ACTION_SECTIONS.search(heading) and left >= 3 and rng.random() < 0.65:
            k = min(left, int(rng.integers(3, 6)))
            bullets = Bullets([[Run(maker.make(heading, kinds=["request", "modal", "expect"]))] for _ in range(k)])
            left -= k
        paras = []
        for size in paragraph_sizes(left, rng):
            if not doc_name_used and rng.random() < 0.15:   # one sentence points to a guide, in italic
                doc_name_used = True
                maker.count += 1
                maker.kinds.append("reference")
                rest = [maker.make(heading) for _ in range(size - 1)]
                paras.append(Paragraph(with_doc_name(ctx) + ([Run(" " + " ".join(rest))] if rest else [])))
                continue
            text = " ".join(maker.make(heading) for _ in range(size))
            paras.append(Paragraph(emphasise(text, ctx, rng)))
        if bullets is not None:                      # a lead-in paragraph first, if there is one
            if paras:
                blocks.append(paras.pop(0))
            blocks.append(bullets)
        blocks += paras
    doc = Document(blocks, meta={"kind": kind, "title": title, "sentences": maker.count, "structures": maker.kinds,
                                 "project": ctx.project})
    problems = ir.validate(doc)
    if problems:
        raise ValueError(f"Generated document is not safe to type: {problems[:3]}")
    return doc


def count_sentences(doc):
    """Sentences in the paragraphs and bullet items of a document (for checks): every '.', '?'
    or '!' that ends a sentence."""
    n = 0
    for b in doc.blocks:
        texts = [b.text] if isinstance(b, Paragraph) else \
            ["".join(r.text for r in item) for item in b.items] if isinstance(b, Bullets) else []
        for t in texts:
            n += len(re.findall(r"[.?!](?=\s|$)", t))
    return n


def file_name(title):
    """A file name a person would give the document: its title, without punctuation."""
    return re.sub(r"[^A-Za-z0-9 ]+", "", title).strip() or "Document"


class OfficeProject(WritingProject):
    """Normal mode's document: generated at the start, then typed section by section by
    tasks.Workspace.write() like the research paper, saved as '<title>.odt' in
    runs/normal/<title>-<time>/ (with outline.json, paper.md, progress.json)."""

    def __init__(self, human, n_sentences=25, kind=None, folder=None, revisions="heuristic", log=print):
        paper = generate_office_doc(human.rng, n_sentences, kind)
        title = paper.meta["title"]
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        folder = os.path.abspath(folder or os.path.join(RUNS_DIR, f"{slugify(title)}-{stamp}"))
        super().__init__(human, folder, os.path.join(folder, file_name(title) + ".odt"), revisions=revisions, log=log)
        self.set_paper(paper)
        log(f"Document to write: {paper.meta['kind']} {title!r} - {paper.meta['sentences']} sentences, "
            f"{paper.word_count()} words, about {self.remaining_seconds() / 60:.1f} min of typing "
            f"(preview in {os.path.join(folder, 'paper.md')})")
