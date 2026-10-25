"""
The two 30-minute runs: a browsing phase, then a writing phase, each of fixed length
(--browse-minutes / --write-minutes, 15 each by default).

normal    visit about --target-sites of the allowed websites (surfing.SiteSource), each for a
          while - reading, following links around it, searching it - now and then copying a
          page's address into a scratch note in Notepad; close Chrome. Then, in the writing
          phase's time: open Writer, build last month's budget sheet in OpenOffice Calc
          (--sheet-minutes: spreadsheet_ops - totals, a SUMPRODUCT summary copied down,
          formats, a column chart; saved as .ods), jot its totals in the note, and write a generated office document of
          exactly --target-sentences sentences (office_writing.OfficeProject) in OpenOffice
          Writer that quotes the totals and pastes the sheet's summary in as a table; save it
          under its title, close Writer and Calc, save the note as notes.txt.
research  search the topic and follow links in the articles, taking notes from every page
          (surfing.ResearchSource + research.project.ResearchProject); close Chrome. Then
          compose a paper from those notes to fit the writing time, type it, save it, close
          Writer.

Phase timing: the browsing deadline leaves CLOSE_RESERVE seconds to close Chrome inside the
phase; writing never starts a block that wouldn't finish SAVE_RESERVE seconds before the end,
which is kept for the last save and closing Writer. When the writing runs ahead of time the
person reads over what they wrote (scrolling up, then back to the end) between sections; when
the document is done early they read it over until it is time to save and close. A phase
that has time left over (Chrome closed a little early) waits it out - the person isn't
touching anything - so each phase takes its full length.
"""

import time

from algorithms.session import TaskError
from research.writer_ops import Op
from surfing import ResearchSource, SiteSource, load_sites
from tasks import Workspace, close_writer, log

CLOSE_RESERVE = 20.0     # seconds kept at the end of browsing to close Chrome
SAVE_RESERVE = 45.0      # seconds kept at the end of writing for the last save and closing Writer
TIDY_RESERVE = 40.0      # normal mode: ... and for closing Calc and saving the scratch note
MAX_SHEET_SHARE = 0.4    # the budget sheet takes at most this share of the writing phase
MIN_SHEET = 60.0         # seconds: less than this and there is no budget sheet


def clock(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def run_normal(human, args):
    """Normal mode: browse (target sites, links noted in Notepad now and then), build the budget
    sheet in Calc, then write a generated document (target sentences) that quotes it."""
    from office_writing import OfficeProject
    from spreadsheet_ops import generate_budget, rows_for_time
    start = time.monotonic()
    browse_end = start + args.browse_minutes * 60
    write_end = browse_end + args.write_minutes * 60
    sheet_seconds = min(args.sheet_minutes, args.write_minutes * MAX_SHEET_SHARE) * 60
    log(f"=== Normal mode: browsing {args.browse_minutes:g} min (target {args.target_sites} sites), then "
        + (f"a budget sheet in Calc ({sheet_seconds / 60:.1f} min) and " if sheet_seconds >= MIN_SHEET else "")
        + f"writing {args.write_minutes:g} min in all (target {args.target_sentences} sentences)")
    ws = Workspace(human, args.profile, args.profile_index, args.revisions)
    ws.scratch = True
    sites = load_sites(args.sites_file) if args.sites_file else None
    browse_phase(human, ws, start, browse_end, SiteSource(human.rng, sites), target_sites=args.target_sites)
    budget, sheet_end = None, browse_end + sheet_seconds
    if sheet_seconds >= MIN_SHEET:
        # Writer first: OpenOffice shows UI Automation only its first window. Opening it can take a
        # minute, so the sheet is sized to the time that is left after it
        attempt("Opening Writer", ws.writer_window)
        budget = generate_budget(human.rng, rows_for_time(sheet_end - time.monotonic(), human.persona.wpm))
    proj = OfficeProject(human, args.target_sentences, revisions=args.revisions, log=log, budget=budget)
    ws.project = proj
    if budget is not None:
        sheet_phase(human, ws, proj, browse_end, sheet_end)
    write_phase(human, ws, proj, time.monotonic(), write_end, reserve=SAVE_RESERVE + TIDY_RESERVE,
                tidy=lambda: tidy_up(ws, proj))
    log(f"=== Normal mode finished in {clock(time.monotonic() - start)} - files in {proj.folder}")


def run_research(human, args, proj):
    """Research mode: browse the topic (notes from every page), then write the paper from them.
    A resumed project whose paper is already composed goes straight to writing."""
    start = time.monotonic()
    browse_end = start + args.browse_minutes * 60 if proj.paper is None else start
    write_end = browse_end + args.write_minutes * 60
    log(f"=== Research mode on {proj.topic!r}: "
        + (f"browsing {args.browse_minutes:g} min, then " if proj.paper is None else "")
        + f"writing {args.write_minutes:g} min")
    ws = Workspace(human, args.profile, args.profile_index, args.revisions, research=proj)
    if proj.paper is None:
        browse_phase(human, ws, start, browse_end, ResearchSource(proj))
        if not proj.has_material():
            raise TaskError("Not enough was read to write a paper")
        human.think("mental")
        proj.compose(target_words=args.words, seconds=max(60.0, write_end - time.monotonic() - SAVE_RESERVE) * 0.85)
    write_phase(human, ws, proj, browse_end, write_end)
    log(f"=== Research {'finished' if proj.finished else 'stopped (out of time)'} in "
        f"{clock(time.monotonic() - start)} - files in {proj.folder}")


def sheet_phase(human, ws, proj, start, end):
    """Build the project's budget sheet in Calc (the optional steps while there is time before
    `end`) and note its totals in the scratch note. Writer is opened before this (OpenOffice shows
    UI Automation only its first window, so the report's window comes first). If the sheet can't
    be built the report's table is typed instead of pasted."""
    from spreadsheet_ops import money
    budget = proj.budget
    log(f"--- Spreadsheet phase ({clock(end - start)}): {len(budget.rows)} expenses for {budget.month}")
    if not attempt("Spreadsheet", lambda: ws.build_sheet(budget, proj.sheet_path, deadline=end)):
        return
    cat, cents, _ = budget.top
    human.think("mental")
    attempt("Note", lambda: ws.jot(f"{budget.month} spend {money(budget.total)}, {cat.lower()} highest "
                                   f"at {money(cents)}"))


def tidy_up(ws, proj):
    """After the report: close Calc (the sheet was saved) and save the scratch note next to
    the report, closing its page."""
    if ws.calc_win is not None:
        attempt("Closing Calc", ws.close_sheet)
    if ws.notepad:
        attempt("Saving the note", lambda: ws.save_note(proj.notes_path))


def attempt(what, action):
    """Run a step that may fail without ending the run (a TaskError or an unexpected UI / COM
    error is logged); the emergency stop still stops everything. True if it went through."""
    try:
        action()
        return True
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        if type(exc).__name__ == "FailSafeException":
            raise
        log(f"{what} stopped ({type(exc).__name__}: {exc})")
        return False


def browse_phase(human, ws, start, end, source, target_sites=None):
    """Browse until `end` (minus the time to close Chrome), close Chrome, wait out the rest."""
    log(f"--- Browsing phase ({clock(end - start)})")
    try:
        ws.browse(end - time.monotonic(), deadline=end - CLOSE_RESERVE, source=source, target_sites=target_sites,
                  close=True)
    except TaskError as exc:                  # e.g. Chrome didn't open: the writing phase still runs
        log(f"Browsing stopped: {exc}")
    wait_until(end, "browsing")


def write_phase(human, ws, proj, start, end, reserve=SAVE_RESERVE, tidy=None):
    """Write the project's document section by section until `end`, reading it over when
    there is time to spare, then save it and close Writer. reserve: seconds kept at the end for
    saving and closing; tidy: called after Writer is closed (normal mode closes Calc and saves
    its note)."""
    rng = human.rng
    left = lambda: end - time.monotonic()    # noqa: E731
    need, have = proj.remaining_seconds(), max(0.0, left() - reserve)
    log(f"--- Writing phase ({clock(end - start)}): about {need / 60:.1f} min of typing for {have / 60:.1f} min")
    if need > have:
        log(f"At {human.persona.wpm:.0f} wpm the whole document probably won't fit - it is saved as far as it "
            f"gets (a smaller --target-sentences or a faster --wpm fits it)")
    failures = 0
    while not proj.finished and left() - reserve > 15:
        before = proj.op_index
        try:
            ws.write(left() - reserve)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:              # TaskError, or an unexpected UI / COM error
            if type(exc).__name__ == "FailSafeException":
                raise
            failures += 1
            log(f"Writing stopped ({type(exc).__name__}: {exc})")
            if failures >= 3:
                break
            continue
        failures = 0
        if proj.op_index == before:           # the next block wouldn't finish in time
            log("Not enough time left for the next part of the document")
            break
        if proj.finished:
            break
        spare = left() - reserve - proj.remaining_seconds() * 1.15
        if spare > 40 and rng.random() < 0.6:     # ahead of time: read over what's written
            ws.review_document(min(spare * float(rng.uniform(0.2, 0.45)), 120.0))
        else:
            human.think("mental", 1.5)            # what comes next
    while left() - reserve > 25 and ws.writer_win is not None:   # done early (or the next part
        # won't fit): read it over and think until it is time to save and close
        ws.review_document(min(left() - reserve - 10, float(rng.uniform(40, 120))))
        pause = min(left() - reserve - 5, float(rng.uniform(5, 25)))
        if pause > 0:
            time.sleep(pause)
    finish_document(human, ws, proj)
    if tidy:
        tidy()
    wait_until(end, "writing")


def finish_document(human, ws, proj):
    """Save the document (Save As under its name if it was never saved, else Ctrl+S or the Save
    button - a finished document was saved by its last step, and is saved again now and
    then out of habit) and close Writer."""
    win = ws.writer_win
    if win is None or not ws._alive(win.NativeWindowHandle) or not ws.back_to(win.NativeWindowHandle):
        log("No Writer window to save and close")
        return
    if not proj.finished or not proj.saved or human.rng.random() < 0.4:
        ex = proj.executor(ws)
        ex.op_save(Op("save", "final"))
        proj.saved = ex.saved
        proj.save_progress()
        log(f"Saved {proj.doc_path} ({ex.summary()})")
        time.sleep(1.0)
    human.think("mental")
    if close_writer(human, win):
        ws.writer_win = ws.writer_doc = None


def wait_until(end, phase):
    """Let the rest of a phase pass without input (the person sits back for a moment)."""
    rest = end - time.monotonic()
    if rest > 0.5:
        log(f"{phase.capitalize()} phase: {rest:.0f} s left - pausing until it ends")
        time.sleep(rest)
    else:
        log(f"{phase.capitalize()} phase ran {-rest:.0f} s over")
