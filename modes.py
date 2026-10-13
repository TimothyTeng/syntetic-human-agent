"""
The two 30-minute runs: a browsing phase, then a writing phase, each of fixed length
(--browse-minutes / --write-minutes, 15 each by default).

normal    visit about --target-sites of the allowed websites (surfing.SiteSource), each for a
          while - reading, following links around it, searching it; close Chrome. Then write a generated office document
          of exactly --target-sentences sentences (office_writing.OfficeProject) in
          OpenOffice Writer, save it under its title and close Writer.
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


def clock(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def run_normal(human, args):
    """Normal mode: browse (target sites), then write a generated document (target sentences)."""
    from office_writing import OfficeProject
    start = time.monotonic()
    browse_end = start + args.browse_minutes * 60
    write_end = browse_end + args.write_minutes * 60
    log(f"=== Normal mode: browsing {args.browse_minutes:g} min (target {args.target_sites} sites), then writing "
        f"{args.write_minutes:g} min (target {args.target_sentences} sentences)")
    ws = Workspace(human, args.profile, args.profile_index, args.revisions)
    sites = load_sites(args.sites_file) if args.sites_file else None
    browse_phase(human, ws, start, browse_end, SiteSource(human.rng, sites), target_sites=args.target_sites)
    proj = OfficeProject(human, args.target_sentences, revisions=args.revisions, log=log)
    ws.project = proj
    write_phase(human, ws, proj, browse_end, write_end)
    log(f"=== Normal mode finished in {clock(time.monotonic() - start)} - document: {proj.doc_path}")


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


def browse_phase(human, ws, start, end, source, target_sites=None):
    """Browse until `end` (minus the time to close Chrome), close Chrome, wait out the rest."""
    log(f"--- Browsing phase ({clock(end - start)})")
    try:
        ws.browse(end - time.monotonic(), deadline=end - CLOSE_RESERVE, source=source, target_sites=target_sites,
                  close=True)
    except TaskError as exc:                  # e.g. Chrome didn't open: the writing phase still runs
        log(f"Browsing stopped: {exc}")
    wait_until(end, "browsing")


def write_phase(human, ws, proj, start, end):
    """Write the project's document section by section until `end`, reading it over when
    there is time to spare, then save it and close Writer."""
    rng = human.rng
    left = lambda: end - time.monotonic()    # noqa: E731
    need, have = proj.remaining_seconds(), max(0.0, left() - SAVE_RESERVE)
    log(f"--- Writing phase ({clock(end - start)}): about {need / 60:.1f} min of typing for {have / 60:.1f} min")
    if need > have:
        log(f"At {human.persona.wpm:.0f} wpm the whole document probably won't fit - it is saved as far as it "
            f"gets (a smaller --target-sentences or a faster --wpm fits it)")
    failures = 0
    while not proj.finished and left() - SAVE_RESERVE > 15:
        before = proj.op_index
        try:
            ws.write(left() - SAVE_RESERVE)
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
        spare = left() - SAVE_RESERVE - proj.remaining_seconds() * 1.15
        if spare > 40 and rng.random() < 0.6:     # ahead of time: read over what's written
            ws.review_document(min(spare * float(rng.uniform(0.2, 0.45)), 120.0))
        else:
            human.think("mental", 1.5)            # what comes next
    while left() - SAVE_RESERVE > 25 and ws.writer_win is not None:   # done early (or the next part
        # won't fit): read it over and think until it is time to save and close
        ws.review_document(min(left() - SAVE_RESERVE - 10, float(rng.uniform(40, 120))))
        pause = min(left() - SAVE_RESERVE - 5, float(rng.uniform(5, 25)))
        if pause > 0:
            time.sleep(pause)
    finish_document(human, ws, proj)
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
