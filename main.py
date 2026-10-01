"""
Demos of the simulated person (the task code lives in tasks.py).

Chrome demo (--process search, default):
  1. Open Chrome (Start menu: Win key or the Start button); if the "Who's using
     Chrome?" picker appears, move to the chosen profile and click it; maximise
  2. Focus the address bar: click it with a learned trajectory, or Ctrl+L
  3. Type a search query and press Enter
  4. Click a search result
  5. "Read" the page: scroll bursts, pauses, small mouse drifts and now and then
     following a line of text with the cursor
  6. Go back: the Back button or Alt+Left
  7. Click a different result and read a little

Writer demo (--process writer): open OpenOffice Writer (it starts on a blank document),
click into the page and type a paragraph with the learned typing model (typos +
corrections, thinking pauses, changes of wording, arrowing back to fix things), then
check the text. (--process word is still accepted for it.)

Research (--process research --topic "..."): a researcher at work. Searches the topic
(Wikipedia, Britannica, then the topic with the main headings of the first page), takes
notes from every page it reads (page text through UI Automation, which part was in view),
then writes a research paper into OpenOffice Writer from those notes - title, abstract,
headings, paragraphs with citations, bold / italic, font sizes, tables, bullets,
references - typed and formatted with shortcuts or the toolbar and menus, section by
section, saving as it goes and now and then going back to a source. Everything is kept in
runs/research/<topic>-<time>/ (notes, page captures, the composed paper as paper.md, the
.odt); --resume DIR carries on.

Compose (--process compose --notes DIR): no UI at all - writes the paper from saved notes
(a research folder, or a folder of page captures such as tests/fixtures/pages) and prints
it as Markdown; --dry-run also lists every Writer action the person would perform.

Session (--process session): a continuous working session of --minutes minutes
(default 15) mixing browsing, writing in Writer, entering expenses in Excel and jotting
notes in Notepad, with idle breaks; windows opened earlier are switched back to
(Alt+Tab) rather than reopened. The person starts fresh (a little faster and more
accurate than their average) and slowly tires; --no-fatigue turns that off.

All demos act as one simulated person (algorithms.behaviour.Persona, sampled from
--seed): typing speed, mouse speed, whether they reach for shortcuts or the mouse,
how long they think and how fast they read. Override single traits with --wpm,
--pause-scale, --temperature, --shortcut-pref.

Usage:
    python main.py                      # uses models/mouse_mdn.npz if present
    python main.py --no-model           # force the fallback generator
    python main.py --query "wikipedia" --read-seconds 20 --seed 42
    python main.py --profile Work --profile-index 1   # 2nd profile matching "Work"
    python main.py --list-profiles                    # print the picker's profiles and exit
    python main.py --process writer                   # type a paragraph into OpenOffice Writer
    python main.py --process writer --wpm 80          # ... as an 80 wpm typist
    python main.py --process writer --wpm 55 --text-file notes.txt --revisions llm
    python main.py --process session --minutes 15 --seed 7
    python main.py --process research --topic "Apollo program" --minutes 45
    python main.py --process research --resume runs/research/apollo-program-20261001-1030
    python main.py --process session --topic "Apollo program"   # session browsing/writing does the research
    python main.py --process compose --notes tests/fixtures/pages --topic "Apollo program" --dry-run

Abort at any time by slamming the mouse into a screen corner.
"""

import argparse
import glob
import json
import os
import sys
import time

from algorithms.behaviour import Human
from algorithms.session import Session, TaskError
from controller import browser
from tasks import (WRITER_PARAGRAPH, Workspace, click_into_document, describe_profiles, log, open_browser,
                   open_writer_document, search_and_read, session_activities, type_into_document)


def make_human(args):
    """The simulated person for this run (persona sampled from --seed, CLI overrides win)."""
    human = Human(seed=args.seed, use_model=not args.no_model, fatigue=not args.no_fatigue,
                  wpm=args.wpm, pause_scale=args.pause_scale,
                  mouse_temperature=args.temperature, shortcut_pref=args.shortcut_pref,
                  click_spread=args.click_spread)
    log(f"Persona: {human.persona.describe()}")
    log(f"Mouse backend: {human.hm.backend} | click timing: {human.hm.clicks.summary()}")
    log("Starting in 3 s - don't touch the mouse/keyboard.")
    time.sleep(3)
    return human


def run_search_demo(human, args):
    open_browser(human, args.profile, args.profile_index)
    search_and_read(human, args.query, args.read_seconds, link_hint=args.link_hint or args.query)
    log("Demo finished.")


def run_writer_demo(human, args):
    text = args.text
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read().strip()
    win, doc = open_writer_document(human)
    click_into_document(human, win, doc)
    type_into_document(human, doc, text, args.revisions)
    log("Writer demo finished (document left open, unsaved).")


def run_session(human, args):
    proj = make_project(human, args)
    ws = Workspace(human, args.profile, args.profile_index, args.revisions, research=proj)
    try:
        Session(human, session_activities(ws), minutes=args.minutes, log=log).run()
    finally:
        if proj is not None:
            for line in proj.diag.summary():
                log(line)


def make_project(human, args):
    """The research project for --topic / --resume (None without either)."""
    from research.project import ResearchProject
    kw = dict(n_sources=args.sources, paraphrase=args.paraphrase, revisions=args.revisions)
    if args.resume:
        return ResearchProject.resume(human, args.resume, log=log, **kw)
    if args.topic:
        proj = ResearchProject(human, args.topic, log=log, **kw)
        log(f"Research folder: {proj.folder}")
        return proj
    return None


def run_research(human, args):
    """Read sources (up to 35-45% of the time), compose the paper to fit the time left, then
    write it section by section, sometimes going back to a source in between."""
    proj = make_project(human, args)
    if proj is None:
        sys.exit("--process research needs --topic (or --resume DIR)")
    ws = Workspace(human, args.profile, args.profile_index, args.revisions, research=proj)
    try:
        research_flow(human, args, proj, ws)
    finally:                                  # the self-checks, also when the run stopped early
        for line in proj.diag.summary():
            log(line)


def research_flow(human, args, proj, ws):
    """
    The research run itself (run_research adds the end-of-run self-check summary):
      1. read: search and take notes until --sources pages are read or the reading share of
         the time is used up (a failed search is logged and the next query tried)
      2. compose the paper to fit ~85% of the time left
      3. write it one section per turn; between sections now and then go back to the
         browser to re-read a bit of a source; stop when the next block wouldn't fit
      4. out of time: save what is written, as a person would before leaving it
    """
    total = args.minutes * 60
    end = time.monotonic() + total
    left = lambda: end - time.monotonic()  # noqa: E731
    failures = 0
    reading_share = 0.35 if total < 45 * 60 else 0.45     # short runs: keep enough time to write
    # 1. read (skipped when resuming a project whose paper is already composed)
    while proj.paper is None and proj.needs_sources() and left() > (1 - reading_share) * total and failures < 3:
        try:
            ws.browse(left())
            failures = 0
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:              # TaskError, or an unexpected UI / COM error
            if type(exc).__name__ == "FailSafeException":
                raise                         # the emergency stop (mouse in a corner) always wins
            failures += 1
            log(f"Search didn't work out ({type(exc).__name__}: {exc}) - trying the next one")
    if not proj.has_material():
        raise TaskError("Not enough was read to write a paper")
    # 2. compose (kept as is when resuming)
    proj.compose(target_words=args.words, seconds=left() * 0.85)
    # 3. write, a section per turn
    while not proj.finished and left() > 15:
        before = proj.op_index
        ws.write(left())
        if proj.op_index == before:                # the next block wouldn't finish in time
            log("Not enough time left for the next part of the paper")
            break
        if not proj.finished and left() > 90 and human.rng.random() < 0.3:
            ws.glance_at_source(float(human.rng.uniform(8, 25)))   # "let me check that in the source"
    # 4. stopped early: save the paper so far (finished papers were saved by their last step)
    if not proj.finished and proj.saved and ws.writer_win is not None and ws.back_to(ws.writer_win.NativeWindowHandle):
        log("Out of time - saving the paper so far")          # a person saves before leaving it
        human.think("glance")
        human.hotkey("ctrl", "s")
        time.sleep(1.0)
    log(f"Research {'finished' if proj.finished else 'stopped (out of time)'} - files in {proj.folder}")


def run_compose(args):
    """Compose offline from saved notes or page captures; print the paper (and, with
    --dry-run, the Writer actions)."""
    import numpy as np
    from research import ir
    from research.compose import ComposeConfig, compose_paper
    from research.extract import parse_capture
    from research.notes import ResearchNotes
    from research.writer_exec import WriterExecutor
    from research.writer_ops import render

    folder = args.notes
    if not folder:
        sys.exit("--process compose needs --notes DIR (a research folder or a folder of page captures)")
    if os.path.exists(os.path.join(folder, "notes.json")):     # a research run's folder
        notes = ResearchNotes.load(folder)
    else:                                                        # a folder of page captures (e.g. test fixtures)
        notes = ResearchNotes(args.topic or "")
        for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
            with open(path, encoding="utf-8") as f:
                notes.add(parse_capture(json.load(f)))
    topic = args.topic or notes.topic
    if not topic:
        sys.exit("Give the topic with --topic")
    cfg = ComposeConfig(target_words=args.words or 800, paraphrase=args.paraphrase)
    doc = compose_paper(notes, topic, np.random.default_rng(args.seed), cfg)
    print(ir.to_markdown(doc))
    ops = render(doc)
    print(f"\n<!-- {doc.word_count()} words, {len(doc.blocks)} blocks, {len(ops)} editing steps -->")
    if args.dry_run:                     # a sampled persona "performs" the steps; actions are listed, not done
        human = Human(seed=args.seed, use_model=False, fatigue=False, shortcut_pref=args.shortcut_pref)
        ex = WriterExecutor(human, None, None, save_path=os.path.abspath("paper.odt"), dry_run=True)
        ex.run(ops)
        print(f"\n<!-- Writer actions (shortcut preference {human.persona.shortcut_pref:.2f}):")
        for action in ex.actions:
            print("  " + action[:110])
        print(f"{ex.summary()} -->")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--process", default="search",
                    choices=["search", "writer", "word", "session", "research", "compose"],
                    help="search = Chrome search demo, writer (or word) = type a paragraph into a new "
                         "OpenOffice Writer document, session = browse + write for --minutes, research = read "
                         "about --topic and write a paper in Writer, compose = write the paper from saved notes "
                         "(no UI)")
    ap.add_argument("--topic", default=None, help="research: what to research and write about")
    ap.add_argument("--sources", type=int, default=4, help="research: how many pages to read before writing")
    ap.add_argument("--words", type=int, default=None,
                    help="research / compose: length of the paper (default: what fits the time left; compose 800)")
    ap.add_argument("--paraphrase", type=int, default=1, choices=[0, 1, 2],
                    help="research: 0 = sentences as read, 1 = light phrase rewrites, 2 = also 'According to ...'")
    ap.add_argument("--resume", default=None, help="research: carry on with the project in this folder")
    ap.add_argument("--notes", default=None, help="compose: research folder or folder of page captures")
    ap.add_argument("--dry-run", action="store_true", help="compose: also list the Writer actions, perform nothing")
    ap.add_argument("--minutes", type=float, default=15, help="session length")
    ap.add_argument("--wpm", type=float, default=None, help="typing speed (default: from the sampled persona)")
    ap.add_argument("--text", default=WRITER_PARAGRAPH, help="writer demo: text to type")
    ap.add_argument("--text-file", default=None, help="writer demo: type the contents of this file instead")
    ap.add_argument("--revisions", default="heuristic", choices=["heuristic", "llm"],
                    help="where changed wordings come from when composing")
    ap.add_argument("--pause-scale", type=float, default=None,
                    help="thinking-pause length while composing (persona default 0.7; 1.0 = timed-essay writers)")
    ap.add_argument("--query", default="history of the internet wikipedia")
    ap.add_argument("--link-hint", default="wikipedia", help="text the first clicked result should contain")
    ap.add_argument("--read-seconds", type=float, default=None,
                    help="seconds on the first result (default: from the page's length)")
    ap.add_argument("--no-model", action="store_true", help="use the fallback generator")
    ap.add_argument("--temperature", type=float, default=None, help="mouse path variety (persona default 1.0)")
    ap.add_argument("--shortcut-pref", type=float, default=None,
                    help="0 = always reaches for the mouse, 1 = always keyboard shortcuts (default: sampled)")
    ap.add_argument("--click-spread", type=float, default=None,
                    help="click scatter around element centres (persona default 0.17; smaller = more precise)")
    ap.add_argument("--no-fatigue", action="store_true",
                    help="keep the person at their average pace (no fresh start, no slowing down)")
    ap.add_argument("--seed", type=int, default=None, help="reproduces the whole run, persona included")
    ap.add_argument("--profile", default=None,
                    help="profile to pick on the 'Who's using Chrome?' screen (profile or account name)")
    ap.add_argument("--profile-index", type=int, default=0,
                    help="which match to use when several profiles share the name (0 = leftmost)")
    ap.add_argument("--list-profiles", action="store_true",
                    help="print the profiles on an open profile picker and exit")
    args = ap.parse_args(argv)

    if args.list_profiles:
        profiles = browser.list_profiles(timeout=1)
        print(describe_profiles(profiles).replace("; ", "\n") if profiles else "Profile picker is not open.")
        return

    if args.process == "compose":
        run_compose(args)
        return
    run = {"search": run_search_demo, "writer": run_writer_demo, "word": run_writer_demo, "session": run_session,
           "research": run_research}[args.process]
    try:
        run(make_human(args), args)
    except TaskError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
