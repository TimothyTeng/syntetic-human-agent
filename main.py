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

Word demo (--process word): open Word, start a blank document, click into the page
and type a paragraph with the learned typing model (typos + corrections, thinking
pauses, changes of wording, arrowing back to fix things), then check the text.

Session (--process session): a continuous working session of --minutes minutes
(default 15) mixing browsing, writing in Word, entering expenses in Excel and jotting
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
    python main.py --process word                     # type a paragraph into Word
    python main.py --process word --wpm 80            # ... as an 80 wpm typist
    python main.py --process word --wpm 55 --text-file notes.txt --revisions llm
    python main.py --process session --minutes 15 --seed 7

Abort at any time by slamming the mouse into a screen corner.
"""

import argparse
import sys
import time

from algorithms.behaviour import Human
from algorithms.session import Session, TaskError
from controller import browser
from tasks import (WORD_PARAGRAPH, Workspace, click_into_document, describe_profiles, log, open_browser,
                   open_word_document, search_and_read, session_activities, type_into_document)


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


def run_word_demo(human, args):
    text = args.text
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read().strip()
    win, doc = open_word_document(human)
    click_into_document(human, win, doc)
    type_into_document(human, doc, text, args.revisions)
    log("Word demo finished (document left open, unsaved).")


def run_session(human, args):
    ws = Workspace(human, args.profile, args.profile_index, args.revisions)
    Session(human, session_activities(ws), minutes=args.minutes, log=log).run()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--process", default="search", choices=["search", "word", "session"],
                    help="search = Chrome search demo, word = type a paragraph into a new Word document, "
                         "session = browse + write for --minutes")
    ap.add_argument("--minutes", type=float, default=15, help="session length")
    ap.add_argument("--wpm", type=float, default=None, help="typing speed (default: from the sampled persona)")
    ap.add_argument("--text", default=WORD_PARAGRAPH, help="word demo: text to type")
    ap.add_argument("--text-file", default=None, help="word demo: type the contents of this file instead")
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

    run = {"search": run_search_demo, "word": run_word_demo, "session": run_session}[args.process]
    try:
        run(make_human(args), args)
    except TaskError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
