"""
Chrome demo for the learned mouse model.

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

Both demos act as one simulated person (algorithms.behaviour.Persona, sampled from
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

Abort at any time by slamming the mouse into a screen corner.
"""

import argparse
import ctypes
import sys
import time

import uiautomation as auto

from algorithms import reading
from algorithms.behaviour import Human
from controller import browser, office, ui_elements

T0 = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - T0:6.1f}s] {msg}", flush=True)


def wait_for_navigation(settle=0.8):
    """Give Chrome a moment to start navigating, then wait for load to finish."""
    time.sleep(settle)
    browser.wait_for_page_load(timeout=20)


# Page chrome / account links a reader wouldn't click as a "result"
NAV_LINK_NAMES = {"skip to main content", "accessibility help", "sign in", "sign out",
                  "settings", "privacy", "terms", "feedback", "read more", "go to google home"}


def usable_links(page_rect=None, exclude=()):
    """Visible links with real text and a clickable size, top-to-bottom."""
    out = []
    for link in browser.find_links():
        l, t, r, b = link["rect"]
        if len(link["name"]) < 4 or (r - l) < 20 or (b - t) < 8:
            continue
        if link["name"].strip().lower() in NAV_LINK_NAMES:
            continue
        if page_rect and not (page_rect[1] + 20 < t and b < page_rect[3] - 5):
            continue
        if link["name"] in exclude:
            continue
        out.append(link)
    return sorted(out, key=lambda k: (k["rect"][1], k["rect"][0]))


def result_links(links):
    """Search-result links: Google puts the result's URL in the link text."""
    return [k for k in links if "http" in k["name"].lower()]


def choose_link(links, human, hint=None, top_n=8):
    """
    Prefer search results over page navigation. With `hint`, take the first
    result containing it; otherwise one of the top results (upper ones likelier).
    Either way the person spends a moment scanning the list first.
    """
    pool = result_links(links) or links
    if not pool:
        return None
    if hint:
        for group in (pool, links):
            matches = [k for k in group if hint.lower() in k["name"].lower()]
            if matches:
                human.visual_search(min(len(pool), top_n))
                return matches[0]
    return human.choose(pool, top_n=top_n)


def reading_params(human):
    """Reading behaviour at this person's reading speed."""
    return reading.ReadingParams(reading_wpm=human.persona.reading_wpm)


def log_trace(words):
    log(f"  traced with the cursor: {' '.join(w['text'] for w in words)[:70]!r}")


def describe_profiles(profiles):
    return "; ".join(f"[{p['index']}] {p['name']}" + (f" ({p['account']})" if p["account"] else "")
                     for p in profiles)


def choose_profile(human, name, index):
    """
    On the profile picker, move to the requested profile card with the learned
    mouse model and click it. With no name given, the first profile is used.
    """
    profiles = browser.list_profiles()
    if not profiles:
        sys.exit("Profile picker is open but no profiles were found")
    log(f"Profile picker shown: {describe_profiles(profiles)}")
    if name:
        target = browser.find_profile(name, index)
        if not target:
            sys.exit(f"No profile matching {name!r} (index {index}). Available: {describe_profiles(profiles)}")
    else:
        target = profiles[0]
        log("No --profile given - using the first one")
    human.visual_search(len(profiles))           # a person scans the cards first
    log(f"Choosing profile [{target['index']}] {target['name']} ({target['account'] or '-'})")
    human.click_rect(target["click_rect"])
    if not browser.wait_for_browser_window(timeout=20):
        sys.exit("Browser window did not open after choosing the profile")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--process", default="search", choices=["search", "word"],
                    help="search = Chrome search demo, word = type a paragraph into a new Word document")
    ap.add_argument("--wpm", type=float, default=None, help="typing speed (default: from the sampled persona)")
    ap.add_argument("--text", default=WORD_PARAGRAPH, help="word demo: text to type")
    ap.add_argument("--text-file", default=None, help="word demo: type the contents of this file instead")
    ap.add_argument("--revisions", default="heuristic", choices=["heuristic", "llm"],
                    help="word demo: where changed wordings come from")
    ap.add_argument("--pause-scale", type=float, default=None,
                    help="thinking-pause length while composing (persona default 0.7; 1.0 = timed-essay writers)")
    ap.add_argument("--query", default="wikipedia")
    ap.add_argument("--link-hint", default=None, help="text the first clicked result should contain (default: query)")
    ap.add_argument("--read-seconds", type=float, default=20)
    ap.add_argument("--no-model", action="store_true", help="use the fallback generator")
    ap.add_argument("--temperature", type=float, default=None, help="mouse path variety (persona default 1.0)")
    ap.add_argument("--shortcut-pref", type=float, default=None,
                    help="0 = always reaches for the mouse, 1 = always keyboard shortcuts (default: sampled)")
    ap.add_argument("--seed", type=int, default=None, help="reproduces the whole run, persona included")
    ap.add_argument("--profile", default=None,
                    help="profile to pick on the 'Who's using Chrome?' screen (profile or account name)")
    ap.add_argument("--profile-index", type=int, default=0,
                    help="which match to use when several profiles share the name (0 = leftmost)")
    ap.add_argument("--list-profiles", action="store_true",
                    help="print the profiles on an open profile picker and exit")
    args = ap.parse_args(argv)
    if args.process == "search":
        if args.list_profiles:
            profiles = browser.list_profiles(timeout=1)
            print(describe_profiles(profiles).replace("; ", "\n") if profiles else "Profile picker is not open.")
            return

        human = make_human(args)

        # 1. Open Chrome
        log("Opening Chrome")
        if not browser.open_chrome(launch=lambda: human.open_app("chrome")):
            sys.exit("Chrome window did not appear")
        time.sleep(1.0)
        if browser.get_profile_picker(timeout=1):
            choose_profile(human, args.profile, args.profile_index)
            time.sleep(1.0)
        browser.maximize_chrome()
        time.sleep(1.0)
        human.think("scan")

        # 2. Address bar: clicked or Ctrl+L, depending on the person
        log("Focusing the address bar")
        if not human.focus_address_bar():
            # Safety: never type unless Chrome's address bar really has keyboard focus
            log("Address bar not focused - refocusing Chrome (Ctrl+L)")
            browser.focus_chrome()
            time.sleep(0.5)
            human.hotkey("ctrl", "l")
            time.sleep(0.3)
            if not browser.address_bar_has_focus():
                sys.exit("Chrome's address bar doesn't have keyboard focus - stopping before typing anything")

        # 3. Search
        log(f"Typing search: {args.query!r}")
        human.select_all_and_type(args.query)      # replace whatever is in the bar
        # Chrome's inline autocomplete can swallow a Backspace (it deletes the highlighted
        # suggestion instead of the typo). If the bar doesn't hold the query, retype it cleanly.
        typed = browser.get_current_url()
        if typed and not typed.startswith(args.query):
            log(f"Address bar shows {typed!r} - retyping the query")
            human.think("confirm")
            human.select_all_and_type(args.query, error_scale=0.0)
        human.think("confirm")
        human.press("enter")
        wait_for_navigation()
        human.think("scan")

        # 4. First result
        page = browser.get_page_rect()
        links = usable_links(page)
        first = choose_link(links, human, hint=args.link_hint or args.query)
        if not first:
            sys.exit("No clickable links found on the results page")
        log(f"Clicking result: {first['name'][:60]!r}")
        human.click_rect(first["rect"])
        wait_for_navigation()

        # 5. Read
        log(f"Reading for ~{args.read_seconds:.0f} s")
        reading.read_page(human.hm, args.read_seconds, browser.get_page_rect(), reading_params(human), log_trace)
        human.think("mental")

        # 6. Back: the toolbar button or Alt+Left, depending on the person
        log("Going back")
        human.go_back()
        wait_for_navigation()
        human.think("scan")

        # 7. A different result
        links = usable_links(browser.get_page_rect(), exclude={first["name"]})
        second = choose_link(links, human)
        if second:
            log(f"Clicking another result: {second['name'][:60]!r}")
            human.click_rect(second["rect"])
            wait_for_navigation()
            reading.read_page(human.hm, args.read_seconds / 2, browser.get_page_rect(),
                              reading_params(human), log_trace)
        else:
            log("No second link found")

        log("Demo finished.")
    else:
        run_word_demo(make_human(args), args)


def make_human(args):
    """The simulated person for this run (persona sampled from --seed, CLI overrides win)."""
    human = Human(seed=args.seed, use_model=not args.no_model, wpm=args.wpm, pause_scale=args.pause_scale,
                  mouse_temperature=args.temperature, shortcut_pref=args.shortcut_pref)
    log(f"Persona: {human.persona.describe()}")
    log(f"Mouse backend: {human.hm.backend} | click timing: {human.hm.clicks.summary()}")
    log("Starting in 3 s - don't touch the mouse/keyboard.")
    time.sleep(3)
    return human


# --- Word demo ---------------------------------------------------------------------------

WORD_PARAGRAPH = (
    "Over the past quarter, our team has focused on improving the reliability of the scheduling "
    "system and reducing the time it takes to onboard new customers. Most of the delays we saw "
    "earlier in the year came from manual data checks, so we introduced a simple validation step "
    "that catches missing fields before a request reaches the operations team. As a result, the "
    "average onboarding time dropped from nine days to just under five, and the number of support "
    "tickets related to incorrect records fell by roughly a third. There is still work to do, "
    "especially around reporting, where several managers have asked for clearer weekly summaries. "
    "For the next quarter, I would suggest that we prioritise the reporting dashboard, review the "
    "remaining manual steps with the operations leads, and set up a short feedback session with "
    "two or three of our largest customers to understand what they would like to see improved."
)

# Word's AutoCorrect turns straight quotes into curly ones; compare text without them
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


WORD_WINDOW_CLASS = "OpusApp"


def word_window_handles():
    """Handles of every open Word window (any state: minimised, Start screen, document)."""
    return {w.NativeWindowHandle for w in auto.GetRootControl().GetChildren()
            if w.ClassName == WORD_WINDOW_CLASS}


def foreground_word_window(timeout=30, ignore=()):
    """
    The Word window that is in front, as a UIA control (None on timeout).
    ignore: handles to skip - pass the Word windows that were already open so only
            the newly launched one is accepted.

    Window titles can't tell Word's states apart: with another document open, the
    Start screen of a new window is already titled "Document2 - Word".
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            win = auto.GetForegroundControl().GetTopLevelControl()
        except Exception:
            win = None
        if win is not None and win.ClassName == WORD_WINDOW_CLASS and win.NativeWindowHandle not in ignore:
            return win
        time.sleep(0.25)
    return None


def word_blank_document_tile(win, timeout=4):
    """The 'Blank document' tile if `win` is showing Word's Start screen, else None."""
    tile = auto.ListItemControl(searchFromControl=win, searchDepth=10, Name="Blank document")
    if tile.Exists(maxSearchSeconds=timeout, searchIntervalSeconds=0.25) and ui_elements.is_visible(tile):
        return tile
    return None


def word_document(win=None, timeout=10):
    """(window, editing surface) of a Word window (default: the one in front). The
    surface is a DocumentControl named after the document, e.g. 'Document1'."""
    win = win or foreground_word_window(timeout)
    if not win:
        return None, None
    doc = auto.DocumentControl(searchFromControl=win)
    if not doc.Exists(maxSearchSeconds=timeout, searchIntervalSeconds=0.25):
        return win, None
    return win, doc


def word_document_has_focus(doc):
    """True if keyboard input would go into this Word document."""
    focused = auto.GetFocusedControl()
    if focused is None or doc is None:
        return False
    if foreground_word_window(timeout=0.5) is None:
        return False
    return focused.ControlType == auto.ControlType.DocumentControl or focused.ClassName == "_WwG"


def word_document_text(doc):
    """Current text of the document (Word ends paragraphs with '\\r'), or None."""
    try:
        return doc.GetTextPattern().DocumentRange.GetText(-1)
    except Exception:
        return None


def run_word_demo(human, args):
    """
    Open Word, start a blank document, click into the page with the learned mouse
    model and type a paragraph with the learned typing model (typos and their
    corrections, thinking pauses, changes of wording, going back to fix things).
    """
    text = args.text
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read().strip()

    # 1. Word + blank document
    already_open = word_window_handles()       # never type into a document that was already open
    log("Opening Word")
    human.open_app("word")
    win = foreground_word_window(timeout=30, ignore=already_open)
    if win is None:
        sys.exit("A new Word window did not appear")
    time.sleep(2.0)                            # Word's Start screen needs a moment to settle
    tile = word_blank_document_tile(win)
    if tile is not None:                       # Start screen: click "Blank document"
        log("Clicking 'Blank document'")
        human.think("scan")                    # a person glances over the Start screen first
        human.click_element(tile)
        deadline = time.time() + 20
        while word_blank_document_tile(win, timeout=0) is not None:
            if time.time() > deadline:
                log("Start screen still showing - pressing Enter")
                office.new_blank_from_start_screen()
                break
            time.sleep(0.25)
    else:
        log("No Start screen - Word opened straight into a document")
    time.sleep(1.0)
    ctypes.windll.user32.ShowWindow(win.NativeWindowHandle, 3)     # SW_MAXIMIZE, this window only
    time.sleep(1.0)

    # 2. Click into the page, like a person would, then make sure typing goes there
    win, doc = word_document(win)
    if doc is None:
        sys.exit("Couldn't find the Word document area")
    l, t, r, b = ui_elements.element_rect(doc)
    top_of_page = (l + (r - l) * 0.25, t + 40, l + (r - l) * 0.75, t + min(160, (b - t) * 0.25))
    log("Clicking into the document")
    human.think("scan")
    human.click_field(top_of_page)
    human.think("glance")
    human.hotkey("ctrl", "end")                # caret at the end of the (empty) document
    if not word_document_has_focus(doc):
        log("Document not focused after click - activating Word")
        win.SetActive()
        time.sleep(0.5)
        if not word_document_has_focus(doc):
            sys.exit("The Word document doesn't have keyboard focus - stopping before typing anything")

    # 3. Type
    words = len(text.split())
    log(f"Typing {words} words at a {human.persona.wpm:.0f} wpm persona (revisions: {args.revisions})")
    t0 = time.monotonic()
    plan = human.type(text, mode="compose", revisions=args.revisions, max_pause=4.0)
    took = time.monotonic() - t0
    tags = {tag: sum(1 for k in plan if k.tag == tag) for tag in ("typo", "fix", "false_start", "lost", "nav")}
    log(f"Typed in {took:.0f} s = {len(text) / 5 / (took / 60):.0f} wpm effective "
        f"({len(plan)} keystrokes; typos {tags['typo']}, correction keys {tags['fix']}, "
        f"false-start keys {tags['false_start']}, lost-thought deletions {tags['lost']}, "
        f"arrow keys for revisions {tags['nav']})")

    # 4. Check what landed in the document
    time.sleep(0.5)
    got = word_document_text(doc)
    if got is None:
        log("Couldn't read the document text back to check it")
    elif got.strip().translate(_QUOTES) == text.strip():
        log("Document text matches exactly")
    else:
        log("Document text differs from the intended text (Word AutoCorrect or a missed key):")
        log(f"  got: {got.strip()[:300]!r}")
    log("Word demo finished (document left open, unsaved).")


if __name__ == "__main__":
    main()
