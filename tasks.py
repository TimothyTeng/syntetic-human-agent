"""
Desktop tasks performed by a simulated person (algorithms.behaviour.Human).

    Chrome   open_browser(), search_and_read() (results in place or in new tabs,
             occasional slips: wrong result, tab closed too early)
    Writer   open_writer_document(), click_into_document(), type_into_document()
             (OpenOffice Writer)
    Excel    open_office_blank(..., "Blank workbook"), focus_sheet(), goto_cell(), type_rows()
    Notepad  open_notepad_note() - always on an empty page, never in an existing note
    Session  Workspace + session_activities(): the same tasks as repeatable
             activities for algorithms.session.Session (the 15-minute run). The
             workspace remembers the windows it opened, so later activities switch
             back to them (Alt+Tab when it is the previous window) instead of opening
             new ones.

Tasks raise TaskError when the UI isn't in the expected state. They never type unless
the target field is confirmed to have keyboard focus.
"""

import ctypes
import os
import re
import time

import uiautomation as auto

from algorithms import reading
from algorithms.mouse_model import scroll_stats
from algorithms.session import Activity, TaskError
from controller import apps, browser, office, ui_elements
from controller import writer as W

T0 = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - T0:6.1f}s] {msg}", flush=True)


# --- Chrome --------------------------------------------------------------------------------

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


# Only results on these sites (or their subdomains) are ever clicked. Search results
# also contain junk pages that copy popular search terms (spam, scams, malware); an
# unattended agent must not wander onto those. Extend the list as needed.
ALLOWED_DOMAINS = (
    "wikipedia.org", "wikimedia.org", "britannica.com", "nationalgeographic.com", "smithsonianmag.com",
    "khanacademy.org", "bbc.co.uk", "bbc.com", "nasa.gov", "nih.gov", "noaa.gov", "history.com",
    "microsoft.com", "google.com", "office.com", "mayoclinic.org", "investopedia.com", "hbr.org",
    "atlassian.com", "asana.com", "mindtools.com", "coursera.org", "w3schools.com", "python.org",
    "realpython.com", "geeksforgeeks.org", "unesco.org", "who.int", "un.org",
)
_URL_HOST = re.compile(r"https?://([A-Za-z0-9.-]+)")


def link_domain(link):
    """Host name of a search result (Google shows the URL in the result's link text), or None."""
    m = _URL_HOST.search(link["name"])
    return m.group(1).lower() if m else None


def is_allowed(link):
    host = link_domain(link)
    return bool(host) and any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)


def result_links(links):
    """Search results on allowed sites (Google puts each result's URL in its link text)."""
    return [k for k in links if is_allowed(k)]


def choose_link(links, human, hint=None, top_n=8):
    """
    Pick a search result on an allowed site (ALLOWED_DOMAINS). With `hint`, take the
    first one containing it; otherwise one of the top ones (upper ones likelier). Either
    way the person spends a moment scanning the list first. None if no allowed result
    is on the page - then nothing is clicked.
    """
    pool = result_links(links)
    if not pool:
        return None
    if hint:
        matches = [k for k in pool if hint.lower() in k["name"].lower()]
        if matches:
            human.visual_search(min(len(pool), top_n))
            return matches[0]
    return human.choose(pool, top_n=top_n)


def reading_params(human):
    """Reading behaviour for this person: their reading speed, and keyboard-minded
    people sometimes scroll with PgDn / arrow keys instead of the wheel."""
    return reading.ReadingParams(reading_wpm=human.persona.reading_wpm,
                                 p_key_scroll=0.5 * human.persona.shortcut_pref,
                                 scroll=scroll_stats.load_default().for_profile(human.persona.scroll_profile))


def read(human, seconds, on_view=None):
    """Read the current page for about `seconds` (it may end early at the bottom of the page).
    on_view: see reading.read_page (research notes record which part of the page was seen)."""
    reading.read_page(human.hm, seconds, browser.get_page_rect(), reading_params(human), log_trace, human=human,
                      on_view=on_view)


def page_reading_time(human, max_seconds, default=30.0):
    """How long this person would read the current page (from its amount of text), capped."""
    secs = reading.reading_time(human.rng, reading_params(human), lo=min(10.0, max_seconds), hi=max_seconds)
    state = browser.page_scroll_state()
    log(f"Page length: ~{browser.page_total_words()} words | scroll position "
        + (f"{state[0]:.0f}%, {state[1]:.0f}% of the page in view" if state else "not reported"))
    return secs if secs is not None else min(default, max_seconds)


def wait_for_page_change(before_title, timeout=6.0):
    """Wait until the active tab's title differs from `before_title`; True if it did."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        title = browser.get_page_title()
        if title and title != before_title:
            wait_for_navigation(settle=0.3)
            return True
        time.sleep(0.25)
    return False


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
        raise TaskError("Profile picker is open but no profiles were found")
    log(f"Profile picker shown: {describe_profiles(profiles)}")
    if name:
        target = browser.find_profile(name, index)
        if not target:
            raise TaskError(f"No profile matching {name!r} (index {index}). Available: {describe_profiles(profiles)}")
    else:
        target = profiles[0]
        log("No --profile given - using the first one")
    human.visual_search(len(profiles))           # a person scans the cards first
    log(f"Choosing profile [{target['index']}] {target['name']} ({target['account'] or '-'})")
    human.click_rect(target["click_rect"])
    if not browser.wait_for_browser_window(timeout=20):
        raise TaskError("Browser window did not open after choosing the profile")


def open_browser(human, profile=None, profile_index=0):
    """Open a new Chrome window the way a person does (picking a profile if asked).
    Returns its window handle."""
    log("Opening Chrome")
    if not browser.open_chrome(launch=lambda: human.open_app("chrome")):
        raise TaskError("Chrome window did not appear")
    time.sleep(1.0)
    if browser.get_profile_picker(timeout=1):
        choose_profile(human, profile, profile_index)
        time.sleep(1.0)
    win = browser.get_chrome_window()
    if win is None:
        raise TaskError("No Chrome browser window after opening Chrome")
    human.place_new_window(win.NativeWindowHandle)
    time.sleep(0.5)
    human.think("scan")
    return win.NativeWindowHandle


P_SUGGESTION = 0.3       # type part of a multi-word query and pick one of Chrome's suggestions


def search(human, query, allow_suggestion=False):
    """
    Focus the address bar (click or Ctrl+L), type the query, Enter.
    allow_suggestion: sometimes type only the first word(s) and pick one of the
                      address bar's suggestions with the arrow keys instead - then the
                      search that runs is Chrome's suggestion, not exactly `query`.
    """
    log("Focusing the address bar")
    if not human.focus_address_bar():
        # Safety: never type unless Chrome's address bar really has keyboard focus
        log("Address bar not focused - refocusing Chrome (Ctrl+L)")
        browser.focus_chrome()
        time.sleep(0.5)
        human.hotkey("ctrl", "l")
        time.sleep(0.3)
        if not browser.address_bar_has_focus():
            raise TaskError("Chrome's address bar doesn't have keyboard focus - stopping before typing anything")

    words = query.split()
    suggest = allow_suggestion and len(words) >= 2 and human.rng.random() < P_SUGGESTION
    text = " ".join(words[:int(human.rng.integers(1, len(words)))]) + " " if suggest else query
    log(f"Typing search: {text!r}" + (" (then picking a suggestion)" if suggest else ""))
    human.select_all_and_type(text)           # replace whatever is in the bar
    # Chrome's inline autocomplete can swallow a Backspace (it deletes the highlighted
    # suggestion instead of the typo). If the bar doesn't hold the text, retype it cleanly.
    typed = browser.get_current_url()
    if typed and not typed.startswith(text.rstrip()):
        log(f"Address bar shows {typed!r} - retyping")
        human.think("confirm")
        human.select_all_and_type(text, error_scale=0.0)
    if suggest:
        human.think("scan", 0.8)              # read the suggestion list
        human.press("down", presses=int(human.rng.integers(1, 3)))
    human.think("confirm")
    human.press("enter")
    wait_for_navigation()
    if suggest:
        log(f"Searched for the suggestion: {browser.get_page_title()[:70]!r}")
    human.think("scan")


# Base rates of slips (per opportunity), multiplied by the persona's error_scale
P_WRONG_RESULT = 0.06    # click the result next to the intended one, notice, go back
P_EARLY_CLOSE = 0.06     # close a result's tab while still reading it, reopen it (Ctrl+Shift+T)


def open_result(human, link):
    """
    Open a search result the way this person does: a plain click, or (new-tab habit)
    Ctrl+click and then switch to the new tab with Ctrl+Tab or by clicking it.
    Returns True if the result is in a new tab.
    """
    if human.rng.random() >= human.persona.new_tab_pref:
        results_title = browser.get_page_title()
        human.click_element(link["control"])          # re-aims if the results page shifted
        if not wait_for_page_change(results_title):
            # Nothing happened - e.g. the click only closed a pop-up covering the page.
            # A person notices and clicks the result again.
            log("The click didn't open the result - clicking it again")
            human.think("scan", 0.6)
            again = find_again(link["name"], browser.get_page_rect())
            if again is None:
                raise TaskError("The result is no longer on the page")
            human.click_element(again["control"])
            if not wait_for_page_change(results_title):
                raise TaskError("Clicking the result didn't open it - something on the page is in the way")
        log(f"Opened: {browser.get_page_title()[:70]!r}")
        return False
    before = len(browser.tab_items())
    human.click_element(link["control"], modifiers=("ctrl",))
    deadline = time.time() + 2.5                      # Chrome can take a moment to add the tab
    while True:
        time.sleep(0.4)
        tabs = browser.tab_items()
        if len(tabs) > before or time.time() >= deadline:
            break
    if len(tabs) <= before:                           # no new tab appeared: it opened in place
        wait_for_navigation()
        return False
    human.think("glance")
    cur = browser.selected_tab_index(tabs)
    if human.prefers_keyboard() or cur is None or cur + 1 >= len(tabs):
        log("Switching to the new tab (Ctrl+Tab)")
        human.hotkey("ctrl", "tab")
    else:
        log("Switching to the new tab (clicking it)")
        human.click_element(tabs[cur + 1])
    wait_for_navigation()
    return True


def close_tab(human):
    """Close the active tab (Ctrl+W or its x button). Refuses if it is the window's last tab,
    since that would close the whole window. Returns True if closed."""
    tabs = browser.tab_items()
    if len(tabs) < 2:
        return False
    cur = browser.selected_tab_index(tabs)
    button = None if human.prefers_keyboard() or cur is None else browser.tab_close_button(tabs[cur])
    if button is not None and ui_elements.is_visible(button):
        human.click_element(button)
    else:
        human.hotkey("ctrl", "w")
    time.sleep(0.5)
    return True


def leave_result(human, new_tab):
    """Done with a result: close its tab, or go back to the results page."""
    if new_tab and close_tab(human):
        log("Closing the result's tab")
    else:
        log("Going back")                             # the Back button or Alt+Left
        human.go_back()
    wait_for_navigation()


def read_result(human, seconds, new_tab, on_page=None, on_view=None):
    """Read the opened result. In a new tab, the person occasionally closes it too early,
    realises, and reopens it with Ctrl+Shift+T.
    on_page: called once the page is open, before reading (research: take notes from it);
    on_view: passed to read()."""
    if on_page:
        on_page()
    log(f"Reading for ~{seconds:.0f} s")
    if new_tab and human.rng.random() < P_EARLY_CLOSE * human.error_level:
        early = float(human.rng.uniform(2, min(8, seconds)))
        read(human, early, on_view)
        if close_tab(human):
            log("Slip: closed the tab too early - reopening it with Ctrl+Shift+T")
            human.think("confirm", 2.5)               # "wait, I wasn't done"
            human.hotkey("ctrl", "shift", "t")
            wait_for_navigation()
            human.think("glance")
        seconds = max(3.0, seconds - early)
    read(human, seconds, on_view)


def maybe_click_wrong_result(human, links, intended):
    """
    Slip: sometimes the person clicks the result next to the one they meant, sees the
    wrong page, and goes back. Returns True if it happened (the results page is shown again).
    """
    pool = result_links(links)                        # only ever slip onto allowed sites too
    if intended not in pool or human.rng.random() >= P_WRONG_RESULT * human.error_level:
        return False
    i = pool.index(intended)
    neighbours = [pool[j] for j in (i - 1, i + 1) if 0 <= j < len(pool)]
    if not neighbours:
        return False
    wrong = neighbours[int(human.rng.integers(len(neighbours)))]
    log(f"Slip: clicked the neighbouring result {wrong['name'][:50]!r}")
    human.click_element(wrong["control"])
    wait_for_navigation()
    human.think("scan", 1.6)                          # realise it's not the page they wanted
    log("Going back to the results")
    human.go_back()
    wait_for_navigation()
    human.think("scan")
    return True


def find_again(name, page_rect=None):
    """The visible link with this exact text (after the page was reloaded), or None."""
    return next((k for k in usable_links(page_rect) if k["name"] == name), None)


def visit_result(human, links, link, read_seconds=None, max_read=90.0, on_page=None, on_view=None):
    """
    Open `link` (sometimes after clicking its neighbour by mistake) and read it.
    read_seconds: None = from the page's length, at most max_read.
    on_page / on_view: see read_result().
    Returns (link actually read, whether it is in a new tab).
    """
    if maybe_click_wrong_result(human, links, link):
        link = find_again(link["name"], browser.get_page_rect())
        if link is None:
            raise TaskError("The intended result is no longer on the results page")
    log(f"Clicking result: {link['name'][:60]!r}")
    new_tab = open_result(human, link)
    if on_page:                       # first (it waits for a slow page), then size the reading time
        on_page()
    read_result(human, read_seconds or page_reading_time(human, max_read), new_tab, on_view=on_view)
    return link, new_tab


def search_and_read(human, query, read_seconds=None, link_hint=None, second_result=True, max_read=90.0,
                    allow_suggestion=False, on_page=None, on_view=None):
    """
    Search, open a result, read it, return to the results and (optionally) read a second
    result. Results open in place or in a new tab (persona habit); slips happen now and then.
    read_seconds: time on the first result; None = from the page's length (at most
                  max_read). The second result gets about half as long.
    link_hint: text the first clicked result should contain (None = any top result).
    allow_suggestion: see search().
    on_page / on_view: see read_result() - called for every result read.
    """
    search(human, query, allow_suggestion and not link_hint)

    links = usable_links(browser.get_page_rect())
    first = choose_link(links, human, hint=link_hint)
    if not first:
        raise TaskError("No search result on an allowed site (see ALLOWED_DOMAINS) - not clicking anything")
    first, new_tab = visit_result(human, links, first, read_seconds, max_read, on_page, on_view)
    if not second_result:
        if new_tab:                                   # don't leave tabs piling up
            leave_result(human, new_tab)
        return
    human.think("mental")
    leave_result(human, new_tab)
    human.think("scan")

    links = usable_links(browser.get_page_rect(), exclude={first["name"]})
    second = choose_link(links, human)
    if not second:
        log("No other result on an allowed site - not opening a second one")
        return
    _, new_tab = visit_result(human, links, second, read_seconds and read_seconds / 2, max_read / 2, on_page,
                              on_view)
    if new_tab:
        leave_result(human, new_tab)


# --- Writer --------------------------------------------------------------------------------

WRITER_PARAGRAPH = (
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

WRITER_APP = "openoffice writer"           # its Start-menu name
EXCEL_WINDOW_CLASS = "XLMAIN"
NOTEPAD_WINDOW_CLASS = "Notepad"


def app_window_handles(window_class):
    """Handles of every open top-level window of this class (visible or minimised)."""
    return set(apps.top_level_windows(window_class))


def foreground_app_window(window_class, timeout=30, ignore=()):
    """
    The window of this class that is in front, as a UIA control (None on timeout).
    ignore: handles to skip - pass the windows that were already open so only a newly
            launched one is accepted.
    """
    deadline = time.time() + timeout
    while True:
        hwnd = apps.foreground_top_level()
        if hwnd and apps.window_class(hwnd) == window_class and hwnd not in ignore:
            win = apps.control_for(hwnd)
            if win is not None:
                return win
        if time.time() >= deadline:
            return None
        time.sleep(0.25)


def start_screen_tile(win, name, timeout=4):
    """An Office Start-screen tile ('Blank document', 'Blank workbook') if shown, else None."""
    tile = auto.ListItemControl(searchFromControl=win, searchDepth=10, Name=name)
    if tile.Exists(maxSearchSeconds=timeout, searchIntervalSeconds=0.25) and ui_elements.is_visible(tile):
        return tile
    return None


def open_office_blank(human, app, window_class, tile_name):
    """
    Open an Office app from the Start menu and start a blank file by clicking its
    Start-screen tile. Returns the NEW window (files already open are never touched).
    """
    already_open = app_window_handles(window_class)
    log(f"Opening {app.capitalize()}")
    human.open_app(app)
    win = foreground_app_window(window_class, timeout=12, ignore=already_open)
    if win is None:
        # Launching an Office app that already shows its Start screen just brings that
        # window back. It holds no document, so it is safe to use; any other window that
        # was already open (it may hold someone's work) is never used.
        fg = foreground_app_window(window_class, timeout=18)
        if fg is not None and start_screen_tile(fg, tile_name, timeout=1) is not None:
            log(f"Using the {app.capitalize()} window that was waiting on its Start screen")
            win = fg
    if win is None:
        raise TaskError(f"A new {app.capitalize()} window did not appear")
    time.sleep(2.0)                            # the Start screen needs a moment to settle
    tile = start_screen_tile(win, tile_name)
    if tile is not None:
        log(f"Clicking {tile_name!r}")
        human.think("scan")                    # a person glances over the Start screen first
        human.click_element(tile)
        deadline = time.time() + 20
        while start_screen_tile(win, tile_name, timeout=0) is not None:
            if time.time() > deadline:
                log("Start screen still showing - pressing Enter")
                office.new_blank_from_start_screen()
                break
            time.sleep(0.25)
    else:
        log(f"No Start screen - {app.capitalize()} opened straight into a file")
    time.sleep(1.0)
    human.place_new_window(win.NativeWindowHandle)
    time.sleep(0.5)
    return win


# Writer ------------------------------------------------------------------------------

def foreground_writer_window(timeout=30, ignore=()):
    """
    The OpenOffice Writer window that is in front, as a UIA control (None on timeout).
    ignore: handles to skip - pass the windows that were already open so only a newly
            opened one is accepted. (All OpenOffice windows share one window class, so the
            title tells Writer from Calc or the Start Center.)
    """
    deadline = time.time() + timeout
    while True:
        hwnd = apps.foreground_top_level()
        if hwnd and W.is_writer_window(hwnd) and hwnd not in ignore:
            win = apps.control_for(hwnd)
            if win is not None:
                return win
        if time.time() >= deadline:
            return None
        time.sleep(0.25)


def writer_document(win=None, timeout=10):
    """(window, editing surface) of a Writer window (default: the one in front). The
    surface is a DocumentControl named after the document, e.g. 'Untitled 1 - OpenOffice
    Document'."""
    win = win or foreground_writer_window(timeout)
    if not win:
        return None, None
    return win, W.document_control(win, timeout)


def open_writer_document(human):
    """
    Open OpenOffice Writer from the Start menu; it starts on a new blank document
    ('Untitled N'). Returns (window, document control) of the NEW window - documents that
    were already open are never touched.
    """
    already_open = set(W.writer_windows())
    log("Opening OpenOffice Writer")
    human.open_app(WRITER_APP)
    win = foreground_writer_window(timeout=40, ignore=already_open)     # the first start is slow
    if win is None:
        raise TaskError("A new OpenOffice Writer window did not appear")
    time.sleep(1.5)                            # the toolbars and the page need a moment to settle
    human.place_new_window(win.NativeWindowHandle)
    time.sleep(0.5)
    win, doc = writer_document(win)
    if doc is None:
        raise TaskError("Couldn't find the Writer document area")
    return win, doc


def reopen_writer_document(human, path):
    """
    Open a saved document to carry on writing it: Writer from the Start menu, then the Open
    dialog (Ctrl+O), the file's path, Enter. Writer replaces the untouched blank document
    with the file. Returns (window, document control).
    """
    open_writer_document(human)
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    log(f"Opening {path}")
    human.think("mental")
    human.hotkey("ctrl", "o")
    time.sleep(1.5)
    human.think("scan", 0.6)
    human.select_all_and_type(os.path.abspath(path))
    human.think("confirm")
    human.press("enter")
    deadline = time.time() + 20
    while time.time() < deadline:
        win = foreground_writer_window(timeout=1)
        if win is not None and stem in apps.window_title(win.NativeWindowHandle).lower():
            time.sleep(1.0)
            win, doc = writer_document(win)
            if doc is not None:
                return win, doc
        time.sleep(0.3)
    raise TaskError(f"Couldn't open {path} in Writer")


def click_into_document(human, win, doc):
    """Click into the page like a person, put the caret at the end, and make sure
    typing will go into this document (TaskError otherwise). The click goes near the top
    of the page (Writer's document area also holds the grey desk around the page)."""
    doc = W.document_control(win, timeout=2) or doc        # the element is replaced when the file is saved
    l, t, r, b = ui_elements.element_rect(doc)
    top_of_page = (l + (r - l) * 0.3, t + 60, l + (r - l) * 0.7, t + min(180, (b - t) * 0.3))
    log("Clicking into the document")
    human.think("scan")
    human.click_field(top_of_page)
    human.think("glance")
    human.hotkey("ctrl", "end")                # caret at the end of the document
    time.sleep(0.3)
    if not W.document_has_focus(win, doc):
        log("Document not focused after click - activating Writer")
        win.SetActive()
        time.sleep(0.5)
        if not W.document_has_focus(win, doc):
            raise TaskError("The Writer document doesn't have keyboard focus - stopping before typing anything")


def type_into_document(human, doc, text, revisions="heuristic", mode="compose"):
    """
    Type `text` at the caret with the learned typing model, then check that the
    document now ends with it. Returns the keystroke plan.
    """
    words = len(text.split())
    log(f"Typing {words} words at a {human.persona.wpm:.0f} wpm persona (revisions: {revisions})")
    t0 = time.monotonic()
    plan = human.type(text, mode=mode, revisions=revisions, max_pause=4.0)
    took = time.monotonic() - t0
    tags = {tag: sum(1 for k in plan if k.tag == tag) for tag in ("typo", "fix", "false_start", "lost", "nav")}
    log(f"Typed in {took:.0f} s = {len(text) / 5 / (took / 60):.0f} wpm effective "
        f"({len(plan)} keystrokes; typos {tags['typo']}, correction keys {tags['fix']}, "
        f"false-start keys {tags['false_start']}, lost-thought deletions {tags['lost']}, "
        f"arrow keys for revisions {tags['nav']})")
    time.sleep(0.5)
    got = W.document_text(doc)
    if got is None:
        log("Couldn't read the document text back to check it")
    elif W.normalise(got).endswith(W.normalise(text)):
        log("Document text matches what was typed")
    else:
        log("Document text differs from the intended text (AutoCorrect, word completion or a missed key):")
        log(f"  got: {got.strip()[-300:]!r}")
    return plan


# --- Excel ---------------------------------------------------------------------------------

def excel_sheet_has_focus(win):
    """True if `win` is in front and keyboard input goes to its worksheet grid."""
    if apps.foreground_window() != win.NativeWindowHandle:
        return False
    try:
        f = auto.GetFocusedControl()
    except Exception:
        return False
    return f is not None and (f.ClassName == "EXCEL7" or f.ControlType == auto.ControlType.DataItemControl)


def focus_sheet(human, win):
    """Make sure typing goes into the worksheet: click into the grid if needed (TaskError if it won't)."""
    if excel_sheet_has_focus(win):
        return
    grid = auto.PaneControl(searchFromControl=win, ClassName="EXCEL7")
    if not grid.Exists(3, 0.25):
        raise TaskError("Couldn't find the Excel worksheet")
    l, t, r, b = ui_elements.element_rect(grid)
    log("Clicking into the worksheet")
    human.click_rect((l + 80, t + 40, min(r, l + 600), min(b, t + 300)))
    if not excel_sheet_has_focus(win):
        raise TaskError("The Excel worksheet doesn't have keyboard focus - stopping before typing anything")


def goto_cell(human, win, ref):
    """Select a cell: type it into the Name Box (mouse people) or Go To with Ctrl+G."""
    box = None if human.prefers_keyboard() else ui_elements.find_element(win, "Name Box", timeout=1)
    if box is not None and ui_elements.is_visible(box):
        human.click_element(box)
    else:
        human.hotkey("ctrl", "g")
        time.sleep(0.5)
    human.think("glance")
    human.type_short(ref)
    human.press("enter")
    time.sleep(0.3)
    if not excel_sheet_has_focus(win):
        raise TaskError(f"Excel didn't return to the worksheet after going to {ref}")


def type_rows(human, rows):
    """Type rows of cells: Tab between cells, Enter at the end of a row (Excel then
    returns to the row's first column)."""
    for row in rows:
        for j, value in enumerate(row):
            human.type_short(str(value))
            human.press("tab" if j + 1 < len(row) else "enter")
        human.think("glance")


EXPENSE_HEADER = ["Date", "Item", "Category", "Amount"]
EXPENSES = [            # no item is the start of another (Excel's AutoComplete would finish it)
    ["2026-09-01", "Taxi to client office", "Travel", "18.40"],
    ["2026-09-02", "Printer paper", "Supplies", "24.90"],
    ["2026-09-03", "Team lunch", "Meals", "86.00"],
    ["2026-09-07", "Train tickets", "Travel", "32.60"],
    ["2026-09-09", "Conference fee", "Training", "250.00"],
    ["2026-09-10", "Coffee beans", "Supplies", "15.75"],
    ["2026-09-14", "Software licence", "IT", "49.00"],
    ["2026-09-15", "Courier delivery", "Postage", "12.30"],
    ["2026-09-18", "Parking", "Travel", "9.00"],
    ["2026-09-21", "Office plants", "Supplies", "38.50"],
    ["2026-09-24", "Client dinner", "Meals", "142.80"],
    ["2026-09-28", "Laptop stand", "IT", "35.99"],
]


# --- Notepad -------------------------------------------------------------------------------

def notepad_text(ctrl):
    """Text of a Notepad editor control (TextPattern, or ValuePattern for classic Notepad), or None."""
    try:
        return ctrl.GetTextPattern().DocumentRange.GetText(-1)
    except Exception:
        pass
    try:
        return ctrl.GetValuePattern().Value
    except Exception:
        return None


def notepad_editor_with_focus(win):
    """The focused editor control if `win` is in front and its editor has focus, else None."""
    if apps.foreground_window() != win.NativeWindowHandle:
        return None
    try:
        f = auto.GetFocusedControl()
    except Exception:
        return None
    if f is None or f.ControlType not in (auto.ControlType.DocumentControl, auto.ControlType.EditControl):
        return None
    return f


def notepad_editor_rect(win):
    for ctrl_type in (auto.DocumentControl, auto.EditControl):
        ed = ctrl_type(searchFromControl=win)
        if ed.Exists(2, 0.25) and ui_elements.is_visible(ed):
            return ui_elements.element_rect(ed)
    return None


def click_into_notepad(human, win):
    """Click into the editor (if it doesn't have focus) and return the focused editor."""
    ed = notepad_editor_with_focus(win)
    if ed is None:
        rect = notepad_editor_rect(win)
        if rect is None:
            raise TaskError("Couldn't find Notepad's editor")
        l, t, r, b = rect
        log("Clicking into Notepad")
        human.click_rect((l + 40, t + 30, l + (r - l) // 2, t + min(200, (b - t) // 2)))
        ed = notepad_editor_with_focus(win)
    if ed is None:
        raise TaskError("Notepad's editor doesn't have keyboard focus - stopping before typing anything")
    return ed


def open_notepad_note(human):
    """
    Open Notepad and get an EMPTY page to write on. Windows 11 Notepad restores the
    previous session's tabs, so if the page it opens on has text, a new tab (Ctrl+N)
    is used instead - existing notes are never typed into. Returns the window.
    """
    log("Opening Notepad")
    human.open_app("notepad")
    win = foreground_app_window(NOTEPAD_WINDOW_CLASS, timeout=15)
    if win is None:
        raise TaskError("Notepad did not come to the front")
    time.sleep(1.0)
    human.place_new_window(win.NativeWindowHandle)
    ed = click_into_notepad(human, win)
    if notepad_text(ed):
        log("Notepad opened on an existing note - starting a new tab")
        human.hotkey("ctrl", "n")
        time.sleep(1.0)
        try:
            ed = click_into_notepad(human, win)
        except TaskError:
            human.press("esc")                 # classic Notepad asks "Save changes?" - cancel it
            raise
        if notepad_text(ed) != "":
            raise TaskError("Couldn't get an empty Notepad page - not typing into an existing note")
    return win


NOTES = [
    "Notes from the planning meeting\n- move the weekly check-in to Thursday\n"
    "- ask finance about the Q4 budget\n- send the draft agenda by Friday\n",
    "To do this week\n- review the onboarding checklist\n- book a room for the workshop\n"
    "- update the project timeline\n",
    "Ideas for the newsletter\n- short interview with the support team\n"
    "- tips for using the new expense tool\n- photos from the team lunch\n",
    "Questions for the vendor call\n- what is the expected uptime\n"
    "- how are support tickets prioritised\n- is there a discount for a two year contract\n",
]


# --- Session activities ----------------------------------------------------------------------

QUERIES = [            # mostly long Wikipedia articles: plenty to read and scroll through
    "history of the internet wikipedia", "industrial revolution wikipedia", "roman empire wikipedia",
    "photosynthesis wikipedia", "great barrier reef wikipedia", "apollo program wikipedia",
    "printing press wikipedia", "renaissance wikipedia", "how to make a pivot table in excel",
    "how to write a project status report", "time management tips", "what is a gantt chart",
]

WRITING = [
    WRITER_PARAGRAPH,
    "The main risk for the next release is the dependency on the new billing provider. Their "
    "sandbox has been unstable for the last two weeks, which has slowed down our integration "
    "testing. We have asked for a dedicated test account and expect an answer by Friday. If it "
    "does not arrive in time, we can still ship the release with the current provider and switch "
    "over in the following sprint.",
    "Thank you to everyone who joined the planning session on Tuesday. We agreed to keep the "
    "weekly check-in short and to move detailed technical discussions into separate meetings. "
    "Action items have been added to the shared tracker, and each owner should update the status "
    "of their items before the next check-in. Please let me know if anything is missing.",
]


def _sentences(paragraph):
    out, cur = [], []
    for word in paragraph.split():
        cur.append(word)
        if word.endswith((".", "!", "?")):
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


class Workspace:
    """
    State shared by the activities of one session: the windows opened so far (so
    they are reused, not reopened), what has been searched and written, and how to
    get back to a window like a person (Alt+Tab when it is the previous window).
    """

    def __init__(self, human, profile=None, profile_index=0, revisions="heuristic", research=None):
        """research: a research.project.ResearchProject - browse() then searches for its
        topic and takes notes from the pages read, write() types its paper."""
        self.human = human
        self.research = research
        self.profile, self.profile_index = profile, profile_index
        self.revisions = revisions
        self.chrome = None               # window handle
        self.writer_win = self.writer_doc = None
        self.queries = list(QUERIES)
        human.rng.shuffle(self.queries)
        self.writing = [_sentences(p) for p in WRITING]
        self.para, self.sent = 0, 0      # next sentence to write
        self.written_any = False
        self.excel_win = None
        self.excel_row = 1               # next empty worksheet row
        self.expense = 0                 # next expense to enter
        self.notepad = None              # window handle
        self.notepad_text = ""           # everything typed into our note so far
        self.note = 0                    # next note to write

    def switch_to(self, hwnd):
        """Bring a window to the front: Alt+Tab if it is next in the switcher, else directly."""
        if apps.foreground_window() == hwnd:
            return True
        order = apps.switcher_windows()
        if len(order) > 1 and order[1] == hwnd:
            log("Switching window with Alt+Tab")
            self.human.think("mental", 0.6)
            self.human.hotkey("alt", "tab")
            deadline = time.time() + 2
            while time.time() < deadline:
                if apps.foreground_window() == hwnd:
                    time.sleep(0.4)            # the window repaints
                    return True
                time.sleep(0.1)
        log("Bringing the window to the front")
        return apps.bring_to_front(hwnd)

    def back_to(self, hwnd):
        """switch_to() a window this session opened; sometimes the person now maximises it."""
        if not self.switch_to(hwnd):
            return False
        if not ctypes.windll.user32.IsZoomed(hwnd) and self.human.rng.random() < 0.25:
            log("Maximising the window")
            self.human.maximize_window(hwnd)
        return True

    @staticmethod
    def _alive(hwnd):
        return bool(hwnd) and bool(ctypes.windll.user32.IsWindow(hwnd))

    # --- activities --------------------------------------------------------------------

    def browse(self, seconds_left):
        human = self.human
        if self._alive(self.chrome) and self.back_to(self.chrome):
            human.think("scan")
        else:
            self.chrome = open_browser(human, self.profile, self.profile_index)
        if self.research is not None:                # research: the project's next query, notes taken
            self.research_browse(seconds_left)
            return
        query = self.queries.pop(0)
        self.queries.append(query)
        search_and_read(human, query, second_result=human.rng.random() < 0.5,
                        max_read=max(15.0, min(90.0, 0.4 * seconds_left)), allow_suggestion=True)

    def research_browse(self, seconds_left):
        """Search for the research topic and take notes from the results read (a researcher
        reads more closely and longer than a casual browser)."""
        proj = self.research
        query, hint = proj.next_query()
        log(f"Researching: {query!r}")
        try:
            # a second result more often than casual browsing (60%), up to 150 s per page; notes are
            # taken as each page opens (on_page) and what scrolls into view is recorded (on_view)
            search_and_read(self.human, query, link_hint=hint, second_result=self.human.rng.random() < 0.6,
                            max_read=max(20.0, min(150.0, 0.5 * seconds_left)),
                            on_page=proj.on_page, on_view=proj.on_view)
        finally:
            proj.save_notes()                         # keep what was read even if the search failed midway

    def glance_at_source(self, seconds):
        """While writing: switch back to the browser and re-read a little of the open page,
        then return to Writer. False if there is no browser window to go back to."""
        if not (self._alive(self.chrome) and self.back_to(self.chrome)):
            return False
        log("Checking something in the source")
        self.human.think("scan")
        read(self.human, seconds)
        return True

    def write(self, seconds_left):
        human = self.human
        # research: no material yet -> go and read first; a finished paper -> nothing left to write
        if self.research is not None and not self.research.has_material():
            log("Nothing to write about yet - researching first")
            self.browse(seconds_left)
            return
        if self.research is not None and self.research.finished:
            raise TaskError("The research paper is finished")
        if self.writer_win is not None and self._alive(self.writer_win.NativeWindowHandle) \
                and self.back_to(self.writer_win.NativeWindowHandle):
            self.writer_doc = W.document_control(self.writer_win) or self.writer_doc   # replaced after a save
            if not W.document_has_focus(self.writer_win, self.writer_doc):
                click_into_document(human, self.writer_win, self.writer_doc)
        else:
            research = self.research
            if research is not None and research.saved and os.path.exists(research.doc_path):
                # resuming a paper that was already saved: open that file, not a blank document
                self.writer_win, self.writer_doc = reopen_writer_document(human, research.doc_path)
            else:
                self.writer_win, self.writer_doc = open_writer_document(human)
            click_into_document(human, self.writer_win, self.writer_doc)
            self.written_any = False
        if self.research is not None:                # the next section of the paper
            self.research.write_next(self, seconds_left * 0.9)
            return
        text = self._next_chunk(seconds_left)
        type_into_document(human, self.writer_doc, text, self.revisions)

    def _next_chunk(self, seconds_left):
        """The next 1-4 sentences that fit the time left; a new paragraph starts with Enter."""
        cps = self.human.persona.wpm * 5 / 60 / 1.8      # effective chars/s including thinking pauses
        sentences = self.writing[self.para]
        chunk = []
        while self.sent < len(sentences) and len(chunk) < 4:
            nxt = sentences[self.sent]
            if chunk and (len(" ".join(chunk + [nxt])) / cps > 0.8 * seconds_left or self.human.rng.random() < 0.3):
                break
            chunk.append(nxt)
            self.sent += 1
        text = " ".join(chunk)
        if self.written_any:
            if self.sent == len(chunk):          # first sentences of a new paragraph
                self.human.press("enter")
            else:
                text = " " + text
        if self.sent >= len(sentences):          # paragraph done: next one (wrapping around)
            self.para = (self.para + 1) % len(self.writing)
            self.sent = 0
        self.written_any = True
        return text


    def spreadsheet(self, seconds_left):
        """Enter a few expenses into an Excel worksheet (a header row first)."""
        human = self.human
        if self.excel_win is not None and self._alive(self.excel_win.NativeWindowHandle) \
                and self.back_to(self.excel_win.NativeWindowHandle):
            focus_sheet(human, self.excel_win)
            goto_cell(human, self.excel_win, f"A{self.excel_row}")
        else:
            self.excel_win = open_office_blank(human, "excel", EXCEL_WINDOW_CLASS, "Blank workbook")
            focus_sheet(human, self.excel_win)
            human.hotkey("ctrl", "home")                    # start at A1
            self.excel_row = 1
        rows = [] if self.excel_row > 1 else [EXPENSE_HEADER]
        n = int(human.rng.integers(2, 5)) if seconds_left > 90 else 1
        for _ in range(n):
            rows.append(EXPENSES[self.expense % len(EXPENSES)])
            self.expense += 1
        log(f"Entering {len(rows)} row(s) from A{self.excel_row}")
        human.think("mental", 0.6)
        if not excel_sheet_has_focus(self.excel_win):
            raise TaskError("The Excel worksheet lost keyboard focus - stopping before typing anything")
        type_rows(human, rows)
        self.excel_row += len(rows)

    def notes(self, seconds_left):
        """Jot down a short note in Notepad (on a page of its own)."""
        human = self.human
        if self.notepad and self._alive(self.notepad) and self.back_to(self.notepad):
            win = auto.ControlFromHandle(self.notepad)
            ed = click_into_notepad(human, win)
            current = (notepad_text(ed) or "").replace("\r\n", "\n").replace("\r", "\n")
            if current.rstrip() != self.notepad_text.rstrip():
                raise TaskError("Notepad shows a different note than ours - not typing into it")
            human.hotkey("ctrl", "end")
            prefix = "\n"
        else:
            win = open_notepad_note(human)
            self.notepad, self.notepad_text, prefix = win.NativeWindowHandle, "", ""
        text = prefix + NOTES[self.note % len(NOTES)]
        self.note += 1
        human.think("mental", 0.6)
        if notepad_editor_with_focus(win) is None:
            raise TaskError("Notepad's editor lost keyboard focus - stopping before typing anything")
        log(f"Writing a note ({len(text.split())} words)")
        human.type_short(text)
        self.notepad_text += text
        ed = notepad_editor_with_focus(win)
        got = (notepad_text(ed) or "").replace("\r\n", "\n").replace("\r", "\n") if ed else None
        log("Note text matches what was typed" if got is not None and got.rstrip() == self.notepad_text.rstrip()
            else "Couldn't confirm the note's text")


def session_activities(ws):
    """The activities of a desktop session, for algorithms.session.Session."""
    return [
        Activity("browse", ws.browse, weight=1.0, min_seconds=45),
        Activity("write", ws.write, weight=0.8, min_seconds=40),
        Activity("spreadsheet", ws.spreadsheet, weight=0.5, min_seconds=40),
        Activity("notes", ws.notes, weight=0.4, min_seconds=30),
    ]
