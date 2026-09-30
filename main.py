"""
Chrome demo for the learned mouse model.

  1. Open Chrome (Start menu); if the "Who's using Chrome?" picker appears,
     move to the chosen profile and click it; maximise the browser window
  2. Move to the address bar with a learned trajectory and click
  3. Type a search query and press Enter
  4. Click a search result
  5. "Read" the page: scroll bursts, pauses and small mouse drifts
  6. Move to the Back button and click it
  7. Click a different result and read a little

Usage:
    python main.py                      # uses models/mouse_mdn.npz if present
    python main.py --no-model           # force the fallback generator
    python main.py --query "wikipedia" --read-seconds 20 --seed 42
    python main.py --profile Work --profile-index 1   # 2nd profile matching "Work"
    python main.py --list-profiles                    # print the picker's profiles and exit

Abort at any time by slamming the mouse into a screen corner.
"""

import argparse
import sys
import time

import numpy as np

from algorithms import human_typing, reading
from algorithms.human_mouse import HumanMouse
from controller import browser, keyboard

T0 = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - T0:6.1f}s] {msg}", flush=True)


def think(rng, median=0.6, sigma=0.5):
    """Short human 'decision' pause between actions (log-normal)."""
    time.sleep(float(median * np.exp(rng.normal(0, sigma))))


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


def choose_link(links, rng, hint=None, top_n=8):
    """
    Prefer search results over page navigation. With `hint`, take the first
    result containing it; otherwise one of the top results (upper ones likelier).
    """
    pool = result_links(links) or links
    if hint:
        for group in (pool, links):
            matches = [k for k in group if hint.lower() in k["name"].lower()]
            if matches:
                return matches[0]
    if not pool:
        return None
    candidates = pool[:top_n]
    weights = 1.0 / (1 + np.arange(len(candidates)))   # people favour the top of the page
    return candidates[rng.choice(len(candidates), p=weights / weights.sum())]


def describe_profiles(profiles):
    return "; ".join(f"[{p['index']}] {p['name']}" + (f" ({p['account']})" if p["account"] else "")
                     for p in profiles)


def choose_profile(hm, name, index):
    """
    On the profile picker, move to the requested profile card with the learned
    mouse model and click it. With no name given, the first profile is used.
    """
    rng = hm.rng
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
    think(rng, 1.0)                              # a person scans the cards first
    log(f"Choosing profile [{target['index']}] {target['name']} ({target['account'] or '-'})")
    hm.click_rect(target["click_rect"])
    if not browser.wait_for_browser_window(timeout=20):
        sys.exit("Browser window did not open after choosing the profile")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", default="wikipedia")
    ap.add_argument("--link-hint", default=None, help="text the first clicked result should contain (default: query)")
    ap.add_argument("--read-seconds", type=float, default=20)
    ap.add_argument("--no-model", action="store_true", help="use the fallback generator")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
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

    hm = HumanMouse(use_model=not args.no_model, temperature=args.temperature, seed=args.seed)
    rng = hm.rng
    log(f"Mouse backend: {hm.backend} | click timing: {hm.clicks.summary()}")
    log("Starting in 3 s - don't touch the mouse/keyboard.")
    time.sleep(3)

    # 1. Open Chrome
    log("Opening Chrome")
    if not browser.open_chrome():
        sys.exit("Chrome window did not appear")
    time.sleep(1.0)
    if browser.get_profile_picker(timeout=1):
        choose_profile(hm, args.profile, args.profile_index)
        time.sleep(1.0)
    browser.maximize_chrome()
    time.sleep(1.0)

    # 2. Address bar
    bar = browser.get_address_bar_rect()
    if not bar:
        sys.exit("Couldn't locate the address bar")
    log("Moving to the address bar")
    hm.click_rect(bar)
    think(rng)

    # Safety: never type unless Chrome's address bar really has keyboard focus
    if not browser.address_bar_has_focus():
        log("Address bar not focused after click - refocusing Chrome (Ctrl+L)")
        browser.focus_chrome()
        time.sleep(0.5)
        keyboard.hotkey("ctrl", "l")
        time.sleep(0.3)
        if not browser.address_bar_has_focus():
            sys.exit("Chrome's address bar doesn't have keyboard focus - stopping before typing anything")

    # 3. Search
    log(f"Typing search: {args.query!r}")
    keyboard.select_all()                      # replace whatever is in the bar
    human_typing.type_like_human(args.query, rng=rng)
    think(rng, 0.4)
    keyboard.press_key("enter")
    wait_for_navigation()
    think(rng, 1.2)

    # 4. First result
    page = browser.get_page_rect()
    links = usable_links(page)
    first = choose_link(links, rng, hint=args.link_hint or args.query)
    if not first:
        sys.exit("No clickable links found on the results page")
    log(f"Clicking result: {first['name'][:60]!r}")
    hm.click_rect(first["rect"])
    wait_for_navigation()

    # 5. Read
    log(f"Reading for ~{args.read_seconds:.0f} s")
    reading.read_page(hm, args.read_seconds, browser.get_page_rect())
    think(rng)

    # 6. Back
    back = browser.get_toolbar_button_rect("Back")
    if back:
        log("Moving to Back button")
        hm.click_rect(back)
    else:
        log("Back button not found - using Alt+Left")
        browser.go_back()
    wait_for_navigation()
    think(rng, 1.0)

    # 7. A different result
    links = usable_links(browser.get_page_rect(), exclude={first["name"]})
    second = choose_link(links, rng)
    if second:
        log(f"Clicking another result: {second['name'][:60]!r}")
        hm.click_rect(second["rect"])
        wait_for_navigation()
        reading.read_page(hm, args.read_seconds / 2, browser.get_page_rect())
    else:
        log("No second link found")

    log("Demo finished.")


if __name__ == "__main__":
    main()
