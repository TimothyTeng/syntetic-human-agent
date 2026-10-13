"""
A browsing session the way people surf: pick a website and spend a while on it - read the
page they land on top-down, follow a few links around the site, search it and open what
they find (a news article, a course), go back - then move on to the next
website, until the time is up, then close the browser.

    surfer = Surfer(human, ws, deadline, SiteSource(human.rng), target_sites=5)
    surfer.run(close=True)

Where visits start (a source's next() gives ("url", address, site) or ("search", query,
site)). Only the sites the event network lets through are used (sites.py), and searches run
on those sites' own search pages, not a search engine:
  SiteSource      normal mode: the websites in a shuffled cycle; a visit starts on one of the
                  site's entry pages (sites.ENTRY_PAGES, or a --sites-file) typed into the
                  address bar, now and then straight with a search of the site
  ResearchSource  research mode: the project's planned searches (the topic on AP News and
                  Coursera, then the topic with the first page's main headings)

During a visit, after each page (choose_move):
  link     follow a link on the page to another page of the same website (article_links:
           real words, not page furniture, citations or shop / account buttons; in research
           mode links that share words with the topic are preferred). The person looks for
           one - scrolling a little if none is in view - rests the cursor on it, then clicks
           (or Ctrl+clicks it into a new tab, as they open search results)
  back     back to the page the link was followed from (Back / Alt+Left, or closing the
           link's tab) and read on there a little
  result   back to the search results and open another one
  search   search the site (its search box when one is in view, else its search address in
           the address bar) and open one of the results - at most twice a visit
The visit ends when its time is up (or nothing more is found to do there); then the next
website. Link chains stay short (at most four hops from the page they started on). A cookie /
privacy banner (AP News asks before showing anything) is accepted when it appears.

Safety: only the allowed sites are opened; search results are picked by their address on the
site (tasks.choose_link); links on pages are filtered on their address where Chrome reports
it, and after every click the address actually reached is checked - anything off the website
being visited is left at once. A Chrome error page (the "Your connection isn't private"
warning, a site the network blocks) counts as a failure: the agent never clicks through it
and moves on to the next website.

Timing: with target_sites (normal mode) each visit gets the time left shared over the
websites still to visit (jittered); without it a visit lasts 1.5-4 min (research: 2.5-6 min).
Within a visit a page is read for its natural reading time, capped at a share of the visit's
time left so there is time for a few pages. Reading stops at the deadline; no new website is
started in the last ~45 s (the time left is spent reading on), and closing the browser
happens after the deadline.
"""

import os
import re
import time
import traceback
from collections import Counter

import numpy as np

import sites as site_list
import tasks
from algorithms import reading
from algorithms.session import TaskError
from controller import browser
from tasks import log

# Pages a normal-mode visit can start on (all on sites.ALLOWED_HOSTS), typed without the scheme
SITES = [url for pages in site_list.ENTRY_PAGES.values() for url in pages]

MOVE_WEIGHTS = {"link": 0.5, "back": 0.2, "result": 0.15, "search": 0.15}
MAX_HOPS = 4                             # links followed in a row from the page a chain started on
MAX_SEARCHES = 2                         # searches of the site in one visit
VISIT_SECONDS = (90.0, 240.0)            # a visit's length without a target number of websites
RESEARCH_VISIT_SECONDS = (150.0, 360.0)

# Link texts that are page furniture, not something a reader follows from the text
SKIP_LINK = re.compile(
    r"^(\[?\d+\]?|\[?[a-z]\]?|\^|edit|\[edit\]|edit source|cite|citation needed|jump to.*|main page|contents|"
    r"current events|random article|about wikipedia|contact us|help|learn to edit|community portal|"
    r"recent changes|upload file|log in|create account|donate|talk|read|view history|view source|"
    r"search|languages?|tools|download as pdf|printable version|permanent link|page information|"
    r"cite this page|what links here|related changes|special pages|archived.*|retrieved.*|isbn.*|"
    r"doi.*|pmid.*|s2cid.*|wayback machine|privacy policy|cookie.*|terms of use|subscribe|sign up|"
    r"sign in|share|print|email|next|previous|more|menu|home|skip to.*|"
    r"(buy|sell|trade|shop|order|checkout|add to (cart|bag|basket)|get started|log ?in|sign ?up|register|"
    r"download|get the app|customi[sz]e|compare|enroll|join|start free trial)\b.*|.*\b(cart|bag)$)$", re.I)
SKIP_HREF = re.compile(r"/wiki/(File|Help|Special|Wikipedia|Talk|Category|Portal|Template|Template_talk|User|"
                       r"Draft|Module|MediaWiki):|#|action=|oldid=|/w/index\.php|\.(pdf|jpe?g|png|svg|zip)$|"
                       r"/(cart|checkout|signin|sign-in|signup|sign-up|login|register|buy|trade|ap/signin|gp/cart)\b",
                       re.I)


def load_sites(path):
    """Addresses from a text file, one per line ('#' starts a comment)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(line)
    return out


class SiteSource:
    """Normal mode: the websites in a shuffled cycle. A visit starts on one of the site's
    pages (typed into the address bar) or, with p_search_first on a searchable site, with a
    search of it. sites: addresses to start from (default SITES), grouped by website."""

    def __init__(self, rng, sites=None, p_search_first=0.2):
        self.rng, self.p_search_first = rng, p_search_first
        self.groups = {}
        for url in sites or SITES:
            host = tasks.url_host(url)
            if host:
                self.groups.setdefault(site_list.site_of(host) or host, []).append(url)
        self.order = list(self.groups)
        rng.shuffle(self.order)
        self.used = {}                   # search key -> queries already searched
        # hosts of a user-supplied site list count as allowed too
        self.extra_hosts = tuple(h for h in self.groups if not tasks.host_allowed(h))

    def next(self):
        site = self.order.pop(0)
        self.order.append(site)
        if self.can_search(site) and self.rng.random() < self.p_search_first:
            return "search", self.query_for(site), site
        pages = self.groups[site]
        return "url", pages[int(self.rng.integers(len(pages)))], site

    def can_search(self, site):
        return site_list.search_key(site) is not None

    def query_for(self, site):
        """A search that suits the site, not searched yet if possible (None: not searchable)."""
        key = site_list.search_key(site)
        if key is None:
            return None
        used = self.used.setdefault(key, set())
        options = [q for q in site_list.SITE_SEARCH[key].queries if q not in used] or site_list.SITE_SEARCH[key].queries
        query = options[int(self.rng.integers(len(options)))]
        used.add(query)
        return query

    def keywords(self):
        return set()


class ResearchSource:
    """Research mode: the project's next planned search (and the site it runs on)."""

    extra_hosts = ()

    def __init__(self, project):
        self.project = project

    def next(self):
        query, site = self.project.next_query()
        log(f"Researching: {query!r} on {site}")
        return "search", query, site

    def can_search(self, site):
        return False                     # one planned search per visit, then links and other results

    def query_for(self, site):
        return None

    def keywords(self):
        """Words of the topic and of the headings read so far."""
        words = set(re.findall(r"[a-z]{4,}", self.project.topic.lower()))
        for src in self.project.notes.sources[:2]:
            for sec in src.sections[:12]:
                words |= set(re.findall(r"[a-z]{5,}", sec.heading.lower()))
        return words


# --- decisions (pure) ------------------------------------------------------------------------

def choose_move(rng, can_link, can_back, can_result, can_search=False, weights=MOVE_WEIGHTS):
    """What to do next on the website: 'link', 'back', 'result' or 'search' (None: nothing is
    possible - the visit is over)."""
    allowed = {"link": can_link, "back": can_back, "result": can_result, "search": can_search}
    names = [k for k in weights if allowed[k]]
    if not names:
        return None
    p = np.array([weights[k] for k in names])
    return names[int(rng.choice(len(names), p=p / p.sum()))]


def article_links(links, page_rect, current_url="", exclude=(), extra_hosts=(), site=None):
    """Links on a page a reader might follow: centred in the text column (not the side
    menus), at least one real word, not page furniture, a citation mark or a shop / account
    button, not already followed (exclude: link texts), and - where Chrome reports the
    address - on an allowed site (site: on that website), a different page and not a file /
    talk / special page."""
    l, t, r, b = page_rect
    col = (l + 0.14 * (r - l), t + 30, r - 0.04 * (r - l), b - 10)
    here = (current_url or "").split("#", 1)[0].rstrip("/").lower()
    out = []
    for k in links:
        name = (k.get("name") or "").strip()
        if not 3 <= len(name) <= 60 or name in exclude or SKIP_LINK.match(name):
            continue
        if not re.search(r"[A-Za-z]{3}", name) or name.lower().startswith(("http", "www.")):
            continue
        cx, cy = k["center"]
        if not (col[0] <= cx <= col[2] and col[1] <= cy <= col[3]):
            continue
        href = k.get("href") or ""
        if href:
            host = tasks.url_host(href)
            if not tasks.host_allowed(host, extra_hosts) or SKIP_HREF.search(href):
                continue
            if site and not site_list.same_site(host, site):
                continue
            if href.split("#", 1)[0].rstrip("/").lower().split("://")[-1] == here.split("://")[-1]:
                continue
        out.append(k)
    return out


def pick_link(links, keywords, page_rect, rng):
    """One of the candidate links: ones sharing words with `keywords` and ones in the upper
    middle of the view (where the reader just was) are likelier."""
    t, b = page_rect[1], page_rect[3]
    w = []
    for k in links:
        words = set(re.findall(r"[a-z]{4,}", k["name"].lower()))
        rel = (k["center"][1] - t) / max(1, b - t)
        w.append((1.0 + 2.0 * len(words & keywords)) * (np.exp(-((rel - 0.4) / 0.35) ** 2) + 0.05))
    w = np.array(w)
    return links[int(rng.choice(len(links), p=w / w.sum()))]


# --- the session -------------------------------------------------------------------------------

class Surfer:
    MIN_ENTRY = 45.0                     # seconds needed to start on a new website
    MIN_HOP = 20.0                       # ... to follow a link / search / open another result
    MAX_FAILURES = 4
    MAX_IDLE = 3                         # moves in a row that found nothing to do end a visit early

    def __init__(self, human, ws, deadline, source, target_sites=None, research=None, max_read=170.0):
        """ws: the tasks.Workspace (its Chrome window is used or opened); deadline:
        time.monotonic() when browsing ends; source: SiteSource / ResearchSource;
        target_sites: websites to visit in the time (paces the visits); research: a
        ResearchProject - notes are taken from every page read."""
        self.human, self.ws, self.deadline, self.source = human, ws, deadline, source
        self.target = target_sites
        self.research = research
        self.on_page = research.on_page if research is not None else None
        self.on_view = research.on_view if research is not None else None
        self.read_fraction = 0.5 if research is not None else None      # researchers read more of a page
        self.max_read = max_read
        self.extra_hosts = getattr(source, "extra_hosts", ())
        self.sites = 0                   # websites visited
        self.pages = 0                   # pages read
        self.visit_end = deadline        # when the current visit is over
        self.home = None                 # the website being visited (a host)
        self.searches = 0                # searches in this visit
        self.read_names = set()          # links / results already opened (not opened again)
        self.stack = []                  # how each link hop was opened from the chain's first page: 'same' / 'tab'
        self.result_tab = False          # the current result was opened in a new tab
        self.on_result = False           # the first page of this chain came from search results
        self.search_site = None          # the site whose search gave those results (None: a search engine)
        self.stats = Counter()

    def left(self):
        return self.deadline - time.monotonic()

    def visit_left(self):
        return min(self.visit_end, self.deadline) - time.monotonic()

    # --- the loop -------------------------------------------------------------------------------

    def run(self, close=False):
        """Browse until the deadline; close: close Chrome afterwards. Returns self.stats."""
        human = self.human
        self.ws.chrome_window()
        failures = 0
        while self.left() > self.MIN_ENTRY:
            kind, what, site = self.source.next()
            self.visit_end = time.monotonic() + self.visit_budget()
            log(f"--- Website {self.sites + 1}" + (f" of {self.target}" if self.target else "")
                + f": {site or tasks.url_host(what)} for ~{self.visit_left():.0f} s")
            try:
                self.visit(kind, what, site)
                failures = 0
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:          # TaskError, or an unexpected UI / COM error
                if type(exc).__name__ == "FailSafeException":
                    raise                     # the emergency stop (mouse in a corner) always wins
                failures += 1
                self.stats["failures"] += 1
                frames = [f for f in traceback.extract_tb(exc.__traceback__) if "site-packages" not in f.filename]
                where = f" at {os.path.basename(frames[-1].filename)}:{frames[-1].lineno} {frames[-1].name}" if frames else ""
                log(f"That didn't work out ({type(exc).__name__}: {exc}{where}) - on to the next website")
                if failures >= self.MAX_FAILURES:
                    log(f"{failures} failures in a row - stopping browsing early")
                    break
                self.recover()
            if self.left() > self.MIN_ENTRY:
                human.think("mental")         # where next?
        if self.left() > 6:                   # too little time for a new website: read on where we are
            log(f"Reading on until the time is up ({self.left():.0f} s)")
            tasks.read(human, self.left() - 2)
        self.tidy_tabs()
        log(f"Browsing done: {self.sites} websites visited" + (f" (target {self.target})" if self.target else "")
            + f", {self.pages} pages read; {self.stats['links']} links followed, {self.stats['results']} search "
              f"results opened, {self.stats['searches']} searches, {self.stats['typed']} addresses typed in, "
              f"{self.stats['backs']} times back" + (f", {self.stats['failures']} failures" if self.stats['failures'] else ""))
        if close:
            human.think("mental")
            if tasks.close_browser(human, self.ws.chrome):
                self.ws.chrome = None
        return self.stats

    def visit_budget(self):
        """Seconds for the next website (see the module docstring)."""
        avail = max(5.0, self.left())
        rng = self.human.rng
        if self.target and self.sites < self.target:
            budget = avail / (self.target - self.sites) * float(np.exp(rng.normal(0, 0.2)))
        else:
            budget = float(rng.uniform(*(RESEARCH_VISIT_SECONDS if self.research is not None else VISIT_SECONDS)))
        if avail - budget < self.MIN_ENTRY + 15:  # too little would be left for another website: stay on
            budget = avail
        return min(budget, avail)

    def visit(self, kind, what, site):
        """One website: land on it (a typed address, or a search of it and a result), read,
        then wander around it until the visit's time is up."""
        self.tidy_tabs()
        self.home = site or tasks.url_host(what)
        self.searches, self.on_result, self.stack = 0, False, []
        if kind == "url":
            self.go_to(what)
            self.read_page()
        elif not self.search(what):
            raise TaskError(f"No search result on {self.home} for {what!r}")
        self.sites += 1
        self.wander()

    def go_to(self, url):
        """Type an address into the address bar and go there (checked to be an allowed site)."""
        human = self.human
        log(f"Going to {url}")
        if not human.focus_address_bar():
            browser.focus_chrome()
            time.sleep(0.4)
            human.hotkey("ctrl", "l")
            time.sleep(0.3)
            if not browser.address_bar_has_focus():
                raise TaskError("Chrome's address bar doesn't have keyboard focus - not typing")
        human.select_all_and_type(url)
        shown = browser.get_current_url() or ""
        if shown and not shown.startswith(url):
            if shown.startswith(url[:max(4, len(url) - 3)]) and len(shown) > len(url):
                human.press("delete")         # Chrome completed it to a longer address: drop that
            else:
                log(f"Address bar shows {shown!r} - retyping")
                human.think("confirm")
                human.select_all_and_type(url, error_scale=0.0)
        human.think("confirm")
        human.press("enter")
        tasks.wait_for_navigation()
        host = tasks.url_host(browser.get_current_url())
        if not tasks.host_allowed(host, self.extra_hosts):
            human.go_back()
            raise TaskError(f"Typed address ended up on {host!r}, not an allowed site - went back")
        self.stats["typed"] += 1
        human.think("scan")

    def search(self, query):
        """Search the website for `query` - with its search box when one is in view (and the
        person is on the site already), else by typing its search address - and open one of
        the results. False if the search couldn't be made or listed nothing to open."""
        human = self.human
        key = site_list.search_key(self.home)
        self.search_site = key
        on_site = site_list.same_site(tasks.url_host(browser.get_current_url()), self.home)
        if not (key and on_site and human.rng.random() < 0.75 and tasks.search_on_page(human, query)):
            tasks.search(human, query, allow_suggestion=key is None, site=key)
        self.searches += 1
        self.stats["searches"] += 1
        if tasks.page_unreachable():
            raise TaskError(f"The search page didn't load ({browser.get_page_title()!r})")
        tasks.accept_consent(human)
        link, links = self.find_result()
        if not link:
            log(f"No result to open for {query!r}")
            return False
        self.open_result(links, link)
        return True

    def find_result(self):
        """(a result to open on the results page, the links seen there): the results can take
        a moment to appear and may start below the page's header - the person waits a moment,
        then scrolls down a little looking for them. (None, links) if there is none."""
        human, links = self.human, []
        for attempt in range(3):
            links = tasks.usable_links(browser.get_page_rect(), exclude=self.read_names,     # not re-read
                                       with_href=self.search_site is not None)
            if tasks.result_links(links, self.search_site):
                return tasks.choose_link(links, human, site=self.search_site), links
            if attempt == 0:
                time.sleep(1.5)
            else:
                reading.scroll_burst(human.rng, tasks.reading_params(human), -1, int(human.rng.integers(3, 6)))
            human.think("scan")
        return None, links

    def open_result(self, links, link):
        """Open a search result (sometimes clicking its neighbour first by mistake) and read it."""
        human = self.human
        if tasks.maybe_click_wrong_result(human, links, link, self.search_site):
            link = tasks.find_again(link["name"], browser.get_page_rect())
            if link is None:
                raise TaskError("The intended result is no longer on the results page")
        log(f"Clicking result: {link['name'][:60]!r}")
        self.result_tab = tasks.open_result(human, link)
        self.on_result, self.stack = True, []
        self.read_names.add(link["name"])
        self.stats["results"] += 1
        self.read_page(new_tab=self.result_tab)

    def page_budget(self):
        """Seconds to spend on the page just opened: its natural reading time, capped at a
        share of the visit's time left (so a few pages fit in a visit)."""
        avail = max(5.0, self.visit_left() - 3)
        natural = tasks.page_reading_time(self.human, self.max_read, read_fraction=self.read_fraction)
        share = 0.75 if self.research is not None else float(self.human.rng.uniform(0.3, 0.6))
        cap = avail * share if avail > 2 * self.MIN_HOP else avail
        return float(np.clip(natural, min(10.0, avail), max(min(10.0, avail), cap)))

    def read_page(self, new_tab=False, notes=True):
        """Accept a privacy banner if one is up, take notes (research), then read the page in
        front. A Chrome error page instead of the site is a failure (never clicked through)."""
        if tasks.page_unreachable():
            raise TaskError(f"Chrome shows an error page ({browser.get_page_title()!r}), not the site")
        tasks.accept_consent(self.human)
        if notes and self.on_page:
            self.on_page()
        budget = self.page_budget()
        log(f"Page {self.pages + 1}: {browser.get_page_title()[:70]!r}")
        res = tasks.read_result(self.human, budget, new_tab, on_view=self.on_view if notes else None)
        self.pages += 1
        return res

    # --- after a page ---------------------------------------------------------------------------

    def wander(self):
        """Follow links around the website, go back, search it, open other results - until
        the visit's time is up or nothing more is found to do there."""
        rng = self.human.rng
        idle = 0
        while self.visit_left() > self.MIN_HOP and idle < self.MAX_IDLE:
            can_result = self.on_result and not self.stack
            can_search = self.searches < MAX_SEARCHES and self.source.can_search(self.home)
            move = choose_move(rng, can_link=len(self.stack) < MAX_HOPS, can_back=bool(self.stack),
                               can_result=can_result, can_search=can_search)
            if move is None:
                break
            if move == "link":
                done = self.follow_link()
            elif move == "back":
                self.go_back()
                done = True
            elif move == "result":
                done = self.another_result()
            else:
                self.tidy_tabs()              # search from the page the visit is on, not a link's tab
                done = self.search(self.source.query_for(self.home))
            idle = 0 if done else idle + 1
        if idle >= self.MAX_IDLE:
            log(f"Nothing more to do on {self.home} - moving on")

    def follow_link(self):
        """Find a link on the page to another page of the website, rest on it, click it and
        read the page. False if there was none to follow (or it led off the website)."""
        human, rng = self.human, self.human.rng
        area = browser.get_page_rect()
        if area is None:
            return False
        keywords = self.source.keywords()
        links = []
        for attempt in range(3):
            links = article_links(browser.find_links(with_href=True), area, browser.get_current_url(),
                                  self.read_names, self.extra_hosts, site=self.home)
            if links:
                break
            if attempt < 2:                   # none in view: scroll on a little, looking for one
                reading.scroll_burst(rng, tasks.reading_params(human), -1, int(rng.integers(2, 6)))
                human.think("scan")
        if not links:
            log("No link on this page worth following here")
            return False
        link = pick_link(links, keywords, area, rng)
        human.visual_search(min(len(links), 8))
        l, t, r, b = link["rect"]
        human.move_to(int(rng.uniform(l + 2, max(l + 3, r - 2))), int((t + b) / 2))     # rest on it
        human.think("confirm", 1.5)
        log(f"Following a link: {link['name'][:50]!r}")
        try:
            new_tab = tasks.open_result(human, link)
        except TaskError as exc:
            log(f"The link didn't open ({exc})")
            return False
        host = tasks.url_host(browser.get_current_url())
        if not tasks.host_allowed(host, self.extra_hosts) or not site_list.same_site(host, self.home):
            log(f"The link went to {host!r}, off {self.home} - leaving it")
            if not (new_tab and tasks.close_tab(human)):
                human.go_back()
            tasks.wait_for_navigation()
            self.read_names.add(link["name"])
            return False
        self.stack.append("tab" if new_tab else "same")
        self.read_names.add(link["name"])
        self.stats["links"] += 1
        self.read_page(new_tab=new_tab)
        return True

    def go_back(self):
        """Back to the page the last link was followed from, and read on there a little."""
        human = self.human
        how = self.stack.pop()
        if how == "tab" and tasks.close_tab(human):
            log("Closing the tab - back to the previous page")
        else:
            log("Going back to the previous page")
            human.go_back()
        tasks.wait_for_navigation()
        self.stats["backs"] += 1
        human.think("scan")
        if self.visit_left() > 15:            # pick up where they were (no new notes: it was read)
            tasks.read(human, min(float(human.rng.uniform(8, 30)), self.visit_left() - 5))

    def another_result(self):
        """Back to the search results and open one not read yet. False if there is none."""
        human = self.human
        tasks.leave_result(human, self.result_tab)
        self.result_tab = False
        self.stats["backs"] += 1
        human.think("scan")
        link, links = self.find_result()
        if not link:
            log("No other result to open")
            self.on_result = False
            return False
        self.open_result(links, link)
        return True

    # --- housekeeping -----------------------------------------------------------------------------

    def tidy_tabs(self):
        """Close the tabs this chain opened (link tabs and a result's tab)."""
        n = sum(1 for h in self.stack if h == "tab") + (1 if self.result_tab else 0)
        for _ in range(n):
            if not tasks.close_tab(self.human):
                break
            log("Closing a tab")
        self.stack, self.result_tab = [], False
        self.on_result = False

    def recover(self):
        """After a failure: Chrome back in front, tabs tidied, a fresh start for the next website."""
        try:
            self.ws.chrome_window()
            self.tidy_tabs()
        except TaskError:
            self.ws.chrome = None
        self.on_result = False
