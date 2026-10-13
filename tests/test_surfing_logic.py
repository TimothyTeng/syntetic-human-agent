"""
The decisions behind browsing, offline: the progressive reader's pure functions
(algorithms/reading.py) and the surfer's choices of where to go next and which links in an
article are worth following (surfing.py).
"""

import unittest
from collections import Counter

import numpy as np

import sites
import tasks
from algorithms import reading
from surfing import SITES, SiteSource, article_links, choose_move, pick_link

PAGE = (0, 100, 1600, 1000)                  # page area on screen: left, top, right, bottom


def link(name, x=700, y=400, href=""):
    return {"name": name, "rect": (x - 40, y - 8, x + 40, y + 8), "center": (x, y), "href": href, "control": None}


class ReadingTest(unittest.TestCase):

    def test_steps_are_a_share_of_the_screen(self):
        """Reading steps scroll 40-80% of a screen; skimming steps are bigger."""
        rng = np.random.default_rng(0)
        read = [reading.step_notches(rng, 900, "read") for _ in range(300)]
        skim = [reading.step_notches(rng, 900, "skim") for _ in range(300)]
        self.assertGreaterEqual(min(read), 3)              # 0.4 * 900 / 100
        self.assertLessEqual(max(read), 8)                 # 0.8 * 900 / 100
        self.assertGreater(np.mean(skim), np.mean(read) + 2)
        self.assertEqual(reading.step_notches(rng, 50, "close"), 1)   # never zero

    def test_only_new_text_is_read(self):
        """After a step down only the words below what was already seen count; scrolling back
        over seen text reveals nothing new."""
        # 20% of the page in view; scrolled from the top to 25% position -> view covers 20..40%
        state = (25.0, 20.0)
        self.assertAlmostEqual(reading.view_top(state), 20.0)
        self.assertAlmostEqual(reading.revealed_words(state, 20.0, 300), 300)    # all new
        self.assertAlmostEqual(reading.revealed_words(state, 30.0, 300), 150)    # half was seen
        self.assertAlmostEqual(reading.revealed_words(state, 45.0, 300), 0)      # all seen before

    def test_pause_follows_the_words(self):
        """More words, or closer reading, take longer; pauses stay in their bounds."""
        rng = np.random.default_rng(1)
        few = np.median([reading.words_pause(rng, 30, 230, 0.45) for _ in range(200)])
        many = np.median([reading.words_pause(rng, 120, 230, 0.45) for _ in range(200)])
        close = np.median([reading.words_pause(rng, 120, 230, 0.85) for _ in range(200)])
        self.assertLess(few, many)
        self.assertLess(many, close)
        self.assertEqual(reading.words_pause(rng, 0, 230, 0.5), 0.4)
        self.assertLessEqual(reading.words_pause(rng, 5000, 230, 0.9), 30.0)

    def test_engagement_is_sticky(self):
        """The engagement level mostly stays the same from one step to the next."""
        rng = np.random.default_rng(2)
        cur, changes = "read", 0
        for _ in range(1000):
            nxt = reading.next_engagement(rng, cur)
            changes += nxt != cur
            cur = nxt
        self.assertLess(changes, 300)
        self.assertGreater(changes, 50)


class SurfingTest(unittest.TestCase):

    def test_moves_respect_what_is_possible(self):
        """Moves only when allowed (None when nothing is: the visit is over); links are the
        commonest move."""
        rng = np.random.default_rng(0)
        self.assertIsNone(choose_move(rng, False, False, False))
        self.assertEqual({choose_move(rng, False, True, False) for _ in range(50)}, {"back"})
        counts = Counter(choose_move(rng, True, True, True, True) for _ in range(2000))
        self.assertEqual(set(counts), {"link", "back", "result", "search"})
        self.assertEqual(counts.most_common(1)[0][0], "link")

    def test_article_links_skip_furniture(self):
        """Citations, edit links, side menus, off-site pages, other subdomains and shop /
        account buttons are never followed."""
        links = [
            link("Electric cars"), link("[12]"), link("edit"), link("Main page", x=100),        # side menu
            link("Battery makers expand", href="https://apnews.com/article/batteries-123"),
            link("Some blog", href="https://example.com/post"),
            link("Wikipedia page", href="https://en.wikipedia.org/wiki/Electric_car"),           # not let through
            link("AWS docs", href="https://docs.aws.amazon.com/lambda/"),                        # other subdomain
            link("Same page", href="https://apnews.com/article/ev-1#comments"),
            link("Buy now"), link("Add to cart"), link("Sign up", href="https://www.coinbase.com/signup"),
            link("Your cart", href="https://www.amazon.com/gp/cart/view.html"),
            link("Cite this page"), link("Charging networks", y=50),                         # above the page
            link("Charging networks lag"),
        ]
        names = [k["name"] for k in article_links(links, PAGE, "https://apnews.com/article/ev-1",
                                                  exclude={"Charging networks lag"})]
        self.assertEqual(names, ["Electric cars", "Battery makers expand"])

    def test_links_stay_on_the_website(self):
        """During a visit only links to the same website are followed (www. or not)."""
        links = [link("Coin prices", href="https://coinmarketcap.com/coins/"),
                 link("CoinGecko learn", href="https://www.coingecko.com/learn"),
                 link("Bitcoin", href="https://www.coinmarketcap.com/currencies/bitcoin/")]
        names = [k["name"] for k in article_links(links, PAGE, "https://coinmarketcap.com/", site="coinmarketcap.com")]
        self.assertEqual(names, ["Coin prices", "Bitcoin"])

    def test_research_prefers_topic_links(self):
        """Links that share words with the topic are picked more often."""
        rng = np.random.default_rng(3)
        links = [link("Apollo 11 mission", y=450), link("Cold War", y=450), link("Television", y=450)]
        picks = Counter(pick_link(links, {"apollo", "mission"}, PAGE, rng)["name"] for _ in range(600))
        self.assertGreater(picks["Apollo 11 mission"], picks["Cold War"] * 1.8)

    def test_pre_selected_sites_are_allowed(self):
        """Every built-in site is on the allowed list, and the source mixes sites and searches,
        every search on an allowed site's own search page."""
        for url in SITES:
            self.assertTrue(tasks.host_allowed(tasks.url_host(url)), url)
        src = SiteSource(np.random.default_rng(0))
        entries = [src.next() for _ in range(200)]
        kinds = Counter(kind for kind, _, _ in entries)
        self.assertGreater(kinds["url"], kinds["search"])
        self.assertGreater(kinds["search"], 10)
        for kind, what, site in entries:
            if kind == "search":
                key = sites.search_key(site)
                self.assertTrue(tasks.host_allowed(tasks.url_host(sites.search_url(key, what))), site)
                self.assertIn(what, sites.SITE_SEARCH[key].queries)
            else:
                self.assertTrue(sites.same_site(tasks.url_host(what), site), (what, site))
        self.assertEqual(src.extra_hosts, ())

    def test_websites_take_turns(self):
        """Every website is visited once before any is visited again."""
        src = SiteSource(np.random.default_rng(1))
        first = [src.next()[2] for _ in range(len(sites.ENTRY_PAGES))]
        self.assertEqual(sorted(first), sorted(sites.ENTRY_PAGES))
        self.assertTrue(src.can_search("apnews.com"))
        self.assertTrue(src.can_search("www.coursera.org"))
        self.assertFalse(set(sites.ENTRY_PAGES) & set(sites.POPUP_HEAVY))
        self.assertFalse(src.can_search("www.apple.com"))
        self.assertIsNone(src.query_for("www.apple.com"))
        own = SiteSource(np.random.default_rng(1), ["www.bbc.com/news", "www.bbc.com/sport"])
        self.assertEqual(own.next()[0], "url")
        self.assertEqual(own.extra_hosts, ("www.bbc.com",))

    def test_only_the_network_sites_are_allowed(self):
        """Exactly the listed hosts, with or without 'www.' - no other subdomains, no look-alikes,
        not the sites behind the certificate warning, no search engine."""
        for host in ("apnews.com", "www.apnews.com", "coinmarketcap.com", "www.coinmarketcap.com",
                     "www.coingecko.com", "coursera.org", "aws.amazon.com", "www.amazon.com", "www.samsung.com"):
            self.assertTrue(tasks.host_allowed(host), host)
        for host in ("en.wikipedia.org", "www.google.com", "docs.aws.amazon.com", "www.amazon.sg",
                     "apnews.com.evil.com", "evilapnews.com", "www.bbc.com", "www.bing.com"):
            self.assertFalse(tasks.host_allowed(host), host)
        self.assertTrue(tasks.host_allowed("www.bbc.com", extra=("www.bbc.com",)))     # added with --sites-file

    def test_site_search(self):
        """Search addresses, the results they list, and results pages recognised as such."""
        self.assertEqual(sites.search_url("apnews.com", "electric vehicles"), "apnews.com/search?q=electric+vehicles")
        self.assertEqual(sites.search_url("www.coursera.org", " data analysis "),
                         "www.coursera.org/search?query=data+analysis")
        self.assertTrue(sites.is_result("https://apnews.com/article/panama-earthquake-e4e5dc85", "apnews.com"))
        self.assertFalse(sites.is_result("https://apnews.com/hub/television", "apnews.com"))
        self.assertTrue(sites.is_result("https://www.coursera.org/learn/excel-data-analysis", "www.coursera.org"))
        self.assertTrue(sites.is_result("https://aws.amazon.com/serverless/", "aws.amazon.com"))
        self.assertFalse(sites.is_result("https://docs.aws.amazon.com/lambda/", "aws.amazon.com"))
        dell = "https://www.dell.com/en-us/shop/laptop-computers/spd/xps13dx13260/dx13260_fixed_61"
        self.assertTrue(sites.is_result(dell, "www.dell.com", "XPS 13 Laptop"))
        self.assertFalse(sites.is_result(dell, "www.dell.com", "Customize & Buy XPS 13 Laptop"))
        self.assertFalse(sites.is_result(dell + "#ratings_section", "www.dell.com", "4.3 (76)"))
        for url in ("https://apnews.com/search?q=climate", "https://www.coursera.org/search?query=excel",
                    "https://www.dell.com/en-us/shop/search/laptop?rt=true", "https://aws.amazon.com/search/?searchQuery=x"):
            self.assertTrue(sites.is_search_url(url), url)
        self.assertFalse(sites.is_search_url("https://apnews.com/article/x"))

    def test_results_on_a_site_search_page(self):
        """One link per result, only real results, in page order."""
        links = [link("World", href="https://apnews.com/hub/world-news"),
                 link("Chinese EV makers move to Africa", href="https://apnews.com/article/ev-africa-1"),
                 link("Chinese EV makers move to Africa", href="https://apnews.com/article/ev-africa-1"),
                 link("Bolivia fuel shortages", href="https://apnews.com/article/bolivia-2")]
        names = [k["name"] for k in tasks.result_links(links, "apnews.com")]
        self.assertEqual(names, ["Chinese EV makers move to Africa", "Bolivia fuel shortages"])

    def test_url_hosts(self):
        self.assertEqual(tasks.url_host("https://apnews.com/article/x"), "apnews.com")
        self.assertEqual(tasks.url_host("www.coursera.org/articles"), "www.coursera.org")
        self.assertIsNone(tasks.url_host("about:blank"))
        self.assertTrue(tasks.host_allowed("evil.com", extra=("evil.com",)))


if __name__ == "__main__":
    unittest.main()
