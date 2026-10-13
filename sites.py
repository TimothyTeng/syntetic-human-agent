"""
The sites the agent may visit and how it searches them.

The event network only lets a short list of sites through (with its certificate installed),
so nothing else is ever opened - not Google, not Wikipedia. Searches therefore run on the
sites' own search pages: the person uses the site's search box, or types the site's search
address with the query (e.g. "apnews.com/search?q=electric+vehicles"), and opens one of the
results it lists.

    url = search_url("apnews.com", "electric vehicles")
    is_result(link["href"], "apnews.com")           # an article the results page lists
    is_search_url(browser.get_current_url())        # a results page (no notes taken from it)

ALLOWED_HOSTS          load without trouble on the network; the only sites ever visited
CERT_WARNING_HOSTS     also let through, but Chrome stops at "Your connection isn't private"
                       first (someone has to click Continue) - not visited unless added with
                       --sites-file, and the agent never clicks through that warning itself
ENTRY_PAGES            the websites visited by default and the pages a visit can start on (all
                       checked to load, with no pop-ups beyond a cookie banner, which is accepted)
SITE_SEARCH            the allowed sites whose search page works from a typed address, with
                       the addresses of their results and searches that suit the site
                       (apple.com builds its results in script - it is browsed, not searched;
                       AWS, Amazon and Dell are only searched when added with --sites-file)
"""

import re
from urllib.parse import quote_plus

ALLOWED_HOSTS = (
    "coinmarketcap.com", "www.coursera.org", "www.coinbase.com", "coingecko.com", "www.samsung.com",
    "aws.amazon.com", "www.amazon.com", "www.apple.com", "apnews.com", "www.dell.com",
)

CERT_WARNING_HOSTS = (
    "archive.org", "arstechnica.com", "aliexpress.com", "arxiv.org", "asana.com", "asus.com", "bbc.com",
    "binance.com", "bing.com", "bit.ly", "bleacherreport.com", "booking.com", "bumble.com",
    "cloud.google.com", "cloudflare.com", "cnet.com", "cnn.com",
)

ENTRY_PAGES = {
    "apnews.com": ["apnews.com", "apnews.com/technology", "apnews.com/business", "apnews.com/science",
                   "apnews.com/world-news", "apnews.com/climate-and-environment"],
    "www.coursera.org": ["www.coursera.org", "www.coursera.org/articles", "www.coursera.org/browse"],
    "www.apple.com": ["www.apple.com", "www.apple.com/newsroom/", "www.apple.com/environment/"],
    "coinmarketcap.com": ["coinmarketcap.com/academy/", "coinmarketcap.com/academy/glossary"],
    "coingecko.com": ["www.coingecko.com/learn"],
    "www.coinbase.com": ["www.coinbase.com/learn", "www.coinbase.com/learn/crypto-basics/what-is-bitcoin"],
}
# Allowed, but not visited by default: pop-ups and chat assistants over the page (AWS, Dell,
# Samsung), or a redirect off the allowed hosts (amazon.com -> amazon.sg from Singapore). The
# price-table pages of CoinMarketCap / CoinGecko are left out too: huge pages that Chrome's
# accessibility tree struggles with. Add any of them back with --sites-file.
POPUP_HEAVY = ("aws.amazon.com", "www.dell.com", "www.samsung.com", "www.amazon.com")


def same_site(host, allowed):
    """True if `host` is `allowed` with or without 'www.' ('coingecko.com' ~ 'www.coingecko.com').
    Other subdomains don't count: the network lets through the listed hosts only."""
    strip = lambda h: h.lower()[4:] if h.lower().startswith("www.") else h.lower()   # noqa: E731
    return bool(host) and bool(allowed) and strip(host) == strip(allowed)


def site_of(host):
    """The ALLOWED_HOSTS entry `host` belongs to, or None."""
    return next((h for h in ALLOWED_HOSTS if same_site(host, h)), None)


class SiteSearch:
    def __init__(self, url, result, queries):
        self.url = url                    # typed address, {q} = the query (spaces as '+')
        self.result = re.compile(result, re.I)
        self.queries = queries


SITE_SEARCH = {
    "apnews.com": SiteSearch(
        "apnews.com/search?q={q}", r"^https?://(www\.)?apnews\.com/article/",
        ["electric vehicles", "artificial intelligence", "climate change", "space exploration",
         "renewable energy", "semiconductors", "cybersecurity", "interest rates", "bitcoin", "smartphones",
         "remote work", "supply chain"]),
    "www.coursera.org": SiteSearch(
        "www.coursera.org/search?query={q}",
        r"^https?://(www\.)?coursera\.org/(learn|specializations|professional-certificates|projects|articles)/",
        ["project management", "data analysis", "python programming", "machine learning", "excel",
         "cloud computing", "time management", "business writing", "digital marketing", "cybersecurity"]),
    "aws.amazon.com": SiteSearch(
        "aws.amazon.com/search/?searchQuery={q}", r"^https?://aws\.amazon\.com/(?!search|marketplace)[a-z]",
        ["machine learning", "serverless", "data lake", "cloud storage", "generative ai", "containers",
         "database migration"]),
    "www.amazon.com": SiteSearch(
        "www.amazon.com/s?k={q}", r"^https?://(www\.)?amazon\.com/([^/?#]+/)?(dp|gp/product)/",
        ["wireless mouse", "laptop stand", "noise cancelling headphones", "usb c hub", "desk lamp",
         "water bottle", "backpack", "mechanical keyboard", "office chair"]),
    "www.dell.com": SiteSearch(
        "www.dell.com/en-us/shop/search/{q}", r"^https?://(www\.)?dell\.com/[a-z-]+/shop/[^#]*/(spd|pd)/",
        ["laptop", "monitor", "gaming laptop", "docking station", "keyboard", "desktop computer"]),
}

# Search results a person reads, not buttons on them ("Customize & Buy", ratings, compare)
_NOT_A_RESULT = re.compile(r"\b(buy|add to (cart|bag|basket)|customi[sz]e|compare|out of 5 stars|reviews?)\b", re.I)

# A site's own search field, by its accessible name ("Search Amazon", "What do you want to learn?")
SEARCH_BOX = re.compile(r"search|what do you want to learn", re.I)


def search_key(host):
    """The SITE_SEARCH key for a host of that site (with or without 'www.'), or None."""
    return next((k for k in SITE_SEARCH if same_site(host, k)), None)


def search_url(site, query):
    """The address that searches `site` (a key of SITE_SEARCH) for `query`."""
    return SITE_SEARCH[site].url.format(q=quote_plus(query.strip()))


def searches():
    """Every (query, site) pair of SITE_SEARCH."""
    return [(q, site) for site, s in SITE_SEARCH.items() for q in s.queries]


def is_result(href, site, name=""):
    """True if a link (its address and text) is one of the results a search of `site` lists."""
    return bool(href) and "#" not in href and bool(SITE_SEARCH[site].result.match(href)) \
        and not _NOT_A_RESULT.search(name or "")


def is_search_url(url):
    """True for one of the sites' own search-results pages."""
    u = re.sub(r"^https?://", "", (url or "").strip(), flags=re.I).lower()
    for s in SITE_SEARCH.values():
        prefix = s.url.split("{q}")[0].lower()
        if u.startswith(prefix) or ("www." + u).startswith(prefix) or u.startswith(prefix.replace("www.", "", 1)):
            return True
    return False
