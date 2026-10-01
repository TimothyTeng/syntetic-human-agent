"""
Research notes: what the person took away from each page they read.

    notes = ResearchNotes(topic="Apollo program")
    src = notes.add(parse_capture(capture, query="apollo program wikipedia"))
    notes.save(folder)            # folder/notes.json
    notes = ResearchNotes.load(folder)

A Source keeps the page's cleaned text by section (Unicode, as read - folding to ASCII
happens when the paper is composed), its small tables, and which part of the
page was actually in view while reading (seen_intervals, % of page height).
"""

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class Para:
    """One paragraph of page text, with where it sat on the page."""
    text: str
    pos: float = 0.0              # where on the page it was (0 = top, 1 = bottom)


@dataclass
class SectionNote:
    """The paragraphs under one heading of a page."""
    heading: str                  # "" = the lead (text before the first heading)
    level: int = 2                # heading level (2 = <h2>); lower = more important
    paras: list = field(default_factory=list)      # [Para]

    def words(self):
        """Number of words in the section's paragraphs."""
        return sum(len(p.text.split()) for p in self.paras)


@dataclass
class TableNote:
    """A small table copied from a page: an infobox (key/value rows) or a data table."""
    kind: str                     # "infobox" | "data"
    rows: list                    # [[cell text]]
    caption: str = ""


@dataclass
class Source:
    """Everything kept from one page read: where it came from, its text by section, its
    tables, and how much of it was actually seen."""
    url: str
    title: str
    site: str                     # display name of the site ("Wikipedia")
    accessed: str                 # ISO date
    query: str = ""               # the search that led to the page
    sections: list = field(default_factory=list)   # [SectionNote]
    tables: list = field(default_factory=list)     # [TableNote]
    seen_intervals: list = field(default_factory=list)  # [[lo, hi]] in % of page height
    words_read: int = 0           # words of the kept text that were in view while reading
    id: int = 0                   # 1, 2, ... in order first read (set by ResearchNotes.add)

    def words(self):
        """Number of words of text kept from the page."""
        return sum(s.words() for s in self.sections)

    def seen(self, pos):
        """Whether the text at `pos` (0..1 of the page) was in view while reading. Pages
        read without position tracking count as fully seen."""
        if not self.seen_intervals:
            return True
        pct = 100.0 * pos
        return any(lo - 2.0 <= pct <= hi + 2.0 for lo, hi in self.seen_intervals)   # 2% slack at the edges

    @classmethod
    def from_dict(cls, d):
        """A Source from its JSON form (asdict output), rebuilding the nested dataclasses."""
        d = dict(d)
        d["sections"] = [SectionNote(s["heading"], s.get("level", 2), [Para(**p) for p in s.get("paras", [])])
                         for s in d.get("sections", [])]
        d["tables"] = [TableNote(**t) for t in d.get("tables", [])]
        return cls(**d)


class ResearchNotes:
    """The notes for one research topic: the Sources read so far, saved as notes.json."""

    def __init__(self, topic, sources=None):
        """Notes on `topic`, optionally starting from a list of Sources."""
        self.topic = topic
        self.sources = list(sources or [])

    def add(self, source):
        """Add a source (a page read again replaces its earlier notes). Returns it."""
        for i, old in enumerate(self.sources):
            if _same_page(old.url, source.url):
                # same page again: keep its id (citations stay stable) and add up what was seen
                source.id = old.id
                source.seen_intervals = _merge(old.seen_intervals + source.seen_intervals)
                source.words_read = max(old.words_read, source.words_read)
                self.sources[i] = source
                return source
        source.id = max((s.id for s in self.sources), default=0) + 1
        self.sources.append(source)
        return source

    def get(self, source_id):
        """The Source with this id, or None."""
        return next((s for s in self.sources if s.id == source_id), None)

    def to_dict(self):
        """Plain dict/list form for JSON."""
        return {"topic": self.topic, "sources": [asdict(s) for s in self.sources]}

    @classmethod
    def from_dict(cls, d):
        """ResearchNotes from the to_dict() form."""
        return cls(d["topic"], [Source.from_dict(s) for s in d.get("sources", [])])

    def save(self, folder):
        """Write folder/notes.json (via a temp file, so a crash never leaves it half written).
        Returns the path."""
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "notes.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1, ensure_ascii=False)
        os.replace(tmp, path)                       # atomic rename over the old file
        return path

    @classmethod
    def load(cls, folder):
        """The notes saved in folder/notes.json."""
        with open(os.path.join(folder, "notes.json"), encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _same_page(a, b):
    """True if two URLs name the same page (ignoring scheme, case, #fragment and a trailing /)."""
    norm = lambda u: u.lower().split("#")[0].rstrip("/").replace("https://", "").replace("http://", "")  # noqa: E731
    return norm(a) == norm(b)


def _merge(intervals):
    """Union of [lo, hi] intervals."""
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]:                # overlaps the last one: extend it
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out
