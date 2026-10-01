"""
Writing the paper from the notes - without a language model.

    doc = compose_paper(notes, "Apollo program", rng, ComposeConfig(target_words=800))

The text is extractive: real sentences from the pages read, chosen and arranged so the
result reads like a careful student's compiled report.

 1. Sentence pool    every paragraph of every source split into sentences, folded to
                     ASCII and filtered (10-40 words, statements only, nothing Writer
                     would autocorrect). Sentences that lean on the one before
                     ("It ...", "However, ...") are only used together with it.
 2. Scores           0.45 TextRank centrality + 0.30 similarity to the topic + 0.15
                     position in its section + 0.10 contains a date or number; x1.5
                     for text that was actually in view while reading.
 3. Outline          body sections = the sources' own top-level headings, clustered
                     across sources (similar names or similar content), strongest 3-5
                     kept; k-means on the sentences if the pages had no usable headings.
 4. Selection        per section, maximal marginal relevance up to a word budget, with
                     near-duplicates across the whole paper dropped.
 5. Arrangement      sentences from one source stay together (history sections in date
                     order), paragraphs of 2-4 sentences, one citation [k] per paragraph.
 6. Light rewriting  a phrase map ("in order to" -> "to") and a safe date move, no
                     word-level synonyms (without grammar knowledge they go wrong).
 7. Frame            abstract, keywords, introduction, key-facts tables (infobox or
                     timeline, and the sources consulted), discussion with key points,
                     conclusion and references, from small sets of templates.
"""

import datetime
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import ir, rank
from .ir import Bullets, Caption, Document, Heading, PageBreak, Paragraph, Run, Subtitle, Table, Title
from .textutil import (STOPWORDS, ascii_fold, clean_line, is_ascii, join_list, load_map, split_sentences,
                       title_case, tokenize)


@dataclass
class ComposeConfig:
    """Knobs for compose_paper(). The defaults give an ~800-word report."""
    target_words: int = 800          # length of the whole paper; sets its size class and section count
    paraphrase: int = 1              # 0 = verbatim, 1 = phrase map + date moves, 2 = also "According to ..."
    min_sentence_words: int = 10     # shorter sentences are usually captions or fragments
    max_sentence_words: int = 40     # longer ones are run-ons that read badly out of context
    max_pool: int = 2500             # cap on pool sentences (shared between sources) - bounds the n*n matrices
    mmr_lambda: float = 0.7          # MMR trade-off: 1 = relevance only, 0 = novelty only
    duplicate_sim: float = 0.6       # cosine similarity above which a sentence repeats one already used
    today: datetime.date = None      # date on the paper and in references (None = today)
    weights: dict = field(default_factory=lambda: {"textrank": 0.45, "topic": 0.30, "position": 0.15, "fact": 0.10})


# --- Templates (one variant is picked per paper, never repeated) ----------------------------------

SUBTITLES = ["A short research report", "A review of {n} online sources", "Research summary, {month_year}"]
ABSTRACT_OPEN = [
    "This report reviews {topic}, drawing on {n} sources including {sites}.",
    "This paper summarises what is known about {topic}, based on {n} sources such as {sites}.",
    "This short report brings together information on {topic} from {n} sources, among them {sites}.",
]
ABSTRACT_CLOSE = [
    "It covers {secs}, and summarises the key facts in tabular form.",
    "The discussion is organised around {secs}, with the main facts collected in tables.",
    "Sections on {secs} follow, together with a table of key facts.",
]
INTRO_ROADMAP = [
    "The sections below examine {secs}.",
    "The rest of this report looks at {secs} in turn.",
    "This report is organised into sections on {secs}, followed by a discussion and a conclusion.",
]
SECTION_OPEN = [
    "This section summarises the main points on {h}.",
    "The sources cover {h} in some detail.",
    "Several points stand out regarding {h}.",
]
SHARED = [
    "Several of the sources emphasise {things}.",
    "Across the sources, {things} come up repeatedly.",
    "A number of themes recur across the material, notably {things}.",
]
FOCUS = [
    "{a} gives most attention to {fa}, whereas {b} concentrates more on {fb}.",
    "While {a} focuses on {fa}, {b} places more weight on {fb}.",
]
AGREE = [
    "The sources broadly agree on the main facts and differ mainly in emphasis.",
    "Where the sources overlap, their accounts are consistent with one another.",
]
CONCLUDE_OPEN = [
    "In summary, the sources reviewed here give a consistent account of {topic}.",
    "Taken together, the sources describe {topic} from several complementary angles.",
    "Overall, the material reviewed provides a broad overview of {topic}.",
]
CONCLUDE_CLOSE = [
    "Further reading of primary sources would help to clarify {weak}.",
    "A closer look at {weak} would be a useful next step.",
    "Future work could examine {weak} in more detail.",
]
EVALUATIVE = re.compile(r"\b(remains?|considered|widely|significan\w*|important|influen\w*|legacy|"
                        r"achievement|one of the|helped|showed|shown|demonstrat\w*|inspir\w*|lasting|"
                        r"advances?|transformed|key|major)\b", re.I)
ATTRIBUTION = ["According to {site}, ", "As {site} notes, ", "{site} reports that "]

FRAME_WORDS = {"tiny": 300, "small": 400, "full": 540}
MINOR_SECTIONS = re.compile(r"^(name|names|naming|etymology|terminology|pronunciation|in popular culture|"
                            r"trivia|gallery|overview|summary|introduction|contents)$", re.I)
HISTORY_HEADINGS = re.compile(r"histor|background|origin|timeline|chronolog|development|early|beginning", re.I)
LEANS_ON_PREVIOUS = re.compile(
    r"^(It|Its|This|These|That|Those|They|Their|Them|He|She|His|Her|Such|However|Also|Moreover|Furthermore|"
    r"Thus|Therefore|Hence|Consequently|Meanwhile|Instead|Nevertheless|Nonetheless|Similarly|Likewise|"
    r"Additionally|Besides|Otherwise|Then|Later|Afterwards|Afterward|Both|Neither|Each|Another|"
    r"In addition|As a result|In contrast|On the other hand|For example|For instance|In turn|At the same time|"
    r"By then|The latter|The former|The same|Despite this|Because of this|After this|Before this|"
    r"The (?:first|second|third|other|rest|remaining),)(?:\b|(?<=,))")
SUBJECTLESS = re.compile(r"^(Flew|Served|Died|Became|Was|Were|Had|Has|Commanded|Piloted|Walked|Born|"
                         r"Retired|Resigned|Joined|Left|Selected|Assigned|Named|Replaced|[A-Z][a-z]+ed)\s"
                         r"(on|in|as|at|from|to|the|a|an|by|for|with|during|after|before)\b")
PERSONAL = re.compile(r"\b(I|me|my|we|our|us|you|your)\b", re.I)
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
NUMBER = re.compile(r"\b\d[\d,.]*\b")
MONTHS = ("January February March April May June July August September October November December").split()
_DATE_PHRASE = re.compile(
    r"^(In|By|On|During|From|Until|Since|Between)\s+((early|mid|late)[ -])?((" + "|".join(MONTHS) + r")\s+)?"
    r"(\d{1,2},?\s+)?(1[0-9]{3}|20[0-9]{2})(s)?,\s+", re.I)
PHRASES = [(re.compile(r"\b" + re.escape(a) + r"\b", re.I), b) for a, b in load_map("phrase_map.tsv")]


# --- 1. Sentence pool -----------------------------------------------------------------------------

@dataclass
class Sent:
    """One candidate sentence in the pool, with where it came from."""
    text: str               # the sentence, folded to ASCII
    src: int                # source id
    group: str              # top-level heading of the source section ("" = lead)
    heading: str            # the actual (sub)section heading
    index: int              # position within its group (0 = first sentence)
    pos: float              # position on the page (0..1)
    seen: bool              # its paragraph was in view while reading
    prev: int = -1          # pool index of the sentence right before it in the same paragraph (-1 = none)
    leans: bool = False     # only makes sense after the previous sentence
    tokens: list = None     # tokenize(text): stems for TF-IDF


def usable_sentence(s, cfg):
    """True if `s` can be lifted into the paper as it is: a complete statement of the right
    length that reads well out of context and is safe to type (ASCII, nothing Writer would
    autocorrect or autoformat)."""
    words = s.split()
    if not (cfg.min_sentence_words <= len(words) <= cfg.max_sentence_words):
        return False
    if not s[:1].isupper() and not s[:1].isdigit():
        return False
    if SUBJECTLESS.match(s) and "," not in " ".join(s.split()[:8]):
        return False                  # a note from a table or list ("Flew on the 1975 ..."), not a sentence
    if not s.endswith((".", '."', ".)")) or s.endswith(("..", " .")):
        return False                  # cut off, or an ellipsis
    if "?" in s or "!" in s or "[" in s or "]" in s or s.count(":") > 1:
        return False                  # questions, exclamations, leftover markup, lists
    if s.count('"') > 2 or s.count("(") != s.count(")") or PERSONAL.search(s):
        return False                  # quotations, broken brackets, the author's own voice
    if not is_ascii(s) or ir.check_text(s):
        return False                  # non-ASCII would be pasted, not typed; Writer would autocorrect it
    return sum(ch.isdigit() for ch in s) < 0.15 * len(s)    # not a row of statistics


def build_pool(notes, cfg):
    """Every usable sentence of every source, in page order, as a list of Sent.

    Each source gets an equal share of cfg.max_pool so one long page can't crowd out the
    rest. Sentences are folded to ASCII before the checks because the typing layer pastes
    anything it can't press as a key; those that still aren't ASCII are dropped."""
    pool = []
    per_source = max(50, cfg.max_pool // max(1, len(notes.sources)))
    for src in notes.sources:
        start = len(pool)
        top_level = min((s.level for s in src.sections if s.heading), default=2) + 1   # sections and subsections
        group, counter = "", Counter()
        for sec in src.sections:
            # sections and subsections each start a group; anything deeper stays in the
            # current one, and text without a heading (the lead) is group ""
            if not sec.heading:
                group = ""
            elif sec.level <= top_level:
                group = sec.heading
            for para in sec.paras:
                prev = -1
                for raw in split_sentences(clean_line(para.text)):
                    if len(pool) - start >= per_source:
                        break
                    text = ascii_fold(raw).strip()
                    if not usable_sentence(text, cfg):
                        prev = -1           # a gap: the next sentence has nothing to lean on
                        continue
                    pool.append(Sent(text=text, src=src.id, group=group, heading=sec.heading,
                                     index=counter[group], pos=para.pos, seen=src.seen(para.pos), prev=prev,
                                     leans=bool(LEANS_ON_PREVIOUS.match(text)), tokens=tokenize(text)))
                    counter[group] += 1
                    prev = len(pool) - 1
    return pool


# --- 2. Scores ------------------------------------------------------------------------------------

def score_pool(pool, topic, cfg, any_seen_tracking):
    """How much each pool sentence deserves a place in the paper.

    score = 0.45 centrality + 0.30 topic similarity + 0.15 position + 0.10 fact (weights from
    cfg.weights), each part scaled to 0..1, then x1.5 for sentences that were in view while
    reading - the person writes about what they actually read. Returns (score, S, X, vocab):
    S is the sentence similarity matrix and X the TF-IDF rows, reused for outline and MMR."""
    X, vocab, idf = rank.tfidf_matrix([s.tokens for s in pool])
    S = rank.cosine_sim(X)
    tr = rank.textrank(S)                          # central = says what many other sentences say
    tr = tr / tr.max() if len(tr) and tr.max() > 0 else tr
    q = rank.transform(tokenize(topic), vocab, idf)
    rel = X @ q                                    # cosine similarity to the topic as a query
    rel = rel / rel.max() if len(rel) and rel.max() > 0 else rel
    # early in its section = summary-like (1, 1/2, 1/3, ...); the lead gets a small bonus
    position = np.array([1.0 / (1 + s.index) + (0.1 if not s.group else 0.0) for s in pool])
    fact = np.array([1.0 if (YEAR.search(s.text) or NUMBER.search(s.text)) else 0.0 for s in pool])
    w = cfg.weights
    score = w["textrank"] * tr + w["topic"] * rel + w["position"] * np.minimum(position, 1.0) + w["fact"] * fact
    if any_seen_tracking:                          # without tracking every sentence counts as seen
        score = score * np.array([1.5 if s.seen else 1.0 for s in pool])
    return score.astype(np.float64), S, X, vocab


def units(pool, score):
    """Selectable units: a sentence on its own, or a leaning sentence together with the one
    before it. [(members, score)]

    A sentence starting "It ..." or "However, ..." is meaningless (or wrong) after some other
    sentence, so it can only be chosen as a pair with the one it leans on. Pairs score a bit
    below their mean as they cost twice the words; chains of leaning sentences are skipped."""
    out = []
    for i, s in enumerate(pool):
        if not s.leans:
            out.append(((i,), score[i]))
        elif s.prev >= 0 and not pool[s.prev].leans:
            out.append(((s.prev, i), 0.9 * (score[s.prev] + score[i]) / 2))
    return out


# --- 3. Outline -----------------------------------------------------------------------------------

@dataclass
class Section:
    """A planned body section of the paper and the pool sentences it may draw on."""
    name: str               # heading as it will be typed ("" = unusable)
    groups: list            # [(src, group heading)]
    members: list           # pool indices
    strength: float = 0.0   # how much good material it has (decides which sections are kept)
    order: float = 0.0      # median page position of its sentences (decides the section order)


def plan_outline(pool, score, X, rng, n_sections, topic):
    """The paper's body sections, in reading order: the sources' own headings merged across
    sources, the strongest `n_sections` kept (k-means topics if too few headings survive)."""
    by_group = defaultdict(list)
    for i, s in enumerate(pool):
        if s.group:
            by_group[(s.src, s.group)].append(i)
    clusters = []                                      # [[group keys]]
    topic_tokens = set(tokenize(topic))
    names = {g: set(tokenize(g[1])) - topic_tokens for g in by_group}   # "Apollo missions" ~ "Missions"
    centroid = {g: _unit(X[idx].mean(axis=0)) for g, idx in by_group.items()}
    # greedy single pass, biggest groups first so they seed the clusters; a group joins the
    # most similar cluster: a similar heading (word Jaccard >= 0.5, always preferred, hence
    # the 1 +) or else similar content (cosine to the cluster centroid >= 0.45)
    for g in sorted(by_group, key=lambda g: -len(by_group[g])):
        best, best_sim = None, 0.0
        for c in clusters:
            if g[0] in {h[0] for h in c}:              # at most one section per source in a cluster
                continue
            jac = max(_jaccard(names[g], names[h]) for h in c)
            cos = float(_unit(sum(centroid[h] for h in c)) @ centroid[g])
            sim = 1.0 + jac if jac >= 0.5 else cos if cos >= 0.45 else 0.0
            if sim > best_sim:
                best, best_sim = c, sim
        if best is None:
            clusters.append([g])
        else:
            best.append(g)
    sections = []
    for c in clusters:
        members = [i for g in c for i in by_group[g]]
        if len(members) < 3:
            continue                                   # too thin to fill a section
        # strength: its best 8 sentences, +50% per extra source covering it (a theme several
        # sources share is central), scaled down if it has fewer than 8 sentences
        top = sorted(score[members], reverse=True)[:8]
        strength = float(np.sum(top)) * (1 + 0.5 * (len({g[0] for g in c}) - 1)) * min(1.0, len(members) / 8)
        heading_votes = Counter(g[1] for g in c)
        name = min(heading_votes, key=lambda h: (-heading_votes[h], len(h)))   # most common, then shortest
        sections.append(Section(name=_section_name(name), groups=c, members=members, strength=strength,
                                order=float(np.median([pool[i].pos for i in members]))))
    sections.sort(key=lambda s: -s.strength)
    sections = [s for s in sections if s.name and not MINOR_SECTIONS.match(s.name)][:n_sections]
    if len(sections) < min(3, n_sections):
        sections = _kmeans_sections(pool, score, X, rng, n_sections)
    return sorted(sections, key=lambda s: s.order)     # pages tend to go background -> details -> legacy


def _kmeans_sections(pool, score, X, rng, k):
    """Sections for pages without usable headings: k-means on the sentences, each cluster
    named after its two most characteristic words."""
    surface = _surface_forms(pool)
    idx = [i for i, s in enumerate(pool) if s.group] or list(range(len(pool)))   # leave the lead to the intro
    if len(idx) < 6:
        return []
    labels, C = rank.kmeans(X[idx], min(k, len(idx) // 3), rng)                 # >= 3 sentences per cluster
    out = []
    for j in range(C.shape[0]):
        members = [idx[m] for m in range(len(idx)) if labels[m] == j]
        if len(members) < 3:
            continue
        counts = Counter(t for i in members for t in pool[i].tokens if not t.isdigit())
        top = [surface.get(t, t) for t, _ in counts.most_common(2)]                # stems back to real words
        out.append(Section(name=title_case(" and ".join(top)), groups=[], members=members,
                           strength=float(np.sum(sorted(score[members], reverse=True)[:8])),
                           order=float(np.median([pool[i].pos for i in members]))))
    return sorted(out, key=lambda s: -s.strength)[:k]


def quoted(name):
    """A section's name inside a sentence: 'the sections on "Background" and "Legacy"'."""
    return f'"{name}"'


def _section_name(heading):
    """A page heading cleaned up for use as a section heading in the paper (ASCII, safe to
    type, title case), or "" if it can't be one."""
    name = ascii_fold(re.sub(r"\s+", " ", heading)).strip(" .:")
    if (not is_ascii(name) or ir.check_text(name) or len(name.split()) > 8 or not name or "|" in name
            or not re.search(r"[A-Za-z]{3}", name)):                 # "1962", "A-C": not a section name
        return ""
    return title_case(name)


def _unit(v):
    """`v` scaled to length 1 (a zero vector is returned as is)."""
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _jaccard(a, b):
    """Overlap of two sets, |a & b| / |a | b| (0 if either is empty)."""
    return len(a & b) / len(a | b) if a and b else 0.0


def _surface_forms(pool):
    """stem -> the most common way it is written."""
    seen = defaultdict(Counter)
    for s in pool:
        for w in re.findall(r"[A-Za-z0-9]+", s.text):
            t = tokenize(w)
            if t:
                seen[t[0]][w.lower()] += 1
    return {t: c.most_common(1)[0][0] for t, c in seen.items()}


# --- 4. Selection ---------------------------------------------------------------------------------

class Selector:
    """Picks units by maximal marginal relevance, remembering everything chosen in the
    paper so nothing is said twice."""

    def __init__(self, pool, score, S, cfg):
        """Set up the selectable units of `pool` with their scores scaled to 0..1 (so they
        trade off evenly against similarity in MMR)."""
        self.pool, self.score, self.S, self.cfg = pool, score, S, cfg
        self.all_units = units(pool, score)
        top = max((u[1] for u in self.all_units), default=1.0) or 1.0
        self.rel = {u[0]: u[1] / top for u in self.all_units}
        self.chosen = []                                # pool indices

    def is_used(self, members):
        """True if any of these pool indices is already in the paper."""
        return any(m in self.chosen for m in members)

    def redundancy(self, members):
        """Highest similarity between the unit's sentences and anything already chosen."""
        if not self.chosen:
            return 0.0
        return float(self.S[np.ix_(list(members), self.chosen)].max())

    def duplicate(self, members):
        """True if the unit repeats something already in the paper: too similar overall, or
        the same year with 4+ shared words - two sources telling the same dated event in
        different words ("In 1961, Kennedy ...") rarely reach the similarity threshold."""
        if self.redundancy(members) > self.cfg.duplicate_sim:
            return True
        nums = {n for m in members for n in YEAR.findall(self.pool[m].text)}
        if not nums:
            return False
        toks = {t for m in members for t in self.pool[m].tokens}
        for c in self.chosen:
            if nums & set(YEAR.findall(self.pool[c].text)) and len(toks & set(self.pool[c].tokens)) >= 4:
                return True
        return False

    def candidates(self, allowed):
        """Unused units lying entirely within the `allowed` pool indices."""
        allowed = set(allowed)
        return [u for u, _ in self.all_units if all(m in allowed for m in u) and not self.is_used(u)]

    def pick(self, allowed, budget_words=None, max_units=None):
        """Choose units among `allowed` pool indices until the word budget / unit count is
        reached. Returns the chosen units (tuples of pool indices).

        Each step takes the unit with the best maximal marginal relevance,
        lam * relevance - (1 - lam) * redundancy, so strong sentences come first but one
        that mostly repeats what is already chosen (anywhere in the paper) gives way."""
        pool_units = self.candidates(allowed)
        picked, words = [], 0
        lam = self.cfg.mmr_lambda
        while pool_units:
            if max_units is not None and len(picked) >= max_units:
                break
            if budget_words is not None and words >= budget_words:
                break
            best = max(pool_units, key=lambda u: lam * self.rel[u] - (1 - lam) * self.redundancy(u))
            pool_units.remove(best)
            if self.is_used(best) or self.duplicate(best):
                continue                                # e.g. a pair whose first half was taken alone
            size = sum(len(self.pool[m].text.split()) for m in best)
            if budget_words is not None and picked and words + size > 1.15 * budget_words:
                continue                                # too long to fit: a shorter one may
            picked.append(best)
            self.chosen += list(best)
            words += size
        return picked


# --- 5. Arrangement and rewriting -----------------------------------------------------------------

CITE = "\x00{}\x00"             # placeholder for a source's reference number (numbered at the end, see
                                # number_citations: [k] goes by first citation, not by source id)


def cite(text, src_id):
    """Put the citation before the sentence's final punctuation: '... in 1969 [3].'"""
    m = re.search(r"([.!?][\"')]*)$", text)
    mark = " " + CITE.format(src_id)
    return text[:m.start()] + mark + m.group(1) if m else text + mark


def rewrite(text, rng, level):
    """Grammar-safe light paraphrase.

    Only whole phrases from phrase_map.tsv are swapped ("in order to" -> "to"), and with
    some chance a leading date moves to the end ("In 1961, X did Y." -> "X did Y in 1961.").
    There are no word-level synonyms: without knowing the grammar they pick the wrong
    sense or form often enough to make the text read worse, not better."""
    if level <= 0:
        return text
    for pattern, repl in PHRASES:
        def sub(m, repl=repl):
            """The replacement, capitalised if it replaces the sentence's first word."""
            if m.start() == 0 and m.group(0)[:1].isupper():
                return repl[:1].upper() + repl[1:]
            return repl
        text = pattern.sub(sub, text)
    # only a simple main clause (no further commas/semicolons) that starts with "The" or a
    # capital, so "In 1961, the ..." doesn't become a sentence starting in lower case
    m = re.match(r"^(In|By) (\d{4}), ((?:The |[A-Z])[^,;]{15,})\.$", text)
    if m and rng.random() < 0.3 and len(m.group(3).split()) <= 25:
        text = f"{m.group(3)} {m.group(1).lower()} {m.group(2)}."
    return text


def lowercase_start(text):
    """'The program ...' -> 'the program ...' (only for common words, never names)."""
    first = text.split(" ", 1)[0]
    return text[:1].lower() + text[1:] if first.lower() in STOPWORDS and first != "I" else text


def arrange(units_, pool, historical, rng):
    """Order units (source runs together, or by date for history sections) and group them
    into paragraphs of 2-4 sentences from one source each. Returns [[pool index]].

    Keeping each source's sentences together and in page order preserves the flow the
    author wrote; jumping between sources sentence by sentence reads like a collage."""
    if historical:
        def year_of(u):
            """Earliest year mentioned in the unit, or None."""
            ys = [int(y) for m in u for y in YEAR.findall(pool[m].text)]
            return min(ys) if ys else None
        # sort by year; an undated sentence inherits the year of the one before it on its page,
        # so it stays next to the event it follows on from (k keeps page order within a year)
        keyed, last = [], 0
        for k, u in enumerate(sorted(units_, key=lambda u: (pool[u[0]].src, pool[u[0]].pos, u[0]))):
            y = year_of(u)
            last = y if y is not None else last
            keyed.append((last, k, u))
        ordered = [u for _, _, u in sorted(keyed)]
    else:
        # the source contributing most goes first, then in page order within each source
        weight = Counter(pool[u[0]].src for u in units_)
        ordered = sorted(units_, key=lambda u: (-weight[pool[u[0]].src], pool[u[0]].src, pool[u[0]].pos, u[0]))
    # cut into paragraphs at every change of source, or after a random 2-4 sentences; a unit
    # (a leaning pair) is never split, so a paragraph can end up one sentence longer
    paras, cur, cur_src, limit = [], [], None, int(rng.integers(2, 5))
    for u in ordered:
        src = pool[u[0]].src
        if cur and (src != cur_src or len(cur) >= limit):
            paras.append(cur)
            cur, limit = [], int(rng.integers(2, 5))
        cur += list(u)
        cur_src = src
    if cur:
        paras.append(cur)
    # a lone sentence after a paragraph of the same source joins it
    merged = []
    for p in paras:
        if merged and len(p) == 1 and pool[p[0]].src == pool[merged[-1][-1]].src:
            merged[-1] += p
        else:
            merged.append(p)
    return merged


# --- 6. The paper ---------------------------------------------------------------------------------

class Composer:
    """Writes one paper: compose() runs the steps above and returns the ir.Document. The
    pool, scores, outline and selector live on the instance while the parts are built."""

    def __init__(self, notes, topic, rng, cfg):
        """A composer for `topic` from `notes`, drawing random choices from `rng`."""
        self.notes, self.topic, self.rng, self.cfg = notes, topic, rng, cfg
        self.today = cfg.today or datetime.date.today()
        self.sources = {s.id: s for s in notes.sources}
        self.used_templates = set()
        self.topic_text = ascii_fold(topic).strip()

    def pick(self, options, **fmt):
        """A random template from `options`, filled in with `fmt`; one already used in this
        paper is only repeated when all of them have been."""
        fresh = [o for o in options if o not in self.used_templates] or options
        choice = fresh[int(self.rng.integers(len(fresh)))]
        self.used_templates.add(choice)
        return choice.format(**fmt)

    def phrase(self, text):
        """`text` for use inside a sentence: common words in lower case, names as they are
        (a word is a name if the sources capitalise it mid-sentence and never write it in
        lower case)."""
        words = ascii_fold(text).split()
        return " ".join(w if (w.isupper() and len(w) > 1) or (w.lower() in self.name_words
                                                              and w.lower() not in self.lower_words)
                        else w.lower() for w in words)

    def sentence(self, i):
        """Pool sentence `i` as it will appear in the paper (lightly paraphrased)."""
        return rewrite(self.pool[i].text, self.rng, self.cfg.paraphrase)

    def paragraph_from(self, members, attribution=False, opener=None):
        """One Paragraph from pool indices; each run of sentences from one source is cited
        once, at its end."""
        texts = [self.sentence(i) for i in members]
        srcs = [self.pool[i].src for i in members]
        # "According to NASA, it ..." would cut a leaning sentence off from what it leans on
        if attribution and self.cfg.paraphrase >= 2 and not self.pool[members[0]].leans:
            texts[0] = self.pick(ATTRIBUTION, site=ascii_fold(self.sources[srcs[0]].site)) + lowercase_start(texts[0])
        for k in range(len(texts)):
            if k + 1 == len(texts) or srcs[k + 1] != srcs[k]:
                texts[k] = cite(texts[k], srcs[k])
        if opener:
            texts.insert(0, opener)
        return Paragraph([Run(" ".join(texts))])

    def compose(self):
        """The whole paper as an ir.Document, checked safe to type (ValueError if not, or if
        the notes have too little usable text).

        Content is chosen in order of importance - introduction, body sections, abstract,
        key points, closing sentence - and only then laid out in reading order. The Selector
        remembers everything chosen, so the most important parts get the best sentences and
        nothing is said twice."""
        cfg, rng = self.cfg, self.rng
        self.pool = build_pool(self.notes, cfg)
        if len(self.pool) < 8:
            raise ValueError(f"Not enough usable text in the notes ({len(self.pool)} sentences)")
        self.lower_words, self.name_words = _word_cases(self.pool)
        self.topic_np = _topic_phrase(self.pool, self.topic_text)
        tracking = any(s.seen_intervals for s in self.notes.sources)    # any page read with scroll tracking
        self.score, self.S, self.X, _ = score_pool(self.pool, self.topic_text, cfg, tracking)
        # paper size: tiny < 500 words < small < 700 < full; the frame (abstract, introduction,
        # tables, discussion, conclusion, references) costs about FRAME_WORDS of it
        self.size = "tiny" if cfg.target_words < 500 else "small" if cfg.target_words < 700 else "full"
        self.small = self.size != "full"
        n_sections = {"tiny": 2, "small": 3}.get(self.size) or (4 if cfg.target_words < 1100 else 5)
        self.sections = plan_outline(self.pool, self.score, self.X, rng, n_sections, self.topic_text)
        self.sel = Selector(self.pool, self.score, self.S, cfg)

        # choose content first (introduction, body, then the extras), lay out afterwards
        lead = [i for i, s in enumerate(self.pool) if not s.group]
        intro_units = self._intro_units(lead)
        small = self.small
        body_budget = max(100, cfg.target_words - FRAME_WORDS[self.size])
        per_section = max(50, body_budget // max(1, len(self.sections)))
        body = [(sec, self.sel.pick(sec.members, budget_words=per_section)) for sec in self.sections]
        body = [(sec, us) for sec, us in body if us]
        secs = [sec.name for sec, _ in body]
        n_abstract = 1 if small else 2
        # abstract from what's left of the leads (the page summaries), else from anywhere
        abstract_units = self.sel.pick(lead or range(len(self.pool)), max_units=n_abstract) or \
            self.sel.pick(range(len(self.pool)), max_units=n_abstract)
        key_points = self._key_points(body) if self.size != "tiny" else []
        strongest = sorted(body, key=lambda b: -b[0].strength)
        closing = self._closing_units()

        # --- layout, in reading order ---
        blocks = []
        n = len(self.notes.sources)
        sites = join_list(sorted({ascii_fold(s.site) for s in self.notes.sources}, key=str.lower)[:3])
        sec_list = join_list([quoted(s) for s in secs]) or "its main aspects"
        blocks.append(Title(title_case(self.topic_text)))
        blocks.append(Subtitle(self.pick(SUBTITLES, n=n, month_year=self.today.strftime("%B %Y"))))
        blocks.append(Paragraph([Run(f"Compiled {self.today.day} {self.today.strftime('%B %Y')}")],
                                align="center", size=10))

        blocks.append(Heading("Abstract", 1))
        abstract = [self.pick(ABSTRACT_OPEN, topic=self.topic_np, n=n, sites=sites)]
        abstract += [self.sentence(i) for u in abstract_units for i in u]
        abstract.append(self.pick(ABSTRACT_CLOSE, secs=sec_list))
        blocks.append(Paragraph([Run(" ".join(abstract))], size=10))
        keywords = self._keywords(secs)
        if keywords:
            blocks.append(Paragraph([Run("Keywords:", bold=True), Run(" " + ", ".join(keywords))], size=10))

        blocks.append(Heading("Introduction", 1))
        intro_members = [i for u in intro_units for i in u]
        if intro_members:
            para = self.paragraph_from(intro_members)
            para.runs = _bold_phrase(para.runs[0].text, self.topic_text)
            blocks.append(para)
        blocks.append(Paragraph([Run(self.pick(INTRO_ROADMAP, secs=sec_list))]))

        for sec, us in body:
            blocks.append(Heading(sec.name, 1))
            historical = bool(HISTORY_HEADINGS.search(sec.name))
            paras = arrange(us, self.pool, historical, rng)
            attrib_at = int(rng.integers(len(paras))) if paras else -1     # at most one "According to" per section
            for k, members in enumerate(paras):
                opener = None
                if k == 0 and rng.random() < 0.35:
                    opener = self.pick(SECTION_OPEN, h=quoted(sec.name))
                blocks.append(self.paragraph_from(members, attribution=(k == attrib_at), opener=opener))

        tables = self._tables()
        if tables:
            blocks.append(Heading("Key Facts", 1))
            for caption, table in tables:
                blocks += [Caption(caption), table]

        blocks.append(Heading("Discussion", 1))
        blocks.append(Paragraph([Run(" ".join(self._discussion()))]))
        if key_points:
            blocks.append(Heading("Key Points", 2))
            blocks.append(Bullets(key_points))

        blocks.append(Heading("Conclusion", 1))
        conclusion = [self.pick(CONCLUDE_OPEN, topic=self.topic_np)]
        if closing:
            members = closing[0]
            conclusion += [self.sentence(i) for i in members]
            conclusion[-1] = cite(conclusion[-1], self.pool[members[0]].src)
        weak = strongest[-1][0].name if strongest else "the topic"      # thinnest section = "further work"
        conclusion.append(self.pick(CONCLUDE_CLOSE, weak=quoted(weak) if body else weak))
        blocks.append(Paragraph([Run(" ".join(conclusion))]))

        # citations are numbered once all the text is in place, then the reference list follows
        doc = Document(blocks, meta={"topic": self.topic_text, "sections": secs})
        numbers = number_citations(doc, [s.id for s in self.notes.sources])
        doc.blocks.append(PageBreak())
        doc.blocks.append(Heading("References", 1))
        for src_id, k in sorted(numbers.items(), key=lambda kv: kv[1]):
            doc.blocks.append(self._reference(k, self.sources[src_id]))
        if not self.small:
            doc.blocks.append(Heading("Appendix: Sources Consulted", 1))
            doc.blocks.append(Caption("Table A1. Web pages read for this report."))
            doc.blocks.append(self._sources_table(numbers))
        doc.meta["references"] = {str(k): self.sources[s].url for s, k in numbers.items()}
        problems = ir.validate(doc)                     # last line of defence before anything is typed
        if problems:
            raise ValueError(f"Composed document is not safe to type: {problems[:3]}")
        return doc

    # --- parts -----------------------------------------------------------------------------

    def _intro_units(self, lead):
        """A defining first sentence ('X is a ...') from a lead, then two more lead sentences."""
        defining = [i for i in lead if self.pool[i].index == 0 and not self.pool[i].leans
                    and re.search(r"\b(is|was|are|were)\b", " ".join(self.pool[i].text.split()[:25]))]
        picked = []
        if defining:
            first_src = self.notes.sources[0].id          # usually the encyclopedia page read first
            best = max(defining, key=lambda i: (self.pool[i].src == first_src, self.score[i]))
            self.sel.chosen.append(best)                  # taken outright, not by MMR
            picked.append((best,))
            same = [i for i in lead if self.pool[i].src == self.pool[best].src]
            picked += self.sel.pick(same, max_units=1 if self.small else 2)
        else:
            picked += self.sel.pick(lead, max_units=3)
        # the first unit's source first, each source in page order
        return sorted(picked, key=lambda u: (self.pool[u[0]].src != self.pool[picked[0][0]].src, u[0])) \
            if picked else []

    def _closing_units(self):
        """A sentence that sums something up ('... remains one of the most ...') for the
        conclusion: evaluative wording and closeness to the topic count, dated narrative
        ('On July 20, 1969, ...') is avoided."""
        candidates = [i for i, s in enumerate(self.pool)
                      if not s.leans and not _DATE_PHRASE.match(s.text) and i not in self.sel.chosen]
        if not candidates:
            return []
        boosted = sorted(candidates,
                         key=lambda i: -self.score[i] * (1.6 if EVALUATIVE.search(self.pool[i].text) else 1.0))
        return self.sel.pick(boosted[:12], max_units=1)

    def _key_points(self, body):
        """One short, strong sentence per body section, labelled with the section name."""
        items = []
        for sec, _ in body:
            short = [i for i in sec.members if len(self.pool[i].text.split()) <= 22 and not self.pool[i].leans]
            got = self.sel.pick(short, max_units=1)
            if got:
                i = got[0][0]
                items.append([Run(f"{sec.name}:", bold=True), Run(" " + cite(self.sentence(i), self.pool[i].src))])
        items = items[:3] if self.small else items
        return items if len(items) >= 2 else []

    def _keywords(self, secs):
        """Up to 6 keywords: the topic, then the names mentioned by the most sources.
        (`secs` is not used.)"""
        words = []
        topic = re.sub(r"^the ", "", self.topic_np)
        for p in [topic] + _shared_names(self.pool, 5, self.topic_text):
            if p.lower() not in {w.lower() for w in words} and not ir.check_text(p):
                words.append(p)
        return words[:6]

    def _discussion(self):
        """Sentences comparing the sources (all from templates): what they share, where two
        of them put their weight, and that they agree."""
        out = []
        shared = [_topic_phrase(self.pool, n) for n in _shared_names(self.pool, 3, self.topic_text)]
        if len(shared) >= 2:
            out.append(self.pick(SHARED, things=join_list(shared)))
        # each source's focus = its two longest sections
        focus = []
        for src in self.notes.sources[:4]:
            heads = sorted({s.heading for s in src.sections if s.heading and _section_name(s.heading)},
                           key=lambda h: -sum(x.words() for x in src.sections if x.heading == h))
            heads = [self.phrase(_section_name(h)) for h in heads[:2]]
            if heads:
                focus.append((ascii_fold(src.site), join_list(heads)))
        distinct = [f for k, f in enumerate(focus) if f[0] not in {g[0] for g in focus[:k]}]   # one per site
        if len(distinct) >= 2 and distinct[0][1] != distinct[1][1]:
            (a, fa), (b, fb) = distinct[:2]
            out.append(self.pick(FOCUS, a=a, fa=fa, b=b, fb=fb))
        out.append(self.pick(AGREE))
        return out

    def _tables(self):
        """Key-fact tables: an infobox from a source, else a timeline of dated events, plus a
        small data table when one is clean enough. [(caption, Table)]"""
        tables = []
        info = self._infobox_table()
        if info:
            tables.append(info)
        # a timeline replaces a missing infobox, and sometimes joins one in a full-size paper
        timeline = self._timeline_table()
        if timeline and (not tables or (not self.small and self.rng.random() < 0.5)):
            tables.append(timeline)
        if len(tables) < (1 if self.small else 2):     # at most 1 table in a small paper, 2 in a full one
            data = self._data_table()
            if data:
                tables.append(data)
        for k, (caption, _) in enumerate(tables):       # number the captions in order
            tables[k] = (f"Table {k + 1}. {caption}", tables[k][1])
        return tables

    def _infobox_table(self):
        """(caption, Table) of attribute/value rows from the first source with a usable
        infobox (3+ short, clean rows), or None."""
        for src in self.notes.sources:
            for t in src.tables:
                if t.kind != "infobox":
                    continue
                rows = []
                for row in t.rows:
                    cells = [c for c in row if c]
                    if len(cells) != 2:
                        continue                        # section titles, images, multi-value rows
                    key, value = (_cell(c) for c in cells)
                    if key and value and len(key.split()) <= 4 and len(value.split()) <= 12 and key != value:
                        rows.append([key, value])
                rows = _unique_rows(rows)[:10]
                if len(rows) >= 3:
                    caption = f"Key facts about {self.topic_np} (source: {ascii_fold(src.site)} " \
                              f"{CITE.format(src.id)})."
                    return caption, Table([["Attribute", "Value"]] + rows)
        return None

    def _timeline_table(self):
        """(caption, Table) of Year / Event / Source built from pool sentences that open with
        a date ("In 1961, ..."), one event per year, the 8 best-scoring; None if under 3."""
        events = {}                                     # year -> (score, event text, source id)
        for i, s in enumerate(self.pool):
            m = _DATE_PHRASE.match(s.text)
            if not m:
                continue
            year = m.group(7) + (m.group(8) or "")      # "1960" or "1960s"
            event = s.text[m.end():].rstrip(".")        # the sentence without its date phrase
            if len(event.split()) > 14:                 # cut a trailing clause, never a list
                cut = re.search(r",\s(which|who|where|while|when|after|before|making|forcing|allowing|"
                                r"a|an|the)\s", event)
                event = event[:cut.start()] if cut else event
            if not (4 <= len(event.split()) <= 14) or LEANS_ON_PREVIOUS.match(event):
                continue                                # too short/long for a cell, or "it ..." without context
            event = _cell(event)
            if event and (year not in events or self.score[i] > events[year][0]):
                events[year] = (self.score[i], event, s.src)
        if len(events) < 3:
            return None
        best = sorted(events.items(), key=lambda kv: -kv[1][0])[:8]
        rows = [[y, e, ascii_fold(self.sources[src].site)] for y, (_, e, src) in sorted(best)]   # by year
        return f"Timeline of {self.topic_np}.", Table([["Year", "Event", "Source"]] + rows)

    def _data_table(self):
        """(caption, Table) copying the first small, clean data table from a source (3-12 rows,
        2-5 columns, every cell short and safe to type), or None."""
        for src in self.notes.sources:
            for t in src.tables:
                if t.kind != "data":
                    continue
                rows = [[_cell(c) for c in row] for row in t.rows]
                if not (3 <= len(rows) <= 12) or not (2 <= len(rows[0]) <= 5):
                    continue
                if any(not c or len(c.split()) > 8 for row in rows for c in row):
                    continue                            # an empty cell may be one _cell() rejected
                return f"Data reported by {ascii_fold(src.site)} {CITE.format(src.id)}.", Table(rows)
        return None

    def _sources_table(self, numbers):
        """Appendix table of the pages read, in reference-number order, with words read."""
        rows = [["No.", "Source", "Site", "Words read"]]
        for src_id, k in sorted(numbers.items(), key=lambda kv: kv[1]):
            src = self.sources[src_id]
            title = _cell(src.title, max_words=8) or "Untitled page"
            rows.append([str(k), title, ascii_fold(src.site), f"{src.words_read:,}"])
        return Table(rows)

    def _reference(self, k, src):
        """Reference-list entry k for a web page, APA-like:
        '[k] Title. Site. Retrieved Month D, YYYY, from URL' (URL left out if it can't be typed)."""
        title = _cell(src.title, max_words=20) or "Untitled page"
        accessed = datetime.date.fromisoformat(src.accessed) if src.accessed else self.today
        date = f"{accessed.strftime('%B')} {accessed.day}, {accessed.year}"
        tail = f". {ascii_fold(src.site)}. Retrieved {date}"
        url = src.url if src.url and is_ascii(src.url) and not ir.check_text(src.url) else ""
        tail += f", from {url}" if url else "."
        return Paragraph([Run(f"[{k}] "), Run(title, italic=True), Run(tail)], size=10)


# --- helpers ----------------------------------------------------------------------------------------

def number_citations(doc, all_source_ids):
    """Replace citation placeholders by reference numbers in order of first citation;
    sources never cited are numbered after. Returns {source id: number}."""
    numbers = {}
    pattern = re.compile("\x00(\\d+)\x00")

    def assign(text):
        """`text` with its placeholders replaced, numbering sources on first sight."""
        for m in pattern.finditer(text):
            numbers.setdefault(int(m.group(1)), len(numbers) + 1)
        return pattern.sub(lambda m: f"[{numbers[int(m.group(1))]}]", text)

    # walk every text-bearing block in document order, so [1] is the first source cited
    # in the text, as a reader expects (table captions count too)
    for b in doc.blocks:
        if isinstance(b, (Title, Subtitle, Heading, Caption)):
            b.text = assign(b.text)
        elif isinstance(b, Paragraph):
            for r in b.runs:
                r.text = assign(r.text)
        elif isinstance(b, Bullets):
            for item in b.items:
                for r in item:
                    r.text = assign(r.text)
        elif isinstance(b, Table):
            b.rows = [[assign(c) for c in row] for row in b.rows]
    for sid in all_source_ids:
        numbers.setdefault(sid, len(numbers) + 1)       # read but never quoted: still listed
    return numbers


def _cell(text, max_words=None):
    """Table-cell text: ASCII, safe to type, first letter capitalised (Writer's AutoCorrect
    would capitalise it anyway), no final period."""
    text = ascii_fold(clean_line(text or "")).strip().rstrip(".")
    if max_words and len(text.split()) > max_words:
        text = " ".join(text.split()[:max_words])
    if not text or not is_ascii(text) or ir.check_text(text) or ir._AUTOFORMAT_START.match(text):
        return ""
    return text[:1].upper() + text[1:]


def _unique_rows(rows):
    """`rows` without repeats of a first cell (case-insensitive), keeping the first."""
    seen, out = set(), []
    for r in rows:
        if r[0].lower() not in seen:
            seen.add(r[0].lower())
            out.append(r)
    return out


def _word_cases(pool):
    """(words the sources write in lower case, words they capitalise mid-sentence)."""
    lower, names = set(), set()
    for sent in pool:
        for k, w in enumerate(re.findall(r"[A-Za-z][A-Za-z'-]*", sent.text)):
            if w.islower():
                lower.add(w)
            elif k > 0 and w[:1].isupper():
                names.add(w.lower())
    return lower, names


def _topic_phrase(pool, topic):
    """How the sources write the topic mid-sentence, with 'the' if they usually use it:
    'Apollo program' -> 'the Apollo program'."""
    pattern = re.compile(r"(\bthe\s+)?\b(" + re.escape(topic) + r")\b", re.I)
    spellings, with_the, total = Counter(), 0, 0
    for sent in pool:
        for m in pattern.finditer(sent.text):
            total += 1
            with_the += bool(m.group(1))
            if m.start(2) > 0:
                spellings[m.group(2)] += 1
    text = spellings.most_common(1)[0][0] if spellings else topic
    numbered = bool(re.search(r"\s(\d+|[IVX]+)$", text))         # "Apollo 11", "Saturn V": no article
    return f"the {text}" if total >= 2 and with_the / total >= 0.5 and not numbered else text


def _bold_phrase(text, phrase):
    """Runs with the first occurrence of `phrase` in bold."""
    m = re.search(re.escape(phrase), text, re.I)
    if not m:
        return [Run(text)]
    runs = [Run(text[:m.start()]), Run(text[m.start():m.end()], bold=True), Run(text[m.end():])]
    return [r for r in runs if r.text]


# a capitalised word or acronym, plus following capitalised words, short numbers and "of"/"for"
# ("Neil Armstrong", "Apollo 11", "Bureau of Standards")
_NAME = re.compile(r"\b([A-Z][A-Za-z]+|[A-Z]{2,})((?:\s+(?:(?:of|for)\s+)?(?:[A-Z][A-Za-z]+|[A-Z]{1,}|\d{1,3}\b))*)")


def _shared_names(pool, k, topic):
    """Names (capitalised phrases, not at a sentence start) mentioned by the most sources."""
    by_src = defaultdict(set)
    counts = Counter()
    topic_l = topic.lower()
    for s in pool:
        for m in _NAME.finditer(s.text):
            if m.start() == 0:
                continue                                # any word is capitalised there
            name = (m.group(1) + m.group(2)).strip()
            if name.split()[0] in MONTHS or name.lower() in STOPWORDS or name.lower() in topic_l or len(name) < 3:
                continue
            by_src[name].add(s.src)
            counts[name] += 1
    # most sources first, multi-word names before single words, then most mentions
    ranked = sorted(counts, key=lambda n: (-len(by_src[n]), -(" " in n), -counts[n]))
    shared = [n for n in ranked if len(by_src[n]) >= 2] or ranked
    out = []
    for n in shared:
        if not any(n in o or o in n for o in out):     # "Armstrong" adds nothing after "Neil Armstrong"
            out.append(n)
        if len(out) >= k:
            break
    return out


def plan_queries(topic, notes=None):
    """Searches a researcher would run: the topic on Wikipedia and Britannica, then the
    topic plus the main headings of the first page read (fallback: generic facets).
    [(query, link hint or None)]"""
    queries = [(f"{topic} wikipedia", "wikipedia"), (f"{topic} britannica", "britannica")]
    facets = []
    if notes and notes.sources:
        first = notes.sources[0]
        top = min((s.level for s in first.sections if s.heading), default=2)
        heads = [s for s in first.sections if s.heading and s.level == top and len(s.heading.split()) <= 4
                 and not MINOR_SECTIONS.match(s.heading)]
        heads.sort(key=lambda s: -s.words())
        facets = [s.heading.lower() for s in heads[:3]]
    for f in facets + ["history", "significance", "facts"]:
        q = f"{topic} {f}"
        if q.lower() not in {x.lower() for x, _ in queries}:
            queries.append((q, None))
    return queries


def compose_paper(notes, topic, rng=None, cfg=None):
    """The research paper (ir.Document) written from `notes`."""
    rng = rng if rng is not None else np.random.default_rng()
    return Composer(notes, topic, rng, cfg or ComposeConfig()).compose()


def words_for_time(seconds, wpm):
    """Roughly how many words this person can write (typing, thinking, formatting) in
    `seconds` - same effective rate as tasks.Workspace._next_chunk, plus ~25% for formatting."""
    chars_per_s = wpm * 5 / 60 / 1.8                 # 5 chars per word; /1.8 for thinking pauses
    return int(seconds * chars_per_s / 6.0 / 1.25)    # ~6 chars per word with the space
