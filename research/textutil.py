"""
Text helpers for research notes: cleaning scraped page text, folding it to ASCII,
splitting sentences and tokenising.

ASCII matters because controller.keyboard.type_char *pastes* any character it can't
press as a plain key. A paper full of en dashes, curly quotes and accented names would
paste in visible chunks, so everything that will be typed goes through ascii_fold()
first, and sentences that still aren't ASCII are left out.
"""

import os
import re
import unicodedata

DATA = os.path.join(os.path.dirname(__file__), "data")


def load_list(name):
    """Non-empty, non-comment lines of a bundled data file."""
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def load_map(name):
    """A bundled tab-separated file as a list of (key, value) pairs, in file order."""
    pairs = []
    for ln in load_list(name):
        if "\t" in ln:
            k, v = ln.split("\t", 1)
            pairs.append((k.strip(), v.strip()))
    return pairs


STOPWORDS = frozenset(load_list("stopwords.txt"))
ABBREVIATIONS = frozenset(load_list("abbreviations.txt"))

# --- ASCII folding ------------------------------------------------------------------------

_FOLD = str.maketrans({
    "–": "-", "—": " - ", "‒": "-", "‐": "-", "‑": "-", "−": "-",
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "…": "...", " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "​": "", "‌": "", "‍": "", "⁠": "", "﻿": "", "­": "",
    "°": " degrees", "×": "x", "·": "-", "•": "-", "½": "half",
    "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss", "ø": "o",
    "Ø": "O", "ł": "l", "Ł": "L", "ð": "d", "þ": "th", "≤": "<=",
    "≥": ">=", "≈": "about ", "±": "+/-", "€": "EUR ", "£": "GBP ",
})


def ascii_fold(text):
    """Closest ASCII spelling: dashes, quotes and spaces mapped, accents dropped.
    Characters with no ASCII form are kept (check with is_ascii())."""
    text = unicodedata.normalize("NFKD", text.translate(_FOLD))        # table first, then split accents off
    text = "".join(ch for ch in text if not unicodedata.combining(ch))   # "é" -> "e" + accent -> "e"
    return re.sub(r"[ ]{2,}", " ", text)


def is_ascii(text):
    """True if every character is plain 7-bit ASCII (so the keyboard can press it)."""
    return all(ord(ch) < 128 for ch in text)


# --- Cleaning scraped text --------------------------------------------------------------------

_CITE = re.compile(
    r"\s?\[(?:\d+(?:[-,–]\s?\d+)*|[a-z]{1,2}|note \d+|nb \d+|n \d+|citation needed|clarification needed|"
    r"when\?|who\?|which\?|where\?|according to whom\?|better source needed|page needed|"
    r"full citation needed|dubious(?: [^\]]*)?|failed verification|verification needed|edit|update)\]",
    re.IGNORECASE)
_PAREN = re.compile(r"\s*\(([^()]*)\)")
_IPA_CHARS = set("ɐɑɒɔəɚɛɜɞɟɡɣɥɨɪ"
                 "ɫɯɰɱɲɳɴɸɹɾʀʁʂʃʈ"
                 "ʉʊʋʌʍʎʒʔʕˈˌːˑθð")
_PAREN_JUNK = re.compile(r"/|listen|pronounced|pronunciation|\bIPA\b|lit\.|romani[sz]ed|born ", re.IGNORECASE)


def strip_citations(text):
    """Remove footnote markers such as [12], [a], [note 3], [citation needed]."""
    return _CITE.sub("", text)


def _bad_paren(inner):
    """True if the text inside a pair of parentheses is noise for a reader: empty, a
    pronunciation (IPA, "listen"), a transliteration, or mostly foreign script."""
    if not inner.strip():
        return True
    if _PAREN_JUNK.search(inner) or any(ch in _IPA_CHARS for ch in inner):
        return True
    non_ascii = sum(1 for ch in ascii_fold(inner) if ord(ch) >= 128)    # what folding can't rescue
    return non_ascii > 0.2 * len(inner) or (";" in inner and non_ascii)


def strip_bad_parens(text):
    """Remove parentheses holding pronunciations, transliterations or foreign script."""
    return _PAREN.sub(lambda m: "" if _bad_paren(m.group(1)) else m.group(0), text)


def clean_line(text):
    """One line of scraped page text, ready to split into sentences."""
    text = strip_bad_parens(strip_citations(text.replace("\ufffc", "")))   # U+FFFC: Chrome's image placeholder
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    text = re.sub(r"([a-z0-9][.!?])([A-Z][a-z])", r"\1 \2", text)    # "1961.[3]Thereafter" -> "1961. Thereafter"
    text = re.sub(r"\(\s+", "(", text)
    return text


def alpha_share(text):
    """Fraction of non-space characters that are letters."""
    chars = [ch for ch in text if not ch.isspace()]
    return sum(ch.isalpha() for ch in chars) / len(chars) if chars else 0.0


# --- Sentences --------------------------------------------------------------------------------

_BOUNDARY = re.compile(r"""[.!?]["')\]]*\s+(?=["'(]?[A-Z0-9])""")


def _ends_with_abbreviation(chunk, following=""):
    """True if `chunk` (text up to and including a period) ends in an abbreviation or an
    initial ("John F. Kennedy"). A single capital followed by a word that typically starts
    a sentence ("Saturn V. The ...", "... V. Dr. Braun") ends the sentence instead."""
    m = re.search(r"([A-Za-z][A-Za-z.]*)\.$", chunk)
    if not m:
        return False
    token = m.group(1)
    if token in ABBREVIATIONS or token.rstrip(".") in ABBREVIATIONS:
        return True
    if len(token) == 1 and token.isupper():
        nxt = re.match(r"[\"'(]?([A-Za-z]+)", following)
        word = nxt.group(1) if nxt else ""
        return not (word.lower() in STOPWORDS or word in ABBREVIATIONS)
    return False


def split_sentences(text):
    """Split a paragraph into sentences (abbreviations, initials and decimals are protected)."""
    text = re.sub(r"\s+", " ", text).strip()
    out, start = [], 0
    # a boundary is end punctuation + space + a capital/digit; "3.5" never matches (no space)
    for m in _BOUNDARY.finditer(text):
        end = m.end()
        piece = text[start:end].rstrip()
        if piece.endswith(".") and _ends_with_abbreviation(piece, text[end:end + 20]):
            continue                    # "Dr." / "F." - keep going, the sentence isn't over
        out.append(piece)
        start = end
    if start < len(text):
        out.append(text[start:].strip())
    return [s for s in out if s]


# --- Tokens -----------------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")


def stem(word):
    """Crude suffix stripping, enough to match 'missions' with 'mission'."""
    if len(word) <= 4 or word.isdigit():
        return word
    # first matching suffix that leaves 4+ letters wins ("based" stays, "landed" -> "land")
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)] + repl
    return word


def tokenize(text):
    """Lower-case content-word stems (stopwords and 1-letter tokens dropped)."""
    return [stem(w) for w in _WORD.findall(text.lower()) if len(w) > 1 and w not in STOPWORDS]


def word_count(text):
    """Number of whitespace-separated words."""
    return len(text.split())


# --- Small formatting helpers -----------------------------------------------------------------

_SMALL = {"a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on", "or", "the", "to", "vs", "with"}


def title_case(text):
    """Headline capitalisation that leaves acronyms and already-capitalised words alone."""
    words = text.split()
    out = []
    for i, w in enumerate(words):
        if 0 < i < len(words) - 1 and w.lower() in _SMALL:
            out.append(w.lower())
        elif w[:1].islower():
            out.append(w[:1].upper() + w[1:])
        else:
            out.append(w)
    return " ".join(out)


def slugify(text, max_len=40):
    """File-name-safe form of `text`: 'Apollo program!' -> 'apollo-program' ("research" if empty)."""
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_fold(text).lower()).strip("-")
    return slug[:max_len].rstrip("-") or "research"


def join_list(items, conj="and"):
    """'a', 'a and b', 'a, b and c'."""
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {conj} " + items[-1]
