"""
Where "change of idea" wording comes from.

A false start types something *other* than the target text, pauses, deletes back
to where it diverged and continues with the real text. The planner asks a provider
for that "something other":

    provider.alternative(target, start, approx_len, rng) -> str | None

`start` is a word-start index in `target`; the returned string is what the writer
first typed at that point (it must differ from target[start:] early on).

HeuristicAlternatives   offline: synonym swaps, different sentence openers,
                        anticipating a later word, abandoned partial words.
LLMAlternatives         one Claude call per text (before typing starts) returns
                        plausible first-draft phrasings at chosen positions;
                        falls back to the heuristics on any failure.
"""

import json
import os
import re

_WORD_RE = re.compile(r"\S+\s*")

SYNONYMS = {
    "think": ["believe", "feel", "guess"], "believe": ["think", "feel"], "need": ["want", "have"],
    "want": ["need", "would like"], "help": ["assist", "support"], "get": ["receive", "have", "grab"],
    "also": ["additionally", "and"], "but": ["however", "though"], "so": ["therefore", "thus"],
    "big": ["large", "huge"], "large": ["big", "huge"], "small": ["little", "minor"],
    "quick": ["brief", "short"], "please": ["kindly", "could you"], "thanks": ["thank you", "cheers"],
    "meeting": ["call", "session", "catch-up"], "important": ["key", "critical", "crucial"],
    "problem": ["issue", "challenge"], "issue": ["problem", "concern"], "start": ["begin", "kick off"],
    "begin": ["start"], "finish": ["complete", "wrap up"], "complete": ["finish"], "show": ["present", "share"],
    "send": ["forward", "share"], "check": ["review", "look at", "verify"], "review": ["check", "go over"],
    "make": ["create", "build"], "create": ["make", "build"], "use": ["utilise", "try"],
    "good": ["great", "nice", "solid"], "great": ["good", "excellent"], "bad": ["poor", "weak"],
    "many": ["lots of", "several", "a lot of"], "some": ["a few", "several"], "very": ["really", "quite"],
    "really": ["very", "truly"], "because": ["since", "as"], "since": ["because", "as"],
    "however": ["but", "that said"], "maybe": ["perhaps", "possibly"], "perhaps": ["maybe"],
    "soon": ["shortly", "later"], "now": ["currently", "today"], "today": ["this morning", "now"],
    "tomorrow": ["later", "next week"], "update": ["status", "news"], "report": ["summary", "document"],
    "idea": ["thought", "plan", "suggestion"], "plan": ["idea", "proposal"], "change": ["update", "modify"],
    "improve": ["enhance", "fix"], "people": ["users", "everyone", "staff"], "team": ["group", "everyone"],
    "should": ["could", "might", "must"], "could": ["should", "might", "can"], "can": ["could", "will"],
    "will": ["can", "would"], "would": ["will", "could"], "might": ["may", "could"], "may": ["might"],
    "many people": ["most people"], "understand": ["see", "get"], "explain": ["describe", "clarify"],
    "ask": ["check with", "request"], "find": ["see", "discover"], "look": ["check", "see"],
    "easy": ["simple", "straightforward"], "hard": ["difficult", "tough"], "difficult": ["hard", "tricky"],
    "fast": ["quick", "rapid"], "slow": ["sluggish", "delayed"], "new": ["latest", "updated"],
    "old": ["previous", "earlier"], "different": ["other", "various"], "same": ["similar", "identical"],
    "first": ["initially", "to begin"], "finally": ["lastly", "in the end"], "then": ["after that", "next"],
    "the": ["this", "our", "a"], "this": ["the", "that"], "that": ["this", "which"], "a": ["the", "one"],
    "we": ["I", "our team"], "I": ["we"], "you": ["your team", "we"], "it": ["this", "that"],
    "is": ["was", "seems"], "are": ["were", "seem"], "was": ["is", "had been"], "have": ["got", "had"],
}

OPENERS = ["I think", "Actually,", "Just", "So", "Also,", "In addition,", "I was wondering if",
           "It seems", "Basically,", "To be honest,", "Well,", "Hopefully", "Please", "I just", "We"]

COMMON = ["the", "it", "we", "this", "and", "so", "but", "i", "to", "there"]


def _words(text):
    return _WORD_RE.findall(text)


def _match_case(word, like):
    if like[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def _split_punct(w):
    core = w.rstrip()
    trail = w[len(core):]
    m = re.match(r"^(\W*)(.*?)(\W*)$", core)
    return m.group(1), m.group(2), m.group(3) + trail


def at_clause_start(target, start):
    before = target[:start].rstrip(" ")
    return start == 0 or not before or before[-1] in ".?!,;:\n"


class HeuristicAlternatives:
    name = "heuristic"

    def prepare(self, text, rng):
        return None

    def positions(self):
        return None

    def alternative(self, target, start, approx_len, rng):
        words = _words(target[start:])
        if not words:
            return None
        n = 1
        while n < len(words) and len("".join(words[:n])) < approx_len:
            n += 1
        seg_words = words[:n]
        strategies = ["synonym", "opener", "anticipate", "abandon"]
        rng.shuffle(strategies)
        for s in strategies:
            alt = getattr(self, "_" + s)(target, start, seg_words, words, rng)
            if alt and not target[start:].startswith(alt):
                return alt
        return None

    # Each strategy returns the text first typed at `start`, or None.

    def _synonym(self, target, start, seg, words, rng):
        idx = [i for i, w in enumerate(seg) if _split_punct(w)[1].lower() in SYNONYMS]
        if not idx:
            return None
        i = idx[rng.integers(len(idx))]
        pre, core, post = _split_punct(seg[i])
        choices = SYNONYMS[core.lower()]
        rep = _match_case(choices[rng.integers(len(choices))], core)
        # the writer notices the change shortly after typing the replaced word
        tail = ""
        if i + 1 < len(words) and rng.random() < 0.4:
            nxt = words[i + 1]
            tail = nxt[:rng.integers(1, len(nxt) + 1)]
        return "".join(seg[:i]) + pre + rep + post + tail

    def _opener(self, target, start, seg, words, rng):
        if not at_clause_start(target, start):
            return None
        op = OPENERS[rng.integers(len(OPENERS))]
        first = seg[0]
        if start == 0 or target[:start].rstrip()[-1:] in ".?!\n":
            op = _match_case(op, "X")
            first = first[:1].lower() + first[1:] if first[:2] != "I " else first
        else:
            op = op[:1].lower() + op[1:]
        extra = first[:rng.integers(1, len(first) + 1)] if rng.random() < 0.5 else ""
        return op + " " + extra

    def _anticipate(self, target, start, seg, words, rng):
        if len(words) < 2:
            return None
        nxt = words[1].rstrip()
        if len(nxt) < 2:
            return None
        return nxt[:rng.integers(2, len(nxt) + 1)]

    def _abandon(self, target, start, seg, words, rng):
        _, core, _ = _split_punct(seg[0])
        opts = SYNONYMS.get(core.lower()) or [w for w in COMMON if w != core.lower()]
        w = _match_case(opts[rng.integers(len(opts))], core or "x")
        return w[:max(1, rng.integers(1, len(w) + 1))]


class LLMAlternatives:
    """
    Ask Claude once per text for plausible first-draft wordings, then serve them.
    Requires the `anthropic` package and credentials (ANTHROPIC_API_KEY or an
    `ant auth login` profile). Any failure -> heuristic fallback, never an exception.
    """

    name = "llm"
    DEFAULT_MODEL = os.environ.get("TYPING_ALT_MODEL", "claude-opus-5")

    def __init__(self, model=None, timeout=20.0, per_words=12, fallback=None):
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self.per_words = per_words
        self.fallback = fallback or HeuristicAlternatives()
        self._alts = {}          # char index -> [alternative strings]
        self.error = None

    def prepare(self, text, rng):
        """Fetch alternatives for `text`. Returns True on success."""
        self._alts = {}
        n_words = len(_words(text))
        want = max(1, round(n_words / self.per_words))
        try:
            import anthropic

            client = anthropic.Anthropic(timeout=self.timeout, max_retries=1)
            prompt = (
                "You are simulating how a person drafts text while typing it. For the final text "
                f"below, propose {want} realistic moments where the writer first typed something "
                "different and then deleted it (a change of wording, a different sentence opener, "
                "a wrong word, a restarted clause).\n"
                "For each, give `anchor`: an exact substring of the final text that starts at a word "
                "boundary (3-8 words, copied verbatim), and `first_draft`: what the writer typed "
                "at that point before changing their mind (2-8 words, must start differently from "
                "the anchor, same tone, may be cut off mid-word).\n\nFinal text:\n" + text
            )
            response = client.beta.messages.create(
                model=self.model,
                max_tokens=2000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={
                    "effort": "low",
                    "format": {
                        "type": "json_schema",
                        "schema": {
                            "type": "object",
                            "properties": {"edits": {"type": "array", "items": {
                                "type": "object",
                                "properties": {"anchor": {"type": "string"},
                                               "first_draft": {"type": "string"}},
                                "required": ["anchor", "first_draft"],
                                "additionalProperties": False}}},
                            "required": ["edits"],
                            "additionalProperties": False,
                        },
                    },
                },
                messages=[{"role": "user", "content": prompt}],
            )
            if response.stop_reason == "refusal":
                raise RuntimeError("model declined")
            raw = next(b.text for b in response.content if b.type == "text")
            for e in json.loads(raw).get("edits", []):
                idx = text.find(e["anchor"])
                draft = e["first_draft"]
                if idx < 0 or not draft or text[idx:].startswith(draft):
                    continue
                self._alts.setdefault(idx, []).append(draft)
            self.error = None
            return bool(self._alts)
        except Exception as exc:     # noqa: BLE001 - any failure means "use heuristics"
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    def positions(self):
        """Word-start indices with LLM alternatives, or None to let the planner choose."""
        return sorted(self._alts) if self._alts else None

    def alternative(self, target, start, approx_len, rng):
        alts = self._alts.get(start)
        if alts:
            return alts[rng.integers(len(alts))]
        return self.fallback.alternative(target, start, approx_len, rng)


def make_provider(kind="heuristic", **kwargs):
    if kind == "llm":
        return LLMAlternatives(**kwargs)
    return HeuristicAlternatives()
