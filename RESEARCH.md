# Research Mode: Algorithms and Findings

Research mode turns the Synthetic Human Agent into a researcher. It runs in five stages:

1. It searches a topic in Chrome.
2. It takes notes from every page it reads.
3. It writes a research paper from those notes.
4. It types the paper into OpenOffice Writer (Apache OpenOffice 4.1) with real keyboard and mouse input. The paper has a title, abstract, headings, paragraphs with citations, bold and italic text, font sizes, tables, bullets and references.
5. It checks its own work through UI Automation.

This document explains how each stage works and why it was built that way. It also records what live testing on a real desktop showed. The behaviour layer that the stages run on (mouse model, typing model, reading, personas) is described in [ALGORITHMS.md](ALGORITHMS.md). The low-level controller is described in [CONTROLLER.md](CONTROLLER.md).

---

## Contents
1. [Constraints and design decisions](#1-constraints-and-design-decisions)
2. [Quick start](#2-quick-start)
3. [Pipeline overview](#3-pipeline-overview)
4. [Reading pages: `extract.py`](#4-reading-pages-extractpy)
5. [Notes and search planning: `notes.py`, `plan_queries`](#5-notes-and-search-planning)
6. [Writing the paper without a language model: `compose.py`, `rank.py`](#6-writing-the-paper-without-a-language-model)
7. [From paper to Writer operations: `ir.py`, `writer_ops.py`, `fake_writer.py`](#7-from-paper-to-writer-operations)
8. [Performing the operations in Writer: `writer_exec.py`, `controller/writer.py`](#8-performing-the-operations-in-writer)
9. [Orchestration: `project.py`, `main.py`, the Session](#9-orchestration)
10. [Self-checks: `diagnostics.py`](#10-self-checks)
11. [Testing](#11-testing)
12. [Findings from live testing](#12-findings-from-live-testing)
13. [Limitations and next steps](#13-limitations-and-next-steps)

---

## 1. Constraints and design decisions

| Constraint | Consequence |
|---|---|
| The target machine has 4 GB RAM. A local language model doesn't fit. | Writing is extractive and classical: TF-IDF, TextRank, MMR and k-means in plain numpy. Peak memory stays in the tens of MB. |
| Internet access is very restricted, so there are no online LLM APIs. | Everything runs offline. The word lists and phrase maps are small files bundled in `research/data/`. |
| The point of the project is realistic human activity. | Every action in Writer is real input: keystrokes, shortcuts, toolbar and menu clicks and dialogs. No UNO, COM or file generation is used. Pages are read through UI Automation, the accessibility interface screen readers use. Nothing is injected into Chrome, and Chrome needs no debugging port. |
| Typing must look like a person typing. | Text goes through the learned typing model (`algorithms/human_typing.py`): typos and fixes, thinking pauses, false starts. One formatted run is typed per call, so the model's corrections never cross a formatting boundary. |
| Anything that can't be typed as plain keys gets pasted (`controller/keyboard.type_char`). | All text is folded to ASCII before typing. Sentences that still contain non-ASCII characters are skipped. |

The writing **reads like a careful student's compiled report**, not original analysis. Every sentence in the body is a real sentence from a source, chosen and arranged by the composer and cited. The frame around those sentences (abstract opening, roadmap, discussion, conclusion) comes from small sets of templates.

---

## 2. Quick start

```bash
python main.py --process research --topic "Apollo program" --minutes 45      # browse, take notes, write the paper
python main.py --process research --resume runs/research/apollo-program-...   # carry on with an earlier run
python main.py --process session --topic "Apollo program" --minutes 30        # the random Session does the research
python main.py --process compose --notes tests/fixtures/pages --topic "Apollo program" --dry-run   # offline, no UI
python -m tools.writer_smoke --shortcut-pref 0.05   # ~3 min live check of every Writer formatting type
python -m tools.capture_page --save my_page         # save the Chrome page in front as a test fixture
python -m tools.probe_research chrome|writer        # the self-checks on their own (read-only)
python -m unittest discover -s tests -t .           # offline tests
```

| Flag | Meaning |
|---|---|
| `--topic` | What to research. |
| `--minutes` | Total time. Reading takes at most 35% of it (45% from 45 minutes up). The rest is writing. |
| `--sources 4` | How many pages to read before writing. |
| `--words` | Paper length. By default it is whatever fits the time left. |
| `--paraphrase 0\|1\|2` | 0 = sentences as read. 1 = phrase rewrites and date moves. 2 = also "According to …". |
| `--resume DIR` | Continue a run: the notes, the composed paper and how far the writing got. |
| `--notes DIR`, `--dry-run` | Compose only. Notes come from a run folder or a folder of page captures. `--dry-run` lists the Writer actions. |

A 30-minute run writes roughly 20 of the paper's 27 blocks at a typical persona's pace. To finish a whole paper, use `--minutes 45` or more, or `--resume`. See [§12](#12-findings-from-live-testing).

Everything a run produces goes to `runs/research/<topic>-<yyyymmdd-hhmm>/`:

| File | Content |
|---|---|
| `captures/NNN.json` | The raw page capture of every page read, in the same format as the test fixtures. |
| `notes.json` | The parsed notes (sources, sections, paragraphs, tables, what was seen). |
| `outline.json` | The composed paper as a document model. It is used for resuming. |
| `paper.md` | A Markdown preview of the paper. |
| `progress.json` | How far the writing got (the operation index), whether the file is saved, and which queries were used. |
| `diagnostics.json` | The self-checks ([§10](#10-self-checks)). |
| `<topic>.odt` | The Writer document (OpenDocument Text). |

---

## 3. Pipeline overview

```
 search (tasks.search_and_read)          ┌────────────────────────────────────────────────┐
   │ result opens                        │ runs/research/<topic>-<time>/                  │
   ▼                                     │   captures/NNN.json  notes.json  outline.json  │
 on_page ─► extract.capture_page() ──────┤   paper.md  progress.json  diagnostics.json    │
   │          (UIA: TextPattern +        │   <topic>.odt                                  │
   │           bulk FindAll)             └────────────────────────────────────────────────┘
   │        extract.parse_capture() ─► notes.Source ─► notes.ResearchNotes
 read_page(on_view) ─► seen intervals (% of the page that was in view)
   │
   ▼  (enough material)
 compose.compose_paper(notes) ─► ir.Document  (title, headings, paragraphs+runs, bullets, tables, references)
   │
   ▼
 writer_ops.render(doc) ─► [Op, Op, …]   (pure: style / bullets / bold / size / align / type / enter / table / save …)
   │                          ▲
   │                          └─ fake_writer.FakeWriter replays them offline (tests)
   ▼
 writer_exec.WriterExecutor.run(ops) ─► Human keystrokes / toolbar and menu clicks ─► OpenOffice Writer
   │   checks: paragraph style and font size at the caret, caret in a table, focus,
   │           each block's text read back, word-completion suggestions dismissed
   ▼
 diagnostics.json + end-of-run summary
```

| Module | Role |
|---|---|
| `research/extract.py` | Reads the page through UIA (`capture_page`) and parses it into notes (`parse_capture`, which is pure). |
| `research/notes.py` | `Source`, `SectionNote`, `Para`, `TableNote` and `ResearchNotes`, with JSON persistence. |
| `research/textutil.py` | ASCII folding, cleaning, sentence splitting, tokenising and the bundled word lists. |
| `research/rank.py` | TF-IDF, cosine similarity, TextRank, MMR and k-means in numpy. |
| `research/compose.py` | Search planning and the paper composer. |
| `research/ir.py` | The paper as data, plus `validate()`, `to_markdown()` and JSON round-trip. |
| `research/writer_ops.py` | Turns the document model into editing operations, tracking Writer's state. |
| `research/fake_writer.py` | An offline model of Writer that executes the operations, for tests. |
| `research/writer_exec.py` | Performs the operations as the persona, with fallbacks and checks. |
| `controller/writer.py` | Writer UIA helpers: toolbar and menu lookup, caret style and font size, focus, tables, text. |
| `research/project.py` | Ties the stages together for `main.py` and the Session. Handles saving and resuming. |
| `research/diagnostics.py` | The self-checks recorded during a run. |

---

## 4. Reading pages (`extract.py`)

### 4.1 Why UI Automation, and how to make it fast
Chrome exposes every page through its accessibility tree, the same interface screen readers use. The existing reading code already used it to trace words with the cursor. Research mode reads three things from that tree:

- **Text: one call.** `TextPattern.RangeFromChild(main).GetText()` returns the whole article as plain text. Block elements are separated by line breaks.
- **Structure: one call per kind.** `ui_elements.find_all_fast()` runs a single filtered `IUIAutomationElement::FindAll` for headings and tables. The older tree walk costs one cross-process call per element and per property, which takes tens of seconds on a long Wikipedia article. `FindAll` returns in 0.2–2 s.
- **Timing.** The capture happens right after the result opens. It is invisible and takes 0.2–3 s, during the "scan" pause a reader makes anyway.

### 4.2 Finding the article
`_main_root()` tries these candidates in order. It takes the first one with at least 150 words (`MIN_MAIN_WORDS`):

1. The **Main landmark** (`LandmarkType` 80002, i.e. `<main>`).
2. An element whose localized control type is `article`.
3. A well-known content id: `mw-content-text`, `bodyContent`, `main-content`, `content`, `main` or `article-body`. Chrome exposes the HTML `id` as the AutomationId.
4. The whole document (method `document`).
5. If the page has no TextPattern at all, the visible text elements (method `text-nodes`).

The method used is stored in the capture and checked by the diagnostics. A weak method means menus may slip into the notes.

### 4.3 Headings, tables and captions
**Headings.** A heading is a text control whose `HeadingLevel` property (UIA id 30173) is not "none". Values 80051–80059 mean h1–h9. On older Windows the fallback is the localized type `heading`. The level then comes from the ARIA `level=` property, defaulting to 2.

`_locate_headings()` finds each heading in the text by **line match**. The line must equal the heading, optionally followed by a short suffix such as `[edit]`. A paragraph that merely *starts* with the same word doesn't count; on Britannica, "Apollo, the United States effort…" would otherwise have swallowed the lead.

**Pages without heading markup.** `_guess_headings()` treats a line as a heading when all of these hold:

- it is 1–8 words long, capitalised, and has no final punctuation, comma or colon;
- it is not right after an image;
- it is followed by a line of at least 15 words.

**Tables.** `_tables()` uses GridPattern and keeps small tables only: 2–40 rows, 2–5 columns, at most 6 tables.

- A 2-column table within 1.5 screen heights of the top, or with an `infobox` class, is an **infobox**.
- Navboxes, sidebars and metadata boxes are skipped.
- The text lines of **every** table, kept or not, are recorded in `table_lines`, so their cells never leak into the paragraphs. Without this, the Wikipedia mission list produced "sentences" like "No spacecraft; observations of liquid hydrogen…".

**Images.** Chrome puts U+FFFC (object replacement character) where an image is. The line after it is the image's caption, and it is dropped. Every U+FFFC is removed from text and cells.

### 4.4 Cleaning (`parse_capture`, pure)
1. **Split into sections** at the located headings. Text before the first heading, or after the page's h1 title, is the *lead*.
2. **Drop boilerplate sections and their subsections.** These are the headings in `data/boilerplate_headings.txt`: references, see also, notes, external links, further reading, contents, related topics, and so on.
3. **Drop junk lines** (`_JUNK_LINE`): `[edit]`, "Main article:", "From Wikipedia", "Retrieved …", "Archived from", ISBN, doi, coordinates, "Cite this", "Share", "Written by", "Last updated", image credits.
4. **Clean each line** (`textutil.clean_line`):
   - strip footnote markers such as `[12]`, `[a]`, `[note 3]` and `[citation needed]`;
   - strip parentheses holding pronunciations, IPA, transliterations or foreign script;
   - restore a missing space after a sentence end ("1961.[3]Thereafter" → "1961. Thereafter");
   - collapse whitespace.
5. **Keep a paragraph** only if it has at least 6 words (`MIN_PARA_WORDS`) and at least 60% of its non-space characters are letters.
6. **Clean table rows.** Rows that span the full width (all cells equal) and empty rows are dropped.
7. **Record each paragraph's position** on the page: its character offset divided by the text length. This is used for seen-tracking.

### 4.5 What was actually read
`reading.read_page(on_view=…)` calls back with Chrome's scroll state `(position %, visible %)` at the start and after every scroll. The part of the page in view is then:

```
top    = position × (100 − visible) / 100        (in % of page height)
seen   = [top, top + visible]
```

Intervals are merged per source. A paragraph counts as *seen* if its position falls inside an interval, within a ±2% margin. Seen sentences get a ranking bonus ([§6.3](#63-scoring)). `words_read` sums the seen paragraphs and feeds the "sources consulted" table. Pages that never report a scroll position count as fully seen.

### 4.6 Timing safeguards (from live testing)
- **Slow sites.** After a click, a slow site leaves the Google results on screen for several seconds. `ResearchProject.on_page` waits up to 10 s for the address bar to leave the search page (`is_search_page`), then for the page to finish loading. If it never leaves, no notes are taken. Notes are **never** taken from a search results page.
- **A tree that isn't ready.** Chrome builds a page's accessibility tree a moment after loading, especially in a freshly opened window. The capture is retried up to 4 times, 1.5 s apart, until the page has at least 150 words.
- **Order of operations.** Notes are taken *before* the reading time is sized, so a slow page doesn't get a reading time computed from the search results.

---

## 5. Notes and search planning

**Notes** (`notes.py`). Each `Source` holds:

- url, title (site suffix removed), site name, access date and the query that found it;
- sections, each a heading and level with paragraphs and their page positions;
- small tables;
- seen intervals and words read.

`ResearchNotes.add()` gives each source a number. A page read twice (same URL, ignoring scheme, `www`, fragment and trailing slash) **replaces** its earlier notes and merges the seen intervals, so it doesn't become a second source.

**Search plan** (`compose.plan_queries`). The queries come in this order:

1. `"<topic> wikipedia"`, with link hint `wikipedia`.
2. `"<topic> britannica"`.
3. `"<topic> <heading>"` for the three longest **top-level** headings of the first page read, excluding minor ones such as "Name", "Etymology" and "Overview".
4. Fallbacks: `history`, `significance`, `facts`.

Results are still chosen through the existing `choose_link` and `ALLOWED_DOMAINS`. A result that needs no hint is one of the top results, with upper ones more likely.

---

## 6. Writing the paper without a language model

`compose_paper(notes, topic, rng, ComposeConfig)` builds an `ir.Document`. Each step below names the function that does it.

### 6.1 Sentence pool (`build_pool`, `usable_sentence`)
Every paragraph is cleaned, split into sentences, and **folded to ASCII**. Dashes, quotes and spaces are mapped and accents dropped. A sentence is kept only if all of these hold:

- it has 10–40 words;
- it starts with a capital letter or a digit, and ends with `.`, `."` or `.)`;
- it has no `?`, `!`, `[`, `]`, at most one colon, at most two double quotes, and balanced parentheses;
- it has no first- or second-person words (I, we, our, you, …);
- it is pure ASCII and passes `ir.check_text`, so Writer's AutoCorrect has nothing to rewrite;
- digits are under 15% of its characters, which excludes lists of figures;
- it is not a subject-less note ("Flew on the 1975 …", "Landed on the Moon …"), unless a comma follows the opening phrase ("Founded in 1958, NASA grew …" is kept).

Sentence splitting (`textutil.split_sentences`) breaks after `.`, `!` or `?` when whitespace and a capital or digit follow. It does not break after:

- an abbreviation from `data/abbreviations.txt`;
- a single-capital initial ("John F. Kennedy"), unless the next word typically starts a sentence ("Saturn V. The …", "… V. Dr. Braun").

Decimals never split, because the split needs whitespace.

**Leaning sentences.** A sentence that only makes sense after the previous one is marked `leans`. Examples: "It …", "This …", "They …", "However, …", "The latter …", "The first, an automated vehicle …". It can only be chosen **together with the sentence right before it** in the same paragraph, as a two-sentence *unit*. If that predecessor isn't usable, the sentence is never chosen. This one rule removes most dangling references without any grammar parsing.

The pool is capped at 2,500 sentences (`max_pool`), shared evenly across sources.

**Groups.** Each sentence belongs to a group: its source plus the nearest heading at level ≤ top level + 1. On Wikipedia that is h2 and h3. Wikipedia h2 sections are often long and mixed, while h3 sections are focused, which makes for better paper sections.

### 6.2 Vectors (`rank.tfidf_matrix`)
Tokens are lower-case alphanumeric words. Stopwords (`data/stopwords.txt`, about 150 words) are removed, and a crude suffix stemmer folds missions → mission and landed → land.

```
tf(t, s)  = 1 + ln(count of t in s)
idf(t)    = ln((1 + N) / (1 + df(t))) + 1          N = sentences, df = sentences containing t
x(s)      = tf · idf, L2-normalised                vocabulary capped at 5,000 terms, float32
S         = X Xᵀ                                   cosine similarity (rows are unit length)
```

### 6.3 Scoring (`score_pool`)
```
score(s) = 0.45 · TextRank(s) / max  +  0.30 · cos(s, topic) / max  +  0.15 · min(1, 1/(1+index) + 0.1·lead)  +  0.10 · has_number
score(s) × 1.5  if s was in view while reading (only when some source has seen-tracking)
```

**TextRank** (`rank.textrank`) is PageRank on the similarity graph:

- edges are pairs with S > 0.1, the diagonal is zeroed, and rows are normalised;
- damping is 0.85, with 30 power iterations;
- *dangling* sentences (no edges) spread their score evenly, so the scores still sum to 1.

Central sentences, which many others resemble, score high.

The other terms:

- **Topic similarity** is the cosine to the topic's TF-IDF vector.
- **Position** favours the first sentences of a section, where encyclopaedias put the key statement, with a bonus for the lead.
- **has_number** favours sentences that contain a year or number, i.e. facts.

A *unit* (a leaning sentence with its predecessor) scores `0.9 × mean` of its two sentences.

### 6.4 Outline: which sections the paper has (`plan_outline`)
The groups are clustered across sources, largest first. A group joins an existing cluster if either condition holds:

- **name overlap:** the Jaccard similarity of the heading tokens (without topic words) is at least 0.5, as with "Missions" and "Lunar missions";
- **content overlap:** the cosine between the group's sentence centroid and the cluster's centroid is at least 0.45, as with "Uses" and "Applications".

A cluster holds at most one group per source.

**Strength** = (sum of the top-8 sentence scores) × (1 + 0.5 per extra source) × min(1, sentences / 8). Clusters with fewer than 3 sentences, and minor sections (Name, Etymology, Overview, Gallery …), are dropped. The strongest *n* are kept, and **ordered by their median page position**, so the paper follows the sources' own order (background before legacy).

Each cluster is named after its most common original heading, in title case.

The number of body sections follows the paper size: 2 for a tiny paper, 3 for a small one, 4 under 1,100 words and 5 above. **k-means fallback:** when the pages have too few usable headings, `rank.kmeans` (k-means++ seeding, 20 iterations, distances computed without building an n×k×d array) clusters the sentences. Each cluster is named after its two most characteristic words.

### 6.5 Selection (`Selector`)
Each body section gets `(target − frame words) / sections` words. **Maximal marginal relevance** picks units one at a time:

```
next = argmax over unchosen units u of   0.7 · relevance(u)  −  0.3 · max cos(u, anything already in the paper)
```

The redundancy term is checked against the **whole paper**, not just the section. A unit is rejected as a duplicate if either holds:

- its cosine to anything chosen is above 0.6;
- it shares a year with a chosen sentence and has at least 4 content words in common, which catches the same fact told in different words.

A unit that would push the section more than 15% over its budget is skipped in favour of a shorter one.

The same selector also picks the introduction, the abstract sentences, one key point per section and the conclusion sentence. So nothing is said twice.

- **Introduction.** It starts with a *defining* sentence: the first sentence of a lead with "is / was / are / were" in its first 25 words, preferring the first source. One or two more lead sentences from the same source follow.
- **Conclusion.** Its sentence favours evaluative wording: remains, considered, widely, significant, legacy, showed, inspired, major … It avoids dated narrative ("On July 20, 1969, …").

### 6.6 Arrangement and citations (`arrange`, `cite`, `number_citations`)
- **Order.** Within a section, units are ordered so that sentences from one source stay together, the source contributing most first, in page order.
- **History sections** (history, background, origin, timeline, development …) are put in **date order**. A sentence without a year inherits the last year seen before it in its source, so it stays next to its context.
- **Paragraphs.** Units are grouped into paragraphs of 2–4 sentences (random length) from one source each. A lone sentence after a paragraph from the same source joins it.
- **Citations.** Each paragraph is cited **once**, at the end of its run of sentences from one source, before the final full stop: `… in 1969 [3].` Placeholders are numbered by **first citation in reading order**, sources never cited are numbered after, and the references list follows that numbering.

### 6.7 Light rewriting (`rewrite`, `paraphrase` 0/1/2)
Only grammar-safe changes are made:

- **Phrase map** (`data/phrase_map.tsv`, about 60 pairs). Examples: "in order to" → "to", "a number of" → "several", "due to the fact that" → "because", "is known as" → "is referred to as". Matches are whole words, and the capital is kept at the start of a sentence.
- **Moving a date.** "In 1969, X happened." becomes "X happened in 1969." This is applied with probability 0.3, only when the rest has no comma and is at most 25 words.
- **Level 2 attribution.** At most one "According to {Site}, …" per section.

**No word-level synonym swaps.** Without part-of-speech and word-sense knowledge they produce wrong English ("bank of the river" → "financial institution of the river"), and a reader notices that far more than a verbatim sentence.

### 6.8 The frame, names and sizes
The paper is laid out in this order:

1. Title and subtitle.
2. A centred 10 pt date line.
3. **Abstract** (10 pt): a template, plus the best one or two unused lead sentences, plus a list of the sections.
4. **Keywords:** in bold.
5. **Introduction**, with the topic in **bold**, followed by a roadmap sentence.
6. The body sections.
7. **Key Facts**: a caption and an infobox table and/or a timeline table and/or a small data table.
8. **Discussion**:
   - names shared across sources;
   - each source's focus, taken from its headings;
   - an agreement sentence;
   - **Key Points**: bullets with bold labels.
9. **Conclusion**.
10. A page break, then **References** at 10 pt with italic titles.
11. **Appendix: Sources Consulted**, a table.

Template variants are picked at random and never repeated within a paper.

**Size tiers.** Short papers get a leaner frame:

| Target words | Tier | Body sections | Frame cost (`FRAME_WORDS`) | Dropped |
|---|---|---|---|---|
| < 500 | tiny | 2 | ≈ 300 | key-point bullets, appendix table, second table; shorter abstract and introduction |
| < 700 | small | 3 | ≈ 400 | appendix table, second table |
| ≥ 700 | full | 4–5 | ≈ 540 | none |

The target comes from the time left when `--words` isn't given:

```
words ≈ seconds × (wpm × 5 / 60 / 1.8) / 6 / 1.25
```

That is the persona's effective characters per second (thinking pauses included, as in `Workspace._next_chunk`), at 6 characters per word, with 25% extra for formatting. The result is clipped to 300–2,000 words.

**Tables.**

- An **infobox** table needs at least 3 rows of key (≤ 4 words) and value (≤ 12 words); it keeps at most 10.
- A **timeline** table needs at least 3 events. Its rows come from sentences that start with a date phrase ("In July 1969, …", "On January 27, 1967, …"). The event text runs from the date to the end of the sentence, is cut at a trailing clause (", which…", ", forcing…"; never inside a list of names), and must be 4–14 words. There is one event per year, at most 8, sorted by year.
- A **data** table must have 3–12 rows, 2–5 columns and every cell at most 8 words.
- Cell text is capitalised in advance, because Writer's AutoCorrect would capitalise it anyway and the read-back must match.

**Proper nouns and articles.** Names keep their case. A word counts as a name if the sources capitalise it mid-sentence and never write it in lower case, so "Moon", "Apollo" and "NASA" stay capitalised while "background" and "missions" don't.

The topic and shared names get "the" when the sources usually write it that way ("the Apollo program", "the Moon"), but numbered names never do ("Apollo 11", "Saturn V"). Section names inside sentences are quoted ('the sections below examine "Astronauts" and "Lunar Landings"'), because headings are often clauses ("Political pressure builds").

### 6.9 Safety for typing: `ir.validate`
Before a paper is accepted, every piece of text must pass these checks:

- it is ASCII;
- it has no line break or tab inside a run, so the typing model's corrections can never cross a paragraph or table cell;
- no paragraph, bullet or cell starts with something Writer's AutoCorrect turns into a list or border: `* `, `- `, `1. `, `a) `, `---`;
- it contains nothing AutoCorrect replaces: `(c)`, `(r)`, `(tm)`, `--`, `->`, `...`, `1/2`, smileys.

Straight quotes becoming curly ones and " - " becoming an en dash are allowed. The read-back normalises them.

### 6.10 Quality to expect
| Reads well | Reads less well (mitigation) |
|---|---|
| Individual sentences, because they are human-written. | Jumps between runs from different sources (mitigated by keeping one source's run together and by date order). |
| A lead-based introduction. | Dangling references (mitigated by leaning-sentence units). |
| Section names from real headings. | The same fact in three wordings (mitigated by MMR, the 0.6 threshold and year+words de-duplication). |
| Infobox and timeline tables, and the references. | Formulaic frame sentences (mitigated by rotating variants and quoted section names). |
| Shared names across sources. | Conclusions summarise only lightly (by design: one evaluative sentence). |

---

## 7. From paper to Writer operations

### 7.1 The document model (`ir.py`)
The paper is a list of blocks:

- `Title`, `Subtitle`, `Heading(level)`, `Caption`;
- `Paragraph(runs, size, align)`, where each `Run(text, bold, italic)` is a stretch of uniformly formatted text;
- `Bullets([[Run]])`;
- `Table(rows, header)`;
- `PageBreak`.

The model can be serialised (`to_dict` / `from_dict`) for resuming, and previewed as Markdown.

### 7.2 Rendering (`writer_ops.render`)
The renderer is a **state machine**. It keeps track of what Writer's state will be at the caret: paragraph style, bullets, bold, italic, font size, whether any *direct* character formatting is there, the paragraph's *direct* alignment, whether the caret is in a table, and whether the paragraph is still empty. It emits an operation only when something must change, the way a person formats while writing.

| Operation | Meaning |
|---|---|
| `style name` | Paragraph style: Title, Subtitle, Heading 1/2/3, Text body, Caption. |
| `bullets on` | Bullets on or off (a toggle: Shift+F12 / Bullets On/Off). In Writer bullets aren't a paragraph style. |
| `bold on` / `italic on` | The desired state (the executor turns it into a toggle). |
| `size pt`, `reset_chars` | Font size, and Default Formatting (Ctrl+M) to clear direct formatting. |
| `align left/center` | Paragraph alignment. |
| `type text mode` | One run. Mode is `compose` from 12 words up, otherwise `transcribe`. |
| `enter`, `page_break` | A new paragraph, or Ctrl+Enter. |
| `table rows cols`, `next_cell`, `exit_table` | Insert a table, Tab to the next cell, Ctrl+End out of it. |
| `checkpoint i`, `save` / `save final` | Block *i* is done (a writing session may stop here), and a save at a section boundary. |

Rules the renderer relies on, each checked in Writer 4.1 ([§12.4](#124-moving-to-openoffice-writer)):

- **Enter picks the style's next style.** Title → Subtitle, Subtitle and headings → Text body, Caption and Text body continue. Bullets, direct character formatting, font size and *direct* alignment carry over. A style's own alignment doesn't: Title and Subtitle are centred by their style, and the paragraph after them is left-aligned.
- **"Not bold" is direct formatting too.** Ctrl+B after bold text stores an explicit not-bold, which carries over Enter like bold does and would un-bold a heading typed next. So once anything was toggled or sized in a paragraph, the next block starts with Ctrl+M.
- **Ctrl+M clears direct alignment as well**, so it is only pressed in a fresh, empty paragraph (in a paragraph with text it would un-centre that text), and alignment is set after it. It keeps bullets and a page break.
- **Applying a paragraph style clears direct alignment** (the style's own applies), and heading styles drop bullets. The alignment keys *set* the alignment (they don't toggle as in Word).
- **Size before alignment.** The keyboard way to the Font Size box tabs out of the Apply Style box, which re-applies the style and so clears direct alignment.
- **Tables.** Insert the table, type the cells with Tab between them, never after the last cell (that would add a row), make header cells bold, then Ctrl+End. Everything is written at the end of the document, so the caret lands in the empty paragraph after the table. Each new cell starts unformatted.
- **After a table or a page break** the caret already sits in a fresh paragraph, so no Enter is needed.
- **Checkpoints and saves.** A checkpoint follows every block, and a save comes before each new Heading 1 section. The final save is always done.

### 7.3 The offline Writer model (`fake_writer.py`)
`FakeWriter` executes the operations with Writer's semantics:

- next styles after Enter, with bullets, direct formatting and direct alignment carried over;
- bold, italic and bullets as **toggles**, alignment keys that set, a style clearing direct alignment and a heading style dropping bullets;
- Ctrl+M asserted to happen only in an empty paragraph, and typing into a title, heading or caption with direct formatting still at the caret asserted not to happen;
- each table cell starting unformatted, Tab in the last cell adding a row, Ctrl+End leaving the table.

The tests check that `FakeWriter().run(render(doc)).result() == expected(doc)` for many random papers. Any renderer bug that would leave formatting on, toggle it the wrong way, un-centre text or break a table fails offline, without Writer.

---

## 8. Performing the operations in Writer

### 8.1 Methods per persona (`writer_exec.WriterExecutor`)
Each operation is done the way *this* person would do it. `human.prefers_keyboard(bias)` draws keyboard or mouse from the persona's shortcut preference, per action. When the first method fails, the next is tried.

| Operation | Keyboard-minded | Mouse-minded | Fallback / check |
|---|---|---|---|
| Heading 1/2/3, Text body | Ctrl+1/2/3, Ctrl+0 | Click the Apply Style box, type the name, Enter | Ctrl+F11 (focuses the Apply Style box). **Check:** the style shown in the Apply Style box. |
| Title, Subtitle, Caption | Ctrl+F11, type the name, Enter | Click the Apply Style box, type, Enter | The other way. Names are typed without typos: a mistyped name would create a new style. |
| Bold, Italic | Ctrl+B / Ctrl+I | Formatting toolbar › Bold / Italic | Ctrl+B / Ctrl+I. Writer doesn't report the state, so the executor's own model decides (synced from the operations done when a stint starts mid-paper). |
| Bullets | Shift+F12 | Formatting toolbar › Bullets On/Off | Shift+F12. |
| Default Formatting | Ctrl+M | Ctrl+M | Always the shortcut (the Format menu route left a carried "not bold" in place). |
| Font size | Ctrl+F11, Tab, Tab (Font Size box), type, Enter | Click the Font Size box, Ctrl+A, type, Enter | The other way. Only typed into a confirmed text field. **Check:** the size shown in the Font Size box. |
| Centre / left | Ctrl+E / Ctrl+L | Centred / Align Left button | Ctrl key. |
| Table | Ctrl+F12 (Insert Table dialog): Tab, columns, Tab, rows, Enter | Insert › Table… | The other way. The dialog's Heading option is unticked if ticked. **Check:** the caret is in a table, or the table count grew. |
| Page break | Ctrl+Enter | Insert › Manual Break… › Page break › OK | Ctrl+Enter. |
| Save | First Save As (Ctrl+Shift+S, full path, `.odt`); after that Ctrl+S | The Save button on the Standard toolbar | After the first save, about 60% of section boundaries also get a save. (F12 in Writer switches numbering on, Ctrl+F12 inserts a table.) |

### 8.2 Guards
- **Focus** (`ensure_focus`, `controller/writer.focus_kind`). Text is only typed when keys go to the page. Writer's UIA `GetFocusedControl()` only ever returns the window, so focus is read from the MSAA FOCUSED state: of the caret's paragraph (the page), of a toolbar box (a field), or of a toolbar button. After a toolbar click Writer goes on marking that button focused while keys still reach the page; that counts as fine. An unclear answer is asked again first (the caret's paragraph is marked a moment after it moves). Otherwise the executor presses Esc, then clicks back into the page; inside a table it stops instead (the click and Ctrl+End would leave the table), and if focus can't be had it raises `TaskError` and types nothing.
- **Text fields** (`_in_field`). Ctrl+A and typing a value (font size, style name) only happen once a toolbar box, or a Writer dialog, *confirmably* has the focus. In Word testing a missed click on the Font Size box followed by Ctrl+A replaced the **whole document**.
- **Style resync.** Before the first run of a paragraph, the style Writer reports is compared with the expected one and re-applied if Writer chose another (in case a "next style" differs from the renderer's table).
- **Word completion.** Writer offers the rest of a long word it has seen (as selected text after the caret) and Enter accepts it. After each run ending in a word, the caret's paragraph is read; if it shows more letters than were typed, Delete dismisses them.
- **Block text read-back** (`project.verify_block`). After every block, the document text, normalised for curly quotes, dashes, bullets, cell markers and breaks, must end with the block's text. Writer only exposes the paragraphs on screen, which is enough: the block was just typed at the end.
- **Tolerant lookups** (`controller/writer._tolerant`). UIA calls fail now and then while Writer redraws (`COMError`). Lookups retry, then report "not found", and the next method is used.
- **Typing is `word_safe`.** AutoCorrect rewrites a misspelled word as soon as a space follows it. If the typing model then fixes its typo itself, it edits text that is no longer what it typed, and a letter goes missing. With `TypingConfig.word_safe`, a typo is always corrected **before its word ends**. See [§12.2](#122-the-lost-letter-investigation).

### 8.3 Timing
- `ResearchProject.block_seconds(i)` estimates the next block: its characters at the persona's effective rate, plus 2 s per formatting step.
- A block is **not started** if it wouldn't finish before the deadline. This is checked before the first block of a writing turn as well as at every checkpoint.
- When time runs out, the person saves (Ctrl+S) before stopping.

---

## 9. Orchestration

### 9.1 `--process research` (`main.run_research` / `research_flow`)
1. **Read.** Browse until `--sources` pages are read, or until the reading share is used up: 35% of the time for runs under 45 minutes, 45% otherwise. Short runs keep enough time to write.
   - Each search is `Workspace.research_browse`. A person reads up to 150 s per page, sometimes a second result, and notes are taken from every page.
   - A failed search is logged and the next one tried. That includes a result that won't open and unexpected UIA errors; the corner-slam emergency stop always ends the run.
2. **Compose** the paper to fit about 85% of the remaining time. It is written to `outline.json` and `paper.md`.
3. **Write** one section per turn (`ResearchProject.write_next`), opening Writer, or reopening the saved `.odt` when resuming.
   - Between sections there is a 30% chance of "checking something in the source": Alt+Tab to Chrome, read 8–25 s, back to Writer.
   - The loop stops when nothing more fits.
4. **Finish.** Save, then print the self-check summary, even if the run stopped early.

### 9.2 In the Session
With `--process session --topic …`, the Markov session's activities change:

- `browse` does research searches and takes notes.
- `write` types the next section of the paper. If nothing has been read yet, it browses first: the person "needs material".

When the paper is finished, `write` raises `TaskError` and the Session drops the activity.

### 9.3 Resuming
`--resume DIR` loads `notes.json`, the composed `outline.json` and `progress.json`. The operation list is re-rendered deterministically from the outline, so the saved operation index points to the same place. Writer is opened, Ctrl+O opens the saved `.odt` in place of the blank document, and writing continues from Ctrl+End. The executor takes its idea of bold / italic / style at the caret from the operations already done.

---

## 10. Self-checks (`diagnostics.py`)
Checks run as the pages and Writer come up. Problems are logged as `Check: …`, everything is written to `diagnostics.json`, and a summary is printed at the end of the run.

| Area | Check | Warning means |
|---|---|---|
| Each page | How the article was found; number of headings and their levels; tables; words kept | "article element wasn't found" means the whole page was read and menus may slip in. "no headings" means sections were guessed. "every heading came back as level 2" means levels may not be exposed. "only N usable words" means the page was mostly furniture. |
| Writer (once) | Paragraph style and font size read-back at the caret; document text readable; focus; Formatting toolbar, its buttons and boxes, the Standard toolbar's Save, the Insert / Format menus found by name | "doesn't report paragraph styles" means the Formatting toolbar is hidden and style changes can't be verified. "toolbar wasn't found" / "buttons weren't found" means clicks fall back to shortcuts (hidden toolbar, narrow window, another UI language). |
| Writing | Methods used per operation; problems; blocks whose text didn't match | Each problem is listed. |

Example end-of-run summary:

```
Pages: 4 read, article found on 4, headings on 4
Writer: style read-back OK, text read-back OK, toolbar OK
Writing: 20/20 blocks matched, 0 formatting problems
0 warnings - details in ...\diagnostics.json
```

---

## 11. Testing

| Test | What it covers | UI? |
|---|---|---|
| `tests/test_research_text.py` | Citation and IPA stripping, ASCII folding, sentence splitting (initials, abbreviations, "Saturn V. Dr."), tokenising, AutoCorrect triggers, TF-IDF normalisation, TextRank, MMR, k-means. | No |
| `tests/test_research_paper.py` | Parsing the fixtures (sections, boilerplate, infobox, lead vs heading, guessed headings, seen-tracking); paper structure over 5 seeds; citations resolve; length within 25% of the target; no sentence twice; no leaning sentence without its predecessor; lean frame for short papers; outline round-trip; **FakeWriter round-trip** over 6 seeds and a paper with every block kind; operation invariants (`next_cell` count = rows × cols − 1); **every typed run replays exactly** through the typing model; executor dry runs for keyboard and mouse personas; stopping at section boundaries; a new executor syncing bold from the operations done. | No |
| `tests/fixtures/pages/*.json` | Three page captures written for the tests: Wikipedia-style with `[n]`, `[edit]`, an IPA pronunciation, an infobox and boilerplate sections; Britannica-style; NASA-style with no heading markup. `python -m tests.fixtures.make_apollo_pages` regenerates them. | No |
| `python -m algorithms.typing_model.selftest` | The typing model's plans still reproduce the text exactly, including with `word_safe`. | No |
| `python -m tools.writer_smoke` | **Live**, about 3 minutes. Writes a mini paper with every block type into a new Writer document, saves it as `.odt`, and checks its `content.xml`: styles, exact heading texts (no accepted word completion), alignment, sizes, bold and italic runs, bullets, table cells and bold header, the paragraph after the table, the page break, and **no direct bold/italic on titles, headings and captions** (a carried-over "not bold" or a style's own formatting toggled). Closes the document afterwards. | Yes |
| `python main.py --process compose --dry-run` | Composing plus the full list of Writer actions for a sampled persona. | No |

---

## 12. Findings from live testing

§12.1–12.3 were found on Windows 11 with Chrome and Microsoft 365 Word on a two-monitor desktop (four full research runs, eight smoke tests and targeted experiments), before the move to OpenOffice Writer; the Word-specific rows are kept as history. §12.4 covers the move.

### 12.1 Bugs found and fixed
| # | Symptom (live) | Root cause | Fix |
|---|---|---|---|
| 1 | Wikipedia read as "whole page, no headings, 0 tables" | `uiautomation` doesn't export `_AutomationClient` at package level, so every bulk `FindAll` failed silently | `auto.uiautomation._AutomationClient` |
| 2 | Image captions and U+FFFC characters in the notes | Chrome's object-replacement character marks images; the next line is the caption | Strip U+FFFC and drop the line after it |
| 3 | "No spacecraft; observations of…" as sentences | Cells of large tables (not captured as tables) are part of the page text | Record every table's lines and drop them from paragraphs |
| 4 | Heading "Abstract" justified | Applying a style resets alignment; Ctrl+L on a left-aligned paragraph *justifies* it | The renderer assumes left after a style; FakeWord models the toggle |
| 5 | Bold on the wrong text (plain words bold, labels plain) | Word turns Bold on by itself in list items, so the executor's assumed state drifted; Esc after a ribbon click dropped a pending Bold | Read and resync from the ribbon toggle state; a ribbon focus counts as fine; read back and repair runs |
| 6 | Caption lost its italics | The format check treated the Caption style's own italic as wrong and "fixed" it | Check runs only in body styles; the smoke test flags explicit bold/italic-off runs |
| 7 | **Whole document replaced by "10"** (mouse persona) | A click on the Font Size box didn't focus it; the following Ctrl+A selected the document | Ctrl+A and typing only into a confirmed text field (`_in_field`) |
| 8 | Apply Styles refused after fix 7 | The Apply Styles pane is a separate floating window of Word's process | Accept the foreground window when it belongs to Word's process |
| 9 | One letter missing ("teting", "obit"), 2 of about 60 blocks | Word's AutoCorrect fixed a misspelled word when a space followed; the typing model's own fix then hit corrected text | `word_safe` typing ([§12.2](#122-the-lost-letter-investigation)) |
| 10 | Notes from `google.com` | A slow site left the results page up when notes were taken; a Ctrl+click tab took more than 0.6 s to appear | Wait for the page to leave the search page; never take notes from search pages; poll for the new tab |
| 11 | "Couldn't read this page's text" on a fresh window | Chrome builds the accessibility tree after load | Retry the capture 4 × 1.5 s |
| 12 | Crash: `COMError` in word tracing and in lookups | UIA elements go stale while pages and Word redraw | Tolerant lookups; tracing skips a stale page; a failed search moves on |
| 13 | A 12-minute run wrote only 8/31 blocks and overran by 100 s; a 30-minute run overran by 2 min and its last blocks weren't saved | Paper frame too big for short runs; no estimate before starting a block; no save when time ran out | Size tiers; block-time estimate (also before the first block); final save |
| 14 | "examine political pressure builds", "the Apollo 1", "November 196", 'examine "1962"' | Clause-like headings, the article rule on numbered names, a regex taking part of a year, year headings | Quoted section names, no article on numbered names, whole-number tokens, section names need a word |
| 15 | Follow-up queries such as "Apollo program political pressure builds" | Queries came from h3 headings | Top-level headings only, without minor ones |

### 12.2 The lost-letter investigation
Two mismatches in about 60 live blocks were single missing letters inside words. Two controlled experiments in a fresh Word document, 6 sentences each:

| Condition | Typos / revisions in the plans | Paragraphs that differed |
|---|---|---|
| Transcribe, typos off | 0 | 0 / 6 |
| Compose, typos ×3 | 55 typos, 145 correction keys, 88 arrow keys, 46 false-start keys | 0 / 6 |

So Word doesn't drop keystrokes, and the plans replay correctly. The remaining explanation is Word's **AutoCorrect** acting on a misspelled word once a separator follows it. An offline measurement over 200 plans of a 34-word text at 3× typos counted the times a misspelled word was followed by a space or punctuation:

| Planner | Exposures | From typos |
|---|---|---|
| Default | 769 | most of them |
| `word_safe` (no late notice, no typing on past a separator) | 250 | 179: the typo itself included the space ("orbi ") |
| `word_safe` and no typo that includes a separator | 53 | **0** (the rest are real words from false starts) |

The model still makes about 10 typos per sentence at 3× typos (1,925 typo keys over the 200 plans), all corrected inside their word. The last full run matched **20/20** blocks.

### 12.3 Measured rates
| Measure | Value |
|---|---|
| Page capture (main landmark, headings, tables) | 0.2–3 s per page (long Wikipedia article: about 2–3 s) |
| Effective writing rate, 46–55 wpm personas, thinking pauses and formatting included | about 18–25 wpm |
| A 30-minute run | 4 sources read in about 10.5 min; a paper of about 460 words composed; 20 of 27 blocks written and saved; 2 look-backs at a source |
| Final 30-minute run self-check | Pages 4/4 article found and headings found; Word read-back OK; 20/20 blocks matched; 0 formatting problems; 0 warnings |
| Smoke test, keyboard and mouse personas | Pass. Bold resyncs happen about twice per run (list items), with no repairs needed. |

### 12.4 Moving to OpenOffice Writer
Research mode was moved from Microsoft Word to Apache OpenOffice Writer 4.1.16. Writer's behaviour was probed live first (UIA tree dumps, then keystroke experiments in a scratch document), and the smoke test was then run for keyboard, mouse and mixed personas until all three passed.

| # | Finding | Consequence |
|---|---|---|
| 1 | The document's TextPattern exists but its calls fail. Each on-screen paragraph is a child element whose MSAA value is its text; tables are TableControl › DataItemControl › paragraph. Only the visible part of the document is in the tree. | Text read-back from the paragraphs on screen; enough for checking the block just typed at the end. |
| 2 | `GetFocusedControl()` always returns the Writer window. The caret's paragraph and a focused toolbar box carry the MSAA FOCUSED state. | Focus is read from those states. |
| 3 | Toggle buttons (Bold, Italic, Centred, Bullets On/Off) never change their reported state. | Bold / italic follow the executor's model; no live resync as in Word. |
| 4 | After a toolbar click the button stays marked FOCUSED (and no paragraph) while keys still reach the page; after a menu click nothing is marked. | `focus_kind` 'toolbar' counts as the page. The first mouse-persona run re-clicked into the page six times and then stopped inside the table before this. |
| 5 | Ctrl+Space types a non-breaking space; Default Formatting is Ctrl+M. | `reset_chars` is Ctrl+M. |
| 6 | Ctrl+M also clears the paragraph's direct alignment; it keeps bullets and a page break. | Ctrl+M only in an empty paragraph; alignment set after it. |
| 7 | Ctrl+B after bold text stores an explicit "not bold" that carries over Enter. Smoke test: the Heading 2 "Details" came out not bold. | Any toggle or size marks the paragraph as having direct formatting; the next block starts with Ctrl+M. FakeWriter asserts on it. |
| 8 | A style's own alignment doesn't carry over Enter (the line after the centred Subtitle was left-aligned while the renderer assumed centred). | Direct alignment tracked separately from the style's. |
| 9 | Tabbing out of the Apply Style box re-applies the style, clearing direct alignment (the centred date line lost its centring when its size was set by keyboard). | Size before alignment. |
| 10 | Format › Default Formatting from the menu left a carried "not bold" in place (Caption and References came out with direct formatting); Ctrl+M never did. | Default Formatting is always Ctrl+M. |
| 11 | Ctrl+E / Ctrl+L set the alignment; they don't toggle as in Word. Title and Subtitle are centred by style. Title → Enter gives Subtitle; Caption → Enter stays Caption. Heading styles replace bullets with outline numbering. | Encoded in `NEXT_STYLE`, `STYLE_ALIGN` and the renderer. |
| 12 | F12 switches numbering on and Ctrl+F12 opens Insert Table; Save As is Ctrl+Shift+S, Open is Ctrl+O. The window's UIA name is empty, so the saved-file check timed out (about 50 s) until the title was read through Win32. | Shortcuts and title reading changed. |
| 13 | Word completion offers the rest of long words already in the document, and Enter accepts it. | Dismissed with Delete; the smoke test's "Introduction" heading follows "introductions" in the abstract to exercise it. |

Smoke tests after the fixes: keyboard persona (shortcut preference 0.95), mouse persona (0.05) and a mixed one (0.6) all **pass**.

---

## 13. Limitations and next steps
- **Writing quality is extractive.** No new sentences are written, only selected and lightly rephrased ones. A small offline paraphraser could be added if the hardware allows one later.
- **Short runs can't finish a paper.** At a realistic 20 wpm a full paper needs 45 minutes or more. `--resume` continues a run.
- **Site differences.** Britannica pages sometimes expose only a preview (about 160–370 words). Pages without heading markup fall back to guessed headings.
- **Writer settings and version.** Built and tested against Apache OpenOffice 4.1.16 (en-GB). Style names, toolbar and menu names are English (`Centred` and `Centered` are both accepted). The Formatting and Standard toolbars must be visible for mouse-minded people; the diagnostics report when they aren't, and shortcuts remain the fallback. Word completion can be switched off in Tools › AutoCorrect Options › Word Completion; it is handled either way.
- **Bold / italic can't be verified live.** Writer doesn't expose those states through UIA, so they follow the executor's model; the smoke test checks them in the saved file.
- **Possible additions:** a caption via Insert › Caption; a table of contents; "write the abstract last" (Ctrl+Home, back to the top); learning the reading share and section count from the persona; using lists from source pages as bullets.
