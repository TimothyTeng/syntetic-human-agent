# Algorithms — Behaviour Layer Reference

`controller/` knows **what** can be done: move to (x, y), click, type. `algorithms/` decides **how** a person would do it: the path the mouse takes, how fast, how long the button stays down, how someone scrolls while reading, and how they type (rhythm, typos, pauses, changes of mind). `main.py` runs two demos: a Chrome search and a Word document.

Detailed guides for the two learned models (training process, design considerations, file-by-file function rundown, usage examples, results):
- [`algorithms/mouse_model/README.md`](algorithms/mouse_model/README.md)
- [`algorithms/typing_model/README.md`](algorithms/typing_model/README.md)

```
main.py ──► algorithms.HumanMouse ──► mouse_model (trained LSTM-MDN, numpy runtime)
        │                          ──► click_stats (learned click timing)
        │                          ──► targeting / player ──► controller.mouse
        ├─► algorithms.reading ─────► HumanMouse + controller.mouse.scroll
        ├─► algorithms.human_typing ► typing_model (planner + learned timing, typos, pauses)
        │                          ──► controller.keyboard (key_down / key_up timeline)
        ├─► controller.browser / apps / ui_elements (find targets on screen)
        └─► controller.office (Word demo: open Word, blank document)
```

---

## 1. Quick start

```bash
pip install -r requirements.txt

# 1) dataset: the BMDD split archive (data/BMDD/boun-mouse-dynamics-dataset.zip + .z01-.z09)
#    is read directly - no extraction needed (see data/README.md)

# 2) train (CPU). Smoke run first, then the real run (~5 min/epoch for 40k strokes)
python -m algorithms.mouse_model.train --data data/BMDD/boun-mouse-dynamics-dataset.zip --out models --max-strokes 5000 --epochs 1
python -m algorithms.mouse_model.train --data data/BMDD/boun-mouse-dynamics-dataset.zip --out models --window-filter browsing --max-strokes 40000 --epochs 25

# 3) check realism against held-out real strokes (BMDD test folders)
python -m algorithms.mouse_model.evaluate --data data/BMDD/boun-mouse-dynamics-dataset.zip --window-filter browsing

# 4) typing model (datasets in data/, see data/README.md). Statistics first (~1 min), then the
#    optional neural timing model (~1-2 min/epoch on CPU), then checks.
python -m algorithms.typing_model.fit_stats --aalto data/aalto_keystrokes/Keystrokes.zip --klicke data/klicke/linking-writing-processes-to-writing-quality.zip
python -m algorithms.typing_model.train --data data/aalto_keystrokes/Keystrokes.zip
python -m algorithms.typing_model.selftest
python -m algorithms.typing_model.evaluate --aalto data/aalto_keystrokes/Keystrokes.zip --klicke data/klicke/linking-writing-processes-to-writing-quality.zip

# 5) run the demos (they work before training too, using fallbacks / built-in defaults)
python main.py                                    # Chrome search demo
python main.py --no-model --query "wikipedia" --read-seconds 20 --seed 42
python main.py --process word                     # type a paragraph into a new Word document (80 wpm)
python main.py --process word --wpm 55 --text-file notes.txt --pause-scale 0.4
```

Abort `main.py` at any time by slamming the mouse into a screen corner (the pyautogui fail-safe).

---

## 2. The mouse model

Full guide: [`algorithms/mouse_model/README.md`](algorithms/mouse_model/README.md).

### Why deep learning, and not reinforcement learning?
RL needs a reward, and there is no natural reward for "moves like a human". What we have is **recorded human data**, so the right tool is a **generative sequence model** trained to imitate it by maximum likelihood.

### Pipeline
1. **`dataset.py`** reads raw logs into a common `Session` format: `t, x, y, button, state, window`. It reads one file at a time, because BMDD is huge (33.8 GB uncompressed). `--data` can be an extracted folder **or the split archive itself**: **`splitzip.py`** reads the ZIP64 multi-part archive and inflates single files on demand.
2. **`strokes.py`** cuts sessions into **strokes**. A stroke is a run of Move events ending in a left-button press, starting from the resting point before the run. It filters out:
   - moves shorter than 15 px
   - strokes outside 0.08–4 s
   - strokes with fewer than 5 or more than 200 samples

   The dwell time (last move → press) and hold time (press → release) are recorded too.
3. **Canonical frame:** each stroke is translated so it starts at (0,0), rotated so the target lies on +x, and scaled so the target sits at (1,0). The pixel distance D is kept as an input. One model then covers every direction, distance and screen resolution.
4. **`features.py`** encodes each step (7 inputs):
   - the previous step (dx, dy, log dt)
   - a first-step flag
   - the remaining vector to the target
   - log D

   The target for each step is the next (dx, dy, log dt) plus an end-of-stroke flag. Values are standardised with statistics from the training set.
5. **`model.py`** (TensorFlow, training only) is a 2-layer LSTM (128 units) followed by a **Mixture Density Network** head. The head has K = 10 diagonal-Gaussian components over (dx, dy, log dt), plus an end-of-stroke probability.
   - The loss is the masked negative log-likelihood.
   - The end-flag bias starts at the data's base rate (about 1/length). That keeps a young model from stopping after one step.
6. **`train.py`** trains it:
   - Validation holds out whole users, so it measures generalisation to new people.
   - Batches are bucketed by length.
   - It uses early stopping and learning-rate decay.
   - It then exports `models/mouse_mdn.npz` and `models/click_stats.json`, and runs a **numpy vs TensorFlow parity check**.
7. **`sampler.py`** (numpy only) is the runtime:
   - It runs the LSTM step by step, samples a mixture component and then a step, and stops when the end flag fires.
   - Any leftover landing error is spread along the path in proportion to arc length, so the stroke ends *exactly* on the target with no visible jump.
   - If the model misses repeatedly, it returns `None` and the fallback generator is used.

### Why train in TensorFlow but run in numpy
TensorFlow costs about 500 MB of RAM and several seconds to import. The exported network is about 200 KB, and one trajectory takes about **10 ms** in numpy. With the model loaded, `main.py` uses about **100 MB** RAM and never imports TensorFlow. That helps the Resource Efficiency score.

### Evaluation (`evaluate.py`)
For held-out real strokes, the trained model and the fallback each make the same start→end move. The script then compares medians and IQR, plus a two-sample **KS statistic** against the real data (0 = same distribution). The metrics are:

| Metric | Meaning |
|---|---|
| `duration` | Seconds from start to click |
| `efficiency` | Straight distance ÷ path length (1 = a straight line) |
| `peak_speed` | Maximum speed in px/s, lightly smoothed |
| `max_deviation` | Largest sideways bulge ÷ distance |
| `overshoot` | Fraction of strokes that pass the target |
| `n_points`, `mean_dt` | Sampling density and rate |

It also fits a Fitts-style line, T = a + b·log2(D+1), for each source. `--plot eval.png` saves example paths and speed profiles if `matplotlib` is installed.

### Current model (`models/mouse_mdn.npz`)
- **Training data:** BMDD `training` folders, **browsing** window category only (`--window-filter browsing`). 40,000 strokes from 60 files; users `user1` and `user10` held out for validation.
- **Training run:** 2×128 LSTM, 10 mixtures. Early stopping after 17 epochs; best validation NLL −2.96. Numpy/TF parity 3e-6.
- **Click timing learned:** hold median 102 ms, dwell median 78 ms.
- **Sampling gate:** the stop signal is accepted only within 25% of the distance to the target (`end_tolerance_frac=0.25`). This was tuned in a sweep: mean KS 0.135, against 0.198 with no gate and 0.254 with a tight 4% gate.
- **Evaluation** against 600 held-out real browsing strokes from the BMDD `*_tests` folders. KS = difference from real data (0 = identical, 1 = completely different):

| Metric | Real median | Model median | KS model | KS fallback |
|---|---|---|---|---|
| duration (s) | 0.81 | 0.72 | **0.11** | 0.35 |
| efficiency | 0.86 | 0.76 | **0.16** | 0.36 |
| peak speed (px/s) | 3355 | 3629 | **0.06** | 0.49 |
| max deviation | 0.13 | 0.11 | **0.08** | 0.50 |
| overshoot | — | — | **0.09** | 0.32 |
| n points | 64 | 46 | **0.17** | 0.29 |
| mean dt (s) | 0.012 | 0.017 | **0.29** | 0.88 |

  The model misses the target on only 1 of 600 strokes; the fallback is used in that case. **Known gap:** sample spacing is about 40% longer than real (mean dt), so strokes have fewer points. Candidates for the next training round: more data (`--max-strokes`) and larger `--units`.
- **Runtime:** about 11 ms to generate one trajectory, about 97 MB RSS, no TensorFlow loaded.

---

## 3. The typing model

Full guide: [`algorithms/typing_model/README.md`](algorithms/typing_model/README.md).

Typing is split into two layers, as in writing research:
- the **motor** layer: how fast each key follows the previous one, how long it is held, and the typos fingers make;
- the **cognitive** layer: where the writer stops to think, deletes and rewrites, or goes back to fix something.

### Datasets (see `data/README.md`)
| Dataset | What it teaches | Why it fits |
|---|---|---|
| **Aalto 136M Keystrokes** (168k typists copying sentences) | Per-key intervals and hold times by bigram, Shift timing, typo types and how they are corrected | Real press/release times plus the target sentence, so every typo and Backspace can be labelled exactly |
| **KLiCKe** (Kaggle *Linking Writing Processes*, 2.5k essays written freely in 30 min) | Pauses by location (in word, between words, after a clause or sentence), deletion sizes, lost-thought deletions, in-text revisions | Real composition, not copying. Letters are masked as `q` but spaces and punctuation survive, so boundaries can still be recovered |

### Pipeline (`algorithms/typing_model/`)
1. **Planner** (`planner.py`, pass 1) decides *which keys* to press. It types the target at the end of the text and keeps a virtual buffer, so the final text is always exactly the target. On the way it injects:
   - **typos** (`errors.py`): adjacent key, omission, doubled key, transposition, case slip. The typist notices after 0–n more keys (at most to the end of the next word), then backspaces, sometimes over-deleting;
   - **false starts / changes of idea** (compose mode): type a different wording from an *alternatives provider*, pause, delete back to where it diverged;
   - **lost train of thought**: a long pause, then delete the last few characters and retype;
   - **in-text revisions**: leave a wrong or missing word, or an unnoticed typo; keep writing; arrow back (at most 60 chars), fix it, arrow forward.
2. **Alternatives** (`alternatives.py`): `heuristic` (default, offline: synonym swaps, sentence openers, anticipating the next word, abandoned partial words) or `llm` (one Claude call before typing returns first-draft phrasings anchored in the text; any failure falls back to heuristics).
3. **Timing** (`planner.py`, pass 2) decides *when*. Each key gets (interval since the previous press, hold) from the motor timer. In compose mode the interval at word, clause and sentence boundaries is replaced by a pause sampled from KLiCKe.
   - `timing.BigramTimer`: per-bigram quantiles of interval / typist median, AR(1) tempo drift, persona variation between typists (hold level, rhythm spread).
   - `sampler.TypingMDN`: 2×128 LSTM + 8-component MDN over (log interval, log hold), conditioned on the previous, current and next key and on the typist's speed. Trained in TensorFlow, run in numpy.
4. **Player** (`human_typing.py`): turns the plan into a `key_down` / `key_up` timeline. Keys can overlap (rollover), and Shift goes down a learned lead time before a capital and comes up after it. The timeline is executed with sub-millisecond waits, and any held key is released in a `finally`. Characters the layout cannot press directly are typed atomically.

Modes: `transcribe` for short inputs (search boxes, fields; rhythm + typos only) and `compose` for longer text (everything). The default picks compose from 12 words up.

### Calibration and checks
- `fit_stats.py` simulates the fitted model on real sentences and scales the typo rate until generated typing uses as many Backspaces per key as real typists did. The generator always fixes its typos, while about 15% of real ones are left.
- `selftest.py` replays random plans through a simulated keyboard (Shift state included) and checks the result is exactly the target.
- `evaluate.py` re-types held-out real sentences at each typist's own speed and on their browser's clock resolution. It reports KS distances and a real-vs-synthetic classifier AUC.

### Current model
**Statistics** (`fit_stats.py`, about 1 min):
- **Motor:** 5,367 Aalto typists (3.8M keystrokes) produced 762 bigram tables.
- **Typos:** typo rate calibrated ×0.67 (e.g. 2.8%/key at 45 wpm, 1.3% at 105 wpm).
- **Composition:** all 2,471 KLiCKe essays.

**Neural timing** (`models/typing_mdn.npz`, 860 KB):
- **Training:** 4,000 typists, 52,722 sentences, 20 epochs on CPU (~1 min each). Best validation NLL 1.743; numpy/TF parity 1.4e-6.
- **Speed calibration:** `wpm_scale` = 1.118, so a 45-wpm persona really types at about 46 wpm.

**Motor realism.** 810 held-out sentences from 150 unseen typists, each re-typed at that typist's measured speed and clock resolution. KS: 0 = same distribution as real. AUC: 0.5 = a gradient-boosting detector cannot tell generated from real.

| Generator | interval KS | hold KS | speed KS | Backspace-rate KS | rollover KS | tempo lag-1 KS | **detector AUC** |
|---|---|---|---|---|---|---|---|
| old placeholder | 0.47 | 1.00 | 0.58 | 0.60 | 0.87 | 0.19 | **1.00** |
| built-in defaults (no data) | 0.20 | 0.47 | 0.10 | 0.05 | 0.63 | 0.38 | **0.99** |
| bigram statistics | 0.06 | 0.03 | 0.07 | 0.05 | 0.13 | 0.09 | **0.66** |
| **LSTM-MDN** (default) | **0.011** | **0.024** | **0.038** | **0.028** | **0.059** | **0.063** | **0.52** |

**Composition realism.** 100 KLiCKe essays re-typed in compose mode at essay-level pause settings (`pause_scale=1.0`). The share of keystrokes preceded by a pause of 2 s or more:

| Location | real | generated |
|---|---|---|
| inside a word | 0.2% | 0.0% |
| between words | 6.4% | 7.6% |
| after a clause | 10.1% | 16.0% |
| after a sentence | 40.2% | 43.8% |
| before a deletion | 12.0% | 17.8% |

Deletion runs per 100 chars: real 3.3, generated 3.9. In-text revisions per 1,000 chars: real 1.3, generated 1.7.

**Correctness and live test:**
- **`selftest`:** 0 mismatches in 600 plans, both at the plan level and on the simulated keyboard.
- **Live typing:** typing into a real window gave the exact final text, including 11 and 17 Backspaces and 4–6 Shift presses. Delivered key timing was within **0.1–0.2 ms** of the plan (median; p95 < 2 ms).

**Runtime:**
- Planning takes about 5 ms per sentence and about 40 ms for an email.
- numpy only; TensorFlow is never imported at runtime.
- The default email pace (`pause_scale=0.7`) comes out at about 30 wpm effective for a 55-wpm persona. That is realistic for composing, since thinking pauses dominate.

**Known gaps:**
- Clause-final pauses and pauses before deletions are a little too frequent.
- The Aalto data is sentence *copying*, so motor timing in free composition is inferred rather than measured.

---

## 4. Function reference

### `algorithms/human_mouse.py` — `HumanMouse`
| Member | Description |
|---|---|
| `HumanMouse(model_path=models/mouse_mdn.npz, stats_path=models/click_stats.json, use_model=True, temperature=1.0, speed=1.0, seed=None)` | Loads the model and click statistics. Falls back automatically if the files are missing. |
| `.backend` | `'lstm-mdn'` or `'fallback'`. |
| `.rng` | numpy Generator, shared by all behaviours so a seed reproduces a whole run. |
| `.plan(start, end)` | Returns a trajectory `[(x, y, dt), ...]` without moving. |
| `.move_to(x, y)` | Moves from the current position along a generated path. |
| `.move_to_rect(rect)` | Moves to a human-chosen point inside `(left, top, right, bottom)` and returns the point. |
| `.drift(max_dist=80, bounds=None)` | Small idle move to a nearby point. |
| `.press_release(button='left')` | Clicks in place, with learned dwell and hold times. |
| `.click_at(x, y)` / `.click_rect(rect)` / `.click_element(uia_ctrl)` | Move, then click. |
| `.double_click_rect(rect)` | Move, then double-click. |

### `algorithms/mouse_model/`
| Function / class | Description |
|---|---|
| `dataset.iter_sessions(dataset, root, split=None, max_files=None, users=None, shuffle=True, seed=0)` | Yields `Session`s. `dataset` is `'bmdd'`, `'balabit'` or `'auto'`. |
| `dataset.peek(dataset, root)` | Prints the first rows of a file, to check the column mapping. |
| `splitzip.SplitZip(zip_path)` / `.names(pattern)` / `.read(name)` / `.extract(name, dest)` | Reader for split `.z01…/.zip` archives. CLI: `python -m algorithms.mouse_model.splitzip list|extract <archive.zip> [--pattern training] [--dest data/bmdd]`. |
| `strokes.segment_strokes(session, gap=0.5, min_dist=15, ..., window_filter=None)` | Session → list of `Stroke(points, times, user, dwell, hold)`. `window_filter='chrome'` keeps only browser clicks (BMDD). |
| `strokes.to_canonical(points)` / `from_canonical(canon, start, D, angle)` | Converts between screen space and the canonical frame. |
| `features.NormStats.fit(strokes)` / `encode_stroke(stroke, stats)` | Builds standardised model inputs and targets. |
| `click_stats.ClickStats.fit(strokes)` / `.load()` / `.save()` / `.sample_hold(rng)` / `.sample_dwell(rng)` | Empirical click timing. |
| `model.build_model(units, layers, mixtures, end_prior)` / `make_loss(K)` / `export_npz(...)` | TensorFlow model, used only during training. |
| `sampler.MouseModel.load(path)` / `.generate(start, end, rng, temperature)` / `.forward_sequence(X)` | Numpy runtime. |
| `fallback.generate(start, end, rng)` | Minimum-jerk path with an arc and noise. No training needed. |
| `train.main()` / `evaluate.main()` | CLI entry points; see Quick start. |

### `algorithms/player.py`
| Function | Description |
|---|---|
| `play_trajectory(points, speed=1.0)` | Plays `(x, y, dt)` points through `controller.mouse.move_to` against a `perf_counter` schedule. Keeps 1 px away from screen corners so the fail-safe isn't triggered by accident. |
| `high_res_timer(ms=1)` | Context manager: sets the Windows timer to 1 ms. The default is 15.6 ms, too coarse for mouse samples about 8 ms apart. |

### `algorithms/targeting.py`
| Function | Description |
|---|---|
| `pick_click_point(rect, rng, spread=0.17, margin=0.15, max_span=260)` | Gaussian scatter around the centre, kept inside an inner margin. Very wide elements are aimed at near their middle. |
| `random_point_in(rect, rng, inset=0.1)` | Uniform point inside a rect. |

### `algorithms/reading.py`
| Function | Description |
|---|---|
| `read_page(hm, duration, page_rect=None, params=ReadingParams())` | Moves the cursor into the text column, then alternates scroll bursts, log-normal reading pauses and small drifts. |
| `scroll_burst(rng, params)` | 1–5 wheel notches in quick succession, mostly down. |
| `reading_area(page_rect)` | The central text column of the page. |
| `ReadingParams` | All reading probabilities and timings in one place. They are **hand-set placeholders**, to be tuned or learned later. |

### `algorithms/human_typing.py`
| Function | Description |
|---|---|
| `type_like_human(text, rng=None, wpm=45, sigma=0.35, mode=None, revisions="heuristic", allow_uncorrected=False, error_scale=1.0, revision_scale=1.0, pause_scale=0.7, max_pause=8.0, dry_run=False)` | Plans and types `text` into the focused window and returns the plan. `wpm` = speed persona; `sigma` scales timing variability (0.35 = typical); `mode` = `"transcribe"` / `"compose"` / auto; `revisions="llm"` asks Claude for first-draft wordings. `pause_scale=1.0` gives timed-essay thinking pauses, which feel slow for emails. |
| `play(plan, rng)` / `build_timeline(plan, rng)` | Executes a plan, or builds its `key_down` / `key_up` event list. |
| `get_model()` | The loaded `TypingModel` (cached). |

### `algorithms/typing_model/`
| Function / class | Description |
|---|---|
| `planner.plan_keystrokes(text, model, rng, cfg, provider)` | Text → list of `Keystroke(key, tag, press, hold, iki, pclass, long)`. `tag` is one of type / typo / cont / fix / false_start / lost / nav / revise. |
| `planner.TypingConfig` | All knobs: `wpm, mode, temperature, error_scale, revision_scale, pause_scale, max_pause, allow_uncorrected, intext_revisions, p_late_notice, max_travel`. |
| `runtime.TypingModel.load(models_dir, use_mdn=True)` | Loads the timer (MDN → bigrams → defaults), `ErrorStats` and `CompositionStats`, each with fallbacks. `.sources` says what was loaded. |
| `runtime.apply_plan(plan)` / `describe(plan)` | Replay a plan into text / print a readable trace (`~` = Backspace, `<` `>` = arrows, `[2.1s]` = long pause). |
| `timing.BigramTimer.fit/load/session(wpm, rng).sample(prev, key, next)` | Empirical motor timing. |
| `sampler.TypingMDN.load(path).session(...)` | Neural motor timing (numpy). |
| `errors.ErrorStats` / `errors.make_typo(target, j, type, rng)` | Typo rates by speed, type mix, detection delay, over-deletion. |
| `composition.CompositionStats.sample_pause(class, base_iki, rng)` / `pause_class(...)` | Composition pauses and revision rates. |
| `alternatives.HeuristicAlternatives` / `LLMAlternatives(model=..., timeout=20)` | Change-of-idea wording providers. The LLM model defaults to `claude-opus-5` (override with the `TYPING_ALT_MODEL` env var) and uses server-side refusal fallbacks. |
| `dataset_aalto.iter_participants(zip, split, max_participants)` / `dataset_klicke.iter_essays(path, max_essays)` | Streaming loaders (zip read in place). |
| `fit_stats.main()` / `train.main()` / `evaluate.main()` / `selftest.run()` | CLI entry points; see Quick start. |

### `main.py`
Two demos, chosen with `--process search` (default) or `--process word`.

**Chrome search demo:**
1. Open Chrome from the Start menu. If the "Who's using Chrome?" picker appears, move to the chosen profile card with the learned mouse model and click it. Then maximize the browser.
2. Move to the address bar and click; type the query; press Enter.
3. Click a result, preferring one that contains the query.
4. Read the page.
5. Move to Back and click it.
6. Click a different result and read.

Flags: `--query`, `--link-hint`, `--read-seconds`, `--no-model`, `--temperature`, `--seed`, and for the profile:
- `--profile NAME`: profile or account name to pick. The default is the first card.
- `--profile-index N`: which match to use when several cards share the name.
- `--list-profiles`: print the picker's profiles with their index, then exit.

After typing the query, the demo reads the address bar back. If Chrome's inline autocomplete swallowed a Backspace, it retypes the query.

**Word demo** (`--process word`):
1. Note which Word windows are already open, then open Word from the Start menu and wait for a *new* Word window to come to the front, so the demo never types into a document that was already open. If that window shows the **Blank document** tile (the Start screen), click it with the learned mouse model and wait for the tile to disappear; Enter is the fallback. Window titles are not used to detect the Start screen: with another document open, it is already titled "Document2 - Word".
2. Maximize that window, click near the top of the page, and press Ctrl+End. It stops if the document doesn't have keyboard focus.
3. Type the paragraph in compose mode: typos and corrections, thinking pauses, changed wordings, arrowing back to fix things.
4. Read the document text back through UI Automation (`TextPattern`) and report whether it matches. Curly quotes from AutoCorrect count as a match. The document is left open and unsaved.

| Function | Description |
|---|---|
| `run_word_demo(hm, args)` | The steps above. |
| `word_window_handles()` | Handles of every open Word window (class `OpusApp`). |
| `foreground_word_window(timeout, ignore)` | The Word window in front, skipping the handles in `ignore`. |
| `word_blank_document_tile(win, timeout)` | The "Blank document" tile if the window shows the Start screen, else None. |
| `word_document(win, timeout)` | `(window, document control)`: the editing surface, a `DocumentControl` named after the document. |
| `word_document_has_focus(doc)` | True if keyboard input would go into the document. |
| `word_document_text(doc)` | The document's text (paragraphs end in `\r`), or None. |
| `WORD_PARAGRAPH` | The built-in 153-word paragraph. |

Word flags: `--wpm` (default 80), `--text`, `--text-file`, `--revisions heuristic|llm`, `--pause-scale` (default 0.7), plus `--seed`, `--no-model` and `--temperature` for the mouse.

Measured run (80 wpm persona, 153 words): 271 s including thinking pauses (40 wpm effective); 1,222 keystrokes with 24 typos; the document text matched exactly.

---

## 5. What is learned vs. hand-set

| Behaviour | Source |
|---|---|
| Mouse path shape, speed profile and sampling rhythm | **Learned** (LSTM-MDN) |
| Click hold time and pre-click dwell | **Learned** (empirical from data) |
| Where inside an element to click | Parametric; the datasets have no element boundaries |
| Scroll bursts and reading pauses | Hand-set (`ReadingParams`); next candidate to learn. Balabit has scroll events. |
| Typing rhythm: key intervals, holds, rollover, Shift timing, typist-to-typist variation | **Learned** (Aalto: bigram statistics, or LSTM-MDN) |
| Typos, when they are noticed, how they are corrected | **Learned** (Aalto), rate calibrated by simulation |
| Thinking pauses, deletion sizes, lost-thought deletions, in-text revision rate and distance | **Learned** (KLiCKe) |
| *What* the changed wording is | Heuristic (synonyms, openers, anticipation) or LLM (`revisions="llm"`) |
| Arrow-key auto-repeat rate, 60-char revision reach, `pause_scale` default 0.7 | Hand-set |

## 6. Next steps
- Compare the browsing-only model with a general one trained without `--window-filter`, using `evaluate.py`.
- Learn scroll timing from Balabit's `Scroll` events, to replace the `ReadingParams` defaults.
- Typing: verify the field's final text after typing into fields that autocomplete (Chrome omnibox inline completion, Word autocorrect), where Backspace can behave differently; mine CoAuthor's real rewrites to improve the heuristic alternatives; learn word-level ctrl+Backspace from free-text data (e.g. Clarkson II).
- A session scheduler that strings tasks together (Word, email, browsing) for the 15-minute run.
