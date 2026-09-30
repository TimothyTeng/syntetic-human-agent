# Typing model

Types text like a person does. It has a realistic per-key rhythm and key overlap, and makes typos and fixes them. It also pauses to think, changes its mind about wording, loses its train of thought, and goes back to fix earlier words. All of this is learned from two public keystroke datasets.

`algorithms/human_typing.py` (`type_like_human`) is the high-level API the agent calls. This folder holds the data loaders, fitting and training code, the planner and the numpy runtime.

```
Aalto 136M keystrokes ─► dataset_aalto.py ─┬► timing.py   (bigram timing)  ─► typing_bigrams.json
   (copying sentences)                     ├► errors.py   (typos)          ─► typing_errors.json
                                           └► features.py + model.py + train.py (TensorFlow) ─► typing_mdn.npz
KLiCKe essay logs      ─► dataset_klicke.py ─► composition.py (pauses, revisions) ─► typing_composition.json
   (free writing)

type_like_human(text) ─► planner.py  pass 1: WHICH keys (typos, false starts, revisions)   ◄─ alternatives.py
                                     pass 2: WHEN      (sampler.py / timing.py + composition pauses)
                      ─► human_typing.play ─► key_down / key_up timeline ─► controller.keyboard
```

---

## 1. Training process and the deep learning used

Typing is modelled in two layers, following keystroke-logging research:

| Layer | Question | Learned from | How |
|---|---|---|---|
| **Motor** | How long after the last key is this key pressed, and for how long is it held? Which typos do fingers make? | Aalto 136M Keystrokes: 168k people copying sentences, with press and release times | LSTM-MDN (deep learning) or bigram statistics; typo statistics |
| **Cognitive** | Where does the writer stop to think? When do they delete and rewrite, or go back into earlier text? | KLiCKe: 2,471 essays written freely in 30 min | Empirical distributions |

### 1a. Motor timing network: LSTM + Mixture Density Network (`features.py`, `model.py`, `train.py`)

**Data.** Each Aalto sentence becomes one sequence of text keystrokes. Non-text keys such as Shift are dropped, but their time is kept in the interval.
- The target for keystroke j is `(log interval_j, log hold_j)`, where the interval is press→press and the hold is press→release. Both are standardised.
- The inputs for keystroke j are:

| input | meaning |
|---|---|
| `emb(prev key), emb(this key), emb(next key)` | 16-dim learned embeddings of 53 key classes (a–z, 0–9, punctuation keys, space, Backspace, Enter, nav). Case is ignored, because timing depends on the physical key |
| `log interval_{j-1}, log hold_{j-1}` | what the fingers just did (standardised) |
| `is_second` | first interval of the sentence |
| `speed` | the typist's standardised log median interval: the **persona**, so one network covers 17–130 wpm typists |

**Network** (about **230k parameters**):
```
[3 × Embedding(53,16) | 4 floats] ─► LSTM(128) ─► LSTM(128) ─► Dense(8 · 5)
                                                       ├─ 8 mixture logits
                                                       ├─ 8 × 2 means      (log interval, log hold)
                                                       └─ 8 × 2 log std-devs
```
- **Why an LSTM:** real typing has memory. Tempo drifts, bursts speed up, and after a slow key the next is often slow too. Keys also interact: rollover (pressing the next key before releasing the previous) depends on the hold *and* the next interval together.
- **Why an MDN:** the interval before a key is multi-modal. It can be fluent (~100 ms), a hesitation (~300 ms) or a correction pause (~700 ms). An MDN outputs the whole distribution, and at runtime we *sample* from it. A regression head would output the mean, which is too regular and easy to detect.
- **Loss:** the masked negative log-likelihood of the real (interval, hold) pair under the 8-component diagonal Gaussian mixture.

**Training** (`train.py`):
- **Data:** 4,000 typists (52,722 sentences) for training; 300 *different* typists for validation (1 in 20 participant ids is held out).
- **Batching:** by similar length, with teacher forcing.
- **Optimiser:** Adam (lr 2e-3, clip-norm 1.0), `ReduceLROnPlateau(0.5, patience 1)`, `EarlyStopping(patience 3, restore best)`. Takes about 1–1.5 min per epoch on CPU.
- **After training:**
  1. **export** to `typing_mdn.npz`;
  2. **numpy/TF parity check** (max abs diff 1.4e-6);
  3. **speed calibration**: generate text at many personas, measure achieved/requested wpm (1.118) and store it as `wpm_scale`, so a 45 wpm persona really types at about 45 wpm.

**Runtime** (`sampler.py`): the same forward pass in numpy. For each key it picks a mixture component, samples (log interval, log hold), and feeds the sample back in as the next "previous" input. The state resets every ~80 keys at a space, because training sequences are sentences.

### 1b. Statistical models (fitted by `fit_stats.py`, about 1 min)

- **`timing.BigramTimer`** (fallback motor model and the source of Shift timing):
  - For each typist, intervals are divided by the typist's median. For each key-class bigram, 41 quantiles of `log(interval/median)` are stored (762 bigrams), plus per-key hold quantiles.
  - Sampling uses an AR(1) Gaussian copula for tempo drift.
  - Persona variation between typists is measured from the data: hold level SD 0.21, rhythm spread SD 0.24.
  - Shift lead and lag samples are also stored here.
- **`errors.ErrorStats`**: every Aalto sentence is replayed against its target text. An *error onset* is the keystroke that turns a correct prefix into a wrong one (104,906 onsets from 5,367 typists). For each onset it records:
  - **type:** adjacent key 26%, other substitution 20%, omission 28%, insertion 9%, transposition 7%, case 10%;
  - **detection delay:** how many more keys were typed before the first Backspace; 48% are noticed immediately;
  - **over-deletion:** extra Backspaces;
  - **error rate by speed**, and relative error rates per key.

  **Calibration:** the generator always fixes its typos (real typists leave about 15%) and samples lengths independently. `fit_stats` therefore simulates the model on real sentences and scales the rate (×0.67) until Backspaces per key match real typing: 0.060 real vs 0.061 generated.
- **`composition.CompositionStats`**: each KLiCKe essay is replayed into a masked text buffer, so every keystroke gets a **pause class** from its context:
  - classes: in word, word end, between words, after clause, after sentence, after paragraph, punctuation, before/in/after a deletion;
  - whether it happened at the end of the text or inside it.

  It stores, per class, P(pause ≥ 2 s) plus short-pause ratios and long-pause seconds. It also stores deletion-run rates and sizes, P(deletion after a long pause) = 0.16, and the in-text revision rate, distance and size.

---

## 2. Considerations

| Topic | Decision and why |
|---|---|
| **Two datasets** | No single public dataset has both *free composition* and *unmasked text with typo labels*. Aalto gives exact motor behaviour and typo labels (it's copying). KLiCKe gives real thinking and revision behaviour (text masked as `q`). |
| **Final text must be exact** | The planner keeps a virtual text buffer and cursor, and every behaviour is expressed as "type something, then fix it back". `selftest.py` replays hundreds of random plans through a simulated keyboard (Shift included) and demands an exact match. `allow_uncorrected=True` opts out. |
| **Transcribe vs compose** | Short inputs (search boxes, fields) get motor rhythm and typos only: nobody "changes their mind" typing a query. Longer text (≥ 12 words) also gets thinking pauses, false starts, lost thoughts and in-text revisions. |
| **Change-of-idea wording** | The data says *when* and *how much* people rewrite, not *what* (the text is masked). The replacement wording comes from `alternatives.py`: offline heuristics by default, or one Claude call per text (`revisions="llm"`, model `claude-opus-5` with server-side refusal fallback). It falls back to heuristics on any error, including a missing package or key. |
| **In-text revisions** | These use arrow keys, which behave the same in every text field. Ctrl+Left word jumps and Home/End differ between apps and line wraps. Trips are capped at 60 characters (`max_travel`); a flaw that couldn't be reached is not planned. |
| **Typos noticed late** | The continuation after a typo is capped at the end of the next word. Aalto had no arrow keys, so its typists sometimes backspaced 20+ characters. In compose mode, a later notice becomes an arrow-key revision instead. |
| **Thinking-pause level** | KLiCKe is timed argumentative essays, which are pause-heavy. The default `pause_scale=0.7` and `max_pause=8 s` suit emails and documents; 1.0 reproduces the essays. |
| **Rollover and Shift** | The player sends real `key_down` / `key_up` events, so keys overlap like real typing. Shift goes down a learned lead before a capital and comes up after it, never while an unshifted key is pressed. Holds stay below the OS auto-repeat delay (0.35 s cap), a key is released before it is pressed again, and the minimum interval is 12 ms. |
| **pyautogui quirk** | `pyautogui.keyDown('A')` presses and releases Shift *inside* that call, which would destroy Shift timing. The player presses the base key (`a`) with an explicit Shift, looked up from the active keyboard layout. Characters the layout can't press directly are typed atomically. |
| **Timing precision** | Hybrid sleep + spin wait against `perf_counter`. Measured on real OS events: median 0.1–0.4 ms off the plan, p95 < 2 ms. |
| **Safety** | Held keys (especially Shift) are always released in a `finally`. The pyautogui fail-safe still works. The Word demo refuses to type unless the document has keyboard focus. |
| **Evaluation fairness** | About 30% of Aalto browsers recorded timestamps on 4/8/16 ms grids. `evaluate.py` rounds generated times to the same grid as the typist they're compared with, and requests the typist's *measured* speed. Aalto's `AVG_WPM_15` runs about 10% below first-to-last-key speed. |
| **Autocomplete / autocorrect** | Chrome's omnibox inline completion can swallow a Backspace; `main.py` checks the bar afterwards. Word AutoCorrect changes are length-preserving (capitalisation, curly quotes), so the buffer model still holds. |
| **Licences** | Aalto: **non-commercial use with attribution**. KLiCKe: Kaggle competition rules. Only derived statistics and weights are stored in `models/`. |

---

## 3. What is in each file

### Data loading
| File | Purpose | Key functions / classes |
|---|---|---|
| `dataset_aalto.py` | Streams Aalto participant files from `Keystrokes.zip`; QWERTY desktop/laptop filter; participant-level split | `iter_participants(zip, split="train"\|"val"\|"all", max_participants, seed)` → `(wpm, [Trial])`; `Trial(sentence, user_input, keys, press, release, names)`; `load_metadata`, `select_participants`, `is_validation(pid)`, `parse_file` |
| `dataset_klicke.py` | Reads `train_logs.csv` from the Kaggle zip and replays each essay into a masked text buffer | `load_logs(path)`, `iter_essays(path_or_df, max_essays, seed)` → `Essay(kind, down, up, char, at_edge, prev, text_len, pos, final_text)`; kinds `INPUT/REMOVE/NAV/OTHER` |
| `keys.py` | QWERTY geometry, shift map, key-class vocabulary | `neighbours(ch)`, `base_key(ch)`, `needs_shift(ch)`, `hand(ch)`, `key_class(ch)`, `timing_token(ch)`, `VOCAB`; special keys `BKSP`, `LEFT`, `RIGHT` |

### Fitting / training (offline)
| File | Purpose | Key functions / classes |
|---|---|---|
| `fit_stats.py` | CLI: fits timing, errors (with calibration) and composition; writes the JSON files | `main()`, `calibrate_error_rate(parts, timer, errors, rng)` |
| `timing.py` | Bigram motor timing (fit + sample) | `BigramTimer.fit(participants)`, `.load/.save/.default()`, `.median_iki(wpm)`, `.sample_shift(rng)`, `.session(wpm, rng, temperature).sample(prev, key, next)` → `(interval, hold)` |
| `errors.py` | Typo statistics (fit + sample) | `ErrorStats.fit(...)`, `.rate(wpm)`, `.p_error(ch, wpm, scale)`, `.sample_type/.sample_delay/.sample_extra(rng)`, `.scale_rate(f)`, `.summary()`; `classify_error(typed, target, p)`; `make_typo(target, j, type, rng)` |
| `composition.py` | Composition pauses and revision rates (fit + sample) | `CompositionStats.fit(essays)`, `.sample_pause(class, base_iki, rng, long_scale, max_pause)` → `(s, is_long)`, `.sample_short(...)`, `.p_long(class)`, `.delete_rate(bucket)`, `.sample_delete_size(bucket, rng)`; `pause_class(ch, prev_text, prev_kind, kind)` |
| `features.py` | Network input/target encoding (shared by train and runtime) | `NormStats` (+ wpm→median regression), `trial_arrays(trial)`, `participant_median(trials)`, `encode_trial(...)`, `make_step_input(...)` |
| `model.py` | Keras LSTM-MDN (training only) | `build_model(units=128, layers=2, mixtures=8, emb=16)`, `split_params`, `make_loss(K)`, `export_npz(...)` |
| `train.py` | CLI: collect → normalise → train → export → parity → speed calibration | `collect(...)`, `fit_stats(rows)`, `encode(rows, stats)`, `Batches(...).dataset()`, `parity_check(...)`, `calibrate_speed(npz_path)`, `main()` |

### Runtime (numpy only)
| File | Purpose | Key functions / classes |
|---|---|---|
| `planner.py` | **The core.** Pass 1 (`_Writer`) decides the key sequence; pass 2 (`assign_timing`) gives every key its interval and hold | `plan_keystrokes(text, model, rng, cfg, provider)` → `[Keystroke]`; `TypingConfig`; `Keystroke(key, tag, press, hold, iki, pclass, long)`; `auto_mode(text)` |
| `sampler.py` | Numpy LSTM-MDN, same `session().sample()` interface as `BigramTimer` | `TypingMDN.load(path)`, `.median_iki(wpm)`, `.session(wpm, rng, temperature)`, `.forward_sequence(tokens, floats)` |
| `alternatives.py` | Change-of-idea wordings | `HeuristicAlternatives`, `LLMAlternatives(model, timeout, per_words)`, `.prepare(text, rng)`, `.positions()`, `.alternative(target, start, approx_len, rng)`; `make_provider(kind)`; `SYNONYMS`, `OPENERS` |
| `runtime.py` | Loads everything with fallbacks; plan utilities | `TypingModel.load(models_dir, use_mdn=True)` (`.timer`, `.errors`, `.comp`, `.sources`); `apply_plan(plan)` → text; `describe(plan)` → readable trace |

### Checks
| File | Purpose |
|---|---|
| `selftest.py` | Random plans at random speeds, error rates and revision rates. Checks the plan replays to the exact text, and that the player's key timeline produces it on a simulated US keyboard with no key pressed in the wrong Shift state |
| `evaluate.py` | Motor realism (KS distances plus a gradient-boosting real-vs-synthetic detector, cross-validated AUC) on unseen Aalto typists, and composition realism on KLiCKe essays. Helpers: `sentence_features`, `timer_resolution`, `measured_wpm`, `placeholder_plan` (the old baseline) |

### Outside this folder
| File | Purpose |
|---|---|
| `algorithms/human_typing.py` | `type_like_human(...)` (plan + play), `play(plan)`, `build_timeline(plan, rng)`, `get_model()` |
| `controller/keyboard.py` | `key_down`, `key_up`, `type_char` (the only way keys reach the OS) |
| `main.py --process word` | Word demo: opens Word, clicks "Blank document", types a paragraph, reads the document back to verify |

### Model files (`models/`)
| File | Contents | From |
|---|---|---|
| `typing_mdn.npz` (860 KB) | embeddings, LSTM and MDN weights, normalisation, wpm regression, `wpm_scale`, JSON config | `train.py` |
| `typing_bigrams.json` | bigram / hold quantiles, speed regressions, tempo rho, Shift lead/lag, persona spread | `fit_stats.py --aalto` |
| `typing_errors.json` | typo rates by wpm (calibrated), type mix, delays, over-deletion, per-key weights | `fit_stats.py --aalto` |
| `typing_composition.json` | pause classes, deletion and revision rates and sizes | `fit_stats.py --klicke` |
| `typing_training_info.json`, `typing_train_log.txt`, `typing_fit_info.json` | run settings, losses, parity, calibration | both |

If any file is missing, `TypingModel.load` falls back in the order MDN → bigrams → built-in defaults, so the agent always runs.

---

## 4. How to use it

### Fit, train and check (from the project root)
```bash
# 1) statistics (~1 min). Either dataset can be given on its own.
python -m algorithms.typing_model.fit_stats --aalto data/aalto_keystrokes/Keystrokes.zip --klicke data/klicke/linking-writing-processes-to-writing-quality.zip

# 2) neural timing model (optional, ~25 min on CPU); smoke run first
python -m algorithms.typing_model.train --data data/aalto_keystrokes/Keystrokes.zip --participants 200 --epochs 2 --out scratch_models
python -m algorithms.typing_model.train --data data/aalto_keystrokes/Keystrokes.zip

# 3) checks
python -m algorithms.typing_model.selftest --n 600
python -m algorithms.typing_model.evaluate --aalto data/aalto_keystrokes/Keystrokes.zip --klicke data/klicke/linking-writing-processes-to-writing-quality.zip
```

### Type into whatever window has focus
```python
import numpy as np
from algorithms import human_typing

rng = np.random.default_rng(7)

# short input: rhythm + typos only (auto-detected, < 12 words)
human_typing.type_like_human("weather in singapore tomorrow", rng=rng, wpm=60)

# longer text: thinking pauses, false starts, revisions
human_typing.type_like_human(open("email.txt").read(), rng=rng, wpm=80, mode="compose")

# snappier: fewer and shorter pauses, fewer rewrites, a careful typist
human_typing.type_like_human(text, rng=rng, wpm=70, pause_scale=0.3, max_pause=3,
                             revision_scale=0.5, error_scale=0.5)

# first-draft wordings from Claude (pip install anthropic + credentials; falls back if unavailable)
human_typing.type_like_human(text, rng=rng, wpm=55, revisions="llm")
```

### Plan without typing (inspect, test, estimate duration)
```python
from algorithms import human_typing
from algorithms.typing_model.runtime import apply_plan, describe

plan = human_typing.type_like_human(text, rng=rng, wpm=80, dry_run=True)
print(describe(plan))            # ~ = Backspace, < > = arrow keys, [2.1s] = long pause
print(f"{plan[-1].press:.0f} s, {len(plan)} keys, exact: {apply_plan(plan) == text}")
print({t: sum(k.tag == t for k in plan) for t in {k.tag for k in plan}})
```
Example trace (80 wpm persona):
```
Over the past quarter, our team has focused on improving ht~~the reliailit
y [2.7s]<<<<<~[1.5s]bi>>>>>of the scheduling [1.8s]z~system and ...
```

### Lower level: your own model directory or config
```python
from algorithms.typing_model.runtime import TypingModel
from algorithms.typing_model.planner import TypingConfig, plan_keystrokes
from algorithms.typing_model.alternatives import HeuristicAlternatives

model = TypingModel.load("models", use_mdn=True)
print(model.sources)     # {'timing': 'typing_mdn.npz', 'errors': 'aalto', 'composition': 'klicke'}
cfg = TypingConfig(wpm=65, mode="compose", max_travel=30, intext_revisions=True)
plan = plan_keystrokes(text, model, rng, cfg, HeuristicAlternatives())
human_typing.play(plan, rng)
```

### Word demo
```bash
python main.py --process word                       # 80 wpm persona, built-in paragraph
python main.py --process word --wpm 55 --text-file notes.txt --pause-scale 0.4
```

---

## 5. Current results

**Motor realism.** 810 sentences from 150 unseen typists, each re-typed at that typist's speed and clock resolution. KS: 0 = same distribution. AUC: 0.5 = a detector can't tell generated from real.

| Generator | interval KS | hold KS | speed KS | Backspace KS | rollover KS | tempo lag-1 KS | **detector AUC** |
|---|---|---|---|---|---|---|---|
| old placeholder | 0.47 | 1.00 | 0.58 | 0.60 | 0.87 | 0.19 | **1.00** |
| built-in defaults | 0.20 | 0.47 | 0.10 | 0.05 | 0.63 | 0.38 | **0.99** |
| bigram statistics | 0.06 | 0.03 | 0.07 | 0.05 | 0.13 | 0.09 | **0.66** |
| **LSTM-MDN** | **0.011** | **0.024** | **0.038** | **0.028** | **0.059** | **0.063** | **0.52** |

**Composition realism.** 100 essays re-typed at essay pause levels. Share of keystrokes preceded by a pause of 2 s or more:

| Location | real | generated |
|---|---|---|
| inside a word | 0.2% | 0.0% |
| between words | 6.4% | 7.6% |
| after a clause | 10.1% | 16.0% |
| after a sentence | 40.2% | 43.8% |

Deletion runs per 100 chars: real 3.3, generated 3.9. In-text revisions per 1,000 chars: real 1.3, generated 1.7.

**Live:**
- **Test window:** exact text; key timing within 0.1–0.4 ms of the plan.
- **Word demo** (153 words, 80 wpm persona): exact text in the document; 271 s (40 wpm effective, which is normal for composing); 1,222 keystrokes including 24 typos, 62 keys of changed wordings and 112 arrow presses for going back to fix things.

**Speed:** planning takes about 5 ms per sentence and about 35 ms per email. TensorFlow is never imported at runtime.

---

## 6. Troubleshooting and ideas

- **Typing feels too slow for a long document:** lower `pause_scale` (e.g. 0.3) and `max_pause` (e.g. 3), or `revision_scale`. `wpm` only sets finger speed; pauses dominate composition time.
- **Wrong characters on a non-US layout:** characters whose physical key differs from US QWERTY are typed atomically, which is correct but has no rollover. Typos use US QWERTY neighbours.
- **Text differs in an app with autocomplete:** read the field back afterwards (see `main.py`: `browser.get_current_url()` for Chrome, `word_document_text()` for Word) and retype if needed.
- **`revisions="llm"` does nothing different:** check `LLMAlternatives().prepare(text, rng)` and its `.error`. Usually the `anthropic` package or credentials are missing.
- **Ideas:**
  - Learn *what* people rewrite from CoAuthor (unmasked text).
  - Validate composition motor timing on Clarkson II (free text typed naturally in daily use).
  - Add ctrl+Backspace word deletion and mouse-click navigation for long-distance revisions.
  - Fine-tune the network on KLiCKe's timing (letters are masked, but intervals and holds are real).
