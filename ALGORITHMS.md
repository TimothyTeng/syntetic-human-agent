# Algorithms — Behaviour Layer Reference

`controller/` knows **what** can be done: move to (x, y), click, type. `algorithms/` decides **how** a person would do it: the path the mouse takes, how fast, how long the button stays down, and how someone scrolls while reading. `main.py` runs a Chrome demo that uses both.

```
main.py ──► algorithms.HumanMouse ──► mouse_model (trained LSTM-MDN, numpy runtime)
        │                          ──► click_stats (learned click timing)
        │                          ──► targeting / player ──► controller.mouse
        ├─► algorithms.reading ─────► HumanMouse + controller.mouse.scroll
        ├─► algorithms.human_typing ► controller.keyboard
        └─► controller.browser / apps / ui_elements (find targets on screen)
```

---

## 1. Quick start

```bash
pip install -r requirements.txt

# 1) dataset: the BMDD split archive (boun-mouse-dynamics-dataset.zip + .z01-.z09) in the
#    project root is read directly - no extraction needed (see data/README.md)

# 2) train (CPU). Smoke run first, then the real run (~5 min/epoch for 40k strokes)
python -m algorithms.mouse_model.train --data boun-mouse-dynamics-dataset.zip --out models --max-strokes 5000 --epochs 1
python -m algorithms.mouse_model.train --data boun-mouse-dynamics-dataset.zip --out models --window-filter browsing --max-strokes 40000 --epochs 25

# 3) check realism against held-out real strokes (BMDD test folders)
python -m algorithms.mouse_model.evaluate --data boun-mouse-dynamics-dataset.zip --window-filter browsing

# 4) run the Chrome demo (works before training too; it uses the fallback generator)
python main.py
python main.py --no-model --query "wikipedia" --read-seconds 20 --seed 42
```

Abort `main.py` at any time by slamming the mouse into a screen corner (the pyautogui fail-safe).

---

## 2. The mouse model

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

## 3. Function reference

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
| `type_like_human(text, rng=None, wpm=45, sigma=0.35)` | **Placeholder:** log-normal per-key delays, with longer gaps after spaces and punctuation. To be replaced by a learned keystroke model. |

### `main.py`
The Chrome demo:
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

---

## 4. What is learned vs. hand-set

| Behaviour | Source |
|---|---|
| Mouse path shape, speed profile and sampling rhythm | **Learned** (LSTM-MDN) |
| Click hold time and pre-click dwell | **Learned** (empirical from data) |
| Where inside an element to click | Parametric; the datasets have no element boundaries |
| Scroll bursts and reading pauses | Hand-set (`ReadingParams`); next candidate to learn. Balabit has scroll events. |
| Typing rhythm | Hand-set placeholder |

## 5. Next steps
- Compare the browsing-only model with a general one trained without `--window-filter`, using `evaluate.py`.
- Learn scroll timing from Balabit's `Scroll` events, to replace the `ReadingParams` defaults.
- A keystroke-dynamics model for typing, e.g. from a public keystroke dataset, behind the same `type_like_human` signature.
- A session scheduler that strings tasks together (Word, email, browsing) for the 15-minute run.
