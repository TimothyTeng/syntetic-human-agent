# Mouse model

Generates human-like mouse movements (path shape, speed profile, sampling rhythm) and click timing. It learns them from real recorded mouse use. `algorithms/human_mouse.py` (`HumanMouse`) is the high-level API the agent calls. This folder holds the data pipeline, the training code and the numpy runtime.

```
BMDD logs ──► dataset.py ──► strokes.py ──► features.py ──► model.py + train.py (TensorFlow)
                                   │                                  │
                                   └► click_stats.py                   ▼
                                                       models/mouse_mdn.npz + click_stats.json
                                                                      │
HumanMouse.move_to / click_rect ──► sampler.py (numpy) ──► player.py ──► controller.mouse
                                  └► fallback.py (no model / model miss)
```

---

## 1. Training process and the deep learning used

### Why a generative sequence model (and not RL or hand-made curves)
- **Not reinforcement learning:** RL needs a reward, and there is no natural reward for "moves like a human". What we do have is **recorded human movements**, so the right tool is a model that imitates them, trained by maximum likelihood.
- **Not hand-made curves:** Bézier curves or minimum-jerk paths (see `fallback.py`) look smooth but too regular. Real movements have overshoot, corrections, uneven sampling and speed profiles that vary from person to person. A learned model reproduces the whole distribution of these, not just an average curve.

### Data preparation
1. **Sessions** (`dataset.py`): raw logs become `Session(t, x, y, button, state, window)` arrays.
   - Files are read one at a time, because BMDD is 33.8 GB uncompressed.
   - The split `.z01…/.zip` archive is read in place by `splitzip.py`.
2. **Strokes** (`strokes.py`): a stroke is the run of Move events that ends in a left-button press, starting from the cursor's resting point.
   - Runs are broken at pauses of more than 0.5 s.
   - Strokes are rejected if they are shorter than 15 px, outside 0.08–4 s, or outside 5–200 samples.
   - The **dwell** (last move → press) and **hold** (press → release) are kept for `click_stats.py`.
3. **Canonical frame:** each stroke is translated to start at (0,0), rotated so the target lies on +x, and scaled so the target sits at (1,0). The pixel distance D is kept as an input. One model then covers every direction, distance and screen resolution.
4. **Features** (`features.py`): each step j sees 7 inputs.

   | input | meaning |
   |---|---|
   | `dx_prev, dy_prev, log dt_prev` | the previous step, standardised (zeros at j = 0) |
   | `is_first` | 1 on the first step |
   | `rem_x, rem_y` | remaining vector to the target (canonical units) |
   | `log D` | log pixel distance of the move, standardised |

   The target is the next `(dx, dy, log dt)` plus an **end-of-stroke** flag. Mean and std come from the training set and are saved in the `.npz`.

### Network: LSTM + Mixture Density Network (MDN)
```
input (7) ─► LSTM(128) ─► LSTM(128) ─► Dense(10·7 + 1)
                                          ├─ 10 mixture logits
                                          ├─ 10 × 3 means        (dx, dy, log dt)
                                          ├─ 10 × 3 log std-devs
                                          └─ 1 end-of-stroke logit
```
About **210k parameters**.
- **The LSTM** carries the state of the movement so far: speeding up, cruising, braking, corrections.
- **The MDN head** outputs a *distribution* over the next step, not a single value. It is a mixture of K = 10 diagonal Gaussians, the same idea as Graves' handwriting generation. Human movement is multi-modal (keep going vs. brake vs. correct sideways). A plain regression head would average those options into an unrealistic mean path.
- **The end-of-stroke Bernoulli** lets the model decide when the movement is over.

**Loss:** the masked negative log-likelihood of the real next step under the mixture, plus binary cross-entropy for the end flag. Padding steps are masked out.

```
NLL = -log Σ_k π_k · N(step | μ_k, diag σ_k²)  +  BCE(end, sigmoid(end_logit))
```

Training details (`train.py`):
- **Teacher forcing:** during training the model sees the real previous step.
- **Held-out users:** validation uses whole users the model never trained on, so it measures generalisation to new people.
- **Length bucketing:** strokes of similar length are batched together (padded to a multiple of 16), which saves compute.
- **Optimiser:** Adam (lr 1e-3, gradient clip-norm 1.0).
- **Schedule:** `ReduceLROnPlateau(0.5, patience 2)` and `EarlyStopping(patience 4, restore best)`.
- **End-flag initialisation:** the end-flag bias starts at the data's base rate (about 1/stroke length). Otherwise a fresh model predicts "stop" 50% of the time and makes one-step strokes.
- **Export:** weights are saved to `models/mouse_mdn.npz`, then a **parity check** runs the numpy runtime on validation strokes and compares with Keras (max abs diff must be < 1e-3).

### Generation (runtime, `sampler.py`)
1. Run the LSTM one step at a time in numpy.
2. At each step, pick a mixture component (softmax / temperature), sample a step from its Gaussian, and un-standardise.
3. The stop signal is only accepted once the cursor is within 25% of the distance of the target (`end_tolerance_frac`). Without this gate, strokes sometimes stop early (mean KS 0.198 vs 0.135 with the gate).
4. Leftover landing error is **spread along the path in proportion to arc length**, so the stroke ends *exactly* on the target with no visible jump.
5. If it fails 3 times, it returns `None`, and `HumanMouse` uses `fallback.generate`.

---

## 2. Considerations

| Topic | Decision and why |
|---|---|
| **Dataset** | BMDD: native Windows logger, 24 users, about 2,550 h of real use, CC BY 4.0. Balabit is supported but is captured over RDP, so its timing and resolution are worse. |
| **Which clicks** | Trained on the `browsing` window category (`--window-filter browsing`), because the demo drives Chrome. A general model can be trained without the filter. |
| **TensorFlow only for training** | TF costs about 500 MB of RAM and seconds to import. The runtime is pure numpy (about 11 ms per trajectory, about 100 MB RSS), which helps the Resource Efficiency score. |
| **Canonical frame** | Rotation/scale invariance means one model for all directions and distances. D is still an input, so long moves behave differently from short ones (Fitts' law). |
| **Temperature** | `<1` gives calmer, more average paths; `>1` gives more varied ones. 1.0 reproduces the data. |
| **Ending exactly on target** | The warp to target is necessary for clicking UI elements. It is spread over the path so no final jump is visible. |
| **Where to click inside an element** | `targeting.pick_click_point`: Gaussian scatter around the centre, kept inside a margin. This is parametric, because the datasets don't record element boundaries. |
| **Playback accuracy** | `player.py` sets the Windows timer to 1 ms (`timeBeginPeriod`), since the default is 15.6 ms and samples are about 8 ms apart. Points are scheduled against `perf_counter`, so timing doesn't drift. |
| **Safety** | The pyautogui fail-safe stays on: slam the mouse into a corner to abort. Generated paths are clamped 1 px away from the corner pixels, so they can't trigger it by accident. |
| **DPI** | `controller/config.py` makes the process per-monitor DPI aware, so UIA rectangles and mouse coordinates agree on scaled displays. |
| **Known gap** | Sample spacing is about 40% longer than real (mean dt KS 0.29), so strokes have fewer points than real ones. |

---

## 3. What is in each file

### Training / data (TensorFlow needed only for `model.py` and `train.py`)
| File | Purpose | Key functions / classes |
|---|---|---|
| `splitzip.py` | Reads split ZIP64 archives (`.z01…/.zip`) in place | `SplitZip(zip_path)`, `.names(pattern)`, `.read(name)`, `.extract(name, dest)`, `.summary()`; CLI `list` / `extract` |
| `dataset.py` | Raw logs → `Session` arrays; BMDD + Balabit | `iter_sessions(dataset, root, split=None, max_files=None, users=None, shuffle=True, seed=0)`, `list_files(...)`, `detect_format(root)`, `peek(dataset, root)`; event codes `MOVE/PRESS/RELEASE/DRAG/SCROLL`, `LEFT/RIGHT` |
| `strokes.py` | Sessions → point-and-click strokes; canonical frame | `segment_strokes(session, gap=0.5, min_dist=15, min_dur=0.08, max_dur=4, min_points=5, max_points=200, window_filter=None)` → `[Stroke(points, times, user, dwell, hold)]`; `to_canonical(points)` → `(canon, D, angle)`; `from_canonical(canon, start, D, angle)` |
| `features.py` | Shared input/target encoding (train + runtime) | `NormStats.fit(strokes)` / `.to_dict()` / `.from_dict()`, `raw_steps(stroke)`, `encode_stroke(stroke, stats)` → `(X (m,7), Y (m,4))`, `make_input(prev, is_first, pos, logd)` |
| `click_stats.py` | Empirical click hold and dwell times | `ClickStats.fit(strokes)`, `.save/.load/.default()`, `.sample_hold(rng)`, `.sample_dwell(rng)`, `.summary()` |
| `model.py` | Keras LSTM-MDN (training only) | `build_model(units=128, layers=2, mixtures=10, end_prior=0.02)`, `split_params(out, K)`, `make_loss(K)`, `export_npz(model, stats, path, ...)` |
| `train.py` | CLI: collect strokes → split by user → train → export → parity check | `collect_strokes(...)`, `split_by_user(strokes, val_frac, rng)`, `StrokeBatches(...).to_tf_dataset()`, `parity_check(...)`, `main()` |
| `evaluate.py` | CLI: real vs generated strokes (KS, medians/IQR, Fitts fit, optional plot) | `stroke_metrics(points, times)`, `trajectory_to_arrays(start, traj)`, `fitts_fit(rows)`, `main()` |

### Runtime (numpy only)
| File | Purpose | Key functions / classes |
|---|---|---|
| `sampler.py` | Runs the exported network and samples trajectories | `MouseModel.load(path)`, `.generate(start, end, rng, temperature=1.0, retries=3, max_error=0.35)` → `[(x, y, dt), ...]` or `None`; `.generate_canonical(distance, rng, ...)`; `.forward_sequence(X)` (parity tests) |
| `fallback.py` | Minimum-jerk path with arc and noise; no training | `generate(start, end, rng, sample_dt=0.008)`, `minimum_jerk(tau)` |

### Used from outside this folder
| File | Purpose |
|---|---|
| `algorithms/human_mouse.py` | `HumanMouse`: `plan`, `move_to`, `move_to_rect`, `drift`, `press_release`, `click_at`, `click_rect`, `click_element`, `double_click_rect`, `.backend`, `.rng` |
| `algorithms/player.py` | `play_trajectory(points, speed=1.0)`, `high_res_timer(ms=1)` |
| `algorithms/targeting.py` | `pick_click_point(rect, rng)`, `random_point_in(rect, rng)` |
| `algorithms/reading.py` | `read_page(hm, duration, page_rect)`: scroll bursts, pauses and drifts built on `HumanMouse` |

### Model files (`models/`)
| File | Contents |
|---|---|
| `mouse_mdn.npz` | LSTM/Dense weights, normalisation stats and a JSON `config` (units, layers, mixtures) |
| `click_stats.json` | Up to 5,000 real hold and dwell samples |
| `training_info.json`, `train_log.txt` | Settings, losses and parity result of the last run |

---

## 4. How to use it

### Train and evaluate (from the project root)
```bash
# check the archive and column mapping
python -m algorithms.mouse_model.splitzip list data/BMDD/boun-mouse-dynamics-dataset.zip
python -c "from algorithms.mouse_model.dataset import peek; peek('bmdd', 'data/BMDD/boun-mouse-dynamics-dataset.zip')"

# smoke run, then the real run (~5 min/epoch on CPU for 40k strokes)
python -m algorithms.mouse_model.train --data data/BMDD/boun-mouse-dynamics-dataset.zip --max-strokes 5000 --epochs 1
python -m algorithms.mouse_model.train --data data/BMDD/boun-mouse-dynamics-dataset.zip --window-filter browsing --max-strokes 40000 --epochs 25

# compare with held-out real strokes (BMDD *_tests folders); --plot needs matplotlib
python -m algorithms.mouse_model.evaluate --data data/BMDD/boun-mouse-dynamics-dataset.zip --window-filter browsing --plot eval.png
```

Useful `train.py` flags: `--max-strokes`, `--max-files`, `--window-filter`, `--epochs`, `--batch`, `--units`, `--layers`, `--mixtures`, `--lr`, `--val-frac`, `--seed`, `--dataset bmdd|balabit|auto`.

### Use it in agent code
```python
from algorithms.human_mouse import HumanMouse
from controller import browser

hm = HumanMouse(seed=42)                 # loads models/mouse_mdn.npz if present
print(hm.backend)                        # 'lstm-mdn' or 'fallback'

hm.click_rect(browser.get_address_bar_rect())   # move to a human-chosen point inside, then click
hm.move_to(800, 450)                            # just move
hm.drift(max_dist=60)                           # small idle movement while "reading"
```

### Generate without moving (e.g. to inspect or plot)
```python
import numpy as np
from algorithms.mouse_model.sampler import MouseModel
from algorithms.mouse_model import fallback

model = MouseModel.load("models/mouse_mdn.npz")
rng = np.random.default_rng(0)
traj = model.generate((200, 300), (1200, 650), rng, temperature=1.0)   # [(x, y, dt), ...]
if traj is None:                                   # rare: model missed the target 3 times
    traj = fallback.generate((200, 300), (1200, 650), rng)
print(len(traj), "points,", sum(dt for _, _, dt in traj), "seconds")
```

### Click timing on its own
```python
from algorithms.mouse_model.click_stats import ClickStats
cs = ClickStats.load("models/click_stats.json")
print(cs.summary(), cs.sample_hold(rng), cs.sample_dwell(rng))
```

---

## 5. Current results

- **Training:** BMDD `training` folders, browsing clicks only, 40,000 strokes (33,012 train / 6,988 val, users 1 and 10 held out). 2×128 LSTM with 10 mixtures. Stopped early after 17 epochs; best validation NLL −2.96; numpy/TF parity 3.3e-6.
- **Click timing:** hold median 102 ms, dwell median 78 ms.

Evaluation against 600 held-out real browsing strokes (KS: 0 = same distribution as real):

| Metric | Real median | Model median | KS model | KS fallback |
|---|---|---|---|---|
| duration (s) | 0.81 | 0.72 | **0.11** | 0.35 |
| efficiency (straightness) | 0.86 | 0.76 | **0.16** | 0.36 |
| peak speed (px/s) | 3355 | 3629 | **0.06** | 0.49 |
| max sideways deviation | 0.13 | 0.11 | **0.08** | 0.50 |
| overshoot | — | — | **0.09** | 0.32 |
| n points | 64 | 46 | **0.17** | 0.29 |
| mean dt (s) | 0.012 | 0.017 | **0.29** | 0.88 |

The model reaches the target on 599 of 600 strokes; the fallback covers the rest.

---

## 6. Troubleshooting and ideas

- **`backend == 'fallback'`:** `models/mouse_mdn.npz` is missing, or `use_model=False` / `--no-model` was used.
- **Clicks land offset on a scaled second monitor:** make sure `controller.config` is imported before pyautogui (the controller package does this).
- **The script stops suddenly:** the fail-safe fired because the mouse hit a screen corner. This is intentional.
- **Next steps:**
  - More strokes or `--units 192` to fix the sparse sampling (mean dt).
  - A general model without `--window-filter` for Office apps.
  - Learn scrolling from Balabit `Scroll` events to replace the hand-set `ReadingParams`.
