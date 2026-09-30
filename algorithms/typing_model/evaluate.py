"""
How human is the generated typing? Compare it with held-out real data.

Motor level (Aalto held-out participants, transcription):
    Every real sentence is re-typed by each generator at the typist's own measured speed, and
    its timestamps are rounded to that typist's browser clock resolution (1-16 ms),
    so the detector cannot win on recording artefacts.
    Reported: KS distance (0 = identical distributions) of per-key interval, hold,
    and per-sentence speed / backspace rate / rollover rate / tempo autocorrelation,
    plus a real-vs-synthetic classifier (gradient boosting on per-sentence keystroke
    features, 5-fold CV ROC-AUC: 0.5 = indistinguishable, 1.0 = trivially detected).

Composition level (KLiCKe essays, compose mode):
    Masked essay texts ("qqq qq.") are re-typed; long-pause rates by location,
    deletion rates and in-text revision rates are compared with the real logs.
    (Composition stats were fitted on the same essays, so this checks that the
    planner reproduces them rather than generalisation.)

Usage:
    python -m algorithms.typing_model.evaluate --aalto data/aalto_keystrokes/Keystrokes.zip \\
        [--klicke data/klicke/linking-writing-processes-to-writing-quality.zip] [--participants 150]
"""

import argparse
import os

import numpy as np
from scipy.stats import ks_2samp

from .composition import LONG_PAUSE, CompositionStats, pause_class
from .dataset_klicke import INPUT, REMOVE
from .errors import ErrorStats
from .keys import BKSP
from .planner import TypingConfig, plan_keystrokes
from .runtime import DEFAULT_DIR, TypingModel
from .timing import BigramTimer

SENT_FEATURES = ["wpm", "bksp_rate", "rollover", "lag1", "iki_med", "iki_iqr", "iki_p90", "hold_med",
                 "hold_iqr", "space_ratio"]


# --- per-sentence features -----------------------------------------------------------------

def sentence_features(keys, press, hold, n_chars):
    press, hold = np.asarray(press, float), np.asarray(hold, float)
    iki = np.diff(press)
    if len(iki) < 5:
        return None
    li = np.log(np.clip(iki, 1e-3, None))
    rel = press + hold
    rollover = float(np.mean(press[1:] < rel[:-1]))
    after_space = np.array([k == " " for k in keys[:-1]])
    sp = np.median(iki[after_space]) / np.median(iki) if after_space.any() else 1.0
    lag1 = float(np.corrcoef(li[:-1], li[1:])[0, 1]) if len(li) > 3 and li.std() > 0 else 0.0
    dur = press[-1] - press[0]
    return {
        "wpm": n_chars / 5 / max(dur / 60, 1e-6),
        "bksp_rate": float(np.mean([k == BKSP for k in keys])),
        "rollover": rollover, "lag1": lag1,
        "iki_med": float(np.median(iki)), "iki_iqr": float(np.subtract(*np.percentile(iki, [75, 25]))),
        "iki_p90": float(np.percentile(iki, 90)),
        "hold_med": float(np.median(hold)), "hold_iqr": float(np.subtract(*np.percentile(hold, [75, 25]))),
        "space_ratio": float(sp),
    }


def timer_resolution(trials):
    """Timestamp granularity (s) of a real typist's browser: 0.001, 0.004, 0.008 or 0.016."""
    ms = np.concatenate([np.round(np.r_[tr.press, tr.release] * 1000).astype(int) for tr in trials])
    for q in (16, 8, 4):
        r = ms % q
        if np.mean((r <= 1) | (r >= q - 1)) > 0.95:
            return q / 1000.0
    return 0.001


def measured_wpm(trials):
    """Median first-to-last-key words per minute over a typist's sentences. (The
    dataset's AVG_WPM_15 is defined differently and runs ~10% lower.)"""
    r = []
    for tr in trials:
        idx = [i for i, k in enumerate(tr.keys) if k is not None]
        if len(idx) >= 8:
            dur = tr.press[idx[-1]] - tr.press[idx[0]]
            r.append(len(tr.user_input or tr.sentence) / 5 / max(dur / 60, 1e-6))
    return float(np.median(r)) if r else 45.0


def quantise(t, q):
    return np.round(np.asarray(t) / q) * q


def placeholder_plan(text, rng, wpm=45, sigma=0.35):
    """The old human_typing placeholder, for comparison."""
    from .planner import Keystroke
    base = 60.0 / (wpm * 5)
    out, t = [], 0.0
    for i, ch in enumerate(text):
        out.append(Keystroke(ch, "type", press=t, hold=0.0))    # pyautogui.write: ~0 hold
        d = base * np.exp(rng.normal(0, sigma))
        d *= 1.3 if ch == " " else (2.0 if ch in ".,;:!?" else 1.0)
        t += d
    return out


# --- motor evaluation ------------------------------------------------------------------------

def motor_eval(zip_path, n_participants, models_dir, seed=0):
    from .dataset_aalto import iter_participants

    rng = np.random.default_rng(seed)
    gens = {"placeholder": None}
    full = TypingModel.load(models_dir)
    gens["defaults"] = TypingModel(BigramTimer.default(), ErrorStats.default(), CompositionStats.default())
    if os.path.exists(os.path.join(models_dir, "typing_bigrams.json")):
        gens["bigram"] = TypingModel.load(models_dir, use_mdn=False)
    if full.sources.get("timing") == "typing_mdn.npz":
        gens["mdn"] = full

    real_iki, real_hold, feats = [], [], {"real": []}
    syn_iki = {g: [] for g in gens}
    syn_hold = {g: [] for g in gens}
    for g in gens:
        feats[g] = []

    for _, trials in iter_participants(zip_path, "val", n_participants, seed, verbose=False):
        q = timer_resolution(trials)          # record generated typing with the same clock
        wpm = measured_wpm(trials)            # the typist's actual speed, measured like generated text
        for tr in trials[:6]:
            idx = [i for i, k in enumerate(tr.keys) if k is not None]
            if len(idx) < 8 or not tr.sentence:
                continue
            keys = [tr.keys[i] for i in idx]
            press, hold = tr.press[idx], np.clip(tr.release[idx] - tr.press[idx], 0, 2)
            f = sentence_features(keys, press, hold, len(tr.user_input or tr.sentence))
            if f is None:
                continue
            feats["real"].append(f)
            real_iki.append(np.diff(press))
            real_hold.append(hold)
            for g, model in gens.items():
                if model is None:
                    plan = placeholder_plan(tr.sentence, rng)
                else:
                    plan = plan_keystrokes(tr.sentence, model, rng, TypingConfig(wpm=wpm, mode="transcribe"))
                p = quantise([k.press for k in plan], q)
                h = quantise([k.press + k.hold for k in plan], q) - p
                sf = sentence_features([k.key for k in plan], p, h, len(tr.sentence))
                if sf is not None:
                    feats[g].append(sf)
                    syn_iki[g].append(np.diff(p))
                    syn_hold[g].append(h)

    ri, rh = np.concatenate(real_iki), np.concatenate(real_hold)
    print(f"\nMotor level: {len(feats['real'])} held-out real sentences\n")
    cols = ["iki", "hold"] + SENT_FEATURES[:4] + ["AUC"]
    print(f"{'generator':12s}" + "".join(f"{c:>10s}" for c in cols))
    results = {}
    for g in gens:
        row = {"iki": ks_2samp(ri, np.concatenate(syn_iki[g])).statistic,
               "hold": ks_2samp(rh, np.concatenate(syn_hold[g])).statistic}
        for c in SENT_FEATURES[:4]:
            row[c] = ks_2samp([f[c] for f in feats["real"]], [f[c] for f in feats[g]]).statistic
        row["AUC"] = detector_auc(feats["real"], feats[g], seed)
        results[g] = row
        print(f"{g:12s}" + "".join(f"{row[c]:10.3f}" for c in cols))
    print("\n(KS columns: 0 = same distribution as real typing. AUC: 0.5 = a classifier cannot tell"
          " generated from real sentences.)")
    return results


def detector_auc(real, syn, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    n = min(len(real), len(syn))
    X = np.array([[f[c] for c in SENT_FEATURES] for f in real[:n] + syn[:n]], float)
    y = np.r_[np.ones(n), np.zeros(n)]
    X = np.nan_to_num(X)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    return float(np.mean(cross_val_score(clf, X, y, cv=5, scoring="roc_auc")))


# --- composition evaluation ------------------------------------------------------------------

def _comp_metrics_real(essay, max_chars):
    prod = np.flatnonzero((essay.kind == INPUT) | (essay.kind == REMOVE))
    iki = np.diff(essay.down[prod])
    stats = {}
    n_chars = 0
    for j in range(1, len(prod)):
        i, ip = prod[j], prod[j - 1]
        if not essay.at_edge[i]:
            continue
        ch = essay.char[i][:1] if essay.kind[i] == INPUT else "\b"
        c = pause_class(ch, essay.prev[i], essay.kind[ip], essay.kind[i])
        s = stats.setdefault(c, [0, 0])
        s[0] += 1
        s[1] += iki[j - 1] >= LONG_PAUSE
        if essay.kind[i] == INPUT:
            n_chars += 1
        if n_chars >= max_chars:
            break
    return stats


def _comp_metrics_syn(plan):
    stats = {}
    for k in plan[1:]:
        if k.pclass == "nav":
            continue
        s = stats.setdefault(k.pclass, [0, 0])
        s[0] += 1
        s[1] += k.iki >= LONG_PAUSE
    return stats


def composition_eval(klicke_path, n_essays, models_dir, max_chars=800, seed=0):
    from .dataset_klicke import iter_essays

    model = TypingModel.load(models_dir)
    rng = np.random.default_rng(seed)
    real, syn = {}, {}
    del_real = del_syn = chars = 0
    nav_syn = 0
    for e in iter_essays(klicke_path, n_essays, seed + 1):
        text = e.final_text[:max_chars]
        if len(text) < 200:
            continue
        for c, (n, l) in _comp_metrics_real(e, max_chars).items():
            real.setdefault(c, [0, 0])
            real[c][0] += n
            real[c][1] += l
        plan = plan_keystrokes(text, model, rng, TypingConfig(wpm=45, mode="compose", pause_scale=1.0,
                                                              max_pause=600))
        for c, (n, l) in _comp_metrics_syn(plan).items():
            syn.setdefault(c, [0, 0])
            syn[c][0] += n
            syn[c][1] += l
        chars += len(text)
        del_syn += sum(1 for i, k in enumerate(plan) if k.key == BKSP and (i == 0 or plan[i - 1].key != BKSP))
        nav_syn += sum(1 for i, k in enumerate(plan) if k.tag == "nav" and (i == 0 or plan[i - 1].tag != "nav"))
    d = model.comp.d
    real_runs = sum(d["delete_rate_per_100"].values()) / 100.0
    print(f"\nComposition level: {n_essays} essays (first {max_chars} chars each)\n")
    print(f"{'pause class':18s}{'real long%':>12s}{'synthetic':>12s}")
    for c in ["in_word", "word_end", "between_words", "after_clause", "after_sentence", "punct",
              "before_delete", "after_delete"]:
        if c in real and c in syn:
            print(f"{c:18s}{100 * real[c][1] / real[c][0]:11.1f}%{100 * syn[c][1] / max(syn[c][0], 1):11.1f}%")
    print(f"\ndeletion runs per 100 chars: real {100 * real_runs:.2f}, synthetic {100 * del_syn / chars:.2f}")
    print(f"in-text revisions per 1000 chars: real {d['intext_rate_per_1000']:.2f} (fixes <=30 chars), "
          f"synthetic {1000 * nav_syn / 2 / chars:.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aalto")
    ap.add_argument("--klicke")
    ap.add_argument("--models", default=DEFAULT_DIR)
    ap.add_argument("--participants", type=int, default=150)
    ap.add_argument("--essays", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.aalto:
        motor_eval(args.aalto, args.participants, args.models, args.seed)
    if args.klicke:
        composition_eval(args.klicke, args.essays, args.models, seed=args.seed)


if __name__ == "__main__":
    main()
