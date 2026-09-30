"""
Fit the statistical typing model (no neural network) from the public datasets.

Usage (from the project root):
    python -m algorithms.typing_model.fit_stats \\
        --aalto data/aalto_keystrokes/Keystrokes.zip \\
        --klicke data/klicke/linking-writing-processes-to-writing-quality.zip

Either dataset can be omitted (only its files are rewritten).

Outputs (in --out, default models/):
    typing_bigrams.json      motor timing: per-bigram interval + per-key hold quantiles,
                             speed regression, tempo drift, Shift lead/lag      (Aalto)
    typing_errors.json       typo rates by speed, typo types, detection delay,
                             over-deletion                                      (Aalto)
    typing_composition.json  pause distributions by location, deletion / revision
                             rates, lost-thought behaviour                      (KLiCKe)
"""

import argparse
import json
import os
import time

import numpy as np

from .composition import CompositionStats
from .errors import ErrorStats
from .timing import BigramTimer


def calibrate_error_rate(parts, timer, errors, rng, rounds=2):
    """
    Scale the typo rate so that generated transcription typing uses as many
    Backspaces per keystroke as the real typists did. (The generator always
    corrects its typos and its typo lengths / detection delays are sampled
    independently, so the raw onset rate over-produces Backspaces.)
    """
    from .composition import CompositionStats
    from .keys import BKSP
    from .planner import TypingConfig, plan_keystrokes
    from .runtime import TypingModel

    real_k = sum(sum(k is not None for k in tr.keys) for _, t in parts for tr in t)
    real_b = sum(sum(k == BKSP for k in tr.keys) for _, t in parts for tr in t)
    target = real_b / max(real_k, 1)
    total = 1.0
    for _ in range(rounds):
        model = TypingModel(timer, errors, CompositionStats.default())
        k = b = 0
        for wpm, trials in parts:
            for tr in trials[:5]:
                plan = plan_keystrokes(tr.sentence, model, rng, TypingConfig(wpm=wpm, mode="transcribe"))
                k += len(plan)
                b += sum(ks.key == BKSP for ks in plan)
        factor = float(np.clip(target / max(b / max(k, 1), 1e-6), 0.3, 3.0))
        errors.scale_rate(factor)
        total *= factor
        print(f"  calibration: real {target:.4f} vs generated {b / max(k, 1):.4f} Backspaces/key -> x{factor:.2f}")
    errors.d["rate_calibration"] = round(total, 4)
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aalto", help="Keystrokes.zip from the Aalto 136M keystrokes dataset")
    ap.add_argument("--klicke", help="Kaggle competition .zip or train_logs.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--participants", type=int, default=6000, help="Aalto participants to use")
    ap.add_argument("--essays", type=int, default=None, help="KLiCKe essays to use (default all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not args.aalto and not args.klicke:
        ap.error("give --aalto and/or --klicke")
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    info = {}

    if args.aalto:
        from .dataset_aalto import iter_participants
        t0 = time.time()
        parts = list(iter_participants(args.aalto, "train", args.participants, args.seed))
        n_keys = sum(len(tr.keys) for _, trials in parts for tr in trials)
        print(f"Aalto: {len(parts)} participants, {n_keys} keystrokes ({time.time() - t0:.0f}s)")

        timer = BigramTimer.fit(parts, rng)
        timer.save(os.path.join(args.out, "typing_bigrams.json"))
        print(f"  timing: {len(timer.iki_bigram)} bigrams, rho={timer.d['rho']:.2f}, "
              f"median IKI at 40/80 wpm = {timer.median_iki(40) * 1000:.0f}/{timer.median_iki(80) * 1000:.0f} ms")

        errors = ErrorStats.fit(parts, rng)
        factor = calibrate_error_rate(parts[:400], timer, errors, rng)
        errors.save(os.path.join(args.out, "typing_errors.json"))
        print(f"  errors (rate x{factor:.2f} after calibration):", errors.summary())
        info["aalto"] = {"participants": len(parts), "keystrokes": n_keys}

    if args.klicke:
        from .dataset_klicke import iter_essays
        t0 = time.time()
        comp = CompositionStats.fit(iter_essays(args.klicke, args.essays, args.seed), rng)
        comp.save(os.path.join(args.out, "typing_composition.json"))
        print(f"KLiCKe ({time.time() - t0:.0f}s):", comp.summary())
        info["klicke"] = {"essays": comp.d["n_essays"], "edge_chars": comp.d["edge_chars"]}

    path = os.path.join(args.out, "typing_fit_info.json")
    prev = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            prev = json.load(f)
    prev.update(info)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prev, f, indent=2)


if __name__ == "__main__":
    main()
