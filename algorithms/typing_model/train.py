"""
Train the LSTM-MDN keystroke-timing model on the Aalto 136M keystrokes dataset
and export it for the numpy runtime.

Usage (from the project root):
    python -m algorithms.typing_model.train --data data/aalto_keystrokes/Keystrokes.zip

Quick smoke run:
    python -m algorithms.typing_model.train --data data/aalto_keystrokes/Keystrokes.zip \\
        --participants 300 --epochs 2

Outputs (in --out):
    typing_mdn.npz              weights + normalisation (loaded by sampler.TypingMDN)
    typing_training_info.json   settings and losses
Validation uses the dataset's held-out participants (dataset_aalto.is_validation).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import keras
import tensorflow as tf

from .dataset_aalto import iter_participants
from .features import N_FLOAT, N_TOK, NormStats, encode_trial, participant_median, trial_arrays
from .model import build_model, export_npz, make_loss


def collect(zip_path, split, n_participants, seed):
    rows = []      # (wpm, median_iki, [(keys, iki, hold), ...])
    for wpm, trials in iter_participants(zip_path, split, n_participants, seed):
        med = participant_median(trials)
        if med is None:
            continue
        seqs = [a for a in (trial_arrays(tr) for tr in trials) if a is not None]
        rows.append((wpm, med, seqs))
    return rows


def fit_stats(rows):
    z = np.concatenate([np.column_stack([np.log(i[1:]), np.log(h[1:])]) for _, _, s in rows for _, i, h in s])
    speeds = np.log([m for _, m, _ in rows])
    wpm = np.log([w for w, _, _ in rows])
    b, a = np.polyfit(wpm, speeds, 1)
    return NormStats(z.mean(0), z.std(0) + 1e-6, speeds.mean(), speeds.std() + 1e-6, [a, b])


def encode(rows, stats):
    out = []
    for _, med, seqs in rows:
        speed = float(stats.speed_from_median(med))
        for keys, iki, hold in seqs:
            out.append(encode_trial(keys, iki, hold, speed, stats))
    return out


class Batches:
    def __init__(self, enc, batch, shuffle=True, seed=0):
        self.enc, self.batch, self.shuffle = enc, batch, shuffle
        self.rng = np.random.default_rng(seed)
        self.lengths = np.array([len(e[0]) for e in enc])
        self._make()

    def _make(self):
        noise = self.rng.uniform(0, 6, len(self.lengths)) if self.shuffle else 0
        order = np.argsort(self.lengths + noise)
        self.batches = [order[i:i + self.batch] for i in range(0, len(order), self.batch)]
        if self.shuffle:
            self.rng.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for idx in self.batches:
            T = int(self.lengths[idx].max())
            tok = np.zeros((len(idx), T, N_TOK), np.int32)
            fl = np.zeros((len(idx), T, N_FLOAT), np.float32)
            y = np.zeros((len(idx), T, 3), np.float32)
            for b, j in enumerate(idx):
                t, f, z = self.enc[j]
                tok[b, :len(t)], fl[b, :len(t)] = t, f
                y[b, :len(t), :2], y[b, :len(t), 2] = z, 1.0
            yield (tok, fl), y
        if self.shuffle:
            self._make()

    def dataset(self):
        spec = ((tf.TensorSpec((None, None, N_TOK), tf.int32), tf.TensorSpec((None, None, N_FLOAT), tf.float32)),
                tf.TensorSpec((None, None, 3), tf.float32))
        return tf.data.Dataset.from_generator(lambda: iter(self), output_signature=spec).prefetch(2)


def parity_check(model, npz_path, enc, n=3):
    from .sampler import TypingMDN
    mm = TypingMDN.load(npz_path)
    worst = 0.0
    for t, f, _ in enc[:n]:
        tf_out = model.predict([t[None], f[None]], verbose=0)[0]
        worst = max(worst, float(np.abs(tf_out - mm.forward_sequence(t, f)).max()))
    print(f"Numpy/TensorFlow parity: max abs diff {worst:.2e} -> {'OK' if worst < 1e-3 else 'MISMATCH'}")
    return worst


CALIBRATION_TEXTS = [
    "Was wondering if you and Natalie connected?", "If Motley is in we are out of money.",
    "Please send me the latest version of the contract.", "I will be out of the office until Monday.",
    "The meeting has been moved to three o'clock tomorrow.", "Thanks for getting back to me so quickly.",
]


def calibrate_speed(npz_path, n=240, seed=0):
    """
    Generated text should come out at the requested words-per-minute. Measure the
    achieved/requested ratio over a range of personas (transcribe mode, typos
    included) and store it in the .npz as `wpm_scale`.
    """
    from .planner import TypingConfig, plan_keystrokes
    from .runtime import TypingModel

    rng = np.random.default_rng(seed)
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays.pop("wpm_scale", None)
    np.savez_compressed(npz_path, **arrays)
    model = TypingModel.load(os.path.dirname(npz_path) or ".")
    ratios = []
    for i in range(n):
        wpm = float(rng.uniform(20, 110))
        text = CALIBRATION_TEXTS[i % len(CALIBRATION_TEXTS)]
        plan = plan_keystrokes(text, model, rng, TypingConfig(wpm=wpm, mode="transcribe"))
        ratios.append(len(text) / 5 / (plan[-1].press / 60) / wpm)
    scale = float(np.median(ratios))
    arrays["wpm_scale"] = np.float32(scale)
    np.savez_compressed(npz_path, **arrays)
    print(f"Speed calibration: generated text ran at x{scale:.3f} the requested wpm -> stored wpm_scale")
    return scale


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Aalto Keystrokes.zip")
    ap.add_argument("--out", default="models")
    ap.add_argument("--participants", type=int, default=4000)
    ap.add_argument("--val-participants", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--units", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--mixtures", type=int, default=8)
    ap.add_argument("--emb", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    os.makedirs(args.out, exist_ok=True)
    keras.utils.set_random_seed(args.seed)

    t0 = time.time()
    train_rows = collect(args.data, "train", args.participants, args.seed)
    val_rows = collect(args.data, "val", args.val_participants, args.seed)
    stats = fit_stats(train_rows)
    enc_train, enc_val = encode(train_rows, stats), encode(val_rows, stats)
    print(f"Train {len(enc_train)} sentences ({len(train_rows)} typists), val {len(enc_val)} "
          f"({len(val_rows)} typists), {sum(len(e[0]) for e in enc_train)} train steps "
          f"[{time.time() - t0:.0f}s]")

    model = build_model(args.units, args.layers, args.mixtures, args.emb)
    model.compile(optimizer=keras.optimizers.Adam(args.lr, clipnorm=1.0), loss=make_loss(args.mixtures))
    model.summary()
    hist = model.fit(
        Batches(enc_train, args.batch, True, args.seed).dataset(),
        validation_data=Batches(enc_val, args.batch, False).dataset(),
        epochs=args.epochs,
        verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                   keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1)],
    )
    npz = os.path.join(args.out, "typing_mdn.npz")
    export_npz(model, stats, npz, args.units, args.layers, args.mixtures, args.emb)
    print("Saved", npz)
    diff = parity_check(model, npz, enc_val or enc_train)
    wpm_scale = calibrate_speed(npz, seed=args.seed)
    info = {"args": vars(args), "n_train": len(enc_train), "n_val": len(enc_val),
            "final_loss": float(hist.history["loss"][-1]),
            "best_val_loss": float(min(hist.history["val_loss"])),
            "parity_max_abs_diff": diff, "wpm_scale": wpm_scale}
    with open(os.path.join(args.out, "typing_training_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
