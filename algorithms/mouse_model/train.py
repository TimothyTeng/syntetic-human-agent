"""
Train the LSTM-MDN mouse model on a public dataset and export it for runtime.

Usage (from the project root):
    python -m algorithms.mouse_model.train --dataset bmdd --data data/bmdd --out models
    python -m algorithms.mouse_model.train --dataset balabit --data data/balabit --out models

Quick smoke run:
    python -m algorithms.mouse_model.train --data data/bmdd --max-strokes 5000 --epochs 2

Outputs (in --out):
    mouse_mdn.npz      weights + normalisation stats (loaded by sampler.MouseModel)
    click_stats.json   empirical click hold / dwell times
    training_info.json settings and final losses
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import keras
import tensorflow as tf

from .click_stats import ClickStats
from .dataset import iter_sessions
from .features import N_IN, NormStats, encode_stroke
from .model import build_model, export_npz, make_loss
from .strokes import segment_strokes


# --- Data ------------------------------------------------------------------------

def collect_strokes(dataset, data_dir, max_strokes, max_files=None, split=None,
                    window_filter=None, seed=0):
    """Read sessions until `max_strokes` point-and-click strokes are collected."""
    strokes, n_files, t0 = [], 0, time.time()
    for sess in iter_sessions(dataset, data_dir, split=split, max_files=max_files, seed=seed):
        n_files += 1
        strokes.extend(segment_strokes(sess, window_filter=window_filter))
        if n_files % 20 == 0:
            print(f"  read {n_files} files, {len(strokes)} strokes ({time.time() - t0:.0f}s)")
        if len(strokes) >= max_strokes:
            break
    print(f"Collected {len(strokes)} strokes from {n_files} files")
    return strokes[:max_strokes]


def split_by_user(strokes, val_frac, rng):
    """Hold out whole users for validation (tests generalisation to new people)."""
    users = sorted({s.user for s in strokes})
    if len(users) < 3:
        idx = rng.permutation(len(strokes))
        n_val = max(1, int(len(strokes) * val_frac))
        val_idx = set(idx[:n_val].tolist())
        return ([s for i, s in enumerate(strokes) if i not in val_idx],
                [s for i, s in enumerate(strokes) if i in val_idx])
    rng.shuffle(users)
    counts = {u: 0 for u in users}
    for s in strokes:
        counts[s.user] += 1
    val_users, n_val = set(), 0
    for u in users:
        if n_val >= val_frac * len(strokes):
            break
        val_users.add(u)
        n_val += counts[u]
    print(f"Validation users: {sorted(val_users)}")
    return ([s for s in strokes if s.user not in val_users],
            [s for s in strokes if s.user in val_users])


class StrokeBatches:
    """
    Batches of similar-length strokes, zero-padded to a multiple of 16 steps.
    y carries a 5th column 'mask' (1 = real step, 0 = padding) used by the loss.
    Iterating yields one epoch; batch order is reshuffled every epoch.
    """

    def __init__(self, encoded, batch_size, shuffle=True, seed=0):
        self.encoded = encoded
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.lengths = np.array([len(x) for x, _ in encoded])
        self._make_batches()

    def _make_batches(self):
        noise = self.rng.uniform(0, 8, len(self.lengths)) if self.shuffle else 0
        order = np.argsort(self.lengths + noise)
        self.batches = [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        if self.shuffle:
            self.rng.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        idx = self.batches[i]
        T = int(np.ceil(self.lengths[idx].max() / 16) * 16)
        X = np.zeros((len(idx), T, N_IN), np.float32)
        Y = np.zeros((len(idx), T, 5), np.float32)
        for b, j in enumerate(idx):
            x, y = self.encoded[j]
            X[b, :len(x)] = x
            Y[b, :len(y), :4] = y
            Y[b, :len(y), 4] = 1.0
        return X, Y

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
        if self.shuffle:
            self._make_batches()

    def to_tf_dataset(self):
        """Wrap as tf.data with a variable-length time axis (batches differ in length)."""
        spec = (tf.TensorSpec((None, None, N_IN), tf.float32), tf.TensorSpec((None, None, 5), tf.float32))
        return tf.data.Dataset.from_generator(lambda: iter(self), output_signature=spec).prefetch(2)


# --- Checks ------------------------------------------------------------------------

def parity_check(model, npz_path, encoded, n=3):
    """Confirm the numpy runtime reproduces the Keras model's outputs."""
    from .sampler import MouseModel
    mm = MouseModel.load(npz_path)
    worst = 0.0
    for x, _ in encoded[:n]:
        tf_out = model.predict(x[None], verbose=0)[0]
        np_out = mm.forward_sequence(x)
        worst = max(worst, float(np.abs(tf_out - np_out).max()))
    status = "OK" if worst < 1e-3 else "MISMATCH"
    print(f"Numpy/TensorFlow parity: max abs diff {worst:.2e} -> {status}")
    return worst


# --- Main ------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="auto", choices=["auto", "bmdd", "balabit"])
    ap.add_argument("--data", required=True, help="folder where the dataset was extracted")
    ap.add_argument("--out", default="models")
    ap.add_argument("--split", default=None, help="path filter, default 'train'/'training'; '' = all files")
    ap.add_argument("--max-strokes", type=int, default=60000)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--window-filter", default=None, help="BMDD: only clicks in windows containing this text")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--units", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--mixtures", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    # Keras progress bars use Unicode; avoid crashes on cp1252 Windows consoles
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    keras.utils.set_random_seed(args.seed)

    # 1. Strokes + click timing
    strokes = collect_strokes(args.dataset, args.data, args.max_strokes, args.max_files,
                              args.split, args.window_filter, args.seed)
    if len(strokes) < 50:
        raise SystemExit("Too few strokes found - check --data / --dataset / --split")
    clicks = ClickStats.fit(strokes, rng)
    clicks.save(os.path.join(args.out, "click_stats.json"))
    print("Click timing:", clicks.summary())

    # 2. Split, normalise, encode
    train, val = split_by_user(strokes, args.val_frac, rng)
    stats = NormStats.fit(train)
    enc_train = [encode_stroke(s, stats) for s in train]
    enc_val = [encode_stroke(s, stats) for s in val]
    print(f"Train {len(enc_train)} strokes / val {len(enc_val)} strokes, "
          f"median length {int(np.median([len(x) for x, _ in enc_train]))} steps")

    # 3. Train
    end_prior = len(enc_train) / sum(len(x) for x, _ in enc_train)  # 1 end flag per stroke
    model = build_model(args.units, args.layers, args.mixtures, end_prior=end_prior)
    model.compile(optimizer=keras.optimizers.Adam(args.lr, clipnorm=1.0), loss=make_loss(args.mixtures))
    model.summary()
    history = model.fit(
        StrokeBatches(enc_train, args.batch, shuffle=True, seed=args.seed).to_tf_dataset(),
        validation_data=StrokeBatches(enc_val, args.batch, shuffle=False).to_tf_dataset(),
        epochs=args.epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
        ],
    )

    # 4. Export + verify
    npz_path = os.path.join(args.out, "mouse_mdn.npz")
    export_npz(model, stats, npz_path, args.units, args.layers, args.mixtures)
    print("Saved", npz_path)
    diff = parity_check(model, npz_path, enc_val or enc_train)

    info = {"args": vars(args), "n_train": len(enc_train), "n_val": len(enc_val),
            "final_loss": float(history.history["loss"][-1]),
            "best_val_loss": float(min(history.history["val_loss"])),
            "parity_max_abs_diff": diff}
    with open(os.path.join(args.out, "training_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
