"""
LSTM + Mixture Density Network for keystroke timing (TensorFlow / Keras) -
TRAINING ONLY. At runtime the weights are run by the numpy sampler (sampler.py).

Per step:
    [emb(prev key), emb(key), emb(next key), floats] -> LSTM(units) x layers
    -> Dense(K*5): [mixture logits (K) | means (K*2) | log std-devs (K*2)]
i.e. a K-component diagonal Gaussian mixture over (log interval, log hold).
"""

import json
import math

import numpy as np
import tensorflow as tf
import keras

from .features import N_FLOAT, N_TOK
from .keys import VOCAB

LOG_SIGMA_MIN, LOG_SIGMA_MAX = -5.0, 3.0


def build_model(units=128, layers=2, mixtures=8, emb=16):
    tok = keras.Input(shape=(None, N_TOK), dtype="int32", name="tokens")
    fl = keras.Input(shape=(None, N_FLOAT), name="floats")
    e = keras.layers.Embedding(len(VOCAB), emb, name="emb")(tok)           # (B,T,3,emb)
    e = keras.layers.Reshape((-1, N_TOK * emb))(e)
    h = keras.layers.Concatenate()([e, fl])
    for i in range(layers):
        h = keras.layers.LSTM(units, return_sequences=True, name=f"lstm{i}")(h)
    out = keras.layers.Dense(mixtures * 5, name="mdn")(h)
    return keras.Model([tok, fl], out, name="typing_mdn")


def split_params(out, K):
    logits = out[..., :K]
    shape = tf.concat([tf.shape(out)[:-1], [K, 2]], 0)
    mu = tf.reshape(out[..., K:3 * K], shape)
    log_sigma = tf.clip_by_value(tf.reshape(out[..., 3 * K:5 * K], shape), LOG_SIGMA_MIN, LOG_SIGMA_MAX)
    return logits, mu, log_sigma


def make_loss(K):
    """Masked NLL. y_true layout: [log iki, log hold, mask]."""
    log_2pi = math.log(2 * math.pi)

    def mdn_loss(y_true, y_pred):
        target, mask = y_true[..., :2], y_true[..., 2]
        logits, mu, log_sigma = split_params(y_pred, K)
        z = (target[..., None, :] - mu) * tf.exp(-log_sigma)
        comp_ll = -0.5 * tf.reduce_sum(z * z, -1) - tf.reduce_sum(log_sigma, -1) - log_2pi
        ll = tf.reduce_logsumexp(tf.nn.log_softmax(logits) + comp_ll, axis=-1)
        return tf.reduce_sum(-ll * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)

    return mdn_loss


def export_npz(model, stats, path, units, layers, mixtures, emb):
    arrays = {"emb": model.get_layer("emb").get_weights()[0]}
    for i in range(layers):
        W, U, b = model.get_layer(f"lstm{i}").get_weights()
        arrays[f"lstm{i}_W"], arrays[f"lstm{i}_U"], arrays[f"lstm{i}_b"] = W, U, b
    Wd, bd = model.get_layer("mdn").get_weights()
    arrays["mdn_W"], arrays["mdn_b"] = Wd, bd
    arrays.update(stats.to_dict())
    config = {"units": units, "layers": layers, "mixtures": mixtures, "emb": emb,
              "vocab": len(VOCAB), "n_float": N_FLOAT, "version": 1}
    arrays["config"] = np.array(json.dumps(config))
    np.savez_compressed(path, **arrays)
