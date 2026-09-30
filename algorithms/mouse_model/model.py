"""
LSTM + Mixture Density Network (TensorFlow / Keras) - TRAINING ONLY.

Architecture (per time step):
    input (7) -> LSTM(units) x layers -> Dense(K*7 + 1)

The Dense output is split into:
    [ mixture logits (K) | means (K*3) | log std-devs (K*3) | end logit (1) ]
i.e. a K-component diagonal Gaussian mixture over the next (dx, dy, log dt)
step, plus a Bernoulli "this is the last step" probability.

At runtime this file is NOT imported - weights are exported to a .npz and run
by the numpy sampler (sampler.py), so the agent never loads TensorFlow.
"""

import json
import math

import numpy as np
import tensorflow as tf
import keras

from .features import N_IN

LOG_SIGMA_MIN, LOG_SIGMA_MAX = -7.0, 5.0


def build_model(units=128, layers=2, mixtures=10, end_prior=0.02):
    """
    Create the Keras model. Output shape: (batch, time, mixtures*7 + 1).

    end_prior: fraction of steps that are the last step of a stroke (~1/length).
               The end-logit bias starts at this base rate; otherwise it starts
               at 50% and freshly trained models stop after one or two steps.
    """
    inp = keras.Input(shape=(None, N_IN), name="steps")
    h = inp
    for i in range(layers):
        h = keras.layers.LSTM(units, return_sequences=True, name=f"lstm{i}")(h)
    dense = keras.layers.Dense(mixtures * 7 + 1, name="mdn")
    out = dense(h)
    model = keras.Model(inp, out, name="mouse_mdn")

    W, b = dense.get_weights()
    b[7 * mixtures] = math.log(end_prior / (1 - end_prior))
    dense.set_weights([W, b])
    return model


def split_params(out, K):
    """Split raw network output into (logits, mu, log_sigma, end_logit) tensors."""
    logits = out[..., :K]
    mu = tf.reshape(out[..., K:4 * K], tf.concat([tf.shape(out)[:-1], [K, 3]], 0))
    log_sigma = tf.reshape(out[..., 4 * K:7 * K], tf.concat([tf.shape(out)[:-1], [K, 3]], 0))
    end_logit = out[..., 7 * K]
    return logits, mu, tf.clip_by_value(log_sigma, LOG_SIGMA_MIN, LOG_SIGMA_MAX), end_logit


def make_loss(K):
    """
    Masked negative log-likelihood.
    y_true layout: [dx, dy, logdt, end, mask]  (mask = 0 on padding steps)
    """
    log_2pi = math.log(2 * math.pi)

    def mdn_loss(y_true, y_pred):
        target, end, mask = y_true[..., :3], y_true[..., 3], y_true[..., 4]
        logits, mu, log_sigma, end_logit = split_params(y_pred, K)

        z = (target[..., None, :] - mu) * tf.exp(-log_sigma)                 # (B,T,K,3)
        comp_ll = -0.5 * tf.reduce_sum(z * z, -1) - tf.reduce_sum(log_sigma, -1) - 1.5 * log_2pi
        ll = tf.reduce_logsumexp(tf.nn.log_softmax(logits) + comp_ll, axis=-1)  # (B,T)
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=end, logits=end_logit)

        per_step = (-ll + bce) * mask
        return tf.reduce_sum(per_step) / tf.maximum(tf.reduce_sum(mask), 1.0)

    return mdn_loss


def export_npz(model, stats, path, units, layers, mixtures):
    """
    Save weights + normalisation stats to a .npz readable by sampler.MouseModel.
    Keras LSTM weights: kernel (in, 4u), recurrent (u, 4u), bias (4u);
    gate order i, f, c, o.
    """
    arrays = {}
    for i in range(layers):
        W, U, b = model.get_layer(f"lstm{i}").get_weights()
        arrays[f"lstm{i}_W"], arrays[f"lstm{i}_U"], arrays[f"lstm{i}_b"] = W, U, b
    Wd, bd = model.get_layer("mdn").get_weights()
    arrays["mdn_W"], arrays["mdn_b"] = Wd, bd
    arrays.update(stats.to_dict())
    config = {"units": units, "layers": layers, "mixtures": mixtures, "n_in": N_IN, "version": 1}
    arrays["config"] = np.array(json.dumps(config))
    np.savez_compressed(path, **arrays)
