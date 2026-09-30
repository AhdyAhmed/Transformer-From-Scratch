"""Scaled dot-product attention (TensorFlow) — Section 3.2.1 of the paper.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

This is the core primitive multi-head attention (Day 5) builds on top of. It
is deliberately kept as a plain function, not a layer: it has no learned
parameters of its own, only the projections around it do.

Mirrors ``torch_impl/model/attention.py``.
"""

from __future__ import annotations

import tensorflow as tf

from tf_impl.model.masking import NEG_INF


def scaled_dot_product_attention(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    mask: tf.Tensor | None = None,
    dropout: tf.keras.layers.Dropout | None = None,
    training: bool | None = None,
):
    """
    Args:
        query: (..., q_len, d_k)
        key:   (..., k_len, d_k)
        value: (..., k_len, d_v)
        mask:  optional bool tensor broadcastable to (..., q_len, k_len).
               True = may attend, False = blocked (see model/masking.py).
        dropout: optional Dropout layer applied to the attention weights
            (applied to weights, not to the output — matches the paper).
        training: passed through to ``dropout``; required whenever
            ``dropout`` is given.

    Returns:
        output: (..., q_len, d_v)
        attn_weights: (..., q_len, k_len) — post-softmax, pre-dropout, so
            they remain valid probability distributions for visualization.
    """
    d_k = tf.cast(tf.shape(query)[-1], tf.float32)
    scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(d_k)

    if mask is not None:
        scores = tf.where(mask, scores, tf.cast(NEG_INF, scores.dtype))

    attn_weights = tf.nn.softmax(scores, axis=-1)

    weights_for_output = attn_weights
    if dropout is not None:
        weights_for_output = dropout(attn_weights, training=training)

    output = tf.matmul(weights_for_output, value)
    return output, attn_weights
