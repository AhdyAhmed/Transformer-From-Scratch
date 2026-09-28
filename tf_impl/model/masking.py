"""Attention mask utilities (TensorFlow).

Convention (shared with ``torch_impl/model/masking.py``):

    True  -> this key position MAY be attended to
    False -> this key position is BLOCKED

All masks are ``tf.bool`` and broadcast against attention scores of shape
``(batch, num_heads, q_len, k_len)``. In the attention layer (Day 4) they are
applied as ``tf.where(mask, scores, NEG_INF)`` before the softmax.
"""

from __future__ import annotations

import tensorflow as tf

# Large negative value used to zero-out blocked positions after softmax.
# -1e9 (not -inf) keeps fully-masked rows finite and avoids NaNs.
NEG_INF = -1e9


def create_padding_mask(seq: tf.Tensor, pad_idx: int = 0) -> tf.Tensor:
    """Mask that blocks attention *to* padding tokens.

    Args:
        seq: token ids, shape ``(batch, seq_len)``.
        pad_idx: id of the padding token.

    Returns:
        bool tensor of shape ``(batch, 1, 1, seq_len)``.
    """
    if seq.shape.rank != 2:
        raise ValueError(f"seq must be 2-D (batch, seq_len), got shape {seq.shape}")
    mask = tf.not_equal(seq, pad_idx)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size) -> tf.Tensor:
    """Causal mask: position *i* may only attend to positions ``<= i``.

    ``size`` may be a Python int or a scalar tensor (so it works inside
    ``@tf.function`` with dynamic sequence lengths).

    Returns:
        bool tensor of shape ``(1, 1, size, size)`` (lower-triangular).
    """
    ones = tf.ones((size, size), dtype=tf.int32)
    # keep lower triangle incl. diagonal: num_lower=-1 (all), num_upper=0
    mask = tf.linalg.band_part(ones, -1, 0)
    return tf.cast(mask, tf.bool)[tf.newaxis, tf.newaxis, :, :]


def create_target_mask(tgt: tf.Tensor, pad_idx: int = 0) -> tf.Tensor:
    """Decoder self-attention mask = padding mask AND look-ahead mask.

    Returns:
        bool tensor of shape ``(batch, 1, tgt_len, tgt_len)``.
    """
    tgt_len = tf.shape(tgt)[1]
    pad_mask = create_padding_mask(tgt, pad_idx)      # (B, 1, 1, T)
    causal = create_look_ahead_mask(tgt_len)          # (1, 1, T, T)
    return tf.logical_and(pad_mask, causal)           # (B, 1, T, T)


def create_masks(src: tf.Tensor, tgt: tf.Tensor, pad_idx: int = 0):
    """Build every mask the encoder-decoder Transformer needs.

    Returns:
        src_mask:   ``(B, 1, 1, S)`` encoder self-attention (blocks source padding).
        tgt_mask:   ``(B, 1, T, T)`` decoder self-attention (padding + causal).
        memory_mask:``(B, 1, 1, S)`` decoder cross-attention (blocks source padding).
    """
    src_mask = create_padding_mask(src, pad_idx)
    tgt_mask = create_target_mask(tgt, pad_idx)
    memory_mask = src_mask
    return src_mask, tgt_mask, memory_mask
