"""Label-smoothed token loss with padding ignored (Day 16)."""
from __future__ import annotations

import tensorflow as tf


class LabelSmoothingLoss:
    """Framework-native callable returning loss over non-padding tokens.

    ``logits`` shape is ``(..., vocab_size)`` and ``targets`` shape is ``(...)``.
    With ``reduction='mean'`` the result is normalized by the number of targets
    not equal to ``pad_idx``, rather than by padded sequence length.
    """
    def __init__(self, label_smoothing: float = 0.1, pad_idx: int = 0,
                 reduction: str = "mean"):
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if reduction not in {"none", "sum", "mean"}:
            raise ValueError("reduction must be 'none', 'sum', or 'mean'")
        self.label_smoothing = float(label_smoothing)
        self.pad_idx = int(pad_idx)
        self.reduction = reduction

    def __call__(self, targets, logits):
        logits = tf.convert_to_tensor(logits)
        targets = tf.cast(targets, tf.int32)
        tf.debugging.assert_equal(tf.shape(logits)[:-1], tf.shape(targets),
                                  message="targets must match logits leading dimensions")
        vocab_size = tf.shape(logits)[-1]
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        flat_log_probs = tf.reshape(log_probs, [-1, vocab_size])
        flat_targets = tf.reshape(targets, [-1])
        valid = tf.not_equal(flat_targets, self.pad_idx)
        safe_targets = tf.where(valid, flat_targets, tf.zeros_like(flat_targets))
        indices = tf.stack([tf.range(tf.shape(safe_targets)[0]), safe_targets], axis=1)
        nll = -tf.gather_nd(flat_log_probs, indices)
        smooth = -tf.reduce_mean(flat_log_probs, axis=-1)
        per_token = (1.0 - self.label_smoothing) * nll + self.label_smoothing * smooth
        per_token *= tf.cast(valid, per_token.dtype)
        per_token = tf.reshape(per_token, tf.shape(targets))
        if self.reduction == "none":
            return per_token
        total = tf.reduce_sum(per_token)
        if self.reduction == "sum":
            return total
        count = tf.reduce_sum(tf.cast(valid, total.dtype))
        return total / tf.maximum(count, 1.0)
