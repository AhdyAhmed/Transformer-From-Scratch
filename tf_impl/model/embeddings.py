"""Token embeddings and sinusoidal positional encoding (TensorFlow).

Together these turn a batch of token ids into a matrix of shape
``(batch, seq_len, d_model)`` ready for the encoder/decoder stacks (Day 8+).
Mirrors ``torch_impl/model/embeddings.py``.

Known, intentional framework difference: PyTorch's ``nn.Embedding`` accepts a
``padding_idx`` that zeros the pad row's gradient automatically. Keras'
``Embedding`` layer has no equivalent, and replicating it would mean writing
a custom gradient — out of scope for a from-scratch *architecture* exercise.
``pad_idx`` is still accepted here (and documented) so the two constructors
match, but on the TF side it currently has no training-time effect.
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf


class TokenEmbedding(tf.keras.layers.Layer):
    """Learned token embedding, scaled by sqrt(d_model) as in the paper."""

    def __init__(self, vocab_size: int, d_model: int, pad_idx: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """x: (batch, seq_len) int token ids -> (batch, seq_len, d_model)."""
        scale = tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        return self.embedding(x) * scale

    def get_config(self):
        cfg = super().get_config()
        cfg.update(vocab_size=self.vocab_size, d_model=self.d_model, pad_idx=self.pad_idx)
        return cfg


class PositionalEncoding(tf.keras.layers.Layer):
    """Fixed sinusoidal positional encoding (Section 3.5 of the paper).

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Precomputed once up to ``max_len`` and stored as a plain ``tf.constant``
    (not a weight) — it's a fixed function of position, never learned or
    checkpointed.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = tf.keras.layers.Dropout(dropout)

        position = np.arange(max_len)[:, np.newaxis].astype(np.float32)
        div_term = np.exp(
            np.arange(0, d_model, 2).astype(np.float32) * (-math.log(10000.0) / d_model)
        )
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe[np.newaxis, ...])  # (1, max_len, d_model)

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        """x: (batch, seq_len, d_model) token embeddings -> same shape, + PE."""
        static_len = x.shape[1]
        if static_len is not None and static_len > self.max_len:
            raise ValueError(
                f"sequence length {static_len} exceeds max_len {self.max_len} "
                "this PositionalEncoding was built for"
            )
        seq_len = tf.shape(x)[1]
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x, training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(d_model=self.d_model, max_len=self.max_len)
        return cfg


class TransformerEmbedding(tf.keras.layers.Layer):
    """Convenience wrapper: TokenEmbedding followed by PositionalEncoding.

    Used by the full model assembly in Day 12; kept as a thin composition so
    each piece stays independently testable.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        pad_idx: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.token_embedding = TokenEmbedding(vocab_size, d_model, pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        return self.positional_encoding(self.token_embedding(x), training=training)
