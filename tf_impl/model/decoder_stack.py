"""Decoder stack (TensorFlow) — Section 3.1 of the paper: "N = 6" identical layers.

    x = embeddings(tgt)
    for layer in layers:
        x = layer(x, memory, tgt_mask, memory_mask)
    x = final_norm(x)
    return x

Mirrors ``Encoder`` (Day 10) exactly, with the extra ``memory``/``memory_mask``
every ``DecoderLayer`` needs for cross-attention. Mirrors
``torch_impl/model/decoder_stack.py``.
"""

from __future__ import annotations

import tensorflow as tf

from tf_impl.model.decoder import DecoderLayer


class Decoder(tf.keras.layers.Layer):
    """A stack of ``num_layers`` independent ``DecoderLayer``s + final LayerNorm.

    Each layer gets its own freshly-initialized weights — a fresh
    ``DecoderLayer(...)`` per list position, for the same reason as the
    encoder stack (Day 10): sharing one layer's weights across all
    positions would collapse the stack's depth.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, d_ff, dropout, norm_first) for _ in range(num_layers)
        ]
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(
        self,
        x: tf.Tensor,
        memory: tf.Tensor,
        tgt_mask: tf.Tensor | None = None,
        memory_mask: tf.Tensor | None = None,
        training: bool | None = None,
    ) -> tf.Tensor:
        """
        Args:
            x: (batch, tgt_len, d_model) — already embedded + positionally encoded target.
            memory: (batch, src_len, d_model) — the encoder stack's output.
            tgt_mask: combined padding + causal mask, (batch, 1, tgt_len, tgt_len).
            memory_mask: src padding mask, (batch, 1, 1, src_len).
            training: passed through to every layer's dropout.

        Returns:
            (batch, tgt_len, d_model)
        """
        for layer in self.dec_layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, training=training)
        return self.final_norm(x)
