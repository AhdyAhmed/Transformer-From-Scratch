"""Encoder layer (TensorFlow) — Section 3.1 of the paper.

One encoder layer is two sublayers, each wrapped in a residual + LayerNorm
block (Day 8):

    x = ResidualConnection(x, self_attention(x, x, x, mask))
    x = ResidualConnection(x, feed_forward(x))

The full encoder stack (N of these layers, Day 10) just repeats this block.

Mirrors ``torch_impl/model/encoder.py``.
"""

from __future__ import annotations

import tensorflow as tf

from tf_impl.model.feed_forward import PositionwiseFeedForward
from tf_impl.model.multi_head_attention import MultiHeadAttention
from tf_impl.model.residual import ResidualConnection


class EncoderLayer(tf.keras.layers.Layer):
    """One encoder block: self-attention sublayer + feed-forward sublayer,
    each wrapped in a residual connection (post-norm or pre-norm per
    ``norm_first``, see ``model/residual.py``).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.self_attn_block = ResidualConnection(d_model, dropout, norm_first)
        self.feed_forward_block = ResidualConnection(d_model, dropout, norm_first)

    def call(self, x: tf.Tensor, mask: tf.Tensor | None = None, training: bool | None = None) -> tf.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) — encoder input (token + positional embeddings
               on layer 1, previous layer's output on later layers).
            mask: optional src padding mask, (batch, 1, 1, seq_len). Blocks attention
               *to* padding positions; there is no causal mask here since the encoder
               sees the whole source sequence at once.
            training: passed through to every dropout layer inside both sublayers.

        Returns:
            (batch, seq_len, d_model)
        """
        x = self.self_attn_block(
            x, lambda t: self.self_attn(t, t, t, mask=mask, training=training), training=training
        )
        x = self.feed_forward_block(
            x, lambda t: self.feed_forward(t, training=training), training=training
        )
        return x
