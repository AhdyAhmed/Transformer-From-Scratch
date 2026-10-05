"""Decoder layer (TensorFlow) — Section 3.1 of the paper.

One decoder layer is three sublayers, each wrapped in its own residual +
LayerNorm block (Day 8):

    x = ResidualConnection(x, self_attention(x, x, x, tgt_mask))       # masked, causal
    x = ResidualConnection(x, cross_attention(x, memory, memory, memory_mask))
    x = ResidualConnection(x, feed_forward(x))

``memory`` is the encoder's output (Day 10) — the decoder's cross-attention
queries it with the decoder's own (masked) hidden state, giving every
decoder position access to the full source sequence without ever seeing
future target tokens.

Mirrors ``torch_impl/model/decoder.py``.
"""

from __future__ import annotations

import tensorflow as tf

from tf_impl.model.feed_forward import PositionwiseFeedForward
from tf_impl.model.multi_head_attention import MultiHeadAttention
from tf_impl.model.residual import ResidualConnection


class DecoderLayer(tf.keras.layers.Layer):
    """One decoder block: masked self-attention, cross-attention into the
    encoder's output, and a feed-forward sublayer — each wrapped in its own
    residual connection (three independent LayerNorms, unlike the encoder
    layer's two).
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
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.self_attn_block = ResidualConnection(d_model, dropout, norm_first)
        self.cross_attn_block = ResidualConnection(d_model, dropout, norm_first)
        self.feed_forward_block = ResidualConnection(d_model, dropout, norm_first)

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
            x: (batch, tgt_len, d_model) — decoder input (target embeddings on
               layer 1, previous layer's output on later layers).
            memory: (batch, src_len, d_model) — the encoder stack's output.
            tgt_mask: combined padding + causal mask for decoder self-attention,
               (batch, 1, tgt_len, tgt_len). See model/masking.create_target_mask.
            memory_mask: src padding mask for cross-attention, (batch, 1, 1, src_len).
            training: passed through to every dropout layer inside all three sublayers.

        Returns:
            (batch, tgt_len, d_model)
        """
        x = self.self_attn_block(
            x,
            lambda t: self.self_attn(t, t, t, mask=tgt_mask, training=training),
            training=training,
        )
        x = self.cross_attn_block(
            x,
            lambda t: self.cross_attn(t, memory, memory, mask=memory_mask, training=training),
            training=training,
        )
        x = self.feed_forward_block(
            x, lambda t: self.feed_forward(t, training=training), training=training
        )
        return x
