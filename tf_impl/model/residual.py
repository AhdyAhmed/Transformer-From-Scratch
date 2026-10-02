"""Residual connection + LayerNorm wrapper (TensorFlow) — Sections 3.1 & 5.4.

Two conventions exist in the literature, both common enough to be worth
supporting rather than picking one and hardcoding it:

    post-norm (the original paper):
        output = LayerNorm(x + Dropout(sublayer(x)))

    pre-norm (used by most modern large-scale Transformers, e.g. GPT-2+):
        output = x + Dropout(sublayer(LayerNorm(x)))

Pre-norm generally trains more stably at depth (gradients reach early layers
without passing through a LayerNorm first), at a small cost in final
performance versus post-norm at the same depth — see ``design.md`` for the
full trade-off discussion. Controlled by ``TransformerConfig.norm_first``.

Mirrors ``torch_impl/model/residual.py``.
"""

from __future__ import annotations

from typing import Callable

import tensorflow as tf


class ResidualConnection(tf.keras.layers.Layer):
    """Wraps one sublayer (self-attention, cross-attention, or the FFN) with
    a residual connection, dropout, and LayerNorm, in whichever order
    ``norm_first`` selects.

    Usage (``sublayer`` is any callable x -> tensor of the same shape; pass
    the outer ``training`` flag through via closure if the sublayer needs
    it, e.g. a dropout-bearing MultiHeadAttention or FeedForward):

        self.self_attn_block = ResidualConnection(d_model, dropout, norm_first)
        ...
        x = self.self_attn_block(
            x, lambda t: self.self_attn(t, t, t, mask=mask, training=training), training=training
        )
    """

    def __init__(self, d_model: int, dropout: float = 0.1, norm_first: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.norm_first = norm_first

    def call(
        self,
        x: tf.Tensor,
        sublayer: Callable[[tf.Tensor], tf.Tensor],
        training: bool | None = None,
    ) -> tf.Tensor:
        if self.norm_first:
            return x + self.dropout(sublayer(self.norm(x)), training=training)
        return self.norm(x + self.dropout(sublayer(x), training=training))
