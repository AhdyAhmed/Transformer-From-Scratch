"""Residual connection + LayerNorm wrapper (PyTorch) — Sections 3.1 & 5.4.

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

Mirrors ``tf_impl/model/residual.py``.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class ResidualConnection(nn.Module):
    """Wraps one sublayer (self-attention, cross-attention, or the FFN) with
    a residual connection, dropout, and LayerNorm, in whichever order
    ``norm_first`` selects.

    Usage (the ``sublayer`` is any callable x -> tensor of the same shape):

        self.self_attn_block = ResidualConnection(d_model, dropout, norm_first)
        ...
        x = self.self_attn_block(x, lambda t: self.self_attn(t, t, t, mask))
    """

    def __init__(self, d_model: int, dropout: float = 0.1, norm_first: bool = False) -> None:
        super().__init__()
        # eps pinned to match tf.keras.layers.LayerNormalization's default elsewhere
        # in this repo (1e-6) exactly — PyTorch's own default is 1e-5, Keras' is
        # 1e-3, and neither matches the other, which would otherwise silently
        # break cross-framework parity once LayerNorm enters those tests.
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if self.norm_first:
            return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))
