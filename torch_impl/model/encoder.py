"""Encoder layer (PyTorch) — Section 3.1 of the paper.

One encoder layer is two sublayers, each wrapped in a residual + LayerNorm
block (Day 8):

    x = ResidualConnection(x, self_attention(x, x, x, mask))
    x = ResidualConnection(x, feed_forward(x))

The full encoder stack (N of these layers, Day 10) just repeats this block.

Mirrors ``tf_impl/model/encoder.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_impl.model.feed_forward import PositionwiseFeedForward
from torch_impl.model.multi_head_attention import MultiHeadAttention
from torch_impl.model.residual import ResidualConnection


class EncoderLayer(nn.Module):
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
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.self_attn_block = ResidualConnection(d_model, dropout, norm_first)
        self.feed_forward_block = ResidualConnection(d_model, dropout, norm_first)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) — encoder input (token + positional embeddings
               on layer 1, previous layer's output on later layers).
            mask: optional src padding mask, (batch, 1, 1, seq_len). Blocks attention
               *to* padding positions; there is no causal mask here since the encoder
               sees the whole source sequence at once.

        Returns:
            (batch, seq_len, d_model)
        """
        x = self.self_attn_block(x, lambda t: self.self_attn(t, t, t, mask))
        x = self.feed_forward_block(x, self.feed_forward)
        return x
