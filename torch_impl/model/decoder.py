"""Decoder layer (PyTorch) — Section 3.1 of the paper.

One decoder layer is three sublayers, each wrapped in its own residual +
LayerNorm block (Day 8):

    x = ResidualConnection(x, self_attention(x, x, x, tgt_mask))       # masked, causal
    x = ResidualConnection(x, cross_attention(x, memory, memory, memory_mask))
    x = ResidualConnection(x, feed_forward(x))

``memory`` is the encoder's output (Day 10) — the decoder's cross-attention
queries it with the decoder's own (masked) hidden state, giving every
decoder position access to the full source sequence without ever seeing
future target tokens.

Mirrors ``tf_impl/model/decoder.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_impl.model.feed_forward import PositionwiseFeedForward
from torch_impl.model.multi_head_attention import MultiHeadAttention
from torch_impl.model.residual import ResidualConnection


class DecoderLayer(nn.Module):
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
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.self_attn_block = ResidualConnection(d_model, dropout, norm_first)
        self.cross_attn_block = ResidualConnection(d_model, dropout, norm_first)
        self.feed_forward_block = ResidualConnection(d_model, dropout, norm_first)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, tgt_len, d_model) — decoder input (target embeddings on
               layer 1, previous layer's output on later layers).
            memory: (batch, src_len, d_model) — the encoder stack's output.
            tgt_mask: combined padding + causal mask for decoder self-attention,
               (batch, 1, tgt_len, tgt_len). See model/masking.create_target_mask.
            memory_mask: src padding mask for cross-attention, (batch, 1, 1, src_len).

        Returns:
            (batch, tgt_len, d_model)
        """
        x = self.self_attn_block(x, lambda t: self.self_attn(t, t, t, tgt_mask))
        x = self.cross_attn_block(
            x, lambda t: self.cross_attn(t, memory, memory, memory_mask)
        )
        x = self.feed_forward_block(x, self.feed_forward)
        return x
