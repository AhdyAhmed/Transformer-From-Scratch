"""Decoder stack (PyTorch) — Section 3.1 of the paper: "N = 6" identical layers.

    x = embeddings(tgt)
    for layer in layers:
        x = layer(x, memory, tgt_mask, memory_mask)
    x = final_norm(x)
    return x

Mirrors ``Encoder`` (Day 10) exactly, with the extra ``memory``/``memory_mask``
every ``DecoderLayer`` needs for cross-attention. Mirrors
``tf_impl/model/decoder_stack.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_impl.model.decoder import DecoderLayer


class Decoder(nn.Module):
    """A stack of ``num_layers`` independent ``DecoderLayer``s + final LayerNorm.

    Each layer is constructed independently so it receives a fresh random
    initialization; copying one initialized layer would produce identical initial
    values even when the parameters do not share storage.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, norm_first)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, tgt_len, d_model) — already embedded + positionally encoded target.
            memory: (batch, src_len, d_model) — the encoder stack's output.
            tgt_mask: combined padding + causal mask, (batch, 1, tgt_len, tgt_len).
            memory_mask: src padding mask, (batch, 1, 1, src_len).

        Returns:
            (batch, tgt_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.final_norm(x)
