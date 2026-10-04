"""Encoder stack (PyTorch) — Section 3.1 of the paper: "N = 6" identical layers.

    x = embeddings(src)
    for layer in layers:
        x = layer(x, mask)
    x = final_norm(x)          # see note below
    return x                    # "memory" the decoder will cross-attend to

Mirrors ``tf_impl/model/encoder_stack.py``.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from torch_impl.model.encoder import EncoderLayer


class Encoder(nn.Module):
    """A stack of ``num_layers`` independent ``EncoderLayer``s.

    Each layer gets its own freshly-initialized weights (``copy.deepcopy``,
    not a shared reference) — six layers sharing one set of weights would
    collapse the stack's capacity to a single layer applied six times.

    Final LayerNorm: in post-norm mode (the paper's default) every
    ``EncoderLayer`` already ends with a LayerNorm, so this final one is
    technically redundant but harmless, and kept for symmetry with pre-norm
    mode, where the last layer's residual *stream* is never itself
    normalized — only the sublayer inputs are — so a model built with
    ``norm_first=True`` genuinely needs this extra norm before its output is
    used downstream (by the decoder's cross-attention).
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
        layer = EncoderLayer(d_model, num_heads, d_ff, dropout, norm_first)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) — already embedded + positionally encoded input.
            mask: optional src padding mask, (batch, 1, 1, seq_len).

        Returns:
            (batch, seq_len, d_model) — the encoder "memory" passed to every
            decoder layer's cross-attention.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)
