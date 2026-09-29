"""Token embeddings and sinusoidal positional encoding (PyTorch).

Together these turn a batch of token ids into a matrix of shape
``(batch, seq_len, d_model)`` ready for the encoder/decoder stacks (Day 8+).
Mirrors ``tf_impl/model/embeddings.py``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Learned token embedding, scaled by sqrt(d_model) as in the paper.

    Section 3.4 of Vaswani et al.: "we multiply those weights by sqrt(d_model)"
    so embedding and positional-encoding magnitudes are comparable before
    they're summed. ``padding_idx`` tells PyTorch to zero the pad row's
    gradient, so it never drifts away from zero during training.
    """

    def __init__(self, vocab_size: int, d_model: int, pad_idx: int = 0) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) int64 token ids -> (batch, seq_len, d_model)."""
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Section 3.5 of the paper).

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Precomputed once up to ``max_len`` and stored as a non-persistent buffer
    (``persistent=False``) so it moves with the module across devices/dtypes
    via ``.to(...)`` but is never written to a checkpoint — it's a fixed
    function of position, not a learned parameter.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1).float()               # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                                    # (d_model/2,)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)        # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) token embeddings -> same shape, + PE."""
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"sequence length {seq_len} exceeds max_len {self.pe.size(1)} "
                "this PositionalEncoding was built for"
            )
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
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
    ) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model, pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.positional_encoding(self.token_embedding(x))
