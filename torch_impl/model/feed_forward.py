"""Position-wise feed-forward network (PyTorch) — Section 3.3 of the paper.

    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

Applied identically (same weights) to every position independently — hence
"position-wise". Expands d_model -> d_ff -> d_model, giving the model extra
representational capacity between attention layers.

Mirrors ``tf_impl/model/feed_forward.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"activation must be 'relu' or 'gelu', got {activation!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) -> same shape.

        Dropout sits between the two linear layers (after the activation),
        matching the paper and the common reference implementations.
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
