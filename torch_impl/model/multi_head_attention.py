"""Multi-head attention (PyTorch) — Section 3.2.2 of the paper.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)

Wraps ``scaled_dot_product_attention`` (Day 4) with the learned input/output
projections and the head split/merge reshape logic. This is where real,
trainable weights first appear, so cross-framework parity (Day 6) needs
matching weights loaded into both implementations rather than the "feed
identical inputs" trick used for the parameter-free Day 3/4 modules.

Mirrors ``tf_impl/model/multi_head_attention.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_impl.model.attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """Splits d_model into num_heads parallel attention heads of size d_k,
    attends independently in each, then concatenates and projects back.

    Used three ways in the full model (Day 9-12):
      - encoder self-attention   (query, key, value all = encoder input)
      - decoder self-attention   (query, key, value all = decoder input, causal-masked)
      - decoder cross-attention  (query = decoder state, key/value = encoder output)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # One combined linear layer per Q/K/V, rather than num_heads separate
        # small ones — mathematically identical to per-head projections once
        # reshaped, but a single matmul is far more efficient.
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attn_weights: torch.Tensor | None = None  # cached from the last forward, for inspection/plots

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)."""
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)."""
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, q_len, d_model)
            key:   (batch, k_len, d_model)
            value: (batch, k_len, d_model)
            mask:  optional bool tensor broadcastable to
                   (batch, num_heads, q_len, k_len) or (batch, 1, q_len, k_len).

        Returns:
            (batch, q_len, d_model)
        """
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        attn_out, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout
        )
        self.attn_weights = attn_weights  # (batch, num_heads, q_len, k_len)

        merged = self._merge_heads(attn_out)
        return self.w_o(merged)
