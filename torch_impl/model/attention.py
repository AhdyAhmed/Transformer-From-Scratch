"""Scaled dot-product attention (PyTorch) — Section 3.2.1 of the paper.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

This is the core primitive multi-head attention (Day 5) builds on top of. It
is deliberately kept as a plain function, not a module: it has no learned
parameters of its own, only the projections around it do.

Mirrors ``tf_impl/model/attention.py``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from torch_impl.model.masking import NEG_INF


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: torch.nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        query: (..., q_len, d_k)
        key:   (..., k_len, d_k)
        value: (..., k_len, d_v)
        mask:  optional bool tensor broadcastable to (..., q_len, k_len).
               True = may attend, False = blocked (see model/masking.py).
        dropout: optional dropout module applied to the attention weights
            (applied to weights, not to the output — matches the paper).

    Returns:
        output: (..., q_len, d_v)
        attn_weights: (..., q_len, k_len) — post-softmax, pre-dropout, so
            they remain valid probability distributions for visualization.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, NEG_INF)

    attn_weights = F.softmax(scores, dim=-1)

    weights_for_output = attn_weights
    if dropout is not None:
        weights_for_output = dropout(attn_weights)

    output = torch.matmul(weights_for_output, value)
    return output, attn_weights
