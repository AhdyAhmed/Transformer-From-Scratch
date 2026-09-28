"""Attention mask utilities (PyTorch).

Convention (shared with ``tf_impl/model/masking.py``):

    True  -> this key position MAY be attended to
    False -> this key position is BLOCKED

All masks are ``torch.bool`` and broadcast against attention scores of shape
``(batch, num_heads, q_len, k_len)``. In the attention layer (Day 4) they are
applied as ``scores.masked_fill(~mask, NEG_INF)`` before the softmax.
"""

from __future__ import annotations

import torch

# Large negative value used to zero-out blocked positions after softmax.
# -1e9 (not -inf) keeps fully-masked rows finite and avoids NaNs.
NEG_INF = -1e9


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Mask that blocks attention *to* padding tokens.

    Args:
        seq: token ids, shape ``(batch, seq_len)``.
        pad_idx: id of the padding token.

    Returns:
        bool tensor of shape ``(batch, 1, 1, seq_len)``.
    """
    if seq.dim() != 2:
        raise ValueError(f"seq must be 2-D (batch, seq_len), got shape {tuple(seq.shape)}")
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """Causal mask: position *i* may only attend to positions ``<= i``.

    Returns:
        bool tensor of shape ``(1, 1, size, size)`` (lower-triangular).
    """
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def create_target_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Decoder self-attention mask = padding mask AND look-ahead mask.

    Returns:
        bool tensor of shape ``(batch, 1, tgt_len, tgt_len)``.
    """
    tgt_len = tgt.size(1)
    pad_mask = create_padding_mask(tgt, pad_idx)                  # (B, 1, 1, T)
    causal = create_look_ahead_mask(tgt_len, device=tgt.device)   # (1, 1, T, T)
    return pad_mask & causal                                      # (B, 1, T, T)


def create_masks(
    src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build every mask the encoder-decoder Transformer needs.

    Returns:
        src_mask:   ``(B, 1, 1, S)`` encoder self-attention (blocks source padding).
        tgt_mask:   ``(B, 1, T, T)`` decoder self-attention (padding + causal).
        memory_mask:``(B, 1, 1, S)`` decoder cross-attention (blocks source padding).
    """
    src_mask = create_padding_mask(src, pad_idx)
    tgt_mask = create_target_mask(tgt, pad_idx)
    memory_mask = src_mask
    return src_mask, tgt_mask, memory_mask
