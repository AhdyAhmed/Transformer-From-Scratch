"""Loss functions used by the Transformer training loop (Day 16)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """Token cross-entropy with label smoothing and padding ignored.

    ``logits`` has shape ``(..., vocab_size)`` and ``targets`` has shape
    ``(...)``. The smoothed target distribution is a mixture of the one-hot
    target and a uniform distribution over the vocabulary. Positions whose
    target equals ``pad_idx`` contribute exactly zero to the loss.

    ``reduction='mean'`` averages over non-padding tokens (not batch/sequence
    positions), which makes loss comparable across differently padded batches.
    """

    def __init__(self, label_smoothing: float = 0.1, pad_idx: int = 0,
                 reduction: str = "mean") -> None:
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if reduction not in {"none", "sum", "mean"}:
            raise ValueError("reduction must be 'none', 'sum', or 'mean'")
        self.label_smoothing = float(label_smoothing)
        self.pad_idx = int(pad_idx)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim < 2:
            raise ValueError("logits must have at least 2 dimensions (..., vocab_size)")
        if tuple(logits.shape[:-1]) != tuple(targets.shape):
            raise ValueError(f"target shape {tuple(targets.shape)} must match logits leading shape {tuple(logits.shape[:-1])}")
        vocab_size = logits.shape[-1]
        if not 0 <= self.pad_idx < vocab_size:
            raise ValueError(f"pad_idx {self.pad_idx} is outside vocabulary size {vocab_size}")

        log_probs = F.log_softmax(logits, dim=-1).reshape(-1, vocab_size)
        flat_targets = targets.reshape(-1).long()
        valid = flat_targets.ne(self.pad_idx)
        safe_targets = flat_targets.masked_fill(~valid, 0)
        nll = -log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        per_token = (1.0 - self.label_smoothing) * nll + self.label_smoothing * smooth
        per_token = per_token * valid.to(per_token.dtype)

        if self.reduction == "none":
            return per_token.reshape(targets.shape)
        total = per_token.sum()
        if self.reduction == "sum":
            return total
        count = valid.sum()
        # Keep an all-padding batch differentiable and finite.
        return total / count.clamp_min(1).to(total.dtype)
