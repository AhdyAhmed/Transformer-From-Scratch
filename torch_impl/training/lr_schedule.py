"""Noam learning-rate schedule from *Attention Is All You Need* (Day 16).

The public ``rate(step)`` API uses one-based optimizer steps: ``step=1`` is
 the learning rate for the first optimizer update. This keeps the paper's
 formula explicit and easy to compare against TensorFlow's zero-based schedule.
"""
from __future__ import annotations

import math


class NoamSchedule:
    """Warmup followed by inverse-square-root decay.

    ``lr = factor * d_model**-0.5 * min(step**-0.5,
                                         step * warmup_steps**-1.5)``
    """

    def __init__(self, d_model: int, warmup_steps: int = 4000, factor: float = 1.0):
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be positive, got {warmup_steps}")
        if factor <= 0:
            raise ValueError(f"factor must be positive, got {factor}")
        self.d_model = int(d_model)
        self.warmup_steps = int(warmup_steps)
        self.factor = float(factor)

    def rate(self, step: int) -> float:
        """Return the learning rate for a one-based optimizer step."""
        if isinstance(step, bool) or int(step) != step or step < 1:
            raise ValueError(f"step must be a positive integer, got {step!r}")
        step = int(step)
        return self.factor * self.d_model ** -0.5 * min(
            step ** -0.5, step * self.warmup_steps ** -1.5
        )

    __call__ = rate

    def apply(self, optimizer, step: int) -> float:
        """Set every optimizer parameter group's LR and return the value."""
        lr = self.rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self) -> dict[str, int | float]:
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps, "factor": self.factor}
