"""Day 16 tests for the Noam schedule and label-smoothed loss."""
import math

import pytest
import torch
import torch.nn.functional as F

from torch_impl.training.lr_schedule import NoamSchedule
from torch_impl.training.losses import LabelSmoothingLoss


def test_noam_schedule_matches_paper_formula_and_warmup_peak():
    schedule = NoamSchedule(d_model=512, warmup_steps=4000)
    expected_first = 512 ** -0.5 * min(1.0, 4000 ** -1.5)
    assert schedule(1) == pytest.approx(expected_first)
    peak = schedule(4000)
    assert all(schedule(step) < peak for step in (1, 100, 1000))
    assert schedule(8000) == pytest.approx(peak / math.sqrt(2))
    assert schedule(4001) < peak


def test_noam_schedule_validates_arguments_and_optimizer_application():
    with pytest.raises(ValueError):
        NoamSchedule(d_model=0)
    with pytest.raises(ValueError):
        NoamSchedule(d_model=32, warmup_steps=0)
    with pytest.raises(ValueError):
        NoamSchedule(d_model=32)(0)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    value = NoamSchedule(32, 10).apply(optimizer, 3)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(value)


def test_label_smoothing_zero_matches_cross_entropy_when_no_padding():
    torch.manual_seed(4)
    logits = torch.randn(2, 3, 7)
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
    actual = LabelSmoothingLoss(0.0, pad_idx=0)(logits, targets)
    expected = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
    assert torch.allclose(actual, expected)


def test_label_smoothing_ignores_padding_and_normalizes_by_valid_tokens():
    torch.manual_seed(5)
    logits = torch.randn(2, 4, 6, requires_grad=True)
    targets = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
    criterion = LabelSmoothingLoss(0.1, pad_idx=0)
    actual = criterion(logits, targets)
    valid_logits = logits[targets.ne(0)]
    valid_targets = targets[targets.ne(0)]
    log_probs = F.log_softmax(valid_logits, dim=-1)
    nll = -log_probs.gather(1, valid_targets[:, None]).squeeze(1)
    expected = ((1 - 0.1) * nll + 0.1 * -log_probs.mean(-1)).mean()
    assert torch.allclose(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[targets.eq(0)], torch.zeros_like(logits.grad[targets.eq(0)]))


def test_label_smoothing_reductions_and_all_padding_batch():
    logits = torch.randn(2, 3, 5, requires_grad=True)
    targets = torch.zeros((2, 3), dtype=torch.long)
    loss_fn = LabelSmoothingLoss(0.1, pad_idx=0)
    assert loss_fn(logits, targets).item() == 0.0
    assert loss_fn(logits, targets).requires_grad
    assert loss_fn(logits, targets).backward() is None
    none = LabelSmoothingLoss(0.1, pad_idx=0, reduction="none")(logits, targets)
    summed = LabelSmoothingLoss(0.1, pad_idx=0, reduction="sum")(logits, targets)
    assert none.shape == targets.shape
    assert summed.item() == 0.0
    with pytest.raises(ValueError):
        LabelSmoothingLoss(1.0)
    with pytest.raises(ValueError):
        LabelSmoothingLoss(0.1, reduction="bad")
