import pytest
import torch
import torch.nn as nn

from torch_impl.model.feed_forward import PositionwiseFeedForward
from torch_impl.model.residual import ResidualConnection

D_MODEL = 16
D_FF = 32


# --------------------------------------------------------------------- #
# PositionwiseFeedForward
# --------------------------------------------------------------------- #

def test_ffn_output_shape():
    ffn = PositionwiseFeedForward(D_MODEL, D_FF, dropout=0.0).eval()
    x = torch.randn(3, 5, D_MODEL)
    out = ffn(x)
    assert out.shape == (3, 5, D_MODEL)


def test_ffn_rejects_unknown_activation():
    with pytest.raises(ValueError, match="activation"):
        PositionwiseFeedForward(D_MODEL, D_FF, activation="tanh")


def test_ffn_applies_same_weights_at_every_position():
    """Position-wise: running one position alone must match running it as
    part of a longer, otherwise-independent sequence."""
    ffn = PositionwiseFeedForward(D_MODEL, D_FF, dropout=0.0).eval()
    x = torch.randn(1, 4, D_MODEL)
    full_out = ffn(x)
    single_out = ffn(x[:, 2:3, :])
    assert torch.allclose(full_out[:, 2:3, :], single_out, atol=1e-6)


def test_ffn_relu_zeroes_negative_preactivations():
    ffn = PositionwiseFeedForward(D_MODEL, D_FF, dropout=0.0, activation="relu")
    with torch.no_grad():
        ffn.linear1.weight.zero_()
        ffn.linear1.bias.fill_(-1.0)  # every pre-activation is -1 -> ReLU kills it
        ffn.linear2.weight.fill_(1.0)
        ffn.linear2.bias.zero_()
    out = ffn(torch.randn(1, 1, D_MODEL))
    assert torch.allclose(out, torch.zeros_like(out))


def test_ffn_gradients_flow():
    ffn = PositionwiseFeedForward(D_MODEL, D_FF, dropout=0.0)
    x = torch.randn(2, 3, D_MODEL, requires_grad=True)
    ffn(x).sum().backward()
    assert x.grad.abs().sum() > 0
    assert ffn.linear1.weight.grad.abs().sum() > 0
    assert ffn.linear2.weight.grad.abs().sum() > 0


# --------------------------------------------------------------------- #
# ResidualConnection
# --------------------------------------------------------------------- #

def test_residual_output_shape_both_modes():
    x = torch.randn(2, 5, D_MODEL)
    identity = lambda t: t
    for norm_first in (True, False):
        block = ResidualConnection(D_MODEL, dropout=0.0, norm_first=norm_first).eval()
        out = block(x, identity)
        assert out.shape == x.shape


def test_post_norm_matches_manual_formula():
    torch.manual_seed(0)
    block = ResidualConnection(D_MODEL, dropout=0.0, norm_first=False).eval()
    x = torch.randn(2, 4, D_MODEL)
    sublayer = lambda t: t * 2 + 1
    out = block(x, sublayer)
    expected = block.norm(x + sublayer(x))
    assert torch.allclose(out, expected, atol=1e-6)


def test_pre_norm_matches_manual_formula():
    torch.manual_seed(0)
    block = ResidualConnection(D_MODEL, dropout=0.0, norm_first=True).eval()
    x = torch.randn(2, 4, D_MODEL)
    sublayer = lambda t: t * 2 + 1
    out = block(x, sublayer)
    expected = x + sublayer(block.norm(x))
    assert torch.allclose(out, expected, atol=1e-6)


def test_pre_and_post_norm_differ():
    torch.manual_seed(0)
    x = torch.randn(1, 3, D_MODEL)
    sublayer = lambda t: t * 2 + 1

    post = ResidualConnection(D_MODEL, dropout=0.0, norm_first=False).eval()
    pre = ResidualConnection(D_MODEL, dropout=0.0, norm_first=True).eval()
    # give them the same LayerNorm weights so the only difference is ordering
    pre.norm.load_state_dict(post.norm.state_dict())

    assert not torch.allclose(post(x, sublayer), pre(x, sublayer), atol=1e-4)


def test_residual_with_identity_sublayer_only_normalizes():
    block = ResidualConnection(D_MODEL, dropout=0.0, norm_first=False).eval()
    x = torch.randn(2, 3, D_MODEL)
    out = block(x, lambda t: torch.zeros_like(t))  # sublayer contributes nothing
    assert torch.allclose(out, block.norm(x), atol=1e-6)


def test_output_is_normalized_in_post_norm_mode():
    """Post-norm output should have ~zero mean / ~unit variance per position
    (LayerNorm's defining property), regardless of the sublayer's output scale."""
    block = ResidualConnection(D_MODEL, dropout=0.0, norm_first=False).eval()
    x = torch.randn(1, 1, D_MODEL) * 100  # large scale on purpose
    out = block(x, lambda t: t)
    assert out.mean(dim=-1).abs().item() < 1e-5
    assert abs(out.std(dim=-1, unbiased=False).item() - 1.0) < 1e-3
