import pytest
import torch

from torch_impl.model.multi_head_attention import MultiHeadAttention
from torch_impl.model.masking import create_target_mask

D_MODEL = 16
NUM_HEADS = 4


def test_rejects_non_divisible_heads():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(d_model=10, num_heads=3)


def test_output_shape_self_attention():
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS, dropout=0.0).eval()
    x = torch.randn(2, 7, D_MODEL)
    out = mha(x, x, x)
    assert out.shape == (2, 7, D_MODEL)


def test_output_shape_cross_attention_different_lengths():
    """query (decoder) and key/value (encoder) can have different seq lengths."""
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS, dropout=0.0).eval()
    query = torch.randn(2, 5, D_MODEL)   # decoder side, len 5
    memory = torch.randn(2, 9, D_MODEL)  # encoder side, len 9
    out = mha(query, memory, memory)
    assert out.shape == (2, 5, D_MODEL)


def test_split_and_merge_heads_are_inverses():
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS)
    x = torch.randn(3, 6, D_MODEL)
    merged_back = mha._merge_heads(mha._split_heads(x))
    assert torch.equal(merged_back, x)


def test_split_heads_shape():
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS)
    x = torch.randn(3, 6, D_MODEL)
    split = mha._split_heads(x)
    assert split.shape == (3, NUM_HEADS, 6, D_MODEL // NUM_HEADS)


def test_caches_attention_weights_for_inspection():
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS, dropout=0.0).eval()
    x = torch.randn(1, 4, D_MODEL)
    mha(x, x, x)
    assert mha.attn_weights is not None
    assert mha.attn_weights.shape == (1, NUM_HEADS, 4, 4)
    assert torch.allclose(mha.attn_weights.sum(-1), torch.ones(1, NUM_HEADS, 4), atol=1e-5)


def test_causal_mask_blocks_future_positions():
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS, dropout=0.0).eval()
    seq = torch.tensor([[1, 2, 3, 4, 5]])
    mask = create_target_mask(seq, pad_idx=0)
    x = torch.randn(1, 5, D_MODEL)
    mha(x, x, x, mask=mask)
    # query position 1 must have ~0 weight on key positions 2, 3, 4
    assert mha.attn_weights[0, :, 1, 2:].abs().max() < 1e-5


def test_gradients_flow_through_all_projections():
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS, dropout=0.0)
    x = torch.randn(2, 5, D_MODEL, requires_grad=True)
    out = mha(x, x, x)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    for proj in (mha.w_q, mha.w_k, mha.w_v, mha.w_o):
        assert proj.weight.grad is not None
        assert proj.weight.grad.abs().sum() > 0


def test_parameter_count():
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS)
    # 4 linear layers, each d_model x d_model + bias
    expected = 4 * (D_MODEL * D_MODEL + D_MODEL)
    actual = sum(p.numel() for p in mha.parameters())
    assert actual == expected
