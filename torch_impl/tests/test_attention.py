import math

import torch

from torch_impl.model.attention import scaled_dot_product_attention
from torch_impl.model.masking import create_target_mask

D_K = 8


def test_output_and_weight_shapes():
    q = torch.randn(2, 4, 5, D_K)   # (batch, heads, q_len, d_k)
    k = torch.randn(2, 4, 6, D_K)
    v = torch.randn(2, 4, 6, D_K)
    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 4, 5, D_K)
    assert weights.shape == (2, 4, 5, 6)


def test_weights_sum_to_one():
    q = torch.randn(2, 4, 5, D_K)
    k = torch.randn(2, 4, 6, D_K)
    v = torch.randn(2, 4, 6, D_K)
    _, weights = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 4, 5), atol=1e-5)


def test_matches_manual_computation():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 3, D_K)
    k = torch.randn(1, 1, 3, D_K)
    v = torch.randn(1, 1, 3, D_K)

    expected_scores = (q @ k.transpose(-2, -1)) / math.sqrt(D_K)
    expected_weights = torch.softmax(expected_scores, dim=-1)
    expected_out = expected_weights @ v

    out, weights = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, expected_out, atol=1e-6)
    assert torch.allclose(weights, expected_weights, atol=1e-6)


def test_masked_positions_get_zero_weight():
    seq = torch.tensor([[1, 2, 3, 0, 0]])  # last two are padding
    mask = create_target_mask(seq, pad_idx=0)  # (1, 1, 5, 5)
    q = k = v = torch.randn(1, 1, 5, D_K)
    _, weights = scaled_dot_product_attention(q, k, v, mask=mask)
    # row 0 (query pos 0) may only attend to key pos 0 -> weight there ~1
    assert weights[0, 0, 0, 0] > 0.999
    assert weights[0, 0, 0, 1:].abs().max() < 1e-5
    # padded key positions (3, 4) get ~0 weight for every non-padded query row
    assert weights[0, 0, :3, 3:].abs().max() < 1e-5


def test_uniform_scores_give_uniform_weights():
    q = torch.zeros(1, 1, 1, D_K)
    k = torch.zeros(1, 1, 4, D_K)
    v = torch.randn(1, 1, 4, D_K)
    out, weights = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(weights, torch.full((1, 1, 1, 4), 0.25), atol=1e-6)
    assert torch.allclose(out[0, 0, 0], v.mean(dim=2)[0, 0], atol=1e-5)


def test_dropout_zeroes_output_when_p_is_one_but_weights_unaffected():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 3, D_K)
    k = torch.randn(1, 1, 3, D_K)
    v = torch.randn(1, 1, 3, D_K)
    dropout = torch.nn.Dropout(p=1.0)
    dropout.train()
    out, weights = scaled_dot_product_attention(q, k, v, dropout=dropout)
    # returned weights are pre-dropout -> still a valid distribution
    assert torch.allclose(weights.sum(-1), torch.ones(1, 1, 3), atol=1e-5)
    # but the output used dropped weights -> everything zeroed at p=1.0
    assert torch.allclose(out, torch.zeros_like(out))


def test_gradients_flow_to_query_key_value():
    q = torch.randn(1, 1, 3, D_K, requires_grad=True)
    k = torch.randn(1, 1, 3, D_K, requires_grad=True)
    v = torch.randn(1, 1, 3, D_K, requires_grad=True)
    out, _ = scaled_dot_product_attention(q, k, v)
    out.sum().backward()
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert k.grad is not None and k.grad.abs().sum() > 0
    assert v.grad is not None and v.grad.abs().sum() > 0
