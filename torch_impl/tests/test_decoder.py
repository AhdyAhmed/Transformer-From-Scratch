import torch

from torch_impl.model.decoder import DecoderLayer
from torch_impl.model.masking import create_padding_mask, create_target_mask

D_MODEL = 16
NUM_HEADS = 4
D_FF = 32
PAD = 0


def test_output_shape():
    layer = DecoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0).eval()
    x = torch.randn(2, 4, D_MODEL)       # decoder side, len 4
    memory = torch.randn(2, 7, D_MODEL)  # encoder side, len 7 (can differ)
    out = layer(x, memory)
    assert out.shape == x.shape


def test_output_shape_with_masks():
    layer = DecoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0).eval()
    tgt_seq = torch.tensor([[1, 2, 3, PAD]])
    src_seq = torch.tensor([[4, 5, 6, 7, PAD, PAD]])
    tgt_mask = create_target_mask(tgt_seq, PAD)
    memory_mask = create_padding_mask(src_seq, PAD)

    x = torch.randn(1, 4, D_MODEL)
    memory = torch.randn(1, 6, D_MODEL)
    out = layer(x, memory, tgt_mask, memory_mask)
    assert out.shape == x.shape


def test_causal_mask_blocks_future_target_positions():
    """Decoder self-attention must not let position i see position > i."""
    layer = DecoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0).eval()
    tgt_seq = torch.tensor([[1, 2, 3, 4, 5]])  # no padding, pure causal case
    tgt_mask = create_target_mask(tgt_seq, PAD)

    torch.manual_seed(0)
    x = torch.randn(1, 5, D_MODEL)
    memory = torch.randn(1, 6, D_MODEL)
    out_a = layer(x, memory, tgt_mask)

    # perturbing only future positions (3, 4) must not change the output at
    # earlier positions (0, 1, 2), since they're causally masked out.
    x_perturbed = x.clone()
    x_perturbed[:, 3:, :] = torch.randn(1, 2, D_MODEL)
    out_b = layer(x_perturbed, memory, tgt_mask)

    assert torch.allclose(out_a[:, :3], out_b[:, :3], atol=1e-5)


def test_memory_mask_blocks_padded_source_positions():
    """Cross-attention must not attend to padded encoder (source) positions."""
    layer = DecoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0).eval()
    src_seq = torch.tensor([[1, 2, 3, PAD, PAD]])
    memory_mask = create_padding_mask(src_seq, PAD)

    torch.manual_seed(0)
    x = torch.randn(1, 4, D_MODEL)
    memory = torch.randn(1, 5, D_MODEL)
    out_a = layer(x, memory, memory_mask=memory_mask)

    memory_perturbed = memory.clone()
    memory_perturbed[:, 3:, :] = torch.randn(1, 2, D_MODEL)  # scramble padded src positions
    out_b = layer(x, memory_perturbed, memory_mask=memory_mask)

    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_cross_attention_actually_uses_memory():
    """Sanity check the other direction: changing *non-padded* memory content
    SHOULD change the output (cross-attention isn't accidentally a no-op)."""
    layer = DecoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0).eval()
    torch.manual_seed(0)
    x = torch.randn(1, 3, D_MODEL)
    memory_a = torch.randn(1, 5, D_MODEL)
    memory_b = torch.randn(1, 5, D_MODEL)
    assert not torch.allclose(layer(x, memory_a), layer(x, memory_b), atol=1e-4)


def test_both_norm_modes_run_and_differ():
    torch.manual_seed(0)
    x = torch.randn(1, 3, D_MODEL)
    memory = torch.randn(1, 4, D_MODEL)

    post = DecoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0, norm_first=False).eval()
    pre = DecoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0, norm_first=True).eval()
    pre.load_state_dict(post.state_dict())

    assert not torch.allclose(post(x, memory), pre(x, memory), atol=1e-4)


def test_gradients_flow_through_all_three_sublayers():
    layer = DecoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    x = torch.randn(2, 4, D_MODEL, requires_grad=True)
    memory = torch.randn(2, 5, D_MODEL, requires_grad=True)
    layer(x, memory).sum().backward()
    assert x.grad.abs().sum() > 0
    assert memory.grad.abs().sum() > 0
    assert layer.self_attn.w_q.weight.grad.abs().sum() > 0
    assert layer.cross_attn.w_q.weight.grad.abs().sum() > 0
    assert layer.feed_forward.linear1.weight.grad.abs().sum() > 0
