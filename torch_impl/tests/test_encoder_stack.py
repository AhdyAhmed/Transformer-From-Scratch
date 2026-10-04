import torch

from torch_impl.model.encoder_stack import Encoder
from torch_impl.model.masking import create_padding_mask

D_MODEL = 16
NUM_HEADS = 4
D_FF = 32
NUM_LAYERS = 3
PAD = 0


def test_output_shape():
    enc = Encoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0).eval()
    x = torch.randn(2, 7, D_MODEL)
    out = enc(x)
    assert out.shape == x.shape


def test_output_shape_with_mask():
    enc = Encoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0).eval()
    seq = torch.tensor([[1, 2, 3, PAD, PAD]])
    mask = create_padding_mask(seq, PAD)
    x = torch.randn(1, 5, D_MODEL)
    out = enc(x, mask)
    assert out.shape == x.shape


def test_layers_have_independent_weights():
    """Six layers sharing one set of weights would collapse the stack's
    capacity to a single layer applied six times — guard against that."""
    enc = Encoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    w0 = enc.layers[0].self_attn.w_q.weight
    w1 = enc.layers[1].self_attn.w_q.weight
    assert w0.data_ptr() != w1.data_ptr()  # not the same underlying storage
    assert not torch.equal(w0, w1)          # and not coincidentally equal either (random init)


def test_depth_actually_changes_the_output():
    """A deeper stack (with different weights per layer) should produce a
    different output than a shallow one — i.e. the extra layers aren't
    silently no-ops."""
    torch.manual_seed(0)
    shallow = Encoder(1, D_MODEL, NUM_HEADS, D_FF, dropout=0.0).eval()
    deep = Encoder(4, D_MODEL, NUM_HEADS, D_FF, dropout=0.0).eval()
    x = torch.randn(1, 5, D_MODEL)
    assert not torch.allclose(shallow(x), deep(x), atol=1e-4)


def test_padding_positions_dont_influence_real_positions():
    """The padding-mask-invariance property (verified per-layer on Day 9)
    must still hold once layers are stacked N deep."""
    enc = Encoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0).eval()
    seq = torch.tensor([[1, 2, 3, PAD, PAD]])
    mask = create_padding_mask(seq, PAD)

    torch.manual_seed(0)
    x = torch.randn(1, 5, D_MODEL)
    out_a = enc(x, mask)

    x_perturbed = x.clone()
    x_perturbed[:, 3:, :] = torch.randn(1, 2, D_MODEL)
    out_b = enc(x_perturbed, mask)

    assert torch.allclose(out_a[:, :3], out_b[:, :3], atol=1e-4)


def test_final_norm_output_is_normalized():
    enc = Encoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0).eval()
    x = torch.randn(1, 1, D_MODEL) * 50
    out = enc(x)
    assert out.mean(dim=-1).abs().item() < 1e-4
    assert abs(out.std(dim=-1, unbiased=False).item() - 1.0) < 1e-2


def test_gradients_flow_through_every_layer():
    enc = Encoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    x = torch.randn(2, 5, D_MODEL, requires_grad=True)
    enc(x).sum().backward()
    assert x.grad.abs().sum() > 0
    for layer in enc.layers:
        assert layer.self_attn.w_q.weight.grad.abs().sum() > 0
        assert layer.feed_forward.linear1.weight.grad.abs().sum() > 0


def test_num_layers_matches_config():
    for n in (1, 2, 6):
        enc = Encoder(n, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
        assert len(enc.layers) == n
