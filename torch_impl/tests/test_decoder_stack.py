import torch

from torch_impl.model.decoder_stack import Decoder
from torch_impl.model.masking import create_padding_mask, create_target_mask

D_MODEL = 16
NUM_HEADS = 4
D_FF = 32
NUM_LAYERS = 3
PAD = 0


def test_output_shape():
    dec = Decoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0).eval()
    x = torch.randn(2, 4, D_MODEL)
    memory = torch.randn(2, 7, D_MODEL)
    out = dec(x, memory)
    assert out.shape == x.shape


def test_layers_have_independent_weights():
    dec = Decoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    w0 = dec.layers[0].self_attn.w_q.weight
    w1 = dec.layers[1].self_attn.w_q.weight
    assert w0.data_ptr() != w1.data_ptr()
    assert not torch.equal(w0, w1)


def test_depth_actually_changes_the_output():
    torch.manual_seed(0)
    shallow = Decoder(1, D_MODEL, NUM_HEADS, D_FF, dropout=0.0).eval()
    deep = Decoder(4, D_MODEL, NUM_HEADS, D_FF, dropout=0.0).eval()
    x = torch.randn(1, 4, D_MODEL)
    memory = torch.randn(1, 5, D_MODEL)
    assert not torch.allclose(shallow(x, memory), deep(x, memory), atol=1e-4)


def test_causal_masking_survives_stacking():
    dec = Decoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0).eval()
    tgt_seq = torch.tensor([[1, 2, 3, 4, 5]])
    tgt_mask = create_target_mask(tgt_seq, PAD)

    torch.manual_seed(0)
    x = torch.randn(1, 5, D_MODEL)
    memory = torch.randn(1, 6, D_MODEL)
    out_a = dec(x, memory, tgt_mask)

    x_perturbed = x.clone()
    x_perturbed[:, 3:, :] = torch.randn(1, 2, D_MODEL)
    out_b = dec(x_perturbed, memory, tgt_mask)

    assert torch.allclose(out_a[:, :3], out_b[:, :3], atol=1e-4)


def test_memory_masking_survives_stacking():
    dec = Decoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0).eval()
    src_seq = torch.tensor([[1, 2, 3, PAD, PAD]])
    memory_mask = create_padding_mask(src_seq, PAD)

    torch.manual_seed(0)
    x = torch.randn(1, 4, D_MODEL)
    memory = torch.randn(1, 5, D_MODEL)
    out_a = dec(x, memory, memory_mask=memory_mask)

    memory_perturbed = memory.clone()
    memory_perturbed[:, 3:, :] = torch.randn(1, 2, D_MODEL)
    out_b = dec(x, memory_perturbed, memory_mask=memory_mask)

    assert torch.allclose(out_a, out_b, atol=1e-4)


def test_final_norm_output_is_normalized():
    dec = Decoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0).eval()
    x = torch.randn(1, 1, D_MODEL) * 50
    memory = torch.randn(1, 3, D_MODEL)
    out = dec(x, memory)
    assert out.mean(dim=-1).abs().item() < 1e-4
    assert abs(out.std(dim=-1, unbiased=False).item() - 1.0) < 1e-2


def test_gradients_flow_through_every_layer():
    dec = Decoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    x = torch.randn(2, 4, D_MODEL, requires_grad=True)
    memory = torch.randn(2, 5, D_MODEL, requires_grad=True)
    dec(x, memory).sum().backward()
    assert x.grad.abs().sum() > 0
    assert memory.grad.abs().sum() > 0
    for layer in dec.layers:
        assert layer.self_attn.w_q.weight.grad.abs().sum() > 0
        assert layer.cross_attn.w_q.weight.grad.abs().sum() > 0
