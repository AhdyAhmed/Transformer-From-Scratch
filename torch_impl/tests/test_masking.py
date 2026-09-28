import pytest
import torch

from torch_impl.model.masking import (
    create_look_ahead_mask,
    create_masks,
    create_padding_mask,
    create_target_mask,
)

PAD = 0

# batch of 2 sequences, right-padded with PAD=0
SEQ = torch.tensor([[5, 6, 7, PAD, PAD],
                    [8, 9, 10, 11, 12]])


def test_padding_mask_shape_and_dtype():
    mask = create_padding_mask(SEQ, PAD)
    assert mask.shape == (2, 1, 1, 5)
    assert mask.dtype == torch.bool


def test_padding_mask_values():
    mask = create_padding_mask(SEQ, PAD)
    assert mask[0, 0, 0].tolist() == [True, True, True, False, False]
    assert mask[1, 0, 0].tolist() == [True] * 5


def test_padding_mask_rejects_wrong_rank():
    with pytest.raises(ValueError):
        create_padding_mask(torch.tensor([1, 2, 3]), PAD)


def test_look_ahead_mask_shape_and_dtype():
    mask = create_look_ahead_mask(4)
    assert mask.shape == (1, 1, 4, 4)
    assert mask.dtype == torch.bool


def test_look_ahead_mask_is_lower_triangular():
    mask = create_look_ahead_mask(4)[0, 0]
    expected = torch.tensor([[1, 0, 0, 0],
                             [1, 1, 0, 0],
                             [1, 1, 1, 0],
                             [1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(mask, expected)


def test_look_ahead_mask_no_future_leakage():
    mask = create_look_ahead_mask(6)[0, 0]
    for i in range(6):
        assert mask[i, i]                 # can see itself
        assert not mask[i, i + 1:].any()  # cannot see the future


def test_target_mask_combines_padding_and_causal():
    mask = create_target_mask(SEQ, PAD)
    assert mask.shape == (2, 1, 5, 5)
    # sequence 0 has real tokens at positions 0..2: row 4 may attend to 0..2 only
    assert mask[0, 0, 4].tolist() == [True, True, True, False, False]
    # row 1 may only attend to positions 0 and 1
    assert mask[0, 0, 1].tolist() == [True, True, False, False, False]
    # sequence 1 has no padding -> pure causal mask
    assert torch.equal(mask[1, 0], create_look_ahead_mask(5)[0, 0])


def test_create_masks_shapes():
    src = torch.tensor([[1, 2, 3, PAD], [4, 5, PAD, PAD]])
    tgt = torch.tensor([[1, 2, PAD], [3, 4, 5]])
    src_mask, tgt_mask, memory_mask = create_masks(src, tgt, PAD)
    assert src_mask.shape == (2, 1, 1, 4)
    assert tgt_mask.shape == (2, 1, 3, 3)
    assert memory_mask.shape == (2, 1, 1, 4)
    assert torch.equal(memory_mask, src_mask)


def test_mask_broadcasts_against_attention_scores():
    scores = torch.randn(2, 8, 5, 5)                 # (B, heads, q, k)
    mask = create_target_mask(SEQ, PAD)
    masked = scores.masked_fill(~mask, -1e9)
    assert masked.shape == scores.shape
    weights = torch.softmax(masked, dim=-1)
    # blocked positions receive ~0 attention, rows still sum to 1
    assert torch.allclose(weights.sum(-1), torch.ones(2, 8, 5), atol=1e-5)
    assert weights[0, :, 1, 2:].abs().max() < 1e-6    # row 1 cannot see 2..4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_masks_follow_input_device():
    tgt = SEQ.cuda()
    assert create_target_mask(tgt, PAD).device == tgt.device
