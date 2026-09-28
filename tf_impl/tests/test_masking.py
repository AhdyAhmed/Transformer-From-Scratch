import numpy as np
import pytest
import tensorflow as tf

from tf_impl.model.masking import (
    create_look_ahead_mask,
    create_masks,
    create_padding_mask,
    create_target_mask,
)

PAD = 0

# batch of 2 sequences, right-padded with PAD=0
SEQ = tf.constant([[5, 6, 7, PAD, PAD],
                   [8, 9, 10, 11, 12]])


def test_padding_mask_shape_and_dtype():
    mask = create_padding_mask(SEQ, PAD)
    assert mask.shape == (2, 1, 1, 5)
    assert mask.dtype == tf.bool


def test_padding_mask_values():
    mask = create_padding_mask(SEQ, PAD).numpy()
    assert mask[0, 0, 0].tolist() == [True, True, True, False, False]
    assert mask[1, 0, 0].tolist() == [True] * 5


def test_padding_mask_rejects_wrong_rank():
    with pytest.raises(ValueError):
        create_padding_mask(tf.constant([1, 2, 3]), PAD)


def test_look_ahead_mask_shape_and_dtype():
    mask = create_look_ahead_mask(4)
    assert mask.shape == (1, 1, 4, 4)
    assert mask.dtype == tf.bool


def test_look_ahead_mask_is_lower_triangular():
    mask = create_look_ahead_mask(4).numpy()[0, 0]
    expected = np.array([[1, 0, 0, 0],
                         [1, 1, 0, 0],
                         [1, 1, 1, 0],
                         [1, 1, 1, 1]], dtype=bool)
    np.testing.assert_array_equal(mask, expected)


def test_look_ahead_mask_no_future_leakage():
    mask = create_look_ahead_mask(6).numpy()[0, 0]
    for i in range(6):
        assert mask[i, i]
        assert not mask[i, i + 1:].any()


def test_target_mask_combines_padding_and_causal():
    mask = create_target_mask(SEQ, PAD).numpy()
    assert mask.shape == (2, 1, 5, 5)
    assert mask[0, 0, 4].tolist() == [True, True, True, False, False]
    assert mask[0, 0, 1].tolist() == [True, True, False, False, False]
    np.testing.assert_array_equal(mask[1, 0], create_look_ahead_mask(5).numpy()[0, 0])


def test_create_masks_shapes():
    src = tf.constant([[1, 2, 3, PAD], [4, 5, PAD, PAD]])
    tgt = tf.constant([[1, 2, PAD], [3, 4, 5]])
    src_mask, tgt_mask, memory_mask = create_masks(src, tgt, PAD)
    assert src_mask.shape == (2, 1, 1, 4)
    assert tgt_mask.shape == (2, 1, 3, 3)
    assert memory_mask.shape == (2, 1, 1, 4)
    np.testing.assert_array_equal(memory_mask.numpy(), src_mask.numpy())


def test_mask_broadcasts_against_attention_scores():
    scores = tf.random.normal((2, 8, 5, 5))          # (B, heads, q, k)
    mask = create_target_mask(SEQ, PAD)
    masked = tf.where(mask, scores, -1e9)
    assert masked.shape == scores.shape
    weights = tf.nn.softmax(masked, axis=-1).numpy()
    np.testing.assert_allclose(weights.sum(-1), np.ones((2, 8, 5)), atol=1e-5)
    assert np.abs(weights[0, :, 1, 2:]).max() < 1e-6


def test_works_inside_tf_function_with_dynamic_length():
    """Masks must build under graph mode with an unknown sequence length."""

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.int32)])
    def build(tgt):
        return create_target_mask(tgt, PAD)

    out = build(tf.constant([[1, 2, 3], [4, 5, PAD]], dtype=tf.int32))
    assert out.shape == (2, 1, 3, 3)
