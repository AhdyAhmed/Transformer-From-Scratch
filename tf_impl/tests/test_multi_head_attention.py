import numpy as np
import pytest
import tensorflow as tf

from tf_impl.model.multi_head_attention import MultiHeadAttention
from tf_impl.model.masking import create_target_mask

D_MODEL = 16
NUM_HEADS = 4


def test_rejects_non_divisible_heads():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(d_model=10, num_heads=3)


def test_output_shape_self_attention():
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS, dropout=0.0)
    x = tf.random.normal((2, 7, D_MODEL))
    out = mha(x, x, x, training=False)
    assert out.shape == (2, 7, D_MODEL)


def test_output_shape_cross_attention_different_lengths():
    """query (decoder) and key/value (encoder) can have different seq lengths."""
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS, dropout=0.0)
    query = tf.random.normal((2, 5, D_MODEL))
    memory = tf.random.normal((2, 9, D_MODEL))
    out = mha(query, memory, memory, training=False)
    assert out.shape == (2, 5, D_MODEL)


def test_split_and_merge_heads_are_inverses():
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS)
    x = tf.random.normal((3, 6, D_MODEL))
    merged_back = mha._merge_heads(mha._split_heads(x))
    np.testing.assert_allclose(merged_back.numpy(), x.numpy(), atol=1e-6)


def test_split_heads_shape():
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS)
    x = tf.random.normal((3, 6, D_MODEL))
    split = mha._split_heads(x)
    assert split.shape == (3, NUM_HEADS, 6, D_MODEL // NUM_HEADS)


def test_caches_attention_weights_for_inspection():
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS, dropout=0.0)
    x = tf.random.normal((1, 4, D_MODEL))
    mha(x, x, x, training=False)
    assert mha.attn_weights is not None
    assert mha.attn_weights.shape == (1, NUM_HEADS, 4, 4)
    np.testing.assert_allclose(
        tf.reduce_sum(mha.attn_weights, axis=-1).numpy(), np.ones((1, NUM_HEADS, 4)), atol=1e-5
    )


def test_causal_mask_blocks_future_positions():
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS, dropout=0.0)
    seq = tf.constant([[1, 2, 3, 4, 5]])
    mask = create_target_mask(seq, pad_idx=0)
    x = tf.random.normal((1, 5, D_MODEL))
    mha(x, x, x, mask=mask, training=False)
    assert np.abs(mha.attn_weights.numpy()[0, :, 1, 2:]).max() < 1e-5


def test_gradients_flow_through_all_projections():
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS, dropout=0.0)
    x = tf.Variable(tf.random.normal((2, 5, D_MODEL)))
    with tf.GradientTape() as tape:
        out = mha(x, x, x, training=True)
        loss = tf.reduce_sum(out)
    grads = tape.gradient(loss, [x, *mha.trainable_variables])
    for g in grads:
        assert g is not None
        assert tf.reduce_sum(tf.abs(g)).numpy() > 0


def test_parameter_count():
    mha = MultiHeadAttention(D_MODEL, NUM_HEADS)
    _ = mha(tf.zeros((1, 1, D_MODEL)), tf.zeros((1, 1, D_MODEL)), tf.zeros((1, 1, D_MODEL)))
    # 4 dense layers, each d_model x d_model + bias
    expected = 4 * (D_MODEL * D_MODEL + D_MODEL)
    actual = sum(int(tf.size(v)) for v in mha.trainable_variables)
    assert actual == expected
