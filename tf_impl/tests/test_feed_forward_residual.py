import numpy as np
import pytest
import tensorflow as tf

from tf_impl.model.feed_forward import PositionwiseFeedForward
from tf_impl.model.residual import ResidualConnection

D_MODEL = 16
D_FF = 32


# --------------------------------------------------------------------- #
# PositionwiseFeedForward
# --------------------------------------------------------------------- #

def test_ffn_output_shape():
    ffn = PositionwiseFeedForward(D_MODEL, D_FF, dropout=0.0)
    x = tf.random.normal((3, 5, D_MODEL))
    out = ffn(x, training=False)
    assert out.shape == (3, 5, D_MODEL)


def test_ffn_rejects_unknown_activation():
    with pytest.raises(ValueError, match="activation"):
        PositionwiseFeedForward(D_MODEL, D_FF, activation="tanh")


def test_ffn_applies_same_weights_at_every_position():
    """Position-wise: running one position alone must match running it as
    part of a longer, otherwise-independent sequence."""
    ffn = PositionwiseFeedForward(D_MODEL, D_FF, dropout=0.0)
    x = tf.random.normal((1, 4, D_MODEL))
    full_out = ffn(x, training=False).numpy()
    single_out = ffn(x[:, 2:3, :], training=False).numpy()
    np.testing.assert_allclose(full_out[:, 2:3, :], single_out, atol=1e-5)


def test_ffn_relu_zeroes_negative_preactivations():
    ffn = PositionwiseFeedForward(D_MODEL, D_FF, dropout=0.0, activation="relu")
    _ = ffn(tf.zeros((1, 1, D_MODEL)))  # build
    ffn.linear1.kernel.assign(tf.zeros_like(ffn.linear1.kernel))
    ffn.linear1.bias.assign(tf.fill(ffn.linear1.bias.shape, -1.0))  # ReLU kills every unit
    ffn.linear2.kernel.assign(tf.ones_like(ffn.linear2.kernel))
    ffn.linear2.bias.assign(tf.zeros_like(ffn.linear2.bias))
    out = ffn(tf.random.normal((1, 1, D_MODEL)), training=False)
    np.testing.assert_allclose(out.numpy(), np.zeros_like(out.numpy()))


def test_ffn_gradients_flow():
    ffn = PositionwiseFeedForward(D_MODEL, D_FF, dropout=0.0)
    x = tf.Variable(tf.random.normal((2, 3, D_MODEL)))
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(ffn(x, training=True))
    grads = tape.gradient(loss, [x, *ffn.trainable_variables])
    for g in grads:
        assert g is not None
        assert tf.reduce_sum(tf.abs(g)).numpy() > 0


# --------------------------------------------------------------------- #
# ResidualConnection
# --------------------------------------------------------------------- #

def test_residual_output_shape_both_modes():
    x = tf.random.normal((2, 5, D_MODEL))
    identity = lambda t: t
    for norm_first in (True, False):
        block = ResidualConnection(D_MODEL, dropout=0.0, norm_first=norm_first)
        out = block(x, identity, training=False)
        assert out.shape == x.shape


def test_post_norm_matches_manual_formula():
    block = ResidualConnection(D_MODEL, dropout=0.0, norm_first=False)
    x = tf.random.normal((2, 4, D_MODEL))
    sublayer = lambda t: t * 2 + 1
    out = block(x, sublayer, training=False).numpy()
    expected = block.norm(x + sublayer(x)).numpy()
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_pre_norm_matches_manual_formula():
    block = ResidualConnection(D_MODEL, dropout=0.0, norm_first=True)
    x = tf.random.normal((2, 4, D_MODEL))
    sublayer = lambda t: t * 2 + 1
    out = block(x, sublayer, training=False).numpy()
    expected = (x + sublayer(block.norm(x))).numpy()
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_pre_and_post_norm_differ():
    x = tf.random.normal((1, 3, D_MODEL))
    sublayer = lambda t: t * 2 + 1

    post = ResidualConnection(D_MODEL, dropout=0.0, norm_first=False)
    pre = ResidualConnection(D_MODEL, dropout=0.0, norm_first=True)
    post(x, sublayer, training=False)  # build norm weights
    pre(x, sublayer, training=False)
    pre.norm.set_weights(post.norm.get_weights())  # same LayerNorm weights, different ordering

    out_post = post(x, sublayer, training=False).numpy()
    out_pre = pre(x, sublayer, training=False).numpy()
    assert not np.allclose(out_post, out_pre, atol=1e-4)


def test_residual_with_identity_sublayer_only_normalizes():
    block = ResidualConnection(D_MODEL, dropout=0.0, norm_first=False)
    x = tf.random.normal((2, 3, D_MODEL))
    out = block(x, lambda t: tf.zeros_like(t), training=False).numpy()
    expected = block.norm(x).numpy()
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_output_is_normalized_in_post_norm_mode():
    """Post-norm output should have ~zero mean / ~unit variance per position
    (LayerNorm's defining property), regardless of the sublayer's output scale."""
    block = ResidualConnection(D_MODEL, dropout=0.0, norm_first=False)
    x = tf.random.normal((1, 1, D_MODEL)) * 100  # large scale on purpose
    out = block(x, lambda t: t, training=False).numpy()
    mean = out.mean(axis=-1)
    std = out.std(axis=-1)
    assert abs(mean.item()) < 1e-4
    assert abs(std.item() - 1.0) < 1e-2
