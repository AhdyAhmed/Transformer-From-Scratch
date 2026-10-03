import numpy as np
import tensorflow as tf

from tf_impl.model.encoder import EncoderLayer
from tf_impl.model.masking import create_padding_mask

D_MODEL = 16
NUM_HEADS = 4
D_FF = 32
PAD = 0


def test_output_shape():
    layer = EncoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    x = tf.random.normal((2, 7, D_MODEL))
    out = layer(x, training=False)
    assert out.shape == x.shape


def test_output_shape_with_padding_mask():
    layer = EncoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    seq = tf.constant([[1, 2, 3, PAD, PAD], [4, 5, 6, 7, PAD]])
    mask = create_padding_mask(seq, PAD)
    x = tf.random.normal((2, 5, D_MODEL))
    out = layer(x, mask=mask, training=False)
    assert out.shape == x.shape


def test_padding_positions_dont_influence_real_positions():
    """Changing only the embedding at a padded position must not change the
    encoder's output at the real (non-padded) positions, since the padding
    mask should block all attention to it."""
    layer = EncoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    seq = tf.constant([[1, 2, 3, PAD, PAD]])
    mask = create_padding_mask(seq, PAD)

    tf.random.set_seed(0)
    x = tf.random.normal((1, 5, D_MODEL))
    out_a = layer(x, mask=mask, training=False).numpy()

    x_perturbed = x.numpy().copy()
    x_perturbed[:, 3:, :] = np.random.randn(1, 2, D_MODEL).astype(np.float32)
    out_b = layer(tf.constant(x_perturbed), mask=mask, training=False).numpy()

    np.testing.assert_allclose(out_a[:, :3], out_b[:, :3], atol=1e-4)


def test_both_norm_modes_run_and_differ():
    x = tf.random.normal((1, 4, D_MODEL))

    post = EncoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0, norm_first=False)
    pre = EncoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0, norm_first=True)
    post(x, training=False)  # build
    pre(x, training=False)
    pre.set_weights(post.get_weights())  # identical weights, different residual ordering

    out_post = post(x, training=False).numpy()
    out_pre = pre(x, training=False).numpy()
    assert out_post.shape == out_pre.shape == x.shape
    assert not np.allclose(out_post, out_pre, atol=1e-4)


def test_gradients_flow_through_both_sublayers():
    layer = EncoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    x = tf.Variable(tf.random.normal((2, 5, D_MODEL)))
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(layer(x, training=True))
    grads = tape.gradient(loss, [x, *layer.trainable_variables])
    for g in grads:
        assert g is not None
        assert tf.reduce_sum(tf.abs(g)).numpy() > 0


def test_caches_self_attention_weights():
    layer = EncoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    x = tf.random.normal((1, 6, D_MODEL))
    layer(x, training=False)
    assert layer.self_attn.attn_weights is not None
    assert layer.self_attn.attn_weights.shape == (1, NUM_HEADS, 6, 6)
