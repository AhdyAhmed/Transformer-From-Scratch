import numpy as np
import tensorflow as tf

from tf_impl.model.decoder import DecoderLayer
from tf_impl.model.masking import create_padding_mask, create_target_mask

D_MODEL = 16
NUM_HEADS = 4
D_FF = 32
PAD = 0


def test_output_shape():
    layer = DecoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    x = tf.random.normal((2, 4, D_MODEL))
    memory = tf.random.normal((2, 7, D_MODEL))
    out = layer(x, memory, training=False)
    assert out.shape == x.shape


def test_output_shape_with_masks():
    layer = DecoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    tgt_seq = tf.constant([[1, 2, 3, PAD]])
    src_seq = tf.constant([[4, 5, 6, 7, PAD, PAD]])
    tgt_mask = create_target_mask(tgt_seq, PAD)
    memory_mask = create_padding_mask(src_seq, PAD)

    x = tf.random.normal((1, 4, D_MODEL))
    memory = tf.random.normal((1, 6, D_MODEL))
    out = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, training=False)
    assert out.shape == x.shape


def test_causal_mask_blocks_future_target_positions():
    """Decoder self-attention must not let position i see position > i."""
    layer = DecoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    tgt_seq = tf.constant([[1, 2, 3, 4, 5]])
    tgt_mask = create_target_mask(tgt_seq, PAD)

    tf.random.set_seed(0)
    x = tf.random.normal((1, 5, D_MODEL))
    memory = tf.random.normal((1, 6, D_MODEL))
    out_a = layer(x, memory, tgt_mask=tgt_mask, training=False).numpy()

    x_perturbed = x.numpy().copy()
    x_perturbed[:, 3:, :] = np.random.randn(1, 2, D_MODEL).astype(np.float32)
    out_b = layer(tf.constant(x_perturbed), memory, tgt_mask=tgt_mask, training=False).numpy()

    np.testing.assert_allclose(out_a[:, :3], out_b[:, :3], atol=1e-4)


def test_memory_mask_blocks_padded_source_positions():
    """Cross-attention must not attend to padded encoder (source) positions."""
    layer = DecoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    src_seq = tf.constant([[1, 2, 3, PAD, PAD]])
    memory_mask = create_padding_mask(src_seq, PAD)

    tf.random.set_seed(0)
    x = tf.random.normal((1, 4, D_MODEL))
    memory = tf.random.normal((1, 5, D_MODEL))
    out_a = layer(x, memory, memory_mask=memory_mask, training=False).numpy()

    memory_perturbed = memory.numpy().copy()
    memory_perturbed[:, 3:, :] = np.random.randn(1, 2, D_MODEL).astype(np.float32)
    out_b = layer(x, tf.constant(memory_perturbed), memory_mask=memory_mask, training=False).numpy()

    np.testing.assert_allclose(out_a, out_b, atol=1e-4)


def test_cross_attention_actually_uses_memory():
    """Sanity check the other direction: changing *non-padded* memory content
    SHOULD change the output (cross-attention isn't accidentally a no-op)."""
    layer = DecoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    x = tf.random.normal((1, 3, D_MODEL))
    memory_a = tf.random.normal((1, 5, D_MODEL))
    memory_b = tf.random.normal((1, 5, D_MODEL))
    out_a = layer(x, memory_a, training=False).numpy()
    out_b = layer(x, memory_b, training=False).numpy()
    assert not np.allclose(out_a, out_b, atol=1e-4)


def test_both_norm_modes_run_and_differ():
    x = tf.random.normal((1, 3, D_MODEL))
    memory = tf.random.normal((1, 4, D_MODEL))

    post = DecoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0, norm_first=False)
    pre = DecoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0, norm_first=True)
    post(x, memory, training=False)  # build
    pre(x, memory, training=False)
    pre.set_weights(post.get_weights())

    out_post = post(x, memory, training=False).numpy()
    out_pre = pre(x, memory, training=False).numpy()
    assert not np.allclose(out_post, out_pre, atol=1e-4)


def test_gradients_flow_through_all_three_sublayers():
    layer = DecoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    x = tf.Variable(tf.random.normal((2, 4, D_MODEL)))
    memory = tf.Variable(tf.random.normal((2, 5, D_MODEL)))
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(layer(x, memory, training=True))
    grads = tape.gradient(loss, [x, memory, *layer.trainable_variables])
    for g in grads:
        assert g is not None
        assert tf.reduce_sum(tf.abs(g)).numpy() > 0
