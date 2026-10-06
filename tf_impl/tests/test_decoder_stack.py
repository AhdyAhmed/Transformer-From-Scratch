import numpy as np
import tensorflow as tf

from tf_impl.model.decoder_stack import Decoder
from tf_impl.model.masking import create_padding_mask, create_target_mask

D_MODEL = 16
NUM_HEADS = 4
D_FF = 32
NUM_LAYERS = 3
PAD = 0


def test_output_shape():
    dec = Decoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    x = tf.random.normal((2, 4, D_MODEL))
    memory = tf.random.normal((2, 7, D_MODEL))
    out = dec(x, memory, training=False)
    assert out.shape == x.shape


def test_layers_have_independent_weights():
    dec = Decoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    dummy = tf.zeros((1, 1, D_MODEL))
    dec(dummy, dummy, training=False)  # build
    w0 = dec.dec_layers[0].self_attn.w_q.kernel
    w1 = dec.dec_layers[1].self_attn.w_q.kernel
    assert w0.ref() != w1.ref()
    assert not np.allclose(w0.numpy(), w1.numpy())


def test_depth_actually_changes_the_output():
    shallow = Decoder(1, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    deep = Decoder(4, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    x = tf.random.normal((1, 4, D_MODEL))
    memory = tf.random.normal((1, 5, D_MODEL))
    out_shallow = shallow(x, memory, training=False).numpy()
    out_deep = deep(x, memory, training=False).numpy()
    assert not np.allclose(out_shallow, out_deep, atol=1e-4)


def test_causal_masking_survives_stacking():
    dec = Decoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    tgt_seq = tf.constant([[1, 2, 3, 4, 5]])
    tgt_mask = create_target_mask(tgt_seq, PAD)

    tf.random.set_seed(0)
    x = tf.random.normal((1, 5, D_MODEL))
    memory = tf.random.normal((1, 6, D_MODEL))
    out_a = dec(x, memory, tgt_mask=tgt_mask, training=False).numpy()

    x_perturbed = x.numpy().copy()
    x_perturbed[:, 3:, :] = np.random.randn(1, 2, D_MODEL).astype(np.float32)
    out_b = dec(tf.constant(x_perturbed), memory, tgt_mask=tgt_mask, training=False).numpy()

    np.testing.assert_allclose(out_a[:, :3], out_b[:, :3], atol=1e-4)


def test_memory_masking_survives_stacking():
    dec = Decoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    src_seq = tf.constant([[1, 2, 3, PAD, PAD]])
    memory_mask = create_padding_mask(src_seq, PAD)

    tf.random.set_seed(0)
    x = tf.random.normal((1, 4, D_MODEL))
    memory = tf.random.normal((1, 5, D_MODEL))
    out_a = dec(x, memory, memory_mask=memory_mask, training=False).numpy()

    memory_perturbed = memory.numpy().copy()
    memory_perturbed[:, 3:, :] = np.random.randn(1, 2, D_MODEL).astype(np.float32)
    out_b = dec(x, tf.constant(memory_perturbed), memory_mask=memory_mask, training=False).numpy()

    np.testing.assert_allclose(out_a, out_b, atol=1e-4)


def test_final_norm_output_is_normalized():
    dec = Decoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    x = tf.random.normal((1, 1, D_MODEL)) * 50
    memory = tf.random.normal((1, 3, D_MODEL))
    out = dec(x, memory, training=False).numpy()
    assert abs(out.mean(axis=-1).item()) < 1e-4
    assert abs(out.std(axis=-1).item() - 1.0) < 1e-2


def test_gradients_flow_through_every_layer():
    dec = Decoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    x = tf.Variable(tf.random.normal((2, 4, D_MODEL)))
    memory = tf.Variable(tf.random.normal((2, 5, D_MODEL)))
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(dec(x, memory, training=True))
    grads = tape.gradient(loss, [x, memory, *dec.trainable_variables])
    for g in grads:
        assert g is not None
        assert tf.reduce_sum(tf.abs(g)).numpy() > 0
