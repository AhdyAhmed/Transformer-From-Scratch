import numpy as np
import tensorflow as tf

from tf_impl.model.encoder_stack import Encoder
from tf_impl.model.masking import create_padding_mask

D_MODEL = 16
NUM_HEADS = 4
D_FF = 32
NUM_LAYERS = 3
PAD = 0


def test_output_shape():
    enc = Encoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    x = tf.random.normal((2, 7, D_MODEL))
    out = enc(x, training=False)
    assert out.shape == x.shape


def test_output_shape_with_mask():
    enc = Encoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    seq = tf.constant([[1, 2, 3, PAD, PAD]])
    mask = create_padding_mask(seq, PAD)
    x = tf.random.normal((1, 5, D_MODEL))
    out = enc(x, mask=mask, training=False)
    assert out.shape == x.shape


def test_layers_have_independent_weights():
    """Six layers sharing one set of weights would collapse the stack's
    capacity to a single layer applied six times — guard against that."""
    enc = Encoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    enc(tf.zeros((1, 1, D_MODEL)), training=False)  # build
    w0 = enc.enc_layers[0].self_attn.w_q.kernel
    w1 = enc.enc_layers[1].self_attn.w_q.kernel
    assert w0.ref() != w1.ref()  # distinct variable objects
    assert not np.allclose(w0.numpy(), w1.numpy())  # and not coincidentally equal (random init)


def test_depth_actually_changes_the_output():
    """A deeper stack (with different weights per layer) should produce a
    different output than a shallow one — i.e. the extra layers aren't
    silently no-ops."""
    shallow = Encoder(1, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    deep = Encoder(4, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    x = tf.random.normal((1, 5, D_MODEL))
    out_shallow = shallow(x, training=False).numpy()
    out_deep = deep(x, training=False).numpy()
    assert not np.allclose(out_shallow, out_deep, atol=1e-4)


def test_padding_positions_dont_influence_real_positions():
    """The padding-mask-invariance property (verified per-layer on Day 9)
    must still hold once layers are stacked N deep."""
    enc = Encoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    seq = tf.constant([[1, 2, 3, PAD, PAD]])
    mask = create_padding_mask(seq, PAD)

    tf.random.set_seed(0)
    x = tf.random.normal((1, 5, D_MODEL))
    out_a = enc(x, mask=mask, training=False).numpy()

    x_perturbed = x.numpy().copy()
    x_perturbed[:, 3:, :] = np.random.randn(1, 2, D_MODEL).astype(np.float32)
    out_b = enc(tf.constant(x_perturbed), mask=mask, training=False).numpy()

    np.testing.assert_allclose(out_a[:, :3], out_b[:, :3], atol=1e-4)


def test_final_norm_output_is_normalized():
    enc = Encoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    x = tf.random.normal((1, 1, D_MODEL)) * 50
    out = enc(x, training=False).numpy()
    assert abs(out.mean(axis=-1).item()) < 1e-4
    assert abs(out.std(axis=-1).item() - 1.0) < 1e-2


def test_gradients_flow_through_every_layer():
    enc = Encoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    x = tf.Variable(tf.random.normal((2, 5, D_MODEL)))
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(enc(x, training=True))
    grads = tape.gradient(loss, [x, *enc.trainable_variables])
    for g in grads:
        assert g is not None
        assert tf.reduce_sum(tf.abs(g)).numpy() > 0


def test_num_layers_matches_config():
    for n in (1, 2, 6):
        enc = Encoder(n, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
        assert len(enc.enc_layers) == n
