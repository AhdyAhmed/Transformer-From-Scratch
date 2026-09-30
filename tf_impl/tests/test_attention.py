import numpy as np
import tensorflow as tf

from tf_impl.model.attention import scaled_dot_product_attention
from tf_impl.model.masking import create_target_mask

D_K = 8


def test_output_and_weight_shapes():
    q = tf.random.normal((2, 4, 5, D_K))
    k = tf.random.normal((2, 4, 6, D_K))
    v = tf.random.normal((2, 4, 6, D_K))
    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 4, 5, D_K)
    assert weights.shape == (2, 4, 5, 6)


def test_weights_sum_to_one():
    q = tf.random.normal((2, 4, 5, D_K))
    k = tf.random.normal((2, 4, 6, D_K))
    v = tf.random.normal((2, 4, 6, D_K))
    _, weights = scaled_dot_product_attention(q, k, v)
    np.testing.assert_allclose(tf.reduce_sum(weights, axis=-1).numpy(),
                                np.ones((2, 4, 5)), atol=1e-5)


def test_matches_manual_computation():
    tf.random.set_seed(0)
    q = tf.random.normal((1, 1, 3, D_K))
    k = tf.random.normal((1, 1, 3, D_K))
    v = tf.random.normal((1, 1, 3, D_K))

    expected_scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(float(D_K))
    expected_weights = tf.nn.softmax(expected_scores, axis=-1)
    expected_out = tf.matmul(expected_weights, v)

    out, weights = scaled_dot_product_attention(q, k, v)
    np.testing.assert_allclose(out.numpy(), expected_out.numpy(), atol=1e-6)
    np.testing.assert_allclose(weights.numpy(), expected_weights.numpy(), atol=1e-6)


def test_masked_positions_get_zero_weight():
    seq = tf.constant([[1, 2, 3, 0, 0]])
    mask = create_target_mask(seq, pad_idx=0)
    q = k = v = tf.random.normal((1, 1, 5, D_K))
    _, weights = scaled_dot_product_attention(q, k, v, mask=mask)
    w = weights.numpy()
    assert w[0, 0, 0, 0] > 0.999
    assert np.abs(w[0, 0, 0, 1:]).max() < 1e-5
    assert np.abs(w[0, 0, :3, 3:]).max() < 1e-5


def test_uniform_scores_give_uniform_weights():
    q = tf.zeros((1, 1, 1, D_K))
    k = tf.zeros((1, 1, 4, D_K))
    v = tf.random.normal((1, 1, 4, D_K))
    out, weights = scaled_dot_product_attention(q, k, v)
    np.testing.assert_allclose(weights.numpy(), np.full((1, 1, 1, 4), 0.25), atol=1e-6)
    np.testing.assert_allclose(out.numpy()[0, 0, 0],
                                tf.reduce_mean(v, axis=2).numpy()[0, 0], atol=1e-5)


def test_dropout_zeroes_output_when_p_is_one_but_weights_unaffected():
    tf.random.set_seed(0)
    q = tf.random.normal((1, 1, 3, D_K))
    k = tf.random.normal((1, 1, 3, D_K))
    v = tf.random.normal((1, 1, 3, D_K))
    dropout = tf.keras.layers.Dropout(rate=1.0)
    out, weights = scaled_dot_product_attention(q, k, v, dropout=dropout, training=True)
    np.testing.assert_allclose(tf.reduce_sum(weights, axis=-1).numpy(),
                                np.ones((1, 1, 3)), atol=1e-5)
    np.testing.assert_allclose(out.numpy(), np.zeros_like(out.numpy()))


def test_gradients_flow_to_query_key_value():
    q = tf.Variable(tf.random.normal((1, 1, 3, D_K)))
    k = tf.Variable(tf.random.normal((1, 1, 3, D_K)))
    v = tf.Variable(tf.random.normal((1, 1, 3, D_K)))
    with tf.GradientTape() as tape:
        out, _ = scaled_dot_product_attention(q, k, v)
        loss = tf.reduce_sum(out)
    grads = tape.gradient(loss, [q, k, v])
    for g in grads:
        assert g is not None
        assert tf.reduce_sum(tf.abs(g)).numpy() > 0


def test_works_inside_tf_function():
    @tf.function
    def run(q, k, v):
        out, _ = scaled_dot_product_attention(q, k, v)
        return out

    q = tf.random.normal((1, 2, 3, D_K))
    out = run(q, q, q)
    assert out.shape == (1, 2, 3, D_K)
