import math

import numpy as np
import pytest
import tensorflow as tf

from tf_impl.model.embeddings import (
    PositionalEncoding,
    TokenEmbedding,
    TransformerEmbedding,
)

D_MODEL = 16
VOCAB = 50
PAD = 0


def test_token_embedding_output_shape():
    emb = TokenEmbedding(VOCAB, D_MODEL, PAD)
    x = tf.random.uniform((4, 10), minval=1, maxval=VOCAB, dtype=tf.int32)
    out = emb(x)
    assert out.shape == (4, 10, D_MODEL)


def test_token_embedding_scaling_factor():
    emb = TokenEmbedding(VOCAB, D_MODEL, PAD)
    x = tf.random.uniform((2, 5), minval=1, maxval=VOCAB, dtype=tf.int32)
    scaled = emb(x).numpy()
    raw = emb.embedding(x).numpy()
    nonzero = raw != 0
    ratio = scaled[nonzero] / raw[nonzero]
    np.testing.assert_allclose(ratio, math.sqrt(D_MODEL), atol=1e-5)


def test_positional_encoding_rejects_odd_d_model():
    with pytest.raises(ValueError, match="even"):
        PositionalEncoding(d_model=15)


def test_positional_encoding_shape():
    pe = PositionalEncoding(D_MODEL, max_len=20, dropout=0.0)
    x = tf.zeros((3, 7, D_MODEL))
    out = pe(x, training=False)
    assert out.shape == (3, 7, D_MODEL)


def test_positional_encoding_position_zero_values():
    """At position 0: sin(0)=0 on even dims, cos(0)=1 on odd dims."""
    pe = PositionalEncoding(D_MODEL, max_len=20, dropout=0.0)
    pos0 = pe.pe.numpy()[0, 0]
    np.testing.assert_allclose(pos0[0::2], np.zeros(D_MODEL // 2), atol=1e-6)
    np.testing.assert_allclose(pos0[1::2], np.ones(D_MODEL // 2), atol=1e-6)


def test_positional_encoding_values_are_bounded():
    pe = PositionalEncoding(D_MODEL, max_len=100, dropout=0.0)
    assert np.abs(pe.pe.numpy()).max() <= 1.0 + 1e-6


def test_positional_encoding_is_added_not_replaced():
    pe = PositionalEncoding(D_MODEL, max_len=20, dropout=0.0)
    x = tf.ones((1, 5, D_MODEL))
    out = pe(x, training=False).numpy()
    expected = x.numpy() + pe.pe.numpy()[:, :5]
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_positional_encoding_rejects_too_long_sequence():
    pe = PositionalEncoding(D_MODEL, max_len=8, dropout=0.0)
    x = tf.zeros((1, 9, D_MODEL))
    with pytest.raises(ValueError, match="exceeds max_len"):
        pe(x, training=False)


def test_positional_encoding_has_no_trainable_weights():
    pe = PositionalEncoding(D_MODEL, max_len=20)
    assert pe.trainable_variables == []  # fixed function of position, not learned


def test_transformer_embedding_end_to_end():
    emb = TransformerEmbedding(VOCAB, D_MODEL, max_len=20, dropout=0.0, pad_idx=PAD)
    x = tf.random.uniform((2, 6), minval=1, maxval=VOCAB, dtype=tf.int32)
    out = emb(x, training=False)
    assert out.shape == (2, 6, D_MODEL)
