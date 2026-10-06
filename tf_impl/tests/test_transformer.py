import numpy as np
import pytest
import tensorflow as tf

from tf_impl.config import TransformerConfig
from tf_impl.model.transformer import Transformer

SRC_VOCAB = 50
TGT_VOCAB = 60
D_MODEL = 16
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 32
PAD = 0


def make_model(**overrides):
    kwargs = dict(
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=20,
        dropout=0.0,
        pad_idx=PAD,
    )
    kwargs.update(overrides)
    return Transformer(**kwargs)


def test_output_logits_shape():
    model = make_model()
    src = tf.random.uniform((2, 7), minval=1, maxval=SRC_VOCAB, dtype=tf.int32)
    tgt = tf.random.uniform((2, 5), minval=1, maxval=TGT_VOCAB, dtype=tf.int32)
    logits = model(src, tgt, training=False)
    assert logits.shape == (2, 5, TGT_VOCAB)


def test_create_masks_convenience_matches_module_level_function():
    from tf_impl.model.masking import create_masks as raw_create_masks

    model = make_model()
    src = tf.constant([[1, 2, 3, PAD]])
    tgt = tf.constant([[4, 5, PAD]])
    src_mask, tgt_mask, memory_mask = model.create_masks(src, tgt)
    expected_src, expected_tgt, expected_mem = raw_create_masks(src, tgt, PAD)
    np.testing.assert_array_equal(src_mask.numpy(), expected_src.numpy())
    np.testing.assert_array_equal(tgt_mask.numpy(), expected_tgt.numpy())
    np.testing.assert_array_equal(memory_mask.numpy(), expected_mem.numpy())


def test_forward_with_masks_runs_end_to_end():
    model = make_model()
    src = tf.constant([[1, 2, 3, PAD, PAD], [4, 5, 6, 7, PAD]])
    tgt = tf.constant([[1, 2, PAD], [3, 4, 5]])
    src_mask, tgt_mask, memory_mask = model.create_masks(src, tgt)
    logits = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask, training=False)
    assert logits.shape == (2, 3, TGT_VOCAB)


def test_encode_decode_match_full_forward():
    """encode()+decode()+output_projection, called manually, must match
    calling the model directly — call() is just a convenience composition."""
    model = make_model()
    src = tf.random.uniform((1, 5), minval=1, maxval=SRC_VOCAB, dtype=tf.int32)
    tgt = tf.random.uniform((1, 4), minval=1, maxval=TGT_VOCAB, dtype=tf.int32)

    full = model(src, tgt, training=False).numpy()
    memory = model.encode(src, training=False)
    manual = model.output_projection(model.decode(tgt, memory, training=False)).numpy()
    np.testing.assert_allclose(full, manual, atol=1e-5)


def test_from_config_builds_matching_architecture():
    config = TransformerConfig(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        dropout=0.0,
        max_seq_len=20,
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        pad_idx=PAD,
    )
    model = Transformer.from_config(config)
    dummy_src = tf.zeros((1, 1), dtype=tf.int32)
    dummy_tgt = tf.zeros((1, 1), dtype=tf.int32)
    model(dummy_src, dummy_tgt, training=False)  # build
    assert len(model.encoder.enc_layers) == NUM_LAYERS
    assert len(model.decoder.dec_layers) == NUM_LAYERS
    assert model.output_projection.units == TGT_VOCAB


def test_share_embeddings_reuses_the_same_layer():
    model = make_model(src_vocab_size=SRC_VOCAB, tgt_vocab_size=SRC_VOCAB, share_embeddings=True)
    assert model.src_embed is model.tgt_embed


def test_share_embeddings_rejects_mismatched_vocab_sizes():
    with pytest.raises(ValueError, match="share_embeddings"):
        make_model(src_vocab_size=SRC_VOCAB, tgt_vocab_size=TGT_VOCAB, share_embeddings=True)


def test_gradients_flow_from_logits_to_embeddings():
    model = make_model()
    src = tf.random.uniform((2, 5), minval=1, maxval=SRC_VOCAB, dtype=tf.int32)
    tgt = tf.random.uniform((2, 4), minval=1, maxval=TGT_VOCAB, dtype=tf.int32)
    with tf.GradientTape() as tape:
        logits = model(src, tgt, training=True)
        loss = tf.reduce_sum(logits)
    grads = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None for g in grads)
    assert any(tf.reduce_sum(tf.abs(g)).numpy() > 0 for g in grads)
