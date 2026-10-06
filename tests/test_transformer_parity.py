"""Weight-matched cross-framework parity for the full Transformer model (Day 12).

This is the capstone parity test: every weight (both embeddings, every
encoder and decoder layer, every LayerNorm, the output projection) is
copied from the PyTorch model into the TensorFlow one via
``tests/weight_sync.py``, and the two must then produce identical logits
for identical (src, tgt) input — including with real padding/causal masks
built through each model's own ``create_masks()``. If this test passes, the
two from-scratch implementations are provably the same computation end to
end, not just two codebases that look alike.

Runs only if both torch and tensorflow are importable in the current
environment; otherwise skipped.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tf = pytest.importorskip("tensorflow")

from tf_impl.model.transformer import Transformer as TFTransformer
from torch_impl.model.transformer import Transformer as TorchTransformer

from tests.weight_sync import copy_transformer

SRC_VOCAB = 30
TGT_VOCAB = 35
D_MODEL = 16
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 32
PAD = 0


def _build_synced_pair(share_embeddings: bool = False):
    tgt_vocab = SRC_VOCAB if share_embeddings else TGT_VOCAB

    torch_model = TorchTransformer(
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=tgt_vocab,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=20,
        dropout=0.0,
        pad_idx=PAD,
        share_embeddings=share_embeddings,
    ).eval()

    tf_model = TFTransformer(
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=tgt_vocab,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=20,
        dropout=0.0,
        pad_idx=PAD,
        share_embeddings=share_embeddings,
    )
    dummy = tf.zeros((1, 1), dtype=tf.int32)
    tf_model(dummy, dummy, training=False)  # build every sub-variable

    copy_transformer(torch_model, tf_model)
    return torch_model, tf_model


def test_full_model_matches_across_frameworks_no_masks():
    torch_model, tf_model = _build_synced_pair()
    rng = np.random.default_rng(0)
    src = rng.integers(1, SRC_VOCAB, size=(2, 6)).astype(np.int64)
    tgt = rng.integers(1, TGT_VOCAB, size=(2, 4)).astype(np.int64)

    with torch.no_grad():
        torch_logits = torch_model(torch.from_numpy(src), torch.from_numpy(tgt)).numpy()
    tf_logits = tf_model(
        tf.constant(src, dtype=tf.int32), tf.constant(tgt, dtype=tf.int32), training=False
    ).numpy()

    np.testing.assert_allclose(torch_logits, tf_logits, atol=1e-3)


def test_full_model_matches_across_frameworks_with_real_masks():
    """The realistic case: variable-length, padded batches with masks built
    through each model's own create_masks() convenience method."""
    torch_model, tf_model = _build_synced_pair()

    src_np = np.array([[1, 2, 3, PAD, PAD], [4, 5, 6, 7, PAD]], dtype=np.int64)
    tgt_np = np.array([[1, 2, PAD], [3, 4, 5]], dtype=np.int64)

    src_torch, tgt_torch = torch.from_numpy(src_np), torch.from_numpy(tgt_np)
    torch_masks = torch_model.create_masks(src_torch, tgt_torch)
    with torch.no_grad():
        torch_logits = torch_model(src_torch, tgt_torch, *torch_masks).numpy()

    src_tf = tf.constant(src_np, dtype=tf.int32)
    tgt_tf = tf.constant(tgt_np, dtype=tf.int32)
    tf_src_mask, tf_tgt_mask, tf_mem_mask = tf_model.create_masks(src_tf, tgt_tf)
    tf_logits = tf_model(
        src_tf, tgt_tf, src_mask=tf_src_mask, tgt_mask=tf_tgt_mask, memory_mask=tf_mem_mask, training=False
    ).numpy()

    np.testing.assert_allclose(torch_logits, tf_logits, atol=1e-3)


def test_full_model_matches_across_frameworks_with_shared_embeddings():
    torch_model, tf_model = _build_synced_pair(share_embeddings=True)
    rng = np.random.default_rng(1)
    src = rng.integers(1, SRC_VOCAB, size=(1, 5)).astype(np.int64)
    tgt = rng.integers(1, SRC_VOCAB, size=(1, 4)).astype(np.int64)  # shared vocab

    with torch.no_grad():
        torch_logits = torch_model(torch.from_numpy(src), torch.from_numpy(tgt)).numpy()
    tf_logits = tf_model(
        tf.constant(src, dtype=tf.int32), tf.constant(tgt, dtype=tf.int32), training=False
    ).numpy()

    np.testing.assert_allclose(torch_logits, tf_logits, atol=1e-3)
