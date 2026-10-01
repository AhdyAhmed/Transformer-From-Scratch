"""Weight-matched cross-framework parity check for MultiHeadAttention.

Unlike the parameter-free Day 3/4 parity tests (positional encoding, raw
scaled dot-product attention), ``MultiHeadAttention`` has real learned
weights — the Q/K/V/output projections — so a fair comparison requires the
*same* weights loaded into both frameworks before comparing outputs. That's
what ``tests/weight_sync.py`` does.

This is the portfolio centerpiece parity test: if it passes, the two
from-scratch implementations are provably doing the same computation, not
just producing superficially similar-looking results.

Runs only if both torch and tensorflow are importable in the current
environment; otherwise skipped.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tf = pytest.importorskip("tensorflow")

from tf_impl.model.masking import create_target_mask as tf_target_mask
from tf_impl.model.multi_head_attention import MultiHeadAttention as TFMHA
from torch_impl.model.masking import create_target_mask as torch_target_mask
from torch_impl.model.multi_head_attention import MultiHeadAttention as TorchMHA

from tests.weight_sync import copy_multi_head_attention

D_MODEL = 16
NUM_HEADS = 4


def _build_weight_synced_pair():
    """A PyTorch and a TensorFlow MultiHeadAttention, identical weights."""
    torch_mha = TorchMHA(D_MODEL, NUM_HEADS, dropout=0.0).eval()

    tf_mha = TFMHA(D_MODEL, NUM_HEADS, dropout=0.0)
    dummy = tf.zeros((1, 1, D_MODEL))
    tf_mha(dummy, dummy, dummy, training=False)  # build so kernel/bias exist

    copy_multi_head_attention(torch_mha, tf_mha)
    return torch_mha, tf_mha


def test_self_attention_matches_across_frameworks():
    torch_mha, tf_mha = _build_weight_synced_pair()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 5, D_MODEL)).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_mha(torch.from_numpy(x), torch.from_numpy(x), torch.from_numpy(x)).numpy()
    tf_out = tf_mha(tf.constant(x), tf.constant(x), tf.constant(x), training=False).numpy()

    np.testing.assert_allclose(torch_out, tf_out, atol=1e-4)
    np.testing.assert_allclose(
        torch_mha.attn_weights.numpy(), tf_mha.attn_weights.numpy(), atol=1e-4
    )


def test_cross_attention_matches_across_frameworks():
    """Different query vs. key/value lengths — the encoder-decoder case."""
    torch_mha, tf_mha = _build_weight_synced_pair()
    rng = np.random.default_rng(1)
    query = rng.normal(size=(2, 3, D_MODEL)).astype(np.float32)
    memory = rng.normal(size=(2, 6, D_MODEL)).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_mha(
            torch.from_numpy(query), torch.from_numpy(memory), torch.from_numpy(memory)
        ).numpy()
    tf_out = tf_mha(
        tf.constant(query), tf.constant(memory), tf.constant(memory), training=False
    ).numpy()

    np.testing.assert_allclose(torch_out, tf_out, atol=1e-4)


def test_masked_self_attention_matches_across_frameworks():
    torch_mha, tf_mha = _build_weight_synced_pair()
    seq_np = np.array([[1, 2, 3, 0, 0]], dtype=np.int64)
    torch_mask = torch_target_mask(torch.from_numpy(seq_np), pad_idx=0)
    tf_mask = tf_target_mask(tf.constant(seq_np, dtype=tf.int32), pad_idx=0)

    rng = np.random.default_rng(2)
    x = rng.normal(size=(1, 5, D_MODEL)).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_mha(
            torch.from_numpy(x), torch.from_numpy(x), torch.from_numpy(x), mask=torch_mask
        ).numpy()
    tf_out = tf_mha(
        tf.constant(x), tf.constant(x), tf.constant(x), mask=tf_mask, training=False
    ).numpy()

    np.testing.assert_allclose(torch_out, tf_out, atol=1e-4)
    # both frameworks should independently agree the masked future is ~0 weight
    np.testing.assert_allclose(torch_mha.attn_weights.numpy()[0, :, 1, 2:], 0.0, atol=1e-5)
    np.testing.assert_allclose(tf_mha.attn_weights.numpy()[0, :, 1, 2:], 0.0, atol=1e-5)


def test_parity_fails_on_purpose_if_weights_are_not_synced():
    """Negative control: without syncing, outputs should NOT match.

    Guards against a trivially-passing parity test (e.g. both frameworks
    collapsing to the same constant regardless of weights).
    """
    torch_mha = TorchMHA(D_MODEL, NUM_HEADS, dropout=0.0).eval()
    tf_mha = TFMHA(D_MODEL, NUM_HEADS, dropout=0.0)

    rng = np.random.default_rng(3)
    x = rng.normal(size=(1, 4, D_MODEL)).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_mha(torch.from_numpy(x), torch.from_numpy(x), torch.from_numpy(x)).numpy()
    tf_out = tf_mha(tf.constant(x), tf.constant(x), tf.constant(x), training=False).numpy()

    assert not np.allclose(torch_out, tf_out, atol=1e-4)
