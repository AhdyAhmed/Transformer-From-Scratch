"""Cross-framework numerical parity for scaled dot-product attention.

Like positional encoding, the raw attention function has no learned
parameters — it's pure math over Q/K/V — so identical NumPy inputs should
produce identical outputs in both frameworks, with no weight-loading needed.
This is a lighter version of the "load matching weights into both
frameworks" parity check the roadmap schedules for Day 6, once multi-head
attention introduces learned projections.

Runs only if both torch and tensorflow are importable in the current
environment; otherwise skipped.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tf = pytest.importorskip("tensorflow")

from tf_impl.model.attention import scaled_dot_product_attention as tf_attention
from tf_impl.model.masking import create_target_mask as tf_target_mask
from torch_impl.model.attention import scaled_dot_product_attention as torch_attention
from torch_impl.model.masking import create_target_mask as torch_target_mask


def test_unmasked_attention_matches_across_frameworks():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(2, 4, 5, 8)).astype(np.float32)
    k = rng.normal(size=(2, 4, 6, 8)).astype(np.float32)
    v = rng.normal(size=(2, 4, 6, 8)).astype(np.float32)

    torch_out, torch_w = torch_attention(torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v))
    tf_out, tf_w = tf_attention(tf.constant(q), tf.constant(k), tf.constant(v))

    np.testing.assert_allclose(torch_out.numpy(), tf_out.numpy(), atol=1e-5)
    np.testing.assert_allclose(torch_w.numpy(), tf_w.numpy(), atol=1e-5)


def test_masked_attention_matches_across_frameworks():
    seq_np = np.array([[1, 2, 3, 0, 0]], dtype=np.int64)
    torch_mask = torch_target_mask(torch.from_numpy(seq_np), pad_idx=0)
    tf_mask = tf_target_mask(tf.constant(seq_np, dtype=tf.int32), pad_idx=0)

    rng = np.random.default_rng(1)
    qkv = rng.normal(size=(1, 1, 5, 8)).astype(np.float32)

    torch_out, torch_w = torch_attention(
        torch.from_numpy(qkv), torch.from_numpy(qkv), torch.from_numpy(qkv), mask=torch_mask
    )
    tf_out, tf_w = tf_attention(tf.constant(qkv), tf.constant(qkv), tf.constant(qkv), mask=tf_mask)

    np.testing.assert_allclose(torch_out.numpy(), tf_out.numpy(), atol=1e-5)
    np.testing.assert_allclose(torch_w.numpy(), tf_w.numpy(), atol=1e-5)
