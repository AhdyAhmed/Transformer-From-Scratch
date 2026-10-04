"""Weight-matched cross-framework parity for the full Encoder stack (Day 10).

Same pattern as every parity test since Day 6: sync every weight (all N
layers' self-attention + FFN + LayerNorms, plus the stack's final
LayerNorm), feed identical input, check the outputs match.

Runs only if both torch and tensorflow are importable in the current
environment; otherwise skipped.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tf = pytest.importorskip("tensorflow")

from tf_impl.model.encoder_stack import Encoder as TFEncoder
from tf_impl.model.masking import create_padding_mask as tf_padding_mask
from torch_impl.model.encoder_stack import Encoder as TorchEncoder
from torch_impl.model.masking import create_padding_mask as torch_padding_mask

from tests.weight_sync import copy_encoder_stack

D_MODEL = 16
NUM_HEADS = 4
D_FF = 32
NUM_LAYERS = 3
PAD = 0


def _build_synced_pair():
    torch_enc = TorchEncoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0).eval()

    tf_enc = TFEncoder(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, dropout=0.0)
    tf_enc(tf.zeros((1, 1, D_MODEL)), training=False)  # build every sub-variable

    copy_encoder_stack(torch_enc, tf_enc)
    return torch_enc, tf_enc


def test_encoder_stack_matches_across_frameworks():
    torch_enc, tf_enc = _build_synced_pair()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 6, D_MODEL)).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_enc(torch.from_numpy(x)).numpy()
    tf_out = tf_enc(tf.constant(x), training=False).numpy()

    np.testing.assert_allclose(torch_out, tf_out, atol=1e-4)


def test_encoder_stack_with_padding_mask_matches_across_frameworks():
    torch_enc, tf_enc = _build_synced_pair()
    seq_np = np.array([[1, 2, 3, PAD, PAD]], dtype=np.int64)
    torch_mask = torch_padding_mask(torch.from_numpy(seq_np), PAD)
    tf_mask = tf_padding_mask(tf.constant(seq_np, dtype=tf.int32), PAD)

    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 5, D_MODEL)).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_enc(torch.from_numpy(x), torch_mask).numpy()
    tf_out = tf_enc(tf.constant(x), mask=tf_mask, training=False).numpy()

    np.testing.assert_allclose(torch_out, tf_out, atol=1e-4)
