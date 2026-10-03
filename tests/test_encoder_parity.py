"""Weight-matched cross-framework parity for EncoderLayer.

Same pattern as ``test_mha_parity.py`` (Day 6) and
``test_ffn_residual_parity.py`` (Day 8), extended to the full encoder block:
sync every weight (self-attention projections, FFN, both LayerNorms), feed
identical input, and check the outputs match — with and without a padding
mask, and in both pre-norm and post-norm mode.

Runs only if both torch and tensorflow are importable in the current
environment; otherwise skipped.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tf = pytest.importorskip("tensorflow")

from tf_impl.model.encoder import EncoderLayer as TFEncoderLayer
from tf_impl.model.masking import create_padding_mask as tf_padding_mask
from torch_impl.model.encoder import EncoderLayer as TorchEncoderLayer
from torch_impl.model.masking import create_padding_mask as torch_padding_mask

from tests.weight_sync import copy_encoder_layer

D_MODEL = 16
NUM_HEADS = 4
D_FF = 32
PAD = 0


def _build_synced_pair(norm_first: bool):
    torch_layer = TorchEncoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0, norm_first=norm_first).eval()

    tf_layer = TFEncoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0, norm_first=norm_first)
    tf_layer(tf.zeros((1, 1, D_MODEL)), training=False)  # build every sub-variable

    copy_encoder_layer(torch_layer, tf_layer)
    return torch_layer, tf_layer


@pytest.mark.parametrize("norm_first", [False, True])
def test_encoder_layer_matches_across_frameworks(norm_first):
    torch_layer, tf_layer = _build_synced_pair(norm_first)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 6, D_MODEL)).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_layer(torch.from_numpy(x)).numpy()
    tf_out = tf_layer(tf.constant(x), training=False).numpy()

    np.testing.assert_allclose(torch_out, tf_out, atol=1e-4)


def test_encoder_layer_with_padding_mask_matches_across_frameworks():
    torch_layer, tf_layer = _build_synced_pair(norm_first=False)
    seq_np = np.array([[1, 2, 3, PAD, PAD]], dtype=np.int64)
    torch_mask = torch_padding_mask(torch.from_numpy(seq_np), PAD)
    tf_mask = tf_padding_mask(tf.constant(seq_np, dtype=tf.int32), PAD)

    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 5, D_MODEL)).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_layer(torch.from_numpy(x), torch_mask).numpy()
    tf_out = tf_layer(tf.constant(x), mask=tf_mask, training=False).numpy()

    np.testing.assert_allclose(torch_out, tf_out, atol=1e-4)
