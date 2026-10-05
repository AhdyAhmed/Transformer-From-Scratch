"""Weight-matched cross-framework parity for DecoderLayer.

Same pattern as every parity test since Day 6: sync every weight
(self-attention, cross-attention, FFN, all three LayerNorms), feed
identical decoder input + encoder memory, check the outputs match — with
and without masks, and in both pre-norm and post-norm mode.

Runs only if both torch and tensorflow are importable in the current
environment; otherwise skipped.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tf = pytest.importorskip("tensorflow")

from tf_impl.model.decoder import DecoderLayer as TFDecoderLayer
from tf_impl.model.masking import create_padding_mask as tf_padding_mask
from tf_impl.model.masking import create_target_mask as tf_target_mask
from torch_impl.model.decoder import DecoderLayer as TorchDecoderLayer
from torch_impl.model.masking import create_padding_mask as torch_padding_mask
from torch_impl.model.masking import create_target_mask as torch_target_mask

from tests.weight_sync import copy_decoder_layer

D_MODEL = 16
NUM_HEADS = 4
D_FF = 32
PAD = 0


def _build_synced_pair(norm_first: bool):
    torch_layer = TorchDecoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0, norm_first=norm_first).eval()

    tf_layer = TFDecoderLayer(D_MODEL, NUM_HEADS, D_FF, dropout=0.0, norm_first=norm_first)
    dummy = tf.zeros((1, 1, D_MODEL))
    tf_layer(dummy, dummy, training=False)  # build every sub-variable

    copy_decoder_layer(torch_layer, tf_layer)
    return torch_layer, tf_layer


@pytest.mark.parametrize("norm_first", [False, True])
def test_decoder_layer_matches_across_frameworks(norm_first):
    torch_layer, tf_layer = _build_synced_pair(norm_first)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 4, D_MODEL)).astype(np.float32)
    memory = rng.normal(size=(2, 6, D_MODEL)).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_layer(torch.from_numpy(x), torch.from_numpy(memory)).numpy()
    tf_out = tf_layer(tf.constant(x), tf.constant(memory), training=False).numpy()

    np.testing.assert_allclose(torch_out, tf_out, atol=1e-4)


def test_decoder_layer_with_masks_matches_across_frameworks():
    torch_layer, tf_layer = _build_synced_pair(norm_first=False)

    tgt_np = np.array([[1, 2, 3, PAD]], dtype=np.int64)
    src_np = np.array([[4, 5, 6, 7, PAD, PAD]], dtype=np.int64)
    torch_tgt_mask = torch_target_mask(torch.from_numpy(tgt_np), PAD)
    torch_mem_mask = torch_padding_mask(torch.from_numpy(src_np), PAD)
    tf_tgt_mask = tf_target_mask(tf.constant(tgt_np, dtype=tf.int32), PAD)
    tf_mem_mask = tf_padding_mask(tf.constant(src_np, dtype=tf.int32), PAD)

    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 4, D_MODEL)).astype(np.float32)
    memory = rng.normal(size=(1, 6, D_MODEL)).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_layer(
            torch.from_numpy(x), torch.from_numpy(memory), torch_tgt_mask, torch_mem_mask
        ).numpy()
    tf_out = tf_layer(
        tf.constant(x), tf.constant(memory), tgt_mask=tf_tgt_mask, memory_mask=tf_mem_mask, training=False
    ).numpy()

    np.testing.assert_allclose(torch_out, tf_out, atol=1e-4)
