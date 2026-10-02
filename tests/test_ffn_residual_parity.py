"""Weight-matched cross-framework parity for PositionwiseFeedForward and
ResidualConnection — the same approach as ``test_mha_parity.py`` (Day 6),
applied to the two modules built on Day 8.

Runs only if both torch and tensorflow are importable in the current
environment; otherwise skipped.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tf = pytest.importorskip("tensorflow")

from tf_impl.model.feed_forward import PositionwiseFeedForward as TFFeedForward
from tf_impl.model.residual import ResidualConnection as TFResidual
from torch_impl.model.feed_forward import PositionwiseFeedForward as TorchFeedForward
from torch_impl.model.residual import ResidualConnection as TorchResidual

from tests.weight_sync import copy_feed_forward, copy_residual_connection

D_MODEL = 16
D_FF = 32


def test_feed_forward_matches_across_frameworks():
    torch_ffn = TorchFeedForward(D_MODEL, D_FF, dropout=0.0).eval()

    tf_ffn = TFFeedForward(D_MODEL, D_FF, dropout=0.0)
    tf_ffn(tf.zeros((1, 1, D_MODEL)))  # build

    copy_feed_forward(torch_ffn, tf_ffn)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 5, D_MODEL)).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_ffn(torch.from_numpy(x)).numpy()
    tf_out = tf_ffn(tf.constant(x), training=False).numpy()

    np.testing.assert_allclose(torch_out, tf_out, atol=1e-4)


@pytest.mark.parametrize("norm_first", [False, True])
def test_residual_connection_matches_across_frameworks(norm_first):
    torch_res = TorchResidual(D_MODEL, dropout=0.0, norm_first=norm_first).eval()

    tf_res = TFResidual(D_MODEL, dropout=0.0, norm_first=norm_first)
    tf_res(tf.zeros((1, 1, D_MODEL)), lambda t: t, training=False)  # build

    copy_residual_connection(torch_res, tf_res)

    rng = np.random.default_rng(1)
    x_np = rng.normal(size=(2, 4, D_MODEL)).astype(np.float32)
    x_torch = torch.from_numpy(x_np)
    x_tf = tf.constant(x_np)

    # identical, deterministic, parameter-free sublayer in both frameworks
    torch_sublayer = lambda t: t * 2.0 + 1.0
    tf_sublayer = lambda t: t * 2.0 + 1.0

    with torch.no_grad():
        torch_out = torch_res(x_torch, torch_sublayer).numpy()
    tf_out = tf_res(x_tf, tf_sublayer, training=False).numpy()

    np.testing.assert_allclose(torch_out, tf_out, atol=1e-4)
