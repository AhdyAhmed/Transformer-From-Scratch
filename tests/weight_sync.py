"""Utilities for loading identical weights into both frameworks' modules so
outputs can be compared for exact numerical parity.

Needed from Day 6 onward, for any module with learned parameters — unlike
the parameter-free checks used for positional encoding and raw attention
(Days 3-4), which could just feed identical inputs and compare.

Only imported from test files that already ``pytest.importorskip`` both
``torch`` and ``tensorflow``, so it's safe for this module to import both
unconditionally; it will simply never be imported in a single-framework
environment.
"""

from __future__ import annotations

import tensorflow as tf
import torch


def copy_linear_to_dense(linear: torch.nn.Linear, dense: tf.keras.layers.Dense) -> None:
    """Copy an ``nn.Linear``'s weights into an already-built Keras ``Dense``.

    PyTorch stores the weight as ``(out_features, in_features)`` and computes
    ``x @ W.T + b``; Keras stores the kernel as ``(in_features, out_features)``
    and computes ``x @ W + b`` — so the weight needs transposing on the way
    across. The bias shape is already identical in both.

    ``dense`` must have been called at least once already (Keras layers build
    their variables lazily on first call), or ``dense.kernel`` won't exist yet.
    """
    weight = linear.weight.detach().cpu().numpy().T  # (in, out)
    bias = linear.bias.detach().cpu().numpy()
    dense.kernel.assign(weight)
    dense.bias.assign(bias)


def copy_multi_head_attention(torch_mha, tf_mha) -> None:
    """Copy every projection (W_Q, W_K, W_V, W_O) from a PyTorch
    ``MultiHeadAttention`` into a same-shaped, already-built TensorFlow one.
    """
    copy_linear_to_dense(torch_mha.w_q, tf_mha.w_q)
    copy_linear_to_dense(torch_mha.w_k, tf_mha.w_k)
    copy_linear_to_dense(torch_mha.w_v, tf_mha.w_v)
    copy_linear_to_dense(torch_mha.w_o, tf_mha.w_o)
