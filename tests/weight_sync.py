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


def copy_feed_forward(torch_ffn, tf_ffn) -> None:
    """Copy both linear layers of a PyTorch ``PositionwiseFeedForward`` into
    an already-built TensorFlow one."""
    copy_linear_to_dense(torch_ffn.linear1, tf_ffn.linear1)
    copy_linear_to_dense(torch_ffn.linear2, tf_ffn.linear2)


def copy_layer_norm(torch_ln: torch.nn.LayerNorm, tf_ln: tf.keras.layers.LayerNormalization) -> None:
    """Copy an ``nn.LayerNorm``'s gamma/beta into an already-built Keras
    ``LayerNormalization``. Both store gamma (scale) and beta (shift) as a
    plain ``(d_model,)`` vector each, so no transpose is needed here —
    unlike the Linear/Dense weight matrices."""
    gamma = torch_ln.weight.detach().cpu().numpy()
    beta = torch_ln.bias.detach().cpu().numpy()
    tf_ln.gamma.assign(gamma)
    tf_ln.beta.assign(beta)


def copy_residual_connection(torch_res, tf_res) -> None:
    """Copy a PyTorch ``ResidualConnection``'s LayerNorm into an
    already-built TensorFlow one. The residual add/dropout logic has no
    weights of its own — only the wrapped LayerNorm does."""
    copy_layer_norm(torch_res.norm, tf_res.norm)


def copy_encoder_layer(torch_layer, tf_layer) -> None:
    """Copy every weight of a PyTorch ``EncoderLayer`` (self-attention,
    feed-forward, and both residual blocks' LayerNorms) into an
    already-built TensorFlow one with the same architecture."""
    copy_multi_head_attention(torch_layer.self_attn, tf_layer.self_attn)
    copy_feed_forward(torch_layer.feed_forward, tf_layer.feed_forward)
    copy_residual_connection(torch_layer.self_attn_block, tf_layer.self_attn_block)
    copy_residual_connection(torch_layer.feed_forward_block, tf_layer.feed_forward_block)


def copy_decoder_layer(torch_layer, tf_layer) -> None:
    """Copy every weight of a PyTorch ``DecoderLayer`` (self-attention,
    cross-attention, feed-forward, and all three residual blocks'
    LayerNorms) into an already-built TensorFlow one with the same
    architecture."""
    copy_multi_head_attention(torch_layer.self_attn, tf_layer.self_attn)
    copy_multi_head_attention(torch_layer.cross_attn, tf_layer.cross_attn)
    copy_feed_forward(torch_layer.feed_forward, tf_layer.feed_forward)
    copy_residual_connection(torch_layer.self_attn_block, tf_layer.self_attn_block)
    copy_residual_connection(torch_layer.cross_attn_block, tf_layer.cross_attn_block)
    copy_residual_connection(torch_layer.feed_forward_block, tf_layer.feed_forward_block)


def copy_encoder_stack(torch_encoder, tf_encoder) -> None:
    """Copy every layer of a PyTorch ``Encoder`` (Day 10) into an
    already-built TensorFlow one with the same number of layers, plus the
    stack's own final LayerNorm."""
    for torch_layer, tf_layer in zip(torch_encoder.layers, tf_encoder.enc_layers):
        copy_encoder_layer(torch_layer, tf_layer)
    copy_layer_norm(torch_encoder.final_norm, tf_encoder.final_norm)
