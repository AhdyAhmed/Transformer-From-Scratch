"""Encoder stack (TensorFlow) — Section 3.1 of the paper: "N = 6" identical layers.

    x = embeddings(src)
    for layer in layers:
        x = layer(x, mask)
    x = final_norm(x)          # see note below
    return x                    # "memory" the decoder will cross-attend to

Mirrors ``torch_impl/model/encoder_stack.py``.
"""

from __future__ import annotations

import tensorflow as tf

from tf_impl.model.encoder import EncoderLayer


class Encoder(tf.keras.layers.Layer):
    """A stack of ``num_layers`` independent ``EncoderLayer``s.

    Each layer gets its own freshly-initialized weights — a fresh
    ``EncoderLayer(...)`` per position in the list, not a Python-level
    reference reused ``num_layers`` times, so the layers don't share
    variables (the straightforward Keras analogue of PyTorch's
    ``copy.deepcopy`` + ``nn.ModuleList`` approach).

    Final LayerNorm: in post-norm mode (the paper's default) every
    ``EncoderLayer`` already ends with a LayerNorm, so this final one is
    technically redundant but harmless, and kept for symmetry with pre-norm
    mode, where the last layer's residual *stream* is never itself
    normalized — only the sublayer inputs are — so a model built with
    ``norm_first=True`` genuinely needs this extra norm before its output is
    used downstream (by the decoder's cross-attention).
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, d_ff, dropout, norm_first) for _ in range(num_layers)
        ]
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x: tf.Tensor, mask: tf.Tensor | None = None, training: bool | None = None) -> tf.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) — already embedded + positionally encoded input.
            mask: optional src padding mask, (batch, 1, 1, seq_len).
            training: passed through to every layer's dropout.

        Returns:
            (batch, seq_len, d_model) — the encoder "memory" passed to every
            decoder layer's cross-attention.
        """
        for layer in self.enc_layers:
            x = layer(x, mask=mask, training=training)
        return self.final_norm(x)
