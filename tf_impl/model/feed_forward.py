"""Position-wise feed-forward network (TensorFlow) — Section 3.3 of the paper.

    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

Applied identically (same weights) to every position independently — hence
"position-wise". Expands d_model -> d_ff -> d_model, giving the model extra
representational capacity between attention layers.

Mirrors ``torch_impl/model/feed_forward.py``.
"""

from __future__ import annotations

import tensorflow as tf


class PositionwiseFeedForward(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if activation not in ("relu", "gelu"):
            raise ValueError(f"activation must be 'relu' or 'gelu', got {activation!r}")

        self.d_model = d_model
        self.d_ff = d_ff
        self.activation_name = activation

        self.linear1 = tf.keras.layers.Dense(d_ff, activation=activation)
        self.linear2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        """x: (batch, seq_len, d_model) -> same shape.

        Dropout sits between the two linear layers (after the activation),
        matching the paper and the common reference implementations.
        """
        h = self.linear1(x)
        h = self.dropout(h, training=training)
        return self.linear2(h)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(d_model=self.d_model, d_ff=self.d_ff, activation=self.activation_name)
        return cfg
