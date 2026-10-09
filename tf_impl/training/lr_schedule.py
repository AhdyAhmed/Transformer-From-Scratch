"""Noam warmup/inverse-square-root schedule (Day 16), for Keras optimizers."""
from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="TransformerFromScratch")
class NoamSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Paper schedule; Keras's zero-based step is shifted to one-based."""

    def __init__(self, d_model: int, warmup_steps: int = 4000, factor: float = 1.0, name: str = "NoamSchedule"):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be positive, got {warmup_steps}")
        if factor <= 0:
            raise ValueError(f"factor must be positive, got {factor}")
        self.d_model = int(d_model)
        self.warmup_steps = int(warmup_steps)
        self.factor = float(factor)
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name):
            step = tf.cast(step, tf.float32) + 1.0
            model_scale = tf.math.rsqrt(tf.cast(self.d_model, tf.float32))
            warmup = tf.cast(self.warmup_steps, tf.float32)
            return self.factor * model_scale * tf.minimum(
                tf.math.rsqrt(step), step * tf.pow(warmup, -1.5)
            )

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps,
                "factor": self.factor, "name": self.name}
