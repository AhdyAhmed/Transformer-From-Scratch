"""Day 16 tests for TensorFlow's Noam schedule and label-smoothed loss."""
import math
import numpy as np
import pytest
import tensorflow as tf

from tf_impl.training.lr_schedule import NoamSchedule
from tf_impl.training.losses import LabelSmoothingLoss


def test_noam_schedule_matches_paper_formula_and_warmup_peak():
    schedule = NoamSchedule(d_model=512, warmup_steps=4000)
    expected_first = 512 ** -0.5 * min(1.0, 4000 ** -1.5)
    assert float(schedule(0).numpy()) == pytest.approx(expected_first, rel=1e-6)
    peak = float(schedule(3999).numpy())
    assert float(schedule(7999).numpy()) == pytest.approx(peak / math.sqrt(2), rel=1e-5)
    assert float(schedule(4000).numpy()) < peak


def test_noam_schedule_validation_and_serialization():
    with pytest.raises(ValueError):
        NoamSchedule(0)
    with pytest.raises(ValueError):
        NoamSchedule(32, 0)
    schedule = NoamSchedule(32, 10, factor=2.0)
    restored = NoamSchedule.from_config(schedule.get_config())
    assert float(restored(4).numpy()) == pytest.approx(float(schedule(4).numpy()))


def test_label_smoothing_zero_matches_sparse_cross_entropy():
    logits = tf.constant(np.random.default_rng(4).normal(size=(2, 3, 7)), tf.float32)
    targets = tf.constant([[1, 2, 3], [4, 5, 6]], tf.int32)
    actual = LabelSmoothingLoss(0.0, pad_idx=0)(targets, logits)
    expected = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True))
    assert float(actual.numpy()) == pytest.approx(float(expected.numpy()), rel=1e-6)


def test_label_smoothing_ignores_padding_and_has_finite_gradients():
    logits = tf.Variable(np.random.default_rng(5).normal(size=(2, 4, 6)), dtype=tf.float32)
    targets = tf.constant([[1, 2, 0, 0], [3, 4, 5, 0]], tf.int32)
    criterion = LabelSmoothingLoss(0.1, pad_idx=0)
    with tf.GradientTape() as tape:
        loss = criterion(targets, logits)
    grads = tape.gradient(loss, logits)
    valid_logits = tf.boolean_mask(logits, targets != 0)
    valid_targets = tf.boolean_mask(targets, targets != 0)
    log_probs = tf.nn.log_softmax(valid_logits, axis=-1)
    nll = -tf.gather(log_probs, valid_targets, axis=1, batch_dims=1)
    expected = tf.reduce_mean(0.9 * nll + 0.1 * -tf.reduce_mean(log_probs, axis=-1))
    assert float(loss.numpy()) == pytest.approx(float(expected.numpy()), rel=1e-6)
    assert bool(tf.reduce_all(tf.math.is_finite(grads)).numpy())
    assert np.allclose(tf.boolean_mask(grads, targets == 0).numpy(), 0.0)


def test_label_smoothing_all_padding_and_reductions():
    logits = tf.Variable(tf.ones((2, 3, 5)))
    targets = tf.zeros((2, 3), dtype=tf.int32)
    assert float(LabelSmoothingLoss(0.1, 0)(targets, logits).numpy()) == 0.0
    assert tuple(LabelSmoothingLoss(0.1, 0, "none")(targets, logits).shape) == (2, 3)
    assert float(LabelSmoothingLoss(0.1, 0, "sum")(targets, logits).numpy()) == 0.0
    with pytest.raises(ValueError):
        LabelSmoothingLoss(1.0)
    with pytest.raises(ValueError):
        LabelSmoothingLoss(0.1, reduction="bad")
