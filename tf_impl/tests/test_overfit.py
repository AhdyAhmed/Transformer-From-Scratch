"""Overfit sanity check (TensorFlow) — Day 13.

The single most important test in this repo. Everything through Day 12 only
checked that shapes, masks, and cross-framework numerics line up — none of
it proves gradients actually flow correctly end to end through a real
training loop. This test trains a tiny Transformer on a handful of fixed
(src, tgt) pairs from a trivial copy task and requires it to memorize them.

If gradients are broken anywhere in the 12-day chain of composed modules —
attention, residuals, embeddings, the output projection — this is where
it would show up as a loss that refuses to go down.

No tokenizer or real dataset yet (those are Day 14-15); this uses random
integer sequences directly, with the model's own ``pad_idx``/masking
utilities, which is all a pure architecture sanity check needs.

Marked ``slow`` (trains for several hundred steps): run with
``pytest -m "not slow"`` to skip it for a quick iteration loop, or
``make test-tf`` (no filter) to include it.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from tf_impl.model.masking import create_target_mask
from tf_impl.model.transformer import Transformer

VOCAB_SIZE = 20
PAD, BOS, EOS = 0, 1, 2
D_MODEL = 32
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 64
SEQ_LEN = 8  # [BOS] + 6 content tokens + [EOS]
NUM_EXAMPLES = 12
MAX_STEPS = 600
LR = 1e-3


def _make_copy_task_batch():
    """A handful of fixed random sequences; src == tgt (copy task).

    No padding token is used anywhere in these sequences on purpose, so
    this test exercises causal masking without also depending on padding
    masking being correct — that part is already covered extensively by
    Days 2-12's own test suites.
    """
    rng = np.random.default_rng(42)
    content = rng.integers(3, VOCAB_SIZE, size=(NUM_EXAMPLES, SEQ_LEN - 2))
    bos = np.full((NUM_EXAMPLES, 1), BOS)
    eos = np.full((NUM_EXAMPLES, 1), EOS)
    seq = np.concatenate([bos, content, eos], axis=1).astype(np.int32)
    return tf.constant(seq), tf.constant(seq)


@pytest.mark.slow
def test_overfits_tiny_copy_task():
    tf.random.set_seed(0)
    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=SEQ_LEN,
        dropout=0.0,  # memorization, not generalization — dropout would only add noise
        pad_idx=PAD,
        share_embeddings=True,  # one vocab, since src and tgt are literally the same sequence
    )

    src, tgt = _make_copy_task_batch()
    decoder_input = tgt[:, :-1]
    labels = tgt[:, 1:]
    tgt_mask = create_target_mask(decoder_input, PAD)  # pure causal here (no padding present)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    losses = []
    for _ in range(MAX_STEPS):
        with tf.GradientTape() as tape:
            logits = model(src, decoder_input, tgt_mask=tgt_mask, training=True)
            loss = loss_fn(labels, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        losses.append(float(loss.numpy()))

    initial_loss, final_loss = losses[0], losses[-1]

    # sanity: an untrained model on 20-way classification should start near
    # ln(20) ~= 3.0 nats; a wildly different starting point would suggest
    # something upstream (init, loss wiring) is already off.
    assert 1.0 < initial_loss < 6.0, f"unexpected initial loss {initial_loss:.3f} (expected ~ln(20)=3.0)"

    assert final_loss < initial_loss * 0.2, (
        f"loss only dropped from {initial_loss:.3f} to {final_loss:.3f} over {MAX_STEPS} steps — "
        "likely a broken gradient somewhere in the model. If this is a false alarm on a slower "
        "machine, try increasing MAX_STEPS."
    )

    logits = model(src, decoder_input, tgt_mask=tgt_mask, training=False)
    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32)).numpy()
    assert accuracy > 0.9, f"token-level accuracy {accuracy:.2%} too low after {MAX_STEPS} overfit steps"
