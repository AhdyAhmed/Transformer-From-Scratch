"""Day 18 tests for the TensorFlow training loop and checkpoint-compatible state."""
import numpy as np
import tensorflow as tf

from data.bpe_tokenizer import BPETokenizer
from data.parallel_data import ParallelDataset
from tf_impl.config import TransformerConfig
from tf_impl.data.dataset import TFBatchLoader
from tf_impl.model.transformer import Transformer
from tf_impl.training.losses import LabelSmoothingLoss
from tf_impl.training.lr_schedule import NoamSchedule
from tf_impl.training.train import run_epoch


def _fixture():
    tokenizer = BPETokenizer()
    tokenizer.train(["a man runs", "a woman walks", "ein mann rennt", "eine frau geht"], 48)
    pairs = [("a man runs", "ein mann rennt"), ("a woman walks", "eine frau geht")]
    dataset = ParallelDataset(pairs, tokenizer, max_len=12)
    loader = TFBatchLoader(dataset, batch_size=2, shuffle=False)
    config = TransformerConfig(
        d_model=16, num_heads=4, num_layers=1, d_ff=32, max_seq_len=12,
        dropout=0.0, src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size, warmup_steps=4, batch_size=2,
    )
    return loader, config


def test_run_epoch_updates_weights_and_returns_finite_metrics():
    tf.keras.utils.set_random_seed(7)
    loader, config = _fixture()
    model = Transformer.from_config(config)
    batch = next(iter(loader))
    masks = model.create_masks(batch.src, batch.tgt_in)
    model(batch.src, batch.tgt_in, *masks, training=False)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=NoamSchedule(config.d_model, config.warmup_steps),
        beta_1=config.adam_beta1, beta_2=config.adam_beta2, epsilon=config.adam_eps,
    )
    optimizer.build(model.trainable_variables)
    before = [v.numpy().copy() for v in model.trainable_variables]
    loss, step, tokens_per_sec, elapsed = run_epoch(
        model, loader, LabelSmoothingLoss(config.label_smoothing, config.pad_idx),
        optimizer=optimizer, global_step=0, grad_clip=config.grad_clip,
    )
    assert step == 1
    assert np.isfinite(loss)
    assert tokens_per_sec > 0 and elapsed > 0
    assert any(not np.array_equal(old, new.numpy()) for old, new in zip(before, model.trainable_variables))
    assert int(optimizer.iterations.numpy()) == 1


def test_validation_epoch_does_not_update_weights():
    tf.keras.utils.set_random_seed(11)
    loader, config = _fixture()
    model = Transformer.from_config(config)
    batch = next(iter(loader))
    masks = model.create_masks(batch.src, batch.tgt_in)
    model(batch.src, batch.tgt_in, *masks, training=False)
    before = [v.numpy().copy() for v in model.trainable_variables]
    loss, step, tokens_per_sec, elapsed = run_epoch(
        model, loader, LabelSmoothingLoss(config.label_smoothing, config.pad_idx),
    )
    assert np.isfinite(loss)
    assert step == 0
    assert tokens_per_sec > 0 and elapsed > 0
    assert all(np.array_equal(old, new.numpy()) for old, new in zip(before, model.trainable_variables))
