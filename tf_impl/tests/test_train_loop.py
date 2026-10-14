"""Day 18 tests for the TensorFlow training loop and checkpoint-compatible state."""
import numpy as np
import pytest
import tensorflow as tf

from data.bpe_tokenizer import BPETokenizer
from data.parallel_data import ParallelDataset
from tf_impl.config import TransformerConfig
from tf_impl.data.dataset import TFBatchLoader
from tf_impl.model.transformer import Transformer
from tf_impl.training.losses import LabelSmoothingLoss
from tf_impl.training.lr_schedule import NoamSchedule
from tf_impl.training.train import (
    build_parser,
    configure_device,
    greedy_decode,
    run_epoch,
    save_loss_curve,
)


def test_training_defaults_to_gpu():
    args = build_parser().parse_args([])
    assert args.device == "gpu"
    assert args.max_train_pairs is None
    assert args.max_val_pairs is None
    assert args.sample_translations == 3


def test_requested_gpu_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda kind: [])
    with pytest.raises(SystemExit, match="no GPU device"):
        configure_device("gpu")


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


def test_greedy_decode_returns_bos_prefixed_bounded_sequence():
    tf.keras.utils.set_random_seed(19)
    loader, config = _fixture()
    model = Transformer.from_config(config)
    batch = next(iter(loader))
    masks = model.create_masks(batch.src, batch.tgt_in)
    model(batch.src, batch.tgt_in, *masks, training=False)

    decoded = greedy_decode(
        model,
        batch.src[0].numpy().tolist(),
        config.bos_idx,
        config.eos_idx,
        config.max_seq_len,
    )

    assert decoded[0] == config.bos_idx
    assert 2 <= len(decoded) <= config.max_seq_len


def test_save_loss_curve_creates_svg(tmp_path):
    history = tmp_path / "history.jsonl"
    history.write_text(
        '{"epoch": 1, "train_loss": 2.0, "val_loss": 2.2}\n'
        '{"epoch": 2, "train_loss": 1.5, "val_loss": 1.8}\n',
        encoding="utf-8",
    )
    output = tmp_path / "curve.svg"

    save_loss_curve(history, output)

    assert output.read_text(encoding="utf-8").lstrip().startswith("<?xml")
