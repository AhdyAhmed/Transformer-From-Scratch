import json
from pathlib import Path

import tensorflow as tf

from data.bpe_tokenizer import BPETokenizer
from tf_impl.config import TransformerConfig
from tf_impl.inference import encode_source, main, validate_tokenizer
from tf_impl.model.transformer import Transformer
from tf_impl.training.lr_schedule import NoamSchedule


def _artifacts(tmp_path):
    tokenizer = BPETokenizer()
    tokenizer.train(["a man sleeps", "ein mann schläft"], vocab_size=32)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    config = TransformerConfig(
        d_model=8,
        num_heads=2,
        num_layers=1,
        d_ff=16,
        dropout=0.0,
        max_seq_len=8,
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        share_embeddings=True,
    )
    config_path = tmp_path / "config.json"
    config.save(config_path)
    model = Transformer.from_config(config)
    dummy = tf.constant([[config.bos_idx, config.eos_idx]], dtype=tf.int32)
    model(dummy, dummy, *model.create_masks(dummy, dummy), training=False)
    model.output_projection.kernel.assign(tf.zeros_like(model.output_projection.kernel))
    bias = tf.zeros_like(model.output_projection.bias)
    bias = tf.tensor_scatter_nd_update(bias, [[config.eos_idx]], [10.0])
    model.output_projection.bias.assign(bias)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=NoamSchedule(config.d_model, config.warmup_steps)
    )
    if hasattr(optimizer, "build"):
        optimizer.build(model.trainable_variables)
    optimizer.iterations.assign(12)
    epoch = tf.Variable(3, dtype=tf.int64, trainable=False, name="epoch")
    best_val_loss = tf.Variable(1.25, trainable=False, name="best_val_loss")
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=optimizer, epoch=epoch, best_val_loss=best_val_loss
    )
    checkpoint_dir = tmp_path / "best"
    manager = tf.train.CheckpointManager(checkpoint, str(checkpoint_dir), max_to_keep=1)
    checkpoint_prefix = manager.save(checkpoint_number=3)
    assert checkpoint_prefix is not None
    checkpoint_prefix = tmp_path / "best" / Path(checkpoint_prefix).name
    return tokenizer, config, tokenizer_path, config_path, checkpoint_prefix


def test_encode_source_keeps_boundaries_and_reports_truncation(tmp_path):
    tokenizer, config, *_ = _artifacts(tmp_path)
    ids, truncated = encode_source("a man sleeps " * 20, tokenizer, config)
    assert len(ids) == config.max_seq_len
    assert ids[0] == config.bos_idx and ids[-1] == config.eos_idx
    assert truncated


def test_inference_cli_restores_full_checkpoint_and_writes_json(
    tmp_path, capsys, monkeypatch
):
    _, _, tokenizer_path, config_path, checkpoint_prefix = _artifacts(tmp_path)
    monkeypatch.setattr("tf_impl.inference.configure_device", lambda _: "CPU")
    output_path = tmp_path / "result.json"
    main([
        "--text", "A man sleeps", "--checkpoint", str(checkpoint_prefix),
        "--config", str(config_path), "--tokenizer", str(tokenizer_path),
        "--device", "cpu", "--max-output-tokens", "4",
        "--output", str(output_path),
    ])
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["framework"] == "tensorflow"
    assert result["device"] == "cpu"
    assert result["output_ended_with_eos"]
    assert result["checkpoint"] == {
        "epoch": 3, "step": 12, "best_validation_loss": 1.25
    }
    assert "Saved:" in capsys.readouterr().out


def test_validate_tokenizer_rejects_vocabulary_mismatch(tmp_path):
    tokenizer, config, *_ = _artifacts(tmp_path)
    config.tgt_vocab_size += 1
    try:
        validate_tokenizer(tokenizer, config)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mismatched tokenizer was accepted")
