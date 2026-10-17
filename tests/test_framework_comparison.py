import json

import pytest

from evaluation.framework_comparison import (
    _tf_model_stats,
    _torch_model_stats,
    checkpoint_bytes,
    implementation_stats,
    read_history,
)
from torch_impl.config import TransformerConfig


def _row(epoch=1, step=2):
    return {
        "epoch": epoch,
        "global_step": step,
        "train_loss": 2.0,
        "val_loss": 2.1,
        "learning_rate": 0.01,
        "train_tokens_per_sec": 100.0,
        "val_tokens_per_sec": 120.0,
    }


def test_read_history_validates_and_preserves_rows(tmp_path):
    path = tmp_path / "history.jsonl"
    rows = [_row(), _row(2, 4)]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    assert read_history(path) == rows


def test_read_history_rejects_missing_fields(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text(json.dumps({"epoch": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        read_history(path)


def test_read_history_rejects_non_increasing_progress(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in [_row(), _row(1, 4)]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        read_history(path)


def test_implementation_stats_excludes_tests_init_blank_and_comments(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "model" / "layer.py").write_text(
        "# comment\n\nx = 1\n    # indented\ny = 2\n", encoding="utf-8"
    )
    (tmp_path / "__init__.py").write_text("ignored = True\n", encoding="utf-8")
    (tmp_path / "tests" / "test_layer.py").write_text("ignored = True\n", encoding="utf-8")
    assert implementation_stats(tmp_path) == {
        "python_files": 1,
        "nonblank_noncomment_lines": 2,
    }


def test_checkpoint_bytes_supports_file_directory_and_missing(tmp_path):
    file_path = tmp_path / "one.bin"
    file_path.write_bytes(b"123")
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    (directory / "a").write_bytes(b"12")
    (directory / "b").write_bytes(b"345")
    assert checkpoint_bytes(file_path) == 3
    assert checkpoint_bytes(directory) == 5
    assert checkpoint_bytes(tmp_path / "missing") is None


def test_model_stats_match_across_frameworks_and_count_float32_bytes(tmp_path):
    config_path = tmp_path / "config.json"
    TransformerConfig(
        d_model=8,
        num_heads=2,
        num_layers=1,
        d_ff=16,
        max_seq_len=8,
        src_vocab_size=16,
        tgt_vocab_size=16,
        share_embeddings=True,
    ).save(config_path)
    torch_parameters, torch_bytes = _torch_model_stats(config_path)
    tf_parameters, tf_bytes = _tf_model_stats(config_path)
    assert torch_parameters == tf_parameters
    assert torch_bytes == tf_bytes == 4 * torch_parameters
