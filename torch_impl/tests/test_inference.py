import json

import pytest
import torch

from data.bpe_tokenizer import BPETokenizer
from torch_impl.config import TransformerConfig
from torch_impl.inference import encode_source, main, resolve_device, validate_tokenizer
from torch_impl.model.transformer import Transformer


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
    model = Transformer.from_config(config)
    with torch.no_grad():
        model.output_projection.weight.zero_()
        model.output_projection.bias.zero_()
        model.output_projection.bias[config.eos_idx] = 10.0
    checkpoint_path = tmp_path / "best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "epoch": 3,
        "global_step": 12,
        "best_val_loss": 1.25,
    }, checkpoint_path)
    return tokenizer, config, tokenizer_path, checkpoint_path


def test_encode_source_keeps_boundaries_and_reports_truncation(tmp_path):
    tokenizer, config, _, _ = _artifacts(tmp_path)
    ids, truncated = encode_source("a man sleeps " * 20, tokenizer, config)
    assert len(ids) == config.max_seq_len
    assert ids[0] == config.bos_idx and ids[-1] == config.eos_idx
    assert truncated


def test_inference_cli_loads_checkpoint_and_writes_json(tmp_path, capsys):
    _, _, tokenizer_path, checkpoint_path = _artifacts(tmp_path)
    output_path = tmp_path / "result.json"
    main([
        "--text", "A man sleeps", "--checkpoint", str(checkpoint_path),
        "--tokenizer", str(tokenizer_path), "--device", "cpu",
        "--max-output-tokens", "4", "--output", str(output_path),
    ])
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["framework"] == "pytorch"
    assert result["device"] == "cpu"
    assert result["output_ended_with_eos"]
    assert result["checkpoint"] == {
        "epoch": 3, "step": 12, "best_validation_loss": 1.25
    }
    assert "Saved:" in capsys.readouterr().out


def test_validate_tokenizer_rejects_vocabulary_mismatch(tmp_path):
    tokenizer, config, _, _ = _artifacts(tmp_path)
    config.src_vocab_size += 1
    try:
        validate_tokenizer(tokenizer, config)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mismatched tokenizer was accepted")


def test_explicit_cuda_rejects_unavailable_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="CUDA requested"):
        resolve_device("cuda")
