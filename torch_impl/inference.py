"""Translate one English sentence with a trained PyTorch checkpoint.

Example:
    python -m torch_impl.inference --text "A man is sleeping on a couch."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from data.bpe_tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, BPETokenizer
from torch_impl.config import TransformerConfig
from torch_impl.model.transformer import Transformer
from torch_impl.training.train import greedy_decode

ROOT = Path(__file__).resolve().parents[1]


def resolve_device(value: str) -> torch.device:
    """Resolve an inference device without silently ignoring explicit CUDA."""
    if value not in {"cuda", "cpu", "auto"}:
        raise ValueError(f"unsupported device {value!r}; choose cuda, cpu, or auto")
    if value == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested for PyTorch inference but is not available")
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else (
        "cpu" if value == "auto" else value
    ))


def validate_tokenizer(tokenizer: BPETokenizer, config: TransformerConfig) -> None:
    """Fail early when a tokenizer cannot be used with this checkpoint."""
    if tokenizer.vocab_size not in {config.src_vocab_size, config.tgt_vocab_size} or (
        config.src_vocab_size != config.tgt_vocab_size
    ):
        raise ValueError(
            f"tokenizer vocabulary ({tokenizer.vocab_size}) does not match "
            f"checkpoint vocabularies (source={config.src_vocab_size}, "
            f"target={config.tgt_vocab_size})"
        )
    expected = {
        PAD_TOKEN: config.pad_idx,
        BOS_TOKEN: config.bos_idx,
        EOS_TOKEN: config.eos_idx,
        UNK_TOKEN: config.unk_idx,
    }
    mismatched = {
        token: (tokenizer.vocab.get(token), token_id)
        for token, token_id in expected.items()
        if tokenizer.vocab.get(token) != token_id
    }
    if mismatched:
        raise ValueError(f"tokenizer special-token ids do not match checkpoint: {mismatched}")


def load_checkpoint(
    checkpoint_path: Path, tokenizer_path: Path, device: torch.device
) -> tuple[Transformer, TransformerConfig, BPETokenizer, dict[str, Any]]:
    """Strictly reconstruct a model and load its saved inference state."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before ``weights_only`` was added.
        payload = torch.load(checkpoint_path, map_location=device)
    required = {
        "model_state_dict", "config", "epoch", "global_step", "best_val_loss"
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {sorted(missing)}")
    config = TransformerConfig.from_dict(payload["config"])
    tokenizer = BPETokenizer.load(tokenizer_path)
    validate_tokenizer(tokenizer, config)
    model = Transformer.from_config(config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    metadata = {
        "epoch": int(payload["epoch"]),
        "step": int(payload["global_step"]),
        "best_validation_loss": float(payload["best_val_loss"]),
    }
    return model, config, tokenizer, metadata


def encode_source(
    text: str, tokenizer: BPETokenizer, config: TransformerConfig
) -> tuple[list[int], bool]:
    """Encode one nonempty source while preserving BOS/EOS under truncation."""
    if not text.strip():
        raise ValueError("source text must not be empty")
    token_ids = tokenizer.encode(text.strip())
    room = config.max_seq_len - 2
    truncated = len(token_ids) > room
    return [config.bos_idx, *token_ids[:room], config.eos_idx], truncated


def translate(
    model: Transformer,
    config: TransformerConfig,
    tokenizer: BPETokenizer,
    text: str,
    device: torch.device,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """Translate one sentence and return a JSON-serializable record."""
    limit = config.max_seq_len if max_output_tokens is None else max_output_tokens
    if not 2 <= limit <= config.max_seq_len:
        raise ValueError(
            f"max_output_tokens must be in [2, {config.max_seq_len}], got {limit}"
        )
    source_ids, truncated = encode_source(text, tokenizer, config)
    output_ids = greedy_decode(
        model, source_ids, config.bos_idx, config.eos_idx, limit, device
    )
    return {
        "framework": "pytorch",
        "device": str(device),
        "source": text.strip(),
        "source_token_count": len(source_ids),
        "source_truncated": truncated,
        "translation": tokenizer.decode(output_ids),
        "output_token_count": len(output_ids),
        "output_ended_with_eos": output_ids[-1] == config.eos_idx,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="English sentence to translate")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "checkpoints/day20-full/torch/best.pt",
    )
    parser.add_argument(
        "--tokenizer", type=Path,
        default=ROOT / "checkpoints/day20-full/tokenizer.json",
    )
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    try:
        model, config, tokenizer, checkpoint = load_checkpoint(
            args.checkpoint, args.tokenizer, device
        )
        result = translate(
            model, config, tokenizer, args.text, device, args.max_output_tokens
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    result["checkpoint"] = checkpoint
    print(result["translation"])
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
