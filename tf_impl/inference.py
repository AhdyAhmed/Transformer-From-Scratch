"""Translate one English sentence with a trained TensorFlow checkpoint.

Example:
    python -m tf_impl.inference --text "A man is sleeping on a couch."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import tensorflow as tf

from data.bpe_tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, BPETokenizer
from tf_impl.config import TransformerConfig
from tf_impl.model.transformer import Transformer
from tf_impl.training.lr_schedule import NoamSchedule
from tf_impl.training.train import configure_device, greedy_decode

ROOT = Path(__file__).resolve().parents[1]


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


def resolve_checkpoint(value: Path) -> str:
    """Resolve a checkpoint-manager directory or an explicit TF prefix."""
    if value.is_dir():
        checkpoint = tf.train.latest_checkpoint(str(value))
        if checkpoint is None:
            raise ValueError(f"no TensorFlow checkpoint found in {value}")
        return checkpoint
    if not Path(str(value) + ".index").is_file():
        raise FileNotFoundError(f"TensorFlow checkpoint not found: {value}")
    return str(value)


def load_checkpoint(
    checkpoint_value: Path, config_path: Path, tokenizer_path: Path
) -> tuple[Transformer, TransformerConfig, BPETokenizer, dict[str, Any], Any]:
    """Restore the full training object graph and return an inference model."""
    if not config_path.is_file():
        raise FileNotFoundError(f"TensorFlow config not found: {config_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")
    config = TransformerConfig.load(config_path)
    tokenizer = BPETokenizer.load(tokenizer_path)
    validate_tokenizer(tokenizer, config)
    model = Transformer.from_config(config)
    dummy = tf.constant([[config.bos_idx, config.eos_idx]], dtype=tf.int32)
    masks = model.create_masks(dummy, dummy)
    model(dummy, dummy, *masks, training=False)

    schedule = NoamSchedule(config.d_model, config.warmup_steps)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=schedule,
        beta_1=config.adam_beta1,
        beta_2=config.adam_beta2,
        epsilon=config.adam_eps,
    )
    if hasattr(optimizer, "build"):
        optimizer.build(model.trainable_variables)
    epoch_var = tf.Variable(0, dtype=tf.int64, trainable=False, name="epoch")
    best_val_var = tf.Variable(
        float("inf"), dtype=tf.float32, trainable=False, name="best_val_loss"
    )
    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch_var,
        best_val_loss=best_val_var,
    )
    status = checkpoint.restore(resolve_checkpoint(checkpoint_value))
    status.assert_existing_objects_matched()
    metadata = {
        "epoch": int(epoch_var.numpy()),
        "step": int(optimizer.iterations.numpy()),
        "best_validation_loss": float(best_val_var.numpy()),
    }
    # Return the checkpoint object so its graph stays alive through inference.
    return model, config, tokenizer, metadata, checkpoint


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
    device_label: str,
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
        model, source_ids, config.bos_idx, config.eos_idx, limit
    )
    return {
        "framework": "tensorflow",
        "device": device_label.lower(),
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
        default=ROOT / "checkpoints/day21-medium/tf/best",
    )
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "checkpoints/day21-medium/tf/config.json",
    )
    parser.add_argument(
        "--tokenizer", type=Path,
        default=ROOT / "checkpoints/day21-medium/tokenizer.json",
    )
    parser.add_argument("--device", choices=("gpu", "cpu", "auto"), default="cpu")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        device_label = configure_device(args.device)
        model, config, tokenizer, checkpoint_metadata, restore_context = load_checkpoint(
            args.checkpoint, args.config, args.tokenizer
        )
        result = translate(
            model, config, tokenizer, args.text, device_label, args.max_output_tokens
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    result["checkpoint"] = checkpoint_metadata
    assert restore_context is not None
    print(result["translation"])
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
