"""Day 23: evaluate PyTorch and TensorFlow checkpoints on a shared corpus."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tensorflow as tf
import torch

from data.bpe_tokenizer import BPETokenizer
from data.parallel_data import ParallelDataset, read_parallel_corpus
from evaluation.metrics import BleuResult, corpus_bleu
from tf_impl.config import TransformerConfig as TFConfig
from tf_impl.model.transformer import Transformer as TFTransformer
from tf_impl.training.lr_schedule import NoamSchedule as TFNoamSchedule
from tf_impl.training.train import configure_device as configure_tf_device
from tf_impl.training.train import greedy_decode as tf_greedy_decode
from torch_impl.config import TransformerConfig as TorchConfig
from torch_impl.model.transformer import Transformer as TorchTransformer
from torch_impl.training.train import greedy_decode as torch_greedy_decode

ROOT = Path(__file__).resolve().parents[1]


def _load_torch_model(checkpoint_path: Path, device: torch.device):
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    config = TorchConfig.from_dict(payload["config"])
    model = TorchTransformer.from_config(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, config, {
        "epoch": int(payload["epoch"]),
        "step": int(payload["global_step"]),
        "best_validation_loss": float(payload["best_val_loss"]),
    }


def _tf_checkpoint_path(value: Path) -> str:
    if value.is_dir():
        checkpoint = tf.train.latest_checkpoint(str(value))
        if checkpoint is None:
            raise ValueError(f"No TensorFlow checkpoint found in {value}")
        return checkpoint
    if not Path(str(value) + ".index").is_file():
        raise ValueError(f"TensorFlow checkpoint not found: {value}")
    return str(value)


def _load_tf_model(config_path: Path, checkpoint_value: Path):
    config = TFConfig.load(config_path)
    model = TFTransformer.from_config(config)
    dummy = tf.constant([[config.bos_idx, config.eos_idx]], dtype=tf.int32)
    masks = model.create_masks(dummy, dummy)
    model(dummy, dummy, *masks, training=False)
    schedule = TFNoamSchedule(config.d_model, config.warmup_steps)
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
    checkpoint_path = _tf_checkpoint_path(checkpoint_value)
    status = checkpoint.restore(checkpoint_path)
    status.assert_existing_objects_matched()
    return model, config, {
        "epoch": int(epoch_var.numpy()),
        "step": int(optimizer.iterations.numpy()),
        "best_validation_loss": float(best_val_var.numpy()),
    }, checkpoint


def _translate_torch(
    model, config, tokenizer: BPETokenizer, pairs: list[tuple[str, str]], device
) -> tuple[list[str], float]:
    dataset = ParallelDataset(pairs, tokenizer, config.max_seq_len)
    started = time.perf_counter()
    predictions = [
        tokenizer.decode(torch_greedy_decode(
            model, src_ids, config.bos_idx, config.eos_idx, config.max_seq_len, device
        ))
        for src_ids, _ in dataset.examples
    ]
    return predictions, time.perf_counter() - started


def _translate_tf(
    model, config, tokenizer: BPETokenizer, pairs: list[tuple[str, str]]
) -> tuple[list[str], float]:
    dataset = ParallelDataset(pairs, tokenizer, config.max_seq_len)
    started = time.perf_counter()
    predictions = [
        tokenizer.decode(tf_greedy_decode(
            model, src_ids, config.bos_idx, config.eos_idx, config.max_seq_len
        ))
        for src_ids, _ in dataset.examples
    ]
    return predictions, time.perf_counter() - started


def _metric_dict(result: BleuResult, seconds: float, count: int) -> dict:
    return {
        "bleu": result.to_dict(),
        "decode_seconds": seconds,
        "sentences_per_second": count / seconds if seconds else 0.0,
    }


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_results(
    output_dir: Path,
    pairs: list[tuple[str, str]],
    torch_predictions: list[str],
    tf_predictions: list[str],
    metrics: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "translations.jsonl").open("w", encoding="utf-8") as handle:
        for (source, reference), torch_prediction, tf_prediction in zip(
            pairs, torch_predictions, tf_predictions
        ):
            handle.write(json.dumps({
                "source": source,
                "reference": reference,
                "pytorch": torch_prediction,
                "tensorflow": tf_prediction,
            }, ensure_ascii=False) + "\n")

    rows = [
        "# Day 23 — PyTorch vs TensorFlow evaluation",
        "",
        f"Shared validation examples: {len(pairs)}",
        "",
        "## Metrics",
        "",
        "| Framework | BLEU-4 | BP | Length ratio | Decode seconds | Sentences/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("pytorch", "PyTorch"), ("tensorflow", "TensorFlow")):
        result = metrics[name]
        bleu = result["bleu"]
        rows.append(
            f"| {label} | {bleu['score']:.4f} | {bleu['brevity_penalty']:.4f} | "
            f"{bleu['length_ratio']:.4f} | {result['decode_seconds']:.2f} | "
            f"{result['sentences_per_second']:.3f} |"
        )
    rows.extend([
        "",
        "BLEU is lowercase corpus BLEU-4 using the repository's Unicode "
        "word/punctuation tokenizer and exponential zero-count smoothing.",
        "",
        "## Qualitative examples",
        "",
        "| Source | Reference | PyTorch | TensorFlow |",
        "|---|---|---|---|",
    ])
    for (source, reference), torch_prediction, tf_prediction in list(zip(
        pairs, torch_predictions, tf_predictions
    ))[:10]:
        rows.append(
            f"| {_escape_cell(source)} | {_escape_cell(reference)} | "
            f"{_escape_cell(torch_prediction)} | {_escape_cell(tf_prediction)} |"
        )
    (output_dir / "comparison.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    multi30k = ROOT / "data" / "multi30k"
    parser.add_argument("--val-src", type=Path, default=multi30k / "val.en")
    parser.add_argument("--val-tgt", type=Path, default=multi30k / "val.de")
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--torch-checkpoint", type=Path, required=True)
    parser.add_argument("--torch-tokenizer", type=Path, required=True)
    parser.add_argument("--torch-device", default="cuda")
    parser.add_argument("--tf-checkpoint", type=Path, required=True)
    parser.add_argument("--tf-config", type=Path, required=True)
    parser.add_argument("--tf-tokenizer", type=Path, required=True)
    parser.add_argument("--tf-device", choices=("cpu", "gpu", "auto"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "day23")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_pairs < 1:
        raise SystemExit("--max-pairs must be positive")
    for path in (
        args.val_src, args.val_tgt, args.torch_checkpoint, args.torch_tokenizer,
        args.tf_config, args.tf_tokenizer,
    ):
        if not path.is_file():
            raise SystemExit(f"Required file not found: {path}")
    configure_tf_device(args.tf_device)
    torch_device = torch.device(args.torch_device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested for PyTorch evaluation but is not available")

    pairs = read_parallel_corpus(args.val_src, args.val_tgt)[:args.max_pairs]
    torch_tokenizer = BPETokenizer.load(args.torch_tokenizer)
    tf_tokenizer = BPETokenizer.load(args.tf_tokenizer)
    torch_model, torch_config, torch_checkpoint = _load_torch_model(
        args.torch_checkpoint, torch_device
    )
    tf_model, tf_config, tf_checkpoint, tf_restore_context = _load_tf_model(
        args.tf_config, args.tf_checkpoint
    )

    torch_predictions, torch_seconds = _translate_torch(
        torch_model, torch_config, torch_tokenizer, pairs, torch_device
    )
    tf_predictions, tf_seconds = _translate_tf(tf_model, tf_config, tf_tokenizer, pairs)
    references = [reference for _, reference in pairs]
    metrics = {
        "evaluation_pairs": len(pairs),
        "metric": {
            "name": "lowercase tokenized corpus BLEU-4",
            "tokenizer": "data.bpe_tokenizer._pretokenize",
            "smoothing": "exponential zero-count smoothing",
        },
        "pytorch": {
            **_metric_dict(corpus_bleu(torch_predictions, references), torch_seconds, len(pairs)),
            "checkpoint": torch_checkpoint,
        },
        "tensorflow": {
            **_metric_dict(corpus_bleu(tf_predictions, references), tf_seconds, len(pairs)),
            "checkpoint": tf_checkpoint,
        },
    }
    write_results(
        args.output_dir, pairs, torch_predictions, tf_predictions, metrics
    )
    # Keep the full checkpoint graph referenced through the end of evaluation.
    assert tf_restore_context is not None
    print(
        f"PyTorch BLEU: {metrics['pytorch']['bleu']['score']:.4f} "
        f"({torch_seconds:.1f}s)"
    )
    print(
        f"TensorFlow BLEU: {metrics['tensorflow']['bleu']['score']:.4f} "
        f"({tf_seconds:.1f}s)"
    )
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
