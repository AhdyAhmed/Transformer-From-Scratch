"""Day 25: generate an evidence-backed PyTorch/TensorFlow comparison."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import tensorflow as tf
import torch

from tf_impl.config import TransformerConfig as TFConfig
from tf_impl.model.transformer import Transformer as TFTransformer
from torch_impl.config import TransformerConfig as TorchConfig
from torch_impl.model.transformer import Transformer as TorchTransformer

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_HISTORY_FIELDS = {
    "epoch",
    "global_step",
    "train_loss",
    "val_loss",
    "learning_rate",
    "train_tokens_per_sec",
    "val_tokens_per_sec",
}


def read_history(path: Path) -> list[dict[str, float]]:
    """Read and validate a training history JSONL file."""
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"training history is empty: {path}")
    previous_epoch = 0
    previous_step = -1
    for row in rows:
        missing = REQUIRED_HISTORY_FIELDS - row.keys()
        if missing:
            raise ValueError(f"history row is missing {sorted(missing)}: {path}")
        epoch = int(row["epoch"])
        step = int(row["global_step"])
        if epoch <= previous_epoch or step <= previous_step:
            raise ValueError(f"history epochs/steps must be strictly increasing: {path}")
        for field in REQUIRED_HISTORY_FIELDS - {"epoch", "global_step"}:
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {field}={value} in {path}")
        previous_epoch, previous_step = epoch, step
    return rows


def implementation_stats(root: Path) -> dict[str, int]:
    """Count implementation files and nonblank, non-comment physical lines."""
    files = sorted(
        path
        for path in root.rglob("*.py")
        if "tests" not in path.parts and path.name != "__init__.py"
    )
    lines = 0
    for path in files:
        lines += sum(
            1
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return {"python_files": len(files), "nonblank_noncomment_lines": lines}


def checkpoint_bytes(path: Path) -> int | None:
    """Return file or recursive directory size, or ``None`` if unavailable."""
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return None


def _torch_model_stats(config_path: Path) -> tuple[int, int]:
    config = TorchConfig.load(config_path)
    model = TorchTransformer.from_config(config)
    parameters = list(model.parameters())
    return (
        sum(parameter.numel() for parameter in parameters),
        sum(parameter.numel() * parameter.element_size() for parameter in parameters),
    )


def _tf_model_stats(config_path: Path) -> tuple[int, int]:
    config = TFConfig.load(config_path)
    model = TFTransformer.from_config(config)
    dummy = tf.constant([[config.bos_idx, config.eos_idx]], dtype=tf.int32)
    masks = model.create_masks(dummy, dummy)
    model(dummy, dummy, *masks, training=False)
    return (
        sum(int(tf.size(variable).numpy()) for variable in model.trainable_variables),
        sum(
            int(tf.size(variable).numpy())
            * int(tf.dtypes.as_dtype(variable.dtype).size)
            for variable in model.trainable_variables
        ),
    )


def _history_stats(rows: list[dict[str, float]]) -> dict[str, float | int]:
    throughput = [float(row["train_tokens_per_sec"]) for row in rows]
    final = rows[-1]
    best = min(rows, key=lambda row: float(row["val_loss"]))
    return {
        "epochs": len(rows),
        "optimizer_steps": int(final["global_step"]),
        "mean_train_tokens_per_sec": statistics.fmean(throughput),
        "median_train_tokens_per_sec": statistics.median(throughput),
        "final_train_tokens_per_sec": throughput[-1],
        "initial_train_loss": float(rows[0]["train_loss"]),
        "final_train_loss": float(final["train_loss"]),
        "best_validation_loss": float(best["val_loss"]),
        "best_validation_epoch": int(best["epoch"]),
    }


def _mib(byte_count: int | None) -> float | None:
    return None if byte_count is None else byte_count / (1024 * 1024)


def build_comparison(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict], list[dict]]:
    torch_history = read_history(args.torch_history)
    tf_history = read_history(args.tf_history)
    evaluation = json.loads(args.evaluation_metrics.read_text(encoding="utf-8"))
    torch_params, torch_param_bytes = _torch_model_stats(args.torch_config)
    tf_params, tf_param_bytes = _tf_model_stats(args.tf_config)

    data: dict[str, Any] = {
        "scope": {
            "controlled_framework_benchmark": False,
            "reason": (
                "The recorded runs differ in hardware, model size, vocabulary, "
                "training-pair count, sequence length, normalization mode, and epochs."
            ),
            "peak_runtime_memory_measured": False,
            "memory_note": (
                "Parameter bytes and checkpoint disk size are exact; peak allocator "
                "or process memory was not captured during the historical runs."
            ),
        },
        "pytorch": {
            "version": torch.__version__,
            "device": "NVIDIA Quadro M2000M CUDA (4 GB)",
            "training_pairs": args.torch_train_pairs,
            "validation_pairs": args.torch_val_pairs,
            "parameters": torch_params,
            "parameter_memory_mib": _mib(torch_param_bytes),
            "checkpoint_size_mib": _mib(checkpoint_bytes(args.torch_checkpoint)),
            "implementation": implementation_stats(ROOT / "torch_impl"),
            "training": _history_stats(torch_history),
            "evaluation": evaluation["pytorch"],
        },
        "tensorflow": {
            "version": tf.__version__,
            "device": "Native Windows CPU",
            "training_pairs": args.tf_train_pairs,
            "validation_pairs": args.tf_val_pairs,
            "parameters": tf_params,
            "parameter_memory_mib": _mib(tf_param_bytes),
            "checkpoint_size_mib": _mib(checkpoint_bytes(args.tf_checkpoint)),
            "implementation": implementation_stats(ROOT / "tf_impl"),
            "training": _history_stats(tf_history),
            "evaluation": evaluation["tensorflow"],
        },
    }
    data["observed_ratios"] = {
        "mean_training_throughput_pytorch_over_tensorflow": (
            data["pytorch"]["training"]["mean_train_tokens_per_sec"]
            / data["tensorflow"]["training"]["mean_train_tokens_per_sec"]
        ),
        "decode_throughput_pytorch_over_tensorflow": (
            data["pytorch"]["evaluation"]["sentences_per_second"]
            / data["tensorflow"]["evaluation"]["sentences_per_second"]
        ),
        "parameter_count_pytorch_over_tensorflow": torch_params / tf_params,
    }
    return data, torch_history, tf_history


def _format_optional(value: float | None) -> str:
    return "not available" if value is None else f"{value:.2f}"


def write_outputs(
    comparison: dict[str, Any],
    torch_history: list[dict],
    tf_history: list[dict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        [row["epoch"] for row in torch_history],
        [row["train_tokens_per_sec"] for row in torch_history],
        marker="o", markersize=3, label="PyTorch · CUDA · full corpus",
    )
    axis.plot(
        [row["epoch"] for row in tf_history],
        [row["train_tokens_per_sec"] for row in tf_history],
        marker="o", markersize=3, label="TensorFlow · CPU · 5k subset",
    )
    axis.set(
        xlabel="Epoch",
        ylabel="Training target tokens / second",
        title="Observed training throughput (different workloads and hardware)",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    chart_path = output_dir / "training_throughput.svg"
    figure.savefig(chart_path)
    plt.close(figure)
    # Matplotlib's SVG backend leaves spaces before many newlines. Normalize
    # generated text so reproducible artifacts also pass ``git diff --check``.
    svg_lines = chart_path.read_text(encoding="utf-8").splitlines()
    chart_path.write_text(
        "\n".join(line.rstrip() for line in svg_lines) + "\n", encoding="utf-8"
    )

    torch_data = comparison["pytorch"]
    tf_data = comparison["tensorflow"]
    rows = [
        "# Day 25 — PyTorch vs TensorFlow comparison",
        "",
        "> **Interpretation boundary:** these are observed project runs, not a "
        "controlled framework benchmark. Hardware, model capacity, data volume, "
        "vocabulary, normalization, sequence length, and epoch count differ.",
        "",
        "## Measured results",
        "",
        "| Metric | PyTorch | TensorFlow |",
        "|---|---:|---:|",
        f"| Device | {torch_data['device']} | {tf_data['device']} |",
        f"| Training pairs | {torch_data['training_pairs']:,} | {tf_data['training_pairs']:,} |",
        f"| Parameters | {torch_data['parameters']:,} | {tf_data['parameters']:,} |",
        f"| Parameter memory (MiB) | {torch_data['parameter_memory_mib']:.2f} | {tf_data['parameter_memory_mib']:.2f} |",
        f"| Best checkpoint size (MiB) | {_format_optional(torch_data['checkpoint_size_mib'])} | {_format_optional(tf_data['checkpoint_size_mib'])} |",
        f"| Mean train tokens/s | {torch_data['training']['mean_train_tokens_per_sec']:,.0f} | {tf_data['training']['mean_train_tokens_per_sec']:,.0f} |",
        f"| Best validation loss | {torch_data['training']['best_validation_loss']:.4f} | {tf_data['training']['best_validation_loss']:.4f} |",
        f"| BLEU-4 (100 shared examples) | {torch_data['evaluation']['bleu']['score']:.4f} | {tf_data['evaluation']['bleu']['score']:.4f} |",
        f"| Decode sentences/s | {torch_data['evaluation']['sentences_per_second']:.3f} | {tf_data['evaluation']['sentences_per_second']:.3f} |",
        f"| Implementation Python files | {torch_data['implementation']['python_files']} | {tf_data['implementation']['python_files']} |",
        f"| Nonblank/non-comment lines | {torch_data['implementation']['nonblank_noncomment_lines']:,} | {tf_data['implementation']['nonblank_noncomment_lines']:,} |",
        "",
        "Checkpoint size includes the saved optimizer/progress state. Parameter "
        "memory assumes the actual trainable-variable dtypes. Peak runtime memory "
        "was not recorded, so no peak-memory claim is made.",
        "",
        "![Observed training throughput](training_throughput.svg)",
        "",
        "## Engineering comparison",
        "",
        "| Area | PyTorch | TensorFlow/Keras |",
        "|---|---|---|",
        "| Model API | `nn.Module.forward`; explicit device movement | `Layer.call`; variables build lazily |",
        "| Training | Direct `zero_grad/backward/step` flow | Explicit `GradientTape` with gradient application |",
        "| Debugging | Eager tensors and Python stack traces were direct | Eager mode was inspectable; lazy variable creation and checkpoint object graphs required extra care |",
        "| Checkpoints | One atomic payload with model, optimizer, RNG, and progress | Trackable object graph plus checkpoint managers; full graph must be recreated before restore |",
        "| GPU in this environment | Native Windows CUDA worked | TensorFlow 2.11+ required Linux/WSL2 for CUDA, so this run used CPU |",
        "| Code verbosity | Slightly fewer measured implementation lines | Slightly more glue for `training=` propagation, variable building, and checkpoint restoration |",
        "",
        "See [`WRITEUP.md`](../../WRITEUP.md) for design decisions and the full "
        "interpretation of these measurements.",
    ]
    (output_dir / "README.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-history", type=Path, default=ROOT / "results/day20-full/history.jsonl")
    parser.add_argument("--tf-history", type=Path, default=ROOT / "results/day22/history.jsonl")
    parser.add_argument("--evaluation-metrics", type=Path, default=ROOT / "results/day23/metrics.json")
    parser.add_argument("--torch-config", type=Path, default=ROOT / "configs/day20-full-torch.json")
    parser.add_argument("--tf-config", type=Path, default=ROOT / "configs/day22-tf.json")
    parser.add_argument("--torch-checkpoint", type=Path, default=ROOT / "checkpoints/day20-full/torch/best.pt")
    parser.add_argument("--tf-checkpoint", type=Path, default=ROOT / "checkpoints/day21-medium/tf/best")
    parser.add_argument("--torch-train-pairs", type=int, default=29_000)
    parser.add_argument("--torch-val-pairs", type=int, default=1_014)
    parser.add_argument("--tf-train-pairs", type=int, default=5_000)
    parser.add_argument("--tf-val-pairs", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/day25")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    required = (
        args.torch_history, args.tf_history, args.evaluation_metrics,
        args.torch_config, args.tf_config,
    )
    for path in required:
        if not path.is_file():
            raise SystemExit(f"Required file not found: {path}")
    for value in (
        args.torch_train_pairs, args.torch_val_pairs,
        args.tf_train_pairs, args.tf_val_pairs,
    ):
        if value < 1:
            raise SystemExit("dataset pair counts must be positive")
    comparison, torch_history, tf_history = build_comparison(args)
    write_outputs(comparison, torch_history, tf_history, args.output_dir)
    print(f"PyTorch parameters: {comparison['pytorch']['parameters']:,}")
    print(f"TensorFlow parameters: {comparison['tensorflow']['parameters']:,}")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
