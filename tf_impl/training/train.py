"""Day 18: end-to-end TensorFlow Transformer training with validation/checkpoints.

Run from the repository root:
  python -m tf_impl.training.train --epochs 1 --batch-size 4 --vocab-size 128
  python -m tf_impl.training.train --resume checkpoints/tf/latest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from data.bpe_tokenizer import BPETokenizer
from data.parallel_data import ParallelDataset, read_parallel_corpus
from tf_impl.config import TransformerConfig
from tf_impl.data.dataset import TFBatchLoader
from tf_impl.model.transformer import Transformer
from tf_impl.training.losses import LabelSmoothingLoss
from tf_impl.training.lr_schedule import NoamSchedule

ROOT = Path(__file__).resolve().parents[2]


def configure_device(device: str) -> str:
    """Select the TensorFlow training device before model construction.

    GPU is the default and is intentionally strict: a missing GPU raises
    instead of silently turning a requested training run into a much slower
    CPU job. ``auto`` retains TensorFlow's normal fallback behavior, while
    ``cpu`` explicitly hides any visible GPUs.
    """
    if device not in {"gpu", "cpu", "auto"}:
        raise ValueError(f"unsupported device {device!r}; choose gpu, cpu, or auto")

    physical_gpus = tf.config.list_physical_devices("GPU")
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        return "CPU"
    if device == "gpu" and not physical_gpus:
        hint = " Install the GPU-enabled TensorFlow environment for this platform."
        if sys.platform == "win32":
            hint = (
                " TensorFlow 2.11+ does not support CUDA on native Windows. "
                "Use a supported Linux/WSL2 GPU host and install "
                "requirements-tf-gpu.txt, or pass --device cpu explicitly."
            )
        raise SystemExit("TensorFlow GPU requested, but no GPU device is available." + hint)

    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            raise RuntimeError(
                "TensorFlow GPU configuration must happen before model or tensor creation"
            ) from exc
    return "GPU" if physical_gpus else "CPU"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_or_train_tokenizer(
    path: Path, train_pairs: list[tuple[str, str]], vocab_size: int,
) -> BPETokenizer:
    if path.exists():
        tokenizer = BPETokenizer.load(path)
        print(f"Loaded tokenizer: {path} ({tokenizer.vocab_size} tokens)")
        return tokenizer
    # Fit on training text only to avoid validation-data leakage.
    corpus = [text for pair in train_pairs for text in pair]
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=vocab_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(path)
    print(f"Trained tokenizer: {path} ({tokenizer.vocab_size} tokens from {len(corpus)} lines)")
    return tokenizer


def greedy_decode(
    model: Transformer,
    src_ids: list[int],
    bos_idx: int,
    eos_idx: int,
    max_len: int,
) -> list[int]:
    """Greedily decode one already-tokenized source sequence."""
    if max_len < 2:
        raise ValueError(f"max_len must be at least 2, got {max_len}")
    src = tf.constant([src_ids], dtype=tf.int32)
    src_mask = tf.expand_dims(tf.expand_dims(src != model.pad_idx, axis=1), axis=1)
    memory = model.encode(src, src_mask, training=False)
    generated = tf.constant([[bos_idx]], dtype=tf.int32)
    for _ in range(max_len - 1):
        _, tgt_mask, memory_mask = model.create_masks(src, generated)
        decoded = model.decode(
            generated, memory, tgt_mask, memory_mask, training=False
        )
        next_token = tf.argmax(
            model.output_projection(decoded[:, -1]), axis=-1, output_type=tf.int32
        )[:, None]
        generated = tf.concat((generated, next_token), axis=1)
        if int(next_token[0, 0].numpy()) == eos_idx:
            break
    return generated[0].numpy().tolist()


def translation_samples(
    model: Transformer,
    pairs: list[tuple[str, str]],
    tokenizer: BPETokenizer,
    config: TransformerConfig,
    count: int,
) -> list[dict[str, str]]:
    """Translate the first deterministic validation examples for progress logs."""
    if count < 0:
        raise ValueError(f"sample translation count must be non-negative, got {count}")
    dataset = ParallelDataset(pairs[:count], tokenizer, config.max_seq_len)
    records = []
    for (source, reference), (src_ids, _) in zip(pairs[:count], dataset.examples):
        predicted_ids = greedy_decode(
            model, src_ids, config.bos_idx, config.eos_idx, config.max_seq_len
        )
        records.append({
            "source": source,
            "reference": reference,
            "prediction": tokenizer.decode(predicted_ids),
        })
    return records


def save_loss_curve(history_path: Path, output_path: Path) -> None:
    """Render train/validation loss history as a portable SVG."""
    rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    epochs = [row["epoch"] for row in rows]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, [row["train_loss"] for row in rows], marker="o", label="train")
    axis.plot(epochs, [row["val_loss"] for row in rows], marker="o", label="validation")
    axis.set(xlabel="Epoch", ylabel="Token loss", title="Training loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="svg")
    plt.close(figure)
    svg = output_path.read_text(encoding="utf-8")
    output_path.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
    )


def run_epoch(
    model: Transformer,
    loader: TFBatchLoader,
    loss_fn: LabelSmoothingLoss,
    optimizer: tf.keras.optimizers.Optimizer | None = None,
    global_step: int = 0,
    grad_clip: float = 1.0,
) -> tuple[float, int, float, float]:
    """Run one teacher-forced training or validation epoch.

    Loss is accumulated by non-padding target-token count, so batches with
    different amounts of padding contribute proportionally to the epoch mean.
    """
    training = optimizer is not None
    total_loss = 0.0
    total_tokens = 0
    started = time.perf_counter()

    for batch in loader:
        if training:
            with tf.GradientTape() as tape:
                src_mask, tgt_mask, memory_mask = model.create_masks(batch.src, batch.tgt_in)
                logits = model(
                    batch.src, batch.tgt_in, src_mask, tgt_mask, memory_mask,
                    training=True,
                )
                loss = loss_fn(batch.tgt_out, logits)
            if not bool(tf.reduce_all(tf.math.is_finite(loss)).numpy()):
                raise FloatingPointError(
                    f"non-finite training loss at step {global_step + 1}: {float(loss.numpy())}"
                )
            gradients = tape.gradient(loss, model.trainable_variables)
            pairs = [(g, v) for g, v in zip(gradients, model.trainable_variables) if g is not None]
            if not pairs:
                raise RuntimeError("No gradients were produced; check the model's trainable variables.")
            grads, variables = zip(*pairs)
            def _gradient_values(gradient):
                return gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient

            if not all(
                bool(tf.reduce_all(tf.math.is_finite(_gradient_values(g))).numpy())
                for g in grads
            ):
                raise FloatingPointError(f"non-finite gradient at step {global_step + 1}")
            if grad_clip > 0:
                grads, _ = tf.clip_by_global_norm(grads, grad_clip)
            optimizer.apply_gradients(zip(grads, variables))
            global_step += 1
        else:
            src_mask, tgt_mask, memory_mask = model.create_masks(batch.src, batch.tgt_in)
            logits = model(
                batch.src, batch.tgt_in, src_mask, tgt_mask, memory_mask,
                training=False,
            )
            loss = loss_fn(batch.tgt_out, logits)
            if not bool(tf.reduce_all(tf.math.is_finite(loss)).numpy()):
                phase = "validation" if not training else "training"
                raise FloatingPointError(f"non-finite {phase} loss at step {global_step}")

        token_count = int(batch.num_tgt_tokens)
        total_loss += float(loss.numpy()) * token_count
        total_tokens += token_count

    elapsed = max(time.perf_counter() - started, 1e-9)
    mean_loss = total_loss / max(total_tokens, 1)
    return mean_loss, global_step, total_tokens / elapsed, elapsed


def _checkpoint_path(value: Path) -> str:
    """Accept a CheckpointManager directory or an explicit TF checkpoint prefix."""
    if value.is_dir():
        latest = tf.train.latest_checkpoint(str(value))
        if latest is None:
            raise SystemExit(f"No TensorFlow checkpoint found in directory: {value}")
        return latest
    if not Path(str(value) + ".index").is_file() and not value.is_file():
        raise SystemExit(f"Checkpoint not found: {value} (expected a TensorFlow checkpoint prefix)")
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    sample = ROOT / "data" / "sample"
    parser.add_argument("--train-src", type=Path, default=sample / "train.en")
    parser.add_argument("--train-tgt", type=Path, default=sample / "train.de")
    parser.add_argument("--val-src", type=Path, default=sample / "val.en")
    parser.add_argument("--val-tgt", type=Path, default=sample / "val.de")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "data" / "tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=512, help="Used only when training a tokenizer")
    parser.add_argument("--config", type=Path, help="Optional TransformerConfig JSON")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "tf")
    parser.add_argument("--resume", type=Path, help="Resume from a checkpoint prefix or manager directory")
    parser.add_argument(
        "--device",
        choices=("gpu", "cpu", "auto"),
        default="gpu",
        help="Defaults to GPU and fails instead of silently falling back; use cpu or auto explicitly",
    )
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-train-pairs", type=int, default=None,
        help="Use the first N aligned training pairs (deterministic subset)",
    )
    parser.add_argument(
        "--max-val-pairs", type=int, default=None,
        help="Use the first N aligned validation pairs (deterministic subset)",
    )
    parser.add_argument(
        "--sample-translations", type=int, default=3,
        help="Validation translations to print and append to translations.jsonl each epoch",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.epochs is not None and args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    for name in ("max_train_pairs", "max_val_pairs"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.sample_translations < 0:
        raise SystemExit("--sample-translations must be non-negative")
    for path in (args.train_src, args.train_tgt, args.val_src, args.val_tgt):
        if not path.is_file():
            raise SystemExit(f"Dataset file not found: {path}")
    selected_device = configure_device(args.device)

    train_pairs = read_parallel_corpus(args.train_src, args.train_tgt)
    val_pairs = read_parallel_corpus(args.val_src, args.val_tgt)
    if args.max_train_pairs is not None:
        train_pairs = train_pairs[:args.max_train_pairs]
    if args.max_val_pairs is not None:
        val_pairs = val_pairs[:args.max_val_pairs]
    if not train_pairs or not val_pairs:
        raise SystemExit("Training and validation datasets must each contain at least one non-empty pair")

    tokenizer = load_or_train_tokenizer(args.tokenizer, train_pairs, args.vocab_size)
    if args.config:
        config = TransformerConfig.load(args.config)
        if config.src_vocab_size != tokenizer.vocab_size or config.tgt_vocab_size != tokenizer.vocab_size:
            raise SystemExit(
                "Config vocabulary sizes must match tokenizer.vocab_size "
                f"({tokenizer.vocab_size}); update the config or tokenizer."
            )
    else:
        config = TransformerConfig(
            src_vocab_size=tokenizer.vocab_size, tgt_vocab_size=tokenizer.vocab_size
        )
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
    if args.max_len is not None:
        config.max_seq_len = args.max_len
    if args.seed is not None:
        config.seed = args.seed
    config.validate()
    if config.max_seq_len < 3:
        raise SystemExit("max_seq_len must be at least 3")
    seed_everything(config.seed)

    train_data = ParallelDataset(train_pairs, tokenizer, config.max_seq_len)
    val_data = ParallelDataset(val_pairs, tokenizer, config.max_seq_len)

    train_loader = TFBatchLoader(train_data, config.batch_size, shuffle=True, seed=config.seed)
    val_loader = TFBatchLoader(val_data, config.batch_size, shuffle=False, seed=config.seed)
    model = Transformer.from_config(config)
    # Build Keras variables before restoring a checkpoint or creating optimizer slots.
    first_batch = next(iter(val_loader))
    src_mask, tgt_mask, memory_mask = model.create_masks(first_batch.src, first_batch.tgt_in)
    model(first_batch.src, first_batch.tgt_in, src_mask, tgt_mask, memory_mask, training=False)

    schedule = NoamSchedule(config.d_model, config.warmup_steps)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=schedule, beta_1=config.adam_beta1, beta_2=config.adam_beta2,
        epsilon=config.adam_eps,
    )
    # Create optimizer slot variables before checkpoint restore. This makes
    # resume strict and avoids silently deferring momentum/variance restoration.
    if hasattr(optimizer, "build"):
        optimizer.build(model.trainable_variables)
    loss_fn = LabelSmoothingLoss(config.label_smoothing, pad_idx=config.pad_idx)
    epoch_var = tf.Variable(0, dtype=tf.int64, trainable=False, name="epoch")
    best_val_var = tf.Variable(float("inf"), dtype=tf.float32, trainable=False, name="best_val_loss")
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=optimizer, epoch=epoch_var, best_val_loss=best_val_var
    )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_manager = tf.train.CheckpointManager(
        checkpoint, str(args.checkpoint_dir / "latest"), max_to_keep=3
    )
    best_manager = tf.train.CheckpointManager(
        checkpoint, str(args.checkpoint_dir / "best"), max_to_keep=1
    )

    start_epoch = 0
    best_val_loss = math.inf
    if args.resume:
        resume_path = _checkpoint_path(args.resume)
        status = checkpoint.restore(resume_path)
        # Existing model/optimizer objects are already built. Keep partial restore
        # warnings visible in development but assert that existing objects match.
        status.assert_existing_objects_matched()
        start_epoch = int(epoch_var.numpy())
        best_val_loss = float(best_val_var.numpy())
        print(f"Resumed {resume_path} at epoch {start_epoch}, optimizer step {int(optimizer.iterations.numpy())}")

    history_path = args.checkpoint_dir / "history.jsonl"
    translations_path = args.checkpoint_dir / "translations.jsonl"
    if not args.resume:
        history_path.unlink(missing_ok=True)
        translations_path.unlink(missing_ok=True)
    config.save(args.checkpoint_dir / "config.json")
    print(
        f"Device: {selected_device} "
        f"({', '.join(d.name for d in tf.config.list_logical_devices())}) | "
        f"parameters: {model.count_params():,} | train pairs: {len(train_data)} | "
        f"validation pairs: {len(val_data)} | batch size: {config.batch_size} | "
        f"epochs: {config.num_epochs}"
    )
    if train_data.num_truncated or val_data.num_truncated:
        print(f"Warning: truncated pairs — train={train_data.num_truncated}, val={val_data.num_truncated}")

    global_step = int(optimizer.iterations.numpy())
    for epoch_index in range(start_epoch, config.num_epochs):
        epoch_number = epoch_index + 1
        train_loader.set_epoch(epoch_index)
        train_loss, global_step, train_tps, train_seconds = run_epoch(
            model, train_loader, loss_fn, optimizer, global_step, config.grad_clip
        )
        val_loss, _, val_tps, val_seconds = run_epoch(model, val_loader, loss_fn)
        current_lr = float(schedule(optimizer.iterations - 1).numpy()) if global_step else float(schedule(0).numpy())
        epoch_var.assign(epoch_number)
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_var.assign(best_val_loss)
            best_manager.save(checkpoint_number=epoch_number)
        latest_manager.save(checkpoint_number=epoch_number)
        print(
            f"epoch {epoch_number:03d}/{config.num_epochs:03d} | step {global_step:06d} | "
            f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | lr {current_lr:.3e} | "
            f"train {train_tps:.1f} tok/s ({train_seconds:.1f}s) | "
            f"val {val_tps:.1f} tok/s ({val_seconds:.1f}s)",
            flush=True,
        )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "epoch": epoch_number, "global_step": global_step,
                "train_loss": train_loss, "val_loss": val_loss,
                "learning_rate": current_lr, "train_tokens_per_sec": train_tps,
                "val_tokens_per_sec": val_tps,
            }) + "\n")
        samples = translation_samples(
            model, val_pairs, tokenizer, config, args.sample_translations
        )
        with translations_path.open("a", encoding="utf-8") as handle:
            for sample in samples:
                record = {"epoch": epoch_number, **sample}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"sample | src: {sample['source']} | ref: {sample['reference']} | "
                    f"pred: {sample['prediction']}",
                    flush=True,
                )
        save_loss_curve(history_path, args.checkpoint_dir / "loss_curve.svg")
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints and logs: {args.checkpoint_dir.resolve()}")


if __name__ == "__main__":
    main()
