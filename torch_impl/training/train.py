"""Day 17: end-to-end PyTorch Transformer training with validation/checkpoints.

Run from the repository root:
  python -m torch_impl.training.train --epochs 1 --batch-size 4 --vocab-size 128
  python -m torch_impl.training.train --config configs/torch-small.json --resume checkpoints/latest.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.bpe_tokenizer import BPETokenizer
from data.parallel_data import ParallelDataset, read_parallel_corpus
from torch_impl.config import TransformerConfig
from torch_impl.data.dataset import TorchBatchLoader
from torch_impl.model.transformer import Transformer
from torch_impl.training.losses import LabelSmoothingLoss
from torch_impl.training.lr_schedule import NoamSchedule

ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_or_train_tokenizer(
    path: Path, train_pairs: list[tuple[str, str]], vocab_size: int
) -> BPETokenizer:
    """Load an existing BPE tokenizer or train it on training text only."""
    if path.exists():
        tokenizer = BPETokenizer.load(path)
        print(f"Loaded tokenizer: {path} ({tokenizer.vocab_size} tokens)")
        return tokenizer
    corpus = [text for pair in train_pairs for text in pair]
    # Fit the tokenizer on training text only to avoid validation-data leakage.
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=vocab_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(path)
    print(f"Trained tokenizer: {path} ({tokenizer.vocab_size} tokens from {len(corpus)} lines)")
    return tokenizer


@torch.inference_mode()
def greedy_decode(
    model: Transformer,
    src_ids: list[int],
    bos_idx: int,
    eos_idx: int,
    max_len: int,
    device: torch.device,
) -> list[int]:
    """Greedily decode one already-tokenized source sequence."""
    if max_len < 2:
        raise ValueError(f"max_len must be at least 2, got {max_len}")
    was_training = model.training
    model.eval()
    try:
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = (src != model.pad_idx).unsqueeze(1).unsqueeze(2)
        memory = model.encode(src, src_mask)
        generated = torch.tensor([[bos_idx]], dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            _, tgt_mask, memory_mask = model.create_masks(src, generated)
            decoded = model.decode(generated, memory, tgt_mask, memory_mask)
            next_token = model.output_projection(decoded[:, -1]).argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if int(next_token.item()) == eos_idx:
                break
        return generated[0].tolist()
    finally:
        model.train(was_training)


def translation_samples(
    model: Transformer,
    pairs: list[tuple[str, str]],
    tokenizer: BPETokenizer,
    config: TransformerConfig,
    device: torch.device,
    count: int,
) -> list[dict[str, str]]:
    """Translate the first deterministic validation examples for progress logs."""
    if count < 0:
        raise ValueError(f"sample translation count must be non-negative, got {count}")
    dataset = ParallelDataset(pairs[:count], tokenizer, config.max_seq_len)
    records = []
    for (source, reference), (src_ids, _) in zip(pairs[:count], dataset.examples):
        predicted_ids = greedy_decode(
            model,
            src_ids,
            config.bos_idx,
            config.eos_idx,
            config.max_seq_len,
            device,
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
    # Matplotlib emits trailing spaces in SVG path data. Normalize the text
    # so generated reports pass Git's whitespace checks and produce clean diffs.
    svg = output_path.read_text(encoding="utf-8")
    output_path.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
    )


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def save_checkpoint(path: Path, model: Transformer, optimizer: torch.optim.Optimizer,
                    config: TransformerConfig, epoch: int, global_step: int,
                    best_val_loss: float, tokenizer_path: Path) -> None:
    """Atomically save model, optimizer, progress, config, and RNG state."""
    _atomic_torch_save({
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "tokenizer_path": str(tokenizer_path),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }, path)


def load_checkpoint(path: Path, model: Transformer, optimizer: torch.optim.Optimizer,
                    device: torch.device) -> dict[str, Any]:
    """Restore a supported training checkpoint and all saved RNG state."""
    # Checkpoints are user-created local files and include Python/NumPy RNG tuples.
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 2.0 compatibility
        payload = torch.load(path, map_location=device)
    if "model_state_dict" not in payload or "optimizer_state_dict" not in payload:
        raise ValueError(f"{path} is not a supported training checkpoint")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if payload.get("torch_rng_state") is not None:
        torch.set_rng_state(payload["torch_rng_state"].cpu())
    if payload.get("numpy_rng_state") is not None:
        np.random.set_state(payload["numpy_rng_state"])
    if payload.get("python_rng_state") is not None:
        random.setstate(payload["python_rng_state"])
    return payload


def run_epoch(model: Transformer, loader: TorchBatchLoader, loss_fn: LabelSmoothingLoss,
              device: torch.device, optimizer: torch.optim.Optimizer | None = None,
              schedule: NoamSchedule | None = None, global_step: int = 0,
              grad_clip: float = 1.0) -> tuple[float, int, float, float]:
    """Run one train or validation epoch and return loss/progress metrics."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    started = time.perf_counter()
    for batch in loader:
        batch = batch.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        src_mask, tgt_mask, memory_mask = model.create_masks(batch.src, batch.tgt_in)
        with torch.set_grad_enabled(training):
            logits = model(batch.src, batch.tgt_in, src_mask, tgt_mask, memory_mask)
            loss = loss_fn(logits, batch.tgt_out)
            if training:
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at step {global_step + 1}: {loss.item()}")
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if schedule is not None:
                    schedule.apply(optimizer, global_step + 1)
                optimizer.step()
                global_step += 1
        n_tokens = batch.num_tgt_tokens
        total_loss += float(loss.detach().item()) * n_tokens
        total_tokens += n_tokens
    elapsed = max(time.perf_counter() - started, 1e-9)
    mean_loss = total_loss / max(total_tokens, 1)
    return mean_loss, global_step, total_tokens / elapsed, elapsed


def build_parser() -> argparse.ArgumentParser:
    """Build the end-to-end PyTorch training CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "torch")
    parser.add_argument("--resume", type=Path, help="Resume from a latest/best checkpoint")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device. Defaults to CUDA and fails if no CUDA GPU is available; "
             "use 'cpu' or 'auto' explicitly to allow CPU execution",
    )
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=None, help="Optional intra-op CPU thread limit")
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
    """Train or resume a PyTorch Transformer from CLI-style arguments."""
    args = build_parser().parse_args(argv)
    if args.epochs is not None and args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    for name in ("max_train_pairs", "max_val_pairs"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.sample_translations < 0:
        raise SystemExit("--sample-translations must be non-negative")
    for p in (args.train_src, args.train_tgt, args.val_src, args.val_tgt):
        if not p.is_file():
            raise SystemExit(f"Dataset file not found: {p}")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available in this PyTorch installation")
    if args.num_threads is not None:
        if args.num_threads < 1:
            raise SystemExit("--num-threads must be positive")
        torch.set_num_threads(args.num_threads)

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
            raise SystemExit("Config vocabulary sizes must match tokenizer.vocab_size "
                             f"({tokenizer.vocab_size}); update the config or tokenizer.")
    else:
        config = TransformerConfig(src_vocab_size=tokenizer.vocab_size,
                                   tgt_vocab_size=tokenizer.vocab_size)
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
    train_loader = TorchBatchLoader(train_data, config.batch_size, shuffle=True, seed=config.seed, device=device)
    val_loader = TorchBatchLoader(val_data, config.batch_size, shuffle=False, seed=config.seed, device=device)

    model = Transformer.from_config(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(config.adam_beta1, config.adam_beta2),
                                 eps=config.adam_eps)
    schedule = NoamSchedule(config.d_model, config.warmup_steps)
    loss_fn = LabelSmoothingLoss(config.label_smoothing, pad_idx=config.pad_idx)
    start_epoch, global_step, best_val_loss = 0, 0, math.inf
    if args.resume:
        if not args.resume.is_file():
            raise SystemExit(f"Checkpoint not found: {args.resume}")
        payload = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = int(payload["epoch"])
        global_step = int(payload["global_step"])
        best_val_loss = float(payload.get("best_val_loss", math.inf))
        print(f"Resumed {args.resume} at epoch {start_epoch}, step {global_step}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.checkpoint_dir / "history.jsonl"
    translations_path = args.checkpoint_dir / "translations.jsonl"
    if not args.resume:
        history_path.unlink(missing_ok=True)
        translations_path.unlink(missing_ok=True)
    config.save(args.checkpoint_dir / "config.json")
    print(f"Device: {device} | parameters: {sum(p.numel() for p in model.parameters()):,} | "
          f"train pairs: {len(train_data)} | validation pairs: {len(val_data)} | "
          f"batch size: {config.batch_size} | epochs: {config.num_epochs}")
    if train_data.num_truncated or val_data.num_truncated:
        print(f"Warning: truncated pairs — train={train_data.num_truncated}, val={val_data.num_truncated}")

    for epoch_index in range(start_epoch, config.num_epochs):
        epoch_number = epoch_index + 1
        train_loader.set_epoch(epoch_index)
        train_loss, global_step, train_tps, train_seconds = run_epoch(
            model, train_loader, loss_fn, device, optimizer, schedule, global_step, config.grad_clip)
        val_loss, _, val_tps, val_seconds = run_epoch(model, val_loader, loss_fn, device)
        current_lr = schedule.rate(max(global_step, 1))
        print(f"epoch {epoch_number:03d}/{config.num_epochs:03d} | step {global_step:06d} | "
              f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | lr {current_lr:.3e} | "
              f"train {train_tps:.1f} tok/s ({train_seconds:.1f}s) | "
              f"val {val_tps:.1f} tok/s ({val_seconds:.1f}s)", flush=True)
        save_checkpoint(args.checkpoint_dir / "latest.pt", model, optimizer, config,
                        epoch_number, global_step, min(best_val_loss, val_loss), args.tokenizer)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(args.checkpoint_dir / "best.pt", model, optimizer, config,
                            epoch_number, global_step, best_val_loss, args.tokenizer)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch_number, "global_step": global_step,
                                     "train_loss": train_loss, "val_loss": val_loss,
                                     "learning_rate": current_lr, "train_tokens_per_sec": train_tps,
                                     "val_tokens_per_sec": val_tps}, ensure_ascii=False) + "\n")
        samples = translation_samples(
            model, val_pairs, tokenizer, config, device, args.sample_translations
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
