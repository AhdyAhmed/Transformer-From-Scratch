"""Day 24: extract and plot attention from trained Transformer checkpoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import torch

from data.bpe_tokenizer import END_OF_WORD, BPETokenizer
from data.parallel_data import ParallelDataset, read_parallel_corpus
from evaluation.evaluate import _load_tf_model, _load_torch_model
from tf_impl.training.train import configure_device as configure_tf_device

ROOT = Path(__file__).resolve().parents[1]


def framework_label(name: str) -> str:
    return {"pytorch": "PyTorch", "tensorflow": "TensorFlow"}.get(
        name, name.title()
    )


@dataclass
class AttentionRecord:
    """One translation and its per-head attention probability matrices."""

    framework: str
    source: str
    reference: str
    translation: str
    layer: int
    source_tokens: list[str]
    target_tokens: list[str]
    encoder_self: np.ndarray
    decoder_self: np.ndarray
    decoder_cross: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "source": self.source,
            "reference": self.reference,
            "translation": self.translation,
            "layer": self.layer,
            "source_tokens": self.source_tokens,
            "target_tokens": self.target_tokens,
            "encoder_self": self.encoder_self.tolist(),
            "decoder_self": self.decoder_self.tolist(),
            "decoder_cross": self.decoder_cross.tolist(),
        }


def token_labels(tokenizer: BPETokenizer, ids: list[int]) -> list[str]:
    """Return readable labels while retaining BPE word-boundary information."""
    labels = []
    for token_id in ids:
        symbol = tokenizer.inverse_vocab.get(token_id, "<unk>")
        labels.append(symbol.replace(END_OF_WORD, "▁"))
    return labels


def resolve_layer(layer: int, num_layers: int) -> int:
    """Resolve a possibly negative layer index and fail with a useful message."""
    resolved = layer if layer >= 0 else num_layers + layer
    if not 0 <= resolved < num_layers:
        raise ValueError(
            f"layer index {layer} is out of range for a {num_layers}-layer model"
        )
    return resolved


def validate_attention(record: AttentionRecord, atol: float = 1e-5) -> None:
    """Validate shapes and the softmax probability invariant for every map."""
    expected = {
        "encoder_self": (len(record.source_tokens), len(record.source_tokens)),
        "decoder_self": (len(record.target_tokens), len(record.target_tokens)),
        "decoder_cross": (len(record.target_tokens), len(record.source_tokens)),
    }
    for name, trailing_shape in expected.items():
        weights = getattr(record, name)
        if weights.ndim != 3 or weights.shape[1:] != trailing_shape:
            raise ValueError(
                f"{name} has shape {weights.shape}; expected "
                f"(heads, {trailing_shape[0]}, {trailing_shape[1]})"
            )
        if not np.isfinite(weights).all():
            raise ValueError(f"{name} contains non-finite values")
        if (weights < -atol).any() or (weights > 1.0 + atol).any():
            raise ValueError(f"{name} contains values outside [0, 1]")
        if not np.allclose(weights.sum(axis=-1), 1.0, atol=atol):
            raise ValueError(f"{name} rows do not sum to one")
    future_attention = np.triu(record.decoder_self, k=1)
    if not np.allclose(future_attention, 0.0, atol=atol):
        raise ValueError("decoder_self leaks attention to future target tokens")


def _torch_array(value: torch.Tensor | None, name: str) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"PyTorch did not cache {name} attention weights")
    return value[0].detach().cpu().numpy().astype(np.float32, copy=True)


@torch.inference_mode()
def extract_torch_attention(
    model,
    config,
    tokenizer: BPETokenizer,
    source: str,
    reference: str,
    device: torch.device,
    layer: int = -1,
) -> AttentionRecord:
    """Greedily translate one sentence and capture a selected layer's maps."""
    layer_index = resolve_layer(layer, len(model.encoder.layers))
    src_ids, _ = ParallelDataset(
        [(source, reference)], tokenizer, config.max_seq_len
    ).examples[0]
    was_training = model.training
    model.eval()
    try:
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = (src != model.pad_idx).unsqueeze(1).unsqueeze(2)
        memory = model.encode(src, src_mask)
        encoder_self = _torch_array(
            model.encoder.layers[layer_index].self_attn.attn_weights,
            "encoder self",
        )
        generated = torch.tensor([[config.bos_idx]], dtype=torch.long, device=device)
        for _ in range(config.max_seq_len - 1):
            _, tgt_mask, memory_mask = model.create_masks(src, generated)
            decoded = model.decode(generated, memory, tgt_mask, memory_mask)
            next_token = model.output_projection(decoded[:, -1]).argmax(
                dim=-1, keepdim=True
            )
            generated = torch.cat((generated, next_token), dim=1)
            if int(next_token.item()) == config.eos_idx:
                break
        # The final decoder call queried every token except the newly predicted one.
        target_ids = generated[0, :-1].tolist()
        decoder_layer = model.decoder.layers[layer_index]
        record = AttentionRecord(
            framework="pytorch",
            source=source,
            reference=reference,
            translation=tokenizer.decode(generated[0].tolist()),
            layer=layer_index,
            source_tokens=token_labels(tokenizer, src_ids),
            target_tokens=token_labels(tokenizer, target_ids),
            encoder_self=encoder_self,
            decoder_self=_torch_array(
                decoder_layer.self_attn.attn_weights, "decoder self"
            ),
            decoder_cross=_torch_array(
                decoder_layer.cross_attn.attn_weights, "decoder cross"
            ),
        )
    finally:
        model.train(was_training)
    validate_attention(record)
    return record


def _tf_array(value: tf.Tensor | None, name: str) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"TensorFlow did not cache {name} attention weights")
    return np.asarray(value[0].numpy(), dtype=np.float32).copy()


def extract_tf_attention(
    model,
    config,
    tokenizer: BPETokenizer,
    source: str,
    reference: str,
    layer: int = -1,
) -> AttentionRecord:
    """TensorFlow equivalent of :func:`extract_torch_attention`."""
    layer_index = resolve_layer(layer, len(model.encoder.enc_layers))
    src_ids, _ = ParallelDataset(
        [(source, reference)], tokenizer, config.max_seq_len
    ).examples[0]
    src = tf.constant([src_ids], dtype=tf.int32)
    src_mask = tf.expand_dims(tf.expand_dims(src != model.pad_idx, 1), 1)
    memory = model.encode(src, src_mask, training=False)
    encoder_self = _tf_array(
        model.encoder.enc_layers[layer_index].self_attn.attn_weights,
        "encoder self",
    )
    generated = tf.constant([[config.bos_idx]], dtype=tf.int32)
    for _ in range(config.max_seq_len - 1):
        _, tgt_mask, memory_mask = model.create_masks(src, generated)
        decoded = model.decode(
            generated, memory, tgt_mask, memory_mask, training=False
        )
        next_token = tf.argmax(
            model.output_projection(decoded[:, -1]),
            axis=-1,
            output_type=tf.int32,
        )[:, None]
        generated = tf.concat((generated, next_token), axis=1)
        if int(next_token[0, 0].numpy()) == config.eos_idx:
            break
    generated_ids = generated[0].numpy().tolist()
    target_ids = generated_ids[:-1]
    decoder_layer = model.decoder.dec_layers[layer_index]
    record = AttentionRecord(
        framework="tensorflow",
        source=source,
        reference=reference,
        translation=tokenizer.decode(generated_ids),
        layer=layer_index,
        source_tokens=token_labels(tokenizer, src_ids),
        target_tokens=token_labels(tokenizer, target_ids),
        encoder_self=encoder_self,
        decoder_self=_tf_array(
            decoder_layer.self_attn.attn_weights, "decoder self"
        ),
        decoder_cross=_tf_array(
            decoder_layer.cross_attn.attn_weights, "decoder cross"
        ),
    )
    validate_attention(record)
    return record


def plot_attention(record: AttentionRecord, output_path: Path) -> None:
    """Plot head-mean encoder, causal decoder, and cross-attention heatmaps."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    longest = max(len(record.source_tokens), len(record.target_tokens))
    figure, axes = plt.subplots(
        1, 3, figsize=(max(15.0, longest * 0.48), 5.8), constrained_layout=True
    )
    panels = (
        (record.encoder_self.mean(axis=0), record.source_tokens,
         record.source_tokens, "Encoder self-attention"),
        (record.decoder_self.mean(axis=0), record.target_tokens,
         record.target_tokens, "Masked decoder self-attention"),
        (record.decoder_cross.mean(axis=0), record.source_tokens,
         record.target_tokens, "Decoder cross-attention"),
    )
    for axis, (weights, x_labels, y_labels, title) in zip(axes, panels):
        image = axis.imshow(weights, cmap="magma", vmin=0.0, aspect="auto")
        axis.set_title(
            f"{title}\n(mean of {record.encoder_self.shape[0]} heads)"
        )
        axis.set_xticks(range(len(x_labels)), x_labels, rotation=60, ha="right", fontsize=7)
        axis.set_yticks(range(len(y_labels)), y_labels, fontsize=7)
        axis.set_xlabel("Key tokens")
        axis.set_ylabel("Query tokens")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        f"{framework_label(record.framework)} · layer {record.layer + 1}\n"
        f"{record.source} → {record.translation}",
        fontsize=11,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_results(records: list[AttentionRecord], output_dir: Path) -> None:
    """Persist exact matrices, plots, and a compact Markdown index."""
    if not records:
        raise ValueError("at least one attention record is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "attention.json").write_text(
        json.dumps([record.to_dict() for record in records], ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    rows = [
        "# Day 24 — Attention visualization",
        "",
        "Each figure uses the selected layer and averages its individual attention "
        "heads for readability. Exact per-head matrices are stored in `attention.json`.",
        "",
    ]
    framework_counts: dict[str, int] = {}
    for record in records:
        index = framework_counts.get(record.framework, 0) + 1
        framework_counts[record.framework] = index
        filename = f"{record.framework}-example-{index}.png"
        plot_attention(record, output_dir / filename)
        rows.extend([
            f"## {framework_label(record.framework)} example {index}",
            "",
            f"- Source: {record.source}",
            f"- Reference: {record.reference}",
            f"- Translation: {record.translation}",
            f"- Layer: {record.layer + 1}",
            "",
            f"![{record.framework} attention example {index}]({filename})",
            "",
        ])
    (output_dir / "README.md").write_text("\n".join(rows), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    multi30k = ROOT / "data" / "multi30k"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-src", type=Path, default=multi30k / "val.en")
    parser.add_argument("--val-tgt", type=Path, default=multi30k / "val.de")
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument(
        "--torch-checkpoint", type=Path,
        default=ROOT / "checkpoints/day20-full/torch/best.pt",
    )
    parser.add_argument(
        "--torch-tokenizer", type=Path,
        default=ROOT / "checkpoints/day20-full/tokenizer.json",
    )
    parser.add_argument("--torch-device", default="cuda")
    parser.add_argument(
        "--tf-checkpoint", type=Path,
        default=ROOT / "checkpoints/day21-medium/tf/best",
    )
    parser.add_argument(
        "--tf-config", type=Path,
        default=ROOT / "checkpoints/day21-medium/tf/config.json",
    )
    parser.add_argument(
        "--tf-tokenizer", type=Path,
        default=ROOT / "checkpoints/day21-medium/tokenizer.json",
    )
    parser.add_argument("--tf-device", choices=("cpu", "gpu", "auto"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/day24")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.examples < 1:
        raise SystemExit("--examples must be positive")
    required = (
        args.val_src, args.val_tgt, args.torch_checkpoint, args.torch_tokenizer,
        args.tf_config, args.tf_tokenizer,
    )
    for path in required:
        if not path.is_file():
            raise SystemExit(f"Required file not found: {path}")
    configure_tf_device(args.tf_device)
    torch_device = torch.device(args.torch_device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested for PyTorch visualization but is not available")
    pairs = read_parallel_corpus(args.val_src, args.val_tgt)[:args.examples]
    if len(pairs) < args.examples:
        raise SystemExit(
            f"requested {args.examples} examples, but the corpus has only {len(pairs)}"
        )

    torch_tokenizer = BPETokenizer.load(args.torch_tokenizer)
    torch_model, torch_config, _ = _load_torch_model(
        args.torch_checkpoint, torch_device
    )
    records = [
        extract_torch_attention(
            torch_model, torch_config, torch_tokenizer, source, reference,
            torch_device, args.layer,
        )
        for source, reference in pairs
    ]

    tf_tokenizer = BPETokenizer.load(args.tf_tokenizer)
    tf_model, tf_config, _, tf_restore_context = _load_tf_model(
        args.tf_config, args.tf_checkpoint
    )
    records.extend(
        extract_tf_attention(
            tf_model, tf_config, tf_tokenizer, source, reference, args.layer
        )
        for source, reference in pairs
    )
    write_results(records, args.output_dir)
    assert tf_restore_context is not None
    print(f"Validated and rendered {len(records)} attention records")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
