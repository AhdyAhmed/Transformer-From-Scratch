# TensorFlow/Keras implementation

The TensorFlow side mirrors the PyTorch architecture without
`keras.layers.MultiHeadAttention` or `model.fit`. It uses custom Keras layers,
an explicit `GradientTape` loop, checkpoint managers, greedy inference, and
framework-specific tests.

Run every command from the repository root.

## Environment

### Native Windows or CPU-only

```powershell
python -m venv .venv-tf
.venv-tf\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-tf.txt
```

### NVIDIA GPU on Linux/WSL2

```bash
python -m venv .venv-tf
source .venv-tf/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-tf-gpu.txt
```

TensorFlow 2.11 and later do not provide native-Windows CUDA support. Training
defaults to strict `--device gpu`; use `--device cpu` intentionally on native
Windows or `--device auto` when fallback is acceptable.

## Quick smoke training

```powershell
python -m tf_impl.training.train --device cpu --epochs 1 --batch-size 4 --vocab-size 64 --max-len 16 --warmup-steps 4 --tokenizer checkpoints/smoke-tf/tokenizer.json --checkpoint-dir checkpoints/smoke-tf
```

Outputs include `latest/` and `best/` checkpoint-manager directories plus
`config.json`, `history.jsonl`, `translations.jsonl`, and `loss_curve.svg`.

## Multi30k training

```powershell
python -m data.download_multi30k

python -m tf_impl.training.train --config configs/day22-tf.json --train-src data/multi30k/train.en --train-tgt data/multi30k/train.de --val-src data/multi30k/val.en --val-tgt data/multi30k/val.de --max-train-pairs 5000 --max-val-pairs 500 --tokenizer checkpoints/day21-medium/tokenizer.json --checkpoint-dir checkpoints/day21-medium/tf --vocab-size 512 --device cpu
```

The completed native-Windows CPU run used 20 epochs. Compact metrics are in
[`results/day22/`](../results/day22/); the full checkpoint remains Git-ignored.

## Resume

`--epochs` is the final target epoch:

```powershell
python -m tf_impl.training.train --resume checkpoints/day21-medium/tf/latest --tokenizer checkpoints/day21-medium/tokenizer.json --checkpoint-dir checkpoints/day21-medium/tf --train-src data/multi30k/train.en --train-tgt data/multi30k/train.de --val-src data/multi30k/val.en --val-tgt data/multi30k/val.de --max-train-pairs 5000 --max-val-pairs 500 --epochs 25 --device cpu
```

The full trackable graph is rebuilt before restore: model variables, optimizer
slots/iterations, epoch, and best validation loss.

## Translate

```powershell
python -m tf_impl.inference --text "A man sleeping in a green room on a couch." --checkpoint checkpoints/day21-medium/tf/best --config checkpoints/day21-medium/tf/config.json --tokenizer checkpoints/day21-medium/tokenizer.json --device cpu --output translation.json
```

Use `--device gpu` on a supported Linux/WSL2 GPU environment. Explicit GPU
requests fail clearly when no TensorFlow GPU is visible.

## Tests

```bash
make test-tf
make test-tf-fast   # skips the slow overfit test

# Direct equivalents
python -m pytest tests tf_impl
python -m pytest tests tf_impl -m "not slow"
```

## Package map

| Path | Responsibility |
|---|---|
| `config.py` | Validated model/training configuration |
| `model/` | Masks, embeddings, attention, FFN, encoder, decoder, full model |
| `data/dataset.py` | Convert shared deterministic batches to `tf.int32` tensors |
| `training/` | Noam schedule, smoothed loss, `GradientTape` training/checkpoint CLI |
| `inference.py` | Full-object-graph restoration and single-sentence translation |
| `tests/` | Component, training, checkpoint, overfit, and inference regression tests |

See the root [`README.md`](../README.md) and [`WRITEUP.md`](../WRITEUP.md) for
architecture, measured results, and framework tradeoffs.
