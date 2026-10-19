# PyTorch implementation

The PyTorch side implements the complete encoder-decoder Transformer without
`nn.Transformer` or `nn.MultiheadAttention`. It includes the model, shared-data
adapter, explicit training loop, atomic checkpointing/resume, greedy inference,
and framework-specific tests.

Run every command from the repository root.

## Environment

```powershell
python -m venv .venv-torch
.venv-torch\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-torch.txt
```

On Linux/macOS, activate with `source .venv-torch/bin/activate`.

PyTorch training defaults to `--device cuda` and fails if CUDA is unavailable.
Pass `--device cpu` for an intentional CPU run or `--device auto` to permit
fallback. The completed full-corpus run used a Quadro M2000M CUDA GPU.

## Quick smoke training

The bundled sample corpus needs no download:

```powershell
python -m torch_impl.training.train --device cpu --num-threads 1 --epochs 1 --batch-size 4 --vocab-size 64 --max-len 16 --warmup-steps 4 --tokenizer checkpoints/smoke-torch/tokenizer.json --checkpoint-dir checkpoints/smoke-torch
```

Outputs include `latest.pt`, `best.pt`, `config.json`, `history.jsonl`,
`translations.jsonl`, and `loss_curve.svg`.

## Multi30k training

```powershell
python -m data.download_multi30k

python -m torch_impl.training.train --config configs/day20-full-torch.json --train-src data/multi30k/train.en --train-tgt data/multi30k/train.de --val-src data/multi30k/val.en --val-tgt data/multi30k/val.de --tokenizer checkpoints/day20-full/tokenizer.json --checkpoint-dir checkpoints/day20-full/torch --vocab-size 2048 --device cuda
```

The checked configuration describes the completed 30-epoch run. Data and
checkpoints are intentionally Git-ignored; compact metrics live under
[`results/day20-full/`](../results/day20-full/).

## Resume

`--epochs` is the final target epoch, not an additional epoch count:

```powershell
python -m torch_impl.training.train --resume checkpoints/day20-full/torch/latest.pt --tokenizer checkpoints/day20-full/tokenizer.json --checkpoint-dir checkpoints/day20-full/torch --train-src data/multi30k/train.en --train-tgt data/multi30k/train.de --val-src data/multi30k/val.en --val-tgt data/multi30k/val.de --epochs 35 --device cuda
```

Resume restores model, optimizer, epoch, global step, best validation loss,
and Python/NumPy/PyTorch random states.

## Translate

```powershell
python -m torch_impl.inference --text "A man sleeping in a green room on a couch." --checkpoint checkpoints/day20-full/torch/best.pt --tokenizer checkpoints/day20-full/tokenizer.json --device cuda --output translation.json
```

The JSON output records token counts, truncation/EOS status, device, and
checkpoint metadata. `--max-output-tokens` may shorten decoding but cannot
exceed the checkpoint's positional capacity.

## Tests

```bash
make test-torch
make test-torch-fast   # skips the slow overfit test

# Direct equivalents
python -m pytest tests torch_impl
python -m pytest tests torch_impl -m "not slow"
```

## Package map

| Path | Responsibility |
|---|---|
| `config.py` | Validated model/training configuration |
| `model/` | Masks, embeddings, attention, FFN, encoder, decoder, full model |
| `data/dataset.py` | Convert shared deterministic batches to `torch.long` tensors |
| `training/` | Noam schedule, label-smoothed loss, training/checkpoint CLI |
| `inference.py` | Strict checkpoint loading and single-sentence translation |
| `tests/` | Component, training, checkpoint, overfit, and inference regression tests |

See the root [`README.md`](../README.md) for architecture, results, and the
cross-framework test strategy.
