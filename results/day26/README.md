# Day 26 — Inference CLI validation

Both standalone framework CLIs were executed against their real best
checkpoints with the same English sentence:

> A man sleeping in a green room on a couch.

| Framework | Device | Translation | EOS | Source truncated |
|---|---|---|---:|---:|
| PyTorch | CUDA | ein mann schläft in einem grünen zimmer auf einem sofa . | yes | no |
| TensorFlow | CPU | ein mann mit einem roten hemd , der einen kopf auf dem boden . | yes | no |

The checked JSON files retain token counts and restored checkpoint metadata:

- [`pytorch.json`](pytorch.json) — epoch 30, step 13,620, best validation loss 2.3910.
- [`tensorflow.json`](tensorflow.json) — epoch 20, step 1,580, best validation loss 3.4132.

The scripts validate checkpoint/tokenizer vocabulary and special-token ids,
preserve BOS/EOS when truncating long sources, bound output length to the
model's positional capacity, and reject an explicit unavailable GPU instead of
silently falling back.

## Reproduce

```bash
python -m torch_impl.inference \
  --text "A man sleeping in a green room on a couch." \
  --device cuda --output results/day26/pytorch.json

python -m tf_impl.inference \
  --text "A man sleeping in a green room on a couch." \
  --device cpu --output results/day26/tensorflow.json
```

TensorFlow may use `--device gpu` on a supported Linux/WSL2 GPU environment.
The current native Windows TensorFlow installation is CPU-only.
