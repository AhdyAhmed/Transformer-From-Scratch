# Day 22 — Extended TensorFlow CPU run

This directory contains the lightweight reproducibility artifacts from the
extended Day 22 TensorFlow run. Model and optimizer checkpoints remain under
`checkpoints/day21-medium/tf/` and are intentionally ignored by Git.

## Configuration

- Framework: TensorFlow 2.21 / Keras 3.13
- Device: native Windows CPU (`/device:CPU:0`; no TensorFlow GPU visible)
- Dataset: first 5,000 Multi30k training pairs and first 500 validation pairs
- Tokenizer: joint English/German from-scratch BPE, 512 tokens, selected training data only
- Model: 64 dimensions, 4 heads, 2 encoder and decoder layers, 256-dimensional FFN
- Parameters: 299,776
- Batch size: 64
- Maximum sequence length: 48 (144 training pairs and 16 validation pairs truncated)
- Epochs: 20 (resumed from the Day 21 epoch-10 latest checkpoint)
- Optimizer steps: 1,580

## Results

| Epoch | Step | Train loss | Validation loss | Train tokens/s |
|---:|---:|---:|---:|---:|
| 1 | 79 | 4.8390 | 4.1279 | 1,542 |
| 5 | 395 | 3.5340 | 3.6506 | 841 |
| 10 | 790 | 3.3631 | 3.4998 | 834 |
| 11 | 869 | 3.3395 | 3.4822 | 1,567 |
| 15 | 1,185 | 3.2747 | 3.4430 | 1,425 |
| 18 | 1,422 | 3.2386 | 3.4134 | 1,352 |
| 20 | 1,580 | **3.2172** | **3.4132** | 1,343 |

During the Day 22 continuation, training loss fell by 4.3% and validation loss
by 2.5%. Across the complete 20 epochs, the reductions are 33.5% and 17.3%.
All metrics remained finite, and direct checkpoint inspection confirmed epoch
20, optimizer step 1,580, and best validation loss 3.413247 in both latest and
best checkpoint managers.

The greedy translations remain generic and do not yet match the full-corpus
PyTorch model. This is consistent with the smaller CPU-bound dataset/model and
512-token vocabulary. Day 22 establishes a longer, resumable TensorFlow
baseline; formal cross-framework evaluation is reserved for Day 23.

Artifacts:

- `tokenizer.json` — exact 512-token vocabulary and learned BPE merges.
- `history.jsonl` — exact metrics for all 20 epochs.
- `translations.jsonl` — three deterministic validation samples per epoch.
- `loss_curve.svg` — complete train/validation loss curve.
