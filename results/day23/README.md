# Day 23 — Shared-corpus evaluation

This directory contains the reproducible evaluation of the best PyTorch and
TensorFlow checkpoints on the same first 100 Multi30k validation examples.

## Metric

The score is lowercase corpus BLEU-4 using the repository's Unicode
word/punctuation tokenizer, clipped corpus n-gram counts, the standard brevity
penalty, and exponential zero-count smoothing.

| Framework | BLEU-4 | 1-gram | 2-gram | 3-gram | 4-gram | BP | Length ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch | **37.8505** | 67.89 | 44.56 | 31.38 | 23.49 | 0.9795 | 0.9798 |
| TensorFlow | **1.6422** | 22.61 | 3.07 | 0.70 | 0.15 | 1.0000 | 1.2710 |

PyTorch used its epoch-30/step-13,620 best checkpoint and decoded the 100
sentences in 51.10 seconds (1.957 sentences/s) on CUDA. TensorFlow used its
epoch-20/step-1,580 best checkpoint and decoded in 364.34 seconds (0.274
sentences/s) on native-Windows CPU.

## Interpretation

This comparison answers which saved project checkpoint currently performs
better; it does not isolate PyTorch versus TensorFlow. The PyTorch checkpoint
was trained on all 29,000 pairs with a 2,048-token vocabulary and 1.92M
parameters. The CPU-bounded TensorFlow checkpoint used 5,000 pairs, a
512-token vocabulary, and 299,776 parameters. Those differences explain much
of the quality and speed gap.

Artifacts:

- `metrics.json` — exact score components, lengths, checkpoint identities, and timing.
- `translations.jsonl` — all 100 source/reference/PyTorch/TensorFlow records.
- `comparison.md` — metrics plus ten representative side-by-side examples.
