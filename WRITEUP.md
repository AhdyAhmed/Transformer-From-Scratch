# Transformer From Scratch — Design and Framework Comparison

This project implements the same encoder-decoder Transformer directly in
PyTorch and TensorFlow/Keras. It intentionally avoids `nn.Transformer`,
`keras.layers.MultiHeadAttention`, `model.fit`, and trainer frameworks so the
attention math, masks, residual paths, loss, schedule, and optimization steps
remain visible.

## What is genuinely comparable

The source implementations are designed for structural and numerical
comparison. They share the BPE tokenizer, parallel-corpus reader, batching
order, special-token ids, boolean-mask convention, configuration fields, and
the mathematical definitions of every model component. Weight-sync tests copy
the same parameters into both models and compare their outputs through the
complete Transformer.

The recorded production training runs are **not** a controlled framework
benchmark. PyTorch trained a 1.92M-parameter pre-norm model on all 29,000
Multi30k pairs using a Quadro M2000M GPU. TensorFlow trained a 300k-parameter
post-norm model on 5,000 pairs using a native Windows CPU. Their speed and BLEU
differences therefore describe these two checkpoints, not an inherent
PyTorch-versus-TensorFlow ranking.

## Core design decisions

### Manual multi-head attention

Each implementation projects queries, keys, and values with one dense matrix
per projection, reshapes the result into heads, applies
`softmax(QKᵀ / sqrt(d_k))`, and merges the heads before the output projection.
One combined projection is equivalent to separate per-head matrices but uses
fewer framework operations. Attention returns and caches the post-softmax,
pre-dropout probabilities, making masking tests and Day 24 visualizations
possible without hooks or architecture changes.

### Fixed sinusoidal position encoding

Position encodings are deterministic and contain no learned parameters. This
makes sequences position-aware while keeping cross-framework parity simple:
identical inputs must produce identical embeddings after matching token tables.
The implementation precomputes values up to `max_seq_len` and rejects sequences
that exceed that configured boundary through the data pipeline.

### Boolean masks with one meaning

Every mask uses `True = may attend` and broadcasts to
`(batch, heads, query_length, key_length)`. Encoder masks block source padding;
decoder self-attention combines target padding with a lower-triangular causal
mask; cross-attention blocks source padding. Tests perturb masked and unmasked
positions independently, which catches wrong-axis and inverted-mask bugs that
shape checks alone miss.

### Configurable pre-norm and post-norm

Both implementations support:

- Post-norm: `norm(x + dropout(sublayer(x)))`, matching the original paper.
- Pre-norm: `x + dropout(sublayer(norm(x)))`, generally easier to optimize at
  depth because the residual gradient has a direct route through the stack.

The stack applies a final LayerNorm in both modes. It is redundant but harmless
after post-norm layers and necessary after a pre-norm residual stream. The
full-corpus PyTorch run used pre-norm; the compact TensorFlow run used
post-norm. That difference is another reason their quality results cannot be
interpreted as framework-only.

LayerNorm epsilon is fixed at `1e-6` in both codebases instead of accepting the
different framework defaults (`1e-5` in PyTorch and `1e-3` in Keras).

### Shared embeddings and from-scratch BPE

English and German use one deterministic BPE vocabulary trained only on the
training split. Source and target embedding tables can therefore be shared.
The tokenizer breaks merge-frequency ties lexicographically, so repeated
training on the same corpus is byte-for-byte reproducible. BOS, EOS, PAD, and
UNK ids are fixed at 1, 2, 0, and 3 respectively and cross-checked against both
framework configurations.

### Noam warmup schedule

Both optimizers use the schedule from the Transformer paper:

```text
learning_rate(step) = d_model^(-1/2) × min(step^(-1/2),
                                                step × warmup^(-3/2))
```

The public PyTorch schedule receives a one-based optimizer step. Keras exposes
zero-based optimizer iterations, so the TensorFlow schedule adds one before
evaluating the same formula. Tests check the warmup slope, peak, decay, and
cross-framework values rather than assuming the APIs number steps identically.

### Label smoothing and padding

The token loss combines target negative log likelihood with the mean negative
log probability across the vocabulary. PAD labels contribute neither loss nor
gradient, and mean reduction divides by the number of real target tokens—not
the padded tensor size. An all-padding input returns differentiable finite zero
instead of NaN. Both runs use smoothing `0.1`.

### Explicit training loops

PyTorch uses `zero_grad`, forward, backward, gradient clipping, and optimizer
step directly. TensorFlow mirrors that sequence with `GradientTape`, explicit
gradient validation/clipping, and `apply_gradients`. This costs more code than
high-level trainers but keeps the two educational implementations aligned and
makes non-finite failures stop immediately.

### Initialization and layout differences

PyTorch explicitly applies Xavier initialization and then re-zeros the padding
embedding row. Keras dense and embedding layers already use Glorot defaults,
while Keras embeddings do not offer PyTorch's `padding_idx` behavior. Weight
synchronization must transpose every PyTorch linear weight because
`nn.Linear.weight` is `(out, in)` while a Keras dense kernel is `(in, out)`.
Bias vectors require no transpose.

## Measured comparison

The generated source of truth is
[`results/day25/metrics.json`](results/day25/metrics.json). The table below is
derived from saved histories, checkpoint files, model variables, and Day 23's
shared 100-example evaluation.

| Metric | PyTorch run | TensorFlow run |
|---|---:|---:|
| Hardware | Quadro M2000M CUDA, 4 GB | Native Windows CPU |
| Training pairs | 29,000 | 5,000 |
| Parameters | 1,915,392 | 299,776 |
| Parameter memory | 7.31 MiB | 1.14 MiB |
| Best checkpoint | 22.09 MiB | 3.49 MiB |
| Mean training throughput | 6,234 tokens/s | 1,256 tokens/s |
| Best validation loss | 2.3910 | 3.4132 |
| BLEU-4 | 37.8505 | 1.6422 |
| Decode throughput | 1.957 sentences/s | 0.274 sentences/s |
| Implementation files | 16 | 16 |
| Nonblank/non-comment lines | 1,234 | 1,338 |

Parameter memory is the exact trainable tensor footprint. Checkpoint size also
contains optimizer and progress state. Historical peak allocator/process
memory was not captured, so the project deliberately makes no peak-memory
claim. The observed 4.96× training and 7.13× decoding throughput ratios mix
hardware, model, sequence, vocabulary, and workload effects.

## API ergonomics and debugging experience

### PyTorch

- `nn.Module.forward` and eager execution made tensor shapes and Python stack
  traces direct during model development.
- Device placement is explicit and easy to audit, but every created tensor
  must be placed consistently.
- A single atomic checkpoint dictionary naturally holds model, optimizer,
  random-number-generator states, configuration, and progress.
- Native Windows CUDA was available for the full-corpus run.

### TensorFlow/Keras

- `Layer.call` composes cleanly, but `training=` must be threaded through every
  dropout-bearing layer.
- Lazy variable creation requires a real build call before inspecting weights,
  creating optimizer slots, or restoring checkpoints.
- `GradientTape` keeps the training mechanics explicit and readable in eager
  mode.
- Checkpoints use a trackable object graph. Day 23 exposed an important trap:
  restoring only the model could appear matched while resolving a training
  checkpoint incorrectly. Recreating the model, optimizer slots, epoch, and
  best-loss variables before restore fixed it and now has a regression test.
- TensorFlow 2.11+ lacks native Windows CUDA support. GPU execution requires a
  supported Linux/WSL2 setup, so the historical TensorFlow run used CPU.

The measured implementation size is close: TensorFlow uses 104 additional
nonblank/non-comment lines. Most of that difference comes from training-mode
propagation, lazy-build handling, and checkpoint graph setup rather than the
Transformer mathematics.

## Testing strategy

Correctness does not depend on the training curves alone. The suite checks:

1. Shapes, masking, residuals, embeddings, and independent stack weights in
   each framework.
2. Exact or tolerance-based cross-framework parity after weight synchronization.
3. Full-model logits with padding and causal masks.
4. Tiny-task overfitting to prove gradients flow end to end.
5. Deterministic BPE, aligned corpora, batching, and teacher-forcing shifts.
6. Schedule, smoothed loss, checkpoint round trips, and resume behavior.
7. Corpus BLEU, full TensorFlow checkpoint restoration, and attention-map
   probability/causality invariants.

## How to run a controlled framework benchmark later

A fair performance claim would require the same tokenizer, train/validation
pairs, model configuration, normalization mode, batch size, maximum length,
precision, warmup, epoch/step count, and comparable GPU class. It should include
warmup iterations, synchronize accelerators around timing, and record peak
allocator memory. Multiple seeded repetitions should report median and spread.
Until that experiment exists, the current comparison remains an honest report
of two validated project checkpoints and their engineering tradeoffs.

## Conclusion

The strongest conclusion is parity of ideas, not victory of one API. PyTorch
was the practical choice for the full native-Windows CUDA run and offered the
simpler checkpoint/debugging path here. TensorFlow/Keras expressed the same
architecture and math successfully, with extra lifecycle discipline around
variables and trackable checkpoints. The shared tests demonstrate that both
implementations compute the intended Transformer when weights and inputs match.
