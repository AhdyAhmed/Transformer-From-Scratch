# Transformer From Scratch — Design Document

**Goal:** Implement the original Transformer architecture (Vaswani et al., *"Attention Is All You Need"*) from scratch in both **PyTorch** and **TensorFlow/Keras**, with clean, well-documented, mirrored codebases that demonstrate deep understanding of the architecture — suitable as a portfolio centerpiece.

---

## 1. Objectives

- Implement every core component manually (no `nn.Transformer` / `keras.layers.MultiHeadAttention` shortcuts) to prove understanding.
- Keep the two implementations **structurally parallel** — same module names, same file layout, same hyperparameters — so a reviewer can compare frameworks side by side.
- Train on a real (small-scale) task to prove correctness, not just unit tests.
- Ship with tests, benchmarks, diagrams, and a polished README — this is a portfolio piece, presentation matters as much as code.

---

## 2. Repository Structure

```
transformer-from-scratch/
├── README.md
├── design.md
├── requirements-torch.txt
├── requirements-tf.txt
├── torch_impl/
│   ├── model/
│   │   ├── embeddings.py         # token + positional embeddings
│   │   ├── attention.py          # scaled dot-product + multi-head attention
│   │   ├── feed_forward.py       # position-wise FFN
│   │   ├── layer_norm.py         # (or use built-in, document choice)
│   │   ├── encoder.py            # encoder layer + stack
│   │   ├── decoder.py            # decoder layer + stack
│   │   └── transformer.py        # full model assembly
│   ├── data/
│   │   ├── tokenizer.py
│   │   └── dataset.py
│   ├── train.py
│   ├── inference.py
│   ├── config.py
│   └── tests/
├── tf_impl/
│   ├── model/                    # same submodule breakdown as torch_impl/model
│   ├── data/
│   ├── train.py
│   ├── inference.py
│   ├── config.py
│   └── tests/
├── notebooks/
│   ├── 01_attention_visualization.ipynb
│   └── 02_framework_comparison.ipynb
└── assets/
    └── diagrams/                 # architecture diagrams, attention heatmaps
```

Mirroring the folder structure across `torch_impl/` and `tf_impl/` is intentional — it lets a visitor open both `attention.py` files and diff the *ideas*, not the boilerplate.

> **Naming note:** the implementation folders are called `torch_impl/` and `tf_impl/` rather than `pytorch/` and `tensorflow/`. A top-level folder named `tensorflow/` shadows the real `tensorflow` package on import (`import tensorflow as tf` would resolve to the local folder), so both are given the `_impl` treatment for symmetry.

### Shared conventions (both frameworks)

- **Mask convention:** boolean masks where `True` = "may attend", `False` = "blocked". Masks broadcast to attention scores of shape `(batch, heads, q_len, k_len)`.
- **Special token ids:** `PAD=0, BOS=1, EOS=2, UNK=3`.
- **Config:** one `TransformerConfig` dataclass per framework with identical fields and defaults (stdlib only), verified by `tests/test_config_parity.py`.
- **Embeddings:** `TokenEmbedding` (scaled by `sqrt(d_model)`) and `PositionalEncoding` (fixed sinusoidal, precomputed up to `max_len`) live in each `model/embeddings.py`. Positional encoding has no learned parameters, so it's checked for exact cross-framework numerical parity in `tests/test_embeddings_parity.py` — attention (Day 6) is where parity gets harder, since it requires loading identical weights into both frameworks.
- **Scaled dot-product attention:** implemented as a plain function (not a layer/module) in each `model/attention.py`, since it has no learned parameters of its own — only the Q/K/V projections wrapped around it in multi-head attention (Day 5) do. It returns both the output and the post-softmax attention weights, so weights stay usable for the visualization notebook (Day 24) and for masking tests. Like positional encoding, it's checked for exact cross-framework numerical parity (`tests/test_attention_parity.py`) since it's still parameter-free; multi-head attention's learned projections will need matching weights loaded into both frameworks to compare, per the original Day 6 plan.
- **Multi-head attention:** `model/multi_head_attention.py` wraps `scaled_dot_product_attention` with four dense/linear layers (`W_Q, W_K, W_V, W_O`, each `d_model x d_model`), implemented as one combined projection per Q/K/V rather than `num_heads` separate small ones — mathematically identical after the reshape, but a single matmul instead of many. The layer caches its last attention weights (`self.attn_weights`) for later inspection/visualization. This is the first module with real learned parameters, which is why exact cross-framework parity moves from 'feed identical inputs' (Days 3-4) to 'load identical weights into both frameworks, then compare outputs' — planned for Day 6.
- **Weight-matched parity (Day 6):** `tests/weight_sync.py` copies a PyTorch `MultiHeadAttention`'s weights into a TensorFlow one. The one subtlety: PyTorch's `nn.Linear.weight` is `(out_features, in_features)` and computes `x @ W.T + b`, while Keras' `Dense.kernel` is `(in_features, out_features)` and computes `x @ W + b` — so the weight transposes on the way across, the bias doesn't. `tests/test_mha_parity.py` verifies this for self-attention, cross-attention (mismatched query/key lengths), and masked attention, and includes a negative control confirming the two frameworks genuinely diverge *without* syncing — otherwise a parity test could pass for the wrong reason (e.g. both outputs collapsing to the same constant). The transpose convention was additionally verified end-to-end with a pure NumPy reimplementation of the whole forward pass before being trusted in the real test.
- **Known framework gap:** PyTorch's `nn.Embedding(padding_idx=...)` zeroes the pad token's gradient automatically; Keras' `Embedding` has no built-in equivalent. The TF `TokenEmbedding` accepts `pad_idx` for interface symmetry but doesn't enforce this yet — documented in `tf_impl/model/embeddings.py` rather than silently diverging.

---

## 3. Architecture Components to Implement

Each of these is a separate, independently testable module in **both** frameworks:

1. **Input Embeddings** — token embedding scaled by `sqrt(d_model)`.
2. **Positional Encoding** — fixed sinusoidal (original paper version); optionally add learned positional embeddings as a toggle for experimentation.
3. **Scaled Dot-Product Attention** — `softmax(QK^T / sqrt(d_k))V`, with support for masking.
4. **Multi-Head Attention** — linear projections, head splitting/merging, from-scratch implementation (no framework fused kernels).
5. **Position-wise Feed-Forward Network** — two linear layers with ReLU/GELU in between.
6. **Residual Connections + Layer Normalization** — pre-norm vs post-norm as a configurable flag (worth documenting the trade-off).
7. **Encoder Layer & Encoder Stack** — self-attention + FFN, N layers.
8. **Decoder Layer & Decoder Stack** — masked self-attention + cross-attention + FFN, N layers.
9. **Masking** — padding mask and look-ahead (causal) mask utilities.
10. **Output Projection + Softmax** — final linear layer to vocab size.
11. **Full Transformer Assembly** — wraps encoder/decoder, embeddings, and generation logic.
12. **Learning Rate Schedule** — the paper's warmup + inverse-sqrt-decay schedule.
13. **Label Smoothing Loss**.

---

## 4. Framework-Specific Notes

| Aspect | PyTorch | TensorFlow/Keras |
|---|---|---|
| Base class | `nn.Module` | `tf.keras.layers.Layer` |
| Model assembly | manual `forward()` | `call()` method, optionally `tf.keras.Model` subclass |
| Masking | boolean tensors, `masked_fill` | `tf.where` / additive `-1e9` mask |
| Training loop | manual loop or `Lightning`-style (keep manual for transparency) | manual `tf.GradientTape` loop (avoid `model.fit()` to keep internals visible) |
| Custom LR schedule | `torch.optim.lr_scheduler.LambdaLR` | `tf.keras.optimizers.schedules.LearningRateSchedule` subclass |
| Autograd | dynamic graph | eager mode + `@tf.function` for the train step |

Design decision: **use eager-style, explicit training loops in both frameworks** rather than `Trainer`/`model.fit()` abstractions — the point of the project is to show the mechanics, not framework convenience APIs.

---

## 5. Task for Validation

Pick one modest, fast-to-train task so both implementations can be proven correct without needing heavy compute:

- **Primary:** Small-scale neural machine translation (e.g., Multi30k EN→DE) — the same task as the original paper, easy to sanity-check with BLEU.
- **Alternative/lighter:** Character-level or toy sequence-copy task, useful for fast unit-level correctness checks before committing to full training.

Both frameworks should train on identical data splits, tokenization, and hyperparameters so results are genuinely comparable.

---

## 6. Testing Strategy

- **Shape tests** — every module checked for correct input/output tensor shapes.
- **Masking tests** — verify masked positions receive ~0 attention weight.
- **Overfit test** — model can memorize a tiny (8–16 sample) dataset to near-zero loss; catches broken gradients.
- **Numerical parity test** — feed identical weights/inputs into both frameworks' attention block and confirm outputs match within floating-point tolerance (great portfolio talking point).
- **End-to-end** — full training run producing a reasonable BLEU/loss curve.

---

## 7. Deliverables for the Portfolio

- Clean READMEs (root + per-framework) with architecture diagram, usage instructions, and results table.
- Attention weight visualizations (heatmaps) via notebook.
- A short **"PyTorch vs TensorFlow" writeup** comparing API ergonomics, debugging experience, and performance — this differentiates the project from generic "transformer from scratch" repos.
- Loss/BLEU curves and sample translations included in README.
- Optional: a small blog-post-style `WRITEUP.md` explaining design decisions (pre-norm vs post-norm, warmup schedule, etc.) — recruiters and engineers reading the repo will value this more than the code itself.

---

## 8. Milestones

1. Scaffold repo structure, shared config format.
2. Implement attention + embeddings in both frameworks; pass shape/parity tests.
3. Implement full encoder/decoder stacks; overfit-test on tiny data.
4. Build data pipeline + tokenizer for translation task.
5. Full training run (PyTorch), tune, record baseline metrics.
6. Full training run (TensorFlow), match PyTorch setup, record metrics.
7. Build visualization notebooks + comparison writeup.
8. Polish READMEs, diagrams, and portfolio presentation.

---

## 9. Stretch Goals (optional, post-MVP)

- Add relative positional encoding or RoPE as an alternative, toggleable module.
- Implement KV-caching for faster autoregressive inference.
- Export both models to ONNX and benchmark inference speed.
- Add a minimal Streamlit/Gradio demo for live translation.
