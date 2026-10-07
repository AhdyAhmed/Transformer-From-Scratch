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
- **Feed-forward network (Day 8):** `model/feed_forward.py` implements `FFN(x) = max(0, xW_1+b_1)W_2+b_2` with dropout between the activation and the second linear layer. An optional GELU variant is supported via an `activation` argument, for later experimentation, though the paper and the default config use ReLU.
- **Residual + LayerNorm wrapper (Day 8):** `model/residual.py`'s `ResidualConnection` wraps any sublayer callable with `norm(x + dropout(sublayer(x)))` (post-norm, the paper) or `x + dropout(sublayer(norm(x)))` (pre-norm), selected by `TransformerConfig.norm_first`. Pre-norm generally trains more stably at depth because gradients reach early layers without passing through a LayerNorm first, at a small cost in final quality versus post-norm at the same depth — both are implemented rather than picking one, so this is a config flag to experiment with rather than a fixed design decision.
- **LayerNorm epsilon pinned across frameworks:** PyTorch's `nn.LayerNorm` defaults to `eps=1e-5` and Keras' `LayerNormalization` defaults to `epsilon=1e-3` — neither matches the other. Both are explicitly set to `1e-6` in this repo so cross-framework parity isn't broken by a default neither side chose on purpose.
- **Parity extended to Day 8's modules:** `tests/weight_sync.py` gained `copy_feed_forward` and `copy_layer_norm`/`copy_residual_connection`, and `tests/test_ffn_residual_parity.py` checks both against identical inputs after weight-syncing, the same pattern as Day 6's `MultiHeadAttention` check. For `ResidualConnection` specifically, the two frameworks are fed the same deterministic, parameter-free sublayer (`t -> 2t+1`) so the test isolates the residual/LayerNorm logic itself rather than depending on yet another weight-synced module.
- **Encoder layer (Day 9):** `model/encoder.py`'s `EncoderLayer` composes Day 5's `MultiHeadAttention`, Day 8's `PositionwiseFeedForward`, and two independent `ResidualConnection` wrappers (self-attention gets one, the FFN gets the other — each needs its own LayerNorm). No causal mask is used here, only an optional padding mask: the encoder sees the whole source sequence at once, unlike the decoder (Day 11), which must not see the future.
- **Padding-mask invariance, tested structurally:** beyond shape checks, `test_encoder.py` in both frameworks verifies that changing *only* the embeddings at padded positions leaves the encoder's output at real positions unchanged. This is the kind of bug a shape-only test suite would never catch (wrong mask broadcast axis, an off-by-one in which positions count as padding, etc.), confirmed independently with a full NumPy reimplementation of the encoder layer before trusting the real test.
- **Parity extended to the full encoder layer:** `weight_sync.copy_encoder_layer` composes the existing `copy_multi_head_attention`, `copy_feed_forward`, and `copy_residual_connection` helpers, and `tests/test_encoder_parity.py` checks the assembled layer end-to-end, with and without a padding mask, in both pre-norm and post-norm mode — the parity-testing pattern from Day 6 now scales to composed modules, not just individual layers.
- **Encoder stack (Day 10):** `model/encoder_stack.py`'s `Encoder` stacks `num_layers` independent `EncoderLayer`s (PyTorch: `copy.deepcopy` per layer inside an `nn.ModuleList`; TensorFlow: a fresh `EncoderLayer(...)` per list position) followed by one more LayerNorm. That final norm is redundant in post-norm mode — every layer already ends with one — but is kept because it's load-bearing in pre-norm mode, where the residual *stream* itself is never normalized, only each sublayer's input is; without it, a pre-norm stack's output would be unnormalized going into the decoder's cross-attention.
- **Properties re-checked after stacking:** three things that are easy to get right at one layer and silently wrong once layers compose are explicitly tested at the stack level, not just the single-layer level: (1) padding-mask invariance still holds N layers deep (verified independently in NumPy before trusting the real test), (2) the N layers have genuinely independent weights rather than N references to one shared layer, and (3) a deeper stack actually produces a different output than a shallow one — i.e. depth isn't a no-op.
- **Parity extended to the full stack:** `weight_sync.copy_encoder_stack` loops `copy_encoder_layer` over every layer and then copies the stack's own final LayerNorm; `tests/test_encoder_stack_parity.py` checks the whole thing end-to-end, with and without a padding mask.
- **Decoder layer (Day 11):** `model/decoder.py`'s `DecoderLayer` has three sublayers, each with its own `ResidualConnection` (three independent LayerNorms, vs. the encoder layer's two): masked self-attention over the decoder's own (so-far-generated) sequence, cross-attention where the query comes from the decoder but key/value come from the encoder's `memory`, and the feed-forward network. Cross-attention is what lets every decoder position see the *entire* source sequence while still never seeing future target tokens — the self-attention sublayer enforces the latter via `tgt_mask` (`create_target_mask`, Day 2), and the cross-attention sublayer enforces source padding via `memory_mask` (reused directly from the encoder's own `create_padding_mask`).
- **Three decoder-specific invariants, tested explicitly:** (1) the causal mask blocks future target positions — perturbing a later position must not change an earlier one's output; (2) the memory mask blocks padded *source* positions — perturbing padded encoder memory must not change the output at all; and (3), the inverse sanity check that's easy to forget: perturbing *non-padded* memory **should** change the output, confirming cross-attention is actually doing something rather than silently collapsing to a no-op. All three were independently verified with a NumPy reimplementation of the full decoder layer before being trusted in the real test suite.
- **Parity extended to the decoder layer:** `weight_sync.copy_decoder_layer` copies both attention sublayers (self and cross — two separate `MultiHeadAttention` instances, not a shared one), the feed-forward network, and all three LayerNorms; `tests/test_decoder_parity.py` checks the assembled layer end-to-end, with and without masks, in both pre-norm and post-norm mode.
- **Decoder stack (Day 12):** `model/decoder_stack.py`'s `Decoder` mirrors the encoder stack (Day 10) exactly — `num_layers` independent `DecoderLayer`s plus a final LayerNorm — with the extra `memory`/`memory_mask` every layer threads through for cross-attention. The same depth-matters, independent-weights, and mask-survives-stacking checks from Day 10 are re-run here, plus the decoder-specific causal/memory invariants from Day 11, now at stack depth rather than a single layer.
- **Full model assembly (Day 12):** `model/transformer.py`'s `Transformer` wires `TransformerEmbedding` (Day 3) → `Encoder` (Day 10) → `Decoder` (Day 12) → a final output projection to `tgt_vocab_size`, with `encode()`/`decode()` exposed separately from `forward()` (`call()` in TF) so the eventual inference/generation code (Day 26) can run the encoder once and the decoder repeatedly without re-encoding. `create_masks()` is a thin convenience method binding `model/masking.create_masks` to the model's own `pad_idx`. `from_config()` builds every piece directly from a `TransformerConfig` (Day 2), which is how `train.py` (Day 17/18) will construct the model — one dataclass, one call, no hyperparameter duplicated by hand at the training script.
- **A bug the build caught on itself:** `Transformer._init_parameters()` applies Xavier-uniform init to every parameter with more than one dimension, matching the paper's reference implementation — but the token embedding's pad row (zeroed at construction, Day 3) has more than one dimension too, so that same pass would silently overwrite it with random values the moment the full model is assembled. Caught while writing this file, fixed by explicitly re-zeroing both embeddings' pad rows right after the Xavier pass, and locked in with a regression test (`test_padding_row_stays_zero_after_full_model_init`). The TensorFlow side has no equivalent risk: Keras' default `glorot_uniform`/`ones`/`zeros` initializers already match the paper's convention without any manual re-init pass, and `tf.keras.layers.Embedding` has no pad row to protect in the first place (the existing documented gap from Day 3).
- **Parity extended to the full model — the capstone check:** `weight_sync.copy_transformer` composes every helper built since Day 6 — both embeddings' tables (skipping the duplicate copy when `share_embeddings` is set), the full encoder stack, the full decoder stack, and the output projection. `tests/test_transformer_parity.py` then checks the two complete models produce identical logits: with no masks, with real padding+causal masks built through each model's own `create_masks()`, and with `share_embeddings=True`. This was additionally verified independently with a full NumPy reimplementation of the entire pipeline — embeddings through output projection — before trusting the real cross-framework test, since a bug anywhere in a 12-day chain of composed modules would be hardest to localize exactly here.
- **Overfit test (Day 13) — the first test that trains anything:** every test through Day 12 checks shapes, masking invariants, and cross-framework numerics, but none of them prove a real training loop — forward pass, loss, backward pass, optimizer step, repeated — actually makes the model better at anything. `torch_impl/tests/test_overfit.py` and `tf_impl/tests/test_overfit.py` close that gap directly: a tiny Transformer (`d_model=32, num_layers=2`) is trained with plain Adam for ~600 full-batch steps on 12 fixed sequences from a trivial copy task (`src == tgt`, so correctness is unambiguous), with no tokenizer or dataset — those come Day 14-15 — since a sanity check for "do gradients flow" doesn't need real data.
- **Why a copy task, and why no padding:** the task is trivial on purpose (memorize 12 fixed sequences, not generalize), so a failure to converge points at the model, not the task's difficulty. No PAD token appears in any sequence, which deliberately isolates causal masking (already exercised) from padding masking (already extensively tested on Days 9-12) — this test exists to catch broken *gradients*, not to re-test masking.
- **What the test actually asserts, and why those particular checks:** (1) the initial loss lands near `ln(vocab_size)` — the expected cross-entropy for a vocab-size-way classifier at random init — catching anything already broken before training starts; (2) the final loss drops by at least 5x, a looser bar than "near zero" deliberately, since this environment can't execute the real training run to tune an exact threshold, and a flaky test that sometimes fails for being 1% too strict is worse than a slightly generous one that still catches a truly broken gradient; (3) token-level accuracy on the memorized sequences exceeds 90%. All three were sanity-checked here: the data-generation and teacher-forcing shift logic were verified mechanically in NumPy, and a cross-entropy loss computed on random logits confirmed it lands at the expected `~ln(20)` baseline — but the actual training run, which needs torch/tensorflow, could not be executed in this sandbox and should be run locally.
- **Marked `slow`, with `-fast` Make targets:** `pytest.ini` registers a `slow` marker and the Makefile gained `test-torch-fast`/`test-tf-fast` (equivalent to `-m "not slow"`), so the ~600-step training test doesn't have to run on every quick iteration loop while still running by default on `make test-torch`/`make test-tf`.
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
