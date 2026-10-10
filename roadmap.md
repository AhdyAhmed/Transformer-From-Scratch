# Transformer From Scratch — Roadmap (30 Days)

Assumes roughly **2–3 focused hours/day**, part-time pace. Adjust up/down by compressing or stretching phases — the phase order shouldn't change since each depends on the last.

---

## Phase 1: Foundations & Scaffolding (Days 1–3)

**Day 1 — Repo setup & theory refresh**
- Create repo structure from `design.md`.
- Re-read "Attention Is All You Need" + one good annotated-implementation blog (e.g. Harvard NLP's "The Annotated Transformer") for reference, not copying.
- Set up environments: `requirements-torch.txt`, `requirements-tf.txt`, virtualenvs.

**Day 2 — Config & utilities**
- Write shared `config.py` (d_model, num_heads, num_layers, d_ff, dropout, vocab size, max seq len) — identical values for both frameworks.
- Write masking utilities (padding mask, look-ahead mask) in both frameworks.
- Write shape-testing harness (`tests/`) skeleton.

**Day 3 — Embeddings & positional encoding**
- Implement token embeddings (scaled by `sqrt(d_model)`) — PyTorch + TensorFlow.
- Implement sinusoidal positional encoding — both frameworks.
- Shape + numerical sanity tests for both.

---

## Phase 2: Attention Mechanism (Days 4–7)

**Day 4 — Scaled dot-product attention**
- Implement core attention function in both frameworks.
- Unit test: verify softmax rows sum to 1, masked positions ≈ 0 weight.

**Day 5 — Multi-head attention**
- Implement Q/K/V projections, head splitting/merging — PyTorch.
- Implement same — TensorFlow.

**Day 6 — Numerical parity check**
- Load identical weights into both attention implementations.
- Feed identical input tensors, confirm outputs match within float tolerance.
- Document this as a portfolio highlight (great README/blog material).

**Day 7 — Buffer/catch-up day**
- Fix any shape mismatches or bugs found in Days 4–6.
- Write unit tests to lock in correctness before moving on.

---

## Phase 3: Encoder & Decoder Stacks (Days 8–13)

**Day 8 — Feed-forward network + residual/layernorm**
- Position-wise FFN in both frameworks.
- Residual connection + LayerNorm wrapper, with pre-norm/post-norm toggle.

**Day 9 — Encoder layer**
- Assemble self-attention + FFN + residuals into one encoder layer, both frameworks.

**Day 10 — Encoder stack**
- Stack N encoder layers; test end-to-end shape flow with dummy data.

**Day 11 — Decoder layer**
- Masked self-attention + cross-attention + FFN, both frameworks.

**Day 12 — Decoder stack + full model assembly**
- Stack N decoder layers.
- Wire embeddings → encoder → decoder → output projection into a full `Transformer` class in both frameworks.

**Day 13 — Overfit test**
- Take 8–16 toy examples (e.g. copy-task or tiny translation pairs).
- Train each framework's model to near-zero loss to catch broken gradients before scaling up.

---

## Phase 4: Data Pipeline & Training Setup (Days 14–18)

**Day 14 — Tokenizer & vocab**
- Pick tokenizer (e.g. subword/BPE via `tokenizers` or `sentencepiece`).
- Build shared vocab so both frameworks train on identical tokenized data.

**Day 15 — Dataset & batching**
- Multi30k (or chosen dataset) loading, padding, batching — both frameworks.
- Verify batch shapes match masking utilities from Day 2.

**Day 16 — Learning rate schedule + label smoothing**
- [x] Implement paper's warmup + inverse-sqrt decay schedule — both frameworks.
- [x] Implement label smoothing loss with padding ignored and token-normalized mean — both frameworks.
- [x] Add tests for formula, warmup peak/decay, padding behavior, reductions, and gradients.
- [x] Document usage and framework step-index conventions.

**Day 17 — Training loop (PyTorch) — completed**
- [x] Manual teacher-forced training loop with train/validation loss, learning-rate and tokens/sec logging.
- [x] Gradient clipping, deterministic epoch shuffling, and non-finite loss guard.
- [x] Atomic latest/best checkpoints including model, optimizer, epoch/step, config, and RNG state.
- [x] Resume training from a checkpoint and regression tests for updates/checkpoint restoration.

**Day 18 — Training loop (TensorFlow)**
- Manual `tf.GradientTape` training loop mirroring PyTorch's structure.
- Checkpointing.

---

## Phase 5: Full Training Runs (Days 19–23)

**Day 19–20 — PyTorch training run**
- Train on full (or reasonably-sized subset of) Multi30k.
- Track loss curves, save checkpoints, log sample translations.

**Day 21–22 — TensorFlow training run**
- Same dataset, same hyperparameters, same number of steps/epochs as PyTorch run.
- Track identical metrics for fair comparison.

**Day 23 — Evaluation**
- Compute BLEU (or chosen metric) for both models.
- Save qualitative translation examples side by side.

---

## Phase 6: Visualization, Comparison & Polish (Days 24–28)

**Day 24 — Attention visualization notebook**
- Extract attention weights, plot heatmaps for a few example sentences.

**Day 25 — Framework comparison notebook/writeup**
- Compare training speed, memory use, debugging experience, code verbosity.
- Draft `WRITEUP.md` covering design decisions (pre-norm vs post-norm, warmup schedule, etc.).

**Day 26 — Inference scripts**
- Clean `inference.py` for both frameworks — load checkpoint, translate a sentence.

**Day 27 — README & diagrams**
- Root README: project overview, architecture diagram, results table, how to run.
- Per-framework READMEs with setup + usage instructions.

**Day 28 — Code cleanup & docstrings**
- Pass over both codebases for consistent naming, comments, type hints.
- Ensure the two implementations are genuinely parallel/comparable file-by-file.

---

## Phase 7: Stretch Goals & Final Polish (Days 29–30)

**Day 29 — Pick 1 stretch goal**
- Options: RoPE/relative positional encoding toggle, KV-caching for inference, ONNX export.
- Implement in whichever framework is more convenient; note the other as "future work" in README.

**Day 30 — Final review**
- Re-run all tests.
- Proofread README/writeup.
- Tag a `v1.0` release, publish repo, share on portfolio/LinkedIn.

---

## Suggested Pace Variants

| Pace | Total duration | Notes |
|---|---|---|
| Intensive (4–5 hrs/day) | ~2 weeks | Compress each phase by ~50%; skip Day 7/29 buffer days |
| Standard (2–3 hrs/day) | 30 days (above) | Default plan |
| Relaxed (weekends only) | ~10–12 weeks | Treat each "day" above as one weekend session |
