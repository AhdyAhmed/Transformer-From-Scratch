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

**Day 18 — Training loop (TensorFlow) — implemented**
- [x] Manual `tf.GradientTape` teacher-forced training loop mirroring PyTorch's epoch metrics and token-weighted loss.
- [x] Validation loss, Noam learning-rate logging, token throughput, deterministic epoch shuffling, gradient clipping, and non-finite loss/gradient guards.
- [x] TensorFlow checkpoint managers for latest/best checkpoints, optimizer/model/epoch state restoration, and JSONL metric history.
- [x] CLI for sample/custom parallel data, config overrides, and checkpoint resume; regression tests for updates and validation behavior.

---

## Phase 5: Full Training Runs (Days 19–23)

**Day 19 — PyTorch subset training run — completed**
- [x] Train on a deterministic 2,000-pair Multi30k subset with CUDA.
- [x] Track train/validation loss, save latest/best checkpoints, and render an SVG loss curve.
- [x] Log deterministic validation translations after every epoch.
- [x] Resume from the epoch-3 checkpoint and complete epoch 4 with optimizer/RNG state restored.

**Day 20 — Extended PyTorch training run — completed**
- [x] Train a 1.92M-parameter pre-norm Transformer on all 29,000/1,014 Multi30k train/validation pairs for 30 CUDA epochs (13,620 steps).
- [x] Expand the joint vocabulary from 512 to 2,048 tokens and optimize from-scratch BPE encoding without changing learned tokenization behavior.
- [x] Reduce training loss from 5.5842 to 2.1063 and validation loss from 4.2692 to 2.3910 without non-finite values, memory errors, or runtime failures.
- [x] Preserve the exact tokenizer, full metric history, 90 greedy samples, loss curve, and atomic latest/best checkpoints; verify epoch/step integrity.
- [x] Retain formal BLEU scoring for Day 23 while confirming strong source-conditioned qualitative improvement.

**Day 21 — TensorFlow CPU subset training run — completed**
- [x] Add deterministic subset controls, per-epoch greedy samples, JSONL translation logs, and SVG loss curves to the TensorFlow CLI.
- [x] Train on 5,000/500 Multi30k train/validation pairs for 10 CPU epochs (790 optimizer steps).
- [x] Reduce training loss from 4.8390 to 3.3631 and validation loss from 4.1279 to 3.4998 without non-finite values or runtime failures.
- [x] Restore epoch 9, optimizer step 711, Adam slots, and best-validation state, then complete epoch 10 in a separate process.
- [x] Preserve the exact tokenizer, full metric history, 30 greedy samples, loss curve, and TensorFlow latest/best checkpoints.

**Day 22 — Extended TensorFlow training run — completed**
- [x] Restore the Day 21 model, Adam slots, epoch 10, optimizer step 790, and best-validation state.
- [x] Continue the identical 5,000/500-pair native-Windows CPU run through epoch 20 and optimizer step 1,580.
- [x] Improve training loss from 3.3631 to 3.2172 and validation loss from 3.4998 to 3.4132 during the continuation.
- [x] Verify that epoch 20 is both the latest and best TensorFlow checkpoint with finite metrics throughout.
- [x] Preserve the exact tokenizer, complete 20-epoch history, 60 greedy samples, and updated loss curve for Day 23 evaluation.

**Day 23 — Evaluation — completed**
- [x] Add a reusable shared-corpus evaluation CLI with strict PyTorch and full-object-graph TensorFlow checkpoint restoration.
- [x] Evaluate both best checkpoints on the same 100 Multi30k validation examples with reproducible lowercase tokenized corpus BLEU-4.
- [x] Record PyTorch BLEU 37.8505 and TensorFlow BLEU 1.6422, including n-gram precisions, brevity penalty, length ratio, and decode throughput.
- [x] Save all 100 source/reference/prediction records and a ten-example side-by-side Markdown comparison.
- [x] Document that unequal training data, vocabulary, model capacity, and hardware prevent interpreting the score gap as framework-only.

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
