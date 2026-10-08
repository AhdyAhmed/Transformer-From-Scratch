# Transformer From Scratch — PyTorch & TensorFlow

A from-scratch implementation of the original Transformer architecture
(*"Attention Is All You Need"*, Vaswani et al., 2017) built independently in
**PyTorch** and **TensorFlow/Keras**, with mirrored code structure so the two
implementations can be compared side by side.

No `nn.Transformer`, no `keras.layers.MultiHeadAttention` — every component
(embeddings, positional encoding, multi-head attention, encoder/decoder
stacks, masking, the warmup LR schedule) is implemented by hand in both
frameworks.

> 🚧 **Status:** Work in progress. Follow along via [`roadmap.md`](roadmap.md)
> for the day-by-day build log, and [`design.md`](design.md) for the full
> architecture and project design.

---

## Why this project

Most "transformer from scratch" repos pick one framework. This one
deliberately implements the **same architecture twice**, so the repo also
serves as a practical PyTorch-vs-TensorFlow comparison — in API ergonomics,
training loop design, and debugging experience. See `WRITEUP.md` (added
later in the build) for that comparison.

## Repository structure

```
Transformer-From-Scratch/
├── README.md
├── design.md              # full architecture & design decisions
├── roadmap.md             # day-by-day build plan
├── requirements-torch.txt
├── requirements-tf.txt
├── pytest.ini / Makefile
├── data/                                # shared, framework-agnostic (see design.md, Day 14)
│   ├── bpe_tokenizer.py                   # from-scratch BPE tokenizer
│   └── build_vocab.py                     # CLI: train + save one shared vocab file
├── tests/                             # framework-agnostic tests
│   ├── test_bpe_tokenizer.py            # Day 14 — pure stdlib, actually runs here
│   ├── weight_sync.py                   # copy weights from a torch module into its tf twin
│   ├── test_mha_parity.py               # weight-matched MultiHeadAttention parity (Day 6)
│   ├── test_ffn_residual_parity.py          # weight-matched FFN/LayerNorm parity (Day 8)
│   ├── test_encoder_parity.py           # weight-matched EncoderLayer parity (Day 9)
│   ├── test_encoder_stack_parity.py         # weight-matched full Encoder parity (Day 10)
│   ├── test_decoder_parity.py           # weight-matched DecoderLayer parity (Day 11)
│   ├── test_transformer_parity.py       # weight-matched FULL MODEL parity (Day 12)
│   └── ...                              # config / embeddings / attention parity
├── torch_impl/                          # PyTorch implementation
│   ├── config.py                          # TransformerConfig (identical to tf_impl/config.py)
│   ├── model/                              # masking, embeddings, attention, encoder, decoder, transformer
│   ├── data/                               # (Day 15) Dataset/DataLoader wrapping data/bpe_tokenizer.py
│   └── tests/
│       └── test_overfit.py                   # Day 13: trains on a tiny copy task, slow-marked
├── tf_impl/                              # TensorFlow implementation, mirrors torch_impl/
├── notebooks/                              # attention visualization, framework comparison
└── assets/diagrams/                         # architecture diagrams, attention heatmaps
```

## Status / Progress

- [x] Day 1 — Repo scaffolding & environment setup
- [x] Day 2 — Config & masking utilities
- [x] Day 3 — Embeddings & positional encoding
- [x] Day 4 — Scaled dot-product attention (+ numerical parity check)
- [x] Day 5 — Multi-head attention
- [x] Day 6 — Weight-matched cross-framework parity check (MultiHeadAttention)
- [ ] Day 7 — Buffer / bug-fix day
- [x] Day 8 — Feed-forward network, residual connections & LayerNorm
- [x] Day 9 — Encoder layer
- [x] Day 10 — Encoder stack
- [x] Day 11 — Decoder layer
- [x] Day 12 — Decoder stack + full Transformer model assembly
- [x] Day 13 — Overfit test (tiny-data sanity check)
- [x] Day 14 — Shared BPE tokenizer & vocabulary
- [ ] Day 15–18 — Dataset/batching, LR schedule, label smoothing, training loops
- [ ] Day 19–23 — Full training runs (PyTorch & TensorFlow)
- [ ] Day 24–28 — Visualization, comparison writeup, polish
- [ ] Day 29–30 — Stretch goal & final release

Full detail on each phase lives in [`roadmap.md`](roadmap.md).

## Setup

**PyTorch environment**
```bash
python -m venv .venv-torch
source .venv-torch/bin/activate
pip install -r requirements-torch.txt
```

**TensorFlow environment**
```bash
python -m venv .venv-tf
source .venv-tf/bin/activate
pip install -r requirements-tf.txt
```

## Running the tests

Each framework's tests run in its own virtualenv; the framework-agnostic tests in
`tests/` run in either.

```bash
# PyTorch env
make test-torch        # everything, including the ~600-step Day 13 overfit test
make test-torch-fast   # == python -m pytest tests torch_impl -m "not slow"

# TensorFlow env
make test-tf            # everything, including the Day 13 overfit test
make test-tf-fast       # == python -m pytest tests tf_impl -m "not slow"
```

`tests/test_config_parity.py`, `tests/test_embeddings_parity.py`, `tests/test_attention_parity.py`, `tests/test_mha_parity.py`, `tests/test_ffn_residual_parity.py` and `tests/test_encoder_parity.py` only run if *both* frameworks are importable in the active environment; otherwise pytest reports them as skipped, which is expected.

The parameter-free ones (config, positional encoding, raw attention) compare math directly on identical inputs. The weight-bearing ones (`MultiHeadAttention`, `PositionwiseFeedForward`, `ResidualConnection`'s LayerNorm, and now the full `EncoderLayer`) use `tests/weight_sync.py` to copy the *same* weights from the PyTorch module into its TensorFlow twin first — accounting for PyTorch's `(out, in)` Linear layout vs. Keras' `(in, out)` Dense layout — before comparing outputs. `test_mha_parity.py` additionally includes a negative control confirming the two frameworks genuinely diverge *without* syncing, so parity can't trivially pass for the wrong reason.

LayerNorm's epsilon is pinned to `1e-6` explicitly on both sides — PyTorch's `nn.LayerNorm` default is `1e-5` and Keras' `LayerNormalization` default is `1e-3`, and leaving either at its default would have silently broken this parity check.

`torch_impl/tests/test_encoder.py` and `tf_impl/tests/test_encoder.py` also each check a structural property, not just shapes: perturbing *only* the padded positions of the input must leave the encoder's output at the real positions unchanged. That's an easy property to silently break (e.g. a mask applied to the wrong axis) and a cheap, high-value thing to test for.

`test_encoder_stack.py` (Day 10) checks that property *survives stacking* N layers deep, and adds two more stack-specific checks: the N layers must have genuinely independent weights (not six references to one set — that would silently collapse the stack's capacity to a single layer), and a deeper stack must actually produce a different output than a shallow one on the same input.

`test_decoder.py` (Day 11) checks three analogous invariants for the decoder layer: the causal mask blocks future target positions (perturbing position 4 must not change the output at position 1), the cross-attention memory mask blocks padded *source* positions, and — the easy-to-get-wrong direction — changing *non-padded* encoder memory actually does change the decoder's output, proving cross-attention isn't accidentally a no-op.

**Day 12 is the full model (forward pass only — no training loop exists yet).** `test_transformer.py` checks the assembled model end-to-end: output logit shapes, `create_masks()` matches the raw masking function, `encode()`+`decode()`+output projection called manually matches calling the model directly, `from_config()` builds the right shapes, and `share_embeddings=True` genuinely reuses one module/layer (checked with `is`, not just equal values) while rejecting mismatched vocab sizes. One regression test exists specifically because of a bug this build caught itself: the PyTorch model's Xavier-init pass touches every >1-D parameter, which would silently re-randomize the token embedding's zeroed pad row (Day 3) the moment the full model is assembled — `_init_parameters()` now explicitly re-zeros it afterward, and a test locks that in. `test_transformer_parity.py` is the capstone: every weight across both embeddings, every encoder/decoder layer, and the output projection is synced, and the two frameworks must then produce identical logits — with no masks, with real padding+causal masks, and with `share_embeddings=True`.

**Day 13 is the first test that actually trains anything.** `test_overfit.py` (in each framework's own `tests/`, not the shared `tests/`, since it needs a real optimizer) builds a tiny Transformer and trains it with Adam for ~600 full-batch steps on 12 fixed sequences from a trivial copy task (no tokenizer or dataset needed yet — that's Day 14-15). It asserts the loss starts near `ln(vocab_size)` (confirming nothing is already broken at initialization), drops by at least 5x, and the model reaches >90% token accuracy on the memorized sequences. This is the first test that would actually catch a subtly-broken gradient anywhere in the 12-day chain of composed modules — everything before it checked shapes, masks, and cross-framework numerics, but never "can this thing learn anything at all". It's marked `slow` and run by `make test-torch`/`make test-tf`; use the `-fast` targets to skip it during quick iteration.

**Day 14's tokenizer tests are the first ones with zero framework dependency.** `data/bpe_tokenizer.py` and `tests/test_bpe_tokenizer.py` are pure stdlib (same as `config.py`) — `python -m pytest tests/test_bpe_tokenizer.py` runs with no torch or tensorflow installed at all, in either virtualenv or a bare Python install. It checks the BPE algorithm against a hand-computable case (`'aaab'` must merge `('a','a')` first — more mergeable pairs than any other adjacent pair), training determinism (train twice on the same corpus, get byte-identical vocab and merges — what lets both frameworks' Day-15 data loaders train independently and still agree), encode/decode roundtripping, and that the tokenizer's special-token ids line up exactly with `config.py`'s `PAD_IDX`/`BOS_IDX`/`EOS_IDX`/`UNK_IDX`.

## Conventions shared by both implementations

- **Masks are boolean, `True` = may attend, `False` = blocked**, and broadcast to
  `(batch, heads, q_len, k_len)`.
- **Special tokens:** `PAD=0, BOS=1, EOS=2, UNK=3`.
- **One `TransformerConfig`** per framework with identical fields/defaults; a test
  fails if they ever drift.

## License

MIT — see [LICENSE](LICENSE).
