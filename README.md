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
├── tests/                             # framework-agnostic tests
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
│   ├── model/
│   │   ├── masking.py               # padding / look-ahead / decoder masks
│   │   ├── embeddings.py            # TokenEmbedding + sinusoidal PositionalEncoding
│   │   ├── attention.py             # scaled dot-product attention
│   │   ├── multi_head_attention.py  # Q/K/V projections + head split/merge
│   │   ├── feed_forward.py          # position-wise FFN
│   │   ├── residual.py              # residual + LayerNorm wrapper (pre/post-norm)
│   │   ├── encoder.py               # EncoderLayer: self-attn + FFN sublayers
│   │   ├── encoder_stack.py         # Encoder: N EncoderLayers + final norm
│   │   ├── decoder.py               # DecoderLayer: masked self-attn + cross-attn + FFN
│   │   ├── decoder_stack.py         # Decoder: N DecoderLayers + final norm
│   │   └── transformer.py           # Transformer: full encoder-decoder model
│   ├── data/                           # tokenizer, dataset (later)
│   └── tests/
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
- [ ] Day 13 — Overfit test (tiny-data sanity check)
- [ ] Day 14–18 — Data pipeline & training setup
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
make test-torch        # == python -m pytest tests torch_impl

# TensorFlow env
make test-tf           # == python -m pytest tests tf_impl
```

`tests/test_config_parity.py`, `tests/test_embeddings_parity.py`, `tests/test_attention_parity.py`, `tests/test_mha_parity.py`, `tests/test_ffn_residual_parity.py` and `tests/test_encoder_parity.py` only run if *both* frameworks are importable in the active environment; otherwise pytest reports them as skipped, which is expected.

The parameter-free ones (config, positional encoding, raw attention) compare math directly on identical inputs. The weight-bearing ones (`MultiHeadAttention`, `PositionwiseFeedForward`, `ResidualConnection`'s LayerNorm, and now the full `EncoderLayer`) use `tests/weight_sync.py` to copy the *same* weights from the PyTorch module into its TensorFlow twin first — accounting for PyTorch's `(out, in)` Linear layout vs. Keras' `(in, out)` Dense layout — before comparing outputs. `test_mha_parity.py` additionally includes a negative control confirming the two frameworks genuinely diverge *without* syncing, so parity can't trivially pass for the wrong reason.

LayerNorm's epsilon is pinned to `1e-6` explicitly on both sides — PyTorch's `nn.LayerNorm` default is `1e-5` and Keras' `LayerNormalization` default is `1e-3`, and leaving either at its default would have silently broken this parity check.

`torch_impl/tests/test_encoder.py` and `tf_impl/tests/test_encoder.py` also each check a structural property, not just shapes: perturbing *only* the padded positions of the input must leave the encoder's output at the real positions unchanged. That's an easy property to silently break (e.g. a mask applied to the wrong axis) and a cheap, high-value thing to test for.

`test_encoder_stack.py` (Day 10) checks that property *survives stacking* N layers deep, and adds two more stack-specific checks: the N layers must have genuinely independent weights (not six references to one set — that would silently collapse the stack's capacity to a single layer), and a deeper stack must actually produce a different output than a shallow one on the same input.

`test_decoder.py` (Day 11) checks three analogous invariants for the decoder layer: the causal mask blocks future target positions (perturbing position 4 must not change the output at position 1), the cross-attention memory mask blocks padded *source* positions, and — the easy-to-get-wrong direction — changing *non-padded* encoder memory actually does change the decoder's output, proving cross-attention isn't accidentally a no-op.

**Day 12 is the full model.** `test_transformer.py` checks the assembled model end-to-end: output logit shapes, `create_masks()` matches the raw masking function, `encode()`+`decode()`+output projection called manually matches calling the model directly, `from_config()` builds the right shapes, and `share_embeddings=True` genuinely reuses one module/layer (checked with `is`, not just equal values) while rejecting mismatched vocab sizes. One regression test exists specifically because of a bug this build caught itself: the PyTorch model's Xavier-init pass touches every >1-D parameter, which would silently re-randomize the token embedding's zeroed pad row (Day 3) the moment the full model is assembled — `_init_parameters()` now explicitly re-zeros it afterward, and a test locks that in. `test_transformer_parity.py` is the capstone: every weight across both embeddings, every encoder/decoder layer, and the output projection is synced, and the two frameworks must then produce identical logits — with no masks, with real padding+causal masks, and with `share_embeddings=True`.

## Conventions shared by both implementations

- **Masks are boolean, `True` = may attend, `False` = blocked**, and broadcast to
  `(batch, heads, q_len, k_len)`.
- **Special tokens:** `PAD=0, BOS=1, EOS=2, UNK=3`.
- **One `TransformerConfig`** per framework with identical fields/defaults; a test
  fails if they ever drift.

## License

MIT — see [LICENSE](LICENSE).
