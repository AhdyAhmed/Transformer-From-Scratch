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
├── tests/                   # framework-agnostic tests
│   ├── weight_sync.py         # copy weights from a torch module into its tf twin
│   ├── test_mha_parity.py     # weight-matched MultiHeadAttention parity (Day 6)
│   ├── test_ffn_residual_parity.py  # weight-matched FFN/LayerNorm parity (Day 8)
│   └── ...                    # config / embeddings / attention parity
├── torch_impl/                # PyTorch implementation
│   ├── config.py                # TransformerConfig (identical to tf_impl/config.py)
│   ├── model/
│   │   ├── masking.py               # padding / look-ahead / decoder masks
│   │   ├── embeddings.py            # TokenEmbedding + sinusoidal PositionalEncoding
│   │   ├── attention.py             # scaled dot-product attention
│   │   ├── multi_head_attention.py  # Q/K/V projections + head split/merge
│   │   ├── feed_forward.py          # position-wise FFN
│   │   └── residual.py              # residual + LayerNorm wrapper (pre/post-norm)
│   ├── data/                 # tokenizer, dataset (later)
│   └── tests/
├── tf_impl/                    # TensorFlow implementation, mirrors torch_impl/
├── notebooks/                   # attention visualization, framework comparison
└── assets/diagrams/              # architecture diagrams, attention heatmaps
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
- [ ] Day 9–13 — Encoder/decoder stacks & overfit test
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

`tests/test_config_parity.py`, `tests/test_embeddings_parity.py`, `tests/test_attention_parity.py`, `tests/test_mha_parity.py` and `tests/test_ffn_residual_parity.py` only run if *both* frameworks are importable in the active environment; otherwise pytest reports them as skipped, which is expected.

The parameter-free ones (config, positional encoding, raw attention) compare math directly on identical inputs. The weight-bearing ones (`MultiHeadAttention`, `PositionwiseFeedForward`, `ResidualConnection`'s LayerNorm) use `tests/weight_sync.py` to copy the *same* weights from the PyTorch module into its TensorFlow twin first — accounting for PyTorch's `(out, in)` Linear layout vs. Keras' `(in, out)` Dense layout — before comparing outputs. `test_mha_parity.py` additionally includes a negative control confirming the two frameworks genuinely diverge *without* syncing, so parity can't trivially pass for the wrong reason.

LayerNorm's epsilon is pinned to `1e-6` explicitly on both sides — PyTorch's `nn.LayerNorm` default is `1e-5` and Keras' `LayerNormalization` default is `1e-3`, and leaving either at its default would have silently broken this parity check.

## Conventions shared by both implementations

- **Masks are boolean, `True` = may attend, `False` = blocked**, and broadcast to
  `(batch, heads, q_len, k_len)`.
- **Special tokens:** `PAD=0, BOS=1, EOS=2, UNK=3`.
- **One `TransformerConfig`** per framework with identical fields/defaults; a test
  fails if they ever drift.

## License

MIT — see [LICENSE](LICENSE).
