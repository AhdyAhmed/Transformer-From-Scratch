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
transformer-from-scratch/
├── README.md
├── design.md              # full architecture & design decisions
├── roadmap.md              # day-by-day build plan
├── requirements-torch.txt
├── requirements-tf.txt
├── pytorch/
│   ├── model/               # embeddings, attention, encoder, decoder, transformer
│   ├── data/                # tokenizer, dataset
│   ├── train.py
│   ├── inference.py
│   ├── config.py
│   └── tests/
├── tensorflow/               # mirrors pytorch/ module-for-module
├── notebooks/                # attention visualization, framework comparison
└── assets/diagrams/          # architecture diagrams, attention heatmaps
```

## Status / Progress

- [x] Day 1 — Repo scaffolding & environment setup
- [ ] Day 2 — Config & masking utilities
- [ ] Day 3 — Embeddings & positional encoding
- [ ] Day 4–7 — Attention mechanism (+ numerical parity check)
- [ ] Day 8–13 — Encoder/decoder stacks & overfit test
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

## License

MIT — see [LICENSE](LICENSE).
