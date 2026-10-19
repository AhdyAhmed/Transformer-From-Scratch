# Day 27 — Documentation validation

Day 27 adds the portfolio-facing documentation layer:

- Root overview, architecture, measured results, quick start, and navigation.
- Accessible scalable architecture diagram at
  [`assets/transformer-architecture.svg`](../../assets/transformer-architecture.svg).
- Dedicated [`torch_impl/README.md`](../../torch_impl/README.md) and
  [`tf_impl/README.md`](../../tf_impl/README.md) setup/usage guides.
- Automated local-link, SVG accessibility/structure, and results-table
  consistency tests.

## Commands actually validated

Both documented one-epoch CPU smoke runs completed on the bundled 24-pair
training corpus and five-pair validation corpus. Each produced a tokenizer,
best/latest checkpoint, configuration, history, translations, and loss curve
under its Git-ignored `checkpoints/smoke-*` directory.

The root PyTorch quick-start inference command then loaded the newly generated
`best.pt` and tokenizer successfully. Translation quality is intentionally not
an acceptance criterion for a one-epoch smoke run; the test proves the copied
setup → train → save → load → infer workflow is operational.

The production-checkpoint inference outputs remain under
[`results/day26/`](../day26/), and the consolidated measured comparison remains
under [`results/day25/`](../day25/).
