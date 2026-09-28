"""Hyperparameter configuration for the Transformer.

This file is intentionally *identical* to ``tf_impl/config.py`` (stdlib only,
no framework imports) so both implementations train with exactly the same
settings. ``tests/test_config_parity.py`` enforces that they never drift.

Defaults describe a small "base-lite" model that trains quickly on Multi30k,
not the full 65M-parameter model from the paper. ``paper_base()`` returns the
original hyperparameters for reference.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# Special token ids, shared by tokenizer, masking and loss code.
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


@dataclass
class TransformerConfig:
    # --- model size ---
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 3          # applies to both encoder and decoder
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 128
    norm_first: bool = False     # False = post-norm (paper), True = pre-norm

    # --- vocabulary ---
    src_vocab_size: int = 8000
    tgt_vocab_size: int = 8000
    pad_idx: int = PAD_IDX
    bos_idx: int = BOS_IDX
    eos_idx: int = EOS_IDX
    unk_idx: int = UNK_IDX
    share_embeddings: bool = False   # tie src/tgt/output embeddings (needs shared vocab)

    # --- optimisation (paper: Adam + warmup / inverse-sqrt decay) ---
    warmup_steps: int = 4000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9
    label_smoothing: float = 0.1
    batch_size: int = 64
    num_epochs: int = 20
    grad_clip: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ #
    @property
    def d_k(self) -> int:
        """Per-head key/query dimension."""
        return self.d_model // self.num_heads

    def validate(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {self.label_smoothing}"
            )
        for name in ("d_model", "num_heads", "num_layers", "d_ff", "max_seq_len"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.share_embeddings and self.src_vocab_size != self.tgt_vocab_size:
            raise ValueError("share_embeddings requires src_vocab_size == tgt_vocab_size")

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerConfig":
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TransformerConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def paper_base(cls, **overrides) -> "TransformerConfig":
        """Hyperparameters of the 'base' model from Vaswani et al. (2017)."""
        params = dict(d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1)
        params.update(overrides)
        return cls(**params)
