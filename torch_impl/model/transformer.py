"""Full Transformer model (PyTorch) — wires together every piece from Days 2-12.

    src -> src_embed -> Encoder -> memory
    tgt -> tgt_embed -> Decoder(memory) -> output_projection -> logits

Mirrors ``tf_impl/model/transformer.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_impl.config import TransformerConfig
from torch_impl.model.decoder_stack import Decoder
from torch_impl.model.embeddings import TransformerEmbedding
from torch_impl.model.encoder_stack import Encoder
from torch_impl.model.masking import create_masks


class Transformer(nn.Module):
    """The complete encoder-decoder Transformer.

    Construct directly, or via ``Transformer.from_config(config)`` to build
    every piece from a single ``TransformerConfig`` (Day 2) — the way
    ``train.py`` (Day 17) will create the model.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
        norm_first: bool = False,
        pad_idx: int = 0,
        share_embeddings: bool = False,
    ) -> None:
        super().__init__()
        if share_embeddings and src_vocab_size != tgt_vocab_size:
            raise ValueError("share_embeddings requires src_vocab_size == tgt_vocab_size")

        self.pad_idx = pad_idx

        self.src_embed = TransformerEmbedding(src_vocab_size, d_model, max_seq_len, dropout, pad_idx)
        self.tgt_embed = (
            self.src_embed
            if share_embeddings
            else TransformerEmbedding(tgt_vocab_size, d_model, max_seq_len, dropout, pad_idx)
        )

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout, norm_first)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout, norm_first)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self._init_parameters()

    def _init_parameters(self) -> None:
        """Xavier/Glorot-uniform init for every weight matrix with dim > 1,
        as in the paper's reference implementation. Biases and 1-D params
        (LayerNorm gamma/beta) are left at their module defaults.

        This blanket pass would also overwrite the token-embedding pad row
        that ``nn.Embedding(padding_idx=...)`` zeroes at construction (Day 3)
        — so it's explicitly re-zeroed afterward, otherwise the embedding's
        "pad row stays zero" guarantee would silently break the moment a
        full model is assembled.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        with torch.no_grad():
            self.src_embed.token_embedding.embedding.weight[self.pad_idx].zero_()
            self.tgt_embed.token_embedding.embedding.weight[self.pad_idx].zero_()

    @classmethod
    def from_config(cls, config: TransformerConfig) -> "Transformer":
        return cls(
            src_vocab_size=config.src_vocab_size,
            tgt_vocab_size=config.tgt_vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            norm_first=config.norm_first,
            pad_idx=config.pad_idx,
            share_embeddings=config.share_embeddings,
        )

    def create_masks(self, src: torch.Tensor, tgt: torch.Tensor):
        """Convenience wrapper around ``model.masking.create_masks`` using
        this model's own ``pad_idx``, so callers don't have to track it
        separately."""
        return create_masks(src, tgt, self.pad_idx)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decoder(self.tgt_embed(tgt), memory, tgt_mask, memory_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            src: (batch, src_len) source token ids.
            tgt: (batch, tgt_len) target token ids, already shifted right by
                the caller for teacher forcing — this module does not shift.
            src_mask, tgt_mask, memory_mask: see ``model/masking.py``; build
                with ``self.create_masks(src, tgt)`` if not already built.

        Returns:
            (batch, tgt_len, tgt_vocab_size) — unnormalized logits. No
            softmax here; the training loss (Day 16) applies it.
        """
        memory = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, memory, tgt_mask, memory_mask)
        return self.output_projection(decoder_out)
