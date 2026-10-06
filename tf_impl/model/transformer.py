"""Full Transformer model (TensorFlow) — wires together every piece from Days 2-12.

    src -> src_embed -> Encoder -> memory
    tgt -> tgt_embed -> Decoder(memory) -> output_projection -> logits

Mirrors ``torch_impl/model/transformer.py``.
"""

from __future__ import annotations

import tensorflow as tf

from tf_impl.config import TransformerConfig
from tf_impl.model.decoder_stack import Decoder
from tf_impl.model.embeddings import TransformerEmbedding
from tf_impl.model.encoder_stack import Encoder
from tf_impl.model.masking import create_masks


class Transformer(tf.keras.Model):
    """The complete encoder-decoder Transformer.

    Construct directly, or via ``Transformer.from_config(config)`` to build
    every piece from a single ``TransformerConfig`` (Day 2) — the way
    ``train.py`` (Day 18) will create the model.

    Unlike the PyTorch side, no manual re-initialization is needed here:
    Keras' default initializers — ``glorot_uniform`` for ``Dense`` and
    ``Embedding``, ``ones``/``zeros`` for ``LayerNormalization``'s
    gamma/beta — already match the paper's Xavier-init convention, and
    Keras' ``Embedding`` layer has no ``padding_idx`` concept to accidentally
    clobber (see the documented framework gap in ``design.md``, Day 3).
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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
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
        self.output_projection = tf.keras.layers.Dense(tgt_vocab_size)

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

    def create_masks(self, src: tf.Tensor, tgt: tf.Tensor):
        """Convenience wrapper around ``model.masking.create_masks`` using
        this model's own ``pad_idx``, so callers don't have to track it
        separately."""
        return create_masks(src, tgt, self.pad_idx)

    def encode(self, src: tf.Tensor, src_mask: tf.Tensor | None = None, training: bool | None = None) -> tf.Tensor:
        return self.encoder(self.src_embed(src, training=training), mask=src_mask, training=training)

    def decode(
        self,
        tgt: tf.Tensor,
        memory: tf.Tensor,
        tgt_mask: tf.Tensor | None = None,
        memory_mask: tf.Tensor | None = None,
        training: bool | None = None,
    ) -> tf.Tensor:
        return self.decoder(
            self.tgt_embed(tgt, training=training),
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            training=training,
        )

    def call(
        self,
        src: tf.Tensor,
        tgt: tf.Tensor,
        src_mask: tf.Tensor | None = None,
        tgt_mask: tf.Tensor | None = None,
        memory_mask: tf.Tensor | None = None,
        training: bool | None = None,
    ) -> tf.Tensor:
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
        memory = self.encode(src, src_mask, training=training)
        decoder_out = self.decode(tgt, memory, tgt_mask, memory_mask, training=training)
        return self.output_projection(decoder_out)
