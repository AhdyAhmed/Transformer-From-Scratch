import math

import pytest
import torch

from torch_impl.model.embeddings import (
    PositionalEncoding,
    TokenEmbedding,
    TransformerEmbedding,
)

D_MODEL = 16
VOCAB = 50
PAD = 0


def test_token_embedding_output_shape():
    emb = TokenEmbedding(VOCAB, D_MODEL, PAD)
    x = torch.randint(1, VOCAB, (4, 10))
    out = emb(x)
    assert out.shape == (4, 10, D_MODEL)


def test_token_embedding_scaling_factor():
    torch.manual_seed(0)
    emb = TokenEmbedding(VOCAB, D_MODEL, PAD)
    x = torch.randint(1, VOCAB, (2, 5))
    scaled = emb(x)
    raw = emb.embedding(x)
    ratio = (scaled / raw)[raw != 0]
    assert torch.allclose(ratio, torch.full_like(ratio, math.sqrt(D_MODEL)), atol=1e-5)


def test_padding_idx_row_starts_at_zero():
    emb = TokenEmbedding(VOCAB, D_MODEL, PAD)
    assert torch.all(emb.embedding.weight[PAD] == 0)


def test_padding_idx_gradient_is_zero():
    emb = TokenEmbedding(VOCAB, D_MODEL, PAD)
    x = torch.tensor([[PAD, 3, 4]])
    emb(x).sum().backward()
    assert torch.all(emb.embedding.weight.grad[PAD] == 0)
    assert torch.any(emb.embedding.weight.grad[3] != 0)


def test_positional_encoding_rejects_odd_d_model():
    with pytest.raises(ValueError, match="even"):
        PositionalEncoding(d_model=15)


def test_positional_encoding_shape():
    pe = PositionalEncoding(D_MODEL, max_len=20, dropout=0.0).eval()
    x = torch.zeros(3, 7, D_MODEL)
    out = pe(x)
    assert out.shape == (3, 7, D_MODEL)


def test_positional_encoding_position_zero_values():
    """At position 0: sin(0)=0 on even dims, cos(0)=1 on odd dims."""
    pe = PositionalEncoding(D_MODEL, max_len=20, dropout=0.0).eval()
    pos0 = pe.pe[0, 0]
    assert torch.allclose(pos0[0::2], torch.zeros(D_MODEL // 2), atol=1e-6)
    assert torch.allclose(pos0[1::2], torch.ones(D_MODEL // 2), atol=1e-6)


def test_positional_encoding_values_are_bounded():
    pe = PositionalEncoding(D_MODEL, max_len=100, dropout=0.0)
    assert pe.pe.abs().max() <= 1.0 + 1e-6


def test_positional_encoding_is_added_not_replaced():
    pe = PositionalEncoding(D_MODEL, max_len=20, dropout=0.0).eval()
    x = torch.ones(1, 5, D_MODEL)
    out = pe(x)
    assert torch.allclose(out, x + pe.pe[:, :5])


def test_positional_encoding_rejects_too_long_sequence():
    pe = PositionalEncoding(D_MODEL, max_len=8, dropout=0.0)
    x = torch.zeros(1, 9, D_MODEL)
    with pytest.raises(ValueError, match="exceeds max_len"):
        pe(x)


def test_positional_encoding_is_not_a_saved_parameter():
    pe = PositionalEncoding(D_MODEL, max_len=20)
    assert "pe" not in pe.state_dict()  # persistent=False -> not checkpointed
    assert "pe" not in dict(pe.named_parameters())  # buffer, not a learned weight


def test_transformer_embedding_end_to_end():
    emb = TransformerEmbedding(VOCAB, D_MODEL, max_len=20, dropout=0.0, pad_idx=PAD).eval()
    x = torch.randint(1, VOCAB, (2, 6))
    out = emb(x)
    assert out.shape == (2, 6, D_MODEL)
