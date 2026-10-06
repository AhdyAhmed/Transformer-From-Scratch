import pytest
import torch

from torch_impl.config import TransformerConfig
from torch_impl.model.transformer import Transformer

SRC_VOCAB = 50
TGT_VOCAB = 60
D_MODEL = 16
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 32
PAD = 0


def make_model(**overrides):
    kwargs = dict(
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=20,
        dropout=0.0,
        pad_idx=PAD,
    )
    kwargs.update(overrides)
    return Transformer(**kwargs).eval()


def test_output_logits_shape():
    model = make_model()
    src = torch.randint(1, SRC_VOCAB, (2, 7))
    tgt = torch.randint(1, TGT_VOCAB, (2, 5))
    logits = model(src, tgt)
    assert logits.shape == (2, 5, TGT_VOCAB)


def test_create_masks_convenience_matches_module_level_function():
    from torch_impl.model.masking import create_masks as raw_create_masks

    model = make_model()
    src = torch.tensor([[1, 2, 3, PAD]])
    tgt = torch.tensor([[4, 5, PAD]])
    src_mask, tgt_mask, memory_mask = model.create_masks(src, tgt)
    expected_src, expected_tgt, expected_mem = raw_create_masks(src, tgt, PAD)
    assert torch.equal(src_mask, expected_src)
    assert torch.equal(tgt_mask, expected_tgt)
    assert torch.equal(memory_mask, expected_mem)


def test_forward_with_masks_runs_end_to_end():
    model = make_model()
    src = torch.tensor([[1, 2, 3, PAD, PAD], [4, 5, 6, 7, PAD]])
    tgt = torch.tensor([[1, 2, PAD], [3, 4, 5]])
    src_mask, tgt_mask, memory_mask = model.create_masks(src, tgt)
    logits = model(src, tgt, src_mask, tgt_mask, memory_mask)
    assert logits.shape == (2, 3, TGT_VOCAB)


def test_encode_decode_match_full_forward():
    """encode()+decode()+output_projection, called manually, must match
    calling forward() directly — forward is just a convenience composition."""
    model = make_model()
    src = torch.randint(1, SRC_VOCAB, (1, 5))
    tgt = torch.randint(1, TGT_VOCAB, (1, 4))

    full = model(src, tgt)
    memory = model.encode(src)
    manual = model.output_projection(model.decode(tgt, memory))
    assert torch.allclose(full, manual, atol=1e-6)


def test_from_config_builds_matching_architecture():
    config = TransformerConfig(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        dropout=0.0,
        max_seq_len=20,
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        pad_idx=PAD,
    )
    model = Transformer.from_config(config)
    assert len(model.encoder.layers) == NUM_LAYERS
    assert len(model.decoder.layers) == NUM_LAYERS
    assert model.output_projection.out_features == TGT_VOCAB
    assert model.src_embed.token_embedding.embedding.embedding_dim == D_MODEL


def test_share_embeddings_reuses_the_same_module():
    model = make_model(src_vocab_size=SRC_VOCAB, tgt_vocab_size=SRC_VOCAB, share_embeddings=True)
    assert model.src_embed is model.tgt_embed


def test_share_embeddings_rejects_mismatched_vocab_sizes():
    with pytest.raises(ValueError, match="share_embeddings"):
        make_model(src_vocab_size=SRC_VOCAB, tgt_vocab_size=TGT_VOCAB, share_embeddings=True)


def test_padding_row_stays_zero_after_full_model_init():
    """Regression test: Transformer._init_parameters applies Xavier init to
    every >1-D parameter, which would also overwrite the token embedding's
    pad row if not explicitly re-zeroed afterward."""
    model = make_model()
    assert torch.all(model.src_embed.token_embedding.embedding.weight[PAD] == 0)
    assert torch.all(model.tgt_embed.token_embedding.embedding.weight[PAD] == 0)


def test_gradients_flow_from_logits_to_embeddings():
    model = make_model()
    src = torch.randint(1, SRC_VOCAB, (2, 5))
    tgt = torch.randint(1, TGT_VOCAB, (2, 4))
    logits = model(src, tgt)
    logits.sum().backward()
    assert model.src_embed.token_embedding.embedding.weight.grad is not None
    assert model.tgt_embed.token_embedding.embedding.weight.grad is not None
    assert model.output_projection.weight.grad.abs().sum() > 0
