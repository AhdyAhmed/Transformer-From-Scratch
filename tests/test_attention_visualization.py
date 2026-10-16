import numpy as np
import pytest

from data.bpe_tokenizer import BPETokenizer
from visualization.attention import (
    AttentionRecord,
    resolve_layer,
    token_labels,
    validate_attention,
)


def _record() -> AttentionRecord:
    encoder = np.full((2, 3, 3), 1 / 3, dtype=np.float32)
    decoder = np.zeros((2, 2, 2), dtype=np.float32)
    decoder[:, 0, 0] = 1.0
    decoder[:, 1, :] = 0.5
    cross = np.full((2, 2, 3), 1 / 3, dtype=np.float32)
    return AttentionRecord(
        framework="test",
        source="source",
        reference="reference",
        translation="translation",
        layer=0,
        source_tokens=["<bos>", "a▁", "<eos>"],
        target_tokens=["<bos>", "b▁"],
        encoder_self=encoder,
        decoder_self=decoder,
        decoder_cross=cross,
    )


def test_validate_attention_accepts_well_formed_probability_maps():
    validate_attention(_record())


@pytest.mark.parametrize("field", ["encoder_self", "decoder_self", "decoder_cross"])
def test_validate_attention_rejects_non_normalized_rows(field):
    record = _record()
    getattr(record, field)[0, 0] = 0.0
    with pytest.raises(ValueError, match="sum to one"):
        validate_attention(record)


def test_validate_attention_rejects_shape_mismatch():
    record = _record()
    record.decoder_cross = record.decoder_cross[:, :, :-1]
    with pytest.raises(ValueError, match="expected"):
        validate_attention(record)


def test_validate_attention_rejects_decoder_future_token_leakage():
    record = _record()
    record.decoder_self[:, 0] = [0.5, 0.5]
    with pytest.raises(ValueError, match="future target tokens"):
        validate_attention(record)


def test_resolve_layer_supports_negative_indices_and_checks_bounds():
    assert resolve_layer(-1, 3) == 2
    assert resolve_layer(0, 3) == 0
    with pytest.raises(ValueError, match="out of range"):
        resolve_layer(3, 3)


def test_token_labels_make_end_of_word_visible():
    tokenizer = BPETokenizer()
    tokenizer.inverse_vocab = {0: "<bos>", 1: "cat</w>"}
    assert token_labels(tokenizer, [0, 1]) == ["<bos>", "cat▁"]
