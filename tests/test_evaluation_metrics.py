import pytest

from evaluation.metrics import corpus_bleu


def test_perfect_corpus_bleu_is_100_and_case_insensitive():
    result = corpus_bleu(
        ["Ein Mann schläft auf einem Sofa."],
        ["ein mann schläft auf einem sofa."],
    )
    assert result.score == pytest.approx(100.0)
    assert result.brevity_penalty == 1.0
    assert result.length_ratio == 1.0


def test_bleu_penalizes_mismatch_without_becoming_non_finite():
    result = corpus_bleu(
        ["ein hund rennt schnell"],
        ["eine frau liest ein buch"],
    )
    assert 0.0 < result.score < 100.0
    assert all(value >= 0.0 for value in result.precisions)


def test_bleu_applies_brevity_penalty():
    result = corpus_bleu(
        ["ein mann schläft"],
        ["ein mann schläft auf einem grünen sofa"],
    )
    assert 0.0 < result.brevity_penalty < 1.0
    assert result.length_ratio < 1.0


def test_bleu_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="same number"):
        corpus_bleu(["one"], [])
    with pytest.raises(ValueError, match="at least one"):
        corpus_bleu([], [])
    with pytest.raises(ValueError, match="max_order"):
        corpus_bleu(["one"], ["one"], max_order=0)
