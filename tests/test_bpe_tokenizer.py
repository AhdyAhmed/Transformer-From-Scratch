"""Tests for the shared from-scratch BPE tokenizer (Day 14).

Unlike every parity test since Day 6, this file needs neither torch nor
tensorflow — ``data/bpe_tokenizer.py`` is pure stdlib — so unlike most of
this repo's later-day tests, these actually run in this environment rather
than only being syntax-checked.
"""

from __future__ import annotations

from data.bpe_tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    END_OF_WORD,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNK_TOKEN,
    BPETokenizer,
)
from torch_impl.config import BOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX

TOY_CORPUS = [
    "a man in a red shirt is climbing a rock",
    "two young children are playing in the park",
    "a dog runs across the green field",
    "the woman is reading a book on the bench",
]


# --------------------------------------------------------------------- #
# Special tokens must agree with the model's own ids (torch_impl/config.py
# is stdlib-only, so this needs no torch install to check)
# --------------------------------------------------------------------- #

def test_special_token_ids_match_config():
    tok = BPETokenizer()
    tok.train(TOY_CORPUS, vocab_size=50)
    assert tok.vocab[PAD_TOKEN] == PAD_IDX == 0
    assert tok.vocab[BOS_TOKEN] == BOS_IDX == 1
    assert tok.vocab[EOS_TOKEN] == EOS_IDX == 2
    assert tok.vocab[UNK_TOKEN] == UNK_IDX == 3


# --------------------------------------------------------------------- #
# Correctness of the BPE algorithm itself, against a hand-computable case
# --------------------------------------------------------------------- #

def test_first_merge_matches_hand_computation():
    """Classic textbook example: 'aaab' pre-tokenizes to symbols
    ['a','a','a','b</w>']. Adjacent pairs: ('a','a') appears twice
    (positions 0-1 and 1-2), ('a','b</w>') appears once. The most frequent
    pair must be ('a','a'), merged into 'aa'."""
    tok = BPETokenizer()
    tok.train(["aaab"], vocab_size=len(SPECIAL_TOKENS) + 10)
    assert tok.merges[0] == ("a", "a")


def test_merge_count_is_bounded_by_corpus_exhaustion():
    """A single short, simple word runs out of mergeable pairs long before
    any large vocab_size is reached — train() must stop early rather than
    looping forever or erroring."""
    tok = BPETokenizer()
    tok.train(["ab"], vocab_size=1000)
    assert tok.vocab_size < 1000


def test_vocab_size_never_exceeded():
    tok = BPETokenizer()
    tok.train(TOY_CORPUS, vocab_size=60)
    assert tok.vocab_size <= 60


def test_rejects_vocab_size_smaller_than_special_tokens():
    import pytest

    tok = BPETokenizer()
    with pytest.raises(ValueError, match="vocab_size"):
        tok.train(TOY_CORPUS, vocab_size=2)


# --------------------------------------------------------------------- #
# Determinism — the property both frameworks' independent training runs
# (Day 15) depend on to end up with identical vocabularies
# --------------------------------------------------------------------- #

def test_training_is_deterministic():
    tok_a = BPETokenizer()
    tok_a.train(TOY_CORPUS, vocab_size=80)
    tok_b = BPETokenizer()
    tok_b.train(TOY_CORPUS, vocab_size=80)
    assert tok_a.vocab == tok_b.vocab
    assert tok_a.merges == tok_b.merges


# --------------------------------------------------------------------- #
# Encode / decode behavior
# --------------------------------------------------------------------- #

def test_encode_returns_ints_decode_roundtrips_known_words():
    tok = BPETokenizer()
    tok.train(TOY_CORPUS, vocab_size=120)

    ids = tok.encode("a red shirt")
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0

    decoded = tok.decode(ids)
    # lowercased + whitespace-normalized, so exact string equality is the
    # right bar here (the corpus was already lowercase, single-spaced)
    assert decoded == "a red shirt"


def test_add_special_tokens_wraps_with_bos_eos():
    tok = BPETokenizer()
    tok.train(TOY_CORPUS, vocab_size=120)
    ids = tok.encode("a dog", add_special_tokens=True)
    assert ids[0] == tok.vocab[BOS_TOKEN]
    assert ids[-1] == tok.vocab[EOS_TOKEN]


def test_decode_skips_special_tokens_by_default():
    tok = BPETokenizer()
    tok.train(TOY_CORPUS, vocab_size=120)
    ids = tok.encode("a dog", add_special_tokens=True)
    decoded = tok.decode(ids)
    assert BOS_TOKEN not in decoded and EOS_TOKEN not in decoded
    assert decoded == "a dog"


def test_unknown_character_falls_back_to_unk():
    tok = BPETokenizer()
    tok.train(TOY_CORPUS, vocab_size=120)
    ids = tok.encode("\u00e9\u00e9\u00e9")  # a character never seen in training
    assert all(i == tok.vocab[UNK_TOKEN] for i in ids)


def test_encode_batch_matches_individual_encode_calls():
    tok = BPETokenizer()
    tok.train(TOY_CORPUS, vocab_size=120)
    texts = ["a dog", "the park"]
    batch = tok.encode_batch(texts)
    individual = [tok.encode(t) for t in texts]
    assert batch == individual


def test_end_of_word_marker_never_leaks_into_decoded_text():
    tok = BPETokenizer()
    tok.train(TOY_CORPUS, vocab_size=120)
    decoded = tok.decode(tok.encode("a man in a red shirt"))
    assert END_OF_WORD not in decoded


# --------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------- #

def test_save_load_roundtrip_preserves_behavior(tmp_path):
    tok = BPETokenizer()
    tok.train(TOY_CORPUS, vocab_size=100)

    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)

    assert loaded.vocab == tok.vocab
    assert loaded.merges == tok.merges

    text = "a man in a red shirt"
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(tok.encode(text)) == tok.decode(tok.encode(text))
