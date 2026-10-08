"""Tests for the shared data pipeline (Day 15): reading, encoding, padding,
teacher-forcing shift, and seeded batching.

Pure stdlib, like ``test_bpe_tokenizer.py`` — no torch or tensorflow needed.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from data.bpe_tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, BPETokenizer, _pretokenize
from data.download_multi30k import fetch_all
from data.parallel_data import (
    BatchIterator,
    ParallelDataset,
    collate,
    pad_sequences,
    read_parallel_corpus,
)

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


def _joint_tokenizer(vocab_size: int = 300) -> BPETokenizer:
    """Joint EN+DE vocab from the sample corpus, as build_vocab would make."""
    corpus = []
    for name in ("train.en", "train.de"):
        corpus.extend((SAMPLE_DIR / name).read_text(encoding="utf-8").splitlines())
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=vocab_size)
    return tok


def _sample_dataset(max_len: int = 64) -> ParallelDataset:
    return ParallelDataset.from_files(
        SAMPLE_DIR / "train.en", SAMPLE_DIR / "train.de", _joint_tokenizer(), max_len=max_len
    )


# --------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------- #

def test_sample_files_are_line_aligned():
    pairs = read_parallel_corpus(SAMPLE_DIR / "train.en", SAMPLE_DIR / "train.de")
    assert len(pairs) == 24
    assert pairs[0][0].startswith("A man in a red shirt")
    assert pairs[0][1].startswith("Ein Mann in einem roten Hemd")


def test_mismatched_line_counts_raise(tmp_path):
    (tmp_path / "a.en").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (tmp_path / "a.de").write_text("eins\nzwei\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different line counts"):
        read_parallel_corpus(tmp_path / "a.en", tmp_path / "a.de")


def test_blank_pairs_are_dropped_and_crlf_is_handled(tmp_path):
    (tmp_path / "a.en").write_bytes(b"hello\r\n\r\nworld\r\n")
    (tmp_path / "a.de").write_bytes(b"hallo\r\nleer\r\nwelt\r\n")
    pairs = read_parallel_corpus(tmp_path / "a.en", tmp_path / "a.de")
    assert pairs == [("hello", "hallo"), ("world", "welt")]


def test_unicode_line_separator_does_not_misalign(tmp_path):
    """str.splitlines() would split on U+2028 and misalign the files."""
    (tmp_path / "a.en").write_text("one\u2028still one\ntwo\n", encoding="utf-8")
    (tmp_path / "a.de").write_text("eins\nzwei\n", encoding="utf-8")
    pairs = read_parallel_corpus(tmp_path / "a.en", tmp_path / "a.de")
    assert len(pairs) == 2


# --------------------------------------------------------------------- #
# Dataset encoding + truncation
# --------------------------------------------------------------------- #

def test_every_example_is_wrapped_in_bos_eos():
    ds = _sample_dataset()
    tok = ds.tokenizer
    bos, eos = tok.vocab[BOS_TOKEN], tok.vocab[EOS_TOKEN]
    assert len(ds) == 24
    for src, tgt in ds.examples:
        assert src[0] == bos and src[-1] == eos
        assert tgt[0] == bos and tgt[-1] == eos


def test_encoded_source_decodes_back_to_the_original_sentence():
    ds = _sample_dataset()
    pairs = read_parallel_corpus(SAMPLE_DIR / "train.en", SAMPLE_DIR / "train.de")
    for (src_ids, tgt_ids), (src_text, tgt_text) in zip(ds.examples[:6], pairs[:6]):
        assert ds.tokenizer.decode(src_ids) == " ".join(_pretokenize(src_text))
        assert ds.tokenizer.decode(tgt_ids) == " ".join(_pretokenize(tgt_text))


def test_truncation_respects_max_len_and_keeps_bos_eos():
    tok = _joint_tokenizer()
    long_text = " ".join(["a man is climbing a rock"] * 20)
    ds = ParallelDataset([(long_text, "ein mann")], tok, max_len=10)
    src, tgt = ds[0]
    assert len(src) == 10
    assert src[0] == tok.vocab[BOS_TOKEN] and src[-1] == tok.vocab[EOS_TOKEN]
    assert len(tgt) < 10  # short side untouched
    assert ds.num_truncated == 1


def test_no_truncation_is_reported_when_everything_fits():
    assert _sample_dataset(max_len=64).num_truncated == 0


def test_max_len_too_small_raises():
    with pytest.raises(ValueError, match="max_len"):
        ParallelDataset([("a", "b")], _joint_tokenizer(), max_len=2)


# --------------------------------------------------------------------- #
# Padding + teacher forcing
# --------------------------------------------------------------------- #

def test_pad_sequences_makes_a_rectangle():
    out = pad_sequences([[1, 2, 3], [4], [5, 6]], pad_idx=0)
    assert out == [[1, 2, 3], [4, 0, 0], [5, 6, 0]]


def test_collate_teacher_forcing_shift():
    # target [BOS=1, a=7, b=8, EOS=2], second target shorter
    batch = collate([([1, 5, 2], [1, 7, 8, 2]), ([1, 6, 2], [1, 9, 2])], pad_idx=0)
    assert batch.tgt_in == [[1, 7, 8], [1, 9, 2]]
    assert batch.tgt_out == [[7, 8, 2], [9, 2, 0]]
    assert batch.src == [[1, 5, 2], [1, 6, 2]]


def test_collate_invariants_on_real_data():
    ds = _sample_dataset()
    pad = ds.pad_idx
    bos, eos = ds.tokenizer.vocab[BOS_TOKEN], ds.tokenizer.vocab[EOS_TOKEN]
    batch = collate([ds[i] for i in range(8)], pad)

    assert len({len(r) for r in batch.src}) == 1  # rectangular
    assert len({len(r) for r in batch.tgt_in}) == 1
    assert len(batch.tgt_in[0]) == len(batch.tgt_out[0])

    for row_in, row_out in zip(batch.tgt_in, batch.tgt_out):
        assert row_in[0] == bos
        assert row_out.count(eos) == 1  # exactly one EOS label per sentence
        eos_pos = row_out.index(eos)
        assert all(t == pad for t in row_out[eos_pos + 1 :])  # only padding after EOS
        # labels are the inputs shifted left by one, wherever both are real
        real = [i for i in range(len(row_out) - 1) if row_out[i] != pad]
        assert all(row_in[i + 1] == row_out[i] for i in real)


def test_num_tgt_tokens_counts_non_pad_labels():
    batch = collate([([1, 5, 2], [1, 7, 8, 2]), ([1, 6, 2], [1, 9, 2])], pad_idx=0)
    assert batch.num_tgt_tokens == 3 + 2  # [7,8,2] + [9,2]


def test_pad_idx_matches_model_config():
    from torch_impl.config import PAD_IDX

    assert _sample_dataset().pad_idx == PAD_IDX


def test_collate_rejects_empty():
    with pytest.raises(ValueError):
        collate([], pad_idx=0)


# --------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------- #

def _flat_src(batches):
    return [tuple(row) for b in batches for row in b.src]


def test_epoch_covers_every_example_exactly_once():
    ds = _sample_dataset()
    it = BatchIterator(ds, batch_size=5, shuffle=True, seed=3)
    seen = []
    for b in it:
        seen.extend(tuple(r) for r in b.src)
    # padding makes rows within a batch longer, so compare after stripping pad
    stripped = sorted(tuple(t for t in row if t != ds.pad_idx) for row in seen)
    expected = sorted(tuple(src) for src, _ in ds.examples)
    assert stripped == expected


def test_len_with_and_without_drop_last():
    ds = _sample_dataset()  # 24 examples
    assert len(BatchIterator(ds, batch_size=5)) == 5  # 4 full + 1 partial
    assert len(BatchIterator(ds, batch_size=5, drop_last=True)) == 4
    assert len(BatchIterator(ds, batch_size=8)) == 3
    assert len(list(BatchIterator(ds, batch_size=5))) == 5
    assert len(list(BatchIterator(ds, batch_size=5, drop_last=True))) == 4


def test_drop_last_batches_are_all_full():
    ds = _sample_dataset()
    for b in BatchIterator(ds, batch_size=5, drop_last=True):
        assert len(b.src) == 5


def test_same_seed_and_epoch_gives_identical_batches():
    ds = _sample_dataset()
    a = BatchIterator(ds, batch_size=4, seed=7)
    b = BatchIterator(ds, batch_size=4, seed=7)
    assert _flat_src(a) == _flat_src(b)


def test_iterating_twice_without_set_epoch_repeats_the_order():
    ds = _sample_dataset()
    it = BatchIterator(ds, batch_size=4, seed=7)
    assert _flat_src(it) == _flat_src(it)


def test_different_epochs_give_different_orders():
    ds = _sample_dataset()
    it = BatchIterator(ds, batch_size=4, seed=7)
    it.set_epoch(0)
    first = _flat_src(it)
    it.set_epoch(1)
    second = _flat_src(it)
    assert first != second


def test_different_seeds_give_different_orders():
    ds = _sample_dataset()
    assert _flat_src(BatchIterator(ds, 4, seed=1)) != _flat_src(BatchIterator(ds, 4, seed=2))


def test_set_epoch_makes_order_reproducible():
    ds = _sample_dataset()
    it = BatchIterator(ds, batch_size=4, seed=7)
    it.set_epoch(3)
    run_one = _flat_src(it)
    it.set_epoch(0)
    _flat_src(it)
    it.set_epoch(3)
    assert _flat_src(it) == run_one


def test_no_shuffle_preserves_dataset_order():
    ds = _sample_dataset()
    it = BatchIterator(ds, batch_size=24, shuffle=False)
    (batch,) = list(it)
    for row, (src, _) in zip(batch.src, ds.examples):
        assert [t for t in row if t != ds.pad_idx] == src


def test_bad_batch_size_raises():
    with pytest.raises(ValueError, match="batch_size"):
        BatchIterator(_sample_dataset(), batch_size=0)


# --------------------------------------------------------------------- #
# Download script (network call itself is untestable offline; everything
# after it — gunzip, naming, directory layout — is tested via file:// URLs)
# --------------------------------------------------------------------- #

def test_fetch_all_unpacks_and_renames_via_file_urls(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    for stem in ("train", "val", "test_2016_flickr"):
        for lang in ("en", "de"):
            with gzip.open(remote / f"{stem}.{lang}.gz", "wb") as f:
                f.write(f"{stem} {lang}\nline two\n".encode("utf-8"))

    out = tmp_path / "out"
    written = fetch_all(base_url=remote.as_uri(), out_dir=out)

    assert len(written) == 6
    assert (out / "train.en").read_text(encoding="utf-8") == "train en\nline two\n"
    assert (out / "test.de").read_text(encoding="utf-8") == "test_2016_flickr de\nline two\n"
    assert not (out / "test_2016_flickr.en").exists()  # renamed to test.*
