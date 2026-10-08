from pathlib import Path

import numpy as np
import tensorflow as tf

from data.bpe_tokenizer import BPETokenizer
from data.parallel_data import BatchIterator, ParallelDataset
from tf_impl.data.dataset import TFBatch, TFBatchLoader
from tf_impl.model.transformer import Transformer

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"
MAX_LEN = 32


def _tokenizer() -> BPETokenizer:
    corpus = []
    for name in ("train.en", "train.de"):
        corpus.extend((SAMPLE_DIR / name).read_text(encoding="utf-8").splitlines())
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=300)
    return tok


def _dataset() -> ParallelDataset:
    return ParallelDataset.from_files(
        SAMPLE_DIR / "train.en", SAMPLE_DIR / "train.de", _tokenizer(), max_len=MAX_LEN
    )


def test_batches_are_int32_tensors_with_expected_shapes():
    loader = TFBatchLoader(_dataset(), batch_size=6, shuffle=False)
    batch = next(iter(loader))
    assert isinstance(batch, TFBatch)
    for t in (batch.src, batch.tgt_in, batch.tgt_out):
        assert t.dtype == tf.int32
        assert t.shape.rank == 2 and t.shape[0] == 6
    assert batch.tgt_in.shape == batch.tgt_out.shape


def test_tensor_values_match_the_shared_iterator():
    ds = _dataset()
    loader = TFBatchLoader(ds, batch_size=5, seed=11)
    reference = BatchIterator(ds, batch_size=5, seed=11)
    for tensor_batch, plain_batch in zip(loader, reference):
        assert tensor_batch.src.numpy().tolist() == plain_batch.src
        assert tensor_batch.tgt_in.numpy().tolist() == plain_batch.tgt_in
        assert tensor_batch.tgt_out.numpy().tolist() == plain_batch.tgt_out
        assert tensor_batch.num_tgt_tokens == plain_batch.num_tgt_tokens


def test_len_and_set_epoch():
    loader = TFBatchLoader(_dataset(), batch_size=5, seed=1)
    assert len(loader) == 5
    loader.set_epoch(0)
    first = [b.src.numpy().tolist() for b in loader]
    loader.set_epoch(1)
    second = [b.src.numpy().tolist() for b in loader]
    assert first != second


def test_batch_shapes_work_with_the_model_masking_utilities():
    """The roadmap's Day 15 check: batch shapes must line up with Day 2's masks."""
    ds = _dataset()
    tok = ds.tokenizer
    model = Transformer(
        src_vocab_size=tok.vocab_size,
        tgt_vocab_size=tok.vocab_size,
        d_model=16,
        num_heads=4,
        num_layers=1,
        d_ff=32,
        max_seq_len=MAX_LEN,
        dropout=0.0,
        pad_idx=ds.pad_idx,
        share_embeddings=True,
    )

    batch = next(iter(TFBatchLoader(ds, batch_size=6, shuffle=False)))
    src_mask, tgt_mask, memory_mask = model.create_masks(batch.src, batch.tgt_in)
    b, s = batch.src.shape
    t = batch.tgt_in.shape[1]
    assert src_mask.shape == (b, 1, 1, s)
    assert tgt_mask.shape == (b, 1, t, t)
    assert memory_mask.shape == (b, 1, 1, s)

    # padded source positions are blocked; real ones are visible
    np.testing.assert_array_equal(
        src_mask.numpy()[:, 0, 0, :], batch.src.numpy() != ds.pad_idx
    )
    # causal: no position sees a later one
    assert not np.triu(tgt_mask.numpy()[0, 0], k=1).any()


def test_real_batch_flows_through_the_model_to_a_finite_loss():
    ds = _dataset()
    tok = ds.tokenizer
    model = Transformer(
        src_vocab_size=tok.vocab_size,
        tgt_vocab_size=tok.vocab_size,
        d_model=16,
        num_heads=4,
        num_layers=1,
        d_ff=32,
        max_seq_len=MAX_LEN,
        dropout=0.0,
        pad_idx=ds.pad_idx,
        share_embeddings=True,
    )

    batch = next(iter(TFBatchLoader(ds, batch_size=6, shuffle=False)))
    src_mask, tgt_mask, memory_mask = model.create_masks(batch.src, batch.tgt_in)
    logits = model(
        batch.src,
        batch.tgt_in,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
        memory_mask=memory_mask,
        training=False,
    )
    assert logits.shape == (*batch.tgt_in.shape, tok.vocab_size)

    # masked cross-entropy over non-pad labels (ignore_index equivalent)
    per_token = tf.keras.losses.sparse_categorical_crossentropy(
        batch.tgt_out, logits, from_logits=True
    )
    keep = tf.cast(tf.not_equal(batch.tgt_out, ds.pad_idx), per_token.dtype)
    loss = tf.reduce_sum(per_token * keep) / tf.reduce_sum(keep)
    assert np.isfinite(loss.numpy()) and loss.numpy() > 0
