"""Both frameworks must train on identical batches in identical order (Day 15).

The shared ``BatchIterator`` is what guarantees this; this test checks the
two tensor-conversion loaders don't break it (a wrong dtype, a transposed
tensor, or a framework-side reshuffle would all show up here).

Runs only if both torch and tensorflow are importable in the current
environment; otherwise skipped.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("tensorflow")

from data.bpe_tokenizer import BPETokenizer
from data.parallel_data import ParallelDataset
from tf_impl.data.dataset import TFBatchLoader
from torch_impl.data.dataset import TorchBatchLoader

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


def _dataset(max_len: int = 32) -> ParallelDataset:
    corpus = []
    for name in ("train.en", "train.de"):
        corpus.extend((SAMPLE_DIR / name).read_text(encoding="utf-8").splitlines())
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=300)
    return ParallelDataset.from_files(
        SAMPLE_DIR / "train.en", SAMPLE_DIR / "train.de", tok, max_len=max_len
    )


@pytest.mark.parametrize("epoch", [0, 1, 4])
def test_both_loaders_yield_identical_batches(epoch):
    ds = _dataset()
    torch_loader = TorchBatchLoader(ds, batch_size=5, seed=123)
    tf_loader = TFBatchLoader(ds, batch_size=5, seed=123)
    torch_loader.set_epoch(epoch)
    tf_loader.set_epoch(epoch)

    torch_batches = list(torch_loader)
    tf_batches = list(tf_loader)
    assert len(torch_batches) == len(tf_batches) == len(torch_loader) == len(tf_loader)

    for tb, fb in zip(torch_batches, tf_batches):
        np.testing.assert_array_equal(tb.src.numpy(), fb.src.numpy())
        np.testing.assert_array_equal(tb.tgt_in.numpy(), fb.tgt_in.numpy())
        np.testing.assert_array_equal(tb.tgt_out.numpy(), fb.tgt_out.numpy())
        assert tb.num_tgt_tokens == fb.num_tgt_tokens
