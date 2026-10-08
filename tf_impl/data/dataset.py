"""TensorFlow batch loader (Day 15).

A thin tensor-conversion layer over the shared, framework-agnostic
``data.parallel_data`` pipeline. It deliberately does *not* build a
shuffling ``tf.data.Dataset`` pipeline: that would shuffle with
TensorFlow's RNG, while ``torch_impl/data/dataset.py`` would shuffle with
torch's, and the two frameworks would then never see the same batches in
the same order — defeating the point of a framework comparison. Batch
composition and order come from ``BatchIterator`` (seeded, shared); this
class only turns each finished batch into ``tf.int32`` tensors.

(Wrapping the same iterator with ``tf.data.Dataset.from_generator`` for
prefetching is a straightforward later optimization if input becomes the
bottleneck; at this repo's scale it won't be.)

Mirrors ``torch_impl/data/dataset.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import tensorflow as tf

from data.parallel_data import BatchIterator, ParallelDataset


@dataclass
class TFBatch:
    """src (B, S), tgt_in (B, T), tgt_out (B, T) — all ``tf.int32``.

    Feed ``src`` and ``tgt_in`` to the model (``model.create_masks(src, tgt_in)``
    builds the masks); compare its logits against ``tgt_out``.
    """

    src: tf.Tensor
    tgt_in: tf.Tensor
    tgt_out: tf.Tensor
    num_tgt_tokens: int


class TFBatchLoader:
    def __init__(
        self,
        dataset: ParallelDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self._iterator = BatchIterator(dataset, batch_size, shuffle, seed, drop_last)

    def set_epoch(self, epoch: int) -> None:
        self._iterator.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self._iterator)

    def __iter__(self) -> Iterator[TFBatch]:
        for batch in self._iterator:
            yield TFBatch(
                src=tf.constant(batch.src, dtype=tf.int32),
                tgt_in=tf.constant(batch.tgt_in, dtype=tf.int32),
                tgt_out=tf.constant(batch.tgt_out, dtype=tf.int32),
                num_tgt_tokens=batch.num_tgt_tokens,
            )
