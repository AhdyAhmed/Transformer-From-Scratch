"""PyTorch batch loader (Day 15).

A thin tensor-conversion layer over the shared, framework-agnostic
``data.parallel_data`` pipeline. It deliberately does *not* use
``torch.utils.data.DataLoader``: that would shuffle with torch's own RNG,
while ``tf_impl/data/dataset.py`` would shuffle with TensorFlow's, and the
two frameworks would then never see the same batches in the same order —
defeating the point of a framework comparison. Batch composition and order
come from ``BatchIterator`` (seeded, shared); this class only turns each
finished batch into ``torch.long`` tensors.

Mirrors ``tf_impl/data/dataset.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch

from data.parallel_data import BatchIterator, ParallelDataset


@dataclass
class TorchBatch:
    """src (B, S), tgt_in (B, T), tgt_out (B, T) — all ``torch.long``.

    Feed ``src`` and ``tgt_in`` to the model (``model.create_masks(src, tgt_in)``
    builds the masks); compare its logits against ``tgt_out``.
    """

    src: torch.Tensor
    tgt_in: torch.Tensor
    tgt_out: torch.Tensor
    num_tgt_tokens: int

    def to(self, device) -> "TorchBatch":
        return TorchBatch(
            self.src.to(device),
            self.tgt_in.to(device),
            self.tgt_out.to(device),
            self.num_tgt_tokens,
        )


class TorchBatchLoader:
    def __init__(
        self,
        dataset: ParallelDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        self._iterator = BatchIterator(dataset, batch_size, shuffle, seed, drop_last)
        self.device = device

    def set_epoch(self, epoch: int) -> None:
        self._iterator.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self._iterator)

    def __iter__(self) -> Iterator[TorchBatch]:
        for batch in self._iterator:
            out = TorchBatch(
                src=torch.tensor(batch.src, dtype=torch.long),
                tgt_in=torch.tensor(batch.tgt_in, dtype=torch.long),
                tgt_out=torch.tensor(batch.tgt_out, dtype=torch.long),
                num_tgt_tokens=batch.num_tgt_tokens,
            )
            yield out.to(self.device) if self.device is not None else out
