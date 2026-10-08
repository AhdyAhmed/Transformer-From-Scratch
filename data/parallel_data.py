"""Parallel-corpus loading, batching and teacher-forcing prep — Day 15.

Shared by both frameworks, for the same reason the tokenizer is (Day 14):
nothing here touches torch or tensorflow. It reads line-aligned source/target
text files, encodes them with the shared BPE tokenizer, pads batches, and
builds the shifted decoder input/target pair for teacher forcing — all in
plain Python lists. Each framework's ``dataset.py`` then only converts the
finished batches into its own tensor type.

That split is what delivers the roadmap goal of "both frameworks train on
identical tokenized data": batch *composition and order* come from one
seeded implementation, not from torch's and tf.data's separate shufflers
(which use different RNGs and could never produce the same order).
``tests/test_data_loader_parity.py`` checks the two loaders really do
yield identical batches.

Pipeline:
    text files --read_parallel_corpus--> (src, tgt) string pairs
               --ParallelDataset-------> [BOS, ids..., EOS] id lists (truncated)
               --BatchIterator/collate-> padded Batch(src, tgt_in, tgt_out)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from data.bpe_tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, BPETokenizer


# ---------------------------------------------------------------------- #
# Reading
# ---------------------------------------------------------------------- #
def _read_lines(path: str | Path) -> list[str]:
    """Read a text file as a list of lines, splitting on ``\\n`` only.

    ``str.splitlines()`` also splits on characters like U+2028 and \\x0b, which
    can silently misalign a parallel corpus if one side contains them.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":  # file ended with a newline
        lines.pop()
    return [line.rstrip("\r") for line in lines]


def read_parallel_corpus(src_path: str | Path, tgt_path: str | Path) -> list[tuple[str, str]]:
    """Read two line-aligned files into (source, target) string pairs.

    Raises if the files have different line counts (a misaligned corpus
    would silently train on wrong translations). Pairs where either side is
    blank are dropped.
    """
    src_lines = _read_lines(src_path)
    tgt_lines = _read_lines(tgt_path)
    if len(src_lines) != len(tgt_lines):
        raise ValueError(
            f"parallel files have different line counts: {src_path} has "
            f"{len(src_lines)}, {tgt_path} has {len(tgt_lines)}"
        )
    return [
        (s.strip(), t.strip())
        for s, t in zip(src_lines, tgt_lines)
        if s.strip() and t.strip()
    ]


# ---------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------- #
class ParallelDataset:
    """Encoded (source ids, target ids) pairs, each wrapped as ``[BOS, ..., EOS]``.

    ``max_len`` bounds the *total* length including BOS/EOS, and should be
    the model's ``max_seq_len`` so every sequence fits its positional
    encoding. Longer sentences are truncated, keeping BOS and EOS intact
    (a truncated target must still end in EOS, or the model never sees a
    stop signal for long sentences). Truncated pairs are counted in
    ``num_truncated`` so it's visible how much data a ``max_len`` is costing.
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        tokenizer: BPETokenizer,
        max_len: int = 128,
    ) -> None:
        if max_len < 3:
            raise ValueError(f"max_len must be at least 3 (BOS + one token + EOS), got {max_len}")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_idx = tokenizer.vocab[PAD_TOKEN]
        self._bos = tokenizer.vocab[BOS_TOKEN]
        self._eos = tokenizer.vocab[EOS_TOKEN]

        self.num_truncated = 0
        self.examples: list[tuple[list[int], list[int]]] = []
        for src_text, tgt_text in pairs:
            src_ids, src_cut = self._encode(src_text)
            tgt_ids, tgt_cut = self._encode(tgt_text)
            self.num_truncated += int(src_cut or tgt_cut)
            self.examples.append((src_ids, tgt_ids))

    @classmethod
    def from_files(
        cls,
        src_path: str | Path,
        tgt_path: str | Path,
        tokenizer: BPETokenizer,
        max_len: int = 128,
    ) -> "ParallelDataset":
        return cls(read_parallel_corpus(src_path, tgt_path), tokenizer, max_len)

    def _encode(self, text: str) -> tuple[list[int], bool]:
        ids = self.tokenizer.encode(text)
        room = self.max_len - 2  # leave space for BOS and EOS
        truncated = len(ids) > room
        return [self._bos, *ids[:room], self._eos], truncated

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.examples[index]


# ---------------------------------------------------------------------- #
# Collation
# ---------------------------------------------------------------------- #
@dataclass
class Batch:
    """One padded batch, as plain nested lists (rectangular).

    src:     (batch, src_len)      encoder input, ``[BOS ... EOS PAD*]``
    tgt_in:  (batch, tgt_len)      decoder input = target without its last position
    tgt_out: (batch, tgt_len)      labels = target without its first position (BOS)
    num_tgt_tokens: count of non-pad label positions, for loss normalization
        and tokens/sec reporting.

    The model does not shift targets itself (see ``Transformer.forward``);
    this is where teacher forcing happens.
    """

    src: list[list[int]]
    tgt_in: list[list[int]]
    tgt_out: list[list[int]]
    num_tgt_tokens: int


def pad_sequences(sequences: list[list[int]], pad_idx: int) -> list[list[int]]:
    """Right-pad every sequence to the length of the longest one."""
    longest = max(len(s) for s in sequences)
    return [s + [pad_idx] * (longest - len(s)) for s in sequences]


def collate(examples: list[tuple[list[int], list[int]]], pad_idx: int) -> Batch:
    """Pad a list of (src, tgt) examples into one teacher-forcing ``Batch``.

    For a target ``[BOS a b EOS PAD PAD]``: decoder input is
    ``[BOS a b EOS PAD]`` and labels are ``[a b EOS PAD PAD]``. The PAD
    labels are ignored by the loss (``ignore_index=pad_idx``, Day 16).
    """
    if not examples:
        raise ValueError("cannot collate an empty list of examples")
    src = pad_sequences([e[0] for e in examples], pad_idx)
    tgt = pad_sequences([e[1] for e in examples], pad_idx)
    tgt_in = [row[:-1] for row in tgt]
    tgt_out = [row[1:] for row in tgt]
    num_tgt_tokens = sum(1 for row in tgt_out for tok in row if tok != pad_idx)
    return Batch(src=src, tgt_in=tgt_in, tgt_out=tgt_out, num_tgt_tokens=num_tgt_tokens)


# ---------------------------------------------------------------------- #
# Batching
# ---------------------------------------------------------------------- #
class BatchIterator:
    """Seeded, epoch-aware batching over a ``ParallelDataset``.

    Order is a pure function of ``(seed, epoch)``: ``set_epoch(n)`` then
    iterating always yields the same batches, on any machine and in either
    framework. Iterating twice *without* calling ``set_epoch`` repeats the
    same order — the training loop is expected to call ``set_epoch(epoch)``
    each epoch (the same convention as PyTorch's ``DistributedSampler``).
    """

    def __init__(
        self,
        dataset: ParallelDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            chunk = indices[start : start + self.batch_size]
            if self.drop_last and len(chunk) < self.batch_size:
                break
            yield collate([self.dataset[i] for i in chunk], self.dataset.pad_idx)
