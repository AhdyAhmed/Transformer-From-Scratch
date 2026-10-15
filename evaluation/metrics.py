"""Dependency-free corpus translation metrics used by Day 23."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from data.bpe_tokenizer import _pretokenize


@dataclass(frozen=True)
class BleuResult:
    score: float
    precisions: tuple[float, ...]
    brevity_penalty: float
    length_ratio: float
    hypothesis_length: int
    reference_length: int

    def to_dict(self) -> dict[str, float | int | list[float]]:
        return {
            "score": self.score,
            "precisions": list(self.precisions),
            "brevity_penalty": self.brevity_penalty,
            "length_ratio": self.length_ratio,
            "hypothesis_length": self.hypothesis_length,
            "reference_length": self.reference_length,
        }


def _ngrams(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i:i + order]) for i in range(len(tokens) - order + 1))


def corpus_bleu(
    hypotheses: list[str], references: list[str], max_order: int = 4
) -> BleuResult:
    """Compute lowercase corpus BLEU with exponential zero-count smoothing.

    Text uses the project's Unicode word/punctuation pre-tokenizer. Modified
    n-gram counts, corpus-level brevity penalty, and exponential smoothing are
    compatible with the standard BLEU formulation while remaining completely
    reproducible without an external metric package.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must contain the same number of items")
    if not hypotheses:
        raise ValueError("at least one hypothesis/reference pair is required")
    if max_order < 1:
        raise ValueError("max_order must be positive")

    correct = [0] * max_order
    total = [0] * max_order
    hypothesis_length = 0
    reference_length = 0
    for hypothesis, reference in zip(hypotheses, references):
        hyp_tokens = _pretokenize(hypothesis)
        ref_tokens = _pretokenize(reference)
        hypothesis_length += len(hyp_tokens)
        reference_length += len(ref_tokens)
        for index in range(max_order):
            order = index + 1
            hyp_counts = _ngrams(hyp_tokens, order)
            ref_counts = _ngrams(ref_tokens, order)
            correct[index] += sum((hyp_counts & ref_counts).values())
            total[index] += sum(hyp_counts.values())

    precisions = []
    smooth = 1.0
    effective_order = 0
    for matches, possible in zip(correct, total):
        if possible == 0:
            precisions.append(0.0)
            continue
        effective_order += 1
        if matches == 0:
            smooth *= 2.0
            precision = 100.0 / (smooth * possible)
        else:
            precision = 100.0 * matches / possible
        precisions.append(precision)

    if hypothesis_length == 0 or effective_order == 0:
        brevity_penalty = 0.0
        score = 0.0
    else:
        brevity_penalty = (
            1.0
            if hypothesis_length >= reference_length
            else math.exp(1.0 - reference_length / hypothesis_length)
        )
        log_precision = sum(
            math.log(value / 100.0) for value in precisions[:effective_order]
        ) / effective_order
        score = 100.0 * brevity_penalty * math.exp(log_precision)

    length_ratio = hypothesis_length / reference_length if reference_length else 0.0
    return BleuResult(
        score=score,
        precisions=tuple(precisions),
        brevity_penalty=brevity_penalty,
        length_ratio=length_ratio,
        hypothesis_length=hypothesis_length,
        reference_length=reference_length,
    )
