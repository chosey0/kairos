from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log2
from typing import Sequence


@dataclass(frozen=True, slots=True)
class TransitionReport:
    counts: dict[tuple[int, int], int]
    probabilities: dict[tuple[int, int], float]
    entropy_by_source: dict[int, float]


def transition_counts(tokens: Sequence[int]) -> dict[tuple[int, int], int]:
    """Count Phase 2 one-step token transitions."""
    counts: Counter[tuple[int, int]] = Counter()
    for current_token, next_token in zip(tokens, tokens[1:], strict=False):
        counts[(current_token, next_token)] += 1
    return dict(sorted(counts.items()))


def transition_report(tokens: Sequence[int]) -> TransitionReport:
    counts = transition_counts(tokens)
    totals_by_source: Counter[int] = Counter()
    for (source, _target), count in counts.items():
        totals_by_source[source] += count

    probabilities = {
        pair: count / totals_by_source[pair[0]]
        for pair, count in counts.items()
        if totals_by_source[pair[0]] > 0
    }
    entropy_by_source: dict[int, float] = {}
    for source in sorted(totals_by_source):
        entropy = 0.0
        for (pair_source, _target), probability in probabilities.items():
            if pair_source == source and probability > 0:
                entropy -= probability * log2(probability)
        entropy_by_source[source] = entropy

    return TransitionReport(
        counts=counts,
        probabilities=dict(sorted(probabilities.items())),
        entropy_by_source=entropy_by_source,
    )
