from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log2, sqrt
from typing import Iterable, Sequence

from .features import ShapeFeatures


@dataclass(frozen=True, slots=True)
class TokenUtilization:
    codebook_size: int
    utilized_count: int
    dead_count: int
    dead_ratio: float
    entropy: float
    histogram: dict[int, int]


def token_utilization(tokens: Iterable[int], *, codebook_size: int) -> TokenUtilization:
    """Measure Phase 1 codebook usage for shape quantization."""
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")

    histogram = dict(sorted(Counter(tokens).items()))
    utilized_count = sum(1 for token in range(codebook_size) if histogram.get(token, 0) > 0)
    dead_count = codebook_size - utilized_count
    total = sum(histogram.values())
    entropy = 0.0
    if total > 0:
        for count in histogram.values():
            probability = count / total
            entropy -= probability * log2(probability)

    return TokenUtilization(
        codebook_size=codebook_size,
        utilized_count=utilized_count,
        dead_count=dead_count,
        dead_ratio=dead_count / codebook_size,
        entropy=entropy,
        histogram=histogram,
    )


def semantic_consistency(tokens: Sequence[int], features: Sequence[ShapeFeatures]) -> dict[int, float]:
    """Measure whether candles assigned to the same shape token are similar."""
    if len(tokens) != len(features):
        raise ValueError("tokens and features must have the same length")

    groups: dict[int, list[ShapeFeatures]] = {}
    for token, feature in zip(tokens, features, strict=True):
        groups.setdefault(token, []).append(feature)

    return {token: _mean_distance(group) for token, group in sorted(groups.items())}


def _mean_distance(features: Sequence[ShapeFeatures]) -> float:
    if len(features) <= 1:
        return 0.0

    vectors = [feature.as_tuple() for feature in features]
    dimensions = range(len(vectors[0]))
    centroid = tuple(sum(vector[index] for vector in vectors) / len(vectors) for index in dimensions)
    distances = []
    for vector in vectors:
        squared = sum((vector[index] - centroid[index]) ** 2 for index in dimensions)
        distances.append(sqrt(squared))
    return sum(distances) / len(distances)
