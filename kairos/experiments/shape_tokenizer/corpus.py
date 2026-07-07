from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time
from math import sqrt
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from kairos.experiments.protocol import SplitProtocol, split_name
from kairos.experiments.shape_tokenizer.baselines import (
    boundary_aware_fit_rows,
    fit_boundary_aware_kmeans,
    is_interior_cell,
    split_masks,
)
from kairos.experiments.shape_tokenizer.vq import (
    coarse_body_quantile_thresholds,
    coarse_class_ids,
)


VOCAB_FULL = "full"
VOCAB_INTERIOR = "interior"
VOCAB_BOUNDARY = "boundary"
VOCAB_SPECS = {
    VOCAB_FULL: (0, 40),
    VOCAB_INTERIOR: (0, 31),
    VOCAB_BOUNDARY: (32, 40),
}


@dataclass(frozen=True, slots=True)
class BigramStats:
    vocab: str
    split: str
    unigram_entropy: float
    conditional_entropy: float
    information_gain: float
    token_count: int
    bigram_count: int


@dataclass(frozen=True, slots=True)
class MutualInformationDecomposition:
    vocab: str
    information_gain: float
    diagonal_contribution: float
    off_diagonal_contribution: float
    bigram_count: int


@dataclass(frozen=True, slots=True)
class SecondOrderStats:
    vocab: str
    bigram_conditional_entropy: float
    trigram_conditional_entropy: float
    second_order_information_gain: float
    bigram_count: int
    trigram_count: int


def read_shape_sample_rows(run_dir: Path) -> list[dict[str, Any]]:
    table_path = run_dir / "tables" / "shape_sample.csv"
    with table_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [_shape_row_from_csv(row) for row in reader]


def _shape_row_from_csv(row: dict[str, str]) -> dict[str, Any]:
    return {
        "timestamp": row["timestamp"],
        "symbol": row["symbol"],
        "lambda_o": float(row["lambda_o"]),
        "lambda_c": float(row["lambda_c"]),
        "s1": float(row["s1"]),
        "s2": float(row["s2"]),
        "is_zero_range": row["is_zero_range"] == "True",
        "is_boundary": row["is_boundary"] == "True",
        "rel_range": _optional_float(row.get("rel_range")),
        "gap": _optional_float(row.get("gap")),
    }


def _optional_float(value: str | None) -> float | None:
    if value in {None, "", "None", "nan"}:
        return None
    return float(value)


def ohlc_rows_from_candles(candles: Iterable[Any]) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": str(candle.timestamp),
            "symbol": str(candle.symbol),
            "open": float(candle.open),
            "high": float(candle.high),
            "low": float(candle.low),
            "close": float(candle.close),
            "volume": int(getattr(candle, "volume", 0) or 0),
        }
        for candle in candles
    ]


def join_shape_ohlc(
    shape_rows: list[dict[str, Any]],
    ohlc_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ohlc_by_key = {
        (row["timestamp"], row["symbol"]): row
        for row in ohlc_rows
    }
    joined = [
        shape_row | ohlc_by_key[(shape_row["timestamp"], shape_row["symbol"])]
        for shape_row in shape_rows
        if (shape_row["timestamp"], shape_row["symbol"]) in ohlc_by_key
    ]
    coverage = len(joined) / len(shape_rows) if shape_rows else 1.0
    return joined, {
        "shape_row_count": len(shape_rows),
        "ohlc_row_count": len(ohlc_rows),
        "joined_row_count": len(joined),
        "join_coverage": coverage,
    }


def token_rows_by_seed(
    rows: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...],
    split: SplitProtocol,
    codebook_size: int = 32,
    eps: float = 1e-3,
) -> dict[int, np.ndarray]:
    return {
        seed: fit_boundary_aware_kmeans(
            rows,
            codebook_size=codebook_size,
            seed=seed,
            split=split,
            eps=eps,
        )[0]
        for seed in seeds
    }


def assign_coarse_classes(
    rows: list[dict[str, Any]],
    *,
    split: SplitProtocol,
    body_bins: int = 4,
    eps: float = 1e-3,
) -> tuple[np.ndarray, dict[str, Any]]:
    coarse = np.full(len(rows), -1, dtype=int)
    interior_rows = boundary_aware_fit_rows(rows, eps=eps)
    masks = split_masks(interior_rows, split=split)
    train_rows = [
        row for row, is_train in zip(interior_rows, masks["train"], strict=True) if is_train
    ]
    thresholds = coarse_body_quantile_thresholds(train_rows, body_bins=body_bins)
    interior_classes = coarse_class_ids(
        interior_rows,
        thresholds=thresholds,
        body_bins=body_bins,
    )
    interior_index = 0
    for row_index, row in enumerate(rows):
        if not row["is_zero_range"] and is_interior_cell(row, eps=eps):
            coarse[row_index] = int(interior_classes[interior_index])
            interior_index += 1
    return coarse, {
        "body_bins": body_bins,
        "body_quantile_thresholds": thresholds.tolist(),
        "train_interior_row_count": len(train_rows),
    }


def compute_williams_fractals(
    rows: list[dict[str, Any]],
    *,
    interval: str,
    n: int = 2,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Return strict Williams Fractal pivots grouped by symbol and 1m session."""
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for group_rows in _fractal_groups(rows, interval=interval):
        ordered = sorted(group_rows, key=lambda row: str(row["timestamp"]))
        for index, row in enumerate(ordered):
            key = (row["timestamp"], row["symbol"])
            high = low = False
            confirmed_at = None
            if n <= index < len(ordered) - n:
                left = ordered[index - n : index]
                right = ordered[index + 1 : index + n + 1]
                pivot_high = float(row["high"])
                pivot_low = float(row["low"])
                high = all(pivot_high > float(item["high"]) for item in (*left, *right))
                low = all(pivot_low < float(item["low"]) for item in (*left, *right))
                if high or low:
                    confirmed_at = ordered[index + n]["timestamp"]
            output[key] = {
                "fractal_high": high,
                "fractal_low": low,
                "fractal_confirmed_at": confirmed_at,
            }
    return output


def _fractal_groups(rows: list[dict[str, Any]], *, interval: str) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        session = str(row["timestamp"])[:10] if interval == "1m" else None
        grouped[(str(row["symbol"]), session)].append(row)
    return list(grouped.values())


def build_corpus_rows(
    joined_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    *,
    canonical_tokens: np.ndarray,
    coarse_classes: np.ndarray,
    fractals: dict[tuple[str, str], dict[str, Any]],
    split: SplitProtocol,
) -> list[dict[str, Any]]:
    label_by_key = {(row["timestamp"], row["symbol"]): row for row in label_rows}
    corpus: list[dict[str, Any]] = []
    for index, row in enumerate(joined_rows):
        key = (row["timestamp"], row["symbol"])
        label = label_by_key.get(key, {})
        fractal = fractals.get(
            key,
            {
                "fractal_high": False,
                "fractal_low": False,
                "fractal_confirmed_at": None,
            },
        )
        corpus.append(
            {
                "timestamp": row["timestamp"],
                "symbol": row["symbol"],
                "split": split_name(row["timestamp"], split),
                "token": int(canonical_tokens[index]),
                "is_boundary": bool(row["is_boundary"]),
                "is_zero_range": bool(row["is_zero_range"]),
                "coarse_class": int(coarse_classes[index]),
                **fractal,
                "rel_range": row.get("rel_range"),
                "gap": row.get("gap"),
                **{
                    name: value
                    for name, value in label.items()
                    if name not in {"timestamp", "symbol", "split", "close", "ma_200"}
                },
            }
        )
    return sorted(corpus, key=lambda item: (item["symbol"], item["timestamp"]))


def grouped_token_sequences(
    rows: list[dict[str, Any]],
    *,
    split: str | None = None,
    interval: str = "1d",
) -> list[list[int]]:
    grouped: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if split is not None and row.get("split") != split:
            continue
        session = str(row["timestamp"])[:10] if interval == "1m" else None
        grouped[(str(row["symbol"]), session)].append(row)
    return [
        [int(row["token"]) for row in sorted(items, key=lambda item: str(item["timestamp"]))]
        for items in grouped.values()
        if items
    ]


def entropy_from_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def information_gain(
    sequences: list[list[int]],
    *,
    vocab: str = VOCAB_FULL,
) -> BigramStats:
    low, high = VOCAB_SPECS[vocab]
    values = list(range(low, high + 1))
    index_by_token = {token: index for index, token in enumerate(values)}
    unigrams = np.zeros(len(values), dtype=int)
    bigrams = np.zeros((len(values), len(values)), dtype=int)
    for sequence in sequences:
        filtered_positions = [
            (position, token)
            for position, token in enumerate(sequence)
            if low <= token <= high
        ]
        for _position, token in filtered_positions:
            unigrams[index_by_token[token]] += 1
        for (prev_position, prev), (next_position, nxt) in zip(
            filtered_positions,
            filtered_positions[1:],
            strict=False,
        ):
            if next_position == prev_position + 1:
                bigrams[index_by_token[prev], index_by_token[nxt]] += 1
    unigram_entropy = entropy_from_counts(unigrams)
    conditional_entropy = conditional_entropy_from_bigrams(bigrams)
    return BigramStats(
        vocab=vocab,
        split="",
        unigram_entropy=unigram_entropy,
        conditional_entropy=conditional_entropy,
        information_gain=unigram_entropy - conditional_entropy,
        token_count=int(unigrams.sum()),
        bigram_count=int(bigrams.sum()),
    )


def bigram_count_matrix(
    sequences: list[list[int]],
    *,
    vocab: str = VOCAB_FULL,
) -> np.ndarray:
    low, high = VOCAB_SPECS[vocab]
    values = list(range(low, high + 1))
    index_by_token = {token: index for index, token in enumerate(values)}
    bigrams = np.zeros((len(values), len(values)), dtype=int)
    for sequence in sequences:
        for prev, nxt in zip(sequence, sequence[1:], strict=False):
            if low <= prev <= high and low <= nxt <= high:
                bigrams[index_by_token[prev], index_by_token[nxt]] += 1
    return bigrams


def mutual_information_decomposition(
    sequences: list[list[int]],
    *,
    vocab: str = VOCAB_FULL,
) -> MutualInformationDecomposition:
    """Decompose adjacent-token mutual information into diagonal/off-diagonal cells."""
    bigrams = bigram_count_matrix(sequences, vocab=vocab)
    total = int(bigrams.sum())
    if total == 0:
        return MutualInformationDecomposition(vocab, 0.0, 0.0, 0.0, 0)

    joint = bigrams / total
    prev_probs = joint.sum(axis=1)
    next_probs = joint.sum(axis=0)
    expected = np.outer(prev_probs, next_probs)
    contributions = np.zeros_like(joint, dtype=float)
    mask = joint > 0
    contributions[mask] = joint[mask] * np.log2(joint[mask] / expected[mask])
    diagonal = float(np.trace(contributions))
    total_mi = float(contributions.sum())
    return MutualInformationDecomposition(
        vocab=vocab,
        information_gain=total_mi,
        diagonal_contribution=diagonal,
        off_diagonal_contribution=float(total_mi - diagonal),
        bigram_count=total,
    )


def run_length_encode(sequence: list[int]) -> list[int]:
    if not sequence:
        return []
    encoded = [sequence[0]]
    for token in sequence[1:]:
        if token != encoded[-1]:
            encoded.append(token)
    return encoded


def rle_token_sequences(sequences: list[list[int]]) -> list[list[int]]:
    return [run_length_encode(sequence) for sequence in sequences]


def token_run_lengths(sequences: list[list[int]]) -> dict[int, list[int]]:
    runs: dict[int, list[int]] = defaultdict(list)
    for sequence in sequences:
        if not sequence:
            continue
        current = sequence[0]
        length = 1
        for token in sequence[1:]:
            if token == current:
                length += 1
            else:
                runs[current].append(length)
                current = token
                length = 1
        runs[current].append(length)
    return dict(runs)


def conditional_entropy_from_bigrams(bigrams: np.ndarray) -> float:
    total = int(bigrams.sum())
    if total == 0:
        return 0.0
    entropy = 0.0
    row_sums = bigrams.sum(axis=1)
    for row_index, row_total in enumerate(row_sums):
        if row_total == 0:
            continue
        entropy += (row_total / total) * entropy_from_counts(bigrams[row_index])
    return float(entropy)


def second_order_information_gain(
    sequences: list[list[int]],
    *,
    vocab: str = VOCAB_FULL,
) -> SecondOrderStats:
    low, high = VOCAB_SPECS[vocab]
    bigrams = bigram_count_matrix(sequences, vocab=vocab)
    contexts: dict[tuple[int, int], list[int]] = defaultdict(
        lambda: [0] * (high - low + 1)
    )
    trigram_count = 0
    for sequence in sequences:
        for first, second, third in zip(
            sequence,
            sequence[1:],
            sequence[2:],
            strict=False,
        ):
            if low <= first <= high and low <= second <= high and low <= third <= high:
                contexts[(first, second)][third - low] += 1
                trigram_count += 1

    trigram_entropy = 0.0
    for counts in contexts.values():
        context_total = sum(counts)
        if context_total:
            trigram_entropy += (context_total / trigram_count) * entropy_from_counts(
                np.array(counts, dtype=int)
            )
    bigram_entropy = conditional_entropy_from_bigrams(bigrams)
    return SecondOrderStats(
        vocab=vocab,
        bigram_conditional_entropy=bigram_entropy,
        trigram_conditional_entropy=float(trigram_entropy),
        second_order_information_gain=float(bigram_entropy - trigram_entropy),
        bigram_count=int(bigrams.sum()),
        trigram_count=trigram_count,
    )


def markov1_transition_matrix(
    sequences: list[list[int]],
    *,
    vocab: str = VOCAB_FULL,
    smoothing: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    low, _high = VOCAB_SPECS[vocab]
    bigrams = bigram_count_matrix(sequences, vocab=vocab).astype(float)
    transitions = bigrams + smoothing
    transitions = transitions / transitions.sum(axis=1, keepdims=True)
    initials = np.zeros(transitions.shape[0], dtype=float)
    for sequence in sequences:
        if sequence:
            first = sequence[0]
            if VOCAB_SPECS[vocab][0] <= first <= VOCAB_SPECS[vocab][1]:
                initials[first - low] += 1.0
    if initials.sum() == 0.0:
        initials += 1.0
    initials = initials / initials.sum()
    return transitions, initials


def generate_markov1_sequences(
    lengths: Iterable[int],
    *,
    transitions: np.ndarray,
    initials: np.ndarray,
    vocab: str = VOCAB_FULL,
    rng: np.random.Generator,
) -> list[list[int]]:
    low, high = VOCAB_SPECS[vocab]
    values = np.arange(low, high + 1, dtype=int)
    generated: list[list[int]] = []
    for length in lengths:
        if length <= 0:
            generated.append([])
            continue
        current = int(rng.choice(values, p=initials))
        sequence = [current]
        for _ in range(length - 1):
            current = int(rng.choice(values, p=transitions[current - low]))
            sequence.append(current)
        generated.append(sequence)
    return generated


def assign_intraday_bucket(timestamp: str | datetime) -> str | None:
    """Assign a KR regular-session timestamp to open/middle/close buckets."""
    stamp = timestamp if isinstance(timestamp, datetime) else _parse_timestamp(timestamp)
    current = stamp.time()
    if time(9, 0) <= current < time(9, 30):
        return "open_30m"
    if time(9, 30) <= current < time(15, 0):
        return "middle"
    if time(15, 0) <= current <= time(15, 30):
        return "close_30m"
    return None


def _parse_timestamp(timestamp: str) -> datetime:
    normalized = timestamp.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")


def surrogate_information_gain(
    sequences: list[list[int]],
    *,
    vocab: str,
    repeats: int = 100,
    seed: int = 7,
) -> dict[str, Any]:
    observed = information_gain(sequences, vocab=vocab).information_gain
    rng = np.random.default_rng(seed)
    surrogate_values: list[float] = []
    for _ in range(repeats):
        shuffled = shuffle_token_sequences(sequences, rng=rng)
        surrogate_values.append(information_gain(shuffled, vocab=vocab).information_gain)
    values = np.array(surrogate_values, dtype=float)
    std = float(values.std())
    return {
        "observed_information_gain": observed,
        "surrogate_mean": float(values.mean()),
        "surrogate_std": std,
        "surrogate_quantile": float(np.mean(values <= observed)),
        "z_score": None if std == 0.0 else float((observed - values.mean()) / std),
        "surrogate_values": surrogate_values,
    }


def shuffle_token_sequences(
    sequences: list[list[int]],
    *,
    rng: np.random.Generator,
) -> list[list[int]]:
    shuffled: list[list[int]] = []
    for sequence in sequences:
        copy = np.array(sequence, dtype=int)
        rng.shuffle(copy)
        shuffled.append(copy.tolist())
    return shuffled


def token_distribution(tokens: Iterable[int], *, vocab_size: int = 41) -> np.ndarray:
    counts = np.zeros(vocab_size, dtype=float)
    for token in tokens:
        value = int(token)
        if 0 <= value < vocab_size:
            counts[value] += 1.0
    total = counts.sum()
    return counts / total if total else counts


def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)
    return float(sqrt(0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)))


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def unigram_kl(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def count_invalid_symbol_adjacencies(rows: list[dict[str, Any]]) -> int:
    ordered = sorted(rows, key=lambda item: str(item["timestamp"]))
    return sum(
        1
        for prev, nxt in zip(ordered, ordered[1:], strict=False)
        if prev["symbol"] != nxt["symbol"]
    )


def count_minute_cross_session_bigrams(rows: list[dict[str, Any]]) -> int:
    count = 0
    for sequence_rows in _fractal_groups(rows, interval="1d"):
        ordered = sorted(sequence_rows, key=lambda row: str(row["timestamp"]))
        count += sum(
            1
            for prev, nxt in zip(ordered, ordered[1:], strict=False)
            if str(prev["timestamp"])[:10] != str(nxt["timestamp"])[:10]
        )
    return count
