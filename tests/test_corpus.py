from __future__ import annotations

from datetime import date, timedelta
from math import log, sqrt

import numpy as np
import pytest

from kairos.experiments.protocol import SplitProtocol
from kairos.experiments.shape_tokenizer.corpus import (
    VOCAB_FULL,
    assign_intraday_bucket,
    assign_coarse_classes,
    bigram_count_matrix,
    compute_williams_fractals,
    entropy_from_counts,
    generate_markov1_sequences,
    grouped_token_sequences,
    information_gain,
    jensen_shannon_distance,
    markov1_transition_matrix,
    mutual_information_decomposition,
    run_length_encode,
    shuffle_token_sequences,
)
from kairos.experiments.shape_tokenizer.labels import (
    compute_label_rows,
    max_drawdown,
)


def split() -> SplitProtocol:
    return SplitProtocol(
        train_start="2020-01-01",
        train_end="2020-12-31",
        validation_start="2021-01-01",
        validation_end="2021-12-31",
        test_start="2022-01-01",
        test_end=None,
        embargo_days=20,
        label_horizons=(1, 5, 20),
        rolling_statistics_rule="test",
    )


def ohlc_row(day: int, close: float, *, symbol: str = "A") -> dict:
    timestamp = (date(2020, 1, 1) + timedelta(days=day)).isoformat()
    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "open": close,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
    }


def test_labels_compute_forward_return_rv_drawdown_and_embargo() -> None:
    rows = [ohlc_row(day, 100 + day) for day in range(260)]

    labels, metrics = compute_label_rows(rows, split=split(), interval="1d")
    first = labels[0]

    assert first["fwd_log_return_5"] == pytest.approx(log(105 / 100))
    expected_rv = sqrt((log(1.02 / 0.98) ** 2) / (4 * log(2)))
    assert first["fwd_rv_5"] == pytest.approx(expected_rv)
    assert labels[4]["trailing_rv_5"] == pytest.approx(expected_rv)
    assert max_drawdown(np.array([100.0, 110.0, 90.0, 95.0])) == pytest.approx(20 / 110)
    assert metrics["thresholds"]["direction_thr_1"] is not None

    short_rows = [ohlc_row(day, 100 + day) for day in range(3)]
    short_labels, _ = compute_label_rows(short_rows, split=split(), interval="1d")
    assert short_labels[-1]["fwd_log_return_1"] is None
    assert short_labels[-1]["label_embargoed_1"]


def test_thresholds_use_train_rows_only() -> None:
    rows = []
    for day in range(40):
        rows.append(ohlc_row(day, 100 + day * 0.1))
    for day in range(370, 390):
        rows.append(
            {
                **ohlc_row(day, 1000 + day * 100),
                "timestamp": (date(2021, 1, 1) + timedelta(days=day - 370)).isoformat(),
            }
        )

    _labels, before = compute_label_rows(rows, split=split(), interval="1d")
    mutated = [dict(row) for row in rows]
    for row in mutated:
        if row["timestamp"].startswith("2021"):
            row["close"] *= 100
            row["high"] *= 100
            row["low"] *= 100
    _labels, after = compute_label_rows(mutated, split=split(), interval="1d")

    assert after["thresholds"]["direction_thr_1"] == before["thresholds"]["direction_thr_1"]


def shape_row(day: int, lambda_o: float, lambda_c: float) -> dict:
    timestamp = (date(2020, 1, 1) + timedelta(days=day)).isoformat()
    return {
        "timestamp": timestamp,
        "symbol": "A",
        "lambda_o": lambda_o,
        "lambda_c": lambda_c,
        "s1": lambda_o,
        "s2": lambda_c,
        "is_zero_range": False,
        "is_boundary": False,
    }


def test_coarse_class_thresholds_use_train_interior_rows_only() -> None:
    rows = [shape_row(day, 0.4, 0.4 + 0.01 * day) for day in range(30)]
    rows += [
        {
            **shape_row(day, 0.1, 0.9),
            "timestamp": (date(2021, 1, 1) + timedelta(days=day - 30)).isoformat(),
        }
        for day in range(30, 36)
    ]

    _classes, before = assign_coarse_classes(rows, split=split())
    mutated = [dict(row) for row in rows]
    for row in mutated:
        if row["timestamp"].startswith("2021"):
            row["lambda_c"] = 0.2
    _classes, after = assign_coarse_classes(mutated, split=split())

    assert after["body_quantile_thresholds"] == before["body_quantile_thresholds"]


def test_bigram_sequences_do_not_cross_symbol_or_minute_session() -> None:
    rows = [
        {"timestamp": "2020-01-01 09:00:00", "symbol": "A", "split": "train", "token": 1},
        {"timestamp": "2020-01-01 09:01:00", "symbol": "A", "split": "train", "token": 2},
        {"timestamp": "2020-01-02 09:00:00", "symbol": "A", "split": "train", "token": 3},
        {"timestamp": "2020-01-01 09:00:00", "symbol": "B", "split": "train", "token": 4},
        {"timestamp": "2020-01-01 09:01:00", "symbol": "B", "split": "train", "token": 5},
    ]

    sequences = grouped_token_sequences(rows, split="train", interval="1m")

    assert sorted(sequences) == [[1, 2], [3], [4, 5]]
    stats = information_gain(sequences, vocab=VOCAB_FULL)
    assert stats.bigram_count == 2


def test_entropy_extremes_and_surrogate_preserves_unigrams() -> None:
    assert entropy_from_counts(np.array([5, 5])) == pytest.approx(1.0)
    assert entropy_from_counts(np.array([10, 0])) == pytest.approx(0.0)

    deterministic = information_gain([[1, 2, 1, 2, 1, 2]], vocab=VOCAB_FULL)
    assert deterministic.conditional_entropy == pytest.approx(0.0)
    assert deterministic.information_gain > 0

    sequences = [[1, 2, 2, 3], [3, 3, 1]]
    shuffled = shuffle_token_sequences(sequences, rng=np.random.default_rng(7))
    assert sorted(token for seq in shuffled for token in seq) == sorted(
        token for seq in sequences for token in seq
    )


def test_mutual_information_decomposition_sums_to_total() -> None:
    sequences = [[1, 1, 2, 1, 2, 2], [2, 1, 1, 1]]

    decomposition = mutual_information_decomposition(sequences, vocab=VOCAB_FULL)

    assert (
        decomposition.diagonal_contribution + decomposition.off_diagonal_contribution
    ) == pytest.approx(decomposition.information_gain)
    assert decomposition.bigram_count == 8


def test_run_length_encode_edges() -> None:
    assert run_length_encode([]) == []
    assert run_length_encode([4]) == [4]
    assert run_length_encode([4, 4, 4]) == [4]
    assert run_length_encode([4, 4, 5, 5, 4]) == [4, 5, 4]


def test_markov1_surrogate_preserves_transition_distribution() -> None:
    sequences = [[0, 1] * 2500, [1, 0] * 2500]
    transitions, initials = markov1_transition_matrix(sequences, vocab=VOCAB_FULL)
    generated = generate_markov1_sequences(
        [5000, 5000],
        transitions=transitions,
        initials=initials,
        vocab=VOCAB_FULL,
        rng=np.random.default_rng(7),
    )

    observed = bigram_count_matrix(sequences, vocab=VOCAB_FULL)[:2, :2].astype(float)
    surrogate = bigram_count_matrix(generated, vocab=VOCAB_FULL)[:2, :2].astype(float)
    observed = observed / observed.sum(axis=1, keepdims=True)
    surrogate = surrogate / surrogate.sum(axis=1, keepdims=True)

    assert surrogate == pytest.approx(observed, abs=0.05)


def test_intraday_bucket_boundaries() -> None:
    assert assign_intraday_bucket("2024-01-02 08:59:00") is None
    assert assign_intraday_bucket("2024-01-02 09:00:00") == "open_30m"
    assert assign_intraday_bucket("2024-01-02 09:29:59") == "open_30m"
    assert assign_intraday_bucket("2024-01-02 09:30:00") == "middle"
    assert assign_intraday_bucket("2024-01-02 14:59:59") == "middle"
    assert assign_intraday_bucket("2024-01-02 15:00:00") == "close_30m"
    assert assign_intraday_bucket("2024-01-02 15:30:00") == "close_30m"
    assert assign_intraday_bucket("2024-01-02 15:31:00") is None


def test_jensen_shannon_distance_bounds() -> None:
    assert jensen_shannon_distance(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0
    assert 0.0 < jensen_shannon_distance(
        np.array([1.0, 0.0]), np.array([0.0, 1.0])
    ) <= 1.0


def test_williams_fractal_strict_tie_confirmation_and_minute_session() -> None:
    rows = [
        {"timestamp": f"2020-01-01 09:0{i}:00", "symbol": "A", "high": high, "low": low}
        for i, (high, low) in enumerate(
            [(1, 1), (2, 2), (5, 0), (3, 3), (2, 2), (9, 9), (1, 1)]
        )
    ]

    fractals = compute_williams_fractals(rows, interval="1m", n=2)
    pivot = fractals[("2020-01-01 09:02:00", "A")]

    assert pivot["fractal_high"] is True
    assert pivot["fractal_low"] is True
    assert pivot["fractal_confirmed_at"] == "2020-01-01 09:04:00"
    assert fractals[("2020-01-01 09:00:00", "A")]["fractal_confirmed_at"] is None

    tied = [dict(row) for row in rows]
    tied[3]["high"] = 5
    tied_fractals = compute_williams_fractals(tied, interval="1m", n=2)
    assert tied_fractals[("2020-01-01 09:02:00", "A")]["fractal_high"] is False

    boundary_rows = [
        {"timestamp": "2020-01-01 09:00:00", "symbol": "B", "high": 1, "low": 1},
        {"timestamp": "2020-01-01 09:01:00", "symbol": "B", "high": 2, "low": 2},
        {"timestamp": "2020-01-02 09:00:00", "symbol": "B", "high": 5, "low": 0},
        {"timestamp": "2020-01-02 09:01:00", "symbol": "B", "high": 2, "low": 2},
        {"timestamp": "2020-01-02 09:02:00", "symbol": "B", "high": 1, "low": 1},
    ]
    no_cross = compute_williams_fractals(boundary_rows, interval="1m", n=2)
    assert no_cross[("2020-01-02 09:00:00", "B")]["fractal_confirmed_at"] is None
