import csv
from pathlib import Path

import numpy as np
import pytest

from kairos.experiments.protocol import SPLIT_PROTOCOL_MINUTE
from kairos.experiments.shape_tokenizer.baselines import (
    MERGED_FEATURE_INPUTS,
    dataset_interval,
    evaluate_baselines,
    finite_shape_rows,
    handcrafted_lambda_bins,
    read_merged_shape_rows,
    reconstruction_mse,
    shape_rows_for_boundary_policy,
    split_masks,
    token_stats,
)


def shape_row(
    day: int,
    s1: float,
    s2: float,
    lambda_o: float,
    lambda_c: float,
    *,
    zero: bool = False,
    boundary: bool = False,
) -> dict:
    return {
        "timestamp": f"2016-01-{day:02d}",
        "symbol": "TEST",
        "s1": s1,
        "s2": s2,
        "lambda_o": lambda_o,
        "lambda_c": lambda_c,
        "is_zero_range": zero,
        "is_boundary": boundary,
    }


def test_token_stats_records_effective_vocab_and_dead_tokens() -> None:
    stats = token_stats(np.array([0, 0, 1]), codebook_size=4)
    assert stats["utilized_token_count"] == 2
    assert stats["dead_token_count"] == 2
    assert stats["effective_vocab_size"] == pytest.approx(2 ** stats["token_entropy"])


def test_handcrafted_lambda_bins_assigns_axis_tokens() -> None:
    rows = [shape_row(1, 0.0, 0.0, 0.10, 0.90)]
    tokens, recon = handcrafted_lambda_bins(rows, bins_per_axis=4)
    assert tokens.tolist() == [3]
    assert recon.shape == (1, 2)


def test_finite_shape_rows_filters_zero_range() -> None:
    rows = [shape_row(1, 0.0, 0.0, 0.5, 0.5), shape_row(2, 0.0, 0.0, 0.5, 0.5, zero=True)]
    assert len(finite_shape_rows(rows)) == 1


def test_shape_rows_for_boundary_policy_can_exclude_boundary_rows() -> None:
    rows = [
        shape_row(1, 0.0, 0.0, 0.5, 0.5),
        shape_row(2, 6.9, -6.9, 1.0, 0.0, boundary=True),
        shape_row(3, 0.0, 0.0, 0.5, 0.5, zero=True),
    ]

    included = shape_rows_for_boundary_policy(rows, boundary_policy="include_boundary")
    excluded = shape_rows_for_boundary_policy(rows, boundary_policy="exclude_boundary")

    assert len(included) == 2
    assert len(excluded) == 1
    assert not excluded[0]["is_boundary"]


def test_split_masks_and_mse() -> None:
    rows = [
        shape_row(1, 0.0, 0.0, 0.5, 0.5),
        shape_row(2, 1.0, 1.0, 0.7, 0.7),
        {**shape_row(3, 2.0, 2.0, 0.9, 0.9), "timestamp": "2021-01-01"},
    ]
    masks = split_masks(rows)
    assert masks["train"].sum() == 2
    assert masks["test"].sum() == 1
    assert reconstruction_mse(np.array([[1.0, 2.0]]), np.array([[1.0, 1.0]])) == pytest.approx(1.0)


def test_dataset_interval_from_dataset_id() -> None:
    assert dataset_interval("d1_kospi_daily") == "1d"
    assert dataset_interval("d2_kr-kospi-kosdaq_daily") == "1d"
    assert dataset_interval("d2_kr-kospi-kosdaq_1m") == "1m"


def test_merged_feature_inputs_cover_kr_daily_and_minute() -> None:
    dataset_ids = [merged.dataset_id for merged in MERGED_FEATURE_INPUTS]
    assert dataset_ids == ["d2_kr-kospi-kosdaq_daily", "d2_kr-kospi-kosdaq_1m"]
    for merged in MERGED_FEATURE_INPUTS:
        assert merged.provider == "kiwoom"
        assert {component.symbol for component in merged.components} == {"KOSPI", "KOSDAQ"}


def test_split_masks_supports_minute_split() -> None:
    rows = [
        {**shape_row(1, 0.0, 0.0, 0.5, 0.5), "timestamp": "2025-07-01 09:00:00"},
        {**shape_row(2, 0.0, 0.0, 0.5, 0.5), "timestamp": "2026-02-02 09:00:00"},
        {**shape_row(3, 0.0, 0.0, 0.5, 0.5), "timestamp": "2026-05-02 09:00:00"},
    ]

    daily_masks = split_masks(rows)
    minute_masks = split_masks(rows, split=SPLIT_PROTOCOL_MINUTE)

    assert daily_masks["train"].sum() == 0
    assert daily_masks["test"].sum() == 3
    assert minute_masks["train"].sum() == 1
    assert minute_masks["validation"].sum() == 1
    assert minute_masks["test"].sum() == 1


def write_shape_sample(run_dir: Path, rows: list[dict]) -> None:
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True)
    fieldnames = [
        "timestamp",
        "symbol",
        "s1",
        "s2",
        "lambda_o",
        "lambda_c",
        "is_zero_range",
        "is_boundary",
    ]
    with (tables_dir / "shape_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def test_read_merged_shape_rows_tags_source_and_sorts_by_timestamp(tmp_path: Path) -> None:
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    write_shape_sample(run_a, [shape_row(2, 0.1, 0.2, 0.4, 0.6), shape_row(4, 0.3, 0.4, 0.5, 0.7)])
    write_shape_sample(run_b, [{**shape_row(3, 0.5, 0.6, 0.6, 0.8), "symbol": "OTHER"}])

    merged_rows = read_merged_shape_rows({"d1_a_daily": run_a, "d1_b_daily": run_b})

    assert [row["timestamp"] for row in merged_rows] == [
        "2016-01-02",
        "2016-01-03",
        "2016-01-04",
    ]
    assert [row["source_dataset_id"] for row in merged_rows] == [
        "d1_a_daily",
        "d1_b_daily",
        "d1_a_daily",
    ]
    assert merged_rows[1]["symbol"] == "OTHER"
    assert merged_rows[0]["s1"] == pytest.approx(0.1)


def test_evaluate_baselines_runs_small_kmeans_and_gmm() -> None:
    rows = [
        shape_row(day, float(day % 3), float((day + 1) % 3), 0.2 + (day % 3) * 0.2, 0.3)
        for day in range(1, 25)
    ]
    metrics, summary = evaluate_baselines(rows, seed=7, codebook_sizes=(2,), bins_per_axis=2)
    assert len(metrics) == 3
    assert len(summary) == 3
    assert {item["model"] for item in summary} == {"kmeans", "gmm", "handcrafted_lambda_bins"}
