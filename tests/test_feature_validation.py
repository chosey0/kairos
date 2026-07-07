import pytest

from kairos.core.data import CandleBar
from kairos.experiments.protocol import D1_INDEX_NAMES, D1_INTERVALS, d1_dataset_id
from kairos.experiments.shape_tokenizer.feature_validation import (
    D1_FEATURE_REQUESTS,
    KIS_INDEX_MINUTE_MAX_PAGES,
    KIS_MAJOR_INDEX_DAILY_MAX_PAGES,
    as_candle,
    quantiles,
    summarize_empty_dataset,
    summarize_dataset,
)


def make_candle(day: int, *, open_: float, high: float, low: float, close: float) -> CandleBar:
    return CandleBar(
        market="TEST",
        symbol="TEST",
        interval="1d",
        timestamp=f"2016-12-{day:02d}",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100,
    )


def test_quantiles_interpolate() -> None:
    stats = quantiles([1.0, 2.0, 3.0, 4.0])
    assert stats["p25"] == pytest.approx(1.75)
    assert stats["p50"] == pytest.approx(2.5)
    assert stats["p75"] == pytest.approx(3.25)


def test_summarize_dataset_records_shape_quality() -> None:
    request = D1_FEATURE_REQUESTS[0]
    candles = (
        make_candle(1, open_=1.0, high=2.0, low=0.0, close=1.5),
        make_candle(2, open_=0.0, high=2.0, low=0.0, close=2.0),
        make_candle(3, open_=1.0, high=1.0, low=1.0, close=1.0),
    )

    metrics, sample_rows, summary_rows = summarize_dataset(request, candles)

    assert metrics["row_count"] == 3
    assert metrics["exceptional_rows"]["zero_range_count"] == 1
    assert metrics["exceptional_rows"]["boundary_count"] == 1
    assert metrics["split_row_counts"]["train"] == 3
    assert metrics["split_row_counts"]["excluded"] == 0
    assert len(sample_rows) == 3
    assert {row["metric"] for row in summary_rows} >= {"row_count", "zero_range_count"}


def test_d1_requests_include_daily_and_minute_openapi_max_history() -> None:
    by_id = {request.dataset_id: request for request in D1_FEATURE_REQUESTS}
    expected_ids = {
        d1_dataset_id(symbol_name, interval)
        for symbol_name in D1_INDEX_NAMES
        for interval in D1_INTERVALS
    }

    assert set(by_id) == expected_ids
    assert len(D1_FEATURE_REQUESTS) == len(expected_ids)
    assert by_id["d1_kosdaq_daily"].broker_symbol == "101"
    assert by_id["d1_spx_daily"].broker_symbol == "SPX"
    assert by_id["d1_spx_1m"].broker_symbol == "SPX"
    assert by_id["d1_nasdaq_daily"].broker_symbol == "COMP"
    assert by_id["d1_nasdaq_1m"].broker_symbol == "COMP"
    assert by_id["d1_kospi_1m"].broker_method == "client.domestic.chart.industry_minute"
    assert by_id["d1_kospi_1m"].request["max_pages"] is None
    assert by_id["d1_kospi_1m"].request["start_date"] == "1990-01-01 000000"
    assert by_id["d1_nasdaq_daily"].request["max_pages"] == KIS_MAJOR_INDEX_DAILY_MAX_PAGES
    assert by_id["d1_dji_1m"].broker_method == "client.overseas.chart.index_minute"
    assert by_id["d1_dji_1m"].broker_symbol == "DJI"
    assert by_id["d1_dji_1m"].request["max_pages"] == KIS_INDEX_MINUTE_MAX_PAGES
    assert by_id["d1_dji_1m"].request["start"] == "1900-01-01 00:00:00"


def test_as_candle_normalizes_overseas_index_minute_timestamp() -> None:
    class MinuteBar:
        business_date = "2026-01-20"
        time = "09:30:00"
        open = 1
        high = 2
        low = 0
        close = 1.5
        volume = 100

    request = {item.dataset_id: item for item in D1_FEATURE_REQUESTS}["d1_dji_1m"]

    candle = as_candle(MinuteBar(), request)

    assert candle.timestamp == "2026-01-20 09:30:00"
    assert candle.interval == "1m"


def test_summarize_empty_dataset_records_skipped_status() -> None:
    request = {item.dataset_id: item for item in D1_FEATURE_REQUESTS}["d1_nasdaq_daily"]

    metrics, sample_rows, summary_rows = summarize_empty_dataset(request, reason="endpoint returned no rows")

    assert metrics["status"] == "skipped"
    assert metrics["skip_reason"] == "endpoint returned no rows"
    assert metrics["row_count"] == 0
    assert metrics["split_row_counts"]["excluded"] == 0
    assert metrics["date_range"] == {"start": None, "end": None}
    assert sample_rows == []
    assert {"metric": "skip_reason", "value": "endpoint returned no rows"} in summary_rows
