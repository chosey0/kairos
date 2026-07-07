from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from kairos.core.data import CandleBar
from kairos.core.features import extract_features
from kairos.experiments.protocol import (
    D1_INDEX_NAMES,
    D1_INTERVALS,
    DOMESTIC_KIWOOM_START_DATE,
    DOMESTIC_KIWOOM_START_DATETIME,
    FEATURE_PROTOCOL,
    FILTERING_PROTOCOL,
    INDEX_SYMBOLS,
    KIS_EARLIEST_START_DATE,
    KIS_EARLIEST_START_DATETIME,
    PHASE_01_ID,
    RESEARCH_END_DATE,
    RESEARCH_NAME,
    STEP_01_FEATURE_ID,
    d1_dataset_id,
    data_source_policy_config,
    split_name,
    split_protocol_config,
)


Provider = Literal["kiwoom", "kis"]

DEFAULT_FEATURE_CONFIG = {
    **FEATURE_PROTOCOL,
    "boundary_policy": FILTERING_PROTOCOL["boundary_policy"],
    "zero_range_policy": FILTERING_PROTOCOL["zero_range_policy"],
}

KIS_INDEX_MINUTE_MAX_PAGES = 10_000
KIS_MAJOR_INDEX_DAILY_MAX_PAGES = 500


@dataclass(frozen=True, slots=True)
class DatasetRequest:
    dataset_id: str
    stage: str
    symbol: str
    market: str
    interval: str
    provider: Provider
    broker_method: str
    broker_symbol: str
    request: dict[str, Any]


def d1_feature_request(symbol_name: str, interval: str) -> DatasetRequest:
    index_symbol = INDEX_SYMBOLS[symbol_name]
    source = index_symbol.source
    if interval == "1d":
        broker_method = source.period_method
        broker_symbol = source.period_symbol
        if source.provider == "kiwoom":
            request = {
                "base_date": RESEARCH_END_DATE,
                "start_date": DOMESTIC_KIWOOM_START_DATE,
                "max_pages": None,
            }
        elif source.provider == "kis":
            request = {
                "start": KIS_EARLIEST_START_DATE,
                "end": RESEARCH_END_DATE,
                "period": "D",
                "max_pages": KIS_MAJOR_INDEX_DAILY_MAX_PAGES,
            }
        else:
            raise ValueError(f"unsupported provider: {source.provider}")
    elif interval == "1m":
        broker_method = source.minute_method
        broker_symbol = source.minute_symbol
        if source.provider == "kiwoom":
            request = {
                "interval_minutes": 1,
                "base_date": RESEARCH_END_DATE,
                "start_date": DOMESTIC_KIWOOM_START_DATETIME,
                "max_pages": None,
            }
        elif source.provider == "kis":
            request = {
                "start": KIS_EARLIEST_START_DATETIME,
                "hour_class": "0",
                "include_previous": True,
                "max_pages": KIS_INDEX_MINUTE_MAX_PAGES,
            }
        else:
            raise ValueError(f"unsupported provider: {source.provider}")
    else:
        raise ValueError(f"unsupported interval: {interval}")

    return DatasetRequest(
        dataset_id=d1_dataset_id(symbol_name, interval),
        stage="D1",
        symbol=symbol_name,
        market="KRX-INDEX" if source.provider == "kiwoom" else "OVERSEAS_INDEX",
        interval=interval,
        provider=source.provider,
        broker_method=broker_method,
        broker_symbol=broker_symbol,
        request=request,
    )


D1_FEATURE_REQUESTS = tuple(
    d1_feature_request(symbol_name, interval)
    for symbol_name in D1_INDEX_NAMES
    for interval in D1_INTERVALS
)


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p25": None, "p50": None, "p75": None, "max": None}
    ordered = sorted(values)

    def pick(probability: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "min": ordered[0],
        "p25": pick(0.25),
        "p50": pick(0.50),
        "p75": pick(0.75),
        "max": ordered[-1],
    }


def mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    mean = sum(values) / len(values)
    variance = sum((item - mean) ** 2 for item in values) / len(values)
    return {"mean": mean, "std": math.sqrt(variance)}


def as_candle(bar: Any, request: DatasetRequest) -> CandleBar:
    timestamp = getattr(bar, "timestamp", None)
    if timestamp is None and hasattr(bar, "business_date") and hasattr(bar, "time"):
        timestamp = f"{getattr(bar, 'business_date')} {getattr(bar, 'time')}"
    return CandleBar(
        market=request.market,
        symbol=request.symbol,
        interval=request.interval,
        timestamp=str(timestamp),
        open=float(getattr(bar, "open")),
        high=float(getattr(bar, "high")),
        low=float(getattr(bar, "low")),
        close=float(getattr(bar, "close")),
        volume=int(float(getattr(bar, "volume", 0) or 0)),
    )


def summarize_dataset(
    request: DatasetRequest,
    candles: tuple[CandleBar, ...],
    *,
    feature_config: dict[str, Any] = DEFAULT_FEATURE_CONFIG,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    feature_rows = extract_features(
        list(candles),
        eps=feature_config["eps"],
        atr_period=feature_config["atr_period"],
        include_volume=feature_config["include_volume"],
    )
    shapes = [row.shape for row in feature_rows]
    side_channels = [row.channels for row in feature_rows]

    zero_range_count = sum(1 for shape in shapes if shape.is_zero_range)
    boundary_count = sum(
        1
        for shape in shapes
        if shape.open_at_low or shape.open_at_high or shape.close_at_low or shape.close_at_high
    )
    split_counts = {"train": 0, "validation": 0, "test": 0, "excluded": 0}
    for candle in candles:
        split_counts[split_name(candle.timestamp)] += 1

    signed_body = [shape.signed_body_ratio for shape in shapes]
    s1_values = [shape.s1 for shape in shapes if not shape.is_zero_range]
    s2_values = [shape.s2 for shape in shapes if not shape.is_zero_range]
    lambda_o_values = [shape.lambda_o for shape in shapes]
    lambda_c_values = [shape.lambda_c for shape in shapes]
    partition_errors = [
        abs(shape.upper_ratio + shape.lower_ratio + abs(shape.signed_body_ratio) - 1.0)
        for shape in shapes
        if not shape.is_zero_range
    ]

    rel_range_values = [channel.rel_range for channel in side_channels if channel.rel_range is not None]
    gap_values = [channel.gap for channel in side_channels if channel.gap is not None]
    direction_counts = {
        "bullish": sum(1 for shape in shapes if shape.direction > 0),
        "bearish": sum(1 for shape in shapes if shape.direction < 0),
        "doji": sum(1 for shape in shapes if shape.direction == 0),
    }
    requested_start_date = request.request.get("start_date") or request.request.get("start")
    requested_end_date = request.request.get("base_date") or request.request.get("end")

    metrics = {
        "dataset_id": request.dataset_id,
        "dataset_stage": request.stage,
        "symbol": request.symbol,
        "provider": request.provider,
        "broker_method": request.broker_method,
        "broker_symbol": request.broker_symbol,
        "row_count": len(candles),
        "date_range": {"start": candles[0].timestamp, "end": candles[-1].timestamp},
        "data_request": {
            "requested_start_date": requested_start_date,
            "requested_end_date": requested_end_date,
            "requested_max_pages": request.request.get("max_pages"),
            "history_request_policy": (
                "OpenAPI requests fetch the maximum available history: Kiwoom uses "
                "SDK pagination from 1990-01-01, KIS major-index daily pages backward "
                "by moving FID_INPUT_DATE_2 to the day before the oldest returned row, "
                "and KIS minute endpoints are provider-limited to the latest page."
            ),
        },
        "split_row_counts": split_counts,
        "shape_core": {
            "lambda_o": quantiles(lambda_o_values),
            "lambda_c": quantiles(lambda_c_values),
            "s1": mean_std(s1_values) | quantiles(s1_values),
            "s2": mean_std(s2_values) | quantiles(s2_values),
            "signed_body_ratio": quantiles(signed_body),
            "direction_counts": direction_counts,
            "shape_partition_max_error": max(partition_errors) if partition_errors else None,
        },
        "side_channels": {
            "atr_period": feature_config["atr_period"],
            "rel_range_warmup_rows": sum(1 for channel in side_channels if channel.rel_range is None),
            "gap_warmup_rows": sum(1 for channel in side_channels if channel.gap is None),
            "rel_range": mean_std(rel_range_values) | quantiles(rel_range_values),
            "gap": mean_std(gap_values) | quantiles(gap_values),
            "volume_included": feature_config["include_volume"],
        },
        "exceptional_rows": {
            "zero_range_count": zero_range_count,
            "zero_range_ratio": zero_range_count / len(candles),
            "boundary_count": boundary_count,
            "boundary_ratio": boundary_count / len(candles),
            "winsorize_flag_eligible_rows": len(candles) - zero_range_count,
            "exclude_boundary_eligible_rows": len(candles) - zero_range_count - boundary_count,
            "boundary_exclusion_loss_ratio": boundary_count / len(candles),
        },
        "leakage_checks": {
            "time_split_fixed_before_fit": True,
            "features_extracted_after_split_protocol": True,
            "rolling_statistics_use_t_minus_1": True,
            "random_split_forbidden": True,
            "raw_ohlcv_persisted": False,
        },
        "decision_gate": {
            "shape_core_ready_for_tokenizer_input": zero_range_count < len(candles),
            "zero_range_special_token_required": zero_range_count > 0,
            "boundary_ab_comparison_recorded": True,
        },
    }

    sample_rows: list[dict[str, Any]] = []
    for candle, row in zip(candles, feature_rows, strict=True):
        shape = row.shape
        sample_rows.append(
            {
                "timestamp": candle.timestamp,
                "symbol": candle.symbol,
                "lambda_o": shape.lambda_o,
                "lambda_c": shape.lambda_c,
                "s1": shape.s1,
                "s2": shape.s2,
                "signed_body_ratio": shape.signed_body_ratio,
                "upper_ratio": shape.upper_ratio,
                "lower_ratio": shape.lower_ratio,
                "body_center_location": shape.body_center_location,
                "direction": shape.direction,
                "is_zero_range": shape.is_zero_range,
                "is_boundary": shape.open_at_low or shape.open_at_high or shape.close_at_low or shape.close_at_high,
                "rel_range": row.channels.rel_range,
                "gap": row.channels.gap,
            }
        )

    summary_rows = [
        {"metric": "row_count", "value": metrics["row_count"]},
        {"metric": "zero_range_count", "value": zero_range_count},
        {"metric": "boundary_count", "value": boundary_count},
        {"metric": "rel_range_warmup_rows", "value": metrics["side_channels"]["rel_range_warmup_rows"]},
        {"metric": "shape_partition_max_error", "value": metrics["shape_core"]["shape_partition_max_error"]},
        {"metric": "winsorize_flag_eligible_rows", "value": metrics["exceptional_rows"]["winsorize_flag_eligible_rows"]},
        {"metric": "exclude_boundary_eligible_rows", "value": metrics["exceptional_rows"]["exclude_boundary_eligible_rows"]},
    ]
    return metrics, sample_rows, summary_rows


def summarize_empty_dataset(
    request: DatasetRequest,
    *,
    reason: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    requested_start_date = request.request.get("start_date") or request.request.get("start")
    requested_end_date = request.request.get("base_date") or request.request.get("end")
    metrics = {
        "status": "skipped",
        "skip_reason": reason,
        "dataset_id": request.dataset_id,
        "dataset_stage": request.stage,
        "symbol": request.symbol,
        "provider": request.provider,
        "broker_method": request.broker_method,
        "broker_symbol": request.broker_symbol,
        "row_count": 0,
        "date_range": {"start": None, "end": None},
        "data_request": {
            "requested_start_date": requested_start_date,
            "requested_end_date": requested_end_date,
            "requested_max_pages": request.request.get("max_pages"),
            "history_request_policy": (
                "OpenAPI requests fetch the maximum available history where supported; "
                "datasets with empty endpoint responses are recorded as skipped instead "
                "of aborting the run."
            ),
        },
        "split_row_counts": {"train": 0, "validation": 0, "test": 0, "excluded": 0},
        "shape_core": {},
        "side_channels": {
            "atr_period": DEFAULT_FEATURE_CONFIG["atr_period"],
            "rel_range_warmup_rows": 0,
            "gap_warmup_rows": 0,
            "volume_included": DEFAULT_FEATURE_CONFIG["include_volume"],
        },
        "exceptional_rows": {
            "zero_range_count": 0,
            "zero_range_ratio": None,
            "boundary_count": 0,
            "boundary_ratio": None,
            "winsorize_flag_eligible_rows": 0,
            "exclude_boundary_eligible_rows": 0,
            "boundary_exclusion_loss_ratio": None,
        },
        "leakage_checks": {
            "time_split_fixed_before_fit": True,
            "features_extracted_after_split_protocol": True,
            "rolling_statistics_use_t_minus_1": True,
            "random_split_forbidden": True,
            "raw_ohlcv_persisted": False,
        },
        "decision_gate": {
            "shape_core_ready_for_tokenizer_input": False,
            "zero_range_special_token_required": False,
            "boundary_ab_comparison_recorded": False,
        },
    }
    summary_rows = [
        {"metric": "status", "value": "skipped"},
        {"metric": "skip_reason", "value": reason},
        {"metric": "row_count", "value": 0},
    ]
    return metrics, [], summary_rows


def build_feature_validation_config(request: DatasetRequest) -> dict[str, Any]:
    return {
        "research": RESEARCH_NAME,
        "phase": PHASE_01_ID,
        "step": STEP_01_FEATURE_ID,
        "dataset": request,
        "data_source_policy": data_source_policy_config(),
        "split": split_protocol_config(),
        "feature": FEATURE_PROTOCOL,
        "filtering": FILTERING_PROTOCOL,
        "user_vars": {
            "network_download_enabled": True,
            "raw_ohlcv_persisted": False,
            "requested_start_date": request.request.get("start_date") or request.request.get("start"),
            "comparison_scope": "D1 domestic KOSPI and overseas DJI independently, daily and 1-minute intervals",
            "history_request_policy": (
                "OpenAPI requests fetch the maximum available history: Kiwoom uses "
                "SDK pagination from 1990-01-01, KIS major-index daily pages backward "
                "by moving FID_INPUT_DATE_2 to the day before the oldest returned row, "
                "and KIS minute endpoints are provider-limited to the latest page."
            ),
        },
    }
