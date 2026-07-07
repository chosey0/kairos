from __future__ import annotations

from collections import defaultdict
from datetime import date
from math import log, sqrt
from typing import Any

import numpy as np

from kairos.experiments.protocol import SplitProtocol, split_name


LABEL_HORIZONS = (1, 5, 20)
RV_HORIZONS = (5, 20)
DRAWDOWN_HORIZON = 20
VOL_EXPANSION_THRESHOLD = 1.5


def parse_trade_date(timestamp: str) -> date:
    return date.fromisoformat(str(timestamp)[:10])


def sorted_symbol_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["symbol"])].append(row)
    return {
        symbol: sorted(symbol_rows, key=lambda item: str(item["timestamp"]))
        for symbol, symbol_rows in grouped.items()
    }


def compute_label_rows(
    ohlc_rows: list[dict[str, Any]],
    *,
    split: SplitProtocol,
    interval: str,
    horizons: tuple[int, ...] = LABEL_HORIZONS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute forward labels without crossing symbol or split boundaries."""
    output: list[dict[str, Any]] = []
    threshold_values: dict[int, float | None] = {}
    drawdown_threshold = None
    regime_terciles: tuple[float, float] | None = None

    grouped = sorted_symbol_rows(ohlc_rows)
    intermediate: dict[tuple[str, str], dict[str, Any]] = {}
    for symbol_rows in grouped.values():
        symbol_labels = _compute_symbol_label_rows(
            symbol_rows,
            split=split,
            interval=interval,
            horizons=horizons,
        )
        for item in symbol_labels:
            intermediate[(item["timestamp"], item["symbol"])] = item

    for horizon in horizons:
        values = [
            abs(float(row[f"fwd_log_return_{horizon}"]))
            for row in intermediate.values()
            if row["split"] == "train"
            and not row[f"label_embargoed_{horizon}"]
            and row[f"fwd_log_return_{horizon}"] is not None
        ]
        threshold_values[horizon] = float(np.median(values)) if values else None

    drawdowns = [
        float(row[f"max_drawdown_{DRAWDOWN_HORIZON}"])
        for row in intermediate.values()
        if row["split"] == "train"
        and not row[f"label_embargoed_{DRAWDOWN_HORIZON}"]
        and row[f"max_drawdown_{DRAWDOWN_HORIZON}"] is not None
    ]
    if drawdowns:
        drawdown_threshold = float(np.quantile(drawdowns, 0.9))

    if interval == "1d":
        train_vols = [
            float(row["trailing_rv_20"])
            for row in intermediate.values()
            if row["split"] == "train" and row["trailing_rv_20"] is not None
        ]
        if train_vols:
            low, high = np.quantile(train_vols, (1 / 3, 2 / 3))
            regime_terciles = (float(low), float(high))

    null_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for key in sorted(intermediate):
        row = intermediate[key]
        enriched = dict(row)
        for horizon in horizons:
            threshold = threshold_values[horizon]
            fwd = row[f"fwd_log_return_{horizon}"]
            if fwd is None or threshold is None:
                enriched[f"direction_thr_{horizon}"] = None
            elif abs(float(fwd)) <= threshold:
                enriched[f"direction_thr_{horizon}"] = 0
            else:
                enriched[f"direction_thr_{horizon}"] = 1 if float(fwd) > 0 else -1

        for horizon in RV_HORIZONS:
            fwd_rv = row[f"fwd_rv_{horizon}"]
            trailing_rv = row[f"trailing_rv_{horizon}"]
            if fwd_rv is None or trailing_rv is None or trailing_rv <= 0:
                enriched[f"vol_expansion_{horizon}"] = None
            else:
                enriched[f"vol_expansion_{horizon}"] = int(
                    fwd_rv / trailing_rv > VOL_EXPANSION_THRESHOLD
                )

        drawdown = row[f"max_drawdown_{DRAWDOWN_HORIZON}"]
        if drawdown is None or drawdown_threshold is None:
            enriched[f"drawdown_event_{DRAWDOWN_HORIZON}"] = None
        else:
            enriched[f"drawdown_event_{DRAWDOWN_HORIZON}"] = int(
                float(drawdown) > drawdown_threshold
            )

        if interval == "1d":
            enriched["regime"] = _regime_label(row, regime_terciles)
        else:
            enriched["regime"] = None

        split_value = str(enriched["split"])
        for label_name, value in enriched.items():
            if _is_label_column(label_name) and value is None:
                null_counts[split_value][label_name] += 1
        output.append(enriched)

    metrics = {
        "thresholds": {
            f"direction_thr_{horizon}": threshold_values[horizon]
            for horizon in horizons
        },
        "drawdown_threshold_20": drawdown_threshold,
        "regime_vol_terciles": regime_terciles,
        "label_null_counts_by_split": {
            split_name_: dict(counts) for split_name_, counts in null_counts.items()
        },
        "regime_defined": interval == "1d",
    }
    return output, metrics


def _compute_symbol_label_rows(
    rows: list[dict[str, Any]],
    *,
    split: SplitProtocol,
    interval: str,
    horizons: tuple[int, ...],
) -> list[dict[str, Any]]:
    closes = np.array([float(row["close"]) for row in rows], dtype=float)
    highs = np.array([float(row["high"]) for row in rows], dtype=float)
    lows = np.array([float(row["low"]) for row in rows], dtype=float)
    parkinson = _parkinson_terms(highs, lows)
    split_values = [split_name(str(row["timestamp"]), split) for row in rows]
    output: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        item: dict[str, Any] = {
            "timestamp": str(row["timestamp"]),
            "symbol": str(row["symbol"]),
            "split": split_values[index],
            "close": float(row["close"]),
        }
        for horizon in horizons:
            embargoed = _label_embargoed(
                rows,
                index,
                horizon=horizon,
                split=split,
                split_values=split_values,
            )
            item[f"label_embargoed_{horizon}"] = embargoed
            if embargoed:
                item[f"fwd_log_return_{horizon}"] = None
                item[f"direction_{horizon}"] = None
            else:
                fwd = log(closes[index + horizon] / closes[index])
                item[f"fwd_log_return_{horizon}"] = fwd
                item[f"direction_{horizon}"] = 1 if fwd > 0 else (-1 if fwd < 0 else 0)

        for horizon in RV_HORIZONS:
            item[f"fwd_rv_{horizon}"] = (
                None
                if item[f"label_embargoed_{horizon}"]
                else _window_rv(parkinson[index + 1 : index + horizon + 1])
            )
            item[f"trailing_rv_{horizon}"] = (
                None if index + 1 < horizon else _window_rv(parkinson[index - horizon + 1 : index + 1])
            )

        if item[f"label_embargoed_{DRAWDOWN_HORIZON}"]:
            item[f"max_drawdown_{DRAWDOWN_HORIZON}"] = None
        else:
            item[f"max_drawdown_{DRAWDOWN_HORIZON}"] = max_drawdown(
                closes[index : index + DRAWDOWN_HORIZON + 1]
            )

        item["ma_200"] = None if index + 1 < 200 else float(np.mean(closes[index - 199 : index + 1]))
        output.append(item)
    return output


def _parkinson_terms(highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
    ratios = np.log(highs / lows)
    return (ratios**2) / (4.0 * np.log(2.0))


def _window_rv(values: np.ndarray) -> float | None:
    if len(values) == 0 or np.any(~np.isfinite(values)):
        return None
    return float(sqrt(float(np.mean(values))))


def max_drawdown(closes: np.ndarray) -> float:
    peak = float(closes[0])
    worst = 0.0
    for close in closes:
        value = float(close)
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst


def _label_embargoed(
    rows: list[dict[str, Any]],
    index: int,
    *,
    horizon: int,
    split: SplitProtocol,
    split_values: list[str],
) -> bool:
    if index + horizon >= len(rows):
        return True
    if split_values[index] == "excluded":
        return True
    if split_values[index + horizon] != split_values[index]:
        return True
    trade_date = parse_trade_date(str(rows[index]["timestamp"]))
    anchors = (
        parse_trade_date(split.train_end),
        parse_trade_date(split.validation_start),
        parse_trade_date(split.validation_end),
        parse_trade_date(split.test_start),
    )
    return any(abs((trade_date - anchor).days) <= split.embargo_days for anchor in anchors)


def _regime_label(
    row: dict[str, Any],
    terciles: tuple[float, float] | None,
) -> str | None:
    ma_200 = row.get("ma_200")
    trailing_rv = row.get("trailing_rv_20")
    if ma_200 is None or trailing_rv is None or terciles is None:
        return None
    trend = "above_ma" if float(row.get("close", ma_200)) >= float(ma_200) else "below_ma"
    low, high = terciles
    vol = "low_vol" if trailing_rv <= low else ("high_vol" if trailing_rv >= high else "mid_vol")
    return f"{trend}_{vol}"


def _is_label_column(name: str) -> bool:
    prefixes = (
        "fwd_log_return_",
        "direction_",
        "direction_thr_",
        "fwd_rv_",
        "trailing_rv_",
        "vol_expansion_",
        "max_drawdown_",
        "drawdown_event_",
        "regime",
    )
    return name.startswith(prefixes)
