from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class CandleBar:
    """Broker-agnostic OHLCV candle used across the research pipeline."""

    market: str
    symbol: str
    interval: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True, slots=True)
class CandleSplit:
    train: tuple[CandleBar, ...]
    val: tuple[CandleBar, ...]
    test: tuple[CandleBar, ...]


def filter_by_min_volume(
    candles: Iterable[CandleBar],
    *,
    min_volume: int,
) -> tuple[CandleBar, ...]:
    """Keep candles whose volume is greater than or equal to ``min_volume``.

    Phase 1 shape experiments use this to remove illiquid / placeholder candles.
    For example, ``min_volume=2`` excludes candles with volume ``0`` or ``1``.
    """
    if min_volume < 0:
        raise ValueError("min_volume must be non-negative")
    return tuple(candle for candle in candles if candle.volume >= min_volume)


def split_by_date(
    candles: Iterable[CandleBar],
    *,
    train_end: str,
    val_end: str,
) -> CandleSplit:
    """Split candles by timestamp using inclusive train/val boundaries."""
    train: list[CandleBar] = []
    val: list[CandleBar] = []
    test: list[CandleBar] = []

    for candle in sorted(candles, key=lambda item: item.timestamp):
        if candle.timestamp <= train_end:
            train.append(candle)
        elif candle.timestamp <= val_end:
            val.append(candle)
        else:
            test.append(candle)

    return CandleSplit(train=tuple(train), val=tuple(val), test=tuple(test))


def split_by_ratio(
    candles: Iterable[CandleBar],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> CandleSplit:
    """Split candles by chronological ratio without shuffling.

    This is useful for minute datasets where absolute date boundaries are not known
    before loading the data. The remaining tail becomes the test split.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    ordered = tuple(sorted(candles, key=lambda item: item.timestamp))
    train_end_index = int(len(ordered) * train_ratio)
    val_end_index = train_end_index + int(len(ordered) * val_ratio)

    return CandleSplit(
        train=ordered[:train_end_index],
        val=ordered[train_end_index:val_end_index],
        test=ordered[val_end_index:],
    )
