"""Candle feature extraction following docs/candlestick-shape-quantization/08-research-roadmap.md.

Shape core (tokenizer input):
    lambda_o = (open - low) / range
    lambda_c = (close - low) / range
    s1 = logit(clip(lambda_o, eps, 1 - eps))
    s2 = logit(clip(lambda_c, eps, 1 - eps))

Side channels (kept separate from the shape token):
    rel_range = ln(range_t / ATR(t-1))
    gap       = (open_t - close_{t-1}) / ATR(t-1)
    vol_spike = ln(volume_t / median(volume over trailing window))

All trailing statistics use information up to t-1 only.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from statistics import median
from typing import Iterable, Sequence

from .data import CandleBar

DEFAULT_EPS = 1e-3
DEFAULT_ATR_PERIOD = 14
DEFAULT_VOLUME_WINDOW = 20


def logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("logit input must be in (0, 1)")
    return log(probability / (1.0 - probability))


def expit(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    exponent = exp(value)
    return exponent / (1.0 + exponent)


@dataclass(frozen=True, slots=True)
class ShapeFeatures:
    """Scale-invariant shape of a single candle.

    ``s1``/``s2`` are the tokenizer input. ``lambda_o``/``lambda_c`` are the raw
    relative positions used for reporting and prototype restoration. Boundary
    flags implement the winsorize + flag policy for marubozu-like candles.
    """

    lambda_o: float
    lambda_c: float
    s1: float
    s2: float
    open_at_low: bool
    open_at_high: bool
    close_at_low: bool
    close_at_high: bool
    is_zero_range: bool

    def as_tuple(self) -> tuple[float, float]:
        return (self.s1, self.s2)

    @property
    def signed_body_ratio(self) -> float:
        return self.lambda_c - self.lambda_o

    @property
    def upper_ratio(self) -> float:
        return 1.0 - max(self.lambda_o, self.lambda_c)

    @property
    def lower_ratio(self) -> float:
        return min(self.lambda_o, self.lambda_c)

    @property
    def body_center_location(self) -> float:
        return (self.lambda_o + self.lambda_c) / 2.0

    @property
    def direction(self) -> float:
        if self.lambda_c > self.lambda_o:
            return 1.0
        if self.lambda_c < self.lambda_o:
            return -1.0
        return 0.0


@dataclass(frozen=True, slots=True)
class SideChannels:
    """Continuous channels aligned with a candle; ``None`` while trailing history is insufficient."""

    rel_range: float | None
    gap: float | None
    vol_spike: float | None


@dataclass(frozen=True, slots=True)
class CandleFeatures:
    shape: ShapeFeatures
    channels: SideChannels


def extract_shape(candle: CandleBar, *, eps: float = DEFAULT_EPS) -> ShapeFeatures:
    """Extract the shape core of one candle. Needs no history.

    Zero-range candles are marked with ``is_zero_range`` and must be excluded
    from tokenizer training (assign a special token downstream).
    Invalid OHLC rows (open/close outside [low, high]) are expected to be
    removed by the Phase 0 cleaning rules; here lambda is clamped defensively.
    """
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")

    total_range = candle.high - candle.low
    if total_range <= 0.0:
        return ShapeFeatures(
            lambda_o=0.5,
            lambda_c=0.5,
            s1=0.0,
            s2=0.0,
            open_at_low=False,
            open_at_high=False,
            close_at_low=False,
            close_at_high=False,
            is_zero_range=True,
        )

    lambda_o = _clamp((candle.open - candle.low) / total_range, 0.0, 1.0)
    lambda_c = _clamp((candle.close - candle.low) / total_range, 0.0, 1.0)

    return ShapeFeatures(
        lambda_o=lambda_o,
        lambda_c=lambda_c,
        s1=logit(_clamp(lambda_o, eps, 1.0 - eps)),
        s2=logit(_clamp(lambda_c, eps, 1.0 - eps)),
        open_at_low=lambda_o <= eps,
        open_at_high=lambda_o >= 1.0 - eps,
        close_at_low=lambda_c <= eps,
        close_at_high=lambda_c >= 1.0 - eps,
        is_zero_range=False,
    )


def extract_shape_batch(
    candles: Iterable[CandleBar], *, eps: float = DEFAULT_EPS
) -> tuple[ShapeFeatures, ...]:
    return tuple(extract_shape(candle, eps=eps) for candle in candles)


def extract_features(
    candles: Sequence[CandleBar],
    *,
    eps: float = DEFAULT_EPS,
    atr_period: int = DEFAULT_ATR_PERIOD,
    volume_window: int = DEFAULT_VOLUME_WINDOW,
    include_volume: bool = False,
) -> tuple[CandleFeatures, ...]:
    """Extract shape core plus side channels for a time-ordered candle series.

    ``candles`` must be sorted by timestamp ascending and belong to a single
    symbol/interval. Side channels are ``None`` until enough trailing history
    exists, so warmup rows stay visible instead of silently leaking.
    """
    if atr_period <= 0:
        raise ValueError("atr_period must be positive")
    if volume_window <= 0:
        raise ValueError("volume_window must be positive")
    _require_sorted(candles)

    true_ranges = _true_ranges(candles)
    features: list[CandleFeatures] = []
    for index, candle in enumerate(candles):
        shape = extract_shape(candle, eps=eps)
        atr = _trailing_atr(true_ranges, index, atr_period)

        rel_range: float | None = None
        gap: float | None = None
        if atr is not None and atr > 0.0:
            total_range = candle.high - candle.low
            if total_range > 0.0:
                rel_range = log(total_range / atr)
            gap = (candle.open - candles[index - 1].close) / atr

        vol_spike: float | None = None
        if include_volume:
            vol_spike = _trailing_vol_spike(candles, index, volume_window)

        features.append(
            CandleFeatures(
                shape=shape,
                channels=SideChannels(rel_range=rel_range, gap=gap, vol_spike=vol_spike),
            )
        )
    return tuple(features)


def _true_ranges(candles: Sequence[CandleBar]) -> tuple[float, ...]:
    ranges: list[float] = []
    for index, candle in enumerate(candles):
        if index == 0:
            ranges.append(max(candle.high - candle.low, 0.0))
            continue
        previous_close = candles[index - 1].close
        ranges.append(
            max(
                candle.high - candle.low,
                abs(candle.high - previous_close),
                abs(candle.low - previous_close),
            )
        )
    return tuple(ranges)


def _trailing_atr(true_ranges: Sequence[float], index: int, period: int) -> float | None:
    # ATR at t averages TR[t-period .. t-1] so bar t never sees its own range.
    if index < period:
        return None
    window = true_ranges[index - period : index]
    return sum(window) / period


def _trailing_vol_spike(candles: Sequence[CandleBar], index: int, window: int) -> float | None:
    if index < window:
        return None
    trailing = [float(candles[position].volume) for position in range(index - window, index)]
    baseline = median(trailing)
    volume = float(candles[index].volume)
    if baseline <= 0.0 or volume <= 0.0:
        return None
    return log(volume / baseline)


def _require_sorted(candles: Sequence[CandleBar]) -> None:
    for index in range(1, len(candles)):
        if candles[index].timestamp < candles[index - 1].timestamp:
            raise ValueError("candles must be sorted by timestamp ascending")


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
