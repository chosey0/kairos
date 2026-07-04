from math import log

import pytest

from kairos.features import (
    DEFAULT_EPS,
    expit,
    extract_features,
    extract_shape,
    logit,
)
from kairos.data import CandleBar


def make_candle(
    day: int,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int = 100,
) -> CandleBar:
    return CandleBar(
        market="TEST",
        symbol="TEST",
        interval="1d",
        timestamp=f"2024-01-{day:02d}",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def flat_candle(day: int, *, volume: int = 100) -> CandleBar:
    # TR stays exactly 1.0: hl range 1.0, |high - prev_close| = |low - prev_close| = 0.5
    return make_candle(day, open_=99.5, high=100.0, low=99.0, close=99.5, volume=volume)


class TestShapeCore:
    def test_lambda_and_logit_values(self):
        shape = extract_shape(make_candle(1, open_=1.0, high=2.0, low=0.0, close=1.5))
        assert shape.lambda_o == pytest.approx(0.5)
        assert shape.lambda_c == pytest.approx(0.75)
        assert shape.s1 == pytest.approx(0.0)
        assert shape.s2 == pytest.approx(log(3.0))
        assert not shape.is_zero_range

    def test_derived_features_match_definitions(self):
        shape = extract_shape(make_candle(1, open_=1.5, high=2.0, low=0.0, close=0.5))
        assert shape.signed_body_ratio == pytest.approx(-0.5)
        assert shape.upper_ratio == pytest.approx(0.25)
        assert shape.lower_ratio == pytest.approx(0.25)
        assert shape.body_center_location == pytest.approx(0.5)
        assert shape.direction == -1.0
        body = abs(shape.signed_body_ratio)
        assert shape.upper_ratio + shape.lower_ratio + body == pytest.approx(1.0)

    def test_boundary_candle_is_flagged_and_winsorized(self):
        shape = extract_shape(make_candle(1, open_=0.0, high=2.0, low=0.0, close=2.0))
        assert shape.open_at_low and shape.close_at_high
        assert not shape.open_at_high and not shape.close_at_low
        assert shape.s1 == pytest.approx(logit(DEFAULT_EPS))
        assert shape.s2 == pytest.approx(logit(1.0 - DEFAULT_EPS))

    def test_zero_range_candle_is_marked(self):
        shape = extract_shape(make_candle(1, open_=1.0, high=1.0, low=1.0, close=1.0))
        assert shape.is_zero_range
        assert shape.as_tuple() == (0.0, 0.0)

    def test_logit_expit_roundtrip(self):
        for probability in (0.001, 0.25, 0.5, 0.9, 0.999):
            assert expit(logit(probability)) == pytest.approx(probability)


class TestSideChannels:
    def test_warmup_rows_have_none_channels(self):
        candles = [flat_candle(day) for day in range(1, 17)]
        features = extract_features(candles, atr_period=14)
        for row in features[:14]:
            assert row.channels.rel_range is None
            assert row.channels.gap is None
        assert features[14].channels.rel_range is not None
        assert features[14].channels.gap is not None

    def test_atr_excludes_current_bar_range(self):
        candles = [flat_candle(day) for day in range(1, 16)]
        # Bar 16 has range 5.0; ATR(t-1) must remain 1.0 from the flat history.
        candles.append(make_candle(16, open_=99.5, high=102.0, low=97.0, close=99.5))
        features = extract_features(candles, atr_period=14)
        spike = features[-1].channels
        assert spike.rel_range == pytest.approx(log(5.0))
        assert spike.gap == pytest.approx(0.0)

    def test_gap_uses_previous_close_and_trailing_atr(self):
        candles = [flat_candle(day) for day in range(1, 16)]
        candles.append(make_candle(16, open_=101.5, high=102.0, low=101.0, close=101.5))
        features = extract_features(candles, atr_period=14)
        assert features[-1].channels.gap == pytest.approx((101.5 - 99.5) / 1.0)

    def test_vol_spike_uses_trailing_median_only(self):
        candles = [flat_candle(day) for day in range(1, 21)]
        candles.append(flat_candle(21, volume=300))
        features = extract_features(candles, volume_window=20, include_volume=True)
        assert features[19].channels.vol_spike is None
        assert features[-1].channels.vol_spike == pytest.approx(log(3.0))

    def test_volume_excluded_by_default(self):
        candles = [flat_candle(day) for day in range(1, 26)]
        features = extract_features(candles)
        assert all(row.channels.vol_spike is None for row in features)

    def test_unsorted_candles_raise(self):
        candles = [flat_candle(2), flat_candle(1)]
        with pytest.raises(ValueError, match="sorted"):
            extract_features(candles)


class TestTrainGuard:
    def test_train_rejects_zero_range_shapes(self):
        from pathlib import Path

        from kairos.train import TrainConfig, train

        shapes = [extract_shape(make_candle(1, open_=1.0, high=1.0, low=1.0, close=1.0))]
        with pytest.raises(ValueError, match="zero-range"):
            train(shapes, config=TrainConfig(output_dir=Path("/tmp/unused")))
