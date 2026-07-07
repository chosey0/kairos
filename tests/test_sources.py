import asyncio
from dataclasses import dataclass
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brokers.kis.auth.cache import TokenRecord

from kairos.experiments.shape_tokenizer.feature_validation import D1_FEATURE_REQUESTS
from kairos.sources import brokers


@dataclass(frozen=True)
class RawTimestampBar:
    timestamp: str
    open: float = 1.0
    high: float = 2.0
    low: float = 0.5
    close: float = 1.5
    volume: int = 100


@dataclass(frozen=True)
class RawIndexMinuteBar:
    business_date: str = "2026-01-20"
    time: str = "09:30:00"
    open: float = 1.0
    high: float = 2.0
    low: float = 0.5
    close: float = 1.5
    volume: int = 100


def test_fetch_index_candles_uses_kiwoom_industry_minute(monkeypatch) -> None:
    request = {item.dataset_id: item for item in D1_FEATURE_REQUESTS}["d1_kospi_1m"]
    calls = []

    class FakeChart:
        async def industry_minute(self, *args, **kwargs):
            calls.append((args, kwargs))
            return [RawTimestampBar("2026-01-20 09:00:00")]

    class FakeClient:
        domestic = type("Domestic", (), {"chart": FakeChart()})()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class FakeKiwoomClient:
        @classmethod
        def from_env(cls):
            return FakeClient()

    monkeypatch.setattr(brokers, "KiwoomClient", FakeKiwoomClient)

    candles = asyncio.run(
        brokers.fetch_index_candles(request, env_path=Path("/tmp/nonexistent.env"))
    )

    assert calls == [
        (
            ("001",),
            {
                "interval_minutes": 1,
                "base_date": "2026-07-03",
                "start_date": "1990-01-01 000000",
                "max_pages": None,
            },
        )
    ]
    assert candles[0].timestamp == "2026-01-20 09:00:00"
    assert candles[0].interval == "1m"


def test_fetch_index_candles_normalizes_kiwoom_minute_date_only_start(monkeypatch) -> None:
    base_request = {item.dataset_id: item for item in D1_FEATURE_REQUESTS}["d1_kospi_1m"]
    request = replace(base_request, request={**base_request.request, "start_date": "1990-01-01"})
    calls = []

    class FakeChart:
        async def industry_minute(self, *args, **kwargs):
            calls.append((args, kwargs))
            return [RawTimestampBar("2026-01-20 09:00:00")]

    class FakeClient:
        domestic = type("Domestic", (), {"chart": FakeChart()})()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class FakeKiwoomClient:
        @classmethod
        def from_env(cls):
            return FakeClient()

    monkeypatch.setattr(brokers, "KiwoomClient", FakeKiwoomClient)

    asyncio.run(
        brokers.fetch_index_candles(request, env_path=Path("/tmp/nonexistent.env"))
    )

    assert calls[0][1]["start_date"] == "1990-01-01 000000"


def test_fetch_index_candles_uses_kis_index_minute(monkeypatch) -> None:
    request = {item.dataset_id: item for item in D1_FEATURE_REQUESTS}["d1_dji_1m"]
    calls = []

    class FakeChart:
        async def index_minute(self, *args, **kwargs):
            calls.append((args, kwargs))
            return [RawIndexMinuteBar()]

    class FakeClient:
        overseas = type("Overseas", (), {"chart": FakeChart()})()

        def __init__(self, *, credentials, token_cache):
            self.credentials = credentials
            self.token_cache = token_cache

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class FakeCredentials:
        @classmethod
        def from_env(cls, *, app_secret_var):
            return {"app_secret_var": app_secret_var}

    monkeypatch.setattr(brokers, "Credentials", FakeCredentials)
    monkeypatch.setattr(brokers, "KisClient", FakeClient)

    candles = asyncio.run(
        brokers.fetch_index_candles(request, env_path=Path("/tmp/nonexistent.env"))
    )

    assert calls == [
        (
            ("DJI",),
            {
                "start": "1900-01-01 00:00:00",
                "hour_class": "0",
                "include_previous": True,
                "max_pages": 10_000,
            },
        )
    ]
    assert candles[0].timestamp == "2026-01-20 09:30:00"
    assert candles[0].interval == "1m"


def test_fetch_index_candles_allows_empty_broker_response(monkeypatch) -> None:
    request = {item.dataset_id: item for item in D1_FEATURE_REQUESTS}["d1_nasdaq_daily"]

    class FakeChart:
        async def major_index(self, *_args, **_kwargs):
            return []

    class FakeClient:
        overseas = type("Overseas", (), {"chart": FakeChart()})()

        def __init__(self, *, credentials, token_cache):
            self.credentials = credentials
            self.token_cache = token_cache

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class FakeCredentials:
        @classmethod
        def from_env(cls, *, app_secret_var):
            return {"app_secret_var": app_secret_var}

    monkeypatch.setattr(brokers, "Credentials", FakeCredentials)
    monkeypatch.setattr(brokers, "KisClient", FakeClient)

    candles = asyncio.run(
        brokers.fetch_index_candles(request, env_path=Path("/tmp/nonexistent.env"))
    )

    assert candles == ()


def test_fetch_index_candles_paginates_kis_major_index_daily(monkeypatch) -> None:
    base_request = {item.dataset_id: item for item in D1_FEATURE_REQUESTS}["d1_nasdaq_daily"]
    request = replace(
        base_request,
        request={
            **base_request.request,
            "start": "2026-06-27",
            "end": "2026-07-03",
            "max_pages": 5,
        },
    )
    calls = []

    class FakeChart:
        async def major_index(self, *args, **kwargs):
            calls.append((args, kwargs))
            if kwargs["end"] == "2026-07-03":
                return [
                    RawTimestampBar("2026-07-01"),
                    RawTimestampBar("2026-07-02"),
                    RawTimestampBar("2026-07-03"),
                ]
            if kwargs["end"] == "2026-06-30":
                return [
                    RawTimestampBar("2026-06-29"),
                    RawTimestampBar("2026-06-30"),
                    RawTimestampBar("2026-07-01"),
                ]
            if kwargs["end"] == "2026-06-28":
                return [
                    RawTimestampBar("2026-06-27"),
                    RawTimestampBar("2026-06-28"),
                ]
            return []

    class FakeClient:
        overseas = type("Overseas", (), {"chart": FakeChart()})()

        def __init__(self, *, credentials, token_cache):
            self.credentials = credentials
            self.token_cache = token_cache

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class FakeCredentials:
        @classmethod
        def from_env(cls, *, app_secret_var):
            return {"app_secret_var": app_secret_var}

    monkeypatch.setattr(brokers, "Credentials", FakeCredentials)
    monkeypatch.setattr(brokers, "KisClient", FakeClient)

    candles = asyncio.run(
        brokers.fetch_index_candles(request, env_path=Path("/tmp/nonexistent.env"))
    )

    assert [call[1]["end"] for call in calls] == [
        "2026-07-03",
        "2026-06-30",
        "2026-06-28",
    ]
    assert [candle.timestamp for candle in candles] == [
        "2026-06-27",
        "2026-06-28",
        "2026-06-29",
        "2026-06-30",
        "2026-07-01",
        "2026-07-02",
        "2026-07-03",
    ]


def test_env_file_token_cache_persists_kis_token(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "KIS_APP_KEY=app-key\n"
        "KIS_APP_SECRET_KEY=secret\n"
        "KEEP_ME=yes\n",
        encoding="utf-8",
    )
    cache = brokers.EnvFileTokenCache(env_path)
    expires_at = datetime.now(UTC) + timedelta(hours=1)

    cache.set(
        "real:app-key",
        TokenRecord(
            access_token="access-token",
            token_type="Bearer",
            expires_at=expires_at,
        ),
    )

    assert cache.get("real:app-key") == TokenRecord(
        access_token="access-token",
        token_type="Bearer",
        expires_at=expires_at,
    )
    assert cache.get("real:other-key") is None
    text = env_path.read_text(encoding="utf-8")
    assert "KEEP_ME=yes" in text
    assert "KIS_ACCESS_TOKEN_CACHE_KEY=real:app-key" in text
    assert "KIS_ACCESS_TOKEN=access-token" in text
    assert "KIS_ACCESS_TOKEN_TYPE=Bearer" in text
    assert f"KIS_ACCESS_TOKEN_EXPIRES_AT={expires_at.isoformat()}" in text


def test_env_file_token_cache_deletes_only_matching_key(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    cache = brokers.EnvFileTokenCache(env_path)
    expires_at = datetime.now(UTC) + timedelta(hours=1)

    cache.set(
        "real:app-key",
        TokenRecord(
            access_token="access-token",
            token_type="Bearer",
            expires_at=expires_at,
        ),
    )
    cache.delete("real:other-key")
    assert cache.get("real:app-key") is not None

    cache.delete("real:app-key")

    text = env_path.read_text(encoding="utf-8")
    assert "KIS_ACCESS_TOKEN" not in text
