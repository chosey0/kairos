from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from brokers.kis import Credentials, KisClient
from brokers.kis._internal.pacing import (
    ContinuationPacer,
    call_with_continuation_pacing,
)
from brokers.kis.auth.cache import TokenRecord
from brokers.kiwoom import KiwoomClient

from kairos.core.data import CandleBar
from kairos.experiments.shape_tokenizer.feature_validation import (
    DatasetRequest,
    as_candle,
)

DATE_ONLY_LENGTH = 10
KIS_TOKEN_CACHE_KEY_ENV = "KIS_ACCESS_TOKEN_CACHE_KEY"
KIS_ACCESS_TOKEN_ENV = "KIS_ACCESS_TOKEN"
KIS_TOKEN_TYPE_ENV = "KIS_ACCESS_TOKEN_TYPE"
KIS_TOKEN_EXPIRES_AT_ENV = "KIS_ACCESS_TOKEN_EXPIRES_AT"
DEFAULT_KIS_MAJOR_INDEX_DAILY_MAX_PAGES = 500
KIS_MAJOR_INDEX_DAILY_MIN_INTERVAL_SECONDS = 0.5
KIS_MAJOR_INDEX_DAILY_RATE_LIMIT_RETRIES = 5


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class EnvFileTokenCache:
    """Persist KIS access tokens in the project `.env` file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self, key: str) -> TokenRecord | None:
        values = _read_env_values(self.path)
        if values.get(KIS_TOKEN_CACHE_KEY_ENV) != key:
            return None
        token = values.get(KIS_ACCESS_TOKEN_ENV)
        token_type = values.get(KIS_TOKEN_TYPE_ENV)
        expires_at = values.get(KIS_TOKEN_EXPIRES_AT_ENV)
        if not token or not token_type or not expires_at:
            return None
        try:
            parsed_expires_at = datetime.fromisoformat(expires_at)
        except ValueError:
            return None
        if parsed_expires_at.tzinfo is None:
            parsed_expires_at = parsed_expires_at.replace(tzinfo=UTC)
        return TokenRecord(
            access_token=token,
            token_type=token_type,
            expires_at=parsed_expires_at,
        )

    def set(self, key: str, record: TokenRecord) -> None:
        save_kis_token_to_env(self.path, cache_key=key, record=record)

    def delete(self, key: str) -> None:
        values = _read_env_values(self.path)
        if values.get(KIS_TOKEN_CACHE_KEY_ENV) != key:
            return
        _update_env_file(
            self.path,
            {
                KIS_TOKEN_CACHE_KEY_ENV: None,
                KIS_ACCESS_TOKEN_ENV: None,
                KIS_TOKEN_TYPE_ENV: None,
                KIS_TOKEN_EXPIRES_AT_ENV: None,
            },
        )


def save_kis_token_to_env(path: Path, *, cache_key: str, record: TokenRecord) -> None:
    """Save a KIS REST access token so later runs reuse it before expiry."""
    _update_env_file(
        path,
        {
            KIS_TOKEN_CACHE_KEY_ENV: cache_key,
            KIS_ACCESS_TOKEN_ENV: record.access_token,
            KIS_TOKEN_TYPE_ENV: record.token_type,
            KIS_TOKEN_EXPIRES_AT_ENV: record.expires_at.isoformat(),
        },
    )


async def fetch_index_candles(
    request: DatasetRequest, *, env_path: Path
) -> tuple[CandleBar, ...]:
    load_env(env_path)
    if request.provider == "kiwoom":
        async with KiwoomClient.from_env() as client:
            if request.broker_method == "client.domestic.chart.industry_daily":
                bars = await client.domestic.chart.industry_daily(
                    request.broker_symbol,
                    base_date=request.request["base_date"],
                    start_date=request.request["start_date"],
                    max_pages=request.request["max_pages"],
                )
            elif request.broker_method == "client.domestic.chart.industry_minute":
                bars = await client.domestic.chart.industry_minute(
                    request.broker_symbol,
                    interval_minutes=request.request["interval_minutes"],
                    base_date=request.request["base_date"],
                    start_date=_kiwoom_minute_start_date(request.request["start_date"]),
                    max_pages=request.request["max_pages"],
                )
            else:
                raise ValueError(
                    f"unsupported Kiwoom broker method: {request.broker_method}"
                )
    elif request.provider == "kis":
        credentials = Credentials.from_env(app_secret_var="KIS_APP_SECRET_KEY")
        async with KisClient(
            credentials=credentials,
            token_cache=EnvFileTokenCache(env_path),
        ) as client:
            if request.broker_method == "client.overseas.chart.major_index":
                bars = await fetch_kis_major_index_daily_pages(client, request)
            elif request.broker_method == "client.overseas.chart.index_minute":
                bars = await client.overseas.chart.index_minute(
                    request.broker_symbol,
                    start=request.request["start"],
                    hour_class=request.request["hour_class"],
                    include_previous=request.request["include_previous"],
                    max_pages=request.request["max_pages"],
                )
            else:
                raise ValueError(
                    f"unsupported KIS broker method: {request.broker_method}"
                )
    else:
        raise ValueError(f"unsupported provider: {request.provider}")

    candles = tuple(
        sorted(
            (as_candle(bar, request) for bar in bars), key=lambda item: item.timestamp
        )
    )
    return candles


async def fetch_kis_major_index_daily_pages(
    client: KisClient,
    request: DatasetRequest,
) -> list[Any]:
    """Fetch KIS major-index daily bars by paging backward with the end date."""
    start_date = date.fromisoformat(request.request["start"])
    page_end = date.fromisoformat(request.request["end"])
    max_pages = int(
        request.request.get("max_pages") or DEFAULT_KIS_MAJOR_INDEX_DAILY_MAX_PAGES
    )
    bars_by_timestamp: dict[str, Any] = {}
    pacer = ContinuationPacer(
        min_interval_seconds=KIS_MAJOR_INDEX_DAILY_MIN_INTERVAL_SECONDS
    )

    for _ in range(max_pages):
        page = await call_with_continuation_pacing(
            pacer,
            lambda: client.overseas.chart.major_index(
                request.broker_symbol,
                start=request.request["start"],
                end=page_end.isoformat(),
                period=request.request["period"],
            ),
            max_rate_limit_retries=KIS_MAJOR_INDEX_DAILY_RATE_LIMIT_RETRIES,
        )
        if not page:
            break

        page_dates = [_bar_date(bar) for bar in page]
        for bar, bar_date in zip(page, page_dates, strict=True):
            if start_date <= bar_date <= page_end:
                bars_by_timestamp[bar.timestamp] = bar

        oldest = min(page_dates)
        if oldest <= start_date:
            break

        next_page_end = oldest - timedelta(days=1)
        if next_page_end >= page_end:
            break
        page_end = next_page_end

    return sorted(bars_by_timestamp.values(), key=lambda bar: bar.timestamp)


def _bar_date(bar: Any) -> date:
    return date.fromisoformat(str(bar.timestamp)[:DATE_ONLY_LENGTH])


def _kiwoom_minute_start_date(value: str) -> str:
    text = value.strip()
    if len(text) == DATE_ONLY_LENGTH:
        return f"{text} 000000"
    return text


def _read_env_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _update_env_file(path: Path, updates: dict[str, str | None]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    seen: set[str] = set()
    next_lines: list[str] = []

    for line in lines:
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            next_lines.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key not in updates:
            next_lines.append(line)
            continue
        seen.add(key)
        value = updates[key]
        if value is not None:
            next_lines.append(f"{key}={value}")

    for key, value in updates.items():
        if key not in seen and value is not None:
            next_lines.append(f"{key}={value}")

    path.write_text("\n".join(next_lines).rstrip() + "\n", encoding="utf-8")
