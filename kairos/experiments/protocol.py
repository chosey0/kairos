from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any, Literal


DEFAULT_PROTOCOL_CONFIG_PATH = Path(__file__).with_name("protocols") / "candlestick_shape_quantization.json"

SourceProvider = Literal["kiwoom", "kis"]
SplitName = Literal["train", "validation", "test", "excluded"]


class ProtocolConfigError(ValueError):
    """Raised when the experiment protocol configuration is inconsistent."""


@dataclass(frozen=True, slots=True)
class BrokerSource:
    provider: SourceProvider
    period_method: str
    minute_method: str
    period_symbol: str
    minute_symbol: str
    master_symbol: str | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class IndexSymbol:
    symbol: str
    name: str
    country: str
    source: BrokerSource


@dataclass(frozen=True, slots=True)
class IndexDataset:
    dataset_id: str
    stage: str
    interval: str
    market_group: str
    symbols: tuple[IndexSymbol, ...]
    purpose: str

    @property
    def provider_set(self) -> set[SourceProvider]:
        return {item.source.provider for item in self.symbols}

    @property
    def broker_mapping(self) -> dict[str, dict[str, str]]:
        return {
            item.symbol: {
                "provider": item.source.provider,
                "period_method": item.source.period_method,
                "minute_method": item.source.minute_method,
                "period_symbol": item.source.period_symbol,
                "minute_symbol": item.source.minute_symbol,
                "master_symbol": item.source.master_symbol,
            }
            for item in self.symbols
        }


@dataclass(frozen=True, slots=True)
class SplitProtocol:
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str | None
    embargo_days: int
    label_horizons: tuple[int, ...]
    rolling_statistics_rule: str


def load_protocol_settings(path: Path | str = DEFAULT_PROTOCOL_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        settings = json.load(handle)
    validate_protocol_settings(settings)
    return settings


def _parse_date(value: str, *, field_name: str) -> date:
    try:
        return date.fromisoformat(value[:10])
    except ValueError as exc:
        raise ProtocolConfigError(f"{field_name} must start with an ISO date: {value!r}") from exc


def validate_split_protocol(split: SplitProtocol) -> None:
    train_start = _parse_date(split.train_start, field_name="split.train_start")
    train_end = _parse_date(split.train_end, field_name="split.train_end")
    validation_start = _parse_date(split.validation_start, field_name="split.validation_start")
    validation_end = _parse_date(split.validation_end, field_name="split.validation_end")
    test_start = _parse_date(split.test_start, field_name="split.test_start")
    test_end = _parse_date(split.test_end, field_name="split.test_end") if split.test_end else None

    if not train_start <= train_end < validation_start <= validation_end < test_start:
        raise ProtocolConfigError("split dates must be ordered train <= validation <= test without overlap")
    if test_end is not None and test_start > test_end:
        raise ProtocolConfigError("split.test_end must be greater than or equal to split.test_start")
    if not split.label_horizons:
        raise ProtocolConfigError("split.label_horizons must not be empty")
    if split.embargo_days < max(split.label_horizons):
        raise ProtocolConfigError("split.embargo_days must cover the maximum label horizon")


def validate_protocol_settings(settings: dict[str, Any]) -> None:
    required_sections = {
        "research",
        "dates",
        "data_source_policy",
        "broker_methods",
        "split",
        "symbols",
        "datasets",
        "feature_protocol",
        "filtering_protocol",
        "user_vars",
        "manifest",
    }
    missing_sections = required_sections - set(settings)
    if missing_sections:
        raise ProtocolConfigError(f"missing protocol sections: {sorted(missing_sections)}")

    split = split_protocol_from_settings(settings)
    validate_split_protocol(split)
    if "split_minute" in settings:
        validate_split_protocol(split_protocol_from_settings(settings, key="split_minute"))

    symbols_config = settings["symbols"]
    if not symbols_config:
        raise ProtocolConfigError("symbols must not be empty")

    provider_methods = settings["broker_methods"]
    for symbol_name, symbol_config in symbols_config.items():
        source = symbol_config.get("source", {})
        provider = source.get("provider")
        if provider not in provider_methods:
            raise ProtocolConfigError(f"{symbol_name} uses unsupported provider: {provider!r}")
        if not source.get("period_symbol"):
            raise ProtocolConfigError(f"{symbol_name} must define source.period_symbol")

    d1 = settings["datasets"].get("d1", {})
    for symbol_name in d1.get("symbols", []):
        if symbol_name not in symbols_config:
            raise ProtocolConfigError(f"D1 references unknown symbol: {symbol_name}")
    for interval in d1.get("intervals", []):
        interval_slug(interval)

    dataset_ids: set[str] = set()
    for group in settings["datasets"].get("groups", []):
        dataset_id = group.get("dataset_id")
        if not dataset_id:
            raise ProtocolConfigError("dataset groups must define dataset_id")
        if dataset_id in dataset_ids:
            raise ProtocolConfigError(f"duplicate dataset_id: {dataset_id}")
        dataset_ids.add(dataset_id)
        for symbol_name in group.get("symbols", []):
            if symbol_name not in symbols_config:
                raise ProtocolConfigError(f"{dataset_id} references unknown symbol: {symbol_name}")


def split_protocol_from_settings(settings: dict[str, Any], *, key: str = "split") -> SplitProtocol:
    split = settings[key]
    return SplitProtocol(
        train_start=split["train_start"],
        train_end=split["train_end"],
        validation_start=split["validation_start"],
        validation_end=split["validation_end"],
        test_start=split["test_start"],
        test_end=split.get("test_end"),
        embargo_days=int(split["embargo_days"]),
        label_horizons=tuple(int(item) for item in split["label_horizons"]),
        rolling_statistics_rule=split["rolling_statistics_rule"],
    )


def interval_slug(interval: str) -> str:
    if interval == "1d":
        return "daily"
    if interval == "1m":
        return "1m"
    raise ValueError(f"unsupported interval: {interval}")


def _broker_source_from_settings(symbol_config: dict[str, Any], settings: dict[str, Any]) -> BrokerSource:
    source = symbol_config["source"]
    provider = source["provider"]
    methods = settings["broker_methods"][provider]
    period_symbol = source["period_symbol"]
    return BrokerSource(
        provider=provider,
        period_method=source.get("period_method", methods["period_method"]),
        minute_method=source.get("minute_method", methods["minute_method"]),
        period_symbol=period_symbol,
        minute_symbol=source.get("minute_symbol", period_symbol.lstrip(".")),
        master_symbol=source.get("master_symbol"),
        note=source.get("note", ""),
    )


def build_index_symbols(settings: dict[str, Any]) -> dict[str, IndexSymbol]:
    return {
        symbol_name: IndexSymbol(
            symbol_name,
            symbol_config["name"],
            symbol_config["country"],
            _broker_source_from_settings(symbol_config, settings),
        )
        for symbol_name, symbol_config in settings["symbols"].items()
    }


def symbols(*names: str, index_symbols: dict[str, IndexSymbol] | None = None) -> tuple[IndexSymbol, ...]:
    symbol_map = index_symbols or INDEX_SYMBOLS
    return tuple(symbol_map[name] for name in names)


def d1_dataset_id(symbol_name: str, interval: str) -> str:
    return f"d1_{symbol_name.lower()}_{interval_slug(interval)}"


def single_index_dataset(
    symbol_name: str,
    interval: str,
    *,
    index_symbols: dict[str, IndexSymbol] | None = None,
) -> IndexDataset:
    symbol_map = index_symbols or INDEX_SYMBOLS
    index_symbol = symbol_map[symbol_name]
    market_group = "kr-single" if index_symbol.country == "KR" else "us-single"
    return IndexDataset(
        d1_dataset_id(symbol_name, interval),
        "D1",
        interval,
        market_group,
        (index_symbol,),
        f"Single-index D1 dataset for {symbol_name} at {interval}.",
    )


def build_dataset_registry(
    settings: dict[str, Any],
    index_symbols: dict[str, IndexSymbol] | None = None,
) -> dict[str, IndexDataset]:
    symbol_map = index_symbols or build_index_symbols(settings)
    d1 = settings["datasets"]["d1"]
    registry = {
        dataset.dataset_id: dataset
        for symbol_name in d1["symbols"]
        for interval in d1["intervals"]
        for dataset in (single_index_dataset(symbol_name, interval, index_symbols=symbol_map),)
    }
    for group in settings["datasets"].get("groups", []):
        dataset = IndexDataset(
            group["dataset_id"],
            group["stage"],
            group["interval"],
            group["market_group"],
            symbols(*group["symbols"], index_symbols=symbol_map),
            group["purpose"],
        )
        if dataset.dataset_id in registry:
            raise ProtocolConfigError(f"duplicate dataset_id: {dataset.dataset_id}")
        registry[dataset.dataset_id] = dataset
    return registry


def split_name(timestamp: str, split: SplitProtocol | None = None) -> SplitName:
    active_split = split or SPLIT_PROTOCOL
    trade_date = _parse_date(timestamp, field_name="timestamp")
    train_start = _parse_date(active_split.train_start, field_name="split.train_start")
    train_end = _parse_date(active_split.train_end, field_name="split.train_end")
    validation_start = _parse_date(active_split.validation_start, field_name="split.validation_start")
    validation_end = _parse_date(active_split.validation_end, field_name="split.validation_end")
    test_start = _parse_date(active_split.test_start, field_name="split.test_start")
    test_end = _parse_date(active_split.test_end, field_name="split.test_end") if active_split.test_end else None

    if train_start <= trade_date <= train_end:
        return "train"
    if validation_start <= trade_date <= validation_end:
        return "validation"
    if trade_date >= test_start and (test_end is None or trade_date <= test_end):
        return "test"
    return "excluded"


def split_section_for_interval(interval: str) -> str:
    if interval == "1d":
        return "split"
    if interval == "1m":
        return "split_minute"
    raise ValueError(f"unsupported interval: {interval}")


def split_protocol_for_interval(interval: str) -> SplitProtocol:
    return SPLIT_PROTOCOL if split_section_for_interval(interval) == "split" else SPLIT_PROTOCOL_MINUTE


def split_protocol_config_for_interval(interval: str) -> dict[str, Any]:
    section = split_section_for_interval(interval)
    return split_protocol_config(split_protocol_for_interval(interval), section=section)


def data_source_policy_config(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    active_settings = settings or DEFAULT_PROTOCOL_SETTINGS
    policy = deepcopy(active_settings["data_source_policy"])
    policy["domestic_kiwoom_start_date"] = active_settings["dates"]["domestic_kiwoom_start_date"]
    policy["overseas_kis_start_date"] = active_settings["dates"]["kis_earliest_start_date"]
    return policy


def split_protocol_config(
    split: SplitProtocol | None = None,
    settings: dict[str, Any] | None = None,
    *,
    section: str = "split",
) -> dict[str, Any]:
    active_split = split or split_protocol_from_settings(settings or DEFAULT_PROTOCOL_SETTINGS, key=section)
    payload = asdict(active_split)
    payload["random_split_forbidden"] = (settings or DEFAULT_PROTOCOL_SETTINGS)[section].get("random_split_forbidden", True)
    return payload


def build_protocol_config(dataset: IndexDataset) -> dict[str, Any]:
    return {
        "research": RESEARCH_NAME,
        "phase": PHASE_00_ID,
        "step": STEP_00_ID,
        "dataset": dataset,
        "data_source_policy": data_source_policy_config(),
        "split": SPLIT_PROTOCOL,
        "feature": deepcopy(FEATURE_PROTOCOL),
        "filtering": deepcopy(FILTERING_PROTOCOL),
        "user_vars": deepcopy(USER_VARS),
    }


def run_directory(
    runs_root: Path,
    dataset: IndexDataset,
    cfg_hash: str,
    *,
    seed: int,
    started_at: str,
    phase_id: str = "",
    step_id: str = "",
) -> Path:
    phase = phase_id or PHASE_00_ID
    step = step_id or STEP_00_ID
    return runs_root / phase / step / dataset.dataset_id / f"cfg-{cfg_hash}" / f"run-{started_at}_seed-{seed}"


def figure_directory(
    figures_root: Path,
    dataset: IndexDataset,
    cfg_hash: str,
    *,
    phase_id: str = "",
    step_id: str = "",
) -> Path:
    phase = phase_id or PHASE_00_ID
    step = step_id or STEP_00_ID
    return figures_root / phase / step / dataset.dataset_id / f"cfg-{cfg_hash}" / "selected"


def build_manifest(
    dataset: IndexDataset,
    config: dict[str, Any],
    cfg_hash: str,
    *,
    seed: int | None = None,
    source_notebook: str | None = None,
) -> dict[str, Any]:
    return {
        "research": RESEARCH_NAME,
        "phase": PHASE_00_ID,
        "step": STEP_00_ID,
        "dataset_id": dataset.dataset_id,
        "dataset_stage": dataset.stage,
        "cfg_hash": cfg_hash,
        "seed": seed,
        "source_notebook": source_notebook or SOURCE_NOTEBOOK,
        "symbols": [asdict(item) for item in dataset.symbols],
        "split": split_protocol_config(),
        "required_outputs": list(REQUIRED_OUTPUTS),
        "download_policy": (
            "disabled in this protocol notebook until explicitly enabled; "
            f"domestic indexes use Kiwoom from {DOMESTIC_KIWOOM_START_DATE} onward and overseas indexes use KIS"
        ),
        "hash_inputs": config,
    }


def empty_data_quality_metrics(dataset: IndexDataset) -> dict[str, Any]:
    return {
        "dataset_id": dataset.dataset_id,
        "dataset_stage": dataset.stage,
        "row_counts_by_symbol": {item.symbol: None for item in dataset.symbols},
        "date_ranges_by_symbol": {item.symbol: {"start": None, "end": None} for item in dataset.symbols},
        "missing_ohlc_rows_by_symbol": {item.symbol: None for item in dataset.symbols},
        "zero_range_rows_by_symbol": {item.symbol: None for item in dataset.symbols},
        "boundary_rows_by_symbol": {item.symbol: None for item in dataset.symbols},
        "split_row_counts_by_symbol": {
            item.symbol: {"train": None, "validation": None, "test": None, "excluded": None} for item in dataset.symbols
        },
        "leakage_checks": {
            "time_split_fixed_before_fit": True,
            "rolling_statistics_use_t_minus_1": True,
            "random_split_forbidden": True,
        },
    }


def _normalized_symbol_candidates(value: Any) -> set[str]:
    if value is None:
        return set()
    text = str(value).strip()
    if not text:
        return set()
    candidates = {text, text.lstrip(".")}
    for separator in ("#", ":"):
        if separator in text:
            suffix = text.rsplit(separator, 1)[-1]
            candidates.add(suffix)
            candidates.add(suffix.lstrip("."))
    return {candidate.upper() for candidate in candidates if candidate}


def _iter_overseas_info_rows(overseas_index_info: Any) -> list[Any]:
    if hasattr(overseas_index_info, "to_dict"):
        return list(overseas_index_info.to_dict("records"))
    if isinstance(overseas_index_info, dict):
        return [overseas_index_info]
    return list(overseas_index_info)


def _field_value(row: Any, field_name: str) -> Any:
    if isinstance(row, dict):
        return row.get(field_name)
    return getattr(row, field_name, None)


def validate_kis_overseas_index_symbols(
    index_symbols: dict[str, IndexSymbol] | None = None,
    overseas_index_info: Any | None = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    symbol_map = index_symbols or INDEX_SYMBOLS
    if overseas_index_info is None:
        from brokers.kis import download_overseas_index_info

        overseas_index_info = download_overseas_index_info()

    available_symbols: set[str] = set()
    for row in _iter_overseas_info_rows(overseas_index_info):
        for field_name in ("symbol", "period_symbol", "minute_symbol"):
            available_symbols.update(_normalized_symbol_candidates(_field_value(row, field_name)))

    checked: list[dict[str, Any]] = []
    missing: dict[str, list[str]] = {}
    for symbol_name, index_symbol in symbol_map.items():
        if index_symbol.source.provider != "kis":
            continue
        required = (
            {"master_symbol": index_symbol.source.master_symbol}
            if index_symbol.source.master_symbol
            else {
                "period_symbol": index_symbol.source.period_symbol,
                "minute_symbol": index_symbol.source.minute_symbol,
            }
        )
        missing_fields = [
            field_name
            for field_name, expected in required.items()
            if not (_normalized_symbol_candidates(expected) & available_symbols)
        ]
        checked.append({"symbol": symbol_name, **required, "missing_fields": missing_fields})
        if missing_fields:
            missing[symbol_name] = missing_fields

    report = {
        "valid": not missing,
        "checked_count": len(checked),
        "available_symbol_count": len(available_symbols),
        "checked": checked,
        "missing": missing,
    }
    if strict and missing:
        raise ProtocolConfigError(f"KIS overseas index symbols missing from master data: {missing}")
    return report


DEFAULT_PROTOCOL_SETTINGS = load_protocol_settings()

RESEARCH_NAME = DEFAULT_PROTOCOL_SETTINGS["research"]["name"]
PHASE_00_ID = DEFAULT_PROTOCOL_SETTINGS["research"]["phases"]["data_protocol"]
PHASE_01_ID = DEFAULT_PROTOCOL_SETTINGS["research"]["phases"]["shape_tokenizer"]
STEP_00_ID = DEFAULT_PROTOCOL_SETTINGS["research"]["steps"]["index_universe_and_split"]
STEP_01_FEATURE_ID = DEFAULT_PROTOCOL_SETTINGS["research"]["steps"]["shape_feature_validation"]
STEP_02_BASELINE_ID = DEFAULT_PROTOCOL_SETTINGS["research"]["steps"]["tokenizer_baselines"]
SOURCE_NOTEBOOK = DEFAULT_PROTOCOL_SETTINGS["research"]["source_notebook"]

DOMESTIC_KIWOOM_START_DATE = DEFAULT_PROTOCOL_SETTINGS["dates"]["domestic_kiwoom_start_date"]
DOMESTIC_KIWOOM_START_DATETIME = DEFAULT_PROTOCOL_SETTINGS["dates"]["domestic_kiwoom_start_datetime"]
KIS_EARLIEST_START_DATE = DEFAULT_PROTOCOL_SETTINGS["dates"]["kis_earliest_start_date"]
KIS_EARLIEST_START_DATETIME = DEFAULT_PROTOCOL_SETTINGS["dates"]["kis_earliest_start_datetime"]
RESEARCH_END_DATE = DEFAULT_PROTOCOL_SETTINGS["dates"]["research_end_date"]

SPLIT_PROTOCOL = split_protocol_from_settings(DEFAULT_PROTOCOL_SETTINGS)
SPLIT_PROTOCOL_MINUTE = split_protocol_from_settings(DEFAULT_PROTOCOL_SETTINGS, key="split_minute")
FEATURE_PROTOCOL = deepcopy(DEFAULT_PROTOCOL_SETTINGS["feature_protocol"])
FILTERING_PROTOCOL = deepcopy(DEFAULT_PROTOCOL_SETTINGS["filtering_protocol"])
USER_VARS = deepcopy(DEFAULT_PROTOCOL_SETTINGS["user_vars"])
REQUIRED_OUTPUTS = tuple(DEFAULT_PROTOCOL_SETTINGS["manifest"]["required_outputs"])

INDEX_SYMBOLS = build_index_symbols(DEFAULT_PROTOCOL_SETTINGS)
D1_INDEX_NAMES = tuple(DEFAULT_PROTOCOL_SETTINGS["datasets"]["d1"]["symbols"])
D1_INTERVALS = tuple(DEFAULT_PROTOCOL_SETTINGS["datasets"]["d1"]["intervals"])
DATASET_REGISTRY = build_dataset_registry(DEFAULT_PROTOCOL_SETTINGS, INDEX_SYMBOLS)
