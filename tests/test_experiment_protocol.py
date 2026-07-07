from copy import deepcopy

import pytest

from kairos.experiments.artifacts import config_hash, find_project_root, to_jsonable
from kairos.experiments.protocol import (
    D1_INDEX_NAMES,
    D1_INTERVALS,
    DATASET_REGISTRY,
    DEFAULT_PROTOCOL_CONFIG_PATH,
    DEFAULT_PROTOCOL_SETTINGS,
    DOMESTIC_KIWOOM_START_DATE,
    FEATURE_PROTOCOL,
    FILTERING_PROTOCOL,
    INDEX_SYMBOLS,
    ProtocolConfigError,
    SPLIT_PROTOCOL_MINUTE,
    SplitProtocol,
    build_manifest,
    build_protocol_config,
    d1_dataset_id,
    data_source_policy_config,
    empty_data_quality_metrics,
    load_protocol_settings,
    split_name,
    split_protocol_config,
    split_protocol_config_for_interval,
    split_protocol_for_interval,
    split_section_for_interval,
    validate_kis_overseas_index_symbols,
    validate_protocol_settings,
)
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NestedConfig:
    name: str
    path: Path


def test_config_hash_ignores_runtime_fields() -> None:
    base = {"dataset": "d1", "seed": 7, "started_at": "a"}
    changed_runtime = {"dataset": "d1", "seed": 37, "started_at": "b"}
    changed_semantic = {"dataset": "d2", "seed": 7, "started_at": "a"}

    assert config_hash(base) == config_hash(changed_runtime)
    assert config_hash(base) != config_hash(changed_semantic)


def test_to_jsonable_handles_dataclass_and_path() -> None:
    assert to_jsonable(NestedConfig("x", Path("a/b"))) == {"name": "x", "path": "a/b"}


def test_project_root_detection() -> None:
    root = find_project_root(Path(__file__))
    assert (root / "pyproject.toml").exists()
    assert (root / "kairos").exists()


def test_protocol_settings_load_from_json() -> None:
    settings = load_protocol_settings()

    assert DEFAULT_PROTOCOL_CONFIG_PATH.exists()
    assert settings["research"]["name"] == "candlestick-shape-quantization"
    assert settings["feature_protocol"] == FEATURE_PROTOCOL
    assert settings["filtering_protocol"] == FILTERING_PROTOCOL
    assert settings["datasets"]["d1"]["symbols"] == list(D1_INDEX_NAMES)


def test_protocol_dataset_sources_and_registry_contracts() -> None:
    dataset_ids = list(DATASET_REGISTRY)

    assert len(dataset_ids) == len(set(dataset_ids))
    assert DOMESTIC_KIWOOM_START_DATE == DEFAULT_PROTOCOL_SETTINGS["dates"]["domestic_kiwoom_start_date"]
    assert INDEX_SYMBOLS["KOSPI"].source.provider == "kiwoom"
    assert INDEX_SYMBOLS["DJI"].source.provider == "kis"
    for symbol_name in D1_INDEX_NAMES:
        for interval in D1_INTERVALS:
            dataset = DATASET_REGISTRY[d1_dataset_id(symbol_name, interval)]
            assert dataset.stage == "D1"
            assert dataset.symbols[0].symbol == symbol_name
            assert dataset.interval == interval
    assert DATASET_REGISTRY["d2_us-nasdaq-spx-dji_daily"].symbols[1].symbol == "SPX"
    assert DATASET_REGISTRY["d3_global-major_daily"].market_group == "global-major"


def test_split_name_uses_start_end_boundaries_and_excludes_gaps() -> None:
    assert split_name("2004-12-31") == "excluded"
    assert split_name("2005-01-01") == "train"
    assert split_name("2016-12-31 15:30:00") == "train"
    assert split_name("2017-01-01") == "validation"
    assert split_name("2021-01-01") == "test"

    split = SplitProtocol(
        train_start="2020-01-01",
        train_end="2020-01-10",
        validation_start="2020-01-20",
        validation_end="2020-01-30",
        test_start="2020-02-10",
        test_end="2020-02-20",
        embargo_days=5,
        label_horizons=(1, 5),
        rolling_statistics_rule="test rule",
    )
    assert split_name("2019-12-31", split) == "excluded"
    assert split_name("2020-01-15", split) == "excluded"
    assert split_name("2020-02-01", split) == "excluded"
    assert split_name("2020-02-21", split) == "excluded"


def test_minute_split_protocol_boundaries() -> None:
    assert split_section_for_interval("1d") == "split"
    assert split_section_for_interval("1m") == "split_minute"
    with pytest.raises(ValueError, match="unsupported interval"):
        split_section_for_interval("1w")

    assert split_protocol_for_interval("1d").train_start == "2005-01-01"
    minute = split_protocol_for_interval("1m")
    assert minute == SPLIT_PROTOCOL_MINUTE

    assert split_name("2025-06-30 15:30:00", minute) == "excluded"
    assert split_name("2025-07-01 09:00:00", minute) == "train"
    assert split_name("2026-01-31", minute) == "train"
    assert split_name("2026-02-01 09:00:00", minute) == "validation"
    assert split_name("2026-04-30", minute) == "validation"
    assert split_name("2026-05-01 09:00:00", minute) == "test"


def test_split_protocol_config_for_interval_uses_matching_section() -> None:
    assert split_protocol_config_for_interval("1d") == split_protocol_config()
    minute_config = split_protocol_config_for_interval("1m")
    assert minute_config["train_start"] == "2025-07-01"
    assert minute_config["test_start"] == "2026-05-01"
    assert minute_config["random_split_forbidden"] is True


def test_protocol_settings_validation_rejects_leaky_minute_split() -> None:
    settings = deepcopy(DEFAULT_PROTOCOL_SETTINGS)
    settings["split_minute"]["train_end"] = "2026-03-01"

    with pytest.raises(ProtocolConfigError, match="ordered"):
        validate_protocol_settings(settings)


def test_protocol_config_manifest_and_metrics_are_built_from_settings() -> None:
    dataset = DATASET_REGISTRY["d1_dji_daily"]
    config = build_protocol_config(dataset)
    manifest = build_manifest(dataset, config, "abc123", seed=7, source_notebook="custom.ipynb")
    metrics = empty_data_quality_metrics(dataset)

    assert config["data_source_policy"] == data_source_policy_config()
    assert config["feature"] == FEATURE_PROTOCOL
    assert config["filtering"] == FILTERING_PROTOCOL
    assert manifest["source_notebook"] == "custom.ipynb"
    assert manifest["split"] == split_protocol_config()
    assert manifest["required_outputs"] == ["experiment_config.json", "metrics.json", "manifest.json"]
    assert metrics["split_row_counts_by_symbol"]["DJI"] == {
        "train": None,
        "validation": None,
        "test": None,
        "excluded": None,
    }


def test_protocol_settings_validation_rejects_leaky_split() -> None:
    settings = deepcopy(DEFAULT_PROTOCOL_SETTINGS)
    settings["split"]["embargo_days"] = 1

    with pytest.raises(ProtocolConfigError, match="embargo_days"):
        validate_protocol_settings(settings)


def test_kis_overseas_index_symbols_validate_against_master_rows() -> None:
    master_rows = [
        {"symbol": symbol.source.master_symbol or f"US#{symbol.source.minute_symbol}"}
        for symbol in INDEX_SYMBOLS.values()
        if symbol.source.provider == "kis"
    ]

    report = validate_kis_overseas_index_symbols(overseas_index_info=master_rows)

    assert report["valid"] is True
    assert report["checked_count"] == sum(1 for symbol in INDEX_SYMBOLS.values() if symbol.source.provider == "kis")
    assert report["missing"] == {}


def test_kis_overseas_index_symbol_validation_reports_missing_symbols() -> None:
    report = validate_kis_overseas_index_symbols(overseas_index_info=[{"symbol": "US#SPX"}], strict=False)

    assert report["valid"] is False
    assert "DJI" in report["missing"]
    with pytest.raises(ProtocolConfigError, match="missing"):
        validate_kis_overseas_index_symbols(overseas_index_info=[{"symbol": "US#SPX"}])
