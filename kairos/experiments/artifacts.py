from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_HASH_EXCLUDED_KEYS = {
    "seed",
    "started_at",
    "finished_at",
    "created_at_utc",
    "duration_seconds",
    "runtime_seconds",
    "hostname",
    "absolute_paths",
}


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "kairos").exists():
            return candidate
    raise RuntimeError("Could not find kairos project root")


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value.__class__.__module__ == "numpy":
        return value.item()
    return value


def strip_hash_excluded(
    value: Any,
    *,
    excluded_keys: Iterable[str] = DEFAULT_HASH_EXCLUDED_KEYS,
) -> Any:
    excluded = set(excluded_keys)
    if isinstance(value, dict):
        return {
            key: strip_hash_excluded(item, excluded_keys=excluded)
            for key, item in value.items()
            if key not in excluded
        }
    if isinstance(value, list):
        return [strip_hash_excluded(item, excluded_keys=excluded) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_hash(config: dict[str, Any], *, length: int = 8) -> str:
    canonical = canonical_json(strip_hash_excluded(config))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latest_run(cfg_dir: Path, *, seed: int | None = None) -> Path:
    pattern = f"run-*_seed-{seed}" if seed is not None else "run-*_seed-*"
    runs = sorted(cfg_dir.glob(pattern))
    if not runs:
        raise FileNotFoundError(f"No run directory found under {cfg_dir}")
    return runs[-1]
