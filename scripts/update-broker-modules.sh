#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

uv lock -P broker-modules
uv sync

uv run python - <<'PY'
from importlib.metadata import version

print(f"broker-modules {version('broker-modules')} synced")
PY
